"""Checkpointed native FEM trajectories with a global RF clock.

This runner never changes the frozen native-FEM integrator. Checkpoint
boundaries preserve the state and the exact integer RF-cycle count. A local
segment clock avoids loss of precision at late global times: every segment
starts on a whole RF period, so its phase offset is exactly the same phi0.
Absolute output times are reconstructed from integer RF sample indices.
A completed trajectory
is a computed result, not a claim of long-time spatial/temporal convergence.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    replace_file(temporary, path)


def replace_file(temporary, target):
    # Windows readers may briefly hold status.json without delete sharing.
    # A concurrent status check must not abort an hour-long computation.
    for attempt in range(20):
        try:
            Path(temporary).replace(target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def write_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    replace_file(temporary, path)


@contextmanager
def run_lock(directory):
    """An OS lock releases automatically after exit/crash; no stale PID lock."""
    path = Path(directory)/"run.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("This run is already active in another process") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if __import__("os").name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)


def is_running(directory):
    if not (Path(directory)/"run.lock").exists():
        return False
    try:
        with run_lock(directory):
            return False
    except RuntimeError:
        return True


class RunStore:
    def __init__(self, directory, settings):
        self.directory, self.settings = Path(directory), settings
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parts = self.directory/"checkpoints"
        self.parts.mkdir(exist_ok=True)
        self.signature = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def freeze(self):
        path = self.directory/"settings.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != self.settings:
                raise ValueError("Run settings/source hashes changed; use a different run name")
        else:
            if list(self.parts.glob("part_*.json")) or (self.directory/"COMPLETED.json").exists():
                raise ValueError("Existing checkpoints have no frozen settings")
            write_json(path, self.settings)

    @property
    def count(self):
        s = self.settings
        return (s["RF_periods"]+s["chunk_RF_periods"]-1)//s["chunk_RF_periods"]

    def bounds(self, index):
        first = index*self.settings["chunk_RF_periods"]
        return first, min(first+self.settings["chunk_RF_periods"], self.settings["RF_periods"])

    def paths(self, index):
        base = self.parts/f"part_{index:05d}"
        return base.with_suffix(".npz"), base.with_suffix(".json"), base.with_suffix(".jsonl")

    def read_part(self, index, initial):
        arrays, metadata, log = self.paths(index)
        info = json.loads(metadata.read_text(encoding="utf-8"))
        first, last = self.bounds(index)
        if info["signature"] != self.signature or info["index"] != index or info["RF_bounds"] != [first, last]:
            raise ValueError(f"Checkpoint belongs to a different case: {metadata}")
        if sha(arrays) != info["arrays_sha256"] or sha(log) != info["crossings_sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {metadata}")
        with np.load(arrays) as data:
            t, y, owners = data["time_s"], data["states"], data["triangles"]
        n = (last-first)*self.settings["samples_per_RF_period"]+1
        frequency = self.settings["physical"]["frequency_Hz"]
        if t.shape != (n,) or y.shape != (n, 4) or owners.shape != (n,):
            raise ValueError("Incomplete checkpoint sample count")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0) or np.any(owners < 0):
            raise ValueError("Invalid checkpoint state/time/ownership")
        if abs(t[0]-first/frequency) > 1e-13 or abs(t[-1]-last/frequency) > 1e-13:
            raise ValueError("Checkpoint changed the absolute RF clock")
        if not np.array_equal(y[0], initial) or not np.array_equal(y[-1], info["final_state"]):
            raise ValueError("Checkpoint state discontinuity")
        if info["status"] != "completed":
            raise ValueError("Non-completed segment cannot be a checkpoint")
        return t, y, owners, info

    def scan(self):
        records, initial, missing = [], np.array(self.settings["initial_state"]), False
        for index in range(self.count):
            _, metadata, _ = self.paths(index)
            if not metadata.exists():
                missing = True
                continue
            if missing:
                raise ValueError("Checkpoint chain contains a gap")
            _, y, _, info = self.read_part(index, initial)
            initial = y[-1].copy()
            records.append(info)
        return records, initial

    def commit(self, index, result, initial):
        if result.run.status != "completed" or not np.array_equal(result.run.states[0], initial):
            raise ValueError("Segment did not complete or changed its initial state")
        arrays, metadata, log = self.paths(index)
        write_npz(arrays, time_s=result.run.time, states=result.run.states, triangles=result.sample_triangles)
        temporary = log.with_name(log.name+".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for crossing in result.crossings:
                stream.write(json.dumps(crossing, allow_nan=False)+"\n")
        replace_file(temporary, log)
        info = dict(signature=self.signature, index=index, RF_bounds=list(self.bounds(index)), status="completed",
                    arrays_sha256=sha(arrays), crossings_sha256=sha(log), final_state=result.run.states[-1].tolist(),
                    initial_state=np.asarray(initial).tolist(), integration_wall_seconds=result.run.wall_seconds,
                    nfev=result.run.nfev, diagnostics=result.diagnostics)
        # JSON is the commit record; an interrupted array/log write is ignored.
        write_json(metadata, info)
        return self.read_part(index, initial)[3]


def join_parts(store):
    n = store.settings["RF_periods"]*store.settings["samples_per_RF_period"]+1
    t, states, owners = np.empty(n), np.empty((n, 4)), np.empty(n, np.int32)
    cursor, initial = 0, np.array(store.settings["initial_state"])
    for index in range(store.count):
        tt, yy, oo, _ = store.read_part(index, initial)
        skip = 0 if index == 0 else 1
        stop = cursor+len(tt)-skip
        t[cursor:stop], states[cursor:stop], owners[cursor:stop] = tt[skip:], yy[skip:], oo[skip:]
        cursor, initial = stop, yy[-1].copy()
    if cursor != n or np.any(np.diff(t) <= 0):
        raise ValueError("Joined trajectory has missing/duplicate time nodes")
    return t, states, owners


def verify_completion(store, marker=None):
    if marker is None:
        marker = json.loads((store.directory/"COMPLETED.json").read_text(encoding="utf-8"))
    if marker["signature"] != store.signature or marker["status"] != "completed":
        raise ValueError("Completion marker does not match the run")
    for name, expected in marker["artifacts"].items():
        if Path(name).name != name or sha(store.directory/name) != expected:
            raise ValueError("Final artifact checksum mismatch: "+name)
    if "source_manifest.json" in marker["artifacts"]:
        sources = json.loads((store.directory/"source_manifest.json").read_text(encoding="utf-8"))
        for name, expected in sources.items():
            if Path(name).name != name or sha(store.directory/"source_snapshot"/name) != expected:
                raise ValueError("Source snapshot checksum mismatch: "+name)
    records, final_state = store.scan()
    if len(records) != store.count:
        raise ValueError("Not all segments have completed")
    with np.load(store.directory/"trajectory.npz") as data:
        t, states, owners = data["time_s"], data["states"], data["triangles"]
    n = store.settings["RF_periods"]*store.settings["samples_per_RF_period"]+1
    end = store.settings["RF_periods"]/store.settings["physical"]["frequency_Hz"]
    if states.shape != (n, 4) or t.shape != (n,) or owners.shape != (n,) or not np.all(np.isfinite(states)) or np.any(np.diff(t) <= 0):
        raise ValueError("Final trajectory is incomplete")
    if t[0] != 0. or abs(t[-1]-end) > 1e-13 or not np.array_equal(states[0], store.settings["initial_state"]) or not np.array_equal(states[-1], final_state):
        raise ValueError("Final trajectory changed its clock or initial/final state")
    return marker


def export_result(store, t, states, owners, make_figures=True):
    from .slow_analysis import centered_period_mean
    from .pilot_dynamics import oscillation_metrics
    write_npz(store.directory/"trajectory.npz", time_s=t, states=states, triangles=owners)
    tm, position = centered_period_mean(states[:, :2], t, store.settings["samples_per_RF_period"])
    stride = store.settings["samples_per_RF_period"]
    tm, position = tm[::stride].copy(), position[::stride].copy()
    write_npz(store.directory/"slow_samples.npz", time_s=tm, positions_m=position)
    summary = dict(status="computed", start_time_s=float(t[0]), end_time_s=float(t[-1]), samples=len(t),
                   simulated_reference_slow_periods=float(t[-1]/store.settings["reference_slow_period_s"]),
                   maximum_full_radius_m=float(np.max(np.linalg.norm(states[:, :2], axis=1))),
                   maximum_slow_abs_y_m=float(np.max(abs(position[:, 1]))),
                   slow_x_oscillations=oscillation_metrics(tm, position, store.settings["reference_slow_period_s"]),
                   convergence_status="not_assessed_for_this_duration",
                   meaning="Trajectory computation completed; 50-period convergence and B/C comparison require separate controls")
    write_json(store.directory/"summary.json", summary)
    artifacts = ["trajectory.npz", "slow_samples.npz", "summary.json", "settings.json"]
    if (store.directory/"source_manifest.json").exists():
        artifacts.append("source_manifest.json")
    if make_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        cut = slice(None, None, max(1, len(tm)//25000))
        clock, xy = tm[cut]/store.settings["reference_slow_period_s"], position[cut]*1e3
        for j, name in enumerate(("x", "y")):
            axes[0, j].plot(clock, xy[:, j], linewidth=.8)
            axes[0, j].set(xlabel="t / reference slow period", ylabel=f"Slow {name} (mm)")
        axes[1, 0].plot(xy[:, 0], xy[:, 1], linewidth=.7)
        axes[1, 0].set(xlabel="Slow x (mm)", ylabel="Slow y (mm)")
        axes[1, 1].plot(clock, np.linalg.norm(xy, axis=1), linewidth=.8)
        axes[1, 1].set(xlabel="t / reference slow period", ylabel="Slow radius (mm)")
        for ax in axes.flat:
            ax.grid(alpha=.25)
        fig.suptitle(f"A: direct native FEM P4, h={store.settings['mesh_um']} µm, phase={store.settings['phase_rad']:.6g} rad")
        fig.savefig(store.directory/"trajectory_overview.png", dpi=160)
        plt.close(fig)
        artifacts.append("trajectory_overview.png")
    return artifacts


def run_case(store, field_factory, *, integrator=None, notify=print, make_figures=True):
    if integrator is None:
        from .direct_fem_dynamics import integrate_native
        integrator = integrate_native
    with run_lock(store.directory):
        store.freeze()
        if (store.directory/"COMPLETED.json").exists():
            result = verify_completion(store)
            notify("[COMPLETED] Existing result verified; no integration was repeated.")
            return result
        records, initial = store.scan()
        s, p = store.settings, store.settings["physical"]
        committed = records[-1]["RF_bounds"][1] if records else 0
        initial_time = committed/p["frequency_Hz"]
        start, prior_wall = time.monotonic(), sum(r["integration_wall_seconds"] for r in records)
        target = s["RF_periods"]/p["frequency_Hz"]
        def emit(stage, simulation_time, message=""):
            elapsed = time.monotonic()-start
            advanced = simulation_time-initial_time
            eta = max(0., (target-simulation_time)*elapsed/advanced) if advanced > 0 else None
            value = dict(status=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
                         progress_percent=min(100., 100*simulation_time/target), simulation_time_s=simulation_time,
                         target_time_s=target, slow_periods=simulation_time/s["reference_slow_period_s"],
                         checkpoint_time_s=committed/p["frequency_Hz"], committed_segments=len(records), total_segments=store.count,
                         elapsed_this_launch_s=elapsed, accumulated_wall_seconds=prior_wall+elapsed,
                         estimated_remaining_s=eta, message=message)
            write_json(store.directory/"status.json", value)
            remaining = "estimating" if eta is None else f"{eta/60:.1f} min"
            line = f"[{stage.upper()}] {value['progress_percent']:6.2f}% | {value['slow_periods']:.3f} slow periods | checkpoints {len(records)}/{store.count} | ETA {remaining} {message}"
            with (store.directory/"run.log").open("a", encoding="utf-8") as stream:
                stream.write(value["updated_utc"]+" "+line+"\n")
            notify(line)
        current = initial_time
        try:
            emit("preparing", current, "Loading existing P4 field; initial trajectory starts at t=0.")
            field = field_factory() if len(records) < store.count or s.get("smoke_test") else None
            for index in range(len(records), store.count):
                first, last = store.bounds(index)
                emit("running", first/p["frequency_Hz"], f"segment {index+1}")
                def progress(value):
                    nonlocal current
                    current = first/p["frequency_Hz"]+float(value["time_s"])
                    emit("running", current)
                # first is an integer RF cycle count. E(t_start+tau) has the
                # same phase phi0 as E(tau), since Omega*t_start=2*pi*first.
                # This is an exact time-coordinate change, not an RF restart.
                result = integrator(field, initial, p["charge_C"], p["mass_kg"], p["frequency_Hz"], s["phase_rad"],
                                    0., (last-first)/p["frequency_Hz"],
                                    method=s["control"]["method"], steps_per_period=s["control"]["steps"],
                                    rtol=s["control"]["rtol"], atol=s["control"]["atol"],
                                    samples_per_period=s["samples_per_RF_period"], progress=progress)
                if abs(result.run.time[0]) > 1e-15 or abs(result.run.time[-1]-(last-first)/p["frequency_Hz"]) > 1e-13:
                    raise ValueError("Native integrator returned an incorrect local time interval")
                n = s["samples_per_RF_period"]
                result.run.time = (first*n+np.arange(len(result.run.time), dtype=np.int64))/(p["frequency_Hz"]*n)
                for crossing in result.crossings:
                    crossing["time_s"] += first/p["frequency_Hz"]
                records.append(store.commit(index, result, initial))
                initial, committed = result.run.states[-1].copy(), last
                current = committed/p["frequency_Hz"]
                del result
                emit("checkpoint_saved", current)
            emit("postprocessing", target, "Joining and checking all samples; no completion marker yet.")
            t, states, owners = join_parts(store)
            extra = []
            if s.get("smoke_test"):
                reference = integrator(field, np.array(s["initial_state"]), p["charge_C"], p["mass_kg"], p["frequency_Hz"], s["phase_rad"], 0., target,
                                       method=s["control"]["method"], steps_per_period=s["control"]["steps"],
                                       rtol=s["control"]["rtol"], atol=s["control"]["atol"], samples_per_period=s["samples_per_RF_period"])
                differences = np.linalg.norm(states[:, :2]-reference.run.states[:, :2], axis=1)/s["R_scale_m"]
                comparison = dict(relative_rms=float(np.sqrt(np.mean(differences**2))), relative_max=float(differences.max()),
                                  clock_difference_max_s=float(np.max(abs(t-reference.run.time))))
                comparison["passed"] = bool(reference.run.status == "completed" and comparison["relative_max"] < 1e-8 and comparison["clock_difference_max_s"] < 1e-13)
                write_json(store.directory/"smoke_comparison.json", comparison)
                if not comparison["passed"]:
                    raise ValueError("Segmented/one-shot native RF smoke comparison failed")
                extra.append("smoke_comparison.json")
            artifacts = export_result(store, t, states, owners, make_figures=make_figures)+extra
            marker = dict(status="completed", signature=store.signature, completed_utc=datetime.now(timezone.utc).isoformat(),
                          start_time_s=float(t[0]), end_time_s=float(t[-1]), RF_periods=s["RF_periods"], segments=store.count,
                          artifacts={name: sha(store.directory/name) for name in artifacts},
                          convergence_status="not_assessed_for_this_duration")
            del t, states, owners
            emit("verifying", target, "Checking final checksums and checkpoint chain.")
            verify_completion(store, marker)
            write_json(store.directory/"COMPLETED.json", marker)
            emit("completed", target, "COMPLETED.json written; result files verified.")
            return marker
        except KeyboardInterrupt:
            emit("interrupted", current, "Restart the same command to resume from the last saved checkpoint.")
            raise
        except Exception as exc:
            (store.directory/"error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            emit("failed", current, str(exc))
            raise
