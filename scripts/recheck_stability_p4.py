"""Recheck unresolved native P4 controls without repeating the N48 stability map."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

# Import the v1 launcher before NumPy so its single-thread BLAS settings apply.
from run_stability_p4 import (
    FREQUENCY, OUTPUT, P4, RADIUS, ROOT, Launch, agreements, integrate_rf, load_field,
    read_json, sha, verify_inputs, write_table,
)

import numpy as np

from three_phase_trap.long_rf_run import run_lock, write_json
from three_phase_trap.pilot_dynamics import PolynomialRF
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.stability_rf_p4_v2 import EscapeAuditError, NativeEscapeV2


RECHECK = OUTPUT.with_name("p4_recheck_v2")
CHUNK = 1000
_native = {}


@dataclass(frozen=True)
class Job:
    launch: Launch
    mesh: int
    level: str
    horizon: int

    @property
    def label(self):
        return f"P4_h{self.mesh}_{self.level}"


def archived_plan(source, refined_control):
    summary_path = source / "summary.json"
    protocol_path = source / "protocol.json"
    old_protocol = read_json(protocol_path)
    summary = read_json(summary_path)
    horizon = summary["final_horizon_RF"]
    if summary["profile"] != "map" or horizon != old_protocol["settings"]["stages"][-1]:
        raise ValueError("Expected the finished v1 stability map")
    hashes, archive, jobs = {}, {}, []

    def load(path):
        hashes[path.relative_to(source).as_posix()] = sha(path)
        return read_json(path)

    load(summary_path)
    load(protocol_path)
    for case in summary["controls"]:
        launch = Launch(case["coupling"], case["radius_mm"], case["start"])
        filename = launch.tag + ".json"
        path = source / "N48" / "refined" / filename
        if not path.exists():
            path = source / "N48" / f"RF{horizon}" / filename
        rows = {"N48": load(path), "N48_strict": load(source / "controls" / "N48_strict" / filename)}
        for mesh in (35, 50):
            for level in ("coarse", "fine"):
                job = Job(launch, mesh, level, horizon)
                row = load(source / "controls" / job.label / filename)
                rows[job.label] = row
                if row["status"] == "numerical_error":
                    jobs.append(job)
        for label, row in rows.items():
            if (row["status"] != case[label + "_status"] or row["escape_RF"] != case[label + "_escape_RF"]
                    or row["horizon_RF"] != horizon or any(row[key] != value for key, value in asdict(launch).items())):
                raise ValueError(f"Archived summary and checkpoint disagree: {launch.tag}/{label}")
        if case["agreement"] == "escape_time_disagrees":
            jobs.extend(Job(launch, mesh, "refined", horizon) for mesh in (35, 50))
        archive[launch] = (case["agreement"], rows)
    controls = {**old_protocol["native"], "refined": refined_control}
    return archive, jobs, controls, hashes


def native(mesh):
    if mesh not in _native:
        _native[mesh] = NativeEscapeV2(load_field(mesh, 4, amplitude=1.), RADIUS)
    return _native[mesh]


def diagnostics_valid(row, first, last, initial):
    status = row["status"]
    if row["segment_RF"] != [first, last] or row["initial_state_SI"] != initial.tolist():
        raise ValueError("P4 checkpoint chain changed its clock or state")
    if status == "numerical_error":
        return
    reached = row["reached_RF"]
    escape = row["escape_RF"]
    if status == "advanced":
        if abs(reached - last) > 1e-8 or escape is not None:
            raise ValueError("Saved segment did not reach its checkpoint")
    elif status != "escaped" or escape != reached or not first <= reached <= last:
        raise ValueError("Invalid terminal P4 segment")
    if not np.all(np.isfinite(row["final_state_SI"])) or len(row["final_state_SI"]) != 4:
        raise ValueError("Invalid checkpoint state")


def native_job(out, job, coefficient, control, legacy_diagnostic=False):
    started = time.perf_counter()
    launch = job.launch
    initial, phase = launch.state()
    qm = launch.charge_mass(coefficient)
    folder = out / "controls" / job.label / launch.tag
    folder.mkdir(parents=True, exist_ok=True)
    crossings = checked = max_stalled = nfev = 0
    maximum = float(np.linalg.norm(initial[:2]))
    # Refined runs are continuous for comparison with the original timing control.
    chunk = job.horizon if job.level == "refined" or legacy_diagnostic else CHUNK
    status, escape, detail, reached = "held", None, None, 0
    for first in range(0, job.horizon, chunk):
        last = min(first + chunk, job.horizon)
        path = folder / f"RF{last:05d}.json"
        if path.exists():
            row = read_json(path)
            diagnostics_valid(row, first, last, initial)
        else:
            record = dict(segment_RF=[first, last], initial_state_SI=initial.tolist())
            try:
                result = native(job.mesh).segment(
                    initial, qm, FREQUENCY, phase, first, last, steps=control["steps"],
                    rtol=control["rtol"], atol=control["atol"], legacy_diagnostic=legacy_diagnostic)
                row = {**record, **result, "status": "escaped" if result["escape_RF"] is not None else "advanced"}
            except EscapeAuditError as exc:
                row = {**record, **exc.diagnostic, "status": "numerical_error", "error": str(exc)}
            except Exception as exc:
                row = {**record, "status": "numerical_error", "error": f"{type(exc).__name__}: {exc}"}
            diagnostics_valid(row, first, last, initial)
            write_json(path, row)
            print(f"{job.label} {launch.tag} RF {first}:{last}: {row['status']}, "
                  f"crossings={row.get('crossings', '?')}", flush=True)
        crossings += row.get("crossings", 0)
        checked += row.get("checked_steps", 0)
        nfev += row.get("nfev", 0)
        max_stalled = max(max_stalled, row.get("max_zero_time_run", 0))
        maximum = max(maximum, row.get("sampled_max_radius_m", 0.))
        reached = row.get("reached_RF", first)
        if row["status"] == "numerical_error":
            status, detail = row["status"], row["error"]
            break
        initial = np.array(row["final_state_SI"])
        if row["status"] == "escaped":
            status, escape = row["status"], row["escape_RF"]
            break
    report = dict(**asdict(launch), model=job.label, status=status, horizon_RF=job.horizon,
                  reached_RF=reached, escape_RF=escape, error=detail, crossings=crossings,
                  checked_steps=checked, max_zero_time_run=max_stalled, nfev=nfev,
                  sampled_max_radius_m=maximum, wall_s=time.perf_counter() - started)
    write_json(folder / "summary.json", report)
    return report


def refined_n48(out, launch, horizon, coefficient, control):
    path = out / f"{launch.tag}_N48_refined.json"
    if path.exists():
        return read_json(path)
    q = RFReconstruction.load(P4 / "h35_P4/diagnostic_N48.json").unit_quadratures
    initial, phase = launch.state()
    try:
        result = integrate_rf(PolynomialRF(q), initial, launch.charge_mass(coefficient), 1.,
                              FREQUENCY, phase, 0., horizon / FREQUENCY, output_samples_per_period=1,
                              max_steps_per_period=48, rtol=control["rtol"], atol=control["atol"], radius=RADIUS)
        escape = None if result.exit_time is None else result.exit_time * FREQUENCY
        row = dict(**asdict(launch), status="held" if escape is None else "escaped", escape_RF=escape,
                   horizon_RF=horizon, nfev=result.nfev, wall_s=result.wall_seconds)
    except Exception as exc:
        row = dict(**asdict(launch), status="numerical_error", escape_RF=None, horizon_RF=horizon,
                   error=f"{type(exc).__name__}: {exc}")
    write_json(path, row)
    return row


def recheck(directory, archive, jobs, coefficient, controls, workers):
    updated = {launch: rows.copy() for launch, (_, rows) in archive.items()}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        try:
            futures = {pool.submit(native_job, directory, job, coefficient, controls[job.level]): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                updated[job.launch][job.label] = future.result()
        except KeyboardInterrupt:
            pool.terminate_workers()
            raise
    reports = []
    for launch, rows in updated.items():
        previous, _ = archive[launch]
        if previous == "escape_time_disagrees":
            rows["N48_refined"] = refined_n48(directory, launch, rows["N48"]["horizon_RF"],
                                             coefficient, controls["refined"])
        agreement, spread = agreements(rows["N48"], [row for label, row in rows.items() if label != "N48"])
        native_rows = [row for label, row in rows.items() if label.startswith("P4_")]
        native_agreement, native_spread = agreements(native_rows[0], native_rows[1:])
        report = dict(**asdict(launch), v1_agreement=previous, agreement=agreement, escape_spread_RF=spread,
                      direct_P4_agreement=native_agreement, direct_P4_escape_spread_RF=native_spread)
        for label, row in rows.items():
            for key in ("status", "escape_RF", "error", "reached_RF", "crossings", "max_zero_time_run"):
                report[f"{label}_{key}"] = row.get(key)
        reports.append(report)
    write_table(directory / "p4_controls_rechecked.csv", reports)
    write_json(directory / "summary.json", dict(cases=reports, new_P4_runs=len(jobs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--diagnose", action="store_true", help="Replay the legacy cap on one failed P4 job")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Expected 1–4 workers")
    out = (args.out or (RECHECK.with_name("p4_legacy_diagnostic") if args.diagnose else RECHECK)).resolve()
    if out == OUTPUT.resolve() or OUTPUT.resolve() in out.parents:
        parser.error("Use a directory outside the frozen v1 results")
    coefficient, reference, hashes = verify_inputs()
    archive, jobs, controls, old_hashes = archived_plan(OUTPUT, reference["refined"])
    for path in (Path(__file__), ROOT / "src/three_phase_trap/stability_rf_p4_v2.py"):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    old_protocol = read_json(OUTPUT / "protocol.json")
    if coefficient != old_protocol["C4_per_volt_m4"] or any(hashes.get(k) != v for k, v in old_protocol["input_sha256"].items()):
        raise ValueError("Frozen v1 inputs have changed")
    if args.diagnose:
        jobs = [max((job for job in jobs if job.mesh == 35 and job.level == "coarse"),
                    key=lambda job: job.launch.radius_mm)]
    protocol = dict(version=2, old_inputs_sha256=old_hashes, sources_sha256=hashes, jobs=[asdict(j) for j in jobs],
                    controls=controls, chunk_RF=CHUNK, legacy_diagnostic=args.diagnose,
                    N48_refined_steps=48)
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        path = out / "protocol.json"
        if path.exists() and read_json(path) != protocol:
            raise ValueError("Recheck protocol changed. Choose a new --out directory")
        if not path.exists():
            write_json(path, protocol)
        print(f"[READY] {len(jobs)} P4 jobs in {out}", flush=True)
        if args.check_only:
            return
        if args.diagnose:
            job = jobs[0]
            result = native_job(out, job, coefficient, controls[job.level], legacy_diagnostic=True)
            write_json(out / "summary.json", result)
            print(result, flush=True)
        else:
            recheck(out, archive, jobs, coefficient, controls, args.workers)
        print(f"[COMPLETE] {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Repeat the same command to resume from saved checkpoints.", flush=True)
        sys.exit(130)
