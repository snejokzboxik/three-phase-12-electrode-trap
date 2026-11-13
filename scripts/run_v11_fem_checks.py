"""Run or resume audited native P4 controls for selected V11 trajectories."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

# Import the V11 runtime before NumPy for its one-thread BLAS settings.
from run_v11_xy import CHARGE, FREQ, MASS, OUT, RADIUS, ROOT

import numpy as np
import scipy
from three_phase_trap import direct_fem
from three_phase_trap.direct_fem import DirectFEMRF, SectorSolution
from three_phase_trap.gmsh_mesh import load_gmsh_mesh
from three_phase_trap.long_rf_run import replace_file, run_lock, sha, write_json
from three_phase_trap.stability_rf_p4_v2 import EscapeAuditError
from three_phase_trap.v11_native_sampling import NativeTrajectory

P4_CACHE = ROOT / "outputs/abc_validation/direct_fem_validation/v2"
DEFAULT_OUTPUT = OUT / "v11_completion/fem_native"
MESHES = (35, 50)
SOLVER = {"steps": 50, "rtol": 2e-9, "atol": [2e-13, 2e-13, 2e-11, 2e-11]}
STATE = ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s")
CASE_COLUMNS = ("job_id", "model", *STATE, "phase_rad", "voltage_V", "mesh_um",
                "representation", "horizon_RF", "reason")
RESULT_COLUMNS = (*CASE_COLUMNS, "status", "escape_RF", "reached_RF", "final_state_SI",
                  "sampled_max_radius_m", "nfev", "checked_steps", "crossings",
                  "max_zero_time_run", "wall_s", "trajectory_path", "reused_from", "error")
_runner = None


def read_json(path):
    return path and json.loads(path.read_text(encoding="utf-8"))


def mesh_path(mesh):
    return ROOT / f"outputs/meshes/gmsh_{mesh}um/trap.msh"


def sector_path(mesh, model):
    folder = OUT / f"fields/h{mesh}_P4_C" if model == "C" else P4_CACHE / f"h{mesh}_P4"
    return folder / "sector_solution.npz"


def archived_path(mesh):
    return OUT / f"numerical_checks/native_C_early_exit/h{mesh}.json"


def preflight():
    """Check dependencies and validate existing FEM files without loading a mesh."""
    import meshio
    import skfem

    inputs = {}
    for mesh in MESHES:
        abc = sector_path(mesh, "A")
        anti = sector_path(mesh, "C")
        abc_manifest = abc.parent / "complete.json"
        anti_manifest = anti.parent / "validation.json"
        geometry = mesh_path(mesh)
        archived = archived_path(mesh)
        for path in (geometry, abc, anti, abc_manifest, anti_manifest, archived):
            if not path.is_file():
                raise FileNotFoundError(f"Missing native V11 input: {path.relative_to(ROOT)}")
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        metadata = read_json(abc_manifest)
        authenticated = {
            geometry: metadata["signature"]["full_mesh_sha256"],
            abc: metadata["artifacts"][abc.name],
            anti: read_json(anti_manifest)["sector_sha256"],
        }
        for path, expected in authenticated.items():
            if inputs[path.relative_to(ROOT).as_posix()] != expected:
                raise ValueError(f"Native V11 input checksum differs: {path}")
    return {
        "input_sha256": inputs,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-fem": skfem.__version__,
                     "meshio": meshio.__version__},
    }


def case_rows(candidates, horizon):
    unique = {}
    for candidate in candidates:
        model = candidate["model"]
        if model not in ("A", "B", "C"):
            raise ValueError(f"Unknown V11 field: {model}")
        values = [float(candidate[name]) for name in (*STATE, "phase_rad", "voltage_V")]
        if not all(map(math.isfinite, values)) or math.hypot(*values[:2]) >= RADIUS or values[-1] <= 0:
            raise ValueError(f"Invalid initial state or voltage: {candidate}")
        values = [0. if value == 0. else value for value in values]
        values[-2] %= math.tau
        physical = (model, *values)
        reasons = unique.setdefault(physical, set())
        if candidate.get("reason"):
            reasons.add(str(candidate["reason"]))
    rows = []
    for physical, reasons in sorted(unique.items()):
        for mesh in MESHES:
            signature = hashlib.sha256(json.dumps([physical, mesh]).encode()).hexdigest()[:20]
            identity = dict(zip(("model", *STATE, "phase_rad", "voltage_V"), physical))
            rows.append(dict(job_id=signature, **identity, mesh_um=mesh, representation="P4",
                             horizon_RF=horizon, reason="; ".join(sorted(reasons))))
    return rows


def write_table(path, columns, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    replace_file(temporary, path)


def reuse_exit(case):
    if case["model"] != "C":
        return None
    path = archived_path(case["mesh_um"])
    old = read_json(path)
    reached = old["escape_RF"]
    if (old["initial_state_SI"] != [case[name] for name in STATE]
            or math.radians(old["phase_deg"]) != case["phase_rad"]
            or old["voltage_V"] != case["voltage_V"]
            or reached is None or reached > case["horizon_RF"]):
        return None
    return dict(**case, status="escaped", escape_RF=reached, reached_RF=reached,
                final_state_SI=old["final_state_SI"], sampled_max_radius_m=old["max_radius_m"],
                nfev=old["nfev"], reused_from=path.relative_to(ROOT).as_posix())


def init_worker(mesh, model):
    global _runner
    sector = SectorSolution.load(sector_path(mesh, model))
    if sector.degree != 4:
        raise ValueError(f"Expected a P4 sector for h{mesh}/{model}")
    geometry = load_gmsh_mesh(mesh_path(mesh))
    rho = direct_fem.RHO
    try:
        if model == "C":
            direct_fem.RHO = -1. + 0j
        field = DirectFEMRF(sector, geometry, amplitude=1.)
    finally:
        direct_fem.RHO = rho
    if model == "B":
        # sin(chi) multiplies the real part of A's complex RF potential.
        field.gx.imag = 0.
        field.gy.imag = 0.
    _runner = NativeTrajectory(field, RADIUS)


def check_segment(row, initial, first, last):
    if row["segment_RF"] != [first, last] or row["initial_state_SI"] != initial.tolist():
        raise ValueError("Native P4 checkpoint trajectory/clock mismatch")
    status = row["status"]
    if status == "numerical_error" and "time_RF" not in row:
        return
    reached = row["reached_RF"]
    ticks = np.asarray(row["time_RF"])
    states = np.asarray(row["states"])
    if (states.shape != (len(ticks), 4) or not np.all(np.isfinite(states))
            or not len(ticks) or abs(ticks[0] - first) > 1e-9 or ticks[-1] != reached
            or not np.all(np.diff(ticks) > 0) or not np.array_equal(states[0], initial)
            or not np.array_equal(states[-1], row["final_state_SI"]) or not first <= reached <= last + 1e-9):
        raise ValueError("Invalid native P4 trajectory checkpoint")
    if status == "retained" and (row["escape_RF"] is not None or reached != last):
        raise ValueError("Native P4 checkpoint missed the segment horizon")
    if status == "escaped" and (row["escape_RF"] != reached
                                or abs(np.linalg.norm(states[-1, :2]) - RADIUS) > 1e-8):
        raise ValueError("Invalid native P4 aperture event")
    if status not in ("retained", "escaped", "numerical_error"):
        raise ValueError("Invalid native P4 checkpoint status")


def run_case(directory, case, segment):
    folder = directory / "checkpoints" / case["job_id"]
    folder.mkdir(parents=True, exist_ok=True)
    initial = np.array([case[name] for name in STATE])
    summary = dict(nfev=0, checked_steps=0, crossings=0, max_zero_time_run=0, wall_s=0.,
                   sampled_max_radius_m=float(np.linalg.norm(initial[:2])))
    for first in range(0, case["horizon_RF"], segment):
        last = min(first + segment, case["horizon_RF"])
        path = folder / f"rf_{last:06d}.json"
        if path.exists():
            result = read_json(path)
        else:
            start = time.perf_counter()
            try:
                result = _runner.segment(initial, CHARGE * case["voltage_V"] / MASS,
                                         FREQ, case["phase_rad"], first, last, **SOLVER)
                result["status"] = "retained" if result["escape_RF"] is None else "escaped"
            except EscapeAuditError as exc:
                result = dict(exc.diagnostic, status="numerical_error", error=str(exc))
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                result = dict(status="numerical_error", error=f"{type(exc).__name__}: {exc}",
                              reached_RF=first, escape_RF=None, final_state_SI=initial.tolist())
            result.update(segment_RF=[first, last], initial_state_SI=initial.tolist(),
                          wall_s=time.perf_counter() - start)
            check_segment(result, initial, first, last)
            write_json(path, result)
        check_segment(result, initial, first, last)
        for name in ("nfev", "checked_steps", "crossings", "wall_s"):
            summary[name] += result.get(name, 0)
        for name in ("max_zero_time_run", "sampled_max_radius_m"):
            summary[name] = max(summary[name], result.get(name, 0))
        initial = np.asarray(result["final_state_SI"])
        if result["status"] != "retained":
            break
    return dict(**case, **summary, status=result["status"], escape_RF=result["escape_RF"],
                reached_RF=result["reached_RF"], final_state_SI=result["final_state_SI"],
                trajectory_path=folder.relative_to(directory).as_posix(), error=result.get("error"))


def run_checks(directory, candidates, workers, *, horizon=1000, segment=25, dry_run=False):
    """Run selected states on h35/h50, preserving old terminal exits and segment state."""
    if min(workers, horizon, segment) < 1:
        raise ValueError("workers, horizon and segment must be positive")
    directory = Path(directory)
    cases = case_rows(candidates, horizon)
    assets = preflight()
    modules = ("v11_native_sampling.py", "stability_rf_p4_v2.py", "stability_rf_p4.py",
               "direct_fem.py", "direct_fem_dynamics.py", "audited_rf.py", "gmsh_mesh.py")
    settings = dict(representation="P4", mesh_um=list(MESHES), horizon_RF=horizon,
                    segment_RF=segment, frequency_Hz=FREQ, mass_kg=MASS, charge_C=CHARGE,
                    aperture_m=RADIUS, trajectory_samples_per_RF=1,
                    solver={"method": "DOP853 (audited element crossings)", **SOLVER},
                    cases=cases, **assets,
                    kernel_sha256={name: sha(ROOT / "src/three_phase_trap" / name) for name in modules})
    if dry_run:
        return settings
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    with run_lock(checkpoints):
        config = directory / "config.json"
        if config.exists():
            if read_json(config) != settings:
                raise ValueError(f"Frozen native V11 case configuration changed: {config}")
        else:
            write_json(config, settings)
        write_table(directory / "initial_states.csv", CASE_COLUMNS, cases)
        results = {}
        pending = {}
        for case in cases:
            checkpoint = directory / "checkpoints" / case["job_id"]
            terminal = checkpoint / "result.json"
            if terminal.exists():
                results[case["job_id"]] = read_json(terminal)
                continue
            archived = reuse_exit(case)
            if archived is not None:
                results[case["job_id"]] = archived
                continue
            pending.setdefault((case["mesh_um"], case["model"]), []).append(case)
        for (mesh, model), jobs in pending.items():
            # A process holds one native field for its lifetime.
            print(f"P4 h{mesh} {model}: {len(jobs)} trajectories", flush=True)
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=init_worker,
                                     initargs=(mesh, model)) as pool:
                futures = [(case, pool.submit(run_case, directory, case, segment)) for case in jobs]
                for case, future in futures:
                    row = future.result()
                    results[case["job_id"]] = row
                    write_json(directory / "checkpoints" / case["job_id"] / "result.json", row)
        ordered = [results[case["job_id"]] for case in cases]
        write_table(directory / "results.csv", RESULT_COLUMNS, ordered)
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="?", type=Path, help="CSV of selected initial states")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--segment", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2))
        return
    if args.candidates is None:
        parser.error("candidates CSV is required unless --preflight is given")
    with args.candidates.open(encoding="utf-8-sig", newline="") as stream:
        results = run_checks(args.output, list(csv.DictReader(stream)), args.workers,
                             horizon=args.horizon, segment=args.segment, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps({"completed": len(results), "directory": str(args.output)}))


if __name__ == "__main__":
    main()
