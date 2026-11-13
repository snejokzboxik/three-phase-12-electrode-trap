"""Run and checkpoint a small direct-P4 FEM trajectory set for V10.

Example (PowerShell):
  $env:PYTHONPATH='src;outputs/octupole_comparison_home/physical_verification_v10'
  python outputs/octupole_comparison_home/physical_verification_v10/run_native_p4.py `
    --mesh 50 --launches deep_az0_dplus early_az22p5_dplus --horizon-rf 10

No new mesh or FEM solve is performed. One field object is loaded per process.
Completed trajectories are individually checkpointed before the next begins.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from native_aperture_integrator import integrate_native_aperture

OUT = HERE / "data" / "native_p4"
CONFIG = json.loads((HERE / "config_v10.json").read_text(encoding="utf-8"))
ION = CONFIG["ion"]
E = 1.602176634e-19
M = float(ION["mass_kg"])
Q = float(ION["charge_C"])
F = float(CONFIG["electrode_drive"]["frequency_Hz"])
APER = float(CONFIG["geometry"]["aperture_radius_m"])


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def source_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_field(mesh_um):
    from three_phase_trap.direct_fem import DirectFEMRF, SectorSolution
    from three_phase_trap.gmsh_mesh import load_gmsh_mesh
    mesh_path = ROOT / f"outputs/meshes/gmsh_{mesh_um}um/trap.msh"
    p4_dir = ROOT / f"outputs/abc_validation/direct_fem_validation/v2/h{mesh_um}_P4"
    solution_path = p4_dir / "sector_solution.npz"
    complete = json.loads((p4_dir / "complete.json").read_text(encoding="utf-8"))
    expected_mesh = complete["signature"]["full_mesh_sha256"]
    if source_sha(mesh_path) != expected_mesh:
        raise RuntimeError(f"{mesh_um} um mesh SHA mismatch")
    expected_p4 = complete["artifacts"]["sector_solution.npz"]
    if source_sha(solution_path) != expected_p4:
        raise RuntimeError(f"{mesh_um} um P4 SHA mismatch")
    sector = SectorSolution.load(solution_path)
    trap = load_gmsh_mesh(mesh_path)
    field = DirectFEMRF(sector, trap, amplitude=250.)
    return field, {
        "mesh_um": mesh_um,
        "mesh_sha256": source_sha(mesh_path),
        "sector_solution_sha256": source_sha(solution_path),
        "element_degree": sector.degree,
        "full_mesh_triangles": len(field.triangles),
        "mapping_error_m": field.mapping_error_m,
        "loaded_saved_p4_solution": True,
        "new_fem_solve": False,
    }


def launch_state(entry):
    az = math.radians(float(entry["azimuth_deg"]))
    if "initial_energy_eV" in entry:
        speed = math.sqrt(2.0*float(entry["initial_energy_eV"])*E/M)
    else:
        # Preserve the exact V7 launch speed for the original matched starts.
        speed = float(ION["initial_speed_m_s"])
    direction = int(entry["direction"])
    radius = float(entry.get("initial_radius_m", CONFIG["ion"]["initial_radius_m"]))
    return np.array([radius*math.cos(az), radius*math.sin(az),
                     -direction*speed*math.sin(az), direction*speed*math.cos(az)])


def save_result(job, result, field_meta):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{job['tag']}" if job.get("tag") else ""
    token = f"{job['mesh_um']}um_{job['model']}_{job['launch_label']}_{job['phase_deg']:g}deg_{job['horizon_rf']}rf{tag}"
    data_path = OUT / f"{token}.npz"
    meta_path = OUT / f"{token}.json"
    tmp = data_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, time_s=result.times, states=result.states)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, data_path)
    meta = {**job, "field": field_meta, "status": result.status,
            "escape_time_s": result.escape_time_s,
            "escape_time_rf": None if result.escape_time_s is None else result.escape_time_s*F,
            "initial_state_SI": launch_state(job["launch"] if "launch" in job else job).tolist(),
            "numerics": result.diagnostics,
            "timeseries_file": data_path.name}
    atomic_json(meta_path, meta)
    return meta


def save_checkpoint(path, time_s, states):
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, time_s=np.asarray(time_s), states=np.asarray(states))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_index():
    rows = []
    for path in sorted(OUT.glob("*.json")):
        if path.name == "index.csv" or ".checkpoint." in path.name or path.name.endswith(".tmp"):
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        # Parallel runs may rebuild the summary while another process is
        # checkpointing. Only completed trajectory metadata are index rows.
        if not isinstance(obj, dict) or "mesh_um" not in obj or "numerics" not in obj:
            continue
        rows.append({
            "mesh_um": obj["mesh_um"], "model": obj["model"],
            "launch_label": obj["launch_label"], "azimuth_deg": obj["azimuth_deg"],
            "direction": obj["direction"], "phase_deg": obj["phase_deg"],
            "voltage_V": obj["voltage_V"], "frequency_Hz": obj["frequency_Hz"],
            "initial_energy_eV": obj["initial_energy_eV"],
            "initial_radius_m": obj["initial_radius_m"], "status": obj["status"],
            "escape_time_rf": obj["escape_time_rf"],
            "max_sampled_radius_m": obj["numerics"]["maximum_sampled_radius_m"],
            "max_dense_probe_radius_m": obj["numerics"].get("maximum_three_point_dense_radius_m"),
            "rms_radius_m": obj["numerics"].get("rms_radius_m"),
            "horizon_rf": obj["horizon_rf"],
            "rtol": obj["numerics"]["rtol"],
            "steps_per_rf_period": obj["numerics"]["steps_per_rf_period"],
            "wall_seconds": obj["numerics"]["wall_seconds"],
            "timeseries_file": obj["timeseries_file"],
        })
    if rows:
        path = OUT / "index.csv"
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, choices=(35, 50), default=50)
    ap.add_argument("--launches", nargs="+", default=("deep_az0_dplus", "early_az22p5_dplus"))
    ap.add_argument("--models", nargs="+", choices=("rotating", "standing"), default=("rotating", "standing"))
    ap.add_argument("--phase-deg", type=float, default=0.0)
    ap.add_argument("--horizon-rf", type=int, default=10)
    ap.add_argument("--steps-per-rf", type=int, default=200)
    ap.add_argument("--rtol", type=float, default=2e-9)
    ap.add_argument("--atol-position", type=float, default=2e-13,
                    help="absolute coordinate tolerance in metres")
    ap.add_argument("--atol-velocity", type=float, default=2e-11,
                    help="absolute velocity tolerance in metres per second")
    ap.add_argument("--samples-per-rf", type=int, default=8)
    ap.add_argument("--chunk-rf", type=int, default=50,
                    help="checkpoint interval; restarted at an exact RF boundary")
    ap.add_argument("--defer-index", action="store_true",
                    help="skip rebuilding the shared summary CSV until a batch ends")
    ap.add_argument("--tag", default="", help="suffix for a distinct numerical-protocol rerun")
    args = ap.parse_args()
    launches = {v["label"]: v for v in CONFIG["initial_conditions_v7"]}
    launches.update({v["label"]: v for v in CONFIG.get("parameter_robustness", [])})
    for label in args.launches:
        if label not in launches:
            raise ValueError(f"Unknown V7 launch label: {label}")
    field, field_meta = load_field(args.mesh)
    base_gx, base_gy = field.gx, field.gy
    print(json.dumps({"field_loaded": field_meta, "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2), flush=True)
    for label in args.launches:
        entry = launches[label]
        for model in args.models:
            tag = f"_{args.tag}" if args.tag else ""
            token = f"{args.mesh}um_{model}_{label}_{args.phase_deg:g}deg_{args.horizon_rf}rf{tag}"
            if (OUT/f"{token}.json").exists():
                saved = json.loads((OUT/f"{token}.json").read_text(encoding="utf-8"))
                expected_atol = [args.atol_position, args.atol_position,
                                 args.atol_velocity, args.atol_velocity]
                if (saved.get("rtol") != args.rtol
                        or saved.get("steps_per_rf_period") != args.steps_per_rf
                        or saved.get("atol", saved.get("numerics", {}).get("atol")) != expected_atol
                        or saved.get("samples_per_rf") != args.samples_per_rf):
                    raise RuntimeError(f"Existing result {token} uses another numerical protocol; choose a new --tag")
                print(f"already checkpointed: {token}", flush=True)
                continue
            voltage = float(CONFIG["electrode_drive"]["rotating_voltage_V"] if model == "rotating"
                            else CONFIG["electrode_drive"]["standing_voltage_V"])
            if model == "rotating":
                field.gx = base_gx*(voltage/250.0)
                field.gy = base_gy*(voltage/250.0)
            else:
                field.gx = base_gx.real * (voltage/250.0)
                field.gy = base_gy.real * (voltage/250.0)
            job = {"study": "V10 direct native P4 aperture pilot", "mesh_um": args.mesh,
                   "model": model, "launch_label": label, "azimuth_deg": float(entry["azimuth_deg"]),
                   "direction": int(entry["direction"]), "phase_deg": args.phase_deg,
                   "voltage_V": voltage, "frequency_Hz": F,
                   "mass_kg": M, "charge_C": Q,
                   "initial_energy_eV": float(entry.get("initial_energy_eV", ION["initial_kinetic_energy_eV"])),
                   "initial_radius_m": float(entry.get("initial_radius_m", ION["initial_radius_m"])),
                   "initial_state_SI": launch_state(entry).tolist(),
                   "horizon_rf": args.horizon_rf, "aperture_radius_m": APER,
                   "normalization": "matched leading m=4 pseudopotential only",
                   "right_censoring": "survived to requested finite horizon; not permanent stability",
                   "rtol": args.rtol, "steps_per_rf_period": args.steps_per_rf,
                   "atol": [args.atol_position, args.atol_position,
                            args.atol_velocity, args.atol_velocity],
                   "samples_per_rf": args.samples_per_rf,
                   "integrator_source_sha256": source_sha(HERE / "native_aperture_integrator.py")}
            if args.tag:
                job["tag"] = args.tag
            print(f"starting {token}", flush=True)
            checkpoint_npz = OUT/f"{token}.checkpoint.npz"
            checkpoint_json = OUT/f"{token}.checkpoint.json"
            signature = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
            if checkpoint_npz.exists() and checkpoint_json.exists():
                saved = json.loads(checkpoint_json.read_text(encoding="utf-8"))
                if saved.get("job_signature") != signature:
                    raise RuntimeError(f"Checkpoint parameters do not match job {token}")
                with np.load(checkpoint_npz, allow_pickle=False) as z:
                    all_t = list(z["time_s"])
                    all_y = list(z["states"])
                current_rf = int(saved["completed_rf"])
                wall_total = float(saved.get("wall_seconds", 0.0))
                steps_total = int(saved.get("checked_steps", 0))
                crossings_total = int(saved.get("element_crossings", 0))
                nfev_total = int(saved.get("nfev", 0))
                min_bound = float(saved.get("minimum_bernstein_membership_bound", math.inf))
            else:
                all_t, all_y = [0.0], [launch_state(entry)]
                current_rf = 0
                wall_total = 0.0; steps_total = 0; crossings_total = 0; nfev_total = 0
                min_bound = math.inf
            status = "retained_to_horizon"
            escape_time_s = None
            last_diag = {}
            while current_rf < args.horizon_rf:
                stop_rf = min(args.horizon_rf, current_rf+max(1, args.chunk_rf))
                result = integrate_native_aperture(
                    field, all_y[-1], Q, M, F, math.radians(args.phase_deg),
                    stop_rf/F, start_time=current_rf/F,
                    steps_per_period=args.steps_per_rf, rtol=args.rtol,
                    atol=(args.atol_position, args.atol_position,
                          args.atol_velocity, args.atol_velocity),
                    samples_per_period=args.samples_per_rf, aperture=APER,
                    progress=lambda d: print(json.dumps({"job": token, **d}), flush=True))
                chunk_t = result.times
                chunk_y = result.states
                all_t.extend(chunk_t[1:].tolist()); all_y.extend(chunk_y[1:].tolist())
                wall_total += result.diagnostics["wall_seconds"]
                steps_total += result.diagnostics["checked_steps"]
                crossings_total += result.diagnostics["element_crossings"]
                nfev_total += result.diagnostics["nfev"]
                min_bound = min(min_bound, result.diagnostics["minimum_bernstein_membership_bound"])
                last_diag = result.diagnostics
                if result.status == "escaped":
                    status = "escaped"
                    escape_time_s = result.escape_time_s
                    current_rf = stop_rf
                else:
                    current_rf = stop_rf
                save_checkpoint(checkpoint_npz, all_t, all_y)
                atomic_json(checkpoint_json, {
                    "job_signature": signature, "completed_rf": current_rf,
                    "status": status, "escape_time_s": escape_time_s,
                    "wall_seconds": wall_total, "checked_steps": steps_total,
                    "element_crossings": crossings_total, "nfev": nfev_total,
                    "minimum_bernstein_membership_bound": min_bound,
                })
                if status == "escaped":
                    break
            from native_aperture_integrator import ApertureResult
            diagnostics = dict(last_diag)
            diagnostics.update({"status": status, "wall_seconds": wall_total,
                                "checked_steps": steps_total, "element_crossings": crossings_total,
                                "nfev": nfev_total,
                                "minimum_bernstein_membership_bound": min_bound,
                                "checkpoint_chunk_rf": args.chunk_rf,
                                "segmented_restart_protocol": True})
            result = ApertureResult(np.asarray(all_t), np.asarray(all_y), status,
                                    escape_time_s, diagnostics)
            meta = save_result(job, result, field_meta)
            checkpoint_npz.unlink(missing_ok=True)
            checkpoint_json.unlink(missing_ok=True)
            if not args.defer_index:
                write_index()
            print(json.dumps({"completed": token, "status": meta["status"],
                              "escape_time_rf": meta["escape_time_rf"],
                              "wall_seconds": meta["numerics"]["wall_seconds"]}), flush=True)
            field.gx, field.gy = base_gx, base_gy
    if not args.defer_index:
        write_index()


if __name__ == "__main__":
    main()
