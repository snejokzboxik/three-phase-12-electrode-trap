"""Matched, checkpointed RF trajectories on validated nominal/displaced P4 fields."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

import field_analysis as fa
import solve_displaced_p4 as pilot

ROOT = fa.ROOT
OUT = fa.OUT
FREQUENCY = 100_000.0
OMEGA = 2 * math.pi * FREQUENCY
PERIOD = 1.0 / FREQUENCY
VOLTAGE = 250.0
APERTURE = 0.008
MZ = 100_000.0
MASS = MZ * fa.AMU
QM = fa.E_CHARGE / MASS
HORIZON_PERIODS = 1000
SEGMENT_PERIODS = 100
SAMPLES_PER_RF = 8


def initial_state(energy_eV: float = 2.0, direction: int = 1) -> tuple[np.ndarray, dict]:
    radius_m = 1e-3
    azimuth = math.pi / 8
    phase = 0.0
    direction = int(direction)
    if direction not in (-1, 1):
        raise ValueError("tangential direction must be -1 or +1")
    speed = math.sqrt(2 * energy_eV * fa.E_CHARGE / MASS)
    state = np.array([radius_m * math.cos(azimuth), radius_m * math.sin(azimuth), -direction * speed * math.sin(azimuth), direction * speed * math.cos(azimuth)], dtype=float)
    metadata = {"mz_Th": MZ, "charge_state": 1, "energy_eV": energy_eV, "r0_m": radius_m, "azimuth_rad": azimuth, "rf_phase_rad": phase, "velocity_kind": "tangential", "tangential_sign": direction, "frequency_Hz": FREQUENCY, "voltage_V": VOLTAGE, "aperture_radius_m": APERTURE, "horizon_RF_periods": HORIZON_PERIODS, "matched_across_geometry_and_model": True}
    return state, metadata


def load_geometry(name: str):
    if name == "nominal":
        return fa.nominal_field()
    if name == "radial_out_10um":
        directory = OUT / "calculations" / "radial_out_10um"
        return fa.field_from_coefficients(directory / "displaced_h35.msh", directory / "p4_coefficients.npz")
    if name.endswith("_h50"):
        pattern = name[:-4]
        directory = OUT / "calculations" / f"{pattern}_h50"
        return fa.field_from_coefficients(directory / "displaced_h50.msh", directory / "p4_coefficients.npz")
    directory = OUT / "calculations" / name
    return fa.field_from_coefficients(directory / "displaced_h35.msh", directory / "p4_coefficients.npz")


def trajectory_key(geometry: str, model: str, tight: bool, energy_eV: float = 2.0, direction: int = 1) -> str:
    base = f"{geometry}_{model}_{'tight' if tight else 'standard'}"
    suffix = "" if energy_eV == 2.0 and direction == 1 else f"_E{energy_eV:g}eV_tangent_{'plus' if direction == 1 else 'minus'}_v2"
    return base + suffix


def write_segment_csv(path: Path, times: np.ndarray, states: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "rf_period", "x_mm", "y_mm", "r_mm", "vx_m_s", "vy_m_s"])
        for t, state in zip(times, states):
            writer.writerow([float(t), float(t * FREQUENCY), float(state[0] * 1e3), float(state[1] * 1e3), float(math.hypot(state[0], state[1]) * 1e3), float(state[2]), float(state[3])])


def integrate_one(field, geometry: str, model: str, tight: bool, runtime: dict, energy_eV: float = 2.0, direction: int = 1) -> dict:
    pilot.require_time(runtime, 300.0, f"starting matched RF trajectory {geometry}/{model}")
    key = trajectory_key(geometry, model, tight, energy_eV, direction)
    trace_dir = OUT / "trajectories" / key
    trace_dir.mkdir(parents=True, exist_ok=True)
    progress_path = trace_dir / "progress.json"
    checkpoint_path = OUT / "runtime" / "trajectory_checkpoints.jsonl"
    initial, initial_meta = initial_state(energy_eV, direction)
    rtol = 2e-11 if tight else 2e-9
    max_step_div = 100 if tight else 50
    atol = (2e-14, 2e-14, 2e-12, 2e-12) if tight else (2e-13, 2e-13, 2e-11, 2e-11)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            return progress
        state = np.asarray(progress["state"], dtype=float)
        t0 = float(progress["time_s"])
        segment_index = int(progress["next_segment"])
        radii = list(progress.get("sampled_radii_m", []))
        nfev = int(progress.get("nfev", 0))
    else:
        state = initial.copy()
        t0 = 0.0
        segment_index = 0
        radii = [float(np.hypot(state[0], state[1]))]
        nfev = 0
        progress = {"key": key, "geometry": geometry, "model": model, "tight": tight, "initial_state": initial.tolist(), "initial_conditions": initial_meta, "time_s": 0.0, "state": state.tolist(), "next_segment": 0, "sampled_radii_m": radii, "nfev": 0, "complete": False}

    if model == "rotating":
        use_cos, use_sin = True, True
    elif model == "standing":
        use_cos, use_sin = False, True
    else:
        raise ValueError(model)

    def rhs(t, y):
        _, ec, es = field.sample_one(float(y[0]), float(y[1]))
        phase = OMEGA * t + initial_meta["rf_phase_rad"]
        rf = VOLTAGE * ((ec * math.cos(phase) if use_cos else 0.0) + es * math.sin(phase))
        acceleration = QM * rf
        return (y[2], y[3], acceleration[0], acceleration[1])

    def aperture_event(t, y):
        return APERTURE - math.hypot(y[0], y[1])
    aperture_event.terminal = True
    aperture_event.direction = -1

    horizon_t = HORIZON_PERIODS * PERIOD
    while t0 < horizon_t - 1e-18:
        pilot.require_time(runtime, 180.0, f"next 100-RF segment of {key}")
        t1 = min(horizon_t, t0 + SEGMENT_PERIODS * PERIOD)
        sample_count = int(round((t1 - t0) * FREQUENCY * SAMPLES_PER_RF))
        t_eval = np.linspace(t0, t1, sample_count + 1)
        started = time.perf_counter()
        solution = solve_ivp(rhs, (t0, t1), state, method="DOP853", rtol=rtol, atol=atol, max_step=1.0 / (FREQUENCY * max_step_div), t_eval=t_eval, events=aperture_event)
        if not solution.success:
            raise RuntimeError(solution.message)
        times = solution.t.copy()
        states = solution.y.T.copy()
        escaped = bool(len(solution.t_events[0]))
        escape_time = float(solution.t_events[0][0]) if escaped else None
        if escaped:
            times = np.r_[times, escape_time]
            states = np.vstack((states, solution.y_events[0][0]))
        radii_segment = np.hypot(states[:, 0], states[:, 1])
        radii.extend(map(float, radii_segment[1:]))
        nfev += int(solution.nfev)
        segment_path = trace_dir / f"segment_{segment_index:03d}.csv"
        write_segment_csv(segment_path, times, states)
        final_state = states[-1]
        t0 = float(times[-1])
        state = final_state
        segment_index += 1
        complete = escaped or t0 >= horizon_t - 1e-18
        progress = {"key": key, "geometry": geometry, "model": model, "tight": tight, "initial_state": initial.tolist(), "initial_conditions": initial_meta, "rtol": rtol, "atol": list(atol), "max_step_RF_fraction": 1.0 / max_step_div, "time_s": t0, "rf_period": t0 * FREQUENCY, "state": state.tolist(), "next_segment": segment_index, "sampled_radii_m": radii, "nfev": nfev, "segment_wall_seconds": float(time.perf_counter() - started), "complete": complete, "status": ("escaped" if escaped else "retained_through_horizon" if complete else "incomplete_checkpoint"), "escape_time_s": escape_time, "escape_time_RF": (escape_time * FREQUENCY if escape_time is not None else None), "right_censor_time_RF": (None if escaped else HORIZON_PERIODS if complete else None), "sampling_rate_per_RF": SAMPLES_PER_RF}
        temp = progress_path.with_suffix(".tmp")
        temp.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, progress_path)
        with checkpoint_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"time_utc": pilot.iso(pilot.utc_now()), "key": key, "segment": segment_index, "status": progress["status"], "rf_period": progress["rf_period"], "segment_wall_seconds": progress["segment_wall_seconds"]}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"[{key}] checkpoint at {progress['rf_period']:.0f} RF; status={progress['status']}; nfev={nfev}", flush=True)
        if complete:
            break

    data_r = np.asarray(progress["sampled_radii_m"], dtype=float)
    progress.update({"max_radius_m": float(np.max(data_r)), "rms_radius_m_sampled": float(np.sqrt(np.mean(data_r * data_r))), "minimum_radius_m_sampled": float(np.min(data_r)), "n_radius_samples": int(len(data_r)), "interpretation": "a 1000-RF survivor is finite-time right-censored; it is not labelled permanently stable"})
    temp = progress_path.with_suffix(".tmp")
    temp.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, progress_path)
    return progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", action="append", required=True, help="nominal, radial_out_10um, or another solved pattern")
    parser.add_argument("--model", action="append", choices=("standing", "rotating"), help="repeat to select models; default is both matched models")
    parser.add_argument("--energy-eV", type=float, default=2.0)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--tight-standing", action="store_true", help="repeat the nominal standing case at tighter DOP853 tolerances")
    args = parser.parse_args()
    runtime = pilot.start_or_resume_budget()
    pilot.require_time(runtime, 600.0, "matched RF trajectory validation")
    summary_rows = []
    models = args.model or ["standing", "rotating"]
    for geometry in args.geometry:
        pilot.require_time(runtime, 600.0, f"load validated field for {geometry}")
        field, meta = load_geometry(geometry)
        for model in models:
            result = integrate_one(field, geometry, model, False, runtime, args.energy_eV, args.direction)
            result["field_representation"] = meta
            pilot.atomic_json(OUT / "trajectories" / trajectory_key(geometry, model, False, args.energy_eV, args.direction) / "summary.json", result)
            summary_rows.append(result)
        if geometry == "nominal" and args.tight_standing and "standing" in models:
            result = integrate_one(field, geometry, "standing", True, runtime, args.energy_eV, args.direction)
            result["field_representation"] = meta
            pilot.atomic_json(OUT / "trajectories" / trajectory_key(geometry, "standing", True, args.energy_eV, args.direction) / "summary.json", result)
            summary_rows.append(result)
    processed = OUT / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    summary_path = processed / "matched_rf_trajectory_summary.json"
    all_rows = []
    for summary_file in sorted((OUT / "trajectories").glob("*/summary.json")):
        item = json.loads(summary_file.read_text(encoding="utf-8"))
        if not item.get("excluded_from_combined_summary", False):
            all_rows.append(item)
    pilot.atomic_json(summary_path, {"protocol": "identical physical start state and 250 V peak RF voltage for all geometries/models within each initial-condition batch", "initial_conditions_by_batch": {"E2_eV_tangent_positive": initial_state(2.0, 1)[1], "CLI_batch": initial_state(args.energy_eV, args.direction)[1]}, "trajectories": all_rows, "retention": "survival through 1000 RF periods, not permanent stability", "escape_time": "first outward r=8 mm crossing; survivors right-censored at 1000 RF"})
    pilot.append_checkpoint({"time_utc": pilot.iso(pilot.utc_now()), "stage": "matched_rf_trajectory_batch_written", "trajectory_count": len(all_rows), "complete_count": sum(r["status"] in ("escaped", "retained_through_horizon") for r in all_rows)})
    print(json.dumps({"summary": str(summary_path.relative_to(ROOT)), "trajectories": len(all_rows), "statuses": {r["key"]: r["status"] for r in all_rows}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
