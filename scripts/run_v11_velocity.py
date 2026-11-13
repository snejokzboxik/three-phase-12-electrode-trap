"""Checkpointed physical-vx/vy map for specified near-center launch positions.

Pilot: python scripts/run_v11_velocity.py --grid 5 --limit 15 --horizon 100
Main:  python scripts/run_v11_velocity.py --grid 9 --horizon 1000 --workers 4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_v11_xy import (OUT, CHARGE, FREQ, MASS, RADIUS, fields,
                        init_worker, write_json, _fields)
from three_phase_trap.pilot_dynamics import integrate_rf


def velocity_job(job):
    # Import module state after the process initializer has populated it.
    import run_v11_xy as xy
    model, state_tuple, phase, horizon, segment, path_text = job
    path = Path(path_text)
    path.mkdir(parents=True, exist_ok=True)
    field, voltage = xy._fields[model]
    initial = np.asarray(state_tuple, float)
    prior = None
    for stop in range(segment, horizon+1, segment):
        target = path / f"rf_{stop:06d}.json"
        if target.exists():
            prior = json.loads(target.read_text(encoding="utf-8"))
            if prior["status"] != "retained":
                return prior
            continue
        t0 = 0 if prior is None else prior["reached_RF"]
        start_state = initial if prior is None else np.asarray(prior["final_state_SI"])
        try:
            result = integrate_rf(field, start_state, CHARGE*voltage, MASS, FREQ, phase,
                                  t0/FREQ, stop/FREQ, output_samples_per_period=1,
                                  max_steps_per_period=50, radius=RADIUS)
            escape = None if result.exit_time is None else result.exit_time*FREQ
            row = {"model": model, "x0_m": initial[0], "y0_m": initial[1],
                   "vx0_m_s": initial[2], "vy0_m_s": initial[3],
                   "E0_eV": float(MASS*(initial[2]**2+initial[3]**2)/(2*CHARGE)),
                   "beta_deg": float(math.degrees(math.atan2(initial[3], initial[2]))),
                   "phase_rad": phase, "voltage_V": voltage,
                   "status": "retained" if escape is None else "escaped",
                   "escape_RF": escape, "reached_RF": stop if escape is None else escape,
                   "final_state_SI": result.states[-1].tolist(),
                   "sampled_max_radius_m": max(float(np.linalg.norm(result.states[:, :2], axis=1).max()),
                                                   0. if prior is None else prior["sampled_max_radius_m"]),
                   "nfev_segment": result.nfev, "wall_s_segment": result.wall_seconds}
        except Exception as exc:
            row = {"model": model, "initial_state_SI": initial.tolist(),
                   "status": "numerical_error", "error": f"{type(exc).__name__}: {exc}", "reached_RF": t0}
        write_json(target, row)
        prior = row
        if row["status"] != "retained":
            return row
    return prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--speed-max", type=float, default=100.)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--segment", type=int, default=100)
    p.add_argument("--phase", type=float, default=0.)
    p.add_argument("--mesh", type=int, choices=(35, 50), default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.grid < 3 or args.speed_max <= 0 or args.horizon % args.segment:
        p.error("Invalid grid, speed range or horizon/segment")
    _, hashes = fields(args.mesh)
    voltage = {"A": 250., "B": 250., "C": 250.}
    config = {"grid": args.grid, "speed_max_m_s": args.speed_max,
              "origins_m": [[0., 0.], [.0005, 0.]], "phase_deg": args.phase,
              "mesh_um": args.mesh, "horizon_RF": args.horizon, "segment_RF": args.segment,
              "voltage_V": voltage, "field_sha256": hashes,
              "mass_kg": MASS, "charge_C": CHARGE, "frequency_Hz": FREQ,
              "aperture_m": RADIUS, "method": "DOP853", "rtol": 2e-9,
              "atol_SI": [2e-13, 2e-13, 2e-11, 2e-11], "max_step_RF": .02}
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder = OUT / "velocity" / tag
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", config)
    speeds = np.linspace(-args.speed_max, args.speed_max, args.grid)
    jobs = []
    for origin_id, (x, y) in enumerate(config["origins_m"]):
        for iy, vy in enumerate(speeds):
            for ix, vx in enumerate(speeds):
                for model in "ABC":
                    jobs.append((model, (x, y, float(vx), float(vy)), math.radians(args.phase),
                                 args.horizon, args.segment,
                                 str(folder / "checkpoints" / f"o{origin_id}_{iy:03d}_{ix:03d}_{model}")))
    if args.limit:
        jobs = jobs[:args.limit]
    with (folder / "initial_states.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "x0_m", "y0_m", "vx0_m_s", "vy0_m_s", "phase_rad"))
        writer.writerows((model, *state, phase) for model, state, phase, *_ in jobs)
    print(json.dumps({"jobs": len(jobs), "folder": str(folder)}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.mesh, voltage)) as pool:
        results = list(pool.map(velocity_job, jobs))
    with (folder / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in results))))
        writer.writeheader(); writer.writerows(results)
    print(json.dumps({"status_counts": {status: sum(row["status"] == status for row in results)
                                        for status in ("retained", "escaped", "numerical_error")}}))


if __name__ == "__main__":
    main()
