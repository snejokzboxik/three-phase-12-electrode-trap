"""Checkpointed zero-velocity XY map. Example:

python scripts/run_v11_xy.py --grid 21 --phase 0 --protocol equal --horizon 1000 --workers 4

Use --limit 3 --horizon 10 for a speed and memory pilot first. Jobs are
independent; a stopped run resumes from its last completed RF segment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.pilot_dynamics import PolynomialRF, integrate_rf
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction

OUT = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11"
FREQ = 100_000.
MASS = 100_000 * 1.66053906660e-27
CHARGE = 1.602176634e-19
RADIUS = .008
_fields = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temp, path)


def fields(mesh):
    apath = ROOT / f"outputs/abc_validation/direct_fem_validation/v2/h{mesh}_P4/diagnostic_N48.json"
    cpath = OUT / f"fields/h{mesh}_P4_C/reconstruction_N48_C.json"
    a = RFReconstruction.load(apath).unit_quadratures
    c = RFReconstruction.load(cpath).unit_quadratures
    # The standing convention of V5/V6 is the sine quadrature of the same
    # three-phase FEM solution, with cosine quadrature removed.
    b = HarmonicQuadratures(np.zeros_like(a.cosine), a.sine.copy(), a.reference_radius)
    return (a, b, c), {"A": sha(apath), "B": sha(apath), "C": sha(cpath)}


def voltages(quadratures, protocol, reference_mm):
    if protocol == "equal":
        return {"A": 250., "B": 250., "C": 250.}
    angle = (np.arange(720)+.137)*2*np.pi/720
    points = reference_mm*1e-3*np.column_stack((np.cos(angle), np.sin(angle)))
    def strength(q):
        ec, es = q.field_quadratures(points)
        return float(np.mean(np.sum(ec*ec+es*es, axis=1)))
    sa, sc = strength(quadratures[0]), strength(quadratures[2])
    return {"A": 250., "B": 250.*math.sqrt(2.), "C": 250.*math.sqrt(sa/sc)}


def init_worker(mesh, voltage):
    global _fields
    q, _ = fields(mesh)
    _fields = {name: PolynomialRF(quad) for name, quad in zip("ABC", q)}
    _fields = {name: (_fields[name], voltage[name]) for name in "ABC"}


def run_job(job):
    model, x, y, phase, horizon, segment, path_text = job
    path = Path(path_text)
    path.mkdir(parents=True, exist_ok=True)
    field, voltage = _fields[model]
    initial = np.array([x, y, 0., 0.])
    prior = None
    started = time.perf_counter()
    for stop in range(segment, horizon+1, segment):
        checkpoint = path / f"rf_{stop:06d}.json"
        if checkpoint.exists():
            prior = json.loads(checkpoint.read_text(encoding="utf-8"))
            if prior["status"] != "retained":
                return prior
            continue
        t0 = 0 if prior is None else prior["reached_RF"]
        state = initial if prior is None else np.asarray(prior["final_state_SI"])
        try:
            result = integrate_rf(field, state, CHARGE*voltage, MASS, FREQ, phase,
                                  t0/FREQ, stop/FREQ, output_samples_per_period=1,
                                  max_steps_per_period=50, radius=RADIUS)
            escape = None if result.exit_time is None else result.exit_time*FREQ
            row = {"model": model, "x0_m": x, "y0_m": y, "vx0_m_s": 0., "vy0_m_s": 0.,
                   "phase_rad": phase, "voltage_V": voltage,
                   "status": "retained" if escape is None else "escaped",
                   "escape_RF": escape, "reached_RF": stop if escape is None else escape,
                   "final_state_SI": result.states[-1].tolist(),
                   "sampled_max_radius_m": max(float(np.linalg.norm(result.states[:, :2], axis=1).max()),
                                                   0. if prior is None else prior["sampled_max_radius_m"]),
                   "nfev_segment": result.nfev, "wall_s_segment": result.wall_seconds}
        except Exception as exc:
            row = {"model": model, "x0_m": x, "y0_m": y, "status": "numerical_error",
                   "error": f"{type(exc).__name__}: {exc}", "reached_RF": t0}
        write_json(checkpoint, row)
        prior = row
        if row["status"] != "retained":
            return row
    return prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=int, default=21)
    p.add_argument("--phase", type=float, default=0.)
    p.add_argument("--protocol", choices=("equal", "matched"), default="equal")
    p.add_argument("--reference-mm", type=float, default=4.)
    p.add_argument("--mesh", type=int, choices=(35, 50), default=50)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--segment", type=int, default=10)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.grid < 3 or args.horizon < 1 or args.segment < 1 or args.horizon % args.segment:
        p.error("grid >= 3; horizon must be positive and divisible by segment")
    q, hashes = fields(args.mesh)
    voltage = voltages(q, args.protocol, args.reference_mm)
    config = {"grid": args.grid, "phase_deg": args.phase, "protocol": args.protocol,
              "reference_radius_mm": args.reference_mm if args.protocol == "matched" else None,
              "mesh_um": args.mesh, "horizon_RF": args.horizon, "segment_RF": args.segment,
              "frequency_Hz": FREQ, "mass_kg": MASS, "charge_C": CHARGE,
              "aperture_m": RADIUS, "voltage_V": voltage, "field_sha256": hashes,
              "solver": {"method": "DOP853", "rtol": 2e-9,
                         "atol_SI": [2e-13, 2e-13, 2e-11, 2e-11], "max_step_RF": .02}}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    directory = OUT / "xy" / key
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Immutable calculation config changed")
    write_json(config_path, config)
    axis = np.linspace(-.0075, .0075, args.grid)
    jobs = []
    for iy, y in enumerate(axis):
        for ix, x in enumerate(axis):
            if math.hypot(x, y) >= .0075:
                continue
            for name in "ABC":
                jobs.append((name, float(x), float(y), math.radians(args.phase), args.horizon,
                             args.segment, str(directory / "checkpoints" / f"{iy:03d}_{ix:03d}_{name}")))
    if args.limit:
        jobs = jobs[:args.limit]
    with (directory / "initial_states.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "x0_m", "y0_m", "vx0_m_s", "vy0_m_s", "phase_rad"))
        writer.writerows((name, x, y, 0., 0., phase) for name, x, y, phase, *_ in jobs)
    print(json.dumps({"jobs": len(jobs), "output": str(directory), "voltages": voltage}), flush=True)
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.mesh, voltage)) as pool:
        for i, result in enumerate(pool.map(run_job, jobs), 1):
            results.append(result)
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} in {time.perf_counter()-started:.1f}s", flush=True)
    if results:
        with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in results))))
            writer.writeheader(); writer.writerows(results)


if __name__ == "__main__":
    main()
