"""User step 7 / detailed stage 6: restartable 3.25 and 10.25 slow-period pilots.

Use --stage probe for the evaluator/performance checks; default all computes
and analyzes the pilots. Completed fingerprinted checkpoints are reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.config import load_config, ParticleConfig
from three_phase_trap.fem_cache import load_fem_cache, fem_fingerprint
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.averaged_dynamics import AveragedRFModel, integrate_averaged
from three_phase_trap.octupole_model import OctupoleModel
from three_phase_trap.pilot_dynamics import AffineP2RF, PolynomialRF, RFRun, integrate_rf, coordinate_metrics
from three_phase_trap.dynamics import integrate_trajectory
from three_phase_trap.micromotion import PhasedFEMField

OUTPUT = ROOT/"outputs/abc_validation/pilot"
_objects = {}


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_models():
    protocol = json.loads((OUTPUT.parent/"protocol.json").read_text(encoding="utf-8"))
    initial = json.loads((OUTPUT.parent/"initial_conditions.json").read_text(encoding="utf-8"))
    p = protocol["pilot"]
    rf = RFReconstruction.load(OUTPUT.parent/"model_b/reconstruction.json", protocol["fem_fingerprint"])
    b = AveragedRFModel(rf, p["charge_C"], p["mass_kg"], p["U0_V"], p["frequency_Hz"])
    coefficient = json.loads((OUTPUT.parent/"model_c/coefficients.json").read_text(encoding="utf-8"))
    c = OctupoleModel(coefficient["C4_per_volt_per_m4"], coefficient["delta4_rad"], b.charge, b.mass, b.amplitude, b.frequency, rf.valid_radius)
    return protocol, initial, b, c


def checkpoint_paths(case):
    tag = f"{case['model']}_phase{case['phase_index']}_{case['level']}_{case['extent']}"
    return OUTPUT/(tag+".npz"), OUTPUT/(tag+".json")


def case_signature(case, settings):
    return hashlib.sha256(json.dumps({"case": case, "numerical_provenance": settings["numerical_provenance"],
                                     "controls": settings["controls"], "periods": settings["RF_periods"]}, sort_keys=True).encode()).hexdigest()


def valid_checkpoint(case, settings):
    arrays, metadata = checkpoint_paths(case)
    if not arrays.exists() or not metadata.exists():
        return False
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if meta["signature"] != case_signature(case, settings):
        raise ValueError(f"Existing checkpoint does not match numerical inputs: {metadata.name}")
    if meta["arrays_sha256"] != sha(arrays):
        raise ValueError(f"Corrupt checkpoint: {arrays.name}")
    return meta["status"] == "completed"


def worker(case):
    settings = json.loads((OUTPUT/"settings.json").read_text(encoding="utf-8"))
    protocol, initial, b, c = load_models()
    p = protocol["pilot"]
    controls = settings["controls"][case["level"]]
    frequency = b.frequency
    end_time = settings["RF_periods"][case["extent"]]/frequency
    start_time = 0.
    initial_state = np.array(initial["cases"][case["phase_index"]]["A_and_A_smooth_full_state"])
    phase = p["phase_rad"][case["phase_index"]]
    previous = None
    parent_sha = None
    if case["extent"] == "extended" and case["model"] in ("A", "A_smooth"):
        parent = {**case, "extent": "pilot"}
        if not valid_checkpoint(parent, settings):
            raise ValueError("Cannot extend an incomplete pilot")
        parent_arrays, parent_meta = checkpoint_paths(parent)
        with np.load(parent_arrays) as data:
            previous = (data["time_s"], data["states"])
        start_time, initial_state = float(previous[0][-1]), previous[1][-1]
        parent_sha = sha(parent_arrays)
    started = time.perf_counter()
    if case["model"] in ("A", "A_smooth"):
        if case["model"] == "A":
            if "raw" not in _objects:
                config = load_config(ROOT/"configs/gmsh_70um.toml")
                _, _, _, interp = load_fem_cache(config, ROOT)
                _objects["raw"] = AffineP2RF(interp, b.amplitude)
            field = _objects["raw"]
        else:
            if "smooth" not in _objects:
                _objects["smooth"] = PolynomialRF(b.quadratures)
            field = _objects["smooth"]
        result = integrate_rf(field, initial_state, b.charge, b.mass, frequency, phase, start_time, end_time,
                              max_steps_per_period=controls["RF_steps_per_period"], rtol=controls["RF_rtol"],
                              atol=controls["RF_atol_SI"], output_samples_per_period=settings["output_samples_per_RF_period"],
                              radius=b.reconstruction.valid_radius)
        t, states, status, nfev, exit_time = result.time, result.states, result.status, result.nfev, result.exit_time
        if previous is not None:
            t, states = np.r_[previous[0], t[1:]], np.vstack((previous[1], states[1:]))
        energy = np.empty(0)
    else:
        slow_model = b if case["model"] == "B" else c
        R, V = p["slow_position_m"], p["slow_velocity_m_per_s"]
        result = integrate_averaged(slow_model, R, V, end_time,
                                    samples=settings["RF_periods"][case["extent"]]*settings["output_samples_per_RF_period"]+1,
                                    rtol=controls["slow_rtol"], atol=controls["slow_dimensionless_atol"],
                                    max_step=settings["estimated_slow_period_s"]/controls["slow_steps_per_period"])
        t, states, energy, status, nfev, exit_time = result.time, result.states, result.energy, result.status, result.evaluations, result.exit_time
    arrays, metadata = checkpoint_paths(case)
    temp = arrays.with_suffix(".tmp.npz")
    np.savez_compressed(temp, time_s=t, states=states, energy_J=energy)
    temp.replace(arrays)
    meta = {"case": case, "signature": case_signature(case, settings), "status": status,
            "nfev_this_segment": int(nfev), "wall_seconds_including_load": time.perf_counter()-started,
            "start_time_this_segment_s": start_time, "end_time_s": float(t[-1]), "exit_time_s": exit_time,
            "max_radius_m": float(np.linalg.norm(states[:, :2], axis=-1).max()),
            "parent_arrays_sha256": parent_sha, "arrays_sha256": sha(arrays),
            "fem_fingerprint": protocol["fem_fingerprint"], "controls": controls,
            "RF_phase_rad": phase if case["model"] in ("A", "A_smooth") else None,
            "state_order": ["x_m", "y_m", "vx_m_per_s", "vy_m_per_s"]}
    write(metadata, meta)
    return {"file": arrays.name, "status": status, "wall_seconds": meta["wall_seconds_including_load"], "nfev": int(nfev)}


def probe(settings, b):
    path = OUTPUT/"probe.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["numerical_provenance"] == settings["numerical_provenance"] and saved["passed"]:
            print("Reusing verified evaluator/performance probe.", flush=True)
            return
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    _, _, _, interp = load_fem_cache(config, ROOT)
    raw, smooth = AffineP2RF(interp, b.amplitude), PolynomialRF(b.quadratures)
    rng = np.random.default_rng(20261023)
    angles, radius = rng.uniform(0, 2*np.pi, 701), np.sqrt(rng.uniform(0, .0012**2, 701))
    points = np.column_stack((radius*np.cos(angles), radius*np.sin(angles)))
    ec, es = b.quadratures.field_quadratures(points)
    errors = {}
    for name, field in (("raw", raw), ("smooth", smooth)):
        computed = np.array([field.quadratures(*point) for point in points])
        if name == "raw":
            a, bb, c = (interp.field_basis(points, p) for p in ("A", "B", "C"))
            ref = np.column_stack((math.sqrt(3)*b.amplitude*(c-bb)/2, b.amplitude*(a-bb/2-c/2)))
        else:
            ref = np.column_stack((ec, es))
        errors[name] = {"absolute_max_V_per_m": float(np.max(abs(computed-ref))),
                        "relative_max_on_global_field_scale": float(np.max(abs(computed-ref))/np.max(abs(ref)))}
    _, initial, _, _ = load_models()
    y0 = initial["cases"][0]["A_and_A_smooth_full_state"]
    control = settings["controls"]["coarse"]
    timings, trajectories = {}, {}
    for name, field in (("A", raw), ("A_smooth", smooth)):
        result = integrate_rf(field, y0, b.charge, b.mass, b.frequency, 0., 0., 64/b.frequency,
                              max_steps_per_period=50, rtol=control["RF_rtol"], atol=control["RF_atol_SI"])
        if result.status != "completed":
            raise RuntimeError("Evaluator probe left the domain")
        timings[name] = {"seconds_for_64_RF_periods": result.wall_seconds, "nfev": result.nfev}
        trajectories[name] = result
    # Cross-check an actual raw trajectory through the original implementation.
    particle = ParticleConfig(mass=b.mass, charge=b.charge, gamma=0., gravity_enabled=False)
    original = integrate_trajectory(None, particle, b.amplitude, b.frequency, initial_position=y0[:2], initial_velocity=y0[2:],
                                    periods=64, escape_radius=.008, rtol=control["RF_rtol"], atol=control["RF_atol_SI"],
                                    max_step_per_period=50, output_samples_per_period=64,
                                    field_function=PhasedFEMField(interp, b.amplitude, b.frequency, 0.))
    trajectory_error = coordinate_metrics(trajectories["A"].states[:, :2], original.state[:2].T, .001)
    passed = errors["raw"]["relative_max_on_global_field_scale"] < 1e-8 and errors["smooth"]["relative_max_on_global_field_scale"] < 1e-12 and trajectory_error["relative_max"] < 1e-5
    value = {"passed": bool(passed), "numerical_provenance": settings["numerical_provenance"], "field_equivalence": errors,
             "raw_trajectory_vs_original": trajectory_error, "timings": timings,
             "estimated_RF_seconds_without_parallelism": sum(v["seconds_for_64_RF_periods"] for v in timings.values())*6*settings["RF_periods"]["pilot"]/64,
             "note": "Timing estimate only; raw discontinuities and parallel CPU load change the actual cost"}
    write(path, value)
    print(json.dumps({"probe_passed": bool(passed), "timings": timings, "raw_equivalence": errors["raw"], "trajectory_equivalence": trajectory_error}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Optimized field equivalence probe failed")


def run_group(cases, settings, workers):
    pending = [case for case in cases if not valid_checkpoint(case, settings)]
    print(f"Checkpoints reused: {len(cases)-len(pending)}; pending: {len(pending)}.", flush=True)
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(worker, case): case for case in pending}
        for job in as_completed(jobs):
            result = job.result()
            print(json.dumps(result), flush=True)
            if result["status"] != "completed":
                raise RuntimeError("Pilot terminated before its requested duration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "probe", "run", "analyze"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol, initial, b, c = load_models()
    scale = float(np.linalg.norm(protocol["pilot"]["slow_position_m"]))
    period = float(4*np.sqrt(b.mass/(2*c.B6))*beta(1/6, 1/2)/(6*scale**2))
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    if fem_fingerprint(config) != protocol["fem_fingerprint"]:
        raise ValueError("FEM fingerprint mismatch")
    kernels = ["src/three_phase_trap/pilot_dynamics.py", "src/three_phase_trap/validation_field_audit.py",
               "src/three_phase_trap/interpolation.py", "src/three_phase_trap/rf_reconstruction.py",
               "src/three_phase_trap/dynamics.py", "src/three_phase_trap/averaged_dynamics.py", "src/three_phase_trap/octupole_model.py"]
    inputs = [OUTPUT.parent/p for p in ("protocol.json", "initial_conditions.json", "model_b/reconstruction.json", "model_c/coefficients.json")]
    import scipy
    provenance = {"fem_fingerprint": protocol["fem_fingerprint"], "basis_sha256": protocol["basis_sha256"],
                  "kernel_source_sha256": {p: sha(ROOT/p) for p in kernels},
                  "worker_source_sha256": hashlib.sha256(inspect.getsource(worker).encode()).hexdigest(),
                  "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs},
                  "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    settings = {"user_step": 7, "detailed_plan_stage": 6, "estimated_slow_period_s": period,
                "RF_periods": {"pilot": math.ceil(3.25*period*b.frequency), "extended": math.ceil(10.25*period*b.frequency)},
                "minimum_requested_slow_periods": {"pilot": 3, "extended": 10},
                "pilot_phases_rad": protocol["pilot"]["phase_rad"], "extended_phase_indices": [0],
                "R_scale_m": scale, "output_samples_per_RF_period": 64,
                "controls": {"coarse": {"RF_steps_per_period": 50, "RF_rtol": 2e-9, "RF_atol_SI": [2e-13, 2e-13, 2e-11, 2e-11],
                                        "slow_steps_per_period": 100, "slow_rtol": 2e-10, "slow_dimensionless_atol": 2e-12},
                             "fine": {"RF_steps_per_period": 100, "RF_rtol": 2e-11, "RF_atol_SI": [2e-15, 2e-15, 2e-13, 2e-13],
                                      "slow_steps_per_period": 200, "slow_rtol": 2e-12, "slow_dimensionless_atol": 2e-14}},
                "primary_filter": "centred one-RF-period trapezoid mean at window centres",
                "numerical_provenance": provenance,
                "scope": "One 70um FEM mesh, two phases at 3.25 slow periods and phase 0 extended to 10.25; no global stability or mesh-convergence claim"}
    existing = OUTPUT/"settings.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing pilot settings differ; retain outputs and use a versioned study directory")
    write(existing, settings)
    print(f"Slow period estimate: {period:.9g} s; RF periods: {settings['RF_periods']}", flush=True)
    if args.stage in ("all", "probe", "run"):
        probe(settings, b)
    if args.stage in ("all", "run"):
        pilot = [{"model": model, "phase_index": phase, "level": level, "extent": "pilot"}
                 for model in ("B", "C", "A_smooth", "A") for phase in ((0,) if model in ("B", "C") else (0, 1)) for level in ("coarse", "fine")]
        run_group(pilot, settings, args.workers)
        extended = [{"model": model, "phase_index": 0, "level": level, "extent": "extended"}
                    for model in ("B", "C", "A_smooth", "A") for level in ("coarse", "fine")]
        run_group(extended, settings, args.workers)
    if args.stage in ("all", "analyze"):
        from analyze_abc_pilot import main as analyze
        analyze()


if __name__ == "__main__":
    main()
