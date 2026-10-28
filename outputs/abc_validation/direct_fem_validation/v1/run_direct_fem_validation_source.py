"""Complete native FEM A verification at original pilot/extended cases.

Freeze inputs, calibrate native-field B/C, then compute direct P4 RF runs at
two meshes and temporal resolutions. Never overwrite historical P2 studies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.direct_fem_dynamics import integrate_native
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.averaged_dynamics import AveragedRFModel, integrate_averaged
from three_phase_trap.octupole_model import OctupoleModel, octupole_ring_coefficients
from three_phase_trap.pilot_dynamics import PolynomialRF, integrate_rf
from three_phase_trap.micromotion import recover_slow_state_with_drift
from build_direct_fem import load_field, input_signature, OUTPUT, BASE
from run_abc_pilot import sha, write

_fields = {}


def prepare():
    import scipy
    protocol = json.loads((BASE/"protocol.json").read_text(encoding="utf-8"))
    old = json.loads((BASE/"pilot/settings.json").read_text(encoding="utf-8"))
    initials = json.loads((BASE/"initial_conditions.json").read_text(encoding="utf-8"))
    inputs = [BASE/"protocol.json", BASE/"pilot/settings.json", BASE/"initial_conditions.json",
              BASE/"model_b/reconstruction.json", BASE/"model_c/coefficients.json",
              ROOT/"outputs/fem_cache/p2_basis.npz", ROOT/"outputs/fem_cache/gmsh_70um_fem_bundle.zip",
              OUTPUT/"h70_P2/full_P2_equivalence.json", OUTPUT/"probe/summary.json"]
    for model in ("B", "C", "A_smooth"):
        for extent in ("pilot", "extended"):
            for level in ("coarse", "fine"):
                for phase in ((0, 1) if model == "A_smooth" and extent == "pilot" else (0,)):
                    inputs += [BASE/f"pilot/{model}_phase{phase}_{level}_{extent}{suffix}" for suffix in (".json", ".npz")]
    for h in (50, 35):
        d = OUTPUT/f"h{h}_P4"
        meta = json.loads((d/"complete.json").read_text(encoding="utf-8"))
        if meta["signature"] != input_signature(h, 4) or not all(sha(d/n) == value for n, value in meta["artifacts"].items()):
            raise ValueError("Native FEM cache not ready/intact")
        inputs += [d/"complete.json"]+[d/n for n in meta["artifacts"]]
    kernel_names = ["direct_fem.py", "direct_fem_dynamics.py", "audited_rf.py", "restart_dynamics.py", "pilot_dynamics.py",
                    "rf_reconstruction.py", "averaged_dynamics.py", "octupole_model.py", "micromotion.py", "slow_analysis.py"]
    settings = {"scope": "Direct, unsmoothed P4 FEM A; same geometry/ABC/physics as original P2; old P2 checkpoints remain historical",
                "meshes_um": [50, 35], "FEM_degree": 4, "geometry_degree": 1,
                "physical": protocol["pilot"], "RF_periods": old["RF_periods"], "R_scale_m": old["R_scale_m"],
                "estimated_slow_period_s": old["estimated_slow_period_s"],
                "full_initial_states": [c["A_and_A_smooth_full_state"] for c in initials["cases"]],
                "controls": {
                    "coarse": {"method": "DOP853", "steps": 50, "rtol": 2e-9, "atol": [2e-13, 2e-13, 2e-11, 2e-11]},
                    "fine": {"method": "DOP853", "steps": 100, "rtol": 2e-11, "atol": [2e-15, 2e-15, 2e-13, 2e-13]},
                    "refined": {"method": "DOP853", "steps": 200, "rtol": 2e-13, "atol": [2e-17, 2e-17, 2e-15, 2e-15]},
                    "radau": {"method": "Radau", "steps": 50, "rtol": 2e-8, "atol": [2e-12, 2e-12, 2e-10, 2e-10]}},
                "samples_per_RF_period": 64, "tolerances": protocol["predeclared_trajectory_tolerances"],
                "field_relative_screening_target": .001,
                "primary_comparisons": ["A_P4 vs frozen B70/C70", "A_P4 vs same-mesh B_P4/C_P4", "A_P4 vs same-mesh diagnostic A_fit"],
                "spatial_control": "P4 h50 vs h35, identical physical full IC and global RF clock, all three original cases",
                "independent_temporal_control": "h35 phase0 pilot: DOP853 T50/T100/T200 and Radau; common geometry audit",
                "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in dict.fromkeys(inputs)},
                "kernel_sha256": {n: sha(ROOT/"src/three_phase_trap"/n) for n in kernel_names},
                "worker_sha256": hashlib.sha256(inspect.getsource(worker).encode()).hexdigest(),
                "slow_worker_sha256": hashlib.sha256(inspect.getsource(slow_models).encode()).hexdigest(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    path = OUTPUT/"validation_settings.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Frozen native validation changed; preserve v1 and use a new version")
    if not path.exists():
        write(path, settings)
    (OUTPUT/"runs").mkdir(exist_ok=True)
    return settings


def paths(case):
    tag = f"{case['model']}_h{case['mesh']}_phase{case['phase']}_{case['level']}_{case['extent']}"
    return OUTPUT/"runs"/(tag+".npz"), OUTPUT/"runs"/(tag+".json")


def signature(case):
    return hashlib.sha256(json.dumps({"case": case, "settings": sha(OUTPUT/"validation_settings.json")}, sort_keys=True).encode()).hexdigest()


def valid(case):
    a, m = paths(case)
    if not a.exists() or not m.exists():
        return False
    meta = json.loads(m.read_text(encoding="utf-8"))
    if meta["signature"] != signature(case) or meta["arrays_sha256"] != sha(a):
        raise ValueError(f"Native trajectory checkpoint mismatch: {m.name}")
    log = a.with_name(a.stem+"_crossings.jsonl")
    if sha(log) != meta["crossings_sha256"]:
        raise ValueError("Native crossing log checksum mismatch")
    return meta["status"] == "completed"


def slow_models(h, settings):
    directory = OUTPUT/f"h{h}_P4"
    p = settings["physical"]
    recon = RFReconstruction.load(directory/"diagnostic_N48.json")
    b = AveragedRFModel(recon, p["charge_C"], p["mass_kg"], p["U0_V"], p["frequency_Hz"])
    with np.load(directory/"common_samples.npz") as data:
        phi = data["phi"]
    radii = np.array([.025, .05, .1, .25, .5, 1, 2, 4, 6, 7, 8])*1e-3
    angle = (np.arange(513)+.739)*2*np.pi/513
    coefficients, calibration = [], []
    for radius in (.0005, .001, .002):
        i = int(np.argmin(abs(radii-radius)))
        cut = slice(i*513, (i+1)*513)
        c4, s4, minus, plus = octupole_ring_coefficients(phi[cut, 0]/b.amplitude, phi[cut, 1]/b.amplitude, angle, radius)
        coefficients.append(minus)
        calibration.append({"radius_m": radius, "d_minus_real_per_volt": float(minus.real), "d_minus_imag_per_volt": float(minus.imag),
                            "counter_rotating_fraction": float(abs(plus)/abs(minus))})
    coefficient = np.mean(coefficients)
    c = OctupoleModel(float(abs(coefficient)), float(-np.angle(coefficient)), b.charge, b.mass, b.amplitude, b.frequency, .008)
    write(directory/"native_C_calibration.json", {"source": "Fourier n4 of DIRECT P4 potential at 0.5/1/2 mm, not trajectory fit",
                                                "C4_per_volt_per_m4": c.coefficient_per_volt, "phase_rad": c.phase, "rings": calibration})
    d = directory/"slow"
    d.mkdir(exist_ok=True)
    for extent in ("pilot", "extended"):
        duration = settings["RF_periods"][extent]/b.frequency
        for model, obj in (("B", b), ("C", c)):
            for level, rtol, atol, steps in (("coarse", 2e-10, 2e-12, 100), ("fine", 2e-12, 2e-14, 200)):
                path = d/f"{model}_{level}_{extent}.npz"
                result = integrate_averaged(obj, p["slow_position_m"], p["slow_velocity_m_per_s"], duration,
                                            samples=settings["RF_periods"][extent]*64+1, rtol=rtol, atol=atol,
                                            max_step=settings["estimated_slow_period_s"]/steps)
                if result.status != "completed":
                    raise ValueError("Native-field slow reference left domain")
                np.savez_compressed(path, time_s=result.time, states=result.states, energy_J=result.energy)
        # Quantify the small initial-map change when refining the FEM while
        # preserving the original PHYSICAL full IC. This does not alter A.
        for phase_index in ((0, 1) if extent == "pilot" else (0,)):
            full = np.array(settings["full_initial_states"][phase_index])
            slow = recover_slow_state_with_drift(b.quadratures, full, b.charge, b.mass, b.omega, p["phase_rad"][phase_index],
                                                 check_points=b.reconstruction.check_points)
            result = integrate_averaged(b, slow[:2], slow[2:], duration, samples=settings["RF_periods"][extent]*64+1,
                                        rtol=2e-12, atol=2e-14, max_step=settings["estimated_slow_period_s"]/200)
            if result.status != "completed":
                raise ValueError("Matched native B left domain")
            np.savez_compressed(d/f"B_matched_phase{phase_index}_{extent}.npz", time_s=result.time, states=result.states,
                                energy_J=result.energy, initial_slow_state=slow)
    write(d/"complete.json", {"settings_sha256": sha(OUTPUT/"validation_settings.json"),
                               "artifacts": {f.name: sha(f) for f in d.glob("*.npz")}})


def worker(case):
    settings = json.loads((OUTPUT/"validation_settings.json").read_text(encoding="utf-8"))
    p = settings["physical"]
    control = settings["controls"][case["level"]]
    start, end = 0., settings["RF_periods"][case["extent"]]/p["frequency_Hz"]
    initial = np.array(settings["full_initial_states"][case["phase"]])
    previous, owners_previous, parent_hash = None, None, None
    if case["extent"] == "extended":
        parent = {**case, "extent": "pilot"}
        if not valid(parent):
            raise ValueError("Native extended case needs its matching pilot")
        parent_path, _ = paths(parent)
        with np.load(parent_path) as data:
            previous = (data["time_s"], data["states"])
            owners_previous = data["triangles"] if "triangles" in data else None
        start, initial = float(previous[0][-1]), previous[1][-1]
        parent_hash = sha(parent_path)
    print("Starting native", case, flush=True)
    started = time.perf_counter()
    if case["model"] == "A":
        h = case["mesh"]
        if h not in _fields:
            _fields[h] = load_field(h, 4, p["U0_V"])
        def progress(value):
            print(json.dumps({"case": case, "progress": value}), flush=True)
        out = integrate_native(_fields[h], initial, p["charge_C"], p["mass_kg"], p["frequency_Hz"], p["phase_rad"][case["phase"]], start, end,
                               method=control["method"], steps_per_period=control["steps"], rtol=control["rtol"], atol=control["atol"], progress=progress)
        result, owners, diagnostics = out.run, out.sample_triangles, out.diagnostics
        crossings = out.crossings
    elif case["model"] == "A_fit":
        rf = RFReconstruction.load(OUTPUT/f"h{case['mesh']}_P4/diagnostic_N48.json")
        result = integrate_rf(PolynomialRF(rf.scaled(p["U0_V"])), initial, p["charge_C"], p["mass_kg"], p["frequency_Hz"], p["phase_rad"][case["phase"]], start, end,
                              max_steps_per_period=control["steps"], rtol=control["rtol"], atol=control["atol"])
        owners, crossings, diagnostics = None, [], {"diagnostic_only": "A_fit has one temporal level; A is independently controlled at multiple levels/methods"}
    else:
        raise ValueError("Unknown native validation model")
    if result.status != "completed" or not np.array_equal(result.states[0], initial):
        raise ValueError("Native RF did not complete or altered initial state")
    arrays, metadata = paths(case)
    t, states = result.time, result.states
    if previous is not None:
        t, states = np.r_[previous[0], t[1:]], np.vstack((previous[1], states[1:]))
        if owners is not None:
            owners = np.r_[owners_previous, owners[1:]]
    temp = arrays.with_suffix(".tmp.npz")
    data = {"time_s": t, "states": states}
    if owners is not None:
        data["triangles"] = owners
    np.savez_compressed(temp, **data)
    temp.replace(arrays)
    log = arrays.with_name(arrays.stem+"_crossings.jsonl")
    log.write_text("".join(json.dumps(v)+"\n" for v in crossings), encoding="utf-8")
    write(metadata, {"case": case, "signature": signature(case), "status": result.status,
                     "arrays_sha256": sha(arrays), "crossings_sha256": sha(log), "segment_start_s": start,
                     "end_s": float(t[-1]), "parent_arrays_sha256": parent_hash,
                     "initial_state_this_segment": initial.tolist(), "global_phase_rad": p["phase_rad"][case["phase"]],
                     "nfev_segment": result.nfev, "wall_seconds_including_load": time.perf_counter()-started,
                     "diagnostics_this_segment": diagnostics})
    return {"case": case, "status": result.status, "seconds": time.perf_counter()-started}


def pilot_cases():
    cases = [{"model": "A", "mesh": h, "phase": phase, "extent": "pilot", "level": level}
             for h in (50, 35) for phase in (0, 1) for level in ("coarse", "fine")]
    cases += [{"model": "A", "mesh": 35, "phase": 0, "extent": "pilot", "level": level} for level in ("refined", "radau")]
    cases += [{"model": "A_fit", "mesh": 35, "phase": phase, "extent": "pilot", "level": "fine"} for phase in (0, 1)]
    return cases


def extended_cases():
    return [{"model": "A", "mesh": h, "phase": 0, "extent": "extended", "level": level} for h in (50, 35) for level in ("coarse", "fine")]+[
        {"model": "A_fit", "mesh": 35, "phase": 0, "extent": "extended", "level": "fine"}]


def run_batch(cases, workers):
    pending = [c for c in cases if not valid(c)]
    weights = {"coarse": 1., "fine": 2., "refined": 4., "radau": 3.}
    pending.sort(key=lambda c: weights[c["level"]], reverse=True)
    print(f"Native cases pending/reused: {len(pending)}/{len(cases)-len(pending)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, case) for case in pending]
        for job in as_completed(futures):
            result = job.result()
            print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "slow", "pilot", "extended"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    settings = prepare()
    if args.stage in ("all", "slow"):
        for h in settings["meshes_um"]:
            path = OUTPUT/f"h{h}_P4/slow/complete.json"
            if path.exists():
                meta = json.loads(path.read_text(encoding="utf-8"))
                if meta["settings_sha256"] != sha(OUTPUT/"validation_settings.json") or not all(sha(path.parent/n) == value for n, value in meta["artifacts"].items()):
                    raise ValueError("Native slow checkpoint mismatch")
            else:
                print(f"Building native-field B/C and initial-map controls h{h}", flush=True)
                slow_models(h, settings)
    if args.stage in ("all", "pilot"):
        run_batch(pilot_cases(), args.workers)
    if args.stage in ("all", "extended"):
        run_batch(extended_cases(), args.workers)


if __name__ == "__main__":
    main()
