"""Checkpointed harmonic-ablation trajectories for the V7 physical study.

All fields are assembled from the existing h35/h50 N48 polynomial fits. No
FEM solve is performed. Work accumulators are integrated as auxiliary ODE
variables so RF-cycle energy transfer is not estimated from sparse samples.
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V4 = ROOT / "outputs/octupole_comparison_home/confinement_maps_v4"
V6 = ROOT / "outputs/octupole_comparison_home/chaos_escape_v6"
FIELD35 = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h35_P4/diagnostic_N48.json"
FIELD50 = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json"
sys.path.insert(0, str(V4))
import run_v4  # noqa: E402

OUT = HERE
MASS_MZ_TH = 100_000.0
CHARGE_STATE = 1
FREQUENCY_HZ = 100_000.0
OMEGA = 2.0 * math.pi * FREQUENCY_HZ
VOLTAGE = {"rotating": 250.0, "standing": 353.5533905932738}
ENERGY_EV = 2.0
R0_MM = 1.0
APERTURE_M = 8e-3
HORIZON_RF = 1_000
SAMPLES_PER_RF = 8
RTOL = 2e-9
ATOL_PHYSICAL = (2e-13, 2e-13, 2e-11, 2e-11)
MAX_STEP_RF_DIV = 50
MAX_WALL_S = 60 * 60
DEADLINE_RESERVE_S = 180
WORKERS_MAX = 4

MODEL_SET = {
    "pure_m4": (4,),
    "m4_m8": (4, 8),
    "m4_m8_m16": (4, 8, 16),
    "full_N48": None,
    "full_N48_minus_m8": "minus8",
    "m4_m8_m16_m20": (4, 8, 16, 20),
    "m4_m8_m20": (4, 8, 20),
    "full_N48_minus_m20": "minus20",
    "full_N48_minus_m16": "minus16",
    "full_N48_minus_m28": "minus28",
    "full_N48_minus_m32": "minus32",
    "full_N48_minus_m40": "minus40",
    "full_N48_minus_m44": "minus44",
}
CORE_REPRESENTATIONS = tuple(list(MODEL_SET)[:5])
STARTS = {
    "deep_az0_dplus": (0.0, 1),
    "boundary_dminus_retained": (9.60205078125, -1),
    "boundary_dminus_escaped": (9.6240234375, -1),
    "boundary_dplus_retained": (10.26123046875, 1),
    "boundary_dplus_escaped": (10.283203125, 1),
    "early_az22p5_dplus": (22.5, 1),
    "late_az17p2266_dplus": (17.2265625, 1),
}
STAGES = {
    "pilot": ("deep_az0_dplus", "early_az22p5_dplus"),
    "boundary": ("boundary_dminus_retained", "boundary_dminus_escaped",
                 "boundary_dplus_retained", "boundary_dplus_escaped"),
    "late": ("late_az17p2266_dplus",),
}

_Q_MODELS = None
_MODE_INDICES = None


def _atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _job_hash(job):
    payload = json.dumps(job, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def make_jobs(stage):
    if stage not in STAGES:
        raise ValueError(stage)
    jobs = []
    for launch in STAGES[stage]:
        az, direction = STARTS[launch]
        for representation in CORE_REPRESENTATIONS:
            for model in ("rotating", "standing"):
                job = {
                    "stage": stage,
                    "launch_label": launch,
                    "representation": representation,
                    "retained_harmonic_orders": list(MODEL_SET[representation]) if isinstance(MODEL_SET[representation], tuple) else MODEL_SET[representation],
                    "model": model,
                    "field_source": "h35_N48_polynomial",
                    "normalization": "matched_leading_m4_pseudopotential",
                    "voltage_V": VOLTAGE[model],
                    "frequency_Hz": FREQUENCY_HZ,
                    "mz_Th": MASS_MZ_TH,
                    "charge_state": CHARGE_STATE,
                    "energy_eV": ENERGY_EV,
                    "r0_mm": R0_MM,
                    "azimuth_deg": az,
                    "phase_deg": 0.0,
                    "direction": direction,
                    "velocity_definition": "tangential; signed v_theta verified",
                    "horizon_RF": HORIZON_RF,
                    "aperture_mm": APERTURE_M * 1e3,
                    "rtol": RTOL,
                    "atol": list(ATOL_PHYSICAL),
                    "max_step_RF_fraction": 1.0 / MAX_STEP_RF_DIV,
                    "samples_per_RF": SAMPLES_PER_RF,
                }
                job["job_id"] = _job_hash(job)
                jobs.append(job)
    return jobs


def _load_source_fields():
    _, q35 = run_v4.load_fields(FIELD35)
    _, q50 = run_v4.load_fields(FIELD50)
    return q35, q50


def _make_q_models(q_source):
    from three_phase_trap.rf_reconstruction import HarmonicQuadratures
    # Ignore coefficient-fit leakage at a relative 1e-8 of m=4. The h50 fit
    # contains m=36 and m=48 residuals near 1e-11 of m=4; retaining them would
    # mislabel numerical leakage as a physical symmetry-breaking harmonic.
    coefficient_floor = max(1e-12, 1e-8 * max(abs(q_source.cosine[4]), abs(q_source.sine[4])))
    available = tuple(i for i in range(1, len(q_source.sine))
                      if max(abs(q_source.cosine[i]), abs(q_source.sine[i])) > coefficient_floor)
    models = {}
    indices = {}
    for rep, selection in MODEL_SET.items():
        if selection is None:
            keep = available
        elif isinstance(selection, str) and selection.startswith("minus"):
            remove_m = int(selection.removeprefix("minus"))
            keep = tuple(m for m in available if m != remove_m)
        else:
            keep = tuple(selection)
        indices[rep] = tuple(m for m in keep if m in available)
        c = np.zeros_like(q_source.cosine)
        s = np.zeros_like(q_source.sine)
        for m in indices[rep]:
            c[m] = q_source.cosine[m]
            s[m] = q_source.sine[m]
        for model in ("rotating", "standing"):
            c_use = c if model == "rotating" else np.zeros_like(c)
            models[(rep, model)] = HarmonicQuadratures(c_use.copy(), s.copy(), q_source.reference_radius)
    return models, indices


def _init_worker(field_path=FIELD35):
    global _Q_MODELS, _MODE_INDICES
    os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    _, q = run_v4.load_fields(Path(field_path))
    _Q_MODELS, _MODE_INDICES = _make_q_models(q)


def _mode_coeffs(q, modes):
    out = []
    for m in modes:
        out.append((m, q.cosine[m], q.sine[m]))
    return out


def integrate_job(job):
    q = _Q_MODELS[(job["representation"], job["model"])]
    modes = _MODE_INDICES[job["representation"]]
    coeffs = _mode_coeffs(q, modes)
    mass = job["mz_Th"] * run_v4.AMU
    charge = CHARGE_STATE * run_v4.E_CHARGE
    qm = charge / mass
    inv_ref = 1.0 / q.reference_radius
    voltage = job["voltage_V"]
    az = math.radians(job["azimuth_deg"])
    phase0 = math.radians(job["phase_deg"])
    initial = run_v4.make_initial(job["mz_Th"], job["r0_mm"], job["energy_eV"] * 1000.0,
                                  az, phase0, float(job["direction"]), "tangent")
    r0 = math.hypot(initial[0], initial[1])
    vtheta = (-initial[2] * initial[1] + initial[3] * initial[0]) / r0
    if job["direction"] * vtheta <= 0:
        raise ValueError(f"velocity sign mismatch: {job['job_id']}")

    nmode = len(coeffs)
    physical_atol = np.asarray(ATOL_PHYSICAL, float)
    atol = np.r_[np.asarray(job.get("atol", physical_atol), float),
                 np.full(2 * nmode, float(job.get("work_atol_eV", 2e-10)))]

    def rhs(t, y):
        x, yy, vx, vy = y[:4]
        z = complex(x, yy) * inv_ref
        lam = OMEGA * t + phase0
        co, si = math.cos(lam), math.sin(lam)
        ax = ay = 0.0
        wkin_dot = np.zeros(nmode)
        wparam_dot = np.zeros(nmode)
        for k, (m, cm, sm) in enumerate(coeffs):
            zm = z ** m
            scale = m * (z ** (m - 1)) * inv_ref
            dpc = cm * scale
            dps = sm * scale
            ecx, ecy = -dpc.real, dpc.imag
            esx, esy = -dps.real, dps.imag
            ex = voltage * (co * ecx + si * esx)
            ey = voltage * (co * ecy + si * esy)
            ax += qm * ex
            ay += qm * ey
            wkin_dot[k] = ex * vx + ey * vy
            pc = (cm * zm).real
            ps = (sm * zm).real
            wparam_dot[k] = voltage * OMEGA * (-pc * si + ps * co)
        return np.r_[vx, vy, ax, ay, wkin_dot, wparam_dot]

    def aperture(_t, y):
        return APERTURE_M - math.hypot(y[0], y[1])
    aperture.terminal = True
    aperture.direction = -1

    period = 1.0 / FREQUENCY_HZ
    horizon_rf = int(job.get("horizon_RF", HORIZON_RF))
    samples_per_rf = int(job.get("samples_per_RF", SAMPLES_PER_RF))
    horizon_s = horizon_rf * period
    t_eval = np.linspace(0.0, horizon_s, horizon_rf * samples_per_rf + 1)
    y0 = np.r_[initial, np.zeros(2 * nmode)]
    started = time.perf_counter()
    run_rtol = float(job.get("rtol", RTOL))
    max_step_fraction = float(job.get("max_step_RF_fraction", 1.0 / MAX_STEP_RF_DIV))
    sol = solve_ivp(rhs, (0.0, horizon_s), y0, method="DOP853", rtol=run_rtol, atol=atol,
                    max_step=period * max_step_fraction, t_eval=t_eval, events=aperture)
    elapsed = time.perf_counter() - started
    if not sol.success:
        raise RuntimeError(sol.message)
    escaped = bool(len(sol.t_events[0]))
    if escaped:
        te = float(sol.t_events[0][0])
        ye = sol.y_events[0][0]
        if len(sol.t) == 0 or abs(sol.t[-1] - te) > 1e-14:
            times = np.r_[sol.t, te]
            states_ext = np.vstack((sol.y.T, ye))
        else:
            times = sol.t
            states_ext = sol.y.T
    else:
        times = sol.t
        states_ext = sol.y.T
        te = None
    states = states_ext[:, :4]
    radius = np.hypot(states[:, 0], states[:, 1])
    radial_velocity = np.sum(states[:, :2] * states[:, 2:4], axis=1) / np.maximum(radius, 1e-30)
    time_rf = times * FREQUENCY_HZ
    lam = OMEGA * times + phase0
    z = (states[:, 0] + 1j * states[:, 1]) * inv_ref
    phi_modes = np.empty((len(times), nmode), float)
    for k, (m, cm, sm) in enumerate(coeffs):
        zm = z ** m
        phi_modes[:, k] = voltage * ((cm * zm).real * np.cos(lam) + (sm * zm).real * np.sin(lam))
    kinetic_eV = 0.5 * mass * np.sum(states[:, 2:4] ** 2, axis=1) / run_v4.E_CHARGE
    hamiltonian_eV = kinetic_eV + np.sum(phi_modes, axis=1)
    lz = mass * (states[:, 0] * states[:, 3] - states[:, 1] * states[:, 2])
    wkin = states_ext[:, 4:4+nmode]
    wparam = states_ext[:, 4+nmode:4+2*nmode]
    kin_closure = float((kinetic_eV[-1] - kinetic_eV[0]) - np.sum(wkin[-1]))
    h_closure = float((hamiltonian_eV[-1] - hamiltonian_eV[0]) - np.sum(wparam[-1]))

    trace_path = OUT / "traces" / f"{job['job_id']}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_tmp = trace_path.with_suffix(".npz.tmp")
    with trace_tmp.open("wb") as f:
        np.savez_compressed(f, time_RF=time_rf, state=states, radius_m=radius,
                            radial_velocity_m_s=radial_velocity, kinetic_eV=kinetic_eV,
                            hamiltonian_eV=hamiltonian_eV, angular_momentum_kg_m2_s=lz,
                            potential_by_mode_eV=phi_modes, work_kinetic_by_mode_eV=wkin,
                            work_hamiltonian_by_mode_eV=wparam, harmonic_orders=np.asarray(modes, int))
        f.flush()
        os.fsync(f.fileno())
    os.replace(trace_tmp, trace_path)

    first_escape_rf = te * FREQUENCY_HZ if escaped else None
    result = {
        "job": job,
        "job_id": job["job_id"],
        "status": "escaped" if escaped else "retained",
        "escape_time_RF": first_escape_rf,
        "right_censored_at_RF": None if escaped else horizon_rf,
        "max_radius_mm": float(np.max(radius) * 1e3),
        "rms_radius_mm": float(math.sqrt(np.trapezoid(radius * radius, times) / max(times[-1] - times[0], 1e-30)) * 1e3),
        "initial_state": initial.tolist(),
        "verified_initial_vtheta_m_s": float(vtheta),
        "solver_status": sol.message,
        "nfev": int(sol.nfev),
        "wall_s": elapsed,
        "trace_file": str(trace_path.relative_to(ROOT)).replace("\\", "/"),
        "trace_samples": int(len(times)),
        "harmonic_orders": list(modes),
        "energy_work_closure_error_eV": kin_closure,
        "hamiltonian_work_closure_error_eV": h_closure,
        "kinetic_work_by_mode_final_eV": {str(m): float(wkin[-1, i]) for i, m in enumerate(modes)},
        "hamiltonian_work_by_mode_final_eV": {str(m): float(wparam[-1, i]) for i, m in enumerate(modes)},
    }
    return result


def _read_done():
    p = OUT / "trajectory_checkpoint.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("status") in ("retained", "escaped"):
                    trace = ROOT / row.get("trace_file", "")
                    if trace.exists():
                        out[row["job_id"]] = row
            except Exception:
                continue
    return out


def _start_deadline():
    path = OUT / "integration_deadline.json"
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
        return float(meta["deadline_unix_s"])
    now = time.time()
    meta = {
        "start_unix_s": now,
        "deadline_unix_s": now + MAX_WALL_S,
        "budget_s": MAX_WALL_S,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "start_rule": "created immediately before the first V7 RF integration",
        "workers": WORKERS_MAX,
    }
    _atomic_json(path, meta)
    return float(meta["deadline_unix_s"])


def _append_result(result):
    comp = OUT / "completed" / f"{result['job_id']}.json"
    _atomic_json(comp, result)
    with (OUT / "trajectory_checkpoint.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"checkpoint {result['job']['launch_label']} {result['job']['representation']} "
          f"{result['job']['model']}: {result['status']}; "
          f"escape={result['escape_time_RF'] if result['escape_time_RF'] is not None else 'censored'} RF; "
          f"wall={result['wall_s']:.1f}s; dK-work={result['energy_work_closure_error_eV']:.2e} eV",
          flush=True)


def _write_csv(rows):
    path = OUT / "processed" / "ablation_trajectory_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["job_id", "stage", "launch_label", "representation", "retained_harmonic_orders", "model",
              "field_source", "normalization", "voltage_V", "frequency_Hz", "mz_Th", "charge_state",
              "energy_eV", "r0_mm", "azimuth_deg", "phase_deg", "direction", "velocity_definition",
              "x0_m", "y0_m", "vx0_m_s", "vy0_m_s", "verified_initial_vtheta_m_s", "status",
              "escape_time_RF", "right_censored_at_RF", "max_radius_mm", "rms_radius_mm", "solver_status",
              "rtol", "max_step_RF_fraction", "samples_per_RF", "trace_samples", "nfev", "wall_s",
              "energy_work_closure_error_eV", "hamiltonian_work_closure_error_eV", "trace_file"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in sorted(rows, key=lambda r: (r["job"]["launch_label"], r["job"]["representation"], r["job"]["model"])):
            j = result["job"]
            state = result["initial_state"]
            row = {**j, **result, "retained_harmonic_orders": json.dumps(j["retained_harmonic_orders"]),
                   "x0_m": state[0], "y0_m": state[1], "vx0_m_s": state[2], "vy0_m_s": state[3]}
            writer.writerow(row)


def run_stage(stage, workers=4):
    if not 1 <= workers <= WORKERS_MAX:
        raise ValueError(f"workers must be 1..{WORKERS_MAX}")
    for sub in ("completed", "traces", "processed"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(stage)
    existing = _read_done()
    todo = [j for j in jobs if j["job_id"] not in existing]
    if todo:
        deadline = _start_deadline()
        print(f"stage={stage}; queued={len(todo)}; workers={workers}; deadline={deadline:.0f}", flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(FIELD35,)) as pool:
            pending = {}
            next_index = 0
            while next_index < len(todo) or pending:
                while next_index < len(todo) and len(pending) < workers:
                    if time.time() + DEADLINE_RESERVE_S >= deadline:
                        print("deadline reserve reached; no more V7 integrations submitted", flush=True)
                        next_index = len(todo)
                        break
                    job = todo[next_index]
                    pending[pool.submit(integrate_job, job)] = job
                    next_index += 1
                if not pending:
                    break
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    pending.pop(future)
                    result = future.result()
                    _append_result(result)
                    existing[result["job_id"]] = result
        if next_index < len(todo):
            print(f"incomplete at global deadline: {len(todo)-next_index} jobs remain", flush=True)
    rows = list(_read_done().values())
    _write_csv(rows)
    print(f"stage={stage}; total_checkpoints={len(rows)}; requested={len(jobs)}", flush=True)


def validate_static():
    _init_worker()
    q35, q50 = _load_source_fields()
    floor35 = max(1e-12, 1e-8 * max(abs(q35.cosine[4]), abs(q35.sine[4])))
    floor50 = max(1e-12, 1e-8 * max(abs(q50.cosine[4]), abs(q50.sine[4])))
    modes35 = tuple(i for i in range(1, len(q35.sine)) if max(abs(q35.cosine[i]), abs(q35.sine[i])) > floor35)
    modes50 = tuple(i for i in range(1, len(q50.sine)) if max(abs(q50.cosine[i]), abs(q50.sine[i])) > floor50)
    if modes35 != modes50:
        raise AssertionError(f"h35/h50 active harmonic orders differ: {modes35}, {modes50}")
    # Verify the reconstructed field is the negative potential gradient for
    # each full-field temporal quadrature, over several off-axis points.
    gradient_error = 0.0
    h = 2e-9
    for q in (q35, q50):
        for point in ((0.72e-3, 1.13e-3), (-2.1e-3, 0.44e-3), (3.2e-3, -1.7e-3)):
            xy = np.asarray(point)
            ec, es = q.field_quadratures(xy[None, :])
            for quad_idx, field in enumerate((ec[0], es[0])):
                fd = np.zeros(2)
                for axis in range(2):
                    delta = np.zeros(2); delta[axis] = h
                    pp = q.potential_quadratures((xy + delta)[None, :])[quad_idx][0]
                    pm = q.potential_quadratures((xy - delta)[None, :])[quad_idx][0]
                    fd[axis] = -(pp - pm) / (2*h)
                gradient_error = max(gradient_error, float(np.linalg.norm(fd-field) / max(1.0, np.linalg.norm(field))))
    c4, s4 = q35.cosine[4], q35.sine[4]
    amp_c = abs(c4) / q35.reference_radius**4
    amp_s = abs(s4) / q35.reference_radius**4
    amp_mismatch = abs(amp_c-amp_s) / max(amp_c,amp_s)
    helicity = []
    for m in modes35:
        ratio = q35.cosine[m] / q35.sine[m] if abs(q35.sine[m]) else complex(float("nan"), 0.)
        helicity.append({"m": m, "c_over_s_real": float(ratio.real), "c_over_s_imag": float(ratio.imag),
                         "c_abs_V_per_V": float(abs(q35.cosine[m])), "s_abs_V_per_V": float(abs(q35.sine[m])),
                         "phase_s_deg": float(math.degrees(np.angle(q35.sine[m]))),
                         "potential_coefficient_V_m_minus_m_per_electrode_V": float(abs(q35.sine[m]) / q35.reference_radius**m)})
    validation = {
        "passed": bool(gradient_error < 2e-6 and amp_mismatch < 1e-10),
        "field_source_h35": str(FIELD35.relative_to(ROOT)),
        "field_source_h50": str(FIELD50.relative_to(ROOT)),
        "reference_radius_m": q35.reference_radius,
        "active_harmonics_h35": list(modes35),
        "active_harmonics_h50": list(modes50),
        "active_harmonic_threshold": "max(1e-12, 1e-8 times the larger m=4 quadrature coefficient)",
        "below_threshold_h50_residuals": {str(m): float(max(abs(q50.cosine[m]), abs(q50.sine[m])))
                                           for m in range(1, len(q50.sine)) if m not in modes50 and
                                           max(abs(q50.cosine[m]), abs(q50.sine[m])) > 0},
        "finite_difference_field_gradient_relative_error_max": gradient_error,
        "m4_rotating_vs_standing_coefficient_relative_difference": amp_mismatch,
        "matched_voltage_ratio_standing_to_rotating": VOLTAGE["standing"] / VOLTAGE["rotating"],
        "harmonics": helicity,
        "phase_convention": {
            "general": "Phi_m=V Re[(c_m cos(lambda)+s_m sin(lambda)) (z^m)], lambda=Omega*t+phi0, z=(x+i y)/Rref",
            "rotating_mode_class_m12n_plus4": "c_m/s_m approximately +i; phase lambda-m*theta; pattern speed +Omega/m",
            "rotating_mode_class_m12n_plus8": "c_m/s_m approximately -i; phase lambda+m*theta; pattern speed -Omega/m",
            "standing": "c_m=0; all spatial harmonics share the same sin(lambda) time factor",
        },
        "field_coefficient_sensitivity_h35_h50": {
            str(m): float(abs(q35.sine[m]-q50.sine[m]) / max(abs(q35.sine[m]),1e-30)) for m in modes35
        },
    }
    _atomic_json(OUT / "processed" / "harmonic_field_validation.json", validation)
    config = {
        "study": "V7 causal harmonic ablation of finite-time escape",
        "scope": "deterministic two-dimensional transverse trajectories through 1000 RF periods; no infinite-time stability claim",
        "source_studies": {
            "V3": "outputs/octupole_comparison_home/confinement_maps_v4/ОТЧЁТ_V4_RU.md",
            "V4": "outputs/octupole_comparison_home/confinement_maps_v4/ОТЧЁТ_V4_RU.md",
            "V5": "outputs/octupole_comparison_home/matched_pseudopotential_v5/REPORT_RU.md",
            "V6": "outputs/octupole_comparison_home/chaos_escape_v6/REPORT_RU.md",
            "h35_N48": str(FIELD35.relative_to(ROOT)),
            "h50_N48": str(FIELD50.relative_to(ROOT)),
        },
        "normalization": {
            "rotating_V": VOLTAGE["rotating"],
            "standing_V": VOLTAGE["standing"],
            "basis": "match only leading measured m=4 pseudopotential; retain actual higher harmonic coefficients",
        },
        "ion": {"charge_state": CHARGE_STATE, "mz_Th": MASS_MZ_TH, "mass_kg": MASS_MZ_TH * run_v4.AMU},
        "initial_conditions": {"energy_eV": ENERGY_EV, "radius_mm": R0_MM, "phase_deg": 0.0,
                               "horizon_RF": HORIZON_RF, "aperture_mm": APERTURE_M*1e3,
                               "velocity": "same tangential construction as V6; signed v_theta checked per job",
                               "pilot_launches": list(STAGES["pilot"]),
                               "boundary_launches": list(STAGES["boundary"]),
                               "optional_late_launch": list(STAGES["late"])},
        "representations": {k: (list(v) if isinstance(v,tuple) else v) for k,v in MODEL_SET.items()},
        "solver": {"method": "DOP853", "rtol": RTOL, "atol_physical": list(ATOL_PHYSICAL),
                   "max_step_RF_fraction": 1/MAX_STEP_RF_DIV, "samples_per_RF": SAMPLES_PER_RF,
                   "escape_event": "first outward crossing r=8 mm; retained states are right-censored at 1000 RF"},
        "diagnostics": ["x,y,vx,vy", "r,vr", "H", "Lz", "kinetic work by spatial harmonic",
                        "explicit Hamiltonian work by spatial harmonic", "stroboscopic states at integer RF periods"],
        "compute_budget": {"max_elapsed_wall_s": MAX_WALL_S, "workers_max": WORKERS_MAX,
                           "checkpoint": "completed trajectory NPZ plus atomic JSON and fsync JSONL journal",
                           "new_FEM_solves": 0},
        "validation": validation,
        "commands": ["python run_v7.py --stage validate", "python run_v7.py --stage pilot --workers 4",
                     "python run_v7.py --stage boundary --workers 4", "python run_v7_validation.py",
                     "python analyze_v7.py", "python analyze_pseudopotential_cross_terms.py",
                     "python render_v7_figures.py", "powershell -ExecutionPolicy Bypass -File reproduce_v7.ps1"],
    }
    _atomic_json(OUT / "config_v7.json", config)
    print(json.dumps({"validation_passed": validation["passed"], "harmonics": modes35,
                      "max_gradient_error": gradient_error, "m4_amplitude_mismatch": amp_mismatch}, indent=2))
    if not validation["passed"]:
        raise SystemExit("static field validation failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "pilot", "boundary", "late"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "validate":
        validate_static()
    else:
        run_stage(args.stage, args.workers)


if __name__ == "__main__":
    main()
