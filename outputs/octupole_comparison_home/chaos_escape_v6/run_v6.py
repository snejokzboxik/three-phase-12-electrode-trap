"""Checkpointed V6 phase-space escape study using the validated h35 N48 field.

The analytical m=4 control is obtained by retaining only the measured fourth
complex-potential coefficient in each RF quadrature. It therefore matches the
FEM/N48 leading harmonic and its phase convention exactly.
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
V5_CSV = ROOT / "outputs/octupole_comparison_home/matched_pseudopotential_v5/processed/matched_strength_balanced_pairs.csv"
FIELD35 = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h35_P4/diagnostic_N48.json"
sys.path.insert(0, str(V4))
import run_v4  # noqa: E402

OUT = HERE
ION_MZ_TH = 100_000.0
FREQUENCY_HZ = 100_000.0
ROTATING_VOLTAGE_V = 250.0
STANDING_VOLTAGE_V = 353.5533905932738
HORIZON_RF = 1_000
APERTURE_M = 8e-3
RTOL = 2e-9
ATOL = (2e-13, 2e-13, 2e-11, 2e-11)
MAX_STEP_DIV = 50
BUDGET_S = 90 * 60
DEADLINE_RESERVE_S = 75
REPRESENTATIONS = ("N48", "pure_m4")
MODELS = ("rotating", "standing")

_Q_N48 = None
_Q_PURE = None


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _job_id(job):
    payload = {k: job[k] for k in sorted(job) if k not in ("job_id", "stage", "sampling_design")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _phase_stage(stage):
    if stage in ("pilot", "scan", "phase-check", "refine"):
        return stage
    raise ValueError(stage)


def _make_job(stage, rep, model, az_deg, phase_deg, direction, *, sampling="balanced"):
    return {
        "stage": stage, "representation": rep, "model": model,
        "field": "h35_N48" if rep == "N48" else "measured_leading_m4",
        "normalization": "matched_leading_pseudopotential",
        "voltage_V": ROTATING_VOLTAGE_V if model == "rotating" else STANDING_VOLTAGE_V,
        "frequency_Hz": FREQUENCY_HZ, "mz_Th": ION_MZ_TH, "charge_state": 1,
        "energy_eV": 2.0, "r0_mm": 1.0,
        "azimuth_deg": float(az_deg), "phase_deg": float(phase_deg),
        "direction": int(direction), "velocity_kind": "tangent",
        "horizon_RF": HORIZON_RF, "aperture_mm": 8.0,
        "rtol": RTOL, "atol": list(ATOL), "max_step_RF_fraction": 1 / MAX_STEP_DIV,
        "samples_per_RF": 8, "sampling_design": sampling,
    }


def jobs_for_stage(stage):
    jobs = []
    if stage == "pilot":
        azimuths = (-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 11.25, 22.5)
        reps = ("N48",)
        phases = (0.0,)
        sample = "pilot_offsets_and_midpoint"
    elif stage == "scan":
        azimuths = np.linspace(0.0, 22.5, 65).tolist()
        reps = REPRESENTATIONS
        phases = (0.0,)
        sample = "balanced_65_point_azimuth_scan"
    elif stage == "phase-check":
        azimuths = {0.0, 5.0, 11.25, 22.5}
        # Add one representative midpoint for each standing-wave transition
        # family at positive tangential direction. Both velocity signs and
        # both requested phases are then tested at these localized angles.
        for rep in REPRESENTATIONS:
            for lo, hi in _transition_brackets(rep, "standing", 1):
                azimuths.add((lo + hi) / 2)
        azimuths = tuple(sorted(azimuths))
        reps = REPRESENTATIONS
        phases = (90.0, 180.0)
        sample = "selected_phase_robustness_scan"
    elif stage == "refine":
        jobs = []
        for rep in REPRESENTATIONS:
            for model in MODELS:
                for direction in (-1, 1):
                    for lo, hi in _transition_brackets(rep, model, direction, max_width=0.025):
                        az = (lo + hi) / 2
                        j = _make_job(stage, rep, model, az, 0.0, direction,
                                      sampling="adaptive_midpoint_of_observed_status_change")
                        j["job_id"] = _job_id(j)
                        jobs.append(j)
        return jobs
    else:
        raise ValueError(stage)
    for rep in reps:
        for model in MODELS:
            for phase in phases:
                for az in azimuths:
                    for direction in (-1, 1):
                        j = _make_job(stage, rep, model, az, phase, direction, sampling=sample)
                        j["job_id"] = _job_id(j)
                        jobs.append(j)
    return jobs


def _transition_brackets(rep, model, direction, max_width=0.0):
    done = read_done()
    rows = []
    for r in done.values():
        j = r.get("job", {})
        if (j.get("representation") == rep and j.get("model") == model
                and int(j.get("direction", 0)) == direction
                and float(j.get("phase_deg", -999)) == 0.0
                and 0.0 <= float(j.get("azimuth_deg", -999)) <= 22.5
                and r.get("status") in ("retained", "escaped")):
            rows.append((float(j["azimuth_deg"]), r["status"]))
    rows = sorted(set(rows))
    brackets = []
    for (a, sa), (b, sb) in zip(rows, rows[1:]):
        if sa != sb and b - a > max_width + 1e-12:
            brackets.append((a, b))
    return brackets


def load_v5_reuse():
    with V5_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def try_v5_reuse(job, rows):
    if job["representation"] != "N48" or job["energy_eV"] != 2.0 or job["r0_mm"] != 1.0:
        return None
    if job["azimuth_deg"] not in (0.0, 22.5) or job["phase_deg"] not in (0.0, 90.0, 180.0, 270.0):
        return None
    match = [r for r in rows if float(r["energy_eV"]) == 2.0 and float(r["r0_mm"]) == 1.0
             and float(r["azimuth_deg"]) == job["azimuth_deg"]
             and float(r["phase_deg"]) == job["phase_deg"]
             and int(float(r["direction"])) == job["direction"]]
    if len(match) != 1:
        return None
    row = match[0]
    model = job["model"]
    if (row["field"] != "h35" or row["normalization"] != "matched_leading_pseudopotential_at_0p5mm"
            or float(row["mz_Th"]) != ION_MZ_TH or int(float(row["charge_state"])) != 1
            or float(row["frequency_Hz"]) != FREQUENCY_HZ or int(float(row["horizon_RF"])) != HORIZON_RF
            or float(row[model + "_voltage_V"]) != job["voltage_V"]
            or row[model + "_solver_status"] in ("", "failed", "paused")):
        return None
    status = row[model + "_status"]
    escape = row[model + "_escape_time_RF"]
    max_r = row[model + "_max_radius_mm"]
    return {
        "job": job, "job_id": job["job_id"], "status": status,
        "escape_time_RF": None if not escape else float(escape),
        "right_censored_at_RF": HORIZON_RF if status == "retained" else None,
        "max_radius_mm": float(max_r), "rms_radius_mm": None,
        "initial_state": run_v4.make_initial(ION_MZ_TH, 1.0, 2000.0,
            math.radians(job["azimuth_deg"]), math.radians(job["phase_deg"]),
            float(job["direction"]), "tangent").tolist(),
        "solver_status": "reused_exact_v5_h35_n48_record",
        "source": "matched_strength_balanced_pairs.csv",
        "samples_per_RF_source": 1,
        "nfev": None, "wall_s": 0.0,
    }


def _init_worker():
    global _Q_N48, _Q_PURE
    os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    _, _Q_N48 = run_v4.load_fields(FIELD35)
    _Q_PURE = {}
    for model in MODELS:
        if model == "rotating":
            base = _Q_N48
            c = np.zeros(5, dtype=complex); s = np.zeros(5, dtype=complex)
            c[4] = base.cosine[4]; s[4] = base.sine[4]
        else:
            base = _Q_N48
            c = np.zeros(5, dtype=complex); s = np.zeros(5, dtype=complex)
            s[4] = base.sine[4]
        from three_phase_trap.rf_reconstruction import HarmonicQuadratures
        _Q_PURE[model] = HarmonicQuadratures(c, s, base.reference_radius)


def integrate_job(job, *, save_trace=False):
    q = _Q_N48 if job["representation"] == "N48" else _Q_PURE[job["model"]]
    model = job["model"]
    if model == "rotating":
        dc = np.polynomial.polynomial.polyder(q.cosine)
        ds = np.polynomial.polynomial.polyder(q.sine)
    else:
        dc = np.zeros(1, dtype=complex)
        ds = np.polynomial.polynomial.polyder(q.sine)
    inv_ref = 1.0 / q.reference_radius
    mass = job["mz_Th"] * run_v4.AMU
    qm = run_v4.E_CHARGE / mass
    omega = 2 * math.pi * job["frequency_Hz"]
    voltage = job["voltage_V"]
    az = math.radians(job["azimuth_deg"])
    phase0 = math.radians(job["phase_deg"])
    initial = run_v4.make_initial(job["mz_Th"], job["r0_mm"], job["energy_eV"] * 1000.0,
                                  az, phase0, float(job["direction"]), "tangent")
    r0 = math.hypot(initial[0], initial[1])
    vtan = (-initial[2] * initial[1] + initial[3] * initial[0]) / r0
    if job["direction"] * vtan <= 0:
        raise ValueError("initial tangential velocity sign check failed")

    def rhs(t, state):
        z = complex(state[0], state[1]) * inv_ref
        dcos = np.polynomial.polynomial.polyval(z, dc) * inv_ref
        dsin = np.polynomial.polynomial.polyval(z, ds) * inv_ref
        lam = omega * t + phase0
        co, si = math.cos(lam), math.sin(lam)
        return (state[2], state[3],
                qm * voltage * (-dcos.real * co - dsin.real * si),
                qm * voltage * (dcos.imag * co + dsin.imag * si))

    def aperture(_t, state):
        return APERTURE_M - math.hypot(state[0], state[1])
    aperture.terminal = True
    aperture.direction = -1
    horizon_s = HORIZON_RF / job["frequency_Hz"]
    samples = HORIZON_RF * job["samples_per_RF"]
    t_eval = np.linspace(0.0, horizon_s, samples + 1)
    run_rtol = float(job.get("rtol", RTOL))
    run_atol = tuple(job.get("atol", ATOL))
    max_step_div = int(round(1.0 / float(job.get("max_step_RF_fraction", 1.0 / MAX_STEP_DIV))))
    tic = time.perf_counter()
    sol = solve_ivp(rhs, (0.0, horizon_s), initial, method="DOP853", rtol=run_rtol, atol=run_atol,
                    max_step=1.0 / (job["frequency_Hz"] * max_step_div), t_eval=t_eval,
                    events=aperture)
    wall = time.perf_counter() - tic
    if not sol.success:
        raise RuntimeError(sol.message)
    times, states = sol.t, sol.y.T
    escaped = bool(len(sol.t_events[0]))
    escape_s = float(sol.t_events[0][0]) if escaped else None
    if escaped:
        times = np.r_[times, escape_s]
        states = np.vstack((states, sol.y_events[0][0]))
    radius = np.hypot(states[:, 0], states[:, 1])
    rms = math.sqrt(float(np.trapezoid(radius * radius, times) / max(times[-1] - times[0], 1e-30)))
    result = {
        "job": job, "job_id": job["job_id"], "status": "escaped" if escaped else "retained",
        "escape_time_RF": escape_s * job["frequency_Hz"] if escaped else None,
        "right_censored_at_RF": None if escaped else HORIZON_RF,
        "max_radius_mm": float(np.max(radius) * 1e3), "rms_radius_mm": rms * 1e3,
        "initial_state": initial.tolist(), "solver_status": sol.message,
        "nfev": int(sol.nfev), "wall_s": wall, "samples_per_RF_source": job["samples_per_RF"],
    }
    if save_trace:
        result["trace"] = {"time_RF": (times * job["frequency_Hz"]).tolist(), "state": states.tolist()}
    return result


def _worker(job):
    return integrate_job(job)


def read_done():
    done = {}
    path = OUT / "trajectory_checkpoint.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("status") in ("retained", "escaped"):
                    done[row["job_id"]] = row
            except Exception:
                continue
    return done


def _mark_deadline():
    p = OUT / "integration_deadline.json"
    if not p.exists():
        now = time.time()
        _atomic_json(p, {"start_unix_s": now, "deadline_unix_s": now + BUDGET_S,
                         "budget_s": BUDGET_S, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                         "start_rule": "created immediately before the first new V6 RF integration"})
    return json.loads(p.read_text(encoding="utf-8"))["deadline_unix_s"]


def _append_result(result):
    OUT.mkdir(parents=True, exist_ok=True)
    comp = OUT / "completed" / (result["job_id"] + ".json")
    _atomic_json(comp, result)
    p = OUT / "trajectory_checkpoint.jsonl"
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
        f.flush(); os.fsync(f.fileno())
    print(f"completed {result['job']['stage']} {result['job']['representation']} {result['job']['model']} "
          f"az={result['job']['azimuth_deg']:+.4f} deg phase={result['job']['phase_deg']:.0f} "
          f"dir={result['job']['direction']:+d}: {result['status']} "
          f"te={result['escape_time_RF'] if result['escape_time_RF'] is not None else 'censored'} "
          f"wall={result['wall_s']:.2f}s", flush=True)


def _write_csv(rows):
    path = OUT / "processed" / "trajectory_results_v6.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["job_id", "stage", "representation", "model", "normalization", "voltage_V", "frequency_Hz",
              "mz_Th", "charge_state", "energy_eV", "r0_mm", "azimuth_deg", "phase_deg", "direction",
              "velocity_kind", "x0_m", "y0_m", "vx0_m_s", "vy0_m_s", "status", "escape_time_RF",
              "right_censored_at_RF", "max_radius_mm", "rms_radius_mm", "solver_status", "rtol",
              "max_step_RF_fraction", "samples_per_RF_source", "nfev", "wall_s", "source"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            j = r["job"]; state = r["initial_state"]
            w.writerow({"job_id": r["job_id"], "stage": j["stage"], "representation": j["representation"],
                "model": j["model"], "normalization": j["normalization"], "voltage_V": j["voltage_V"],
                "frequency_Hz": j["frequency_Hz"], "mz_Th": j["mz_Th"], "charge_state": j["charge_state"],
                "energy_eV": j["energy_eV"], "r0_mm": j["r0_mm"], "azimuth_deg": j["azimuth_deg"],
                "phase_deg": j["phase_deg"], "direction": j["direction"], "velocity_kind": j["velocity_kind"],
                "x0_m": state[0], "y0_m": state[1], "vx0_m_s": state[2], "vy0_m_s": state[3],
                "status": r["status"], "escape_time_RF": r["escape_time_RF"],
                "right_censored_at_RF": r["right_censored_at_RF"], "max_radius_mm": r["max_radius_mm"],
                "rms_radius_mm": r.get("rms_radius_mm"), "solver_status": r["solver_status"],
                "rtol": j["rtol"], "max_step_RF_fraction": j["max_step_RF_fraction"],
                "samples_per_RF_source": r.get("samples_per_RF_source"), "nfev": r.get("nfev"),
                "wall_s": r.get("wall_s"), "source": r.get("source", "new V6 integration")})


def run_stage(stage, workers=4):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "completed").mkdir(exist_ok=True)
    jobs = jobs_for_stage(_phase_stage(stage))
    existing = read_done()
    reuse = load_v5_reuse()
    todo = []
    for job in jobs:
        if job["job_id"] in existing:
            continue
        cached = try_v5_reuse(job, reuse)
        if cached is not None:
            _append_result(cached)
            existing[cached["job_id"]] = cached
        else:
            todo.append(job)
    if todo:
        deadline = _mark_deadline()
        print(f"stage={stage} queued={len(todo)} worker_count={workers} deadline_unix_s={deadline:.0f}", flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            pending = {}
            next_idx = 0
            while next_idx < len(todo) or pending:
                while next_idx < len(todo) and len(pending) < workers:
                    if time.time() + DEADLINE_RESERVE_S >= deadline:
                        print("deadline reserve reached; no further integrations will be submitted", flush=True)
                        next_idx = len(todo)
                        break
                    job = todo[next_idx]
                    pending[pool.submit(_worker, job)] = job
                    next_idx += 1
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    job = pending.pop(future)
                    result = future.result()
                    _append_result(result)
                    existing[result["job_id"]] = result
            if next_idx < len(todo):
                print(f"left incomplete at deadline: {len(todo)-next_idx} jobs", flush=True)
    all_rows = list(read_done().values())
    _write_csv(all_rows)
    print(f"stage={stage} complete_rows={len(all_rows)} requested={len(jobs)}", flush=True)


def dry_validation():
    _init_worker()
    q = _Q_N48
    c4, s4 = q.cosine[4], q.sine[4]
    amp_rot = abs(c4) / q.reference_radius**4
    amp_st = abs(s4) / q.reference_radius**4
    mismatch = abs(amp_rot - amp_st) / max(amp_rot, amp_st)
    qpr = _Q_PURE["rotating"]
    qps = _Q_PURE["standing"]
    xys = np.array([[0.72e-3, 1.13e-3], [-2.1e-3, 0.44e-3], [3.2e-3, -1.7e-3]])
    max_gradient_rel = 0.0
    h = 1e-9
    for quad in (qpr, qps):
        for angle in (0.13, 1.7, 4.2):
            for xy in xys:
                pc, ps = quad.potential_quadratures(np.array([xy + [h, 0], xy - [h, 0], xy + [0, h], xy - [0, h]]))
                ec, es = quad.field_quadratures(xy[None, :])
                for vals, efield in ((pc, ec[0]), (ps, es[0])):
                    grad = np.array([(vals[0] - vals[1]) / (2*h), (vals[2] - vals[3]) / (2*h)])
                    max_gradient_rel = max(max_gradient_rel, float(np.linalg.norm(grad + efield) / max(1.0, np.linalg.norm(efield))))
    ratio_pseudopotential = (STANDING_VOLTAGE_V**2 * amp_st**2 / 2) / (ROTATING_VOLTAGE_V**2 * amp_rot**2)
    # Exact q4 time law written in angular form: phi_rot=A r^4 sin(lambda-4theta);
    # phi_stand=A r^4 cos(4theta) sin(lambda), lambda=omega*t+phi.
    theta = np.array([0.0, 0.17, 0.9])
    tphase = np.linspace(0, 2*math.pi, 512, endpoint=False)
    rot_norm = []
    stand_norm = []
    helix_error = 0.0
    for th in theta:
        xy = np.array([[0.73e-3 * math.cos(th), 0.73e-3 * math.sin(th)]])
        erc, ers = qpr.field_quadratures(xy)
        esc, ess = qps.field_quadratures(xy)
        phases = tphase
        er = ROTATING_VOLTAGE_V * (erc[0][None, :] * np.cos(phases)[:, None]
                                   + ers[0][None, :] * np.sin(phases)[:, None])
        es = STANDING_VOLTAGE_V * (esc[0][None, :] * np.cos(phases)[:, None]
                                   + ess[0][None, :] * np.sin(phases)[:, None])
        rot_norm.append(float(np.mean(np.sum(er * er, axis=1))))
        stand_norm.append(float(np.mean(np.sum(es * es, axis=1))))
        for lam in (0.27, 1.19, 2.83):
            xy_t = np.array([[0.73e-3 * math.cos(th), 0.73e-3 * math.sin(th)]])
            phi_t = qpr.potential_quadratures(xy_t)[0][0] * math.cos(lam) + qpr.potential_quadratures(xy_t)[1][0] * math.sin(lam)
            shifted = th - lam / 4
            xy_0 = np.array([[0.73e-3 * math.cos(shifted), 0.73e-3 * math.sin(shifted)]])
            phi_0 = qpr.potential_quadratures(xy_0)[0][0]
            # At general phase lambda the cosine/sine quadratures rotate as one pattern.
            helix_error = max(helix_error, abs(phi_t - phi_0) / max(abs(phi_t), abs(phi_0), 1e-30))
    numeric_pseudo_ratio = float(np.mean(stand_norm) / np.mean(rot_norm))
    harmonic_spectrum=[]
    for n in range(1,len(q.cosine)):
        ar=math.sqrt(abs(q.cosine[n])**2+abs(q.sine[n])**2)
        ast=abs(q.sine[n])
        if max(ar,ast)>1e-10:
            ratios={}
            for rmm in (0.5,1.0,2.0,4.0,8.0):
                rho=rmm*1e-3/q.reference_radius
                ratios[str(rmm)]={
                    "rotating_combined_field_relative_to_m4":float((n*ar/(4*math.sqrt(abs(c4)**2+abs(s4)**2)))*rho**(n-4)),
                    "standing_field_relative_to_m4":float((n*ast/(4*abs(s4)))*rho**(n-4))}
            harmonic_spectrum.append({"spatial_order_m":n,"normalized_rotating_potential_coefficient":float(ar),
                "normalized_standing_potential_coefficient":float(ast),"field_contribution_ratios_by_radius_mm":ratios})
    test = {"passed": bool(mismatch < 1e-10 and abs(ratio_pseudopotential - 1) < 1e-10
            and abs(numeric_pseudo_ratio - 1) < 1e-10 and max_gradient_rel < 2e-6 and helix_error < 1e-10),
            "field_source": str(FIELD35.relative_to(ROOT)), "reference_radius_m": q.reference_radius,
            "m4_cosine_coefficient": [float(c4.real), float(c4.imag)],
            "m4_sine_coefficient": [float(s4.real), float(s4.imag)],
            "m4_amplitude_V_m4_per_electrode_V": float((amp_rot + amp_st) / 2),
            "relative_rotating_vs_standing_m4_amplitude_mismatch": float(mismatch),
            "matched_pseudopotential_ratio_standing_over_rotating": float(ratio_pseudopotential),
            "numerical_time_averaged_pseudopotential_ratio_standing_over_rotating": numeric_pseudo_ratio,
            "time_average_E2_rotating_per_test_position_V2_m2": rot_norm,
            "time_average_E2_standing_per_test_position_V2_m2": stand_norm,
            "rotating_pattern_helicity_test_relative_error": helix_error,
            "max_analytical_field_gradient_relative_error": max_gradient_rel,
            "measured_h35_N48_spatial_harmonics":harmonic_spectrum,
            "phase_convention": {"lambda": "Omega*t + initial_RF_phase",
                "rotating_potential_per_applied_V": "A4*r^4*sin(lambda - 4*theta)",
                "standing_potential_per_applied_V": "A4*r^4*cos(4*theta)*sin(lambda)",
                "equivalent_requested_forms": "rotating cos(4theta-Omega*t+pi/2); standing cos(4theta)cos(Omega*t-pi/2) at phase zero"}}
    _atomic_json(OUT / "processed" / "analytical_field_validation.json", test)
    config = {
        "study": "V6 azimuth-dependent escape and rotating-frame dynamics",
        "scientific_scope": "deterministic finite-time transverse dynamics; no inference of infinite-time stability or population probabilities",
        "source_studies": {
            "V4_report": "outputs/octupole_comparison_home/confinement_maps_v4/ОТЧЁТ_V4_RU.md",
            "V4_runner": "outputs/octupole_comparison_home/confinement_maps_v4/run_v4.py",
            "V4_pairs": "outputs/octupole_comparison_home/confinement_maps_v4/matched_pairs_equal_peak.csv",
            "V5_report": "outputs/octupole_comparison_home/matched_pseudopotential_v5/REPORT_RU.md",
            "V5_pairs": "outputs/octupole_comparison_home/matched_pseudopotential_v5/processed/matched_strength_balanced_pairs.csv",
            "field_h35": str(FIELD35.relative_to(ROOT)),
            "field_h50": "outputs/abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json",
        },
        "ion": {"charge_state": 1, "mz_Th": ION_MZ_TH, "mass_kg": ION_MZ_TH * run_v4.AMU},
        "rf": {"frequency_Hz": FREQUENCY_HZ, "omega_rad_s": 2 * math.pi * FREQUENCY_HZ,
               "rotating_voltage_V": ROTATING_VOLTAGE_V, "standing_voltage_V": STANDING_VOLTAGE_V,
               "normalization": "matched measured leading m=4 pseudopotential; other harmonics not matched"},
        "initial_conditions": {"energy_eV": 2.0, "radius_mm": 1.0, "aperture_mm": 8.0,
            "horizon_RF": HORIZON_RF, "primary_azimuth_grid_deg": {"start": 0, "end": 22.5, "samples": 65},
            "directions": [-1, 1], "pilot_offsets_deg": [-5, -1, -0.1, 0, 0.1, 1, 5, 11.25, 22.5],
            "phase_check_deg": [90, 180]},
        "solver": {"method": "DOP853", "rtol": RTOL, "atol": list(ATOL),
                   "max_step_RF_fraction": 1 / MAX_STEP_DIV, "radial_output_samples_per_RF": 8,
                   "escape_event": "first terminal outward crossing of r=8 mm"},
        "analytical_control": {"potential_order": 4, "amplitude_calibration": "complex m=4 coefficient in h35 N48 potential quadratures",
            "rotating_potential_per_V": "A4*r^4*sin(Omega*t + phi - 4*theta)",
            "standing_potential_per_V": "A4*r^4*cos(4*theta)*sin(Omega*t + phi)"},
        "compute_budget": {"max_elapsed_wall_s": BUDGET_S, "workers": 4,
            "deadline_file": "integration_deadline.json", "checkpoint": "atomic per-trajectory JSON plus fsync JSONL journal",
            "no_new_fem_or_mesh_calculation": True},
        "run_commands": ["python run_v6.py --stage validate", "python run_v6.py --stage pilot --workers 4",
                         "python run_v6.py --stage scan --workers 4", "python run_v6.py --stage refine --workers 4",
                         "python run_v6.py --stage phase-check --workers 4"],
    }
    _atomic_json(OUT / "study_config_v6.json", config)
    if not test["passed"]:
        raise AssertionError(json.dumps(test, indent=2))
    print(json.dumps(test, indent=2), flush=True)
    return test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "pilot", "scan", "phase-check", "refine"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    if args.stage == "validate":
        dry_validation()
    else:
        run_stage(args.stage, args.workers)


if __name__ == "__main__":
    main()
