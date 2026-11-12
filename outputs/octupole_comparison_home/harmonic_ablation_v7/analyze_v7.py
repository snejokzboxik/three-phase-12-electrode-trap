"""Reduce V7 harmonic-ablation checkpoints to reproducible scientific tables."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V6 = ROOT / "outputs/octupole_comparison_home/chaos_escape_v6"
sys.path.insert(0, str(HERE))
import run_v7


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    unique = {}
    for row in rows:
        unique[row["job_id"]] = row
    return list(unique.values())


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(value):
    return "" if value is None else value


def source_models():
    q35, q50 = run_v7._load_source_fields()
    models35, indices = run_v7._make_q_models(q35)
    models50, _ = run_v7._make_q_models(q50)
    return q35, q50, models35, models50, indices


def effective_potential_tables(q35, q50, models35, models50, indices):
    mass = run_v7.MASS_MZ_TH * run_v7.run_v4.AMU
    charge = run_v7.CHARGE_STATE * run_v7.run_v4.E_CHARGE
    radii_mm = (1.0, 2.0, 4.0, 6.0, 7.0, 8.0)
    angles = np.linspace(0.0, 2 * math.pi, 1440, endpoint=False)
    summary = []
    detail = []
    arrays = {}
    for resolution, qdict in (("h35", models35), ("h50", models50)):
        for rep in run_v7.MODEL_SET:
            for model in ("rotating", "standing"):
                q = qdict[(rep, model)]
                voltage = run_v7.VOLTAGE[model]
                curve_values = []
                for rmm in radii_mm:
                    x = rmm * 1e-3 * np.cos(angles)
                    y = rmm * 1e-3 * np.sin(angles)
                    points = np.column_stack((x, y))
                    psi = q.pseudo(points, charge, mass, run_v7.OMEGA) * voltage**2 / charge
                    psi_ev = np.asarray(psi, float)
                    curve_values.append(psi_ev)
                    imin, imax = int(np.argmin(psi_ev)), int(np.argmax(psi_ev))
                    summary.append({
                        "field_resolution": resolution,
                        "representation": rep,
                        "model": model,
                        "radius_mm": rmm,
                        "mean_pseudopotential_eV": float(np.mean(psi_ev)),
                        "minimum_pseudopotential_eV": float(psi_ev[imin]),
                        "minimum_angle_deg": float(math.degrees(angles[imin]) % 360),
                        "maximum_pseudopotential_eV": float(psi_ev[imax]),
                        "maximum_angle_deg": float(math.degrees(angles[imax]) % 360),
                        "peak_to_peak_over_mean": float((np.max(psi_ev)-np.min(psi_ev))/max(np.mean(psi_ev),1e-30)),
                        "sampled_angles": len(angles),
                    })
                    # Preserve only the aperture angular cuts used in the
                    # publication figure. The full fitted coefficients and
                    # angular extrema remain available for all radii/models.
                    if resolution == "h35" and abs(rmm-8.0) < 1e-12:
                        for theta, val in zip(angles, psi_ev):
                            detail.append({"field_resolution": resolution, "representation": rep,
                                           "model": model, "radius_mm": rmm,
                                           "angle_deg": float(math.degrees(theta)),
                                           "pseudopotential_eV": float(val)})
                arrays[f"{resolution}_{rep}_{model}"] = np.asarray(curve_values)

    write_csv(HERE / "processed" / "effective_pseudopotential_summary.csv", summary,
              ["field_resolution", "representation", "model", "radius_mm", "mean_pseudopotential_eV",
               "minimum_pseudopotential_eV", "minimum_angle_deg", "maximum_pseudopotential_eV",
               "maximum_angle_deg", "peak_to_peak_over_mean", "sampled_angles"])
    write_csv(HERE / "processed" / "effective_pseudopotential_angular_samples.csv", detail,
              ["field_resolution", "representation", "model", "radius_mm", "angle_deg", "pseudopotential_eV"])
    np.savez_compressed(HERE / "processed" / "effective_pseudopotential_curves.npz",
                        radii_mm=np.asarray(radii_mm), angles_deg=np.degrees(angles), **arrays)

    # Local adiabaticity estimate eta = q ||grad E||/(M Omega^2), using the
    # operator norm of the field Jacobian. This is a diagnostic of pseudopotential
    # applicability, not a universal stability criterion.
    eta_rows = []
    for rep in run_v7.MODEL_SET:
        for model in ("rotating", "standing"):
            q = models35[(rep, model)]
            for rmm in (1.0, 4.0, 6.0, 8.0):
                a = np.linspace(0, 2*math.pi, 256, endpoint=False)
                pts = np.column_stack((rmm*1e-3*np.cos(a), rmm*1e-3*np.sin(a)))
                ec, es = q.field_quadratures(pts)
                jc, js = q.jacobian_quadratures(pts)
                vals = []
                for ph in np.linspace(0, 2*math.pi, 64, endpoint=False):
                    e = run_v7.VOLTAGE[model] * (ec*np.cos(ph)+es*np.sin(ph))
                    j = run_v7.VOLTAGE[model] * (jc*np.cos(ph)+js*np.sin(ph))
                    # largest singular value at each sampled angle
                    norms = np.linalg.svd(j, compute_uv=False)[:, 0]
                    vals.extend((charge*norms/(mass*run_v7.OMEGA**2)).tolist())
                vals = np.asarray(vals)
                eta_rows.append({"representation": rep, "model": model, "radius_mm": rmm,
                                 "eta_min_sampled": float(np.min(vals)),
                                 "eta_median_sampled": float(np.median(vals)),
                                 "eta_max_sampled": float(np.max(vals)),
                                 "interpretation": "local pseudopotential adiabaticity diagnostic; not a stability proof"})
    write_csv(HERE / "processed" / "adiabaticity_diagnostic.csv", eta_rows,
              ["representation", "model", "radius_mm", "eta_min_sampled", "eta_median_sampled",
               "eta_max_sampled", "interpretation"])

    # Compare the angular pseudopotential summaries between h35 and h50 fits.
    keyed = {}
    for row in summary:
        key = (row["representation"], row["model"], row["radius_mm"])
        keyed.setdefault(key, {})[row["field_resolution"]] = row
    conv = []
    for (rep, model, radius), vv in keyed.items():
        if set(vv) != {"h35", "h50"}:
            continue
        a, b = vv["h35"], vv["h50"]
        conv.append({"representation": rep, "model": model, "radius_mm": radius,
                     "mean_delta_h50_minus_h35_eV": b["mean_pseudopotential_eV"]-a["mean_pseudopotential_eV"],
                     "min_delta_h50_minus_h35_eV": b["minimum_pseudopotential_eV"]-a["minimum_pseudopotential_eV"],
                     "max_delta_h50_minus_h35_eV": b["maximum_pseudopotential_eV"]-a["maximum_pseudopotential_eV"],
                     "relative_min_change": (b["minimum_pseudopotential_eV"]-a["minimum_pseudopotential_eV"])/max(abs(a["minimum_pseudopotential_eV"]),1e-30)})
    write_csv(HERE / "processed" / "h35_h50_pseudopotential_sensitivity.csv", conv,
              ["representation", "model", "radius_mm", "mean_delta_h50_minus_h35_eV",
               "min_delta_h50_minus_h35_eV", "max_delta_h50_minus_h35_eV", "relative_min_change"])
    # Dense radial values are direct evaluations of the fitted potential and
    # the sampled angular minimum; points are calculated, not interpolated.
    dense_rows = []
    dense_radii_mm = np.linspace(0.5, 8.0, 151)
    dense_angles = np.linspace(0.0, 2 * math.pi, 1440, endpoint=False)
    for rep in run_v7.MODEL_SET:
        for model in ("rotating", "standing"):
            q = models35[(rep, model)]
            voltage = run_v7.VOLTAGE[model]
            for rmm in dense_radii_mm:
                points = np.column_stack((rmm*1e-3*np.cos(dense_angles), rmm*1e-3*np.sin(dense_angles)))
                psi = np.asarray(q.pseudo(points, charge, mass, run_v7.OMEGA) * voltage**2 / charge, float)
                imin = int(np.argmin(psi))
                dense_rows.append({"representation": rep, "model": model, "radius_mm": float(rmm),
                                   "minimum_pseudopotential_eV": float(psi[imin]),
                                   "mean_pseudopotential_eV": float(np.mean(psi)),
                                   "maximum_pseudopotential_eV": float(np.max(psi)),
                                   "minimum_angle_deg": float(math.degrees(dense_angles[imin])),
                                   "sampled_angles": len(dense_angles)})
    write_csv(HERE / "processed" / "effective_pseudopotential_radial_minima.csv", dense_rows,
              ["representation", "model", "radius_mm", "minimum_pseudopotential_eV",
               "mean_pseudopotential_eV", "maximum_pseudopotential_eV", "minimum_angle_deg", "sampled_angles"])
    return summary, eta_rows, conv


def trace_summaries(checkpoints, models35, indices):
    event_rows, work_rows, poincare_rows, exposure_rows = [], [], [], []
    trace_cache = {}
    for row in checkpoints:
        job = row["job"]
        path = ROOT / row["trace_file"]
        with np.load(path) as data:
            t = data["time_RF"]
            state = data["state"]
            rmm = data["radius_m"] * 1e3
            vr = data["radial_velocity_m_s"]
            kinetic = data["kinetic_eV"]
            hamiltonian = data["hamiltonian_eV"]
            lz = data["angular_momentum_kg_m2_s"]
            wk = data["work_kinetic_by_mode_eV"]
            wh = data["work_hamiltonian_by_mode_eV"]
            modes = data["harmonic_orders"].astype(int)
            phi = data["potential_by_mode_eV"]
        qs = models35[(job["representation"], job["model"])]
        voltage = job["voltage_V"]
        charge = run_v7.CHARGE_STATE * run_v7.run_v4.E_CHARGE
        mass = run_v7.MASS_MZ_TH * run_v7.run_v4.AMU
        psi_path = qs.pseudo(state[:, :2], charge, mass, run_v7.OMEGA) * voltage**2 / charge
        theta_deg = np.degrees(np.arctan2(state[:,1],state[:,0])) % 360.0
        reach = {}
        for threshold in (2.0, 4.0, 6.0, 7.0, 7.9, 8.0):
            ix = np.flatnonzero(rmm >= threshold)
            reach[f"first_r{threshold:g}_mm_RF"] = float(t[ix[0]]) if len(ix) else None
            reach[f"theta_at_first_r{threshold:g}_mm_deg"] = float(theta_deg[ix[0]]) if len(ix) else None
        # Spatial-harmonic field-envelope ratios along this exact trajectory.
        env_by_m = {}
        z = (state[:, 0] + 1j*state[:, 1]) / qs.reference_radius
        for m in modes:
            dc = m * qs.cosine[m] * z**(m-1) / qs.reference_radius
            ds = m * qs.sine[m] * z**(m-1) / qs.reference_radius
            ec = np.column_stack((-dc.real, dc.imag))
            es = np.column_stack((-ds.real, ds.imag))
            env_by_m[int(m)] = np.sqrt(np.sum(ec*ec+es*es, axis=1))
        base4 = np.maximum(env_by_m.get(4, np.zeros(len(t))), 1e-99)
        max_ratios = {}
        first_ratios = {}
        for m in modes:
            ratio = env_by_m[int(m)] / base4
            max_ratios[str(int(m))] = float(np.max(ratio))
            for cutoff in (0.1, 0.5, 1.0):
                ix = np.flatnonzero(ratio >= cutoff)
                first_ratios[f"first_E{int(m)}_over_E4_ge_{cutoff:g}_RF"] = float(t[ix[0]]) if len(ix) else None
        final_wk = {str(int(m)): float(wk[-1, i]) for i, m in enumerate(modes)}
        final_wh = {str(int(m)): float(wh[-1, i]) for i, m in enumerate(modes)}
        event = {
            "job_id": row["job_id"], "launch_label": job["launch_label"],
            "representation": job["representation"], "model": job["model"],
            "status": row["status"], "escape_time_RF": row["escape_time_RF"],
            "max_radius_mm": row["max_radius_mm"], "rms_radius_mm": row["rms_radius_mm"],
            "peak_kinetic_energy_eV": float(np.max(kinetic)),
            "kinetic_energy_at_end_eV": float(kinetic[-1]),
            "hamiltonian_at_end_eV": float(hamiltonian[-1]),
            "hamiltonian_range_eV": float(np.max(hamiltonian)-np.min(hamiltonian)),
            "Lz_start_kg_m2_s": float(lz[0]), "Lz_end_kg_m2_s": float(lz[-1]),
            "max_abs_radial_velocity_m_s": float(np.max(np.abs(vr))),
            "pseudopotential_along_path_min_eV": float(np.min(psi_path)),
            "pseudopotential_along_path_max_eV": float(np.max(psi_path)),
            "pseudopotential_at_end_eV": float(psi_path[-1]),
            "theta_at_end_deg": float(theta_deg[-1]),
            "kinetic_work_closure_eV": row["energy_work_closure_error_eV"],
            "hamiltonian_work_closure_eV": row["hamiltonian_work_closure_error_eV"],
            "final_kinetic_work_by_mode_eV": json.dumps(final_wk, sort_keys=True),
            "final_explicit_H_work_by_mode_eV": json.dumps(final_wh, sort_keys=True),
            "max_field_envelope_ratio_to_m4": json.dumps(max_ratios, sort_keys=True),
            **reach, **first_ratios,
        }
        event_rows.append(event)
        for i, m in enumerate(modes):
            work_rows.append({"job_id": row["job_id"], "launch_label": job["launch_label"],
                              "representation": job["representation"], "model": job["model"],
                              "harmonic_m": int(m), "final_kinetic_work_eV": float(wk[-1, i]),
                              "final_explicit_hamiltonian_work_eV": float(wh[-1, i]),
                              "status": row["status"], "escape_time_RF": row["escape_time_RF"]})
        # Integer-period stroboscopic state; data are at 8 samples per RF period.
        for i, trf in enumerate(t):
            nearest = round(float(trf))
            if abs(float(trf)-nearest) < 1e-7:
                x, y, vx, vy = state[i]
                poincare_rows.append({"job_id": row["job_id"], "launch_label": job["launch_label"],
                                      "representation": job["representation"], "model": job["model"],
                                      "status": row["status"], "time_RF": nearest,
                                      "x_mm": x*1e3, "y_mm": y*1e3, "vx_m_s": vx, "vy_m_s": vy,
                                      "radius_mm": rmm[i], "radial_velocity_m_s": vr[i],
                                      "kinetic_energy_eV": kinetic[i], "hamiltonian_eV": hamiltonian[i],
                                      "angular_momentum_kg_m2_s": lz[i], "pseudopotential_eV": psi_path[i]})
        exposure_rows.append({"job_id": row["job_id"], "launch_label": job["launch_label"],
                              "representation": job["representation"], "model": job["model"],
                              "max_pseudopotential_sampled_along_path_eV": float(np.max(psi_path)),
                              "min_pseudopotential_sampled_along_path_eV": float(np.min(psi_path)),
                              "maximum_sampled_radius_mm": float(np.max(rmm)),
                              "max_field_envelope_ratios": json.dumps(max_ratios, sort_keys=True)})
        trace_cache[row["job_id"]] = (job, t, state, rmm, kinetic, hamiltonian, lz, psi_path, wk, wh, modes)

    fields = list(event_rows[0]) if event_rows else []
    write_csv(HERE / "processed" / "trajectory_event_diagnostics.csv", event_rows, fields)
    write_csv(HERE / "processed" / "harmonic_work_by_trajectory.csv", work_rows,
              ["job_id", "launch_label", "representation", "model", "harmonic_m",
               "final_kinetic_work_eV", "final_explicit_hamiltonian_work_eV", "status", "escape_time_RF"])
    write_csv(HERE / "processed" / "poincare_sections.csv", poincare_rows,
              ["job_id", "launch_label", "representation", "model", "status", "time_RF", "x_mm", "y_mm",
               "vx_m_s", "vy_m_s", "radius_mm", "radial_velocity_m_s", "kinetic_energy_eV",
               "hamiltonian_eV", "angular_momentum_kg_m2_s", "pseudopotential_eV"])
    write_csv(HERE / "processed" / "trajectory_pseudopotential_exposure.csv", exposure_rows,
              ["job_id", "launch_label", "representation", "model", "max_pseudopotential_sampled_along_path_eV",
               "min_pseudopotential_sampled_along_path_eV", "maximum_sampled_radius_mm", "max_field_envelope_ratios"])
    return event_rows, work_rows, trace_cache


def v6_reproduction(checkpoints):
    with (V6 / "processed" / "trajectory_results_v6.csv").open(encoding="utf-8-sig", newline="") as f:
        old = list(csv.DictReader(f))
    rows = []
    for new in checkpoints:
        j = new["job"]
        if j["representation"] not in ("full_N48", "pure_m4"):
            continue
        old_rep = "N48" if j["representation"] == "full_N48" else "pure_m4"
        matches = [r for r in old if r["representation"] == old_rep and r["model"] == j["model"]
                   and abs(float(r["azimuth_deg"])-j["azimuth_deg"]) < 1e-10
                   and int(float(r["direction"])) == j["direction"] and float(r["phase_deg"]) == 0.0
                   and float(r["energy_eV"]) == run_v7.ENERGY_EV and float(r["r0_mm"]) == run_v7.R0_MM]
        if len(matches) != 1:
            continue
        oldrow = matches[0]
        dt = None
        if oldrow["escape_time_RF"] and new["escape_time_RF"]:
            dt = float(new["escape_time_RF"])-float(oldrow["escape_time_RF"])
        rows.append({"job_id": new["job_id"], "launch_label": j["launch_label"],
                     "representation_v7": j["representation"], "model": j["model"],
                     "azimuth_deg": j["azimuth_deg"], "direction": j["direction"],
                     "v6_status": oldrow["status"], "v7_status": new["status"],
                     "status_agrees": oldrow["status"] == new["status"],
                     "v6_escape_time_RF": oldrow["escape_time_RF"],
                     "v7_escape_time_RF": new["escape_time_RF"],
                     "delta_escape_time_v7_minus_v6_RF": dt,
                     "v6_source": oldrow.get("source", "")})
    write_csv(HERE / "processed" / "v6_reproduction.csv", rows,
              ["job_id", "launch_label", "representation_v7", "model", "azimuth_deg", "direction",
               "v6_status", "v7_status", "status_agrees", "v6_escape_time_RF", "v7_escape_time_RF",
               "delta_escape_time_v7_minus_v6_RF", "v6_source"])
    return rows


def validation_tables(checkpoints, pseudo_convergence):
    by_key = {}
    for r in checkpoints:
        j = r["job"]
        key = (j["launch_label"], j["representation"], j["model"], j.get("field_source"),
               float(j["azimuth_deg"]), int(j["direction"]))
        by_key[key] = r
    strict_pairs = []
    h50_pairs = []
    for r in checkpoints:
        j=r["job"]
        if j["stage"] == "strict_h35":
            key=(j["launch_label"],j["representation"],j["model"],"h35_N48_polynomial",float(j["azimuth_deg"]),int(j["direction"]))
            ref=next((x for x in checkpoints if x["job"]["stage"] in ("pilot","boundary")
                      and (x["job"]["launch_label"],x["job"]["representation"],x["job"]["model"],
                           x["job"].get("field_source"),float(x["job"]["azimuth_deg"]),int(x["job"]["direction"]))==key),None)
            if ref:
                dt=None if not r["escape_time_RF"] or not ref["escape_time_RF"] else float(r["escape_time_RF"])-float(ref["escape_time_RF"])
                strict_pairs.append({"launch_label":j["launch_label"],"representation":j["representation"],"model":j["model"],
                    "standard_status":ref["status"],"strict_status":r["status"],"classification_agrees":ref["status"]==r["status"],
                    "standard_escape_RF":ref["escape_time_RF"],"strict_escape_RF":r["escape_time_RF"],"delta_strict_minus_standard_RF":dt,
                    "strict_rtol":j["rtol"],"strict_max_step_RF_fraction":j["max_step_RF_fraction"]})
        if j.get("field_resolution")=="h50":
            key=(j["launch_label"],j["representation"],j["model"],"h35_N48_polynomial",float(j["azimuth_deg"]),int(j["direction"]))
            ref=next((x for x in checkpoints if x["job"]["stage"] in ("pilot","boundary")
                      and (x["job"]["launch_label"],x["job"]["representation"],x["job"]["model"],
                           x["job"].get("field_source"),float(x["job"]["azimuth_deg"]),int(x["job"]["direction"]))==key),None)
            if ref:
                dt=None if not r["escape_time_RF"] or not ref["escape_time_RF"] else float(r["escape_time_RF"])-float(ref["escape_time_RF"])
                h50_pairs.append({"launch_label":j["launch_label"],"representation":j["representation"],"model":j["model"],
                    "h35_status":ref["status"],"h50_status":r["status"],"classification_agrees":ref["status"]==r["status"],
                    "h35_escape_RF":ref["escape_time_RF"],"h50_escape_RF":r["escape_time_RF"],"delta_h50_minus_h35_RF":dt})
    write_csv(HERE/"processed"/"strict_tolerance_comparison.csv",strict_pairs,
              ["launch_label","representation","model","standard_status","strict_status","classification_agrees",
               "standard_escape_RF","strict_escape_RF","delta_strict_minus_standard_RF","strict_rtol","strict_max_step_RF_fraction"])
    write_csv(HERE/"processed"/"h35_h50_trajectory_comparison.csv",h50_pairs,
              ["launch_label","representation","model","h35_status","h50_status","classification_agrees",
               "h35_escape_RF","h50_escape_RF","delta_h50_minus_h35_RF"])
    return strict_pairs,h50_pairs


def integral_diagnostics(checkpoints):
    selected = [
        ("deep_az0_dplus","pure_m4","rotating","pilot"),
        ("early_az22p5_dplus","full_N48","rotating","pilot"),
        ("early_az22p5_dplus","full_N48","standing","pilot"),
    ]
    out=[]
    charge=run_v7.CHARGE_STATE*run_v7.run_v4.E_CHARGE
    omega=run_v7.OMEGA
    for launch,rep,model,stage in selected:
        row=next(r for r in checkpoints if r["job"]["launch_label"]==launch and r["job"]["representation"]==rep
                 and r["job"]["model"]==model and r["job"]["stage"]==stage)
        with np.load(ROOT/row["trace_file"]) as data:
            h=data["hamiltonian_eV"]
            lz=data["angular_momentum_kg_m2_s"]
            t=data["time_RF"]
            k=h-omega/4*lz/charge
        out.append({"launch_label":launch,"representation":rep,"model":model,"status":row["status"],
                    "escape_time_RF":row["escape_time_RF"],
                    "K_definition_eV":"H_eV - Omega*Lz/(4*elementary_charge_C)",
                    "K_initial_eV":float(k[0]),"K_max_abs_drift_eV":float(np.max(abs(k-k[0]))),
                    "K_relative_max_drift":float(np.max(abs(k-k[0]))/max(abs(k[0]),run_v7.ENERGY_EV)),
                    "H_peak_to_peak_eV":float(np.max(h)-np.min(h)),
                    "Lz_peak_to_peak_kg_m2_s":float(np.max(lz)-np.min(lz)),
                    "time_samples":int(len(t)),
                    "interpretation":"Only the pure rotating m=4 case has the continuous rotational symmetry required for exact K conservation."})
    write_csv(HERE/"processed"/"integral_diagnostics_v7.csv",out,
              ["launch_label","representation","model","status","escape_time_RF","K_definition_eV","K_initial_eV",
               "K_max_abs_drift_eV","K_relative_max_drift","H_peak_to_peak_eV","Lz_peak_to_peak_kg_m2_s",
               "time_samples","interpretation"])
    return out


def main():
    checkpoints = read_jsonl(HERE / "trajectory_checkpoint.jsonl")
    if not checkpoints:
        raise SystemExit("No V7 trajectory checkpoints found")
    q35, q50, models35, models50, indices = source_models()
    pseudo, eta, convergence = effective_potential_tables(q35, q50, models35, models50, indices)
    event_rows, work_rows, traces = trace_summaries(checkpoints, models35, indices)
    reproduction = v6_reproduction(checkpoints)
    strict_rows,h50_rows=validation_tables(checkpoints,convergence)
    invariants=integral_diagnostics(checkpoints)
    escaped = sum(row["status"] == "escaped" for row in checkpoints)
    summary = {
        "study": "V7 harmonic ablation",
        "checkpointed_trajectories": len(checkpoints),
        "escaped": escaped,
        "retained_right_censored_at_1000_RF": len(checkpoints)-escaped,
        "all_initial_velocity_signs_verified": all(row["verified_initial_vtheta_m_s"]*row["job"]["direction"] > 0 for row in checkpoints),
        "max_abs_kinetic_work_closure_error_eV": max(abs(row["energy_work_closure_error_eV"]) for row in checkpoints),
        "max_abs_hamiltonian_work_closure_error_eV": max(abs(row["hamiltonian_work_closure_error_eV"]) for row in checkpoints),
        "v6_reproduction_pairs": len(reproduction),
        "v6_classifications_agree": sum(r["status_agrees"] for r in reproduction),
        "v6_max_abs_escape_time_delta_RF": max((abs(float(r["delta_escape_time_v7_minus_v6_RF"]))
            for r in reproduction if r["delta_escape_time_v7_minus_v6_RF"] not in (None,"")), default=None),
        "strict_h35_classification_agreements": sum(r["classification_agrees"] for r in strict_rows),
        "strict_h35_cases": len(strict_rows),
        "strict_h35_max_abs_escape_time_delta_RF": max((abs(r["delta_strict_minus_standard_RF"]) for r in strict_rows
            if r["delta_strict_minus_standard_RF"] is not None),default=None),
        "h35_h50_classification_agreements": sum(r["classification_agrees"] for r in h50_rows),
        "h35_h50_cases": len(h50_rows),
        "h35_h50_max_abs_escape_time_delta_RF": max((abs(r["delta_h50_minus_h35_RF"]) for r in h50_rows
            if r["delta_h50_minus_h35_RF"] is not None),default=None),
        "pure_rotating_m4_jacobi_max_abs_drift_eV": next(r["K_max_abs_drift_eV"] for r in invariants
            if r["representation"]=="pure_m4" and r["model"]=="rotating"),
        "global_deadline": json.loads((HERE/"integration_deadline.json").read_text(encoding="utf-8")),
        "interpretation_scope": "finite-time deterministic outcomes; adaptive/selected starts are not population samples",
    }
    (HERE / "processed" / "analysis_summary_v7.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
