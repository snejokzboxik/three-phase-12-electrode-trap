"""User step 8: fixed-time U_s=s*U, f_s=s*f high-frequency test.

Reuses verified s=1 smooth pilots and B, without overwriting them. New RF
trajectories retain every harmonic of the same reconstructed scalar potential.
Raw P2 trajectories are not a converged reference (see user step 7).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.averaged_dynamics import AveragedRFModel
from three_phase_trap.micromotion import matched_full_state_with_drift
from three_phase_trap.pilot_dynamics import PolynomialRF, integrate_rf, coordinate_metrics, oscillation_metrics
from three_phase_trap.slow_analysis import centered_period_mean, local_rf_trend
from run_abc_pilot import load_models, sha, write, valid_checkpoint, checkpoint_paths
from analyze_abc_pilot import csv_write, inverse_batches, relative_period_difference

OUTPUT = ROOT/"outputs/abc_validation/frequency_limit"
PILOT = OUTPUT.parent/"pilot"


def scaled_model(base, multiplier):
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Expected a positive frequency multiplier")
    return AveragedRFModel(base.reconstruction, base.charge, base.mass,
                           base.amplitude*multiplier, base.frequency*multiplier)


def sample_at_centres(time_values, values, target_time):
    """Select exact nested time nodes, rejecting extrapolation or phase shifts."""
    indices = np.searchsorted(time_values, target_time)
    indices = np.minimum(indices, len(time_values)-1)
    previous = np.maximum(indices-1, 0)
    indices = np.where(abs(time_values[previous]-target_time) < abs(time_values[indices]-target_time), previous, indices)
    if np.max(abs(time_values[indices]-target_time)) > 1e-12:
        raise ValueError("Target times must be nested in the centred output grid")
    return values[indices]


def paths(case):
    tag = f"A_smooth_s{case['s']}_phase{case['phase_index']}_{case['level']}"
    return OUTPUT/(tag+".npz"), OUTPUT/(tag+".json")


def signature(case, settings):
    return hashlib.sha256(json.dumps({"case": case, "settings": settings}, sort_keys=True).encode()).hexdigest()


def valid(case, settings):
    arrays, metadata = paths(case)
    if not arrays.exists() or not metadata.exists():
        return False
    info = json.loads(metadata.read_text(encoding="utf-8"))
    if info["signature"] != signature(case, settings) or info["arrays_sha256"] != sha(arrays):
        raise ValueError(f"Checkpoint inputs/hash mismatch: {arrays.name}")
    return info["status"] == "completed"


def worker(case):
    settings = json.loads((OUTPUT/"settings.json").read_text(encoding="utf-8"))
    protocol, initial, base, _ = load_models()
    model = scaled_model(base, case["s"])
    p = protocol["pilot"]
    phase = p["phase_rad"][case["phase_index"]]
    state = matched_full_state_with_drift(model.quadratures, p["slow_position_m"], p["slow_velocity_m_per_s"],
                                         model.charge, model.mass, model.omega, phase)
    control = settings["controls"][case["level"]]
    result = integrate_rf(PolynomialRF(model.quadratures), state, model.charge, model.mass, model.frequency,
                          phase, 0., settings["duration_s"], output_samples_per_period=settings["samples_per_RF_period"],
                          max_steps_per_period=control["RF_steps_per_period"], rtol=control["RF_rtol"],
                          atol=control["RF_atol_SI"], radius=model.reconstruction.valid_radius)
    arrays, metadata = paths(case)
    temp = arrays.with_suffix(".tmp.npz")
    np.savez_compressed(temp, time_s=result.time, states=result.states)
    temp.replace(arrays)
    info = {"case": case, "status": result.status, "signature": signature(case, settings),
            "arrays_sha256": sha(arrays), "nfev": result.nfev, "wall_seconds": result.wall_seconds,
            "frequency_Hz": model.frequency, "amplitude_V": model.amplitude,
            "full_initial_state": state.tolist(), "slow_initial_position_m": p["slow_position_m"],
            "slow_initial_velocity_m_per_s": p["slow_velocity_m_per_s"], "RF_phase_rad": phase,
            "exit_time_s": result.exit_time, "end_time_s": float(result.time[-1]),
            "max_radius_m": float(np.linalg.norm(result.states[:, :2], axis=-1).max()),
            "state_order": ["x_m", "y_m", "vx_m_per_s", "vy_m_per_s"]}
    write(metadata, info)
    return {"file": arrays.name, "status": result.status, "nfev": result.nfev, "seconds": result.wall_seconds}


def load_case(case, settings):
    if case["s"] == 1:
        old_case = {"model": "A_smooth", "phase_index": case["phase_index"], "level": case["level"], "extent": "pilot"}
        old_settings = json.loads((PILOT/"settings.json").read_text(encoding="utf-8"))
        if not valid_checkpoint(old_case, old_settings):
            raise ValueError("Base pilot checkpoint is not valid")
        arrays, metadata = checkpoint_paths(old_case)
        if settings["base_checkpoint_sha256"][arrays.name] != sha(arrays):
            raise ValueError("Changed s=1 checkpoint")
    else:
        if not valid(case, settings):
            raise ValueError(f"Missing checkpoint {case}")
        arrays, metadata = paths(case)
    with np.load(arrays) as data:
        t, states = data["time_s"], data["states"]
    if abs(t[-1]-settings["duration_s"]) > 1e-13:
        raise ValueError("Frequency runs must have identical physical duration")
    return t, states


def prepare():
    protocol, initial, base, c = load_models()
    old = json.loads((PILOT/"settings.json").read_text(encoding="utf-8"))
    for group in ("kernel_source_sha256", "input_sha256"):
        for name, expected in old["numerical_provenance"][group].items():
            if sha(ROOT/name) != expected:
                raise ValueError(f"Base pilot no longer matches current source/input: {name}")
    kernel_names = ("pilot_dynamics.py", "micromotion.py", "slow_analysis.py", "rf_reconstruction.py", "averaged_dynamics.py")
    input_names = ("protocol.json", "initial_conditions.json", "model_b/reconstruction.json", "model_c/coefficients.json")
    import scipy
    settings = {"user_step": 8, "detailed_plan_stage": 7, "multipliers": [1, 2, 4],
                "duration_s": old["RF_periods"]["pilot"]/base.frequency,
                "estimated_slow_period_s": old["estimated_slow_period_s"],
                "base_RF_periods": old["RF_periods"]["pilot"], "samples_per_RF_period": 64,
                "R_scale_m": old["R_scale_m"], "controls": old["controls"],
                "phases_rad": protocol["pilot"]["phase_rad"], "initial_map": "drift_corrected",
                "common_comparison_margin_s": 1.5/base.frequency,
                "common_comparison_step_s": 1/(64*base.frequency),
                "field": "unchanged smooth potential reconstruction, all modes through n=48",
                "fem_fingerprint": protocol["fem_fingerprint"], "basis_sha256": protocol["basis_sha256"],
                "raw_A_status": "inconclusive: step-7 raw P2 trajectories did not converge; no new raw A trajectory claim",
                "base_checkpoint_sha256": {},
                "provenance": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                               "kernels": {n: sha(ROOT/"src/three_phase_trap"/n) for n in kernel_names},
                               "inputs": {n: sha(OUTPUT.parent/n) for n in input_names},
                               "worker_sha256": hashlib.sha256((inspect.getsource(worker)+inspect.getsource(scaled_model)).encode()).hexdigest()}}
    for phase in (0, 1):
        generated = matched_full_state_with_drift(base.quadratures, protocol["pilot"]["slow_position_m"],
                                                protocol["pilot"]["slow_velocity_m_per_s"], base.charge, base.mass,
                                                base.omega, protocol["pilot"]["phase_rad"][phase])
        if not np.allclose(generated, initial["cases"][phase]["A_and_A_smooth_full_state"], rtol=0., atol=1e-18):
            raise ValueError("Initial map differs from the base pilot")
        for level in ("coarse", "fine"):
            case = {"model": "A_smooth", "phase_index": phase, "level": level, "extent": "pilot"}
            if not valid_checkpoint(case, old):
                raise ValueError("Cannot reuse an invalid base pilot")
            array, _ = checkpoint_paths(case)
            settings["base_checkpoint_sha256"][array.name] = sha(array)
    for level in ("coarse", "fine"):
        case = {"model": "B", "phase_index": 0, "level": level, "extent": "pilot"}
        if not valid_checkpoint(case, old):
            raise ValueError("Cannot reuse an invalid B trajectory")
        array, _ = checkpoint_paths(case)
        settings["base_checkpoint_sha256"][array.name] = sha(array)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/"settings.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Frequency settings changed; preserve outputs and use a versioned study")
    write(path, settings)
    return settings


def invariance_audit(settings):
    """Fixed spatial samples: RF-fit discrepancy is not a spatial error bound."""
    from three_phase_trap.config import load_config
    from three_phase_trap.fem_cache import load_fem_cache, fem_fingerprint
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    protocol, initial, base, _ = load_models()
    if fem_fingerprint(config) != protocol["fem_fingerprint"] or sha(ROOT/"outputs/fem_cache/p2_basis.npz") != settings["basis_sha256"]:
        raise ValueError("FEM inputs changed")
    _, _, _, interp = load_fem_cache(config, ROOT)
    theta = 2*np.pi*np.arange(721)/721
    points = .001*np.column_stack((np.cos(theta), np.sin(theta)))
    ea, eb, ec = (interp.field_basis(points, name) for name in ("A", "B", "C"))
    raw_base = (np.sqrt(3)*base.amplitude*(ec-eb)/2, base.amplitude*(ea-eb/2-ec/2))
    rows, initial_rows = [], []
    angles = np.arange(128)*2*np.pi/128
    for s in settings["multipliers"]:
        model = scaled_model(base, s)
        smooth = model.quadratures.field_quadratures(points)
        raw = tuple(s*v for v in raw_base)
        raw_scale = np.mean(sum(np.sum(v*v, axis=-1) for v in raw))
        relative_difference = np.sqrt(np.mean(sum(np.sum((a-b)**2, axis=-1) for a, b in zip(raw, smooth)))/raw_scale)
        psi_raw = model.charge**2*sum(np.sum(v*v, axis=-1) for v in raw)/(4*model.mass*model.omega**2)
        potential_change = np.max(abs(model.potential(points)-base.potential(points)))/np.max(abs(base.potential(points)))
        force_change = np.max(np.linalg.norm(model.force(points)-base.force(points), axis=-1))/np.max(np.linalg.norm(base.force(points), axis=-1))
        jc, js = model.quadratures.jacobian_quadratures(np.array(protocol["pilot"]["slow_position_m"]))
        jac = np.cos(angles)[:, None, None]*jc+np.sin(angles)[:, None, None]*js
        eta = 2*abs(model.charge)/(model.mass*model.omega**2)*np.linalg.norm(jac, ord=2, axis=(-2, -1)).max()
        e0c, e0s = model.quadratures.field_quadratures(np.array(protocol["pilot"]["slow_position_m"]))
        e0 = np.cos(angles)[:, None]*e0c+np.sin(angles)[:, None]*e0s
        xi = abs(model.charge)/(model.mass*model.omega**2)*np.linalg.norm(e0, axis=-1).max()
        rows.append({"s": s, "frequency_Hz": model.frequency, "amplitude_V": model.amplitude,
                     "potential_relative_change": float(potential_change), "force_relative_change": float(force_change),
                     "micromotion_leading_amplitude_at_R0_m": float(xi), "eta_J_sampled_at_R0": float(eta),
                     "raw_fit_vector_RMS_relative_at_1mm": float(relative_difference),
                     "raw_fit_potential_RMS_relative_at_1mm": float(np.sqrt(np.mean((psi_raw-model.potential(points))**2))/np.sqrt(np.mean(psi_raw**2))),
                     "raw_A_trajectory_status": "inconclusive_from_step7_not_recomputed"})
        if max(potential_change, force_change) > 1e-12:
            raise RuntimeError("U/f fixed must preserve leading potential and force")
        for index, phase in enumerate(settings["phases_rad"]):
            state = matched_full_state_with_drift(model.quadratures, protocol["pilot"]["slow_position_m"],
                                                 protocol["pilot"]["slow_velocity_m_per_s"], model.charge, model.mass, model.omega, phase)
            initial_rows.append({"s": s, "phase_index": index, "phase_rad": phase, "full_state": state.tolist(),
                                 "slow_position_m": protocol["pilot"]["slow_position_m"],
                                 "slow_velocity_m_per_s": protocol["pilot"]["slow_velocity_m_per_s"]})
    csv_write(OUTPUT/"scaling_checks.csv", rows)
    write(OUTPUT/"initial_conditions.json", {"map": "drift_corrected", "cases": initial_rows})
    return rows


def analyze(settings):
    protocol, initial, base, c = load_models()
    tolerance = protocol["predeclared_trajectory_tolerances"]
    budget = {"relative_rms": tolerance["relative_position_rms"]*tolerance["numerical_budget_fraction"],
              "relative_max": tolerance["relative_position_max"]*tolerance["numerical_budget_fraction"],
              "relative_period": tolerance["relative_period"]*tolerance["numerical_budget_fraction"]}
    N, scale, period = settings["samples_per_RF_period"], settings["R_scale_m"], settings["estimated_slow_period_s"]
    slow = {}
    for level in ("coarse", "fine"):
        path = PILOT/f"B_phase0_{level}_pilot.npz"
        if sha(path) != settings["base_checkpoint_sha256"][path.name]:
            raise ValueError("Changed B input")
        with np.load(path) as data:
            slow[level] = (data["time_s"], data["states"], data["energy_J"])
    # All s and phases use the same physical times, with room for 3*T_base windows.
    base_time = slow["fine"][0]
    common_time = base_time[3*N//2:-3*N//2]
    bpos = slow["fine"][1][3*N//2:-3*N//2, :2]
    bcoarse = slow["coarse"][1][3*N//2:-3*N//2, :2]
    bperiod = oscillation_metrics(common_time, bpos, period)
    bnum = coordinate_metrics(bcoarse, bpos, scale)
    bnum["relative_period"] = relative_period_difference(oscillation_metrics(common_time, bcoarse, period), bperiod)
    b_ok = all(bnum[k] is not None and bnum[k] <= budget[k] for k in budget)
    rows, numerical, filters, saved_positions = [], [], [], {}
    for phase_index, phase in enumerate(settings["phases_rad"]):
        for s in settings["multipliers"]:
            model = scaled_model(base, s)
            positions, full_fine = {}, None
            for level in ("coarse", "fine"):
                t, states = load_case({"s": s, "phase_index": phase_index, "level": level}, settings)
                mt, means = centered_period_mean(states[:, :2], t, N)
                positions[level] = sample_at_centres(mt, means, common_time)
                if level == "fine":
                    full_fine = (t, states)
            num = coordinate_metrics(positions["coarse"], positions["fine"], scale)
            fine_period = oscillation_metrics(common_time, positions["fine"], period)
            num["relative_period"] = relative_period_difference(oscillation_metrics(common_time, positions["coarse"], period), fine_period)
            ode_ok = all(num[k] is not None and num[k] <= budget[k] for k in budget)
            numerical.append({"s": s, "phase_index": phase_index, **num, "passed": ode_ok})
            t, states = full_fine
            variants = {}
            for window in (2, 3):
                wt, wm = centered_period_mean(states[:, :2], t, N*window)
                variants[f"window_{window}_vs_1"] = (sample_at_centres(wt, wm, common_time), positions["fine"], common_time)
            t32, m32 = centered_period_mean(states[::2, :2], t[::2], N//2)
            # The s=1 32-sample series contains alternate base output nodes.
            variants["resolution_32_vs_64"] = (sample_at_centres(t32, m32, common_time[::2]), positions["fine"][::2], common_time[::2])
            trend_t, trend = local_rf_trend(states[:, :2], t, N, 2)
            variants["local_trend_vs_mean"] = (sample_at_centres(trend_t, trend, common_time), positions["fine"], common_time)
            filter_ok = True
            max_filter_rms, max_filter_error = 0., 0.
            for name, (values, reference, variant_time) in variants.items():
                m = coordinate_metrics(values, reference, scale)
                m["relative_period"] = relative_period_difference(oscillation_metrics(variant_time, values, period), fine_period)
                ok = all(m[k] is not None and m[k] <= budget[k] for k in budget)
                filters.append({"s": s, "phase_index": phase_index, "check": name, **m, "passed": ok})
                max_filter_rms, max_filter_error = max(max_filter_rms, m["relative_rms"]), max(max_filter_error, m["relative_max"])
                filter_ok = filter_ok and ok
            # Independent truncated-coordinate extraction on 8 base samples/RF period.
            inverse_time = common_time[::8]
            raw_selected = sample_at_centres(t, states, inverse_time)
            inverse = inverse_batches(model, raw_selected, inverse_time, phase)
            inv = coordinate_metrics(inverse[:, :2], positions["fine"][::8], scale)
            inv["relative_period"] = relative_period_difference(oscillation_metrics(inverse_time, inverse[:, :2], period), fine_period)
            inv_ok = all(inv[k] is not None and inv[k] <= budget[k] for k in budget)
            filters.append({"s": s, "phase_index": phase_index, "check": "inverse_drift_vs_mean", **inv, "passed": inv_ok})
            max_filter_rms, max_filter_error = max(max_filter_rms, inv["relative_rms"]), max(max_filter_error, inv["relative_max"])
            filter_ok = filter_ok and inv_ok
            err = coordinate_metrics(positions["fine"], bpos, scale)
            pe = relative_period_difference(fine_period, bperiod)
            physical_ok = err["relative_rms"] <= tolerance["relative_position_rms"] and err["relative_max"] <= tolerance["relative_position_max"] and pe is not None and pe <= tolerance["relative_period"]
            numerics_ok = ode_ok and b_ok and filter_ok
            rows.append({"s": s, "phase_index": phase_index, "frequency_Hz": model.frequency, "amplitude_V": model.amplitude,
                         **err, "relative_period_error": pe, "period_s": fine_period["mean_period_s"],
                         "numerical_RMS_change_sum": num["relative_rms"]+bnum["relative_rms"],
                         "max_filter_RMS_change": max_filter_rms, "max_filter_max_change": max_filter_error,
                         "numerics_passed": numerics_ok, "status": ("passed" if physical_ok else "failed") if numerics_ok else "inconclusive"})
            saved_positions[f"A_smooth_s{s}_phase{phase_index}"] = positions["fine"]
            print(json.dumps(rows[-1]), flush=True)
            del states, full_fine, variants
    csv_write(OUTPUT/"metrics.csv", rows)
    csv_write(OUTPUT/"numerical_checks.csv", numerical)
    csv_write(OUTPUT/"filter_checks.csv", filters)
    # Common grid only: no duplicate full-resolution filtered arrays per variant.
    np.savez_compressed(OUTPUT/"comparison.npz", time_s=common_time, B_position_m=bpos, **saved_positions)
    trends = []
    for index in (0, 1):
        subset = [r for r in rows if r["phase_index"] == index]
        errors = np.array([r["relative_rms"] for r in subset])
        slopes = (-np.diff(np.log(errors))/np.log(2)).tolist()
        trends.append({"phase_index": index, "RMS_decreases": bool(np.all(np.diff(errors) < 0)),
                       "observed_pairwise_RMS_orders": slopes,
                       "all_numerics_passed": all(r["numerics_passed"] for r in subset),
                       "errors_resolved_above_numerical_and_filter_changes": all(r["relative_rms"] > 10*max(r["numerical_RMS_change_sum"], r["max_filter_RMS_change"]) for r in subset)})
    scaling = invariance_audit(settings)
    passed = all(r["RMS_decreases"] and r["all_numerics_passed"] and r["errors_resolved_above_numerical_and_filter_changes"] for r in trends)
    bundle = ROOT/"outputs/fem_cache/gmsh_70um_fem_bundle.zip"
    bundle_expected = json.loads(bundle.with_name("bundle_checksum.json").read_text(encoding="utf-8"))["sha256"]
    bundle_actual = sha(bundle)
    if bundle_actual != bundle_expected:
        raise ValueError("Original FEM archive checksum changed")
    summary = {"status": "smooth_high_frequency_trend_verified_raw_FEM_unresolved" if passed else "frequency_limit_inconclusive",
               "smooth_trend_verified": passed, "rows": rows, "trends": trends, "B_numerical_change": bnum,
               "scaling": scaling, "scope": settings["raw_A_status"], "settings_sha256": sha(OUTPUT/"settings.json"),
               "basis_sha256_after": sha(ROOT/"outputs/fem_cache/p2_basis.npz"),
               "original_FEM_bundle_sha256_after": bundle_actual}
    write(OUTPUT/"summary.json", summary)
    plot(settings, common_time, bpos, saved_positions, rows)
    report(settings, summary)
    progress_path = OUTPUT.parent/"progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["user_step_8_frequency_limit"] = summary["status"]
    write(progress_path, progress)
    from run_abc_initial_checks import write_report
    write_report(OUTPUT.parent, protocol, progress)
    manifest = {"script_sha256": sha(Path(__file__)), "artifacts": {str(p.relative_to(OUTPUT)).replace("\\", "/"): sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "manifest.json"}}
    write(OUTPUT/"manifest.json", manifest)
    return summary


def plot(settings, t, bpos, positions, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for phase in (0, 1):
        ax = axes[0, phase]
        for s, color in zip((1, 2, 4), ("tab:blue", "tab:orange", "tab:green")):
            error = np.linalg.norm(positions[f"A_smooth_s{s}_phase{phase}"]-bpos, axis=-1)/settings["R_scale_m"]
            ax.plot(t/settings["estimated_slow_period_s"], 100*error, label=f"s={s}", color=color, linewidth=1)
        ax.axhline(3, color="gray", linestyle=":", label="max tolerance 3%")
        ax.set(xlabel="time / estimated slow period", ylabel="vector coordinate error, %", title=f"RF phase = {phase} * pi/2")
        ax.legend()
        subset = [r for r in rows if r["phase_index"] == phase]
        ss = np.array([r["s"] for r in subset])
        ax = axes[1, phase]
        rms = np.array([r["relative_rms"] for r in subset])
        ax.loglog(ss, 100*rms, "o-", label="A_smooth - B RMS")
        ax.loglog(ss, 100*np.array([r["relative_max"] for r in subset]), "s-", label="maximum")
        ax.loglog(ss, 100*np.array([r["max_filter_RMS_change"]+r["numerical_RMS_change_sum"] for r in subset]), "x--", label="ODE + filter changes (diagnostic)")
        ax.loglog(ss, 100*rms[0]/ss, ":", color="gray", label="1/s guide")
        ax.set(xlabel="s (both f and U multiplied)", ylabel="error / fixed 1 mm, %", xticks=[1, 2, 4], xticklabels=["1", "2", "4"])
        ax.legend(fontsize=8)
    fig.suptitle("Fixed geometry, slow initial state and physical duration; smooth RF only")
    fig.savefig(OUTPUT/"frequency_limit.png", dpi=170)
    plt.close(fig)


def report(settings, summary):
    lines = ["# Пункт 8: контролируемый высокочастотный предел", "", f"Статус: `{summary['status']}`.", "",
             "Серия f_s=s f, U_s=s U0 при s=1,2,4: 100/200/400 кГц и 250/500/1000 В.",
             "Геометрия, q=+1e−12 Кл, m=1e−15 кг, R0=(1 мм,0), V0=0 неизменны; DC, трение и гравитация отсутствуют.",
             f"Обе фазы 0, π/2 рассчитаны до одного физического времени {settings['duration_s']:.8g} с (около 3,25 Ts).", "",
             "## Что проверено", "",
             "Ψ=q²(|Ec|²+|Es|²)/(4mΩ²): при Ec,Es→s(Ec,Es), Ω→sΩ потенциал и сила B не меняются.",
             "Ведущая амплитуда микродвижения |qE|/(mΩ²) уменьшается как 1/s; параметр η_J также как 1/s.",
             "Здесь η_J=2|q| max_φ ||Jc cosφ+Js sinφ||₂/(mΩ²), максимум по 128 фазам при R0.",
             "Это явно определённый локальный диагностический параметр, не глобальная гарантия адиабатичности.",
             "Полные старты заново получены по drift_corrected для каждого s; одинаковы именно медленные условия.",
             "Измеренное расхождение включает остаток этой конечнопорядковой начальной карты; оно не выделяет отдельно поправку к усреднённому гамильтониану.",
             "Полная RF-скорость не обязана уменьшаться: ∂tξ имеет ведущий масштаб U/Ω=const.", "",
             "Используется то же гладкое скалярное восстановление до n=48, без удаления мод.",
             "При s=1 повторно использованы проверенные checkpoint пункта 7; B одинаков для всех s и не пересчитывался.",
             "Сравнение на общей физической сетке; с каждого конца исключены 1,5 базового RF-периода.",
             "Центрированное среднее использует собственный T_RF(s). Временные сдвиги и амплитуды не подгонялись.", "",
             "## Результаты A_smooth–B", "",
             "| s | фаза | RMS, % | максимум, % | ошибка периода, % | статус |", "|---:|---:|---:|---:|---:|---|"]
    for r in summary["rows"]:
        pe = "—" if r["relative_period_error"] is None else f"{100*r['relative_period_error']:.6g}"
        lines.append(f"| {r['s']} | {r['phase_index']}·π/2 | {100*r['relative_rms']:.6g} | {100*r['relative_max']:.6g} | {pe} | {r['status']} |")
    lines += ["", "Критерии сохранены: RMS≤1%, максимум≤3%, период≤1%; масштаб координат — фиксированный 1 мм.", "",
              "## Численная проверка", "",
              "Каждый новый RF-случай рассчитан при max_step=T/50 и T/100, rtol=2e−9 и 2e−11,",
              "atol SI для (x,y,vx,vy) соответственно (2e−13,2e−13,2e−11,2e−11) и (2e−15,2e−15,2e−13,2e−13).",
              "Выход — 64 отсчёта за RF-период. Проверены 32/64 отсчёта, окна 1/2/3 T, локальный RF-тренд",
              "и обратное drift_corrected-преобразование. Таблицы numerical_checks.csv, filter_checks.csv содержат RMS, максимум и изменение периода.",
              "Численные бюджеты: 0,1% RMS, 0,3% максимум, 0,1% период для каждого источника отдельно.",
              "Разброс способов фильтрации — диагностика, не доказанная абсолютная погрешность.", ""]
    for trend in summary["trends"]:
        lines.append(f"Фаза {trend['phase_index']}·π/2: RMS убывает={trend['RMS_decreases']}; наблюдаемые порядки "
                     f"−log₂(δ(2s)/δ(s)) = {', '.join(f'{v:.4g}' for v in trend['observed_pairwise_RMS_orders'])}; "
                     f"численные проверки пройдены={trend['all_numerics_passed']}.")
    lines += ["", "Три значения частоты проверяют тенденцию на данном интервале; строгий универсальный степенной закон не заявляется.",
              "Сводка отделяет физическую ошибку усреднения от изменений при уточнении ODE и фильтров.", "",
              "## Исходный FEM A и пространственная неопределённость", "",
              "Длительные исходные P2-траектории A не достигли ODE-сходимости в пункте 7. Здесь они не объявляются эталоном",
              "и не пересчитываются: вывод относится только к A_smooth–B. Сохранён статус inconclusive для проверки по исходному FEM.",
              "scaling_checks.csv сравнивает сырое FEM-поле и гладкое восстановление на одной окружности r=1 мм (721 угол).",
              "Относительное расхождение поля и Ψ не уменьшается при данном масштабировании: геометрия и сетка прежние.",
              "Это расхождение двух представлений, не абсолютная ошибка FEM и не оценка ошибки траектории.",
              "Увеличение частоты не заменяет проверку FEM на сетках 50/35 мкм и выяснение чувствительности исходного A.",
              "Результат пункта 7 на 10,25 Ts при s=1 остаётся в силе: рост векторной ошибки может ограничивать время применимости.",
              "Эта серия выполнена на 3,25 Ts и не доказывает согласие на произвольно долгом времени или глобальную устойчивость.", "",
              "## Воспроизведение и файлы", "", "```powershell", "C:/Python314/python.exe scripts/run_abc_frequency_limit.py --workers 2", "```", "",
              "`--stage run` только рассчитывает недостающие checkpoint; `--stage analyze` анализирует готовые.",
              "settings.json фиксирует SHA исходных данных и вычислительных ядер, параметры и длительность.",
              "initial_conditions.json содержит полные старты; новые NPZ — t,x,y,vx,vy в SI; comparison.npz — общая временная сетка и медленные координаты.",
              "scaling_checks.csv — инвариантность Ψ/силы, η_J, микродвижение и пространственное расхождение; manifest.json — SHA результатов.",
              "Исходный FEM-базис и архив не изменялись. Повторный запуск проверяет подписи и хеши перед использованием checkpoint.", ""]
    (OUTPUT/"FREQUENCY_LIMIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def preview(settings):
    """Read completed cases only; do not issue verdicts or change progress."""
    N = settings["samples_per_RF_period"]
    with np.load(PILOT/"B_phase0_fine_pilot.npz") as data:
        target = data["time_s"][3*N//2:-3*N//2]
        bpos = data["states"][3*N//2:-3*N//2, :2]
    rows = []
    for phase in (0, 1):
        for s in settings["multipliers"]:
            levels = [level for level in ("fine", "coarse") if s == 1 or valid({"s": s, "phase_index": phase, "level": level}, settings)]
            if not levels:
                continue
            case = {"s": s, "phase_index": phase, "level": levels[0]}
            t, states = load_case(case, settings)
            mt, means = centered_period_mean(states[:, :2], t, N)
            rows.append({**case, **coordinate_metrics(sample_at_centres(mt, means, target), bpos, settings["R_scale_m"])})
    print(json.dumps({"preliminary_only": True, "available": rows}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "run", "analyze", "preview"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    settings = prepare()
    print(f"Fixed physical duration: {settings['duration_s']} s; multipliers: {settings['multipliers']}", flush=True)
    if args.stage in ("all", "run"):
        invariance_audit(settings)
        cases = [{"s": s, "phase_index": phase, "level": level} for s in (2, 4) for phase in (0, 1) for level in ("coarse", "fine")]
        pending = [case for case in cases if not valid(case, settings)]
        # Start costly fine/high-frequency jobs first to avoid a long last job.
        pending.sort(key=lambda case: case["s"]*(2 if case["level"] == "fine" else 1), reverse=True)
        print(f"New/reused checkpoints: {len(pending)}/{len(cases)-len(pending)} (s=1 reused separately).", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(worker, case) for case in pending]
            for job in as_completed(jobs):
                result = job.result()
                print(json.dumps(result), flush=True)
                if result["status"] != "completed":
                    raise RuntimeError("Frequency trajectory did not complete")
    if args.stage in ("all", "analyze"):
        summary = analyze(settings)
        print(summary["status"], flush=True)
    if args.stage == "preview":
        preview(settings)


if __name__ == "__main__":
    main()
