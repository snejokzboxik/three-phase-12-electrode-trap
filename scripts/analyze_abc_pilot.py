"""Analyze saved user-step-7 trajectories without rerunning RF dynamics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.slow_analysis import centered_period_mean, local_rf_trend
from three_phase_trap.micromotion import recover_slow_state_with_drift
from three_phase_trap.pilot_dynamics import coordinate_metrics, oscillation_metrics
from run_abc_pilot import OUTPUT, load_models, checkpoint_paths, valid_checkpoint, sha, write


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_case(model, phase, level, extent, settings, prefer_events=True):
    case = {"model": model, "phase_index": phase if model in ("A", "A_smooth") else 0, "level": level, "extent": extent}
    if not valid_checkpoint(case, settings):
        raise ValueError(f"Missing complete checkpoint: {case}")
    arrays, metadata = checkpoint_paths(case)
    if model == "A" and prefer_events:
        event_arrays = OUTPUT/"edge_resolved"/f"A_phase{phase}_{level}_{extent}.npz"
        event_metadata = event_arrays.with_suffix(".json")
        other_level = "fine" if level == "coarse" else "coarse"
        other_metadata = OUTPUT/"edge_resolved"/f"A_phase{phase}_{other_level}_{extent}.json"
        if event_arrays.exists() and event_metadata.exists() and other_metadata.exists():
            value = json.loads(event_metadata.read_text(encoding="utf-8"))
            other = json.loads(other_metadata.read_text(encoding="utf-8"))
            if value["status"] == "completed" and other["status"] == "completed":
                if value["provenance"] != other["provenance"] or value["provenance"]["raw_pilot_numerical_provenance"] != settings["numerical_provenance"]:
                    raise ValueError("Event-resolved levels do not share the same numerical inputs")
                if value["arrays_sha256"] != sha(event_arrays):
                    raise ValueError("Corrupt edge-resolved checkpoint")
                arrays, metadata = event_arrays, event_metadata
                refined = OUTPUT/"edge_resolved/A_phase0_refined_pilot.npz"
                if phase == 0 and extent == "pilot" and refined.exists() and refined.with_suffix(".json").exists():
                    info = json.loads(refined.with_suffix(".json").read_text(encoding="utf-8"))
                    if info["status"] == "completed":
                        if info["arrays_sha256"] != sha(refined) or info["provenance"]["edge_resolved_source_sha256"] != value["provenance"]["edge_resolved_source_sha256"]:
                            raise ValueError("Third refinement does not match the edge-resolved field")
                        arrays = refined if level == "fine" else OUTPUT/"edge_resolved/A_phase0_fine_pilot.npz"
                        metadata = arrays.with_suffix(".json")
    with np.load(arrays) as data:
        t, states, energy = data["time_s"], data["states"], data["energy_J"]
    return t, states, energy, json.loads(metadata.read_text(encoding="utf-8"))


def relative_period_difference(a, b):
    if a["mean_period_s"] is None or b["mean_period_s"] is None:
        return None
    return abs(a["mean_period_s"]-b["mean_period_s"])/b["mean_period_s"]


def inverse_batches(b, states, t, phase):
    out = np.empty_like(states)
    for start in range(0, len(t), 4096):
        stop = start+4096
        out[start:stop] = recover_slow_state_with_drift(b.quadratures, states[start:stop], b.charge, b.mass, b.omega,
                                                       b.omega*t[start:stop]+phase, check_points=b.reconstruction.check_points)
    return out


def preview():
    """Read only completed fine checkpoints; no final verdict or progress update."""
    settings = json.loads((OUTPUT/"settings.json").read_text(encoding="utf-8"))
    rows = []
    for extent in ("pilot", "extended"):
        for phase in ((0, 1) if extent == "pilot" else (0,)):
            case = {"model": "A_smooth", "phase_index": phase, "level": "fine", "extent": extent}
            if not valid_checkpoint(case, settings):
                continue
            t, state, _, _ = load_case("A_smooth", phase, "fine", extent, settings)
            _, means = centered_period_mean(state[:, :2], t, settings["output_samples_per_RF_period"])
            half = settings["output_samples_per_RF_period"]//2
            for model in ("B", "A"):
                other_case = {"model": model, "phase_index": phase if model == "A" else 0, "level": "fine", "extent": extent}
                if valid_checkpoint(other_case, settings):
                    _, other, _, _ = load_case(model, phase, "fine", extent, settings)
                    reference = other[half:-half, :2] if model == "B" else centered_period_mean(other[:, :2], t, 2*half)[1]
                    rows.append({"extent": extent, "phase": phase, "pair": "A_smooth-"+model,
                                 **coordinate_metrics(means, reference, settings["R_scale_m"])})
                    if model == "A":
                        coarse_case = {**other_case, "level": "coarse"}
                        if valid_checkpoint(coarse_case, settings):
                            _, coarse, _, _ = load_case(model, phase, "coarse", extent, settings)
                            coarse_mean = centered_period_mean(coarse[:, :2], t, 2*half)[1]
                            rows.append({"extent": extent, "phase": phase, "pair": "A_coarse-A_fine",
                                         **coordinate_metrics(coarse_mean, reference, settings["R_scale_m"])})
    print(json.dumps({"preliminary_only": True, "available_comparisons": rows}, indent=2), flush=True)


def main(extent="all"):
    settings = json.loads((OUTPUT/"settings.json").read_text(encoding="utf-8"))
    protocol, initial, b, c = load_models()
    tol = protocol["predeclared_trajectory_tolerances"]
    budget = {"relative_rms": tol["relative_position_rms"]*tol["numerical_budget_fraction"],
              "relative_max": tol["relative_position_max"]*tol["numerical_budget_fraction"],
              "relative_period": tol["relative_period"]*tol["numerical_budget_fraction"]}
    N, scale, period = settings["output_samples_per_RF_period"], settings["R_scale_m"], settings["estimated_slow_period_s"]
    extents = ("pilot", "extended") if extent == "all" else (extent,)
    metrics_rows, numerical_rows, filter_rows, period_rows, accumulation_rows, details, ordinary_convergence = [], [], [], [], [], [], []
    artifacts = {}
    for length in extents:
        for phase_index in ((0, 1) if length == "pilot" else (0,)):
            phase = protocol["pilot"]["phase_rad"][phase_index]
            print(f"Analyzing {length}, RF phase {phase_index}...", flush=True)
            slow_positions, diagnostics, convergence, full_for_plot = {}, {}, {}, {}
            common_time = None
            raw_inputs = {}
            for model in ("B", "C", "A_smooth", "A"):
                t, states, energy, meta = load_case(model, phase_index, "fine", length, settings)
                tc, coarse, _, coarse_meta = load_case(model, phase_index, "coarse", length, settings)
                if len(tc) != len(t) or np.max(abs(tc-t)) > 1e-13:
                    raise ValueError("Coarse/fine time grids differ")
                if model in ("A", "A_smooth"):
                    times, means = centered_period_mean(states, t, N)
                    _, coarse_means = centered_period_mean(coarse, tc, N)
                    position, coarse_position = means[:, :2], coarse_means[:, :2]
                    np.savez_compressed(OUTPUT/f"slow_{model}_phase{phase_index}_{length}.npz", time_s=times, mean_states=means,
                                        effective_energy_in_B_J=b.energy(means))
                    # This effective energy is diagnostic only; RF lab energy is not conserved.
                    windows = []
                    for window in (2, 3):
                        tw, mw = centered_period_mean(states[:, :2], t, N*window)
                        offset = (window-1)*N//2
                        windows.append({"window_periods": window, **coordinate_metrics(mw, position[offset:-offset], scale)})
                    t32, mean32 = centered_period_mean(states[::2, :2], t[::2], N//2)
                    # The 32-sample grid is nested in the 64-sample output.
                    discretization = coordinate_metrics(mean32, position[::2], scale)
                    tr, trend = local_rf_trend(states[:, :2], t, N, 2)
                    trend_difference = coordinate_metrics(trend, position[N//2:-N//2], scale)
                    stride = N//8
                    selected = np.arange(N//2, len(t)-N//2, stride)
                    inverse = inverse_batches(b, states[selected], t[selected], phase)
                    inverse_difference = coordinate_metrics(inverse[:, :2], position[::stride], scale)
                    np.savez_compressed(OUTPUT/f"inverse_{model}_phase{phase_index}_{length}.npz",
                                        time_s=t[selected], inverse_states=inverse)
                    filter_row = {"extent": length, "phase_index": phase_index, "model": model,
                                  "resolution_32_vs_64_rms": discretization["relative_rms"],
                                  "window_2_vs_1_rms": windows[0]["relative_rms"], "window_3_vs_1_rms": windows[1]["relative_rms"],
                                  "local_trend_vs_mean_rms": trend_difference["relative_rms"],
                                  "inverse_vs_mean_rms": inverse_difference["relative_rms"]}
                    filter_row["within_RMS_budget"] = bool(max(filter_row[k] for k in filter_row if k.endswith("_rms")) <= budget["relative_rms"])
                    filter_rows.append(filter_row)
                    # Both preregistered phases put the phase-zero section on an exact output node.
                    offset = int(round(((-phase/(2*np.pi)) % 1)*N))
                    ids = np.arange(offset, len(t), N)
                    locked = states[ids]
                    locked_inverse = inverse_batches(b, locked, t[ids], phase)
                    phase_error = np.angle(np.exp(1j*(b.omega*t[ids]+phase)))
                    poincare = [{"time_s": float(tt), "RF_phase_error_rad": float(err),
                                 "x_full_m": float(yy[0]), "y_full_m": float(yy[1]), "vx_full_m_per_s": float(yy[2]), "vy_full_m_per_s": float(yy[3]),
                                 "x_inverse_m": float(zz[0]), "y_inverse_m": float(zz[1])}
                                for tt, err, yy, zz in zip(t[ids], phase_error, locked, locked_inverse)]
                    csv_write(OUTPUT/f"poincare_{model}_phase{phase_index}_{length}.csv", poincare)
                    raw_inputs[model] = {"filter": filter_row, "full_radii_max_m": float(np.linalg.norm(states[:, :2], axis=-1).max()),
                                         "full_ODE_coordinate_change": coordinate_metrics(coarse[:, :2], states[:, :2], scale),
                                         "phase_slice_error_max_rad": float(np.max(abs(phase_error))),
                                         "raw_integrator": "edge-resolved" if "triangle_crossings" in meta else "ordinary DOP853",
                                         "triangle_crossings_fine": meta.get("triangle_crossings")}
                else:
                    times, position, coarse_position = t[N//2:-N//2], states[N//2:-N//2, :2], coarse[N//2:-N//2, :2]
                    raw_inputs[model] = {"energy_drift_max_relative": float(np.max(abs(energy-energy[0]))/abs(energy[0]))}
                if common_time is None:
                    common_time = times
                elif len(times) != len(common_time) or np.max(abs(times-common_time)) > 1e-13:
                    raise ValueError("Model comparison times differ")
                fine_period = oscillation_metrics(times, position, period)
                coarse_period = oscillation_metrics(times, coarse_position, period)
                ode = coordinate_metrics(coarse_position, position, scale)
                period_change = relative_period_difference(coarse_period, fine_period)
                is_converged = bool(ode["relative_rms"] <= budget["relative_rms"] and ode["relative_max"] <= budget["relative_max"]
                                    and period_change is not None and period_change <= budget["relative_period"])
                convergence[model] = {**ode, "relative_period_change": period_change, "passed": is_converged}
                numerical_rows.append({"extent": length, "phase_index": phase_index, "model": model, **convergence[model],
                                       "RF_step_denominator_coarse": coarse_meta["controls"]["RF_steps_per_period"] if model in ("A", "A_smooth") else None,
                                       "RF_step_denominator_fine": meta["controls"]["RF_steps_per_period"] if model in ("A", "A_smooth") else None})
                diagnostics[model] = fine_period
                period_rows.append({"extent": length, "phase_index": phase_index, "model": model,
                                    "mean_period_s": fine_period["mean_period_s"], "period_spread_s": fine_period["period_spread_s"],
                                    "positive_peaks": len(fine_period["positive_peak_times_s"]), "x_half_range_m": fine_period["x_half_range_m"],
                                    "elapsed_over_measured_period": float(t[-1]/fine_period["mean_period_s"]) if fine_period["mean_period_s"] else None})
                slow_positions[model] = position
                del states, coarse, energy
            assert common_time is not None
            ta, ordinary_fine, _, _ = load_case("A", phase_index, "fine", length, settings, prefer_events=False)
            _, ordinary_coarse, _, _ = load_case("A", phase_index, "coarse", length, settings, prefer_events=False)
            _, of_mean = centered_period_mean(ordinary_fine[:, :2], ta, N)
            _, oc_mean = centered_period_mean(ordinary_coarse[:, :2], ta, N)
            ordinary_convergence.append({"extent": length, "phase_index": phase_index,
                                         **coordinate_metrics(oc_mean, of_mean, scale),
                                         "ordinary_fine_vs_selected_A": coordinate_metrics(of_mean, slow_positions["A"], scale)})
            del ordinary_fine, ordinary_coarse, of_mean, oc_mean
            for a, reference in (("A", "A_smooth"), ("A_smooth", "B"), ("B", "C"), ("A", "B"), ("A", "C"), ("A_smooth", "C")):
                m = coordinate_metrics(slow_positions[a], slow_positions[reference], scale)
                pe = relative_period_difference(diagnostics[a], diagnostics[reference])
                aa, bb = diagnostics[a]["positive_peak_times_s"], diagnostics[reference]["positive_peak_times_s"]
                count = min(len(aa), len(bb))
                delays = np.array(aa[:count])-np.array(bb[:count])
                numerics_ok = convergence[a]["passed"] and convergence[reference]["passed"]
                filters_ok = all(raw_inputs[k]["filter"]["within_RMS_budget"] for k in (a, reference) if k in ("A", "A_smooth"))
                enough_time = all(diagnostics[k]["mean_period_s"] and common_time[-1]/diagnostics[k]["mean_period_s"] >= settings["minimum_requested_slow_periods"][length] for k in (a, reference))
                physical_pass = m["relative_rms"] <= tol["relative_position_rms"] and m["relative_max"] <= tol["relative_position_max"] and pe is not None and pe <= tol["relative_period"]
                status = ("passed" if physical_pass else "failed") if numerics_ok and filters_ok and enough_time else "inconclusive"
                row = {"extent": length, "phase_index": phase_index, "pair": a+"-"+reference, **m,
                       "relative_period_error": pe, "x_half_range_difference_on_Rscale": abs(diagnostics[a]["x_half_range_m"]-diagnostics[reference]["x_half_range_m"])/scale,
                       "last_positive_peak_delay_s": float(delays[-1]) if count else None,
                       "numerical_RMS_change_sum": convergence[a]["relative_rms"]+convergence[reference]["relative_rms"],
                       "ODE_converged": bool(numerics_ok), "filter_diagnostics_passed": bool(filters_ok),
                       "minimum_duration_met": bool(enough_time), "status": status}
                metrics_rows.append(row)
                error = np.linalg.norm(slow_positions[a]-slow_positions[reference], axis=-1)/scale
                for cycle in range(int(np.ceil(common_time[-1]/period))):
                    selected = (common_time >= cycle*period) & (common_time < (cycle+1)*period)
                    if np.any(selected):
                        accumulation_rows.append({"extent": length, "phase_index": phase_index, "pair": a+"-"+reference,
                                                  "estimated_cycle_index": cycle, "t_start_s": float(common_time[selected][0]),
                                                  "t_end_s": float(common_time[selected][-1]),
                                                  "relative_rms": float(np.sqrt(np.mean(error[selected]**2))), "relative_max": float(error[selected].max())})
            details.append({"extent": length, "phase_index": phase_index, "period_diagnostics": diagnostics,
                            "numerical_convergence": convergence, "model_diagnostics": raw_inputs})
            np.savez_compressed(OUTPUT/f"comparison_phase{phase_index}_{length}.npz", time_s=common_time,
                                **{model+"_slow_position_m": value for model, value in slow_positions.items()})
            plot(OUTPUT, length, phase_index, common_time, slow_positions, period, convergence["A"]["passed"])
            del slow_positions
    csv_write(OUTPUT/f"metrics_{extent}.csv", metrics_rows)
    csv_write(OUTPUT/f"numerical_checks_{extent}.csv", numerical_rows)
    csv_write(OUTPUT/f"filter_checks_{extent}.csv", filter_rows)
    csv_write(OUTPUT/f"period_amplitude_{extent}.csv", period_rows)
    csv_write(OUTPUT/f"accumulation_{extent}.csv", accumulation_rows)
    refinement = None
    refined_arrays = OUTPUT/"edge_resolved/A_phase0_refined_pilot.npz"
    refined_meta = refined_arrays.with_suffix(".json")
    if refined_arrays.exists() and refined_meta.exists():
        rm = json.loads(refined_meta.read_text(encoding="utf-8"))
        if rm["arrays_sha256"] != sha(refined_arrays):
            raise ValueError("Corrupt third refinement")
        with np.load(refined_arrays) as data:
            rt, rs = data["time_s"], data["states"]
        with np.load(OUTPUT/"edge_resolved/A_phase0_fine_pilot.npz") as data:
            ft, fs = data["time_s"], data["states"]
        _, rmean = centered_period_mean(rs[:, :2], rt, N)
        _, fmean = centered_period_mean(fs[:, :2], ft, N)
        cycles = []
        centers = rt[N//2:-N//2]
        for cycle in range(int(np.ceil(centers[-1]/period))):
            select = (centers >= cycle*period) & (centers < (cycle+1)*period)
            if np.any(select):
                cycles.append({"cycle": cycle, **coordinate_metrics(rmean[select], fmean[select], scale)})
        refinement = {"levels": [100, 200], "phase_index": 0, "extent": "pilot",
                      **coordinate_metrics(rmean, fmean, scale), "by_estimated_cycle": cycles,
                      "refined_source_provenance": rm["provenance"]}
        write(OUTPUT/"third_refinement_check.json", refinement)
        del rt, rs, ft, fs, rmean, fmean
    source_files = [Path(__file__), ROOT/"scripts/run_abc_pilot.py", ROOT/"scripts/run_abc_pilot_events.py", ROOT/"scripts/run_abc_pilot_refined.py", ROOT/"src/three_phase_trap/event_resolved_rf.py",
                    ROOT/"src/three_phase_trap/slow_analysis.py", ROOT/"src/three_phase_trap/micromotion.py", ROOT/"src/three_phase_trap/pilot_dynamics.py"]
    raw_inconclusive = any(row["status"] == "inconclusive" and "A-" in row["pair"] for row in metrics_rows)
    summary = {"status": "pilot_computed_raw_FEM_inconclusive" if raw_inconclusive else "pilot_computed_analysis_complete_spatial_accuracy_unverified",
               "analyzed_extent": extent, "all_requested_pilot_runs_complete": extent == "all",
               "settings": settings, "predeclared_tolerances": tol, "numerical_budgets": budget,
               "metrics": metrics_rows, "details": details, "FEM_mesh_convergence_verified": False,
               "ordinary_raw_convergence": ordinary_convergence,
               "third_refinement": refinement,
               "scope": "Single-mesh trajectory agreement is not absolute FEM accuracy; no global stability inference",
               "analysis_source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in source_files}}
    write(OUTPUT/f"summary_{extent}.json", summary)
    report(OUTPUT, summary)
    if extent == "all":
        write(OUTPUT/"summary.json", summary)
        for source, target in (("metrics_all.csv", "metrics.csv"), ("numerical_checks_all.csv", "numerical_checks.csv"),
                               ("filter_checks_all.csv", "filter_checks.csv"), ("period_amplitude_all.csv", "period_amplitude.csv"),
                               ("accumulation_all.csv", "accumulation.csv")):
            (OUTPUT/target).write_bytes((OUTPUT/source).read_bytes())
        progress_path = OUTPUT.parent/"progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["user_step_7_long_pilot"] = summary["status"]
        write(progress_path, progress)
        from run_abc_initial_checks import write_report
        write_report(OUTPUT.parent, protocol, progress)
    print(json.dumps({"analyzed_extent": extent, "metrics": metrics_rows}, indent=2), flush=True)


def plot(output, extent, phase, time, positions, period, raw_converged):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    step = max(1, len(time)//5000)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for model in ("A", "A_smooth", "B", "C"):
        xy = positions[model][::step]*1000
        label = "A (ODE unresolved)" if model == "A" and not raw_converged else model
        axes[0].plot(xy[:, 0], xy[:, 1], linewidth=1, label=label)
        axes[1].plot(time[::step]*1000, xy[:, 0], linewidth=1, label=label)
    axes[0].set(xlabel="X (mm)", ylabel="Y (mm)", title="Slow vector trajectories")
    axes[0].axis("equal")
    axes[1].set(xlabel="t (ms)", ylabel="X (mm)", title=f"{extent}, RF phase index {phase}")
    for a, b in (("A", "A_smooth"), ("A_smooth", "B"), ("B", "C"), ("A", "C")):
        error = np.linalg.norm(positions[a]-positions[b], axis=-1)/.001
        axes[2].semilogy(time[::step]/period, np.maximum(error[::step], 1e-16), label=a+" - "+b)
    axes[2].axhline(.01, linestyle=":", color="grey", label="1% reference (RMS target)")
    axes[2].axhline(.03, linestyle="--", color="grey", label="3% maximum target")
    axes[2].set(xlabel="t / estimated slow period", ylabel="Instantaneous vector error / 1 mm", title="Unaligned trajectory differences")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output/f"trajectories_phase{phase}_{extent}.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model in ("A_smooth", "B", "C"):
        axes[0].plot(time[::step]*1000, positions[model][::step, 1]*1e6, label=model)
    axes[0].set(xlabel="t (ms)", ylabel="Y (micrometres)", title="Smooth RF and averaged trajectories")
    error = np.linalg.norm(positions["A_smooth"]-positions["B"], axis=-1)/.001
    cum_rms = np.sqrt(np.cumsum(error**2)/np.arange(1, len(error)+1))
    cum_max = np.maximum.accumulate(error)
    axes[1].plot(time[::step]/period, 100*cum_rms[::step], label="Cumulative vector RMS, %")
    axes[1].plot(time[::step]/period, 100*cum_max[::step], label="Cumulative vector maximum, %")
    axes[1].axhline(1, linestyle=":", color="C0", label="1% RMS criterion")
    axes[1].axhline(3, linestyle=":", color="C1", label="3% maximum criterion")
    axes[1].set(xlabel="t / estimated slow period", ylabel="Error / 1 mm (%)", title="A_smooth - B: accumulation with time")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output/f"averaging_phase{phase}_{extent}.png", dpi=180)
    plt.close(fig)


def report(output, summary):
    s = summary["settings"]
    lines = ["# Пункт 7: длительное сравнение A, A_smooth, B и C", "",
             f"Статус: `{summary['status']}`. Проанализирован набор: `{summary['analyzed_extent']}`.", "",
             "Проверки B–C проходят на обоих интервалах. Для гладкого полного поля A_smooth усреднение B проходит",
             "критерии на 3,25 медленного периода, но превышает их на 10,25 периода из-за накопления векторной ошибки.",
             "Исходное FEM-движение A пока не стало численно сходящимся эталоном; все сравнения с ним — inconclusive.", "",
             "## Условия и временной интервал", "",
             "Пилот: m=1e−15 кг, q=+1e−12 Кл, U0=250 В, f=100 кГц, без DC, трения и гравитации.",
             "Общая медленная пара R0=(1 мм,0), V0=0. A и A_smooth начинают с одинаковой полной пары",
             "из initial_conditions.json, включая drift_corrected; фаза задаётся как Ωt+φ0 для всех групп.",
             f"Оценка медленного периода C: {s['estimated_slow_period_s']:.10g} с.",
             f"Обе RF-фазы 0 и π/2: {s['RF_periods']['pilot']} RF-периодов ({s['RF_periods']['pilot']/100000:.6g} с, около 3,25 медленных периодов).",
             f"Фаза 0 дополнительно: {s['RF_periods']['extended']} RF-периодов ({s['RF_periods']['extended']/100000:.6g} с, около 10,25 медленных периодов).",
             "Дополнительная четверть периода позволяет измерить внутренние максимумы без выдуманного начального отсчёта после обрезки окна.", "",
             "A использует исходный разрывный градиент P2. Точная аффинная форма внутри каждого треугольника",
             "проверена против исходного интерполятора и короткой траектории; сглаживания поля A нет (probe.json).",
             "A_smooth использует все моды гладкого восстановления B до N=48. B и C интегрируются отдельно.",
             "DOP853: RF max_step=T_RF/50 и T_RF/100, rtol=2e−9 и 2e−11; абсолютные допуски SI раздельны для R,V.",
             "Для B/C также два шага и два допуска. Все настройки и SHA входных данных — в settings.json и метаданных checkpoint.",
             "Для 10-периодного RF-расчёта продолжение начинается с конца 3-периодного; глобальная фаза времени сохраняется.", "",
             "## Сравнения медленных координат", "",
             "Основной фильтр A/A_smooth — центрированное трапецеидальное среднее за один RF-период.",
             "B/C сравниваются в центрах тех же окон. Масштаб ошибок всегда 1 мм; временной сдвиг и амплитуда не подгонялись.",
             "Критерии из исходного протокола: RMS координат ≤1%, максимум ≤3%, ошибка периода ≤1%.",
             "Период измерен по положительным максимумам X с локальным уточнением сплайном; сохранены все времена максимумов и разброс периодов.", "",
             "| набор | RF-фаза | пара | RMS, % | максимум, % | ошибка периода, % | статус |",
             "|---|---:|---|---:|---:|---:|---|"]
    for row in summary["metrics"]:
        pe = f"{row['relative_period_error']*100:.5g}" if row["relative_period_error"] is not None else "нет оценки"
        lines.append(f"| {row['extent']} | {row['phase_index']} | {row['pair']} | {row['relative_rms']*100:.6g} | {row['relative_max']*100:.6g} | {pe} | {row['status']} |")
    lines += ["", "RF-фаза в таблице дана индексом: 0 означает φ0=0, 1 означает φ0=π/2.",
              "`passed` относится к этим параметрам, интервалу и данной FEM-сетке. `failed` означает превышение",
              "критерия после пройденных численных проверок. `inconclusive` означает недостаточную сходимость,",
              "диагностику фильтра или длительность; это не доказательство физического отказа приближения.", "",
              "## Сходимость исходного FEM-движения", "",
              "Обычный DOP853 без локализации переходов между элементами не прошёл проверку T/50→T/100.",
              "Эти расчёты сохранены; они не используются как доказанный эталон полного движения.",
              "Дополнительно реализован вариант с событиями λi(R)=0 для трёх барицентрических координат:",
              "решение останавливается на ребре, координата и скорость сохраняются, далее берётся поле соседнего элемента.",
              "Полевые скачки не сглаживаются; поле не меняется. Метод проверен по независимому аналитическому решению со скачком силы.",
              "Когда имеются оба уровня edge_resolved, они выбраны для A вместо обычного решения.",
              "Для фазы 0 дополнительно выполнен T/200 с rtol=2e−13; при наличии он сравнивается с T/100.", "",
              "| набор | фаза | RMS изменения обычного DOP853, % | максимум, % |",
              "|---|---:|---:|---:|"]
    for row in summary["ordinary_raw_convergence"]:
        lines.append(f"| {row['extent']} | {row['phase_index']} | {row['relative_rms']*100:.6g} | {row['relative_max']*100:.6g} |")
    ref = summary.get("third_refinement")
    if ref is not None:
        lines += ["", f"После локализации рёбер при переходе T/100→T/200 (фаза 0, 3,25 периода) RMS изменения = {ref['relative_rms']*100:.6g}%, максимум = {ref['relative_max']*100:.6g}%.",
                  "Изменение по отдельным оценочным циклам сохранено в third_refinement_check.json."]
    lines += ["", "Если эта сходимость не достигает бюджета, сравнения с A имеют статус inconclusive.",
              "По ним нельзя заключить, что физическое временное усреднение неверно или что ошибка FEM равна наблюдаемой разнице.",
              "Причину оставшейся чувствительности исходного P2-поля необходимо проверять отдельно, включая более гладкое/точное FEM-представление и сходимость по сетке.", "",
              "## Разделение ошибок и ограничения", "",
              "- A–A_smooth: влияние замены исходного P2-поля гладким представлением при одинаковом полном старте.",
              "- A_smooth–B: временное усреднение для одного гладкого RF-поля, включая остаток начального преобразования.",
              "- B–C: отбрасывание высших пространственных гармоник и отличие коэффициента ведущей моды.",
              "- A–C: суммарная цепочка приближений на этой сетке.", "",
              "Численные бюджеты каждого отдельного источника: RMS ≤0,1%, максимум ≤0,3%, период ≤0,1%.",
              "Проверены coarse/fine, 32/64 отсчёта, окна 1/2/3 RF-периода, альтернативный локальный тренд",
              "и обратное преобразование микродвижения. Разброс методов — диагностика, не независимая истинная погрешность.",
              "При оценке периода сохранённая вариация между циклами может включать реальную эволюцию орбиты.",
              "Энергия B/C записана и проверена. Энергия B, подставленная в усреднённое состояние A, — только",
              "дополнительная диагностика; энергия в полном RF-поле не обязана сохраняться.", "",
              "Почти совпадающий медленный период не гарантирует совпадения векторных координат.",
              "Графики averaging_*.png показывают Y(t), накопленный RMS и максимум для A_smooth–B отдельно от неопределённых FEM-траекторий.", "",
              "Сходимость по FEM-сетке 50/35 мкм пока не выполнена. Совпадение A/B/C на одной сетке",
              "не доказывает абсолютную точность электростатики или глобальную устойчивость ловушки.", "",
              "## Данные и продолжение", "",
              "- `metrics.csv`, `numerical_checks.csv`, `filter_checks.csv`, `period_amplitude.csv`, `accumulation.csv`: итоговые метрики полного набора.",
              "- `*_pilot.npz`, `*_extended.npz`: t,x,y,vx,vy; для B/C энергия; рядом JSON со статусом, допусками и SHA.",
              "- `slow_*.npz`: полные векторные средние; `inverse_*.npz`: независимая диагностика на 8 отсчётах за RF-период.",
              "- `poincare_*.csv`: фазовые срезы; `comparison_*.npz`, `trajectories_*.png`: сравнения и графики.",
              "- `summary.json`: полный результат; `summary_pilot.json`/`summary_extended.json` при отдельных анализах.", "",
              "```powershell", "C:/Python314/python.exe scripts/run_abc_pilot.py", "C:/Python314/python.exe scripts/run_abc_pilot.py --stage analyze", "```", "",
              "Готовые checkpoint проверяются по входам, вычислительному коду и SHA массивов; повторный запуск не пересчитывает завершённые случаи.",
              "Уточнения A воспроизводятся через scripts/run_abc_pilot_events.py (--phase-index 0/1, --level coarse/fine, --extent pilot),",
              "затем scripts/run_abc_pilot_refined.py. Для extended сохранён также отдельный coarse edge_resolved; в сравнении 10-периодного A используются два обычных уровня с соответствующим статусом неопределённости.",
              "После этих уточнений повторно запустите --stage analyze; NPZ исходных обычных запусков не перезаписываются.",
              "Следующий пункт 8 краткого списка — контролируемый высокочастотный предел (этап 7 плана).",
              "Если численная сходимость или точность поля не позволяет вывод, сначала требуется закрыть соответствующую неопределённость."]
    (output/f"PILOT_REPORT_{summary['analyzed_extent']}.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if summary["analyzed_extent"] == "all":
        (output/"PILOT_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extent", choices=("all", "pilot", "extended"), default="all")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    preview() if args.preview else main(args.extent)
