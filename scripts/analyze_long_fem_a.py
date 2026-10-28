"""Analyze an existing long native A trajectory without rerunning full RF A.

Compute cheap B/C references with temporal controls, compare unaligned slow
positions, and quantify observed turning-line drift. A's long-time global
convergence remains unverified unless separate long RF controls exist.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.long_rf_run import RunStore, verify_completion, sha, write_json, write_npz
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.averaged_dynamics import AveragedRFModel, integrate_averaged
from three_phase_trap.octupole_model import OctupoleModel
from three_phase_trap.micromotion import recover_slow_state_with_drift
from three_phase_trap.pilot_dynamics import coordinate_metrics, oscillation_metrics
from three_phase_trap.slow_analysis import centered_period_mean, local_rf_trend
from three_phase_trap.trajectory_diagnostics import position_errors, turning_orientations
from analyze_abc_pilot import csv_write, relative_period_difference, inverse_batches

BASE = ROOT/"outputs/abc_validation"
VERIFIED = BASE/"direct_fem_validation/v2"
DEFAULT_RUN = BASE/"long_A/v1/A_P4_h35_phase0_fine_50Ts_from_zero"


def signature(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def prepare(run, directory):
    original = json.loads((run/"settings.json").read_text(encoding="utf-8"))
    verify_completion(RunStore(run, original))
    if original["mesh_um"] != 35 or original["phase_index"] != 0:
        raise ValueError("This analysis is explicitly scoped to h35, phase zero")
    inputs = [run/n for n in ("settings.json", "COMPLETED.json", "summary.json", "trajectory.npz", "slow_samples.npz")]
    paths = [VERIFIED/"h35_P4/diagnostic_N48.json", VERIFIED/"h35_P4/native_C_calibration.json",
             BASE/"model_b/reconstruction.json", BASE/"model_c/coefficients.json"]
    verified_manifest = json.loads((VERIFIED/"manifest.json").read_text(encoding="utf-8"))["artifacts"]
    verified_settings = json.loads((VERIFIED/"validation_settings.json").read_text(encoding="utf-8"))
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        expected = verified_manifest.get(name, verified_settings["input_sha256"].get(name))
        if expected is None or sha(path) != expected:
            raise ValueError("Reference model provenance mismatch: "+name)
    inputs += paths+[VERIFIED/"validation_settings.json"]
    for extent in ("pilot", "extended"):
        inputs += [VERIFIED/f"runs/A_h35_phase0_fine_{extent}.npz", VERIFIED/f"runs/A_h35_phase0_fine_{extent}.json"]
    kernels = {}
    for name, expected in original["source_sha256"].items():
        if sha(ROOT/name) != expected:
            raise ValueError("Original long-run source changed: "+name)
        kernels[name] = expected
    kernels["src/three_phase_trap/trajectory_diagnostics.py"] = sha(ROOT/"src/three_phase_trap/trajectory_diagnostics.py")
    end = original["RF_periods"]/original["physical"]["frequency_Hz"]
    settings = dict(run=run.relative_to(ROOT).as_posix(), physical=original["physical"], full_initial_state=original["initial_state"],
                    phase_rad=original["phase_rad"], reference_slow_period_s=original["reference_slow_period_s"],
                    RF_periods=original["RF_periods"], duration_s=end, samples_per_RF_period=64, R_scale_m=original["R_scale_m"],
                    physical_tolerances=verified_settings["tolerances"],
                    windows_slow_periods=[3.25, 10.25, 20., 25., 30., 40., 50.],
                    reference_sampling="two intervals per RF period; cubic interpolation to all RF-filtered A nodes, with nested-grid control",
                    reference_controls={"coarse": dict(rtol=2e-10, atol=2e-12, steps_per_slow_period=100),
                                        "fine": dict(rtol=2e-12, atol=2e-14, steps_per_slow_period=200)},
                    input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
                    source_sha256=kernels, analysis_source_sha256=sha(Path(__file__)),
                    A_global_convergence_50Ts_verified=False)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"settings.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Frozen analysis changed; use a new --version")
    if not path.exists():
        write_json(path, settings)
    (directory/"references").mkdir(exist_ok=True)
    return settings


def models(settings):
    p = settings["physical"]
    b = AveragedRFModel(RFReconstruction.load(VERIFIED/"h35_P4/diagnostic_N48.json"), p["charge_C"], p["mass_kg"], p["U0_V"], p["frequency_Hz"])
    calibration = json.loads((VERIFIED/"h35_P4/native_C_calibration.json").read_text(encoding="utf-8"))
    c = OctupoleModel(calibration["C4_per_volt_per_m4"], calibration["phase_rad"], b.charge, b.mass, b.amplitude, b.frequency, .008)
    old_b = AveragedRFModel(RFReconstruction.load(BASE/"model_b/reconstruction.json"), b.charge, b.mass, b.amplitude, b.frequency)
    old = json.loads((BASE/"model_c/coefficients.json").read_text(encoding="utf-8"))
    old_c = OctupoleModel(old["C4_per_volt_per_m4"], old["delta4_rad"], b.charge, b.mass, b.amplitude, b.frequency, .008)
    return dict(B35=b, C35=c, B70=old_b, C70=old_c, B35_matched=b)


def reference_path(directory, model, level):
    return directory/"references"/f"{model}_{level}.npz"


def reference_valid(directory, settings, model, level):
    path = reference_path(directory, model, level)
    if not path.exists() or not path.with_suffix(".json").exists():
        return False
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["signature"] != signature(settings) or meta["arrays_sha256"] != sha(path) or meta["status"] != "completed":
        raise ValueError("Slow reference checkpoint mismatch: "+path.name)
    return True


def reference_worker(directory_string, model, level):
    directory = Path(directory_string)
    s = json.loads((directory/"settings.json").read_text(encoding="utf-8"))
    obj = models(s)[model]
    p, control = s["physical"], s["reference_controls"][level]
    initial = np.r_[p["slow_position_m"], p["slow_velocity_m_per_s"]]
    if model == "B35_matched":
        initial = recover_slow_state_with_drift(obj.quadratures, np.array(s["full_initial_state"]), obj.charge, obj.mass, obj.omega,
                                              s["phase_rad"], check_points=obj.reconstruction.check_points)
    started = time.perf_counter()
    result = integrate_averaged(obj, initial[:2], initial[2:], s["duration_s"], samples=2*s["RF_periods"]+1,
                                rtol=control["rtol"], atol=control["atol"], max_step=s["reference_slow_period_s"]/control["steps_per_slow_period"])
    representation_difference = result.states[0]-initial
    if result.status != "completed" or np.any(abs(representation_difference) > 8*abs(np.spacing(initial))):
        raise ValueError("Slow reference failed or changed its initial state")
    path = reference_path(directory, model, level)
    write_npz(path, time_s=result.time, states=result.states, energy_J=result.energy)
    write_json(path.with_suffix(".json"), dict(signature=signature(s), arrays_sha256=sha(path), status="completed", initial_state=initial.tolist(),
                                             initial_representation_difference=representation_difference.tolist(),
                                             model=model, level=level, evaluations=result.evaluations, wall_seconds=time.perf_counter()-started))
    return dict(model=model, level=level, seconds=time.perf_counter()-started)


def load_reference(directory, settings, model, level):
    if not reference_valid(directory, settings, model, level):
        raise ValueError("Missing slow reference: "+model+" "+level)
    with np.load(reference_path(directory, model, level)) as data:
        return data["time_s"], data["states"], data["energy_J"]


def period(t, position, s):
    return oscillation_metrics(t, position, s["reference_slow_period_s"])


def filter_controls(raw_time, raw_states, time_mean, mean, b, s):
    rows = []
    def add(name, ta, a, ref):
        metrics = coordinate_metrics(a, ref, s["R_scale_m"])
        periods = relative_period_difference(period(ta[::64], a[::64], s), period(ta[::64], ref[::64], s))
        rows.append(dict(control=name, **metrics, relative_period_change=periods))
    for n in (2, 3):
        tw, filtered = centered_period_mean(raw_states[:, :2], raw_time, 64*n)
        offset = (n-1)*32
        add(f"window{n}_vs_window1", tw, filtered, mean[offset:-offset])
    t32, mean32 = centered_period_mean(raw_states[::2, :2], raw_time[::2], 32)
    add("32_vs_64_output_intervals", t32, mean32, mean[::2])
    tt, trend = local_rf_trend(raw_states[:, :2], raw_time, 64, 2)
    add("local_RF_trend_vs_period_mean", tt, trend, mean[32:-32])
    ids = np.arange(32, len(raw_time)-32, 64)
    inverse = inverse_batches(b, raw_states[ids], raw_time[ids], s["phase_rad"])
    m = coordinate_metrics(inverse[:, :2], mean[::64], s["R_scale_m"])
    pe = relative_period_difference(period(raw_time[ids], inverse[:, :2], s), period(time_mean[::64], mean[::64], s))
    rows.append(dict(control="inverse_drift_map_vs_period_mean", **m, relative_period_change=pe))
    return rows


def analyze(run, directory, s):
    print("Reading full A and checking its period average...", flush=True)
    with np.load(run/"trajectory.npz") as data:
        raw_time, raw = data["time_s"], data["states"]
    tm, mean = centered_period_mean(raw[:, :2], raw_time, 64)
    with np.load(run/"slow_samples.npz") as data:
        if not np.array_equal(data["time_s"], tm[::64]) or not np.array_equal(data["positions_m"], mean[::64]):
            raise ValueError("Stored slow samples do not match the frozen period filter")
    tol = s["physical_tolerances"]
    budget = dict(relative_rms=tol["relative_position_rms"]*tol["numerical_budget_fraction"],
                  relative_max=tol["relative_position_max"]*tol["numerical_budget_fraction"],
                  relative_period=tol["relative_period"]*tol["numerical_budget_fraction"])
    prefix_checks = []
    for extent in ("pilot", "extended"):
        with np.load(VERIFIED/f"runs/A_h35_phase0_fine_{extent}.npz") as data:
            old_t, old_y = data["time_s"], data["states"]
        n = len(old_t)
        if np.max(abs(old_t-raw_time[:n])) > 1e-13:
            raise ValueError("Old/new A comparison clocks differ")
        old_tm, old_mean = centered_period_mean(old_y[:, :2], old_t, 64)
        m = coordinate_metrics(mean[:len(old_mean)], old_mean, s["R_scale_m"])
        prefix_checks.append(dict(extent=extent, **m, full_RF_change=coordinate_metrics(raw[:n, :2], old_y[:, :2], s["R_scale_m"])))
    print("Checking filter sensitivity over the full 50-period interval...", flush=True)
    objects = models(s)
    filters = filter_controls(raw_time, raw, tm, mean, objects["B35"], s)
    del raw_time, raw, old_t, old_y, old_mean
    tplot, aplot = tm[::64], mean[::64]
    tp, turning_xy, angles, orientation = turning_orientations(tplot, aplot, s["reference_slow_period_s"])
    orientation["fitted_deg_per_reference_slow_period"] = orientation["fitted_rate_deg_per_s"]*s["reference_slow_period_s"]
    turning_rows = [dict(time_s=float(t), reference_slow_periods=float(t/s["reference_slow_period_s"]), x_m=float(p[0]), y_m=float(p[1]),
                         radius_m=float(np.linalg.norm(p)), line_orientation_deg=float(np.rad2deg(a))) for t, p, a in zip(tp, turning_xy, angles)]
    csv_write(directory/"turning_points.csv", turning_rows)
    controls, references = [], {}
    for name in ("B35", "C35", "B70", "C70"):
        t, y, energy = load_reference(directory, s, name, "fine")
        tc, yc, _ = load_reference(directory, s, name, "coarse")
        if np.max(abs(tc-t)) > 1e-13:
            raise ValueError("Reference control clocks differ")
        m = coordinate_metrics(yc[:, :2], y[:, :2], s["R_scale_m"])
        pe = relative_period_difference(period(tc, yc[:, :2], s), period(t, y[:, :2], s))
        drift = float(np.max(abs(energy-energy[0]))/abs(energy[0]))
        interpolant = CubicSpline(t[::2], y[::2, :2], axis=0)
        interpolation = coordinate_metrics(interpolant(t[1:-1:2]), y[1:-1:2, :2], s["R_scale_m"])
        controls.append(dict(model=name, **m, relative_period_change=pe, energy_max_relative_drift=drift,
                             interpolation_relative_rms=interpolation["relative_rms"], interpolation_relative_max=interpolation["relative_max"],
                             passed=bool(m["relative_rms"] <= budget["relative_rms"] and m["relative_max"] <= budget["relative_max"]
                                         and pe is not None and pe <= budget["relative_period"] and drift <= 1e-8 and interpolation["relative_max"] <= budget["relative_max"])))
        references[name] = CubicSpline(t, y[:, :2], axis=0)
    t, y, _ = load_reference(directory, s, "B35_matched", "fine")
    references["B35_matched"] = CubicSpline(t, y[:, :2], axis=0)
    f, ts = s["physical"]["frequency_Hz"], s["reference_slow_period_s"]
    windows = [(n, min(s["duration_s"], math.ceil(n*ts*f)/f)) for n in s["windows_slow_periods"] if n*ts <= s["duration_s"]+1e-12]
    a_periods = {n: period(tplot[tplot <= end-.5/f+1e-13], aplot[tplot <= end-.5/f+1e-13], s) for n, end in windows}
    rows, crossings, curve_arrays = [], {}, dict(time_s=tplot, A=aplot)
    for name, interpolant in references.items():
        print("Comparing full-resolution slow A with "+name+"...", flush=True)
        reference = interpolant(tm)
        errors = position_errors(mean, reference, s["R_scale_m"])
        square_sum = np.cumsum(errors["vector"]**2)
        curve_arrays[name] = reference[::64].copy()
        cumulative = np.sqrt(square_sum/np.arange(1, len(tm)+1))
        first = {}
        for title, values, threshold in (("prefix_RMS", cumulative, tol["relative_position_rms"]), ("pointwise_max", errors["vector"], tol["relative_position_max"])):
            mask = values > threshold
            first[title] = dict(time_s=float(tm[np.argmax(mask)]), reference_slow_periods=float(tm[np.argmax(mask)]/ts)) if mask.any() else None
        crossings[name] = first
        for n, end in windows:
            stop = int(np.searchsorted(tm, end-.5/f+1e-13, side="right"))
            selected = tplot <= end-.5/f+1e-13
            pe = relative_period_difference(a_periods[n], period(tplot[selected], curve_arrays[name][selected], s))
            rms, maximum = float(np.sqrt(square_sum[stop-1]/stop)), float(errors["vector"][:stop].max())
            rows.append(dict(reference=name, requested_reference_periods=n, end_time_s=end, filtered_samples=stop,
                             relative_rms=rms, relative_max=maximum, relative_period_error=pe,
                             x_relative_rms=float(np.sqrt(np.mean(errors["x"][:stop]**2))),
                             y_relative_rms=float(np.sqrt(np.mean(errors["y"][:stop]**2))),
                             radial_relative_rms=float(np.sqrt(np.mean(errors["radial"][:stop]**2))),
                             radial_relative_max=float(np.max(abs(errors["radial"][:stop]))),
                             observed_within_physical_tolerances=bool(rms <= tol["relative_position_rms"] and maximum <= tol["relative_position_max"] and pe is not None and pe <= tol["relative_period"]),
                             validation_status="provisional_A_global_convergence_at_50Ts_unverified"))
        del reference, errors, square_sum, cumulative
    secondary = []
    for a, b in (("B35", "C35"), ("B70", "C70"), ("B35_matched", "B35")):
        secondary.append(dict(pair=a+"-"+b, **coordinate_metrics(curve_arrays[a], curve_arrays[b], s["R_scale_m"])))
    dense_period = period(tm, mean, s)
    period_resolution = relative_period_difference(dense_period, period(tplot, aplot, s))
    raw_summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    metadata = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run/"checkpoints").glob("part_*.json"))]
    audit = dict(segments=len(metadata), checked_steps=sum(m["diagnostics"]["checked_steps"] for m in metadata),
                 crossings=sum(m["diagnostics"]["crossings"] for m in metadata),
                 position_nudges=sum(m["diagnostics"]["position_nudges"] for m in metadata),
                 velocity_nudges=sum(m["diagnostics"]["velocity_nudges"] for m in metadata),
                 hidden_exits_detected=sum(m["diagnostics"]["hidden_exits_detected"] for m in metadata),
                 integration_wall_seconds=sum(m["integration_wall_seconds"] for m in metadata))
    filter_ok = all(r["relative_rms"] <= budget["relative_rms"] and r["relative_max"] <= budget["relative_max"] and r["relative_period_change"] is not None and r["relative_period_change"] <= budget["relative_period"] for r in filters)
    prefix_ok = all(r["relative_rms"] <= budget["relative_rms"] and r["relative_max"] <= budget["relative_max"] for r in prefix_checks)
    summary = dict(trajectory_integrity="passed", A_global_convergence_50Ts_verified=False,
                   scope="One h35 P4 full-RF A trajectory, phase zero; no long A time-step/mesh control computed here",
                   run_summary=raw_summary, geometry_audit=audit, filter_controls=filters, filters_passed=filter_ok,
                   reference_controls=controls, reference_controls_passed=all(r["passed"] for r in controls),
                   old_verified_prefix_checks=prefix_checks, old_prefix_agreement_passed=prefix_ok,
                   dense_vs_RF_sampled_period_relative_change=period_resolution, orientation=orientation,
                   comparisons=rows, secondary_comparisons=secondary, threshold_crossings=crossings,
                   physical_tolerances=tol, numerical_budgets=budget,
                   interpretation="Observed unaligned coordinate mismatch; do not promote the single 50Ts A trajectory to a globally converged reference")
    csv_write(directory/"comparison_windows.csv", rows)
    csv_write(directory/"reference_controls.csv", controls)
    csv_write(directory/"filter_controls.csv", filters)
    write_npz(directory/"comparison_curves.npz", **curve_arrays)
    write_json(directory/"summary.json", summary)
    plots(directory, curve_arrays, tp, angles, turning_xy, s)
    report(directory, summary)
    artifacts = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write_json(directory/"manifest.json", dict(artifacts=artifacts, count=len(artifacts)))
    print(json.dumps(dict(integrity="passed", A_global_50Ts_convergence="not_verified", filters_passed=filter_ok,
                          main_results=[r for r in rows if r["requested_reference_periods"] == 50. and r["reference"] in ("B35", "C35", "B70", "C70")],
                          orientation=orientation, report=str(directory/"LONG_A_ANALYSIS.md")), indent=2), flush=True)


def plots(directory, curves, tp, angles, turning_xy, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    t, a, b, c = curves["time_s"], curves["A"], curves["B35"], curves["C35"]
    ts, scale = s["reference_slow_period_s"], s["R_scale_m"]
    cut = slice(None, None, max(1, len(t)//30000))
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for j in range(2):
        for name in ("A", "B35", "C35"):
            axes[0, j].plot(t[cut]/ts, curves[name][cut, j]*1e3, label=name, linewidth=.8)
        axes[0, j].set(ylabel=("Slow x (mm)" if j == 0 else "Slow y (mm)"), xlabel="t / reference slow period")
    for name in ("A", "B35", "C35"):
        axes[1, 0].plot(t[cut]/ts, np.linalg.norm(curves[name][cut], axis=1)*1e3, label=name, linewidth=.8)
    axes[1, 0].set(xlabel="t / reference slow period", ylabel="Slow radius (mm)")
    for name in ("B35", "C35", "B70"):
        axes[1, 1].plot(t[cut]/ts, np.linalg.norm(a[cut]-curves[name][cut], axis=1)/scale*100, label="A - "+name, linewidth=.8)
    axes[1, 1].axhline(3., color="gray", linestyle="--", label="3% maximum tolerance")
    axes[1, 1].set(xlabel="t / reference slow period", ylabel="Vector difference / 1 mm (%)")
    axes[2, 0].plot(tp/ts, np.rad2deg(angles), label="Observed turning-line angle")
    axes[2, 0].set(xlabel="t / reference slow period", ylabel="Orientation modulo pi, unwrapped (deg)")
    radii = np.linalg.norm(turning_xy, axis=1)
    axes[2, 1].plot(tp/ts, (radii-radii[0])*1e6, label="A: radial turning-amplitude change")
    axes[2, 1].set(xlabel="t / reference slow period", ylabel="Change from first radial maximum (µm)")
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Observed 50-period A trajectory: direct FEM P4, h=35 µm, RF phase 0; A long-time convergence not yet controlled")
    fig.savefig(directory/"long_A_comparison.png", dpi=160)
    fig.savefig(directory/"long_A_comparison.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    xy = a[cut]*1e3
    segments = np.stack((xy[:-1], xy[1:]), axis=1)
    collection = LineCollection(segments, cmap="viridis", linewidth=.6)
    collection.set_array(t[cut][:-1]/ts)
    ax.add_collection(collection)
    ax.plot(b[cut, 0]*1e3, b[cut, 1]*1e3, "k--", linewidth=1, label="B35 / C35 (nearly overlapping)")
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="Slow x (mm)", ylabel="Slow y (mm)", title="Unaligned slow trajectories: A over 50 reference periods")
    ax.legend()
    fig.colorbar(collection, ax=ax, label="t / reference slow period")
    ax.grid(alpha=.25)
    fig.savefig(directory/"long_A_xy.png", dpi=180)
    fig.savefig(directory/"long_A_xy.pdf")
    plt.close(fig)


def report(directory, s):
    lines = ["# Анализ A за 50 медленных периодов", "",
             "Целостность исходной траектории подтверждена. Здесь анализируется один запуск P4 h35 с одним временным уровнем. **Глобальная временная и межсеточная сходимость A на 50 Ts пока не проверена; наблюдаемые ошибки не являются окончательным верифицированным физическим вердиктом.**", "",
             "A не пересчитывался. Заново рассчитаны только быстрые модели B/C, с двумя уровнями точности и контролем энергии. Исходные полные RF-файлы и COMPLETED.json сохранены.", "",
             "Нормировка координат — фиксированные 1 мм. Фазового, временного или пространственного совмещения траекторий нет. Основной фильтр — центрированное среднее за RF-период на 64 интервалах. Ошибки координат вычисляются на всех узлах фильтрованной траектории; B/C интерполируются с проверенной вложенной сетки. Периоды в таблице измерены на одном отсчёте медленных координат за RF-период; его согласованность с полным выводом проверена отдельно.", "",
             "## Наблюдаемое расхождение", "", "| Интервал, Ts | Модель | RMS, % | Максимум, % | Период, % | RMS по радиусу, % | В заданных физических допусках |", "|---:|---|---:|---:|---:|---:|---|"]
    for r in s["comparisons"]:
        if r["reference"] in ("B35", "C35"):
            lines.append(f"| {r['requested_reference_periods']:g} | A–{r['reference']} | {100*r['relative_rms']:.7g} | {100*r['relative_max']:.7g} | {100*r['relative_period_error']:.7g} | {100*r['radial_relative_rms']:.7g} | {r['observed_within_physical_tolerances']} |")
    o, raw = s["orientation"], s["run_summary"]
    lines += ["", "## Форма движения", "",
              f"Максимальный полный радиус в сохранённых отсчётах: {raw['maximum_full_radius_m']*1e3:.9g} мм. Максимум абсолютной медленной y: {raw['maximum_slow_abs_y_m']*1e6:.7g} мкм. Средний период x по исходному отчёту: {raw['slow_x_oscillations']['mean_period_s']*1e3:.10g} мс.", "",
              f"По {o['turning_points']} внутренним радиальным максимумам угол линии колебаний изменился от {o['first_orientation_deg']:.6g}° до {o['last_orientation_deg']:.6g}°. Диагностический линейный наклон: {o['fitted_deg_per_reference_slow_period']:.6g}°/Ts; RMS остатка угловой подгонки {o['angular_fit_residual_rms_deg']:.6g}°. Размах радиальных амплитуд: {o['radial_turning_span_m']*1e6:.6g} мкм. Подгонка угла не используется для исправления сравнения A–B/C и не доказывает физический механизм дрейфа.", "",
              "## Контроли и ограничения", "",
              f"Численные контроли B/C: {s['reference_controls_passed']}; контроли фильтров: {s['filters_passed']}; согласованность первых 3,25/10,25 Ts с ранее проверенным A: {s['old_prefix_agreement_passed']}. Подробные числа — reference_controls.csv, filter_controls.csv и summary.json.", "",
              f"Относительное изменение периода A при переходе от полного фильтрованного вывода к одному отсчёту за RF-период: {s['dense_vs_RF_sampled_period_relative_change']:.6g}. Геометрический аудит: {s['geometry_audit']['checked_steps']} шагов, {s['geometry_audit']['crossings']} переходов, сдвигов координат {s['geometry_audit']['position_nudges']}, скоростей {s['geometry_audit']['velocity_nudges']}.", "",
              "Результат показывает наблюдаемую эволюцию на одной численной траектории. Для строгого вывода о точности полной модели A на 50 Ts нужны отдельные длительные проверки шага и сетки. Согласие с прежним эталоном на первых 10,25 Ts этой проверки на оставшиеся времена не заменяет. Новые многочасовые RF-контроли этим анализом не запускались.", "",
              "B35/C35 относятся к уточнённому полю P4; B70/C70 — исходные замороженные модели. B35_matched проверяет только влияние карты начального микродвижения. Их полные численные сравнения, первые пересечения допусков и разделение ошибки на x/y/радиус сохранены в summary.json и comparison_windows.csv.", "",
              "Воспроизведение: `C:/Python314/python.exe scripts/analyze_long_fem_a.py --stage all --workers 4`. Проверка артефактов: `--stage verify`. Итоговый manifest.json фиксирует SHA данных и анализа.", ""]
    (directory/"LONG_A_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--stage", choices=("all", "references", "analyze", "verify"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    if not args.version.isalnum():
        parser.error("Expected a simple analysis version name")
    run = args.run.resolve()
    directory = run/"analysis"/args.version
    settings = prepare(run, directory)
    if args.stage == "verify":
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["artifacts"].items():
            if sha(ROOT/name) != expected:
                raise ValueError("Analysis artifact changed: "+name)
        print(json.dumps(dict(status="passed", artifacts_verified=len(manifest["artifacts"]))), flush=True)
        return
    if args.stage in ("all", "references"):
        cases = [(model, level) for model in ("B35", "C35", "B70", "C70") for level in ("coarse", "fine")]+[("B35_matched", "fine")]
        pending = [(model, level) for model, level in cases if not reference_valid(directory, settings, model, level)]
        print(f"Slow references pending/reused: {len(pending)}/{len(cases)-len(pending)}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(reference_worker, str(directory), model, level) for model, level in pending]
            for job in as_completed(jobs):
                print(json.dumps(job.result()), flush=True)
    if args.stage in ("all", "analyze"):
        analyze(run, directory, settings)


if __name__ == "__main__":
    main()
