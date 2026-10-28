"""Diagnostic step 2: common-state restarts and finite perturbation growth.

Saved full trajectories are read only. All restarted runs keep the global RF
clock, the raw P2 field and all four initial coordinates exactly as recorded.
Results are separate from raw_edge_audit and the original pilot directories.
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
from three_phase_trap.audited_rf import integrate_audited, _local
from three_phase_trap.config import load_config
from three_phase_trap.fem_cache import load_fem_cache, fem_fingerprint
from three_phase_trap.pilot_dynamics import AffineP2RF, coordinate_metrics
from three_phase_trap.restart_dynamics import integrate_radau, scaled_state_distance
from three_phase_trap.slow_analysis import centered_period_mean
from run_abc_pilot import load_models, sha, write
from analyze_abc_pilot import csv_write

BASE = ROOT/"outputs/abc_validation"
OUTPUT = BASE/"raw_restart_diagnosis/v1"
HISTORY = {"audited100": "raw_edge_audit/v2/A_phase0_fine_pilot.npz",
           "legacy100": "pilot/edge_resolved/A_phase0_fine_pilot.npz",
           "legacy200": "pilot/edge_resolved/A_phase0_refined_pilot.npz"}
_cache = {}


def histories():
    result = {}
    for name, relative in HISTORY.items():
        path = BASE/relative
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if info["status"] != "completed" or info["arrays_sha256"] != sha(path):
            raise ValueError(f"History checkpoint failed validation: {relative}")
        with np.load(path) as data:
            result[name] = (data["time_s"], data["states"])
    t = result["audited100"][0]
    if any(not np.array_equal(t, value[0]) for value in result.values()):
        raise ValueError("History comparison requires identical time nodes")
    if any(not np.array_equal(result["audited100"][1][0], value[1][0]) for value in result.values()):
        raise ValueError("Historical trajectories have different initial states")
    return result


def raw_field(settings):
    if "raw" not in _cache:
        config = load_config(ROOT/"configs/gmsh_70um.toml")
        if fem_fingerprint(config) != settings["fem_fingerprint"]:
            raise ValueError("FEM fingerprint mismatch")
        if sha(ROOT/"outputs/fem_cache/p2_basis.npz") != settings["basis_sha256"]:
            raise ValueError("FEM basis changed")
        _, _, _, interp = load_fem_cache(config, ROOT)
        _cache["raw"] = AffineP2RF(interp, settings["physical"]["U0_V"])
    return _cache["raw"]


def prepare():
    protocol, initial, model, c = load_models()
    old = histories()
    t = old["audited100"][0]
    windows = {}
    for name, clock, periods in (("early", .010, 512), ("onset", .024, 512),
                                 ("rapid", .030, 512), ("long", .020, 2048)):
        index = int(np.argmin(abs(t-clock)))
        if abs(t[index]-clock) > 1e-13:
            raise ValueError("Restart must use an exact saved output node")
        common = old["audited100"][1][index]
        plus = old["legacy100"][1][index]
        minus = common-(plus-common)
        windows[name] = {"start_s": float(t[index]), "RF_periods": periods,
                         "end_s": float(t[index]+periods/model.frequency), "history_index": index,
                         "initial_states": {"common": common.tolist(), "plus": plus.tolist(), "minus": minus.tolist()}}
    controls = {"dop100": {"method": "DOP853", "steps": 100, "rtol": 2e-11, "atol": [2e-15, 2e-15, 2e-13, 2e-13]},
                "dop200": {"method": "DOP853", "steps": 200, "rtol": 2e-13, "atol": [2e-17, 2e-17, 2e-15, 2e-15]},
                "dop400": {"method": "DOP853", "steps": 400, "rtol": 4e-14, "atol": [2e-18, 2e-18, 2e-16, 2e-16]},
                "radau50": {"method": "Radau", "steps": 50, "rtol": 2e-9, "atol": [2e-13, 2e-13, 2e-11, 2e-11]},
                "radau100": {"method": "Radau", "steps": 100, "rtol": 2e-11, "atol": [2e-15, 2e-15, 2e-13, 2e-13]}}
    sources = ("audited_rf.py", "restart_dynamics.py", "pilot_dynamics.py", "interpolation.py",
               "validation_field_audit.py", "fem_cache.py")
    input_paths = [relative for relative in HISTORY.values()]
    input_paths += [str(Path(relative).with_suffix(".json")).replace("\\", "/") for relative in HISTORY.values()]
    input_paths += ["raw_edge_audit/v2/A_phase0_fine_pilot_crossings.jsonl"]
    import scipy
    settings = {"diagnostic_step": 2, "physical": protocol["pilot"], "fem_fingerprint": protocol["fem_fingerprint"],
                "basis_sha256": protocol["basis_sha256"], "windows": windows, "controls": controls,
                "output_samples_per_RF_period": 64, "R_scale_m": .001,
                "V_scale_m_per_s": float(np.sqrt(2*model.potential(np.array([.001, 0.]))/model.mass)),
                "coordinate_budgets": {"relative_rms": .001, "relative_max": .003},
                "history_replay_relative_tolerance": .1, "minimum_signal_to_numerical_change": 10.,
                "source_sha256": {n: sha(ROOT/"src/three_phase_trap"/n) for n in sources},
                "input_sha256": {relative: sha(BASE/relative) for relative in input_paths},
                "protocol_sha256": sha(BASE/"protocol.json"),
                "worker_sha256": hashlib.sha256(inspect.getsource(worker).encode()).hexdigest(),
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "scope": "Phase 0, one 70um P2 field; local restart accuracy and sensitivity, no mesh/chaos/global-convergence claim"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/"settings.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Restart study inputs changed; preserve results and use a new versioned folder")
    write(path, settings)
    return settings


def case_paths(case):
    tag = f"{case['window']}_{case['control']}_{case['initial']}"
    return OUTPUT/(tag+".npz"), OUTPUT/(tag+".json")


def signature(case, settings):
    return hashlib.sha256(json.dumps({"case": case, "settings": settings}, sort_keys=True).encode()).hexdigest()


def valid(case, settings):
    arrays, metadata = case_paths(case)
    if not arrays.exists() or not metadata.exists():
        return False
    record = json.loads(metadata.read_text(encoding="utf-8"))
    if record["signature"] != signature(case, settings) or sha(arrays) != record["arrays_sha256"]:
        raise ValueError(f"Restart checkpoint signature/hash mismatch: {arrays.name}")
    return record["status"] == "completed"


def worker(case):
    settings = json.loads((OUTPUT/"settings.json").read_text(encoding="utf-8"))
    window, p, control = settings["windows"][case["window"]], settings["physical"], settings["controls"][case["control"]]
    raw = raw_field(settings)
    initial = np.array(window["initial_states"][case["initial"]])
    method = integrate_audited if control["method"] == "DOP853" else integrate_radau
    print(f"Starting {case}", flush=True)
    result = method(raw, initial, p["charge_C"], p["mass_kg"], p["frequency_Hz"], p["phase_rad"][0],
                    window["start_s"], window["end_s"], max_steps_per_period=control["steps"],
                    rtol=control["rtol"], atol=control["atol"], samples_per_period=64)
    if not np.array_equal(result.run.states[0], initial):
        raise ValueError("Restart did not preserve all four initial components")
    arrays, metadata = case_paths(case)
    temp = arrays.with_suffix(".tmp.npz")
    np.savez_compressed(temp, time_s=result.run.time, states=result.run.states, sample_triangles=result.sample_triangles)
    temp.replace(arrays)
    info = {"case": case, "signature": signature(case, settings), "status": result.run.status,
            "arrays_sha256": sha(arrays), "initial_state": initial.tolist(), "start_s": window["start_s"],
            "end_s": float(result.run.time[-1]), "global_RF_phase_at_start_rad": float((2*np.pi*p["frequency_Hz"]*window["start_s"]+p["phase_rad"][0])%(2*np.pi)),
            "diagnostics": result.diagnostics, "crossings": result.crossings,
            "nfev": result.run.nfev, "wall_seconds": result.run.wall_seconds}
    write(metadata, info)
    return {"case": case, "status": info["status"], "seconds": info["wall_seconds"], "crossings": len(result.crossings)}


def all_cases():
    cases = []
    for name in ("early", "onset", "rapid", "long"):
        controls = ("dop200", "dop400") if name == "long" else ("dop100", "dop200", "dop400", "radau50", "radau100")
        cases.extend({"window": name, "control": control, "initial": "common"} for control in controls)
        cases.extend({"window": name, "control": "dop400", "initial": label} for label in ("plus", "minus"))
        cases.append({"window": name, "control": "dop200", "initial": "plus"})
    return cases


def load_case(window, control, initial, settings):
    case = {"window": window, "control": control, "initial": initial}
    if not valid(case, settings):
        raise ValueError(f"Incomplete restart case: {case}")
    arrays, metadata = case_paths(case)
    with np.load(arrays) as data:
        return data["time_s"], data["states"]


def historical_analysis(settings, raw):
    old = histories()
    t, base = old["audited100"]
    threshold_rows, bins, plot_data = [], [], {"time_s": t}
    for name in ("legacy100", "legacy200"):
        difference = old[name][1]-base
        position = np.linalg.norm(difference[:, :2], axis=1)/settings["R_scale_m"]
        phase_space = scaled_state_distance(old[name][1], base, settings["R_scale_m"], settings["V_scale_m_per_s"])
        plot_data[name+"_position_error"] = position
        for threshold in (1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, .001, .01):
            indices = np.flatnonzero(position > threshold)
            threshold_rows.append({"pair": name+"-audited100", "threshold_on_1mm": threshold,
                                   "first_time_s": float(t[indices[0]]) if len(indices) else None})
        width = 64*64
        for start in range(0, len(t)-1, width):
            stop = min(start+width, len(t))
            bins.append({"pair": name+"-audited100", "start_s": float(t[start]), "end_s": float(t[stop-1]),
                         "R_relative_rms": float(np.sqrt(np.mean(position[start:stop]**2))),
                         "R_relative_max": float(position[start:stop].max()),
                         "scaled_state_rms": float(np.sqrt(np.mean(phase_space[start:stop]**2)))})
    events = [json.loads(line) for line in (BASE/"raw_edge_audit/v2/A_phase0_fine_pilot_crossings.jsonl").read_text().splitlines()]
    edges = []
    p = settings["physical"]
    for index, event in enumerate(events):
        state = np.array(event["state"])
        normal = raw.gradient[event["from_triangle"], event["face"]]
        normal = normal/np.linalg.norm(normal)
        normal_velocity = float(normal@state[2:])
        angle = 2*np.pi*p["frequency_Hz"]*event["time_s"]+p["phase_rad"][0]
        before = _local(raw, event["from_triangle"], state[:2], angle)[0]
        after = _local(raw, event["to_triangle"], state[:2], angle)[0]
        delta_a = p["charge_C"]/p["mass_kg"]*(after-before)
        if normal_velocity == 0:
            gain = None
        else:
            saltation = np.eye(4)
            saltation[2:, :2] += settings["R_scale_m"]/settings["V_scale_m_per_s"]*np.outer(delta_a, normal)/normal_velocity
            gain = float(np.linalg.norm(saltation, ord=2))
        edges.append({"index": index, "time_s": event["time_s"], "from_triangle": event["from_triangle"],
                      "to_triangle": event["to_triangle"], "radius_m": float(np.linalg.norm(state[:2])),
                      "normal_velocity_m_per_s": normal_velocity, "acceleration_jump_m_per_s2": float(np.linalg.norm(delta_a)),
                      "scaled_saltation_norm": gain})
    csv_write(OUTPUT/"historical_thresholds.csv", threshold_rows)
    csv_write(OUTPUT/"historical_growth_bins.csv", bins)
    csv_write(OUTPUT/"crossing_sensitivity.csv", edges)
    np.savez_compressed(OUTPUT/"historical_growth.npz", **plot_data)
    return old, edges


def comparison_metrics(actual, reference, t, settings):
    full = coordinate_metrics(actual[:, :2], reference[:, :2], settings["R_scale_m"])
    _, ma = centered_period_mean(actual[:, :2], t, 64)
    _, mb = centered_period_mean(reference[:, :2], t, 64)
    slow = coordinate_metrics(ma, mb, settings["R_scale_m"])
    distance = scaled_state_distance(actual, reference, settings["R_scale_m"], settings["V_scale_m_per_s"])
    return {"full_R_rms": full["relative_rms"], "full_R_max": full["relative_max"],
            "slow_R_rms": slow["relative_rms"], "slow_R_max": slow["relative_max"],
            "scaled_state_rms": float(np.sqrt(np.mean(distance**2))), "scaled_state_max": float(distance.max())}


def analyze(settings):
    old, edges = historical_analysis(settings, raw_field(settings))
    rows, sensitivity, replay, local_details = [], [], [], []
    diagnostics = {}
    for name, window in settings["windows"].items():
        t, reference = load_case(name, "dop400", "common", settings)
        t200, common200 = load_case(name, "dop200", "common", settings)
        if not np.array_equal(t, t200):
            raise ValueError("Restart reference time grids differ")
        pairs = [("dop200-dop400", common200, reference)]
        if name != "long":
            _, d100 = load_case(name, "dop100", "common", settings)
            _, r50 = load_case(name, "radau50", "common", settings)
            _, r100 = load_case(name, "radau100", "common", settings)
            pairs += [("dop100-dop400", d100, reference), ("radau50-radau100", r50, r100),
                      ("radau100-dop400", r100, reference)]
        _, plus = load_case(name, "dop400", "plus", settings)
        _, minus = load_case(name, "dop400", "minus", settings)
        _, plus200 = load_case(name, "dop200", "plus", settings)
        pairs.append(("perturbed_plus_dop200-dop400", plus200, plus))
        local_rows = []
        for label, actual, baseline in pairs:
            m = comparison_metrics(actual, baseline, t, settings)
            passed = m["slow_R_rms"] <= settings["coordinate_budgets"]["relative_rms"] and m["slow_R_max"] <= settings["coordinate_budgets"]["relative_max"]
            row = {"window": name, "pair": label, **m, "within_coordinate_budget": bool(passed)}
            rows.append(row)
            local_rows.append(row)
        numerical_change = max(r["slow_R_rms"] for r in local_rows if r["pair"] in ("dop200-dop400", "perturbed_plus_dop200-dop400"))
        initial_norm = float(scaled_state_distance(plus[0], reference[0], settings["R_scale_m"], settings["V_scale_m_per_s"]))
        norm_plus = scaled_state_distance(plus, reference, settings["R_scale_m"], settings["V_scale_m_per_s"])
        norm_minus = scaled_state_distance(minus, reference, settings["R_scale_m"], settings["V_scale_m_per_s"])
        perturb_m = comparison_metrics(plus, reference, t, settings)
        linear_sum = np.linalg.norm(plus[:, :2]+minus[:, :2]-2*reference[:, :2], axis=1)
        odd = np.linalg.norm(plus[:, :2]-minus[:, :2], axis=1)
        odd_scale = float(np.sqrt(np.mean(odd**2)))
        symmetry = float(np.sqrt(np.mean(linear_sum**2))/odd_scale) if odd_scale > 0 else None
        sensitivity.append({"window": name, "start_s": window["start_s"], "end_s": window["end_s"],
                            "initial_position_difference_m": float(np.linalg.norm(plus[0, :2]-reference[0, :2])),
                            "initial_velocity_difference_m_per_s": float(np.linalg.norm(plus[0, 2:]-reference[0, 2:])),
                            "initial_scaled_state_difference": initial_norm,
                            "final_scaled_state_difference": float(norm_plus[-1]),
                            "max_scaled_state_difference": float(norm_plus.max()),
                            "final_amplification": float(norm_plus[-1]/initial_norm) if initial_norm else None,
                            "max_amplification": float(norm_plus.max()/initial_norm) if initial_norm else None,
                            "plus_minus_antisymmetry_defect": symmetry,
                            "slow_position_perturbation_RMS": perturb_m["slow_R_rms"],
                            "strict_level_position_RMS_change": numerical_change,
                            "signal_to_numerical_change": perturb_m["slow_R_rms"]/numerical_change if numerical_change else None})
        start = window["history_index"]
        history_time = old["audited100"][0][start:start+len(t)]
        if len(history_time) != len(t) or np.max(abs(history_time-t)) > 1e-13:
            raise ValueError("Historical and restarted clocks do not match")
        original_base = old["audited100"][1][start:start+len(t)]
        original_plus = old["legacy100"][1][start:start+len(t)]
        historical_difference = original_plus[:, :2]-original_base[:, :2]
        restarted_difference = plus[:, :2]-reference[:, :2]
        residual = coordinate_metrics(restarted_difference, historical_difference, settings["R_scale_m"])
        gap_scale = float(np.sqrt(np.mean(np.sum(historical_difference**2, axis=1))))
        recovery_error = residual["relative_rms"]*settings["R_scale_m"]/gap_scale if gap_scale else None
        base_replay = comparison_metrics(reference, original_base, t, settings)
        plus_replay = comparison_metrics(plus, original_plus, t, settings)
        replay.append({"window": name, "historical_gap_RMS_on_1mm": gap_scale/settings["R_scale_m"],
                       "restart_gap_residual_RMS_on_1mm": residual["relative_rms"],
                       "residual_relative_to_historical_gap_RMS": recovery_error,
                       "common_replay_RMS": base_replay["slow_R_rms"], "plus_replay_RMS": plus_replay["slow_R_rms"],
                       "within_declared_replay_tolerance": bool(recovery_error is not None and recovery_error <= settings["history_replay_relative_tolerance"])})
        stride = 8
        diagnostics[name] = {"time_s": t[::stride], "initial_perturbation_growth": norm_plus[::stride],
                             "negative_perturbation_growth": norm_minus[::stride],
                             "common_dop200_vs_400_R_error": np.linalg.norm(common200[::stride, :2]-reference[::stride, :2], axis=1)/settings["R_scale_m"],
                             "historical_position_gap": np.linalg.norm(historical_difference[::stride], axis=1)/settings["R_scale_m"],
                             "restarted_position_gap": np.linalg.norm(restarted_difference[::stride], axis=1)/settings["R_scale_m"]}
        np.savez_compressed(OUTPUT/f"diagnostics_{name}.npz", **diagnostics[name])
        local_details.append({"window": name, "all_local_coordinate_checks_passed": all(r["within_coordinate_budget"] for r in local_rows)})
        print(json.dumps({"window": name, "local": local_details[-1], "sensitivity": sensitivity[-1], "replay": replay[-1]}), flush=True)
    csv_write(OUTPUT/"local_method_comparison.csv", rows)
    csv_write(OUTPUT/"perturbation_growth.csv", sensitivity)
    csv_write(OUTPUT/"history_replay.csv", replay)
    all_local = all(r["within_coordinate_budget"] for r in rows)
    resolved = [r for r in sensitivity if r["signal_to_numerical_change"] is not None and r["signal_to_numerical_change"] >= settings["minimum_signal_to_numerical_change"]]
    recovered = [r for r in replay if r["within_declared_replay_tolerance"]]
    summary = {"status": "local_restarts_converged_sensitivity_observed_global_raw_FEM_unverified" if all_local and len(resolved) == len(sensitivity) and len(recovered) == len(replay) else "restart_diagnosis_partial_or_inconclusive",
               "all_local_coordinate_checks_passed": all_local, "local_details": local_details,
               "resolved_perturbation_windows": [r["window"] for r in resolved],
               "recovered_historical_gap_windows": [r["window"] for r in recovered],
               "sensitivity": sensitivity, "replay": replay,
               "largest_scaled_saltation_norms": sorted([e for e in edges if e["scaled_saltation_norm"] is not None], key=lambda e: e["scaled_saltation_norm"], reverse=True)[:12],
               "basis_sha256_after": sha(ROOT/"outputs/fem_cache/p2_basis.npz"),
               "original_bundle_sha256_after": sha(ROOT/"outputs/fem_cache/gmsh_70um_fem_bundle.zip"),
               "scope": settings["scope"]}
    if summary["basis_sha256_after"] != settings["basis_sha256"]:
        raise ValueError("FEM basis changed during restart study")
    expected_bundle = json.loads((ROOT/"outputs/fem_cache/bundle_checksum.json").read_text(encoding="utf-8"))["sha256"]
    if summary["original_bundle_sha256_after"] != expected_bundle:
        raise ValueError("Original FEM bundle changed")
    write(OUTPUT/"summary.json", summary)
    plot(settings, diagnostics)
    report(settings, summary, rows)
    progress_path = BASE/"progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["raw_FEM_restart_diagnosis"] = summary["status"]
    write(progress_path, progress)
    write(OUTPUT/"manifest.json", {"driver_sha256": sha(Path(__file__)),
                                  "artifacts": {p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "manifest.json"}})


def plot(settings, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    with np.load(OUTPUT/"historical_growth.npz") as history:
        for key in ("legacy100", "legacy200"):
            axes[0, 0].semilogy(history["time_s"][::16], np.maximum(history[key+"_position_error"][::16], 1e-16), label=key+" - audited100")
    for window in settings["windows"].values():
        axes[0, 0].axvline(window["start_s"], color="gray", alpha=.3, linewidth=.8)
    axes[0, 0].set(xlabel="global time, s", ylabel="position difference / 1 mm", title="Saved full trajectories")
    axes[0, 0].legend(fontsize=8)
    for name, value in data.items():
        dt = value["time_s"]-value["time_s"][0]
        axes[0, 1].semilogy(dt, np.maximum(value["common_dop200_vs_400_R_error"], 1e-16), label=name)
        initial = value["initial_perturbation_growth"][0]
        if initial:
            axes[1, 0].semilogy(dt, value["initial_perturbation_growth"]/initial, label=name)
    axes[0, 1].set(xlabel="time since common restart, s", ylabel="position difference / 1 mm", title="Same initial state: DOP200 vs DOP400")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set(xlabel="time since restart, s", ylabel="scaled state amplification", title="Small historical initial difference, same DOP400")
    axes[1, 0].legend(fontsize=8)
    selected = data["long"]
    axes[1, 1].semilogy(selected["time_s"], np.maximum(selected["historical_position_gap"], 1e-16), label="original historical gap")
    axes[1, 1].semilogy(selected["time_s"], np.maximum(selected["restarted_position_gap"], 1e-16), "--", label="replayed from two initial states")
    axes[1, 1].set(xlabel="global time, s", ylabel="position difference / 1 mm", title="Restart at 0.020 s: gap replay")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Raw P2 FEM, phase 0: local accuracy and finite-time sensitivity")
    fig.savefig(OUTPUT/"restart_diagnosis.png", dpi=170)
    plt.close(fig)


def report(settings, summary, rows):
    lines = ["# Пункт 2: локализация расхождения и перезапуски из общего состояния", "",
             f"Статус: `{summary['status']}`.", "",
             "Используется исходное P2-поле сетки 70 мкм, фаза 0; FEM не пересчитывался.",
             "Сравниваются новые расчёты из строго одинаковых четырёх начальных компонент, с сохранением глобального RF-времени.",
             "Отдельно вводится начальное различие, взятое из двух исторических траекторий, и его отрицание.", "",
             "## Основной результат", "",
             f"Проверок численного бюджета координат: {len(rows)}; выполнено: {sum(r['within_coordinate_budget'] for r in rows)}.",
             "Возмущение различимо на фоне уточнения минимум в 10 раз на участках: "+", ".join(summary["resolved_perturbation_windows"])+".",
             "Историческое расхождение воспроизведено с остатком RMS не более 10% на участках: "+", ".join(summary["recovered_historical_gap_windows"])+".",
             "Согласие методов после общего старта и рост заранее существовавшего различия проверяются отдельно. Если один из критериев не выполнен, вывод по соответствующему участку ограничен.",
             "Воспроизведение относится к паре legacy100–audited100; для legacy200 построена история роста, но отдельное воспроизведение его стартового возмущения не выполнялось.",
             "Малые различия новых расчётов — эмпирический контроль численной точности на выбранных участках, а не строгая верхняя граница абсолютной ошибки.",
             "Точное попадание в прежнюю траекторию на всём интервале от t=0 и правильность эффективного потенциала этим опытом не установлены.", "",
             "## Участки и масштабы", "",
             "R_scale=1 мм; V_scale=sqrt(2Ψ_B(R0)/m), фиксированный масштаб скорости, не мгновенная скорость частицы.",
             f"V_scale={settings['V_scale_m_per_s']:.12g} м/с. Норма состояния: sqrt(|δr|²/R_scale²+|δv|²/V_scale²).", "",
             "| участок | начало, с | конец, с | RF-периодов |", "|---|---:|---:|---:|"]
    for name, value in settings["windows"].items():
        lines.append(f"| {name} | {value['start_s']:.8g} | {value['end_s']:.8g} | {value['RF_periods']} |")
    lines += ["", "Участки выбраны по сохранённым данным до новых запусков. Это диагностическая выборка одного случая, не независимая статистическая проверка всей ловушки.", "",
              "## Одинаковые начальные условия: сравнение методов", "",
              "DOP853: T/100, T/200, T/400 с rtol=2e−11,2e−13,4e−14 и отдельными SI-atol координат/скоростей (settings.json).",
              "Radau: T/50,T/100, rtol=2e−9,2e−11; точный Якобиан внутри элемента. Стадии, оценка ошибки и интерполянт отличаются от DOP853.",
              "Оба метода используют общий проверяющий механизм геометрии и выбора элемента; это независимая временная дискретизация, а не полностью независимый программный код.",
              "Выход — 64 отсчёта на RF-период. Медленная позиция выделена центрированным средним за один RF-период.",
              "Численные бюджеты координат сохранены: RMS≤0,1%, максимум≤0,3% фиксированного 1 мм.",
              "На коротком участке нельзя независимо измерить медленный период; проверка периода здесь не заявляется.", "",
              "| участок | сравнение | RMS медленной позиции, % | максимум, % | в бюджете |", "|---|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['window']} | {r['pair']} | {100*r['slow_R_rms']:.6g} | {100*r['slow_R_max']:.6g} | {r['within_coordinate_budget']} |")
    lines += ["", "## Усиление начального различия", "",
              "plus — точное полное состояние старого T/100 в момент перезапуска; common — состояние аудируемого T/100.",
              "minus=common−(plus−common). Преобразование медленного движения при перезапуске не выполняется; сравниваются полные физические состояния.",
              "plus проверен при T/200 и T/400; minus — дополнительная диагностика симметрии, без отдельной проверки его ODE-сходимости.", "",
              "| участок | начальная норма различия | максимальное усиление | конечное усиление | сигнал / изменение при уточнении |", "|---|---:|---:|---:|---:|"]
    for r in summary["sensitivity"]:
        lines.append(f"| {r['window']} | {r['initial_scaled_state_difference']:.6g} | {r['max_amplification']:.6g} | {r['final_amplification']:.6g} | {r['signal_to_numerical_change']:.6g} |")
    lines += ["", "Максимум и конец различаются из-за колебаний. Эти отношения — конечное усиление конкретного возмущения на указанном интервале; они не являются показателем Ляпунова и не доказывают хаос.", "",
              "Начальные различия координат и скоростей отдельно записаны в perturbation_growth.csv. Диагностика antisymmetry сравнивает plus+minus−2common с plus−minus; у rapid и long дефект порядка единицы. Линейность реакции на всём интервале не установлена; для minus отдельный уровень уточнения не рассчитывался.", "",
              "## Воспроизведение исторического расхождения", "",
              "Сравнивается разность новых траекторий plus−common с разностью исторических legacy100−audited100.",
              "Диагностический ориентир заранее записан в settings.json: RMS остатка≤10% RMS исторического различия; сигнал должен превышать изменения при уточнении минимум в 10 раз.", "",
              "| участок | RMS исторического различия / 1 мм | RMS остатка / RMS исторического различия | воспроизведено в пределах 10% |", "|---|---:|---:|---|"]
    for r in summary["replay"]:
        lines.append(f"| {r['window']} | {r['historical_gap_RMS_on_1mm']:.6g} | {r['residual_relative_to_historical_gap_RMS']:.6g} | {r['within_declared_replay_tolerance']} |")
    lines += ["", "## Границы элементов и ограничения вывода", "",
              "crossing_sensitivity.csv содержит нормальную скорость и скачок ускорения на всех переходах исходного аудируемого пилота.",
              "Для невырожденного пересечения стационарной границы с непрерывным состоянием диагностируется матрица S=I+(f_plus−f_minus)n_state^T/(n_state^T f_minus).",
              "Здесь n_state=(n_x,n_y,0,0), f=(v,a); единственный ненулевой блок добавки — δa·n^T/(n·v). Норма вычислена после масштабирования (r/R_scale,v/V_scale).",
              "При малой нормальной скорости чувствительность времени перехода может быть большой. Отдельные нормы матриц не перемножались для оценки общего усиления и сами по себе не устанавливают причину роста.",
              "Источник формулы: [Kong et al., Saltation Matrices](https://arxiv.org/abs/2306.06862). Это анализ чувствительности кусочно-гладкой численной модели, а не утверждение о физической ловушке.",
              "Даже согласие коротких перезапусков не доказывает точность всей траектории из t=0 на 0,07506 с.",
              "Физическая точность сырого P2-поля и сходимость по сеткам 50/35 мкм остаются непроверенными. Статус A для глобального сравнения A–B/A–C не изменён на passed.", "",
              "## Воспроизведение", "", "```powershell", "C:/Python314/python.exe scripts/diagnose_raw_fem_restarts.py --stage run --workers 4",
              "C:/Python314/python.exe scripts/diagnose_raw_fem_restarts.py --stage analyze", "```", "",
              "Повторный запуск проверяет подписи и SHA и использует готовые checkpoint. Исходные пилоты, FEM-базис и вычислительный архив сохранены.", ""]
    (OUTPUT/"RESTART_DIAGNOSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "run", "analyze", "history"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    settings = prepare()
    if args.stage == "history":
        historical_analysis(settings, raw_field(settings))
        return
    if args.stage in ("all", "run"):
        pending = [case for case in all_cases() if not valid(case, settings)]
        weights = {"dop100": 1, "dop200": 2, "dop400": 4, "radau50": 4, "radau100": 8}
        pending.sort(key=lambda c: settings["windows"][c["window"]]["RF_periods"]*weights[c["control"]], reverse=True)
        print(f"Restart cases pending/reused: {len(pending)}/{len(all_cases())-len(pending)}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(worker, case) for case in pending]
            for job in as_completed(jobs):
                result = job.result()
                print(json.dumps(result), flush=True)
                if result["status"] != "completed":
                    raise RuntimeError("Restart case did not complete")
    if args.stage in ("all", "analyze"):
        analyze(settings)


if __name__ == "__main__":
    main()
