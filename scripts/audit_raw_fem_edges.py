"""Audit raw-P2 element crossings without recomputing FEM or replacing pilots.

--extent probe uses 64 RF periods; pilot uses the existing 3.25-Ts interval.
The audited method records every transition and checks every numerical dense
curve. This stage does not establish long-time or mesh convergence of A.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.audited_rf import integrate_audited
from three_phase_trap.config import load_config
from three_phase_trap.fem_cache import load_fem_cache, fem_fingerprint
from three_phase_trap.pilot_dynamics import AffineP2RF, coordinate_metrics
from three_phase_trap.slow_analysis import centered_period_mean
from run_abc_pilot import load_models, sha, write

OUTPUT = ROOT/"outputs/abc_validation/raw_edge_audit/v2"


def calculate(case, protocol, initial, model, raw):
    original = json.loads((OUTPUT.parents[1]/"pilot/settings.json").read_text(encoding="utf-8"))
    control = ({"RF_steps_per_period": 200, "RF_rtol": 2e-13,
                "RF_atol_SI": [2e-17, 2e-17, 2e-15, 2e-15]} if case["level"] == "refined" else original["controls"]["fine"])
    periods = 64 if case["extent"] == "probe" else original["RF_periods"]["pilot"]
    source_names = ("audited_rf.py", "pilot_dynamics.py", "interpolation.py", "event_resolved_rf.py")
    provenance = {"fem_fingerprint": protocol["fem_fingerprint"], "basis_sha256": protocol["basis_sha256"],
                  "sources": {n: sha(ROOT/"src/three_phase_trap"/n) for n in source_names},
                  "initial_conditions_sha256": sha(OUTPUT.parents[1]/"initial_conditions.json"),
                  "worker_sha256": hashlib.sha256(inspect.getsource(calculate).encode()).hexdigest(),
                  "python": platform.python_version(), "numpy": np.__version__}
    import scipy
    provenance["scipy"] = scipy.__version__
    identity = {"case": case, "controls": control, "periods": periods, "provenance": provenance}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    tag = f"A_phase{case['phase_index']}_{case['level']}_{case['extent']}"
    arrays, metadata = OUTPUT/(tag+".npz"), OUTPUT/(tag+".json")
    transitions = OUTPUT/(tag+"_crossings.jsonl")
    if metadata.exists():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record["signature"] != signature:
            raise ValueError("Audit source/parameters changed; preserve results and use a versioned folder")
        if record["status"] == "completed" and sha(arrays) == record["arrays_sha256"] and sha(transitions) == record["crossings_sha256"]:
            print(f"Reusing verified {tag}", flush=True)
            return record
        if record["status"] != "completed":
            raise ValueError(f"Prior diagnostic failure retained: {metadata.name}; inspect it before rerunning")
        raise ValueError("Audit checkpoint checksum mismatch")
    print(f"Auditing {tag}: {periods} RF periods", flush=True)
    phase = protocol["pilot"]["phase_rad"][case["phase_index"]]
    with transitions.open("w", encoding="utf-8", newline="\n") as stream:
        def log(value):
            stream.write(json.dumps(value, allow_nan=False)+"\n")
            stream.flush()
        try:
            result = integrate_audited(raw, initial["cases"][case["phase_index"]]["A_and_A_smooth_full_state"],
                                       model.charge, model.mass, model.frequency, phase, 0., periods/model.frequency,
                                       max_steps_per_period=control["RF_steps_per_period"], rtol=control["RF_rtol"],
                                       atol=control["RF_atol_SI"], on_crossing=log,
                                       progress=lambda value: print(json.dumps(value), flush=True))
        except Exception as error:
            write(metadata, {**identity, "signature": signature, "status": "audit_failed",
                             "error_type": type(error).__name__, "message": str(error)})
            raise
    run = result.run
    np.savez_compressed(arrays, time_s=run.time, states=run.states, sample_triangles=result.sample_triangles)
    record = {**identity, "signature": signature, "status": run.status,
              "arrays_sha256": sha(arrays), "crossings_sha256": sha(transitions),
              "diagnostics": result.diagnostics, "nfev": run.nfev, "wall_seconds": run.wall_seconds,
              "max_radius_m": float(np.linalg.norm(run.states[:, :2], axis=-1).max()),
              "end_time_s": float(run.time[-1]), "initial_state": run.states[0].tolist()}
    if case["extent"] == "pilot" and case["level"] == "fine":
        previous_path = OUTPUT.parents[1]/"pilot/edge_resolved"/f"A_phase{case['phase_index']}_fine_pilot.npz"
        previous_meta = json.loads(previous_path.with_suffix(".json").read_text(encoding="utf-8"))
        if previous_meta["provenance"]["edge_resolved_source_sha256"] != provenance["sources"]["event_resolved_rf.py"] or sha(previous_path) != previous_meta["arrays_sha256"]:
            raise ValueError("Legacy comparison checkpoint/source mismatch")
        with np.load(previous_path) as old:
            t, states = old["time_s"], old["states"]
        if np.max(abs(t-run.time)) > 1e-13:
            raise ValueError("Legacy/audited comparison grids differ")
        means = centered_period_mean(run.states[:, :2], run.time, 64)[1]
        old_means = centered_period_mean(states[:, :2], t, 64)[1]
        record["vs_legacy_same_nominal_controls"] = {
            "legacy_arrays_sha256": sha(previous_path),
            "full_positions": coordinate_metrics(run.states[:, :2], states[:, :2], .001),
            "centred_slow_positions": coordinate_metrics(means, old_means, .001),
            "interpretation": "Difference of numerical methods, not an absolute trajectory error or convergence proof"}
    write(metadata, record)
    print(json.dumps({"case": case, "diagnostics": result.diagnostics, "seconds": run.wall_seconds,
                      "comparison": record.get("vs_legacy_same_nominal_controls")}), flush=True)
    return record


def report():
    records = [json.loads(p.read_text(encoding="utf-8")) for p in OUTPUT.glob("A_*.json")]
    lines = ["# Аудит переходов между FEM-треугольниками: пункт 1", "",
             "FEM-базис 70 мкм не пересчитывался. Прежний event_resolved_rf.py и его результаты сохранены.",
             "Новый метод: src/three_phase_trap/audited_rf.py; запуск: scripts/audit_raw_fem_edges.py.", "",
             "## Механизм", "",
             "DOP853 продвигается по одному шагу. Его полином плотного вывода степени 7 переводится в барицентрические полиномы Бернштейна.",
             "Выпуклая оболочка коэффициентов и рекурсивное подразделение проверяют весь сохраняемый участок шага, включая промежутки между отсчётами.",
             "Внутренние экстремумы позволяют найти выход и возврат при одинаковых знаках на концах; сохраняется только префикс до первого выхода.",
             "Это численная проверка интерполянта с явным барицентрическим допуском, не строгая оценка ошибки точной траектории.",
             "На границе проверяется весь набор инцидентных треугольников. Направление определяется скоростью и следующими производными.",
             "Совпадение первых производных не считается достаточным: при нескольких допустимых элементах проверяется эквивалентность полных локальных RF-законов.",
             "Положение не сдвигается, скорость не меняется. Неоднозначные продолжения и неразрешённые геометрические случаи завершаются диагностической ошибкой.",
             "Журнал crossings.jsonl хранит время, оба элемента, барицентрические координаты, состояние, скачок ускорения и признаки вершины/пропуска по знакам концов.", "",
             "## Аналитические проверки", "",
             "14 специальных тестов проверяют выход-возврат (включая воспроизведение пропуска настоящим solve_ivp), узкий выход, касание, движение вдоль ребра, вершины и независимые аналитические траектории.",
             "Отдельный пример проверяет неоднозначность, которая проявляется лишь в пятой производной координаты. Полный набор проекта: 70 тестов пройдены.", "",
             "## Выполненные FEM-проверки", "",
             "| случай | статус | проверено шагов | переходы | выходы, невидимые по знакам концов | минимум барицентрической границы |", "|---|---|---:|---:|---:|---:|"]
    details = []
    for record in sorted(records, key=lambda r: str(r["case"])):
        case, d = record["case"], record.get("diagnostics", {})
        label = f"phase{case['phase_index']}, {case['level']}, {case['extent']}"
        lines.append(f"| {label} | {record['status']} | {d.get('checked_RK_steps', '—')} | {d.get('crossings', '—')} | {d.get('endpoint_misses_detected', '—')} | {d.get('minimum_certified_barycentric_bound', '—')} |")
        if "vs_legacy_same_nominal_controls" in record:
            difference = record["vs_legacy_same_nominal_controls"]["centred_slow_positions"]
            details += ["", f"Для {label} отличие от прежнего метода с теми же номинальными настройками: RMS={100*difference['relative_rms']:.6g}%, max={100*difference['relative_max']:.6g}% от фиксированного 1 мм."]
        if record["status"] != "completed":
            details += ["", f"Диагностическая остановка: {record.get('error_type')}: {record.get('message')}"]
    lines += details
    lines += ["", "## Ограничение вывода", "",
              "Аудит геометрии и аналитические тесты не устанавливают длительную ODE-сходимость A и пространственную сходимость FEM.",
              "Сравнение с прежним методом не является независимым точным решением. Нулевой счётчик пропущенных событий в конкретном запуске не исключает другие причины прежней чувствительности.",
              "Пункты 2 (локализация роста расхождения) и 3 (пространственная сходимость) предыдущего плана остаются отдельными задачами.", "",
              "Воспроизведение: `C:/Python314/python.exe scripts/audit_raw_fem_edges.py --extent pilot --level fine --phase-index 0`.",
              "Быстрая проверка: `--extent probe --phase-index 1` (64 RF-периода). Все файлы сохраняются отдельно от pilot/edge_resolved.", ""]
    (OUTPUT/"EDGE_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--level", choices=("fine", "refined"), default="fine")
    parser.add_argument("--extent", choices=("probe", "pilot"), default="probe")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol, initial, model, _ = load_models()
    study_inputs = {"protocol_sha256": sha(OUTPUT.parents[1]/"protocol.json"),
                    "initial_conditions_sha256": sha(OUTPUT.parents[1]/"initial_conditions.json"),
                    "physical_parameters": protocol["pilot"], "basis_sha256": protocol["basis_sha256"]}
    binding = OUTPUT/"inputs.json"
    if binding.exists() and json.loads(binding.read_text(encoding="utf-8")) != study_inputs:
        raise ValueError("Audit inputs changed; use a new versioned study folder")
    write(binding, study_inputs)
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    if fem_fingerprint(config) != protocol["fem_fingerprint"] or sha(ROOT/"outputs/fem_cache/p2_basis.npz") != protocol["basis_sha256"]:
        raise ValueError("FEM fingerprint or basis checksum mismatch")
    _, _, _, interp = load_fem_cache(config, ROOT)
    try:
        calculate(vars(args), protocol, initial, model, AffineP2RF(interp, model.amplitude))
    finally:
        report()
    write(OUTPUT/"integrity.json", {"basis_sha256_after": sha(ROOT/"outputs/fem_cache/p2_basis.npz"),
                                   "original_bundle_sha256_after": sha(ROOT/"outputs/fem_cache/gmsh_70um_fem_bundle.zip")})
    write(OUTPUT/"manifest.json", {"driver_sha256": sha(Path(__file__)),
                                  "artifacts": {p.name: sha(p) for p in sorted(OUTPUT.iterdir())
                                                if p.is_file() and p.name != "manifest.json"}})


if __name__ == "__main__":
    main()
