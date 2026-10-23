"""Run numbered user steps 1-3: protocol, smooth benchmarks, single-mesh audit.

These are stages 0-2 in docs/ABC_VALIDATION_PLAN.md. No FEM rebuild occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from three_phase_trap.config import load_config
from three_phase_trap.fem_cache import fem_fingerprint,load_fem_cache
from three_phase_trap.validation_benchmarks import run_benchmarks
from three_phase_trap.validation_field_audit import run_field_audit


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def provenance(config):
    files=["scripts/run_abc_initial_checks.py",*list(str(p.relative_to(ROOT)).replace("\\","/") for p in (ROOT/"src/three_phase_trap").glob("*.py"))]
    return {"fem_fingerprint":fem_fingerprint(config),"python":platform.python_version(),
            "versions":{n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-fem","matplotlib")},
            "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("all","protocol","benchmarks","field-audit"),default="all")
    args=parser.parse_args()
    output=ROOT/"outputs/abc_validation"
    output.mkdir(parents=True,exist_ok=True)
    config=load_config(ROOT/"configs/gmsh_70um.toml")
    config=replace(config,particle=replace(config.particle,mass=1e-15,charge=1e-12,gamma=0.,gravity_enabled=False))
    protocol={"user_steps":[1,2,3],"detailed_plan_stages":[0,1,2],
              "scope":"Protocol, smooth analytic benchmarks, existing-mesh spatial audit; no physical FEM trajectory comparison yet",
              "fem_fingerprint":fem_fingerprint(config),
              "basis_sha256":json.loads((ROOT/"outputs/fem_cache/metadata.json").read_text())["arrays_sha256"],
              "pilot":{"mass_kg":1e-15,"charge_C":1e-12,"gamma":0.,"gravity":False,"Udc_V":0.,
                       "U0_V":250.,"frequency_Hz":100000.,"slow_position_m":[.001,0.],"slow_velocity_m_per_s":[0.,0.],"phase_rad":[0.,1.5707963267948966]},
              "predeclared_trajectory_tolerances":{"relative_position_rms":.01,"relative_position_max":.03,"relative_period":.01,"numerical_budget_fraction":.1},
              "benchmark_settings":{"mass":1.,"charge":1.,"omega_base":20.,"frequency_multipliers":[1,2,4],"max_steps_per_period":[50,100],"output_samples_per_period":64,
                                    "dimensionless":True,"duration":"at least one estimated slow period, rounded to an integer base RF period"},
              "audit_settings":{"harmonic_orders":[8,16,24,32],"field_fit_discrepancy_candidate_threshold":.01,
                                "spatial_accuracy":"Not established by a single-mesh fit; refined mesh required"}}
    path=output/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError("Existing preregistered protocol differs; use an explicitly versioned study directory")
    write(path,protocol)
    progress=json.loads((output/"progress.json").read_text()) if (output/"progress.json").exists() else {}
    print("Verifying and loading the existing FEM cache...",flush=True)
    _,_,_,interpolator=load_fem_cache(config,ROOT)
    progress["user_step_1_protocol"]="passed"
    write(output/"progress.json",progress)
    if args.stage in ("all","benchmarks"):
        summary=run_benchmarks(output/"benchmarks")
        summary["provenance"]=provenance(config)
        write(output/"benchmarks/summary.json",summary)
        progress["user_step_2_smooth_benchmarks"]=summary["status"]
        write(output/"progress.json",progress)
    if args.stage in ("all","field-audit"):
        print("Auditing raw P2 field on independent circles...",flush=True)
        summary=run_field_audit(interpolator,config,output/"field_audit")
        summary["provenance"]=provenance(config)
        write(output/"field_audit/summary.json",summary)
        progress["user_step_3_spatial_audit"]=summary["status"]
        write(output/"progress.json",progress)
    write_report(output,protocol,progress)
    print(json.dumps(progress),flush=True)


def write_report(output,protocol,progress):
    lines=["# Первые три пункта проверки A/B/C", "", "Нумерация пользователя 1–3 соответствует этапам 0–2 подробного плана.",
           "Реальные FEM-траектории моделей A/B/C на этом этапе не сравнивались. Готовый FEM-кэш не пересчитывался.",
           "", "## 1. Протокол", "", "Пилот: m=1e−15 кг, q=+1e−12 Кл, U0=250 В, f=100 кГц, без DC, трения и гравитации.",
           "R0=(1 мм,0), V0=0, RF-фазы 0 и π/2. Критерии записаны до запуска в `protocol.json`:",
           "RMS координат ≤1%, максимум ≤3%, ошибка периода ≤1%, численная погрешность ≤10% каждого допуска.",
           "Это критерии будущего сравнения FEM-траекторий; величины ошибок гладких эталонов ниже имеют отдельный смысл.",
           "", "## 2. Гладкие эталоны", ""]
    file=output/"benchmarks/summary.json"
    if file.exists():
        benchmarks=json.loads(file.read_text())
        lines += ["Интегратор полного движения проверен по независимой матричной экспоненте для вращающегося квадруполя.",
                  "Также проверены нормировка усреднения, знак силы, обратимость ведущего преобразования начальных условий,",
                  "энергия усреднённой системы, шаг ODE и чувствительность периодного фильтра к частоте дискретизации.",
                  "Величины эталонов безразмерные: они проверяют алгоритм, не режим конкретной частицы.",
                  "", "| n | фаза | множитель f и U | относительный RMS координаты | относительный максимум | относительная ошибка ODE при уменьшении шага |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for r in benchmarks["cases"]:
            lines.append(f"| {r['order']} | {r['phase_rad']:.3f} | {r['frequency_multiplier']} | {r['averaged_vs_slow']['relative_rms']:.5g} | {r['averaged_vs_slow']['relative_max']:.5g} | {r['ode_step_convergence']['relative_max']:.3g} |")
        exact=max(r["quadrupole_exact_error"]["relative_max"] for r in benchmarks["cases"] if r["quadrupole_exact_error"])
        ode=max(r["ode_step_convergence"]["relative_max"] for r in benchmarks["cases"])
        lines += ["", f"Максимальное относительное расхождение RF-квадруполя с матричной экспонентой: {exact:.4g};",
                  f"максимальное изменение RF-координаты при уменьшении шага и ужесточении допусков: {ode:.4g}."]
        lines += ["", "При увеличении частоты амплитуда повышалась пропорционально, так что ведущий псевдопотенциал сохранялся.",
                  "Конечное различие полного и усреднённого движения здесь является ошибкой ведущего усреднения и его начальных поправок,",
                  "а не ошибкой решения RF-уравнения. Прохождение эталонов не означает, что каждая строка укладывается в будущий допуск 1%.",
                  "Длительность эталонов — около одного медленного периода; будущий FEM-пилот требует не менее трёх."]
    else:
        lines.append("Ещё не выполнено.")
    file=output/"field_audit/summary.json"
    lines += ["", "## 3. Пространственный аудит сетки 70 мкм", ""]
    if file.exists():
        audit=json.loads(file.read_text())
        holdout=audit["order32_holdout"]
        lines += ["Квадратуры и гармоники извлечены из нового кэша. Потенциал аппроксимирован полиномами до порядков 8,16,24,32;",
                  "проверочные точки имеют другие углы. Для радиусов 0,5 и 6 мм обучающие и проверочные окружности совпадают, но узлы различны.",
                  "В таблице приведено расхождение сырого P2-поля и независимой по углам гармонической аппроксимации N=32.",
                  "Это диагностическая оценка чувствительности представления, **не абсолютная ошибка FEM**.",
                  "", "| r, мм | RF RMS, В/м | FEM–fit RMS / FEM RMS | C4 из потенциала, В/м⁴ | η внутри элементов, максимум 64 фаз | ξmax, м |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for r,h in zip(audit["radii"],holdout):
            lines.append(f"| {r['radius_m']*1e3:g} | {r['rf_vector_rms_V_per_m']:.6g} | {h['field_relative_rms_discrepancy']:.6g} | {r['C4_amplitude_from_potential_V_per_m4']:.6g} | {r['eta_elementwise_cycle_sample_max']:.6g} | {r['micromotion_amplitude_max_m']:.6g} |")
        lines += ["", "Кандидатные проверенные радиусы с расхождением FEM–fit ниже 1%: "+str([r*1e3 for r in audit["candidate_radii_below_1percent_fit_discrepancy_m"]])+" мм.",
                  "Они не образуют автоматически непрерывный интервал доказанной точности.",
                  "η рассчитана по производной внутри треугольника и не включает скачки на рёбрах; ξ — ведущая амплитуда микродвижения.",
                  "Данные о скачках сохранены отдельно в `field_audit/interelement_jumps.csv`.",
                  "Производные на межэлементных границах и в вершинах не являются единственными значениями производной сырого поля.",
                  "На этапе первоначального аудита гладкое восстановление использовалось только для диагностики; B построена позднее, см. продолжение.",
                  "Полевая точность и сходимость силы остаются **не установленными** до сравнения с более мелкой сеткой."]
        pilot=audit["pilot_1mm_representation_discrepancy"]
        lines += ["", f"Для пилотного радиуса 1 мм расхождение представлений составляет {pilot*100:.4f}%;",
                  "бюджет 0,1% задан для RMS координат, поэтому локальное расхождение поля не равно ошибке траектории.",
                  "Это не измерение абсолютной ошибки FEM, но оснований объявить пилот количественно проверенным пока нет.",
                  "Аудит N≤32 на r=8 мм требует дополнительной проверки устойчивости по порядку гармонического восстановления:",
                  "малое расхождение N=32 с исходным FEM само по себе не устанавливает сходимость восстановления.",
                  "", "| Диапазон r, мм | число проверенных рёбер | медиана относительного скачка поля | 95-й процентиль |",
                  "|---|---:|---:|---:|"]
        for row in audit["jump_radial_bins"]:
            lines.append(f"| {row['r_min_mm']:g}–{row['r_max_mm']:g} | {row['sampled_facets']} | {row['relative_jump_median']:.5g} | {row['relative_jump_p95']:.5g} |")
    else:
        lines.append("Ещё не выполнено.")
    if progress.get("user_step_8_frequency_limit") in ("smooth_high_frequency_trend_verified_raw_FEM_unresolved", "frequency_limit_inconclusive"):
        next_stage = "Пункт 8 рассчитан: см. `frequency_limit/FREQUENCY_LIMIT_REPORT.md`. Серия f и U ×1,2,4 использует одно физическое время и общие медленные начальные условия; вывод по гладкому A_smooth не заменяет нерешённую проверку исходного FEM A. Далее — пространственная сходимость и разрешение неопределённости raw P2/FEM."
    elif str(progress.get("user_step_7_long_pilot", "")).startswith("pilot_computed_"):
        next_stage = "Пункт 7 рассчитан: см. `pilot/PILOT_REPORT.md`. Гладкое A_smooth–B проходит критерии за 3,25 медленного периода, но не за 10,25; сравнения с исходным FEM A пока inconclusive из-за несходимости его траекторий. Далее требуется контроль высокочастотного предела и отдельное разрешение неопределённости raw P2/FEM."
    elif progress.get("user_step_6_initial_conditions_and_filters") == "implemented_diagnostics_passed_long_pilot_unverified":
        next_stage = "Пункт 6 выполнен: см. `filter_checks/INITIAL_CONDITIONS_REPORT.md` и `initial_conditions.json`. Далее — длительный пилот A/A_smooth/B/C. Пока выполнены только короткие RF-диагностики A_smooth, без траектории исходного FEM A."
    elif str(progress.get("user_step_5_model_c", "")).startswith("implemented_"):
        next_stage = "Пункты 4–5 выполнены отдельно: см. `model_b/MODEL_B_REPORT.md` и `model_c/MODEL_C_REPORT.md`. Далее — общие начальные условия и выделение медленного движения; A–B пока не сравнивались."
    elif str(progress.get("user_step_4_model_b", "")).startswith("implemented_"):
        next_stage = "Пункт 4 выполнен отдельно: см. `model_b/MODEL_B_REPORT.md`. Следующий этап — модель C; A–B пока не сравнивались."
    else:
        next_stage = "Следующий этап — проверенное представление силы модели B с отдельным контролем ошибки восстановления."
    lines += ["", "## Продолжение", "", next_stage,
              "Перед количественным выводом о применимости усреднения требуется пространственная сходимость.",
              "Код запуска: `python scripts/run_abc_initial_checks.py`; отдельные этапы: `--stage benchmarks` и `--stage field-audit`.",
              "Результаты, ошибки, исходные состояния, временные ряды и происхождение данных сохранены в JSON/CSV/NPZ рядом с отчётом.",
              "Готовый FEM-кэш и архив исходного FEM-расчёта не изменены."]
    (output/"INITIAL_CHECKS_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
