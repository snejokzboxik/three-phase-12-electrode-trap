"""Read-only preview or final analysis of diagnostic step 3, no FEM solves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.spatial_convergence import relative_rms
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.pilot_dynamics import coordinate_metrics, oscillation_metrics
from three_phase_trap.slow_analysis import centered_period_mean
from run_abc_pilot import sha, write
from analyze_abc_pilot import csv_write
from run_raw_fem_spatial import OUTPUT, BASE, check_complete


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def coordinate_comparison(actual, reference, t, scale, averaged=False):
    full = coordinate_metrics(actual[:, :2], reference[:, :2], scale)
    if averaged:
        slow = full
    else:
        _, a = centered_period_mean(actual[:, :2], t, 64)
        _, b = centered_period_mean(reference[:, :2], t, 64)
        slow = coordinate_metrics(a, b, scale)
    return {"full_position_rms": full["relative_rms"], "full_position_max": full["relative_max"],
            "slow_position_rms": slow["relative_rms"], "slow_position_max": slow["relative_max"]}


def field_analysis(settings):
    samples, levels, origins, jumps, fits, representation = {}, [], [], [], [], []
    n = settings["holdout_count"]
    for h in settings["meshes_um"]:
        d = OUTPUT/f"h{h}"
        if not check_complete(d/"field_complete.json", settings):
            raise ValueError(f"Field checkpoint h={h} is incomplete")
        samples[h] = data = read_arrays(d/"common_samples.npz")
        meta = json.loads((d/"field_complete.json").read_text(encoding="utf-8"))
        rf = RFReconstruction.load(d/"reconstruction_N48.json")
        q = rf.unit_quadratures
        c4 = float(np.sqrt((abs(q.cosine[4])**2+abs(q.sine[4])**2)/2)/q.reference_radius**4)
        levels.append({"mesh_um": h, "triangles": meta["triangles"], "P2_dofs": meta["dofs"],
                       "C4_per_volt_per_m4": c4, "fingerprint": meta["fingerprint"],
                       "discrete_scaled_residual_max": max(meta["discrete_scaled_residuals"].values()),
                       "DC_superposition_max_V": meta["DC_superposition_max_V"]})
        regions = [(float(r), slice(i*n, (i+1)*n)) for i, r in enumerate(settings["holdout_radii_m"])]
        regions += [(None, slice(int(data["ring_points_count"]), len(data["points_m"])))]
        for radius, cut in regions:
            tag = "ring" if radius is not None else "saved_pilot_path"
            rep = {"mesh_um": h, "region": tag, "radius_m": radius,
                   "raw_RF_rms_V_per_m": float(np.sqrt(np.mean(np.sum(data["raw_field"][cut]**2, axis=(1, 2)))/2)),
                   "raw_fit_field_relative_rms": relative_rms(data["raw_field"][cut], data["N48_field"][cut]),
                   "raw_fit_psi_relative_rms": relative_rms(data["raw_psi"][cut], data["N48_psi"][cut]),
                   "raw_elementwise_fit_force_relative_rms": relative_rms(data["raw_force_elementwise"][cut], data["N48_force"][cut])}
            representation.append(rep)
            for label in ("N40", "N56", "alternative"):
                fits.append({"mesh_um": h, "region": tag, "radius_m": radius, "comparison": label+"-N48",
                             **{quantity+"_relative_rms": relative_rms(data[label+"_"+quantity][cut], data["N48_"+quantity][cut])
                                for quantity in ("field", "psi", "force")}})
        origin = read_arrays(d/"origin_limits.npz")
        field = origin["one_sided_field"]
        spread = np.sqrt(np.sum((field[:, None]-field[None, :])**2, axis=(2, 3))/2)
        origins.append({"mesh_um": h, "incident_elements": len(field),
                        "max_one_sided_origin_RF_rms_V_per_m": float(origin["one_sided_rf_rms"].max()),
                        "max_pairwise_origin_RF_rms_difference_V_per_m": float(spread.max()),
                        "smooth_origin_RF_rms_V_per_m": float(np.sqrt(sum(np.sum(e*e) for e in rf.scaled(settings["physical"]["U0_V"]).field_quadratures(np.zeros(2)))/2))})
        facets = read_arrays(d/"facet_samples.npz")
        for low, high in zip((0., .075, .15, .375, .75, 1.5, 3., 5., 7.), (.075, .15, .375, .75, 1.5, 3., 5., 7., 8.)):
            cut = (facets["radius_m"] >= low*1e-3) & (facets["radius_m"] < high*1e-3)
            if not cut.any():
                continue
            jump, reference = facets["jump_rf_rms"][cut], facets["mean_field_rf_rms"][cut]
            good = reference > 0
            relative = jump[good]/reference[good]
            jumps.append({"mesh_um": h, "r_min_mm": low, "r_max_mm": high, "facets": int(cut.sum()),
                          "absolute_jump_RMS_V_per_m": float(np.sqrt(np.mean(jump**2))),
                          "jump_RMS_over_field_RMS": float(np.linalg.norm(jump)/np.linalg.norm(reference)),
                          "pointwise_relative_jump_median": float(np.median(relative)),
                          "pointwise_relative_jump_p95": float(np.quantile(relative, .95)),
                          "max_tangential_jump_V_per_m": float(facets["tangential_jump_rf_rms"][cut].max())})
    mesh_rows = []
    for a, b in ((70, 50), (50, 35), (70, 35)):
        if not np.array_equal(samples[a]["points_m"], samples[b]["points_m"]):
            raise ValueError("Mesh fields must be compared at exactly common physical points")
        for radius, cut in regions:
            row = {"coarse_um": a, "fine_um": b, "region": "ring" if radius is not None else "saved_pilot_path", "radius_m": radius}
            for quantity in ("phi", "field", "psi", "force_elementwise", "jacobian"):
                row["raw_"+quantity+"_relative_rms"] = relative_rms(samples[a]["raw_"+quantity][cut], samples[b]["raw_"+quantity][cut])
            for quantity in ("phi", "field", "psi", "force"):
                row["smooth_"+quantity+"_relative_rms"] = relative_rms(samples[a]["N48_"+quantity][cut], samples[b]["N48_"+quantity][cut])
            mesh_rows.append(row)
    for name, rows in (("mesh_levels", levels), ("mesh_field_comparison", mesh_rows), ("representation_discrepancy", representation),
                       ("reconstruction_sensitivity", fits), ("origin_limits", origins), ("facet_jump_bins", jumps)):
        csv_write(OUTPUT/(name+".csv"), rows)
    return samples, levels, origins, mesh_rows, representation, fits


def trajectory_analysis(settings):
    paths, comparisons, numerical, average_pairs = {}, [], [], []
    budget = settings["coordinate_budgets"]
    original_tolerances = json.loads((BASE/"protocol.json").read_text(encoding="utf-8"))["predeclared_trajectory_tolerances"]
    period_budget = original_tolerances["relative_period"]*original_tolerances["numerical_budget_fraction"]
    def add(rows, h, label, a, b, averaged=False, period_check=False):
        if not np.allclose(a["time_s"], b["time_s"], rtol=0., atol=1e-13):
            raise ValueError("Trajectory clocks differ")
        m = coordinate_comparison(a["states"], b["states"], a["time_s"], settings["R_scale_m"], averaged)
        row = {"mesh_um": h, "comparison": label, **m,
               "within_coordinate_budget": bool(m["slow_position_rms"] <= budget["relative_rms"] and m["slow_position_max"] <= budget["relative_max"]),
               "period_relative_difference": None, "within_period_budget": None}
        if period_check:
            t = a["time_s"]
            if averaged:
                pa, pb, tc = a["states"][:, :2], b["states"][:, :2], t
            else:
                tc, pa = centered_period_mean(a["states"][:, :2], t, 64)
                _, pb = centered_period_mean(b["states"][:, :2], t, 64)
            expected = settings["estimated_slow_period_s"]
            aa = oscillation_metrics(tc, pa, expected)["mean_period_s"]
            bb = oscillation_metrics(tc, pb, expected)["mean_period_s"]
            period_error = abs(aa/bb-1) if aa is not None and bb is not None else None
            row["period_relative_difference"] = period_error
            row["within_period_budget"] = bool(period_error is not None and period_error <= period_budget)
        rows.append(row)
    for h in settings["meshes_um"]:
        d = OUTPUT/f"h{h}"
        if not check_complete(d/"trajectories_complete.json", settings):
            raise ValueError(f"Incomplete trajectory checkpoint h={h}")
        paths[h] = data = {path.stem: read_arrays(path) for path in d.glob("*.npz") if path.stem.startswith(("A_smooth_", "B_", "local_"))}
        for model in ("B", "A_smooth"):
            add(numerical, h, model+" coarse-fine", data[model+"_coarse"], data[model+"_fine"], model == "B", period_check=True)
        add(numerical, h, "local raw DOP200-DOP400", data["local_raw_dop200"], data["local_raw_dop400"])
        add(numerical, h, "local smooth T100-T200", data["local_smooth_100"], data["local_smooth_200"])
        add(comparisons, h, "local raw-smooth", data["local_raw_dop400"], data["local_smooth_200"])
        t = data["A_smooth_fine"]["time_s"]
        tc, slow = centered_period_mean(data["A_smooth_fine"]["states"][:, :2], t, 64)
        averaged = data["B_fine"]["states"][32:-32, :2]
        m = coordinate_metrics(slow, averaged, settings["R_scale_m"])
        # Period is diagnostic for this smooth pair; raw local windows cannot
        # determine a secular period independently.
        expected = settings["estimated_slow_period_s"]
        a_period = oscillation_metrics(tc, slow, expected)
        b_period = oscillation_metrics(tc, averaged, expected)
        average_pairs.append({"mesh_um": h, "slow_position_relative_rms": m["relative_rms"],
                              "slow_position_relative_max": m["relative_max"], "RF_phase": 0.,
                              "smooth_period": a_period, "B_period": b_period})
    for a, b in ((70, 50), (50, 35), (70, 35)):
        for model in ("B", "A_smooth", "local_raw", "local_smooth"):
            tag = model+"_fine" if model in ("B", "A_smooth") else ("local_raw_dop400" if model == "local_raw" else "local_smooth_200")
            add(comparisons, f"{a}-{b}", model+" mesh comparison", paths[a][tag], paths[b][tag], model == "B", period_check=model in ("B", "A_smooth"))
    csv_write(OUTPUT/"trajectory_numerical_checks.csv", numerical)
    csv_write(OUTPUT/"trajectory_comparisons.csv", comparisons)
    write(OUTPUT/"smooth_averaging_by_mesh.json", average_pairs)
    return paths, numerical, comparisons, average_pairs


def plots(settings, fields, representations, mesh_rows, origins, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    for h in settings["meshes_um"]:
        rows = [r for r in representations if r["mesh_um"] == h and r["region"] == "ring"]
        r = [v["radius_m"]*1e3 for v in rows]
        ax[0, 0].loglog(r, [100*v["raw_fit_field_relative_rms"] for v in rows], "o-", label=f"{h} um")
        ax[1, 0].loglog(r, [100*v["raw_elementwise_fit_force_relative_rms"] for v in rows], "o-", label=f"{h} um")
    for a, b in ((70, 50), (50, 35)):
        rows = [r for r in mesh_rows if r["coarse_um"] == a and r["fine_um"] == b and r["region"] == "ring"]
        r = [v["radius_m"]*1e3 for v in rows]
        for mode, style in (("raw", "-"), ("smooth", "--")):
            ax[0, 1].loglog(r, [max(100*v[mode+"_field_relative_rms"], 1e-12) for v in rows], style, label=f"{mode} {a}-{b}")
    ax[0, 0].set(xlabel="radius, mm", ylabel="RF field difference, %", title="Raw field vs potential-based N48 fit")
    ax[0, 1].set(xlabel="radius, mm", ylabel="common-point RF field difference, %", title="Refinement: different meshes, same points")
    ax[1, 0].set(xlabel="radius, mm", ylabel="force difference, %", title="Elementwise grad(Psi_raw) vs smooth force")
    ax[1, 1].loglog([v["mesh_um"] for v in origins], [v["max_one_sided_origin_RF_rms_V_per_m"] for v in origins], "o-", label="max one-sided RF RMS at origin")
    ax[1, 1].set(xlabel="mesh size, um", ylabel="RF field, V/m", title="Central vertex: raw gradient has directional limits")
    for a in ax.flat:
        a.grid(alpha=.2)
        a.legend(fontsize=8)
    fig.savefig(OUTPUT/"spatial_field_convergence.png", dpi=160)
    plt.close(fig)
    if paths is None:
        return
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    for a, b in ((70, 50), (50, 35), (70, 35)):
        for model, axis in (("B_fine", ax[0, 0]), ("A_smooth_fine", ax[0, 1])):
            t = paths[a][model]["time_s"]
            diff = paths[a][model]["states"][:, :2]-paths[b][model]["states"][:, :2]
            axis.plot(t[::64]*1e3, 100*np.linalg.norm(diff[::64], axis=-1)/settings["R_scale_m"], label=f"{a}-{b} um")
    for h, data in paths.items():
        t = data["local_raw_dop400"]["time_s"]
        d = data["local_raw_dop400"]["states"][:, :2]-data["local_smooth_200"]["states"][:, :2]
        ax[1, 0].plot(t[::16]*1e3, 100*np.linalg.norm(d[::16], axis=-1)/settings["R_scale_m"], label=f"{h} um")
        d = data["local_raw_dop200"]["states"][:, :2]-data["local_raw_dop400"]["states"][:, :2]
        ax[1, 1].semilogy(t[::16]*1e3, np.maximum(100*np.linalg.norm(d[::16], axis=-1)/settings["R_scale_m"], 1e-14), label=f"{h} um")
    for axis, title in zip(ax.flat, ("B: mesh differences, common slow start", "Full smooth RF: mesh differences, common full start", "Local raw vs smooth, common full restart", "Local raw ODE refinement T200 vs T400")):
        axis.set(xlabel="global time, ms", ylabel="position difference / 1 mm, %", title=title)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.savefig(OUTPUT/"spatial_trajectory_convergence.png", dpi=160)
    plt.close(fig)


def report(settings, summary, levels, origins, mesh_rows, representation, numerical, comparisons, average_pairs):
    fine_pair = {r["comparison"]: r for r in comparisons if r["mesh_um"] == "50-35"}
    lines = ["# Пункт 3: пространственное представление и сетки 70/50/35 мкм", "",
             f"Статус: `{summary['status']}`.", "",
             f"Между сетками 50 и 35 мкм RMS-различие траекторий B = {100*fine_pair['B mesh comparison']['slow_position_rms']:.6g}%, A_smooth = {100*fine_pair['A_smooth mesh comparison']['slow_position_rms']:.6g}% масштаба 1 мм за 3,25 Ts.",
             f"Для локального сырого A межсеточное RMS-различие = {100*fine_pair['local_raw mesh comparison']['slow_position_rms']:.6g}%; бюджет координат 0,1% не выполнен. Численные проверки ODE при этом проходят.",
             "Гладкая ветвь получила межсеточный контроль в выбранном пилоте; исходный A ещё не подтверждён как пространственно сходящийся эталон.", "",
             "Новые FEM-решения рассчитаны отдельно. Геометрия, электроды, фазовая последовательность и физические параметры прежние.",
             "h=70/50/35 мкм в центре, внешний размер 350/250/175 мкм: уточняются также граничные многоугольники и внешняя область.",
             "Во всех случаях аффинная геометрия и P2-решение, симметричная сборка 12 секторов. Это три невложенные сетки.", "",
             "## FEM и гладкое представление", "",
             "| h, мкм | треугольники | P2 DOF | C4/U0, м⁻⁴ | масштабированный невязочный максимум |",
             "|---:|---:|---:|---:|---:|"]
    for r in levels:
        lines.append(f"| {r['mesh_um']} | {r['triangles']} | {r['P2_dofs']} | {r['C4_per_volt_per_m4']:.12g} | {r['discrete_scaled_residual_max']:.4g} |")
    lines += ["", "Проверены дискретные уравнения на свободных DOF и тождество A+B+C=DC. Малая невязка не ограничивает ошибку дискретизации.",
              "Потенциал восстанавливается тем же методом, что модель B: N48, все моды сохранены без проекции на симметрию и без подгонки к траекториям.",
              "Отдельно проверены N40, N56 и другая обучающая сетка. Исходная модель B при 70 мкм воспроизведена численно.",
              "Сравнения выполняются в одних физических точках: 513 углов на каждой окружности и 7507 точек сохранённой траектории A (один отсчёт на RF-период).",
              "Нормировка относительного различия RF-поля: sqrt(sum(|δEc|²+|δEs|²)/sum(|Ec_ref|²+|Es_ref|²)); ref — более мелкая сетка или N48 для сравнения представлений.",
              "Ψ=q²(|Ec|²+|Es|²)/(4mΩ²); гладкая сила =−q²(JcᵀEc+JsᵀEs)/(2mΩ²).",
              "raw_force_elementwise — только производная внутри треугольника; скачки и межэлементные распределительные слагаемые в неё не включены. Она не используется для интегрирования B.", "",
              "## Различия поля и представлений", "",
              "| r, мм | raw 70–50, % | raw 50–35, % | smooth 70–50, % | smooth 50–35, % |",
              "|---:|---:|---:|---:|---:|"]
    for radius in settings["holdout_radii_m"]:
        a, b = [next(r for r in mesh_rows if r["coarse_um"] == c and r["fine_um"] == f and r["radius_m"] == radius) for c, f in ((70, 50), (50, 35))]
        lines.append(f"| {radius*1e3:g} | {100*a['raw_field_relative_rms']:.6g} | {100*b['raw_field_relative_rms']:.6g} | {100*a['smooth_field_relative_rms']:.6g} | {100*b['smooth_field_relative_rms']:.6g} |")
    lines += ["", "Полные таблицы поля, Ψ, Якобиана и силы: mesh_field_comparison.csv. Сопоставление raw–N48 для каждой сетки: representation_discrepancy.csv.",
              "Критерий 0,1% для полевого сравнения зафиксирован до новых расчётов как диагностический ориентир. Он не является оценкой абсолютной ошибки FEM и не равен критерию координат.",
              f"Максимум различия N48-полей/Ψ/силы 50–35 на проверенных окружностях r≤1 мм: {100*summary['smooth_pilot_disk_max_mesh_difference']:.6g}%.",
              f"Максимум чувствительности N48 к N40/N56 и другой обучающей сетке при 35 мкм, r≤1 мм (поле/Ψ/сила): {100*summary['fine_reconstruction_pilot_disk_max_sensitivity']:.6g}%.", "",
              "## Центральная вершина и скачки", "",
              "В физической гладкой задаче E(0,t)=0 следует из инвариантности граничной задачи при повороте на 90°: E(0,t)=R90 E(0,t).",
              "Градиент непрерывного P2-потенциала может иметь разные односторонние значения. Они вычислены в одной точке (0,0) из всех прилегающих элементов без выбора произвольной стороны и без сдвигов координаты.", "",
              "| h, мкм | соседних элементов | max одностороннего RF RMS, В/м | max различия сторон RF RMS, В/м | N48 RF RMS в центре, В/м |",
              "|---:|---:|---:|---:|---:|"]
    for r in origins:
        lines.append(f"| {r['mesh_um']} | {r['incident_elements']} | {r['max_one_sided_origin_RF_rms_V_per_m']:.6g} | {r['max_pairwise_origin_RF_rms_difference_V_per_m']:.6g} | {r['smooth_origin_RF_rms_V_per_m']:.6g} |")
    lines += ["", "Нормальные/касательные скачки на рёбрах сохранены в facet_samples.npz; сводка — facet_jump_bins.csv. Внутри 1,5 мм взяты все рёбра, дальше каждое 13-е до 8 мм.",
              "Квантили скачков получены в разных физических точках разных сеток: это диагностика, а не строгая межсеточная оценка. Относительные ошибки около нуля поля требуют осторожной интерпретации; абсолютные значения сохранены.", "",
              "Выявленная центральная особенность — свойство численного представления градиента. Её наличие само по себе не устанавливает единственную причину всего длительного расхождения траекторий.", "",
              "## Траектории и независимый контроль ODE", "",
              "Полные RF-пилоты A_smooth и усреднённые B: фаза 0, 7506 RF-периодов =0,07506 с (≈3,25 медленного периода), по два уровня ODE.",
              "Для B одна медленная начальная пара; для полного RF одна сохранённая физическая начальная четвёрка на всех сетках. Пересогласование микродвижения отдельно для каждой сетки не выполняется: так измеряется влияние самого поля.",
              "Сырой A: локальный перезапуск 0,030–0,03512 с из того же полного состояния и глобальной RF-фазы на всех сетках, T/200,T/400 с геометрическим аудитом шага.",
              "Это не полная траектория raw A от t=0. Численные критерии: RMS≤0,1%, max≤0,3% масштаба 1 мм, без совмещения фаз или сдвига времени.", "",
              "| h, мкм | проверка ODE | RMS медленной позиции, % | max, % | в бюджете |", "|---|---|---:|---:|---|"]
    for r in numerical:
        lines.append(f"| {r['mesh_um']} | {r['comparison']} | {100*r['slow_position_rms']:.6g} | {100*r['slow_position_max']:.6g} | {r['within_coordinate_budget']} |")
    lines += ["", "| h / пара сеток | сравнение | RMS позиции, % | max, % | в бюджете 0,1%/0,3% |", "|---|---|---:|---:|---|"]
    for r in comparisons:
        lines.append(f"| {r['mesh_um']} | {r['comparison']} | {100*r['slow_position_rms']:.6g} | {100*r['slow_position_max']:.6g} | {r['within_coordinate_budget']} |")
    lines += ["", "Для B используется непосредственно медленная позиция; для RF — центрированное среднее за один период, 64 выходных интервала на RF-период.",
              "Для полных гладких пилотов также проверены периоды: допустимое изменение 0,1% = 10% исходного допуска периода 1%; значения в CSV. Короткому raw-перезапуску проверка медленного периода не приписывается.", "",
              "| h, мкм | A_smooth–B RMS, % | max, % |", "|---:|---:|---:|"]
    for r in average_pairs:
        lines.append(f"| {r['mesh_um']} | {100*r['slow_position_relative_rms']:.6g} | {100*r['slow_position_relative_max']:.6g} |")
    lines += ["", "Эта таблица характеризует выбранный пилот с общими стартами. Согласие гладкого поля на сетках не заменяет проверку исходного A.", "",
              "## Что установлено и что остаётся", "",
              f"Все численные проверки выбранных траекторий проходят бюджет: {summary['all_trajectory_numerical_checks_passed']}.",
              f"Гладкие поле/Ψ/сила 50–35 и контроль восстановления проходят диагностический порог в выборке r≤1 мм: {summary['smooth_spatial_screening_passed']}.",
              f"Гладкие траектории B и A_smooth 50–35 проходят бюджет координат: {summary['smooth_pilot_mesh_coordinate_checks_passed']}.",
              f"Сырой RF-gradient 50–35 проходит порог 0,1% на всех проверенных окружностях r≤1 мм: {summary['raw_pilot_disk_field_screening_passed']}.",
              "Три сетки дают эмпирическую проверку выбранных величин; строгая абсолютная оценка ошибки и независимый гранично-элементный/аналитический эталон не получены.",
              "Не доказаны глобальная ODE-сходимость сырого A на всём пилоте, физический хаос или универсальная применимость эффективного потенциала.",
              "Расширение радиусов/скоростей/фаз, длинный интервал 10,25 Ts на новых сетках и отдельный контроль более высокого порядка геометрии остаются вне этой диагностической выборки.",
              "Если raw A остаётся чувствительным, дальнейший вывод надо строить через проверенное пространственное представление и/или более точный FEM, с отдельной оценкой A–A_smooth; простое совпадение A_smooth–B не закрывает эту оценку.", "",
              "## Графики", "", "[Поле, сила и центральные градиенты](spatial_field_convergence.png).",
              "[Траектории и контроль ODE](spatial_trajectory_convergence.png).", "",
              "## Воспроизведение", "", "```powershell", "C:/Python314/python.exe scripts/run_raw_fem_spatial.py --stage all",
              "C:/Python314/python.exe scripts/analyze_raw_fem_spatial.py", "```", "",
              "Расчёты сеток выполняются последовательно в отдельных процессах для освобождения памяти. Готовые checkpoint проверяются по SHA; новые базисы находятся в outputs/fem_cache/variants/ по fingerprint.",
              "Исходный кэш 70 мкм, архив и старые траектории сохранены; их SHA проверены после расчёта. v1 — неудачный предварительный запуск записи JSON, научные результаты — v2.", ""]
    (OUTPUT/"SPATIAL_DIAGNOSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--fields-only", action="store_true", help="Analyze ready fields without assigning a trajectory/final study status")
    args = parser.parse_args()
    settings = json.loads((OUTPUT/"settings.json").read_text(encoding="utf-8"))
    if args.preview:
        for h in settings["meshes_um"]:
            d = OUTPUT/f"h{h}"
            field_ok = check_complete(d/"field_complete.json", settings)
            row = {"mesh_um": h, "field_complete": field_ok,
                   "trajectories_complete": check_complete(d/"trajectories_complete.json", settings)}
            if field_ok:
                data = read_arrays(d/"common_samples.npz")
                origin = read_arrays(d/"origin_limits.npz")
                row["origin_max_RF_rms_V_per_m"] = float(origin["one_sided_rf_rms"].max())
                row["raw_fit_field_RMS_percent"] = {}
                q = RFReconstruction.load(d/"reconstruction_N48.json").unit_quadratures
                row["C4_per_volt_per_m4"] = float(np.sqrt((abs(q.cosine[4])**2+abs(q.sine[4])**2)/2)/q.reference_radius**4)
                for radius in (.000025, .0001, .001):
                    index = int(np.argmin(abs(np.array(settings["holdout_radii_m"])-radius)))
                    cut = slice(index*settings["holdout_count"], (index+1)*settings["holdout_count"])
                    row["raw_fit_field_RMS_percent"][str(radius)] = 100*relative_rms(data["raw_field"][cut], data["N48_field"][cut])
            if row["trajectories_complete"]:
                raw = read_arrays(d/"local_raw_dop400.npz")
                smooth = read_arrays(d/"local_smooth_200.npz")
                check = read_arrays(d/"local_raw_dop200.npz")
                row["local_raw_smooth_position_RMS_percent"] = 100*coordinate_comparison(raw["states"], smooth["states"], raw["time_s"], settings["R_scale_m"])["slow_position_rms"]
                row["local_raw_numerical_position_RMS_percent"] = 100*coordinate_comparison(raw["states"], check["states"], raw["time_s"], settings["R_scale_m"])["slow_position_rms"]
                if h != 70 and check_complete(OUTPUT/"h70/trajectories_complete.json", settings):
                    for model in ("B", "A_smooth"):
                        current = read_arrays(d/(model+"_fine.npz"))
                        original = read_arrays(OUTPUT/"h70"/(model+"_fine.npz"))
                        row[model+"_mesh_change_vs70_RMS_percent"] = 100*coordinate_comparison(current["states"], original["states"], current["time_s"], settings["R_scale_m"], averaged=model == "B")["slow_position_rms"]
            print(json.dumps(row))
        return
    fields, levels, origins, mesh_rows, rep, fits = field_analysis(settings)
    fine = [r for r in mesh_rows if r["coarse_um"] == 50 and r["fine_um"] == 35 and r["region"] == "ring" and r["radius_m"] <= .00100000001]
    smooth_max = max(r["smooth_"+q+"_relative_rms"] for r in fine for q in ("field", "psi", "force"))
    raw_max = max(r["raw_field_relative_rms"] for r in fine)
    reconstruction_max = max(r[q+"_relative_rms"] for r in fits if r["mesh_um"] == 35 and r["region"] == "ring" and r["radius_m"] <= .00100000001 for q in ("field", "psi", "force"))
    if args.fields_only:
        plots(settings, fields, rep, mesh_rows, origins, None)
        value = {"status": "field_stage_analyzed_without_trajectory_status", "smooth_pilot_disk_max_mesh_difference": smooth_max,
                 "fine_reconstruction_pilot_disk_max_sensitivity": reconstruction_max, "raw_pilot_disk_max_mesh_field_difference": raw_max,
                 "levels": levels, "origins": origins}
        write(OUTPUT/"field_stage_summary.json", value)
        print(json.dumps(value), flush=True)
        return
    paths, numerical, comparisons, avg = trajectory_analysis(settings)
    spatial_ok = max(smooth_max, reconstruction_max) <= settings["field_comparison_target"]
    pilots_ok = all(r["within_coordinate_budget"] and r["within_period_budget"] is not False for r in comparisons if r["mesh_um"] == "50-35" and r["comparison"].startswith(("B ", "A_smooth ")))
    numerical_ok = all(r["within_coordinate_budget"] and r["within_period_budget"] is not False for r in numerical)
    summary = {"status": "smooth_spatial_and_pilot_mesh_checks_passed_raw_global_unverified" if spatial_ok and pilots_ok and numerical_ok else "spatial_refinement_computed_remaining_uncertainties",
               "all_trajectory_numerical_checks_passed": numerical_ok, "smooth_spatial_screening_passed": spatial_ok,
               "smooth_pilot_mesh_coordinate_checks_passed": pilots_ok, "raw_pilot_disk_field_screening_passed": raw_max <= settings["field_comparison_target"],
               "smooth_pilot_disk_max_mesh_difference": smooth_max, "fine_reconstruction_pilot_disk_max_sensitivity": reconstruction_max,
               "raw_pilot_disk_max_mesh_field_difference": raw_max, "levels": levels, "origins": origins,
               "scope": settings["scope"], "input_integrity": {}}
    for name, expected in settings["input_sha256"].items():
        actual = sha(ROOT/name)
        if actual != expected:
            raise ValueError(f"Original input was changed: {name}")
        summary["input_integrity"][name] = actual
    for name, expected in settings["kernel_sha256"].items():
        if sha(ROOT/"src/three_phase_trap"/name) != expected:
            raise ValueError(f"Numeric kernel changed: {name}")
    write(OUTPUT/"summary.json", summary)
    plots(settings, fields, rep, mesh_rows, origins, paths)
    report(settings, summary, levels, origins, mesh_rows, rep, numerical, comparisons, avg)
    progress = json.loads((BASE/"progress.json").read_text(encoding="utf-8"))
    progress["raw_FEM_spatial_diagnosis"] = summary["status"]
    write(BASE/"progress.json", progress)
    artifacts = {str(p.relative_to(OUTPUT)).replace("\\", "/"): sha(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write(OUTPUT/"manifest.json", {"builder_sha256": sha(ROOT/"scripts/run_raw_fem_spatial.py"), "analysis_sha256": sha(Path(__file__)), "artifacts": artifacts})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("input_integrity", "levels")}), flush=True)


if __name__ == "__main__":
    main()
