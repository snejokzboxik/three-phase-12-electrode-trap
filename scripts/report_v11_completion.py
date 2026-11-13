"""Validate and summarize measured V11 outputs without integrating trajectories."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = "ABC"
STATE_FIELDS = ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s", "phase_rad")
STATUSES = ("retained", "escaped", "numerical_error")
STAGE_TITLES = {
    "baseline": "Сохранённые карты фаз 0° и 45°",
    "phase90": "Фазовые XY-карты",
    "matched": "Карты при согласованной нормировке",
    "boundary_area": "Площадь удержания и границы",
    "velocity": "Карты начальных скоростей",
    "long": "Долгосрочное удержание",
    "fem": "Проверки h35/h50 и прямым P4",
}


@dataclass
class Dataset:
    stage: str
    kind: str
    path: Path
    config: dict
    rows: dict
    initial: dict
    allow_numerical_errors: bool


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_text(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_table(path, rows, columns):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def state_key(row):
    model = row["model"]
    state = tuple(float(row[field]) for field in STATE_FIELDS)
    if model not in MODEL_NAMES or len(model) != 1 or not all(map(math.isfinite, state)):
        raise ValueError(f"Invalid model or initial state: {row}")
    return (model, *[round(value, 14) for value in state])


def horizon(row, config):
    return float(row.get("horizon_RF") or config["horizon_RF"])


def sample_key(row, dataset):
    key = state_key(row)
    if dataset.kind == "fem":
        mesh = int(row.get("mesh_um") or dataset.config["mesh_um"])
        representation = row.get("representation") or dataset.config.get("representation", "N48")
        return (*key, mesh, representation.upper(), horizon(row, dataset.config))
    return key


def index_rows(rows, dataset, filename):
    result = {}
    for row in rows:
        key = sample_key(row, dataset)
        if key in result:
            raise ValueError(f"Duplicate initial state in {dataset.path / filename}: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"No records in {dataset.path / filename}")
    return result


def load_dataset(spec):
    path = (ROOT / spec["directory"]).resolve()
    config = read_json(path / "config.json")
    dataset = Dataset(spec["stage"], spec["kind"], path, config, {}, {}, spec.get("allow_numerical_errors", False))
    dataset.initial = index_rows(read_csv(path / "initial_states.csv"), dataset, "initial_states.csv")
    dataset.rows = index_rows(read_csv(path / "results.csv"), dataset, "results.csv")
    if dataset.initial.keys() != dataset.rows.keys():
        missing = len(dataset.initial.keys() - dataset.rows.keys())
        extra = len(dataset.rows.keys() - dataset.initial.keys())
        raise ValueError(f"Result states differ from initial states in {path}: missing={missing}, extra={extra}")
    if dataset.kind == "xy":
        validate_grid(dataset)
    if dataset.kind == "velocity":
        validate_velocity(dataset)
    for key, row in dataset.rows.items():
        try:
            validate_result(row, config)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid output in {path} for {key}: {exc}") from exc
    return dataset


def validate_result(row, config):
    status = row["status"]
    target = horizon(row, config)
    reached = float(row["reached_RF"])
    aperture = config.get("aperture_m", .008)
    expected_voltage = config.get("voltage_V")
    if isinstance(expected_voltage, dict):
        expected_voltage = expected_voltage[row["model"]]
    if expected_voltage is not None and row.get("voltage_V"):
        if not math.isclose(float(row["voltage_V"]), float(expected_voltage), rel_tol=1e-12):
            raise ValueError("voltage differs from declared configuration")
    if status not in STATUSES or not math.isfinite(target) or not 0 <= reached <= target + 1e-7:
        raise ValueError(f"invalid status, horizon or reached time: {status}, {target}, {reached}")
    if status == "numerical_error":
        return
    state = json.loads(row["final_state_SI"])
    if len(state) != 4 or not all(map(math.isfinite, state)):
        raise ValueError("invalid terminal state")
    final_radius = math.hypot(*state[:2])
    if row.get("sampled_max_radius_m"):
        maximum = float(row["sampled_max_radius_m"])
        if not final_radius - 1e-8 <= maximum <= aperture + 1e-8:
            raise ValueError("invalid sampled maximum radius")
    if status == "retained":
        if not math.isclose(reached, target, rel_tol=0, abs_tol=1e-7) or row.get("escape_RF"):
            raise ValueError("retained row has an escape time or has not reached the horizon")
        if final_radius > aperture + 1e-8:
            raise ValueError("retained row ends outside the aperture")
    else:
        escaped = float(row["escape_RF"])
        if not 0 < escaped <= target + 1e-7 or not math.isclose(escaped, reached, rel_tol=0, abs_tol=1e-7):
            raise ValueError("invalid escape time")
        if not math.isclose(final_radius, aperture, rel_tol=0, abs_tol=1e-8):
            raise ValueError("escaped row does not end on the aperture")


def validate_grid(dataset):
    config = dataset.config
    grid = config["grid"]
    if grid < 3 or grid % 2 != 1:
        raise ValueError(f"Expected an odd grid of size at least three in {dataset.path}")
    half = (grid - 1) // 2
    step = .0075 / half
    phase = math.radians(config["phase_deg"])
    expected = {
        (model, round(i*step, 14), round(j*step, 14), 0., 0., round(phase, 14))
        for i in range(-half, half + 1) for j in range(-half, half + 1)
        if i*i + j*j < half*half for model in MODEL_NAMES
    }
    if expected != dataset.rows.keys():
        raise ValueError(f"Incorrect {grid}×{grid} XY sampling domain in {dataset.path}")


def validate_velocity(dataset):
    config = dataset.config
    grid = config["grid"]
    maximum = config["speed_max_m_s"]
    step = 2*maximum / (grid - 1)
    speeds = [round(-maximum + step*i, 14) for i in range(grid)]
    phase = round(math.radians(config["phase_deg"]), 14)
    expected = {(model, round(x, 14), round(y, 14), vx, vy, phase)
                for x, y in config["origins_m"] for vx in speeds for vy in speeds
                if (vx, vy) != (0, 0) or not config.get("zero_velocity_excluded") for model in MODEL_NAMES}
    if expected != dataset.rows.keys():
        raise ValueError(f"Incomplete velocity grid in {dataset.path}")


def settings_key(config):
    fields = ("protocol", "reference_radius_mm", "mesh_um", "horizon_RF", "frequency_Hz",
              "mass_kg", "charge_C", "aperture_m", "voltage_V", "field_sha256", "solver")
    return json.dumps({field: config.get(field) for field in fields}, sort_keys=True)


def compare_phases(xy, path):
    xy = sorted(xy, key=lambda dataset: dataset.config["phase_deg"])
    phases = [dataset.config["phase_deg"] for dataset in xy]
    if len(set(phases)) != len(phases) or len({settings_key(ds.config) for ds in xy}) != 1:
        raise ValueError("Phase maps must have distinct phases and identical non-phase settings")
    by_position = [{key[:-1]: row["status"] for key, row in ds.rows.items()} for ds in xy]
    if any(set(rows) != set(by_position[0]) for rows in by_position):
        raise ValueError("Phase maps have different physical launch states")
    phase_columns = [f"phase_{phase:g}" for phase in phases]
    transitions = []
    for model, x, y, vx, vy in sorted(by_position[0]):
        statuses = [rows[(model, x, y, vx, vy)] for rows in by_position]
        if len(set(statuses)) > 1:
            transitions.append({"model": model, "x0_m": x, "y0_m": y,
                                **dict(zip(phase_columns, statuses))})
    write_table(path, transitions, ("model", "x0_m", "y0_m", *phase_columns))
    return len(transitions)


def area_table(xy, out):
    areas = []
    for dataset in sorted(xy, key=lambda ds: (ds.config["protocol"], ds.config["phase_deg"], ds.config["grid"])):
        config = dataset.config
        grid = config["grid"]
        step = 15 / (grid - 1)
        for model in MODEL_NAMES:
            counts = Counter(row["status"] for key, row in dataset.rows.items() if key[0] == model)
            areas.append({"protocol": config["protocol"], "phase_deg": config["phase_deg"],
                          "reference_radius_mm": config.get("reference_radius_mm"), "model": model,
                          "grid": grid, "step_mm": step,
                          "sampled": counts.total(), **{status: counts[status] for status in STATUSES},
                          "retained_cell_area_mm2": counts["retained"] * step**2,
                          "upper_cell_area_mm2": (counts["retained"] + counts["numerical_error"]) * step**2})
    if areas:
        write_table(out / "area_resolution.csv", areas, tuple(areas[0]))
    return areas


def boundary_table(datasets, coarse, out):
    edges = []
    for dataset in datasets:
        config = dataset.config
        source = next(ds for ds in coarse if ds.config["phase_deg"] == config["phase_deg"])
        measured = source.rows | dataset.rows
        for row in read_csv(dataset.path / "edges.csv"):
            span = 1000 * math.hypot(float(row["x_left_m"]) - float(row["x_right_m"]),
                                     float(row["y_left_m"]) - float(row["y_right_m"]))
            if not math.isclose(span, float(row["width_mm"]), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Transition interval width differs from endpoints in {dataset.path}")
            if {row["status_left"], row["status_right"]} != {"retained", "escaped"}:
                raise ValueError(f"Transition interval requires observed opposite outcomes in {dataset.path}")
            for side in ("left", "right"):
                key = (row["model"], round(float(row[f"x_{side}_m"]), 14),
                       round(float(row[f"y_{side}_m"]), 14), 0., 0., round(math.radians(config["phase_deg"]), 14))
                if key not in measured or measured[key]["status"] != row[f"status_{side}"]:
                    raise ValueError(f"Transition interval endpoint lacks matching measurement in {dataset.path}: {key}")
            edges.append({"dataset": dataset.path.name, "phase_deg": config["phase_deg"],
                          **{key: row[key] for key in ("model", "x_left_m", "y_left_m", "x_right_m", "y_right_m",
                                                       "status_left", "status_right", "width_mm")}})
    columns = ("dataset", "phase_deg", "model", "x_left_m", "y_left_m", "x_right_m", "y_right_m",
               "status_left", "status_right", "width_mm")
    write_table(out / "boundary_transitions.csv", edges, columns)
    return len(edges)


def survival_table(datasets, out):
    survival = []
    for dataset in datasets:
        cohorts = defaultdict(list)
        for key, row in dataset.rows.items():
            initial = dataset.initial[key]
            group = (key[0], round(math.degrees(key[-1]), 9))
            cohorts[(*group, "all")].append(row)
            if initial.get("stratum") and initial["stratum"] != "all":
                cohorts[(*group, initial["stratum"])].append(row)
        horizons = dataset.config.get("known_horizons", [1000, 3000, 10000, 100000])
        for (model, phase, stratum), rows in sorted(cohorts.items()):
            for time in horizons:
                if time > min(horizon(row, dataset.config) for row in rows):
                    continue
                escaped = sum(row["status"] == "escaped" and float(row["escape_RF"]) <= time for row in rows)
                censored = sum(row["status"] == "numerical_error" and float(row["reached_RF"]) < time for row in rows)
                retained = len(rows) - escaped - censored
                survival.append({"dataset": dataset.path.name, "model": model, "phase_deg": phase,
                                 "stratum": stratum, "horizon_RF": time, "cohort": len(rows),
                                 "retained": retained, "escaped": escaped, "censored": censored,
                                 "fraction_lower": retained / len(rows),
                                 "fraction_upper": (retained + censored) / len(rows)})
    if survival:
        write_table(out / "survival.csv", survival, tuple(survival[0]))
    return survival


def fem_table(fem, primary, out):
    references = {}
    for dataset in primary:
        if dataset.config.get("mesh_um") != 50:
            continue
        for key, row in dataset.rows.items():
            if not row.get("voltage_V"):
                continue
            ref_key = (*key, float(row["voltage_V"]), horizon(row, dataset.config))
            old = references.get(ref_key)
            if old is not None:
                if old["status"] != row["status"]:
                    raise ValueError(f"Different reference outcomes for {ref_key}")
                if row["status"] == "escaped" and not math.isclose(float(old["escape_RF"]), float(row["escape_RF"]),
                                                                   rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f"Different reference escape times for {ref_key}")
            references[ref_key] = row
    comparisons = []
    for dataset in fem:
        for key, row in dataset.rows.items():
            model, x, y, vx, vy, phase, mesh, representation, target = key
            configured = dataset.config.get("voltage_V")
            if isinstance(configured, dict):
                configured = configured[model]
            voltage = float(row.get("voltage_V") or configured)
            ref_key = (model, x, y, vx, vy, phase, voltage, target)
            if ref_key not in references:
                raise ValueError(f"No matching h50 N48 reference for {ref_key} in {dataset.path}")
            reference = references[ref_key]
            escape_delta = None
            if row["status"] == reference["status"] == "escaped":
                escape_delta = float(row["escape_RF"]) - float(reference["escape_RF"])
            comparisons.append(dict(zip(("model", *STATE_FIELDS), key[:6]), dataset=dataset.path.name,
                                    voltage_V=voltage, mesh_um=mesh, representation=representation,
                                    horizon_RF=target, baseline_status=reference["status"],
                                    status=row["status"], baseline_escape_RF=reference.get("escape_RF"),
                                    escape_RF=row.get("escape_RF"), escape_delta_RF=escape_delta))
    if comparisons:
        write_table(out / "fem_comparison.csv", comparisons, tuple(comparisons[0]))
    return comparisons


def report_plots(xy, areas, survival, velocity, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"retained": "#32875e", "escaped": "#d44e4e", "numerical_error": "#777777"}
    legend = [Line2D([], [], marker="s", linestyle="none", color=color, label=status)
              for status, color in colors.items()]
    for dataset in xy:
        config = dataset.config
        if config["grid"] != 21 or dataset.stage not in {"phase90", "baseline", "matched"}:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for model, ax in zip(MODEL_NAMES, axes):
            x, y, samples = zip(*[(key[1]*1000, key[2]*1000, colors[row["status"]])
                                  for key, row in dataset.rows.items() if key[0] == model])
            ax.scatter(x, y, c=samples, marker="s", s=22)
            ax.set(title=f"{model}, {config['voltage_V'][model]:.5g} V", aspect="equal",
                   xlabel="x₀ (mm)", ylabel="y₀ (mm)", xlim=(-7.8, 7.8), ylim=(-7.8, 7.8))
        fig.legend(handles=legend, loc="outside lower center", ncol=3)
        fig.suptitle(f"{config['protocol']}, RF phase {config['phase_deg']:g}°, {config['horizon_RF']} RF periods")
        fig.savefig(out / f"xy_{dataset.path.name}.png", dpi=200)
        plt.close(fig)
    if areas:
        for protocol, radius in sorted({(row["protocol"], row["reference_radius_mm"]) for row in areas}):
            selected = [row for row in areas if (row["protocol"], row["reference_radius_mm"]) == (protocol, radius)]
            phases = sorted({row["phase_deg"] for row in selected})
            fig, axs = plt.subplots(1, len(phases), squeeze=False, figsize=(4.5*len(phases), 4), layout="constrained")
            for ax, phase in zip(axs.flat, phases):
                for model in MODEL_NAMES:
                    rows = sorted((row for row in selected if row["model"] == model and row["phase_deg"] == phase),
                                  key=lambda row: row["grid"])
                    ax.plot([row["grid"] for row in rows], [row["retained_cell_area_mm2"] for row in rows], "o-", label=model)
                ax.set(xlabel="Grid size", ylabel="Retained cell area (mm²)", title=f"RF phase {phase:g}°")
                ax.set_xticks(sorted({row["grid"] for row in selected}))
                ax.legend()
            label = protocol if radius is None else f"{protocol}_{radius:g}mm"
            fig.savefig(out / f"area_resolution_{label}.png", dpi=200)
            plt.close(fig)
    if survival:
        sets = sorted({row["dataset"] for row in survival})
        for name in sets:
            selected = [row for row in survival if row["dataset"] == name and row["stratum"] == "all"]
            phases = sorted({row["phase_deg"] for row in selected})
            fig, axs = plt.subplots(1, len(phases), squeeze=False, figsize=(4.5*len(phases), 4), layout="constrained")
            for ax, phase in zip(axs.flat, phases):
                for model in MODEL_NAMES:
                    rows = sorted((row for row in selected if row["model"] == model and row["phase_deg"] == phase),
                                  key=lambda row: row["horizon_RF"])
                    times = [row["horizon_RF"] for row in rows]
                    lower = [row["fraction_lower"] for row in rows]
                    upper = [row["fraction_upper"] for row in rows]
                    line, = ax.plot(times, lower, "o-", label=model)
                    ax.fill_between(times, lower, upper, alpha=.2, color=line.get_color())
                ax.set(xscale="log", xlabel="Horizon (RF periods)", ylabel="Selected cohort retained fraction",
                       ylim=(-.025, 1.025), title=f"RF phase {phase:g}°")
                ax.legend()
            fig.savefig(out / f"survival_{name}.png", dpi=200)
            plt.close(fig)
    for dataset in velocity:
        origins = sorted({key[1:3] for key in dataset.rows})
        phases = sorted({key[-1] for key in dataset.rows})
        for phase_number, phase in enumerate(phases):
            fig, axs = plt.subplots(len(origins), 3, squeeze=False, figsize=(11, 3.4*len(origins)), layout="constrained")
            for (x, y), axes in zip(origins, axs):
                for model, ax in zip(MODEL_NAMES, axes):
                    points = [(key[3], key[4], colors[row["status"]]) for key, row in dataset.rows.items()
                              if key[:3] == (model, x, y) and key[-1] == phase]
                    if points:
                        vx, vy, markers = zip(*points)
                        ax.scatter(vx, vy, c=markers, marker="s", s=18)
                    ax.set(xlabel="vₓ (m/s)", ylabel="vᵧ (m/s)", aspect="equal",
                           title=f"{model}, ({x*1000:g}, {y*1000:g}) mm")
                    ax.legend(handles=legend, fontsize=7, loc="best")
            fig.suptitle(f"RF phase {math.degrees(phase):g}°, horizon {dataset.config['horizon_RF']} RF")
            fig.savefig(out / f"velocity_{dataset.path.name}_{phase_number}.png", dpi=200)
            plt.close(fig)


def render_report(datasets, manifest, transitions, areas, boundaries, survival, comparisons):
    stages = {name: [] for name in ("baseline", *manifest.get("required_stages", STAGE_TITLES))}
    for dataset in datasets:
        stages.setdefault(dataset.stage, []).append(dataset)
    summary = []
    report = ["# V11 — ход исследования", "",
              "Удержание означает отсутствие зарегистрированного выхода за апертуру до конечного горизонта.", ""]
    for stage, inputs in stages.items():
        report.extend((f"## {STAGE_TITLES.get(stage, stage)}", ""))
        details = manifest.get("stage_details", {}).get(stage)
        if details:
            report.extend((details, ""))
        if not inputs:
            report.extend(("Ожидает расчёта.", ""))
            continue
        report.extend(("| Данные | Стартов | Удержано | Вышло | Численных ошибок |",
                       "|---|---:|---:|---:|---:|"))
        for dataset in inputs:
            counts = Counter(row["status"] for row in dataset.rows.values())
            location = dataset.path.relative_to(ROOT).as_posix()
            report.append(f"| `{location}` | {counts.total()} | {counts['retained']} | "
                          f"{counts['escaped']} | {counts['numerical_error']} |")
            summary.append({"stage": stage, "kind": dataset.kind, "directory": location,
                            "initial": counts.total(), **{status: counts[status] for status in STATUSES},
                            "allow_numerical_errors": dataset.allow_numerical_errors})
        report.append("")
        if stage == "phase90" and transitions is not None:
            report.extend((f"Смены статусов по фазам на общих стартах — {transitions}. См. `phase_transitions.csv`.", ""))
        if stage == "boundary_area":
            report.extend((f"Выявлено интервалов между двумя измеренными противоположными исходами — {boundaries}. "
                           "См. `boundary_transitions.csv`. В интервале возможны дополнительные смены статуса.", ""))
        if stage == "long" and survival:
            late = [row for row in survival if row["stratum"] == "all"]
            if late:
                report.extend(("Доли S(N) относятся к заранее зафиксированной выборке. "
                               "Они представлены по фазам, схемам и стратам в `survival.csv`.", ""))
                report.extend(("| Фаза, ° | Схема | Горизонт, RF | Удержано / выборка | Неопределённо |",
                               "|---:|---|---:|---:|---:|"))
                endpoint = max(row["horizon_RF"] for row in late)
                for row in late:
                    if row["horizon_RF"] == endpoint:
                        report.append(f"| {row['phase_deg']:g} | {row['model']} | {endpoint} | "
                                      f"{row['retained']} / {row['cohort']} | {row['censored']} |")
                report.append("")
        if stage == "fem" and comparisons:
            groups = defaultdict(list)
            for row in comparisons:
                groups[(row["mesh_um"], row["representation"])].append(row)
            report.extend(("Исходы сопоставлены с базовым h50 N48 для одинаковых стартов, фаз, напряжений и горизонтов.", "",
                           "| Вариант | Совпало | Разошлось | Неопределённо | Макс. сдвиг выхода, RF |",
                           "|---|---:|---:|---:|---:|"))
            for (mesh, representation), rows in sorted(groups.items()):
                valid = [row for row in rows if "numerical_error" not in (row["status"], row["baseline_status"])]
                agreed = sum(row["status"] == row["baseline_status"] for row in valid)
                deltas = [abs(row["escape_delta_RF"]) for row in valid if row["escape_delta_RF"] is not None]
                delta = f"{max(deltas):.6g}" if deltas else "—"
                report.append(f"| h{mesh} {representation} | {agreed} | {len(valid)-agreed} | "
                              f"{len(rows)-len(valid)} | {delta} |")
            report.extend(("", "Подробности сохранены в `fem_comparison.csv`.", ""))
    if areas:
        report.extend(("## Клеточные оценки площади", "",
                       "Оценка равна числу удержанных узлов внутри круга начальных состояний, умноженному на квадрат шага. "
                       "Краевые клетки включаются целиком. Ряд разрешений приведён в `area_resolution.csv`.", "",
                       "| Протокол | Фаза, ° | Сетка | A, мм² | B, мм² | C, мм² |",
                       "|---|---:|---:|---:|---:|---:|"))
        by_grid = defaultdict(dict)
        for row in areas:
            radius = row["reference_radius_mm"]
            title = row["protocol"] if radius is None else f"{row['protocol']} ({radius:g} мм)"
            value = f"{row['retained_cell_area_mm2']:.3f}"
            if row["numerical_error"]:
                value += f"–{row['upper_cell_area_mm2']:.3f}"
            by_grid[(title, row["phase_deg"], row["grid"])][row["model"]] = value
        for (protocol, phase, grid), counts in sorted(by_grid.items()):
            report.append(f"| {protocol} | {phase:g} | {grid}×{grid} | " + " | ".join(counts[m] for m in MODEL_NAMES) + " |")
        report.append("")
        differences = []
        resolved = {(row["protocol"], row["phase_deg"], row["model"], row["grid"]): row["retained_cell_area_mm2"]
                    for row in areas if row["protocol"] == "equal" and not row["numerical_error"]}
        for phase in sorted({key[1] for key in resolved}):
            for model in MODEL_NAMES:
                coarse, dense = (resolved.get(("equal", phase, model, grid)) for grid in (41, 61))
                if coarse is not None and dense is not None:
                    differences.append((phase, model, dense - coarse))
        if differences:
            report.extend(("Изменение клеточной оценки площади при переходе 41×41 → 61×61 приведено без экстраполяции.", "",
                           "| Фаза, ° | Схема | Изменение, мм² |", "|---:|---|---:|"))
            for phase, model, delta in differences:
                report.append(f"| {phase:g} | {model} | {delta:+.3f} |")
            report.append("")
    return "\n".join(report), summary


def validate_stage(stage, datasets, coarse):
    phases = {0., 45., 90.}
    inputs = [ds for ds in datasets if ds.stage == stage]
    if not inputs:
        raise ValueError(f"No complete datasets for required stage {stage}")
    if stage == "phase90" and {ds.config["phase_deg"] for ds in coarse} != phases:
        raise ValueError("phase90 requires phases 0°, 45° and 90°")
    if stage == "boundary_area":
        dense = [ds for ds in inputs if ds.kind == "xy"]
        grids = {(ds.config["phase_deg"], ds.config["grid"]) for ds in dense + coarse}
        boundaries = {ds.config["phase_deg"] for ds in inputs if ds.kind == "boundary"}
        if grids != {(phase, grid) for phase in phases for grid in (21, 41, 61)} or boundaries != phases:
            raise ValueError("boundary_area requires three phases, each with 21/41/61 grids and sampled brackets")
        if len({settings_key(ds.config) for ds in dense + coarse}) != 1:
            raise ValueError("Area convergence maps must have identical non-phase settings")
    if stage in {"velocity", "long", "matched"}:
        if {ds.config["phase_deg"] for ds in inputs} != phases:
            raise ValueError(f"{stage} requires phases 0°, 45° and 90°")
    if stage == "long":
        for ds in inputs:
            horizons = [1000, 3000, 10000, 100000]
            if ds.config["known_horizons"] != horizons or ds.config["horizon_RF"] != horizons[-1]:
                raise ValueError(f"Unexpected long-term horizons in {ds.path}")
            cases = defaultdict(set)
            for key, row in ds.initial.items():
                cases[key[0]].add((key[1:], row["stratum"]))
            if set(cases) != set(MODEL_NAMES) or len({frozenset(group) for group in cases.values()}) != 1:
                raise ValueError(f"Long-term cohorts differ between models in {ds.path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-stage", action="append", default=[])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.run_dir / "report_inputs.json")
    out = args.run_dir / "postprocessing"
    out.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(spec) for spec in manifest["maps"]]
    if len({(dataset.stage, dataset.path) for dataset in datasets}) != len(datasets):
        raise ValueError("A stage cannot declare the same dataset more than once")
    by_kind = defaultdict(list)
    for dataset in datasets:
        by_kind[dataset.kind].append(dataset)
    coarse = [dataset for dataset in by_kind["xy"] if dataset.stage in {"baseline", "phase90"}]
    if any(dataset.config["grid"] != 21 or dataset.config["protocol"] != "equal" for dataset in coarse):
        raise ValueError("phase90 inputs must be 21×21 equal-voltage XY maps")
    for stage in args.require_stage:
        validate_stage(stage, datasets, coarse)
    transitions = compare_phases(coarse, out / "phase_transitions.csv") if len(coarse) > 1 else None
    matched = [dataset for dataset in by_kind["xy"] if dataset.stage == "matched"]
    if len(matched) > 1:
        compare_phases(matched, out / "matched_phase_transitions.csv")
    areas = area_table(by_kind["xy"], out)
    boundaries = boundary_table(by_kind["boundary"], coarse, out) if by_kind["boundary"] else 0
    survival = survival_table(by_kind["long"], out)
    comparisons = fem_table(by_kind["fem"], by_kind["xy"] + by_kind["velocity"] + by_kind["boundary"], out)
    body, summary = render_report(datasets, manifest, transitions, areas, boundaries, survival, comparisons)
    atomic_text(out / "REPORT_RU.md", body)
    atomic_text(out / "validation.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if not args.no_plots:
        report_plots(by_kind["xy"], areas, survival, by_kind["velocity"], out)
    for stage in args.require_stage:
        records = [row for row in summary if row["stage"] == stage]
        failures = [row["directory"] for row in records if row["numerical_error"] and not row["allow_numerical_errors"]]
        if failures:
            parser.error(f"numerical errors in required stage {stage}: {', '.join(failures)}")
    print(out / "REPORT_RU.md")


if __name__ == "__main__":
    main()
