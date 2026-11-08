"""Plot published P4 transfer results from the saved analysis tables and states."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/effective_reconstruction/p4_transfer/v1"
COLORS = {"A": "#1d3c58", "B": "#ac5f20", "E": "#007e70", "h50": "#75788a"}


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    launches = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))["launches"]
    cases = [row["case"] for row in summary["cases"]]
    analysis = directory / "analysis"
    diagnostics = {(row["case"], row["filter"]): row for row in read_rows(analysis / "slow_diagnostics.csv")}
    comparisons = {(row["case"], row["model"]): row for row in read_rows(analysis / "comparisons.csv")}
    controls = {(row["case"], row["control"]): row for row in read_rows(analysis / "numerical_controls.csv")}
    x = np.arange(len(cases))
    labels = [f"{launches[c]['radius_mm']:.2f}".replace(".", ",") + f" · {launches[c]['sequence']}" for c in cases]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .18,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 5.65), layout="constrained")
    ax = axes[0, 0]
    for mesh, marker, offset, color in ((35, "o", -.13, COLORS["A"]), (50, "x", 0, COLORS["h50"])):
        rates = [float(diagnostics[(f"{case}_h{mesh}_fine", "mean")]["fitted_rate_deg_per_s"]) for case in cases]
        ax.scatter(x + offset, rates, marker=marker, color=color, s=33, label=f"P4 h{mesh}")
    ax.scatter(x + .13, [row["predicted_rate_deg_s"] for row in summary["cases"]], marker="D", facecolors="none",
               edgecolors=COLORS["E"], s=34, label="E")
    ax.axhline(0, color="#aaa", lw=.8)
    ax.set(title="а  Скорость поворота", xticks=x, xticklabels=labels, ylabel="Скорость, °/с")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[0, 1]
    for model, offset in (("B", -.17), ("E", .17)):
        values = [float(comparisons[(case, model)]["rms_pct"]) for case in cases]
        ax.bar(x + offset, values, width=.28, color=COLORS[model], label=model)
    ax.scatter(x, [float(controls[(case, "space_fine")]["rms_pct"]) for case in cases], marker="_", s=100,
               color=COLORS["h50"], label="h35–h50")
    ax.axhline(1, color="#666", lw=.9, ls="--")
    ax.set(xticks=x, xticklabels=labels, ylabel="Векторная RMS, % от 1 мм", yscale="log", ylim=(7e-5, 3))
    ax.set_title("б  Ошибка траектории", pad=23)
    ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1), frameon=False)

    radius = max(launch["radius_mm"] for launch in launches.values())
    selected = [case for case in cases if launches[case]["radius_mm"] == radius]
    for ax, case, panel in zip(axes[1], selected, ("в", "г"), strict=True):
        launch = launches[case]
        angle = launch["azimuth_rad"]
        transverse = np.array([-np.sin(angle), np.cos(angle)])
        with np.load(analysis / f"{case}_comparison.npz") as data:
            time_ms = data["time_s"] * 1e3
            for name, key, style in (("A", "RF_mean_m", "-"), ("B", "B", ":"), ("E", "E", "--")):
                xy = data[key][:, :2]
                ax.plot(time_ms, (xy @ transverse) * 1e6, style, lw=1.25, color=COLORS[name], label=name)
        radius_label = f"{launch['radius_mm']:.2f}".replace(".", ",")
        ax.set(title=f"{panel}  {radius_label} мм · {launch['sequence']}", xlabel="Время, мс",
               ylabel="Поперечная координата, мкм")
    axes[1, 0].legend(ncol=3, fontsize=8, loc="upper left")
    output = directory / "report"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "p4_transfer.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT)
    plot(parser.parse_args().directory)
