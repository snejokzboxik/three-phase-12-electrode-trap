"""Draw the classical Mathieu stability wedge alongside the computed RF map."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import mathieu_a, mathieu_b


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/stability_p4/mathieu_comparison_v1"
INK = "#263643"
LIMIT = "#a04742"


def mathieu_panel(ax, edge, apex):
    q = np.linspace(0, 1.03, 700)
    lower = -mathieu_a(0, q)
    upper = mathieu_b(1, q)
    band_q = np.linspace(0, edge, 550)
    ceiling = np.minimum(-mathieu_a(0, band_q), mathieu_b(1, band_q))
    apex_a = float(mathieu_b(1, apex))
    ax.fill_between(band_q, 0, np.maximum(ceiling, 0), color="#c9e6e0", zorder=1)
    for values, selected in ((lower, q <= apex), (upper, q >= apex)):
        ax.plot(q, np.where(selected, values, np.nan), color=INK, lw=1.7)
        ax.plot(q, np.where(~selected, values, np.nan), color=INK, lw=1, ls=(0, (4, 3)))
    ax.plot(q, .30*q, color="#79736d", lw=1, ls=(0, (3, 3)))
    ax.text(.205, .086, r"пример $a/q=\mathrm{const}$", fontsize=8.5, color="#635d57", rotation=20)
    ax.hlines(apex_a, 0, apex, color="#76808b", lw=.8, ls=":")
    ax.vlines(apex, 0, apex_a, color="#76808b", lw=.8, ls=":")
    ax.plot(apex, apex_a, "o", ms=4, color=INK)
    ax.annotate(f"вершина\n({apex:.3f}; {apex_a:.3f})".replace(".", ","), (apex, apex_a),
                xytext=(.49, .300), ha="center", va="center", fontsize=8.7,
                arrowprops=dict(arrowstyle="-", color=INK, lw=.75))
    ax.text(.49, .044, "устойчивы\nобе оси", ha="center", color="#185d53", fontsize=10)
    ax.text(.335, .205, "устойчива\nтолько ось $x$", ha="center", color=INK, fontsize=9)
    ax.text(.864, .184, "устойчива\nтолько ось $y$", ha="center", color=INK, fontsize=9)
    ax.plot(edge, 0, "o", ms=4.5, color=LIMIT, clip_on=False, zorder=5)
    ax.annotate("0,908", (edge, 0), xytext=(.98, .052), color=LIMIT, ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=LIMIT, lw=.75))
    ax.set(xlim=(0, 1.035), ylim=(0, .338), xlabel=r"$|q_{\mathrm{M}}|$", ylabel=r"$a\geq 0$",
           xticks=np.arange(0, 1.01, .2), yticks=(0., .1, .2, .3))
    ax.set_title("а  Идеальный квадруполь · первая область", loc="left", pad=11)


def capture_panel(fig, ax, cells, edge):
    q = np.array([float(row["q_M_ref"]) for row in cells])
    radius = np.array([float(row["radius_mm"]) for row in cells])
    starts = np.array([int(row["starts"]) for row in cells])
    octupole = np.array([int(row["N48_held"]) for row in cells])
    quadrupole = np.array([int(row["quadrupole_held"]) for row in cells])
    coarse = np.array([row["coarse"] == "True" for row in cells])
    for mask, marker in ((coarse, "s"), (~coarse, "D")):
        image = ax.scatter(q[mask], radius[mask], c=octupole[mask] / starts[mask], marker=marker,
                           s=50, cmap="viridis", vmin=0, vmax=1, linewidths=.45, edgecolors=INK, zorder=3)
    full = quadrupole == starts
    ax.scatter(q[full], radius[full], s=96, facecolors="none", edgecolors=INK, linewidths=.8, zorder=4,
               label="квадруполь: 6 из 6")
    ax.axvline(edge, color=LIMIT, lw=1.05, ls=(0, (4, 3)), zorder=2)
    ax.text(edge * 1.20, 1.57, "0,908 · граница\nквадруполя", color=LIMIT, fontsize=9)
    ax.legend(loc="center right", bbox_to_anchor=(1, .85), frameon=False, fontsize=8.5)
    ax.grid(alpha=.22, linewidth=.65)
    ax.set(xscale="log", ylim=(.40, 2.12), yticks=sorted(set(radius)),
           xlabel=r"$|q_{\mathrm{M}}^{\mathrm{ref}}|$ при $a=0$", ylabel=r"$r_0$, мм")
    ax.set_title("б  Удержание до 10 000 RF-периодов", loc="left", pad=11)
    bar = fig.colorbar(image, ax=ax, pad=.025, ticks=(0, 1/6, .5, 1), label="12 электродов: доля удержанных")
    bar.set_ticklabels(("0", "1/6", "1/2", "1"))


def interpolate_capture(q, radius, fraction, q_eval, radius_eval):
    """Interpolate within the measured domain in log(q), then in launch radius."""
    launch_radii = np.unique(radius)
    along_rows = []
    for launch_radius in launch_radii:
        selected = np.flatnonzero(radius == launch_radius)
        selected = selected[np.argsort(q[selected])]
        curve = PchipInterpolator(np.log(q[selected]), fraction[selected], extrapolate=False)
        along_rows.append(curve(np.log(q_eval)))
    return PchipInterpolator(launch_radii, np.asarray(along_rows), axis=0, extrapolate=False)(radius_eval)


def interpolated_capture_panel(ax, cells, edge):
    q = np.array([float(row["q_M_ref"]) for row in cells])
    radius = np.array([float(row["radius_mm"]) for row in cells])
    fraction = np.array([int(row["N48_held"]) / int(row["starts"]) for row in cells])
    q_grid = np.geomspace(q.min(), q.max(), 650)
    r_grid = np.linspace(radius.min(), radius.max(), 360)
    values = interpolate_capture(q, radius, fraction, q_grid, r_grid)
    ax.contourf(q_grid, r_grid, values, levels=(.5, 5/6, 1), colors=("#e6f3ef", "#c9e6e0"))
    contours = ax.contour(q_grid, r_grid, values, levels=(.5, 5/6), colors=(INK, "#30776c"),
                          linewidths=(1.65, 1.15), linestyles=("solid", "dashed"))
    ax.clabel(contours, fmt={.5: "50%", 5/6: "≈83%"}, inline=True, fontsize=9,
              manual=((2.7, 1.44), (5.5, .86)))
    ax.scatter(q, radius, s=15, c=fraction, cmap="Greys", vmin=0, vmax=1, linewidths=.55, edgecolors=INK,
               label="рассчитанные точки", zorder=4)
    ax.axvline(edge, color=LIMIT, lw=1.05, ls=(0, (4, 3)), zorder=3, label="граница Матьё: 0,908")
    ax.legend(loc="center right", bbox_to_anchor=(1, .85), frameon=False, fontsize=8.5)
    ax.text(.11, 1.61, "область\nудержания", ha="center", color="#185d53", fontsize=10)
    ax.set(xscale="log", xlim=(q.min() / 1.25, q.max() * 1.2), ylim=(.41, 2.11), yticks=np.unique(radius),
           xlabel=r"$|q_{\mathrm{M}}^{\mathrm{ref}}|$ при $a=0$", ylabel=r"$r_0$, мм")
    ax.set_title("б  12 электродов · интерполяция удержания", loc="left", pad=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=DEFAULT)
    parser.add_argument("--interpolated", action="store_true", help="Draw estimated capture contours")
    args = parser.parse_args()
    with (args.directory / "matched_cells.csv").open(encoding="utf-8", newline="") as stream:
        cells = list(csv.DictReader(stream))
    edge = brentq(lambda q: mathieu_b(1, q), .8, 1.)
    apex = brentq(lambda q: mathieu_b(1, q) + mathieu_a(0, q), .6, .8)
    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "stix", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.6), layout="constrained", width_ratios=(.95, 1.20))
    mathieu_panel(axes[0], edge, apex)
    if args.interpolated:
        interpolated_capture_panel(axes[1], cells, edge)
        name = "mathieu_style_interpolated"
    else:
        capture_panel(fig, axes[1], cells, edge)
        name = "mathieu_style_diagram"
    for suffix in ("png", "svg"):
        fig.savefig(args.directory / f"{name}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)
    print(f"q_apex={apex:.9f}  a_apex={mathieu_b(1, apex):.9f}  q_edge={edge:.9f}")


if __name__ == "__main__":
    main()
