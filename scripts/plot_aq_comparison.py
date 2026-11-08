"""Publication plots of the calculated (a,q) maps, with a small-DC zoom."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import PchipInterpolator

from run_stability_aq import DEFAULT, first_band


def interpolation(grid, a, q, x, y):
    missing = ~np.isfinite(grid)
    # Supply a value only for constructing the interpolant. The neighboring
    # cells are masked below, so the replacement is never shown as a result.
    filled = grid.copy()
    for i in range(len(q)):
        good = ~missing[:, i]
        if not good.any():
            raise ValueError("A full RF column is unresolved")
        filled[~good, i] = np.interp(a[~good], a[good], filled[good, i])
    z = PchipInterpolator(a, PchipInterpolator(q, filled, axis=1)(x), axis=0)(y)
    ai = np.clip(np.searchsorted(a, y) - 1, 0, len(a) - 2)
    qi = np.clip(np.searchsorted(q, x) - 1, 0, len(q) - 2)
    corners = (missing[:-1, :-1] | missing[1:, :-1] | missing[:-1, 1:] | missing[1:, 1:])
    return np.ma.array(z, mask=corners[np.ix_(ai, qi)])


def draw(directory, data, a, q, horizon, filled):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey="row", layout="constrained")
    qq, aa = np.meshgrid(q, a)
    x = np.linspace(q[0], q[-1], 500)
    y = np.unique(np.r_[np.linspace(a[0], .04, 160), np.linspace(.04, a[-1], 360)])
    for col, (name, grid) in enumerate(data.items()):
        missing = ~np.isfinite(grid)
        z = interpolation(grid, a, q, x, y) if filled else None
        for row in range(2):
            ax = axes[row, col]
            if filled:
                ax.contourf(x, y, z, levels=(.5, 5/6, 1.000001), colors=("#dcf0eb", "#9acdc2"), corner_mask=False)
                ax.contour(x, y, z, levels=(.5, 5/6), colors=("#293d4a", "#20766b"),
                           linewidths=(1.2, 1.1), linestyles=("-", "--"), corner_mask=False)
                ax.scatter(qq[~missing], aa[~missing], s=9, color="#344e5e", alpha=.5, zorder=4)
            else:
                scatter = ax.scatter(qq, aa, c=grid, s=29 if row == 0 else 43, cmap="viridis",
                                     vmin=0, vmax=1, linewidths=.35, edgecolors="#455261", zorder=4)
            ax.scatter(qq[missing], aa[missing], c="#a94046", marker="x", s=45, zorder=5)
            first_band(ax)
            ax.set(xlim=(0, q[-1] * 1.035), ylim=((-0.008, a[-1] + .012) if row == 0 else (-.001, .041)),
                   ylabel=r"$a$" if col == 0 else None)
            ax.grid(alpha=.17)
            if row == 0:
                ax.set_title(name)
            else:
                ax.set_xlabel(r"$|q_{\mathrm{M}}^{\mathrm{ref}}|$")
    keys = [Line2D([], [], color="#a94046", ls="--", label="Первая область Матьё"),
            Line2D([], [], color="#a94046", ls="", marker="x", label="Чувствительно к шагу")]
    if filled:
        keys += [Line2D([], [], color="#293d4a", label="50%"), Line2D([], [], color="#20766b", ls="--", label="5/6")]
    axes[0, 0].legend(handles=keys, fontsize=8, loc="upper center", bbox_to_anchor=(.5, .92))
    if not filled:
        fig.colorbar(scatter, ax=axes, label="Доля удержанных стартов", shrink=.86)
    title = "Интерполированные контуры" if filled else "Рассчитанные точки"
    fig.suptitle(f"{title} · старт 1 мм · {horizon:,} RF-периодов".replace(",", " "))
    prefix = "aq_contours_article" if filled else "aq_points_article"
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"{prefix}.{suffix}", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=DEFAULT / "RF1000")
    args = parser.parse_args()
    with (args.directory / "capture_cells.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    a = np.array(sorted({float(r["a"]) for r in rows}))
    q = np.array(sorted({float(r["q_M_ref"]) for r in rows}))
    keyed = {(r["model"], float(r["a"]), float(r["q_M_ref"])): r for r in rows}

    def fraction(row):
        return np.nan if int(row["unresolved"]) or row["step_check_disagreement"] == "True" else int(row["held"]) / 6

    grids = {label: np.array([[fraction(keyed[model, v, u]) for u in q] for v in a])
             for model, label in (("N48", "12 электродов · N48 + DC"), ("quadrupole", "Идеальный квадруполь + DC"))}
    horizon = int(args.directory.name.removeprefix("RF"))
    for filled in (False, True):
        draw(args.directory, grids, a, q, horizon, filled)


if __name__ == "__main__":
    main()
