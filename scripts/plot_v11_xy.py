"""Render measured XY classes only; never interpolate unknown binary states."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11"


def save(fig, root):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(root.with_suffix("."+ext), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    folder = a.directory.resolve()
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    with (folder / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    n = config["grid"]
    axis = np.linspace(-7.5, 7.5, n)
    step = axis[1]-axis[0]
    arrays = {model: np.full((n, n), -1, int) for model in "ABC"}
    times = {model: np.full((n, n), np.nan) for model in "ABC"}
    radii = {model: np.full((n, n), np.nan) for model in "ABC"}
    for row in rows:
        ix = int(round((float(row["x0_m"])*1000+7.5)/step))
        iy = int(round((float(row["y0_m"])*1000+7.5)/step))
        model = row["model"]
        arrays[model][iy, ix] = {"retained": 1, "escaped": 0}.get(row["status"], -1)
        if row.get("escape_RF"):
            times[model][iy, ix] = float(row["escape_RF"])
        if row.get("sampled_max_radius_m"):
            radii[model][iy, ix] = 1000*float(row["sampled_max_radius_m"])
    title = (f"φ={config['phase_deg']:g}°, {config['protocol']}, {config['horizon_RF']} RF; "
             f"h{config['mesh_um']}, Δ={step:.3g} mm, 100000 Th, +e, 100 kHz, rₐ=8 mm")
    cmap = ListedColormap(["#c0c0c0", "#d44e4e", "#32875e"])
    norm = BoundaryNorm([-1.5, -.5, .5, 1.5], cmap.N)
    figs = folder / "figures"
    figs.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, model in zip(axs, "ABC"):
        ax.imshow(arrays[model], origin="lower", extent=(-7.5-step/2, 7.5+step/2)*2,
                  interpolation="none", cmap=cmap, norm=norm)
        ax.set(title=f"{model}: {config['voltage_V'][model]:.4g} V", xlabel="x₀ (mm)", ylabel="y₀ (mm)", aspect="equal")
    fig.suptitle(title)
    save(fig, figs / "xy_binary_three")
    # Exclusive classes are assigned only where all three measured statuses exist.
    exclusive = np.full((n, n), -1, int)
    valid = np.logical_and.reduce([arrays[m] >= 0 for m in "ABC"])
    bits = arrays["A"] + 2*arrays["B"] + 4*arrays["C"]
    exclusive[valid] = bits[valid]
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    colors = ["#b7b7b7", "#dddddd", "#d55e59", "#4c78a8", "#ba80b9", "#38a66e", "#edae49", "#765f9e", "#222222"]
    ax.imshow(exclusive+1, origin="lower", extent=(-7.5-step/2, 7.5+step/2)*2,
              interpolation="none", cmap=ListedColormap(colors), vmin=0, vmax=8)
    ax.set(xlabel="x₀ (mm)", ylabel="y₀ (mm)", title="A/B/C bit mask: 1/2/4; gray=unknown")
    fig.suptitle(title)
    save(fig, figs / "xy_comparison_bits")
    for label, group, limits in (("escape_RF", times, (0, config["horizon_RF"])),
                                  ("max_radius_mm", radii, (0, 8))):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
        for ax, model in zip(axs, "ABC"):
            image = ax.imshow(group[model], origin="lower", extent=(-7.5-step/2, 7.5+step/2)*2,
                              interpolation="none", vmin=limits[0], vmax=limits[1])
            ax.set(title=f"{model}: {config['voltage_V'][model]:.4g} V", xlabel="x₀ (mm)", ylabel="y₀ (mm)", aspect="equal")
            fig.colorbar(image, ax=ax)
        fig.suptitle(title)
        save(fig, figs / ("xy_"+label))
    print(f"Rendered {len(rows)} measured rows to {figs}")


if __name__ == "__main__":
    main()
