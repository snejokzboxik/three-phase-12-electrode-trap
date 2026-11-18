"""Plot the uniform-orientation startup scans and long-horizon controls."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
RES, FIG = OUT / "results", OUT / "figures"


def read(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def main():
    coarse = read("startup_capture_results.json")["frequency_sweep"]
    refined = read("startup_capture_refinement.json")["rows"]
    long = read("startup_capture_long_horizon.json")["rows"]
    selected = {round(r["omega"], 10): r for r in coarse}
    selected.update({round(r["omega"], 10): r for r in refined})
    selected.update({round(r["omega"], 10): r for r in long})
    rows = sorted(selected.values(), key=lambda r: r["omega"])
    markers = {1500: "o", 4000: "s", 10000: "D"}
    colors = {1500: "#2874a6", 4000: "#af7ac5", 10000: "#c0392b"}

    fig, (ax, zoom) = plt.subplots(2, 1, figsize=(8.0, 7.2), gridspec_kw={"height_ratios": [1, 1.05]})
    for target, xlim in ((ax, (0.3, 4.15)), (zoom, (0.38, 0.56))):
        target.axvspan(0.3887674821559735, 4.05, color="#d5f5e3", alpha=0.65,
                       label="аналитическая joint linear stability band")
        for horizon in sorted(markers):
            subset = [r for r in rows if r["rf_periods"] == horizon and xlim[0] <= r["omega"] <= xlim[1]]
            if not subset:
                continue
            target.scatter([r["omega"] for r in subset],
                           [r["orientation_grid_capture_fraction"] for r in subset],
                           marker=markers[horizon], color=colors[horizon], s=38,
                           label=f"{horizon} RF; uniform grid")
        target.set_xlim(*xlim)
        target.set_ylim(-0.035, 1.04)
        target.set_ylabel("captured grid fraction")
        target.grid(alpha=0.25)
    ax.set_title("RO3 finite neutral dipole: spin-up from zero initial spin")
    zoom.set_xlabel("dimensionless drive frequency Ω")
    ax.legend(fontsize=8, loc="upper right")
    zoom.legend(fontsize=8, loc="upper right")
    fig.text(0.01, 0.005,
             "Zero means no capture on the sampled phase grid, not proof of an empty basin. Fractions are deterministic grid quadrature, not probabilities.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "startup_capture_frequency.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
