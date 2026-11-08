"""Build figures from the v1 tables and saved trajectories."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


STYLE = {
    "A": ("#153653", "-"),
    "B": ("#a6611a", "--"),
    "C": ("#728392", ":"),
    "D": ("#b24873", "-."),
    "E": ("#007f71", (0, (4, 2))),
}


def rows(out, filename):
    with (out / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def line(ax, x, y, label):
    color, style = STYLE[label]
    ax.plot(x, y, color=color, linestyle=style, label=label)


def save(fig, path):
    fig.savefig(path, dpi=230, facecolor="white")
    plt.close(fig)


def potential(out):
    synthetic = json.loads((out / "synthetic_fit.json").read_text(encoding="utf-8"))
    rf = json.loads((out / "frozen_models.json").read_text(encoding="utf-8"))
    reference = synthetic["known"]
    radius = np.linspace(0., 1.25, 350)
    energy = lambda model: model["U_at_1mm_J"] * radius**model["p"]
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.8), layout="constrained")
    ax = axes[0]
    line(ax, radius, 1e18 * energy(reference), "C")
    line(ax, radius, 1e18 * energy(rf["D"]), "D")
    ax.plot(radius, 1e18 * energy(synthetic["fit"]), "k--", lw=1, label="Fit on C")
    ax.set(xlabel="Radius [mm]", ylabel="Potential energy [aJ]")
    ax.legend()
    ax = axes[1]
    delta = (energy(rf["D"]) - energy(reference)) * 1e24
    line(ax, radius, delta, "D")
    ax.plot(radius, (energy(synthetic["fit"]) - energy(reference)) * 1e24, "k--", lw=1, label="Fit on C")
    ax.set(xlabel="Radius [mm]", ylabel="Difference from C [yJ]")
    ax.legend()
    save(fig, out / "potential.png")


def trajectories(out):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.1), layout="constrained")
    for index, (tag, title) in enumerate((("p4_phase0", "P4 phase 0 / calibration"),
                                          ("p4_phase1", "P4 phase π/2 / holdout"))):
        with np.load(out / f"{tag}_comparison.npz") as data:
            t, xy = 1e3 * data["time_s"], data["RF_mean_m"]
            ax = axes[index, 0]
            line(ax, t, xy[:, 1] * 1e6, "A")
            for name in ("B", "D", "E"):
                line(ax, t, data[name][:, 1] * 1e6, name)
            ax.set(title=title, xlabel="Time [ms]", ylabel="Transverse position [μm]")
            ax = axes[index, 1]
            line(ax, xy[:, 0] * 1e3, xy[:, 1] * 1e6, "A")
            for name in ("B", "D", "E"):
                line(ax, data[name][:, 0] * 1e3, data[name][:, 1] * 1e6, name)
            ax.set(xlabel="Longitudinal position [mm]", ylabel="Transverse position [μm]")
    axes[0, 0].legend(ncol=4, loc="upper left")
    save(fig, out / "trajectories.png")


def precession(out):
    diagnostics = rows(out, "slow_diagnostics.csv")
    comparisons = rows(out, "comparisons.csv")
    tag = ("oct075", "oct100", "oct125", "oct125_reverse")
    x = np.arange(len(tag))
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.05), layout="constrained")
    ax = axes[0]
    for offset, filt, marker in ((-.12, "mean", "o"), (0., "strobe", "+"), (.12, "inverse", "x")):
        rate = [float(next(r for r in diagnostics if r["case"] == name and r["filter"] == filt)
                      ["fitted_rate_deg_per_s"]) for name in tag]
        ax.plot(x + offset, rate, marker=marker, linestyle="none", label=filt)
    predicted = [float(next(r for r in comparisons if r["case"] == name and r["model"] == "E")
                       ["predicted_orientation_rate_deg_s"]) for name in tag]
    ax.plot(x, predicted, ls="none", marker="D", markerfacecolor="none", color=STYLE["E"][0], label="E prediction")
    ax.axhline(0., lw=.8, color="#888888")
    ax.set(xticks=x, xticklabels=["0.75", "1.00", "1.25", "1.25 rev."], xlabel="Launch radius [mm]",
           ylabel="Orientation rate [°/s]")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax = axes[1]
    for offset, model in ((-.25, "C"), (0., "D"), (.25, "E")):
        values = [float(next(r for r in comparisons if r["case"] == name and r["model"] == model)["rms_pct"]) for name in tag]
        ax.bar(x + offset, values, width=.24, color=STYLE[model][0], label=model)
    ax.axhline(1., lw=.8, color="#888888", ls="--")
    ax.set(xticks=x, xticklabels=["0.75", "1.00", "1.25", "1.25 rev."], xlabel="Launch radius [mm]",
           ylabel="Vector RMS [% of 1 mm]", yscale="log")
    ax.legend(ncol=3)
    save(fig, out / "analytic_control.png")


def spectrum(out):
    """Project onto harmonics of one measured cycle, for waveform diagnostics."""
    result = []
    for path in sorted(out.glob("*_comparison.npz")):
        case = path.stem.removesuffix("_comparison")
        with np.load(out / f"{case}_mean.npz") as ref:
            turns = ref["turning_points"]
        begin, end = turns[0, 0], turns[2, 0]
        axis = turns[0, 1:3] / turns[0, 3]
        clock = np.linspace(begin, end, 4096, endpoint=False)
        with np.load(path) as data:
            for key in sorted(set(data.files) - {"time_s"}):
                x = CubicSpline(data["time_s"], data[key][:, :2] @ axis)(clock)
                harmonics = abs(np.fft.rfft(x)) * (2 / len(x))
                result.append(dict(case=case, model="A" if key == "RF_mean_m" else key,
                                   cycle_begin_s=begin, cycle_end_s=end, frequency_Hz=1 / (end - begin),
                                   X1_m=harmonics[1], X3_over_X1=harmonics[3] / harmonics[1],
                                   X5_over_X1=harmonics[5] / harmonics[1]))
    with (out / "waveform_spectrum.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .18, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.25})
    potential(args.out)
    trajectories(args.out)
    precession(args.out)
    spectrum(args.out)


if __name__ == "__main__":
    main()
