"""Audit the expanded DC-free map and draw point and contour variants."""
import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from run_stability_p4 import OUTPUT, ROOT, Launch, read_json, sha
from three_phase_trap.long_rf_run import write_json


DEFAULT = OUTPUT.with_name("extended_v2")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(directory, cells):
    protocol_path = directory / "protocol.json"
    protocol = read_json(protocol_path)
    summary = read_json(directory / "summary.json")
    for name, digest in protocol["reused_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Archived input changed: {name}")
    if len(cells) != summary["history"][-1]["cells"]:
        raise ValueError("Cell table and completion summary differ")
    horizon = protocol["horizons_RF"][-1]
    sources = {}
    for path in [*(ROOT / name for name in protocol["reused_sha256"]),
                 *(directory / "N48" / f"RF{horizon}").glob("*.json")]:
        row = read_json(path)
        key = Launch(row["coupling"], row["radius_mm"], row["start"])
        if key in sources or row["horizon_RF"] != horizon:
            raise ValueError(f"Duplicate or inconsistent final result: {path}")
        sources[key] = row["status"]
    for row in cells:
        statuses = Counter(sources[Launch(float(row["coupling"]), float(row["radius_mm"]), i)] for i in range(6))
        for column, status in (("held", "held"), ("escaped", "escaped"), ("unresolved", "numerical_error")):
            if int(row[column]) != statuses[status]:
                raise ValueError(f"Cell table differs from individual trajectories: {row}")
    if len(sources) != 6 * len(cells):
        raise ValueError("Unexpected number of trajectories")
    return protocol, summary


def render(directory, cells, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    q = np.array([float(row["q_M_ref"]) for row in cells])
    r = np.array([float(row["radius_mm"]) for row in cells])
    held = np.array([int(row["held"]) / 6 for row in cells])
    resolved = np.array([int(row["unresolved"]) == 0 for row in cells])
    logq = np.log10(q)
    mesh = tri.Triangulation(logq, r)
    vertices = mesh.triangles
    # Avoid filling poorly supported spans of the adaptive scan.
    mesh.set_mask((np.ptp(logq[vertices], axis=1) > .53) | (np.ptp(r[vertices], axis=1) > .66)
                  | (~resolved[vertices]).any(axis=1))
    interpolator = tri.LinearTriInterpolator(mesh, held)
    x = np.linspace(logq.min(), logq.max(), 660)
    y = np.linspace(r.min(), r.max(), 460)
    xx, yy = np.meshgrid(x, y)
    z = interpolator(xx, yy)
    horizon = protocol["horizons_RF"][-1]
    aperture_mm = 1000 * protocol["aperture_m"]
    for contour in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.5), layout="constrained")
        for index, ax in enumerate(axes):
            if contour:
                ax.contourf(10**x, y, z, levels=(-1e-6, .5, 5/6, 1.000001),
                            colors=("#f0edf3", "#d9efea", "#91cbbf"), corner_mask=False)
                lines = ax.contour(10**x, y, z, levels=(.5, 5/6), colors=("#384552", "#1d7266"),
                                   linestyles=("solid", "dashed"), linewidths=(1.35, 1.1))
                if index:
                    ax.clabel(lines, fmt={.5: "50%", 5/6: "5/6"}, fontsize=8)
                ax.scatter(q[resolved], r[resolved], s=6, color="#304a55", alpha=.5, zorder=3)
            else:
                artist = ax.scatter(q[resolved], r[resolved], c=held[resolved], s=22, cmap="viridis",
                                    vmin=0, vmax=1, linewidths=.28, edgecolors="#334753", zorder=3)
            if np.any(~resolved):
                ax.scatter(q[~resolved], r[~resolved], s=22, marker="x", c="#a43e42", zorder=5)
            ax.axhline(aperture_mm, color="#a94046", ls=":", lw=1.15, label="Апертура 8 мм")
            ax.set(xscale="log", ylabel="Стартовый радиус, мм",
                   xlabel=r"$|q_{\mathrm{M}}^{\mathrm{ref}}|$ при $a=0$",
                   ylim=(5 if index else 0, aperture_mm + .12))
            if index:
                visible = q[r >= 5]
                ax.set_xlim(visible.min() / 1.25, visible.max() * 1.25)
            ax.grid(alpha=.17)
            ax.legend(loc="lower left" if index else "upper right", fontsize=8)
        axes[0].set_title("Полная сетка")
        axes[1].set_title("Увеличение у апертуры")
        if not contour:
            fig.colorbar(artist, ax=axes, label="Доля удержанных")
        kind = "Интерполированные контуры 50% и 5/6" if contour else "Рассчитанные точки"
        fig.suptitle(f"{kind} · N48 · {horizon:,} RF-периодов".replace(",", " "))
        name = "capture_contours" if contour else "capture_points"
        fig.savefig(directory / (name + ".png"), dpi=235)
        fig.savefig(directory / (name + ".svg"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    cells = read_rows(args.directory / "extended_cells.csv")
    protocol, summary = audit(args.directory, cells)
    brackets = read_rows(args.directory / "radial_boundary_brackets.csv")
    transitions = read_rows(args.directory / "q_boundary_brackets.csv")
    result = dict(cells=len(cells), radii=len({r["radius_mm"] for r in cells}),
                  full=sum(int(r["held"]) == 6 for r in cells), zero=sum(int(r["escaped"]) == 6 for r in cells),
                  mixed=sum(0 < int(r["held"]) < 6 and not int(r["unresolved"]) for r in cells),
                  trajectories=summary["history"][-1]["trajectories"], q_reentries=sum(r["direction"] == "reentry" for r in transitions),
                  radial_half_brackets=[r for r in brackets if float(r["level"]) == .5],
                  uppermost_cells=[r for r in cells if float(r["radius_mm"]) >= 7.9])
    write_json(args.directory / "capture_analysis.json", result)
    render(args.directory, cells, protocol)
    print(f"Verified {result['trajectories']} trajectories in {result['cells']} cells", flush=True)


if __name__ == "__main__":
    main()
