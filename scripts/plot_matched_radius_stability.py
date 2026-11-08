"""Evaluate the ideal quadrupole on the expanded N48 launch grid at a=0."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import matplotlib.tri as tri
import numpy as np

from compare_mathieu_stability import (
    FIRST_EDGE, PHASE_GRID, fundamental, initial_state, position_bound, quadrupole,
)
from plot_extended_stability import DEFAULT as EXTENDED, audit, read_rows
from run_floquet_pilot import parallel
from run_stability_p4 import OUTPUT, ROOT, read_json, sha, write_table
from three_phase_trap.long_rf_run import write_json


DEFAULT = OUTPUT.with_name("matched_radius_v2")


def finite_floquet_bound(q, maps, state, periods):
    """Use the two coordinates at the same time to tighten the radial bound."""
    axis_bounds = []
    sampled = []
    for axis, matrices in enumerate(maps):
        initial = state[[axis, axis + 2]]
        bound = position_bound(q, matrices, initial)
        if bound is None:
            return None
        axis_bounds.append(bound)
    radial_bound = np.hypot(*axis_bounds)
    if radial_bound < 1 - 1e-7:
        return radial_bound
    for axis, matrices in enumerate(maps):
        initial = state[[axis, axis + 2]]
        monodromy = matrices[-1]
        c = np.trace(monodromy) / 2
        angle = np.arccos(c)
        conjugate = (monodromy @ initial - c * initial) / np.sqrt(1 - c**2)
        phase = angle * np.arange(periods)
        basis = matrices[:, 0, :]
        sampled.append(np.outer(np.cos(phase), basis @ initial) +
                       np.outer(np.sin(phase), basis @ conjugate))
    max_radius = np.max(np.hypot(*sampled))
    # On each interval use the interpolation remainder for the vector
    # position, with ||R''|| <= 2*pi^2*q*max(||R||).
    error_factor = 2 * np.pi**2 * q * (PHASE_GRID[1] - PHASE_GRID[0])**2 / 8
    return max_radius / (1 - error_factor)


def solve_group(protocol, rows):
    coupling = float(rows[0]["coupling"])
    q = 4 * coupling / (protocol["C4_per_volt_m4"] * protocol["aperture_m"]**4)
    cache = {}
    outcomes = []
    for row in rows:
        state, phase = initial_state(protocol, row)
        periods = int(row["horizon_RF"])
        results = []
        for tight in (False, True):
            key = phase, tight
            if key not in cache:
                cache[key] = [fundamental(sign * q, phase, tight) for sign in (1, -1)]
            maps = cache[key]
            bound = finite_floquet_bound(q, maps, state, periods)
            result = (dict(status="held", escape_RF=None) if bound is not None and bound < 1 - 1e-7
                      else quadrupole(q, state, phase, periods, maps, tight=tight))
            results.append(result)
        coarse, fine = results
        status = fine["status"] if coarse["status"] == fine["status"] else "unresolved"
        time_difference = (abs(coarse["escape_RF"] - fine["escape_RF"])
                           if coarse["escape_RF"] is not None and fine["escape_RF"] is not None else None)
        outcomes.append((coupling, float(row["radius_mm"]), int(row["start"]), status, time_difference))
    return outcomes


def draw(out, cells, horizon, aperture):
    q = np.array([row["q_M_ref"] for row in cells])
    radius = np.array([row["radius_mm"] for row in cells])
    logq = np.log10(q)
    mesh = tri.Triangulation(logq, radius)
    vertices = mesh.triangles
    sparse = (np.ptp(logq[vertices], axis=1) > .53) | (np.ptp(radius[vertices], axis=1) > .66)
    sides = q[vertices] > FIRST_EDGE
    across_mathieu = sides.any(axis=1) & ~sides.all(axis=1)
    x = np.linspace(logq.min(), logq.max(), 720)
    y = np.linspace(radius.min(), radius.max(), 460)
    xx, yy = np.meshgrid(x, y)
    cmap = plt.get_cmap("viridis", 7)
    norm = BoundaryNorm(np.arange(-.5, 7.5), cmap.N)
    for contours in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout="constrained", sharex="row", sharey=True)
        for col, (model, title) in enumerate((("N48", "12 электродов · N48"), ("quadrupole", "Идеальный квадруполь"))):
            held = np.array([row[model + "_held"] for row in cells])
            resolved = np.array([not row.get(model + "_unresolved", 0) for row in cells])
            if contours:
                mask = sparse | (~resolved[vertices]).any(axis=1)
                if model == "quadrupole":
                    mask |= across_mathieu
                mesh.set_mask(mask)
                z = tri.LinearTriInterpolator(mesh, held / 6)(xx, yy)
            axes[0, col].set_title(title)
            for row, ax in enumerate(axes[:, col]):
                if contours:
                    ax.contourf(10**x, y, z, levels=(-1e-6, .5, 5/6, 1.000001),
                                colors=("#f0edf3", "#d9efea", "#91cbbf"), corner_mask=False)
                    ax.contour(10**x, y, z, levels=(.5, 5/6), colors=("#384552", "#1d7266"),
                               linestyles=("solid", "dashed"), linewidths=(1.25, 1.1))
                    ax.scatter(q[resolved], radius[resolved], s=5, color="#304a55", alpha=.48, zorder=3)
                else:
                    points = ax.scatter(q[resolved], radius[resolved], c=held[resolved], s=16, cmap=cmap, norm=norm,
                                        linewidths=.25, edgecolors="#334753", zorder=3)
                if not resolved.all():
                    ax.scatter(q[~resolved], radius[~resolved], s=20, marker="x", c="#a43e42", zorder=4)
                ax.axvline(FIRST_EDGE, color="#ab4745", ls=":", lw=1.25)
                ax.axhline(aperture, color="#56616b", ls=":", lw=.9)
                ax.set(xscale="log", ylim=(0, aperture + .15))
                if not row:
                    ax.set_xlim(q.min() / 1.15, q.max() * 1.15)
                else:
                    ax.set_xlim(.012, 2.5)
                    ax.set_xlabel(r"$|q_{\rm M}^{\rm ref}|$ (увеличение у границы Матьё)")
                ax.grid(alpha=.18)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"Стартовый радиус $r_0$, мм")
        legend = [Line2D([], [], color="#ab4745", ls=":", label="Первая граница Матьё")]
        if contours:
            legend += [Line2D([], [], color="#384552", label="50%"),
                       Line2D([], [], color="#1d7266", ls="--", label="5 из 6")]
        else:
            fig.colorbar(points, ax=axes, ticks=range(7), label="Удержанные старты из 6", shrink=.86)
        axes[0, 1].legend(handles=legend, loc="upper right", fontsize=8)
        kind = "Интерполированные контуры" if contours else "Рассчитанные точки"
        fig.suptitle(f"{kind} · a = 0 · {horizon:,} RF-периодов".replace(",", " "))
        for suffix in ("png", "svg"):
            fig.savefig(out / f"radius_{'contours' if contours else 'points'}.{suffix}", dpi=230)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source = EXTENDED / "extended_cells.csv"
    cells = read_rows(source)
    extended, _ = audit(EXTENDED, cells)
    original = OUTPUT / "protocol.json"
    protocol = read_json(original)
    horizon = extended["horizons_RF"][-1]
    if extended["starts"] != protocol["full_starts"] or extended["aperture_m"] != protocol["aperture_m"]:
        raise ValueError("Expanded scan has a different launch or aperture protocol")

    groups = defaultdict(list)
    for row in cells:
        for start in range(6):
            groups[float(row["coupling"])].append({**row, "start": start})
    print(f"Checking quadrupole on {len(cells)} cells and {len(groups)} RF settings", flush=True)
    outcomes = defaultdict(list)
    checks = []
    for group in parallel(solve_group, [(protocol, rows) for rows in groups.values()], args.workers):
        for coupling, radius, start, status, time_difference in group:
            outcomes[coupling, radius].append(status)
            if status == "unresolved" or (time_difference is not None and time_difference > 1e-5):
                checks.append(dict(coupling=coupling, radius_mm=radius, start=start,
                                   status=status, escape_time_difference_RF=time_difference))
    matched = []
    for row in cells:
        q, coupling, radius = (float(row[k]) for k in ("q_M_ref", "coupling", "radius_mm"))
        result = outcomes[coupling, radius]
        if len(result) != 6 or int(row["unresolved"]):
            raise ValueError(f"Incomplete comparison at q={q}, radius={radius}")
        matched.append(dict(q_M_ref=q, radius_mm=radius, N48_held=int(row["held"]),
                            quadrupole_held=result.count("held"), quadrupole_unresolved=result.count("unresolved")))

    args.out.mkdir(parents=True, exist_ok=True)
    write_table(args.out / "radius_matched_cells.csv", matched)
    if checks:
        write_table(args.out / "quadrupole_step_checks.csv", sorted(checks, key=lambda row: (
            row["coupling"], row["radius_mm"], row["start"])))
    write_json(args.out / "summary.json", dict(a=0, horizon_RF=horizon, aperture_mm=1000 * protocol["aperture_m"],
               cells=len(matched), starts_per_model=6 * len(matched), first_Mathieu_edge=FIRST_EDGE,
               unresolved_quadrupole=sum(row["quadrupole_unresolved"] for row in matched),
               quadrupole_step_flags=len(checks),
               full_cells={m: sum(r[m + "_held"] == 6 for r in matched) for m in ("N48", "quadrupole")},
               input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (source, EXTENDED / "protocol.json", original)}))
    draw(args.out, matched, horizon, 1000 * protocol["aperture_m"])
    print(f"Wrote matched radius maps to {args.out}", flush=True)


if __name__ == "__main__":
    main()
