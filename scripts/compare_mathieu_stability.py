"""Compare the archived P4/N48 capture map to a matched ideal quadrupole."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import mathieu_b

from run_stability_p4 import OUTPUT, ROOT, read_json, sha, write_table
from three_phase_trap.long_rf_run import write_json


DEFAULT = OUTPUT.with_name("mathieu_comparison_v1")
PHASE_GRID = np.linspace(0, 1, 129)
FIRST_EDGE = brentq(lambda q: mathieu_b(1, q), .8, 1.)


def fundamental(q, phase, tight):
    """One RF period in (x/R, dx/dn), where n counts RF periods."""
    scale = 2 * np.pi**2 * q

    def rhs(n, flat):
        x0, x1, v0, v1 = flat
        k = scale * np.sin(2 * np.pi * n + phase)
        return v0, v1, -k * x0, -k * x1

    sol = solve_ivp(rhs, (0., 1.), np.eye(2).ravel(), t_eval=PHASE_GRID, method="DOP853",
                    max_step=1 / (64 if tight else 32), rtol=2e-13 if tight else 2e-11,
                    atol=2e-15 if tight else 2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(-1, 2, 2)


def position_bound(q, matrices, initial):
    """Floquet envelope valid for every period in the elliptic regime."""
    period = matrices[-1]
    c = np.trace(period) / 2
    if abs(c) >= 1 or abs(np.linalg.det(period) - 1) > 1e-9:
        return None
    sine = np.sqrt(1 - c**2)
    companion = (period @ initial - c * initial) / sine
    rows = matrices[:, 0, :]
    envelope = np.hypot(rows @ initial, rows @ companion).max()
    # Linear interpolation remainder for z'' = -k z gives this between-node bound.
    error_factor = 2 * np.pi**2 * abs(q) * (PHASE_GRID[1] - PHASE_GRID[0])**2 / 8
    return float(envelope / (1 - error_factor)) if error_factor < 1 else None


def quadrupole(q, initial, phase, periods, maps, *, tight):
    bounds = [position_bound(sign * q, m, initial[[i, i + 2]])
              for i, (sign, m) in enumerate(zip((1, -1), maps, strict=True))]
    if all(b is not None for b in bounds):
        bound = float(np.hypot(*bounds))
        if bound < 1 - 1e-7:
            return dict(status="held", escape_RF=None, method="floquet_envelope", radius_bound_over_aperture=bound)

    def rhs(n, state):
        x, y, vx, vy = state
        k = 2 * np.pi**2 * q * np.sin(2 * np.pi * n + phase)
        return vx, vy, -k * x, k * y

    def escape(_, state):
        return np.hypot(*state[:2]) - 1

    escape.terminal = True
    escape.direction = 1
    sol = solve_ivp(rhs, (0., periods), initial, events=escape, method="DOP853",
                    max_step=1 / (48 if tight else 24), rtol=2e-12 if tight else 2e-10,
                    atol=2e-14 if tight else 2e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    if len(sol.t_events[0]):
        return dict(status="escaped", escape_RF=float(sol.t_events[0][0]), method="DOP853_event",
                    radius_bound_over_aperture=None)
    return dict(status="held", escape_RF=None, method="DOP853_event", radius_bound_over_aperture=None)


def initial_state(protocol, row):
    azimuth, phase, vt = protocol["full_starts"][int(row["start"])]
    c, s = np.cos(azimuth), np.sin(azimuth)
    radius, frequency = protocol["aperture_m"], protocol["reference_frequency_Hz"]
    pos = float(row["radius_mm"]) * 1e-3 / radius
    vel = vt / (frequency * radius)
    return np.array([pos*c, pos*s, -vel*s, vel*c]), phase


def solve_comparison(protocol, source):
    conversion = 4 / (protocol["C4_per_volt_m4"] * protocol["aperture_m"]**4)
    comparisons, axes, cache = [], {}, {}
    for row in source:
        coupling, radius = (float(row[key]) for key in ("coupling", "radius_mm"))
        q = conversion * coupling
        initial, phase = initial_state(protocol, row)
        results = []
        for tight in (False, True):
            key = (q, phase, tight)
            if key not in cache:
                cache[key] = (fundamental(q, phase, tight), fundamental(-q, phase, tight))
            maps = cache[key]
            results.append(quadrupole(q, initial, phase, int(row["horizon_RF"]), maps, tight=tight))
        coarse, fine = results
        if coarse["status"] != fine["status"]:
            raise RuntimeError(f"Unresolved quadrupole classification: {row}")
        if fine["escape_RF"] is not None:
            if abs(coarse["escape_RF"] - fine["escape_RF"]) > 1e-5:
                raise RuntimeError(f"Unresolved quadrupole escape time: {row}")
        trace = float(np.trace(cache[(q, phase, True)][0][-1]))
        band = abs(trace) < 2
        if q <= 1.5 and band != (q < FIRST_EDGE):
            raise RuntimeError(f"Monodromy and known first Mathieu band disagree at q={q}")
        if q in axes and band != axes[q]["mathieu_bounded_at_a0"]:
            raise RuntimeError(f"Mathieu result depends on initial RF phase at q={q}")
        axes[q] = dict(lambda4=coupling, q_M_ref=q, a=0., mathieu_bounded_at_a0=band, monodromy_trace=trace)
        comparisons.append(dict(radius_mm=radius, start=int(row["start"]), lambda4=coupling, q_M_ref=q, a=0.,
                                eta_octupole_launch=24 * coupling * (radius * 1e-3 / protocol["aperture_m"])**2,
                                horizon_RF=int(row["horizon_RF"]), N48_status=row["status"],
                                N48_escape_RF=float(row["escape_RF"]) if row["escape_RF"] else None,
                                **{f"quadrupole_{key}": value for key, value in fine.items()}))
    return comparisons, [axes[q] for q in sorted(axes)]


def cell_results(rows, coarse_lambda):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["q_M_ref"], row["radius_mm"])].append(row)
    return [dict(q_M_ref=q, radius_mm=r, lambda4=group[0]["lambda4"],
                 eta_octupole_launch=group[0]["eta_octupole_launch"], coarse=group[0]["lambda4"] in coarse_lambda,
                 starts=len(group), N48_held=sum(x["N48_status"] == "held" for x in group),
                 quadrupole_held=sum(x["quadrupole_status"] == "held" for x in group))
            for (q, r), group in sorted(groups.items())]


def area_estimate(cells):
    """Nearest-grid-cell quadrature within the original rectangular domain."""
    coarse = [row for row in cells if row["coarse"]]
    q = sorted({row["q_M_ref"] for row in coarse})
    radius = sorted({row["radius_mm"] for row in coarse})
    if len(coarse) != len(q) * len(radius):
        raise ValueError("The common coarse grid is incomplete")

    def weights(x):
        x = np.asarray(x)
        edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
        return np.diff(edges) / (x[-1] - x[0])

    radial = dict(zip(radius, weights(radius), strict=True))
    result = {}
    for mode, axis in (("linear_q", q), ("log_q", np.log(q))):
        horizontal = dict(zip(q, weights(axis), strict=True))
        result[mode] = {}
        for model in ("N48", "quadrupole"):
            held = np.array([row[f"{model}_held"] / row["starts"] for row in coarse])
            w = np.array([horizontal[row["q_M_ref"]] * radial[row["radius_mm"]] for row in coarse])
            result[mode][model] = dict(full_capture_area_fraction=float(w @ (held == 1)),
                                       mean_capture_fraction=float(w @ held))
    return result


def plot(path, cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4), sharex=True, sharey=True, layout="constrained")
    for ax, model, name in zip(axes, ("N48", "quadrupole"), ("12-electrode trap, N48 from P4", "Ideal quadrupole"), strict=True):
        for coarse, marker in ((True, "s"), (False, "D")):
            subset = [row for row in cells if row["coarse"] == coarse]
            artist = ax.scatter([row["q_M_ref"] for row in subset], [row["radius_mm"] for row in subset],
                                c=[row[f"{model}_held"] / row["starts"] for row in subset], marker=marker,
                                s=58, cmap="viridis", vmin=0, vmax=1, edgecolors="#484848", linewidths=.35)
        ax.axvline(FIRST_EDGE, color="#b43947", linestyle="--", lw=1, label="First Mathieu edge: 0.908")
        ax.set(xscale="log", title=name, xlabel=r"Reference quadrupole $|q_{\rm M}|$ at $a=0$")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, .91))
        ax.grid(alpha=.15)
    axes[0].set(ylabel="Launch radius [mm]", yticks=sorted({row["radius_mm"] for row in cells}))
    fig.colorbar(artist, ax=axes, label="Held fraction at 10,000 RF periods")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    source = OUTPUT / "n48_trajectories.csv"
    protocol_path = OUTPUT / "protocol.json"
    protocol = read_json(protocol_path)
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 420 or any(row["status"] not in {"held", "escaped"} for row in rows):
        raise ValueError("Expected the completed N48 map with resolved classifications")
    boundary = OUTPUT.with_name("p4_recheck_v2") / "summary.json"
    args.out.mkdir(parents=True, exist_ok=False)
    comparisons, axis = solve_comparison(protocol, rows)
    cells = cell_results(comparisons, set(protocol["couplings"]))
    brackets = []
    for row in read_json(boundary)["cases"]:
        r, coupling = row["radius_mm"], row["coupling"]
        brackets.append(dict(radius_mm=r, lambda4=coupling, q_M_ref=4 * coupling /
                             (protocol["C4_per_volt_m4"] * protocol["aperture_m"]**4),
                             eta_octupole_launch=24 * coupling * (r * 1e-3 / protocol["aperture_m"])**2,
                             P4_result=row["direct_P4_agreement"]))
    write_table(args.out / "matched_trajectories.csv", comparisons)
    write_table(args.out / "matched_cells.csv", cells)
    write_table(args.out / "mathieu_axis.csv", axis)
    write_table(args.out / "direct_p4_brackets.csv", brackets)
    counts = {}
    for model in ("N48", "quadrupole"):
        counts[model] = {scope: dict(cells=len(group), starts=sum(row["starts"] for row in group),
                                     all_held_cells=sum(row[f"{model}_held"] == row["starts"] for row in group),
                                     held_starts=sum(row[f"{model}_held"] for row in group))
                         for scope, group in (("coarse", [c for c in cells if c["coarse"]]), ("all", cells))}
    write_json(args.out / "summary.json", dict(first_mathieu_edge=FIRST_EDGE, counts=counts,
               area=area_estimate(cells), input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in
                                                        (source, protocol_path, boundary, Path(__file__))}))
    plot(args.out / "matched_maps.png", cells)
    print((args.out / "summary.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
