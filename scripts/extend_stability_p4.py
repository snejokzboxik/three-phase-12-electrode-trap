"""Extend the finite-time N48 capture map with measured boundary refinements."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import sys
import time

from run_stability_p4 import (
    OUTPUT, ROOT, Launch, agreements, compute, grid_rows, read_json, sha,
    transition_intervals, verify_inputs, write_table,
)

import numpy as np
from scipy.interpolate import PchipInterpolator

from recheck_stability_p4 import Job, native_job
from run_floquet_pilot import parallel
from three_phase_trap.long_rf_run import run_lock, write_json


DEFAULT = OUTPUT.with_name("extended_v2")
LEVELS = (.5, 5 / 6, 1.)
RADII = (.1, .2, .3, .4, .5, .65, .75, .875, 1., 1.125, 1.25, 1.5, 1.75, 2.,
         2.4, 2.8, 3.2, 3.6, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.35, 7.6, 7.75, 7.9, 7.95)
PROFILES = {
    "dense": (RADII, (.055, .105, .15, .20, .26, .34, .46, .68), (1000, 10000), 3),
    "quick": ((.15, .3, .5, .75, 1., 1.25, 1.5, 2., 3., 4., 5., 6., 7., 7.6, 7.95),
              (.065, .13, .20, .30, .45, .68), (1000, 10000), 1),
    "smoke": ((.2, 1., 2.4, 7.95), (.08, .18, .26, .4), (16, 32), 1),
}


def archived_outcomes(horizon):
    """Read the completed map without editing its files or re-integrating it."""
    outcomes, inputs = {}, {}
    if horizon != 10000:
        return outcomes, inputs
    for directory in (OUTPUT / "N48" / f"RF{horizon}", OUTPUT / "N48" / "refined"):
        for path in sorted(directory.glob("*.json")):
            row = read_json(path)
            launch = Launch(row["coupling"], row["radius_mm"], row["start"])
            if row["horizon_RF"] != horizon or launch in outcomes or row["status"] not in ("held", "escaped"):
                raise ValueError(f"Invalid archived checkpoint: {path}")
            outcomes[launch] = row
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    if len(outcomes) != 420:
        raise ValueError("Expected all 420 trajectories of the previous map")
    return outcomes, inputs


def initial_cells(radii, etas, anchors, factor, aperture):
    cells = {(r, eta / (24 * (r * 1e-3 / aperture)**2)) for r in radii for eta in etas}
    for r in radii:
        if r >= 2:
            cells.update((r, q / factor) for q in anchors)
        if r >= 7:
            cells.update((r, eta / (24 * (r * 1e-3 / aperture)**2)) for eta in (.005, .012, .025))
    return cells


def evaluate(out, cells, existing, horizons, control, workers, coefficient):
    starts = [Launch(coupling, radius, i) for radius, coupling in sorted(cells) for i in range(6)]
    pending = [launch for launch in starts if launch not in existing]
    directory = out / "N48"
    first, last = horizons
    # A completed final checkpoint can be loaded directly on restart.
    final = directory / f"RF{last}"
    needs_prefix = [launch for launch in pending if not (final / (launch.tag + ".json")).exists()]
    prefix = compute(directory / f"RF{first}", [(launch, first, None) for launch in needs_prefix],
                     "N48", control, workers, coefficient)
    results = compute(final, [(launch, last, prefix.get(launch)) for launch in pending],
                      "N48", control, workers, coefficient)
    return {**existing, **results}


def axis_brackets(rows, axis, factor, anchors=()):
    """Brackets based only on resolved adjacent cells, including re-entries."""
    scan = defaultdict(list)
    for row in rows:
        r, lam = row["radius_mm"], row["coupling"]
        if axis == "q":
            scan[r].append((factor * lam, row))
        elif lam in anchors:
            scan[lam].append((r, row))
    for fixed, line in sorted(scan.items()):
        line.sort(key=lambda item: item[0])
        for (left, a), (right, b) in zip(line, line[1:]):
            if a["unresolved"] or b["unresolved"]:
                continue
            for level in LEVELS:
                fa, fb = a["survival_lower"], b["survival_lower"]
                if (fa >= level) == (fb >= level):
                    continue
                yield dict(axis=axis, fixed=fixed if axis == "q" else factor * fixed,
                           fixed_coupling=None if axis == "q" else fixed, level=level,
                           left=left, right=right, left_held=a["held"], right_held=b["held"],
                           direction="loss" if fa >= level else "reentry")


def refine_cells(brackets, factor):
    """Select up to two distinct broad brackets per row or vertical cut."""
    by_line = defaultdict(dict)
    for row in brackets:
        lo, hi = row["left"], row["right"]
        horizontal = row["axis"] == "q"
        width = np.log(hi / lo) if horizontal else hi - lo
        if width < (.075 if horizontal else .035):
            continue
        point = ((row["fixed"], np.sqrt(lo * hi) / factor) if horizontal else
                 (round((lo + hi) / 2, 6), row["fixed_coupling"]))
        line = by_line[row["axis"], row["fixed"]]
        line[point] = max(width, line.get(point, 0))
    return {point for group in by_line.values()
            for point in sorted(group, key=lambda p: (-group[p], p))[:2]}


def plotted_branches(ax, rows, brackets):
    """Draw guides only through consecutive radii with a single observed loss."""
    sampled = defaultdict(set)
    for row in rows:
        sampled[row["radius_mm"]].add(row["coupling"])
    radii = sorted(radius for radius, couplings in sampled.items() if len(couplings) >= 3)
    for level, color in ((.5, "#303746"), (5/6, "#147f72")):
        grouped = defaultdict(list)
        for row in brackets:
            if row["level"] == level:
                grouped[row["fixed"]].append(row)
        segments = [[]]
        for radius in radii:
            points = grouped.get(radius, [])
            if len(points) == 1 and points[0]["direction"] == "loss":
                point = points[0]
                segments[-1].append((radius, np.sqrt(point["left"] * point["right"])))
            elif segments[-1]:
                segments.append([])
        label = "Оценка 50%" if level == .5 else "Оценка 5/6"
        for segment in segments:
            if len(segment) < 2:
                continue
            r, q = np.asarray(segment).T
            rr = np.linspace(r[0], r[-1], max(60, len(r) * 14))
            guide = np.exp(PchipInterpolator(r, np.log(q))(rr))
            ax.plot(guide, rr, color=color, lw=1.25, label=label)
            label = "_nolegend_"


def draw(out, rows, q_brackets, radial_brackets, aperture):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, lower in zip(axes, (0., 4.), strict=True):
        plotted_branches(ax, rows, q_brackets)
        valid = [row for row in rows if not row["unresolved"] and row["radius_mm"] >= lower]
        unresolved = [row for row in rows if row["unresolved"] and row["radius_mm"] >= lower]
        markers = ax.scatter([row["q_M_ref"] for row in valid], [row["radius_mm"] for row in valid],
                             c=[row["survival_lower"] for row in valid], vmin=0, vmax=1,
                             cmap="viridis", s=19, edgecolors="#4b5661", linewidths=.3, zorder=3)
        if unresolved:
            ax.scatter([row["q_M_ref"] for row in unresolved], [row["radius_mm"] for row in unresolved],
                       color="#a43b39", marker="x", s=24, label="Численно неопределено")
        ax.axhline(1000 * aperture, color="#a43b39", ls=":", lw=1, label="Апертура")
        if lower:
            band = [row for row in radial_brackets if row["level"] == .5 and row["right"] >= lower]
            if band:
                ax.errorbar([row["fixed"] for row in band], [(row["left"] + row["right"]) / 2 for row in band],
                            yerr=[(row["right"] - row["left"]) / 2 for row in band], fmt="none",
                            color="#a65f23", capsize=3, lw=1.1, label="Интервал 50% по радиусу", zorder=4)
        ax.set(xscale="log", ylim=(lower, 1000 * aperture + .14),
               xlabel="Общий параметр Матьё |q|", ylabel="Стартовый радиус, мм")
        if valid and lower:
            lo = min(row["q_M_ref"] for row in valid)
            hi = max(row["q_M_ref"] for row in valid)
            ax.set_xlim(lo / 1.35, hi * 1.35)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="lower left" if lower else "upper right")
    axes[0].set_title("Расширенная карта N48")
    axes[1].set_title("Увеличение у апертуры")
    fig.colorbar(markers, ax=axes, label="Доля удержанных стартов")
    fig.savefig(out / "extended_map.png", dpi=220)
    fig.savefig(out / "extended_map.svg")
    plt.close(fig)


def direct_controls(out, results, workers, reference, coefficient, protocol):
    pairs = list(transition_intervals(results))
    chosen = []
    for test in (lambda r: r < .5, lambda r: r >= 6):
        eligible = [pair for pair in pairs if test(pair[0].radius_mm)]
        if eligible:
            closest = min(eligible, key=lambda p: (-p[0].radius_mm if p[0].radius_mm >= 6 else p[0].radius_mm,
                                                   p[1].coupling / p[0].coupling, p[0].start))
            chosen.extend(closest)
    directory = out / "direct_checks"
    directory.mkdir(exist_ok=True)
    horizon = protocol["horizons_RF"][-1]
    strict = compute(directory / "N48_strict", [(launch, horizon, None) for launch in chosen], "N48",
                     protocol["N48_strict"], workers, coefficient)
    jobs = [Job(launch, mesh, level, horizon) for launch in chosen for mesh in (35, 50) for level in ("coarse", "fine")]
    records = parallel(native_job, [(directory, job, coefficient, reference[job.level]) for job in jobs], min(workers, 3))
    summary = []
    for launch in chosen:
        variants = [row for row in records if all(row[k] == value for k, value in asdict(launch).items())]
        agreement, spread = agreements(results[launch], [strict[launch], *variants])
        summary.append(dict(**asdict(launch), agreement=agreement, escape_spread_RF=spread,
                            N48_status=results[launch]["status"], N48_strict_status=strict[launch]["status"],
                            **{row["model"] + "_status": row["status"] for row in variants}))
    if summary:
        write_table(out / "direct_controls.csv", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="dense")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, help="Adaptive refinement passes")
    parser.add_argument("--p4-controls", action="store_true", help="Add selected direct checks after N48")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    radii, etas, horizons, rounds = PROFILES[args.profile]
    if args.rounds is not None:
        rounds = args.rounds
    if not 1 <= args.workers <= 16 or not 0 <= rounds <= 8:
        parser.error("Expected 1–16 workers and 0–8 refinement rounds")
    out = (args.out or (DEFAULT if args.profile == "dense" else DEFAULT.with_name(args.profile + "_extended_v2"))).resolve()
    if out == OUTPUT.resolve() or OUTPUT.resolve() in out.parents:
        parser.error("Use an output directory outside the archived map")
    coefficient, reference, hashes = verify_inputs()
    source = read_json(OUTPUT / "protocol.json")
    if coefficient != source["C4_per_volt_m4"] or any(hashes.get(k) != v for k, v in source["input_sha256"].items()):
        raise ValueError("Input differs from the archived N48 map")
    hashes[(OUTPUT / "protocol.json").relative_to(ROOT).as_posix()] = sha(OUTPUT / "protocol.json")
    for path in (Path(__file__), ROOT / "scripts/recheck_stability_p4.py", ROOT / "scripts/run_floquet_pilot.py",
                 ROOT / "src/three_phase_trap/stability_rf_p4_v2.py"):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    existing, archive_hashes = archived_outcomes(horizons[-1])
    aperture = source["aperture_m"]
    factor = 4 / (coefficient * aperture**4)
    anchor_q = (.01, .02, factor * source["couplings"][0], .08, .16, .32, .64)
    anchors = {q / factor for q in anchor_q}
    cells = initial_cells(radii, etas, anchor_q, factor, aperture)
    protocol = dict(version=2, profile=args.profile, horizons_RF=list(horizons), aperture_m=aperture,
                    radii_mm=list(radii), initial_eta=list(etas), anchor_q=list(anchor_q), rounds=rounds,
                    thresholds=list(LEVELS), N48=source["N48"], N48_strict=source["N48_strict"],
                    starts=source["full_starts"], source_sha256=hashes, reused_sha256=archive_hashes)
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        path = out / "protocol.json"
        if path.exists() and read_json(path) != protocol:
            raise ValueError("Protocol differs; choose a new --out directory")
        if not path.exists():
            write_json(path, protocol)
        new_cells = cells - {(launch.radius_mm, launch.coupling) for launch in existing}
        print(f"[READY] {len(new_cells)} new seed cells x 6; {len(existing)} archived trajectories; "
              f"{rounds} adaptive passes; output: {out}", flush=True)
        if args.check_only:
            return
        started = time.perf_counter()
        results = existing
        history = []
        for stage in range(rounds + 1):
            results = evaluate(out, cells, results, horizons, protocol["N48"], args.workers, coefficient)
            rows = [{**row, "q_M_ref": factor * row["coupling"]} for row in grid_rows(results)]
            horizontal = list(axis_brackets(rows, "q", factor))
            vertical = list(axis_brackets(rows, "radius", factor, anchors))
            history.append(dict(round=stage, cells=len(rows), trajectories=len(results)))
            print(f"[PASS {stage}] {len(rows)} cells, {len(horizontal)} q brackets, {len(vertical)} radial brackets", flush=True)
            write_table(out / "extended_cells.csv", rows)
            if horizontal:
                write_table(out / "q_boundary_brackets.csv", horizontal)
            if vertical:
                write_table(out / "radial_boundary_brackets.csv", vertical)
            draw(out, rows, horizontal, vertical, aperture)
            cells = refine_cells([*horizontal, *vertical], factor) - {(r["radius_mm"], r["coupling"]) for r in rows}
            if not cells:
                break
        checks = direct_controls(out, results, args.workers, reference, coefficient, protocol) if args.p4_controls else []
        write_json(out / "summary.json", dict(
            history=history, reused_trajectories=len(existing),
            held=sum(r["status"] == "held" for r in results.values()),
            escaped=sum(r["status"] == "escaped" for r in results.values()),
            unresolved=sum(r["status"] == "numerical_error" for r in results.values()),
            direct_controls=checks, wall_this_run_s=time.perf_counter() - started))
        print("[COMPLETE] " + str(out), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Repeat the same command to resume saved trajectories.", flush=True)
        sys.exit(130)
