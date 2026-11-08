"""Matched (a,q) capture maps for ideal static quadrupole bias and full RF."""
import argparse
from collections import Counter
from pathlib import Path
import sys

from run_stability_p4 import FREQUENCY, OUTPUT, P4, RADIUS, ROOT, Launch, read_json, sha, write_table

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.special import mathieu_a, mathieu_b

from compare_mathieu_stability import FIRST_EDGE, PHASE_GRID, position_bound
from run_floquet_pilot import parallel
from three_phase_trap.long_rf_run import run_lock, write_json
from three_phase_trap.pilot_dynamics import PolynomialRF
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction


DEFAULT = OUTPUT.with_name("aq_ideal_dc_v1")
A_GRID = (0., .002, .01, .025, .05, .10, .16, .22, .30, .40)
EXTRA_Q = (.12, .30, .58, .706, .82, FIRST_EDGE, 1.15)
_field = None


def normalized_rf(coefficient):
    global _field
    if _field is None:
        q = RFReconstruction.load(P4 / "h35_P4/diagnostic_N48.json").unit_quadratures
        scale = coefficient * RADIUS**4
        _field = PolynomialRF(HarmonicQuadratures(q.cosine / scale, q.sine / scale, 1.))
    return _field


def transverse_coefficients(a, q, phase, n):
    """a_x=a, q_x=-q in the sine convention and RF periods as time."""
    return np.pi**2 * (a + 2 * q * np.sin(2 * np.pi * n + phase))


def quadrupole_bound(a, q, phase, state):
    bounds = []
    for axis, sign in enumerate((1, -1)):
        def rhs(n, y):
            k = sign * transverse_coefficients(a, q, phase, n)
            return y[2], y[3], -k * y[0], -k * y[1]

        sol = solve_ivp(rhs, (0., 1.), np.eye(2).ravel(), t_eval=PHASE_GRID, method="DOP853",
                        max_step=1 / 48, rtol=3e-13, atol=3e-15)
        if not sol.success:
            raise RuntimeError(sol.message)
        bound = position_bound(abs(q) + abs(a) / 2, sol.y.T.reshape(-1, 2, 2), state[[axis, axis + 2]])
        if bound is None:
            return None
        bounds.append(bound)
    return float(np.hypot(*bounds))


def trajectory(model, a, q, coefficient, horizon, start, tight, previous=None):
    factor = 4 / (coefficient * RADIUS**4)
    units = np.array([RADIUS, RADIUS, RADIUS * FREQUENCY, RADIUS * FREQUENCY])
    state_si, phase = Launch(q / factor, 1., start).state()
    initial = state_si / units if previous is None else np.asarray(previous["final_state"])
    offset = previous["horizon_RF"] if previous else 0
    static = np.pi**2 * a
    if model == "quadrupole":
        bound = quadrupole_bound(a, q, phase, initial)
        if bound is not None and bound < 1 - 1e-7:
            return dict(status="held", escape_RF=None, method="floquet_bound", final_state=None)

        def rhs(n, z):
            k = transverse_coefficients(a, q, phase, n)
            return z[2], z[3], -k * z[0], k * z[1]
    else:
        rf = normalized_rf(coefficient)
        scale = 4 * np.pi**2 * q / factor

        def rhs(n, z):
            x, y, vx, vy = z
            rc = np.hypot(x, y)
            # Trial stages can go beyond the terminal circle before root location.
            xc, yc = (x / rc, y / rc) if rc > 1 else (x, y)
            cx, cy, sx, sy = rf.quadratures(xc, yc)
            angle = 2 * np.pi * n + phase
            co, si = np.cos(angle), np.sin(angle)
            return vx, vy, scale * (cx * co + sx * si) - static * x, scale * (cy * co + sy * si) + static * y

    def escape(_, z):
        return np.hypot(z[0], z[1]) - 1

    escape.direction, escape.terminal = 1, True
    duration = horizon - offset
    sol = solve_ivp(rhs, (0., duration), initial, t_eval=[duration], events=escape, method="DOP853",
                    max_step=1 / (24 if tight else 12), rtol=2e-11 if tight else 2e-9,
                    atol=2e-13 if tight else 2e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    if len(sol.t_events[0]):
        return dict(status="escaped", escape_RF=offset + float(sol.t_events[0][0]), method="DOP853",
                    final_state=None)
    return dict(status="held", escape_RF=None, method="DOP853", final_state=sol.y[:, -1].tolist())


def cell(directory, previous, coefficient, horizon, a, q, tight=False):
    name = f"a{a:.12f}_q{q:.12f}" + ("_tight" if tight else "")
    path = directory / (name + ".json")
    if path.exists():
        return read_json(path)
    prior_path = previous / path.name
    old = {(r["model"], r["start"]): r for r in read_json(prior_path)} if prior_path.exists() else {}
    rows = []
    for model in ("N48", "quadrupole"):
        for start in range(6):
            earlier = old.get((model, start))
            if earlier and (earlier["status"] == "escaped" or earlier["method"] == "floquet_bound"):
                result = {k: earlier[k] for k in ("status", "escape_RF", "method", "final_state")}
            else:
                resume = earlier if earlier and earlier["status"] == "held" else None
                try:
                    result = trajectory(model, a, q, coefficient, horizon, start, tight, previous=resume)
                except (ValueError, RuntimeError) as exc:
                    result = dict(status="numerical_error", reason=str(exc), escape_RF=None,
                                  method="failed", final_state=None)
            rows.append(dict(model=model, start=start, a=a, q_M_ref=q, horizon_RF=horizon, **result))
    write_json(path, rows)
    return rows


def control_locations(grid, a_values, q_values):
    changes = []
    for iq, q in enumerate(q_values):
        for ia, a in enumerate(a_values[1:], 1):
            if grid[ia, iq] != grid[ia - 1, iq]:
                changes.append((a, q))
    # Deterministic coverage across RF settings; up to six cells.
    if len(changes) > 6:
        changes = [changes[i] for i in np.linspace(0, len(changes) - 1, 6).round().astype(int)]
    return changes


def first_band(ax):
    q = np.linspace(0, FIRST_EDGE, 420)
    ax.plot(q, np.minimum(-mathieu_a(0, q), mathieu_b(1, q)), color="#a94046", lw=1.25, ls="--",
            label="Первая область Матьё")


def draw(directory, arrays, a, q, horizon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for contours in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.4), sharex=True, sharey=True, layout="constrained")
        qq, aa = np.meshgrid(q, a)
        for ax, (name, grid) in zip(axes, arrays.items(), strict=True):
            if contours:
                x = np.linspace(q[0], q[-1], 420)
                y = np.linspace(a[0], a[-1], 360)
                # Piecewise-linear cells when the grid contains numerical gaps.
                if np.isfinite(grid).all():
                    first = PchipInterpolator(q, grid, axis=1)(x)
                    z = PchipInterpolator(a, first, axis=0)(y)
                    ax.contourf(x, y, z, levels=(.5, 5/6, 1.000001), colors=("#daeeea", "#91cbc0"))
                    ax.contour(x, y, z, levels=(.5, 5/6), colors=("#253641", "#24786e"),
                               linewidths=(1.3, 1.), linestyles=("solid", "dashed"))
                else:
                    ax.contourf(q, a, np.ma.masked_invalid(grid), levels=(.5, 5/6, 1.000001),
                                colors=("#daeeea", "#91cbc0"), corner_mask=False)
                ax.scatter(qq, aa, s=8, c="#364c59", alpha=.48, zorder=4)
            else:
                artist = ax.scatter(qq, aa, c=grid, s=43, cmap="viridis", vmin=0, vmax=1,
                                    linewidths=.3, edgecolors="#44515f", zorder=4)
                missing = ~np.isfinite(grid)
                ax.scatter(qq[missing], aa[missing], marker="x", c="#a94046", zorder=5)
            first_band(ax)
            ax.legend(fontsize=8, loc="upper left")
            ax.set(title=name, xlabel=r"$|q_{\mathrm{M}}^{\mathrm{ref}}|$",
                   xlim=(0, q[-1] * 1.035), ylim=(-.008, a[-1] + .012))
            ax.grid(alpha=.16)
        axes[0].set_ylabel(r"$a$ общего квадрупольного DC")
        if not contours:
            fig.colorbar(artist, ax=axes, label="Доля удержанных стартов")
        caption = "контуры 50% и 5/6" if contours else "рассчитанные точки"
        fig.suptitle(f"Старт 1 мм · {horizon:,} RF-периодов · {caption}".replace(",", " "))
        name = "aq_contours" if contours else "aq_points"
        fig.savefig(directory / (name + ".png"), dpi=235)
        fig.savefig(directory / (name + ".svg"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--horizon", type=int, choices=(1000, 10000), default=1000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Expected 1–16 workers")
    original = OUTPUT / "protocol.json"
    archive = read_json(original)
    coefficient = archive["C4_per_volt_m4"]
    factor = 4 / (coefficient * RADIUS**4)
    q_values = sorted({*EXTRA_Q, *(lam * factor for lam in archive["couplings"][:5])})
    rf_path = P4 / "h35_P4/diagnostic_N48.json"
    old_hash = archive["input_sha256"][rf_path.relative_to(ROOT).as_posix()]
    if sha(rf_path) != old_hash:
        raise ValueError("Archived RF field changed")
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        protocol = dict(a=list(A_GRID), q=q_values, aperture_m=RADIUS, radius_mm=1.,
                        static="ideal Udc (x^2-y^2)/R^2 shared by both fields",
                        hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in
                                (original, rf_path, Path(__file__), ROOT / "scripts/compare_mathieu_stability.py",
                                 ROOT / "scripts/run_stability_p4.py")})
        config = directory / "protocol.json"
        if config.exists() and read_json(config) != protocol:
            raise ValueError("Use a new --out directory for a different (a,q) protocol")
        if not config.exists():
            write_json(config, protocol)
        h = args.horizon
        out = directory / f"RF{h}"
        cells = out / "cells"
        cells.mkdir(parents=True, exist_ok=True)
        prior = directory / "RF1000" / "cells" if h > 1000 else directory / "no_prior"
        print(f"[READY] {len(A_GRID)*len(q_values)} cells, 6 starts per model, {h} RF periods", flush=True)
        jobs = [(cells, prior, coefficient, h, a, q) for a in A_GRID for q in q_values]
        rows = [r for batch in parallel(cell, jobs, args.workers) for r in batch]
        groups = {}
        for r in rows:
            groups.setdefault((r["a"], r["q_M_ref"], r["model"]), []).append(r["status"])
        counts = {}
        table = []
        for (a, q, model), states in sorted(groups.items()):
            held, escaped, error = (states.count(k) for k in ("held", "escaped", "numerical_error"))
            counts[(a, q, model)] = held / 6 if not error else np.nan
            table.append(dict(a=a, q_M_ref=q, model=model, held=held, escaped=escaped, unresolved=error))
        n48 = np.array([[counts[a, q, "N48"] for q in q_values] for a in A_GRID])
        spot = control_locations(n48, A_GRID, q_values)
        extra = [r for batch in parallel(cell, [(cells, prior, coefficient, h, a, q, True) for a, q in spot],
                                        args.workers) for r in batch]
        baseline = {(r["a"], r["q_M_ref"], r["model"], r["start"]): r["status"] for r in rows}
        mismatches = []
        for row in extra:
            a, q, model, start = (row[k] for k in ("a", "q_M_ref", "model", "start"))
            if row["status"] != baseline[a, q, model, start]:
                mismatches.append(dict(a=a, q_M_ref=q, model=model, start=start, baseline=baseline[a, q, model, start],
                                       tightened=row["status"]))
        mismatch_cells = {(r["a"], r["q_M_ref"], r["model"]) for r in mismatches}
        for key in mismatch_cells:
            counts[key] = np.nan
        write_table(out / "capture_cells.csv", [{**row, "step_check_disagreement":
                    (row["a"], row["q_M_ref"], row["model"]) in mismatch_cells} for row in table])
        write_table(out / "trajectories.csv", rows)
        write_table(out / "step_checks.csv", extra) if extra else None
        arrays = {name: np.array([[counts[a, q, model] for q in q_values] for a in A_GRID])
                  for name, model in (("12 электродов · N48 + DC", "N48"), ("Идеальный квадруполь + DC", "quadrupole"))}
        draw(out, arrays, A_GRID, q_values, h)
        write_json(out / "summary.json", dict(cells=len(groups)//2, horizon_RF=h, starts_per_model=len(rows)//2,
                   outcomes={m: dict(Counter(r["status"] for r in rows if r["model"] == m))
                             for m in ("N48", "quadrupole")}, step_checked_cells=len(spot), step_disagreements=mismatches))
        print((out / "summary.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Repeat the same command to reuse completed cells.", flush=True)
        sys.exit(130)
