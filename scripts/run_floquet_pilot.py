"""Find noncentral periodic RF orbits and check selected spectra on native P4."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

# The existing launcher sets BLAS threads to one before loading NumPy.
from run_stability_p4 import FREQUENCY, OUTPUT, P4, RADIUS, ROOT, Launch, read_json, sha, verify_inputs, write_table

import numpy as np

from recheck_stability_p4 import native
from three_phase_trap.floquet_orbits import (
    IDENTITY, PHASE, ShootingFailure, SmoothRFOrbit, circular_seed, shoot, shortest_period, slow_seed, spectrum,
)
from three_phase_trap.long_rf_run import run_lock, write_json
from three_phase_trap.pilot_dynamics import PolynomialRF, integrate_rf
from three_phase_trap.rf_reconstruction import RFReconstruction


DEFAULT = ROOT / "outputs/floquet_orbits/v1"
ROTATE_BACK = np.kron(np.eye(2), [[0., 1.], [-1., 0.]])
N48_PERIODS = (4, 16, 20, 24, 28, 32)
SAMPLES = 8
CLOSE = 4e-9
UNITS = np.array([RADIUS, RADIUS, RADIUS * FREQUENCY, RADIUS * FREQUENCY])


def analytic_check():
    cosine = np.zeros(5, complex)
    sine = cosine.copy()
    cosine[4], sine[4] = 1j, 1.
    model = SmoothRFOrbit(cosine, sine)
    state = circular_seed(.55)
    end, tangent, _ = model.flow(state, .55, 4, steps=64)
    closure = float(np.linalg.norm(end - state, ord=np.inf))
    rho = float(max(abs(np.linalg.eigvals(tangent))))
    expected = np.exp(2 * np.pi * np.sqrt(2))
    if closure > 1e-8 or abs(rho / expected - 1) > 2e-7:
        raise RuntimeError("Analytic circular-octupole Floquet control failed")
    return dict(closure=closure, spectral_radius=rho, analytic_spectral_radius=expected,
                control="exact rotating circle, four RF periods")


def n48_model(coefficient):
    q = RFReconstruction.load(P4 / "h35_P4/diagnostic_N48.json").unit_quadratures
    if q.reference_radius != RADIUS:
        raise ValueError("Unexpected N48 reference radius")
    return SmoothRFOrbit(q.cosine / (coefficient * RADIUS**4), q.sine / (coefficient * RADIUS**4))


def n48_orbit(model, coupling, k, previous):
    def valid(z):
        return .035 < z[0] < .45 and abs(z[1]) < .1 and np.linalg.norm(z[2:]) < .85

    if k == 4:
        guess = circular_seed(coupling)

        def evaluate(z):
            end, jac, _ = model.flow(z, coupling, 1, steps=24)
            return ROTATE_BACK @ end - z, ROTATE_BACK @ jac - IDENTITY

        state = shoot(evaluate, guess, valid, tolerance=2e-12)
    else:
        state = None
        # Reversibility at phase pi/2 suggests y=vx=0 at both endpoints
        # of the half orbit. The full-period shooting below is mandatory.
        keep, vanish = [0, 3], [1, 2]

        def lift(u):
            return np.array([u[0], 0., 0., u[1]])

        def half(u):
            end, jac, _ = model.flow(lift(u), coupling, k / 2, steps=24)
            return end[vanish], jac[np.ix_(vanish, keep)]

        guess = slow_seed(coupling, k)[keep]
        seeds = [np.array([guess[0] * r, guess[1] * v])
                 for r, v in ((1., 1.), (.94, .8), (1.06, 1.25), (1., .5))]
        if previous is not None:
            seeds.insert(0, np.array(previous["initial"])[keep] * np.sqrt(previous["lambda4"] / coupling))
        failures = []
        for seed in seeds:
            try:
                state = lift(shoot(half, seed, lambda u: valid(lift(u)), tolerance=2e-11))
                break
            except ShootingFailure as exc:
                failures.append(str(exc))
        if state is None:
            raise ShootingFailure(f"Half-orbit search failed: {failures[-1]}")

    def full(z):
        end, jac, _ = model.flow(z, coupling, k, steps=48)
        return end - z, jac - IDENTITY

    state = shoot(full, state, valid, tolerance=CLOSE / 4, iterations=6)
    end, matrix, samples = model.flow(state, coupling, k, steps=64, samples=SAMPLES)
    closure = float(np.linalg.norm(end - state, ord=np.inf))
    if closure > CLOSE:
        raise ShootingFailure(f"Full return failed independent step check: {closure:.3g}")
    period = shortest_period(samples, SAMPLES, CLOSE * 4)
    if period != k:
        raise ShootingFailure(f"Converged to repeated lower period {period}")
    row = spectrum(matrix, defect_limit=2e-5)
    _, lower_matrix, _ = model.flow(state, coupling, k, steps=48)
    rho_lower = float(max(abs(np.linalg.eigvals(lower_matrix))))
    rho_change = float(abs(np.log(rho_lower / row["spectral_radius"])))
    if rho_change > 2e-5:
        row["classification"] = "numerical_unresolved"
    values, directions = np.linalg.eig(matrix)
    vector = directions[:, np.argmax(abs(values))]
    direction = max((vector.real.copy(), vector.imag.copy()), key=np.linalg.norm)
    direction /= np.linalg.norm(direction)
    perturbation = 1e-8 * direction
    disturbed, _, _ = model.flow(state + perturbation, coupling, k, steps=64, variational=False)
    undisturbed, _, _ = model.flow(state, coupling, k, steps=64, variational=False)
    predicted = matrix @ perturbation
    perturbation_error = float(np.linalg.norm(disturbed - undisturbed - predicted) / np.linalg.norm(predicted))
    return dict(status="closed", lambda4=coupling, period_RF=k, initial=state.tolist(),
                closure=closure, log_rho_step_change=rho_change, perturbation_error=perturbation_error,
                perturbation_direction=direction.tolist(),
                launch_radius_mm=float(np.linalg.norm(state[:2]) * RADIUS * 1e3),
                radius_max_mm=float(np.linalg.norm(samples[:, :2], axis=1).max() * RADIUS * 1e3),
                initial_speed_m_s=float(np.linalg.norm(state[2:]) * RADIUS * FREQUENCY),
                samples=samples.tolist(), **row)


def n48_branch(directory, coefficient, grid, k):
    model = n48_model(coefficient)
    previous, results = None, []
    for i, coupling in enumerate(grid):
        path = directory / f"k{k:02d}_p{i:02d}.json"
        if path.exists():
            row = read_json(path)
        else:
            try:
                row = n48_orbit(model, coupling, k, previous)
            except ShootingFailure as exc:
                row = dict(status="no_closed_orbit_found", lambda4=coupling, period_RF=k, reason=str(exc))
            write_json(path, row)
            print(f"N48 k={k} lambda={coupling:.6f}: {row['status']} {row.get('classification', '')}", flush=True)
        if row["status"] == "closed":
            previous = row
        results.append(row)
    return results


def finite_difference(flow, state, increment):
    return np.column_stack([(flow(state + dx) - flow(state - dx)) / (2 * increment)
                            for dx in IDENTITY * increment])


def p4_orbit(directory, coefficient, reference, row, mesh):
    coupling, k = row["lambda4"], row["period_RF"]
    path = directory / f"h{mesh}_k{k:02d}_lambda{coupling:.12f}.json"
    if path.exists():
        return read_json(path)
    field = native(mesh)
    qm = Launch(coupling, 1., 0).charge_mass(coefficient)
    state = np.array(row["initial"])
    centre = state.copy()
    checks = []
    increment = 1e-8 if k == 4 else 5e-7
    try:
        for level in ("coarse", "fine"):
            control = reference[level]

            def flow(z, periods=k, start=0):
                result = field.segment(z * UNITS, qm, FREQUENCY, PHASE, start, start + periods, steps=control["steps"],
                                       rtol=control["rtol"], atol=control["atol"])
                if result["escape_RF"] is not None:
                    raise ShootingFailure("Direct P4 flow escaped before return")
                return np.array(result["final_state_SI"]) / UNITS

            def target(z):
                return ROTATE_BACK @ flow(z, 1) if k == 4 else flow(z)

            def evaluate(z):
                return target(z) - z, finite_difference(target, z, increment) - IDENTITY

            nearby = lambda z: np.linalg.norm(z - centre, ord=np.inf) < .012
            state = shoot(evaluate, state, nearby, tolerance=3e-8, iterations=5)
            closure = float(np.linalg.norm(flow(state) - state, ord=np.inf))
            if closure > 1e-7 and k == 4:
                def full(z):
                    return flow(z) - z, finite_difference(flow, z, increment) - IDENTITY

                state = shoot(full, state, nearby, tolerance=3e-8, iterations=5)
                closure = float(np.linalg.norm(flow(state) - state, ord=np.inf))
            if closure > 1e-7:
                raise ShootingFailure(f"Direct P4 closure check failed: {closure:.3g}")
            for step in ((increment,) if level == "coarse" else (increment, increment / 2)):
                if k == 4:
                    # A full-orbit difference amplifies the perturbation by
                    # roughly 7200. Compose four shorter tangent maps instead.
                    point, matrix = state, IDENTITY
                    for start in range(4):
                        local = lambda z: flow(z, 1, start)
                        matrix = finite_difference(local, point, step) @ matrix
                        point = local(point)
                else:
                    matrix = finite_difference(flow, state, step)
                info = spectrum(matrix, defect_limit=.006)
                checks.append(dict(level=level, finite_difference_step=step, closure=closure, **info))
        logs = [np.log(check["spectral_radius"]) for check in checks]
        label = checks[-1]["classification"]
        if len({check["classification"] for check in checks}) != 1 or np.ptp(logs) > .02:
            label = "numerical_unresolved"
        result = dict(status="closed", lambda4=coupling, period_RF=k, mesh_um=mesh, classification=label,
                      spectral_radius=checks[-1]["spectral_radius"], initial=state.tolist(),
                      displacement_from_N48=float(np.linalg.norm(state - centre, ord=np.inf)), checks=checks)
    except (ShootingFailure, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        result = dict(status="direct_check_unresolved", lambda4=coupling, period_RF=k, mesh_um=mesh,
                      reason=str(exc), checks=checks)
    write_json(path, result)
    print(f"P4 h{mesh} k={k} lambda={coupling:.6f}: {result['status']}", flush=True)
    return result


def selected_for_p4(rows, lo, hi, smoke):
    eligible = [r for r in rows if r["status"] == "closed" and r["classification"] != "numerical_unresolved"]
    centre = (lo + hi) / 2
    circles = [r for r in eligible if r["period_RF"] == 4]
    selected = [min(circles, key=lambda r: abs(r["lambda4"] - centre))] if circles else []
    slow = [r for r in eligible if r["period_RF"] > 4 and .7 < r["launch_radius_mm"] < 1.3]
    for target in ((centre,) if smoke else (lo, hi)):
        if slow:
            candidate = min(slow, key=lambda r: abs(r["lambda4"] - target) + .15 * abs(r["launch_radius_mm"] - 1))
            selected.append(candidate)
            slow.remove(candidate)
    return selected


def neighborhood(directory, coefficient, row, horizon):
    """Compare two nearby starts and a nominal rest launch on the same RF section."""
    coupling, k = row["lambda4"], row["period_RF"]
    path = directory / f"near_k{k:02d}_lambda{coupling:.12f}.json"
    if path.exists():
        return read_json(path)
    q = RFReconstruction.load(P4 / "h35_P4/diagnostic_N48.json").unit_quadratures
    field = PolynomialRF(q)
    state = np.array(row["initial"])
    delta = 1e-4 * np.array(row["perturbation_direction"])
    nominal = np.array([1e-3 / RADIUS, 0., 0., 0.])
    cases = (("minus", state - delta), ("plus", state + delta), ("rest_1mm", nominal))
    results = []
    for name, initial in cases:
        record = dict(case=name, lambda4=coupling, period_RF=k, horizon_RF=horizon,
                      initial_state_SI=(initial * UNITS).tolist())
        try:
            run = integrate_rf(field, initial * UNITS, Launch(coupling, 1., 0).charge_mass(coefficient),
                               1., FREQUENCY, PHASE, 0., horizon / FREQUENCY, output_samples_per_period=4,
                               max_steps_per_period=24, rtol=2e-11,
                               atol=(2e-15, 2e-15, 2e-13, 2e-13), radius=RADIUS)
            escaped = None if run.exit_time is None else run.exit_time * FREQUENCY
            record.update(status="held" if escaped is None else "escaped", escape_RF=escaped,
                          sampled_max_radius_mm=float(1e3 * np.linalg.norm(run.states[:, :2], axis=1).max()))
        except (RuntimeError, ValueError) as exc:
            record.update(status="numerical_error", reason=str(exc))
        results.append(record)
    write_json(path, results)
    return results


def parallel(function, arguments, workers):
    if not arguments:
        return []
    with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as pool:
        try:
            futures = [pool.submit(function, *args) for args in arguments]
            return [future.result() for future in as_completed(futures)]
        except KeyboardInterrupt:
            pool.terminate_workers()
            raise


def plot(directory, rows, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    markers = dict(zip(N48_PERIODS, ("D", "s", "^", "o", "P", "v"), strict=True))
    colors = {"linearly_elliptic": "#007e70", "linearly_unstable": "#b54143",
              "near_unit_or_degenerate": "#aa6c20", "numerical_unresolved": "#777777"}
    closed = [row for row in rows if row["status"] == "closed"]
    for row in closed:
        for ax, metric in zip(axes, ("radius_max_mm", "spectral_radius"), strict=True):
            ax.scatter(row["lambda4"], row[metric], c=colors[row["classification"]],
                       marker=markers[row["period_RF"]], s=29)
    for ax in axes:
        ax.axvspan(lo, hi, color="#dbe2ec", alpha=.35, zorder=0)
        ax.set_xlabel("RF coupling λ₄")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Maximum orbit radius [mm]")
    axes[1].set(yscale="log", ylabel="Largest Floquet multiplier modulus")
    if closed:
        labels = sorted({row["classification"] for row in closed})
        axes[0].legend(handles=[Line2D([], [], ls="", marker="o", color=colors[label],
                                      label=label.replace("_", " ")) for label in labels], fontsize=8)
        periods = sorted({row["period_RF"] for row in closed})
        axes[1].legend(handles=[Line2D([], [], ls="", marker=markers[k], color="#57616e",
                                      label=f"{k} RF") for k in periods], fontsize=8, ncol=2)
    fig.savefig(directory / "floquet_branches.png", dpi=220)
    plt.close(fig)


def paired_checks(candidates, checks):
    results = []
    for row in candidates:
        pair = [c for c in checks if c["lambda4"] == row["lambda4"] and c["period_RF"] == row["period_RF"]]
        complete = len(pair) == 2 and all(c["status"] == "closed" for c in pair)
        spread = None
        agrees = False
        if complete:
            spread = float(np.ptp(np.log([row["spectral_radius"], *(c["spectral_radius"] for c in pair)])))
            agrees = all(c["classification"] == row["classification"] for c in pair) and spread <= .02
        results.append(dict(lambda4=row["lambda4"], period_RF=row["period_RF"],
                            N48_classification=row["classification"], P4_both_closed=complete,
                            P4_matches_N48=agrees, log_spectral_radius_spread=spread))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--p4-workers", type=int, default=2)
    parser.add_argument("--n48-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12 or not 1 <= args.p4_workers <= 4:
        parser.error("Expected 1–12 N48 workers and 1–4 P4 workers")
    directory = (args.out or (DEFAULT.with_name("smoke_v1") if args.smoke_test else DEFAULT)).resolve()
    coefficient, controls, hashes = verify_inputs()
    anchors = OUTPUT.with_name("p4_recheck_v2") / "summary.json"
    lo, hi = sorted(r["coupling"] for r in read_json(anchors)["cases"] if r["radius_mm"] == 1)
    grid = [(lo + hi) / 2] if args.smoke_test else [.4, *np.linspace(lo, hi, 9).tolist(), .72]
    periods = (4, 24) if args.smoke_test else N48_PERIODS
    for path in (anchors, Path(__file__), ROOT / "scripts/recheck_stability_p4.py",
                 ROOT / "src/three_phase_trap/floquet_orbits.py",
                 ROOT / "src/three_phase_trap/stability_rf_p4_v2.py"):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    protocol = dict(version=1, phase_rad=PHASE, grid=grid, periods=list(periods), prior_transition=[lo, hi],
                    input_sha256=hashes, smooth_closure_tolerance=CLOSE, native_closure_tolerance=1e-7,
                    native_meshes_um=[35, 50], native_steps=["coarse", "fine"],
                    neighborhood_horizon_RF=64 if args.smoke_test else 1000)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        path = directory / "protocol.json"
        if path.exists() and read_json(path) != protocol:
            raise ValueError("Frozen pilot protocol differs; choose a new --out directory")
        if not path.exists():
            write_json(path, protocol)
        write_json(directory / "analytic_control.json", analytic_check())
        print(f"[READY] {len(grid) * len(periods)} N48 searches; {directory}", flush=True)
        if args.check_only:
            return
        smooth = directory / "N48"
        smooth.mkdir(exist_ok=True)
        batches = parallel(n48_branch, [(smooth, coefficient, grid, k) for k in periods], args.workers)
        rows = sorted((row for batch in batches for row in batch), key=lambda row: (row["period_RF"], row["lambda4"]))
        write_table(directory / "n48_orbits.csv", [{k: v for k, v in row.items() if k != "samples"} for row in rows])
        plot(directory, rows, lo, hi)
        candidates = selected_for_p4(rows, lo, hi, args.smoke_test)
        near = parallel(neighborhood, [(directory, coefficient, row, protocol["neighborhood_horizon_RF"])
                                       for row in candidates if row["period_RF"] > 4], args.workers)
        if near:
            write_table(directory / "neighborhood_checks.csv", [item for group in near for item in group])
        if not args.n48_only:
            direct = directory / "P4"
            direct.mkdir(exist_ok=True)
            checks = parallel(p4_orbit, [(direct, coefficient, controls, row, mesh)
                                        for row in candidates for mesh in (35, 50)], args.p4_workers)
            checks.sort(key=lambda row: (row["period_RF"], row["lambda4"], row["mesh_um"]))
            if checks:
                write_table(directory / "p4_orbit_checks.csv", checks)
            write_json(directory / "summary.json", dict(
                n48_searched=len(rows), n48_closed=sum(row["status"] == "closed" for row in rows), P4=checks,
                paired_checks=paired_checks(candidates, checks),
                scope="Local linear spectra for selected periodic orbits; capture map is a separate observable"))
        print("[COMPLETE] " + str(directory), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Repeat the same command to reuse completed orbit searches.", flush=True)
        sys.exit(130)
