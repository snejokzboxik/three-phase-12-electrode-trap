"""Finite-time survival map for the ideal RF trap, with native P4 controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import platform
import sys
import time

# Each process integrates a single 4D ODE. Avoid nested BLAS thread pools.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy

from build_direct_fem import OUTPUT as FIELD_CACHE, input_signature, load_field
from run_effective_reconstruction_v1 import P4, ROOT, read_json, write_table
from three_phase_trap.long_rf_run import run_lock, sha, write_json
from three_phase_trap.pilot_dynamics import PolynomialRF, integrate_rf
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.stability_rf_p4 import NativeEscape


OUTPUT = ROOT / "outputs/stability_p4/v1"
RADIUS = .008
FREQUENCY = 100_000.
MASS = 1e-15
CHARGE = 1e-12
RADII_MM = (.5, .75, 1., 1.25, 2.)
REST = tuple((angle, phase, 0.) for angle in (0., math.pi / 8) for phase in (0., math.pi / 2))
VELOCITY = ((math.pi / 8, 0., -.02), (math.pi / 8, 0., .02))
PROFILES = {
    "map": dict(grid=13, stages=[1000, 10000], p4_cases=6, refine=True),
    "pilot": dict(grid=9, stages=[1000, 10000], p4_cases=4, refine=True),
    "smoke": dict(grid=3, stages=[32, 64], p4_cases=2, refine=True),
    "benchmark": dict(grid=8, stages=[750], p4_cases=0, refine=False),
}
SMOOTH_STEPS = 12
_field = None
_native = None
_coefficient = None


@dataclass(frozen=True)
class Launch:
    coupling: float
    radius_mm: float
    start: int

    @property
    def tag(self):
        return f"r{self.radius_mm:g}_k{self.coupling:.13e}_i{self.start}"

    def state(self):
        angle, phase, velocity = (REST + VELOCITY)[self.start]
        co, si = math.cos(angle), math.sin(angle)
        radius = self.radius_mm * 1e-3
        return np.array((radius * co, radius * si, -velocity * si, velocity * co)), phase

    def charge_mass(self, coefficient):
        return self.coupling * (2 * np.pi * FREQUENCY)**2 / (coefficient * RADIUS**2)


def verify_inputs():
    reference = read_json(P4 / "validation_settings.json")
    runtime = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    if runtime != reference["versions"]:
        raise ValueError(f"Use the verified P4 runtime: {reference['versions']}; found {runtime}")
    paths = {Path(__file__), ROOT / "scripts/build_direct_fem.py",
             ROOT / "scripts/run_effective_reconstruction_v1.py",
             ROOT / "src/three_phase_trap/stability_rf_p4.py",
             ROOT / "src/three_phase_trap/long_rf_run.py", ROOT / "src/three_phase_trap/rf_reconstruction.py",
             P4 / "validation_settings.json"}
    manifest = read_json(P4 / "manifest.json")["artifacts"]
    for name in ("diagnostic_N48.json", "native_C_calibration.json"):
        path = P4 / "h35_P4" / name
        if sha(path) != manifest[path.relative_to(ROOT).as_posix()]:
            raise ValueError(f"Archived P4 input changed: {path}")
        paths.add(path)
    for name, expected in reference["kernel_sha256"].items():
        path = ROOT / "src/three_phase_trap" / name
        if sha(path) != expected:
            raise ValueError(f"Frozen P4 kernel changed: {path}")
        paths.add(path)
    for mesh in (35, 50):
        directory = FIELD_CACHE / f"h{mesh}_P4"
        complete = directory / "complete.json"
        meta = read_json(complete)
        if meta["signature"] != input_signature(mesh, 4):
            raise ValueError(f"P4 cache signature differs: {directory}")
        paths.add(complete)
        for name, expected in meta["artifacts"].items():
            path = directory / name
            if sha(path) != expected:
                raise ValueError(f"P4 cache checksum mismatch: {path}")
            paths.add(path)
        paths.update(ROOT / f"outputs/meshes/gmsh_{mesh}um" / name for name in ("trap.msh", "sector.msh"))
    coefficient = read_json(P4 / "h35_P4/native_C_calibration.json")["C4_per_volt_per_m4"]
    return coefficient, reference["controls"], {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def init_worker(model, coefficient, mesh=None):
    global _field, _native, _coefficient
    _coefficient = coefficient
    if model == "N48":
        reconstruction = RFReconstruction.load(P4 / "h35_P4/diagnostic_N48.json")
        _field = PolynomialRF(reconstruction.unit_quadratures)
    else:
        _native = NativeEscape(load_field(mesh, 4, amplitude=1.), RADIUS)


def integrate_job(launch, horizon, control, previous, model):
    started = time.perf_counter()
    try:
        initial, phase = launch.state()
        offset = previous["horizon_RF"] if previous else 0
        if previous:
            initial = np.array(previous["final_state_SI"])
        remaining = horizon - offset
        qm = launch.charge_mass(_coefficient)
        if model == "N48":
            result = integrate_rf(_field, initial, qm, 1., FREQUENCY, phase, 0., remaining / FREQUENCY,
                                  max_steps_per_period=control["steps"], rtol=control["rtol"], atol=control["atol"],
                                  output_samples_per_period=1, radius=RADIUS)
            exit_rf = None if result.exit_time is None else result.exit_time * FREQUENCY
            state = result.states[-1]
            maximum = float(np.max(np.linalg.norm(result.states[:, :2], axis=1)))
            nfev = result.nfev
        else:
            exit_rf, state, maximum, nfev = _native.run(
                initial, qm, FREQUENCY, phase, remaining, steps=control["steps"],
                rtol=control["rtol"], atol=control["atol"])
        return dict(**asdict(launch), horizon_RF=horizon, status="held" if exit_rf is None else "escaped",
                    escape_RF=None if exit_rf is None else offset + exit_rf, final_state_SI=state.tolist(),
                    sampled_max_radius_m=max(maximum, previous["sampled_max_radius_m"] if previous else 0.),
                    nfev=nfev, integrated_RF=remaining if exit_rf is None else exit_rf,
                    wall_s=time.perf_counter() - started)
    except Exception as exc:
        return dict(**asdict(launch), horizon_RF=horizon, status="numerical_error",
                    error=f"{type(exc).__name__}: {exc}", escape_RF=None, integrated_RF=0., nfev=0,
                    wall_s=time.perf_counter() - started)


def compute(directory, jobs, model, control, workers, coefficient, mesh=None):
    """One atomic checkpoint per stage of each trajectory; restart skips those files."""
    directory.mkdir(parents=True, exist_ok=True)
    outcomes, pending = {}, []
    for launch, horizon, previous in jobs:
        path = directory / (launch.tag + ".json")
        if path.exists():
            row = read_json(path)
            if row["horizon_RF"] != horizon or any(row[k] != v for k, v in asdict(launch).items()):
                raise ValueError(f"Checkpoint differs from scan protocol: {path}")
            outcomes[launch] = row
        elif previous and previous["status"] == "escaped":
            row = {**previous, "horizon_RF": horizon, "integrated_RF": 0., "wall_s": 0., "nfev": 0}
            write_json(path, row)
            outcomes[launch] = row
        else:
            # A longer stage can retry a numerical error from t=0.
            pending.append((launch, horizon, previous if previous and previous["status"] == "held" else None))
    label = f"{directory.parent.name}/{directory.name}"
    if pending:
        print(f"[{label}] {len(pending)} pending, {len(outcomes)} reused/early escapes", flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)), initializer=init_worker,
                                 initargs=(model, coefficient, mesh)) as pool:
            try:
                futures = {pool.submit(integrate_job, launch, horizon, control, prev, model): launch
                           for launch, horizon, prev in pending}
                for index, future in enumerate(as_completed(futures), 1):
                    launch = futures[future]
                    row = future.result()
                    write_json(directory / (launch.tag + ".json"), row)
                    outcomes[launch] = row
                    if index % 10 == 0 or index == len(pending) or model == "P4":
                        print(f"[{label}] {index}/{len(pending)} new results saved", flush=True)
            except KeyboardInterrupt:
                pool.terminate_workers()
                raise
    return outcomes


def grid_rows(outcomes):
    groups = {}
    for launch, result in outcomes.items():
        groups.setdefault((launch.radius_mm, launch.coupling), []).append(result)
    for (radius, coupling), group in sorted(groups.items()):
        held = sum(row["status"] == "held" for row in group)
        escaped = sum(row["status"] == "escaped" for row in group)
        yield dict(radius_mm=radius, coupling=coupling, horizon_RF=group[0]["horizon_RF"],
                   held=held, escaped=escaped, unresolved=len(group) - held - escaped,
                   survival_lower=held / len(group), survival_upper=1 - escaped / len(group))


def transition_intervals(outcomes):
    """Adjacent coarse points with resolved opposite outcomes for the same launch."""
    rays = {}
    for launch, row in outcomes.items():
        rays.setdefault((launch.radius_mm, launch.start), []).append((launch, row["status"]))
    for ray in rays.values():
        ray.sort(key=lambda item: item[0].coupling)
        for (lo, left), (hi, right) in zip(ray, ray[1:]):
            if {left, right} == {"held", "escaped"}:
                yield lo, hi


def refinement(outcomes, starts):
    # At most one midpoint per radius, picked from the first resolved transition.
    chosen = {}
    for lo, hi in sorted(transition_intervals(outcomes), key=lambda pair: (pair[0].radius_mm, pair[0].coupling)):
        chosen.setdefault(lo.radius_mm, math.sqrt(lo.coupling * hi.coupling))
    return [Launch(coupling, radius, i) for radius, coupling in chosen.items() for i in range(starts)]


def candidates(outcomes, cap):
    if not cap:
        return []
    pairs = {}
    for lo, hi in sorted(transition_intervals(outcomes), key=lambda p: (p[0].radius_mm, p[0].coupling, p[0].start)):
        pairs.setdefault(lo.radius_mm, (lo, hi))
    radii = sorted(pairs)
    count = min(cap // 2, len(radii))
    indices = np.linspace(0, len(radii) - 1, count).round().astype(int)
    selected = [launch for i in indices for launch in pairs[radii[i]]]
    if not selected:
        # If no boundary is bracketed, use at most one example of each class.
        for status in ("held", "escaped"):
            pool = [launch for launch, row in outcomes.items() if row["status"] == status]
            if pool:
                selected.append(max(pool, key=lambda launch: (launch.radius_mm, launch.coupling, -launch.start)))
    return selected


def agreements(n48, controls):
    statuses = {row["status"] for row in [n48, *controls]}
    if "numerical_error" in statuses:
        return "numerical_error", None
    if len(statuses) != 1:
        return "classification_disagrees", None
    if statuses == {"held"}:
        return "agrees_held", None
    times = [row["escape_RF"] for row in [n48, *controls]]
    spread = max(times) - min(times)
    return ("agrees_escaped" if spread <= max(1., .05 * min(times)) else "escape_time_disagrees"), spread


def plot_map(path, rows, horizon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.3, 4.5), layout="constrained")
    resolved = [r for r in rows if not r["unresolved"]]
    uncertain = [r for r in rows if r["unresolved"]]
    if resolved:
        img = ax.scatter([r["coupling"] for r in resolved], [r["radius_mm"] for r in resolved],
                         c=[r["survival_lower"] for r in resolved], vmin=0, vmax=1, cmap="viridis",
                         s=82, marker="s", edgecolors="#343434", linewidths=.35)
        fig.colorbar(img, ax=ax, label="Fraction held")
    if uncertain:
        ax.scatter([r["coupling"] for r in uncertain], [r["radius_mm"] for r in uncertain],
                   marker="x", color="crimson", label="Numerical result unresolved")
        ax.legend()
    ax.set(xscale="log", xlabel=r"Coupling $\lambda_4=|q|C_4R_{\rm esc}^2/(m\Omega^2)$",
           ylabel="Launch radius [mm]", title=f"N48 (P4): survival to {horizon:,} RF periods")
    ax.grid(alpha=.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def study(directory, profile_name, workers, native_workers, check_only):
    coefficient, reference_controls, hashes = verify_inputs()
    profile = PROFILES[profile_name]
    base = CHARGE / MASS * coefficient * 250. * RADIUS**2 / (2 * np.pi * FREQUENCY)**2
    if profile_name == "benchmark":
        couplings = list(base * np.geomspace(.75, 1.5, profile["grid"]))
        radii, starts = (1.,), 4
    elif profile_name == "smoke":
        couplings, radii, starts = [base, base * 400, base * 16000], (2.,), 4
    else:
        couplings = list(base * np.geomspace(1., 16000., profile["grid"]))
        radii, starts = RADII_MM, 6 if profile_name == "map" else 4
    smooth = dict(steps=SMOOTH_STEPS, rtol=2e-9, atol=[2e-13, 2e-13, 2e-11, 2e-11])
    strict = dict(steps=2 * SMOOTH_STEPS, rtol=2e-11, atol=[2e-15, 2e-15, 2e-13, 2e-13])
    protocol = dict(schema=1, profile=profile_name, settings=profile, input_sha256=hashes,
                    C4_per_volt_m4=coefficient, reference_frequency_Hz=FREQUENCY, aperture_m=RADIUS,
                    couplings=couplings, radii_mm=list(radii), full_starts=[list(start) for start in (REST + VELOCITY)[:starts]],
                    N48=smooth, N48_strict=strict, native={level: reference_controls[level] for level in ("coarse", "fine")})
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        path = directory / "protocol.json"
        if path.exists() and read_json(path) != protocol:
            raise ValueError("Saved protocol differs; use --out with a new directory")
        if not path.exists():
            write_json(path, protocol)
        launch_list = [Launch(k, r, i) for r in radii for k in couplings for i in range(starts)]
        print(f"[READY] {len(launch_list)} coarse trajectories; horizons {profile['stages']}; "
              f"workers N48/P4 = {workers}/{native_workers}; {directory}", flush=True)
        if check_only:
            return
        started = time.perf_counter()
        horizons = profile["stages"]
        previous = {}
        for horizon in horizons:
            jobs = [(launch, horizon, previous.get(launch)) for launch in launch_list]
            previous = compute(directory / "N48" / f"RF{horizon}", jobs, "N48", smooth, workers, coefficient)
            rows = list(grid_rows(previous))
            write_table(directory / f"map_RF{horizon}.csv", rows)
            plot_map(directory / f"map_RF{horizon}.png", rows, horizon)
        coarse = previous
        horizon = horizons[-1]
        extra = refinement(coarse, starts) if profile["refine"] else []
        refined = compute(directory / "N48" / "refined", [(launch, horizon, None) for launch in extra],
                          "N48", smooth, workers, coefficient)
        combined = {**coarse, **refined}
        write_table(directory / "n48_trajectories.csv", [combined[launch] for launch in sorted(combined, key=lambda l: l.tag)])
        rows = list(grid_rows(combined))
        write_table(directory / "map_refined.csv", rows)
        plot_map(directory / "map_refined.png", rows, horizon)
        smooth_wall = time.perf_counter() - started

        chosen = candidates(combined, profile["p4_cases"])
        jobs = [(launch, horizon, None) for launch in chosen]
        control_results = {}
        if chosen:
            control_results["N48_strict"] = compute(directory / "controls" / "N48_strict", jobs, "N48", strict,
                                                   workers, coefficient)
            for mesh in (35, 50):
                for level in ("coarse", "fine"):
                    label = f"P4_h{mesh}_{level}"
                    control = reference_controls[level]
                    control_results[label] = compute(directory / "controls" / label, jobs, "P4", control,
                                                     native_workers, coefficient, mesh)
        reports = []
        for launch in chosen:
            controls = [rows[launch] for rows in control_results.values()]
            status, spread = agreements(combined[launch], controls)
            report = dict(**asdict(launch), horizon_RF=horizon, agreement=status, escape_spread_RF=spread,
                          N48_status=combined[launch]["status"], N48_escape_RF=combined[launch]["escape_RF"])
            for label, result in zip(control_results, controls, strict=True):
                report[label + "_status"] = result["status"]
                report[label + "_escape_RF"] = result["escape_RF"]
                report[label + "_error"] = result.get("error")
            reports.append(report)
        if reports:
            write_table(directory / "p4_controls.csv", reports)
        summary = dict(profile=profile_name, final_horizon_RF=horizon, coarse_trajectories=len(coarse),
                       refined_trajectories=len(refined), n48_counts={status: sum(r["status"] == status for r in combined.values())
                                                                    for status in ("held", "escaped", "numerical_error")},
                       n48_wall_this_run_s=smooth_wall, total_wall_this_run_s=time.perf_counter() - started,
                       controls=reports, scope="Only the listed P4 controls have independent mesh/time comparisons")
        write_json(directory / "summary.json", summary)
        print(f"[COMPLETE] {directory / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="map")
    parser.add_argument("--workers", type=int, default=min(8, os.process_cpu_count() or 1))
    parser.add_argument("--p4-workers", type=int, default=2)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.p4_workers <= 8:
        parser.error("Expected 1–32 N48 workers and 1–8 P4 workers")
    default = OUTPUT if args.profile == "map" else OUTPUT.with_name(args.profile + "_v1")
    study((args.out or default).resolve(), args.profile, args.workers, args.p4_workers, args.check_only)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Repeat the same command to reuse saved trajectories.", flush=True)
        sys.exit(130)
