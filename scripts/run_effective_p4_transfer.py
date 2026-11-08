"""Run or resume the preregistered radius/azimuth/helicity P4 transfer test."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
import math
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from build_direct_fem import OUTPUT as FIELD_CACHE, input_signature, load_field
from run_effective_reconstruction_v1 import (
    DEFAULT_OUTPUT, LIMITS, P4, ROOT, compare, diagnostics, filters, read_json,
    velocity_pair, write_table,
)
from three_phase_trap.averaged_dynamics import AveragedRFModel
from three_phase_trap.effective_reconstruction_v1 import LENGTH, RadialPower, integrate_effective, turning_data
from three_phase_trap.long_rf_run import RunStore, is_running, run_case, run_lock, sha, write_json
from three_phase_trap.micromotion import matched_full_state_with_drift
from three_phase_trap.octupole_model import OctupoleModel
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction
from three_phase_trap.trajectory_diagnostics import position_errors


OUTPUT = ROOT / "outputs/effective_reconstruction/p4_transfer/v1"
LAUNCHES = ((.75, .23, .7), (1.25, .31, 1.1))  # mm, launch azimuth, RF phase
MESHES = (35, 50)
LEVELS = ("fine", "coarse")
HORIZON = 3.25
RATE_FRACTION = .05
NUMERICAL_LIMITS = {name: value / 10 for name, value in LIMITS.items()}
SWEEP_HZ = (80000, 100000, 140000)


def reverse_native(field):
    """Swap the B/C electrode drives: chi -> pi - chi on the native P4 field."""
    result = copy(field)
    result.gx = field.gx.conj()
    result.gy = field.gy.conj()
    result.potential_coefficients = field.potential_coefficients.conj()
    return result


def reverse_reconstruction(reconstruction):
    q = reconstruction.unit_quadratures
    swapped = HarmonicQuadratures(-q.cosine, q.sine, q.reference_radius)
    return RFReconstruction(swapped, reconstruction.valid_radius, {"drive": "ACB"})


def drive_models(physical):
    reconstruction = RFReconstruction.load(P4 / "h35_P4/diagnostic_N48.json")
    parameters = [physical[key] for key in ("charge_C", "mass_kg", "U0_V", "frequency_Hz")]
    return {1: AveragedRFModel(reconstruction, *parameters),
            -1: AveragedRFModel(reverse_reconstruction(reconstruction), *parameters)}


def freeze(path, data):
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f"Frozen experiment differs: {path}. Use a new output directory.")
    else:
        write_json(path, data)


def preflight(smoke):
    reference_path = P4 / "validation_settings.json"
    frozen_path = DEFAULT_OUTPUT / "frozen_models.json"
    calibration_path = P4 / "h35_P4/native_C_calibration.json"
    reconstruction_path = P4 / "h35_P4/diagnostic_N48.json"
    reference, fitted = read_json(reference_path), read_json(frozen_path)
    for manifest_path, files in (
        (DEFAULT_OUTPUT / "delivery_sha256.json", (frozen_path,)),
        (P4 / "manifest.json", (reference_path, calibration_path, reconstruction_path)),
    ):
        archived = read_json(manifest_path)
        archived = archived.get("artifacts", archived)
        for path in files:
            if sha(path) != archived[path.relative_to(ROOT).as_posix()]:
                raise ValueError(f"Archived input checksum mismatch: {path}")
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    if versions != reference["versions"]:
        raise ValueError(f"Runtime differs from verified P4: {versions}; expected {reference['versions']}")
    paths = {reference_path, frozen_path, calibration_path, reconstruction_path,
             Path(__file__).resolve(), ROOT / "scripts/build_direct_fem.py",
             ROOT / "scripts/run_effective_reconstruction_v1.py"}
    package = ROOT / "src/three_phase_trap"
    for name, expected in reference["kernel_sha256"].items():
        path = package / name
        if sha(path) != expected:
            raise ValueError(f"Verified numerical kernel changed: {path}")
        paths.add(path)
    paths.update(package / name for name in ("long_rf_run.py", "effective_reconstruction_v1.py", "octupole_model.py",
                                             "rf_reconstruction.py", "averaged_dynamics.py", "micromotion.py",
                                             "slow_analysis.py", "trajectory_diagnostics.py"))
    for mesh in ((50,) if smoke else MESHES):
        directory = FIELD_CACHE / f"h{mesh}_P4"
        complete = directory / "complete.json"
        meta = read_json(complete)
        if meta["signature"] != input_signature(mesh, 4):
            raise ValueError(f"Field signature mismatch: {directory}")
        paths.add(complete)
        for name, expected in meta["artifacts"].items():
            if sha(directory / name) != expected:
                raise ValueError(f"Field checksum mismatch: {directory / name}")
            paths.add(directory / name)
        paths.update(ROOT / f"outputs/meshes/gmsh_{mesh}um" / name for name in ("trap.msh", "sector.msh"))
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}

    p = {key: reference["physical"][key] for key in ("mass_kg", "charge_C", "gamma", "gravity", "Udc_V", "U0_V", "frequency_Hz")}
    models = drive_models(p)
    power = RadialPower(fitted["C"]["p"], fitted["C"]["U_at_1mm_J"])
    launches, runs = {}, []
    for radius, azimuth, phase in (LAUNCHES[:1] if smoke else LAUNCHES):
        direction = np.array([np.cos(azimuth), np.sin(azimuth)])
        initial = np.r_[radius * LENGTH * direction, 0., 0.]
        period = float(power.period(radius * LENGTH, p["mass_kg"]))
        cycles = 64 if smoke else math.ceil(HORIZON * period * p["frequency_Hz"])
        for sign, order in ((1, "ABC"), (-1, "ACB")):
            key = f"r{round(100 * radius):03d}_{order}"
            b = models[sign]
            full = matched_full_state_with_drift(b.quadratures, initial[:2], initial[2:],
                                                 b.charge, b.mass, b.omega, phase)
            launch = dict(case=key, radius_mm=radius, azimuth_rad=azimuth, phase_rad=phase,
                          helicity=sign, sequence=order, direction=direction.tolist(), initial_state=full.tolist(),
                          initial_slow_SI=initial.tolist(), reference_slow_period_s=period, RF_periods=cycles)
            launches[key] = launch
            for mesh in ((50,) if smoke else MESHES):
                for level in (("coarse",) if smoke else LEVELS):
                    name = f"{key}_h{mesh}_{level}"
                    runs.append(dict(**launch, run=name, mesh_um=mesh, level=level,
                                     physical=p, R_scale_m=LENGTH, control=reference["controls"][level],
                                     samples_per_RF_period=64, chunk_RF_periods=32 if smoke else 1500,
                                     smoke_test=smoke, versions=versions, input_sha256=hashes))
    config = dict(schema=1, physical=p, launches=launches, runs=[r["run"] for r in runs], input_sha256=hashes,
                  slow_periods=None if smoke else HORIZON, C=fitted["C"], E_gamma_mm_s_inv=fitted["E_gamma_mm_s_inv"],
                  rate_budget_fraction=RATE_FRACTION, physical_limits_pct=LIMITS,
                  numerical_limits_pct=NUMERICAL_LIMITS, frequency_sweep_Hz=list(SWEEP_HZ),
                  drive_rule="ABC: native field; ACB: conjugate native complex quadratures, same phase and azimuth")
    return config, runs, models, power


def run_native(directory, settings):
    name = settings["run"]
    store = RunStore(directory / "runs" / name, settings)

    def field_factory():
        field = load_field(settings["mesh_um"], 4, settings["physical"]["U0_V"])
        return field if settings["helicity"] == 1 else reverse_native(field)

    run_case(store, field_factory, notify=lambda line: print(f"{name} {line}", flush=True), make_figures=False)
    return name


def run_batch(directory, runs, workers):
    failures = []
    # Start longer, fine cases first. All independent cases are attempted.
    ordered = sorted(runs, key=lambda s: s["RF_periods"] * s["control"]["steps"], reverse=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        try:
            futures = {pool.submit(run_native, directory, settings): settings["run"] for settings in ordered}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    name = futures[future]
                    print(f"[ERROR] {name}: {exc}", flush=True)
                    failures.append(name)
        except KeyboardInterrupt:
            pool.terminate_workers()
            raise
    if failures:
        raise RuntimeError(f"Failed runs: {', '.join(failures)}. See error.txt; rerun to resume.")


def model_predictions(directory, config, models, power):
    physical = config["physical"]
    mass, frequency = physical["mass_kg"], physical["frequency_Hz"]
    predictions = {}
    rows = []
    for name, launch in config["launches"].items():
        full_clock = np.arange(launch["RF_periods"] + 1) / frequency
        clock = full_clock[1:-1]  # same once-per-RF nodes as v1's centered mean
        gamma = launch["helicity"] * config["E_gamma_mm_s_inv"]
        initial = launch["initial_slow_SI"]
        prediction = {
            "B": integrate_effective(full_clock, initial, mass, power, fem_model=models[1])[1:-1],
            "E": integrate_effective(full_clock, initial, mass, power, gamma_mm=gamma)[1:-1],
        }
        for model, states in prediction.items():
            result, _ = turning_data(clock, states, launch["reference_slow_period_s"])
            rows.append(dict(case=name, model=model, gamma_mm_s_inv=gamma if model == "E" else 0.,
                             **result, rate_deg_per_period=result["fitted_rate_deg_per_s"] * result["period_s"]))
        predictions[name] = (clock, prediction)
    path = directory / "predictions.csv"
    candidate = path.with_suffix(".tmp")
    write_table(candidate, rows)
    if path.exists():
        if sha(candidate) != sha(path):
            raise ValueError(f"Frozen forecasts differ: {path}")
        candidate.unlink()
    else:
        candidate.replace(path)
    return predictions, rows


def convergence_row(name, kind, first, second, budget):
    t, x, d = first
    u, y, e = second
    if len(t) != len(u) or not np.allclose(t, u, rtol=0, atol=1e-13):
        raise ValueError(f"Control changed RF output clock: {name}")
    delta = position_errors(x, y, LENGTH)["vector"]
    drift_delta = abs(d["fitted_rate_deg_per_s"] - e["fitted_rate_deg_per_s"])
    error = dict(rms_pct=100 * np.sqrt(np.mean(delta**2)), max_pct=100 * max(delta),
                 period_error_pct=100 * abs(d["period_s"] / e["period_s"] - 1))
    return dict(case=name, control=kind, **error, angle_rate_difference_deg_s=drift_delta,
                angle_rate_budget_deg_s=budget,
                same_sign=np.sign(d["fitted_rate_deg_per_s"]) == np.sign(e["fitted_rate_deg_per_s"]),
                passed=all(error[k] <= limit for k, limit in NUMERICAL_LIMITS.items()) and drift_delta <= budget
                and np.sign(d["fitted_rate_deg_per_s"]) == np.sign(e["fitted_rate_deg_per_s"]))


def analyze(directory, runs, models, predictions, forecasts):
    analysis = directory / "analysis"
    analysis.mkdir(exist_ok=True)
    measured, slow, comparisons, controls, verdicts = {}, [], [], [], []
    for settings in runs:
        label = settings["run"]
        with np.load(directory / "runs" / label / "trajectory.npz") as arrays:
            series = filters(arrays["time_s"], arrays["states"], models[settings["helicity"]], settings["phase_rad"], 64)
        observed = diagnostics(analysis, label, series, settings["reference_slow_period_s"], slow)
        measured[label] = (*series["mean"], observed)
    for name, (clock, prediction) in predictions.items():
        t, actual, observed = measured[f"{name}_h35_fine"]
        if len(t) != len(clock) or not np.allclose(t, clock, rtol=0, atol=1e-13):
            raise ValueError(f"Prediction and native trajectories use different clocks: {name}")
        expected = next(row["fitted_rate_deg_per_s"] for row in forecasts if row["case"] == name and row["model"] == "E")
        rate = observed["fitted_rate_deg_per_s"]
        budget = RATE_FRACTION * abs(expected)
        case_controls = []
        for kind, one, two in (
            ("time_h35", "h35_coarse", "h35_fine"), ("time_h50", "h50_coarse", "h50_fine"),
            ("space_fine", "h50_fine", "h35_fine"), ("space_coarse", "h50_coarse", "h35_coarse"),
        ):
            case_controls.append(convergence_row(name, kind, measured[f"{name}_{one}"], measured[f"{name}_{two}"], budget))
        controls.extend(case_controls)
        axis = np.array(next(s["direction"] for s in runs if s["case"] == name))
        b, e = compare(analysis, name, t, actual, prediction, observed, axis)
        comparisons.extend((b, e))
        numerical_rms = max(row["rms_pct"] for row in case_controls)
        numerical_rate = max(row["angle_rate_difference_deg_s"] for row in case_controls)
        rate_improvement = abs(b["orientation_rate_error_deg_s"]) - abs(e["orientation_rate_error_deg_s"])
        verdicts.append(dict(case=name, converged=all(row["passed"] for row in case_controls),
                             measured_rate_deg_s=rate, predicted_rate_deg_s=expected, rate_budget_deg_s=budget,
                             rate_agrees=bool(abs(rate - expected) <= budget and np.sign(rate) == np.sign(expected)),
                             B_passes=b["passes"], E_passes=e["passes"],
                             E_improves_RMS_over_B_above_resolution=bool(b["rms_pct"] - e["rms_pct"] > 2 * numerical_rms),
                             E_improves_rate_over_B_above_resolution=bool(rate_improvement > 2 * numerical_rate)))
    for row in verdicts:
        if not row["converged"]:
            row["verdict"] = "inconclusive_numerical_controls"
        elif all(row[key] for key in ("rate_agrees", "E_passes", "E_improves_RMS_over_B_above_resolution",
                                     "E_improves_rate_over_B_above_resolution")):
            row["verdict"] = "supports_E"
        else:
            row["verdict"] = "E_criteria_not_met"
    write_table(analysis / "slow_diagnostics.csv", slow)
    write_table(analysis / "comparisons.csv", comparisons)
    write_table(analysis / "numerical_controls.csv", controls)
    write_table(analysis / "verdicts.csv", verdicts)
    return verdicts


def frequency_sweep(directory, config):
    """Independently measure the velocity-odd acceleration at fixed U0 and launch."""
    from csv import DictReader

    directory.mkdir(exist_ok=True)
    calibration = read_json(P4 / "h35_P4/native_C_calibration.json")
    p = config["physical"]
    rows = []
    for frequency in SWEEP_HZ:
        target = directory / f"f{frequency}"
        target.mkdir(exist_ok=True)
        path = target / "velocity_pair.csv"
        completed = target / "COMPLETED.json"
        if not completed.exists():
            model = OctupoleModel(calibration["C4_per_volt_per_m4"], calibration["phase_rad"],
                                  p["charge_C"], p["mass_kg"], p["U0_V"], frequency, .008)
            velocity_pair(target, model)
            write_json(completed, {file.name: sha(file) for file in sorted(target.glob("velocity_pair*"))})
        for name, digest in read_json(completed).items():
            if sha(target / name) != digest:
                raise ValueError(f"Analytic RF artifact checksum mismatch: {target / name}")
        with path.open(encoding="utf-8", newline="") as stream:
            estimates = [float(row["gamma_from_pair_s_inv"]) for row in DictReader(stream) if row["gamma_from_pair_s_inv"]]
        if len(estimates) != 4 or min(estimates) <= 0 or not np.all(np.isfinite(estimates)):
            raise ValueError(f"Unexpected velocity-pair results: {path}")
        predicted = config["E_gamma_mm_s_inv"] * (p["frequency_Hz"] / frequency)**3
        rows.append(dict(frequency_Hz=frequency, predicted_s_inv=predicted, measured_s_inv=np.mean(estimates),
                         measured_min_s_inv=min(estimates), measured_max_s_inv=max(estimates),
                         max_relative_error=max(abs(np.array(estimates) / predicted - 1))))
    slope, _ = np.polyfit(np.log(SWEEP_HZ), np.log([row["measured_s_inv"] for row in rows]), 1)
    write_table(directory / "frequency_scaling.csv", rows)
    return dict(fitted_frequency_exponent=float(slope), predicted_exponent=-3,
                max_relative_gamma_error=float(max(row["max_relative_error"] for row in rows)))


def show_status(directory):
    config = directory / "protocol.json"
    if not config.exists():
        print("Experiment has not started.")
        return
    for name in read_json(config)["runs"]:
        folder = directory / "runs" / name
        if (folder / "COMPLETED.json").exists():
            print(f"{name}: completed")
        elif (folder / "status.json").exists():
            status = read_json(folder / "status.json")
            stage = status["status"] if is_running(folder) or status["status"] == "failed" else "interrupted"
            print(f"{name}: {stage} {status['progress_percent']:.2f}% "
                  f"({status['committed_segments']}/{status['total_segments']} checkpoints)")
        else:
            print(f"{name}: pending")
    if (directory / "summary.json").exists():
        print("Comparison and frequency sweep completed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--smoke-test", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    directory = args.out or (OUTPUT.with_name("smoke_v1") if args.smoke_test else OUTPUT)
    directory = directory.resolve()
    if args.status:
        show_status(directory)
        return
    config, runs, models, power = preflight(args.smoke_test)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        freeze(directory / "protocol.json", config)
        print(f"[READY] {len(runs)} native P4 runs; output: {directory}", flush=True)
        for name, launch in config["launches"].items():
            print(f"{name}: {launch['RF_periods']} RF periods, azimuth={launch['azimuth_rad']} rad", flush=True)
        if not args.smoke_test:
            predictions, forecasts = model_predictions(directory, config, models, power)
        if args.check_only:
            return
        run_batch(directory, runs, args.workers)
        if not args.smoke_test:
            verdicts = analyze(directory, runs, models, predictions, forecasts)
            sweep = frequency_sweep(directory / "frequency_sweep", config)
            rates = {row["case"]: row["measured_rate_deg_s"] for row in verdicts}
            pairs = []
            for abc in (key for key in rates if key.endswith("_ABC")):
                acb = abc.removesuffix("_ABC") + "_ACB"
                pairs.append(dict(radius_mm=config["launches"][abc]["radius_mm"],
                                  odd_rate_deg_s=(rates[abc] - rates[acb]) / 2,
                                  even_rate_deg_s=(rates[abc] + rates[acb]) / 2,
                                  sign_reversed=bool(rates[abc] * rates[acb] < 0)))
            write_json(directory / "summary.json", dict(cases=verdicts, helicity_pairs=pairs, frequency_sweep=sweep))
            print(f"[COMPLETE] Comparison and numerical controls: {directory / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Run the same command to resume saved checkpoints.", flush=True)
        raise SystemExit(130)
