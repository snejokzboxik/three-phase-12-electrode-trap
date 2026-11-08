"""Reproduce the v1 study in a new output directory. No FEM runs or writes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.averaged_dynamics import AveragedRFModel
from three_phase_trap.effective_reconstruction_v1 import (
    LENGTH, RadialPower, angular_momentum_fit, check_split, fit_power, integrate_effective,
    octupole_gamma, turning_data,
)
from three_phase_trap.micromotion import matched_full_state_with_drift, recover_slow_state_with_drift
from three_phase_trap.octupole_model import OctupoleModel
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.slow_analysis import centered_period_mean
from three_phase_trap.trajectory_diagnostics import position_errors


P4 = ROOT / "outputs/abc_validation/direct_fem_validation/v2"
DEFAULT_OUTPUT = ROOT / "outputs/effective_reconstruction/v1"
RF_TRAIN = ("p4_phase0",)
RF_TEST = ("p4_phase1",)
C_TRAIN = (.5, 1., 1.25)
C_TEST = (.75, .9, 1.1)
ANALYTIC_TEST = (
    ("oct075", .75, .23, .7, 1),
    ("oct100", 1., 0., 0., 1),
    ("oct125", 1.25, .31, 1.1, 1),
    ("oct125_reverse", 1.25, -.31, 1.1, -1),
)
LIMITS = dict(rms_pct=1., max_pct=3., period_error_pct=1.)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def power_row(model):
    return dict(p=model.exponent, U_at_1mm_J=model.energy_at_mm_J, B_SI=model.coefficient)


def synthetic(out, mass, known):
    check_split(C_TRAIN, C_TEST)
    rows = []
    for amplitude in C_TRAIN + C_TEST:
        radius = amplitude * LENGTH
        expected = known.period(radius, mass)
        time = np.linspace(0., 4.25 * expected, 4201)
        state = np.array([radius, 0., 0., 0.])
        actual = integrate_effective(time, state, mass, known)
        observed, turns = turning_data(time, actual, expected)
        energy = mass * np.sum(actual[:, 2:]**2, axis=1) / 2 + known.potential(np.linalg.norm(actual[:, :2], axis=1))
        row = dict(radius_input_mm=amplitude, split="train" if amplitude in C_TRAIN else "test",
                   **observed, exact_period_s=expected, relative_energy_span=np.ptp(energy) / energy[0])
        rows.append(row)
        np.savez_compressed(out / f"C_{amplitude:g}mm.npz", time_s=time, states_SI=actual,
                            turning_points=turns, energy_J=energy)
    train = [row for row in rows if row["split"] == "train"]
    reconstructed, fit = fit_power([row["amplitude_m"] for row in train], [row["period_s"] for row in train], mass)
    for row in rows:
        radius = row["radius_input_mm"] * LENGTH
        expected = row["exact_period_s"]
        time = np.linspace(0., 4.25 * expected, 4201)
        row["predicted_period_s"] = reconstructed.period(radius, mass)
        row["prediction_relative_error"] = row["predicted_period_s"] / expected - 1
        predicted = integrate_effective(time, [radius, 0., 0., 0.], mass, reconstructed)
        path = out / f"C_{row['radius_input_mm']:g}mm.npz"
        with np.load(path) as z:
            errors = position_errors(z["states_SI"][:, :2], predicted[:, :2], LENGTH)
        row["prediction_coordinate_rms_pct"] = 100 * np.sqrt(np.mean(errors["vector"]**2))
    write_table(out / "synthetic.csv", rows)
    if abs(reconstructed.exponent - known.exponent) > 2e-6 or max(abs(row["prediction_relative_error"]) for row in rows) > 2e-6:
        raise RuntimeError("Synthetic control failed. Do not proceed to RF fitting.")
    result = dict(known=power_row(known), fit=power_row(reconstructed), diagnostics=fit,
                  p_error=reconstructed.exponent - known.exponent,
                  U_relative_error=reconstructed.energy_at_mm_J / known.energy_at_mm_J - 1)
    write_json(out / "synthetic_fit.json", result)
    print("C control:", result, flush=True)
    return reconstructed


def read_p4(mesh, phase_index, level, extent, settings, used):
    tag = f"A_h{mesh}_phase{phase_index}_{level}_{extent}"
    path = P4 / "runs" / (tag + ".npz")
    meta_path = path.with_suffix(".json")
    meta = read_json(meta_path)
    for source, expected in ((path, meta["arrays_sha256"]),
                             (path.with_name(tag + "_crossings.jsonl"), meta["crossings_sha256"])):
        actual = sha(source)
        if actual != expected:
            raise ValueError(f"P4 provenance failure: {source}")
        used[str(source.relative_to(ROOT))] = actual
    used[str(meta_path.relative_to(ROOT))] = sha(meta_path)
    if extent == "extended":
        parent = path.with_name(tag.replace("_extended", "_pilot") + ".npz")
        actual = sha(parent)
        if actual != meta["parent_arrays_sha256"]:
            raise ValueError("Changed parent trajectory")
        used[str(parent.relative_to(ROOT))] = actual
    with np.load(path) as arrays:
        t, states = arrays["time_s"], arrays["states"]
    n = settings["RF_periods"][extent] * 64
    if (meta["status"] != "completed" or len(t) != n + 1 or not np.all(np.isfinite(states))
            or not np.array_equal(states[0], settings["full_initial_states"][phase_index])
            or not np.allclose(t, np.arange(n + 1) / (64 * settings["physical"]["frequency_Hz"]), rtol=0, atol=1e-13)):
        raise ValueError(f"Invalid P4 launch or clock: {tag}")
    return t, states


def inverse(model, t, states, phase):
    return recover_slow_state_with_drift(model.quadratures, states, model.charge, model.mass,
                                        model.omega, model.omega * t + phase,
                                        check_points=model.reconstruction.check_points)


def filters(t, states, model, phase, n):
    """Uniform once-per-RF samples, except a phase-locked clock for the strobe."""
    tm, ym = centered_period_mean(states[:, :2], t, n)
    ts, ys = centered_period_mean(states, t, 2 * n)
    td, yd = centered_period_mean(ys, ts, 2 * n)
    strobe_offset = int(round((-phase / (2 * np.pi) % 1) * n)) % n
    strobe = slice(strobe_offset, None, n)
    indices = np.arange(n, len(t) - n, n)
    return dict(mean=(tm[n//2::n], ym[n//2::n]), double_mean=(td[::n], yd[::n]),
                strobe=(t[strobe], states[strobe]), inverse=(t[indices], inverse(model, t[indices], states[indices], phase)))


def diagnostics(out, label, series, expected, rows):
    for name, (t, states) in series.items():
        observed, turns = turning_data(t, states, expected)
        row = dict(case=label, filter=name, **observed)
        row["precession_deg_per_period"] = observed["fitted_rate_deg_per_s"] * observed["period_s"]
        if name == "double_mean":
            row.update(angular_momentum_fit(states))
        if name != "mean":
            tm, ym = series["mean"]
            common = (t >= tm[0]) & (t <= tm[-1])
            delta = position_errors(states[common, :2], CubicSpline(tm, ym[:, :2])(t[common]), LENGTH)
            row["filter_difference_rms_pct"] = 100 * np.sqrt(np.mean(delta["vector"]**2))
            row["filter_difference_max_pct"] = 100 * np.max(delta["vector"])
        rows.append(row)
        np.savez_compressed(out / f"{label}_{name}.npz", time_s=t, states_SI=states, turning_points=turns)
    return rows[-len(series)]


def compare(out, label, t, actual, models, observed, direction):
    results = []
    for name, prediction in models.items():
        errors = position_errors(prediction[:, :2], actual[:, :2], LENGTH)
        transverse = (prediction[:, :2] - actual[:, :2]) @ np.array([-direction[1], direction[0]]) / LENGTH
        observed_prediction, _ = turning_data(t, prediction, observed["period_s"])
        _, turns = turning_data(t, actual, observed["period_s"])
        at_turns = CubicSpline(t, prediction[:, :2])(turns[:, 0])
        angle_errors = .5 * np.angle(np.exp(2j * (np.arctan2(at_turns[:, 1], at_turns[:, 0]) - turns[:, 4])))
        rate_error = observed_prediction["fitted_rate_deg_per_s"] - observed["fitted_rate_deg_per_s"]
        period_error = 100 * abs(observed_prediction["period_s"] / observed["period_s"] - 1)
        result = dict(case=label, model=name, rms_pct=100 * np.sqrt(np.mean(errors["vector"]**2)),
                      max_pct=100 * np.max(errors["vector"]), period_error_pct=period_error,
                      radial_rms_pct=100 * np.sqrt(np.mean(errors["radial"]**2)),
                      transverse_rms_pct=100 * np.sqrt(np.mean(transverse**2)),
                      transverse_max_pct=100 * np.max(abs(transverse)),
                      amplitude_error_um=1e6 * (observed_prediction["amplitude_m"] - observed["amplitude_m"]),
                      predicted_orientation_rate_deg_s=observed_prediction["fitted_rate_deg_per_s"],
                      orientation_rms_deg=np.rad2deg(np.sqrt(np.mean(angle_errors**2))),
                      orientation_max_deg=np.rad2deg(np.max(abs(angle_errors))),
                      orientation_rate_error_deg_s=rate_error,
                      orientation_drift_error_deg=rate_error * (observed["last_time_s"] - observed["first_time_s"]))
        result["passes"] = all(result[key] <= threshold for key, threshold in LIMITS.items())
        results.append(result)
    np.savez_compressed(out / f"{label}_comparison.npz", time_s=t, RF_mean_m=actual[:, :2], **models)
    return results


def analytic_rf(model, initial_slow, phase, helicity, periods, step_divisor, *, time_direction=1):
    """Full Newton dynamics in the pure analytic rotating harmonic.

    Opposite helicity is implemented as y-reflection of the drive. Clock is
    RF periods and length is mm. No FEM or effective force in the RF ODE.
    """
    parity = np.array([1, helicity, 1, helicity])
    full = parity * matched_full_state_with_drift(
        model.reconstruction.scaled(model.amplitude), (initial_slow * parity)[:2], (initial_slow * parity)[2:],
        model.charge, model.mass, model.omega, phase)
    initial = full / [LENGTH, LENGTH, LENGTH * model.frequency, LENGTH * model.frequency]
    scale = 4 * model.charge * model.C4 * LENGTH**2 / (model.mass * model.frequency**2)

    def rhs(clock, state):
        x, y, vx, vy = state
        chi = 2 * np.pi * clock + phase + model.phase
        z = complex(x, -helicity * y)
        acceleration = scale * complex(-np.sin(chi), np.cos(chi)) * z**3
        return vx, vy, acceleration.real, helicity * acceleration.imag

    clock = time_direction * np.arange(periods * 16 + 1) / 16
    sol = solve_ivp(rhs, (0., clock[-1]), initial, t_eval=clock, method="DOP853", max_step=1 / step_divisor,
                    rtol=3e-13, atol=3e-15)
    if not sol.success:
        raise RuntimeError(sol.message)
    return clock / model.frequency, sol.y.T * [LENGTH, LENGTH, LENGTH * model.frequency, LENGTH * model.frequency]


def velocity_pair(out, model):
    """Estimate the odd transverse acceleration in the same analytic RF field."""
    rows = []
    for half_window in (32, 64):
        for phase in (.21, 1.31):
            observations = []
            for vx in (-.12, .12):
                initial = [LENGTH, 0., vx, 0.]
                tf, yf = analytic_rf(model, initial, phase, 1, half_window, 40)
                tb, yb = analytic_rf(model, initial, phase, 1, half_window, 40, time_direction=-1)
                time = np.r_[tb[:0:-1], tf]
                states = np.vstack((yb[:0:-1], yf))
                scale = tf[-1]
                u = time / scale
                slow = np.polynomial.polynomial.polyvander(u, 7)
                rf = np.polynomial.polynomial.polyvander(u, 5)
                chi = model.omega * time + phase + model.phase
                design = np.column_stack((slow, *(rf * wave(n * chi)[:, None]
                                                  for n in (1, 2, 3) for wave in (np.cos, np.sin))))
                coefficients = np.linalg.lstsq(design, states[:, :2], rcond=None)[0]
                trend = coefficients[:3] * np.array([1., 1 / scale, 2 / scale**2])[:, None]
                residual = states[:, :2] - design @ coefficients
                observations.append(trend)
                np.savez_compressed(out / f"velocity_pair_{half_window}_{phase}_{vx:+.2f}.npz",
                                    time_s=time, states_SI=states, slow_trend_at_0=trend)
                rows.append(dict(half_window_RF=half_window, phase_rad=phase, launch_vx_m_s=vx,
                                 x_m=trend[0, 0], y_m=trend[0, 1], vx_m_s=trend[1, 0], vy_m_s=trend[1, 1],
                                 ax_m_s2=trend[2, 0], ay_m_s2=trend[2, 1],
                                 position_fit_residual_rms_m=np.sqrt(np.mean(residual**2))))
            delta = observations[1] - observations[0]
            rows[-1]["pair_separation_m"] = np.linalg.norm(delta[0])
            rows[-1]["pair_delta_ay_m_s2"] = delta[2, 1]
            rows[-1]["gamma_from_pair_s_inv"] = delta[2, 1] / delta[1, 0]
    write_table(out / "velocity_pair.csv", rows)


def symmetry_scales(b, c):
    q = b.quadratures
    # Allowed signed azimuthal orders in Im(exp(i*chi) sum d_n conj(z)^n).
    c4 = abs(q.sine[4] - 1j * q.cosine[4]) / (2 * q.reference_radius**4)
    c8 = abs(q.sine[8] + 1j * q.cosine[8]) / (2 * q.reference_radius**8)
    c16 = abs(q.sine[16] - 1j * q.cosine[16]) / (2 * q.reference_radius**16)
    b14 = 32 * c.charge**2 * c8**2 / (c.mass * c.omega**2)
    return dict(C4_V_m4=c4, C8_V_m8=c8, C16_V_m16=c16, B14_field_J_m14=b14,
                r14_over_r6_at_1mm=b14 * LENGTH**8 / c.B6,
                leading_angular_over_r6_at_1mm=8 * c16 / c4 * LENGTH**12)


def record_p4(out, mesh, phase_index, level, extent, label, settings, model, used, rows):
    t, raw = read_p4(mesh, phase_index, level, extent, settings, used)
    phase = settings["physical"]["phase_rad"][phase_index]
    series = filters(t, raw, model, phase, 64)
    observed = diagnostics(out, label, series, settings["estimated_slow_period_s"], rows)
    return inverse(model, t[:1], raw[:1], phase)[0], series["mean"], observed


def predict(t, initial, b, mass, known, reconstructed, gamma):
    clock = np.r_[0., t]
    models = dict(C=integrate_effective(clock, initial, mass, known)[1:],
                  D=integrate_effective(clock, initial, mass, reconstructed)[1:],
                  E=integrate_effective(clock, initial, mass, known, gamma_mm=gamma)[1:])
    if b is not None:
        models["B"] = integrate_effective(clock, initial, mass, known, fem_model=b)[1:]
    return models


def run_rf(out, settings, c, b, known, used):
    rows, comparisons = [], []
    check_split(RF_TRAIN, RF_TEST + tuple(case[0] for case in ANALYTIC_TEST))
    gamma = octupole_gamma(c.charge, c.mass, c.C4, c.omega)
    write_json(out / "symmetry_scales.json", symmetry_scales(b, c))
    label = RF_TRAIN[0]
    initial, (t, mean), observed = record_p4(out, 35, 0, "fine", "extended", label, settings, b, used, rows)
    reconstructed, fit = fit_power([observed["amplitude_m"]], [observed["period_s"]], c.mass, exponent=6)
    map_sensitivity = []
    for row in rows:
        fitted, _ = fit_power([row["amplitude_m"]], [row["period_s"]], c.mass, exponent=6)
        map_sensitivity.append(dict(filter=row["filter"], B6_J_m6=fitted.coefficient))
    write_json(out / "frozen_models.json", dict(
        C=power_row(known), D=power_row(reconstructed), train=RF_TRAIN, test=RF_TEST,
        analytic_holdout=[case[0] for case in ANALYTIC_TEST], D_diagnostics=fit,
        D_filter_sensitivity=map_sensitivity, E_gamma_mm_s_inv=gamma,
        E_G_mm_kg_per_s=c.mass * gamma, E_gamma_r4_m4_s_inv=gamma / LENGTH**4,
        E_fit_to_trajectory=False))
    initials = {label: initial.tolist()}
    comparisons.extend(compare(out, label, t, mean, predict(t, initial, b, c.mass, known, reconstructed, gamma),
                               observed, initial[:2] / np.linalg.norm(initial[:2])))
    label = RF_TEST[0]
    initial, (t, mean), observed = record_p4(out, 35, 1, "fine", "pilot", label, settings, b, used, rows)
    initials[label] = initial.tolist()
    comparisons.extend(compare(out, label, t, mean, predict(t, initial, b, c.mass, known, reconstructed, gamma),
                               observed, initial[:2] / np.linalg.norm(initial[:2])))
    for label, mesh, phase, level, extent in (
        ("p4_phase0_h50", 50, 0, "fine", "extended"),
        ("p4_phase0_coarse", 35, 0, "coarse", "extended"),
        ("p4_phase1_h50", 50, 1, "fine", "pilot"),
        ("p4_phase1_coarse", 35, 1, "coarse", "pilot"),
    ):
        record_p4(out, mesh, phase, level, extent, label, settings, b, used, rows)
    write_table(out / "slow_diagnostics.csv", rows)
    write_table(out / "comparisons.csv", comparisons)
    print("P4 comparisons complete. Running analytic RF controls.", flush=True)
    velocity_pair(out, c)
    analytic_model = AveragedRFModel(c.reconstruction, c.charge, c.mass, c.amplitude, c.frequency)
    for label, radius, angle, phase, helicity in ANALYTIC_TEST:
        direction = np.array([np.cos(angle), np.sin(angle)])
        initial = np.r_[radius * LENGTH * direction, 0., 0.]
        initials[label] = initial.tolist()
        expected = known.period(radius * LENGTH, c.mass)
        periods = round(3.25 * expected * c.frequency)
        parity = np.array([1, helicity, 1, helicity])
        runs = {}
        for suffix, divisor in (("", 20), ("_coarse", 10)):
            raw_time, raw = analytic_rf(c, initial, phase, helicity, periods, divisor)
            if not suffix:
                np.savez_compressed(out / f"{label}_RF.npz", time_s=raw_time, states_SI=raw)
            series = filters(raw_time, raw * parity, analytic_model, phase, 16)
            series = {name: (t, states * parity[:states.shape[1]]) for name, (t, states) in series.items()}
            observed = diagnostics(out, label + suffix, series, expected, rows)
            runs[suffix] = series["mean"]
            if not suffix:
                t, mean = series["mean"]
                comparisons.extend(compare(out, label, t, mean, predict(t, initial, None, c.mass, known, reconstructed, helicity * gamma),
                                           observed, direction))
        delta = position_errors(runs[""][1], runs["_coarse"][1], LENGTH)["vector"]
        rows[-len(series)]["analytic_fine_coarse_rms_pct"] = 100 * np.sqrt(np.mean(delta**2))
        print(f"{label}: analytic RF fine/coarse RMS = {100 * np.sqrt(np.mean(delta**2)):.3g}%", flush=True)
        write_table(out / "slow_diagnostics.csv", rows)
        write_table(out / "comparisons.csv", comparisons)
    write_json(out / "slow_initial_states.json", initials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "synthetic"), default="all")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings_path = P4 / "validation_settings.json"
    calibration_path = P4 / "h35_P4/native_C_calibration.json"
    reconstruction_path = P4 / "h35_P4/diagnostic_N48.json"
    settings, calibration = read_json(settings_path), read_json(calibration_path)
    p = settings["physical"]
    c = OctupoleModel(calibration["C4_per_volt_per_m4"], calibration["phase_rad"],
                      p["charge_C"], p["mass_kg"], p["U0_V"], p["frequency_Hz"], .008)
    known = RadialPower(6., c.B6 * LENGTH**6)
    used = {str(path.relative_to(ROOT)): sha(path) for path in (
        settings_path, calibration_path, reconstruction_path, Path(__file__).resolve(),
        ROOT / "src/three_phase_trap/effective_reconstruction_v1.py",
        ROOT / "src/three_phase_trap/micromotion.py",
        ROOT / "src/three_phase_trap/slow_analysis.py",
        ROOT / "src/three_phase_trap/trajectory_diagnostics.py",
        ROOT / "src/three_phase_trap/rf_reconstruction.py",
        ROOT / "src/three_phase_trap/averaged_dynamics.py",
        ROOT / "src/three_phase_trap/octupole_model.py",
    )}
    write_json(out / "configuration.json", dict(version=1, stage=args.stage, physical=p,
               C_train_mm=C_TRAIN, C_test_mm=C_TEST, RF_train=RF_TRAIN, RF_test=RF_TEST,
               analytic_cases=ANALYTIC_TEST, limits=LIMITS, scale_m=LENGTH,
               runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)))
    synthetic(out, c.mass, known)
    if args.stage == "all":
        b = AveragedRFModel(RFReconstruction.load(reconstruction_path), c.charge, c.mass, c.amplitude, c.frequency)
        run_rf(out, settings, c, b, known, used)
    write_json(out / "input_sha256.json", used)
    write_json(out / "result_sha256.json", {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})


if __name__ == "__main__":
    main()
