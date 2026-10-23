"""User step 6 / detailed stage 5: common initial states and slow extraction.

Includes short (32 RF periods) A_smooth filter diagnostics, not the long
physical A/B/C pilot. The original FEM cache is loaded read-only.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from three_phase_trap.config import load_config, ParticleConfig
from three_phase_trap.fem_cache import load_fem_cache, fem_fingerprint
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.averaged_dynamics import AveragedRFModel, integrate_averaged
from three_phase_trap.dynamics import integrate_trajectory
from three_phase_trap.micromotion import (matched_full_state, recover_slow_state, matched_full_state_with_drift,
                                        recover_slow_state_with_drift, PhasedFEMField)
from three_phase_trap.slow_analysis import centered_period_mean, local_rf_trend, poincare_samples
from three_phase_trap.validation_benchmarks import ideal_field, quadrupole_exact
from three_phase_trap.validation_field_audit import rf_quadratures
from three_phase_trap.voltages import phase_voltages


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(actual, reference, scale):
    error = np.linalg.norm(actual-reference, axis=-1)/scale
    return {"relative_rms": float(np.sqrt(np.mean(error**2))), "relative_max": float(np.max(error))}


def exact_quadrupole_slow(time, full_initial, amplitude, omega, phase):
    """Independent low-frequency spectral part of the exact rotating system.

    Rotation splits every eigenmode into lab frequencies Im(lambda)+/-Omega/2.
    Keep the branch |frequency|<Omega/2, discard its fast partner. This is
    independent of both the leading micromotion transform and box filtering.
    q=m=1; only the stable, nondegenerate benchmark is used here.
    """
    w, k = omega/2, 2*amplitude
    J = np.array([[0., -1.], [1., 0.]])
    M = np.block([[np.zeros((2, 2)), np.eye(2)], [w*w*np.eye(2)-k*np.diag([1., -1.]), -2*w*J]])
    co, si = np.cos(phase/2), np.sin(phase/2)
    Q = np.array([[co, -si], [si, co]])
    u = Q.T@full_initial[:2]
    du = Q.T@full_initial[2:]-w*J@u
    eigenvalues, vectors = np.linalg.eig(M)
    amplitudes = np.linalg.solve(vectors, np.r_[u, du])
    output = np.zeros((len(time), 4), complex)
    for value, vector, a in zip(eigenvalues, vectors.T, amplitudes):
        if abs(value.real) > 1e-10:
            raise RuntimeError("Quadrupole spectral benchmark is unstable")
        for sign in (-1, 1):
            frequency = value+sign*1j*w
            if abs(frequency.imag) < w:
                projected = Q@((np.eye(2)-sign*1j*J)/2)@vector[:2]*a
                position = np.exp(frequency*np.asarray(time))[:, None]*projected
                output += np.column_stack((position, frequency*position))
    if np.max(abs(output.imag)) > 1e-10:
        raise RuntimeError("Exact slow spectral projection is not real")
    return output.real


def benchmark_checks(output):
    records = []
    R, V = np.array([1., .1]), np.array([0., .03])
    scale = float(np.linalg.norm(R))
    for multiplier in (1, 2, 4):
        amplitude, omega = 2.*multiplier, 20.*multiplier
        q = ideal_field(2, amplitude)
        for phase in (0., np.pi/2):
            for scheme, forward, backward in (("leading", matched_full_state, recover_slow_state),
                                               ("drift_corrected", matched_full_state_with_drift, recover_slow_state_with_drift)):
                full0 = forward(q, R, V, 1., 1., omega, phase)
                t = np.arange(16*128+1)*2*np.pi/omega/128
                exact = quadrupole_exact(t, full0, amplitude, omega, phase)
                exact_slow = exact_quadrupole_slow(t, full0, amplitude, omega, phase)
                recovered = backward(q, exact, 1., 1., omega, omega*t+phase)
                tm, mean = centered_period_mean(exact[:, :2], t, 128)
                tp, trend = local_rf_trend(exact[:, :2], t, 128, 2)
                truth = CubicSpline(t, exact_slow[:, :2], axis=0)
                r_error = float(np.linalg.norm(exact_slow[0, :2]-R)/scale)
                v_error = float(np.linalg.norm(exact_slow[0, 2:]-V)/(scale*2*amplitude/omega))
                records.append({"transform": scheme, "frequency_multiplier": multiplier, "phase_rad": phase,
                                "initial_slow_position_remainder_relative": r_error,
                                "initial_slow_velocity_remainder_on_R_omega_slow": v_error,
                                "box_mean_vs_exact_slow": metrics(mean, truth(tm), scale),
                                "local_rf_trend_vs_exact_slow": metrics(trend, truth(tp), scale),
                                "inverse_micromotion_vs_exact_slow": metrics(recovered[:, :2], exact_slow[:, :2], scale)})
                np.savez_compressed(output/f"quadrupole_s{multiplier}_phase{round(phase*2/np.pi)}_{scheme}.npz",
                                    time=t, exact_full=exact, exact_slow=exact_slow, inverse_slow=recovered,
                                    mean_time=tm, mean_position=mean, trend_time=tp, trend_position=trend)
    return records


def synthetic_checks(output, frequency):
    t = np.arange(16*128+1)/frequency/128
    u = t*frequency
    slow = np.column_stack((.001+2e-6*u+3e-8*u*u, .0002-1e-6*u+2e-8*u*u))
    ac = np.column_stack((3e-7+2e-9*u, -2e-7+1e-9*u))
    ass = np.column_stack((1e-7-1e-9*u, 2.5e-7+1e-9*u))
    full = slow+ac*np.cos(2*np.pi*u)[:, None]+ass*np.sin(2*np.pi*u)[:, None]
    truth = CubicSpline(t, slow, axis=0)
    records = []
    for samples in (32, 64, 128):
        step = 128//samples
        time, values = t[::step], full[::step]
        tm, mean = centered_period_mean(values, time, samples)
        tp, trend = local_rf_trend(values, time, samples, 2)
        records.append({"samples_per_RF_period": samples,
                        "box_vs_known_slow": metrics(mean, truth(tm), .001),
                        "local_rf_trend_vs_known_slow": metrics(trend, truth(tp), .001)})
    np.savez_compressed(output/"synthetic_truth.npz", time=t, known_slow=slow, synthetic_full=full)
    return records


def short_smooth_checks(output, b, R, V, phases, settings):
    scale, frequency = float(np.linalg.norm(R)), b.frequency
    records, plot_data = [], []
    particle = ParticleConfig(mass=b.mass, charge=b.charge, gamma=0., gravity_enabled=False)
    duration = settings["short_RF_periods"]/frequency
    slow = integrate_averaged(b, R, V, duration, samples=settings["short_RF_periods"]*128+1,
                              rtol=2e-12, atol=2e-14, max_step=duration/400)
    if slow.status != "completed":
        raise RuntimeError("Short slow reference did not complete")
    for phase in phases:
        print(f"Short A_smooth filter check, RF phase={phase:.6g} rad...", flush=True)
        initial = matched_full_state_with_drift(b.quadratures, R, V, b.charge, b.mass, b.omega, phase)
        def field(p, t):
            return b.quadratures.field(b.reconstruction.check_points(p), b.omega*t+phase)
        results = []
        for max_step, rtol, atol in ((50, 2e-9, 2e-13), (100, 2e-11, 2e-15)):
            result = integrate_trajectory(None, particle, b.amplitude, frequency,
                                          initial_position=initial[:2], initial_velocity=initial[2:],
                                          simulation_duration=duration, escape_radius=b.reconstruction.valid_radius,
                                          rtol=rtol, atol=atol, max_step_per_period=max_step,
                                          output_samples_per_period=128, field_function=field)
            if result.termination != "completed":
                raise RuntimeError("Short A_smooth diagnostic did not complete")
            results.append(result)
        coarse, fine = results
        t, state = fine.time, fine.state.T
        inverse = recover_slow_state_with_drift(b.quadratures, state, b.charge, b.mass, b.omega, b.omega*t+phase,
                                      check_points=b.reconstruction.check_points)
        reference = CubicSpline(slow.time, slow.states[:, :2], axis=0)
        filtered = {}
        windows, resolution = [], []
        for periods in (1, 2, 3):
            tm, mean = centered_period_mean(state[:, :2], t, 128*periods)
            _, filtered_reference = centered_period_mean(slow.states[:, :2], slow.time, 128*periods)
            filtered[periods] = (tm, mean)
            windows.append({"window_RF_periods": periods,
                            "mean_vs_B_at_centre": metrics(mean, reference(tm), scale),
                            "mean_vs_identically_filtered_B": metrics(mean, filtered_reference, scale)})
        tm, mean = filtered[1]
        mean_reference = CubicSpline(tm, mean, axis=0)
        for samples in (32, 64):
            step = 128//samples
            ts, ms = centered_period_mean(state[::step, :2], t[::step], samples)
            resolution.append({"samples_per_RF_period": samples, **metrics(ms, mean_reference(ts), scale)})
        tp, trend = local_rf_trend(state[:, :2], t, 128, 2)
        inverse_reference = CubicSpline(t, inverse[:, :2], axis=0)
        window_differences = []
        for periods in (2, 3):
            tw, mw = filtered[periods]
            window_differences.append({"window_RF_periods": periods, **metrics(mw, mean_reference(tw), scale)})
        pts, full_pts = poincare_samples(t, state, frequency, phase)
        _, low_resolution_pts = poincare_samples(t[::2], state[::2], frequency, phase)
        slow_pts = recover_slow_state_with_drift(b.quadratures, full_pts, b.charge, b.mass, b.omega, b.omega*pts+phase,
                                      check_points=b.reconstruction.check_points)
        point_rows = [{"time_s": float(tt), "RF_phase_mod_2pi": float(np.mod(b.omega*tt+phase, 2*np.pi)),
                       "x_full_m": float(y[0]), "y_full_m": float(y[1]), "vx_full_m_per_s": float(y[2]), "vy_full_m_per_s": float(y[3]),
                       "x_inverse_m": float(z[0]), "y_inverse_m": float(z[1]),
                       "vx_inverse_m_per_s": float(z[2]), "vy_inverse_m_per_s": float(z[3])} for tt, y, z in zip(pts, full_pts, slow_pts)]
        tag = f"phase{round(phase*2/np.pi)}"
        csv_write(output/f"poincare_{tag}.csv", point_rows)
        np.savez_compressed(output/f"smooth_short_{tag}.npz", time_s=t, full_states=state,
                            coarse_states=coarse.state.T, inverse_states=inverse, B_states=slow.states,
                            mean_time_s=tm, mean_position_m=mean, trend_time_s=tp, trend_position_m=trend,
                            two_period_time_s=filtered[2][0], two_period_mean_m=filtered[2][1],
                            three_period_time_s=filtered[3][0], three_period_mean_m=filtered[3][1])
        record = {"phase_rad": phase, "duration_s": duration, "RF_periods": settings["short_RF_periods"],
                  "ode_coordinate_change": metrics(coarse.state[:2].T, state[:, :2], scale),
                  "output_resolution_checks": resolution, "window_checks": windows,
                  "window_change_vs_one_period": window_differences,
                  "mean_vs_inverse_micromotion": metrics(mean, inverse_reference(tm), scale),
                  "trend_vs_inverse_micromotion": metrics(trend, inverse_reference(tp), scale),
                  "trend_vs_B": metrics(trend, reference(tp), scale),
                  "inverse_micromotion_vs_B": metrics(inverse[:, :2], slow.states[:, :2], scale),
                  "poincare_interpolation_64_vs_128": metrics(low_resolution_pts[:, :2], full_pts[:, :2], scale),
                  "poincare_count": len(pts)}
        records.append(record)
        plot_data.append((phase, t, state, slow.states, tm, mean, inverse))
    return records, plot_data


def main():
    output = ROOT/"outputs/abc_validation/filter_checks"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output.parent/"protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pilot = protocol["pilot"]
    b_path = output.parent/"model_b/reconstruction.json"
    c_path = output.parent/"model_c/coefficients.json"
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    fingerprint = fem_fingerprint(config)
    if fingerprint != protocol["fem_fingerprint"]:
        raise ValueError("Protocol FEM fingerprint mismatch")
    rf = RFReconstruction.load(b_path, fingerprint)
    b = AveragedRFModel(rf, pilot["charge_C"], pilot["mass_kg"], pilot["U0_V"], pilot["frequency_Hz"])
    R, V = np.array(pilot["slow_position_m"]), np.array(pilot["slow_velocity_m_per_s"])
    budget = protocol["predeclared_trajectory_tolerances"]["relative_position_rms"]*protocol["predeclared_trajectory_tolerances"]["numerical_budget_fraction"]
    settings = {"user_step": 6, "detailed_plan_stage": 5, "reference_field": "amplitude-scaled B reconstruction",
                "selected_transform": "drift_corrected", "leading_transform_retained_as_benchmark": True,
                "full_IC_reference_for_A_and_A_smooth": "same smooth B field, same full state; isolate spatial representation error",
                "short_RF_periods": 32, "short_output_samples_per_RF_period": 128,
                "short_max_steps_per_RF_period": [50, 100], "short_rtol": [2e-9, 2e-11], "short_atol_SI": [2e-13, 2e-15],
                "window_RF_periods": [1, 2, 3], "output_resolution_samples": [32, 64, 128],
                "default_filter": "centred one-RF-period trapezoid mean, no endpoint extrapolation",
                "alternative_filter": "local quadratic trend + RF quadratures and linear RF drift over 2 periods",
                "poincare_target_RF_phase_rad": 0., "coordinate_diagnostic_budget": budget,
                "velocity_benchmark_budget_on_R_omega_slow": budget,
                "benchmark_frequencies": [20, 40, 80], "benchmark_amplitudes": [2, 4, 8],
                "approximation_order": "leading xi plus delta_xi=-2q/(m Omega^4)*(partial_t J)*V and its full material derivative using A_B; not a complete next-order averaging Hamiltonian",
                "scope": "Short A_smooth diagnostics and exact/synthetic benchmarks only; no raw FEM A trajectory or long pilot"}
    write(output/"settings.json", settings)
    print("Verifying original FEM cache and common RF phase adapter...", flush=True)
    _, _, _, interp = load_fem_cache(config, ROOT)
    initial_states = []
    maximum_phase_error = 0.
    for phase in pilot["phase_rad"]:
        full = matched_full_state_with_drift(b.quadratures, R, V, b.charge, b.mass, b.omega, phase)
        inverse = recover_slow_state_with_drift(b.quadratures, full, b.charge, b.mass, b.omega, phase, check_points=rf.check_points)
        shifted = PhasedFEMField(interp, b.amplitude, b.frequency, phase)
        probe = np.array([[.001, 0.], [.0006, .0007], [-.002, .001]])
        ec, es = rf_quadratures(interp, probe, b.amplitude)
        for time in (0., .137/b.frequency, .83/b.frequency):
            expected = ec*np.cos(b.omega*time+phase)+es*np.sin(b.omega*time+phase)
            maximum_phase_error = max(maximum_phase_error, float(np.max(abs(shifted(probe, time)-expected))/np.max(abs(expected))))
        raw_ec, raw_es = rf_quadratures(interp, R, b.amplitude)
        raw_E = raw_ec*np.cos(phase)+raw_es*np.sin(phase)
        raw_dtE = b.omega*(-raw_ec*np.sin(phase)+raw_es*np.cos(phase))
        raw_full_pilot = np.r_[R-b.charge*raw_E/(b.mass*b.omega**2), V-b.charge*raw_dtE/(b.mass*b.omega**2)]
        leading_full = matched_full_state(b.quadratures, R, V, b.charge, b.mass, b.omega, phase)
        initial_states.append({"leading_full_state_for_comparison": matched_full_state(b.quadratures, R, V, b.charge, b.mass, b.omega, phase).tolist(),
                               "RF_phase_rad": phase, "RF_time_shift_s": phase/b.omega,
                               "phase_group_voltages_at_t0_V": {p: float(u) for p, u in phase_voltages(phase/b.omega, b.amplitude, b.frequency).items()},
                               "B_and_C_slow_state": np.r_[R, V].tolist(), "A_and_A_smooth_full_state": full.tolist(),
                               "micromotion_position_correction_m": (full[:2]-R).tolist(),
                               "micromotion_velocity_correction_m_per_s": (full[2:]-V).tolist(),
                               "inverse_recovered_slow_state": inverse.tolist(),
                               "inverse_roundtrip_position_error_m": float(np.linalg.norm(inverse[:2]-R)),
                               "inverse_roundtrip_velocity_error_m_per_s": float(np.linalg.norm(inverse[2:]-V)),
                               "raw_FEM_vs_smooth_leading_IC_position_difference_m": float(np.linalg.norm(raw_full_pilot[:2]-leading_full[:2])),
                               "raw_FEM_vs_smooth_leading_IC_velocity_difference_m_per_s": float(np.linalg.norm(raw_full_pilot[2:]-leading_full[2:])),
                               "drift_refinement_position_change_m": float(np.linalg.norm(full[:2]-leading_full[:2])),
                               "drift_refinement_velocity_change_m_per_s": float(np.linalg.norm(full[2:]-leading_full[2:]))})
    # An additional nonzero slow velocity exercises the convective term.
    control_V, control_phase = np.array([.002, -.003]), .731
    control_full = matched_full_state_with_drift(b.quadratures, R, control_V, b.charge, b.mass, b.omega, control_phase)
    control_inverse = recover_slow_state_with_drift(b.quadratures, control_full, b.charge, b.mass, b.omega, control_phase, check_points=rf.check_points)
    ic = {"fem_fingerprint": fingerprint, "physical_parameters": pilot, "settings": settings,
          "state_order": ["x_m", "y_m", "vx_m_per_s", "vy_m_per_s"], "cases": initial_states,
          "nonzero_velocity_control": {"slow_state": np.r_[R, control_V].tolist(), "phase_rad": control_phase,
                                       "full_state": control_full.tolist(), "recovered_state": control_inverse.tolist()},
          "raw_FEM_IC_note": "A and A_smooth share drift-corrected smooth-reference IC; raw-vs-smooth leading corrections isolate spatial mismatch separately from drift refinement",
          "warning": "Algebraic roundtrip accuracy is not the physical accuracy of either truncated transform"}
    write(output.parent/"initial_conditions.json", ic)
    print("Checking exact quadrupole slow modes and known-truth synthetic filters...", flush=True)
    benchmarks = benchmark_checks(output)
    synthetic = synthetic_checks(output, b.frequency)
    short, plot_data = short_smooth_checks(output, b, R, V, pilot["phase_rad"], settings)
    algorithm_pass = maximum_phase_error < 1e-12 and max(row["inverse_roundtrip_position_error_m"] for row in initial_states) < 1e-12
    algorithm_pass &= max(row["inverse_roundtrip_velocity_error_m_per_s"] for row in initial_states) < 1e-8
    algorithm_pass &= np.linalg.norm(control_inverse[:2]-R) < 1e-12 and np.linalg.norm(control_inverse[2:]-control_V) < 1e-8
    algorithm_pass &= max(row["local_rf_trend_vs_known_slow"]["relative_max"] for row in synthetic) < 1e-10
    selected_benchmarks = [row for row in benchmarks if row["transform"] == "drift_corrected"]
    init_pass = max(max(row["initial_slow_position_remainder_relative"], row["initial_slow_velocity_remainder_on_R_omega_slow"]) for row in selected_benchmarks) < budget
    filter_budget_values = []
    for row in short:
        filter_budget_values += [row["ode_coordinate_change"]["relative_rms"], row["mean_vs_inverse_micromotion"]["relative_rms"]]
        filter_budget_values += [v["relative_rms"] for v in row["output_resolution_checks"]+row["window_change_vs_one_period"]]
    benchmark_filter_pass = max(row["box_mean_vs_exact_slow"]["relative_rms"] for row in selected_benchmarks) < budget
    filter_pass = max(filter_budget_values) < budget and benchmark_filter_pass
    files = ["scripts/run_abc_initial_conditions.py", "src/three_phase_trap/micromotion.py", "src/three_phase_trap/slow_analysis.py",
             "src/three_phase_trap/validation_benchmarks.py", "src/three_phase_trap/rf_reconstruction.py",
             "src/three_phase_trap/dynamics.py", "src/three_phase_trap/averaged_dynamics.py"]
    summary = {"status": "implemented_diagnostics_passed_long_pilot_unverified" if algorithm_pass and init_pass and filter_pass else "implemented_diagnostic_budget_not_met",
               "algorithm_checks_passed": bool(algorithm_pass), "benchmark_initialization_budget_passed": bool(init_pass),
               "short_filter_budget_passed": bool(filter_pass), "coordinate_diagnostic_budget": budget,
               "benchmark_filter_budget_passed": bool(benchmark_filter_pass),
               "RF_phase_adapter_relative_max_error": maximum_phase_error, "quadrupole_benchmarks": benchmarks,
               "synthetic_filter_checks": synthetic, "short_smooth_checks": short,
               "scope": settings["scope"], "raw_FEM_A_trajectory_computed": False, "long_physical_pilot_computed": False,
               "fem_fingerprint": fingerprint, "basis_sha256": protocol["basis_sha256"],
               "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-fem", "matplotlib")},
               "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
               "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in (protocol_path, b_path, c_path)}}
    write(output/"summary.json", summary)
    plot(output, plot_data)
    report(output, summary, initial_states)
    progress_path = output.parent/"progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["user_step_6_initial_conditions_and_filters"] = summary["status"]
    write(progress_path, progress)
    from run_abc_initial_checks import write_report
    write_report(output.parent, protocol, progress)
    print(json.dumps({"status": summary["status"], "phase_error": maximum_phase_error,
                      "initial_position_remainder_max": max(r["initial_slow_position_remainder_relative"] for r in selected_benchmarks),
                      "initial_velocity_remainder_max": max(r["initial_slow_velocity_remainder_on_R_omega_slow"] for r in selected_benchmarks),
                      "short_filter_budget_max": max(filter_budget_values), "pilot_initial_states": initial_states}, indent=2), flush=True)
    if not algorithm_pass:
        raise RuntimeError("Initialization/filter algorithm check failed; inspect summary.json")


def plot(output, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for phase, t, state, slow, tm, mean, inverse in data:
        B = CubicSpline(t, slow[:, :2], axis=0)
        axes[0].plot(t*1e6, (state[:, 0]-slow[:, 0])*1e6, label=f"raw RF - B, phase {phase:.2f}")
        axes[1].semilogy(tm*1e6, np.maximum(np.linalg.norm(mean-B(tm), axis=-1)/.001, 1e-16), label=f"period mean - B, phase {phase:.2f}")
        axes[1].semilogy(t*1e6, np.maximum(np.linalg.norm(inverse[:, :2]-slow[:, :2], axis=-1)/.001, 1e-16), linestyle="--", label=f"inverse - B, phase {phase:.2f}")
    axes[0].set(xlabel="t (microseconds)", ylabel="X difference (micrometres)", title="Short A_smooth diagnostic: 32 RF periods")
    axes[1].axhline(.001, color="grey", linestyle=":", label="0.1% numerical reference")
    axes[1].set(xlabel="t at window centre (microseconds)", ylabel="Position discrepancy / 1 mm", title="Extraction methods; not a long-pilot validation")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output/"initialization_and_filters.png", dpi=180)
    plt.close(fig)


def report(output, summary, initial):
    lines = ["# Пункт 6: начальные условия и выделение медленного движения", "",
             "Пункт 6 краткого списка соответствует этапу 5 подробного плана.", f"Статус: `{summary['status']}`.", "",
             "## Согласование начальных условий", "", "```text", "χ=Ωt+φ0, Ω=2πf, β=q/(mΩ²)",
             "ξ0(R,t)=−β E(R,t)", "ξd(R,V,t)=−2β/Ω² · (∂t J(R,t)) V, Jij=∂Ei/∂Rj",
             "r0=R0+ξ0+ξd", "v0=V0+Dt(ξ0+ξd)",
             "Dt=∂t+V·∇R+AB·∇V, AB=−∇ΨB/m", "```", "",
             "Сохранены два варианта: исходный ведущий ξ0 и выбранный `drift_corrected` с добавкой ξd.",
             "ξd учитывает изменение амплитуды RF-поля вдоль медленного движения. При двукратном",
             "дифференцировании ξ0 возникает член −2q/(mΩ²)·(∂tJ)V; в ξd выбран противоположный быстрый отклик.",
             "Скорость вычислена полной производной: добавка равна 2βJV−2β/Ω²·(H∂tE[V,V]+∂tJ·AB).",
             "Это не полная поправка следующего порядка к усреднённому гамильтониану и не точное преобразование.",
             "B и C начинают с общей медленной пары R0=(1 мм,0), V0=0.",
             "Для A и A_smooth выбрано одно и то же полное начальное состояние по гладкому полю B.",
             "Поэтому будущее сравнение A–A_smooth не смешивает замену поля с разными начальными состояниями.",
             "Несогласованность этой поправки с сырым FEM-полем сохранена отдельно; она не занулена.",
             "При V0=0 полная скорость v0 обычно не равна нулю: она содержит скорость микродвижения.", "",
             "| RF-фаза, рад | x0, м | y0, м | vx0, м/с | vy0, м/с |", "|---|---:|---:|---:|---:|"]
    for row in initial:
        y = row["A_and_A_smooth_full_state"]
        lines.append(f"| {row['RF_phase_rad']:.6g} | {y[0]:.12g} | {y[1]:.12g} | {y[2]:.12g} | {y[3]:.12g} |")
    lines += ["", "Ведущий вариант доступен через `matched_full_state` / `recover_slow_state`.",
              "Выбранный вариант — `matched_full_state_with_drift` / `recover_slow_state_with_drift` в `micromotion.py`.",
              "Для произвольной полной пары (r,v) сначала обращается ведущая карта, затем итеративно учитывается ξd.",
              "В каждой итерации решается r=R−qE(R)/(mΩ²) и линейная система для V с матрицей I−qJ/(mΩ²).",
              "Область поля, сходимость и обусловленность проверяются; глобальная единственность вне режима малой поправки не заявляется.",
              "Есть отдельный контроль с ненулевой V0. Данные в `../initial_conditions.json`; порядок состояния (x,y,vx,vy).", "",
              "## Общая фаза RF", "",
              "`PhasedFEMField` использует t→t+φ0/Ω одновременно для всех трёх фаз напряжения.",
              f"Согласие с непосредственной комбинацией сырых FEM-квадратур: {summary['RF_phase_adapter_relative_max_error']:.4g} на общем масштабе поля.", "",
              "## Независимый эталон начального преобразования", "",
              "Для вращающегося квадруполя использовано точное решение линейной системы через матричную экспоненту.",
              "Медленная часть независимо выделена спектральным разложением во вращающейся системе:",
              "из каждой моды взята лабораторная ветвь с частотой меньше Ω/2. Это не инверсия приближённого преобразования.",
              "Разница этой точной медленной пары и заданной (R0,V0) оценивает остаток начального приближения.", "",
              "| преобразование | множитель f и U | фаза | остаток R / Rscale | остаток V / (Rscale·ωslow) | RMS среднего против точной медленной части |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary["quadrupole_benchmarks"]:
        lines.append(f"| {row['transform']} | {row['frequency_multiplier']} | {row['phase_rad']:.4g} | {row['initial_slow_position_remainder_relative']:.4g} | {row['initial_slow_velocity_remainder_on_R_omega_slow']:.4g} | {row['box_mean_vs_exact_slow']['relative_rms']:.4g} |")
    lines += ["", "У ведущей карты на самом жёстком эталоне остаток V достигает 0,309% масштаба Rscale·ωslow.",
              "С ξd максимум уменьшился до 0,0300%; ориентир 0,1% не изменялся. Этот исходный провал сохранён в таблице.",
              f"Проверка остатка выбранной карты по ориентиру 0,1% пройдена: {summary['benchmark_initialization_budget_passed']}.",
              "Это эталон реализации, не универсальная оценка остатка для любой геометрии и начального состояния.", "",
              "## Фильтры и короткая проверка A_smooth", "",
              "Основной вариант: центрированное трапецеидальное среднее за один RF-период.",
              "Время результата — центр окна. Начальные/конечные пол-окна отбрасываются без экстраполяции.",
              "Сохранены проверки 32/64/128 отсчётов на период, окон 1/2/3 периода, сравнение с обратным",
              "преобразованием микродвижения и альтернативным локальным трендом с RF-квадратурами.",
              "Синтетический сигнал с известной квадратичной медленной частью и дрейфующей RF-амплитудой",
              "проверяет фильтр независимо от уравнения движения. Простое среднее имеет конечный оконный сдвиг;",
              "приближённое удаление микродвижения также не объявляется точной медленной траекторией.", "",
              "Полное гладкое поле A_smooth проинтегрировано лишь на 32 RF-периода (0,32 мс) для каждой фазы.",
              "Это значительно меньше медленного периода около 23 мс; длительный физический пилот не выполнен.",
              "Отдельно сохранены сравнения среднего с B в центре окна и с таким же оконным средним B.", "",
              "| фаза | RMS уточнения ODE | RMS дискретизации 32→128 | RMS окна 1→3 | RMS среднее–обратное преобразование |",
              "|---|---:|---:|---:|---:|"]
    for row in summary["short_smooth_checks"]:
        lines.append(f"| {row['phase_rad']:.4g} | {row['ode_coordinate_change']['relative_rms']:.4g} | {row['output_resolution_checks'][0]['relative_rms']:.4g} | {row['window_change_vs_one_period'][-1]['relative_rms']:.4g} | {row['mean_vs_inverse_micromotion']['relative_rms']:.4g} |")
    lines += ["", f"Короткие проверки фильтра по бюджету RMS координат 0,1% пройдены: {summary['short_filter_budget_passed']}.",
              "Все координатные метрики короткого пилота нормированы на заранее заданный Rscale=1 мм.",
              "Точки Пуанкаре сохранены при Ωt+φ0=0 mod 2π, с полным состоянием и обратным преобразованием.",
              "Это фазовые срезы; сами по себе они не равны временному среднему. Проверено влияние интерполяции.", "",
              "## Продолжение и воспроизведение", "",
              "Далее — пункт 7 краткого списка: длительное сравнение A, A_smooth, B и C минимум на трёх медленных периодах.",
              "Сходимость FEM по сетке остаётся открытой. Сохранённые начальные состояния и выбранный фильтр нужны для следующего расчёта.", "",
              "```powershell", "C:/Python314/python.exe scripts/run_abc_initial_conditions.py", "```", "",
              "`summary.json` содержит проверки, версии и SHA; `settings.json` — настройки, зафиксированные до расчёта.",
              "NPZ содержат точные эталоны, известную синтетическую медленную часть, короткие RF-траектории, B и результаты фильтрации.",
              "Файлы `quadrupole_*_leading.npz` и `quadrupole_*_drift_corrected.npz` разделяют две карты; `legacy_leading_run/` хранит первоначальный запуск.",
              "CSV `poincare_phase*.csv` содержат фазовые срезы. График: `initialization_and_filters.png`.",
              "Исходные FEM-кэш, сетка, архив и результаты B/C не изменялись."]
    (output/"INITIAL_CONDITIONS_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
