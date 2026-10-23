"""User step 4 (detailed stage 3): reconstruct and verify model B, no FEM rebuild.

Run from the repository: C:/Python314/python.exe scripts/run_abc_model_b.py
This does not compare physical RF trajectories A with averaged trajectories B.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.special import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from three_phase_trap.config import load_config
from three_phase_trap.fem_cache import load_fem_cache, fem_fingerprint
from three_phase_trap.effective_potential import mean_square_field_from_basis, time_average_field_squared, effective_potential_from_mean_square
from three_phase_trap.rf_reconstruction import RFReconstruction, fit_harmonic_quadratures
from three_phase_trap.averaged_dynamics import AveragedRFModel, integrate_averaged
from three_phase_trap.validation_field_audit import rf_quadratures, rf_potential_quadratures


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ring_points(radii, count, offset):
    angles = (np.arange(count)+offset)*2*np.pi/count
    return np.concatenate([r*np.column_stack((np.cos(angles), np.sin(angles))) for r in radii])


def relative_vector_rms(actual, reference):
    return float(np.sqrt(np.mean(np.sum((actual-reference)**2, axis=-1))/np.mean(np.sum(reference**2, axis=-1))))


def main():
    output = ROOT/"outputs/abc_validation/model_b"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT/"outputs/abc_validation/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pilot = protocol["pilot"]
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    config = replace(config, particle=replace(config.particle, mass=pilot["mass_kg"], charge=pilot["charge_C"]))
    fingerprint = fem_fingerprint(config)
    if fingerprint != protocol["fem_fingerprint"]:
        raise ValueError("FEM does not match the preregistered protocol")
    print("Loading and verifying original FEM cache (read only)...", flush=True)
    _, _, _, interp = load_fem_cache(config, ROOT)
    reference_radius, valid_radius = .008, .008
    settings = {"user_step": 4, "detailed_plan_stage": 3,
                "orders": [16, 24, 32, 40, 48], "selected_order": 48,
                "training_radii_m": [.0005, .0015, .003, .0045, .006, .0075],
                "training_angles": 256, "training_angle_offset": .317,
                "alternative_training_radii_m": [.0007, .0021, .0035, .005, .0065, .0077],
                "alternative_training_angles": 384, "alternative_training_angle_offset": .431,
                "holdout_radii_m": [.0001, .00025, .0005, .001, .002, .004, .006, .007, .008],
                "holdout_angles": 513, "holdout_angle_offset": .739,
                "reference_radius_m": reference_radius, "computational_domain_radius_m": valid_radius,
                "fit_input": "RF scalar potential at U0=1 V; all orders retained, no symmetry projection",
                "force_finite_difference_steps_m": [2e-7, 1e-7, 5e-8],
                "integrator": "DOP853, dimensionless state; no damping/gravity/DC",
                "energy_drift_tolerance": 1e-8, "force_fd_tolerance": 1e-6,
                "reconstruction_sensitivity_target": .001,
                "spatial_accuracy": "Single-mesh consistency only; FEM mesh convergence remains unverified"}
    write(output/"settings.json", settings)
    train = ring_points(settings["training_radii_m"], 256, .317)
    pc, ps = rf_potential_quadratures(interp, train, 1.)
    fits, conditions = {}, {}
    for order in settings["orders"]:
        fits[order], conditions[order] = fit_harmonic_quadratures(train, pc, ps, order, reference_radius)
    alternative_train = ring_points(settings["alternative_training_radii_m"], 384, .431)
    ac, ass = rf_potential_quadratures(interp, alternative_train, 1.)
    alternative, alt_condition = fit_harmonic_quadratures(alternative_train, ac, ass, 48, reference_radius)
    metadata = {"fem_fingerprint": fingerprint, "basis_sha256": protocol["basis_sha256"],
                "fit_order": 48, "all_modes_retained": True,
                "training_settings": settings, "design_condition": conditions[48],
                "spatial_accuracy": "not verified by a refined FEM mesh"}
    reconstruction = RFReconstruction(fits[48], valid_radius, metadata)
    reconstruction.save(output/"reconstruction.json")
    RFReconstruction(alternative, valid_radius, {**metadata, "training_grid": "alternative", "design_condition": alt_condition}).save(output/"alternative_reconstruction.json")
    # Reload the delivered artifact, rather than use an unsaved in-memory fit.
    reconstruction = RFReconstruction.load(output/"reconstruction.json", fingerprint)
    model = AveragedRFModel(reconstruction, pilot["charge_C"], pilot["mass_kg"], pilot["U0_V"], pilot["frequency_Hz"])
    alt_model = AveragedRFModel(RFReconstruction(alternative, valid_radius, metadata), model.charge, model.mass, model.amplitude, model.frequency)
    order_models = {n: AveragedRFModel(RFReconstruction(q, valid_radius, metadata), model.charge, model.mass, model.amplitude, model.frequency) for n, q in fits.items()}
    radii = settings["holdout_radii_m"]
    points = ring_points(radii, 513, .739)
    raw_c, raw_s = rf_quadratures(interp, points, model.amplitude)
    raw_psi = effective_potential_from_mean_square((np.sum(raw_c**2+raw_s**2, axis=-1))/2, model.charge, model.mass, model.omega)
    fit_c, fit_s = model.quadratures.field_quadratures(points)
    fit_psi, fit_force = model.potential(points), model.force(points)
    raw_phi_c, raw_phi_s = rf_potential_quadratures(interp, points, model.amplitude)
    fit_phi_c, fit_phi_s = model.quadratures.potential_quadratures(points)
    rows = []
    for i, radius in enumerate(radii):
        cut = slice(i*513, (i+1)*513)
        p = points[cut]
        norm_field = np.mean(np.sum(raw_c[cut]**2+raw_s[cut]**2, axis=-1))
        for order, comparison in order_models.items():
            cc, cs = comparison.quadratures.field_quadratures(p)
            rows.append({"radius_m": radius, "order": order, "design_condition": conditions[order],
                         "raw_vs_selected_field_relative_rms": float(np.sqrt(np.mean(np.sum((fit_c[cut]-raw_c[cut])**2+(fit_s[cut]-raw_s[cut])**2, axis=-1))/norm_field)),
                         "raw_vs_selected_potential_relative_rms": float(np.sqrt(np.mean((fit_phi_c[cut]-raw_phi_c[cut])**2+(fit_phi_s[cut]-raw_phi_s[cut])**2)/np.mean(raw_phi_c[cut]**2+raw_phi_s[cut]**2))),
                         "raw_vs_selected_psi_relative_rms": float(np.linalg.norm(fit_psi[cut]-raw_psi[cut])/np.linalg.norm(raw_psi[cut])),
                         "order_vs_selected_field_relative_rms": float(np.sqrt(np.mean(np.sum((cc-fit_c[cut])**2+(cs-fit_s[cut])**2, axis=-1))/np.mean(np.sum(fit_c[cut]**2+fit_s[cut]**2, axis=-1)))),
                         "order_vs_selected_force_relative_rms": relative_vector_rms(comparison.force(p), fit_force[cut]),
                         "alternative_training_force_relative_rms": relative_vector_rms(alt_model.force(p), fit_force[cut]),
                         "alternative_training_psi_relative_rms": float(np.linalg.norm(alt_model.potential(p)-fit_psi[cut])/np.linalg.norm(fit_psi[cut]))})
    csv_write(output/"reconstruction_checks.csv", rows)
    print("Checking fixed-point averaging, force gradient and slow dynamics...", flush=True)
    # On identical raw FEM points, verify all three averaging implementations.
    check_points = points[::79]
    bases = {p: interp.field_basis(check_points, p) for p in ("A", "B", "C")}
    exact_mean = mean_square_field_from_basis(bases, model.amplitude)
    rc, rs = rf_quadratures(interp, check_points, model.amplitude)
    quadrature_mean = np.sum(rc**2+rs**2, axis=-1)/2
    numerical_mean = time_average_field_squared(interp, check_points, model.amplitude, model.frequency, samples=64)
    field_scale = float(np.max(exact_mean))
    # Independent phase integral of the reconstructed field, with nonzero phase.
    ec, es = model.quadratures.field_quadratures(check_points)
    phase_mean = np.zeros(len(check_points))
    for phase in np.arange(64)*2*np.pi/64+.371:
        E = ec*np.cos(phase)+es*np.sin(phase)
        phase_mean += np.sum(E*E, axis=-1)/64
    smooth_psi_numeric = effective_potential_from_mean_square(phase_mean, model.charge, model.mass, model.omega)
    checks = {"raw_quadratures_vs_existing_exact_average_relative_max": float(np.max(abs(quadrature_mean-exact_mean))/field_scale),
              "raw_numerical_vs_existing_exact_average_relative_max": float(np.max(abs(numerical_mean-exact_mean))/field_scale),
              "reconstructed_psi_vs_phase_integral_relative_max": float(np.max(abs(smooth_psi_numeric-model.potential(check_points)))/np.max(smooth_psi_numeric))}
    rng = np.random.default_rng(20261023)
    # Interior points: all finite-difference stencil points stay in the disk.
    a = rng.uniform(0, 2*np.pi, 100)
    r = np.sqrt(rng.uniform(0, .0078**2, 100))
    fd_points = r[:, None]*np.column_stack((np.cos(a), np.sin(a)))
    analytic = model.force(fd_points)
    force_scale = float(np.max(np.linalg.norm(analytic, axis=-1)))
    fd_records = []
    for step in settings["force_finite_difference_steps_m"]:
        fd = np.column_stack([-(model.potential(fd_points+step*np.eye(2)[j])-model.potential(fd_points-step*np.eye(2)[j]))/(2*step) for j in range(2)])
        fd_records.append({"step_m": step, "max_error_N": float(np.max(np.linalg.norm(fd-analytic, axis=-1))),
                           "relative_max_on_fixed_force_scale": float(np.max(np.linalg.norm(fd-analytic, axis=-1))/force_scale)})
    checks["force_vs_energy_gradient"] = fd_records
    # Estimate a time scale only, not a calibration or verification of model C.
    q = model.quadratures
    c4_scale = np.sqrt((abs(q.cosine[4])**2+abs(q.sine[4])**2)/2)/reference_radius**4
    b6_scale = 8*model.charge**2*c4_scale**2/(model.mass*model.omega**2)
    R0, V0 = np.array(pilot["slow_position_m"]), np.array(pilot["slow_velocity_m_per_s"])
    length = np.linalg.norm(R0)
    period_estimate = float(4*np.sqrt(model.mass/(2*b6_scale))*beta(1/6, 1/2)/(6*length**2))
    duration = 3*period_estimate
    coarse = integrate_averaged(model, R0, V0, duration, samples=2401, rtol=2e-10, atol=2e-12, max_step=period_estimate/100)
    fine = integrate_averaged(model, R0, V0, duration, samples=2401, rtol=2e-12, atol=2e-14, max_step=period_estimate/200)
    alternate = integrate_averaged(alt_model, R0, V0, duration, samples=2401, rtol=2e-12, atol=2e-14, max_step=period_estimate/200)
    if any(result.status != "completed" for result in (coarse, fine, alternate)):
        raise RuntimeError("Pilot model B left its reconstruction domain")
    energy_scale = max(abs(float(fine.energy[0])), model.mass*(length/duration)**2)
    trajectory_checks = {"initial_position_m": R0.tolist(), "initial_velocity_m_per_s": V0.tolist(),
                         "duration_s": duration, "estimated_slow_period_s": period_estimate,
                         "time_scale_source": "leading coefficient used only to estimate integration duration",
                         "steps_per_estimated_slow_period": [100, 200],
                         "rtol": [2e-10, 2e-12], "dimensionless_atol": [2e-12, 2e-14],
                         "length_scale_m": length, "time_scale_s": duration,
                         "energy_scale_J": energy_scale, "nfev": [coarse.evaluations, fine.evaluations],
                         "status": fine.status, "max_radius_m": float(np.max(np.linalg.norm(fine.states[:, :2], axis=-1))),
                         "relative_energy_drift_max": float(np.max(abs(fine.energy-fine.energy[0]))/energy_scale),
                         "ode_relative_position_rms": float(np.sqrt(np.mean(np.sum((coarse.states[:, :2]-fine.states[:, :2])**2, axis=-1)))/length),
                         "ode_relative_position_max": float(np.max(np.linalg.norm(coarse.states[:, :2]-fine.states[:, :2], axis=-1))/length),
                         "alternative_training_relative_position_rms": float(np.sqrt(np.mean(np.sum((alternate.states[:, :2]-fine.states[:, :2])**2, axis=-1)))/length),
                         "alternative_training_relative_position_max": float(np.max(np.linalg.norm(alternate.states[:, :2]-fine.states[:, :2], axis=-1))/length)}
    np.savez_compressed(output/"slow_trajectory_checks.npz", time_s=fine.time, states=fine.states,
                        coarse_states=coarse.states, alternative_training_states=alternate.states, energy_J=fine.energy)
    np.savez_compressed(output/"holdout_samples.npz", points_m=points, raw_Ec=raw_c, raw_Es=raw_s,
                        smooth_Ec=fit_c, smooth_Es=fit_s, raw_psi_J=raw_psi, smooth_psi_J=fit_psi, smooth_force_N=fit_force)
    numeric_budget = protocol["predeclared_trajectory_tolerances"]["relative_position_rms"]*protocol["predeclared_trajectory_tolerances"]["numerical_budget_fraction"]
    checks_passed = (max(v for k, v in checks.items() if k != "force_vs_energy_gradient") < 1e-12
                     and fd_records[-1]["relative_max_on_fixed_force_scale"] < settings["force_fd_tolerance"]
                     and trajectory_checks["relative_energy_drift_max"] < settings["energy_drift_tolerance"]
                     and trajectory_checks["ode_relative_position_rms"] < numeric_budget)
    sensitivity_passed = (all(max(row["order_vs_selected_force_relative_rms"], row["alternative_training_force_relative_rms"],
                                  row["alternative_training_psi_relative_rms"]) < settings["reconstruction_sensitivity_target"]
                              for row in rows if row["order"] == 40)
                          and trajectory_checks["alternative_training_relative_position_rms"] < numeric_budget)
    relevant = ["scripts/run_abc_model_b.py", "src/three_phase_trap/rf_reconstruction.py", "src/three_phase_trap/averaged_dynamics.py",
                "src/three_phase_trap/validation_field_audit.py", "src/three_phase_trap/effective_potential.py", "src/three_phase_trap/interpolation.py", "src/three_phase_trap/fem_cache.py"]
    pilot_row = next(row for row in rows if row["radius_m"] == length and row["order"] == 40)
    summary = {"status": "implemented_checks_passed_spatial_accuracy_unverified" if checks_passed and sensitivity_passed else "model_b_checks_failed",
               "algorithm_checks_passed": bool(checks_passed), "physical_RF_trajectory_comparison_performed": False,
               "reconstruction_sensitivity_checks_passed": bool(sensitivity_passed),
               "fem_fingerprint": fingerprint, "basis_sha256": protocol["basis_sha256"],
               "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in relevant},
               "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
               "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-fem", "matplotlib")},
               "physical_parameters": pilot, "selected_order": 48, "valid_radius_m": valid_radius,
               "alternative_design_condition": alt_condition, "fixed_point_checks": checks,
               "slow_integrator_checks": trajectory_checks, "pilot_radius_reconstruction_checks": pilot_row,
               "reconstruction_sensitivity_target": .001,
               "spatial_uncertainty": "raw FEM vs fit is a discrepancy, not an absolute FEM error; refined-mesh comparison still required",
               "force_comparison_to_raw_FEM": "No global smooth force is inferred by differentiating discontinuous raw P2 squared fields",
               "holdout_note": "Angles are independent. Some ring radii coincide with fitting rings; this is not an independent mesh."}
    write(output/"summary.json", summary)
    plot(output, rows, fine, coarse, alternate, length)
    report(output, summary, rows)
    progress_path = ROOT/"outputs/abc_validation/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["user_step_4_model_b"] = summary["status"]
    write(progress_path, progress)
    from run_abc_initial_checks import write_report
    write_report(output.parent, protocol, progress)
    print(json.dumps({"status": summary["status"], "checks": checks, "trajectory": trajectory_checks}, indent=2), flush=True)
    if not checks_passed or not sensitivity_passed:
        raise RuntimeError("One or more model B checks failed; inspect summary.json")


def plot(output, rows, fine, coarse, alternate, length):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r["order"] == 40]
    radius = np.array([r["radius_m"] for r in selected])*1000
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for key, label in (("raw_vs_selected_field_relative_rms", "raw FEM vs fit: E"),
                       ("order_vs_selected_force_relative_rms", "N40 vs N48: force"),
                       ("alternative_training_force_relative_rms", "training grids: force")):
        axes[0].loglog(radius, [max(r[key], 1e-16) for r in selected], "o-", label=label)
    axes[0].axhline(.001, color="grey", linestyle=":", label="0.1% reference")
    axes[0].set(xlabel="r (mm)", ylabel="Relative RMS discrepancy", title="Single-mesh consistency")
    axes[0].legend(fontsize=7)
    axes[1].plot(fine.time, fine.states[:, 0]*1e3, label="X: slow model B")
    axes[1].plot(fine.time, fine.states[:, 1]*1e3, label="Y: slow model B")
    axes[1].set(xlabel="t (s)", ylabel="Position (mm)", title="Three estimated slow periods")
    axes[1].legend(fontsize=7)
    energy = abs(fine.energy-fine.energy[0])/max(abs(fine.energy[0]), 1e-300)
    axes[2].semilogy(fine.time, np.maximum(energy, 1e-16), label="Energy drift")
    for other, label in ((coarse, "ODE refinement / R0"), (alternate, "Training variation / R0")):
        axes[2].semilogy(fine.time, np.maximum(np.linalg.norm(other.states[:, :2]-fine.states[:, :2], axis=-1)/length, 1e-16), label=label)
    axes[2].set(xlabel="t (s)", ylabel="Relative discrepancy", title="Model B internal checks")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output/"model_b_checks.png", dpi=180)
    plt.close(fig)


def report(output, summary, rows):
    t = summary["slow_integrator_checks"]
    c = summary["fixed_point_checks"]
    lines = ["# Пункт 4: модель B на основе FEM-потенциала", "",
             "Выполнен пункт 4 краткого списка, то есть этап 3 подробного плана. Модель C ещё не строилась.", "",
             f"Статус: `{summary['status']}`.", "",
             "## Формулы и реализация", "",
             "```text", "E(R,t) = Ec(R) cos(Ωt+φ0) + Es(R) sin(Ωt+φ0), Ω=2πf",
             "ΨB = q² (|Ec|²+|Es|²)/(4mΩ²)",
             "FB = −q² (JcᵀEc + JsᵀEs)/(2mΩ²), Jij=∂Ei/∂Rj", "```", "",
             "Ψ — энергия в джоулях; F — сила в ньютонах. Усреднение по времени при фиксированной R.",
             "Для гладкого поля формула силы точно равна −∇Ψ. Применимость этого ведущего приближения",
             "к полному RF-движению будет проверяться отдельно. DC, трение и гравитация здесь отсутствуют.", "",
             "Квадратуры скалярного потенциала восстановлены по FEM при U0=1 В гармоническими",
             "полиномами всех порядков 0…48. Симметрия принудительно не навязывается,",
             "углового усреднения и замены на r⁶ нет. E и J вычисляются аналитически.",
             "Коэффициенты и область r≤8 мм сохранены в `reconstruction.json`. Эта область —",
             "ограничение вычислений, а не доказанная область точности FEM. Наружу модель не экстраполируется.",
             "`RFReconstruction.field` задаёт согласованное полное поле для будущего контроля A_smooth.", "",
             "## Проверки формул и интегратора", "",
             f"- Квадратуры против прежней точной трёхфазной формулы: {c['raw_quadratures_vs_existing_exact_average_relative_max']:.3g}.",
             f"- Численное усреднение исходного FEM по периоду: {c['raw_numerical_vs_existing_exact_average_relative_max']:.3g}.",
             f"- Ψ восстановленного поля против численного фазового интеграла: {c['reconstructed_psi_vs_phase_integral_relative_max']:.3g}.",
             f"- Сила против конечных разностей Ψ: {c['force_vs_energy_gradient'][-1]['relative_max_on_fixed_force_scale']:.3g} на общем фиксированном масштабе силы.",
             f"- Интегрирование B на {t['duration_s']:.8g} с: три оценочных медленных периода по {t['estimated_slow_period_s']:.8g} с.",
             f"- Максимальный относительный дрейф энергии: {t['relative_energy_drift_max']:.3g}.",
             f"- Изменение координат при уточнении ODE: RMS={t['ode_relative_position_rms']:.3g}, максимум={t['ode_relative_position_max']:.3g}, нормировка R0=1 мм.",
             f"- Смена обучающих радиусов/углов: RMS координат={t['alternative_training_relative_position_rms']:.3g}, максимум={t['alternative_training_relative_position_max']:.3g}.", "",
             "Интегратор работает с безразмерным состоянием (R/L, V·duration/L), затем возвращает SI.",
             "Настройки coarse/fine, масштабы и число вычислений сохранены в summary.json.",
             "Неуспешное решение вызывает ошибку; выход из круга даёт отдельный статус domain_exit и",
             "точное время события. Граничное продолжение силы применяется только к пробным стадиям RK.", "",
             "## Пространственная неопределённость", "",
             "Точки проверки имеют независимые углы; некоторые радиусы совпадают с обучающими.",
             "Ни одна из этих проверок не заменяет сравнение разных FEM-сеток.", "",
             f"Проверки чувствительности восстановления пройдены: {summary['reconstruction_sensitivity_checks_passed']}. Ориентир 0,1% для силы/Ψ на контрольных кольцах и RMS координат пилота.", "",
             "| r, мм | FEM–fit E, % | FEM–fit Ψ, % | N40–48 F, % | другая подгонка F, % |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["order"] == 40:
            lines.append(f"| {row['radius_m']*1e3:g} | {row['raw_vs_selected_field_relative_rms']*100:.6g} | {row['raw_vs_selected_psi_relative_rms']*100:.6g} | {row['order_vs_selected_force_relative_rms']*100:.6g} | {row['alternative_training_force_relative_rms']*100:.6g} |")
    lines += ["", "FEM–fit — расхождение двух представлений одной сетки, а не измеренная абсолютная ошибка.",
              "N40–48 и другая подгонка показывают устойчивость выбранного восстановления; это также",
              "не оценка истинной ошибки FEM. Около центра локальная относительная ошибка растёт при малом",
              "абсолютном поле. Производная сырого поэлементного Ψ не использована как глобальная сила.", "",
              "Численный бюджет 0,1% относится к ошибке сравниваемых координат. Локальное расхождение",
              "E или F с этим ориентиром — сигнал для аудита, а не автоматический процент ошибки траектории.",
              "Физические траектории A и A_smooth пока не рассчитаны. Поэтому точность усреднения",
              "и соответствие статье пока не подтверждены. Следующие этапы — C, общие начальные условия,",
              "контроль A–A_smooth и A_smooth–B; перед итоговым выводом нужна пространственная сходимость.", "",
              "## Воспроизведение и файлы", "", "```powershell", "C:/Python314/python.exe scripts/run_abc_model_b.py", "```", "",
              "- `settings.json`: настройки до расчёта; исходный protocol.json не изменён.",
              "- `reconstruction.json`: готовое поле U0=1 В, fingerprint, нормировка и домен.",
              "- `alternative_reconstruction.json`: коэффициенты второй независимой выборки подгонки.",
              "- `summary.json`: параметры, версии, SHA исходников, проверки и ограничения.",
              "- `reconstruction_checks.csv`: все порядки и радиусы.",
              "- `holdout_samples.npz`: исходное/восстановленное поле, Ψ и гладкая сила.",
              "- `slow_trajectory_checks.npz`: t, состояния B, уточнение ODE, другая подгонка, энергия.",
              "- `model_b_checks.png`: пространственные и динамические проверки.", "",
              "Исходный FEM-кэш, сетка и переносимый архив не пересчитывались и не перезаписывались."]
    (output/"MODEL_B_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
