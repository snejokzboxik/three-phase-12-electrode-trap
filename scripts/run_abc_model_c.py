"""User step 5 (detailed stage 4): field-calibrated analytical octupole C.

No mesh/FEM rebuild and no physical RF trajectory comparison is performed.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from three_phase_trap.config import load_config
from three_phase_trap.fem_cache import load_fem_cache, fem_fingerprint
from three_phase_trap.rf_reconstruction import RFReconstruction
from three_phase_trap.averaged_dynamics import AveragedRFModel
from three_phase_trap.octupole_model import OctupoleModel, octupole_ring_coefficients
from three_phase_trap.validation_field_audit import rf_potential_quadratures


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = ROOT/"outputs/abc_validation/model_c"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT/"outputs/abc_validation/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pilot = protocol["pilot"]
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    fingerprint = fem_fingerprint(config)
    if fingerprint != protocol["fem_fingerprint"]:
        raise ValueError("FEM fingerprint differs from the original protocol")
    b_path = ROOT/"outputs/abc_validation/model_b/reconstruction.json"
    b_alt_path = b_path.with_name("alternative_reconstruction.json")
    reconstruction = RFReconstruction.load(b_path, fingerprint)
    b = AveragedRFModel(reconstruction, pilot["charge_C"], pilot["mass_kg"], pilot["U0_V"], pilot["frequency_Hz"])
    b_alt = AveragedRFModel(RFReconstruction.load(b_alt_path, fingerprint), b.charge, b.mass, b.amplitude, b.frequency)
    settings = {"user_step": 5, "detailed_plan_stage": 4,
                "calibration_radii_m": [.00125, .00225, .00325, .00425, .00525, .00625],
                "calibration_angles": 768, "calibration_angle_offset": .187,
                "coefficient_estimator": "equal-weight complex mean of normalized n=4 co-rotating Fourier coefficients over all six circles",
                "calibration_input": "raw scalar P2 FEM potential quadratures at U0=1 V, no trajectory fitting",
                "holdout_angles": 721, "holdout_angle_offset": .613,
                "holdout_radii_m": np.unique(np.round(np.r_[np.geomspace(.000025, .0005, 20), .0001, .00025, np.arange(5, 81)*.0001], 12)).tolist(),
                "local_shape_tolerance": .01,
                "shape_tolerance_note": "1% maximum local potential and vector-force error relative to pure C on each circle; independent of the future 1% trajectory RMS criterion",
                "coefficient_stability_tolerance": .001,
                "counter_rotating_fraction_tolerance": 1e-6,
                "force_fd_tolerance": 1e-7,
                "fixed_error_scale_radius_m": .001,
                "domain_radius_m": reconstruction.valid_radius,
                "range_note": "sampled radii/angles only; no continuum certificate or FEM mesh-convergence claim"}
    write(output/"settings.json", settings)
    print("Verifying and loading original FEM cache; extracting n=4 on six circles...", flush=True)
    _, _, _, interp = load_fem_cache(config, ROOT)
    angles = (np.arange(settings["calibration_angles"])+.187)*2*np.pi/settings["calibration_angles"]
    calibration = []
    desired, reverse = [], []
    for radius in settings["calibration_radii_m"]:
        points = radius*np.column_stack((np.cos(angles), np.sin(angles)))
        pc, ps = rf_potential_quadratures(interp, points, 1.)
        cc, ss, dm, dp = octupole_ring_coefficients(pc, ps, angles, radius)
        desired.append(dm)
        reverse.append(dp)
        calibration.append({"radius_m": radius, "c4_cosine_real_per_m4": float(cc.real), "c4_cosine_imag_per_m4": float(cc.imag),
                            "c4_sine_real_per_m4": float(ss.real), "c4_sine_imag_per_m4": float(ss.imag),
                            "desired_real_per_m4": float(dm.real), "desired_imag_per_m4": float(dm.imag),
                            "C4_per_volt_per_m4": float(abs(dm)), "delta4_rad": float(-np.angle(dm)),
                            "counter_rotating_fraction": float(abs(dp)/abs(dm))})
    d = np.mean(desired)
    coefficient, delta = float(abs(d)), float(-np.angle(d))
    c = OctupoleModel(coefficient, delta, b.charge, b.mass, b.amplitude, b.frequency, reconstruction.valid_radius)
    stability = float(np.max(abs(np.array(desired)-d))/abs(d))
    reverse_fraction = float(np.max(abs(np.array(reverse)))/abs(d))
    for row, dm in zip(calibration, desired):
        row["complex_coefficient_relative_deviation"] = float(abs(dm-d)/abs(d))
    csv_write(output/"calibration_rings.csv", calibration)
    # Independently compare calibration with both full-potential reconstructions.
    def reconstructed_coefficient(rf):
        q = rf.unit_quadratures
        return (q.sine[4]-1j*q.cosine[4])/(2*q.reference_radius**4)
    d_b, d_alt = reconstructed_coefficient(reconstruction), reconstructed_coefficient(b_alt.reconstruction)
    coefficients = {"format_version": 1, "fem_fingerprint": fingerprint, "basis_sha256": protocol["basis_sha256"],
                    "RF_potential_definition": "Phi4=C4*r^4*sin(Omega*t+phi0-4*theta+delta4)",
                    "Omega_definition": "2*pi*voltage_frequency_Hz, not pattern angular speed",
                    "C4_per_volt_per_m4": coefficient, "C4_at_pilot_V_per_m4": c.C4,
                    "delta4_rad": delta, "delta4_equivalence": "modulo 2*pi, C4/U0 positive",
                    "d_minus_per_volt_real_per_m4": float(d.real), "d_minus_per_volt_imag_per_m4": float(d.imag),
                    "B6_at_pilot_J_per_m6": c.B6, "B6_definition": "8*q^2*(U0*C4_per_volt)^2/(m*Omega^2)",
                    "force_definition": "-6*B6*|R|^4*R", "origin_m": [0., 0.], "lower_modes_retained": False,
                    "physical_parameters": pilot, "computational_domain_radius_m": c.valid_radius,
                    "calibration_settings": settings, "coefficient_stability_relative_max": stability,
                    "counter_rotating_fraction_max": reverse_fraction,
                    "calibration_vs_model_b_coefficient_relative": float(abs(d-d_b)/abs(d)),
                    "calibration_vs_alternative_b_coefficient_relative": float(abs(d-d_alt)/abs(d)),
                    "uncertainty_note": "Ring spread measures single-mesh consistency, not absolute physical or FEM error"}
    write(output/"coefficients.json", coefficients)
    print("Comparing B and C on independent radii and angles, including angular errors...", flush=True)
    a = (np.arange(settings["holdout_angles"])+.613)*2*np.pi/settings["holdout_angles"]
    radial = np.column_stack((np.cos(a), np.sin(a)))
    tangential = np.column_stack((-np.sin(a), np.cos(a)))
    radii = np.array(settings["holdout_radii_m"])
    points = np.concatenate([r*radial for r in radii])
    psi_b, psi_c = b.potential(points), c.potential(points)
    force_b, force_c = b.force(points), c.force(points)
    psi_alt, force_alt = b_alt.potential(points), b_alt.force(points)
    psi_scale = float(c.potential([.001, 0.]))
    force_scale = float(np.linalg.norm(c.force([.001, 0.])))
    rows = []
    for i, r in enumerate(radii):
        sl = slice(i*len(a), (i+1)*len(a))
        pb, pc, fb, fc = psi_b[sl], psi_c[sl], force_b[sl], force_c[sl]
        delta_psi = pb-pc
        delta_force = np.linalg.norm(fb-fc, axis=-1)
        cpsi, cf = float(pc[0]), float(np.linalg.norm(fc[0]))
        alternate_psi_error = float(np.max(abs(psi_alt[sl]-pc))/cpsi)
        alternate_force_error = float(np.max(np.linalg.norm(force_alt[sl]-fc, axis=-1))/cf)
        potential_error, force_error = float(np.max(abs(delta_psi))/cpsi), float(np.max(delta_force)/cf)
        rows.append({"radius_m": float(r), "psi_C_J": cpsi, "force_C_N": cf,
                     "potential_relative_rms": float(np.sqrt(np.mean(delta_psi**2))/cpsi),
                     "potential_relative_max": potential_error,
                     "force_relative_rms": float(np.sqrt(np.mean(delta_force**2))/cf),
                     "force_relative_max": force_error,
                     "potential_absolute_error_max_J": float(np.max(abs(delta_psi))),
                     "force_absolute_error_max_N": float(np.max(delta_force)),
                     "potential_error_on_fixed_1mm_scale": float(np.max(abs(delta_psi))/psi_scale),
                     "force_error_on_fixed_1mm_scale": float(np.max(delta_force)/force_scale),
                     "psi_B_angular_peak_to_peak_over_mean": float(np.ptp(pb)/np.mean(pb)),
                     "force_B_tangential_max_over_force_C": float(np.max(abs(np.sum(fb*tangential, axis=-1)))/cf),
                     "alternative_B_potential_relative_max": alternate_psi_error,
                     "alternative_B_force_relative_max": alternate_force_error,
                     "passes_sampled_local_1pct": bool(max(potential_error, force_error) <= .01),
                     "passes_with_both_B_reconstructions": bool(max(potential_error, force_error, alternate_psi_error, alternate_force_error) <= .01)})
    csv_write(output/"radial_errors.csv", rows)
    origin_force = b.force([0., 0.])
    origin = {"psi_B_J": float(b.potential([0., 0.])), "psi_C_J": 0., "force_B_N": origin_force.tolist(), "force_C_N": [0., 0.],
              "potential_error_on_fixed_1mm_scale": float(b.potential([0., 0.])/psi_scale),
              "force_error_on_fixed_1mm_scale": float(np.linalg.norm(origin_force)/force_scale),
              "local_relative_error": "undefined at origin; C potential and force both vanish"}
    # Independent normalization/force checks, including the measured RF phase.
    rng = np.random.default_rng(20261023)
    pts = rng.uniform(-.0045, .0045, (47, 2))
    h = 5e-8
    finite_difference = np.column_stack([-(c.potential(pts+h*np.eye(2)[j])-c.potential(pts-h*np.eye(2)[j]))/(2*h) for j in range(2)])
    analytical_force = c.force(pts)
    scaled_q = c.reconstruction.scaled(c.amplitude)
    norm_p = float(np.max(c.potential(pts)))
    norm_f = float(np.max(np.linalg.norm(analytical_force, axis=-1)))
    phi0, time = .371, .0000023
    phi_c, phi_s = scaled_q.potential_quadratures(pts)
    actual_phi = phi_c*np.cos(c.omega*time+phi0)+phi_s*np.sin(c.omega*time+phi0)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    expected_phi = c.C4*np.sum(pts**2, axis=-1)**2*np.sin(c.omega*time+phi0-4*theta+c.phase)
    checks = {"RF_phase_formula_relative_max": float(np.max(abs(actual_phi-expected_phi))/np.max(abs(expected_phi))),
              "analytic_vs_quadrature_psi_relative_max": float(np.max(abs(c.potential(pts)-scaled_q.pseudo(pts, c.charge, c.mass, c.omega)))/norm_p),
              "analytic_vs_quadrature_force_relative_max": float(np.max(np.linalg.norm(analytical_force-scaled_q.force(pts, c.charge, c.mass, c.omega), axis=-1))/norm_f),
              "analytic_vs_finite_difference_force_relative_max": float(np.max(np.linalg.norm(analytical_force-finite_difference, axis=-1))/norm_f)}
    numerical_checks_passed = (max(checks[k] for k in checks if k != "analytic_vs_finite_difference_force_relative_max") < 1e-12
                               and checks["analytic_vs_finite_difference_force_relative_max"] < settings["force_fd_tolerance"])
    calibration_checks_passed = stability < .001 and reverse_fraction < 1e-6
    passed = [row for row in rows if row["passes_with_both_B_reconstructions"]]
    blocks = []
    for row in rows:
        if row["passes_with_both_B_reconstructions"]:
            if not blocks or blocks[-1]["closed"]:
                blocks.append({"first_radius_m": row["radius_m"], "last_radius_m": row["radius_m"], "closed": False})
            else:
                blocks[-1]["last_radius_m"] = row["radius_m"]
        elif blocks:
            blocks[-1]["closed"] = True
    for block in blocks:
        del block["closed"]
    first_fail_after_pass = next((row for row in rows if passed and row["radius_m"] > passed[0]["radius_m"] and not row["passes_with_both_B_reconstructions"]), None)
    # An isolated analytic n=8 term contributes an isotropic r^14 energy.
    # Cross terms remain in the actual B-C comparison above; this is diagnostic only.
    qb = b.quadratures
    B14 = float(b.charge**2*8**2*(abs(qb.cosine[8])**2+abs(qb.sine[8])**2)/(4*b.mass*b.omega**2*qb.reference_radius**16))
    higher_modes = {"isolated_n8_B14_J_per_m14": B14,
                    "isolated_n8_potential_over_C_at_4mm": B14*.004**8/c.B6,
                    "isolated_n8_force_over_C_at_4mm": (14/6)*B14*.004**8/c.B6,
                    "note": "Isolated n8 term only; actual B retains all cross terms and angular structure"}
    # Preserve all sampled pass/fail points; a single outer limit does not certify a continuous disk.
    files = ["scripts/run_abc_model_c.py", "src/three_phase_trap/octupole_model.py", "src/three_phase_trap/rf_reconstruction.py",
             "src/three_phase_trap/averaged_dynamics.py", "src/three_phase_trap/validation_field_audit.py",
             "src/three_phase_trap/interpolation.py", "src/three_phase_trap/fem_cache.py", "configs/gmsh_70um.toml"]
    summary = {"status": "implemented_spatial_comparison_complete_FEM_and_dynamics_unverified" if numerical_checks_passed and calibration_checks_passed else "model_c_checks_failed",
               "numerical_checks_passed": bool(numerical_checks_passed), "calibration_checks_passed": bool(calibration_checks_passed),
               "checks": checks, "coefficients": coefficients, "origin_check": origin,
               "higher_mode_diagnostics": higher_modes, "sampled_pass_blocks": blocks,
               "sampled_pass_radii_m": [row["radius_m"] for row in passed],
               "sampled_fail_radii_m": [row["radius_m"] for row in rows if not row["passes_with_both_B_reconstructions"]],
               "last_sampled_pass_radius_m": passed[-1]["radius_m"] if passed else None,
               "first_sampled_fail_after_pass": first_fail_after_pass,
               "range_claim": "Local potential/force agreement on sampled circles only; no guarantee of continuum or trajectory error",
               "physical_RF_trajectory_comparison_performed": False, "FEM_mesh_convergence_verified": False,
               "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-fem", "matplotlib")},
               "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
               "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in (protocol_path, b_path, b_alt_path)}}
    write(output/"summary.json", summary)
    np.savez_compressed(output/"holdout_samples.npz", points_m=points, angles_rad=a, radii_m=radii,
                        psi_B_J=psi_b, psi_C_J=psi_c, force_B_N=force_b, force_C_N=force_c,
                        psi_alternative_B_J=psi_alt, force_alternative_B_N=force_alt)
    plot(output, rows)
    report(output, summary, rows)
    progress_path = ROOT/"outputs/abc_validation/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["user_step_5_model_c"] = summary["status"]
    write(progress_path, progress)
    from run_abc_initial_checks import write_report
    write_report(output.parent, protocol, progress)
    print(json.dumps({"status": summary["status"], "C4_per_volt": coefficient, "C4_at_250V": c.C4,
                      "delta4_rad": delta, "B6_at_pilot": c.B6, "ring_stability": stability,
                      "counter_rotating_fraction": reverse_fraction, "checks": checks,
                      "last_sampled_pass_radius_m": summary["last_sampled_pass_radius_m"],
                      "first_sampled_fail_after_pass": first_fail_after_pass}, indent=2), flush=True)
    if not numerical_checks_passed or not calibration_checks_passed:
        raise RuntimeError("Model C checks failed; inspect summary.json")


def plot(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.array([row["radius_m"] for row in rows])*1000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, label in (("potential_relative_max", "Potential max |B-C| / C"), ("force_relative_max", "Vector force max |B-C| / C"),
                       ("psi_B_angular_peak_to_peak_over_mean", "B potential angular span / mean")):
        axes[0].loglog(r, [max(row[key], 1e-16) for row in rows], label=label)
    axes[0].axhline(.01, color="grey", linestyle=":", label="1% local shape threshold")
    axes[0].set(xlabel="r (mm)", ylabel="Relative discrepancy", title="Sampled B-C spatial comparison")
    for key, label in (("potential_error_on_fixed_1mm_scale", "Potential error / C(1 mm)"),
                       ("force_error_on_fixed_1mm_scale", "Force error / |F_C(1 mm)|")):
        axes[1].loglog(r, [max(row[key], 1e-30) for row in rows], label=label)
    axes[1].set(xlabel="r (mm)", ylabel="Error on fixed physical scale", title="Absolute-error normalization near the origin")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output/"model_c_comparison.png", dpi=180)
    plt.close(fig)


def report(output, s, rows):
    c, checks = s["coefficients"], s["checks"]
    lines = ["# Пункт 5: аналитическая октупольная модель C", "",
             "Выполнен пункт 5 краткого списка (этап 4 подробного плана).", f"Статус: `{s['status']}`.", "",
             "## Определение и калибровка", "", "```text", "Φ4 = C4 r⁴ sin(Ωt+φ0−4θ+δ4)",
             "C4 = U0·κ4, Ω=2πf", "ΨC = B6 r⁶, B6=8q²C4²/(mΩ²)", "FC = −6 B6 |R|⁴ R", "```", "",
             "Φ — электрический потенциал в В; Ψ — потенциальная энергия в Дж.",
             "κ4 имеет единицы м⁻⁴; B6 — Дж/м⁶. Ω относится к напряжению, скорость рисунка — Ω/4.",
             "Калибровка выполнена по исходному скалярному P2 FEM-потенциалу при U0=1 В:",
             "шесть окружностей r=1,25; 2,25; 3,25; 4,25; 5,25; 6,25 мм, по 768 углов.",
             "Использовано комплексное среднее нормированных коэффициентов с одинаковым весом окружностей, без весов r⁸.",
             "Траектории для калибровки не использовались. Низшие моды, сдвиг центра и высшие гармоники в C не перенесены.", "",
             "Для Φc=Re(c4(x+iy)⁴), Φs=Re(s4(x+iy)⁴):", "```text",
             "d−=(s4−i c4)/2=κ4 exp(−iδ4), d+=(s4+i c4)/2", "```", "",
             "d− описывает выбранное направление вращения; d+ измеряет примесь обратного вращения.", "",
             f"- κ4 = {c['C4_per_volt_per_m4']:.12g} м⁻⁴.",
             f"- C4 при U0=250 В = {c['C4_at_pilot_V_per_m4']:.12g} В/м⁴.",
             f"- δ4 = {c['delta4_rad']:.12g} рад (по модулю 2π).",
             f"- B6 при m=1e−15 кг, q=1e−12 Кл, f=100 кГц = {c['B6_at_pilot_J_per_m6']:.12g} Дж/м⁶.",
             f"- Максимальное комплексное отклонение между окружностями = {c['coefficient_stability_relative_max']:.4g}.",
             f"- Примесь обратного вращения = {c['counter_rotating_fraction_max']:.4g}.",
             f"- Расхождение коэффициента с восстановлением B = {c['calibration_vs_model_b_coefficient_relative']:.4g}.", "",
             "Разброс на окружностях — проверка согласованности одной сетки, не абсолютная погрешность коэффициента.", "",
             "## Проверки формул", ""]
    lines += [f"- `{key}`: {value:.4g}." for key, value in checks.items()]
    lines += ["", "## Независимое пространственное сравнение B–C", "",
              "На каждом радиусе проверен 721 угол, отличный от углов калибровки и проверки B.",
              "Относительная ошибка Ψ нормирована на ΨC(r), векторной силы — на |FC(r)|.",
              "Для кандидатной области выбран отдельный критерий: максимум обеих ошибок ≤1%",
              "для основного и альтернативного восстановления B. Это локальный критерий формы,",
              "а не обещание ошибки траектории ≤1%. Параметры критерия сохранены до вычисления.", "",
              "| r, мм | max abs(ΔΨ)/ΨC, % | max ‖ΔF‖/‖FC‖, % | угловой размах ΨB / среднее, % |",
              "|---|---:|---:|---:|"]
    for row in rows:
        if row is rows[0] or row is rows[-1] or any(abs(row["radius_m"]-v) < 1e-12 for v in (.0001, .0005, .001, .002, .003, .004, .005, .006, .007)):
            lines.append(f"| {row['radius_m']*1e3:.4g} | {row['potential_relative_max']*100:.6g} | {row['force_relative_max']*100:.6g} | {row['psi_B_angular_peak_to_peak_over_mean']*100:.6g} |")
    for block in s["sampled_pass_blocks"]:
        lines += ["", f"На проверенных кольцах критерий пройден от {block['first_radius_m']*1e3:.6g} до {block['last_radius_m']*1e3:.6g} мм."]
    failure = s["first_sampled_fail_after_pass"]
    if failure:
        lines.append(f"Первое последующее проверенное кольцо с превышением допуска: {failure['radius_m']*1e3:g} мм; max ΔΨ/ΨC={failure['potential_relative_max']*100:.5g}%, max |ΔF|/|FC|={failure['force_relative_max']*100:.5g}%.")
    origin = s["origin_check"]
    h = s["higher_mode_diagnostics"]
    lines += ["", "Малая угловая зависимость сама по себе не гарантирует точность r⁶:",
              "изолированная гармоника RF-потенциала n=8 даёт радиальный вклад Ψ8=B14·r¹⁴.",
              f"При r=4 мм этот вклад составляет {h['isolated_n8_potential_over_C_at_4mm']*100:.5g}% от ΨC и {h['isolated_n8_force_over_C_at_4mm']*100:.5g}% от силы C.",
              "Это объясняет основную поправку к радиальной зависимости в этой области. Полное сравнение выше сохраняет и перекрёстные члены."]
    lines += ["", "Это результаты конечной пространственной выборки; непрерывный диск и точная граница применимости не доказаны.",
              "Полная таблица всех радиусов, RMS/максимумов, абсолютных ошибок, тангенциальной силы и второй подгонки — в radial_errors.csv.",
              "В начале координат ΨC=0 и FC=0, локальные относительные ошибки не определены.",
              f"Для B в центре Ψ={origin['psi_B_J']:.4g} Дж; |F|={np.linalg.norm(origin['force_B_N']):.4g} Н.",
              "Эти малые остатки низших мод не перенесены в C; абсолютные ошибки и нормировка на C(1 мм) сохранены отдельно.", "",
              "## Что это позволяет заключить", "",
              "Проверена замена пространственной структуры гладкого эффективного потенциала B ведущим r⁶-потенциалом C",
              "на указанных точках. Упрощение не является заменой октуполя квадруполем (у квадруполя ведущая Ψ∝r²).",
              "Сходимость FEM по сетке и применимость временного усреднения к полному RF-движению пока не проверены.",
              "Следующий пункт 6 краткого списка — общие начальные условия и выделение медленного движения (этап 5 плана).", "",
              "## Воспроизведение", "", "```powershell", "C:/Python314/python.exe scripts/run_abc_model_c.py", "```", "",
              "Файлы: coefficients.json (коэффициенты, формулы, нормировка, fingerprint), calibration_rings.csv,",
              "radial_errors.csv, holdout_samples.npz, model_c_comparison.png, settings.json, summary.json (версии и SHA).",
              "Исходные FEM-кэш, сетка, архив и результаты B не изменены."]
    (output/"MODEL_C_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
