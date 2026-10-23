"""Reproducible end-to-end analyses for the 12-electrode trap."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .config import ParticleConfig, ProjectConfig
from .dynamics import TrajectoryResult, integrate_trajectory
from .effective_potential import classify_center, effective_potential_from_mean_square, mean_square_field_from_basis, time_average_field_squared
from .fields import field_at_time, potential_at_time
from .fem import FEMBasisSolutions, element_electric_field, solve_instantaneous, superpose_field, superpose_potential
from .fem_cache import load_or_build_fem_state
from .geometry import build_geometry
from .interpolation import FEMInterpolator
from .mesh import build_mesh
from .multipoles import estimate_radial_exponent, fit_multipoles, fit_rotation, harmonic_orientation, polar_sample_points
from .stability import run_stability_scan, write_stability_csv
from .visualization import save_phase_voltage_diagnostics, save_potential_animation, save_static_snapshots, save_trajectory_plots
from .voltages import phase_voltages, voltage_matrix


def make_state(config: ProjectConfig, project_root: Path | None = None) -> tuple[object, object, FEMBasisSolutions, FEMInterpolator]:
    """Load the exact fingerprinted P2 cache, creating it only when absent."""
    root = Path.cwd() if project_root is None else Path(project_root)
    return load_or_build_fem_state(config, root)


def ensure_output_dir(config: ProjectConfig, project_root: Path) -> Path:
    output_dir = config.output_dir(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_fem(config: ProjectConfig, project_root: Path, state: tuple[object, object, FEMBasisSolutions, FEMInterpolator] | None = None) -> dict[str, object]:
    output_dir = ensure_output_dir(config, project_root)
    geometry, trap_mesh, solutions, interpolator = state or make_state(config, project_root)
    times = np.arange(7, dtype=float) / 12.0 * config.rf.period
    rows = []
    for t_over_T, time in zip(np.arange(7, dtype=float) / 12.0, times):
        phase = phase_voltages(float(time), config.rf.U0, config.rf.frequency)
        direct = solve_instantaneous(trap_mesh, phase)
        basis = superpose_potential(solutions, phase)
        difference = direct - basis
        direct_field = element_electric_field(trap_mesh, direct, basis=solutions.basis)
        basis_field = superpose_field(solutions, phase)
        field_difference = direct_field - basis_field
        rms = float(np.sqrt(np.mean(difference**2)))
        direct_rms = float(np.sqrt(np.mean(direct**2)))
        rows.append((t_over_T, float(np.max(np.abs(difference))), rms, rms / max(direct_rms, 1.0e-30), float(np.max(np.linalg.norm(field_difference, axis=1))), float(np.sqrt(np.mean(field_difference**2)))))
    np.savetxt(output_dir / "direct_fem_validation.csv", np.asarray(rows), delimiter=",", header="t_over_T,max_abs_potential_error_V,rms_potential_error_V,relative_rms_potential_error,max_abs_field_error_V_per_m,rms_field_error_V_per_m", comments="")
    summary = {"mesh_vertices": int(len(trap_mesh.points)), "mesh_triangles": int(len(trap_mesh.triangles)), "max_potential_error": float(max(row[1] for row in rows)), "max_field_error": float(max(row[4] for row in rows))}
    (output_dir / "fem_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"geometry": geometry, "trap_mesh": trap_mesh, "solutions": solutions, "interpolator": interpolator, "summary": summary}


def run_visuals(config: ProjectConfig, project_root: Path, state: tuple[object, object, FEMBasisSolutions, FEMInterpolator] | None = None) -> dict[str, object]:
    output_dir = ensure_output_dir(config, project_root)
    geometry, _, _, interpolator = state or make_state(config, project_root)
    voltage_summary = save_phase_voltage_diagnostics(output_dir, config.rf)
    snapshots = save_static_snapshots(output_dir, interpolator, geometry, config.rf, config.simulation.grid_size)
    animation = save_potential_animation(output_dir, interpolator, geometry, config.rf, config.simulation.frames_per_period, config.simulation.animation_periods, max(81, config.simulation.grid_size // 2))
    return {"voltage_summary": voltage_summary, "animation": animation, "snapshots": snapshots}


def _multipole_samples(interpolator: FEMInterpolator, geometry, config: ProjectConfig) -> tuple[np.ndarray, np.ndarray, float]:
    maximum_radius = geometry.electrode_orbit_radius - geometry.electrode_radius
    radii = np.linspace(0.00055, min(0.0048, 0.55 * maximum_radius), 10)
    angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    points = polar_sample_points(radii, angles)
    return points, angles, float(np.max(radii))


def run_multipoles(config: ProjectConfig, project_root: Path, state: tuple[object, object, FEMBasisSolutions, FEMInterpolator] | None = None) -> dict[str, object]:
    output_dir = ensure_output_dir(config, project_root)
    geometry, _, _, interpolator = state or make_state(config, project_root)
    points, _, reference_radius = _multipole_samples(interpolator, geometry, config)
    times = np.linspace(0.0, config.rf.period, 25, endpoint=False)
    rows = []
    coefficient_by_time: list[dict[int, object]] = []
    orders = tuple(range(1, 9))
    for time in times:
        potential = interpolator.potential(points, float(time), config.rf.U0, config.rf.frequency)
        coefficients = fit_multipoles(points, potential, orders, reference_radius)
        coefficient_by_time.append(coefficients)
        for order in orders:
            item = coefficients[order]
            rows.append((time / config.rf.period, order, item.a, item.b, item.amplitude, item.relative_amplitude))
    np.savetxt(output_dir / "multipole_spectrum.csv", np.asarray(rows), delimiter=",", header="t_over_T,order,a_V,b_V,amplitude_V,relative_amplitude", comments="")
    average_amplitudes = {order: float(np.mean([coefficients[order].amplitude for coefficients in coefficient_by_time])) for order in orders}
    dominant_order = max(average_amplitudes, key=average_amplitudes.get)
    maximum = average_amplitudes[dominant_order]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar(list(average_amplitudes), [average_amplitudes[o] / max(maximum, 1e-30) for o in average_amplitudes], color="#4c78a8")
    ax.set(xlabel="Multipole order m", ylabel="Mean relative amplitude", title="FEM-fitted multipole spectrum")
    ax.set_xticks(list(average_amplitudes))
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "multipole_spectrum.png", dpi=150)
    plt.close(fig)
    return {"dominant_order": int(dominant_order), "average_amplitudes": average_amplitudes, "reference_radius": reference_radius, "points": points, "coefficient_by_time": coefficient_by_time}


def run_rotation(config: ProjectConfig, project_root: Path, multipole_result: dict[str, object], state: tuple[object, object, FEMBasisSolutions, FEMInterpolator] | None = None) -> dict[str, object]:
    output_dir = ensure_output_dir(config, project_root)
    geometry, _, _, interpolator = state or make_state(config, project_root)
    points = multipole_result["points"]
    reference_radius = float(multipole_result["reference_radius"])
    dominant_order = int(multipole_result["dominant_order"])
    times = np.linspace(0.0, 2.0 * config.rf.period, 97, endpoint=False)
    orientations = []
    amplitudes = []
    for time in times:
        potential = interpolator.potential(points, float(time), config.rf.U0, config.rf.frequency)
        coefficient = fit_multipoles(points, potential, (dominant_order,), reference_radius)[dominant_order]
        orientations.append(harmonic_orientation(coefficient))
        amplitudes.append(coefficient.amplitude)
    # The physical orientation is periodic modulo 2π/m.  Unwrap the harmonic
    # phase first, then divide by m; unwrapping the physical angle directly
    # would miss the shorter π/2 branch period for m=4.
    phase_fit = fit_rotation(times, np.asarray(orientations) * dominant_order)
    fit = dict(phase_fit)
    fit["angular_rate"] = phase_fit["angular_rate"] / dominant_order
    fit["intercept"] = phase_fit["intercept"] / dominant_order
    fit["rate_uncertainty"] = phase_fit["rate_uncertainty"] / dominant_order
    fit["residual_rms"] = phase_fit["residual_rms"] / dominant_order
    fit["unwrapped_orientation"] = phase_fit["unwrapped_orientation"] / dominant_order
    fit["residuals"] = phase_fit["residuals"] / dominant_order
    unwrapped = fit["unwrapped_orientation"]
    residuals = fit["residuals"]
    rows = np.column_stack((times / config.rf.period, times, orientations, unwrapped, amplitudes, residuals))
    np.savetxt(output_dir / "multipole_rotation.csv", rows, delimiter=",", header="t_over_T,time_s,raw_orientation_rad,unwrapped_orientation_rad,amplitude_V,residual_rad", comments="")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(times / config.rf.period, unwrapped, label="Fitted orientation", color="#4c78a8")
    expected = fit["intercept"] + config.rf.omega / dominant_order * times
    ax.plot(times / config.rf.period, expected, "--", label=f"Ω/m reference (m={dominant_order})", color="#e45756")
    ax.set(xlabel="Time / RF period", ylabel="Physical orientation (rad)", title=f"Dominant m={dominant_order} multipole rotation")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "multipole_rotation.png", dpi=150)
    plt.close(fig)
    return {"dominant_order": dominant_order, "fit": fit, "orientations": np.asarray(orientations), "times": times}


def run_radial_scaling(config: ProjectConfig, project_root: Path, state: tuple[object, object, FEMBasisSolutions, FEMInterpolator] | None = None) -> dict[str, object]:
    output_dir = ensure_output_dir(config, project_root)
    geometry, _, _, interpolator = state or make_state(config, project_root)
    radii = np.linspace(0.0007, min(0.0048, 0.56 * (geometry.electrode_orbit_radius - geometry.electrode_radius)), 18)
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    times = np.linspace(0.0, config.rf.period, 16, endpoint=False)
    rows = []
    mean_field = []
    for radius in radii:
        points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        samples = []
        for time in times:
            samples.append(np.linalg.norm(interpolator.field(points, float(time), config.rf.U0, config.rf.frequency), axis=1))
        magnitude = np.asarray(samples)
        mean_value = float(np.mean(magnitude))
        mean_field.append(mean_value)
        rows.append((radius, mean_value, float(np.std(magnitude))))
    exponent, log_residual = estimate_radial_exponent(radii, np.asarray(mean_field))
    np.savetxt(output_dir / "radial_field_scaling.csv", np.asarray(rows), delimiter=",", header="radius_m,mean_field_magnitude_V_per_m,field_magnitude_std_V_per_m", comments="")
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.loglog(radii, mean_field, "o", label="FEM samples")
    fit_line = np.exp(np.polyfit(np.log(radii), np.log(mean_field), 1)[1]) * radii**exponent
    ax.loglog(radii, fit_line, "--", label=f"fit p={exponent:.3f}")
    ax.set(xlabel="Radius (m)", ylabel="Mean |E| (V/m)", title="Near-center electric-field radial scaling")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "radial_field_scaling.png", dpi=150)
    plt.close(fig)
    return {"exponent": exponent, "log_residual": log_residual, "radii": radii, "mean_field": np.asarray(mean_field)}


def run_effective_potential(config: ProjectConfig, project_root: Path, state: tuple[object, object, FEMBasisSolutions, FEMInterpolator] | None = None) -> dict[str, object]:
    output_dir = ensure_output_dir(config, project_root)
    geometry, _, solutions, interpolator = state or make_state(config, project_root)
    display_radius = min(0.020, geometry.outer_domain_radius * 0.84)
    axis, xx, yy, mask, basis_potential, basis_field = __import__("three_phase_trap.visualization", fromlist=["prepare_grid"]).prepare_grid(interpolator, display_radius, config.simulation.grid_size)
    basis_fields = {phase: basis_field[i] for i, phase in enumerate(("A", "B", "C"))}
    mean_squared = mean_square_field_from_basis(basis_fields, config.rf.U0)
    effective = effective_potential_from_mean_square(mean_squared, config.particle.charge, config.particle.mass, config.rf.omega)
    classification, hessian = classify_center(effective, axis, axis)
    radial_radii = np.linspace(0.0, min(0.007, display_radius * 0.75), 70)
    radial_angles = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
    radial_values = []
    for angle in radial_angles:
        points = np.column_stack((radial_radii * np.cos(angle), radial_radii * np.sin(angle)))
        points[0] = [1.0e-12, 0.0]
        mean_sq = time_average_field_squared(interpolator, points, config.rf.U0, config.rf.frequency, samples=120)
        eff = effective_potential_from_mean_square(mean_sq, config.particle.charge, config.particle.mass, config.rf.omega)
        radial_values.append(eff)
    radial_values = np.asarray(radial_values)
    np.savetxt(output_dir / "effective_potential_radial.csv", np.column_stack((radial_radii, radial_values.T)), delimiter=",", header="radius_m,angle_0_deg,angle_45_deg,angle_90_deg,angle_135_deg", comments="")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    image = axes[0].imshow(effective, origin="lower", extent=(axis[0], axis[-1], axis[0], axis[-1]), cmap="magma")
    axes[0].set_title(f"Effective RF potential: {classification}")
    axes[0].set(xlabel="x (m)", ylabel="y (m)")
    axes[0].set_aspect("equal")
    from .visualization import draw_electrodes
    draw_electrodes(axes[0], geometry)
    fig.colorbar(image, ax=axes[0], label="Effective potential (J)")
    center_index = int(np.argmin(np.abs(axis)))
    axes[1].plot(axis, effective[center_index], label="horizontal")
    axes[1].plot(axis, effective[:, center_index], label="vertical")
    axes[1].set(xlabel="Coordinate (m)", ylabel="Effective potential (J)", title="Center cuts")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    for angle, values in zip(radial_angles, radial_values):
        axes[2].plot(radial_radii, values, label=f"{np.degrees(angle):.0f}°")
    axes[2].set(xlabel="Radius (m)", ylabel="Effective potential (J)", title="Radial cuts")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.savefig(output_dir / "effective_potential.png", dpi=150)
    plt.close(fig)
    return {"classification": classification, "hessian_eigenvalues": hessian, "axis": axis, "effective": effective, "mean_squared": mean_squared, "radial_radii": radial_radii, "radial_values": radial_values}


def _write_trajectory_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    with (output_dir / "trajectory_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_trajectory(config: ProjectConfig, project_root: Path, state: tuple[object, object, FEMBasisSolutions, FEMInterpolator] | None = None) -> dict[str, object]:
    output_dir = ensure_output_dir(config, project_root)
    _, _, _, interpolator = state or make_state(config, project_root)
    base = integrate_trajectory(interpolator, config.particle, config.rf.U0, config.rf.frequency, periods=config.simulation.trajectory_periods, escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol)
    damped_gamma = max(config.particle.gamma, 0.02 * config.particle.mass * config.rf.omega)
    damped_particle = replace(config.particle, gamma=damped_gamma)
    damped = integrate_trajectory(interpolator, damped_particle, config.rf.U0, config.rf.frequency, periods=config.simulation.trajectory_periods, escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol)
    save_trajectory_plots(output_dir, base, "base gamma")
    rows = []
    for label, particle, result in (("base", config.particle, base), ("damped", damped_particle, damped)):
        rows.append({"case": label, "gamma": particle.gamma, "max_radius_m": result.max_radius, "final_radius_m": result.final_radius, "escaped": int(result.escaped), "bounded": int(result.bounded), "simulation_duration_s": result.simulation_duration, "periods_simulated": result.periods_simulated, "termination": result.termination})
    _write_trajectory_summary(output_dir, rows)
    return {"base": base, "damped": damped, "summary": rows}


def run_stability(config: ProjectConfig, project_root: Path, state: tuple[object, object, FEMBasisSolutions, FEMInterpolator] | None = None) -> dict[str, object]:
    output_dir = ensure_output_dir(config, project_root)
    _, _, _, interpolator = state or make_state(config)
    rows = run_stability_scan(interpolator, config)
    write_stability_csv(rows, output_dir / "stability_scan.csv")
    # A point is shown bounded only when every tested initial position for the
    # same (U0, f_rf) pair remains below the explicit escape radius.
    matrix = np.full((len(config.stability.frequency_values), len(config.stability.U0_values)), np.nan)
    for j, U0 in enumerate(config.stability.U0_values):
        for i, frequency in enumerate(config.stability.frequency_values):
            pair = [row for row in rows if row.U0 == U0 and row.f_rf == frequency]
            matrix[i, j] = float(all(row.bounded for row in pair))
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1, extent=(min(config.stability.U0_values), max(config.stability.U0_values), min(config.stability.frequency_values), max(config.stability.frequency_values)))
    fig.colorbar(image, ax=ax, ticks=[0, 1], label="All tested starts bounded (1=yes)")
    ax.set(xlabel="U0 (V)", ylabel="RF frequency (Hz)", title="Exploratory full-dynamics stability scan")
    fig.savefig(output_dir / "stability_map.png", dpi=150)
    plt.close(fig)
    return {"rows": rows, "matrix": matrix}


def run_mesh_convergence(config: ProjectConfig, project_root: Path) -> dict[str, object]:
    if config.mesh.backend == "gmsh":
        raise ValueError("Gmsh convergence requires separately generated meshes/configurations at each resolution.")
    output_dir = ensure_output_dir(config, project_root)
    resolutions = np.array([1.35, 1.0, 0.78]) * config.mesh.resolution
    rows = []
    geometry = build_geometry(config.geometry.electrode_count, config.geometry.electrode_orbit_radius, config.geometry.electrode_radius, config.geometry.outer_domain_radius)
    points = polar_sample_points(np.linspace(0.001, 0.004, 8), np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False))
    for resolution in resolutions:
        trap_mesh = build_mesh(geometry, float(resolution), config.mesh.boundary_points_per_electrode, config.mesh.outer_boundary_points, config.mesh.interior_seed)
        solutions = solve_basis(trap_mesh)
        interpolator = FEMInterpolator(solutions)
        coefficients = fit_multipoles(points, interpolator.potential(points, 0.0, config.rf.U0, config.rf.frequency), tuple(range(1, 7)), 0.004)
        rows.append((resolution, len(trap_mesh.points), len(trap_mesh.triangles), coefficients[4].amplitude))
    np.savetxt(output_dir / "mesh_convergence.csv", np.asarray(rows), delimiter=",", header="resolution_m,vertices,triangles,m4_amplitude_at_reference_V", comments="")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot([row[1] for row in rows], [row[3] for row in rows], "o-")
    ax.set(xlabel="Mesh vertices", ylabel="m=4 amplitude at reference radius (V)", title="Basic mesh-convergence check")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "mesh_convergence.png", dpi=150)
    plt.close(fig)
    return {"rows": rows}


def write_final_report(output_dir: Path, config: ProjectConfig, fem_result: dict[str, object], multipoles: dict[str, object], rotation: dict[str, object], radial: dict[str, object], effective: dict[str, object], trajectories: dict[str, object], stability: dict[str, object], convergence: dict[str, object]) -> Path:
    amps = multipoles["average_amplitudes"]
    relative = {order: value / max(amps[multipoles["dominant_order"]], 1e-30) for order, value in amps.items()}
    rotation_fit = rotation["fit"]
    base: TrajectoryResult = trajectories["base"]
    damped: TrajectoryResult = trajectories["damped"]
    convergence_rows = convergence["rows"]
    bounded_rows = [row for row in stability["rows"] if row.bounded]
    if bounded_rows:
        bounded_examples = "; ".join(f"U0={row.U0:g} V, f={row.f_rf:g} Hz, start=({row.initial_x:g},{row.initial_y:g}) m" for row in bounded_rows)
    else:
        bounded_examples = "none"
    convergence_relative_change = (max(row[3] for row in convergence_rows) - min(row[3] for row in convergence_rows)) / max(np.mean([row[3] for row in convergence_rows]), 1e-30)
    lines = [
        "# Three-phase 12-electrode trap simulation report", "",
        "## Project", "", "This report is generated from the independent `three-phase-12-electrode-trap` implementation.", "",
        "## Numerical methods", "", "The model uses a sampled circular free-space domain, linear triangular scikit-fem Laplace solves with a grounded far outer boundary, three reusable 1-V basis solutions, piecewise-linear potential interpolation, harmonic least-squares multipole fitting, exact three-phase time averaging, and `scipy.integrate.solve_ivp` for full RF trajectories.", "",
        "## Results", "",
        f"- Dominant average multipole: **m={multipoles['dominant_order']}**.",
        "- Mean relative amplitudes (normalized to the dominant order): " + ", ".join(f"m={order}: {relative[order]:.4g}" for order in range(1, 7)) + ".",
        f"- Measured physical orientation rate: {rotation_fit['angular_rate']:.6g} rad/s ± {rotation_fit['rate_uncertainty']:.3g}; Ω/m reference: {config.rf.omega / int(multipoles['dominant_order']):.6g} rad/s.",
        f"- Orientation-fit RMS residual: {rotation_fit['residual_rms']:.4g} rad.",
        f"- Radial field exponent: p={radial['exponent']:.6g}; log-fit RMS residual={radial['log_residual']:.4g}.",
        f"- Effective-potential center classification: **{effective['classification']}**; Hessian eigenvalues of the plotted proxy: {effective['hessian_eigenvalues']}.",
        f"- Base full-dynamics trajectory: {'bounded' if base.bounded else 'escaped'}, r_max={base.max_radius:.6g} m, simulated {base.periods_simulated:.3g} RF periods, gamma={config.particle.gamma:.6g}.",
        f"- Damped comparison: {'bounded' if damped.bounded else 'escaped'}, r_max={damped.max_radius:.6g} m, gamma={damped_gamma_text(damped, trajectories)}.",
        f"- Stability scan bounded cells (all tested initial positions): {sum(bool(row) for row in np.asarray(stability['matrix']).ravel())}/{np.asarray(stability['matrix']).size}.",
        f"- Individual bounded starts in the scan (not a global-stability claim): {bounded_examples}.",
        f"- Undamped base result: {'bounded' if base.bounded else 'escaped'}; damping comparison: {'bounded' if damped.bounded else 'escaped'}.",
        f"- Mesh convergence: m=4 amplitude at the reference radius ranges from {min(row[3] for row in convergence_rows):.6g} to {max(row[3] for row in convergence_rows):.6g} V; fractional range {convergence_relative_change:.3g}.", "",
        "These numerical results describe this discretization and parameter set. A bounded trajectory is reported only when the entire integrated path stays below `escape_radius`; damping-assisted localization is listed separately from the gamma=0 case.", "",
        "## Main limitations", "", "The circular electrode surfaces are represented by sampled boundaries and a Delaunay triangulation; the outer boundary is finite and grounded; P1 element fields are piecewise constant; interpolation is not extrapolated into electrodes or outside the mesh; the parameter scan is intentionally small; and trajectory conclusions depend on the chosen particle charge-to-mass ratio, RF frequency, duration, tolerances, and escape criterion.", "",
    ]
    path = output_dir / "final_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def damped_gamma_text(damped: TrajectoryResult, trajectories: dict[str, object]) -> str:
    rows = trajectories.get("summary", [])
    for row in rows:
        if row.get("case") == "damped":
            return f"{float(row['gamma']):.6g}"
    return "unknown"


def run_all(config: ProjectConfig, project_root: Path) -> dict[str, object]:
    state = make_state(config, project_root)
    fem = run_fem(config, project_root, state)
    run_visuals(config, project_root, state)
    multipoles = run_multipoles(config, project_root, state)
    rotation = run_rotation(config, project_root, multipoles, state)
    radial = run_radial_scaling(config, project_root, state)
    effective = run_effective_potential(config, project_root, state)
    trajectories = run_trajectory(config, project_root, state)
    stability = run_stability(config, project_root, state)
    convergence = run_mesh_convergence(config, project_root)
    report = write_final_report(ensure_output_dir(config, project_root), config, fem, multipoles, rotation, radial, effective, trajectories, stability, convergence)
    return {"fem": fem, "multipoles": multipoles, "rotation": rotation, "radial": radial, "effective": effective, "trajectories": trajectories, "stability": stability, "convergence": convergence, "report": report}
