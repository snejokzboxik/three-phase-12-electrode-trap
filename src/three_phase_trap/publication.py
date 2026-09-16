"""Publication-quality figures and extended localization/filter analyses."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from .config import ProjectConfig
from .dynamics import integrate_trajectory, trajectory_metrics
from .effective_potential import effective_potential_from_mean_square, mean_square_field_from_basis
from .geometry import TrapGeometry
from .interpolation import FEMInterpolator
from .mass_filter import choose_representative_masses, estimate_fwhm, run_mass_frequency_map, run_mass_transmission_scan, write_mass_transmission_csv
from .multipoles import fit_multipoles, fit_rotation, harmonic_orientation, polar_sample_points
from .visualization import PHASE_COLORS, draw_electrodes, prepare_grid


_LOCALIZATION_WORKER_INTERPOLATOR = None
_LOCALIZATION_WORKER_CONFIG = None


def _initialize_localization_worker(config: ProjectConfig) -> None:
    """Build independent read-only FEM state in a spawned worker process."""
    global _LOCALIZATION_WORKER_INTERPOLATOR, _LOCALIZATION_WORKER_CONFIG
    from .pipeline import make_state
    _LOCALIZATION_WORKER_CONFIG = config
    _LOCALIZATION_WORKER_INTERPOLATOR = make_state(config)[3]


def _run_localization_case_worker(arguments):
    stage, periods, U0, frequency, radius, angle_deg, velocity = arguments
    return _localization_case(_LOCALIZATION_WORKER_INTERPOLATOR, _LOCALIZATION_WORKER_CONFIG, stage, periods, U0, frequency, radius, angle_deg, velocity)


def publication_dir(project_root: Path) -> Path:
    path = Path(project_root) / "outputs" / "publication_study"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=320, facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _mm(value: float) -> float:
    return value * 1000.0


def publication_geometry(output_dir: Path, geometry: TrapGeometry, aperture: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    for electrode in geometry.electrodes:
        x, y = map(_mm, electrode.center)
        ax.add_patch(Circle((x, y), _mm(geometry.electrode_radius), facecolor=PHASE_COLORS[electrode.phase], edgecolor="black", linewidth=1.0))
        ax.text(x, y, str(electrode.index), ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.add_patch(Circle((0, 0), _mm(aperture), fill=False, linestyle="--", linewidth=1.2, color="#444444", label="central aperture"))
    ax.scatter([0], [0], color="black", s=18, zorder=4)
    handles = [Patch(facecolor=PHASE_COLORS[p], edgecolor="black", label=f"Phase {p}") for p in ("A", "B", "C")]
    handles.append(Line2D([], [], color="#444444", linestyle="--", label="central aperture"))
    ax.legend(handles=handles, loc="upper left", frameon=True)
    ax.text(0.02, 0.02, "Phase order around orbit:\nABCABCABCABC\n\nU_A = U₀ sin(Ωt)\nU_B = U₀ sin(Ωt − 120°)\nU_C = U₀ sin(Ωt + 120°)", transform=ax.transAxes, fontsize=9, va="bottom", bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"})
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="12-electrode three-phase rotating-field geometry")
    ax.set_aspect("equal")
    limit = _mm(geometry.outer_domain_radius * 0.58)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(alpha=0.22)
    save_figure(fig, output_dir, "fig01_geometry")


def _multipole_orientation(interpolator: FEMInterpolator, geometry: TrapGeometry, config: ProjectConfig, time: float, order: int = 4) -> float:
    radii = np.linspace(0.0007, min(0.0045, 0.5 * (geometry.electrode_orbit_radius - geometry.electrode_radius)), 8)
    points = polar_sample_points(radii, np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False))
    coefficient = fit_multipoles(points, interpolator.potential(points, time, config.rf.U0, config.rf.frequency), (order,), float(radii[-1]))[order]
    return harmonic_orientation(coefficient)


def publication_rotating_potential(output_dir: Path, interpolator: FEMInterpolator, geometry: TrapGeometry, config: ProjectConfig, rotation_fit: dict[str, object] | None = None) -> None:
    axis, _, _, _, basis_potential, _ = prepare_grid(interpolator, min(0.019, geometry.outer_domain_radius * 0.82), 161)
    times = np.arange(8, dtype=float) / 8.0 * config.rf.period
    voltages = np.column_stack([config.rf.U0 * np.sin(2 * np.pi * config.rf.frequency * times + offset) for offset in (0.0, -2 * np.pi / 3, 2 * np.pi / 3)])
    frames = np.einsum("tp,pyx->tyx", voltages, basis_potential)
    norm = Normalize(float(np.nanmin(frames)), float(np.nanmax(frames)))
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.6), constrained_layout=True)
    for index, ax in enumerate(axes.ravel()):
        image = ax.imshow(frames[index], origin="lower", extent=(_mm(axis[0]), _mm(axis[-1]), _mm(axis[0]), _mm(axis[-1])), cmap="coolwarm", norm=norm)
        ax.contour(axis * 1000.0, axis * 1000.0, np.ma.masked_invalid(frames[index]), levels=np.linspace(norm.vmin, norm.vmax, 9)[1:-1], colors="black", linewidths=0.35, alpha=0.7)
        for electrode in geometry.electrodes:
            ax.add_patch(Circle(tuple(_mm(v) for v in electrode.center), _mm(geometry.electrode_radius), facecolor=PHASE_COLORS[electrode.phase], edgecolor="black", linewidth=0.5))
            ax.text(_mm(electrode.center[0]), _mm(electrode.center[1]), str(electrode.index), ha="center", va="center", fontsize=5, color="white")
        if rotation_fit is not None:
            orientation = float(rotation_fit["intercept"]) + float(rotation_fit["angular_rate"]) * times[index]
        else:
            orientation = _multipole_orientation(interpolator, geometry, config, float(times[index]))
        length = _mm(0.0055)
        ax.plot([0, length * np.cos(orientation)], [0, length * np.sin(orientation)], color="black", linewidth=1.5)
        ax.text(0.03, 0.96, f"t/T = {index}/8", transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.75, "pad": 2})
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Potential (V)", shrink=0.92)
    fig.suptitle("Continuous rotation of the dominant m=4 potential structure", fontsize=14)
    save_figure(fig, output_dir, "fig02_rotating_potential")
    # Reuse the same fixed normalization for the animation.
    animation_fig, animation_ax = plt.subplots(figsize=(6.0, 5.5), constrained_layout=True)
    animation_image = animation_ax.imshow(frames[0], origin="lower", extent=(_mm(axis[0]), _mm(axis[-1]), _mm(axis[0]), _mm(axis[-1])), cmap="coolwarm", norm=norm, animated=True)
    animation_ax.set_aspect("equal")
    animation_ax.set(xlabel="x (mm)", ylabel="y (mm)")
    for electrode in geometry.electrodes:
        animation_ax.add_patch(Circle(tuple(_mm(v) for v in electrode.center), _mm(geometry.electrode_radius), facecolor=PHASE_COLORS[electrode.phase], edgecolor="black", linewidth=0.6))
    label = animation_ax.text(0.03, 0.96, "", transform=animation_ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.8})
    def update(index: int):
        animation_image.set_data(frames[index % len(frames)])
        label.set_text(f"t/T = {(index % len(frames)) / 8:.3f}")
        return animation_image, label
    animation = FuncAnimation(animation_fig, update, frames=32, interval=90, blit=False)
    animation.save(output_dir / "rotating_field_publication.gif", writer=PillowWriter(fps=12), dpi=110)
    if shutil.which("ffmpeg"):
        try:
            animation.save(output_dir / "rotating_field_publication.mp4", writer="ffmpeg", dpi=130)
        except Exception:
            pass
    plt.close(animation_fig)


def _radial_effective_data(interpolator: FEMInterpolator, config: ProjectConfig, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    values = []
    for radius in radii:
        points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        fields = {phase: interpolator.field_basis(points, phase) for phase in ("A", "B", "C")}
        mean_squared = mean_square_field_from_basis(fields, config.rf.U0)
        values.append(float(np.mean(effective_potential_from_mean_square(mean_squared, config.particle.charge, config.particle.mass, config.rf.omega))))
    return radii, np.asarray(values)


def publication_effective_potential(output_dir: Path, interpolator: FEMInterpolator, geometry: TrapGeometry, config: ProjectConfig) -> dict[str, float | np.ndarray]:
    axis, _, _, _, _, basis_field = prepare_grid(interpolator, min(0.018, geometry.outer_domain_radius * 0.78), 241)
    mean_squared = mean_square_field_from_basis({phase: basis_field[i] for i, phase in enumerate(("A", "B", "C"))}, config.rf.U0)
    effective = effective_potential_from_mean_square(mean_squared, config.particle.charge, config.particle.mass, config.rf.omega)
    radii = np.linspace(0.00045, min(0.0058, 0.68 * (geometry.electrode_orbit_radius - geometry.electrode_radius)), 42)
    radii, radial_values = _radial_effective_data(interpolator, config, radii)
    reference = 0.004
    scaled = radii / reference
    design = np.column_stack((scaled**6, scaled**8, scaled**10))
    coefficients, *_ = np.linalg.lstsq(design, radial_values, rcond=None)
    fit_values = design @ coefficients
    A6 = float(coefficients[0] / reference**6)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    image = axes[0].imshow(effective, origin="lower", extent=(_mm(axis[0]), _mm(axis[-1]), _mm(axis[0]), _mm(axis[-1])), cmap="magma")
    axes[0].set(xlabel="x (mm)", ylabel="y (mm)", title="Effective RF potential")
    axes[0].set_aspect("equal")
    fig.colorbar(image, ax=axes[0], label="Effective potential (J)")
    zoom = max(0.003, 0.35 * _mm(geometry.electrode_orbit_radius - geometry.electrode_radius))
    center = len(axis) // 2
    half = max(2, int(zoom / _mm(axis[1] - axis[0])))
    axes[1].imshow(effective[center-half:center+half+1, center-half:center+half+1], origin="lower", extent=(-zoom, zoom, -zoom, zoom), cmap="magma")
    axes[1].set(xlabel="x (mm)", ylabel="y (mm)", title="Central flat-bottom zoom")
    axes[1].set_aspect("equal")
    axes[2].plot(_mm(radii), radial_values, "o", ms=3, label="angular average")
    axes[2].plot(_mm(radii), fit_values, "-", label=rf"$A_6 r^6 + A_8 r^8 + A_{{10}}r^{{10}}$; $A_6={A6:.2e}$")
    axes[2].axvspan(0, _mm(geometry.electrode_orbit_radius - geometry.electrode_radius), color="#66a61e", alpha=0.10, label="central free region")
    axes[2].set(xlabel="Radius (mm)", ylabel="Effective potential (J)", title="Higher-order central well")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"Higher-order local minimum: A₆ {'> 0' if A6 > 0 else '< 0'}", fontsize=14)
    save_figure(fig, output_dir, "fig03_effective_potential")
    np.savetxt(output_dir / "effective_potential_fit.csv", np.column_stack((radii, radial_values, fit_values)), delimiter=",", header="radius_m,effective_potential_J,fit_J", comments="")
    return {"A6": A6, "radii": radii, "radial_values": radial_values, "fit_values": fit_values, "effective": effective, "axis": axis}


def _localization_row(result, metrics: dict[str, float | str], U0: float, frequency: float, radius: float, angle_deg: float, velocity: tuple[float, float]) -> dict[str, float | int | str]:
    return {"U0_V": U0, "frequency_Hz": frequency, "initial_radius_m": radius, "initial_angle_deg": angle_deg, "initial_vx_m_per_s": velocity[0], "initial_vy_m_per_s": velocity[1], "max_radius_m": metrics["max_radius_m"], "rms_radius_m": metrics["rms_radius_m"], "median_radius_m": metrics["median_radius_m"], "final_radius_m": metrics["final_radius_m"], "escape_time_s": metrics["escape_time_s"], "periods_survived": metrics["periods_survived"], "max_kinetic_energy_J": metrics["max_kinetic_energy_J"], "micromotion_amplitude_m": metrics["micromotion_amplitude_m"], "classification": metrics["classification"], "bounded": int(result.bounded), "escaped": int(result.escaped)}


def _localization_key(row: dict[str, object]) -> tuple[str, ...]:
    return (str(row["stage"]), f"{float(row['U0_V']):.12g}", f"{float(row['frequency_Hz']):.12g}", f"{float(row['initial_radius_m']):.12g}", f"{float(row['initial_angle_deg']):.12g}", f"{float(row['initial_vx_m_per_s']):.12g}", f"{float(row['initial_vy_m_per_s']):.12g}")


def _read_checkpoint(path: Path) -> dict[tuple[str, ...], dict[str, object]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        # Ignore incompatible legacy rows rather than silently treating an old
        # non-checkpointed scan as a completed staged case.
        if "stage" not in row:
            continue
        for key in ("U0_V", "frequency_Hz", "initial_radius_m", "initial_angle_deg", "initial_vx_m_per_s", "initial_vy_m_per_s", "max_radius_m", "rms_radius_m", "median_radius_m", "final_radius_m", "escape_time_s", "periods_survived", "max_kinetic_energy_J", "micromotion_amplitude_m"):
            if key in row:
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
        for key in ("bounded", "escaped", "stability_eligible"):
            if key in row:
                row[key] = int(float(row[key]))
        result[_localization_key(row)] = row
    return result


def _write_checkpoint(path: Path, rows: dict[tuple[str, ...], dict[str, object]]) -> None:
    ordered = sorted(rows.values(), key=lambda row: (_localization_key(row)[0], float(row["U0_V"]), float(row["frequency_Hz"]), float(row["initial_radius_m"]), float(row["initial_angle_deg"])))
    if not ordered:
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def _localization_case(interpolator: FEMInterpolator, config: ProjectConfig, stage: str, periods: int, U0: float, frequency: float, radius: float, angle_deg: float, velocity: tuple[float, float]) -> dict[str, object]:
    angle = np.deg2rad(angle_deg)
    position = (radius * np.cos(angle), radius * np.sin(angle))
    result = integrate_trajectory(
        interpolator, config.particle, U0, frequency,
        initial_position=position, initial_velocity=velocity, periods=periods,
        escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol,
        atol=config.simulation.trajectory_atol, max_step_per_period=config.localization.max_step_per_period,
        output_samples_per_period=4,
    )
    row = _localization_row(result, trajectory_metrics(result, config.particle, config.simulation.escape_radius), U0, frequency, radius, angle_deg, velocity)
    row.update({"stage": stage, "requested_periods": periods, "stability_eligible": int(stage.startswith(("B_", "C_")))})
    if stage.startswith("A_"):
        row["classification"] = "screening survivor" if result.bounded else "screening escape"
    return row


def run_localization_scan(output_dir: Path, interpolator: FEMInterpolator, config: ProjectConfig) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    """Run a checkpointed hierarchical localization scan.

    Stage A screens the full coarse grid for 75 periods. Stage B reruns only
    Stage-A survivors for 1000 periods. Stage C follows only the strongest
    Stage-B candidate regions for 10000 periods. A Stage-A survivor is never
    treated as a stability conclusion.
    """
    checkpoint_path = output_dir / "localization_checkpoint.csv"
    completed = _read_checkpoint(checkpoint_path)
    stage_a_name = f"A_{config.localization.stage_a_periods}_periods"
    stage_b_name = f"B_{config.localization.stage_b_periods}_periods"
    stage_c_name = f"C_{config.localization.stage_c_periods}_periods"
    # A prior interrupted 75-period screening run is stronger than the new
    # 50-period minimum. Reuse those completed cases under the current Stage-A
    # label instead of recomputing them.
    remapped = {}
    for row in completed.values():
        if str(row["stage"]).startswith("A_") and float(row.get("requested_periods", 0.0)) >= config.localization.stage_a_periods:
            row["stage"] = stage_a_name
        remapped[_localization_key(row)] = row
    completed = remapped
    total_a = len(config.localization.U0_values) * len(config.localization.frequency_values) * len(config.localization.initial_radii) * len(config.localization.initial_angles_deg)
    def execute(stage: str, periods: int, cases: list[tuple[float, float, float, float, tuple[float, float]]]) -> None:
        pending = []
        for U0, frequency, radius, angle_deg, velocity in cases:
            probe = {"stage": stage, "U0_V": U0, "frequency_Hz": frequency, "initial_radius_m": radius, "initial_angle_deg": angle_deg, "initial_vx_m_per_s": velocity[0], "initial_vy_m_per_s": velocity[1]}
            key = _localization_key(probe)
            if key not in completed:
                pending.append((key, (stage, periods, U0, frequency, radius, angle_deg, velocity)))
        if not pending:
            return
        if config.localization.workers <= 1:
            results = ((_localization_case(interpolator, config, *arguments), key) for key, arguments in pending)
            for row, key in results:
                completed[key] = row
                _write_checkpoint(checkpoint_path, completed)
                print(f"{stage} checkpoint: {len(completed)} completed rows", flush=True)
            return
        with ProcessPoolExecutor(max_workers=config.localization.workers, initializer=_initialize_localization_worker, initargs=(config,)) as pool:
            futures = {pool.submit(_run_localization_case_worker, arguments): key for key, arguments in pending}
            for future in as_completed(futures):
                key = futures[future]
                completed[key] = future.result()
                _write_checkpoint(checkpoint_path, completed)
                print(f"{stage} checkpoint: {len(completed)} completed rows", flush=True)
    stage_a_cases = [(U0, frequency, radius, angle, (0.0, 0.0)) for U0 in config.localization.U0_values for frequency in config.localization.frequency_values for radius in config.localization.initial_radii for angle in config.localization.initial_angles_deg]
    execute(stage_a_name, config.localization.stage_a_periods, stage_a_cases)
    stage_a_rows = [row for row in completed.values() if row["stage"] == stage_a_name]
    stage_b_cases = [(float(row["U0_V"]), float(row["frequency_Hz"]), float(row["initial_radius_m"]), float(row["initial_angle_deg"]), (0.0, 0.0)) for row in stage_a_rows if int(row["bounded"]) == 1]
    execute(stage_b_name, config.localization.stage_b_periods, stage_b_cases)
    stage_b_rows = [row for row in completed.values() if row["stage"] == stage_b_name and int(row["bounded"]) == 1]
    strongest = sorted(stage_b_rows, key=lambda row: float(row["rms_radius_m"]))[:config.localization.stage_c_candidates]
    stage_c_cases = [(float(row["U0_V"]), float(row["frequency_Hz"]), float(row["initial_radius_m"]), float(row["initial_angle_deg"]), (0.0, 0.0)) for row in strongest]
    execute(stage_c_name, config.localization.stage_c_periods, stage_c_cases)
    with (output_dir / "localization_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        ordered = sorted(completed.values(), key=lambda row: (_localization_key(row)[0], float(row["U0_V"]), float(row["frequency_Hz"]), float(row["initial_radius_m"]), float(row["initial_angle_deg"])))
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(ordered)
    matrix = np.zeros((len(config.localization.frequency_values), len(config.localization.U0_values)))
    for j, U0 in enumerate(config.localization.U0_values):
        for i, frequency in enumerate(config.localization.frequency_values):
            pair = [row for row in stage_b_rows if row["U0_V"] == U0 and row["frequency_Hz"] == frequency]
            matrix[i, j] = len(pair) / max(len(config.localization.initial_radii) * len(config.localization.initial_angles_deg), 1)
    return list(completed.values()), {"matrix": matrix, "total": total_a, "stage_a_count": len(stage_a_rows), "stage_b_count": len(stage_b_rows), "stage_c_count": len([row for row in completed.values() if row["stage"] == stage_c_name]), "checkpoint": checkpoint_path}


def publication_localization_figure(output_dir: Path, config: ProjectConfig, scan: dict[str, object]) -> None:
    matrix = np.asarray(scan["matrix"])
    fig, ax = plt.subplots(figsize=(7.8, 5.7), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1, extent=(_mm(min(config.localization.U0_values)), _mm(max(config.localization.U0_values)), min(config.localization.frequency_values) / 1000, max(config.localization.frequency_values) / 1000))
    fig.colorbar(image, ax=ax, label="Stage-B bounded fraction (1000 periods)")
    ax.set(xlabel="RF amplitude U₀ (kV)", ylabel="RF frequency (kHz)", title=f"Staged localization map: Stage B / 1000 periods ({int(scan['total'])} Stage-A cases screened)")
    save_figure(fig, output_dir, "fig04_localization_map")


def run_velocity_probe(output_dir: Path, interpolator: FEMInterpolator, config: ProjectConfig, scan_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    eligible = [row for row in scan_rows if int(row.get("stability_eligible", 0)) == 1 and int(row["bounded"]) == 1]
    bounded = eligible or [row for row in scan_rows if int(row["bounded"]) == 1]
    if bounded:
        best = min(bounded, key=lambda row: float(row["rms_radius_m"]))
    else:
        best = min(scan_rows, key=lambda row: float(row["max_radius_m"]))
    U0 = float(best["U0_V"]); frequency = float(best["frequency_Hz"]); radius = float(best["initial_radius_m"])
    rows = []
    for angle_deg in config.localization.initial_angles_deg:
        angle = np.deg2rad(angle_deg)
        position = (radius * np.cos(angle), radius * np.sin(angle))
        tangent = np.array((-np.sin(angle), np.cos(angle)))
        for speed in config.localization.transverse_velocity_scales:
            for sign in ((1,) if speed == 0 else (-1, 1)):
                velocity = tuple(float(v) for v in sign * speed * tangent)
                result = integrate_trajectory(interpolator, config.particle, U0, frequency, initial_position=position, initial_velocity=velocity, periods=config.localization.stage_b_periods, escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol, max_step_per_period=config.localization.max_step_per_period, output_samples_per_period=4)
                row = _localization_row(result, trajectory_metrics(result, config.particle, config.simulation.escape_radius), U0, frequency, radius, angle_deg, velocity)
                row.update({"stage": f"B_velocity_probe_{config.localization.stage_b_periods}_periods", "requested_periods": config.localization.stage_b_periods, "stability_eligible": 1})
                rows.append(row)
    with (output_dir / "localization_velocity_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def _poincare(result, frequency: float) -> tuple[np.ndarray, np.ndarray]:
    sample_times = np.arange(0.0, result.time[-1] + 0.5 / frequency, 1.0 / frequency)
    x = np.interp(sample_times, result.time, result.state[0])
    y = np.interp(sample_times, result.time, result.state[1])
    return x, y


def publication_trajectory_examples(output_dir: Path, interpolator: FEMInterpolator, config: ProjectConfig, scan_rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    eligible = [row for row in scan_rows if int(row.get("stability_eligible", 0)) == 1]
    bounded = [row for row in eligible if int(row["bounded"]) == 1]
    screening_only = not bool(bounded)
    if screening_only:
        bounded = [row for row in scan_rows if int(row["bounded"]) == 1]
    localized = min(bounded, key=lambda row: float(row["rms_radius_m"])) if bounded else min(scan_rows, key=lambda row: float(row["max_radius_m"]))
    escaped = max(scan_rows, key=lambda row: float(row["initial_radius_m"]) if row["escaped"] else -1.0)
    marginal_candidates = [row for row in scan_rows if row["escaped"] and float(row["initial_radius_m"]) < float(escaped["initial_radius_m"])]
    marginal = min(marginal_candidates, key=lambda row: abs(float(row["max_radius_m"]) - 0.8 * config.simulation.escape_radius)) if marginal_candidates else escaped
    selections = (("localized", localized), ("marginal", marginal), ("escaping", escaped))
    results = {}
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 10.5), constrained_layout=True)
    for column, (label, row) in enumerate(selections):
        angle = np.deg2rad(float(row["initial_angle_deg"]))
        position = (float(row["initial_radius_m"]) * np.cos(angle), float(row["initial_radius_m"]) * np.sin(angle))
        result = integrate_trajectory(interpolator, config.particle, float(row["U0_V"]), float(row["frequency_Hz"]), initial_position=position, initial_velocity=(0, 0), periods=config.localization.stage_b_periods, escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol, max_step_per_period=config.localization.max_step_per_period, output_samples_per_period=8)
        results[label] = result
        x, y = result.state[0] * 1000, result.state[1] * 1000
        axes[0, column].plot(x, y, linewidth=0.7)
        axes[0, column].scatter([x[0]], [y[0]], color="green", s=18)
        basis = "screening candidate" if screening_only and label == "localized" else "1000-period validation"
        axes[0, column].set(title=f"{label} ({basis})\nU₀={row['U0_V']/1000:g} kV, f={row['frequency_Hz']/1000:g} kHz", xlabel="x (mm)", ylabel="y (mm)")
        axes[0, column].set_aspect("equal")
        radii = np.hypot(x, y)
        axes[1, column].plot(result.time * float(row["frequency_Hz"]), radii, linewidth=0.7)
        axes[1, column].axhline(config.simulation.escape_radius * 1000, color="red", linestyle="--", linewidth=0.8)
        axes[1, column].set(xlabel="RF periods", ylabel="r (mm)")
        px, py = _poincare(result, float(row["frequency_Hz"]))
        axes[2, column].plot(px * 1000, py * 1000, ".", ms=2)
        axes[2, column].set(xlabel="x (mm)", ylabel="y (mm)", title="stroboscopic")
        axes[2, column].set_aspect("equal")
    fig.suptitle("Localized, marginal, and escaping trajectories")
    save_figure(fig, output_dir, "fig05_trajectory_examples")
    return {"results": results, "rows": {label: row for label, row in selections}}


def publication_mass_transmission(output_dir: Path, config: ProjectConfig, rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    masses = np.array([float(row["mass_over_charge_kg_per_C"]) for row in rows])
    fractions = np.array([float(row["transmission_fraction"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.plot(masses, fractions, "o-", color="#4c78a8", linewidth=1.5)
    ax.set_xscale("log")
    ax.set(xlabel="Mass / charge (kg/C)", ylabel="Transmission fraction", title=f"Finite-flight mass transmission at {config.mass_filter.reference_frequency/1000:g} kHz")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    metrics = estimate_fwhm(rows)
    if np.isfinite(metrics["peak_mass_kg"]):
        peak_mq = metrics["peak_mass_kg"] / config.particle.charge
        ax.axvline(peak_mq, color="#e45756", linestyle="--", linewidth=0.9, label="peak")
        ax.legend()
    save_figure(fig, output_dir, "fig06_mass_transmission")
    return metrics


def publication_mass_frequency_map(output_dir: Path, config: ProjectConfig, rows: list[dict[str, float | int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = np.array(sorted({float(row["mass_over_charge_kg_per_C"]) for row in rows}))
    frequencies = np.array(sorted({float(row["frequency_Hz"]) for row in rows}))
    matrix = np.full((len(frequencies), len(masses)), np.nan)
    for row in rows:
        i = int(np.where(frequencies == float(row["frequency_Hz"]))[0][0])
        j = int(np.where(masses == float(row["mass_over_charge_kg_per_C"]))[0][0])
        matrix[i, j] = float(row["survival_fraction"])
    fig, ax = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1, extent=(np.log10(masses[0]), np.log10(masses[-1]), frequencies[0] / 1000, frequencies[-1] / 1000))
    ax.set_xticks(np.log10(masses), labels=[f"{v:.2g}" for v in masses])
    ax.set(xlabel="Mass / charge (kg/C; logarithmic)", ylabel="RF frequency (kHz)", title="Mass-frequency survival map")
    fig.colorbar(image, ax=ax, label="Survival fraction")
    save_figure(fig, output_dir, "fig07_mass_frequency_map")
    return masses, frequencies, matrix


def publication_mass_trajectories(output_dir: Path, interpolator: FEMInterpolator, config: ProjectConfig, masses: tuple[float, float, float]) -> dict[str, object]:
    labels = ("best candidate", "lighter comparison", "heavier comparison")
    colors = ("#2ca02c", "#d62728", "#1f77b4")
    duration = config.mass_filter.length / config.mass_filter.axial_velocity
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    results = {}
    for mass, label, color in zip(masses, labels, colors):
        particle = config.particle.__class__(mass=mass, charge=config.particle.charge, gamma=0.0, gravity_enabled=False, gravity=config.particle.gravity, initial_position=(0.0005, 0), initial_velocity=(0, 0))
        result = integrate_trajectory(interpolator, particle, config.rf.U0, config.mass_filter.reference_frequency, initial_position=(0.0005, 0), initial_velocity=(0, 0), simulation_duration=duration, escape_radius=config.mass_filter.exit_aperture, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol, max_step_per_period=10, output_samples_per_period=8)
        results[label] = result
        axes[0].plot(result.state[0] * 1000, result.state[1] * 1000, color=color, linewidth=0.8, label=f"{label}: m/q={mass/config.particle.charge:.2g}")
        axes[1].plot(result.time * config.mass_filter.reference_frequency, np.hypot(result.state[0], result.state[1]) * 1000, color=color, linewidth=0.8, label=label)
    axes[0].set(xlabel="x (mm)", ylabel="y (mm)", title="Same entrance, different m/q")
    axes[0].set_aspect("equal")
    axes[1].axhline(config.mass_filter.exit_aperture * 1000, color="black", linestyle="--", linewidth=0.8, label="aperture")
    axes[1].set(xlabel="RF periods", ylabel="r (mm)", title="Finite-flight radial response")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    save_figure(fig, output_dir, "fig08_mass_selection_trajectories")
    return results


def publication_summary(output_dir: Path, interpolator: FEMInterpolator, geometry: TrapGeometry, config: ProjectConfig, effective: dict[str, object], trajectory: dict[str, object], transmission: list[dict[str, float | int | str]], mass_map: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)
    # A: geometry.
    for electrode in geometry.electrodes:
        axes[0, 0].add_patch(Circle(tuple(_mm(v) for v in electrode.center), _mm(geometry.electrode_radius), facecolor=PHASE_COLORS[electrode.phase], edgecolor="black", linewidth=0.6))
        axes[0, 0].text(_mm(electrode.center[0]), _mm(electrode.center[1]), str(electrode.index), ha="center", va="center", fontsize=5, color="white")
    axes[0, 0].set(xlabel="x (mm)", ylabel="y (mm)", title="A  ABC geometry"); axes[0, 0].set_aspect("equal"); axes[0, 0].set_xlim(-15, 15); axes[0, 0].set_ylim(-15, 15)
    # B: one rotating potential frame.
    axis, _, _, _, bp, _ = prepare_grid(interpolator, 0.018, 121)
    frame = np.einsum("p,pyx->yx", np.array([0.0, -np.sqrt(3)/2, np.sqrt(3)/2]) * config.rf.U0, bp)
    axes[0, 1].imshow(frame, origin="lower", extent=(_mm(axis[0]), _mm(axis[-1]), _mm(axis[0]), _mm(axis[-1])), cmap="coolwarm", norm=Normalize(np.nanmin(frame), np.nanmax(frame)))
    axes[0, 1].set(xlabel="x (mm)", ylabel="y (mm)", title="B  rotating potential"); axes[0, 1].set_aspect("equal")
    # C: effective radial profile.
    axes[0, 2].plot(_mm(effective["radii"]), effective["radial_values"], color="#7a0177")
    axes[0, 2].set(xlabel="r (mm)", ylabel="effective potential (J)", title="C  higher-order well"); axes[0, 2].grid(alpha=0.25)
    # D: representative trajectory.
    local = trajectory["results"]["localized"]
    axes[1, 0].plot(local.state[0] * 1000, local.state[1] * 1000, linewidth=0.7)
    axes[1, 0].set(xlabel="x (mm)", ylabel="y (mm)", title="D  full RF trajectory"); axes[1, 0].set_aspect("equal")
    # E: transmission.
    axes[1, 1].plot([row["mass_over_charge_kg_per_C"] for row in transmission], [row["transmission_fraction"] for row in transmission], "o-")
    axes[1, 1].set_xscale("log"); axes[1, 1].set(xlabel="m/q (kg/C)", ylabel="transmission", title="E  mass transmission"); axes[1, 1].grid(alpha=0.25)
    # F: mass-frequency map.
    masses, frequencies, matrix = mass_map
    axes[1, 2].imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1, extent=(np.log10(masses[0]), np.log10(masses[-1]), frequencies[0]/1000, frequencies[-1]/1000))
    axes[1, 2].set(xlabel="log₁₀(m/q)", ylabel="f (kHz)", title="F  mass-frequency survival")
    fig.suptitle("Three-phase 12-electrode trap: field, dynamics, and filtering study", fontsize=15)
    save_figure(fig, output_dir, "fig09_summary")


def write_publication_report(output_dir: Path, config: ProjectConfig, localization_rows: list[dict[str, float | int | str]], velocity_rows: list[dict[str, float | int | str]], effective: dict[str, object], transmission: list[dict[str, float | int | str]], mass_metrics: dict[str, float], mass_map: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Path:
    stage_a_rows = [row for row in localization_rows if str(row["stage"]).startswith("A_")]
    stage_b_rows = [row for row in localization_rows if str(row["stage"]).startswith("B_") and "velocity_probe" not in str(row["stage"])]
    stage_c_rows = [row for row in localization_rows if str(row["stage"]).startswith("C_")]
    eligible_rows = [row for row in localization_rows if int(row.get("stability_eligible", 0)) == 1]
    bounded = [row for row in eligible_rows if int(row["bounded"]) == 1]
    strongest = min(bounded, key=lambda row: float(row["rms_radius_m"])) if bounded else None
    map_matrix = mass_map[2]
    best_map = float(np.nanmax(map_matrix))
    velocity_bounded = sum(int(row["bounded"]) for row in velocity_rows)
    lines = [
        "# Publication study report", "",
        "This report summarizes the P2-FEM, long-duration localization, and quasi-2D finite-flight mass-filter analyses. It does not assume that the geometry is a stable trap or mass filter.", "",
        "## Numerical upgrade", "",
        "The electrostatic solve now uses quadratic triangular scikit-fem elements. P2 interpolation evaluates the quadratic shape functions and their exact within-element gradients; visualization fields are sampled on dense regular grids. Direct FEM and basis-superposition validation remain at approximately machine precision.", "",
        "## Localization", "",
        f"The staged localization study screened {len(stage_a_rows)} Stage-A trajectories for {config.localization.stage_a_periods} RF periods, advanced {len(stage_b_rows)} Stage-A survivors to {config.localization.stage_b_periods} periods, and followed {len(stage_c_rows)} strongest Stage-B regions to {config.localization.stage_c_periods} periods. The grid spans {len(config.localization.U0_values)} RF amplitudes, {len(config.localization.frequency_values)} frequencies, {len(config.localization.initial_radii)} radii, and {len(config.localization.initial_angles_deg)} angles with gamma=0.",
        f"Stability-eligible bounded trajectories (Stage B/C only): {len(bounded)}/{len(eligible_rows)}. The strongest eligible tested case was {strongest if strongest else 'none'}. Stage-A survivors are screening candidates only and are not used for stability conclusions.",
        f"The velocity robustness probe contained {len(velocity_rows)} additional cases; {velocity_bounded} remained bounded.", "",
        "## Effective potential", "",
        f"The fitted central potential has A6={float(effective['A6']):.6e} J/m^6, so A6 is {'positive' if float(effective['A6']) > 0 else 'not positive'} in the fitted range. The appropriate interpretation is a flat-bottom/higher-order local minimum when the radial fit is positive; a Hessian-only test is not used.", "",
        "## Mass-filter concept", "",
        f"The finite-flight model uses L={config.mass_filter.length:g} m, vz={config.mass_filter.axial_velocity:g} m/s, and flight time={config.mass_filter.length/config.mass_filter.axial_velocity:g} s. The main transmission scan launches {int(transmission[0]['launched'])} particles per mass at fixed charge and reference frequency.",
        f"Peak transmission={mass_metrics['peak_transmission']:.4g}; peak m/q={mass_metrics['peak_mass_kg']/config.particle.charge:.6g} kg/C. FWHM={mass_metrics['fwhm_kg']/config.particle.charge if np.isfinite(mass_metrics['fwhm_kg']) else float('nan'):.6g} kg/C; resolving power={mass_metrics['resolving_power']:.4g}.",
        f"The best survival fraction in the mass-frequency map is {best_map:.4g}. These coarse results {'show a candidate transmission window' if mass_metrics['peak_transmission'] > 0 else 'do not show transmission'}; a narrow mass-filter claim requires the convergence checks listed below.", "",
        "## Publication interpretation and remaining work", "",
        "The strongest current figures are fig01_geometry, fig02_rotating_potential, fig03_effective_potential, fig05_trajectory_examples, and fig06_mass_transmission. Before publication-level claims, repeat the scans with mesh/timestep/ODE-tolerance convergence, more entrance samples, longer selected flights, and a systematic interaction-time/resolution study. The quasi-2D filter model assumes constant axial velocity and ignores axial fringe fields.", "",
    ]
    path = output_dir / "publication_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_publication_study(config: ProjectConfig, project_root: Path) -> dict[str, object]:
    """Run the extended P2/publication/mass-filter workflow."""
    from .pipeline import make_state, run_multipoles, run_rotation
    output_dir = publication_dir(project_root)
    geometry, trap_mesh, solutions, interpolator = make_state(config)
    state = (geometry, trap_mesh, solutions, interpolator)
    publication_geometry(output_dir, geometry, config.mass_filter.entrance_aperture)
    multipoles = run_multipoles(config, project_root, state)
    rotation = run_rotation(config, project_root, multipoles, state)
    publication_rotating_potential(output_dir, interpolator, geometry, config, rotation["fit"])
    effective = publication_effective_potential(output_dir, interpolator, geometry, config)
    localization_rows, localization_summary = run_localization_scan(output_dir, interpolator, config)
    publication_localization_figure(output_dir, config, localization_summary)
    velocity_rows = run_velocity_probe(output_dir, interpolator, config, localization_rows)
    trajectory_examples = publication_trajectory_examples(output_dir, interpolator, config, localization_rows)
    transmission = run_mass_transmission_scan(interpolator, config)
    write_mass_transmission_csv(transmission, output_dir / "mass_transmission.csv")
    mass_metrics = publication_mass_transmission(output_dir, config, transmission)
    map_rows = run_mass_frequency_map(interpolator, config)
    mass_map = publication_mass_frequency_map(output_dir, config, map_rows)
    representative = choose_representative_masses(transmission)
    mass_trajectories = publication_mass_trajectories(output_dir, interpolator, config, representative)
    publication_summary(output_dir, interpolator, geometry, config, effective, trajectory_examples, transmission, mass_map)
    report = write_publication_report(output_dir, config, localization_rows, velocity_rows, effective, transmission, mass_metrics, mass_map)
    with (output_dir / "mass_frequency_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(map_rows[0]))
        writer.writeheader()
        writer.writerows(map_rows)
    return {"output_dir": output_dir, "localization": localization_rows, "velocity_probe": velocity_rows, "effective": effective, "transmission": transmission, "mass_metrics": mass_metrics, "mass_map": mass_map, "trajectory_examples": trajectory_examples, "mass_trajectories": mass_trajectories, "report": report, "multipoles": multipoles, "rotation": rotation}
