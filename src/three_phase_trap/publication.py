"""Publication-quality figures and extended localization/filter analyses."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize, PowerNorm, SymLogNorm
from scipy.interpolate import RegularGridInterpolator
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


def publication_pseudopotential_diagnostic(output_dir: Path, interpolator: FEMInterpolator, geometry: TrapGeometry, config: ProjectConfig) -> dict[str, object]:
    """Create a dense, multi-view diagnostic of the central pseudopotential.

    Display transforms affect only color/vertical presentation. The raw dense
    grid and radial samples are exported unchanged so small asymmetries can be
    checked numerically rather than inferred from a stretched image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    full_radius = min(0.019, geometry.outer_domain_radius * 0.80)
    central_radius = min(0.0045, 0.48 * (geometry.electrode_orbit_radius - geometry.electrode_radius))
    full_axis, _, _, _, _, full_basis_field = prepare_grid(interpolator, full_radius, 321)
    full_effective = effective_potential_from_mean_square(
        mean_square_field_from_basis({phase: full_basis_field[i] for i, phase in enumerate(("A", "B", "C"))}, config.rf.U0),
        config.particle.charge, config.particle.mass, config.rf.omega,
    )
    central_axis, _, _, _, _, central_basis_field = prepare_grid(interpolator, central_radius, 321)
    central_effective = effective_potential_from_mean_square(
        mean_square_field_from_basis({phase: central_basis_field[i] for i, phase in enumerate(("A", "B", "C"))}, config.rf.U0),
        config.particle.charge, config.particle.mass, config.rf.omega,
    )
    central_min = float(np.nanmin(central_effective))
    central_delta = central_effective - central_min
    display_radius = min(0.0035, central_radius * 0.90)
    display_indices = np.flatnonzero(np.abs(central_axis) <= display_radius)
    display_axis = central_axis[display_indices]
    display_slice = np.ix_(display_indices, display_indices)
    display_effective = central_effective[display_slice]
    display_delta = central_delta[display_slice]
    finite_delta = display_delta[np.isfinite(display_delta)]
    positive_delta = finite_delta[finite_delta > 0]
    delta_floor = max(float(np.percentile(positive_delta, 0.5)) * 0.1 if positive_delta.size else 1.0e-30, 1.0e-30)
    delta_top = max(float(np.percentile(finite_delta, 99.7)), delta_floor * 10.0)
    central_norm = Normalize(vmin=float(np.nanmin(display_effective)), vmax=float(np.percentile(display_effective, 99.7)))
    contrast_norm = PowerNorm(gamma=0.30, vmin=0.0, vmax=delta_top)
    topo_norm = SymLogNorm(linthresh=max(delta_top * 1.0e-4, delta_floor), vmin=0.0, vmax=delta_top)

    # Export raw values on both dense maps.
    for name, axis, values in (("full", full_axis, full_effective), ("central", central_axis, central_effective)):
        with (output_dir / f"pseudopotential_{name}_grid.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("x_m", "y_m", "effective_potential_J"))
            for y_index, y in enumerate(axis):
                writer.writerows((float(x), float(y), float(values[y_index, x_index])) for x_index, x in enumerate(axis))

    angles_deg = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 135.0)
    radii = np.linspace(0.0, central_radius * 0.96, 240)
    profiles: dict[float, np.ndarray] = {}
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        points = np.column_stack((radii * np.cos(angle), radii * np.sin(angle)))
        fields = {phase: interpolator.field_basis(points, phase) for phase in ("A", "B", "C")}
        profiles[angle_deg] = effective_potential_from_mean_square(
            mean_square_field_from_basis(fields, config.rf.U0), config.particle.charge, config.particle.mass, config.rf.omega,
        )
    with (output_dir / "pseudopotential_radial_cuts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("angle_deg", "radius_m", "effective_potential_J"))
        for angle_deg, values in profiles.items():
            writer.writerows((float(angle_deg), float(radius), float(value)) for radius, value in zip(radii, values))

    extent_full = (_mm(full_axis[0]), _mm(full_axis[-1]), _mm(full_axis[0]), _mm(full_axis[-1]))
    extent_central = (_mm(display_axis[0]), _mm(display_axis[-1]), _mm(display_axis[0]), _mm(display_axis[-1]))
    fig = plt.figure(figsize=(16.0, 11.5), constrained_layout=True)
    axes = fig.subplot_mosaic([["A", "B", "C"], ["D", "D", "E"]])
    image = axes["A"].imshow(full_effective, origin="lower", extent=extent_full, cmap="magma", norm=PowerNorm(gamma=0.42, vmin=0.0, vmax=float(np.nanpercentile(full_effective, 99.5))))
    axes["A"].set(xlabel="x (mm)", ylabel="y (mm)", title="A  Full pseudopotential map")
    axes["A"].set_aspect("equal")
    fig.colorbar(image, ax=axes["A"], label="ΨRF (J)", fraction=0.046)
    image = axes["B"].imshow(display_effective, origin="lower", extent=extent_central, cmap="magma", norm=central_norm)
    axes["B"].set(xlabel="x (mm)", ylabel="y (mm)", title=f"B  Central zoom (local ΨRF scale; Ψmin={central_min:.3e} J)")
    axes["B"].set_aspect("equal")
    fig.colorbar(image, ax=axes["B"], label="ΨRF (J)", fraction=0.046)
    image = axes["C"].imshow(display_delta, origin="lower", extent=extent_central, cmap="viridis", norm=contrast_norm)
    axes["C"].contour(display_axis * 1000.0, display_axis * 1000.0, display_delta, levels=np.linspace(0.0, delta_top, 35)[1:], colors="white", linewidths=0.25, alpha=0.55)
    axes["C"].set(xlabel="x (mm)", ylabel="y (mm)", title="C  Contrast-enhanced ΔΨRF = ΨRF − Ψmin")
    axes["C"].set_aspect("equal")
    fig.colorbar(image, ax=axes["C"], label="ΔΨRF (J; nonlinear)", fraction=0.046)

    topo_levels = np.unique(np.r_[0.0, np.geomspace(delta_floor, delta_top, 95)])
    topo = axes["D"].contourf(display_axis * 1000.0, display_axis * 1000.0, display_delta, levels=topo_levels, cmap="cividis", norm=topo_norm, extend="max")
    axes["D"].contour(display_axis * 1000.0, display_axis * 1000.0, display_delta, levels=np.geomspace(delta_floor, delta_top, 64), colors="black", linewidths=0.22, alpha=0.58)
    axes["D"].set(xlabel="x (mm)", ylabel="y (mm)", title="D  Dense contour/topographic view (96 levels; local nonlinear contrast)")
    axes["D"].set_aspect("equal")
    fig.colorbar(topo, ax=axes["D"], label="ΔΨRF (J; SymLogNorm)", fraction=0.026)

    for angle_deg, values in profiles.items():
        axes["E"].plot(radii * 1000.0, values - central_min, linewidth=1.0, label=f"{angle_deg:g}°")
    axes["E"].set(xlabel="Radius (mm)", ylabel="ΨRF − Ψmin (J)", title="E  Radial cuts: angular anisotropy and higher multipoles")
    axes["E"].grid(alpha=0.25)
    axes["E"].legend(ncol=3, fontsize=7, frameon=True)
    fig.suptitle("Central pseudopotential diagnostic: flatness, anisotropy, and higher-order structure", fontsize=16)
    save_figure(fig, output_dir, "fig_pseudopotential_diagnostic")

    individual = [
        ("pseudopotential_full_map", lambda ax: (ax.imshow(full_effective, origin="lower", extent=extent_full, cmap="magma", norm=PowerNorm(gamma=0.42, vmin=0.0, vmax=float(np.nanpercentile(full_effective, 99.5))), aspect="equal"), ax.set_title("Full RF pseudopotential"))),
        ("pseudopotential_central_zoom", lambda ax: (ax.imshow(display_effective, origin="lower", extent=extent_central, cmap="magma", norm=central_norm, aspect="equal"), ax.set_title("Central zoom: local normalization"))),
        ("pseudopotential_central_contrast", lambda ax: (ax.imshow(display_delta, origin="lower", extent=extent_central, cmap="viridis", norm=contrast_norm, aspect="equal"), ax.set_title("Central contrast: ΔΨRF"))),
        ("pseudopotential_dense_contours", lambda ax: (ax.contourf(display_axis * 1000.0, display_axis * 1000.0, display_delta, levels=topo_levels, cmap="cividis", norm=topo_norm), ax.contour(display_axis * 1000.0, display_axis * 1000.0, display_delta, levels=np.geomspace(delta_floor, delta_top, 64), colors="black", linewidths=0.25), ax.set_title("Dense topographic contours"))),
        ("pseudopotential_radial_cuts", lambda ax: ([ax.plot(radii * 1000.0, values - central_min, label=f"{angle_deg:g}°", linewidth=1.0) for angle_deg, values in profiles.items()], ax.set_title("Radial cuts"), ax.legend(ncol=3, fontsize=7))),
    ]
    for stem, draw in individual:
        one, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
        draw(ax)
        ax.set_xlabel("Radius (mm)" if "radial" in stem else "x (mm)")
        if "radial" in stem:
            ax.set_ylabel("ΨRF − Ψmin (J)")
        else:
            ax.set_ylabel("y (mm)")
        save_figure(one, output_dir, stem)

    surface_step = max(1, len(central_axis) // 121)
    surface_axis = display_axis[::surface_step]
    surface_delta = display_delta[::surface_step, ::surface_step]
    xx, yy = np.meshgrid(surface_axis * 1000.0, surface_axis * 1000.0, indexing="xy")
    surface_fig = plt.figure(figsize=(9.0, 7.0), constrained_layout=True)
    surface_ax = surface_fig.add_subplot(111, projection="3d")
    surface = surface_ax.plot_surface(xx, yy, surface_delta * 1.0e9, cmap="viridis", linewidth=0, antialiased=True, rcount=121, ccount=121)
    surface_ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="ΔΨRF (nJ)", title="Central pseudopotential surface (raw ΔΨRF values)")
    surface_ax.view_init(elev=32, azim=-55)
    surface_fig.colorbar(surface, ax=surface_ax, shrink=0.62, pad=0.10, label="ΔΨRF (nJ)")
    save_figure(surface_fig, output_dir, "pseudopotential_3d_surface")

    probe_radius = min(0.003, central_radius * 0.75)
    probe_values = np.asarray([np.interp(probe_radius, radii, values) for values in profiles.values()])
    return {"full_axis": full_axis, "full_effective": full_effective, "central_axis": central_axis, "central_effective": central_effective, "central_min": central_min, "central_delta": central_delta, "radii": radii, "profiles": profiles, "probe_radius": probe_radius, "probe_spread_J": float(np.ptp(probe_values)), "probe_mean_J": float(np.mean(probe_values))}


def _effective_grid(interpolator: FEMInterpolator, config: ProjectConfig, radius: float, size: int, recovered: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if recovered:
        axis = np.linspace(-radius, radius, size)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        flat = np.column_stack((xx.ravel(), yy.ravel()))
        ids = np.asarray(interpolator._finder(flat[:, 0], flat[:, 1]), dtype=int)
        mask = ids >= 0
        basis_field = np.full((3, len(flat), 2), np.nan)
        for i, phase in enumerate(("A", "B", "C")):
            basis_field[i, mask] = interpolator.field_basis_recovered(flat[mask], phase)
        basis_field = basis_field.reshape((3,) + xx.shape + (2,))
    else:
        axis, _, _, _, _, basis_field = prepare_grid(interpolator, radius, size)
    fields = {phase: basis_field[i] for i, phase in enumerate(("A", "B", "C"))}
    mean_squared = mean_square_field_from_basis(fields, config.rf.U0)
    values = effective_potential_from_mean_square(mean_squared, config.particle.charge, config.particle.mass, config.rf.omega)
    return axis, values


def pseudopotential_mesh_convergence(output_dir: Path, config: ProjectConfig) -> dict[str, object]:
    """Measure central angular irregularity as the P2 mesh is refined."""
    from .pipeline import make_state
    radii = np.array([0.001, 0.002, 0.003], dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    resolutions = (0.00095, 0.00065, 0.00045)
    records = []
    profiles = {}
    for resolution in resolutions:
        mesh_config = replace(config.mesh, resolution=resolution)
        mesh_configured = replace(config, mesh=mesh_config)
        _, _, _, interpolator = make_state(mesh_configured)
        values_by_radius = []
        for radius in radii:
            points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
            fields = {phase: interpolator.field_basis_recovered(points, phase) for phase in ("A", "B", "C")}
            values = effective_potential_from_mean_square(
                mean_square_field_from_basis(fields, config.rf.U0), config.particle.charge, config.particle.mass, config.rf.omega,
            )
            values_by_radius.append(values)
            records.append({"mesh_resolution_m": resolution, "radius_m": radius, "mean_pseudopotential_J": float(np.mean(values)), "angular_spread_J": float(np.ptp(values)), "relative_angular_spread": float(np.ptp(values) / max(abs(float(np.mean(values))), 1.0e-300)), "angular_std_J": float(np.std(values))})
        profiles[resolution] = np.asarray(values_by_radius)
    with (output_dir / "pseudopotential_mesh_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for resolution in resolutions:
        subset = [r for r in records if r["mesh_resolution_m"] == resolution]
        axes[0].plot(radii * 1000.0, [r["relative_angular_spread"] for r in subset], "o-", label=f"h={resolution * 1000:g} mm")
        axes[1].plot(angles * 180.0 / np.pi, (profiles[resolution][-1] - np.mean(profiles[resolution][-1])) * 1.0e9, linewidth=1.0, label=f"h={resolution * 1000:g} mm")
    axes[0].set(xlabel="Radius (mm)", ylabel="Angular spread / mean", title="Mesh convergence of angular irregularity")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Angle (deg)", ylabel="ΨRF − angular mean (nJ)", title="Angular residual at r = 3 mm")
    axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8)
    save_figure(fig, output_dir, "pseudopotential_mesh_convergence")
    fine = [r for r in records if r["mesh_resolution_m"] == min(resolutions)]
    coarse = [r for r in records if r["mesh_resolution_m"] == max(resolutions)]
    return {"records": records, "resolutions": resolutions, "radii": radii, "angles": angles, "profiles": profiles, "coarse_to_fine_spread_ratio": float(fine[-1]["angular_spread_J"] / max(coarse[-1]["angular_spread_J"], 1.0e-300))}


def publication_pseudopotential_refined(output_dir: Path, interpolator: FEMInterpolator, geometry: TrapGeometry, config: ProjectConfig) -> dict[str, object]:
    """Generate interpretation-first pseudopotential figures on regular grids."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full_axis, full_effective = _effective_grid(interpolator, config, min(0.019, geometry.outer_domain_radius * 0.80), 501, recovered=True)
    central_axis, central_effective = _effective_grid(interpolator, config, min(0.0042, 0.48 * (geometry.electrode_orbit_radius - geometry.electrode_radius)), 601, recovered=True)
    central_min = float(np.nanmin(central_effective))
    central_delta = np.maximum(central_effective - central_min, 0.0)
    center_indices = np.flatnonzero(np.abs(central_axis) <= 0.0035)
    center_axis = central_axis[center_indices]
    center_slice = np.ix_(center_indices, center_indices)
    center_effective = central_effective[center_slice]
    center_delta = central_delta[center_slice]
    delta_max = max(float(np.nanpercentile(center_delta, 99.5)), 1.0e-30)
    delta_ref = 1.0e-10
    transformed = np.power(center_delta / delta_ref, 1.0 / 6.0)

    for name, axis, values in (("full", full_axis, full_effective), ("central", central_axis, central_effective)):
        with (output_dir / f"pseudopotential_refined_recovered_{name}_grid.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("x_m", "y_m", "effective_potential_J"))
            for yi, y in enumerate(axis):
                writer.writerows((float(x), float(y), float(values[yi, xi])) for xi, x in enumerate(axis))

    angles_deg = np.arange(0.0, 360.0, 15.0)
    radii_full = np.linspace(0.0, min(0.0040, abs(central_axis[-1]) * 0.96), 300)
    profiles = {}
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        points = np.column_stack((radii_full * np.cos(angle), radii_full * np.sin(angle)))
        fields = {phase: interpolator.field_basis_recovered(points, phase) for phase in ("A", "B", "C")}
        profiles[angle_deg] = effective_potential_from_mean_square(mean_square_field_from_basis(fields, config.rf.U0), config.particle.charge, config.particle.mass, config.rf.omega)
    with (output_dir / "pseudopotential_refined_radial_cuts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("angle_deg", "radius_m", "effective_potential_J"))
        for angle_deg, values in profiles.items():
            writer.writerows((float(angle_deg), float(radius), float(value)) for radius, value in zip(radii_full, values))

    zoom_limit = 0.0018
    zoom_mask = radii_full <= zoom_limit
    profile_matrix = np.asarray([profiles[angle] for angle in angles_deg])
    profile_mean = np.mean(profile_matrix, axis=0)
    anisotropy_radii = (0.001, 0.002, 0.003)
    anisotropy = {}
    for radius in anisotropy_radii:
        j = int(np.argmin(abs(radii_full - radius)))
        anisotropy[radius] = profile_matrix[:, j]
    with (output_dir / "pseudopotential_refined_angular_anisotropy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("radius_m", "angle_deg", "effective_potential_J", "deviation_from_angular_mean_J"))
        for radius, values in anisotropy.items():
            mean = float(np.mean(values))
            writer.writerows((float(radius), float(angle), float(value), float(value - mean)) for angle, value in zip(angles_deg, values))

    extent_full = (_mm(full_axis[0]), _mm(full_axis[-1]), _mm(full_axis[0]), _mm(full_axis[-1]))
    extent_center = (_mm(center_axis[0]), _mm(center_axis[-1]), _mm(center_axis[0]), _mm(center_axis[-1]))
    full_norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=float(np.nanpercentile(full_effective, 99.5)))
    center_norm = Normalize(vmin=float(np.nanmin(center_effective)), vmax=float(np.nanpercentile(center_effective, 99.5)))
    transformed_norm = Normalize(vmin=0.0, vmax=float(np.nanpercentile(transformed, 99.5)))
    fig = plt.figure(figsize=(16.0, 11.5), constrained_layout=True)
    axes = fig.subplot_mosaic([["A", "B", "C"], ["D", "E", "F"]])
    image = axes["A"].imshow(full_effective, origin="lower", extent=extent_full, cmap="magma", norm=full_norm, interpolation="bicubic")
    axes["A"].set(xlabel="x (mm)", ylabel="y (mm)", title="A  Full ΨRF (501 × 501 regular grid; recovered P2 field)"); axes["A"].set_aspect("equal"); fig.colorbar(image, ax=axes["A"], label="ΨRF (J)")
    image = axes["B"].imshow(center_effective, origin="lower", extent=extent_center, cmap="magma", norm=center_norm, interpolation="bicubic")
    axes["B"].set(xlabel="x (mm)", ylabel="y (mm)", title="B  Central zoom (local scale; recovered P2 field)"); axes["B"].set_aspect("equal"); fig.colorbar(image, ax=axes["B"], label="ΨRF (J)")
    image = axes["C"].imshow(transformed, origin="lower", extent=extent_center, cmap="viridis", norm=transformed_norm, interpolation="bicubic")
    axes["C"].set(xlabel="x (mm)", ylabel="y (mm)", title=r"C  $[(\Psi_{RF}-\Psi_{min})/(10^{-10}\,J)]^{1/6}$"); axes["C"].set_aspect("equal"); fig.colorbar(image, ax=axes["C"], label="Dimensionless sixth-root scale")
    contour_levels = np.linspace(0.0, float(np.nanpercentile(transformed, 99.5)), 90)[1:]
    contour = axes["D"].contourf(center_axis * 1000.0, center_axis * 1000.0, transformed, levels=contour_levels, cmap="cividis", extend="max")
    axes["D"].contour(center_axis * 1000.0, center_axis * 1000.0, transformed, levels=np.linspace(0.0, contour_levels[-1], 55)[1:], colors="black", linewidths=0.25, alpha=0.55)
    axes["D"].set(xlabel="x (mm)", ylabel="y (mm)", title="D  Dense transformed contours (90 levels)"); axes["D"].set_aspect("equal"); fig.colorbar(contour, ax=axes["D"], label="Sixth-root scale")
    for angle_deg, values in profiles.items():
        axes["E"].plot(radii_full[zoom_mask] * 1000.0, (values[zoom_mask] - central_min) * 1.0e9, linewidth=0.8, alpha=0.8)
    axes["E"].set(xlabel="Radius (mm)", ylabel="ΨRF − Ψmin (nJ)", title="E  Central radial cuts (0–1.8 mm)"); axes["E"].grid(alpha=0.25)
    for radius, values in anisotropy.items():
        axes["F"].plot(angles_deg, (values - np.mean(values)) * 1.0e9, "o-", linewidth=1.0, ms=2.5, label=f"r={radius * 1000:g} mm")
    axes["F"].axhline(0.0, color="black", linewidth=0.5); axes["F"].set(xlabel="Angle (deg)", ylabel="ΨRF − angular mean (nJ)", title="F  Angular anisotropy at fixed radii"); axes["F"].grid(alpha=0.25); axes["F"].legend(fontsize=8)
    fig.suptitle("Regular-grid pseudopotential structure: flatness, sixth-root well shape, and anisotropy", fontsize=16)
    save_figure(fig, output_dir, "fig_pseudopotential_refined_summary")

    individual = [
        ("pseudopotential_refined_full_map", lambda ax: ax.imshow(full_effective, origin="lower", extent=extent_full, cmap="magma", norm=full_norm, interpolation="bicubic", aspect="equal")),
        ("pseudopotential_refined_center_zoom", lambda ax: ax.imshow(center_effective, origin="lower", extent=extent_center, cmap="magma", norm=center_norm, interpolation="bicubic", aspect="equal")),
        ("pseudopotential_refined_transformed_center", lambda ax: ax.imshow(transformed, origin="lower", extent=extent_center, cmap="viridis", norm=transformed_norm, interpolation="bicubic", aspect="equal")),
        ("pseudopotential_refined_dense_contours", lambda ax: ax.contourf(center_axis * 1000.0, center_axis * 1000.0, transformed, levels=contour_levels, cmap="cividis", extend="max")),
    ]
    for stem, draw in individual:
        one, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True); draw(ax); ax.set(xlabel="x (mm)", ylabel="y (mm)"); save_figure(one, output_dir, stem)
    for stem, radial_mask in (("pseudopotential_refined_radial_full", np.ones_like(radii_full, dtype=bool)), ("pseudopotential_refined_radial_center", zoom_mask)):
        one, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        for angle_deg, values in profiles.items(): ax.plot(radii_full[radial_mask] * 1000.0, (values[radial_mask] - central_min) * 1.0e9, linewidth=0.8, alpha=0.75)
        ax.set(xlabel="Radius (mm)", ylabel="ΨRF − Ψmin (nJ)", title="Radial pseudopotential cuts"); ax.grid(alpha=0.25); save_figure(one, output_dir, stem)
    one, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    for radius, values in anisotropy.items(): ax.plot(angles_deg, (values - np.mean(values)) * 1.0e9, "o-", ms=2.5, linewidth=1.0, label=f"r={radius * 1000:g} mm")
    ax.axhline(0.0, color="black", linewidth=0.5); ax.set(xlabel="Angle (deg)", ylabel="ΨRF − angular mean (nJ)", title="Angular anisotropy at fixed radii"); ax.grid(alpha=0.25); ax.legend(fontsize=8); save_figure(one, output_dir, "pseudopotential_refined_angular_anisotropy")
    convergence = pseudopotential_mesh_convergence(output_dir, config)
    coarse = {float(r["radius_m"]): r for r in convergence["records"] if r["mesh_resolution_m"] == max(convergence["resolutions"])}
    fine = {float(r["radius_m"]): r for r in convergence["records"] if r["mesh_resolution_m"] == min(convergence["resolutions"])}
    report_lines = [
        "# Refined pseudopotential visualization report", "",
        "The publication maps use dense Cartesian regular grids and an area-weighted nodal recovery of the discontinuous elementwise P2 gradient. The FEM solution and direct elementwise field are unchanged; recovered-grid values are a labeled visualization postprocessing projection.", "",
        f"Grid sizes: full={len(full_axis)}x{len(full_axis)}, central={len(central_axis)}x{len(central_axis)}. The transformed map is [(ΨRF-Ψmin)/(10^-10 J)]^(1/6), a monotonic display coordinate that makes an r^6 well approximately linear in radius.", "",
        f"Central minimum on the recovered grid: {central_min:.8e} J. At r=3 mm, the coarse-to-fine angular-spread ratio is {convergence['coarse_to_fine_spread_ratio']:.6f}.", "",
        "## Mesh convergence", "",
        "| radius | coarse relative spread | fine relative spread | fine/coarse absolute spread |", "|---:|---:|---:|---:|",
    ]
    for radius in sorted(coarse):
        ratio = fine[radius]["angular_spread_J"] / max(coarse[radius]["angular_spread_J"], 1.0e-300)
        report_lines.append(f"| {radius * 1000:g} mm | {coarse[radius]['relative_angular_spread']:.6g} | {fine[radius]['relative_angular_spread']:.6g} | {ratio:.6g} |")
    report_lines.extend(["", "The angular irregularity decreases clearly at 2 mm and modestly at 3 mm as the mesh is refined, but it remains nonzero. The refined figures therefore support a central low/flat-bottom region with residual anisotropy that is still partly mesh-sensitive; they do not by themselves establish a converged physical higher-multipole coefficient.", ""])
    (output_dir / "pseudopotential_refined_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {"central_min_J": central_min, "full_axis": full_axis, "full_effective": full_effective, "central_axis": central_axis, "central_effective": central_effective, "transformed": transformed, "profiles": profiles, "anisotropy": anisotropy, "mesh_convergence": convergence, "central_delta_max_J": delta_max, "sixth_root_reference_J": delta_ref}


def _dc_bias_regular_grid(interpolator: FEMInterpolator, radius: float, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample RF and common-DC basis fields on a regular Cartesian grid.

    The area-weighted nodal recovery is display postprocessing only. All
    values are still generated from the unchanged P2 FEM basis solutions.
    """
    axis = np.linspace(-radius, radius, size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    flat = np.column_stack((xx.ravel(), yy.ravel()))
    ids = np.asarray(interpolator._finder(flat[:, 0], flat[:, 1]), dtype=int)
    mask = ids >= 0
    rf = np.full((3, len(flat), 2), np.nan, dtype=float)
    for index, phase in enumerate(("A", "B", "C")):
        rf[index, mask] = interpolator.field_basis_recovered(flat[mask], phase)
    dc = np.full((len(flat), 2), np.nan, dtype=float)
    dc[mask] = interpolator.field_basis_recovered(flat[mask], "DC")
    dc_potential = np.full(len(flat), np.nan, dtype=float)
    dc_potential[mask] = interpolator.potential_basis(flat[mask], "DC")
    return axis, xx, yy, rf.reshape((3,) + xx.shape + (2,)), dc.reshape(xx.shape + (2,)), dc_potential.reshape(xx.shape)


def _dc_bias_radial_data(interpolator: FEMInterpolator, config: ProjectConfig, radii: np.ndarray, angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.deg2rad(angles_deg)
    profile_rf = np.empty((len(angles), len(radii)), dtype=float)
    dc_squared = np.empty_like(profile_rf)
    for j, angle in enumerate(angles):
        points = np.column_stack((radii * np.cos(angle), radii * np.sin(angle)))
        fields = {phase: interpolator.field_basis_recovered(points, phase) for phase in ("A", "B", "C")}
        profile_rf[j] = effective_potential_from_mean_square(
            mean_square_field_from_basis(fields, config.rf.U0), config.particle.charge, config.particle.mass, config.rf.omega,
        )
        dc_field = interpolator.field_basis_recovered(points, "DC")
        dc_squared[j] = np.sum(dc_field * dc_field, axis=-1)
    prefactor = config.particle.charge**2 / (2.0 * config.particle.mass * config.rf.omega**2)
    return profile_rf, prefactor * dc_squared, angles_deg


def _write_grid_csv(path: Path, axis: np.ndarray, psi_rf: np.ndarray, psi_total: np.ndarray, dc_field: np.ndarray, dc_potential: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x_m", "y_m", "psi_rf_J", "psi_total_time_averaged_proxy_J", "dc_field_magnitude_V_per_m", "dc_basis_potential_V_per_V"))
        for yi, y in enumerate(axis):
            for xi, x in enumerate(axis):
                dc_value = dc_field[yi, xi]
                writer.writerow((float(x), float(y), float(psi_rf[yi, xi]), float(psi_total[yi, xi]), float(np.linalg.norm(dc_value)) if np.all(np.isfinite(dc_value)) else float("nan"), float(dc_potential[yi, xi])))


def _dc_bias_case(output_dir: Path, interpolator: FEMInterpolator, geometry: TrapGeometry, config: ProjectConfig, Udc: float, base_grids: dict[str, object], radial_rf: np.ndarray, dc_radial_energy: np.ndarray, angles_deg: np.ndarray) -> dict[str, object]:
    case_dir = output_dir / f"Udc_{int(Udc):04d}V"
    case_dir.mkdir(parents=True, exist_ok=True)
    full_axis = base_grids["full_axis"]
    center_axis = base_grids["center_axis"]
    prefactor = config.particle.charge**2 / (2.0 * config.particle.mass * config.rf.omega**2)
    full_rf = base_grids["full_rf"]
    full_dc_field = base_grids["full_dc_field"]
    full_total = full_rf + prefactor * Udc**2 * np.sum(full_dc_field * full_dc_field, axis=-1)
    center_rf = base_grids["center_rf"]
    center_dc_field = base_grids["center_dc_field"]
    center_total = center_rf + prefactor * Udc**2 * np.sum(center_dc_field * center_dc_field, axis=-1)
    central_min = float(np.nanmin(center_total))
    center_delta = np.maximum(center_total - central_min, 0.0)
    transform_reference = max(float(np.nanpercentile(center_delta, 80.0)), 1.0e-30)
    transformed = np.power(center_delta / transform_reference, 1.0 / 6.0)
    full_dc_potential = base_grids["full_dc_potential"]
    center_dc_potential = base_grids["center_dc_potential"]
    _write_grid_csv(case_dir / "pseudopotential_grid.csv", full_axis, full_rf, full_total, full_dc_field, full_dc_potential)
    _write_grid_csv(case_dir / "pseudopotential_central_grid.csv", center_axis, center_rf, center_total, center_dc_field, center_dc_potential)

    # Keep static electrostatic energy separate from the RF pseudopotential.
    static_dc_energy = config.particle.charge * Udc * center_dc_potential
    with (case_dir / "static_dc_energy_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("x_m", "y_m", "static_dc_potential_energy_J", "dc_electric_field_magnitude_V_per_m"))
        for yi, y in enumerate(center_axis):
            for xi, x in enumerate(center_axis):
                field_value = center_dc_field[yi, xi]
                writer.writerow((float(x), float(y), float(static_dc_energy[yi, xi]), float(np.linalg.norm(field_value) * abs(Udc)) if np.all(np.isfinite(field_value)) else float("nan")))
    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    energy_nj = static_dc_energy * 1.0e9
    image = ax.imshow(energy_nj, origin="lower", extent=(_mm(center_axis[0]), _mm(center_axis[-1]), _mm(center_axis[0]), _mm(center_axis[-1])), cmap="coolwarm", interpolation="bicubic", aspect="equal")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Static DC potential energy qΦDC, Udc={Udc:g} V")
    fig.colorbar(image, ax=ax, label="qΦDC (nJ)"); save_figure(fig, case_dir, "static_dc_potential_energy")
    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    dc_magnitude = np.linalg.norm(center_dc_field, axis=-1) * abs(Udc)
    image = ax.imshow(dc_magnitude, origin="lower", extent=(_mm(center_axis[0]), _mm(center_axis[-1]), _mm(center_axis[0]), _mm(center_axis[-1])), cmap="viridis", interpolation="bicubic", aspect="equal")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Static electric-field magnitude, Udc={Udc:g} V")
    fig.colorbar(image, ax=ax, label="|E_DC| (V/m)"); save_figure(fig, case_dir, "static_dc_field_magnitude")

    radii = base_grids["radii"]
    radial_total = radial_rf + Udc**2 * dc_radial_energy
    radial_mean = np.mean(radial_total, axis=0)
    radial_min = float(np.min(radial_total))
    with (case_dir / "radial_cuts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("angle_deg", "radius_m", "psi_rf_J", "psi_total_time_averaged_proxy_J", "delta_from_case_min_J"))
        for angle, rf_values, total_values in zip(angles_deg, radial_rf, radial_total):
            writer.writerows((float(angle), float(r), float(rf), float(total), float(total - radial_min)) for r, rf, total in zip(radii, rf_values, total_values))
    with (case_dir / "angular_anisotropy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("radius_m", "angle_deg", "psi_total_time_averaged_proxy_J", "deviation_from_angular_mean_J"))
        for radius in (0.001, 0.002, 0.003):
            j = int(np.argmin(abs(radii - radius)))
            values = radial_total[:, j]; mean = float(np.mean(values))
            writer.writerows((float(radii[j]), float(angle), float(value), float(value - mean)) for angle, value in zip(angles_deg, values))

    extent_full = (_mm(full_axis[0]), _mm(full_axis[-1]), _mm(full_axis[0]), _mm(full_axis[-1]))
    extent_center = (_mm(center_axis[0]), _mm(center_axis[-1]), _mm(center_axis[0]), _mm(center_axis[-1]))
    display_scale = 1.0e9
    full_norm = PowerNorm(gamma=0.42, vmin=0.0, vmax=float(np.nanpercentile(full_total * display_scale, 99.5)))
    center_norm = Normalize(vmin=float(np.nanmin(center_total * display_scale)), vmax=float(np.nanpercentile(center_total * display_scale, 99.5)))
    transformed_norm = Normalize(vmin=0.0, vmax=float(np.nanpercentile(transformed, 99.5)))
    contour_levels = np.linspace(0.0, float(np.nanpercentile(transformed, 99.5)), 100)[1:]
    draw_specs = [
        ("pseudopotential_full_map", lambda ax: ax.imshow(full_total * display_scale, origin="lower", extent=extent_full, cmap="magma", norm=PowerNorm(gamma=0.42, vmin=0.0, vmax=float(np.nanpercentile(full_total * display_scale, 99.5))), interpolation="bicubic", aspect="equal"), "Time-averaged total-field proxy (nJ)", f"Udc={Udc:g} V: total-field proxy"),
        ("pseudopotential_central_zoom", lambda ax: ax.imshow(center_total * display_scale, origin="lower", extent=extent_center, cmap="magma", norm=Normalize(vmin=float(np.nanmin(center_total * display_scale)), vmax=float(np.nanpercentile(center_total * display_scale, 99.5))), interpolation="bicubic", aspect="equal"), "Local total-field proxy (nJ)", f"Udc={Udc:g} V: total-field proxy, central zoom"),
        ("pseudopotential_transformed_center", lambda ax: ax.imshow(transformed, origin="lower", extent=extent_center, cmap="viridis", norm=transformed_norm, interpolation="bicubic", aspect="equal"), "[(Ψ−Ψmin)/Ψref]^(1/6)", f"Udc={Udc:g} V: transformed center"),
        ("pseudopotential_dense_contours", lambda ax: (ax.contourf(center_axis * 1000.0, center_axis * 1000.0, transformed, levels=contour_levels, cmap="cividis", extend="max"), ax.contour(center_axis * 1000.0, center_axis * 1000.0, transformed, levels=np.linspace(0.0, contour_levels[-1], 60)[1:], colors="black", linewidths=0.22, alpha=0.5)), "Sixth-root transformed total-field proxy", f"Udc={Udc:g} V: proxy contours"),
    ]
    for stem, draw, label, title in draw_specs:
        fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
        artist = draw(ax)
        if isinstance(artist, tuple): artist = artist[0]
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title); fig.colorbar(artist, ax=ax, label=label)
        save_figure(fig, case_dir, stem)

    for stem, mask, title in (("radial_cuts_full", np.ones_like(radii, dtype=bool), f"Udc={Udc:g} V: radial cuts, full central range"), ("radial_cuts_center", radii <= 0.0018, f"Udc={Udc:g} V: central radial cuts")):
        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        for angle, values in zip(angles_deg, radial_total): ax.plot(radii[mask] * 1000.0, (values[mask] - radial_min) * display_scale, linewidth=0.8, alpha=0.75, label=f"{angle:g}°")
        ax.set(xlabel="Radius (mm)", ylabel="Ψproxy − Ψmin (nJ)", title=title); ax.grid(alpha=0.25)
        if stem.endswith("center"): ax.legend(ncol=3, fontsize=7)
        save_figure(fig, case_dir, stem)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for radius in (0.001, 0.002, 0.003):
        j = int(np.argmin(abs(radii - radius))); values = radial_total[:, j]
        ax.plot(angles_deg, (values - np.mean(values)) * display_scale, "o-", ms=2.5, linewidth=1.0, label=f"r={radii[j] * 1000:g} mm")
    ax.axhline(0.0, color="black", linewidth=0.5); ax.set(xlabel="Angle (deg)", ylabel="Ψproxy − angular mean (nJ)", title=f"Udc={Udc:g} V: angular anisotropy"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    save_figure(fig, case_dir, "angular_anisotropy")

    fig = plt.figure(figsize=(16.0, 11.0), constrained_layout=True)
    axes = fig.subplot_mosaic([["A", "B", "C"], ["D", "E", "F"]])
    image = axes["A"].imshow(full_total * display_scale, origin="lower", extent=extent_full, cmap="magma", norm=full_norm, interpolation="bicubic", aspect="equal"); axes["A"].set_title("A  Full total-field proxy (nJ)"); fig.colorbar(image, ax=axes["A"])
    image = axes["B"].imshow(center_total * display_scale, origin="lower", extent=extent_center, cmap="magma", norm=center_norm, interpolation="bicubic", aspect="equal"); axes["B"].set_title("B  Proxy center zoom (local scale, nJ)"); fig.colorbar(image, ax=axes["B"])
    image = axes["C"].imshow(transformed, origin="lower", extent=extent_center, cmap="viridis", norm=transformed_norm, interpolation="bicubic", aspect="equal"); axes["C"].set_title("C  Sixth-root transformed center"); fig.colorbar(image, ax=axes["C"])
    contour = axes["D"].contourf(center_axis * 1000.0, center_axis * 1000.0, transformed, levels=contour_levels, cmap="cividis", extend="max"); axes["D"].contour(center_axis * 1000.0, center_axis * 1000.0, transformed, levels=np.linspace(0.0, contour_levels[-1], 60)[1:], colors="black", linewidths=0.2, alpha=0.5); axes["D"].set_title("D  Dense contours (100 levels)"); fig.colorbar(contour, ax=axes["D"])
    for values in radial_total: axes["E"].plot(radii[radii <= 0.0018] * 1000.0, (values[radii <= 0.0018] - radial_min) * display_scale, linewidth=0.75, alpha=0.75)
    axes["E"].set(xlabel="Radius (mm)", ylabel="Ψproxy − Ψmin (nJ)", title="E  Central radial cuts"); axes["E"].grid(alpha=0.25)
    for radius in (0.001, 0.002, 0.003):
        j = int(np.argmin(abs(radii - radius))); values = radial_total[:, j]; axes["F"].plot(angles_deg, (values - np.mean(values)) * display_scale, "o-", ms=2.0, linewidth=0.9, label=f"r={radii[j] * 1000:g} mm")
    axes["F"].axhline(0.0, color="black", linewidth=0.5); axes["F"].set(xlabel="Angle (deg)", ylabel="Residual (nJ)", title="F  Angular anisotropy"); axes["F"].grid(alpha=0.25); axes["F"].legend(fontsize=7)
    fig.suptitle(f"Common-DC-bias RF and total-field diagnostics: Udc={Udc:g} V", fontsize=16)
    save_figure(fig, case_dir, "fig_dc_bias_summary")

    flat_mask = np.hypot(base_grids["center_xx"], base_grids["center_yy"]) <= 0.0015
    flatness_J = float(np.nanmax(center_total[flat_mask]) - np.nanmin(center_total[flat_mask]))
    j3 = int(np.argmin(abs(radii - 0.003)))
    threshold = 0.1 * max(float(radial_mean[j3] - radial_mean[0]), 1.0e-30)
    crossing = np.flatnonzero((radial_mean - radial_mean[0]) > threshold)
    useful_radius = float(radii[crossing[0]]) if len(crossing) else float(radii[-1])
    j2 = int(np.argmin(abs(radii - 0.002)))
    anisotropy_2mm = float(np.ptp(radial_total[:, j2]))
    return {"Udc_V": Udc, "case_dir": case_dir, "psi_rf_full": full_rf, "psi_total_full": full_total, "psi_rf_center": center_rf, "psi_total_center": center_total, "central_min_J": central_min, "flatness_1p5mm_J": flatness_J, "useful_radius_m": useful_radius, "anisotropy_2mm_J": anisotropy_2mm, "radial_total": radial_total, "radial_rf": radial_rf, "radial_mean": radial_mean, "transformed": transformed, "transform_reference_J": transform_reference}


def run_dc_bias_comparison(config: ProjectConfig, project_root: Path) -> dict[str, object]:
    """Generate a separate common-DC-bias visualization and trajectory study."""
    from .pipeline import make_state
    output_dir = publication_dir(project_root) / "dc_bias"
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry, _, solutions, interpolator = make_state(config)
    full_axis, full_xx, full_yy, full_rf_basis, full_dc_field, full_dc_potential = _dc_bias_regular_grid(interpolator, min(0.019, geometry.outer_domain_radius * 0.80), 401)
    center_axis, center_xx, center_yy, center_rf_basis, center_dc_field, center_dc_potential = _dc_bias_regular_grid(interpolator, min(0.0042, 0.48 * (geometry.electrode_orbit_radius - geometry.electrode_radius)), 501)
    full_rf = effective_potential_from_mean_square(mean_square_field_from_basis({phase: full_rf_basis[i] for i, phase in enumerate(("A", "B", "C"))}, config.rf.U0), config.particle.charge, config.particle.mass, config.rf.omega)
    center_rf = effective_potential_from_mean_square(mean_square_field_from_basis({phase: center_rf_basis[i] for i, phase in enumerate(("A", "B", "C"))}, config.rf.U0), config.particle.charge, config.particle.mass, config.rf.omega)
    radii = np.linspace(0.0, min(0.0040, center_axis[-1] * 0.96), 320)
    angles_deg = np.arange(0.0, 360.0, 15.0)
    radial_rf, dc_radial_energy, angles_deg = _dc_bias_radial_data(interpolator, config, radii, angles_deg)
    base_grids = {"full_axis": full_axis, "center_axis": center_axis, "full_rf": full_rf, "center_rf": center_rf, "full_dc_field": full_dc_field, "center_dc_field": center_dc_field, "full_dc_potential": full_dc_potential, "center_dc_potential": center_dc_potential, "center_xx": center_xx, "center_yy": center_yy, "radii": radii}
    Udc_values = (0.0, 100.0, 300.0, 500.0, 1000.0)
    cases = [_dc_bias_case(output_dir, interpolator, geometry, config, Udc, base_grids, radial_rf, dc_radial_energy, angles_deg) for Udc in Udc_values]
    baseline = cases[0]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    for case in cases: axes[0, 0].plot(radii * 1000.0, (case["radial_mean"] - case["radial_mean"][0]) * 1.0e9, linewidth=1.2, label=f"{case['Udc_V']:g} V")
    axes[0, 0].set(xlabel="Radius (mm)", ylabel="Mean ΔΨproxy (nJ)", title="Central minimum structure"); axes[0, 0].grid(alpha=0.25); axes[0, 0].legend()
    axes[0, 1].plot([c["Udc_V"] for c in cases], [c["flatness_1p5mm_J"] * 1.0e9 for c in cases], "o-"); axes[0, 1].set(xlabel="Udc (V)", ylabel="Range inside r≤1.5 mm (nJ)", title="Central flatness metric"); axes[0, 1].grid(alpha=0.25)
    for case in cases:
        values = case["radial_total"][:, int(np.argmin(abs(radii - 0.002)))]
        axes[1, 0].plot(angles_deg, (values - np.mean(values)) * 1.0e9, linewidth=1.0, label=f"{case['Udc_V']:g} V")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.5); axes[1, 0].set(xlabel="Angle (deg)", ylabel="Residual at r=2 mm (nJ)", title="Angular anisotropy comparison"); axes[1, 0].grid(alpha=0.25); axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot([c["Udc_V"] for c in cases], [c["useful_radius_m"] * 1000.0 for c in cases], "o-"); axes[1, 1].set(xlabel="Udc (V)", ylabel="Illustrative radius (mm)", title="Useful localization region"); axes[1, 1].grid(alpha=0.25)
    fig.suptitle("Common DC-bias comparison: RF pseudopotential versus total-field proxy", fontsize=15)
    save_figure(fig, output_dir, "fig_dc_bias_comparison")

    summary_fig, summary_ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for case in cases: summary_ax.plot(radii * 1000.0, (case["radial_mean"] - case["radial_mean"][0]) * 1.0e9, linewidth=1.4, label=f"Udc={case['Udc_V']:g} V")
    summary_ax.set(xlabel="Radius (mm)", ylabel="Angular-mean ΔΨproxy (nJ)", title="DC-bias effect on the central well"); summary_ax.grid(alpha=0.25); summary_ax.legend()
    save_figure(summary_fig, output_dir, "fig_dc_bias_central_well")

    trajectory_rows = []
    for Udc in (0.0, 300.0, 1000.0):
        result = integrate_trajectory(interpolator, config.particle, config.rf.U0, config.rf.frequency, initial_position=(0.00025, 0.0), initial_velocity=(0.0, 0.0), periods=40, escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol, max_step_per_period=10, output_samples_per_period=8, Udc=Udc)
        metrics = trajectory_metrics(result, config.particle, config.simulation.escape_radius)
        trajectory_rows.append({"Udc_V": Udc, **metrics})
    with (output_dir / "trajectory_subset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0])); writer.writeheader(); writer.writerows(trajectory_rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for Udc in (0.0, 300.0, 1000.0):
        result = integrate_trajectory(interpolator, config.particle, config.rf.U0, config.rf.frequency, initial_position=(0.00025, 0.0), initial_velocity=(0.0, 0.0), periods=40, escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol, max_step_per_period=10, output_samples_per_period=8, Udc=Udc)
        ax.plot(result.time * config.rf.frequency, np.hypot(result.state[0], result.state[1]) * 1000.0, label=f"Udc={Udc:g} V")
    ax.set(xlabel="RF periods", ylabel="Radius (mm)", title="Exploratory 40-period trajectory subset"); ax.grid(alpha=0.25); ax.legend(); save_figure(fig, output_dir, "fig_dc_bias_trajectory_subset")

    flatness_rows = [{"Udc_V": float(case["Udc_V"]), "central_min_J": float(case["central_min_J"]), "flatness_range_1p5mm_J": float(case["flatness_1p5mm_J"]), "anisotropy_spread_2mm_J": float(case["anisotropy_2mm_J"]), "useful_radius_m": float(case["useful_radius_m"])} for case in cases]
    with (output_dir / "dc_bias_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flatness_rows[0])); writer.writeheader(); writer.writerows(flatness_rows)
    rf_difference = max(float(np.nanmax(np.abs(case["psi_rf_center"] - baseline["psi_rf_center"]))) for case in cases)
    report_lines = [
        "# Common DC-bias comparison", "",
        "The zero-DC study is preserved. This additional group uses a fifth FEM basis solution, `DC`, with all 12 electrode groups at 1 V and the outer boundary grounded. The full field is reconstructed as Udc·Edc plus the original three-phase RF field; no time-step FEM solves are used.", "",
        f"Bias values: {', '.join(f'{v:g} V' for v in Udc_values)}. Regular plotting grids: full={len(full_axis)}×{len(full_axis)}, center={len(center_axis)}×{len(center_axis)}. Grid fields use area-weighted nodal recovery for readable visualization only; raw P2/direct FEM data are unchanged.", "",
        f"RF-only invariance check: maximum absolute change in ΨRF across all bias cases is {rf_difference:.6e} J (it is the same RF basis calculation by construction).", "",
        "## Interpretation", "",
        "A common DC bias does not change the RF pseudopotential ΨRF qualitatively or quantitatively: the RF phases and RF basis solutions are unchanged. It does add a static electric field and therefore a static force. For visualization, the plotted `Ψproxy = q²⟨|E_RF+E_DC|²⟩/(2mΩ²)` equals ΨRF + q²Udc²|Edc|²/(2mΩ²), because the RF/DC cross term averages to zero over a period; this is a total-field diagnostic, not a claim that static DC energy is an RF pseudopotential.", "",
        "| Udc (V) | central range r≤1.5 mm (nJ) | angular spread at 2 mm (nJ) | illustrative useful radius (mm) |", "|---:|---:|---:|---:|",
    ]
    for row in flatness_rows: report_lines.append(f"| {row['Udc_V']:g} | {row['flatness_range_1p5mm_J'] * 1e9:.6g} | {row['anisotropy_spread_2mm_J'] * 1e9:.6g} | {row['useful_radius_m'] * 1000:.6g} |")
    report_lines += ["", "The central RF well remains a flat-bottom/higher-order minimum in the zero-DC component. Whether the total-field proxy looks less flat or more anisotropic at larger Udc is a quantitative consequence of the added static field term and is summarized by the table and figures above; it should not be confused with a new RF multipole. A common DC bias is therefore not intrinsically beneficial for localization: it may shift the equilibrium and alter confinement through static forces, so the small trajectory subset is exploratory only and is not a stability conclusion.", "", "The trajectory subset uses 40 RF periods at U0=1000 V, f=10000 Hz, r0=0.25 mm, v0=0. It is intentionally small and does not resume or modify the paused full localization checkpoint.", ""]
    report_path = output_dir / "dc_bias_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"output_dir": output_dir, "cases": cases, "report": report_path, "trajectory_rows": trajectory_rows, "rf_invariance_max_J": rf_difference, "solutions": solutions}


def validate_dc_superposition(output_dir: Path, trap_mesh, solutions, config: ProjectConfig) -> dict[str, float]:
    """Compare DC+RF basis reconstruction against independent direct FEM solves."""
    from .fem import element_electric_field, solve_dirichlet
    from .fields import field_at_time_with_dc, potential_at_time_with_dc
    from .voltages import phase_voltages

    records = []
    potential_sq_sum = 0.0
    field_sq_sum = 0.0
    potential_count = 0
    field_count = 0
    max_potential_error = 0.0
    max_field_error = 0.0
    bias_values = (100.0, 500.0, 1000.0)
    phase_fractions = (0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0)
    for Udc in bias_values:
        for fraction in phase_fractions:
            time = fraction * config.rf.period
            rf_values = phase_voltages(time, config.rf.U0, config.rf.frequency)
            electrode_values = {index: float(Udc + rf_values[("A", "B", "C")[index % 3]]) for index in trap_mesh.electrode_nodes}
            direct_potential, _ = solve_dirichlet(trap_mesh, electrode_values)
            direct_field = element_electric_field(trap_mesh, direct_potential, basis=solutions.basis)
            reconstructed_potential = potential_at_time_with_dc(solutions, time, Udc, config.rf.U0, config.rf.frequency)
            reconstructed_field = field_at_time_with_dc(solutions, time, Udc, config.rf.U0, config.rf.frequency)
            potential_error = np.asarray(direct_potential - reconstructed_potential)
            field_error = np.linalg.norm(direct_field - reconstructed_field, axis=-1)
            max_phi = float(np.max(np.abs(potential_error)))
            rms_phi = float(np.sqrt(np.mean(potential_error**2)))
            max_e = float(np.max(field_error))
            rms_e = float(np.sqrt(np.mean(field_error**2)))
            records.append({"Udc_V": Udc, "phase_fraction_T": fraction, "max_potential_error_V": max_phi, "rms_potential_error_V": rms_phi, "max_field_error_V_per_m": max_e, "rms_field_error_V_per_m": rms_e})
            max_potential_error = max(max_potential_error, max_phi)
            max_field_error = max(max_field_error, max_e)
            potential_sq_sum += float(np.sum(potential_error**2)); potential_count += potential_error.size
            field_sq_sum += float(np.sum(field_error**2)); field_count += field_error.size
    summary = {
        "cases": float(len(records)),
        "max_potential_error_V": max_potential_error,
        "rms_potential_error_V": float(np.sqrt(potential_sq_sum / max(potential_count, 1))),
        "max_electric_field_error_V_per_m": max_field_error,
        "rms_electric_field_error_V_per_m": float(np.sqrt(field_sq_sum / max(field_count, 1))),
    }
    with (output_dir / "dc_superposition_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (output_dir / "dc_superposition_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def finalize_dc_bias_outputs(config: ProjectConfig, project_root: Path) -> dict[str, object]:
    """Validate and clarify the completed DC study using its saved outputs.

    This deliberately does not repeat the regular-grid sampling or trajectory
    study. It reads the already completed case CSVs and runs only the requested
    independent direct-FEM superposition check.
    """
    from .pipeline import make_state
    output_dir = publication_dir(project_root) / "dc_bias"
    bias_values = (0, 100, 300, 500, 1000)
    loaded = {}
    center_data = {}
    for bias in bias_values:
        case_dir = output_dir / f"Udc_{bias:04d}V"
        grid = np.genfromtxt(case_dir / "pseudopotential_grid.csv", delimiter=",", names=True)
        central = np.genfromtxt(case_dir / "pseudopotential_central_grid.csv", delimiter=",", names=True)
        radial = np.genfromtxt(case_dir / "radial_cuts.csv", delimiter=",", names=True)
        angle = np.genfromtxt(case_dir / "angular_anisotropy.csv", delimiter=",", names=True)
        x = np.unique(central["x_m"]); y = np.unique(central["y_m"])
        shape = (len(y), len(x))
        loaded[bias] = {name: grid[name] for name in grid.dtype.names}
        center_data[bias] = {
            "x": x, "y": y,
            "psi_rf": central["psi_rf_J"].reshape(shape),
            "proxy": central["psi_total_time_averaged_proxy_J"].reshape(shape),
            "dc_field_basis": central["dc_field_magnitude_V_per_m"].reshape(shape),
            "dc_potential_basis": central["dc_basis_potential_V_per_V"].reshape(shape),
            "radial": radial,
            "angular": angle,
        }

    baseline = center_data[0]
    representative = center_data[1000]
    proxy_delta = representative["proxy"] - baseline["psi_rf"]
    x_mm, y_mm = baseline["x"] * 1000.0, baseline["y"] * 1000.0
    center_extent = (x_mm[0], x_mm[-1], y_mm[0], y_mm[-1])
    xx_mm, yy_mm = np.meshgrid(x_mm, y_mm, indexing="xy")
    central_disk_2mm = np.hypot(xx_mm, yy_mm) <= 2.0
    for bias in bias_values:
        case_dir = output_dir / f"Udc_{bias:04d}V"
        item = center_data[bias]
        # Re-label the primary pseudopotential maps to show RF-only ΨRF.
        full = loaded[bias]
        full_x, full_y = np.unique(full["x_m"]), np.unique(full["y_m"])
        full_psi_rf = full["psi_rf_J"].reshape((len(full_y), len(full_x)))
        full_proxy = full["psi_total_time_averaged_proxy_J"].reshape((len(full_y), len(full_x)))
        fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
        image = ax.imshow(full_psi_rf * 1e9, origin="lower", extent=(full_x[0]*1000, full_x[-1]*1000, full_y[0]*1000, full_y[-1]*1000), cmap="magma", norm=PowerNorm(gamma=0.45, vmin=0, vmax=float(np.nanpercentile(full_psi_rf*1e9, 99.5))), interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"RF pseudopotential ΨRF, Udc={bias} V (unchanged)"); fig.colorbar(image, ax=ax, label="ΨRF (nJ)"); save_figure(fig, case_dir, "pseudopotential_full_map")
        fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
        image = ax.imshow(full_proxy * 1e9, origin="lower", extent=(full_x[0]*1000, full_x[-1]*1000, full_y[0]*1000, full_y[-1]*1000), cmap="magma", norm=PowerNorm(gamma=0.42, vmin=0, vmax=float(np.nanpercentile(full_proxy*1e9, 99.5))), interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Heuristic total-field proxy, Udc={bias} V"); fig.colorbar(image, ax=ax, label="q²⟨|ERF+EDC|²⟩/(2mΩ²) (nJ)"); save_figure(fig, case_dir, "total_field_proxy_full_map")
        fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
        image = ax.imshow(item["psi_rf"] * 1e9, origin="lower", extent=center_extent, cmap="magma", norm=Normalize(vmin=float(np.nanmin(item["psi_rf"]*1e9)), vmax=float(np.nanpercentile(item["psi_rf"]*1e9, 99.5)), clip=True), interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"RF pseudopotential ΨRF central zoom, Udc={bias} V"); fig.colorbar(image, ax=ax, label="ΨRF (nJ)"); save_figure(fig, case_dir, "pseudopotential_central_zoom")
        delta = np.maximum(item["psi_rf"] - float(np.nanmin(item["psi_rf"])), 0.0)
        ref = max(float(np.nanpercentile(delta, 80)), 1.0e-30)
        transformed = (delta / ref) ** (1.0 / 6.0)
        fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
        image = ax.imshow(transformed, origin="lower", extent=center_extent, cmap="viridis", interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"RF-only sixth-root center view, Udc={bias} V"); fig.colorbar(image, ax=ax, label="[(ΨRF−Ψmin)/Ψref]^(1/6)"); save_figure(fig, case_dir, "pseudopotential_transformed_center")
        levels = np.linspace(0.0, float(np.nanpercentile(transformed, 99.5)), 100)[1:]
        fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
        image = ax.contourf(x_mm, y_mm, transformed, levels=levels, cmap="cividis", extend="max"); ax.contour(x_mm, y_mm, transformed, levels=np.linspace(0, levels[-1], 60)[1:], colors="black", linewidths=0.22, alpha=0.5)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"RF-only dense contours, Udc={bias} V (100 levels)"); ax.set_aspect("equal"); fig.colorbar(image, ax=ax, label="Sixth-root transformed RF scale"); save_figure(fig, case_dir, "pseudopotential_dense_contours")
        # Preserve and expose the heuristic as a separate total-field proxy.
        proxy_fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
        image = ax.imshow(item["proxy"] * 1e9, origin="lower", extent=center_extent, cmap="magma", interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Heuristic total-field proxy, Udc={bias} V"); proxy_fig.colorbar(image, ax=ax, label="q²⟨|ERF+EDC|²⟩/(2mΩ²) (nJ)"); save_figure(proxy_fig, case_dir, "total_field_proxy_central_zoom")
        static_energy_delta = config.particle.charge * bias * (item["dc_potential_basis"] - float(item["dc_potential_basis"][len(y_mm)//2, len(x_mm)//2]))
        fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
        image = ax.imshow(static_energy_delta * 1e9, origin="lower", extent=center_extent, cmap="coolwarm", interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Static DC energy relative to center, Udc={bias} V"); fig.colorbar(image, ax=ax, label="qΔΦDC (nJ)"); save_figure(fig, case_dir, "static_dc_potential_energy")
        dc_field = item["dc_field_basis"] * abs(bias)
        fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
        image = ax.imshow(dc_field, origin="lower", extent=center_extent, cmap="viridis", interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Static DC field magnitude, Udc={bias} V"); fig.colorbar(image, ax=ax, label="|EDC| (V/m)"); save_figure(fig, case_dir, "static_dc_field_magnitude")

    # Independent direct FEM validation across three nonzero offsets and six phases.
    geometry, trap_mesh, solutions, _ = make_state(config)
    validation = validate_dc_superposition(output_dir, trap_mesh, solutions, config)

    # Six-panel reader-first summary. Differential panels use fJ to avoid
    # plotting near-identical values with misleading Matplotlib axis offsets.
    fig = plt.figure(figsize=(16.5, 10.5), constrained_layout=True)
    axes = fig.subplot_mosaic([["A", "B", "C"], ["D", "E", "F"]])
    im = axes["A"].imshow(baseline["psi_rf"] * 1e9, origin="lower", extent=center_extent, cmap="magma", interpolation="bicubic", aspect="equal")
    axes["A"].set_title("A  Udc=0: RF pseudopotential ΨRF"); fig.colorbar(im, ax=axes["A"], label="ΨRF (nJ)")
    im = axes["B"].imshow(representative["proxy"] * 1e9, origin="lower", extent=center_extent, cmap="magma", interpolation="bicubic", aspect="equal")
    axes["B"].set_title("B  Udc=1000 V: total-field proxy"); fig.colorbar(im, ax=axes["B"], label="Heuristic proxy (nJ)")
    delta_zoom = np.where(central_disk_2mm, proxy_delta * 1e15, np.nan)
    limit = max(float(np.nanmax(delta_zoom)), 1e-30)
    zoom_extent = (-2.0, 2.0, -2.0, 2.0)
    im = axes["C"].imshow(delta_zoom, origin="lower", extent=center_extent, cmap="magma", norm=Normalize(vmin=0, vmax=limit), interpolation="nearest", aspect="equal")
    axes["C"].set_xlim(zoom_extent[0], zoom_extent[1]); axes["C"].set_ylim(zoom_extent[2], zoom_extent[3])
    axes["C"].set_title("C  Proxy − ΨRF, r≤2 mm (fJ)"); fig.colorbar(im, ax=axes["C"], label="Added static-field proxy term (fJ)")

    radial0 = center_data[0]["radial"]
    radial1 = center_data[1000]["radial"]
    radii = np.unique(radial0["radius_m"])
    means0 = np.array([np.mean(radial0["psi_rf_J"][radial0["radius_m"] == r]) for r in radii])
    means1 = np.array([np.mean(radial1["psi_total_time_averaged_proxy_J"][radial1["radius_m"] == r]) for r in radii])
    axes["D"].axhline(0.0, color="black", linewidth=0.6, label="Udc=0 difference")
    axes["D"].plot(radii*1e3, (means1-means0)*1e15, label="Udc=1000 V proxy − ΨRF", linewidth=1.4)
    axes["D"].set(xlabel="Radius (mm)", ylabel="Proxy increment (fJ)", title="D  Radial differential (RF well itself is unchanged)"); axes["D"].grid(alpha=0.25); axes["D"].legend(fontsize=8)

    angular0 = center_data[0]["angular"]
    angular1 = center_data[1000]["angular"]
    radius2_0 = float(angular0["radius_m"][np.argmin(abs(angular0["radius_m"]-0.002))])
    radius2_1 = float(angular1["radius_m"][np.argmin(abs(angular1["radius_m"]-0.002))])
    mask0, mask1 = np.isclose(angular0["radius_m"], radius2_0), np.isclose(angular1["radius_m"], radius2_1)
    order0 = np.argsort(angular0["angle_deg"][mask0]); order1 = np.argsort(angular1["angle_deg"][mask1])
    angle_delta = angular1["angle_deg"][mask1][order1]
    residual_delta = (angular1["deviation_from_angular_mean_J"][mask1][order1] - angular0["deviation_from_angular_mean_J"][mask0][order0]) * 1e15
    axes["E"].plot(angle_delta, residual_delta, "o-", ms=2.5, linewidth=1.0, color="#8b3a62")
    axes["E"].axhline(0, color="black", linewidth=0.5); axes["E"].set(xlabel="Angle (deg)", ylabel="Change in anisotropy residual (fJ)", title="E  Anisotropy change at r≈2 mm: 1000 V − 0 V"); axes["E"].grid(alpha=0.25)

    trajectory_rows = list(csv.DictReader((output_dir / "trajectory_subset.csv").open("r", newline="", encoding="utf-8")))
    for row in trajectory_rows: row["periods_survived"] = float(row["periods_survived"]); row["Udc_V"] = float(row["Udc_V"])
    axes["F"].bar([str(int(row["Udc_V"])) for row in trajectory_rows], [row["periods_survived"] for row in trajectory_rows], color=("#426b8a", "#c98238", "#a85252"))
    axes["F"].set(xlabel="Common DC bias (V)", ylabel="RF periods to escape / simulated", title="F  Exploratory 40-period trajectory subset"); axes["F"].grid(axis="y", alpha=0.25)
    fig.suptitle("Common-DC-bias comparison: RF potential, DC field, and exploratory dynamics", fontsize=15)
    save_figure(fig, output_dir, "fig_dc_bias_summary")

    # Replace earlier absolute-scale comparison charts with explicit deltas.
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    for bias in bias_values[1:]:
        item = center_data[bias]
        delta = item["proxy"] - baseline["psi_rf"]
        axes[0, 0].plot(bias, float(np.nanmax(delta[central_disk_2mm]))*1e15, "o", label=f"{bias} V")
    axes[0, 0].plot([0], [0], "ko", label="0 V")
    axes[0, 0].set(xlabel="Udc (V)", ylabel="Max proxy increment for r≤2 mm (fJ)", title="Central proxy increment vs. bias"); axes[0, 0].grid(alpha=0.25); axes[0, 0].legend()
    for bias in bias_values:
        radial = center_data[bias]["radial"]
        radii_b = np.unique(radial["radius_m"])
        means_b = np.array([np.mean(radial["psi_total_time_averaged_proxy_J"][radial["radius_m"] == r]) for r in radii_b])
        mean_base = np.array([np.mean(center_data[0]["radial"]["psi_rf_J"][center_data[0]["radial"]["radius_m"] == r]) for r in radii_b])
        if bias:
            axes[0, 1].plot(radii_b*1e3, (means_b-mean_base)*1e15, label=f"{bias} V")
    axes[0, 1].set(xlabel="Radius (mm)", ylabel="Proxy − ΨRF (fJ)", title="Radial differential (Udc=0 is zero)"); axes[0, 1].grid(alpha=0.25); axes[0, 1].legend()
    for bias in bias_values[1:]:
        angle_data = center_data[bias]["angular"]
        ref_data = center_data[0]["angular"]
        mask_b = np.isclose(angle_data["radius_m"], radius2_1); mask_r = np.isclose(ref_data["radius_m"], radius2_0)
        ob = np.argsort(angle_data["angle_deg"][mask_b]); orf = np.argsort(ref_data["angle_deg"][mask_r])
        axes[1, 0].plot(angle_data["angle_deg"][mask_b][ob], (angle_data["deviation_from_angular_mean_J"][mask_b][ob]-ref_data["deviation_from_angular_mean_J"][mask_r][orf])*1e15, label=f"{bias} V")
    axes[1, 0].axhline(0, color="black", linewidth=0.5); axes[1, 0].set(xlabel="Angle (deg)", ylabel="Change at r≈2 mm (fJ)", title="Angular-anisotropy differentials vs. 0 V"); axes[1, 0].grid(alpha=0.25); axes[1, 0].legend()
    dc_magnitudes = [np.nanmax(center_data[b]["dc_field_basis"])*b for b in bias_values]
    axes[1, 1].plot(bias_values, dc_magnitudes, "o-"); axes[1, 1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1, 1].set(xlabel="Udc (V)", ylabel="Max static |EDC| in central grid (V/m)", title="Static field magnitude (not RF pseudopotential)"); axes[1, 1].grid(alpha=0.25)
    fig.suptitle("DC-bias differences relative to the unchanged zero-DC RF solution", fontsize=14)
    save_figure(fig, output_dir, "fig_dc_bias_comparison")

    report_path = output_dir / "dc_bias_report.md"
    existing_metrics = list(csv.DictReader((output_dir / "dc_bias_metrics.csv").open("r", newline="", encoding="utf-8")))
    max_proxy_delta = float(np.nanmax(proxy_delta))
    max_proxy_delta_2mm = float(np.nanmax(proxy_delta[central_disk_2mm]))
    proxy_relative = max_proxy_delta_2mm / max(float(np.ptp(baseline["psi_rf"][central_disk_2mm])), 1e-300)
    max_dc_potential = float(np.nanmax(np.abs(config.particle.charge * 1000.0 * representative["dc_potential_basis"])))
    report_lines = [
        "# Common DC-bias comparison", "",
        "This is an additional comparison; the zero-DC results remain preserved. A fifth FEM basis solution `ΦDC` uses all 12 electrodes at 1 V and the same grounded outer boundary. The full instantaneous field is `E(t)=Udc·EDC+ERF(t)` and is what the trajectory integrator uses. No per-time-step FEM solves are performed.", "",
        "Quantities are kept distinct: (1) RF pseudopotential `ΨRF=q²⟨|ERF|²⟩/(2mΩ²)`; (2) static electrostatic energy `qΦDC` (saved separately); (3) total instantaneous electric field `Udc·EDC+ERF(t)`; and (4) the optional heuristic `q²⟨|ERF+Udc·EDC|²⟩/(2mΩ²)=ΨRF+q²Udc²|EDC|²/(2mΩ²)`. The latter is explicitly a total-field proxy, not the RF pseudopotential. The RF/DC cross term averages to zero because the three RF sinusoids have zero mean.", "",
        f"The direct-FEM check used {int(validation['cases'])} independent direct solves: Udc=100, 500, and 1000 V at six RF phases each. Across all solves: max/RMS potential error = {validation['max_potential_error_V']:.8e} / {validation['rms_potential_error_V']:.8e} V; max/RMS electric-field error = {validation['max_electric_field_error_V_per_m']:.8e} / {validation['rms_electric_field_error_V_per_m']:.8e} V/m. Per-case results are in `dc_superposition_validation.csv`.", "",
        f"The RF-only pseudopotential is unchanged by common Udc (constructed from the same RF basis). At 1000 V, the proxy increment is {max_proxy_delta_2mm:.8e} J ({max_proxy_delta_2mm*1e15:.6g} fJ) for r≤2 mm, only {proxy_relative:.6g} of the central RF-pseudopotential range. Across the wider displayed central grid (to 4.2 mm), its maximum is {max_proxy_delta:.8e} J. It is negligible on the central-well scale. The static field and static potential energy are distinct and available in each case directory; maximum |qΦDC| in the displayed 1000-V central grid is {max_dc_potential:.8e} J, mostly an offset rather than a measure of trapping curvature.", "",
        "## Bias metrics", "", "| Udc (V) | central range r≤1.5 mm (nJ) | angular spread at 2 mm (nJ) | illustrative useful radius (mm) |", "|---:|---:|---:|---:|",
    ]
    for row in existing_metrics:
        report_lines.append(f"| {float(row['Udc_V']):g} | {float(row['flatness_range_1p5mm_J'])*1e9:.8g} | {float(row['anisotropy_spread_2mm_J'])*1e9:.8g} | {float(row['useful_radius_m'])*1000:.6g} |")
    report_lines.extend([
        "", "The RF center remains the same flat-bottom higher-order minimum; common DC does not change its RF multipoles or RF pseudopotential. The added static field changes the exact instantaneous force but makes a negligible contribution to the central time-averaged squared-field proxy for these values. The present numerical evidence does not show a meaningful DC-based expansion of the useful central region or a localization benefit.", "",
        "## Exploratory trajectory subset", "", "One initial condition was tested for Udc=0, 300, and 1000 V over a maximum of 40 periods (U0=1000 V, f=10000 Hz, r0=0.25 mm, v0=0), with the existing ODE tolerances. All three trajectories escaped: 17.7157, 16.5632, and 16.4942 RF periods, respectively. This small probe suggests slightly earlier escape with DC bias for this one case only; it is not a stability result.", "",
        "Figures and grids are grouped under `outputs/publication_study/dc_bias/`; each Udc case includes RF-only pseudopotential, total-field proxy, static DC energy/field views, radial cuts, anisotropy, and CSV grids. The concise six-panel `fig_dc_bias_summary` distinguishes these quantities.", "",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"output_dir": output_dir, "validation": validation, "report": report_path, "max_proxy_increment_J": max_proxy_delta, "relative_proxy_increment": proxy_relative}


def generate_combined_secular_effective_potential(config: ProjectConfig, project_root: Path) -> dict[str, object]:
    """Build physically combined RF+DC potential figures from saved basis grids.

    No FEM solve or field resampling is performed here. The saved zero-DC RF
    pseudopotential and common-DC unit-voltage basis potential are composed as
    U_eff = Psi_RF + q*Udc*Phi_dc_basis.
    """
    source_dir = publication_dir(project_root) / "dc_bias" / "Udc_0000V"
    output_dir = publication_dir(project_root) / "dc_bias" / "combined_effective_potential"
    output_dir.mkdir(parents=True, exist_ok=True)
    charge = float(config.particle.charge)
    aperture = float(config.simulation.escape_radius)
    biases = (0, 300, 500, 1000)

    def load_grid(path: Path):
        data = np.genfromtxt(path, delimiter=",", names=True)
        x, y = np.unique(data["x_m"]), np.unique(data["y_m"])
        shape = (len(y), len(x))
        return x, y, data["psi_rf_J"].reshape(shape), data["dc_basis_potential_V_per_V"].reshape(shape)

    full_x, full_y, full_psi, full_phi = load_grid(source_dir / "pseudopotential_grid.csv")
    center_x, center_y, center_psi, center_phi = load_grid(source_dir / "pseudopotential_central_grid.csv")
    full_extent = (full_x[0] * 1e3, full_x[-1] * 1e3, full_y[0] * 1e3, full_y[-1] * 1e3)
    center_extent = (center_x[0] * 1e3, center_x[-1] * 1e3, center_y[0] * 1e3, center_y[-1] * 1e3)
    full_xx, full_yy = np.meshgrid(full_x, full_y, indexing="xy")
    center_xx, center_yy = np.meshgrid(center_x, center_y, indexing="xy")
    full_radius = np.hypot(full_xx, full_yy)
    center_radius = np.hypot(center_xx, center_yy)
    interp_full_psi = RegularGridInterpolator((full_y, full_x), full_psi, bounds_error=False, fill_value=np.nan)
    interp_full_phi = RegularGridInterpolator((full_y, full_x), full_phi, bounds_error=False, fill_value=np.nan)
    interp_center_psi = RegularGridInterpolator((center_y, center_x), center_psi, bounds_error=False, fill_value=np.nan)
    interp_center_phi = RegularGridInterpolator((center_y, center_x), center_phi, bounds_error=False, fill_value=np.nan)

    def sample_polar(radii: np.ndarray, angles: np.ndarray, bias: float) -> np.ndarray:
        rr, aa = np.meshgrid(radii, angles, indexing="xy")
        xx, yy = rr * np.cos(aa), rr * np.sin(aa)
        points = np.column_stack((yy.ravel(), xx.ravel()))
        use_center = rr.ravel() <= min(center_x[-1], center_y[-1]) * 0.98
        psi = np.empty(len(points), dtype=float)
        phi = np.empty(len(points), dtype=float)
        psi[use_center] = interp_center_psi(points[use_center])
        phi[use_center] = interp_center_phi(points[use_center])
        psi[~use_center] = interp_full_psi(points[~use_center])
        phi[~use_center] = interp_full_phi(points[~use_center])
        return (psi + charge * bias * phi).reshape(rr.shape).T

    # Resolve the central region with the denser saved grid and use the full
    # grid only outside it when locating the minimum within the aperture.
    center_search = (center_radius <= min(aperture, center_x[-1])) & np.isfinite(center_psi) & np.isfinite(center_phi)
    outer_search = (full_radius <= aperture) & (full_radius > min(center_x[-1], center_y[-1]) * 0.98) & np.isfinite(full_psi) & np.isfinite(full_phi)
    center_value = float(interp_center_psi([[0.0, 0.0]])[0])
    central_radii_for_test = np.array([0.0001, 0.00025, 0.0005])
    angles_dense = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    measurement_radii = np.array([0.0005, 0.001, 0.002, 0.003])
    measurement_angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    aperture_angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    radial_radii = np.linspace(0.0, aperture, 321)
    radial_angles = np.deg2rad(np.arange(0.0, 180.0, 15.0))
    results: dict[int, dict[str, object]] = {}
    metrics_rows: list[dict[str, object]] = []

    for bias in biases:
        case_dir = output_dir / f"Udc_{bias:04d}V"
        case_dir.mkdir(parents=True, exist_ok=True)
        central_eff = center_psi + charge * bias * center_phi
        full_eff = full_psi + charge * bias * full_phi
        central_minimum = float(np.nanmin(central_eff[center_search]))
        central_min_index = np.nanargmin(np.where(center_search, central_eff, np.nan))
        ciy, cix = np.unravel_index(central_min_index, central_eff.shape)
        central_min_xy = (float(center_x[cix]), float(center_y[ciy]))
        if np.any(outer_search):
            outer_values = np.where(outer_search, full_eff, np.nan)
            outer_idx = np.nanargmin(outer_values)
            oiy, oix = np.unravel_index(outer_idx, full_eff.shape)
            outer_minimum = float(full_eff[oiy, oix])
            if outer_minimum < central_minimum:
                min_value = outer_minimum
                min_xy = (float(full_x[oix]), float(full_y[oiy]))
            else:
                min_value, min_xy = central_minimum, central_min_xy
        else:
            min_value, min_xy = central_minimum, central_min_xy

        ring_values = sample_polar(central_radii_for_test, angles_dense, bias)
        center_phi_value = float(interp_center_phi([[0.0, 0.0]])[0])
        center_energy = center_value + charge * bias * center_phi_value
        local_margin = float(np.min(ring_values - center_energy))
        half_mm_values = sample_polar(np.array([0.0005]), measurement_angles, bias)[0]
        local_tol = max(abs(float(np.mean(half_mm_values) - center_energy)) * 1.0e-7, 1.0e-30)
        center_is_local_min = bool(local_margin > local_tol)

        point_data = sample_polar(measurement_radii, measurement_angles, bias)
        aperture_values = sample_polar(np.array([aperture]), aperture_angles, bias)[0]
        aperture_barrier = float(np.min(aperture_values))
        well_depth = aperture_barrier - min_value
        radial_data = sample_polar(radial_radii, radial_angles, bias)
        radial_mean = np.mean(radial_data, axis=1)
        radial_min = np.min(radial_data, axis=1)
        radial_max = np.max(radial_data, axis=1)
        radial_center = float(sample_polar(np.array([0.0]), np.array([0.0]), bias)[0, 0])

        # Exact sampled export: absolute Ueff and center-referenced differences
        # are both retained; all display transforms are explicitly secondary.
        with (case_dir / "combined_secular_effective_potential_radial.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("radius_m", "Ueff_angular_mean_J", "Ueff_angular_min_J", "Ueff_angular_max_J", "Ueff_mean_minus_center_J"))
            writer.writerows((float(r), float(m), float(lo), float(hi), float(m-radial_center)) for r, m, lo, hi in zip(radial_radii, radial_mean, radial_min, radial_max))
        with (case_dir / "combined_secular_effective_potential_point_values.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("radius_m", "Ueff_angular_mean_J", "Ueff_angular_min_J", "Ueff_angular_max_J", "Ueff_mean_minus_center_J"))
            for r, vals in zip(measurement_radii, point_data):
                writer.writerow((float(r), float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)), float(np.mean(vals)-radial_center)))
        with (case_dir / "combined_secular_effective_potential_angular_anisotropy.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("radius_m", "angle_deg", "Ueff_J", "deviation_from_angular_mean_J"))
            for r, values in zip(measurement_radii[1:], point_data[1:]):
                for angle, value in zip(np.rad2deg(measurement_angles), values):
                    writer.writerow((float(r), float(angle), float(value), float(value-np.mean(values))))

        result = {"bias": bias, "case_dir": case_dir, "central_eff": central_eff, "full_eff": full_eff,
                  "minimum_J": min_value, "minimum_xy_m": min_xy, "central_minimum_J": central_minimum,
                  "center_J": radial_center, "aperture_barrier_J": aperture_barrier, "well_depth_J": well_depth,
                  "local_minimum": center_is_local_min, "local_margin_J": local_margin,
                  "radial_radii": radial_radii, "radial_mean": radial_mean, "radial_min": radial_min,
                  "radial_max": radial_max, "point_data": point_data}
        results[bias] = result

        full_vmax = float(np.nanpercentile(full_eff, 99.5) * 1e9)
        full_vmin = float(np.nanmin(full_eff) * 1e9)
        if full_vmax <= full_vmin:
            full_vmax = full_vmin + 1e-12
        center_delta_pj = np.maximum(central_eff - central_minimum, 0.0) * 1e12
        center_vmax = max(float(np.nanpercentile(center_delta_pj[center_search], 99.5)), 1e-12)
        transform = np.power(center_delta_pj, 1.0 / 6.0)  # (ΔUeff / 1 pJ)^(1/6)
        transform_vmax = max(float(np.nanpercentile(transform[center_search], 99.5)), 1e-12)
        contour_levels = np.linspace(0.0, center_vmax, 101)[1:]
        x_mm, y_mm = center_x * 1e3, center_y * 1e3
        styles = {"full": ("magma", PowerNorm(gamma=0.45, vmin=max(0.0, full_vmin), vmax=full_vmax)),
                  "center": ("magma", Normalize(vmin=0.0, vmax=center_vmax)),
                  "transform": ("viridis", Normalize(vmin=0.0, vmax=transform_vmax))}

        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        image = ax.imshow(full_eff * 1e9, origin="lower", extent=full_extent, cmap=styles["full"][0], norm=styles["full"][1], interpolation="bicubic", aspect="equal")
        ax.add_patch(Circle((0.0, 0.0), aperture * 1e3, fill=False, color="white", linestyle="--", linewidth=0.8, alpha=0.85))
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Combined secular effective potential, Udc={bias} V")
        fig.colorbar(image, ax=ax, label="Ueff (nJ)"); save_figure(fig, case_dir, "combined_secular_effective_potential_full_map")

        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        image = ax.imshow(center_delta_pj, origin="lower", extent=center_extent, cmap=styles["center"][0], norm=styles["center"][1], interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Center zoom: Ueff − central minimum, Udc={bias} V")
        fig.colorbar(image, ax=ax, label="Ueff − central minimum (pJ)"); save_figure(fig, case_dir, "combined_secular_effective_potential_center_zoom")

        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        image = ax.imshow(transform, origin="lower", extent=center_extent, cmap=styles["transform"][0], norm=styles["transform"][1], interpolation="bicubic", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Transformed combined well, Udc={bias} V")
        fig.colorbar(image, ax=ax, label="[(Ueff − Umin) / 1 pJ]^(1/6)"); save_figure(fig, case_dir, "combined_secular_effective_potential_transformed_center")

        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        image = ax.contourf(x_mm, y_mm, center_delta_pj, levels=contour_levels, cmap="cividis", extend="max")
        ax.contour(x_mm, y_mm, center_delta_pj, levels=np.linspace(0.0, center_vmax, 41)[1:], colors="black", linewidths=0.22, alpha=0.5)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Dense combined-effective-potential contours, Udc={bias} V"); ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label="Ueff − central minimum (pJ)"); save_figure(fig, case_dir, "combined_secular_effective_potential_dense_contours")

        for suffix, radial_mask, title in (("full", radial_radii <= min(aperture, 0.00408), "Central radial profiles"), ("aperture", radial_radii <= aperture, "Radial profiles to aperture")):
            fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
            for angle_deg, vals in zip(np.arange(0.0, 180.0, 15.0), radial_data.T):
                ax.plot(radial_radii[radial_mask]*1e3, (vals[radial_mask]-radial_center)*1e12, linewidth=0.8, alpha=0.78, label=f"{angle_deg:g}°")
            ax.set(xlabel="Radius (mm)", ylabel="Ueff(r) − Ueff(0) (pJ)", title=f"{title}, Udc={bias} V"); ax.grid(alpha=0.25)
            if suffix == "full": ax.legend(ncol=3, fontsize=7)
            save_figure(fig, case_dir, f"combined_secular_effective_potential_radial_{suffix}")

        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        for r_idx in (1, 2, 3):
            vals = point_data[r_idx]
            ax.plot(np.rad2deg(measurement_angles), (vals-np.mean(vals))*1e15, linewidth=1.0, label=f"r={measurement_radii[r_idx]*1e3:g} mm")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set(xlabel="Angle (degrees)", ylabel="Ueff − angular mean (fJ)", title=f"Angular anisotropy of combined well, Udc={bias} V"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        save_figure(fig, case_dir, "combined_secular_effective_potential_angular_anisotropy")

        # Six-panel summary retains the established visual grammar.
        fig = plt.figure(figsize=(16.0, 11.0), constrained_layout=True)
        axes = fig.subplot_mosaic([["A", "B", "C"], ["D", "E", "F"]])
        im = axes["A"].imshow(full_eff*1e9, origin="lower", extent=full_extent, cmap=styles["full"][0], norm=styles["full"][1], interpolation="bicubic", aspect="equal")
        axes["A"].add_patch(Circle((0.0, 0.0), aperture*1e3, fill=False, color="white", linestyle="--", linewidth=0.8)); axes["A"].set_title("A  Full combined potential")
        fig.colorbar(im, ax=axes["A"], label="Ueff (nJ)")
        im = axes["B"].imshow(center_delta_pj, origin="lower", extent=center_extent, cmap=styles["center"][0], norm=styles["center"][1], interpolation="bicubic", aspect="equal")
        axes["B"].set_title("B  Central zoom (local scale)"); fig.colorbar(im, ax=axes["B"], label="Ueff − Umin (pJ)")
        im = axes["C"].imshow(transform, origin="lower", extent=center_extent, cmap=styles["transform"][0], norm=styles["transform"][1], interpolation="bicubic", aspect="equal")
        axes["C"].set_title("C  Sixth-root transformed center"); fig.colorbar(im, ax=axes["C"], label="[(ΔUeff)/(1 pJ)]^(1/6)")
        im = axes["D"].contourf(x_mm, y_mm, center_delta_pj, levels=contour_levels, cmap="cividis", extend="max")
        axes["D"].contour(x_mm, y_mm, center_delta_pj, levels=np.linspace(0.0, center_vmax, 41)[1:], colors="black", linewidths=0.2, alpha=0.5)
        axes["D"].set_title("D  Dense contours (100 filled levels)"); axes["D"].set_aspect("equal"); fig.colorbar(im, ax=axes["D"], label="ΔUeff (pJ)")
        for angle_deg, vals in zip(np.arange(0.0, 180.0, 15.0), radial_data.T):
            mask = radial_radii <= min(0.002, 0.00408)
            axes["E"].plot(radial_radii[mask]*1e3, (vals[mask]-radial_center)*1e12, linewidth=0.75, alpha=0.75, label=f"{angle_deg:g}°")
        axes["E"].set(xlabel="Radius (mm)", ylabel="Ueff(r) − Ueff(0) (pJ)", title="E  Central radial cuts (r≤2 mm)"); axes["E"].grid(alpha=0.25)
        for r_idx in (1, 2, 3):
            vals = point_data[r_idx]
            axes["F"].plot(np.rad2deg(measurement_angles), (vals-np.mean(vals))*1e15, linewidth=0.9, label=f"r={measurement_radii[r_idx]*1e3:g} mm")
        axes["F"].axhline(0, color="black", linewidth=0.5); axes["F"].set(xlabel="Angle (degrees)", ylabel="Residual (fJ)", title="F  Angular anisotropy"); axes["F"].grid(alpha=0.25); axes["F"].legend(fontsize=7)
        fig.suptitle(f"Combined secular effective potential: Udc={bias} V; q={charge:+.3g} C", fontsize=16)
        save_figure(fig, case_dir, "fig_combined_secular_effective_potential_summary")

        row: dict[str, object] = {"Udc_V": bias, "minimum_x_m": min_xy[0], "minimum_y_m": min_xy[1], "minimum_radius_m": float(np.hypot(*min_xy)),
                                  "Ueff_minimum_J": min_value, "Ueff_center_J": radial_center,
                                  "aperture_radius_m": aperture, "minimum_aperture_energy_J": aperture_barrier,
                                  "well_depth_to_aperture_J": well_depth, "center_is_local_minimum": center_is_local_min,
                                  "minimum_centered_ring_rise_J": local_margin}
        for idx, r in enumerate(measurement_radii):
            label = f"r_{r*1e3:g}mm"
            row[f"Ueff_{label}_angular_mean_J"] = float(np.mean(point_data[idx]))
            row[f"Ueff_{label}_angular_min_J"] = float(np.min(point_data[idx]))
            row[f"Ueff_{label}_angular_max_J"] = float(np.max(point_data[idx]))
        metrics_rows.append(row)

    baseline_depth = float(results[0]["well_depth_J"])
    for row in metrics_rows:
        bias = int(row["Udc_V"])
        depth = float(row["well_depth_to_aperture_J"])
        delta_pct = 0.0 if baseline_depth == 0.0 else 100.0 * (depth-baseline_depth)/abs(baseline_depth)
        row["well_depth_change_vs_0DC_percent"] = delta_pct
        row["confinement_change_vs_0DC"] = (
            "reference" if bias == 0 else
            "stronger" if delta_pct > 1.0 else
            "weaker" if delta_pct < -1.0 else
            "slightly stronger (negligible, <1%)" if delta_pct > 0.0 else
            "slightly weaker (negligible, <1%)" if delta_pct < 0.0 else
            "no change"
        )
    fields = list(metrics_rows[0])
    with (output_dir / "combined_secular_effective_potential_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics_rows)

    # Compare center-referenced radial curves so the arbitrary DC energy
    # offset does not hide real shape changes.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), constrained_layout=True)
    palette = {0: "#202020", 300: "#2878b5", 500: "#e07a19", 1000: "#b23a48"}
    for bias in biases:
        item = results[bias]
        rel = (item["radial_mean"]-item["center_J"])
        axes[0].plot(item["radial_radii"]*1e3, rel*1e9, color=palette[bias], linewidth=1.6, label=f"Udc={bias} V")
        axes[1].plot(item["radial_radii"]*1e3, rel*1e12, color=palette[bias], linewidth=1.6, label=f"Udc={bias} V")
    axes[0].axvline(aperture*1e3, linestyle="--", linewidth=0.8, color="#666666", label="aperture")
    axes[0].set(xlim=(0, aperture*1e3), xlabel="Radius (mm)", ylabel="Angular mean Ueff(r) - Ueff(0) (nJ)", title="Full radial comparison (center-referenced)"); axes[0].grid(alpha=0.25); axes[0].legend(ncol=3, fontsize=8)
    central_values_pj = np.concatenate([
        (item["radial_mean"][item["radial_radii"] <= 0.002] - item["center_J"]) * 1e12
        for item in results.values()
    ])
    central_low, central_high = float(np.nanmin(central_values_pj)), float(np.nanmax(central_values_pj))
    central_pad = max((central_high-central_low)*0.08, 1e-6)
    axes[1].set(xlim=(0, 2.0), ylim=(central_low-central_pad, central_high+central_pad), xlabel="Radius (mm)", ylabel="Angular mean Ueff(r) - Ueff(0) (pJ)", title="Central zoom (r≤2 mm)"); axes[1].grid(alpha=0.25); axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("Combined secular effective potential: radial comparison", fontsize=14)
    save_figure(fig, output_dir, "fig_combined_secular_effective_potential_radial_comparison")

    # Small DC-induced shape differences are shown separately from the large
    # common energy offset and the almost-overlapping absolute radial curves.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), constrained_layout=True)
    base = results[0]
    central_difference_values_zj = []
    for bias in biases[1:]:
        item = results[bias]
        delta_shape = (item["radial_mean"]-item["center_J"]) - (base["radial_mean"]-base["center_J"])
        axes[0].plot(item["radial_radii"]*1e3, delta_shape*1e15, color=palette[bias], linewidth=1.5, label=f"Udc={bias} V")
        axes[1].plot(item["radial_radii"]*1e3, delta_shape*1e21, color=palette[bias], linewidth=1.5, label=f"Udc={bias} V")
        central_mask = item["radial_radii"] <= 0.002
        central_difference_values_zj.append(delta_shape[central_mask]*1e21)
    axes[0].axhline(0.0, color="black", linewidth=0.6); axes[0].set(xlim=(0, aperture*1e3), xlabel="Radius (mm)", ylabel="Change in center-referenced Ueff (fJ)", title="Full-range DC-induced shape difference from 0 V"); axes[0].grid(alpha=0.25); axes[0].legend(ncol=3, fontsize=8)
    central_diff = np.concatenate(central_difference_values_zj)
    diff_low, diff_high = float(np.nanmin(central_diff)), float(np.nanmax(central_diff))
    diff_pad = max((diff_high-diff_low)*0.08, 1e-9)
    axes[1].axhline(0.0, color="black", linewidth=0.6); axes[1].set(xlim=(0, 2.0), ylim=(diff_low-diff_pad, diff_high+diff_pad), xlabel="Radius (mm)", ylabel="Change in center-referenced Ueff (zJ)", title="Central zoom: small differences in zeptojoules"); axes[1].grid(alpha=0.25); axes[1].legend(ncol=3, fontsize=8)
    fig.suptitle("Effect of common DC on combined secular-well shape", fontsize=14)
    save_figure(fig, output_dir, "fig_combined_secular_effective_potential_radial_differences")

    report_lines = [
        "# Combined secular effective potential with common DC bias", "",
        "The plotted quantity is the physically combined secular energy `Ueff(x,y) = Psi_RF(x,y) + q*Phi_DC(x,y)`, with `Phi_DC(x,y) = Udc*Phi_dc_basis(x,y)`. `Psi_RF` remains the RF pseudopotential; the static energy contribution is exactly `q*Phi_DC`. The old time-averaged total-field proxy is not used in any figure or metric here. This post-processing reuses the saved, validated RF and common-DC basis grids and does not rerun FEM.", "",
        f"Configured particle charge: `q = {charge:+.8e} C` ({'positive' if charge > 0 else 'negative' if charge < 0 else 'zero'}). The chosen escape/aperture radius is `{aperture*1e3:.6g} mm`. Absolute `Ueff` values include the DC energy offset; radial comparison plots use `Ueff(r)-Ueff(0)` to remove that spatially irrelevant constant while retaining the exact absolute values in the CSV files. Values at fixed radii are angular means, with the full sampled angular range reported in brackets below. The minimum is the lowest saved-grid value inside the aperture (central high-resolution grid for r≤4.08 mm, full grid outside); its location is therefore grid-resolved, not a continuous optimizer estimate.", "",
        "| Udc (V) | min position (mm) | Ueff,min (nJ) | Ueff at 0.5 mm (nJ; min–max) | 1 mm (nJ; min–max) | 2 mm (nJ; min–max) | 3 mm (nJ; min–max) | Ueff(0) (nJ) | aperture well depth (nJ) | center local min? | depth vs 0 DC | confinement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---|"
    ]
    for row in metrics_rows:
        bias = int(row["Udc_V"])
        coords = f"({float(row['minimum_x_m'])*1e3:.4g}, {float(row['minimum_y_m'])*1e3:.4g})"
        cells = []
        for r in (0.5, 1.0, 2.0, 3.0):
            key = f"Ueff_r_{r:g}mm"
            mean, lo, hi = (float(row[f"{key}_{stat}_J"]) for stat in ("angular_mean", "angular_min", "angular_max"))
            cells.append(f"{mean*1e9:.7g} ({lo*1e9:.7g}–{hi*1e9:.7g})")
        report_lines.append(f"| {bias} | {coords} | {float(row['Ueff_minimum_J'])*1e9:.8g} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {float(row['Ueff_center_J'])*1e9:.8g} | {float(row['well_depth_to_aperture_J'])*1e9:.8g} | {'yes' if row['center_is_local_minimum'] else 'no'} | {float(row['well_depth_change_vs_0DC_percent']):+.5g}% | {row['confinement_change_vs_0DC']} |")
    base_row = metrics_rows[0]
    strongest_bias_row = metrics_rows[-1]
    shape_changes = []
    for r in (0.5, 1.0, 2.0, 3.0):
        key = f"Ueff_r_{r:g}mm_angular_mean_J"
        base_rise = float(base_row[key]) - float(base_row["Ueff_center_J"])
        biased_rise = float(strongest_bias_row[key]) - float(strongest_bias_row["Ueff_center_J"])
        percent = 100.0 * (biased_rise-base_rise)/max(abs(base_rise), 1e-300)
        shape_changes.append(f"{r:g} mm: {biased_rise-base_rise:+.4e} J ({percent:+.4g}% of the zero-DC radial rise)")
    report_lines.extend(["", "A local minimum is tested by comparing the center with all angles on 0.10, 0.25, and 0.50 mm rings of the saved central grid. Aperture well depth is the minimum energy on the escape-radius circumference minus the minimum energy in the aperture; it is the conservative barrier over the sampled escape directions. The confinement comparison uses this barrier and marks changes within 1% of the zero-DC depth as not meaningful.", "",
        "## Physical interpretation", "",
        "A common DC offset does not change `Psi_RF`; that RF-only pseudopotential remains exactly the same saved basis-derived quantity. It does change the combined secular effective potential by adding the static term `q*Udc*Phi_dc_basis`, which includes a large nearly constant offset plus any position-dependent variation. Only the latter changes the well shape/confinement. At Udc=1000 V, the aperture well-depth change relative to 0 V is " + f"{float(strongest_bias_row['well_depth_change_vs_0DC_percent']):+.5g}% (below the stated 1% practical-significance threshold). Center-referenced radial-shape changes at the requested radii are: " + "; ".join(shape_changes) + ". The absolute DC energy offset itself is not a confinement effect. On this basis, common DC has a numerically measurable but not physically meaningful confinement effect for this configuration and bias range.", "",
        "## Outputs", "",
        "Each `Udc_XXXXV` directory contains the full map, local-normalized center zoom, sixth-root transformed center, dense contours, radial profiles, angular anisotropy, six-panel summary, and CSV data. The all-bias radial comparison is `fig_combined_secular_effective_potential_radial_comparison.png/.pdf`; the DC-induced center-referenced differences are isolated in `fig_combined_secular_effective_potential_radial_differences.png/.pdf`. Summary metrics are `combined_secular_effective_potential_metrics.csv`. Existing proxy outputs in the parent directories were not changed.", ""])
    report_path = output_dir / "combined_secular_effective_potential_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"output_dir": output_dir, "report": report_path, "metrics": metrics_rows, "biases": biases}


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
