"""Static plots and fixed-normalization RF potential animation."""

from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from .config import RFConfig
from .dynamics import TrajectoryResult
from .geometry import TrapGeometry
from .interpolation import FEMInterpolator
from .voltages import voltage_matrix


PHASE_COLORS = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}


def draw_electrodes(ax, geometry: TrapGeometry, annotate: bool = True) -> None:
    for electrode in geometry.electrodes:
        patch = Circle(electrode.center, geometry.electrode_radius, facecolor=PHASE_COLORS[electrode.phase], edgecolor="black", linewidth=0.8, alpha=0.9)
        ax.add_patch(patch)
        if annotate:
            ax.text(electrode.center[0], electrode.center[1], str(electrode.index), ha="center", va="center", fontsize=7, color="white", weight="bold")


def prepare_grid(interpolator: FEMInterpolator, radius: float, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-radius, radius, size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    flat = np.column_stack((xx.ravel(), yy.ravel()))
    triangle_ids = np.asarray(interpolator._finder(flat[:, 0], flat[:, 1]), dtype=int)
    mask = triangle_ids >= 0
    basis_potential = np.full((3, len(flat)), np.nan)
    basis_field = np.full((3, len(flat), 2), np.nan)
    for i, phase in enumerate(("A", "B", "C")):
        if np.any(mask):
            basis_potential[i, mask] = interpolator.potential_basis(flat[mask], phase)
            basis_field[i, mask] = interpolator.field_basis(flat[mask], phase)
    return axis, xx, yy, mask.reshape(xx.shape), basis_potential.reshape((3,) + xx.shape), basis_field.reshape((3,) + xx.shape + (2,))


def save_phase_voltage_diagnostics(output_dir: Path, rf: RFConfig, samples: int = 720) -> dict[str, float]:
    times = np.linspace(0.0, 2.0 * rf.period, samples + 1)
    values = voltage_matrix(times, rf.U0, rf.frequency)
    path = output_dir / "phase_voltages.csv"
    np.savetxt(path, np.column_stack((times / rf.period, times, values, values.sum(axis=1))), delimiter=",", header="t_over_T,time_s,U_A_V,U_B_V,U_C_V,sum_V", comments="")
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for i, phase in enumerate(("A", "B", "C")):
        ax.plot(times / rf.period, values[:, i], label=f"U_{phase}", color=PHASE_COLORS[phase])
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set(xlabel="Time / RF period", ylabel="Voltage (V)", title="Continuous three-phase RF voltages")
    ax.legend(ncol=3)
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "phase_voltages.png", dpi=150)
    plt.close(fig)
    return {"max_three_phase_sum_error": float(np.max(np.abs(values.sum(axis=1))))}


def _plot_frame(ax, values: np.ndarray, axis: np.ndarray, geometry: TrapGeometry, title: str, cmap: str = "coolwarm", norm: Normalize | None = None, contours: bool = False) -> None:
    image = ax.imshow(values, origin="lower", extent=(axis[0], axis[-1], axis[0], axis[-1]), cmap=cmap, norm=norm)
    if contours:
        finite = np.isfinite(values)
        if np.count_nonzero(finite) > 20:
            levels = np.linspace(np.nanmin(values), np.nanmax(values), 8)[1:-1]
            if len(levels):
                ax.contour(axis, axis, np.ma.masked_invalid(values), levels=levels, colors="black", linewidths=0.35, alpha=0.6)
    draw_electrodes(ax, geometry)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    return image


def save_static_snapshots(output_dir: Path, interpolator: FEMInterpolator, geometry: TrapGeometry, rf: RFConfig, grid_size: int = 121) -> dict[str, object]:
    display_radius = min(0.022, geometry.outer_domain_radius * 0.92)
    axis, _, _, mask, basis_potential, basis_field = prepare_grid(interpolator, display_radius, grid_size)
    times = np.arange(7, dtype=float) / 12.0 * rf.period
    voltage_values = voltage_matrix(times, rf.U0, rf.frequency)
    potential_frames = np.einsum("tp,pyx->tyx", voltage_values, basis_potential)
    field_frames = np.einsum("tp,pyxk->tyxk", voltage_values, basis_field)
    magnitude_frames = np.linalg.norm(field_frames, axis=-1)
    vnorm = Normalize(vmin=float(np.nanmin(potential_frames)), vmax=float(np.nanmax(potential_frames)))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        if i < len(times):
            image = _plot_frame(ax, potential_frames[i], axis, geometry, f"t/T = {i}/12", norm=vnorm, contours=False)
        else:
            ax.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Potential (V)")
    fig.suptitle("Potential snapshots over one RF period")
    fig.savefig(output_dir / "potential_snapshots.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), constrained_layout=True)
    step = max(1, grid_size // 17)
    qx, qy = np.meshgrid(axis[::step], axis[::step], indexing="xy")
    for i, ax in enumerate(axes.ravel()):
        if i < len(times):
            image = _plot_frame(ax, magnitude_frames[i], axis, geometry, f"t/T = {i}/12", cmap="viridis")
            vectors = field_frames[i, ::step, ::step]
            good = np.isfinite(vectors).all(axis=-1)
            ax.quiver(qx[good], qy[good], vectors[..., 0][good], vectors[..., 1][good], color="white", alpha=0.75, scale=1.0e6, width=0.002)
        else:
            ax.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Electric-field magnitude (V/m)")
    fig.suptitle("Electric-field magnitude and sparse field vectors")
    fig.savefig(output_dir / "field_snapshots.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        if i < len(times):
            image = _plot_frame(ax, potential_frames[i], axis, geometry, f"t/T = {i}/12", norm=vnorm, contours=True)
        else:
            ax.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Potential (V)")
    fig.suptitle("Equipotential snapshots over one RF period")
    fig.savefig(output_dir / "equipotential_snapshots.png", dpi=150)
    plt.close(fig)
    return {"axis": axis, "mask": mask, "basis_potential": basis_potential, "basis_field": basis_field, "times": times, "potential_frames": potential_frames, "field_frames": field_frames}


def save_potential_animation(output_dir: Path, interpolator: FEMInterpolator, geometry: TrapGeometry, rf: RFConfig, frames_per_period: int, periods: int, grid_size: int = 101) -> dict[str, float | bool]:
    display_radius = min(0.022, geometry.outer_domain_radius * 0.92)
    axis, _, _, _, basis_potential, _ = prepare_grid(interpolator, display_radius, grid_size)
    frame_count = int(frames_per_period * periods)
    times = np.arange(frame_count, dtype=float) / frames_per_period * rf.period
    voltage_values = voltage_matrix(times, rf.U0, rf.frequency)
    frames = np.einsum("tp,pyx->tyx", voltage_values, basis_potential)
    norm = Normalize(vmin=float(np.nanmin(frames)), vmax=float(np.nanmax(frames)))
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    image = ax.imshow(frames[0], origin="lower", extent=(axis[0], axis[-1], axis[0], axis[-1]), cmap="coolwarm", norm=norm, animated=True)
    draw_electrodes(ax, geometry)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    colorbar = fig.colorbar(image, ax=ax, label="Potential (V)")
    del colorbar
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", color="black", bbox={"facecolor": "white", "alpha": 0.75, "pad": 3})

    def update(i: int):
        image.set_data(frames[i])
        text.set_text(f"t/T = {times[i] / rf.period:.3f}\nU_A = {voltage_values[i, 0]:+.1f} V\nU_B = {voltage_values[i, 1]:+.1f} V\nU_C = {voltage_values[i, 2]:+.1f} V")
        return image, text

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000.0 / max(frames_per_period / 2, 1), blit=False)
    gif_path = output_dir / "potential_animation.gif"
    animation.save(gif_path, writer=PillowWriter(fps=max(15, frames_per_period // 6)), dpi=90)
    mp4_created = False
    if shutil.which("ffmpeg"):
        try:
            animation.save(output_dir / "potential_animation.mp4", writer="ffmpeg", dpi=100, bitrate=1800)
            mp4_created = True
        except Exception:
            mp4_created = False
    plt.close(fig)
    return {"fixed_vmin": float(norm.vmin), "fixed_vmax": float(norm.vmax), "frame_count": frame_count, "mp4_created": mp4_created}


def save_trajectory_plots(output_dir: Path, result: TrajectoryResult, title_suffix: str = "") -> None:
    t = result.time
    x, y, vx, vy = result.state
    r = np.hypot(x, y)
    title = f" ({title_suffix})" if title_suffix else ""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot(x, y, linewidth=0.8)
    ax.scatter([x[0]], [y[0]], color="green", label="start", zorder=3)
    ax.scatter([x[-1]], [y[-1]], color="red", label="finish", zorder=3)
    ax.set(xlabel="x (m)", ylabel="y (m)", title=f"Particle trajectory{title}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "trajectory_xy.png", dpi=150)
    plt.close(fig)
    for value, label, filename in ((x, "x (m)", "trajectory_x_t.png"), (y, "y (m)", "trajectory_y_t.png"), (r, "radius (m)", "trajectory_radius_t.png")):
        fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
        ax.plot(t, value, linewidth=0.8)
        ax.set(xlabel="Time (s)", ylabel=label, title=f"{label} versus time{title}")
        ax.grid(alpha=0.25)
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
