"""Time-averaged RF effective-potential calculations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .interpolation import FEMInterpolator


@dataclass(frozen=True)
class EffectivePotentialResult:
    mean_field_squared: np.ndarray
    effective_potential: np.ndarray
    classification: str
    hessian_eigenvalues: tuple[float, float]


def mean_square_field_from_basis(fields: dict[str, np.ndarray], U0: float) -> np.ndarray:
    """Use exact three-phase time averaging, equivalent to one RF-period quadrature.

    For phase offsets 0, -2π/3, +2π/3,
    <|E|²> = U0²/2 (|EA|²+|EB|²+|EC|² - EA·EB - EA·EC - EB·EC).
    """
    ea, eb, ec = (np.asarray(fields[p], dtype=float) for p in ("A", "B", "C"))
    norms = np.sum(ea * ea, axis=-1) + np.sum(eb * eb, axis=-1) + np.sum(ec * ec, axis=-1)
    dots = np.sum(ea * eb, axis=-1) + np.sum(ea * ec, axis=-1) + np.sum(eb * ec, axis=-1)
    return 0.5 * U0**2 * (norms - dots)


def effective_potential_from_mean_square(mean_field_squared: np.ndarray, charge: float, mass: float, omega: float) -> np.ndarray:
    """Return the ponderomotive potential q²<E²>/(2mΩ²).

    This normalization reproduces q²E0²/(4mΩ²) for a single field
    E(t)=E0 cos(Ωt), because <E²>=E0²/2.
    """
    return charge**2 * np.asarray(mean_field_squared) / (2.0 * mass * omega**2)


def time_average_field_squared(interpolator: FEMInterpolator, points: np.ndarray, U0: float, frequency: float, samples: int = 240) -> np.ndarray:
    """Numerically average the actual reconstructed field over one full period."""
    times = np.arange(samples, dtype=float) / samples / frequency
    values = np.zeros(np.asarray(points).shape[:-1], dtype=float)
    for time in times:
        field = interpolator.field(points, time, U0, frequency)
        values += np.sum(field * field, axis=-1)
    return values / samples


def classify_center(field_squared_grid: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> tuple[str, tuple[float, float]]:
    """Classify the center from the local Hessian of the effective-potential proxy."""
    values = np.asarray(field_squared_grid, dtype=float)
    ix = int(np.argmin(np.abs(x_axis)))
    iy = int(np.argmin(np.abs(y_axis)))
    if ix < 1 or ix + 1 >= len(x_axis) or iy < 1 or iy + 1 >= len(y_axis):
        return "undetermined", (float("nan"), float("nan"))
    dx = float(x_axis[ix + 1] - x_axis[ix])
    dy = float(y_axis[iy + 1] - y_axis[iy])
    dxx = (values[iy, ix + 1] - 2 * values[iy, ix] + values[iy, ix - 1]) / dx**2
    dyy = (values[iy + 1, ix] - 2 * values[iy, ix] + values[iy - 1, ix]) / dy**2
    dxy = (values[iy + 1, ix + 1] - values[iy + 1, ix - 1] - values[iy - 1, ix + 1] + values[iy - 1, ix - 1]) / (4 * dx * dy)
    eigenvalues = np.linalg.eigvalsh(np.array([[dxx, dxy], [dxy, dyy]]))
    # P1 gradients are piecewise constant, so a Hessian assembled from four
    # neighboring pixels can be dominated by triangle-edge jumps.  Use the
    # local directional stencil for the categorical decision and retain the
    # Hessian eigenvalues as a diagnostic rather than over-interpreting them.
    center = float(values[iy, ix])
    neighbors = np.array([values[iy, ix - 1], values[iy, ix + 1], values[iy - 1, ix], values[iy + 1, ix]])
    tolerance = max(abs(center) * 1.0e-6, np.nanmax(np.abs(values)) * 1.0e-12, 1.0e-30)
    if np.all(neighbors > center + tolerance):
        classification = "local minimum"
    elif np.all(neighbors < center - tolerance):
        classification = "local maximum"
    elif np.any(neighbors < center - tolerance) and np.any(neighbors > center + tolerance):
        classification = "saddle"
    else:
        classification = "approximately flat"
    return classification, (float(eigenvalues[0]), float(eigenvalues[1]))
