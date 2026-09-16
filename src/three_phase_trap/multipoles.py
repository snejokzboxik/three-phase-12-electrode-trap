"""Numerical harmonic/multipole extraction from sampled FEM potentials."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MultipoleCoefficient:
    order: int
    a: float
    b: float
    amplitude: float
    relative_amplitude: float


def polar_sample_points(radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii, dtype=float)
    angles = np.asarray(angles, dtype=float)
    rr, tt = np.meshgrid(radii, angles, indexing="ij")
    return np.column_stack((rr.ravel() * np.cos(tt.ravel()), rr.ravel() * np.sin(tt.ravel())))


def fit_multipoles(points: np.ndarray, potential: np.ndarray, orders: tuple[int, ...] | list[int], reference_radius: float) -> dict[int, MultipoleCoefficient]:
    """Fit Φ=Σ (r/R)^m [a_m cos(mθ)+b_m sin(mθ)]."""
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    values = np.asarray(potential, dtype=float).reshape(-1)
    if len(xy) != len(values):
        raise ValueError("Points and potential samples must have equal length.")
    if reference_radius <= 0:
        raise ValueError("reference_radius must be positive.")
    r = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    columns: list[np.ndarray] = []
    labels: list[tuple[int, str]] = []
    for order in orders:
        if order < 1:
            raise ValueError("Only positive multipole orders are supported.")
        scale = (r / reference_radius) ** order
        columns.extend((scale * np.cos(order * theta), scale * np.sin(order * theta)))
        labels.extend(((int(order), "a"), (int(order), "b")))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    result: dict[int, MultipoleCoefficient] = {}
    for order in orders:
        a = float(coefficients[labels.index((int(order), "a"))])
        b = float(coefficients[labels.index((int(order), "b"))])
        result[int(order)] = MultipoleCoefficient(int(order), a, b, float(np.hypot(a, b)), 0.0)
    maximum = max((item.amplitude for item in result.values()), default=0.0)
    if maximum > 0:
        result = {order: MultipoleCoefficient(item.order, item.a, item.b, item.amplitude, item.amplitude / maximum) for order, item in result.items()}
    return result


def harmonic_orientation(coefficient: MultipoleCoefficient) -> float:
    """Return the physical angle of the cosine maximum for one harmonic."""
    return float(np.arctan2(coefficient.b, coefficient.a) / coefficient.order)


def fit_rotation(times: np.ndarray, orientations: np.ndarray) -> dict[str, float | np.ndarray]:
    """Fit an unwrapped physical harmonic orientation to θ(t)=ωt+θ₀."""
    times = np.asarray(times, dtype=float)
    raw = np.asarray(orientations, dtype=float)
    unwrapped = np.unwrap(raw)
    design = np.column_stack((times, np.ones_like(times)))
    slope, intercept = np.linalg.lstsq(design, unwrapped, rcond=None)[0]
    predicted = slope * times + intercept
    residuals = unwrapped - predicted
    dof = max(len(times) - 2, 1)
    variance = float(np.sum(residuals**2) / dof)
    covariance = variance * np.linalg.inv(design.T @ design)
    return {
        "angular_rate": float(slope),
        "intercept": float(intercept),
        "rate_uncertainty": float(np.sqrt(max(covariance[0, 0], 0.0))),
        "residual_rms": float(np.sqrt(np.mean(residuals**2))),
        "unwrapped_orientation": unwrapped,
        "residuals": residuals,
    }


def estimate_radial_exponent(radii: np.ndarray, field_magnitudes: np.ndarray) -> tuple[float, float]:
    """Fit log(|E|)=p log(r)+constant and return (p, RMS residual)."""
    r = np.asarray(radii, dtype=float).reshape(-1)
    e = np.asarray(field_magnitudes, dtype=float).reshape(-1)
    keep = np.isfinite(r) & np.isfinite(e) & (r > 0) & (e > 0)
    if keep.sum() < 3:
        raise ValueError("At least three positive finite radial field samples are required.")
    design = np.column_stack((np.log(r[keep]), np.ones(keep.sum())))
    slope, intercept = np.linalg.lstsq(design, np.log(e[keep]), rcond=None)[0]
    residuals = np.log(e[keep]) - (slope * np.log(r[keep]) + intercept)
    return float(slope), float(np.sqrt(np.mean(residuals**2)))

