"""Robust piecewise-linear interpolation of FEM potential and fields."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.tri as mtri

from .fem import FEMBasisSolutions
from .fields import field_at_time, potential_at_time
from .voltages import phase_voltages


class InterpolationDomainError(ValueError):
    """Raised when a query lies outside free-space FEM triangles."""


@dataclass
class FEMInterpolator:
    solutions: FEMBasisSolutions

    def __post_init__(self) -> None:
        points = self.solutions.points
        self._x = points[:, 0]
        self._y = points[:, 1]
        self._triangles = self.solutions.triangles
        self._triangulation = mtri.Triangulation(self._x, self._y, self._triangles)
        self._finder = self._triangulation.get_trifinder()

    def _query_triangles(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = np.asarray(points, dtype=float).shape[:-1]
        flat = np.asarray(points, dtype=float).reshape(-1, 2)
        if not np.all(np.isfinite(flat)):
            raise InterpolationDomainError("Interpolation query contains non-finite coordinates.")
        triangle_ids = np.asarray(self._finder(flat[:, 0], flat[:, 1]), dtype=int)
        if np.any(triangle_ids < 0):
            bad = flat[triangle_ids < 0][0]
            raise InterpolationDomainError(f"Point ({bad[0]:.6g}, {bad[1]:.6g}) is outside the free-space FEM domain.")
        return flat, triangle_ids

    def _barycentric(self, flat: np.ndarray, triangle_ids: np.ndarray) -> np.ndarray:
        vertices = self._triangles[triangle_ids]
        p0 = np.column_stack((self._x[vertices[:, 0]], self._y[vertices[:, 0]]))
        p1 = np.column_stack((self._x[vertices[:, 1]], self._y[vertices[:, 1]]))
        p2 = np.column_stack((self._x[vertices[:, 2]], self._y[vertices[:, 2]]))
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = flat - p0
        denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / denominator
        w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / denominator
        return np.column_stack((1.0 - w1 - w2, w1, w2))

    def potential_basis(self, points: np.ndarray, phase: str) -> np.ndarray:
        flat, ids = self._query_triangles(points)
        weights = self._barycentric(flat, ids)
        vertices = self._triangles[ids]
        values = self.solutions.potentials[phase][vertices]
        result = np.sum(weights * values, axis=1)
        return result.reshape(np.asarray(points).shape[:-1])

    def field_basis(self, points: np.ndarray, phase: str) -> np.ndarray:
        flat, ids = self._query_triangles(points)
        values = self.solutions.fields[phase][ids]
        return values.reshape(np.asarray(points).shape[:-1] + (2,))

    def potential(self, points: np.ndarray, t: float, U0: float, frequency: float) -> np.ndarray:
        values = sum(float(phase_voltages(t, U0, frequency)[phase]) * self.potential_basis(points, phase) for phase in ("A", "B", "C"))
        return values

    def field(self, points: np.ndarray, t: float, U0: float, frequency: float) -> np.ndarray:
        values = sum(float(phase_voltages(t, U0, frequency)[phase]) * self.field_basis(points, phase) for phase in ("A", "B", "C"))
        return values

    def valid(self, points: np.ndarray) -> bool:
        try:
            self._query_triangles(points)
        except InterpolationDomainError:
            return False
        return True
