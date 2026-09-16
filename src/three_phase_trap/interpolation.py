"""Robust piecewise-linear interpolation of FEM potential and fields."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.tri as mtri

from .fem import FEMBasisSolutions, p2_standard_element_dofs
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
        self._element_dofs = p2_standard_element_dofs(self.solutions.trap_mesh, self.solutions.basis)
        triangle_points = np.stack((self._x[self._triangles], self._y[self._triangles]), axis=-1)
        gradients_p1_p2 = np.array([
            np.column_stack((
                np.linalg.solve(p_i[1:] - p_i[0], np.array([1.0, 0.0])),
                np.linalg.solve(p_i[1:] - p_i[0], np.array([0.0, 1.0])),
            ))
            for p_i in triangle_points
        ])
        self._gradients_lambda = np.concatenate(((-gradients_p1_p2[:, [0]] - gradients_p1_p2[:, [1]]), gradients_p1_p2), axis=1)
        self._triangulation = mtri.Triangulation(self._x, self._y, self._triangles)
        self._finder = self._triangulation.get_trifinder()
        self._neighbors = np.full((len(self._triangles), 3), -1, dtype=int)
        edge_owner: dict[tuple[int, int], tuple[int, int]] = {}
        for triangle_id, vertices in enumerate(self._triangles):
            for opposite, edge in enumerate(((vertices[1], vertices[2]), (vertices[2], vertices[0]), (vertices[0], vertices[1]))):
                key = tuple(sorted((int(edge[0]), int(edge[1]))))
                previous = edge_owner.get(key)
                if previous is None:
                    edge_owner[key] = (triangle_id, opposite)
                else:
                    other_triangle, other_opposite = previous
                    self._neighbors[triangle_id, opposite] = other_triangle
                    self._neighbors[other_triangle, other_opposite] = triangle_id
        # Trajectory RHS calls are single-point queries and normally remain in
        # the same triangle for many Runge--Kutta stages. Keep the last cell
        # as a fast path; the exact triangulation finder remains the fallback
        # whenever a step crosses an element edge.
        self._last_triangle_id: int | None = None

    def _query_triangles(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = np.asarray(points, dtype=float).shape[:-1]
        flat = np.asarray(points, dtype=float).reshape(-1, 2)
        if not np.all(np.isfinite(flat)):
            raise InterpolationDomainError("Interpolation query contains non-finite coordinates.")
        if flat.shape[0] == 1 and self._last_triangle_id is not None:
            triangle_id = self._last_triangle_id
            for _ in range(12):
                cached = np.array([triangle_id], dtype=int)
                weights = self._barycentric(flat, cached)
                if np.all(weights[0] >= -1.0e-10):
                    self._last_triangle_id = triangle_id
                    return flat, cached
                next_triangle = int(self._neighbors[triangle_id, int(np.argmin(weights[0]))])
                if next_triangle < 0:
                    break
                triangle_id = next_triangle
        triangle_ids = np.asarray(self._finder(flat[:, 0], flat[:, 1]), dtype=int)
        if np.any(triangle_ids < 0):
            bad = flat[triangle_ids < 0][0]
            raise InterpolationDomainError(f"Point ({bad[0]:.6g}, {bad[1]:.6g}) is outside the free-space FEM domain.")
        if flat.shape[0] == 1:
            self._last_triangle_id = int(triangle_ids[0])
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
        l1, l2, l3 = weights.T
        shape_values = np.column_stack((l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1))
        dofs = self._element_dofs[ids]
        values = self.solutions.potentials[phase][dofs]
        result = np.sum(shape_values * values, axis=1)
        return result.reshape(np.asarray(points).shape[:-1])

    def _field_values(self, flat: np.ndarray, ids: np.ndarray, phase: str) -> np.ndarray:
        """Evaluate one phase's P2 field after triangle lookup/barycentrics."""
        weights = self._barycentric(flat, ids)
        gradients_lambda = self._gradients_lambda[ids]
        l1, l2, l3 = weights.T
        shape_gradients = np.stack((
            (4 * l1 - 1)[:, None] * gradients_lambda[:, 0],
            (4 * l2 - 1)[:, None] * gradients_lambda[:, 1],
            (4 * l3 - 1)[:, None] * gradients_lambda[:, 2],
            4 * (l2[:, None] * gradients_lambda[:, 0] + l1[:, None] * gradients_lambda[:, 1]),
            4 * (l3[:, None] * gradients_lambda[:, 1] + l2[:, None] * gradients_lambda[:, 2]),
            4 * (l1[:, None] * gradients_lambda[:, 2] + l3[:, None] * gradients_lambda[:, 0]),
        ), axis=1)
        values = self.solutions.potentials[phase][self._element_dofs[ids]]
        return -np.einsum("nk,nkd->nd", values, shape_gradients)

    def _field_values_all_phases(self, flat: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Evaluate all three phase fields from one shared P2 geometry query."""
        weights = self._barycentric(flat, ids)
        gradients_lambda = self._gradients_lambda[ids]
        l1, l2, l3 = weights.T
        shape_gradients = np.stack((
            (4 * l1 - 1)[:, None] * gradients_lambda[:, 0],
            (4 * l2 - 1)[:, None] * gradients_lambda[:, 1],
            (4 * l3 - 1)[:, None] * gradients_lambda[:, 2],
            4 * (l2[:, None] * gradients_lambda[:, 0] + l1[:, None] * gradients_lambda[:, 1]),
            4 * (l3[:, None] * gradients_lambda[:, 1] + l2[:, None] * gradients_lambda[:, 2]),
            4 * (l1[:, None] * gradients_lambda[:, 2] + l3[:, None] * gradients_lambda[:, 0]),
        ), axis=1)
        values = np.stack([self.solutions.potentials[phase][self._element_dofs[ids]] for phase in ("A", "B", "C")], axis=0)
        return -np.einsum("pnk,nkd->pnd", values, shape_gradients)

    def field_basis(self, points: np.ndarray, phase: str) -> np.ndarray:
        flat, ids = self._query_triangles(points)
        return self._field_values(flat, ids, phase).reshape(np.asarray(points).shape[:-1] + (2,))

    def _recovered_vertex_field(self, phase: str) -> np.ndarray:
        """Return an area-weighted continuous nodal recovery of the P2 gradient.

        The FEM solution remains unchanged. This is a standard postprocessing
        projection used only for regular-grid visualization, replacing the
        discontinuous elementwise gradient by shared vertex values.
        """
        cache_name = f"_recovered_{phase}"
        cached = getattr(self, cache_name, None)
        if cached is not None:
            return cached
        vertices = self._triangles
        points = np.column_stack((self._x, self._y))
        accumulated = np.zeros((len(points), 2), dtype=float)
        weights_accumulated = np.zeros(len(points), dtype=float)
        for triangle_id, triangle in enumerate(vertices):
            p = points[triangle]
            area = 0.5 * abs(float(np.cross(p[1] - p[0], p[2] - p[0])))
            gradients_lambda = self._gradients_lambda[triangle_id]
            dofs = self._element_dofs[triangle_id]
            values = self.solutions.potentials[phase][dofs]
            for local_vertex in range(3):
                barycentric = np.zeros(3, dtype=float)
                barycentric[local_vertex] = 1.0
                l1, l2, l3 = barycentric
                shape_gradients = np.stack((
                    (4 * l1 - 1) * gradients_lambda[0],
                    (4 * l2 - 1) * gradients_lambda[1],
                    (4 * l3 - 1) * gradients_lambda[2],
                    4 * (l2 * gradients_lambda[0] + l1 * gradients_lambda[1]),
                    4 * (l3 * gradients_lambda[1] + l2 * gradients_lambda[2]),
                    4 * (l1 * gradients_lambda[2] + l3 * gradients_lambda[0]),
                ), axis=0)
                field = -np.einsum("k,kd->d", values, shape_gradients)
                vertex = int(triangle[local_vertex])
                accumulated[vertex] += area * field
                weights_accumulated[vertex] += area
        recovered = accumulated / np.maximum(weights_accumulated[:, None], 1.0e-300)
        setattr(self, cache_name, recovered)
        return recovered

    def field_basis_recovered(self, points: np.ndarray, phase: str) -> np.ndarray:
        """Evaluate the area-weighted continuous nodal-gradient recovery."""
        flat, ids = self._query_triangles(points)
        weights = self._barycentric(flat, ids)
        recovered = self._recovered_vertex_field(phase)
        values = recovered[self._triangles[ids]]
        return np.einsum("nk,nkd->nd", weights, values).reshape(np.asarray(points).shape[:-1] + (2,))

    def potential(self, points: np.ndarray, t: float, U0: float, frequency: float) -> np.ndarray:
        values = sum(float(phase_voltages(t, U0, frequency)[phase]) * self.potential_basis(points, phase) for phase in ("A", "B", "C"))
        return values

    def field(self, points: np.ndarray, t: float, U0: float, frequency: float) -> np.ndarray:
        flat, ids = self._query_triangles(points)
        voltages = phase_voltages(t, U0, frequency)
        phase_fields = self._field_values_all_phases(flat, ids)
        values = np.einsum("p,pnd->nd", np.array([voltages[phase] for phase in ("A", "B", "C")], dtype=float), phase_fields)
        return values.reshape(np.asarray(points).shape[:-1] + (2,))

    def field_recovered(self, points: np.ndarray, t: float, U0: float, frequency: float) -> np.ndarray:
        """Evaluate the continuous recovered field for visualization only."""
        flat, ids = self._query_triangles(points)
        weights = self._barycentric(flat, ids)
        vertex_fields = np.stack([self._recovered_vertex_field(phase)[self._triangles[ids]] for phase in ("A", "B", "C")], axis=0)
        values = np.einsum("p,pnd->nd", np.array([phase_voltages(t, U0, frequency)[phase] for phase in ("A", "B", "C")], dtype=float), np.einsum("nk,pnkd->pnd", weights, vertex_fields))
        return values.reshape(np.asarray(points).shape[:-1] + (2,))

    def valid(self, points: np.ndarray) -> bool:
        try:
            self._query_triangles(points)
        except InterpolationDomainError:
            return False
        return True
