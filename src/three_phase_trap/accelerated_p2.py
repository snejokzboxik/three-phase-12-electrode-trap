"""Exact, vectorized evaluation of a quadratic P2 FEM field.

This module does not interpolate or recover the field.  It only changes the
algebraic organization of the same elementwise P2 gradient used by
``FEMInterpolator``.  Triangle location is delegated to the validated
scikit-fem locator and barycentric inclusion is checked by the same triangle
geometry.
"""

from __future__ import annotations

import numpy as np

from .interpolation import FEMInterpolator, InterpolationDomainError


class ExactAcceleratedP2:
    """Exact P2 evaluator with scalar and batched phase-aware field calls."""

    def __init__(self, reference: FEMInterpolator):
        self.reference = reference
        self._x = np.asarray(reference._x, dtype=float)
        self._y = np.asarray(reference._y, dtype=float)
        self._triangles = np.asarray(reference._triangles, dtype=np.int64)
        self._dofs = np.asarray(reference._element_dofs, dtype=np.int64)
        self._g = np.asarray(reference._gradients_lambda, dtype=float)
        self._finder = reference._finder
        self._neighbors = np.asarray(reference._neighbors, dtype=np.int64)
        self._last_triangle_id: int | None = None

        # The P2 gradient is affine in barycentric coordinates.  For each
        # phase and element, store G0 + l1*G1 + l2*G2 + l3*G3.  This is
        # algebraically identical to FEMInterpolator._field_values_all_phases.
        self._coeff = {}
        for phase in ("A", "B", "C", "DC"):
            values = np.asarray(reference.solutions.potentials[phase], dtype=float)[self._dofs]
            v0, v1, v2, v3, v4, v5 = values.T
            g0, g1, g2 = self._g[:, 0], self._g[:, 1], self._g[:, 2]
            c0 = -(v0[:, None] * g0 + v1[:, None] * g1 + v2[:, None] * g2)
            c1 = 4.0 * (v0[:, None] * g0 + v3[:, None] * g1 + v5[:, None] * g2)
            c2 = 4.0 * (v1[:, None] * g1 + v3[:, None] * g0 + v4[:, None] * g2)
            c3 = 4.0 * (v2[:, None] * g2 + v4[:, None] * g1 + v5[:, None] * g0)
            self._coeff[phase] = np.stack((c0, c1, c2, c3), axis=1)

    def _barycentric(self, points: np.ndarray, ids: np.ndarray) -> np.ndarray:
        vertices = self._triangles[ids]
        p0 = np.column_stack((self._x[vertices[:, 0]], self._y[vertices[:, 0]]))
        p1 = np.column_stack((self._x[vertices[:, 1]], self._y[vertices[:, 1]]))
        p2 = np.column_stack((self._x[vertices[:, 2]], self._y[vertices[:, 2]]))
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = points - p0
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        return np.column_stack((1.0 - w1 - w2, w1, w2))

    def _locate_scalar(self, point: np.ndarray) -> tuple[int, np.ndarray]:
        point = np.asarray(point, dtype=float).reshape(1, 2)
        if not np.all(np.isfinite(point)):
            raise InterpolationDomainError("Interpolation query contains non-finite coordinates.")
        tid = self._last_triangle_id
        if tid is not None:
            for _ in range(16):
                w = self._barycentric(point, np.array([tid], dtype=np.int64))[0]
                if np.all(w >= -1.0e-10):
                    return tid, w
                nxt = int(self._neighbors[tid, int(np.argmin(w))])
                if nxt < 0:
                    break
                tid = nxt
        found = int(np.asarray(self._finder(point[:, 0], point[:, 1]), dtype=int)[0])
        if found < 0:
            raise InterpolationDomainError(
                f"Point ({point[0, 0]:.6g}, {point[0, 1]:.6g}) is outside the free-space FEM domain."
            )
        self._last_triangle_id = found
        return found, self._barycentric(point, np.array([found], dtype=np.int64))[0]

    def _locate_batch(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = np.asarray(points, dtype=float).reshape(-1, 2)
        if not np.all(np.isfinite(flat)):
            raise InterpolationDomainError("Interpolation query contains non-finite coordinates.")
        ids = np.asarray(self._finder(flat[:, 0], flat[:, 1]), dtype=np.int64)
        if np.any(ids < 0):
            bad = flat[ids < 0][0]
            raise InterpolationDomainError(
                f"Point ({bad[0]:.6g}, {bad[1]:.6g}) is outside the free-space FEM domain."
            )
        return flat, self._barycentric(flat, ids), ids

    @staticmethod
    def _rf_voltages(t: float | np.ndarray, U0: float, frequency: float, phase0: np.ndarray | float = 0.0) -> np.ndarray:
        omega_t = 2.0 * np.pi * frequency * np.asarray(t) + np.asarray(phase0)
        return float(U0) * np.stack((
            np.sin(omega_t),
            np.sin(omega_t - 2.0 * np.pi / 3.0),
            np.sin(omega_t + 2.0 * np.pi / 3.0),
        ), axis=-1)

    def field(self, points: np.ndarray, t: float, U0: float, frequency: float, phase0: float = 0.0) -> np.ndarray:
        """Evaluate the exact RF P2 field at scalar or batched points."""
        arr = np.asarray(points, dtype=float)
        scalar = arr.reshape(-1, 2).shape[0] == 1 and arr.ndim == 1
        flat, weights, ids = self._locate_batch(arr)
        v = self._rf_voltages(t, U0, frequency, phase0).reshape(3)
        c = v[0] * self._coeff["A"][ids] + v[1] * self._coeff["B"][ids] + v[2] * self._coeff["C"][ids]
        w4 = np.column_stack((np.ones(len(weights)), weights))
        out = -np.einsum("nk,nkd->nd", w4, c)
        if scalar:
            return out[0]
        return out.reshape(arr.shape[:-1] + (2,))

    def field_batch(self, points: np.ndarray, t: float, U0: float, frequency: float, phase0: np.ndarray) -> np.ndarray:
        """Exact field for many simultaneous states with distinct initial phases."""
        flat, weights, ids = self._locate_batch(points)
        v = self._rf_voltages(t, U0, frequency, phase0)
        c = v[:, 0, None, None] * self._coeff["A"][ids]
        c = c + v[:, 1, None, None] * self._coeff["B"][ids]
        c = c + v[:, 2, None, None] * self._coeff["C"][ids]
        w4 = np.column_stack((np.ones(len(weights)), weights))
        return -np.einsum("nk,nkd->nd", w4, c)

    def field_with_dc(self, points: np.ndarray, t: float, Udc: float, U0: float, frequency: float, phase0: float = 0.0) -> np.ndarray:
        arr = np.asarray(points, dtype=float)
        flat, weights, ids = self._locate_batch(arr)
        v = self._rf_voltages(t, U0, frequency, phase0).reshape(3)
        c = v[0] * self._coeff["A"][ids] + v[1] * self._coeff["B"][ids] + v[2] * self._coeff["C"][ids]
        c = c + float(Udc) * self._coeff["DC"][ids]
        w4 = np.column_stack((np.ones(len(weights)), weights))
        out = -np.einsum("nk,nkd->nd", w4, c)
        return out[0] if arr.ndim == 1 else out.reshape(arr.shape[:-1] + (2,))
