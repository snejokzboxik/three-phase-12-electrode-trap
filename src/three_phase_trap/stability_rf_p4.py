"""Native P4 escape integration with a terminal event on the audited dense curve.

The frozen trajectory integrator is kept separate. This driver reuses its
force, element ownership and dense-curve audit for DOP853, but stores only
the terminal state and step statistics.
"""
import math

import numpy as np
from scipy.integrate import DOP853
from scipy.optimize import brentq

from .audited_rf import (
    EPS, GeometryAuditError, bernstein_value, dense_barycentric, first_exit,
    membership_bound, split_bernstein,
)
from .direct_fem_dynamics import clock_membership_tolerance, forward_cell, local_rhs


DEGREE = 7
SQUARE_DEGREE = 2 * DEGREE
PRODUCT_WEIGHTS = np.array([
    [math.comb(DEGREE, i) * math.comb(DEGREE, j) / math.comb(SQUARE_DEGREE, i + j)
     for j in range(DEGREE + 1)] for i in range(DEGREE + 1)
])
PRODUCT_INDEX = np.add.outer(np.arange(DEGREE + 1), np.arange(DEGREE + 1)).ravel()
RADIAL_POWER = np.array([
    [math.comb(SQUARE_DEGREE, i) * math.comb(SQUARE_DEGREE - i, k - i) * (-1.)**(k - i) if i <= k else 0.
     for i in range(SQUARE_DEGREE + 1)] for k in range(SQUARE_DEGREE + 1)
])


def first_radial_exit(points, radius):
    """First outward crossing of a disk by a degree-seven Bernstein curve."""
    xy = points / radius
    if np.max(np.sum(xy**2, axis=1)) < 1 - 1024 * EPS:
        return None
    square = np.bincount(PRODUCT_INDEX, weights=((xy @ xy.T) * PRODUCT_WEIGHTS).ravel())
    gap = 1 - square
    tolerance = 2048 * EPS
    if gap[0] < -tolerance:
        raise GeometryAuditError("Dense step starts outside escape disk")
    if np.min(gap) >= -tolerance:
        return None
    # Roots of the derivative partition the curve into monotone intervals.
    # Use control points for values and independently audit the retained prefix.
    derivative = RADIAL_POWER @ (gap - gap[0])
    derivative = np.polynomial.polynomial.polyder(derivative)
    critical = np.polynomial.polynomial.polyroots(derivative)
    knots = sorted(float(root.real) for root in critical if abs(root.imag) < 1e-8 and 0 < root.real < 1)
    lo, value = 0., gap[0]
    for hi in (*knots, 1.):
        next_value = float(bernstein_value(gap, hi))
        if next_value < -tolerance:
            root = lo if value <= 0 else brentq(lambda u: float(bernstein_value(gap, u)), lo, hi, xtol=1e-15)
            prefix, _ = split_bernstein(gap[:, None], root)
            membership_bound(prefix, tolerance)
            return float(root)
        lo, value = hi, next_value
    membership_bound(gap[:, None], tolerance)
    return None


class NativeEscape:
    def __init__(self, field, radius):
        self.field = field
        self.radius = radius
        self.vertices = field.points[field.triangles]
        vertex_radius = np.linalg.norm(self.vertices, axis=2)
        self.near_exit = vertex_radius.max(axis=1) >= radius * (1 - 1e-10)
        self.maximum_gradient = float(np.max(np.linalg.norm(field.gradient, axis=-1)))

    def run(self, initial, qm, frequency, phase, periods, *, steps, rtol, atol):
        field, end = self.field, periods / frequency
        omega, base_tolerance = 2 * np.pi * frequency, 256 * EPS
        cell = int(field._finder(*initial[:2]))
        if cell < 0 or np.linalg.norm(initial[:2]) >= self.radius:
            raise GeometryAuditError("Invalid P4 escape initial point")
        cell = forward_cell(field, cell, initial, 0., qm, omega, phase, base_tolerance)

        def create(clock, state, owner):
            return DOP853(local_rhs(field, owner, qm, omega, phase), clock, state, end,
                          max_step=1 / (frequency * steps), rtol=rtol, atol=atol)

        solver = create(0., initial, cell)
        crossings, nfev, zero_time = 0, 0, 0
        maximum = float(np.linalg.norm(initial[:2]))
        while solver.status == "running":
            previous = solver.t
            solver.step()
            if solver.status == "failed":
                raise GeometryAuditError("Native P4 escape integration failed")
            dense = solver.dense_output()
            bary = dense_barycentric(dense, field.origin[cell], field.gradient[cell])
            speed = max(np.linalg.norm(solver.y[2:]), np.linalg.norm(dense.y_old[2:]))
            tolerance = clock_membership_tolerance(solver.t, speed, self.maximum_gradient, base_tolerance)
            boundary = first_exit(bary, tolerance)
            fraction = 1. if boundary is None else boundary[0]
            if self.near_exit[cell]:
                retained = bary if boundary is None else split_bernstein(bary, fraction)[0]
                radial = first_radial_exit(retained @ self.vertices[cell], self.radius)
                if radial is not None:
                    clock = previous + (solver.t - previous) * fraction * radial
                    state = dense(clock)
                    if abs(np.linalg.norm(state[:2]) / self.radius - 1) > 1e-8:
                        raise GeometryAuditError("Native P4 radial event failed its state check")
                    return clock * frequency, state, self.radius, nfev + solver.nfev
            stop = solver.t if boundary is None else previous + (solver.t - previous) * fraction
            state = solver.y if boundary is None else dense(stop)
            maximum = max(maximum, float(np.linalg.norm(state[:2])))
            if boundary is not None and stop < end:
                other = forward_cell(field, cell, state, stop, qm, omega, phase, tolerance)
                if other == cell:
                    raise GeometryAuditError("Native P4 crossing did not change owner")
                crossings += 1
                zero_time = zero_time + 1 if stop == previous else 0
                if zero_time > 8 or crossings > 100000:
                    raise GeometryAuditError("Native P4 boundary chatter")
                nfev += solver.nfev
                cell, solver = other, create(stop, state, other)
        return None, solver.y, maximum, nfev + solver.nfev
