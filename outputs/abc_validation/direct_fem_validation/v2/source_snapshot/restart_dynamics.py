"""Independent Radau time discretization for short raw-FEM restart tests.

Geometry checks and boundary ownership are shared with audited_rf. The ODE
method, error estimator, stages and dense interpolant differ from DOP853.
No smoothing of the P2 field or changes to stored pilot trajectories.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import Radau

from .audited_rf import (EPS, AuditedRFRun, GeometryAuditError, _barycentric,
                         _rhs, bernstein_value, choose_forward_triangle,
                         first_exit, membership_bound, split_bernstein)
from .interpolation import InterpolationDomainError
from .pilot_dynamics import RFRun

ELEVATE_CUBIC = np.array([[math.comb(j, k)/math.comb(7, k) if k <= j else 0.
                           for k in range(4)] for j in range(8)])


def radau_barycentric(dense, origin, gradients):
    """Exact degree elevation of SciPy's cubic dense polynomial to degree 7."""
    if not hasattr(dense, "Q") or np.shape(dense.Q) != (4, 3):
        raise GeometryAuditError("Unsupported Radau dense-output representation")
    power = np.vstack((dense.y_old[:2]-origin, dense.Q[:2].T))@gradients.T
    power[0, 0] += 1
    return ELEVATE_CUBIC@power


def affine_state_jacobian(raw, triangle, qm, omega, phase):
    c, s = raw.coefficients[0][triangle], raw.coefficients[1][triangle]
    jc, js = c[[1, 2, 4, 5]].reshape(2, 2), s[[1, 2, 4, 5]].reshape(2, 2)
    def jacobian(t, state):
        jac = np.zeros((4, 4))
        jac[0, 2] = jac[1, 3] = 1.
        jac[2:, :2] = qm*(jc*math.cos(omega*t+phase)+js*math.sin(omega*t+phase))
        return jac
    return jacobian


def integrate_radau(raw, initial, charge, mass, frequency, phase, start_time, end_time, *,
                    max_steps_per_period=50, rtol=2e-9,
                    atol=(2e-13, 2e-13, 2e-11, 2e-11), samples_per_period=64,
                    barycentric_tolerance=256*EPS, max_crossings=100000):
    initial = np.asarray(initial, float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("Expected finite four-component initial state")
    if mass <= 0 or frequency <= 0 or end_time <= start_time or max_steps_per_period < 1:
        raise ValueError("Invalid Radau controls")
    expected = (end_time-start_time)*frequency*samples_per_period
    count = round(expected)
    if count < 1 or abs(expected-count) > 1e-6:
        raise ValueError("Expected an integer number of output sampling steps")
    grid = np.linspace(start_time, end_time, count+1)
    states, owners = np.empty((count+1, 4)), np.empty(count+1, np.int32)
    omega, qm = 2*np.pi*frequency, charge/mass
    triangle = int(raw.interpolator._finder(initial[0], initial[1]))
    if triangle < 0:
        raise InterpolationDomainError("Restart outside the FEM domain")
    triangle, _ = choose_forward_triangle(raw, triangle, initial, start_time, qm, omega, phase, barycentric_tolerance)
    states[0], owners[0] = initial, triangle
    def create(clock, state, cell):
        return Radau(_rhs(raw, cell, qm, omega, phase), clock, state, end_time,
                     jac=affine_state_jacobian(raw, cell, qm, omega, phase),
                     max_step=1/(frequency*max_steps_per_period), rtol=rtol, atol=atol)
    solver = create(start_time, initial, triangle)
    cursor, nfev, steps, zero_time = 1, 0, 0, 0
    lower, output_lower = math.inf, float(np.min(_barycentric(raw, triangle, initial[:2])))
    crossings = []
    started = time.perf_counter()
    while solver.status == "running":
        previous = solver.t
        solver.step()
        if solver.status == "failed":
            raise GeometryAuditError("Radau step failed")
        dense = solver.dense_output()
        bary = radau_barycentric(dense, raw.origin[triangle], raw.gradient[triangle])
        steps += 1
        if steps == 1:
            for u in (.173, .617, 1.):
                ref = _barycentric(raw, triangle, dense(previous+u*(solver.t-previous))[:2])
                if np.max(abs(bernstein_value(bary, u)-ref)) > barycentric_tolerance:
                    raise GeometryAuditError("Radau dense conversion disagrees with actual dense output")
        event = first_exit(bary, barycentric_tolerance)
        fraction = 1. if event is None else event[0]
        retained, _ = split_bernstein(bary, fraction)
        lower = min(lower, membership_bound(retained, barycentric_tolerance))
        stop = solver.t if event is None else previous+(solver.t-previous)*fraction
        end_cursor = int(np.searchsorted(grid, stop, side="right"))
        if end_cursor > cursor:
            samples = dense(grid[cursor:end_cursor]).T
            weights = (samples[:, :2]-raw.origin[triangle])@raw.gradient[triangle].T
            weights[:, 0] += 1
            output_lower = min(output_lower, float(np.min(weights)))
            if np.min(weights) < -barycentric_tolerance:
                raise GeometryAuditError("Radau output outside the logged element")
            states[cursor:end_cursor], owners[cursor:end_cursor] = samples, triangle
            cursor = end_cursor
        if event is not None and stop < end_time:
            state = dense(stop)
            next_cell, _ = choose_forward_triangle(raw, triangle, state, stop, qm, omega, phase, barycentric_tolerance)
            if next_cell == triangle:
                raise GeometryAuditError("Radau exit did not change forward ownership")
            crossings.append({"time_s": float(stop), "from_triangle": triangle, "to_triangle": next_cell,
                              "state": state.tolist(), "endpoint_sign_test_would_miss": event[2]})
            zero_time = zero_time+1 if previous == stop else 0
            if len(crossings) > max_crossings or zero_time > 8:
                raise GeometryAuditError("Radau boundary chattering")
            nfev += solver.nfev
            triangle = next_cell
            solver = create(stop, state, triangle)
            if np.any(solver.y != state):
                raise GeometryAuditError("State changed on Radau restart")
    nfev += solver.nfev
    if cursor != len(grid):
        raise GeometryAuditError("Incomplete Radau output")
    diagnostics = {"method": "Radau", "checked_RK_steps": steps, "crossings": len(crossings),
                   "minimum_certified_barycentric_bound": lower, "minimum_output_barycentric": output_lower,
                   "barycentric_tolerance": barycentric_tolerance,
                   "endpoint_misses_detected": sum(c["endpoint_sign_test_would_miss"] for c in crossings),
                   "geometry_checker_shared_with_DOP853": True}
    return AuditedRFRun(RFRun(grid, states, "completed", nfev, time.perf_counter()-started, None), owners, crossings, diagnostics)


def scaled_state_distance(a, b, length_scale, velocity_scale):
    """Euclidean distance in (r/L,v/V); fixed nonzero physical scales."""
    if length_scale <= 0 or velocity_scale <= 0:
        raise ValueError("State scales must be positive")
    scale = np.array([length_scale, length_scale, velocity_scale, velocity_scale])
    return np.linalg.norm((np.asarray(a)-np.asarray(b))/scale, axis=-1)
