"""Whole-step checked native higher-order FEM dynamics.

Reuses only dense-curve geometry algebra from audited_rf; force and Jacobian
are the native higher-order element polynomials, without an affine surrogate.
Unresolved grazing/ownership is reported rather than regularized by nudges.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.integrate import DOP853, Radau

from .audited_rf import (EPS, GeometryAuditError, BoundaryAmbiguity, AuditedRFRun,
                         dense_barycentric, bernstein_value, first_exit,
                         membership_bound, split_bernstein)
from .restart_dynamics import radau_barycentric
from .pilot_dynamics import RFRun


def local_field(field, cell, state, angle):
    c0, c1, s0, s1 = field.local_quadratures(*field.local_coordinates(cell, state[0], state[1]))
    co, si = math.cos(angle), math.sin(angle)
    ec, es = np.array([c0, c1]), np.array([s0, s1])
    return ec*co+es*si, -ec*si+es*co, ec, es


def forward_cell(field, hint, state, clock, qm, omega, phase, tolerance):
    queue, seen, viable, unresolved = [hint], set(), [], []
    angle = omega*clock+phase
    while queue:
        cell = int(queue.pop())
        if cell < 0 or cell in seen:
            continue
        seen.add(cell)
        bary = field.gradient[cell]@(state[:2]-field.origin[cell])
        bary[0] += 1
        if bary.min() < -tolerance:
            continue
        faces = np.flatnonzero(bary <= tolerance)
        for face in faces:
            queue.append(int(field.neighbors[cell, face]))
        if not len(faces):
            return cell
        e, de, ec, es = local_field(field, cell, state, angle)
        jac = field.jacobian(state[0], state[1], angle, cell=cell)
        acceleration = qm*e
        jerk = qm*(omega*de+jac@state[2:])
        field_scale = np.linalg.norm(ec)+np.linalg.norm(es)
        scales = (np.linalg.norm(state[2:]), abs(qm)*field_scale,
                  abs(qm)*(omega*field_scale+np.linalg.norm(jac)*np.linalg.norm(state[2:])))
        signs, undecided = [], False
        for face in faces:
            g = field.gradient[cell, face]
            found = False
            for derivative, scale in zip((state[2:], acceleration, jerk), scales):
                value = float(g@derivative)
                if abs(value) > 128*EPS*np.linalg.norm(g)*scale:
                    signs.append(value)
                    found = True
                    break
            if not found:
                undecided = True
        if signs and min(signs) < 0:
            continue
        if undecided:
            unresolved.append(cell)
        else:
            viable.append(cell)
    if len(viable) == 1 and not unresolved:
        return viable[0]
    candidates = list(dict.fromkeys(viable+unresolved))
    if not candidates:
        raise BoundaryAmbiguity(f"No native forward cell t={clock}, state={state.tolist()}")
    if qm != 0:
        # Equality of cubic polynomials is tested on an unisolvent 10-point
        # lattice, in one common physical neighborhood. This is not an ODE
        # query outside a cell and never changes the physical state.
        length = min(1/np.max(np.linalg.norm(field.gradient[c], axis=1)) for c in candidates)
        uv = np.array([(i/3, j/3) for total in range(4) for i in range(total+1) for j in [total-i]])
        xy = state[:2]+length*(uv-.3)
        laws = []
        for cell in candidates:
            laws.append(np.array([field.local_quadratures(*field.local_coordinates(cell, *p)) for p in xy]))
        for law in laws[1:]:
            scale = max(np.linalg.norm(law), np.linalg.norm(laws[0]), np.finfo(float).tiny)
            if np.linalg.norm(law-laws[0]) > 4096*EPS*scale:
                raise BoundaryAmbiguity(f"Native higher-order grazing unresolved t={clock}, cells={candidates}")
    return hint if hint in candidates else candidates[0]


def local_rhs(field, cell, qm, omega, phase):
    origin = field.origin[cell]
    a = field.inverse[cell]
    cx, cy = field.gx[cell], field.gy[cell]
    def rhs(t, state):
        dx, dy = state[0]-origin[0], state[1]-origin[1]
        u, v = a[0, 0]*dx+a[0, 1]*dy, a[1, 0]*dx+a[1, 1]*dy
        gx, gy = field.cubic(cx, u, v), field.cubic(cy, u, v)
        co, si = math.cos(omega*t+phase), math.sin(omega*t+phase)
        return np.array([state[2], state[3], -qm*(gx.imag*co+gx.real*si), -qm*(gy.imag*co+gy.real*si)])
    return rhs


def clock_membership_tolerance(clock, speed, maximum_barycentric_gradient, base):
    """Bound displacement from representing a crossing in global float64 time.

    Dense roots use a local step fraction, while restarts use global time.
    Eight global-time ULPs conservatively cover forming/subtracting that clock
    and evaluating its position. This is a geometry envelope, never a nudge.
    """
    return base+8*abs(float(np.spacing(clock)))*speed*maximum_barycentric_gradient


def integrate_native(field, initial, charge, mass, frequency, phase, start_time, end_time, *,
                     method="DOP853", steps_per_period=100, rtol=2e-11,
                     atol=(2e-15, 2e-15, 2e-13, 2e-13), samples_per_period=64,
                     tolerance=256*EPS, radius=.008, progress=None):
    initial = np.asarray(initial, float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)) or mass <= 0 or frequency <= 0 or end_time <= start_time:
        raise ValueError("Invalid native RF problem")
    count = round((end_time-start_time)*frequency*samples_per_period)
    if count < 1 or abs(count-(end_time-start_time)*frequency*samples_per_period) > 1e-6:
        raise ValueError("Output must span integer RF sampling intervals")
    grid = np.linspace(start_time, end_time, count+1)
    states, owners = np.empty((count+1, 4)), np.empty(count+1, np.int32)
    qm, omega = charge/mass, 2*np.pi*frequency
    maximum_gradient = float(np.max(np.linalg.norm(field.gradient, axis=-1)))
    cell = int(field._finder(initial[0], initial[1]))
    if cell < 0:
        raise GeometryAuditError("Native initial point outside vacuum")
    initial_tolerance = clock_membership_tolerance(start_time, np.linalg.norm(initial[2:]), maximum_gradient, tolerance)
    cell = forward_cell(field, cell, initial, start_time, qm, omega, phase, initial_tolerance)
    states[0], owners[0] = initial, cell
    def create(t, state, i):
        kwargs = dict(max_step=1/(frequency*steps_per_period), rtol=rtol, atol=atol)
        if method == "DOP853":
            return DOP853(local_rhs(field, i, qm, omega, phase), t, state, end_time, **kwargs)
        if method != "Radau":
            raise ValueError("Expected DOP853 or Radau")
        def jac(clock, y):
            out = np.zeros((4, 4))
            out[0, 2] = out[1, 3] = 1.
            out[2:, :2] = qm*field.jacobian(y[0], y[1], omega*clock+phase, cell=i)
            return out
        return Radau(local_rhs(field, i, qm, omega, phase), t, state, end_time, jac=jac, **kwargs)
    solver = create(start_time, initial, cell)
    cursor, steps, nfev = 1, 0, 0
    lower, sample_lower = math.inf, math.inf
    crossings = []
    started = last_progress = time.perf_counter()
    zero_time = 0
    maximum_tolerance = initial_tolerance
    while solver.status == "running":
        previous = solver.t
        solver.step()
        if solver.status == "failed":
            raise GeometryAuditError("Native RF integrator failed")
        dense = solver.dense_output()
        convert = dense_barycentric if method == "DOP853" else radau_barycentric
        bary = convert(dense, field.origin[cell], field.gradient[cell])
        speed = max(float(np.linalg.norm(solver.y[2:])), float(np.linalg.norm(dense(previous)[2:])))
        step_tolerance = clock_membership_tolerance(solver.t, speed, maximum_gradient, tolerance)
        maximum_tolerance = max(maximum_tolerance, step_tolerance)
        steps += 1
        if steps == 1:
            for u in (.173, .617, 1.):
                p = dense(previous+u*(solver.t-previous))[:2]
                actual = field.gradient[cell]@(p-field.origin[cell])
                actual[0] += 1
                if np.max(abs(bernstein_value(bary, u)-actual)) > tolerance:
                    raise GeometryAuditError("Native dense polynomial conversion mismatch")
        event = first_exit(bary, step_tolerance)
        fraction = 1. if event is None else event[0]
        retained, _ = split_bernstein(bary, fraction)
        lower = min(lower, membership_bound(retained, step_tolerance))
        stop = solver.t if event is None else previous+(solver.t-previous)*fraction
        end = int(np.searchsorted(grid, stop, side="right"))
        if end > cursor:
            output = dense(grid[cursor:end]).T
            weights = (output[:, :2]-field.origin[cell])@field.gradient[cell].T
            weights[:, 0] += 1
            sample_lower = min(sample_lower, float(weights.min()))
            if weights.min() < -step_tolerance:
                raise GeometryAuditError("Native output outside its element")
            if np.any(np.linalg.norm(output[:, :2], axis=-1) > radius):
                raise GeometryAuditError("Native RF left the declared validation disk")
            states[cursor:end], owners[cursor:end] = output, cell
            cursor = end
        if event is not None and stop < end_time:
            state = dense(stop)
            other = forward_cell(field, cell, state, stop, qm, omega, phase, step_tolerance)
            if other == cell:
                raise GeometryAuditError("Native exit did not change cell ownership")
            crossings.append({"time_s": float(stop), "from": cell, "to": other,
                              "state": state.tolist(), "hidden_from_endpoint_test": event[2],
                              "barycentric_tolerance": step_tolerance})
            zero_time = zero_time+1 if stop == previous else 0
            if zero_time > 8 or len(crossings) > 100000:
                raise GeometryAuditError("Native RF boundary chatter")
            nfev += solver.nfev
            cell = other
            solver = create(stop, state, cell)
            if not np.array_equal(solver.y, state):
                raise GeometryAuditError("Native crossing changed state")
        now = time.perf_counter()
        if progress is not None and now-last_progress > 30:
            progress({"time_s": float(stop), "end_s": end_time, "steps": steps, "crossings": len(crossings)})
            last_progress = now
    nfev += solver.nfev
    if cursor != len(grid):
        raise GeometryAuditError("Native RF output incomplete")
    diagnostics = {"method": method, "checked_steps": steps, "crossings": len(crossings),
                   "minimum_barycentric_bound": lower, "minimum_sample_barycentric": sample_lower,
                   "barycentric_tolerance": tolerance, "position_nudges": 0, "velocity_nudges": 0,
                   "maximum_clock_aware_barycentric_tolerance": maximum_tolerance,
                   "clock_envelope_physical_distance_m": (maximum_tolerance-tolerance)/maximum_gradient,
                   "hidden_exits_detected": sum(c["hidden_from_endpoint_test"] for c in crossings),
                   "force": "native higher-order FEM polynomial gradient; no harmonic reconstruction"}
    return AuditedRFRun(RFRun(grid, states, "completed", nfev, time.perf_counter()-started, None), owners, crossings, diagnostics)
