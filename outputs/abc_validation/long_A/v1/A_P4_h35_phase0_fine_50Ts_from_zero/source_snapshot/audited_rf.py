"""Stepwise raw-P2 RF integration with whole-step geometric checks.

Each degree-seven DOP853 dense curve is transformed to barycentric Bernstein
polynomials. Their convex-hull bounds certify element membership up to an
explicit dimensionless tolerance. Interior extrema locate exits even when
both step endpoints are inside. Unresolved bounds/ownership fail explicitly.
The legacy event_resolved_rf.py remains the immutable comparison method.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.integrate import DOP853
from scipy.optimize import brentq

from .interpolation import InterpolationDomainError
from .pilot_dynamics import RFRun

EPS = np.finfo(float).eps
POWER_FROM_BERNSTEIN = np.array([
    [math.comb(7, i)*math.comb(7-i, k-i)*(-1.)**(k-i) if i <= k else 0.
     for i in range(8)] for k in range(8)])


class GeometryAuditError(RuntimeError):
    """The interpolant/element ownership could not be resolved safely."""


class BoundaryAmbiguity(GeometryAuditError):
    """Uniqueness of forward continuation has not been established."""


def bernstein_value(coefficients, x):
    b = np.array(coefficients, dtype=float, copy=True)
    for n in range(len(b)-1, 0, -1):
        b[:n] = (1-x)*b[:n]+x*b[1:n+1]
    return b[0]


def split_bernstein(coefficients, x=.5):
    b = np.array(coefficients, dtype=float, copy=True)
    left, right = [b[0].copy()], [b[-1].copy()]
    for n in range(len(b)-1, 0, -1):
        b = (1-x)*b[:n]+x*b[1:n+1]
        left.append(b[0].copy())
        right.append(b[-1].copy())
    return np.array(left), np.array(right[::-1])


def membership_bound(coefficients, tolerance, max_depth=40):
    """Conservative lower bound over the whole curve; never a sampling test.

    Roundoff in float64 coefficient construction is covered only by the stated
    tolerance. This certifies the numerical dense curve, not the exact ODE.
    """
    stack, lower = [(coefficients, 0)], math.inf
    while stack:
        b, depth = stack.pop()
        minimum = float(np.min(b))
        if minimum >= -tolerance:
            lower = min(lower, minimum)
            continue
        if np.min(b[0]) < -tolerance or np.min(b[-1]) < -tolerance:
            raise GeometryAuditError("Accepted dense curve leaves its assigned triangle")
        if depth == max_depth:
            raise GeometryAuditError("Bernstein membership bound unresolved; reduce the RK step")
        left, right = split_bernstein(b)
        stack.extend(((right, depth+1), (left, depth+1)))
    return lower


def first_exit(coefficients, tolerance):
    """Return (fraction, face, endpoint_sign_test_would_miss) or None.

    Locate minima via derivative roots, then certify the retained prefix by
    independent Bernstein subdivision. A missed/ill-conditioned root therefore
    causes an explicit audit failure instead of acceptance outside the element.
    """
    candidates = []
    for face in range(3):
        b = coefficients[:, face]
        if float(np.min(b)) >= -tolerance:
            continue
        power = POWER_FROM_BERNSTEIN@b
        derivative = np.polynomial.polynomial.polyder(power)
        roots = np.polynomial.polynomial.polyroots(derivative)
        critical = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < 1)
        knots = [0., *critical, 1.]
        values = [float(bernstein_value(b, x)) for x in knots]
        if values[0] < -tolerance:
            raise GeometryAuditError("Step starts outside its assigned triangle")
        for index in range(1, len(knots)):
            if values[index] < -tolerance:
                lo, hi = knots[index-1], knots[index]
                # A tiny negative entry residual may be within the geometric
                # tolerance. Ownership must be changed at that same state.
                root = lo if values[index-1] <= 0 else brentq(
                    lambda u: float(bernstein_value(b, u)), lo, hi,
                    xtol=1e-15, rtol=8*EPS)
                candidates.append((float(root), face, bool(values[-1] >= 0)))
                break
    if not candidates:
        membership_bound(coefficients, tolerance)
        return None
    result = min(candidates)
    prefix, _ = split_bernstein(coefficients, result[0])
    membership_bound(prefix, tolerance)
    return result


def dense_barycentric(dense, origin, gradients):
    """Translate SciPy's alternating Horner dense form without curve fitting.

    The dependency on F/y_old is checked structurally and against actual dense
    evaluation. A changed SciPy representation must not silently pass through.
    """
    if not hasattr(dense, "F") or np.shape(dense.F) != (7, 4):
        raise GeometryAuditError("Unsupported DOP853 dense-output representation")
    b = np.zeros((1, 2))
    for i, f in enumerate(reversed(dense.F)):
        b += f[:2]
        n = len(b)
        expanded = np.zeros((n+1, 2))
        if i % 2 == 0:
            expanded[1:] = b*(np.arange(1, n+1)/n)[:, None]
        else:
            expanded[:-1] = b*(1-np.arange(n)/n)[:, None]
        b = expanded
    b += dense.y_old[:2]-origin
    barycentric = b@gradients.T
    barycentric[:, 0] += 1
    return barycentric


def _barycentric(raw, triangle, position):
    value = raw.gradient[triangle]@(position-raw.origin[triangle])
    value[0] += 1
    return value


def _local(raw, triangle, position, angle):
    dx, dy = position-raw.origin[triangle]
    c, s = raw.coefficients[0][triangle], raw.coefficients[1][triangle]
    ec = np.array([c[0]+c[1]*dx+c[2]*dy, c[3]+c[4]*dx+c[5]*dy])
    es = np.array([s[0]+s[1]*dx+s[2]*dy, s[3]+s[4]*dx+s[5]*dy])
    jc, js = c[[1, 2, 4, 5]].reshape(2, 2), s[[1, 2, 4, 5]].reshape(2, 2)
    co, si = math.cos(angle), math.sin(angle)
    return co*ec+si*es, -si*ec+co*es, co*jc+si*js, -si*jc+co*js


def choose_forward_triangle(raw, hint, state, clock, qm, omega, phase, tolerance):
    """Search the entire incident star, then inspect forward Taylor signs.

    Position is never nudged and velocity never changed. Distinct viable
    accelerations at a boundary rest state are an ambiguity, not an arbitrary
    choice of the triangulation finder. Multiple surviving cells are accepted
    only when their complete local RF laws agree, or the charge is zero.
    """
    queue, seen, viable = [hint], set(), []
    while queue:
        triangle = int(queue.pop())
        if triangle in seen or triangle < 0:
            continue
        seen.add(triangle)
        bary = _barycentric(raw, triangle, state[:2])
        if np.min(bary) < -tolerance:
            continue
        boundary = np.flatnonzero(bary <= tolerance)
        for face in boundary:
            neighbor = int(raw.neighbors[triangle, face])
            if neighbor >= 0:
                queue.append(neighbor)
        e, de, jac, djac = _local(raw, triangle, state[:2], omega*clock+phase)
        ec, es, jc, js = _local(raw, triangle, state[:2], 0.)
        velocity = state[2:]
        acceleration = qm*e
        jerk = qm*(omega*de+jac@velocity)
        snap = qm*(-omega**2*e+2*omega*djac@velocity+jac@acceleration)
        derivatives = (velocity, acceleration, jerk, snap)
        field_scale = np.linalg.norm(ec)+np.linalg.norm(es)
        jacobian_scale = np.linalg.norm(jc)+np.linalg.norm(js)
        # Bound cancellation using the input terms, not the small residual
        # after e.g. partial_t E and J*v almost exactly cancel at grazing.
        scales = (np.linalg.norm(velocity), abs(qm)*field_scale,
                  abs(qm)*(omega*field_scale+jacobian_scale*np.linalg.norm(velocity)),
                  abs(qm)*(omega**2*field_scale+2*omega*jacobian_scale*np.linalg.norm(velocity)
                           +jacobian_scale*np.linalg.norm(acceleration)))
        admissible = True
        for face in boundary:
            gradient = raw.gradient[triangle, face]
            for derivative, scale in zip(derivatives, scales):
                sign = float(gradient@derivative)
                arithmetic = 128*EPS*np.linalg.norm(gradient)*scale
                if abs(sign) > arithmetic:
                    if sign < 0:
                        admissible = False
                    break
            if not admissible:
                break
        if admissible:
            viable.append((triangle, acceleration, jerk, snap))
    if not viable:
        if any(np.any(raw.neighbors[i] < 0) for i in seen):
            raise InterpolationDomainError("No forward triangle at the FEM domain boundary")
        raise BoundaryAmbiguity(f"No forward cell at t={clock:.17g}, state={state.tolist()}, cells={sorted(seen)}")
    for candidate in viable[1:]:
        for value, reference in zip(candidate[1:], viable[0][1:]):
            norm = max(float(np.linalg.norm(value)), float(np.linalg.norm(reference)), np.finfo(float).tiny)
            if np.linalg.norm(value-reference) > 512*EPS*norm:
                raise BoundaryAmbiguity(f"Different forward continuations at t={clock:.17g}, cells={[v[0] for v in viable]}")
    if len(viable) > 1 and qm != 0:
        reference_cell = viable[0][0]
        reference = _local(raw, reference_cell, state[:2], 0.)
        length = 1/np.max(np.linalg.norm(raw.gradient[reference_cell], axis=1))
        reference_law = np.concatenate((reference[0], reference[1], (length*reference[2]).ravel(), (length*reference[3]).ravel()))
        for candidate in viable[1:]:
            values = _local(raw, candidate[0], state[:2], 0.)
            law = np.concatenate((values[0], values[1], (length*values[2]).ravel(), (length*values[3]).ravel()))
            scale = max(np.linalg.norm(law), np.linalg.norm(reference_law), np.finfo(float).tiny)
            if np.linalg.norm(law-reference_law) > 1024*EPS*scale:
                raise BoundaryAmbiguity(f"Equal low-order jets but distinct RF laws at t={clock:.17g}, cells={[v[0] for v in viable]}")
    chosen = hint if any(v[0] == hint for v in viable) else min(v[0] for v in viable)
    return chosen, len(viable)


def _rhs(raw, triangle, qm, omega, phase):
    x0, y0 = raw.origin[triangle]
    c0, c1, c2, c3, c4, c5 = (float(v) for v in raw.coefficients[0][triangle])
    s0, s1, s2, s3, s4, s5 = (float(v) for v in raw.coefficients[1][triangle])
    def rhs(t, y):
        dx, dy = y[0]-x0, y[1]-y0
        co, si = math.cos(omega*t+phase), math.sin(omega*t+phase)
        return np.array((y[2], y[3], qm*((c0+c1*dx+c2*dy)*co+(s0+s1*dx+s2*dy)*si),
                         qm*((c3+c4*dx+c5*dy)*co+(s3+s4*dx+s5*dy)*si)))
    return rhs


@dataclass
class AuditedRFRun:
    run: RFRun
    sample_triangles: np.ndarray
    crossings: list[dict]
    diagnostics: dict


def integrate_audited(raw, initial, charge, mass, frequency, phase, start_time, end_time, *,
                      max_steps_per_period=100, rtol=2e-11,
                      atol=(2e-15, 2e-15, 2e-13, 2e-13), samples_per_period=64,
                      barycentric_tolerance=256*EPS, max_crossings=100000,
                      first_step=None, progress=None, on_crossing=None):
    initial = np.asarray(initial, float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("Expected a finite four-component initial state")
    if not np.all(np.isfinite([charge, mass, frequency, phase, start_time, end_time, rtol, barycentric_tolerance])):
        raise ValueError("Non-finite integration parameters")
    if mass <= 0 or frequency <= 0 or end_time <= start_time or max_steps_per_period < 1 or rtol <= 0 or barycentric_tolerance <= 0:
        raise ValueError("Invalid integration controls")
    expected = (end_time-start_time)*frequency*samples_per_period
    count = round(expected)
    if count < 1 or abs(expected-count) > 1e-6:
        raise ValueError("Output must span an integer number of sampling steps")
    grid = np.linspace(start_time, end_time, count+1)
    output, owners = np.empty((count+1, 4)), np.empty(count+1, dtype=np.int32)
    omega, qm = 2*math.pi*frequency, charge/mass
    triangle = int(raw.interpolator._finder(initial[0], initial[1]))
    if triangle < 0:
        raise InterpolationDomainError("Initial point outside FEM free space")
    triangle, equivalents = choose_forward_triangle(raw, triangle, initial, start_time, qm, omega, phase, barycentric_tolerance)
    output[0], owners[0] = initial, triangle
    cursor, nfev, checked, slow_checks, zero_time = 1, 0, 0, 0, 0
    certified_lower, sampled_lower = math.inf, float(np.min(_barycentric(raw, triangle, initial[:2])))
    crossings = []
    state, clock = initial.copy(), start_time
    started = last_progress = time.perf_counter()
    max_step = 1/(frequency*max_steps_per_period)
    solver = DOP853(_rhs(raw, triangle, qm, omega, phase), clock, state, end_time,
                    max_step=max_step, rtol=rtol, atol=atol, first_step=first_step)
    while solver.status == "running":
        previous = solver.t
        solver.step()
        if solver.status == "failed":
            raise GeometryAuditError("DOP853 step failed")
        dense = solver.dense_output()
        bary = dense_barycentric(dense, raw.origin[triangle], raw.gradient[triangle])
        checked += 1
        if checked == 1:
            for u in (.173, .617, 1.):
                reference = _barycentric(raw, triangle, dense(previous+u*(solver.t-previous))[:2])
                if np.max(abs(bernstein_value(bary, u)-reference)) > barycentric_tolerance:
                    raise GeometryAuditError("Dense-output coefficient conversion does not match SciPy")
        slow_checks += int(np.min(bary) < -barycentric_tolerance)
        event = first_exit(bary, barycentric_tolerance)
        fraction = 1. if event is None else event[0]
        retained, _ = split_bernstein(bary, fraction)
        certified_lower = min(certified_lower, membership_bound(retained, barycentric_tolerance))
        stop = solver.t if event is None else previous+(solver.t-previous)*fraction
        end_cursor = int(np.searchsorted(grid, stop, side="right"))
        if end_cursor > cursor:
            samples = dense(grid[cursor:end_cursor]).T
            weights = (samples[:, :2]-raw.origin[triangle])@raw.gradient[triangle].T
            weights[:, 0] += 1
            minimum = float(np.min(weights))
            sampled_lower = min(sampled_lower, minimum)
            if minimum < -barycentric_tolerance:
                raise GeometryAuditError("Output point outside its logged element")
            output[cursor:end_cursor], owners[cursor:end_cursor] = samples, triangle
            cursor = end_cursor
        if event is not None and stop < end_time:
            state = dense(stop)
            following, equivalents = choose_forward_triangle(raw, triangle, state, stop, qm, omega, phase, barycentric_tolerance)
            if following == triangle:
                raise GeometryAuditError("Exit found but forward ownership did not change")
            weights = _barycentric(raw, triangle, state[:2])
            new_weights = _barycentric(raw, following, state[:2])
            force_before = qm*_local(raw, triangle, state[:2], omega*stop+phase)[0]
            force_after = qm*_local(raw, following, state[:2], omega*stop+phase)[0]
            crossings.append({"time_s": float(stop), "from_triangle": triangle, "to_triangle": following,
                              "face": event[1], "endpoint_sign_test_would_miss": event[2],
                              "at_vertex": bool(np.sum(abs(weights) <= barycentric_tolerance) >= 2),
                              "equivalent_forward_cells": equivalents, "state": state.tolist(),
                              "from_barycentric": weights.tolist(), "to_barycentric": new_weights.tolist(),
                              "acceleration_jump_m_per_s2": float(np.linalg.norm(force_after-force_before)),
                              "position_jump_m": 0., "velocity_jump_m_per_s": 0.})
            zero_time = zero_time+1 if stop == previous else 0
            if len(crossings) > max_crossings or zero_time > 8:
                raise GeometryAuditError("Excessive crossings or zero-time boundary chattering")
            nfev += solver.nfev
            triangle, clock = following, stop
            solver = DOP853(_rhs(raw, triangle, qm, omega, phase), clock, state, end_time,
                            max_step=max_step, rtol=rtol, atol=atol)
            crossings[-1]["position_jump_m"] = float(np.linalg.norm(solver.y[:2]-state[:2]))
            crossings[-1]["velocity_jump_m_per_s"] = float(np.linalg.norm(solver.y[2:]-state[2:]))
            if crossings[-1]["position_jump_m"] or crossings[-1]["velocity_jump_m_per_s"]:
                raise GeometryAuditError("State changed during triangle restart")
            if on_crossing is not None:
                on_crossing(crossings[-1])
        now = time.perf_counter()
        if progress is not None and now-last_progress >= 30:
            progress({"time_s": float(stop), "checked_RK_steps": checked, "crossings": len(crossings),
                      "endpoint_misses_detected": sum(c["endpoint_sign_test_would_miss"] for c in crossings)})
            last_progress = now
    nfev += solver.nfev
    if cursor != len(grid):
        raise GeometryAuditError("Incomplete output grid")
    diagnostics = {"checked_RK_steps": checked, "steps_requiring_root_checks": slow_checks,
                   "crossings": len(crossings), "endpoint_misses_detected": sum(c["endpoint_sign_test_would_miss"] for c in crossings),
                   "vertex_crossings": sum(c["at_vertex"] for c in crossings),
                   "minimum_certified_barycentric_bound": certified_lower,
                   "minimum_output_barycentric": sampled_lower,
                   "barycentric_tolerance": barycentric_tolerance,
                   "no_position_or_velocity_nudging": True,
                   "scope": "Numerical dense-curve membership audit, not exact-ODE or FEM spatial convergence"}
    run = RFRun(grid, output, "completed", nfev, time.perf_counter()-started, None)
    return AuditedRFRun(run, owners, crossings, diagnostics)
