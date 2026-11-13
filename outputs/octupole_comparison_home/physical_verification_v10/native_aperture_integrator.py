"""Native P4 FEM RF integrator with an audited outward-aperture event.

The force is evaluated from the saved P4 finite-element polynomial gradients.
Triangle crossings use the existing whole-step Bernstein geometry audit.
The aperture is a separate physical first-passage event at r=8 mm; it is not
the FEM-domain boundary. No FEM solve or harmonic reconstruction is used.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import DOP853
from scipy.optimize import brentq

from three_phase_trap.audited_rf import (
    EPS, GeometryAuditError, dense_barycentric, first_exit,
    membership_bound, split_bernstein,
)
from three_phase_trap.direct_fem_dynamics import (
    clock_membership_tolerance, forward_cell, local_rhs,
)


@dataclass
class ApertureResult:
    times: np.ndarray
    states: np.ndarray
    status: str
    escape_time_s: float | None
    diagnostics: dict


def _aperture_root(dense, left: float, right: float, radius: float):
    """Find the earliest outward aperture crossing on a short dense step.

    Steps are capped at T/200. A 32-segment bracket scan bounds all plausible
    crossings for this RF-resolved step; brackets are then refined with Brent.
    The 2x step refinement audit is performed by the caller via a second run.
    """
    if right <= left:
        return None
    knots = np.linspace(left, right, 33)
    values = np.array([np.dot(dense(float(t))[:2], dense(float(t))[:2])-radius*radius
                       for t in knots])
    for j in range(len(knots)-1):
        if values[j] <= 0.0 and values[j+1] >= 0.0:
            root = brentq(lambda t: float(np.dot(dense(t)[:2], dense(t)[:2])-radius*radius),
                          float(knots[j]), float(knots[j+1]), xtol=5e-324,
                          rtol=8*EPS)
            state = dense(root)
            radial_speed = float(np.dot(state[:2], state[2:]) / radius)
            if radial_speed <= 0:
                continue
            return float(root), state
        if values[j] > 0.0:
            raise GeometryAuditError("Dense trajectory started an accepted step outside the aperture")
    return None


def integrate_native_aperture(field, initial, charge, mass, frequency, phase,
                              end_time, *, start_time=0.0,
                              steps_per_period=200, rtol=2e-9,
                              atol=(2e-13, 2e-13, 2e-11, 2e-11),
                              samples_per_period=8, aperture=8e-3,
                              tolerance=256*EPS, progress=None):
    initial = np.asarray(initial, float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("Expected a finite (x,y,vx,vy) initial state")
    if mass <= 0 or frequency <= 0 or end_time <= start_time or steps_per_period < 1:
        raise ValueError("Invalid native RF problem")
    if np.linalg.norm(initial[:2]) >= aperture:
        raise ValueError("Initial position is outside the open aperture")

    duration_periods = (end_time-start_time)*frequency
    sample_count = int(round(duration_periods*samples_per_period))
    if sample_count < 1 or abs(sample_count-duration_periods*samples_per_period) > 1e-7:
        raise ValueError("Requested output must span an integer number of sample intervals")
    sample_grid = np.linspace(start_time, end_time, sample_count+1)
    out_t, out_y = [float(start_time)], [initial.copy()]
    qm, omega = charge/mass, 2*math.pi*frequency
    maximum_gradient = float(np.max(np.linalg.norm(field.gradient, axis=-1)))
    cell = int(field._finder(initial[0], initial[1]))
    if cell < 0:
        raise GeometryAuditError("Initial point outside FEM vacuum")
    cell = forward_cell(field, cell, initial, start_time, qm, omega, phase,
                        clock_membership_tolerance(start_time, np.linalg.norm(initial[2:]),
                                                   maximum_gradient, tolerance))

    def create(t, state, owner):
        return DOP853(local_rhs(field, owner, qm, omega, phase), t, state, end_time,
                      max_step=1/(frequency*steps_per_period), rtol=rtol, atol=atol)

    solver = create(start_time, initial, cell)
    steps = crossings = nfev = 0
    min_bary_bound = math.inf
    max_tolerance = tolerance
    max_dense_radius = float(np.linalg.norm(initial[:2]))
    started = time.perf_counter()
    next_sample = 1
    escape_time = None
    while solver.status == "running":
        prev = float(solver.t)
        solver.step()
        if solver.status == "failed":
            raise GeometryAuditError("Native DOP853 integration failed")
        dense = solver.dense_output()
        bary = dense_barycentric(dense, field.origin[cell], field.gradient[cell])
        speed = max(float(np.linalg.norm(solver.y[2:])), float(np.linalg.norm(dense(prev)[2:])))
        step_tol = clock_membership_tolerance(solver.t, speed, maximum_gradient, tolerance)
        max_tolerance = max(max_tolerance, step_tol)
        face_event = first_exit(bary, step_tol)
        face_fraction = 1.0 if face_event is None else face_event[0]
        face_time = prev+(solver.t-prev)*face_fraction
        face_prefix, _ = split_bernstein(bary, face_fraction)
        min_bary_bound = min(min_bary_bound, membership_bound(face_prefix, step_tol))

        # Bernstein position controls enclose the entire dense curve segment.
        # Most deep-trap steps are certified well inside the aperture and skip
        # root scanning altogether; only possible aperture approaches are split.
        position_controls = face_prefix @ field.points[field.triangles[cell]]
        radial_possible = float(np.max(np.linalg.norm(position_controls, axis=1))) >= aperture
        radial_event = (_aperture_root(dense, prev, face_time, aperture)
                        if radial_possible else None)
        stop = float(solver.t if radial_event is None else radial_event[0])
        stop_state = None if radial_event is None else radial_event[1]
        physical_step_stop = stop if radial_event is not None else face_time
        for t_probe in (prev, 0.5*(prev+physical_step_stop), physical_step_stop):
            max_dense_radius = max(max_dense_radius,
                                   float(np.linalg.norm(dense(t_probe)[:2])))
        next_time = sample_grid[next_sample] if next_sample < len(sample_grid) else math.inf
        while next_time <= stop+4*abs(float(np.spacing(stop))):
            state = dense(float(next_time))
            out_t.append(float(next_time)); out_y.append(state.copy())
            next_sample += 1
            next_time = sample_grid[next_sample] if next_sample < len(sample_grid) else math.inf
        if radial_event is not None:
            out_t.append(stop); out_y.append(stop_state.copy())
            escape_time = stop
            nfev += solver.nfev
            break

        if face_event is not None:
            state = dense(face_time)
            other = forward_cell(field, cell, state, face_time, qm, omega, phase, step_tol)
            if other == cell:
                raise GeometryAuditError("Native FE element exit did not change ownership")
            crossings += 1
            if crossings > 100000:
                raise GeometryAuditError("Excessive element crossings")
            nfev += solver.nfev
            cell = other
            solver = create(face_time, state, cell)
            if not np.array_equal(solver.y, state):
                raise GeometryAuditError("Element crossing changed the physical state")
        steps += 1
        if progress is not None and steps % 5000 == 0:
            progress({"time_s": float(solver.t), "steps": steps, "crossings": crossings,
                      "elapsed_s": time.perf_counter()-started})
    if escape_time is None:
        nfev += solver.nfev
        if next_sample != len(sample_grid):
            raise GeometryAuditError("Stroboscopic/sample output incomplete")
        status = "retained_to_horizon"
    else:
        status = "escaped"
    states = np.asarray(out_y)
    radii = np.linalg.norm(states[:, :2], axis=1)
    sampled_duration = max(float(out_t[-1]-out_t[0]), np.finfo(float).tiny)
    rms_radius = float(np.sqrt(np.trapezoid(radii*radii, x=np.asarray(out_t))/sampled_duration))
    diagnostics = {
        "method": "DOP853 native P4 element gradients; dense aperture first-passage",
        "status": status,
        "checked_steps": steps,
        "element_crossings": crossings,
        "nfev": nfev,
        "max_step_s": 1/(frequency*steps_per_period),
        "steps_per_rf_period": steps_per_period,
        "rtol": rtol,
        "atol": list(atol),
        "aperture_radius_m": aperture,
        "aperture_event_method": "dense DOP853 radius bracketing, 32 subintervals per accepted step, Brent root",
        "escape_radial_velocity_m_s": None if escape_time is None else float(np.dot(states[-1,:2],states[-1,2:])/aperture),
        "maximum_sampled_radius_m": float(radii.max()),
        "maximum_three_point_dense_radius_m": max_dense_radius,
        "rms_radius_m": rms_radius,
        "minimum_sampled_radius_m": float(radii.min()),
        "minimum_bernstein_membership_bound": float(min_bary_bound),
        "maximum_clock_aware_barycentric_tolerance": max_tolerance,
        "wall_seconds": time.perf_counter()-started,
        "position_nudges": 0,
        "velocity_nudges": 0,
    }
    return ApertureResult(np.asarray(out_t), states, status, escape_time, diagnostics)
