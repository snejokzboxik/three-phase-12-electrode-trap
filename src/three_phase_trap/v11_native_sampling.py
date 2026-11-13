"""Trajectory-sampling variant of the frozen native P4 escape audit."""
import numpy as np
from scipy.integrate import DOP853

from .audited_rf import EPS, GeometryAuditError, dense_barycentric, first_exit, split_bernstein
from .direct_fem_dynamics import clock_membership_tolerance, forward_cell, local_rhs
from .stability_rf_p4 import NativeEscape, first_radial_exit
from .stability_rf_p4_v2 import EscapeAuditError


class NativeTrajectory(NativeEscape):
    def segment(self, initial, qm, frequency, phase, first_rf, last_rf, *,
                steps, rtol, atol):
        """Sample the audited curve at whole periods on the absolute RF clock."""
        field = self.field
        start, end = first_rf / frequency, last_rf / frequency
        omega, base_tolerance = 2 * np.pi * frequency, 256 * EPS
        if end <= start or np.linalg.norm(initial[:2]) >= self.radius:
            raise GeometryAuditError("Invalid P4 escape segment")
        cell = int(field._finder(*initial[:2]))
        if cell < 0:
            raise GeometryAuditError("Native P4 segment starts outside vacuum")
        tolerance = clock_membership_tolerance(start, np.linalg.norm(initial[2:]), self.maximum_gradient, base_tolerance)
        cell = forward_cell(field, cell, initial, start, qm, omega, phase, tolerance)

        def create(clock, state, owner):
            return DOP853(local_rhs(field, owner, qm, omega, phase), clock, state, end,
                          max_step=1 / (frequency * steps), rtol=rtol, atol=atol)

        solver = create(start, initial, cell)
        crossings = nfev = stalled = max_stalled = checked_steps = 0
        maximum = float(np.linalg.norm(initial[:2]))
        sample_clock = [float(first_rf)]
        samples = [initial.copy()]
        tick = int(np.floor(first_rf)) + 1

        def sample_prefix(dense, stop):
            nonlocal tick
            # Sample only the portion of the step that was accepted by the
            # element and aperture audits.
            while tick / frequency <= stop:
                sample_clock.append(float(tick))
                samples.append(dense(tick / frequency))
                tick += 1

        def record(clock, state, escape=None):
            reached = float(last_rf) if clock == end and escape is None else clock * frequency
            if abs(reached - sample_clock[-1]) <= 8 * abs(float(np.spacing(reached))):
                sample_clock[-1], samples[-1] = reached, state
            else:
                sample_clock.append(reached)
                samples.append(state)
            return dict(reached_RF=reached, escape_RF=escape, final_state_SI=state.tolist(),
                        crossings=crossings, checked_steps=checked_steps, max_zero_time_run=max_stalled,
                        sampled_max_radius_m=maximum, nfev=nfev + solver.nfev,
                        time_RF=sample_clock, states=np.asarray(samples).tolist())

        while solver.status == "running":
            previous = solver.t
            solver.step()
            if solver.status == "failed":
                raise GeometryAuditError("Native P4 escape integration failed")
            checked_steps += 1
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
                    maximum = self.radius
                    sample_prefix(dense, clock)
                    return record(clock, state, clock * frequency)
            stop = solver.t if boundary is None else previous + (solver.t - previous) * fraction
            state = solver.y if boundary is None else dense(stop)
            maximum = max(maximum, float(np.linalg.norm(state[:2])))
            sample_prefix(dense, stop)
            if boundary is None or stop >= end:
                continue
            other = forward_cell(field, cell, state, stop, qm, omega, phase, tolerance)
            if other == cell:
                raise GeometryAuditError("Native P4 crossing did not change owner")
            crossings += 1
            stalled = stalled + 1 if stop == previous else 0
            max_stalled = max(max_stalled, stalled)
            if stalled > 8:
                diagnostic = record(stop, state)
                raise EscapeAuditError("zero_time_crossings",
                                       {**diagnostic, "from_cell": cell, "to_cell": other})
            nfev += solver.nfev
            cell, solver = other, create(stop, state, other)
        return record(end, solver.y)
