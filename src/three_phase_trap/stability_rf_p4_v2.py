"""Long native P4 escape checks, with separate crossing and stall diagnostics."""
import numpy as np
from scipy.integrate import DOP853

from .audited_rf import EPS, GeometryAuditError, dense_barycentric, first_exit, split_bernstein
from .direct_fem_dynamics import clock_membership_tolerance, forward_cell, local_rhs
from .stability_rf_p4 import NativeEscape, first_radial_exit


class EscapeAuditError(GeometryAuditError):
    """The audit could not advance the physical clock."""

    def __init__(self, reason, diagnostic):
        super().__init__(reason)
        self.diagnostic = diagnostic


class NativeEscapeV2(NativeEscape):
    def segment(self, initial, qm, frequency, phase, first_rf, last_rf, *,
                steps, rtol, atol, legacy_diagnostic=False):
        """Advance with absolute RF time. The legacy limit is opt-in for diagnosis."""
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

        def record(clock, state, escape=None):
            return dict(reached_RF=clock * frequency, escape_RF=escape, final_state_SI=state.tolist(),
                        crossings=crossings, checked_steps=checked_steps, max_zero_time_run=max_stalled,
                        sampled_max_radius_m=maximum, nfev=nfev + solver.nfev)

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
                    return record(clock, state, clock * frequency)
            stop = solver.t if boundary is None else previous + (solver.t - previous) * fraction
            state = solver.y if boundary is None else dense(stop)
            maximum = max(maximum, float(np.linalg.norm(state[:2])))
            if boundary is None or stop >= end:
                continue
            other = forward_cell(field, cell, state, stop, qm, omega, phase, tolerance)
            if other == cell:
                raise GeometryAuditError("Native P4 crossing did not change owner")
            crossings += 1
            stalled = stalled + 1 if stop == previous else 0
            max_stalled = max(max_stalled, stalled)
            if stalled > 8 or (legacy_diagnostic and crossings > 100000):
                reason = "zero_time_crossings" if stalled > 8 else "legacy_total_crossing_cap"
                raise EscapeAuditError(reason, {**record(stop, state), "from_cell": cell, "to_cell": other})
            nfev += solver.nfev
            cell, solver = other, create(stop, state, other)
        return record(end, solver.y)
