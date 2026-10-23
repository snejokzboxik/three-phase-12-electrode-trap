"""Raw P2 RF integration with explicitly localized triangle-edge crossings.

Within a triangle, extend that triangle's affine field only for trial RK
stages. Stop at the first edge event, keep position/velocity continuous, and
restart in the adjacent triangle. No field smoothing or impulse is introduced.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.integrate import solve_ivp

from .pilot_dynamics import RFRun
from .interpolation import InterpolationDomainError


def integrate_resolved(raw, initial, charge, mass, frequency, phase, start_time, end_time, *,
                        max_steps_per_period=50, rtol=2e-9,
                        atol=(2e-13, 2e-13, 2e-11, 2e-11), samples_per_period=64,
                        max_crossings=100000, progress=None):
    initial = np.asarray(initial, float)
    omega, qm = 2*math.pi*frequency, charge/mass
    count = round((end_time-start_time)*frequency*samples_per_period)
    grid = np.linspace(start_time, end_time, count+1)
    output = np.empty((count+1, 4))
    output[0] = initial
    cursor, total_nfev, crossing_count = 1, 0, 0
    t0, state = start_time, initial.copy()
    triangle = int(raw.interpolator._finder(state[0], state[1]))
    if triangle < 0:
        raise InterpolationDomainError("Invalid event-resolved initial position")
    started = time.perf_counter()
    last_progress = started
    while t0 < end_time:
        x0, y0 = raw.origin[triangle]
        g = raw.gradient[triangle]
        g1x, g1y, g2x, g2y = g[1, 0], g[1, 1], g[2, 0], g[2, 1]
        coefs_c, coefs_s = raw.coefficients[0][triangle], raw.coefficients[1][triangle]
        c0, c1, c2, c3, c4, c5 = (float(v) for v in coefs_c)
        s0, s1, s2, s3, s4, s5 = (float(v) for v in coefs_s)
        def rhs(t, y):
            dx, dy = y[0]-x0, y[1]-y0
            co, si = math.cos(omega*t+phase), math.sin(omega*t+phase)
            return np.array((y[2], y[3], qm*((c0+c1*dx+c2*dy)*co+(s0+s1*dx+s2*dy)*si),
                              qm*((c3+c4*dx+c5*dy)*co+(s3+s4*dx+s5*dy)*si)))
        def edge0(t, y):
            return 1-(g1x+g2x)*(y[0]-x0)-(g1y+g2y)*(y[1]-y0)
        def edge1(t, y):
            return g1x*(y[0]-x0)+g1y*(y[1]-y0)
        def edge2(t, y):
            return g2x*(y[0]-x0)+g2y*(y[1]-y0)
        events = (edge0, edge1, edge2)
        for event in events:
            event.direction, event.terminal = -1, True
        solution = solve_ivp(rhs, (t0, end_time), state, method="DOP853", dense_output=True,
                             rtol=rtol, atol=atol, max_step=1/(frequency*max_steps_per_period), events=events)
        total_nfev += solution.nfev
        if not solution.success:
            raise RuntimeError(f"Edge-resolved RF integration failed: {solution.message}")
        stop = float(solution.t[-1])
        end_cursor = int(np.searchsorted(grid, stop, side="right"))
        if end_cursor > cursor:
            output[cursor:end_cursor] = solution.sol(grid[cursor:end_cursor]).T
            cursor = end_cursor
        if solution.status == 0:
            if cursor < len(grid):
                output[cursor:] = solution.sol(grid[cursor:]).T
            return RFRun(grid, output, "completed", total_nfev, time.perf_counter()-started, None), crossing_count
        triggered = next(i for i, times in enumerate(solution.t_events) if len(times))
        next_triangle = int(raw.neighbors[triangle, triggered])
        if next_triangle < 0:
            raise InterpolationDomainError("Event-resolved trajectory reached the FEM boundary")
        state = solution.y[:, -1].copy()
        crossing_count += 1
        if crossing_count > max_crossings:
            raise RuntimeError("Too many triangle crossings; possible grazing/chattering in raw P2 field")
        # A vertex can touch more than two triangles. If the neighboring cell's
        # entry derivative points out through another face, the next event at
        # the same time traverses that face without advancing or altering state.
        if stop == t0 and crossing_count > 8 and next_triangle == triangle:
            raise RuntimeError("Repeated zero-time triangle event")
        triangle, t0 = next_triangle, stop
        now = time.perf_counter()
        if progress is not None and now-last_progress >= 30:
            progress({"time_s": t0, "RF_periods": t0*frequency, "crossings": crossing_count, "nfev": total_nfev})
            last_progress = now
    return RFRun(grid, output, "completed", total_nfev, time.perf_counter()-started, None), crossing_count
