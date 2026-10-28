"""Exact scalar RF evaluators and restartable full-RF pilot integration.

Raw P2 gradients remain discontinuous; no projection or smoothing is applied.
The affine evaluator is an algebraic optimization of the existing interpolator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

from .interpolation import InterpolationDomainError
from .validation_field_audit import element_jacobians


class AffineP2RF:
    def __init__(self, interpolator, amplitude):
        self.interpolator = interpolator
        triangles = interpolator.solutions.triangles
        vertices = interpolator.solutions.points[triangles]
        self.origin = vertices[:, 0].copy()
        self.gradient = interpolator._gradients_lambda
        self.neighbors = interpolator._neighbors
        ids = np.arange(len(triangles))
        ja, jb, jc = (element_jacobians(interpolator, ids, p) for p in ("A", "B", "C"))
        jcos = math.sqrt(3)*amplitude*(jc-jb)/2
        jsin = amplitude*(ja-jb/2-jc/2)
        a, b, c = (interpolator.solutions.fields[p] for p in ("A", "B", "C"))
        ecos, esin = math.sqrt(3)*amplitude*(c-b)/2, amplitude*(a-b/2-c/2)
        centers = vertices.mean(axis=1)
        # [Ex offset, Ex,x, Ex,y, Ey offset, Ey,x, Ey,y], at triangle vertex 0.
        self.coefficients = []
        for E, J in ((ecos, jcos), (esin, jsin)):
            base = E-np.einsum("nij,nj->ni", J, centers-self.origin)
            self.coefficients.append(np.column_stack((base[:, 0], J[:, 0, :], base[:, 1], J[:, 1, :])))
        self.last_id = None
        self.calls = self.fallbacks = 0

    def quadratures(self, x, y):
        self.calls += 1
        i = self.last_id
        if i is not None:
            for _ in range(12):
                dx, dy = x-self.origin[i, 0], y-self.origin[i, 1]
                g = self.gradient[i]
                l1, l2 = g[1, 0]*dx+g[1, 1]*dy, g[2, 0]*dx+g[2, 1]*dy
                l0 = 1-l1-l2
                if min(l0, l1, l2) >= -1e-10:
                    break
                opposite = 0 if l0 <= l1 and l0 <= l2 else (1 if l1 <= l2 else 2)
                i = int(self.neighbors[i, opposite])
                if i < 0:
                    i = None
                    break
            else:
                i = None
        if i is None:
            self.fallbacks += 1
            i = int(self.interpolator._finder(x, y))
            if i < 0:
                raise InterpolationDomainError("Pilot point outside raw FEM free space")
            dx, dy = x-self.origin[i, 0], y-self.origin[i, 1]
        self.last_id = i
        c, s = self.coefficients[0][i], self.coefficients[1][i]
        return (c[0]+c[1]*dx+c[2]*dy, c[3]+c[4]*dx+c[5]*dy,
                s[0]+s[1]*dx+s[2]*dy, s[3]+s[4]*dx+s[5]*dy)


class PolynomialRF:
    def __init__(self, quadratures):
        self.reference_radius = quadratures.reference_radius
        self.coefficients = tuple(tuple(complex(v) for v in np.polynomial.polynomial.polyder(c)/quadratures.reference_radius)
                                  for c in (quadratures.cosine, quadratures.sine))

    def quadratures(self, x, y):
        z = complex(x, y)/self.reference_radius
        out = []
        for coefficients in self.coefficients:
            value = 0j
            for coefficient in reversed(coefficients):
                value = value*z+coefficient
            out.extend((-value.real, value.imag))
        return tuple(out)


@dataclass
class RFRun:
    time: np.ndarray
    states: np.ndarray
    status: str
    nfev: int
    wall_seconds: float
    exit_time: float | None


def integrate_rf(field, initial, charge, mass, frequency, phase, start_time, end_time, *,
                 output_samples_per_period=64, max_steps_per_period=50,
                 rtol=2e-9, atol=(2e-13, 2e-13, 2e-11, 2e-11), radius=.008):
    """Full SI RF integration, DOP853. Restart uses global time in the RF phase.

    A radial boundary continuation is used only for trial stages needed to
    locate an exit event; all accepted output stops at that event.
    """
    initial = np.asarray(initial, float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("Expected finite full initial state")
    if mass <= 0 or frequency <= 0 or end_time <= start_time or radius <= 0 or np.linalg.norm(initial[:2]) >= radius:
        raise ValueError("Invalid integration interval, parameters or initial domain")
    omega, qm = 2*math.pi*frequency, charge/mass
    count = round((end_time-start_time)*frequency*output_samples_per_period)
    if abs(count-(end_time-start_time)*frequency*output_samples_per_period) > 1e-6:
        raise ValueError("Output interval must span an integer number of RF sampling steps")
    times = np.linspace(start_time, end_time, count+1)
    def rhs(t, state):
        x, y, vx, vy = state
        r = math.hypot(x, y)
        if r > radius:
            x, y = x*radius/r, y*radius/r
        cx, cy, sx, sy = field.quadratures(x, y)
        co, si = math.cos(omega*t+phase), math.sin(omega*t+phase)
        return np.array((vx, vy, qm*(cx*co+sx*si), qm*(cy*co+sy*si)))
    def exit_disk(t, state):
        return radius-math.hypot(state[0], state[1])
    exit_disk.direction, exit_disk.terminal = -1, True
    started = time.perf_counter()
    result = solve_ivp(rhs, (start_time, end_time), initial, method="DOP853", t_eval=times,
                       rtol=rtol, atol=atol, max_step=1/(frequency*max_steps_per_period), events=exit_disk)
    if not result.success:
        raise RuntimeError(f"RF integration failed: {result.message}")
    t, states = result.t, result.y.T
    exit_time = None
    if len(result.t_events[0]):
        exit_time = float(result.t_events[0][0])
        if abs(t[-1]-exit_time) < 1e-14:
            states[-1] = result.y_events[0][0]
        else:
            t = np.r_[t, exit_time]
            states = np.vstack((states, result.y_events[0][0]))
    return RFRun(t, states, "completed" if exit_time is None else "domain_exit", result.nfev,
                 time.perf_counter()-started, exit_time)


def coordinate_metrics(actual, reference, scale):
    difference = np.linalg.norm(np.asarray(actual)-np.asarray(reference), axis=-1)/scale
    return {"relative_rms": float(np.sqrt(np.mean(difference**2))), "relative_max": float(difference.max())}


def oscillation_metrics(time_values, position, expected_period):
    """Positive X maxima refined by a cubic spline, no path alignment or fitting.

    At least two interior positive maxima are required for a period estimate.
    The initial turning point is not invented after trimming the RF window.
    """
    t, x = np.asarray(time_values), np.asarray(position)[:, 0]
    step = float(np.median(np.diff(t)))
    peaks, _ = find_peaks(x, distance=max(1, int(.7*expected_period/step)), prominence=.1*np.ptp(x))
    spline = CubicSpline(t, x)
    refined = []
    for i in peaks:
        fit = minimize_scalar(lambda tt: -float(spline(tt)), bounds=(t[i-1], t[i+1]),
                              method="bounded", options={"xatol": 1e-14})
        if fit.success and fit.fun < 0:
            refined.append(float(fit.x))
    periods = np.diff(refined)
    return {"positive_peak_times_s": refined, "periods_s": periods.tolist(),
            "mean_period_s": float(periods.mean()) if len(periods) else None,
            "period_spread_s": float(np.ptp(periods)) if len(periods) else None,
            "x_half_range_m": float(np.ptp(x)/2)}
