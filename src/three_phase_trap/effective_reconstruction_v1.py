"""Inverse period law and effective-dynamics diagnostics for ideal ABC v1.

All external quantities are SI. The potential is parameterized by its
energy at a fixed reference length, so fitting the exponent preserves units.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import beta

from .trajectory_diagnostics import turning_orientations


LENGTH = 1e-3


@dataclass(frozen=True)
class RadialPower:
    exponent: float
    energy_at_mm_J: float

    @property
    def coefficient(self):
        """Return B in J m**(-p)."""
        return self.energy_at_mm_J / LENGTH**self.exponent

    def potential(self, radius):
        return self.energy_at_mm_J * (np.asarray(radius) / LENGTH)**self.exponent

    def period(self, amplitude, mass):
        p = self.exponent
        return (4 * LENGTH * np.sqrt(mass / (2 * self.energy_at_mm_J))
                * beta(1 / p, .5) / p * (np.asarray(amplitude) / LENGTH)**(1 - p / 2))


def fit_power(amplitudes, periods, mass, *, exponent=None):
    """Unweighted least squares in log period. Reject rank-deficient free fits."""
    a, t = np.asarray(amplitudes, float), np.asarray(periods, float)
    if (a.ndim != 1 or a.shape != t.shape or not len(a)
            or not np.isfinite(mass) or mass <= 0
            or not np.all(np.isfinite([a, t])) or np.any(a <= 0) or np.any(t <= 0)):
        raise ValueError("Expected paired positive SI amplitudes/periods and positive mass")
    log_a, log_t = np.log(a / LENGTH), np.log(t)
    design = np.column_stack((np.ones(len(a)), log_a))
    if exponent is None:
        if np.linalg.matrix_rank(design) != 2:
            raise ValueError("A free exponent requires distinct amplitudes")
        intercept, slope = np.linalg.lstsq(design, log_t, rcond=None)[0]
        exponent = 2 * (1 - slope)
    else:
        intercept = np.mean(log_t - (1 - exponent / 2) * log_a)
    if not np.isfinite(exponent) or exponent <= 1:
        raise ValueError("Expected a confining differentiable power, p > 1")
    unit_period = RadialPower(exponent, 1.).period(LENGTH, mass)
    model = RadialPower(float(exponent), float((unit_period / np.exp(intercept))**2))
    return model, dict(log_design_rank=int(np.linalg.matrix_rank(design)),
                       log_design_singular_values=np.linalg.svd(design, compute_uv=False).tolist(),
                       log_period_residuals=np.log(t / model.period(a, mass)).tolist())


def check_split(train, test):
    """Group by physical initial condition, including RF phase and drive."""
    if not train or not test or set(train) & set(test):
        raise ValueError("Train and test must be nonempty and disjoint by launch")


def turning_data(time, states, expected_period):
    t, points, angles, summary = turning_orientations(time, states[:, :2], expected_period)
    periods = t[2:] - t[:-2]
    radii = np.linalg.norm(points, axis=1)
    summary.update(period_s=float(periods.mean()), period_span_s=float(np.ptp(periods)),
                   frequency_Hz=float(1 / periods.mean()), amplitude_m=float(radii.mean()))
    return summary, np.column_stack((t, points, radii, angles))


def integrate_effective(time, state, mass, power, *, gamma_mm=0., fem_model=None):
    """Use the common physical clock. gamma_mm multiplies (r/mm)^4 J V in acceleration."""
    t = np.asarray(time)
    time_scale = power.period(LENGTH, mass)
    initial = np.asarray(state) / [LENGTH, LENGTH, LENGTH / time_scale, LENGTH / time_scale]

    def rhs(_, y):
        xy, v = y[:2], y[2:]
        r = np.hypot(*xy)
        if fem_model is None:
            k = power.exponent * power.energy_at_mm_J * time_scale**2 / (mass * LENGTH**2)
            acceleration = -k * r**(power.exponent - 2) * xy if r else np.zeros(2)
        else:
            acceleration = fem_model.force(LENGTH * xy) * time_scale**2 / (mass * LENGTH)
        gyro = gamma_mm * time_scale * r**4 * np.array([-v[1], v[0]])
        return np.r_[v, acceleration + gyro]

    solution = solve_ivp(rhs, (0., (t[-1] - t[0]) / time_scale), initial, method="DOP853",
                         t_eval=(t - t[0]) / time_scale, rtol=3e-13, atol=3e-15,
                         max_step=1 / 160)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T * [LENGTH, LENGTH, LENGTH / time_scale, LENGTH / time_scale]


def octupole_gamma(charge, mass, C4, omega, helicity=1):
    """Coefficient G(1 mm)/m, in s^-1, for J(x,y)=(-y,x)."""
    return helicity * 2 * (12 * charge * C4 * LENGTH**2 / mass)**2 / omega**3


def angular_momentum_fit(states):
    """Estimate gamma independently from L_z = constant + gamma r^6/(6 L^4).

    L_z is angular momentum per mass. Residuals are diagnostics, not IID errors.
    """
    xy, v = states[:, :2], states[:, 2:]
    angular_momentum = xy[:, 0] * v[:, 1] - xy[:, 1] * v[:, 0]
    predictor = LENGTH**2 * np.sum((xy / LENGTH)**2, axis=1)**3 / 6
    design = np.column_stack((np.ones(len(states)), predictor))
    intercept, gamma = np.linalg.lstsq(design, angular_momentum, rcond=None)[0]
    residual = angular_momentum - design @ [intercept, gamma]
    return dict(gamma_mm_s_inv=float(gamma), intercept_m2_per_s=float(intercept),
                residual_rms_m2_per_s=float(np.sqrt(np.mean(residual**2))))
