"""Conservative model B for a smooth, potential-based RF reconstruction.

This is the leading high-frequency approximation, not an exact effective
Hamiltonian of the full RF dynamics. No DC, damping or gravity is included.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .rf_reconstruction import RFReconstruction


@dataclass
class AveragedRFModel:
    reconstruction: RFReconstruction
    charge: float
    mass: float
    amplitude: float
    frequency: float

    def __post_init__(self):
        if not np.all(np.isfinite([self.charge, self.mass, self.amplitude, self.frequency])):
            raise ValueError("Physical parameters must be finite")
        if self.mass <= 0 or self.frequency <= 0:
            raise ValueError("Mass and frequency must be positive")
        self.quadratures = self.reconstruction.scaled(self.amplitude)
        self.omega = 2*np.pi*self.frequency

    def potential(self, points):
        """q² (|Ec|²+|Es|²)/(4 m Omega²), in joules."""
        return self.quadratures.pseudo(self.reconstruction.check_points(points), self.charge, self.mass, self.omega)

    def force(self, points):
        """-q²/(2m Omega²) (Jc.T Ec + Js.T Es), in newtons."""
        return self.quadratures.force(self.reconstruction.check_points(points), self.charge, self.mass, self.omega)

    def energy(self, states):
        state = np.asarray(states, dtype=float)
        return self.mass*np.sum(state[..., 2:]**2, axis=-1)/2+self.potential(state[..., :2])


@dataclass
class AveragedTrajectory:
    time: np.ndarray
    states: np.ndarray
    energy: np.ndarray
    status: str
    evaluations: int
    exit_time: float | None


def integrate_averaged(model, initial_position, initial_velocity, duration, *,
                       samples=2401, rtol=2e-10, atol=2e-12, max_step=None):
    """DOP853 in dimensionless coordinates; return physical SI quantities.

    atol applies to dimensionless (R/L, V*duration/L), L=|R0| or radius/100.
    Output includes the exact domain-exit event. For rejected/trial RK stages
    outside the disk only, the force is evaluated at its radial projection on
    the boundary, so no unaudited polynomial extrapolation enters the output.
    A domain exit is a distinct incomplete status, never successful completion.
    """
    R = model.reconstruction.check_points(initial_position)
    V = np.asarray(initial_velocity, dtype=float)
    if R.shape != (2,) or V.shape != (2,) or not np.all(np.isfinite(V)):
        raise ValueError("Initial position and velocity must be finite 2-vectors")
    radius = model.reconstruction.valid_radius
    if np.linalg.norm(R) >= radius:
        raise ValueError("Initial position must be strictly inside reconstruction disk")
    if not np.isfinite(duration) or duration <= 0 or samples < 2:
        raise ValueError("Expected positive duration and at least two samples")
    if max_step is None:
        max_step = duration/600
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive and finite")
    length = max(float(np.linalg.norm(R)), radius/100)

    def rhs(t, y):
        xy = y[:2]*length
        norm = np.linalg.norm(xy)
        if norm > radius:
            xy = xy*(radius/norm)
        acceleration = model.force(xy)/model.mass
        return np.r_[y[2:], acceleration*duration**2/length]

    def exit_disk(t, y):
        return radius/length-np.linalg.norm(y[:2])

    exit_disk.terminal = True
    exit_disk.direction = -1
    solution = solve_ivp(rhs, (0., 1.), np.r_[R/length, V*duration/length],
                         method="DOP853", rtol=rtol, atol=atol,
                         max_step=max_step/duration, events=exit_disk,
                         t_eval=np.linspace(0., 1., samples))
    if not solution.success:
        raise RuntimeError(f"Model B integration failed: {solution.message}")
    time, states = solution.t*duration, solution.y.T.copy()
    exit_time = None
    if len(solution.t_events[0]):
        exit_time = float(solution.t_events[0][0]*duration)
        if abs(time[-1]-exit_time) <= 8*np.finfo(float).eps*duration:
            time[-1] = exit_time
            states[-1] = solution.y_events[0][0]
        else:
            time = np.r_[time, exit_time]
            states = np.vstack((states, solution.y_events[0][0]))
    states[:, :2] *= length
    states[:, 2:] *= length/duration
    return AveragedTrajectory(time, states, model.energy(states),
                              "domain_exit" if exit_time is not None else "completed",
                              solution.nfev, exit_time)
