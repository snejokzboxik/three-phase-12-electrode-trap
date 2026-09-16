"""Full time-dependent charged-particle dynamics in the reconstructed RF field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp

from .config import ParticleConfig
from .interpolation import FEMInterpolator, InterpolationDomainError


FieldFunction = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class TrajectoryResult:
    time: np.ndarray
    state: np.ndarray
    max_radius: float
    final_radius: float
    escaped: bool
    bounded: bool
    simulation_duration: float
    periods_simulated: float
    termination: str


def particle_rhs(
    time: float,
    state: np.ndarray,
    particle: ParticleConfig,
    U0: float,
    frequency: float,
    interpolator: FEMInterpolator | None = None,
    field_function: FieldFunction | None = None,
) -> np.ndarray:
    """Evaluate m r¨ = q E(r,t) - γ r˙ + m g."""
    position = np.asarray(state[:2], dtype=float)
    velocity = np.asarray(state[2:4], dtype=float)
    if field_function is not None:
        electric_field = np.asarray(field_function(position, time), dtype=float)
    elif interpolator is not None:
        try:
            electric_field = np.asarray(interpolator.field(position.reshape(1, 2), time, U0, frequency)[0], dtype=float)
        except InterpolationDomainError as error:
            raise InterpolationDomainError("Particle reached an invalid FEM interpolation region.") from error
    else:
        raise ValueError("Either interpolator or field_function must be supplied.")
    if electric_field.shape != (2,) or not np.all(np.isfinite(electric_field)):
        raise InterpolationDomainError("Particle field evaluation returned a non-finite or invalid vector.")
    gravity = np.array([0.0, -particle.gravity if particle.gravity_enabled else 0.0])
    acceleration = (particle.charge / particle.mass) * electric_field - (particle.gamma / particle.mass) * velocity + gravity
    return np.array([velocity[0], velocity[1], acceleration[0], acceleration[1]], dtype=float)


def integrate_trajectory(
    interpolator: FEMInterpolator | None,
    particle: ParticleConfig,
    U0: float,
    frequency: float,
    initial_position: tuple[float, float] | None = None,
    initial_velocity: tuple[float, float] | None = None,
    periods: float | None = None,
    simulation_duration: float | None = None,
    escape_radius: float = 0.020,
    rtol: float = 2.0e-7,
    atol: float = 1.0e-11,
    field_function: FieldFunction | None = None,
) -> TrajectoryResult:
    """Integrate the full RF equation and classify confinement explicitly.

    The escape event is ``r >= escape_radius``.  Queries that enter an
    electrode or leave the FEM triangulation raise ``InterpolationDomainError``
    instead of silently returning an extrapolated field.
    """
    if simulation_duration is None:
        if periods is None:
            raise ValueError("Provide periods or simulation_duration.")
        simulation_duration = float(periods) / frequency
    else:
        simulation_duration = float(simulation_duration)
        periods = simulation_duration * frequency
    if initial_position is None:
        initial_position = particle.initial_position
    if initial_velocity is None:
        initial_velocity = particle.initial_velocity
    y0 = np.array([initial_position[0], initial_position[1], initial_velocity[0], initial_velocity[1]], dtype=float)
    if not np.all(np.isfinite(y0)):
        raise ValueError("Initial particle state must be finite.")
    if escape_radius <= 0:
        raise ValueError("escape_radius must be positive.")

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        return particle_rhs(time, state, particle, U0, frequency, interpolator, field_function)

    def escape_event(time: float, state: np.ndarray) -> float:
        del time
        return float(np.hypot(state[0], state[1]) - escape_radius)

    escape_event.terminal = True
    escape_event.direction = 1
    period = 1.0 / frequency
    sample_count = max(1001, int(np.ceil(float(periods) * 60.0)) + 1)
    t_eval = np.linspace(0.0, simulation_duration, sample_count)
    try:
        solution = solve_ivp(rhs, (0.0, simulation_duration), y0, t_eval=t_eval, events=escape_event, rtol=rtol, atol=atol, max_step=period / 30.0, method="DOP853")
        termination = "escape_radius" if solution.status == 1 else "completed"
    except InterpolationDomainError:
        # This is an explicit failure mode.  Callers can distinguish it from
        # a physically bounded integration and choose a smaller escape radius.
        raise
    radii = np.hypot(solution.y[0], solution.y[1])
    escaped = bool(termination == "escape_radius" or np.any(radii >= escape_radius * (1.0 - 1e-8)))
    return TrajectoryResult(solution.t, solution.y, float(np.max(radii)), float(radii[-1]), escaped, not escaped, float(solution.t[-1]), float(solution.t[-1] * frequency), termination)

