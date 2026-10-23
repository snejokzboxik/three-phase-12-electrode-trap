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
    Udc: float = 0.0,
) -> np.ndarray:
    """Evaluate m r¨ = q E(r,t) - γ r˙ + m g."""
    position = np.asarray(state[:2], dtype=float)
    velocity = np.asarray(state[2:4], dtype=float)
    if field_function is not None:
        electric_field = np.asarray(field_function(position, time), dtype=float)
    elif interpolator is not None:
        try:
            if Udc == 0.0:
                electric_field = np.asarray(interpolator.field(position.reshape(1, 2), time, U0, frequency)[0], dtype=float)
            else:
                electric_field = np.asarray(interpolator.field_with_dc(position.reshape(1, 2), time, Udc, U0, frequency)[0], dtype=float)
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
    max_step_per_period: int = 30,
    output_samples_per_period: float = 60.0,
    field_function: FieldFunction | None = None,
    Udc: float = 0.0,
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
        # DOP853 evaluates internal Runge--Kutta stages beyond an event before
        # it can localize the root.  Keep those provisional stages inside the
        # explicitly defined confinement disk so an otherwise valid escape is
        # not confused with an invalid FEM interpolation query.  The accepted
        # solution is still terminated by escape_event below.
        if field_function is None and interpolator is not None:
            radius = float(np.hypot(state[0], state[1]))
            if radius >= escape_radius:
                safe_radius = escape_radius * (1.0 - 1.0e-8)
                safe_position = np.asarray(state[:2], dtype=float) * safe_radius / max(radius, 1.0e-30)
                guarded_state = np.asarray(state, dtype=float).copy()
                guarded_state[:2] = safe_position
                return particle_rhs(time, guarded_state, particle, U0, frequency, interpolator, field_function, Udc)
        return particle_rhs(time, state, particle, U0, frequency, interpolator, field_function, Udc)

    def escape_event(time: float, state: np.ndarray) -> float:
        del time
        return float(np.hypot(state[0], state[1]) - escape_radius)

    escape_event.terminal = True
    escape_event.direction = 1
    period = 1.0 / frequency
    sample_count = max(1001, int(np.ceil(float(periods) * output_samples_per_period)) + 1)
    t_eval = np.linspace(0.0, simulation_duration, sample_count)
    try:
        solution = solve_ivp(rhs, (0.0, simulation_duration), y0, t_eval=t_eval, events=escape_event, rtol=rtol, atol=atol, max_step=period / max(int(max_step_per_period), 1), method="DOP853")
        if not solution.success:
            raise RuntimeError(f"Trajectory integration failed: {solution.message}")
        termination = "escape_radius" if solution.status == 1 else "completed"
    except InterpolationDomainError:
        # This is an explicit failure mode.  Callers can distinguish it from
        # a physically bounded integration and choose a smaller escape radius.
        raise
    time_values = solution.t
    state_values = solution.y
    if solution.status == 1 and solution.t_events and len(solution.t_events[0]) and solution.y_events and len(solution.y_events[0]):
        event_time = float(solution.t_events[0][0])
        event_state = np.asarray(solution.y_events[0][0], dtype=float)
        if len(time_values) == 0 or event_time > time_values[-1] + 1.0e-15:
            time_values = np.append(time_values, event_time)
            state_values = np.column_stack((state_values, event_state))
    radii = np.hypot(state_values[0], state_values[1])
    escaped = bool(termination == "escape_radius" or np.any(radii >= escape_radius * (1.0 - 1e-8)))
    return TrajectoryResult(time_values, state_values, float(np.max(radii)), float(radii[-1]), escaped, not escaped, float(time_values[-1]), float(time_values[-1] * frequency), termination)


def trajectory_metrics(result: TrajectoryResult, particle: ParticleConfig, aperture: float | None = None) -> dict[str, float | str]:
    """Calculate quantitative localization metrics from an integrated path."""
    position = result.state[:2]
    velocity = result.state[2:4]
    radius = np.hypot(position[0], position[1])
    rms_radius = float(np.sqrt(np.mean(radius**2)))
    kinetic_energy = 0.5 * particle.mass * np.sum(velocity * velocity, axis=0)
    metric_aperture = result.max_radius if aperture is None else aperture
    if result.escaped:
        classification = "escaped"
    elif rms_radius <= 0.25 * metric_aperture:
        classification = "strongly localized"
    elif rms_radius <= 0.60 * metric_aperture:
        classification = "weakly localized"
    else:
        classification = "marginal"
    return {
        "max_radius_m": float(result.max_radius),
        "rms_radius_m": rms_radius,
        "median_radius_m": float(np.median(radius)),
        "final_radius_m": float(result.final_radius),
        "escape_time_s": float(result.time[-1]) if result.escaped else float("nan"),
        "periods_survived": float(result.periods_simulated),
        "max_kinetic_energy_J": float(np.max(kinetic_energy)),
        "rms_kinetic_energy_J": float(np.sqrt(np.mean(kinetic_energy**2))),
        "micromotion_amplitude_m": float(0.25 * (np.ptp(position[0]) + np.ptp(position[1]))) if len(result.time) > 4 else float("nan"),
        "classification": classification,
    }
