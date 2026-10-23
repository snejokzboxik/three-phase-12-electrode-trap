"""Quasi-2D finite-flight-time mass-filter analysis."""

from __future__ import annotations

from dataclasses import replace
import csv
from pathlib import Path
import numpy as np

from .config import ProjectConfig
from .dynamics import integrate_trajectory, trajectory_metrics
from .interpolation import FEMInterpolator


def ensemble_initial_states(config: ProjectConfig) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Build a small reproducible entrance ensemble in position and velocity."""
    cfg = config.mass_filter
    states = []
    for radius in cfg.ensemble_radii:
        for angle_deg in cfg.ensemble_angles_deg:
            angle = np.deg2rad(angle_deg)
            position = (radius * np.cos(angle), radius * np.sin(angle))
            tangent = np.array((-np.sin(angle), np.cos(angle)))
            for speed in cfg.ensemble_velocity_scales:
                if speed == 0:
                    signs = (1,)
                else:
                    signs = (-1, 1)
                for sign in signs:
                    velocity = tuple(float(v) for v in sign * speed * tangent)
                    states.append((tuple(float(v) for v in position), velocity))
    return states


def flight_time(config: ProjectConfig) -> float:
    if config.mass_filter.length <= 0 or config.mass_filter.axial_velocity <= 0:
        raise ValueError("Mass-filter length and axial velocity must be positive.")
    return config.mass_filter.length / config.mass_filter.axial_velocity


def run_mass_transmission_scan(interpolator: FEMInterpolator, config: ProjectConfig) -> list[dict[str, float | int | str]]:
    """Launch the same entrance ensemble for each mass at a reference RF frequency."""
    states = ensemble_initial_states(config)
    duration = flight_time(config)
    rows: list[dict[str, float | int | str]] = []
    for multiplier in config.mass_filter.mass_multipliers:
        mass = config.particle.mass * multiplier
        particle = replace(config.particle, mass=mass, gamma=0.0, gravity_enabled=False)
        transmitted = 0
        metrics_list = []
        for position, velocity in states:
            result = integrate_trajectory(
                interpolator, particle, config.rf.U0, config.mass_filter.reference_frequency,
                initial_position=position, initial_velocity=velocity,
                simulation_duration=duration, escape_radius=config.mass_filter.exit_aperture,
                rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol,
                max_step_per_period=10, output_samples_per_period=8,
            )
            is_transmitted = int(not result.escaped)
            transmitted += is_transmitted
            metrics_list.append(trajectory_metrics(result, particle, config.mass_filter.exit_aperture))
        rows.append({
            "mass_kg": mass, "mass_over_charge_kg_per_C": mass / config.particle.charge,
            "mass_multiplier": multiplier, "U0_V": config.rf.U0,
            "frequency_Hz": config.mass_filter.reference_frequency,
            "flight_time_s": duration, "launched": len(states), "transmitted": transmitted,
            "transmission_fraction": transmitted / len(states),
            "mean_rms_radius_m": float(np.mean([item["rms_radius_m"] for item in metrics_list])),
            "mean_max_radius_m": float(np.mean([item["max_radius_m"] for item in metrics_list])),
        })
    return rows


def write_mass_transmission_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_mass_frequency_map(interpolator: FEMInterpolator, config: ProjectConfig) -> list[dict[str, float | int]]:
    """Estimate survival fraction across mass and frequency with a reduced ensemble."""
    states = ensemble_initial_states(config)
    # The map uses the zero-velocity, two-radius, two-angle subset to keep the
    # exploratory grid tractable; the full ensemble is used for the main curve.
    states = [state for state in states if state[1] == (0.0, 0.0) and state[0][0] >= 0.0][:4]
    if not states:
        states = ensemble_initial_states(config)[:4]
    duration = flight_time(config)
    rows: list[dict[str, float | int]] = []
    for frequency in config.mass_filter.map_frequencies:
        for multiplier in config.mass_filter.mass_multipliers:
            mass = config.particle.mass * multiplier
            particle = replace(config.particle, mass=mass, gamma=0.0, gravity_enabled=False)
            survived = 0
            periods = duration * frequency
            for position, velocity in states:
                result = integrate_trajectory(
                    interpolator, particle, config.rf.U0, frequency,
                    initial_position=position, initial_velocity=velocity,
                    simulation_duration=duration, escape_radius=config.mass_filter.exit_aperture,
                    rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol,
                    max_step_per_period=10, output_samples_per_period=4,
                )
                survived += int(not result.escaped)
            rows.append({"mass_kg": mass, "mass_over_charge_kg_per_C": mass / config.particle.charge, "mass_multiplier": multiplier, "frequency_Hz": frequency, "flight_time_s": duration, "flight_periods": periods, "survival_fraction": survived / len(states), "launched": len(states), "survived": survived})
    return rows


def choose_representative_masses(rows: list[dict[str, float | int | str]]) -> tuple[float, float, float]:
    fractions = np.array([float(row["transmission_fraction"]) for row in rows])
    masses = np.array([float(row["mass_kg"]) for row in rows])
    best = int(np.argmax(fractions))
    lower = masses[max(0, best - 2)]
    upper = masses[min(len(masses) - 1, best + 2)]
    return float(masses[best]), float(lower), float(upper)


def estimate_fwhm(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    masses = np.array([float(row["mass_kg"]) for row in rows])
    transmission = np.array([float(row["transmission_fraction"]) for row in rows])
    peak_index = int(np.argmax(transmission))
    peak = float(transmission[peak_index])
    half = peak / 2.0
    above = masses[transmission >= half]
    if len(above) < 2 or peak <= 0:
        return {"peak_mass_kg": float(masses[peak_index]), "peak_transmission": peak, "fwhm_kg": float("nan"), "resolving_power": float("nan")}
    width = float(above[-1] - above[0])
    return {"peak_mass_kg": float(masses[peak_index]), "peak_transmission": peak, "fwhm_kg": width, "resolving_power": float(masses[peak_index] / width) if width > 0 else float("nan")}

