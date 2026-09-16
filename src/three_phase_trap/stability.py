"""Explicit trajectory-based stability classification and parameter scans."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import numpy as np

from .config import ProjectConfig
from .dynamics import TrajectoryResult, integrate_trajectory
from .interpolation import FEMInterpolator


@dataclass(frozen=True)
class StabilityRow:
    U0: float
    f_rf: float
    mass: float
    charge: float
    gamma: float
    initial_x: float
    initial_y: float
    r_max: float
    r_final: float
    escaped: bool
    bounded: bool


def run_stability_scan(interpolator: FEMInterpolator, config: ProjectConfig) -> list[StabilityRow]:
    rows: list[StabilityRow] = []
    duration_periods = min(config.simulation.trajectory_periods, 20)
    for U0 in config.stability.U0_values:
        for frequency in config.stability.frequency_values:
            for initial in config.stability.initial_positions:
                result = integrate_trajectory(
                    interpolator,
                    config.particle,
                    U0=U0,
                    frequency=frequency,
                    initial_position=initial,
                    periods=duration_periods,
                    escape_radius=config.simulation.escape_radius,
                    rtol=config.simulation.trajectory_rtol,
                    atol=config.simulation.trajectory_atol,
                )
                rows.append(StabilityRow(U0, frequency, config.particle.mass, config.particle.charge, config.particle.gamma, initial[0], initial[1], result.max_radius, result.final_radius, result.escaped, result.bounded))
    return rows


def write_stability_csv(rows: list[StabilityRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("U0", "f_rf", "mass", "charge", "gamma", "initial_x", "initial_y", "r_max", "r_final", "escaped", "bounded"))
        for row in rows:
            writer.writerow((row.U0, row.f_rf, row.mass, row.charge, row.gamma, row.initial_x, row.initial_y, row.r_max, row.r_final, int(row.escaped), int(row.bounded)))

