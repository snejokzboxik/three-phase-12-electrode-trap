"""Continuous analytical three-phase RF voltage functions."""

from __future__ import annotations

import numpy as np


PHASE_OFFSETS = {"A": 0.0, "B": -2.0 * np.pi / 3.0, "C": 2.0 * np.pi / 3.0}


def phase_voltages(t: float | np.ndarray, U0: float, frequency: float) -> dict[str, np.ndarray | float]:
    """Evaluate continuous sinusoidal phase voltages at arbitrary times."""
    omega = 2.0 * np.pi * frequency
    return {phase: U0 * np.sin(omega * np.asarray(t) + offset) for phase, offset in PHASE_OFFSETS.items()}


def voltage_matrix(times: np.ndarray, U0: float, frequency: float) -> np.ndarray:
    values = phase_voltages(times, U0, frequency)
    return np.column_stack([values[p] for p in ("A", "B", "C")])


def phase_periodicity_error(t: np.ndarray, U0: float, frequency: float) -> float:
    T = 1.0 / frequency
    first = voltage_matrix(t, U0, frequency)
    shifted = voltage_matrix(t + T, U0, frequency)
    return float(np.max(np.abs(first - shifted)))

