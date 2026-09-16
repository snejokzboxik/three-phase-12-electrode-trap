"""Time-dependent field reconstruction from the three FEM basis fields."""

from __future__ import annotations

import numpy as np

from .fem import FEMBasisSolutions, superpose_field, superpose_potential
from .voltages import phase_voltages


def potential_at_time(solutions: FEMBasisSolutions, t: float, U0: float, frequency: float) -> np.ndarray:
    return superpose_potential(solutions, phase_voltages(t, U0, frequency))


def field_at_time(solutions: FEMBasisSolutions, t: float, U0: float, frequency: float) -> np.ndarray:
    return superpose_field(solutions, phase_voltages(t, U0, frequency))


def potential_at_time_with_dc(solutions: FEMBasisSolutions, t: float, Udc: float, U0: float, frequency: float) -> np.ndarray:
    """Reconstruct the full potential from the RF and common-DC bases."""
    return float(Udc) * solutions.potentials["DC"] + superpose_potential(solutions, phase_voltages(t, U0, frequency))


def field_at_time_with_dc(solutions: FEMBasisSolutions, t: float, Udc: float, U0: float, frequency: float) -> np.ndarray:
    """Reconstruct the full electric field from the RF and common-DC bases."""
    return float(Udc) * solutions.fields["DC"] + superpose_field(solutions, phase_voltages(t, U0, frequency))


def field_magnitude(field: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(field), axis=-1)
