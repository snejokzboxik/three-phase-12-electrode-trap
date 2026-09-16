"""Parameterized electrode geometry and exact ABC phase assignment."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


PHASE_NAMES = ("A", "B", "C")


@dataclass(frozen=True)
class Electrode:
    index: int
    angle: float
    center: tuple[float, float]
    phase: str


@dataclass(frozen=True)
class TrapGeometry:
    num_electrodes: int
    electrode_orbit_radius: float
    electrode_radius: float
    outer_domain_radius: float
    electrodes: tuple[Electrode, ...]


def phase_for_electrode(index: int) -> str:
    """Return the prescribed repeating phase sequence ABCABCABCABC."""
    return PHASE_NAMES[index % 3]


def build_geometry(
    num_electrodes: int = 12,
    electrode_orbit_radius: float = 0.010,
    electrode_radius: float = 0.0015,
    outer_domain_radius: float = 0.025,
) -> TrapGeometry:
    if num_electrodes != 12:
        raise ValueError("The three-phase trap geometry must contain exactly 12 electrodes.")
    angles = 2.0 * np.pi * np.arange(num_electrodes) / num_electrodes
    electrodes = tuple(
        Electrode(
            index=int(i),
            angle=float(angle),
            center=(float(electrode_orbit_radius * np.cos(angle)), float(electrode_orbit_radius * np.sin(angle))),
            phase=phase_for_electrode(int(i)),
        )
        for i, angle in enumerate(angles)
    )
    return TrapGeometry(num_electrodes, electrode_orbit_radius, electrode_radius, outer_domain_radius, electrodes)


def phase_indices(geometry: TrapGeometry) -> dict[str, np.ndarray]:
    return {phase: np.array([e.index for e in geometry.electrodes if e.phase == phase], dtype=int) for phase in PHASE_NAMES}


def electrode_mask(points: np.ndarray, geometry: TrapGeometry) -> np.ndarray:
    """Return whether points lie inside at least one circular electrode."""
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    centers = np.array([e.center for e in geometry.electrodes])
    distance2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.any(distance2 <= geometry.electrode_radius**2, axis=1)

