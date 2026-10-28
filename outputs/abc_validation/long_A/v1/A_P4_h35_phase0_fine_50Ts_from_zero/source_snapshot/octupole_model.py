"""Leading rotating octupole and its conservative r^6 pseudopotential."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .rf_reconstruction import HarmonicQuadratures, RFReconstruction


def octupole_ring_coefficients(cosine_potential, sine_potential, angles, radius):
    """Fourier extraction from a uniformly sampled circle (no duplicated end).

    Phi4=Re(c4 (x+iy)^4) cos(chi)+Re(s4 (x+iy)^4) sin(chi).
    The desired rotation is C4 r^4 sin(chi-4theta+delta4).
    Its coefficient d_minus=(s4-i*c4)/2=C4 exp(-i*delta4).
    The counter-rotating component is d_plus=(s4+i*c4)/2.
    Returned coefficients include the voltage amplitude of the input potential.
    """
    a = np.asarray(angles, dtype=float)
    pc, ps = np.asarray(cosine_potential), np.asarray(sine_potential)
    if a.ndim != 1 or len(a) < 17 or pc.shape != a.shape or ps.shape != a.shape:
        raise ValueError("Need equally sized one-dimensional circle samples")
    if not np.isfinite(radius) or radius <= 0 or not np.all(np.isfinite([a, pc, ps])):
        raise ValueError("Expected finite samples and a positive radius")
    gap = np.diff(np.r_[a, a[0]+2*np.pi])
    if not np.allclose(gap, 2*np.pi/len(a), rtol=1e-9, atol=1e-12):
        raise ValueError("Angles must be uniformly increasing over one circle")
    c4, s4 = (2*np.mean(p*np.exp(-4j*a))/radius**4 for p in (pc, ps))
    return c4, s4, (s4-1j*c4)/2, (s4+1j*c4)/2


@dataclass
class OctupoleModel:
    """Phi4=C4 r^4 sin(Omega*t+phi0-4theta+delta4).

    coefficient_per_volt is C4/U0 in m^-4, positive; phase is delta4.
    Only the n=4 co-rotating mode is retained, centered on the geometric origin.
    The reconstruction property also provides the RF field and domain interface
    used by integrate_averaged; potential/force below are independent formulas.
    """
    coefficient_per_volt: float
    phase: float
    charge: float
    mass: float
    amplitude: float
    frequency: float
    valid_radius: float

    def __post_init__(self):
        if not np.all(np.isfinite([self.coefficient_per_volt, self.phase, self.charge,
                                   self.mass, self.amplitude, self.frequency, self.valid_radius])):
            raise ValueError("Parameters must be finite")
        if self.coefficient_per_volt <= 0 or self.mass <= 0 or self.frequency <= 0 or self.valid_radius <= 0:
            raise ValueError("Coefficient, mass, frequency and domain radius must be positive")
        self.omega = 2*np.pi*self.frequency
        self.C4 = self.amplitude*self.coefficient_per_volt
        self.B6 = 8*self.charge**2*self.C4**2/(self.mass*self.omega**2)
        c, s = np.zeros(5, complex), np.zeros(5, complex)
        d = self.coefficient_per_volt*self.valid_radius**4*np.exp(-1j*self.phase)
        c[4], s[4] = 1j*d, d
        self.reconstruction = RFReconstruction(HarmonicQuadratures(c, s, self.valid_radius),
                                              self.valid_radius, {"representation": "pure rotating n=4; geometric origin"})

    def potential(self, points):
        xy = self.reconstruction.check_points(points)
        return self.B6*np.sum(xy*xy, axis=-1)**3

    def force(self, points):
        xy = self.reconstruction.check_points(points)
        return -6*self.B6*np.sum(xy*xy, axis=-1)[..., None]**2*xy

    def energy(self, states):
        state = np.asarray(states)
        return self.mass*np.sum(state[..., 2:]**2, axis=-1)/2+self.potential(state[..., :2])
