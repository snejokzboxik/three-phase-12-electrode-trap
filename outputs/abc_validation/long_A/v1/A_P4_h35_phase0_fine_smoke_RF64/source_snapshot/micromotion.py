"""Leading near-identity slow/full coordinate transform and common RF phase.

Algebraic inversion of this truncated transform does not remove its physical
higher-order error. Coordinates are metres, velocities metres per second.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _parameters(charge, mass, omega):
    if not np.all(np.isfinite([charge, mass, omega])) or mass <= 0 or omega <= 0:
        raise ValueError("Expected finite charge and positive mass/frequency")
    return charge/(mass*omega**2)


def _instantaneous(model, points, omega, angle):
    ec, es = model.field_quadratures(points)
    jc, js = model.jacobian_quadratures(points)
    angle = np.asarray(angle)
    co, si = np.cos(angle), np.sin(angle)
    E = ec*co[..., None]+es*si[..., None]
    dtE = omega*(-ec*si[..., None]+es*co[..., None])
    J = jc*co[..., None, None]+js*si[..., None, None]
    return E, dtE, J


def matched_full_state(model, position, velocity, charge, mass, omega, phase):
    """r=R-qE/(mΩ²), v=V-q[∂tE+J V]/(mΩ²), including convection.

    model contains amplitude-scaled smooth quadratures. phase may be a scalar
    or the array Ω*t+phi0; batched input has shape (..., 2).
    """
    factor = _parameters(charge, mass, omega)
    R, V = np.broadcast_arrays(np.asarray(position, float), np.asarray(velocity, float))
    if R.shape[-1] != 2 or not np.all(np.isfinite(R)) or not np.all(np.isfinite(V)) or not np.all(np.isfinite(phase)):
        raise ValueError("Expected finite 2-vector positions/velocities and RF phases")
    E, dtE, J = _instantaneous(model, R, omega, phase)
    return np.concatenate((R-factor*E, V-factor*(dtE+np.einsum("...ij,...j->...i", J, V))), axis=-1)


def recover_slow_state(model, full_state, charge, mass, omega, phase, *, check_points=None,
                       coordinate_tolerance=1e-14, max_iterations=30):
    """Invert the leading transform by Newton iteration, then solve for V.

    A local inverse is sought near the full coordinate; global uniqueness is
    not claimed. Non-convergence, singularity and invalid domains raise errors.
    check_points can be RFReconstruction.check_points for production use.
    """
    factor = _parameters(charge, mass, omega)
    state = np.asarray(full_state, float)
    if state.shape[-1] != 4 or not np.all(np.isfinite(state)) or not np.all(np.isfinite(phase)):
        raise ValueError("Expected finite (..., 4) states and phases")
    if coordinate_tolerance <= 0 or max_iterations < 1:
        raise ValueError("Expected positive coordinate tolerance and iteration count")
    r, v = state[..., :2], state[..., 2:]
    R = r.copy()
    for _ in range(max_iterations):
        if check_points is not None:
            check_points(R)
        E, dtE, J = _instantaneous(model, R, omega, phase)
        matrix = np.eye(2)-factor*J
        if np.any(np.linalg.cond(matrix) > 1e12):
            raise RuntimeError("Singular or ill-conditioned micromotion coordinate transform")
        residual = R-factor*E-r
        if np.max(np.linalg.norm(residual, axis=-1)) <= coordinate_tolerance:
            break
        R -= np.linalg.solve(matrix, residual[..., None])[..., 0]
    else:
        raise RuntimeError("Slow initial coordinate recovery failed to converge")
    V = np.linalg.solve(matrix, (v+factor*dtE)[..., None])[..., 0]
    return np.concatenate((R, V), axis=-1)


def _drift_correction(model, R, V, charge, mass, omega, phase):
    """First slow-amplitude derivative in the RF particular solution.

    delta_xi=-2 q/(m Omega^4) (partial_t J) V. Its total derivative is
    2 beta J V -2 beta/Omega^2 [H_(partial_t E)[V,V]+(partial_t J) A],
    beta=q/(m Omega^2), A=-grad(Psi)/m. This addresses slow amplitude drift;
    it is not the complete next-order averaging transformation/Hamiltonian.
    """
    beta = _parameters(charge, mass, omega)
    jc, js = model.jacobian_quadratures(R)
    co, si = np.cos(phase), np.sin(phase)
    J = jc*co[..., None, None]+js*si[..., None, None]
    dtJ = omega*(-jc*si[..., None, None]+js*co[..., None, None])
    z = (R[..., 0]+1j*R[..., 1])/model.reference_radius
    direction = V[..., 0]+1j*V[..., 1]
    hessian_along_V = []
    for coefficients in (model.cosine, model.sine):
        d3 = np.polynomial.polynomial.polyval(z, np.polynomial.polynomial.polyder(coefficients, 3))/model.reference_radius**3
        value = d3*direction**2
        hessian_along_V.append(np.stack((-value.real, value.imag), axis=-1))
    hdt = omega*(-hessian_along_V[0]*si[..., None]+hessian_along_V[1]*co[..., None])
    acceleration = model.force(R, charge, mass, omega)/mass
    delta_r = -2*beta/omega**2*np.einsum("...ij,...j->...i", dtJ, V)
    delta_v = 2*beta*np.einsum("...ij,...j->...i", J, V)-2*beta/omega**2*(hdt+np.einsum("...ij,...j->...i", dtJ, acceleration))
    return np.concatenate((delta_r, delta_v), axis=-1)


def matched_full_state_with_drift(model, position, velocity, charge, mass, omega, phase):
    """Leading transform plus the explicitly declared slow-amplitude correction."""
    R, V = np.broadcast_arrays(np.asarray(position, float), np.asarray(velocity, float))
    phase = np.asarray(phase)
    leading = matched_full_state(model, R, V, charge, mass, omega, phase)
    return leading+_drift_correction(model, R, V, charge, mass, omega, phase)


def recover_slow_state_with_drift(model, full_state, charge, mass, omega, phase, *,
                                  check_points=None, coordinate_tolerance=1e-14, max_iterations=30):
    """Local fixed-point inversion of the drift-corrected transform.

    Each update solves the leading near-identity inverse. Velocity convergence
    is measured on Omega*coordinate_tolerance; failure is explicit.
    """
    state, phase = np.asarray(full_state, float), np.asarray(phase)
    slow = recover_slow_state(model, state, charge, mass, omega, phase,
                              check_points=check_points, coordinate_tolerance=coordinate_tolerance)
    for _ in range(max_iterations):
        correction = _drift_correction(model, slow[..., :2], slow[..., 2:], charge, mass, omega, phase)
        updated = recover_slow_state(model, state-correction, charge, mass, omega, phase,
                                     check_points=check_points, coordinate_tolerance=coordinate_tolerance)
        if (np.max(np.linalg.norm(updated[..., :2]-slow[..., :2], axis=-1)) <= coordinate_tolerance
                and np.max(np.linalg.norm(updated[..., 2:]-slow[..., 2:], axis=-1)) <= omega*coordinate_tolerance):
            return updated
        slow = updated
    raise RuntimeError("Drift-corrected slow state recovery failed to converge")


@dataclass
class PhasedFEMField:
    """Shift time identically for all electrode groups: t -> t+phi0/Omega.

    Use the same full initial state as A_smooth when isolating A-A_smooth.
    No smoothed-field derivatives are substituted into the raw FEM field.
    """
    interpolator: object
    amplitude: float
    frequency: float
    phase: float = 0.

    def __post_init__(self):
        if not np.all(np.isfinite([self.amplitude, self.frequency, self.phase])) or self.frequency <= 0:
            raise ValueError("Expected finite amplitude/phase and positive frequency")

    def __call__(self, points, time):
        return self.interpolator.field(points, time+self.phase/(2*np.pi*self.frequency), self.amplitude, self.frequency)
