"""Exact planar rigid-rod multipole dynamics for the V16.1 controls.

Coordinates are normalized as in the parent V16 package. A field term is
``(order, kind, kappa, omega, delta)`` where kind is ``RO`` (travelling) or
``SO`` (standing). The moments are raw normalized line-charge moments
``mu_j = <s**j>``; for a uniform rod mu_2=ell**2/12 and mu_4=ell**4/80.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
import math
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Harmonic:
    order: int
    kind: str
    kappa: float
    omega: float = 1.0
    delta: float = 0.0

    def __post_init__(self):
        if self.order < 1 or self.kind.upper() not in {"RO", "SO"}:
            raise ValueError("harmonic must have positive order and kind RO or SO")
        object.__setattr__(self, "kind", self.kind.upper())


def uniform_rod_moments(ell: float, max_order: int = 4) -> np.ndarray:
    """Return moments mu_j=<s**j> for a centered uniformly charged rod."""
    out = np.zeros(max_order + 1, dtype=float)
    out[0] = 1.0
    for j in range(2, max_order + 1, 2):
        out[j] = ell**j / ((j + 1) * 2**j)
    return out


def rod_multipole_jet(x: float, y: float, psi: float, ell: float,
                      order: int, moments=None):
    """Complex polynomial P=< (Z+s exp(i psi))**order > and its jet.

    Returns P, P_x, P_y, P_psi, P_xx, P_xy, P_yy, P_xpsi, P_ypsi,
    P_psipsi. The exact polynomial derivatives are used throughout.
    """
    mu = uniform_rod_moments(ell, order) if moments is None else np.asarray(moments)
    z = complex(x, y)
    p = px = pxx = pp = ppp = pxp = 0j
    for j in range(order + 1):
        muj = mu[j] if j < len(mu) else 0.0
        if muj == 0:
            continue
        n = order - j
        c = comb(order, j) * muj * np.exp(1j*j*psi)
        p += c * z**n
        if n:
            px += c*n*z**(n-1)
            pp += 1j*j*c*z**n
            pxp += 1j*j*c*n*z**(n-1)
        if n >= 2:
            pxx += c*n*(n-1)*z**(n-2)
        if n == 0:
            pp += 1j*j*c
    py = 1j*px
    pxy = 1j*pxx
    pyy = -pxx
    pyp = 1j*pxp
    ppp = sum((-(j*j)*comb(order,j)*(mu[j] if j < len(mu) else 0.0)
               * np.exp(1j*j*psi)*z**(order-j) for j in range(order+1)), 0j)
    return p, px, py, pp, pxx, pxy, pyy, pxp, pyp, ppp


def potential_jet(x: float, y: float, psi: float, tau: float, ell: float,
                  harmonics, moments=None):
    """Return U, grad(U) in (x,y,psi), and its 3x3 Hessian."""
    value = 0.0
    grad = np.zeros(3)
    hess = np.zeros((3, 3))
    for h in harmonics:
        jet = rod_multipole_jet(x, y, psi, ell, h.order, moments)
        p, px, py, pp, pxx, pxy, pyy, pxp, pyp, ppp = jet
        if h.kind == "RO":
            fac = h.kappa * np.exp(-1j*(h.omega*tau-h.delta))
        else:
            fac = h.kappa * np.cos(h.omega*tau-h.delta)
        value += np.real(fac*p)
        grad += np.real(fac*np.array([px, py, pp]))
        hess += np.real(fac*np.array([[pxx, pxy, pxp],
                                      [pxy, pyy, pyp],
                                      [pxp, pyp, ppp]]))
    return float(value), grad, hess


def rigid_rhs(tau: float, state, harmonics, ell: float, gamma_trans: float,
              gamma_rot: float, inertia: float | None = None, moments=None):
    x, y, vx, vy, psi, spin = state
    U, g, _ = potential_jet(x, y, psi, tau, ell, harmonics, moments)
    I = ell*ell/12 if inertia is None else inertia
    return np.array([vx, vy, -g[0]-gamma_trans*vx,
                     -g[1]-gamma_trans*vy, spin,
                     -g[2]/I-gamma_rot*spin], dtype=float)


def lab_jacobian(tau: float, state, harmonics, ell: float, gamma_trans: float,
                 gamma_rot: float, inertia: float | None = None, moments=None):
    """Exact Jacobian of the six-dimensional lab-frame ODE."""
    _, _, H = potential_jet(*state[:2], state[4], tau, ell, harmonics, moments)
    I = ell*ell/12 if inertia is None else inertia
    J = np.zeros((6, 6))
    J[0, 2] = J[1, 3] = J[4, 5] = 1.0
    J[2, 0], J[2, 1], J[2, 4], J[2, 2] = -H[0,0], -H[0,1], -H[0,2], -gamma_trans
    J[3, 0], J[3, 1], J[3, 4], J[3, 3] = -H[1,0], -H[1,1], -H[1,2], -gamma_trans
    J[5, 0], J[5, 1], J[5, 4], J[5, 5] = -H[2,0]/I, -H[2,1]/I, -H[2,2]/I, -gamma_rot
    return J


def ro_sync_solution(kappa: float, ell: float, gamma_rot: float, *,
                     omega: float = 1.0, delta: float = 0.0,
                     moments=None, inertia: float | None = None):
    """Exact centered RO synchronous solution and its stable torque branch."""
    mu = uniform_rod_moments(ell, 4) if moments is None else np.asarray(moments)
    I = ell*ell/12 if inertia is None else inertia
    w = omega/4
    mu4 = mu[4]
    amp = kappa*mu4
    if amp == 0 or I == 0:
        return {"exists": False, "reason": "zero rotational torque or inertia"}
    ratio = gamma_rot*I*w/(4*amp)
    if abs(ratio) > 1:
        return {"exists": False, "ratio": ratio,
                "gamma_rot_critical": 4*abs(amp)/(I*abs(w))}
    # Stable branch is selected by U_phi_phi>0, i.e. -16 amp cos(chi)>0.
    a = math.asin(float(np.clip(ratio, -1, 1)))
    candidates = [a, math.pi-a]
    chi = next(c for c in candidates if -amp*math.cos(c) > 0)
    phi = (chi-delta)/4
    return {"exists": True, "omega_frame": w, "phase_chi": chi,
            "phi": phi, "lab_spin": w,
            "gamma_rot_critical": 4*abs(amp)/(I*abs(w)),
            "rot_stiffness": -16*amp*math.cos(chi)/I,
            "inertia": I}


def ro_rotating_jacobian(kappa: float, ell: float, gamma_trans: float,
                         gamma_rot: float, phase_chi: float,
                         omega: float = 1.0, moments=None,
                         inertia: float | None = None):
    """Constant Jacobian in co-rotating coordinates at centered RO lock.

    State order is (ux,uy,ux',uy',dphi,dspin). The COM block includes both
    Coriolis and laboratory-frame drag terms.
    """
    mu = uniform_rod_moments(ell, 4) if moments is None else np.asarray(moments)
    w = omega/4
    b = 6*kappa*mu[2]
    # Rotate axes to put the quadratic COM Hessian on principal axes.
    # Its eigenvalues are (+2 b, -2 b), independent of the selected phase.
    J = np.zeros((6,6))
    J[0,2] = J[1,3] = J[4,5] = 1.0
    J[2:4,0:2] = np.diag([w*w-2*b, w*w+2*b]) - gamma_trans*w*np.array([[0,-1],[1,0]])
    J[2:4,2:4] = -gamma_trans*np.eye(2)-2*w*np.array([[0,-1],[1,0]])
    I = ell*ell/12 if inertia is None else inertia
    K = -16*kappa*mu[4]*math.cos(phase_chi)/I
    # K=-d(torque/I)/dphi is the positive spring constant, hence acceleration
    # contains -K*dphi.
    J[5,4] = -K
    J[5,5] = -gamma_rot
    return J


def direct_ro_monodromy(kappa: float, ell: float, gamma_trans: float,
                         gamma_rot: float, *, omega: float = 1.0,
                         delta: float = 0.0, rtol: float = 2e-12,
                         atol: float = 2e-14):
    """Integrate lab variational equations around the relative orbit.

    The physical rod orbit closes after T=4*pi/|omega| modulo its pi
    head-tail symmetry. This returned matrix is the lab-frame monodromy on
    that quotient period.
    """
    soln = ro_sync_solution(kappa, ell, gamma_rot, omega=omega, delta=delta)
    if not soln["exists"]:
        raise ValueError("no synchronous RO solution for these parameters")
    T = 4*math.pi/abs(omega)
    w, phi = soln["omega_frame"], soln["phi"]
    hs = [Harmonic(4, "RO", kappa, omega, delta)]
    I = ell*ell/12
    y0 = np.array([0.,0.,0.,0.,phi, w])
    def aug_rhs(t, aug):
        y = aug[:6]
        M = aug[6:].reshape(6,6)
        dy = rigid_rhs(t,y,hs,ell,gamma_trans,gamma_rot,I)
        dM = lab_jacobian(t,y,hs,ell,gamma_trans,gamma_rot,I) @ M
        return np.r_[dy,dM.ravel()]
    init = np.r_[y0,np.eye(6).ravel()]
    out = solve_ivp(aug_rhs,(0,T),init,method="DOP853",rtol=rtol,atol=atol,
                    max_step=T/1000)
    if not out.success:
        raise RuntimeError(out.message)
    M = out.y[6:,-1].reshape(6,6)
    return {"period": T, "monodromy": M, "multipliers": np.linalg.eigvals(M),
            "endpoint_residual": float(np.linalg.norm(out.y[:6,-1]-
              np.array([0.,0.,0.,0.,phi+w*T,w]),ord=np.inf)), "solution": soln}


def orbit_monodromy(initial_state, harmonics, ell: float, gamma_trans: float,
                    gamma_rot: float, period: float, inertia: float | None=None,
                    *, rtol: float=2e-12, atol: float=2e-14,
                    steps_per_period: int=1000):
    """Lab-frame 6D monodromy along a supplied periodic/relative-periodic orbit."""
    I=ell*ell/12 if inertia is None else inertia
    y0=np.asarray(initial_state,float)
    def aug(t,z):
        y=z[:6]; M=z[6:].reshape(6,6)
        return np.r_[rigid_rhs(t,y,harmonics,ell,gamma_trans,gamma_rot,I),
                     (lab_jacobian(t,y,harmonics,ell,gamma_trans,gamma_rot,I)@M).ravel()]
    init=np.r_[y0,np.eye(6).ravel()]
    sol=solve_ivp(aug,(0,period),init,method="DOP853",rtol=rtol,atol=atol,
                  max_step=period/steps_per_period)
    if not sol.success: raise RuntimeError(sol.message)
    M=sol.y[6:,-1].reshape(6,6)
    return {"monodromy":M,"multipliers":np.linalg.eigvals(M),
            "endpoint":sol.y[:6,-1],"integration_success":bool(sol.success)}

