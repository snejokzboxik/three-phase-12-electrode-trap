"""Shooting and full Floquet analysis for centered periodic rigid-rod locks."""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from .dynamics import Harmonic, lab_jacobian, potential_jet, rigid_rhs


def find_sync_orbits(harmonic: Harmonic, ell: float, gamma_rot: float,
                     *, gamma_trans: float = 0.0, inertia=None,
                     phase_seeds: int = 12, rtol: float = 3e-11):
    """Find centered, drive-periodic relative rotations for both directions.

    The relative orientation is phi=psi-(omega/order)*tau. Shooting is done
    over one RF period; returned roots are deduplicated modulo 2*pi/order.
    This does not assume that a standing-wave orbit has constant phase.
    """
    m, w_rf = harmonic.order, harmonic.omega
    Tdrive = 2*math.pi/abs(w_rf)
    wsync = w_rf/m
    I = ell*ell/12 if inertia is None else inertia
    if I <= 0:
        raise ValueError("positive inertia required")

    def shoot(q, direction):
        phi0, u0 = q
        w = direction*abs(wsync)
        def rhs(t, s):
            phi, u = s
            _, g, _ = potential_jet(0.,0.,phi+w*t,t,ell,[harmonic])
            return [u, -g[2]/I-gamma_rot*(u+w)]
        sol=solve_ivp(rhs,(0,Tdrive),[phi0,u0],method="DOP853",rtol=rtol,
                      atol=rtol*1e-2,max_step=Tdrive/100)
        return np.array([sol.y[0,-1]-phi0, sol.y[1,-1]-u0])

    found=[]
    for direction in (1,-1):
        roots=[]
        for seed in np.linspace(0,2*math.pi/m,phase_seeds,endpoint=False):
            out=root(lambda q:shoot(q,direction),[seed,0.],tol=1e-10)
            if not out.success or np.linalg.norm(shoot(out.x,direction),ord=np.inf)>2e-8:
                continue
            phase=out.x[0]%(2*math.pi/m)
            spin=direction*abs(wsync)+out.x[1]
            distance=lambda r: abs((phase-r[0]+math.pi/m)%(2*math.pi/m)-math.pi/m)
            if not any(distance(r)<2e-6 and abs(spin-r[1])<2e-7 for r in roots):
                roots.append((phase,spin))
        for phase,spin in roots:
            found.append({"direction":direction,"phase":float(phase),"spin":float(spin),
                          "rf_period":Tdrive,"inertia":I,
                          "shooting_residual":float(np.linalg.norm(shoot(
                             [phase,spin-direction*abs(wsync)],direction),ord=np.inf))})
    return found


def full_floquet_for_sync(harmonic: Harmonic, ell: float, gamma_trans: float,
                          gamma_rot: float, orbit, *, rtol: float=2e-11,
                          atol: float=2e-13):
    """Integrate the full six-state variational system over a physical period.

    A uniform rod is head-tail symmetric: after N RF periods, orientation
    must advance by an integer multiple of pi. The lab-frame state Jacobian
    then closes, even when the first RF-period orientation is not itself a
    fixed point (as for standing fields).
    """
    m=harmonic.order
    n_periods=m//math.gcd(m,2)
    T=orbit["rf_period"]*n_periods
    y0=np.array([0.,0.,0.,0.,orbit["phase"],orbit["spin"]])
    I=orbit["inertia"]
    def aug(t,z):
        y=z[:6]; M=z[6:].reshape(6,6)
        dy=rigid_rhs(t,y,[harmonic],ell,gamma_trans,gamma_rot,I)
        dM=lab_jacobian(t,y,[harmonic],ell,gamma_trans,gamma_rot,I)@M
        return np.r_[dy,dM.ravel()]
    init=np.r_[y0,np.eye(6).ravel()]
    sol=solve_ivp(aug,(0,T),init,method="DOP853",rtol=rtol,atol=atol,
                  max_step=min(orbit["rf_period"]/100,T/500))
    if not sol.success:
        raise RuntimeError(sol.message)
    M=sol.y[6:,-1].reshape(6,6)
    yend=sol.y[:6,-1]
    mean_spin=math.copysign(abs(harmonic.omega/harmonic.order),orbit["direction"])
    expected=y0.copy(); expected[4]+=mean_spin*T
    residual=float(np.linalg.norm(yend-expected,ord=np.inf))
    return {"period":T,"monodromy":M,"multipliers":np.linalg.eigvals(M),
            "closure_residual":residual,"endpoint":yend}

