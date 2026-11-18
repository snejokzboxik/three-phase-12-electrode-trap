"""Universal signed-charge planar multipole model for V17B.

Coordinates/moments are normalized to the trap length R.  The reference
charge q_ref=sum(abs(q_i)) keeps neutral distributions well-defined.  The
field coupling kappa_m already includes q_ref and voltage/geometry factors.
The exact polynomial jet and lab ODE are reused from the validated V16.1 core.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import comb
import math
import numpy as np
from scipy.integrate import solve_ivp
from v16_1.dynamics import Harmonic, potential_jet, rigid_rhs, lab_jacobian

@dataclass(frozen=True)
class ChargeMoments:
    q_ref: float
    moments: np.ndarray
    def __post_init__(self):
        if self.q_ref <= 0 or not np.isfinite(self.q_ref):
            raise ValueError("q_ref must be positive; choose sum(abs(q_i))")
        object.__setattr__(self, "moments", np.asarray(self.moments, dtype=complex))


def charge_moments(charges, positions, max_order: int, *, R: float = 1.0) -> ChargeMoments:
    """Complex moments from body-fixed charge sites relative to the center of mass.

    ``positions`` must be supplied relative to the mechanical COM; this module
    does not infer the COM from the charge distribution. ``q_ref=sum(abs(q))``
    keeps neutral charge profiles well-defined.
    """
    q = np.asarray(charges, dtype=float)
    s = np.asarray(positions, dtype=complex)
    if q.ndim != 1 or s.shape != q.shape or not len(q):
        raise ValueError("charges and positions must be nonempty 1D arrays of equal length")
    if R <= 0 or max_order < 0:
        raise ValueError("R must be positive and max_order nonnegative")
    qref = float(np.sum(np.abs(q)))
    if qref == 0:
        raise ValueError("all charges are zero")
    u = s / R
    out = np.array([np.sum((q/qref) * u**j) for j in range(max_order+1)], dtype=complex)
    return ChargeMoments(qref, out)


def expanded_polynomial(Z: complex, psi: float, order: int, moments) -> complex:
    """P_m=< (Z+s exp(i psi))^m > using signed moments C_j."""
    c = np.asarray(moments, dtype=complex)
    return sum(comb(order,j) * (c[j] if j < len(c) else 0j) *
               Z**(order-j) * np.exp(1j*j*psi) for j in range(order+1))


def direct_polynomial(Z: complex, psi: float, order: int, charges, positions, *, R=1.0) -> complex:
    """Independent point-charge sum; normalized by sum(abs(q_i))."""
    q=np.asarray(charges,float); s=np.asarray(positions,complex)/R
    return np.sum((q/np.sum(np.abs(q))) * (Z+s*np.exp(1j*psi))**order)


def synchronous_branch(order: int, kappa: float, omega: float, moments,
                       gamma_rot: float, inertia: float, *, delta: float=0.0):
    """Centered exact RO lock if C_(m-1)=0; choose stable phase branch."""
    c=np.asarray(moments,dtype=complex); m=order; w=omega/m
    cm=c[m] if m < len(c) else 0j
    cprev=c[m-1] if m-1 < len(c) else 0j
    if abs(cprev)>1e-12:
        return {"exists":False,"reason":"nonzero C_(m-1) gives nonzero force at Z=0"}
    amp=kappa*cm
    # Put complex C_m phase into the body phase variable.
    if abs(amp)==0 or inertia<=0:
        return {"exists":False,"reason":"zero orientation torque or inertia"}
    theta=np.angle(amp)+delta
    ratio=gamma_rot*inertia*w/(m*abs(amp))
    if abs(ratio)>1:
        return {"exists":False,"reason":"rotational torque below drag", "gamma_rot_critical":m*abs(amp)/(inertia*abs(w))}
    a=math.asin(float(np.clip(ratio,-1,1)))
    # phase chi=m*phi+theta. The minimum has cos(chi+arg amp)=-sign handling.
    chi0=math.pi-a if amp.real >= 0 and abs(amp.imag)<1e-12 else None
    # Robust phase: torque m|amp|sin(chi+arg amp)=gamma I w, stable cos(...)<0.
    phase_eff=math.pi-a
    phi=(phase_eff-theta)/m
    K=-m*m*abs(amp)*math.cos(phase_eff)/inertia
    return {"exists":True,"omega_frame":w,"phase":phi,"phase_eff":phase_eff,
            "lab_spin":w,"gamma_rot_critical":m*abs(amp)/(inertia*abs(w)),
            "rot_stiffness":K,"inertia":inertia,"c_m_abs":abs(cm),"_unused":chi0}


def rotating_jacobian(order: int, kappa: float, omega: float, moments,
                      gamma_trans: float, gamma_rot: float, inertia: float,
                      phase: float, *, delta: float=0.0):
    """Constant 6D Jacobian in co-rotating coordinates at a centered RO lock.

    Requires C_(m-1)=0. State order is (u_x,u_y,u'_x,u'_y,dpsi,dspin).
    """
    c=np.asarray(moments,complex); m=order; w=omega/m
    if abs(c[m-1] if m-1<len(c) else 0)>1e-12:
        raise ValueError("centered lock requires C_(m-1)=0")
    cquad=c[m-2] if m>=2 and m-2<len(c) else 0j
    b=comb(m,2)*kappa*cquad*np.exp(1j*((m-2)*phase+delta))
    # Re(b u^2) has Cartesian Hessian with eigenvalues +/-2|b|.
    H=np.array([[2*b.real,-2*b.imag],[-2*b.imag,-2*b.real]],float)
    J=np.zeros((6,6)); J[0,2]=J[1,3]=J[4,5]=1
    rot=np.array([[0.,-1.],[1.,0.]])
    J[2:4,0:2]=w*w*np.eye(2)-H-gamma_trans*w*rot
    J[2:4,2:4]=-gamma_trans*np.eye(2)-2*w*rot
    cm=c[m] if m<len(c) else 0j
    amp=kappa*cm*np.exp(1j*delta)
    chi=m*phase+np.angle(amp)
    K=-m*m*abs(amp)*math.cos(chi)/inertia
    J[5,4]=-K; J[5,5]=-gamma_rot
    return J, {"b_complex":b,"b_abs":abs(b),"rot_stiffness":K,"omega_frame":w}


def lab_monodromy(order: int, kappa: float, omega: float, moments,
                  gamma_trans: float, gamma_rot: float, inertia: float,
                  phase: float, *, delta: float=0.0, rtol=2e-12, atol=2e-14):
    """Independent DOP853 variational monodromy over m RF periods.

    m drive periods return a generic polar body orientation by 2pi; this also
    restores the linearized coefficients. No head-tail quotient is assumed.
    """
    m=order; w=omega/m; period=2*math.pi*m/abs(omega)
    hs=[Harmonic(m,"RO",kappa,omega,delta)]
    y0=np.array([0.,0.,0.,0.,phase,w])
    def aug(t, z):
        y=z[:6]; M=z[6:].reshape(6,6)
        return np.r_[rigid_rhs(t,y,hs,1.0,gamma_trans,gamma_rot,inertia,moments),
                     (lab_jacobian(t,y,hs,1.0,gamma_trans,gamma_rot,inertia,moments)@M).ravel()]
    sol=solve_ivp(aug,(0.,period),np.r_[y0,np.eye(6).ravel()],method="DOP853",
                  rtol=rtol,atol=atol,max_step=period/2000)
    if not sol.success: raise RuntimeError(sol.message)
    M=sol.y[6:,-1].reshape(6,6)
    endpoint=sol.y[:6,-1]
    expected=y0.copy(); expected[4]=phase+w*period
    return {"period":period,"monodromy":M,"multipliers":np.linalg.eigvals(M),
            "endpoint":endpoint,"endpoint_residual":float(np.linalg.norm(endpoint-expected,np.inf))}
