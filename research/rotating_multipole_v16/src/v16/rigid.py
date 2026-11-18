from __future__ import annotations

import math
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from .models import Field


@lru_cache(maxsize=128)
def rod_quadrature(length_ratio: float, kind: str = "uniform", nodes: int = 24):
    """Body-axis charge nodes, with signed weights normalized to total charge 1."""
    g,w=leggauss(nodes)
    s=0.5*length_ratio*g
    w=0.5*w
    k=kind.lower()
    if k=="uniform":
        q=np.ones_like(s)
    elif k=="symmetric_quad":
        q=1.0+0.8*(g*g-1/3)
    elif k=="dipolar":
        q=1.0+0.7*g
    elif k=="quadrupolar":
        q=1.0+0.8*(g*g-1/3)
    else:
        raise ValueError(kind)
    weights=w*q
    weights=weights/np.sum(weights)
    return s,weights


def rod_potential_force_torque(x: float,y: float,psi: float,tau: float,
                               field: Field,length_ratio: float,
                               kind: str="uniform",nodes: int=24):
    """Potential, center force and body torque per total charge/mass scale."""
    s,q=rod_quadrature(length_ratio,kind,nodes)
    c,ss=math.cos(psi),math.sin(psi)
    xx=x+s*c; yy=y+s*ss
    U,gx,gy=field.evaluate(xx,yy,tau)
    fx=-float(np.dot(q,gx)); fy=-float(np.dot(q,gy))
    torque=-float(np.dot(q,s*(-ss*gx+c*gy)))
    return float(np.dot(q,U)),fx,fy,torque


def rod_rhs(tau: float,state,field: Field,*,length_ratio: float,
            charge_model: str="uniform",lambda_translation: float=0.0,
            lambda_rotation: float=0.0,nodes: int=24):
    x,y,vx,vy,psi,spin=state
    _,fx,fy,mz=rod_potential_force_torque(x,y,psi,tau,field,length_ratio,
                                          charge_model,nodes)
    Ibar=length_ratio*length_ratio/12.0
    return np.array([vx,vy,fx-lambda_translation*vx,fy-lambda_translation*vy,
                     spin,mz/Ibar-lambda_rotation*spin],float)


def rod_energy_and_J(state,tau: float,field: Field,*,length_ratio: float,
                     charge_model: str="uniform",nodes: int=24):
    x,y,vx,vy,psi,spin=state
    U,_,_,_=rod_potential_force_torque(x,y,psi,tau,field,length_ratio,
                                      charge_model,nodes)
    Ibar=length_ratio*length_ratio/12.0
    H=0.5*(vx*vx+vy*vy)+0.5*Ibar*spin*spin+U
    L=x*vy-y*vx+Ibar*spin
    return float(H),float(H-field.omega_ro/4.0*L)


def symmetric_rod_lock_prediction(kappa: float,length_ratio: float,
                                  damping: float,omega_ro: float=1.0):
    """Exact center-fixed, uniform-rod rotational-locking prediction.

    For a uniform line charge centered at the origin in a pure RO field,
    U= kappa*L^4/80*cos(4*phi), phi=psi-omega_ro*tau/4. The rod inertia is
    L^2/12 in the chosen mass normalization. The lab-frame drag acts on the
    absolute spin, including the frame rate.
    """
    L=length_ratio; I=L*L/12; w=omega_ro/4
    amplitude=kappa*L**4/80
    torque_max=4*abs(amplitude)
    required=damping*I*w
    can_lock=(damping>0 and torque_max>=required and torque_max>0)
    threshold=(2.4*abs(kappa)*L*L) if kappa!=0 else 0.0
    if not can_lock:
        return {"can_lock":False,"threshold_lambda":threshold,
                "torque_max":torque_max,"required_drag_torque":required,
                "inertia":I,"field_rate":w}
    ratio=np.clip(required/(4*amplitude),-1.0,1.0)
    # Stable branch has d(torque)/d(phi)<0. For signed kappa, select the
    # phase with the appropriate sine sign and negative torque slope.
    candidates=[]
    for n in range(4):
        base=(math.asin(ratio) if kappa>0 else math.asin(ratio))
        for angle in (base,math.pi-base):
            phi=(angle+2*math.pi*n)/4
            slope=16*amplitude*math.cos(4*phi)/I
            if slope<0:
                candidates.append((phi,slope))
    phi,slope=candidates[0]
    eig=np.roots([1.0,damping,-slope])
    period=4*math.pi/abs(omega_ro)  # rod's head-tail orientation repeats after 2 RF cycles
    multipliers=np.exp(eig*period)
    return {"can_lock":True,"stable_phase":float(phi),"threshold_lambda":threshold,
            "torque_max":torque_max,"required_drag_torque":required,"inertia":I,
            "field_rate":w,"linear_eigenvalues":[complex(x) for x in eig],
            "two_rf_multipliers":[complex(x) for x in multipliers]}


def centered_rod_averaged_potential(psi: float,tau: float,kappa: float,
                                    length_ratio: float,delta: float=0.0):
    """Closed-form uniform-rod potential for the pure rotating octupole."""
    return kappa*length_ratio**4/80*math.cos(4*psi-tau+delta)
