from __future__ import annotations

import math
import numpy as np
from scipy.integrate import solve_ivp
from .models import Field, field_rhs


def rk4_step(fun, t: float, y, h: float, *args):
    k1 = fun(t, y, *args)
    k2 = fun(t+h/2, y+h*k1/2, *args)
    k3 = fun(t+h/2, y+h*k2/2, *args)
    k4 = fun(t+h, y+h*k3, *args)
    return y+h*(k1+2*k2+2*k3+k4)/6


def integrate_dop853(initial, field: Field, *, rf_count: float,
                     damping: float = 0.0, rtol: float = 2e-11,
                     atol: float = 2e-13, aperture: float = 1.0,
                     sample_step: float | None = None):
    """High-accuracy reference integration with an interpolated exit event."""
    y0 = np.asarray(initial, dtype=float)
    def event(t, y):
        return aperture*aperture-y[0]*y[0]-y[1]*y[1]
    event.terminal = True
    event.direction = -1
    # Time is tau=Omega*t. rf_count is measured against the reference period
    # 2*pi; detuned drives do not imply a common Poincare period.
    period = 2*math.pi
    t1 = rf_count*period
    if sample_step is None:
        max_step = period/80
    else:
        max_step = sample_step
    sol = solve_ivp(lambda t,y: field_rhs(t,y,field,damping), (0,t1), y0,
                    method="DOP853", rtol=rtol, atol=atol,
                    events=event, dense_output=True, max_step=max_step)
    return sol


def integrate_ensemble(initial_states, field: Field, *, rf_count: int,
                       steps_per_rf: int = 128, damping: float = 0.0,
                       aperture: float = 1.0, initial_phase=None):
    """Vectorized fixed-step RK4 scout with first-exit detection.

    `initial_states` has shape (N,4). Escape times use chord interpolation and
    are intended for screening, not precision boundary measurements.
    Bounded-at-horizon is a finite observation, never a stability proof.
    """
    states = np.array(initial_states, dtype=float, copy=True)
    n = states.shape[0]
    phase0 = np.zeros(n) if initial_phase is None else np.asarray(initial_phase,dtype=float)
    if phase0.shape != (n,):
        raise ValueError("initial_phase must have one value per initial state")
    exit_t = np.full(n, np.nan)
    active = np.ones(n, dtype=bool)
    r0 = np.hypot(states[:,0], states[:,1])
    already = r0 >= aperture
    exit_t[already] = 0.0
    active[already] = False
    max_r = r0.copy()
    # tau-based horizon is independent of the field frequencies. In particular,
    # detuned drives must not be silently rationalized into a Poincare period.
    period = 2*math.pi
    h = period/steps_per_rf
    total_steps = int(rf_count*steps_per_rf)
    y = states
    for it in range(total_steps):
        ids = np.flatnonzero(active)
        if ids.size == 0:
            break
        old = y[ids]
        t = phase0[ids]+it*h
        def f(tt, s):
            x,yc,vx,vy = s.T
            _,gx,gy = field.evaluate(x,yc,tt)
            return np.stack((vx,vy,-gx-damping*vx,-gy-damping*vy),axis=1)
        k1=f(t,old); k2=f(t+h/2,old+h*k1/2)
        k3=f(t+h/2,old+h*k2/2); k4=f(t+h,old+h*k3)
        new=old+h*(k1+2*k2+2*k3+k4)/6
        rr=np.hypot(new[:,0],new[:,1]); max_r[ids]=np.maximum(max_r[ids],rr)
        crossed=rr>=aperture
        if np.any(crossed):
            oldr=np.hypot(old[crossed,0],old[crossed,1])
            frac=np.clip((aperture-oldr)/(rr[crossed]-oldr+1e-300),0,1)
            exit_t[ids[crossed]]=(it+frac)*h
            active[ids[crossed]]=False
        y[ids[~crossed]]=new[~crossed]
    return {"state":y, "exit_tau":exit_t, "max_radius":max_r,
            "status":np.where(np.isfinite(exit_t),"EXIT","BOUNDED_HORIZON")}


def largest_lyapunov(initial, field: Field, *, rf_count: int,
                     damping: float = 0.0, rtol: float = 2e-10,
                     atol: float = 2e-12, aperture: float = 1.0):
    """QR-renormalized largest finite-time Lyapunov exponent per tau unit."""
    state=np.asarray(initial,dtype=float).copy()
    Q=np.eye(4)
    log_growth=np.zeros(4); cumulative=[]
    T=2*math.pi
    def aug_rhs(t,z):
        y=z[:4]; mat=z[4:].reshape(4,4)
        f=field_rhs(t,y,field,damping)
        _,_,_,hxx,hxy,hyy=field.evaluate(y[0],y[1],t,hessian=True)
        A=np.array([[0,0,1,0],[0,0,0,1],[-hxx,-hxy,-damping,0],[-hxy,-hyy,0,-damping]],float)
        return np.r_[f,(A@mat).ravel()]
    points=[]
    for n in range(int(rf_count)):
        z0=np.r_[state,Q.ravel()]
        sol=solve_ivp(aug_rhs,(n*T,(n+1)*T),z0,method="DOP853",rtol=rtol,atol=atol,max_step=T/60)
        z=sol.y[:,-1]; state=z[:4]
        if state[0]**2+state[1]**2>=aperture**2:
            return {"status":"EXIT","rf":n+1,"partial_sums":np.asarray(cumulative),"state":state}
        Q,R=np.linalg.qr(z[4:].reshape(4,4))
        d=np.maximum(np.abs(np.diag(R)),1e-300)
        log_growth+=np.log(d)
        exponents=log_growth/(T*(n+1))
        cumulative.append(float(np.max(exponents)))
        points.append(state.copy())
    spectrum=log_growth/(T*int(rf_count))
    return {"status":"BOUNDED_HORIZON","state":state,"exponent":float(np.max(spectrum)),
            "spectrum":spectrum,"cumulative":np.asarray(cumulative),"poincare":np.asarray(points)}
