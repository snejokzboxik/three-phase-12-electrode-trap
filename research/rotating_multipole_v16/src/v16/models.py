from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Field:
    """Dimensionless 2-D electrostatic drive. RF rates use tau units."""

    model: str = "RO"
    kappa: float = 0.0
    kappa_so: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    delta_ro: float = 0.0
    delta_so: float = 0.0
    phase_q: float = 0.0
    omega_ro: float = 1.0
    omega_so: float = 1.0
    omega_q: float = 1.0
    kappa8: float = 0.0
    omega8: float | None = None
    delta8: float = 0.0
    kappa10: float = 0.0
    omega10: float | None = None
    delta10: float = 0.0

    def __post_init__(self) -> None:
        name = self.model.upper()
        if name not in {"RO", "SO", "QRO", "QSO"}:
            raise ValueError(f"unknown field model: {self.model}")
        if self.omega_ro == 0 or self.omega_so == 0 or self.omega_q == 0:
            raise ValueError("drive frequencies must be nonzero")
        if self.omega8 is None:
            object.__setattr__(self, "omega8", 2.0*self.omega_ro)
        if self.omega8 == 0:
            raise ValueError("order-8 drive frequency must be nonzero")
        if self.omega10 is None:
            object.__setattr__(self, "omega10", 2.5*self.omega_ro)
        if self.omega10 == 0:
            raise ValueError("order-10 drive frequency must be nonzero")
        object.__setattr__(self, "model", name)

    def evaluate(self, x, y, tau: float, *, hessian: bool = False):
        """Return potential and gradient; optionally the symmetric Hessian.

        Scalar inputs return scalars and length-N arrays return length-N arrays.
        The rotating octupole is Re[z^4 exp(-i(omega_ro*tau-delta_ro))].
        """
        x, y = np.asarray(x), np.asarray(y)
        a = x**4 - 6.0*x*x*y*y + y**4
        b = 4.0*x*y*(x*x-y*y)
        ax = 4.0*x**3 - 12.0*x*y*y
        ay = 4.0*y**3 - 12.0*x*x*y
        bx = 12.0*x*x*y - 4.0*y**3
        by = 4.0*x**3 - 12.0*x*y*y
        val = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=float)
        gx = np.zeros_like(val)
        gy = np.zeros_like(val)
        hxx = np.zeros_like(val)
        hxy = np.zeros_like(val)
        hyy = np.zeros_like(val)

        if self.model in {"RO", "QRO"}:
            phase = self.omega_ro*tau - self.delta_ro
            c, s = np.cos(phase), np.sin(phase)
            val = val + self.kappa*(a*c + b*s)
            gx = gx + self.kappa*(ax*c + bx*s)
            gy = gy + self.kappa*(ay*c + by*s)
            hxx = hxx + self.kappa*(12*(x*x-y*y)*c + 24*x*y*s)
            hxy = hxy + self.kappa*(-24*x*y*c + 12*(x*x-y*y)*s)
            hyy = hyy + self.kappa*(12*(y*y-x*x)*c - 24*x*y*s)

        if self.model in {"SO", "QSO"}:
            phase = self.omega_so*tau - self.delta_so
            q = self.kappa_so*np.cos(phase)
            val = val + q*a
            gx = gx + q*ax
            gy = gy + q*ay
            hxx = hxx + q*12*(x*x-y*y)
            hxy = hxy - q*24*x*y
            hyy = hyy + q*12*(y*y-x*x)

        if self.model in {"QRO", "QSO"}:
            qamp = self.alpha + self.beta*np.cos(self.omega_q*tau + self.phase_q)
            qval = x*x-y*y
            val = val + qamp*qval
            gx = gx + 2*qamp*x
            gy = gy - 2*qamp*y
            hxx = hxx + 2*qamp
            hyy = hyy - 2*qamp

        for order, coefficient, omega, delta in (
            (8,self.kappa8,self.omega8,self.delta8),
            (10,self.kappa10,self.omega10,self.delta10),
        ):
            if coefficient == 0.0:
                continue
            z = x+1j*y
            phase = np.exp(-1j*(omega*tau-delta))
            zm2 = z**(order-2)*phase
            zm1 = z*zm2
            zm = z*zm1
            val = val+coefficient*np.real(zm)
            gx = gx+coefficient*order*np.real(zm1)
            gy = gy+coefficient*order*np.real(1j*zm1)
            hxx = hxx+coefficient*order*(order-1)*np.real(zm2)
            hxy = hxy+coefficient*order*(order-1)*np.real(1j*zm2)
            hyy = hyy-coefficient*order*(order-1)*np.real(zm2)

        result = (val, gx, gy)
        if hessian:
            return result + (hxx, hxy, hyy)
        return result


def field_rhs(tau: float, state, field: Field, damping: float = 0.0):
    """Lab-frame point-particle equations, y=(x,y,vx,vy)."""
    x, y, vx, vy = state
    _, gx, gy = field.evaluate(x, y, tau)
    return np.array([vx, vy, -gx-damping*vx, -gy-damping*vy], dtype=float)


def lab_energy(state, tau: float, field: Field) -> float:
    x, y, vx, vy = state
    v, _, _ = field.evaluate(x, y, tau)
    return 0.5*(vx*vx+vy*vy)+float(v)


def angular_momentum(state) -> float:
    x, y, vx, vy = state
    return float(x*vy-y*vx)


def rotating_invariant(state, tau: float, field: Field) -> float:
    """J=H-omega_field*Lz; conserved only for a pure rigidly rotating RO."""
    if field.model != "RO":
        raise ValueError("J is only asserted for the pure rotating octupole")
    for coefficient,omega,ratio,order in ((field.kappa8,field.omega8,2.0,8),
                                          (field.kappa10,field.omega10,2.5,10)):
        if coefficient != 0.0 and not math.isclose(omega,ratio*field.omega_ro,
                                                    rel_tol=0.0,abs_tol=1e-14):
            raise ValueError(f"order-{order} perturbation must co-rotate to preserve the RO invariant")
    return lab_energy(state, tau, field) - field.omega_ro/4.0*angular_momentum(state)


def rotating_rhs(tau: float, state, field: Field, damping: float = 0.0):
    """RO equations in a frame rotating at omega_ro/4.

    The lab drag acts on qdot+omega*Jq, not on qdot alone.
    State is q=(x,y), qdot=(u,v) in the rotating frame.
    """
    if field.model != "RO":
        raise ValueError("autonomous rotating frame is specific to pure RO")
    for coefficient,omega,ratio,order in ((field.kappa8,field.omega8,2.0,8),
                                          (field.kappa10,field.omega10,2.5,10)):
        if coefficient != 0.0 and not math.isclose(omega,ratio*field.omega_ro,
                                                    rel_tol=0.0,abs_tol=1e-14):
            raise ValueError(f"rotating-frame dynamics require a co-rotating order-{order} term")
    x, y, u, v = state
    w = field.omega_ro/4.0
    _,gx,gy = field.evaluate(x,y,0.0)
    # q'' = -grad(V) + w^2 q - 2w J q' - lambda(q'+w Jq)
    axr = -gx+w*w*x + (2*w+damping)*v + damping*w*y
    ayr = -gy+w*w*y - (2*w+damping)*u - damping*w*x
    return np.array([u, v, axr, ayr], dtype=float)


def rotating_hamiltonian(state, field: Field) -> float:
    """Autonomous K=J in rotating coordinates for a pure RO."""
    if field.model != "RO":
        raise ValueError("only pure RO has this autonomous Hamiltonian")
    for coefficient,omega,ratio,order in ((field.kappa8,field.omega8,2.0,8),
                                          (field.kappa10,field.omega10,2.5,10)):
        if coefficient != 0.0 and not math.isclose(omega,ratio*field.omega_ro,
                                                    rel_tol=0.0,abs_tol=1e-14):
            raise ValueError(f"autonomous Hamiltonian requires a co-rotating order-{order} term")
    x, y, u, v = state
    pot, _, _ = field.evaluate(x, y, 0.0)
    w = field.omega_ro/4.0
    return 0.5*(u*u+v*v)+float(pot)-0.5*w*w*(x*x+y*y)


def equilibrium_spectrum(kappa: float, delta: float = 0.0):
    """Analytic spectrum at one nonzero rotating-frame equilibrium.

    Returns its radius and eigenvalues for w=1/4. Every such equilibrium is
    a saddle-centre (one real pair and one imaginary pair).
    """
    if kappa == 0:
        raise ValueError("kappa=0 has no nonzero stationary point")
    r = 1.0/(8.0*math.sqrt(abs(kappa)))
    # Hessian eigenvalues along radial/tangential axes, including the
    # centrifugal term: +1/8 and -1/4. The Coriolis block couples them.
    a, b, w = 1.0/8.0, -1.0/4.0, 1.0/4.0
    roots2 = np.roots([1.0, a+b+4*w*w, a*b])
    eig = []
    for q in roots2:
        root = np.sqrt(complex(q))
        eig.extend([complex(root), complex(-root)])
    return {"radius": r, "delta": delta, "lambda": eig}

