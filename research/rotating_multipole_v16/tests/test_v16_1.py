import math
import numpy as np

from v16_1.dynamics import (Harmonic, direct_ro_monodromy, lab_jacobian,
                            potential_jet, rigid_rhs, ro_rotating_jacobian,
                            ro_sync_solution, uniform_rod_moments)
from v16_1.ro_parameter_maps import profile_moments, spectra


class D2:
    """Tiny independent second-order forward-mode AD scalar for validation."""
    def __init__(self, v, g=None, h=None):
        self.v = v
        self.g = np.zeros(3, dtype=complex) if g is None else np.asarray(g, complex)
        self.h = np.zeros((3,3), dtype=complex) if h is None else np.asarray(h, complex)
    @staticmethod
    def var(v, i):
        g=np.zeros(3,complex); g[i]=1
        return D2(v,g)
    def __add__(a,b):
        b=b if isinstance(b,D2) else D2(b)
        return D2(a.v+b.v,a.g+b.g,a.h+b.h)
    __radd__=__add__
    def __neg__(a): return D2(-a.v,-a.g,-a.h)
    def __sub__(a,b): return a+-b if isinstance(b,D2) else a+(-b)
    def __mul__(a,b):
        b=b if isinstance(b,D2) else D2(b)
        return D2(a.v*b.v,a.g*b.v+b.g*a.v,
                  a.h*b.v+b.h*a.v+np.outer(a.g,b.g)+np.outer(b.g,a.g))
    __rmul__=__mul__
    def __pow__(a,n):
        out=D2(1)
        for _ in range(n): out=out*a
        return out


def ad_exp(a):
    v=np.exp(a.v)
    return D2(v,v*a.g,v*(a.h+np.outer(a.g,a.g)))


def test_multipole_energy_gradient_and_hessian_match_independent_ad():
    ell=.37; tau=.63; x,y,psi=.17,-.23,.42
    positions=np.array([-.5*ell,-.19*ell,.19*ell,.5*ell])
    weights=np.array([.17,.33,.33,.17])
    mu=np.array([np.sum(weights*positions**j) for j in range(5)])
    hs=[Harmonic(4,"RO",.31,.8,.2),Harmonic(2,"SO",-.17,1.3,-.1)]
    U,g,H=potential_jet(x,y,psi,tau,ell,hs,mu)
    X,Y,P=map(D2.var,(x,y,psi),(0,1,2))
    z=X+1j*Y; ep=ad_exp(1j*P)
    ud=D2(0)
    for h in hs:
        poly=D2(0)
        for s,w in zip(positions,weights): poly=poly+w*(z+s*ep)**h.order
        factor=(h.kappa*np.exp(-1j*(h.omega*tau-h.delta)) if h.kind=="RO"
                else h.kappa*np.cos(h.omega*tau-h.delta))
        ud=ud+factor*poly
    assert abs(U-ud.v.real)<2e-13
    assert np.allclose(g,ud.g.real,rtol=2e-12,atol=2e-13)
    assert np.allclose(H,ud.h.real,rtol=3e-12,atol=4e-13)


def test_uniform_rod_moments_and_ro_com_characteristic_limit():
    ell=.2; mu=uniform_rod_moments(ell,4)
    assert math.isclose(mu[2],ell**2/12,rel_tol=1e-15)
    assert math.isclose(mu[4],ell**4/80,rel_tol=1e-15)
    # In the undamped rotating frame, det(sI-A_COM)=(s^2+w^2)^2-4b^2.
    k=.1044; w=.25; b=6*k*mu[2]
    A=ro_rotating_jacobian(k,ell,0.,.002,math.pi)
    p=np.poly(A[:4,:4])
    expected=np.array([1.,0.,2*w*w,0.,w**4-4*b*b])
    assert np.allclose(p,expected,rtol=2e-10,atol=1e-13)


def test_full_ro_jacobian_and_lab_monodromy_match_rotating_frame():
    k=.1044; ell=.2; gt=.05; gr=.004
    locked=ro_sync_solution(k,ell,gr)
    assert locked["exists"]
    A=ro_rotating_jacobian(k,ell,gt,gr,locked["phase_chi"])
    mon=direct_ro_monodromy(k,ell,gt,gr)
    assert mon["endpoint_residual"]<2e-11
    T=mon["period"]
    # The center-frame basis rotates by pi in one physical rod period, so
    # the four translational multipliers acquire a minus sign in the lab.
    expected=np.r_[-np.exp(np.linalg.eigvals(A[:4,:4])*T),
                    np.exp(np.linalg.eigvals(A[4:,4:])*T)]
    assert np.allclose(np.sort_complex(mon["multipliers"]),
                       np.sort_complex(expected),rtol=2e-8,atol=3e-9)


def test_rotational_and_translational_stability_are_independent_blocks():
    k=.1044; ell=.2
    gr=.004; gt=.05
    locked=ro_sync_solution(k,ell,gr)
    A=ro_rotating_jacobian(k,ell,gt,gr,locked["phase_chi"])
    assert np.max(np.linalg.eigvals(A[:4,:4]).real)<0
    assert np.max(np.linalg.eigvals(A[4:,4:]).real)<0
    # Exceeding only the COM threshold destabilizes translation while the
    # angular lock remains stable; exceeding the torque threshold removes it.
    Aout=ro_rotating_jacobian(k,ell,0.,gr,locked["phase_chi"])
    assert np.max(np.linalg.eigvals(Aout[:4,:4]).real)>0
    assert np.max(np.linalg.eigvals(Aout[4:,4:]).real)<0
    assert not ro_sync_solution(k,ell,gr*4)["exists"]


def test_exact_symmetric_rod_has_no_linear_com_angle_cross_terms():
    hs=[Harmonic(4,"RO",.1044)]
    ell=.2; locked=ro_sync_solution(.1044,ell,.004)
    t=.73; psi=locked["phi"]+.25*t
    y=np.array([0.,0.,0.,0.,psi,.25])
    J=lab_jacobian(t,y,hs,ell,.05,.004)
    assert np.max(np.abs(J[np.ix_([2,3,5],[4,0,1])][0:2,0])) == 0.0
    assert np.allclose(J[2:4,4],0,atol=1e-14)
    assert np.allclose(J[5,:2],0,atol=1e-14)


def test_general_six_state_lab_jacobian_matches_independent_finite_difference():
    hs=[Harmonic(2,"RO",.07,.9,.2),Harmonic(4,"SO",-.13,1.1,-.1)]
    ell=.23; y=np.array([.11,-.08,.03,-.04,.37,.16]); t=.81; eps=2e-7
    J=lab_jacobian(t,y,hs,ell,.031,.017)
    Jfd=np.empty((6,6))
    for j in range(6):
        d=np.zeros(6); d[j]=eps
        Jfd[:,j]=(rigid_rhs(t,y+d,hs,ell,.031,.017)-
                  rigid_rhs(t,y-d,hs,ell,.031,.017))/(2*eps)
    assert np.allclose(J,Jfd,rtol=1e-7,atol=2e-8)


def test_symmetric_charge_profile_moments():
    ell=.3; a=.8
    mu2,mu4=profile_moments(ell,a)
    assert np.isclose(mu2,ell**2/12*(1+4*a/15),rtol=1e-14)
    assert np.isclose(mu4,ell**4/80*(1+8*a/21),rtol=1e-14)


def test_ro_spectrum_invariant_to_drive_phase_and_direction():
    base=np.sort_complex(spectra(.8,.3,.2,.05,0.,1,0.)[0])
    for direction in (-1,1):
        for delta in (0.,math.pi/8,math.pi/4,math.pi/2):
            eig=np.sort_complex(spectra(.8,.3,.2,.05,0.,direction,delta)[0])
            assert np.max(np.abs(eig-base)) < 1e-14

