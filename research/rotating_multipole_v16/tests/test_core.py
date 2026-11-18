import math
import numpy as np
from scipy.integrate import solve_ivp

from v16.models import (Field, field_rhs, lab_energy, rotating_invariant,
                        rotating_rhs, rotating_hamiltonian)
from v16.integrate import integrate_dop853, integrate_ensemble
from v16.integrate import largest_lyapunov
from v16.rigid import (rod_potential_force_torque, rod_rhs, rod_energy_and_J,
                       symmetric_rod_lock_prediction, centered_rod_averaged_potential)


def _finite_difference(fun, q, eps=1e-6):
    return np.array([(fun(q+np.eye(len(q))[i]*eps)-fun(q-np.eye(len(q))[i]*eps))/(2*eps)
                     for i in range(len(q))])


def test_analytic_gradients_and_hessians_all_models():
    models=[Field("RO",kappa=.37,delta_ro=.4,omega_ro=-1),
            Field("RO",kappa=.37,delta_ro=.4,omega_ro=-1,kappa8=.008,delta8=.13,
                  kappa10=.003,delta10=.21),
            Field("SO",kappa_so=-.2,delta_so=.17),
            Field("QRO",kappa=.11,alpha=.3,beta=-.09,phase_q=.2),
            Field("QSO",kappa_so=.08,alpha=-.1,beta=.13)]
    q=np.array([.31,-.27]); tau=.83; eps=1e-6
    for f in models:
        v,gx,gy,hxx,hxy,hyy=f.evaluate(*q,tau,hessian=True)
        g=np.array([gx,gy])
        gn=_finite_difference(lambda z:f.evaluate(*z,tau)[0],q,eps)
        assert np.allclose(g,gn,rtol=2e-7,atol=2e-8)
        H=np.array([[hxx,hxy],[hxy,hyy]])
        # Gradient finite differences at q, with a direct Hessian check.
        for i in range(2):
            qp=q.copy(); qm=q.copy(); qp[i]+=eps; qm[i]-=eps
            gp=np.array(f.evaluate(*qp,tau)[1:]); gm=np.array(f.evaluate(*qm,tau)[1:])
            assert np.allclose(H[:,i],(gp-gm)/(2*eps),rtol=3e-6,atol=2e-7)


def test_fourfold_symmetry_and_limits():
    f=Field("RO",kappa=.23,delta_ro=.6)
    v=f.evaluate(.23,-.14,.77)[0]
    assert math.isclose(v,f.evaluate(.14,.23,.77)[0],rel_tol=1e-13,abs_tol=1e-13)
    assert Field("RO",kappa=0).evaluate(.4,-.2,1.3)[0] == 0
    q=Field("QRO",kappa=0,alpha=.2,beta=.1)
    expected=(.2+.1*math.cos(.7))*(.4**2-.2**2)
    assert math.isclose(q.evaluate(.4,.2,.7)[0],expected,rel_tol=1e-13)
    g=Field("QSO",kappa_so=0,alpha=.2,beta=.1)
    assert math.isclose(q.evaluate(.4,.2,.7)[0],g.evaluate(.4,.2,.7)[0],rel_tol=1e-13)


def test_conservative_ro_invariant_and_rotating_frame_agree():
    f=Field("RO",kappa=.1044,delta_ro=.37)
    y0=np.array([.09,.03,2e-4,-1e-4]); w=f.omega_ro/4
    sol=integrate_dop853(y0,f,rf_count=80,rtol=2e-12,atol=2e-14)
    tt=np.linspace(0,sol.t[-1],801); traj=sol.sol(tt).T
    Js=np.array([rotating_invariant(y,t,f) for y,t in zip(traj,tt)])
    assert np.ptp(Js)<2e-9
    q0=np.array([y0[0],y0[1],y0[2]+w*y0[1],y0[3]-w*y0[0]])
    rot=solve_ivp(lambda t,z:rotating_rhs(t,z,f),(0,sol.t[-1]),q0,method="DOP853",
                  rtol=2e-12,atol=2e-14,max_step=2*math.pi/100)
    lab_end=traj[-1]
    ang=-w*sol.t[-1]; c,s=math.cos(ang),math.sin(ang)
    R=np.array([[c,-s],[s,c]])
    q_lab=np.r_[R@lab_end[:2],R@lab_end[2:]-w*np.array([-(R@lab_end[:2])[1],(R@lab_end[:2])[0]])]
    assert np.allclose(rot.y[:,-1],q_lab,rtol=2e-8,atol=2e-9)
    kh=np.array([rotating_hamiltonian(z,f) for z in rot.y.T])
    assert np.ptp(kh)<2e-9


def test_rotating_invariant_sign_reversal():
    f1=Field("RO",kappa=.03,omega_ro=1)
    f2=Field("RO",kappa=.03,omega_ro=-1)
    y=np.array([.2,.1,.03,-.02])
    assert math.isfinite(rotating_invariant(y,.37,f1))
    assert rotating_invariant(y,.37,f1)!=rotating_invariant(y,.37,f2)


def test_batch_integrator_matches_reference_and_phase_sampling():
    f=Field("RO",kappa=.01)
    starts=np.array([[.2,.0,0,.001],[.3,.0,0,.0]])
    batch=integrate_ensemble(starts,f,rf_count=3,steps_per_rf=512,initial_phase=[0,.4])
    for i in range(2):
        ref=integrate_dop853(starts[i],Field("RO",kappa=.01,delta_ro=-[0,.4][i]),rf_count=3)
        if np.isnan(batch["exit_tau"][i]):
            assert np.allclose(batch["state"][i],ref.y[:,-1],rtol=5e-5,atol=2e-6)
        else:
            assert batch["exit_tau"][i] >= 0


def test_rod_force_torque_energy_gradient_and_ro_invariant():
    f=Field("RO",kappa=.5)
    q=np.array([.03,-.02,.41]); t=.72; L=.2; eps=1e-6
    U,fx,fy,tau=rod_potential_force_torque(*q[:2],q[2],t,f,L,"uniform",32)
    def potential(z): return rod_potential_force_torque(z[0],z[1],z[2],t,f,L,"uniform",32)[0]
    grad=_finite_difference(potential,q,eps)
    assert np.allclose(np.array([-fx,-fy,-tau]),grad,rtol=2e-6,atol=3e-8)
    y0=np.array([0.,0.,0.,0.,.3,.1])
    sol=solve_ivp(lambda t,z:rod_rhs(t,z,f,length_ratio=L), (0,20*2*math.pi),y0,
                  method="DOP853",rtol=2e-11,atol=2e-13,max_step=2*math.pi/80,dense_output=True)
    ts=np.linspace(0,sol.t[-1],201)
    Js=[rod_energy_and_J(sol.sol(t),t,f,length_ratio=L)[1] for t in ts]
    assert np.ptp(Js)<2e-9


def test_stationary_point_is_saddle_centre():
    from v16.models import equilibrium_spectrum
    for k in [.015,.1044,-.2]:
        out=equilibrium_spectrum(k)
        assert math.isclose(out["radius"],1/(8*math.sqrt(abs(k))))
        lam=np.asarray(out["lambda"])
        assert np.any(np.abs(lam.real)>.3)
        assert np.any(np.abs(lam.imag)>.49)


def test_rod_fourth_moment_lock_threshold_and_floquet_multipliers():
    k=.1044; L=.2; Omega=2*math.pi*100_000
    threshold=Omega*2.4*abs(k)*L*L
    locked=symmetric_rod_lock_prediction(k,L,3000/Omega)
    unlocked=symmetric_rod_lock_prediction(k,L,10000/Omega)
    assert math.isclose(threshold,6297.259642267669,rel_tol=1e-12)
    assert locked["can_lock"] and not unlocked["can_lock"]
    assert max(abs(z) for z in locked["two_rf_multipliers"])<1
    phi=locked["stable_phase"]
    assert math.cos(4*phi)<0
    assert math.isclose(4*k*L**4/80*math.sin(4*phi),
                        (3000/Omega)*(L*L/12)/4,rel_tol=1e-12)
    psi=.43; tau=.8
    U,fx,fy,mz=rod_potential_force_torque(0,0,psi,tau,Field("RO",kappa=k),L)
    Ua=centered_rod_averaged_potential(psi,tau,k,L)
    expected=4*k*L**4/80*math.sin(4*psi-tau)
    assert math.isclose(U,Ua,rel_tol=1e-12,abs_tol=1e-14)
    assert math.hypot(fx,fy)<1e-14
    assert math.isclose(mz,expected,rel_tol=2e-9,abs_tol=1e-14)


def test_qr_lyapunov_recovers_known_saddle_exponent():
    r=1/(8*math.sqrt(.1044)); f=Field("RO",kappa=.1044)
    # Lab-frame state corresponding to the rotating-frame saddle-centre.
    le=largest_lyapunov([r,0.,0.,.25*r],f,rf_count=10)
    assert le["status"]=="BOUNDED_HORIZON"
    assert abs(le["exponent"]-1/math.sqrt(8))<.02


def test_corotating_order8_harmonic_preserves_ro_invariant_and_frame():
    f=Field("RO",kappa=.1044,kappa8=.006,omega_ro=-1,delta_ro=.23,delta8=-.17,
            kappa10=.002,delta10=.37)
    y0=np.array([.08,.025,1e-4,-2e-4]); w=f.omega_ro/4
    sol=solve_ivp(lambda t,y:field_rhs(t,y,f),(0,20*2*math.pi),y0,
                  method="DOP853",rtol=2e-12,atol=2e-14,
                  max_step=2*math.pi/100,dense_output=True)
    ts=np.linspace(0,sol.t[-1],201)
    Js=[rotating_invariant(sol.sol(t),t,f) for t in ts]
    assert np.ptp(Js)<2e-9
    q0=np.array([y0[0],y0[1],y0[2]+w*y0[1],y0[3]-w*y0[0]])
    rot=solve_ivp(lambda t,z:rotating_rhs(t,z,f),(0,sol.t[-1]),q0,
                  method="DOP853",rtol=2e-12,atol=2e-14,
                  max_step=2*math.pi/100)
    lab=sol.y[:,-1]
    ang=-w*sol.t[-1]; c,s=math.cos(ang),math.sin(ang)
    R=np.array([[c,-s],[s,c]])
    pos=R@lab[:2]
    vel=R@lab[2:]-w*np.array([-pos[1],pos[0]])
    assert np.allclose(rot.y[:,-1],np.r_[pos,vel],rtol=2e-8,atol=2e-9)

    bad=Field("RO",kappa=.1044,kappa8=.006,omega_ro=1,omega8=1)
    try:
        rotating_invariant(y0,0,bad)
    except ValueError:
        pass
    else:
        raise AssertionError("detuned order-8 term must not be treated as autonomous RO")

