import numpy as np
from v16_1.dynamics import Harmonic, potential_jet
from v17_b.model import charge_moments, synchronous_branch, rotating_jacobian


def test_neutral_dipole_ro3_has_centered_lock_and_localization():
    a=0.1; k=.03; w=1.; I=(2*a)**2/12
    c=charge_moments([1,-1],[a,-a],4).moments
    lock=synchronous_branch(3,k,w,c,.02,I)
    assert lock['exists']
    J,b=rotating_jacobian(3,k,w,c,.05,.02,I,lock['phase'])
    assert np.isclose(b['b_abs'],3*k*a)
    assert max(np.linalg.eigvals(J).real)<0


def test_generalized_jet_against_finite_difference():
    a=.1; c=charge_moments([1,-1],[a,-a],4).moments
    hs=[Harmonic(3,'RO',.03,1.)]
    q=np.array([.023,-.031,.77]); tau=.42; h=1e-5
    _,g,H=potential_jet(*q[:2],q[2],tau,1.,hs,c)
    def u(v): return potential_jet(v[0],v[1],v[2],tau,1.,hs,c)[0]
    gn=np.zeros(3); hn=np.zeros((3,3))
    for i in range(3):
        ei=np.eye(3)[i]*h
        gn[i]=(u(q+ei)-u(q-ei))/(2*h)
        for j in range(3):
            ej=np.eye(3)[j]*h
            hn[i,j]=(u(q+ei+ej)-u(q+ei-ej)-u(q-ei+ej)+u(q-ei-ej))/(4*h*h)
    assert np.max(np.abs(g-gn))<2e-9
    assert np.max(np.abs(H-hn))<2e-7
