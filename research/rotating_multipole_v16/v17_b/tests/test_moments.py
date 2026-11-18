import numpy as np
from v17_b.model import charge_moments, expanded_polynomial, direct_polynomial


def test_neutral_dipole_uses_absolute_charge_reference():
    a=0.17
    m=charge_moments([1,-1],[a,-a],5)
    assert m.q_ref == 2
    assert np.allclose(m.moments, [0,a,0,a**3,0,a**5])


def test_complex_multipole_expansion_matches_direct_charge_sum():
    q=[2.,-1.,-1.]
    s=[0.11+0.03j,-0.07+0.02j,-0.04-0.05j]
    moments=charge_moments(q,s,7).moments
    Z=0.041-0.019j; psi=0.83
    for m in range(1,8):
        assert abs(expanded_polynomial(Z,psi,m,moments)-direct_polynomial(Z,psi,m,q,s))<2e-14
