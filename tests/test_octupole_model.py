import numpy as np
import pytest
from scipy.special import beta

from three_phase_trap.octupole_model import OctupoleModel, octupole_ring_coefficients
from three_phase_trap.averaged_dynamics import integrate_averaged


def test_circle_extraction_separates_rotation_and_ignores_other_orders():
    angles = (np.arange(97)+.17)*2*np.pi/97
    desired, reverse = 2.3*np.exp(-.4j), .07*np.exp(.8j)
    c4, s4 = 1j*(desired-reverse), desired+reverse
    for radius in (.3, .7, 1.2):
        z = radius*np.exp(1j*angles)
        pc = (c4*z**4+(.05+.02j)*z**8).real+1.7
        ps = (s4*z**4+(.03-.01j)*z**16).real-.3
        actual_c, actual_s, actual_d, actual_reverse = octupole_ring_coefficients(pc, ps, angles, radius)
        assert actual_c == pytest.approx(c4, rel=1e-12, abs=1e-12)
        assert actual_s == pytest.approx(s4, rel=1e-12, abs=1e-12)
        assert actual_d == pytest.approx(desired, rel=1e-12, abs=1e-12)
        assert actual_reverse == pytest.approx(reverse, rel=1e-12, abs=1e-12)
    with pytest.raises(ValueError, match="uniformly"):
        octupole_ring_coefficients(pc, ps, angles**2, radius)


def test_octupole_phase_normalization_force_and_geometric_origin():
    model = OctupoleModel(1.3e8, .37, -1e-12, 1e-15, 250., 1e5, .008)
    points = np.array([[.001, .0002], [-.003, .0007], [.0001, -.0003], [0., 0.]])
    chi = .81
    radius = np.linalg.norm(points, axis=-1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    pc, ps = model.reconstruction.scaled(model.amplitude).potential_quadratures(points)
    phi = pc*np.cos(chi)+ps*np.sin(chi)
    assert np.allclose(phi, model.C4*radius**4*np.sin(chi-4*theta+model.phase), rtol=1e-13, atol=1e-16)
    expected_B6 = 8*model.charge**2*(250*1.3e8)**2/(1e-15*(2*np.pi*1e5)**2)
    assert model.B6 == pytest.approx(expected_B6, rel=1e-14)
    assert model.potential([0., 0.]) == 0.
    assert np.array_equal(model.force([0., 0.]), [0., 0.])
    assert np.allclose(model.force(points), -6*expected_B6*radius[:, None]**4*points, rtol=1e-14, atol=0.)
    # Isotropic averaged energy does not depend on RF orientation or sign of q.
    other = OctupoleModel(1.3e8, -.6, 1e-12, 1e-15, 250., 1e5, .008)
    assert np.allclose(model.potential(points), other.potential(points), rtol=0., atol=0.)


def test_octupole_integrator_against_energy_integral_period():
    model = OctupoleModel(.7, .2, 1., 1., 1., 20/(2*np.pi), 2.)
    amplitude = .8
    period = float(4*np.sqrt(model.mass/(2*model.B6))*beta(1/6, 1/2)/(6*amplitude**2))
    trajectory = integrate_averaged(model, [amplitude, 0.], [0., 0.], period,
                                    samples=401, rtol=1e-11, atol=1e-13, max_step=period/300)
    assert trajectory.status == "completed"
    assert trajectory.states[100, 0] == pytest.approx(0., abs=1e-9)
    assert trajectory.states[200, 0] == pytest.approx(-amplitude, abs=1e-9)
    assert trajectory.states[-1, 0] == pytest.approx(amplitude, abs=1e-9)
    assert trajectory.states[-1, 2] == pytest.approx(0., abs=1e-9)
    assert np.max(abs(trajectory.energy/trajectory.energy[0]-1)) < 1e-9
