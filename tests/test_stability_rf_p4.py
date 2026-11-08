import numpy as np
import pytest

from three_phase_trap.direct_fem_dynamics import integrate_native
from three_phase_trap.stability_rf_p4 import NativeEscape, first_radial_exit

from test_direct_fem import analytic_field


def test_radial_event_detects_excursion_with_both_endpoints_inside():
    k = np.arange(8)
    x = .5 + 4 * k / 7 - 4 * k * (k - 1) / (7 * 6)
    curve = np.column_stack((x, np.zeros_like(x)))
    assert first_radial_exit(curve, 1.) == pytest.approx((1 - np.sqrt(.5)) / 2, abs=1e-12)
    assert first_radial_exit(curve, 2.) is None


def test_native_terminal_event_crosses_at_ballistic_time():
    field = analytic_field()
    initial = np.array([.1, .11, .4, 0.])
    run = NativeEscape(field, .6)
    escaped, state, _, _ = run.run(initial, 0., 1., 0., 2, steps=20, rtol=1e-10, atol=1e-12)
    expected = (np.sqrt(.6**2 - .11**2) - .1) / .4
    assert escaped == pytest.approx(expected, abs=2e-12)
    assert np.linalg.norm(state[:2]) == pytest.approx(.6, abs=1e-13)


def test_native_survivor_matches_frozen_trajectory_integrator():
    field = analytic_field()
    initial = np.array([-.3, .13, .4, -.05])
    kwargs = dict(rtol=2e-11, atol=2e-13)
    expected = integrate_native(field, initial, .01, 1., 1., .42, 0., 1., steps_per_period=50, radius=.9, **kwargs)
    escaped, state, _, _ = NativeEscape(field, .9).run(initial, .01, 1., .42, 1, steps=50, **kwargs)
    assert escaped is None
    np.testing.assert_allclose(state, expected.run.states[-1], atol=1e-13, rtol=0)
