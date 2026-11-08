import numpy as np
import pytest

from test_direct_fem import analytic_field
from three_phase_trap.direct_fem_dynamics import integrate_native
from three_phase_trap.stability_rf_p4_v2 import NativeEscapeV2


def test_segment_uses_absolute_rf_clock():
    field = analytic_field()
    initial = np.array([-.3, .13, .25, -.03])
    native = NativeEscapeV2(field, .9)
    kwargs = dict(rtol=2e-11, atol=2e-13)
    for first, last in ((0, 1), (1, 2)):
        actual = native.segment(initial, .01, 1., .42, first, last, steps=50, **kwargs)
        expected = integrate_native(field, initial, .01, 1., 1., .42, first, last,
                                    steps_per_period=50, radius=.9, **kwargs)
        assert actual["escape_RF"] is None
        assert actual["reached_RF"] == last
        assert actual["crossings"] == len(expected.crossings)
        np.testing.assert_allclose(actual["final_state_SI"], expected.run.states[-1], atol=1e-13, rtol=0)
        initial = np.array(actual["final_state_SI"])


def test_escape_time_is_global_after_resume():
    native = NativeEscapeV2(analytic_field(), .6)
    initial = np.array([.1, .11, .4, 0.])
    settings = dict(steps=20, rtol=1e-10, atol=1e-12)
    first = native.segment(initial, 0., 1., 0., 0, 1, **settings)
    assert first["escape_RF"] is None
    second = native.segment(np.array(first["final_state_SI"]), 0., 1., 0., 1, 2, **settings)
    expected = (np.sqrt(.6**2 - .11**2) - .1) / .4
    assert second["escape_RF"] == pytest.approx(expected, abs=2e-12)
    assert second["reached_RF"] == second["escape_RF"]
