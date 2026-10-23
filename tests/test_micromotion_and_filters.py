import numpy as np
import pytest

from three_phase_trap.micromotion import (matched_full_state, recover_slow_state, PhasedFEMField,
                                        matched_full_state_with_drift, recover_slow_state_with_drift)
from three_phase_trap.slow_analysis import centered_period_mean, local_rf_trend, poincare_samples
from three_phase_trap.validation_benchmarks import ideal_field
from three_phase_trap.voltages import phase_voltages


def test_full_velocity_is_derivative_of_full_position_with_convection():
    model = ideal_field(4, 1.2)
    R, V, omega, phase = np.array([.8, .3]), np.array([.07, -.04]), 20., .7
    state = matched_full_state(model, R, V, 1., 1., omega, phase)
    h = 1e-6
    def r(t):
        return R+V*t-model.field(R+V*t, omega*t+phase)/omega**2
    numerical_velocity = (r(h)-r(-h))/(2*h)
    assert np.allclose(state[2:], numerical_velocity, rtol=0., atol=2e-10)
    ec, es = model.field_quadratures(R)
    velocity_without_convection = V-(-ec*np.sin(phase)+es*np.cos(phase))/omega
    assert np.linalg.norm(state[2:]-velocity_without_convection) > 1e-5


def test_batched_inverse_transform_and_singular_transform_failure():
    model = ideal_field(4, 1.2)
    R = np.array([[.8, .3], [.5, -.2], [-.6, .4]])
    V = np.array([[.07, -.04], [.02, .03], [-.03, .06]])
    phases = np.array([0., .7, 2.1])
    full = matched_full_state(model, R, V, 1., 1., 20., phases)
    slow = recover_slow_state(model, full, 1., 1., 20., phases)
    assert np.allclose(slow, np.column_stack((R, V)), rtol=0., atol=2e-14)
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        recover_slow_state(ideal_field(2, 1.), [.1, .1, 0., 0.], 1., 1., np.sqrt(2), 0.)


def test_drift_corrected_velocity_matches_material_derivative():
    model = ideal_field(4, 1.2)
    R, V, omega, phase = np.array([.8, .3]), np.array([.07, -.04]), 20., .7
    A = model.force(R, 1., 1., omega)
    full = matched_full_state_with_drift(model, R, V, 1., 1., omega, phase)
    h = 1e-6
    def r(t):
        return matched_full_state_with_drift(model, R+V*t+.5*A*t*t, V+A*t, 1., 1., omega, phase+omega*t)[:2]
    assert np.allclose(full[2:], (r(h)-r(-h))/(2*h), rtol=0., atol=2e-10)


def test_drift_corrected_batched_inverse_transform():
    model = ideal_field(4, 1.2)
    R = np.array([[.8, .3], [.5, -.2], [-.6, .4]])
    V = np.array([[.07, -.04], [.02, .03], [-.03, .06]])
    phases = np.array([0., .7, 2.1])
    full = matched_full_state_with_drift(model, R, V, 1., 1., 20., phases)
    slow = recover_slow_state_with_drift(model, full, 1., 1., 20., phases)
    assert np.allclose(slow, np.column_stack((R, V)), rtol=0., atol=2e-12)


def test_trapezoid_window_bias_and_central_timestamp_are_known():
    time = np.arange(257)/64
    values = np.column_stack((time**2, 2*time+np.cos(2*np.pi*time)))
    centres, mean = centered_period_mean(values, time, 64)
    assert np.array_equal(centres, time[32:-32])
    assert np.allclose(mean[:, 0], centres**2+1/12+(1/64)**2/6, rtol=0., atol=1e-13)
    assert np.allclose(mean[:, 1], 2*centres, rtol=0., atol=1e-13)
    with pytest.raises(ValueError, match="uniform"):
        centered_period_mean(values, time+1e-3*time**2, 64)


def test_local_trend_recovers_quadratic_with_drifting_rf_amplitude():
    time = np.arange(513)/64
    slow = np.column_stack((1.+.2*time+.01*time**2, -.2+.01*time-.03*time**2))
    rf = np.column_stack(((.1+.03*time)*np.cos(2*np.pi*time), (.04-.01*time)*np.sin(2*np.pi*time)))
    centres, trend = local_rf_trend(slow+rf, time, 64, 2)
    expected = np.column_stack((1.+.2*centres+.01*centres**2, -.2+.01*centres-.03*centres**2))
    assert np.allclose(trend, expected, rtol=0., atol=1e-13)


def test_poincare_uses_total_rf_phase_and_cubic_interpolation():
    frequency, phase = 2., .41
    time = np.linspace(0., 1.5, 601)
    states = np.column_stack((time**3, time**2, 3*time**2, 2*time))
    sampled_time, sampled = poincare_samples(time, states, frequency, phase)
    assert np.allclose((2*np.pi*frequency*sampled_time+phase)/(2*np.pi), [1., 2., 3.], atol=1e-14)
    assert np.allclose(sampled[:, 0], sampled_time**3, rtol=0., atol=1e-14)
    assert sampled_time[0] > time[0] and sampled_time[-1] < time[-1]


def test_fem_phase_adapter_shifts_all_group_voltages_together():
    class Basis:
        def field(self, points, time, amplitude, frequency):
            u = phase_voltages(time, amplitude, frequency)
            return np.asarray(points)*0+np.array([u["A"]+2*u["B"], u["C"]-3*u["B"]])
    field = PhasedFEMField(Basis(), 250., 1e5, 1.7)
    t = .13/1e5
    chi = 2*np.pi*1e5*t+1.7
    expected = 250*np.array([np.sin(chi)+2*np.sin(chi-2*np.pi/3), np.sin(chi+2*np.pi/3)-3*np.sin(chi-2*np.pi/3)])
    assert np.allclose(field([.001, .002], t), expected, rtol=1e-14, atol=1e-12)
