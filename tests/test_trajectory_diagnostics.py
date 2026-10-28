import numpy as np
import pytest

from three_phase_trap.trajectory_diagnostics import position_errors, turning_orientations


def test_rotated_orbit_keeps_radius_but_has_large_unaligned_position_error():
    t = np.linspace(0., 8*np.pi, 8001)
    ref = np.column_stack((np.cos(t), np.zeros_like(t)))
    angle = .2
    actual = np.cos(t)[:, None]*np.array([np.cos(angle), np.sin(angle)])
    errors = position_errors(actual, ref, 1.)
    assert np.max(abs(errors["radial"])) < 1e-14
    assert np.max(errors["vector"]) == pytest.approx(2*np.sin(angle/2))
    assert np.max(abs(errors["y"])) == pytest.approx(np.sin(angle))


def test_orientation_unwrap_does_not_confuse_half_cycle_sign_flip_with_precession():
    period, rate = 1., .02
    t = np.linspace(0., 15., 15001)
    xy = .001*np.cos(2*np.pi*t)[:, None]*np.column_stack((np.cos(rate*t), np.sin(rate*t)))
    times, positions, angles, result = turning_orientations(t, xy, period)
    assert len(times) >= 28
    assert result["fitted_rate_rad_per_s"] == pytest.approx(rate, abs=1e-8)
    assert np.max(abs(angles-rate*times)) < 1e-8
    assert result["radial_turning_span_m"] < 1e-11


def test_stationary_line_has_zero_orientation_drift():
    t = np.linspace(0., 5., 5001)
    angle = -.3
    xy = .001*np.cos(2*np.pi*t)[:, None]*np.array([np.cos(angle), np.sin(angle)])
    _, _, _, result = turning_orientations(t, xy, 1.)
    assert abs(result["fitted_rate_rad_per_s"]) < 1e-10
    assert result["first_orientation_deg"] == pytest.approx(np.rad2deg(angle), abs=1e-9)
