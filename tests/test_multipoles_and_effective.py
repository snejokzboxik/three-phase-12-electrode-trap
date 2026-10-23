import numpy as np

from three_phase_trap.effective_potential import effective_potential_from_mean_square, mean_square_field_from_basis
from three_phase_trap.multipoles import estimate_radial_exponent, fit_multipoles, fit_rotation, polar_sample_points


def test_multipole_fit_recovers_quadrupole():
    points = polar_sample_points(np.linspace(0.2, 0.9, 8), np.linspace(0, 2 * np.pi, 80, endpoint=False))
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    values = (r / 0.9) ** 2 * np.cos(2 * theta)
    result = fit_multipoles(points, values, tuple(range(1, 7)), 0.9)
    assert result[2].relative_amplitude > 0.999999
    assert all(result[m].relative_amplitude < 1e-10 for m in (1, 3, 4, 5, 6))


def test_multipole_fit_recovers_rotated_octupole_like_order_four():
    points = polar_sample_points(np.linspace(0.15, 0.85, 7), np.linspace(0, 2 * np.pi, 96, endpoint=False))
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    values = (r / 0.85) ** 4 * (0.6 * np.cos(4 * theta) - 0.8 * np.sin(4 * theta))
    result = fit_multipoles(points, values, (1, 2, 3, 4, 5, 6), 0.85)
    assert result[4].relative_amplitude > 0.999999
    assert np.isclose(result[4].a, 0.6, atol=1e-10)
    assert np.isclose(result[4].b, -0.8, atol=1e-10)


def test_radial_exponent_fit():
    radii = np.linspace(0.1, 1.0, 20)
    exponent, residual = estimate_radial_exponent(radii, 3.0 * radii**3)
    assert np.isclose(exponent, 3.0)
    assert residual < 1e-12


def test_three_phase_mean_square_field_matches_direct_average():
    fields = {"A": np.array([[1.0, 2.0]]), "B": np.array([[2.0, -1.0]]), "C": np.array([[-1.0, 0.5]])}
    analytic = mean_square_field_from_basis(fields, 4.0)
    times = np.linspace(0.0, 2 * np.pi, 20001, endpoint=False)
    voltages = np.column_stack((np.sin(times), np.sin(times - 2 * np.pi / 3), np.sin(times + 2 * np.pi / 3))) * 4.0
    vectors = voltages[:, 0, None] * fields["A"] + voltages[:, 1, None] * fields["B"] + voltages[:, 2, None] * fields["C"]
    numeric = np.mean(np.sum(vectors**2, axis=-1), axis=0)
    assert np.allclose(analytic, numeric, rtol=2e-4)


def test_rotation_fit_and_periodic_field_scaling():
    times = np.linspace(0.0, 1.0, 40)
    fit = fit_rotation(times, 0.7 + 2.5 * times)
    assert np.isclose(fit["angular_rate"], 2.5)

