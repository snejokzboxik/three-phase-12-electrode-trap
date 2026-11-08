import numpy as np
import pytest

from three_phase_trap.floquet_orbits import SmoothRFOrbit, circular_seed, shortest_period, spectrum


@pytest.fixture
def octupole():
    cosine = np.zeros(5, complex)
    sine = cosine.copy()
    cosine[4], sine[4] = 1j, 1.
    return SmoothRFOrbit(cosine, sine)


def test_exact_rotating_circle_closes_and_has_known_spectral_radius(octupole):
    initial = circular_seed(.55)
    final, matrix, states = octupole.flow(initial, .55, 4, steps=64, samples=8)
    np.testing.assert_allclose(final, initial, atol=1e-9, rtol=0)
    assert shortest_period(states, 8, 1e-8) == 4
    result = spectrum(matrix, defect_limit=2e-5)
    assert result["classification"] == "linearly_unstable"
    assert result["spectral_radius"] == pytest.approx(np.exp(2 * np.pi * np.sqrt(2)), rel=1e-8)


def test_variational_matrix_matches_perturbed_rf_flow(octupole):
    initial = np.array([.12, -.023, .05, .01])
    delta = np.array([1., .5, -.25, .2]) * 1e-6
    _, matrix, _ = octupole.flow(initial, .45, 3, steps=48)
    plus, _, _ = octupole.flow(initial + delta, .45, 3, steps=48, variational=False)
    minus, _, _ = octupole.flow(initial - delta, .45, 3, steps=48, variational=False)
    np.testing.assert_allclose(matrix @ delta, (plus - minus) / 2, atol=5e-13, rtol=0)


def test_central_monodromy_is_degenerate(octupole):
    _, matrix, _ = octupole.flow(np.zeros(4), .5, 2, steps=24)
    expected = np.eye(4)
    expected[:2, 2:] = 2 * np.eye(2)
    np.testing.assert_allclose(matrix, expected, atol=1e-14, rtol=0)
    assert spectrum(matrix, defect_limit=1e-8)["classification"] == "near_unit_or_degenerate"
