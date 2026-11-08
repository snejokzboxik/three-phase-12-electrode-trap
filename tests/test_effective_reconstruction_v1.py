import numpy as np
import pytest
from scipy.special import beta

from three_phase_trap.effective_reconstruction_v1 import (
    LENGTH, RadialPower, angular_momentum_fit, check_split, fit_power, integrate_effective,
    octupole_gamma, turning_data,
)


def test_power_reconstructed_from_integrated_training_launches():
    mass = 1e-15
    truth = RadialPower(6., 2.2e-17)
    observations = []
    for radius in (.55e-3, .8e-3, 1.2e-3):
        period = truth.period(radius, mass)
        t = np.linspace(0., 3.25 * period, 1800)
        states = integrate_effective(t, [radius, 0., 0., 0.], mass, truth)
        measured, _ = turning_data(t, states, period)
        observations.append([measured["amplitude_m"], measured["period_s"]])
    recovered, diagnostic = fit_power(*np.array(observations).T, mass)
    assert diagnostic["log_design_rank"] == 2
    assert recovered.exponent == pytest.approx(6., abs=2e-7)
    assert recovered.energy_at_mm_J == pytest.approx(truth.energy_at_mm_J, rel=2e-7)
    assert recovered.period(1.05e-3, mass) == pytest.approx(truth.period(1.05e-3, mass), rel=2e-7)


def test_rank_and_launch_split():
    with pytest.raises(ValueError, match="distinct amplitudes"):
        fit_power([LENGTH, LENGTH], [.02, .02], 1e-15)
    _, diagnostics = fit_power([LENGTH], [.02], 1e-15, exponent=6)
    assert diagnostics["log_design_rank"] == 1
    with pytest.raises(ValueError, match="disjoint"):
        check_split(["phase0", "phase1"], ["phase0"])
    check_split(["phase0"], ["phase1"])


def test_si_coefficient_and_frequency_scaling():
    p, mass, coefficient, radius = 6., 1e-15, 22., .9e-3
    model = RadialPower(p, coefficient * LENGTH**p)
    period = 4 * np.sqrt(mass / (2 * coefficient)) * radius**(1 - p / 2) * beta(1 / p, .5) / p
    assert model.period(radius, mass) == pytest.approx(period, rel=1e-13)
    assert model.potential(radius) == pytest.approx(coefficient * radius**p, rel=1e-13, abs=0.)
    assert model.coefficient == pytest.approx(coefficient)
    gamma = octupole_gamma(1e-12, mass, 3e10, 2 * np.pi * 1e5)
    assert gamma > 0.
    assert octupole_gamma(1e-12, mass, 3e10, 2 * np.pi * 2e5, -1) == pytest.approx(-gamma / 8)


@pytest.mark.parametrize("gamma", [0., 1.])
def test_energy_and_angular_invariants(gamma):
    mass = 1e-15
    model = RadialPower(6., 2.2e-17)
    states = integrate_effective(np.linspace(0., .12, 1801), [LENGTH, 0., 0., 0.], mass, model, gamma_mm=gamma)
    radius = np.linalg.norm(states[:, :2], axis=1)
    energy = mass * np.sum(states[:, 2:]**2, axis=1) / 2 + model.potential(radius)
    assert np.ptp(energy) / energy[0] < 1e-10
    assert angular_momentum_fit(states)["gamma_mm_s_inv"] == pytest.approx(gamma, abs=1e-10)
    if gamma == 0.:
        assert np.all(states[:, 1] == 0.)
