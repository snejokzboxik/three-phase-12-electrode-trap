import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from three_phase_trap.averaged_dynamics import AveragedRFModel
from three_phase_trap.direct_fem import DirectFEMRF
from three_phase_trap.direct_fem_dynamics import local_rhs
from three_phase_trap.effective_reconstruction_v1 import RadialPower, integrate_effective, turning_data
from three_phase_trap.long_rf_run import write_json
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction


@pytest.fixture
def transfer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_effective_p4_transfer")


def test_drive_swap_is_pi_minus_phase_in_native_force(transfer):
    gx = np.zeros((1, 10), complex)
    gy = np.zeros_like(gx)
    gx[0, :3] = [2 + 3j, 1 - 4j, 3 - 2j]
    gy[0, :3] = [3 - 1j, 1 - 2j, -4 + 4j]
    native = SimpleNamespace(origin=np.zeros((1, 2)), inverse=np.eye(2)[None], gx=gx, gy=gy,
                             potential_coefficients=gx, cubic=DirectFEMRF.cubic)
    reversed_native = transfer.reverse_native(native)
    state = [.03, .04, 0., 0.]
    t, omega, phase = .001, 200., .7
    expected = local_rhs(native, 0, 1., omega, np.pi - phase)(-t, state)
    np.testing.assert_allclose(local_rhs(reversed_native, 0, 1., omega, phase)(t, state), expected, atol=1e-14)
    assert native.gx[0, 0] == 2 + 3j
    quadratures = HarmonicQuadratures(np.array([0., 1 + 2j]), np.array([0., 3 + 4j]), 1.)
    reconstruction = RFReconstruction(quadratures, 1., {})
    reverse = transfer.reverse_reconstruction(reconstruction).unit_quadratures
    np.testing.assert_allclose(reverse.field(state[:2], omega * t + phase),
                               quadratures.field(state[:2], np.pi - omega * t - phase), atol=1e-14)


def test_analysis_accepts_held_out_gyroscopic_trajectory_and_writes_summary(transfer, tmp_path):
    frequency, mass = 20000., 1e-15
    power = RadialPower(6., 2.2e-17)
    period = power.period(.001, mass)
    cycles = int(np.ceil(3.25 * period * frequency))
    raw_clock = np.arange(cycles * 64 + 1) / (64 * frequency)
    clock = np.arange(cycles + 1) / frequency
    initial = [1e-3, 0., 0., 0.]
    actual = integrate_effective(raw_clock, initial, mass, power, gamma_mm=1.2)
    model_e = actual[64:-64:64]
    model_b = integrate_effective(clock, initial, mass, power)[1:-1]
    metrics, _ = turning_data(clock[1:-1], model_e, period)
    q = HarmonicQuadratures(np.zeros(2), np.zeros(2), .008)
    b = AveragedRFModel(RFReconstruction(q, .008, {}), 1e-12, mass, 0., frequency)
    runs = []
    for mesh in transfer.MESHES:
        for level in transfer.LEVELS:
            label = f"case_h{mesh}_{level}"
            folder = tmp_path / "runs" / label
            folder.mkdir(parents=True)
            np.savez(folder / "trajectory.npz", time_s=raw_clock, states=actual)
            runs.append(dict(run=label, case="case", helicity=1, phase_rad=0., direction=[1., 0.],
                             reference_slow_period_s=period))
    verdicts = transfer.analyze(tmp_path, runs, {1: b}, {"case": (clock[1:-1], {"B": model_b, "E": model_e})},
                                [dict(case="case", model="E", fitted_rate_deg_per_s=metrics["fitted_rate_deg_per_s"])])
    assert verdicts[0]["verdict"] == "supports_E"
    write_json(tmp_path / "summary.json", verdicts)
