"""Physical invariants and time alignment for the controlled frequency study."""
from pathlib import Path

import numpy as np
import pytest

from three_phase_trap.averaged_dynamics import AveragedRFModel
from three_phase_trap.micromotion import matched_full_state
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction
from three_phase_trap.slow_analysis import centered_period_mean


def study_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    import run_abc_frequency_limit
    return run_abc_frequency_limit


def test_controlled_limit_preserves_slow_force_and_reduces_micromotion(monkeypatch):
    study = study_module(monkeypatch)
    cosine = np.zeros(9, complex)
    sine = np.zeros(9, complex)
    cosine[4], sine[4] = .7+.2j, -.1-.8j
    cosine[8], sine[8] = .02-.03j, .04+.01j
    reconstruction = RFReconstruction(HarmonicQuadratures(cosine, sine, 1.), 1., {})
    base = AveragedRFModel(reconstruction, 2., 3., 4., 5.)
    R, V, phase = np.array([.4, -.2]), np.zeros(2), .63
    original = matched_full_state(base.quadratures, R, V, base.charge, base.mass, base.omega, phase)
    original_jc, original_js = base.quadratures.jacobian_quadratures(R)
    original_j = original_jc*np.cos(phase)+original_js*np.sin(phase)
    original_eta = np.linalg.norm(original_j, 2)/base.omega**2
    for s in (1, 2, 4):
        model = study.scaled_model(base, s)
        assert np.allclose(model.potential(R), base.potential(R), rtol=1e-14, atol=0.)
        assert np.allclose(model.force(R), base.force(R), rtol=1e-14, atol=0.)
        full = matched_full_state(model.quadratures, R, V, model.charge, model.mass, model.omega, phase)
        assert np.allclose(full[:2]-R, (original[:2]-R)/s, rtol=1e-10, atol=1e-17)
        # Velocity of fast motion stays finite; copying/dividing all four
        # original full coordinates would produce a different slow problem.
        assert np.allclose(full[2:], original[2:], rtol=1e-14, atol=0.)
        jc, js = model.quadratures.jacobian_quadratures(R)
        eta = np.linalg.norm(jc*np.cos(phase)+js*np.sin(phase), 2)/model.omega**2
        assert eta == pytest.approx(original_eta/s, rel=1e-14)
    with pytest.raises(ValueError):
        study.scaled_model(base, 0.)


def test_centred_comparison_uses_identical_physical_times_for_every_frequency(monkeypatch):
    study = study_module(monkeypatch)
    base_time = np.arange(8*64+1)/64
    target = base_time[96:-96]
    for s in (1, 2, 4):
        time = np.arange(8*64*s+1)/(64*s)
        values = np.column_stack((.3*time+np.sin(2*np.pi*s*time), -.2*time+np.cos(2*np.pi*s*time)))
        centre, mean = centered_period_mean(values, time, 64)
        actual = study.sample_at_centres(centre, mean, target)
        expected = np.column_stack((.3*target, -.2*target))
        assert np.allclose(actual, expected, rtol=0., atol=2e-13)
        with pytest.raises(ValueError, match="nested"):
            study.sample_at_centres(centre, mean, target+1e-4)
        with pytest.raises(ValueError, match="nested"):
            study.sample_at_centres(centre, mean, np.array([-1.]))
