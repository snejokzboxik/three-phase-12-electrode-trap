import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_mathieu_stability")


def test_first_mathieu_edge_is_recovered_by_monodromy(comparison):
    edge = comparison.FIRST_EDGE
    assert edge == pytest.approx(.90804633, abs=1e-7)
    for q, expected in ((edge - .001, True), (edge + .001, False)):
        for phase in (0., np.pi / 2):
            period = comparison.fundamental(q, phase, tight=True)[-1]
            assert bool(abs(np.trace(period)) < 2) == expected
            assert np.linalg.det(period) == pytest.approx(1., abs=1e-11)


def test_floquet_bound_covers_full_time_trajectory(comparison):
    q, phase = .445, .3
    state = np.array([.15, .2, .02, .01])
    maps = [comparison.fundamental(s * q, phase, tight=True) for s in (1, -1)]
    result = comparison.quadrupole(q, state, phase, 60, maps, tight=True)
    assert result["method"] == "floquet_envelope"

    def rhs(n, y):
        k = 2 * np.pi**2 * q * np.sin(2 * np.pi * n + phase)
        return y[2], y[3], -k * y[0], k * y[1]

    numerical = solve_ivp(rhs, (0., 60.), state, t_eval=np.linspace(0., 60., 6001),
                          method="DOP853", rtol=1e-12, atol=1e-14)
    assert numerical.success
    assert np.linalg.norm(numerical.y[:2], axis=0).max() <= result["radius_bound_over_aperture"]
