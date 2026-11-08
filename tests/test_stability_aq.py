import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq


def test_shared_dc_reduces_to_exact_static_saddle(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    study = importlib.import_module("run_stability_aq")
    a = .05
    x0, y0 = np.cos(np.pi / 8) / 8, np.sin(np.pi / 8) / 8
    rate = np.pi * np.sqrt(a)
    exact = brentq(lambda n: np.hypot(x0 * np.cos(rate * n), y0 * np.cosh(rate * n)) - 1, 0., 15.)
    for model in ("N48", "quadrupole"):
        result = study.trajectory(model, a, 0., 132167886.53016192, 1000, 2, True)
        assert result["status"] == "escaped"
        assert result["escape_RF"] == pytest.approx(exact, abs=1e-10)
