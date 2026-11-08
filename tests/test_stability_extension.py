import importlib
from pathlib import Path

import pytest


@pytest.fixture
def extension(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("extend_stability_p4")


def cell(r, q, held, unresolved=0):
    return dict(radius_mm=r, coupling=q, held=held, unresolved=unresolved, survival_lower=held / 6)


def test_horizontal_brackets_preserve_reentry_and_skip_unresolved(extension):
    rows = [cell(1., q, held, error) for q, held, error in
            ((1., 6, 0), (2., 0, 0), (4., 6, 0), (8., 0, 1), (16., 0, 0))]
    brackets = list(extension.axis_brackets(rows, "q", 1.))
    middle = [row for row in brackets if row["level"] == .5]
    assert [(b["left"], b["right"], b["direction"]) for b in middle] == [(1., 2., "loss"), (2., 4., "reentry")]
    assert extension.refine_cells(brackets, 1.) == {(1., 2**.5), (1., 8**.5)}


def test_vertical_refinement_keeps_fixed_coupling(extension):
    rows = [cell(7., .12, 6), cell(7.8, .12, 0), cell(7.4, .2, 6)]
    brackets = list(extension.axis_brackets(rows, "radius", 2., {.12}))
    assert len(brackets) == 3
    assert all(row["fixed"] == .24 and row["direction"] == "loss" for row in brackets)
    assert extension.refine_cells(brackets, 2.) == {(7.4, .12)}
