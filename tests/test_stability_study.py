import numpy as np

from three_phase_trap.stability_study import (
    COARSE_FREQUENCY_HZ,
    COARSE_U0_V,
    PRIMARY_ANGLES_DEG,
    PRIMARY_RADII_M,
    SECONDARY_STATES,
    _boundary_midpoints,
    _case_id,
    _cell_edges,
    _make_cases,
    _pair_summary,
    _read_checkpoint,
    _checkpoint_append,
    _stage_c_cases,
    _stage_c_selection,
)


def test_coarse_ensemble_has_requested_parameter_and_initial_condition_coverage():
    assert len(COARSE_U0_V) == 10
    assert len(COARSE_FREQUENCY_HZ) == 9
    assert len(PRIMARY_RADII_M) == 7
    assert PRIMARY_ANGLES_DEG == (0.0, 22.5, 45.0, 67.5)
    assert len(SECONDARY_STATES) == 4
    cases = _make_cases(COARSE_U0_V[0], COARSE_FREQUENCY_HZ[0], "coarse")
    assert len(cases) == 32
    assert sum(case["ensemble"] == "primary" for case in cases) == 28
    assert sum(case["ensemble"] == "velocity_probe" for case in cases) == 4


def test_checkpoint_append_and_resume_deduplicates_by_completed_case_id(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    row = {"case_id": "abc", "periods_survived": 3.0, "bounded": 0}
    _checkpoint_append(path, row)
    _checkpoint_append(path, row)
    loaded = _read_checkpoint(path)
    assert len(loaded) == 1
    assert loaded["abc"]["periods_survived"] == 3.0


def test_pair_summary_requires_all_tested_stage_b_initial_conditions_for_strict_stability():
    cases = _make_cases(250.0, 5000.0, "coarse")
    rows = []
    for case in cases:
        a_id = _case_id("coarse", "A_50", 250.0, 5000.0, case["initial_r_m"], case["initial_theta_deg"], (case["initial_vx_m_per_s"], case["initial_vy_m_per_s"]))
        # The production trajectory key is stage-independent; this compact
        # fixture creates it explicitly to exercise the summary join.
        from three_phase_trap.stability_study import _trajectory_id
        t_id = _trajectory_id("coarse", 250.0, 5000.0, case["initial_r_m"], case["initial_theta_deg"], (case["initial_vx_m_per_s"], case["initial_vy_m_per_s"]))
        rows.append({**case, "case_id": a_id, "trajectory_id": t_id, "stage": "A_50", "stage_a_survived": 1, "periods_survived": 50.0, "r_max_m": 0.001})
        rows.append({**case, "case_id": "b_" + a_id, "trajectory_id": t_id, "stage": "B_1000", "bounded": 1, "periods_survived": 1000.0, "r_max_m": 0.001, "r_rms_m": 0.0005})
    summary = _pair_summary(rows, "coarse")[0]
    assert summary["stage_b_outcomes_complete"] == 1
    assert summary["strict_stability"] == 1
    assert summary["stage_b_survival_fraction"] == 1.0


def test_boundary_refinement_adds_only_edges_where_survival_changes():
    first_u, second_u = COARSE_U0_V[:2]
    first_f, second_f = COARSE_FREQUENCY_HZ[:2]
    summary = [
        {"U0_V": first_u, "f_rf_Hz": first_f, "stage_b_outcomes_complete": 1, "strict_stability": 0, "stage_b_survival_fraction": 0.0},
        {"U0_V": first_u, "f_rf_Hz": second_f, "stage_b_outcomes_complete": 1, "strict_stability": 1, "stage_b_survival_fraction": 1.0},
    ]
    points = _boundary_midpoints(summary)
    assert len(points) == 1
    assert np.isclose(points[0][0], first_u)
    assert np.isclose(points[0][1], np.sqrt(first_f * second_f))


def test_log_plot_cell_edges_use_geometric_boundaries_for_positive_grid():
    values = np.asarray((5.0, 10.0, 40.0))
    edges = _cell_edges(values)
    assert np.all(edges > 0)
    assert np.all(np.diff(edges) > 0)
    assert np.allclose(edges[1:-1], np.sqrt(values[:-1] * values[1:]))


def test_limited_long_duration_selection_is_two_high_survival_candidates_and_one_boundary_point():
    summary = [
        {"grid_level": "coarse", "U0_V": 1000.0, "f_rf_Hz": 10000.0, "stage_b_outcomes_complete": 1, "strict_stability": 1, "stage_b_survival_fraction": 1.0},
        {"grid_level": "coarse", "U0_V": 2000.0, "f_rf_Hz": 20000.0, "stage_b_outcomes_complete": 1, "strict_stability": 1, "stage_b_survival_fraction": 1.0},
        {"grid_level": "coarse", "U0_V": 500.0, "f_rf_Hz": 5000.0, "stage_b_outcomes_complete": 1, "strict_stability": 0, "stage_b_survival_fraction": 0.0},
        {"grid_level": "coarse", "U0_V": 1500.0, "f_rf_Hz": 15000.0, "stage_b_outcomes_complete": 1, "strict_stability": 0, "stage_b_survival_fraction": 0.5},
        {"grid_level": "refined", "U0_V": 1250.0, "f_rf_Hz": 12000.0, "stage_b_outcomes_complete": 1, "strict_stability": 0, "stage_b_survival_fraction": 0.5},
    ]
    selection = _stage_c_selection(summary, [])
    assert len(selection) == 3
    assert sum(group == "high_survival_candidate" for _, _, group in selection) == 2
    assert sum(group == "near_boundary" for _, _, group in selection) == 1
    assert (1250.0, 12000.0) not in {(u, f) for u, f, _ in selection}


def test_limited_long_duration_cases_choose_only_one_trajectory_per_parameter_point():
    selection = [(1000.0, 10000.0, "high_survival_candidate"), (1500.0, 15000.0, "near_boundary")]
    rows = []
    for U0, frequency in [(1000.0, 10000.0), (1500.0, 15000.0)]:
        for index, radius in enumerate((0.00025, 0.0005)):
            rows.append({
                "grid_level": "coarse", "U0_V": U0, "f_rf_Hz": frequency,
                "stage": "B_1000", "bounded": 1, "ensemble": "primary",
                "r_rms_m": 0.0001 + index * 0.0001, "initial_r_m": radius,
                "initial_theta_deg": 0.0, "initial_vx_m_per_s": 0.0, "initial_vy_m_per_s": 0.0,
            })
    cases = _stage_c_cases(selection, rows)
    assert len(cases) == 2
    assert len({(case["U0_V"], case["f_rf_Hz"]) for case in cases}) == 2
    assert [case["initial_r_m"] for case in cases] == [0.00025, 0.00025]
