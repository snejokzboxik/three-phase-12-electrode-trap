"""Validate the six completed, matched h50 P4 robustness pilot records."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "native_p4"
OUT = HERE / "processed" / "parameter_robustness_validation.json"

CASES = {
    "robust_E1_r1_az22p5_dplus": (1.0, 0.001, "retained_to_horizon", "retained_to_horizon", None),
    "robust_E3_r1_az22p5_dplus": (3.0, 0.001, "retained_to_horizon", "escaped", 87.69244444665038),
    "robust_E2_r2_az22p5_dplus": (2.0, 0.002, "retained_to_horizon", "escaped", 41.61426736765448),
}
EXPECTED_V = {"rotating": 250.0, "standing": 353.5533905932738}


def main() -> None:
    checks = []
    for label, (energy, radius, expected_rot, expected_stand, expected_stand_escape) in CASES.items():
        pair = {}
        for model in ("rotating", "standing"):
            path = DATA / f"50um_{model}_{label}_0deg_1000rf.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["mesh_um"] == 50
            assert record["field"]["loaded_saved_p4_solution"] is True
            assert record["field"]["new_fem_solve"] is False
            assert np.isclose(record["voltage_V"], EXPECTED_V[model], rtol=0, atol=1e-12)
            assert record["frequency_Hz"] == 100000.0
            assert record["initial_energy_eV"] == energy
            assert np.isclose(record["initial_radius_m"], radius, rtol=0, atol=1e-15)
            assert record["azimuth_deg"] == 22.5 and record["phase_deg"] == 0.0
            assert record["direction"] == 1 and record["horizon_rf"] == 1000
            assert record["normalization"] == "matched leading m=4 pseudopotential only"
            assert record["numerics"]["method"].startswith("DOP853 native P4")
            assert record["numerics"]["rtol"] == 2e-9
            assert record["numerics"]["steps_per_rf_period"] == 200
            trajectory_path = DATA / record["timeseries_file"]
            assert trajectory_path.is_file()
            with np.load(trajectory_path, allow_pickle=False) as trajectory:
                times = trajectory["time_s"]
                states = trajectory["states"]
            assert states.shape == (len(times), 4) and len(times) > 1
            assert np.all(np.isfinite(times)) and np.all(np.isfinite(states))
            assert np.all(np.diff(times) > 0) and times[0] == 0.0
            state = np.asarray(record["initial_state_SI"], dtype=float)
            assert state.shape == (4,) and np.all(np.isfinite(state))
            np.testing.assert_array_equal(states[0], state)
            if record["status"] == "retained_to_horizon":
                assert record["escape_time_rf"] is None
                assert record["numerics"]["maximum_three_point_dense_radius_m"] < 0.008
                assert np.isclose(times[-1], 1000.0/100000.0, rtol=0, atol=2e-13)
            else:
                assert record["status"] == "escaped"
                assert 0.0 < record["escape_time_rf"] < 1000.0
                assert record["numerics"]["escape_radial_velocity_m_s"] > 0.0
                assert abs(np.hypot(states[-1, 0], states[-1, 1])-0.008) < 2e-10
                assert abs(times[-1]-record["escape_time_s"]) < 1e-12
            pair[model] = {"record": record, "state": state}

        np.testing.assert_array_equal(pair["rotating"]["state"], pair["standing"]["state"])
        assert pair["rotating"]["record"]["status"] == expected_rot
        assert pair["standing"]["record"]["status"] == expected_stand
        if expected_stand_escape is not None:
            assert abs(pair["standing"]["record"]["escape_time_rf"]-expected_stand_escape) < 1e-3
        checks.append({
            "launch_label": label, "matched_initial_state_verified": True,
            "rotating_status": expected_rot, "standing_status": expected_stand,
            "rotating_escape_time_rf": pair["rotating"]["record"]["escape_time_rf"],
            "standing_escape_time_rf": pair["standing"]["record"]["escape_time_rf"],
            "mesh_um": 50, "finite_horizon_rf": 1000,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "passed", "case_count": len(checks), "trajectory_count": 2*len(checks),
        "validation": "matched physical launch states, saved h50 native P4 fields, protocol, finite-time outcomes and censoring metadata",
        "checks": checks,
    }, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "case_count": len(checks), "trajectory_count": 2*len(checks)}, indent=2))


if __name__ == "__main__":
    main()
