from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads((HERE / "config_v10.json").read_text(encoding="utf-8"))
    sources = {
        "h35_mesh": "outputs/meshes/gmsh_35um/trap.msh",
        "h50_mesh": "outputs/meshes/gmsh_50um/trap.msh",
        "h35_p4": "outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz",
        "h50_p4": "outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz",
        "h35_n48": "outputs/abc_validation/direct_fem_validation/v2/h35_P4/diagnostic_N48.json",
        "h50_n48": "outputs/abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json",
        "v7_config": "outputs/octupole_comparison_home/harmonic_ablation_v7/config_v7.json",
        "v7_integrator": "outputs/octupole_comparison_home/harmonic_ablation_v7/run_v7.py",
        "v7_records": "outputs/octupole_comparison_home/harmonic_ablation_v7/processed/ablation_trajectory_results.csv",
        "direct_fem_source": "src/three_phase_trap/direct_fem.py",
        "direct_dynamics_source": "src/three_phase_trap/direct_fem_dynamics.py",
    }
    expected = {
        "h35_mesh": config["fem"]["mesh_artifacts"]["h35_msh_sha256"],
        "h50_mesh": config["fem"]["mesh_artifacts"]["h50_msh_sha256"],
        "h35_p4": config["fem"]["mesh_artifacts"]["h35_p4_sha256"],
        "h50_p4": config["fem"]["mesh_artifacts"]["h50_p4_sha256"],
    }
    observed = {name: sha256(ROOT / rel) for name, rel in sources.items()}
    mismatches = {name: {"expected": value, "observed": observed[name]}
                  for name, value in expected.items() if observed[name] != value}

    with (ROOT / sources["v7_records"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    v7_states = {}
    for item in config["initial_conditions_v7"]:
        key = item["label"]
        hits = [r for r in rows if r["representation"] == "full_N48"
                and r["field_source"] == "h35_N48_polynomial"
                and abs(float(r["azimuth_deg"]) - item["azimuth_deg"]) < 1e-12
                and int(r["direction"]) == item["direction"]
                and abs(float(r["phase_deg"]) - item["phase_deg"]) < 1e-12]
        if not hits:
            raise ValueError(f"V7 source state not found: {key}")
        state = hits[0]
        expected_state = (item["azimuth_deg"], item["direction"], item["phase_deg"],
                          item["x_m"], item["y_m"], item["vx_m_s"], item["vy_m_s"])
        source_state = (float(state["azimuth_deg"]), int(state["direction"]), float(state["phase_deg"]),
                        float(state["x0_m"]), float(state["y0_m"]), float(state["vx0_m_s"]), float(state["vy0_m_s"]))
        if any(abs(a - b) > 2e-14 * max(1.0, abs(a), abs(b)) for a, b in zip(expected_state, source_state)):
            raise ValueError(f"V10 initial state does not match V7 source row: {key}")
        v7_states[key] = {"source_status": state["status"],
                          "source_escape_RF": state["escape_time_RF"] or None,
                          "source_field": state["field_source"], "state_matches": True}

    result = {
        "source_commit": config["source_commit"],
        "source_hashes": {name: {"path": rel, "sha256": observed[name]} for name, rel in sources.items()},
        "config_mesh_and_solution_hashes_match": not mismatches,
        "mismatches": mismatches,
        "V7_launch_conditions_verified": v7_states,
        "V7_full_N48_rows_read": len(rows),
        "new_FEM_solves": 0,
        "status": "PASS" if not mismatches else "FAIL",
    }
    (HERE / "audit_stage_A.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "source_files": len(sources),
                      "verified_V7_launches": len(v7_states), "V7_rows": len(rows)}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
