"""Refine the RO3 from-rest capture transition near Omega=0.5."""
from __future__ import annotations

import json
from pathlib import Path

from v17_b.startup_capture import orientation_sweep

OUT = Path(__file__).resolve().parent
RES = OUT / "results"


def main():
    points = [0.535, 0.540, 0.545]
    path = RES / "startup_capture_refinement.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"] if path.exists() else []
    done = {round(r["omega"], 12) for r in rows}
    rows.extend(orientation_sweep(omega, 1500, 360) for omega in points if round(omega, 12) not in done)
    rows.sort(key=lambda r: r["omega"])
    coarse_rows = json.loads((RES / "startup_capture_results.json").read_text(encoding="utf-8"))["frequency_sweep"]
    long_rows = json.loads((RES / "startup_capture_long_horizon.json").read_text(encoding="utf-8"))["rows"]
    available = coarse_rows + rows + long_rows
    best_by_frequency = {}
    for item in available:
        key = round(item["omega"], 10)
        if key not in best_by_frequency or item["rf_periods"] > best_by_frequency[key]["rf_periods"]:
            best_by_frequency[key] = item
    available = list(best_by_frequency.values())
    result = {
        "status": "COMPLETED_REFINED_STARTUP_FREQUENCY_SLICE",
        "horizon_rf": 1500,
        "orientation_grid_count": 360,
        "unique_relative_phase_count": 120,
        "classification": "deterministic uniform phase grid; fractions are grid quadrature, not probabilities",
        "rows": rows,
        "transition_bracket_on_available_samples": {
            "largest_tested_frequency_with_capture": max((r["omega"] for r in available if r["capture_count"] > 0), default=None),
            "smallest_tested_frequency_with_zero_capture": min((r["omega"] for r in available if r["capture_count"] == 0), default=None),
            "note": "Mixed 1500-RF fine grid and 10000-RF endpoint controls; not a bifurcation threshold.",
        },
        "limitations": ["same finite neutral dipole and fixed dimensionless coefficients as startup_capture_results.json",
                        "1500-RF horizon for this local refinement; selected endpoints require longer continuation"],
    }
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"transition_bracket": result["transition_bracket_on_available_samples"],
                      "rows": [{"omega": r["omega"], "count": r["capture_count"],
                                "fraction": r["orientation_grid_capture_fraction"],
                                "eta": r["eta_drag_to_lock_threshold"]} for r in rows]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
