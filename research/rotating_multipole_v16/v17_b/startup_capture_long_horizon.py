"""10,000-RF startup checks at the capture window and its controls."""
from __future__ import annotations

import json
from pathlib import Path

from v17_b.startup_capture import orientation_sweep

OUT = Path(__file__).resolve().parent
RES = OUT / "results"


def main():
    # These points cover: captured start, just-above sampled loss of capture,
    # and a high target frequency despite an existing local lock branch.
    points = [0.545]
    path = RES / "startup_capture_long_horizon.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"] if path.exists() else []
    done = {round(r["omega"], 12) for r in rows}
    rows.extend(orientation_sweep(omega, 10000, 360) for omega in points if round(omega, 12) not in done)
    rows.sort(key=lambda r: r["omega"])
    result = {
        "status": "COMPLETED_10000_RF_ORIENTATION_CHECKS",
        "horizon_rf": 10000,
        "orientation_grid_count": 360,
        "unique_relative_phase_count": 120,
        "classification": "deterministic uniform phase grid; not a binomial basin probability",
        "rows": rows,
        "limits": ["exact centered 1-DOF reduction of the full RO3 point-charge model",
                   "COM perturbations tested separately by the full 6D calculation",
                   "ideal harmonic field and dimensionless coefficients"],
    }
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"horizon_rf": 10000,
                      "summary": [{"omega": r["omega"], "count": r["capture_count"],
                                   "fraction": r["orientation_grid_capture_fraction"],
                                   "eta": r["eta_drag_to_lock_threshold"], "nfev": r["nfev"]}
                                  for r in rows]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
