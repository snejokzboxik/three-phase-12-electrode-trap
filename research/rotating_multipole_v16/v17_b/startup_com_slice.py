"""One-dimensional full-6D COM perturbation slice through the RO3 start basin."""
from __future__ import annotations

import json
from pathlib import Path

from v17_b.startup_capture import full_6d_spinup

OUT = Path(__file__).resolve().parent
RES = OUT / "results"


def main():
    offsets = [0.001, 0.001546875, 0.0015625, 0.0035]
    rows = [full_6d_spinup(0.5, 3.141592653589793 / 3, 1000, offset)
            for offset in offsets]
    result = {
        "status": "COMPLETED_1D_COM_PERTURBATION_SLICE",
        "field": "RO3, omega=0.5, kappa3=0.03",
        "initial_orientation": "pi/3",
        "initial_spin": 0.0,
        "initial_velocity_rule": "vx=0.25*d, vy=0; x=d, y=-0.75*d",
        "horizon_rf": 1000,
        "rows": rows,
        "interpretation": "A single radial slice in the chosen COM position/velocity direction; not a multidimensional basin volume.",
    }
    (RES / "startup_com_slice.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": [{"d": r["com_offset_parameter"], "status": r["status"],
                                "escape_time": r.get("escape_time"),
                                "max_com_radius": r.get("max_com_radius_all"),
                                "nfev": r.get("nfev")} for r in rows]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
