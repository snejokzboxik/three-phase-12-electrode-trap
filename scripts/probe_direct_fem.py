"""Independent time-integration probe for native FEM, no final physics status."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from build_direct_fem import load_field, BASE, OUTPUT
from run_abc_pilot import write, sha
from three_phase_trap.direct_fem_dynamics import integrate_native
from three_phase_trap.pilot_dynamics import integrate_rf, coordinate_metrics


def main():
    p = json.loads((BASE/"protocol.json").read_text(encoding="utf-8"))["pilot"]
    initial = np.array(json.loads((BASE/"initial_conditions.json").read_text(encoding="utf-8"))["cases"][0]["A_and_A_smooth_full_state"])
    field = load_field(70, 4, p["U0_V"])
    d = OUTPUT/"probe"
    d.mkdir(exist_ok=True)
    results, records = {}, {}
    for method in ("DOP853", "Radau", "ordinary_DOP853"):
        if method == "ordinary_DOP853":
            result = integrate_rf(field, initial, p["charge_C"], p["mass_kg"], p["frequency_Hz"], 0., 0., 128/p["frequency_Hz"],
                                  max_steps_per_period=100, rtol=2e-11, atol=[2e-15, 2e-15, 2e-13, 2e-13])
            diagnostics = {"whole_step_geometry": False}
        else:
            out = integrate_native(field, initial, p["charge_C"], p["mass_kg"], p["frequency_Hz"], 0., 0., 128/p["frequency_Hz"],
                                   method=method, steps_per_period=100 if method == "DOP853" else 50,
                                   rtol=2e-11 if method == "DOP853" else 2e-8,
                                   atol=[2e-15, 2e-15, 2e-13, 2e-13] if method == "DOP853" else [2e-12, 2e-12, 2e-10, 2e-10])
            result, diagnostics = out.run, out.diagnostics
        path = d/(method+".npz")
        np.savez_compressed(path, time_s=result.time, states=result.states)
        records[method] = {"wall_seconds": result.wall_seconds, "diagnostics": diagnostics, "sha256": sha(path)}
        results[method] = result.states
        print(method, result.wall_seconds, diagnostics, flush=True)
    metrics = {k: coordinate_metrics(v[:, :2], results["DOP853"][:, :2], .001) for k, v in results.items()}
    write(d/"summary.json", {"records": records, "comparisons_to_audited_DOP853": metrics,
                             "scope": "128 RF-period probe; not the long validation", "direct_field_degree": 4})
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
