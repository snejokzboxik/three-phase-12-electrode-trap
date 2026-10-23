"""Third raw-FEM ODE level: explicit edge events, T_RF/200, tighter tolerances.

Retains both earlier levels; does not replace an unconverged result silently.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from run_abc_pilot import OUTPUT, load_models, sha, write
from three_phase_trap.config import load_config
from three_phase_trap.fem_cache import load_fem_cache
from three_phase_trap.pilot_dynamics import AffineP2RF
from three_phase_trap.event_resolved_rf import integrate_resolved


def main():
    protocol, initial, b, c = load_models()
    settings = json.loads((OUTPUT/"settings.json").read_text(encoding="utf-8"))
    output = OUTPUT/"edge_resolved"
    arrays = output/"A_phase0_refined_pilot.npz"
    metadata = arrays.with_suffix(".json")
    provenance = {"raw_pilot_numerical_provenance": settings["numerical_provenance"],
                  "edge_resolved_source_sha256": sha(ROOT/"src/three_phase_trap/event_resolved_rf.py"),
                  "driver_sha256": sha(Path(__file__))}
    if arrays.exists() and metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["provenance"] == provenance and saved["arrays_sha256"] == sha(arrays):
            print("Reusing third refinement", flush=True)
            return
        raise ValueError("Existing third refinement differs")
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    _, _, _, interp = load_fem_cache(config, ROOT)
    control = {"RF_steps_per_period": 200, "RF_rtol": 2e-13,
               "RF_atol_SI": [2e-17, 2e-17, 2e-15, 2e-15]}
    print("Starting third refinement (phase 0, 3.25 slow periods)...", flush=True)
    result, crossings = integrate_resolved(AffineP2RF(interp, b.amplitude),
                                           initial["cases"][0]["A_and_A_smooth_full_state"],
                                           b.charge, b.mass, b.frequency, 0., 0., settings["RF_periods"]["pilot"]/b.frequency,
                                           max_steps_per_period=200, rtol=control["RF_rtol"], atol=control["RF_atol_SI"],
                                           progress=lambda value: print(json.dumps(value), flush=True))
    np.savez_compressed(arrays, time_s=result.time, states=result.states, energy_J=np.empty(0))
    write(metadata, {"status": result.status, "provenance": provenance, "arrays_sha256": sha(arrays),
                     "triangle_crossings": crossings, "nfev": result.nfev, "wall_seconds": result.wall_seconds,
                     "controls": control, "phase_index": 0, "level": "refined", "extent": "pilot"})
    print(json.dumps({"status": result.status, "wall_seconds": result.wall_seconds, "crossings": crossings}), flush=True)


if __name__ == "__main__":
    main()
