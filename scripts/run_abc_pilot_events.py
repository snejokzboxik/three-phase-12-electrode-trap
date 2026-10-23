"""Resolve raw FEM edge crossings after ordinary RF ODE convergence failed.

Independent versioned diagnostics; original pilot checkpoints stay unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from run_abc_pilot import OUTPUT, load_models, sha, write
from three_phase_trap.config import load_config
from three_phase_trap.fem_cache import load_fem_cache
from three_phase_trap.pilot_dynamics import AffineP2RF
from three_phase_trap.event_resolved_rf import integrate_resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--level", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--extent", choices=("probe", "pilot", "extended"), default="pilot")
    args = parser.parse_args()
    protocol, initial, b, c = load_models()
    original_settings = json.loads((OUTPUT/"settings.json").read_text(encoding="utf-8"))
    output = OUTPUT/"edge_resolved"
    output.mkdir(exist_ok=True)
    tag = f"A_phase{args.phase_index}_{args.level}_{args.extent}"
    arrays, metadata = output/(tag+".npz"), output/(tag+".json")
    provenance = {"raw_pilot_numerical_provenance": original_settings["numerical_provenance"],
                  "edge_resolved_source_sha256": sha(ROOT/"src/three_phase_trap/event_resolved_rf.py"),
                  "driver_sha256": sha(Path(__file__))}
    if arrays.exists() and metadata.exists():
        old = json.loads(metadata.read_text(encoding="utf-8"))
        if old["provenance"] == provenance and old["arrays_sha256"] == sha(arrays) and old["status"] == "completed":
            print(f"Reusing {tag}", flush=True)
            return
        raise ValueError("Existing event-resolved result differs; preserve it in a versioned folder")
    config = load_config(ROOT/"configs/gmsh_70um.toml")
    _, _, _, interp = load_fem_cache(config, ROOT)
    raw = AffineP2RF(interp, b.amplitude)
    control = original_settings["controls"][args.level]
    rf_periods = 64 if args.extent == "probe" else original_settings["RF_periods"][args.extent]
    initial_state = initial["cases"][args.phase_index]["A_and_A_smooth_full_state"]
    print(f"Starting edge-resolved {tag}, {rf_periods} RF periods...", flush=True)
    result, crossings = integrate_resolved(raw, initial_state, b.charge, b.mass, b.frequency,
                                           protocol["pilot"]["phase_rad"][args.phase_index], 0., rf_periods/b.frequency,
                                           max_steps_per_period=control["RF_steps_per_period"], rtol=control["RF_rtol"],
                                           atol=control["RF_atol_SI"], samples_per_period=64,
                                           progress=lambda value: print(json.dumps(value), flush=True))
    np.savez_compressed(arrays, time_s=result.time, states=result.states, energy_J=np.empty(0))
    record = {"status": result.status, "provenance": provenance, "arrays_sha256": sha(arrays),
              "triangle_crossings": crossings, "nfev": result.nfev, "wall_seconds": result.wall_seconds,
              "end_time_s": float(result.time[-1]), "controls": control, "phase_index": args.phase_index,
              "level": args.level, "extent": args.extent}
    write(metadata, record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
