"""Compare completed V11 phase maps at exactly matching XY coordinates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11"


def load(folder):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    with (folder / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_key = {(row["model"], float(row["x0_m"]), float(row["y0_m"])): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError(f"Duplicate physical state in {folder}")
    return config, by_key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folders", nargs="+", type=Path)
    args = p.parse_args()
    maps = [load(folder) for folder in args.folders]
    configs = [c for c, _ in maps]
    fields = ("grid", "protocol", "mesh_um", "horizon_RF", "frequency_Hz", "mass_kg",
              "charge_C", "aperture_m", "voltage_V", "field_sha256")
    if len({tuple(json.dumps(c[k], sort_keys=True) for k in fields) for c in configs}) != 1:
        raise ValueError("Non-phase parameters differ")
    keys = set(maps[0][1])
    if any(set(rows) != keys for _, rows in maps):
        raise ValueError("Phase maps have different physical initial coordinates")
    phases = [c["phase_deg"] for c in configs]
    if len(set(phases)) != len(phases):
        raise ValueError("Duplicate phase")
    summary = []
    transitions = []
    for model in "ABC":
        model_keys = sorted((key for key in keys if key[0] == model), key=lambda k:(k[2], k[1]))
        for phase, (_, rows) in zip(phases, maps):
            statuses = [rows[key]["status"] for key in model_keys]
            summary.append({"model": model, "phase_deg": phase, "retained": statuses.count("retained"),
                            "escaped": statuses.count("escaped"),
                            "numerical_error": statuses.count("numerical_error"), "initial": len(statuses)})
        for key in model_keys:
            statuses = [rows[key]["status"] for _, rows in maps]
            if len(set(statuses)) > 1:
                transitions.append({"model": model, "x0_m": key[1], "y0_m": key[2],
                                    **{f"phase_{phase:g}": status for phase, status in zip(phases, statuses)}})
    folder = OUT / "phase_comparison"
    folder.mkdir(exist_ok=True)
    for filename, rows in (("summary.csv", summary), ("transitions.csv", transitions)):
        if rows:
            with (folder / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"phases_deg": phases, "summary": summary, "transitions": len(transitions)}))


if __name__ == "__main__":
    main()
