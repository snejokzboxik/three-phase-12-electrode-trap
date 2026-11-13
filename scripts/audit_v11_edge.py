"""Reintegrate all first-map coordinates where A or C escaped on h35 N48."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_v11_xy import OUT, fields, init_worker, run_job


def main():
    source = OUT / "xy/4ceec3a67c8fdfcd/results.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_position = {}
    for row in rows:
        by_position.setdefault((float(row["x0_m"]), float(row["y0_m"])), {})[row["model"]] = row
    positions = [position for position, pair in by_position.items()
                 if pair["A"]["status"] == "escaped" or pair["C"]["status"] == "escaped"]
    q, hashes = fields(35)
    voltage = {"A": 250., "B": 250., "C": 250.}
    directory = OUT / "numerical_checks/h35_edge_equal_phase0"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source.relative_to(ROOT)), "positions": positions,
                "models": list("ABC"), "mesh_um": 35, "voltage_V": voltage,
                "phase_deg": 0., "horizon_RF": 1000, "field_sha256": hashes}
    (directory / "config.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    jobs = [(model, float(x), float(y), 0., 1000, 100,
             str(directory / "checkpoints" / f"{i:03d}_{model}"))
            for i, (x, y) in enumerate(positions) for model in "ABC"]
    print(f"h35 edge audit: {len(positions)} positions, {len(jobs)} trajectories", flush=True)
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker, initargs=(35, voltage)) as pool:
        results = list(pool.map(run_job, jobs))
    with (directory / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in results))))
        writer.writeheader(); writer.writerows(results)
    disagreements = []
    for row in results:
        old = by_position[(row["x0_m"], row["y0_m"])][row["model"]]
        if old["status"] != row["status"]:
            disagreements.append({"model": row["model"], "x0_m": row["x0_m"], "y0_m": row["y0_m"],
                                  "h50_status": old["status"], "h35_status": row["status"]})
    (directory / "comparison.json").write_text(json.dumps({"disagreements": disagreements,
                            "trajectories": len(results)}, indent=2)+"\n", encoding="utf-8")
    print(f"disagreements={len(disagreements)}", flush=True)


if __name__ == "__main__":
    main()
