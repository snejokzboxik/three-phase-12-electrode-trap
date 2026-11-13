"""Refine measured class transitions from a complete 21x21 map at 61-grid spacing.

Only new physical states on observed neighbor transitions are calculated.
Unknown cells remain unknown; this is not a filled 61x61 map.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_v11_xy import OUT, fields, init_worker, run_job, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    source = args.source.resolve()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config["grid"] != 21 or config["horizon_RF"] != 1000:
        p.error("Expected completed 21x21, 1000 RF source map")
    with (source / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_position = {}
    for row in rows:
        position = (round(float(row["x0_m"])*1000, 9), round(float(row["y0_m"])*1000, 9))
        by_position.setdefault(position, {})[row["model"]] = row["status"]
    if len(by_position) != 305 or any(set(statuses) != set("ABC") for statuses in by_position.values()):
        raise ValueError("Source map incomplete")
    new = set()
    for (x, y), a in by_position.items():
        for dx, dy in ((.75, 0.), (0., .75)):
            end = (round(x+dx, 9), round(y+dy, 9))
            b = by_position.get(end)
            if b is None or not any(a[m] != b[m] for m in "ABC"):
                continue
            for fraction in (1/3, 2/3):
                px, py = round(x+fraction*dx, 9), round(y+fraction*dy, 9)
                if math.hypot(px, py) < 7.5:
                    new.add((px, py))
    all_positions = sorted(new, key=lambda xy:(xy[1], xy[0]))
    positions = all_positions[:args.limit] if args.limit else all_positions
    q, hashes = fields(config["mesh_um"])
    if hashes != config["field_sha256"]:
        raise ValueError("Source field checksum changed")
    key = hashlib.sha256(json.dumps({"source": source.name, "positions": all_positions}, sort_keys=True).encode()).hexdigest()[:16]
    folder = OUT / "boundary_refinement" / key
    folder.mkdir(parents=True, exist_ok=True)
    settings = {"source_map": str(source.relative_to(ROOT)), "positions_mm": all_positions,
                "step_mm": .25, "phase_deg": config["phase_deg"],
                "voltage_V": config["voltage_V"], "mesh_um": config["mesh_um"],
                "horizon_RF": 1000, "field_sha256": hashes}
    write_json(folder / "config.json", settings)
    jobs = [(model, x/1000, y/1000, math.radians(config["phase_deg"]), 1000, 100,
             str(folder / "checkpoints" / f"{i:04d}_{model}"))
            for i, (x, y) in enumerate(positions) for model in "ABC"]
    with (folder / "initial_states.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "x0_m", "y0_m", "vx0_m_s", "vy0_m_s", "phase_rad"))
        writer.writerows((m, x, y, 0., 0., phase) for m, x, y, phase, *_ in jobs)
    print(json.dumps({"new_positions": len(positions), "jobs": len(jobs), "folder": str(folder)}), flush=True)
    if args.dry_run or not jobs:
        return
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(config["mesh_um"], config["voltage_V"])) as pool:
        results = list(pool.map(run_job, jobs))
    with (folder / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in results))))
        writer.writeheader(); writer.writerows(results)
    print(json.dumps({"completed": len(results), "numerical_errors": sum(row["status"] == "numerical_error" for row in results)}))


if __name__ == "__main__":
    main()
