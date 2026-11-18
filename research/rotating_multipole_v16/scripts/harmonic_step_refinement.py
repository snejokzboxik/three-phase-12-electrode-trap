from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DATA = OUT / "harmonic_step_refinement.json"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v16.models import Field
from v16.integrate import integrate_ensemble
from run_search import sobol_initials

KAPPA4 = 0.1044
N = 64
HORIZON = 300
SEEDS = (6128, 16128)
TASKS = [{"seed": seed, "order": order, "epsilon": epsilon, "delta": delta}
         for seed in SEEDS
         for order, epsilon, delta in ((0, 0.0, 0.0),
                                       (8, 0.3, math.pi/2),
                                       (10, 0.3, math.pi/2))]


def digest(row):
    packed = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def task_id(t, steps):
    return f"seed={t['seed']}:m={t['order']}:eps={t['epsilon']}:delta={t['delta']:.8f}:steps={steps}"


def run(t, steps):
    init, phases = sobol_initials(N, t["seed"], KAPPA4, "quartic_scaled")
    order = t["order"]
    field = Field("RO", kappa=KAPPA4,
                  kappa8=t["epsilon"]*KAPPA4 if order == 8 else 0.0,
                  omega8=2.0, delta8=t["delta"] if order == 8 else 0.0,
                  kappa10=t["epsilon"]*KAPPA4 if order == 10 else 0.0,
                  omega10=2.5, delta10=t["delta"] if order == 10 else 0.0)
    result = integrate_ensemble(init, field, rf_count=HORIZON,
                                steps_per_rf=steps, initial_phase=phases)
    mask = (~np.isfinite(result["exit_tau"])).tolist()
    return {**t, "task_id": task_id(t, steps), "model": "RO+harmonic",
            "n": N, "horizon_rf": HORIZON, "steps_per_rf": steps,
            "bounded": int(sum(mask)), "bounded_fraction": float(np.mean(mask)),
            "survivor_mask": mask,
            "median_exit_rf": float(np.nanmedian(result["exit_tau"])/(2*math.pi))
                              if np.any(np.isfinite(result["exit_tau"])) else None}


def main():
    previous = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {"rows": []}
    rows = {r["task_id"]: r for r in previous.get("rows", [])}
    atomic = DATA.with_suffix(DATA.suffix + ".tmp")
    for steps in (96, 192):
        for task in TASKS:
            ident = task_id(task, steps)
            if ident not in rows:
                start = time.perf_counter()
                row = run(task, steps)
                row["elapsed_s"] = time.perf_counter() - start
                row["sha256"] = digest({k:v for k,v in row.items() if k != "sha256"})
                rows[ident] = row
                payload = {"status": "RUNNING", "updated_local": time.strftime('%Y-%m-%d %H:%M:%S %z'),
                           "rows": list(rows.values())}
                atomic.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                atomic.replace(DATA)
                print(json.dumps({"task_id": ident, "bounded": row["bounded"],
                                  "elapsed_s": round(row["elapsed_s"], 1)}), flush=True)
    output = []
    for seed in SEEDS:
        subset = [r for r in rows.values() if r["seed"] == seed]
        by = {(r["order"], r["steps_per_rf"]): r for r in subset}
        for order in (8,10):
            low_base = by[(0,96)]; high_base = by[(0,192)]
            low = by[(order,96)]; high = by[(order,192)]
            output.append({"seed": seed, "order": order, "epsilon": 0.3,
                "delta": math.pi/2, "baseline_mask_disagreement_96_vs_192": int(np.count_nonzero(
                    np.asarray(low_base["survivor_mask"]) != np.asarray(high_base["survivor_mask"]))),
                "perturbed_mask_disagreement_96_vs_192": int(np.count_nonzero(
                    np.asarray(low["survivor_mask"]) != np.asarray(high["survivor_mask"]))),
                "bounded_fraction_96": low["bounded_fraction"],
                "bounded_fraction_192": high["bounded_fraction"],
                "bounded_fraction_baseline_192": high_base["bounded_fraction"],
                "lost_from_baseline_192": int(np.count_nonzero(np.asarray(high_base["survivor_mask"])
                                                            & ~np.asarray(high["survivor_mask"]))),
                "gained_vs_baseline_192": int(np.count_nonzero(~np.asarray(high_base["survivor_mask"])
                                                              & np.asarray(high["survivor_mask"])))} )
    final = {"status": "COMPLETED", "n": N, "horizon_rf": HORIZON,
             "steps_per_rf": [96,192], "interpretation": "step refinement for the largest selected perturbations; finite-time screen only",
             "comparisons": output, "rows": list(rows.values())}
    atomic.write_text(json.dumps(final, indent=2), encoding="utf-8")
    atomic.replace(DATA)
    print(json.dumps({"status": "COMPLETED", "comparisons": output}, indent=2), flush=True)


if __name__ == "__main__":
    main()
