from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "harmonic_sensitivity.jsonl"
SUMMARY = ROOT / "results" / "harmonic_sensitivity_summary.json"


def digest(row):
    clean = {k: v for k, v in row.items() if k != "sha256"}
    packed = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def main():
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(rows, 1):
        if row.get("sha256") != digest(row):
            raise RuntimeError(f"SHA-256 mismatch on line {i}")
    baselines = {r["seed"]: r for r in rows if r["order"] == 0}
    groups = defaultdict(list)
    for row in rows:
        if row["order"]:
            groups[(row["order"], row["epsilon"], row["delta"])].append(row)
    cases = []
    for (order, epsilon, delta), items in sorted(groups.items()):
        items = sorted(items, key=lambda x: x["seed"])
        cases.append({"order": order, "epsilon": epsilon, "relative_phase_rad": delta,
            "scramble_fractions": [x["bounded_fraction"] for x in items],
            "mean_bounded_fraction": sum(x["bounded_fraction"] for x in items)/len(items),
            "scramble_sd": (sum((x["bounded_fraction"]-sum(y["bounded_fraction"] for y in items)/len(items))**2
                                 for x in items)/max(1,len(items)-1))**0.5 if len(items)>1 else None,
            "mean_lost_from_same_seed_baseline": sum(x["lost_from_baseline"] for x in items)/len(items),
            "mean_gained_vs_same_seed_baseline": sum(x["gained_vs_baseline"] for x in items)/len(items),
            "mean_same_seed_baseline_fraction": sum(baselines[x["seed"]]["bounded_fraction"] for x in items)/len(items),
            "n_scrambles": len(items)})
    phase_groups = defaultdict(list)
    for case in cases:
        phase_groups[(case["order"], case["epsilon"])].append(case)
    amplitude = []
    for (order, epsilon), items in sorted(phase_groups.items()):
        vals = [x["mean_bounded_fraction"] for x in items]
        amplitude.append({"order": order, "epsilon": epsilon,
                          "phase_mean_bounded_fraction": sum(vals)/len(vals),
                          "phase_range_bounded_fraction": [min(vals), max(vals)],
                          "mean_same_seed_baseline_fraction": sum(x["mean_same_seed_baseline_fraction"] for x in items)/len(items),
                          "mean_lost_from_baseline_per_seed": sum(x["mean_lost_from_same_seed_baseline"] for x in items)/len(items),
                          "mean_gained_vs_baseline_per_seed": sum(x["mean_gained_vs_same_seed_baseline"] for x in items)/len(items),
                          "phase_cases": len(items)})
    payload = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    payload.update({"validated_rows": len(rows), "baseline_fractions_by_seed": {
        str(seed): r["bounded_fraction"] for seed, r in baselines.items()},
        "results_by_order_amplitude_phase": cases,
        "phase_averages_by_order_amplitude": amplitude,
        "replication_note": "Two independent scrambled Sobol sets; SD is descriptive with only two scrambles.",
        "interpretation": "Finite-time survivor sensitivity, conditional on the stated initial-state law; not a stability boundary or chaos diagnosis."})
    tmp = SUMMARY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SUMMARY)
    print(json.dumps({"validated_rows": len(rows), "baseline_fractions_by_seed": payload["baseline_fractions_by_seed"],
                      "phase_averages_by_order_amplitude": amplitude}, indent=2))


if __name__ == "__main__":
    main()
