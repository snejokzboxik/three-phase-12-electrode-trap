"""Audit direction labels, initial states, segment continuity, and rebuild summary."""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def actual_sign(state):
    x, y, vx, vy = map(float, state)
    radius = math.hypot(x, y)
    if radius == 0:
        return 0, 0.0
    projection = vx * (-y / radius) + vy * (x / radius)
    return (1 if projection > 0 else -1 if projection < 0 else 0), projection


def requested_sign(key: str, metadata: dict) -> int:
    if "tangent_minus" in key or "_d-1" in key:
        return -1
    if "tangent_plus" in key or "_d+1" in key:
        return 1
    return int(metadata.get("tangential_sign", 1))


def atomic_json(path: Path, data: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_segment(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty trajectory segment: {path}")
    def state(row):
        return [float(row["x_mm"]) * 1e-3, float(row["y_mm"]) * 1e-3, float(row["vx_m_s"]), float(row["vy_m_s"])]
    return rows, state(rows[0]), state(rows[-1])


def main():
    root = OUT / "trajectories"
    audits = []
    summaries = []
    progress_by_key = {}
    for progress_path in sorted(root.glob("*/progress.json")):
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        key = progress_path.parent.name
        state = data.get("initial_state")
        metadata = data.get("initial_conditions", {})
        if not state:
            continue
        actual, projection = actual_sign(state)
        requested = requested_sign(key, metadata)
        metadata_sign = int(metadata.get("tangential_sign", actual))
        flagged = bool(data.get("excluded_from_science", False) or actual != requested or metadata_sign != actual)
        data.update({"requested_tangential_sign_from_run_key": requested, "actual_tangential_sign_from_initial_state": actual, "initial_tangential_velocity_projection_m_s": projection, "direction_audit_passed": not flagged})
        if flagged:
            data["excluded_from_science"] = True
            data["excluded_from_combined_summary"] = True
            data["exclusion_reason"] = data.get("exclusion_reason", "requested sign, path label, and actual initial velocity do not agree")
        atomic_json(progress_path, data)
        progress_by_key[key] = data

        segments = sorted(progress_path.parent.glob("segment_*.csv"))
        continuity_ok = True
        max_boundary_state_gap = 0.0
        segment_row_count = 0
        first_segment_sign = None
        prior_final = None
        for segment in segments:
            rows, first, last = read_segment(segment)
            segment_row_count += len(rows)
            if first_segment_sign is None:
                first_segment_sign = actual_sign(first)[0]
                if npclose_state(first, state, atol=2e-8):
                    pass
                else:
                    continuity_ok = False
            elif prior_final is not None:
                gap = max(abs(a - b) for a, b in zip(first, prior_final))
                max_boundary_state_gap = max(max_boundary_state_gap, gap)
                if gap > 2e-7:
                    continuity_ok = False
            prior_final = last
        if first_segment_sign is not None and first_segment_sign != actual:
            continuity_ok = False
        audits.append({"key": key, "requested_sign": requested, "actual_sign_from_initial_state": actual, "metadata_sign": metadata_sign, "tangential_velocity_projection_m_s": projection, "direction_audit_passed": not flagged, "excluded_from_science": flagged, "segment_files": len(segments), "segment_rows": segment_row_count, "segment_initial_direction": first_segment_sign, "segment_continuity_passed": continuity_ok, "maximum_adjacent_segment_state_gap": max_boundary_state_gap, "progress_path": str(progress_path.relative_to(ROOT))})

    for summary_path in sorted(root.glob("*/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        key = summary_path.parent.name
        audit = next((x for x in audits if x["key"] == key), None)
        if audit:
            data["requested_tangential_sign_from_run_key"] = audit["requested_sign"]
            data["actual_tangential_sign_from_initial_state"] = audit["actual_sign_from_initial_state"]
            data["direction_audit_passed"] = audit["direction_audit_passed"]
            data["excluded_from_science"] = audit["excluded_from_science"]
            data["excluded_from_combined_summary"] = audit["excluded_from_science"]
        atomic_json(summary_path, data)
        if not data.get("excluded_from_combined_summary", False):
            summaries.append(data)

    summary_path = OUT / "processed" / "matched_rf_trajectory_summary.json"
    combined = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    combined["trajectories"] = summaries
    combined["direction_audit"] = "See trajectory_direction_audit.json; records with sign mismatch are preserved but excluded."
    atomic_json(summary_path, combined)
    audit_output = {
        "trajectory_folders_audited": len(audits),
        "direction_pass_count": sum(x["direction_audit_passed"] for x in audits),
        "direction_excluded_count": sum(x["excluded_from_science"] for x in audits),
        "segment_continuity_pass_count": sum(x["segment_continuity_passed"] for x in audits),
        "segments_audited": sum(x["segment_files"] for x in audits),
        "valid_combined_summary_count": len(summaries),
        "audits": audits,
        "combined_summary_path": str(summary_path.relative_to(ROOT)),
    }
    out = OUT / "processed" / "trajectory_direction_audit.json"
    atomic_json(out, audit_output)
    print(json.dumps(audit_output, indent=2), flush=True)


def npclose_state(a, b, atol):
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= atol


if __name__ == "__main__":
    main()
