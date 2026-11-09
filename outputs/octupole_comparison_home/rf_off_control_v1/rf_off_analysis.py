#!/usr/bin/env python3
"""Analytical RF-off control for the completed V2 octupole trajectory set.

This module does not integrate trajectories. It reads validated RF outcomes,
reconstructs each stored physical launch, and computes free-flight aperture
crossings from the exact line-circle intersection.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.pdfbase.pdfmetrics import stringWidth

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
V2 = PROJECT / "outputs" / "octupole_comparison_home" / "v2"
V1 = PROJECT / "outputs" / "octupole_comparison_home" / "v1"
OUT = HERE
EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27
F_RF_HZ = 100_000.0
APERTURE_M = 0.008
HORIZONS_RF = (300, 1000, 10000)
MODEL_ORDER = ("rotating", "standing_equal_peak")
MODEL_LABEL = {"rotating": "Rotating", "standing_equal_peak": "Standing"}
MODEL_COLOR = {"rotating": "#1769AA", "standing_equal_peak": "#D97904", "free": "#3C4652"}
MASS_COLORS = {50000.0: "#1769AA", 100000.0: "#C4473A", 150000.0: "#6A5AA6", 200000.0: "#17805D"}


def first_outward_circle_crossing(x0: float, y0: float, vx: float, vy: float,
                                  radius: float = APERTURE_M) -> float:
    """First t >= 0 at which p0 + v*t reaches the circle, or +inf if stationary.

    The function assumes the initial point is on or inside the aperture. A
    cancellation-safe root is used for inward launches; a boundary launch
    moving inward returns the later outward crossing.
    """
    vals = (x0, y0, vx, vy, radius)
    if not all(math.isfinite(v) for v in vals) or radius <= 0:
        raise ValueError("finite coordinates/velocities and positive radius required")
    a = vx * vx + vy * vy
    p2 = x0 * x0 + y0 * y0
    r2 = radius * radius
    scale = max(r2, p2, 1.0e-300)
    tol = 64.0 * math.ulp(scale)
    if p2 > r2 + tol:
        raise ValueError("initial position lies outside the aperture")
    if a == 0.0:
        return math.inf
    b = x0 * vx + y0 * vy
    c = p2 - r2
    if abs(c) <= tol:
        if b >= 0.0:
            return 0.0
        # At the boundary and moving inward: return the opposite-side exit.
        return -2.0 * b / a
    disc = b * b - a * c
    disc_tol = 64.0 * math.ulp(max(b * b, abs(a * c), 1.0e-300))
    if disc < -disc_tol:
        return math.inf
    root = math.sqrt(max(0.0, disc))
    if b >= 0.0:
        # Equivalent to (-b + root)/a, without subtracting nearly equal values.
        t = c / (-b - root)
    else:
        t = (-b + root) / a
    if t < 0.0 and t >= -64.0 * math.ulp(max(abs(t), 1.0)):
        t = 0.0
    return t if t >= 0.0 else math.inf


def retention_at_horizon(crossing_s: float, horizon_s: float) -> str:
    """Classify a free trajectory; a crossing at the horizon counts as escape."""
    if horizon_s < 0 or not math.isfinite(horizon_s):
        raise ValueError("horizon must be finite and nonnegative")
    eps = 32.0 * math.ulp(max(abs(horizon_s), 1.0e-300))
    return "escaped" if crossing_s <= horizon_s + eps else "retained"


def physical_initial_state(row: dict[str, str]) -> dict[str, float]:
    """Reconstruct the exact V2 launch from stored r, azimuth, speed and sign."""
    r0 = float(row["r0_m"])
    az = float(row["azimuth_rad"])
    speed = float(row["speed_m_s"])
    sign = float(row["velocity_sign"])
    if speed < 0 or sign not in (-1.0, 1.0):
        raise ValueError("invalid V2 tangential launch metadata")
    vt = sign * speed
    return {
        "x0_m": r0 * math.cos(az),
        "y0_m": r0 * math.sin(az),
        "vx0_m_s": -vt * math.sin(az),
        "vy0_m_s": vt * math.cos(az),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_start_key(row: dict[str, str]) -> tuple:
    return tuple(float(row[k]) for k in (
        "mz_Th", "r0_m", "energy_eV", "azimuth_rad", "rf_phase_rad",
        "velocity_sign", "speed_m_s")) + (int(float(row["horizon_RF"])),)


def _format(v: float) -> str:
    return f"{v:.9g}"


def audit_inputs() -> tuple[list[dict], dict, list[dict]]:
    """Audit the V2 result CSV against its append-only checkpoint journal."""
    with (V2 / "study_config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    csv_rows = read_csv(V2 / "trajectory_results.csv")
    journal = []
    with (V2 / "trajectory_checkpoint.jsonl").open(encoding="utf-8-sig") as checkpoint:
        for line_no, line in enumerate(checkpoint, 1):
            if not line.strip():
                continue
            try:
                journal.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid checkpoint JSON at line {line_no}: {e}") from e

    valid = [r for r in journal if r.get("status") in ("retained", "escaped") and r.get("job_id")]
    checkpoint_unresolved = [r for r in journal if r.get("status") not in ("retained", "escaped")]
    journal_by_id: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        journal_by_id[r["job_id"]].append(r)
    conflicts = {jid: sorted({r["status"] for r in items}) for jid, items in journal_by_id.items()
                 if len({r["status"] for r in items}) > 1}
    csv_by_id: dict[str, dict] = {}
    csv_valid_input = [r for r in csv_rows if r.get("status") in ("retained", "escaped")]
    csv_duplicates: dict[str, list[dict]] = defaultdict(list)
    for r in csv_valid_input:
        if r.get("status") not in ("retained", "escaped"):
            continue
        if "unit_error" in r.get("normalization", "").lower() or "unit_error" in r.get("job_id", "").lower():
            continue
        csv_by_id[r["job_id"]] = r
        csv_duplicates[r["job_id"]].append(r)
    csv_conflicts = [jid for jid, dup in csv_duplicates.items() if len({x["status"] for x in dup}) > 1]
    if conflicts:
        raise ValueError(f"conflicting valid statuses in checkpoint: {len(conflicts)} job IDs")
    if csv_conflicts:
        raise ValueError(f"conflicting duplicate statuses in V2 CSV: {len(csv_conflicts)} job IDs")
    if set(csv_by_id) != set(journal_by_id):
        raise ValueError("trajectory_results.csv job IDs do not match valid checkpoint IDs")
    mismatch = [jid for jid, r in csv_by_id.items() if r["status"] != journal_by_id[jid][-1]["status"]]
    if mismatch:
        raise ValueError(f"CSV/checkpoint latest status mismatch: {len(mismatch)} job IDs")

    valid_rows = list(csv_by_id.values())
    unresolved = [r for r in csv_rows if not r.get("status", "").startswith(("retained", "escaped"))]
    energy_relerr = []
    for r in valid_rows:
        mz = float(r["mz_Th"])
        mass = mz * AMU_KG  # charge state is +1, so Th maps to physical ion mass.
        speed = float(r["speed_m_s"])
        e_calc = 0.5 * mass * speed * speed / EV_J
        e_req = float(r["energy_eV"])
        energy_relerr.append(abs(e_calc - e_req) / max(abs(e_req), 1e-30))
        if int(float(r["horizon_RF"])) != 300:
            raise ValueError("V2 primary table contains an unexpected RF horizon")
        if r["model"] not in MODEL_ORDER:
            raise ValueError(f"unexpected model in V2 trajectory table: {r['model']}")
    starts: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in valid_rows:
        starts[make_start_key(r)][r["model"]] = r
    unpaired = [key for key, mods in starts.items() if set(mods) != set(MODEL_ORDER)]
    if unpaired:
        raise ValueError(f"unpaired V2 starts: {len(unpaired)}")
    audit = {
        "source_csv": str((V2 / "trajectory_results.csv").relative_to(PROJECT)),
        "source_checkpoint": str((V2 / "trajectory_checkpoint.jsonl").relative_to(PROJECT)),
        "checkpoint_records": len(journal),
        "valid_checkpoint_records_including_repeats": len(valid),
        "unique_valid_job_ids": len(journal_by_id),
        "duplicate_valid_checkpoint_rows": len(valid) - len(journal_by_id),
        "conflicting_duplicate_job_ids": len(conflicts),
        "csv_valid_unique_rows": len(csv_by_id),
        "csv_valid_rows_before_job_id_dedup": len(csv_valid_input),
        "duplicate_valid_csv_rows": len(csv_valid_input) - len(csv_by_id),
        "conflicting_duplicate_csv_job_ids": len(csv_conflicts),
        "csv_checkpoint_job_id_sets_equal": set(csv_by_id) == set(journal_by_id),
        "csv_status_matches_last_valid_checkpoint_record": True,
        "unresolved_rows_in_v2_csv": len(unresolved),
        "unresolved_or_failed_checkpoint_rows": len(checkpoint_unresolved),
        "paired_unique_physical_starts": len(starts),
        "all_physical_starts_have_both_rf_models": not unpaired,
        "models": dict(Counter(r["model"] for r in valid_rows)),
        "rf_status_counts": dict(Counter(r["status"] for r in valid_rows)),
        "horizons_RF": dict(Counter(int(float(r["horizon_RF"])) for r in valid_rows)),
        "nonzero_energy_starts": sum(1 for k in starts if k[2] > 0),
        "zero_energy_radial_starts": sum(1 for k in starts if k[2] == 0),
        "max_relative_energy_conversion_error": max(energy_relerr, default=0),
        "physical_mass_conversion": "m_ion = (m/z in Th) * atomic_mass_unit for charge state +1; V2 speed checked against E=mv^2/2",
        "unit_error_auxiliary_records_excluded": 2,
        "v2_field_and_rf_conditions": {
            "frequency_Hz": config["frequency_Hz"],
            "aperture_radius_m": config["aperture_radius_m"],
            "equal_peak_voltage_V": config["equal_peak_voltage_V"],
            "charge_state": config["charge_state"],
            "source_field": config["source_field"],
        },
    }
    return valid_rows, audit, list(starts.keys())


def calculate_outputs() -> dict:
    rows, audit, _ = audit_inputs()
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        groups[make_start_key(r)][r["model"]] = r

    matched, ballistic, energy_cells, radial_cells, escape_pairs = [], [], [], [], []
    cross = Counter()
    outcomes = Counter()
    for key, model_rows in sorted(groups.items()):
        mz, r0, e_req, az, phase, sign, speed, horizon_rf = key
        representative = model_rows["rotating"]
        state = physical_initial_state(representative)
        # Verify both model rows encode identical physical x0,y0,vx0,vy0.
        for m in MODEL_ORDER[1:]:
            other = physical_initial_state(model_rows[m])
            if any(not math.isclose(state[k], other[k], rel_tol=0, abs_tol=2e-15)
                   for k in state):
                raise ValueError(f"model launch mismatch at start {key}")
        t_cross = first_outward_circle_crossing(
            state["x0_m"], state["y0_m"], state["vx0_m_s"], state["vy0_m_s"], APERTURE_M)
        mass_kg = mz * AMU_KG
        row = {
            "start_id": "|".join(_format(v) if isinstance(v, float) else str(v) for v in key),
            "mz_Th": mz, "physical_mass_kg": mass_kg, "r0_m": r0, "energy_eV": e_req,
            "azimuth_rad": az, "rf_phase_rad": phase, "velocity_sign": sign,
            "speed_m_s": speed, "horizon_RF": horizon_rf,
            **state,
            "ballistic_first_escape_s": "" if math.isinf(t_cross) else t_cross,
            "ballistic_first_escape_RF_period": "" if math.isinf(t_cross) else t_cross * F_RF_HZ,
        }
        for h in HORIZONS_RF:
            T = h / F_RF_HZ
            free_status = retention_at_horizon(t_cross, T)
            row[f"rf_off_status_{h}RF"] = free_status
            row[f"rf_off_retained_{h}RF"] = int(free_status == "retained")
        for model in MODEL_ORDER:
            rf = model_rows[model]
            rf_status = rf["status"]
            free_status = row["rf_off_status_300RF"]
            rf_escape = float(rf["escape_period"]) if rf["escape_period"] not in ("", None) else None
            ret_rf = int(rf_status == "retained")
            ret_free = int(free_status == "retained")
            delta = ret_rf - ret_free
            row[f"{model}_status"] = rf_status
            row[f"{model}_escape_RF_period"] = "" if rf_escape is None else rf_escape
            row[f"delta_P_{model}_minus_free"] = delta
            row[f"{model}_outcome_vs_free"] = f"RF_{rf_status}__free_{free_status}"
            cross[(model, rf_status, free_status)] += 1
            outcomes[f"{model}|RF-retained-free-retained"] += int(rf_status == "retained" and free_status == "retained")
            outcomes[f"{model}|RF-retained-free-escaped"] += int(rf_status == "retained" and free_status == "escaped")
            outcomes[f"{model}|RF-escaped-free-retained"] += int(rf_status == "escaped" and free_status == "retained")
            outcomes[f"{model}|RF-escaped-free-escaped"] += int(rf_status == "escaped" and free_status == "escaped")
            if rf_status == "escaped" and free_status == "escaped":
                escape_pairs.append({
                    "model": model, "mz_Th": mz, "r0_m": r0, "energy_eV": e_req,
                    "azimuth_rad": az, "rf_phase_rad": phase, "velocity_sign": sign,
                    "ballistic_escape_RF_period": t_cross * F_RF_HZ,
                    "rf_escape_RF_period": rf_escape,
                    "rf_minus_ballistic_period": rf_escape - t_cross * F_RF_HZ,
                    "initial_start_id": row["start_id"],
                })
        matched.append(row)
        ballistic.append({
            "start_id": row["start_id"], "mz_Th": mz, "r0_m": r0, "energy_eV": e_req,
            "azimuth_rad": az, "rf_phase_rad": phase, "velocity_sign": sign,
            "x0_m": state["x0_m"], "y0_m": state["y0_m"],
            "vx0_m_s": state["vx0_m_s"], "vy0_m_s": state["vy0_m_s"],
            "speed_m_s": speed, "first_outward_crossing_s": "" if math.isinf(t_cross) else t_cross,
            "first_outward_crossing_RF_period": "" if math.isinf(t_cross) else t_cross * F_RF_HZ,
            "status_300RF": row["rf_off_status_300RF"],
            "status_1000RF": row["rf_off_status_1000RF"],
            "status_10000RF": row["rf_off_status_10000RF"],
        })

    # Aggregate retention at exact evaluated energy/radius cells; no interpolation.
    energy_groups: dict[tuple, list[dict]] = defaultdict(list)
    radial_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in matched:
        if float(r["energy_eV"]) > 0:
            energy_groups[(r["mz_Th"], r["energy_eV"])].append(r)
        else:
            radial_groups[(r["mz_Th"], r["r0_m"])].append(r)
    for (mz, energy), items in sorted(energy_groups.items()):
        p_free = sum(x["rf_off_retained_300RF"] for x in items) / len(items)
        for model in MODEL_ORDER:
            ret = sum(x[f"{model}_status"] == "retained" for x in items)
            energy_cells.append({
                "mz_Th": mz, "energy_eV": energy, "energy_meV": energy * 1000,
                "model": model, "n_physical_starts": len(items),
                "rf_retained": ret, "rf_escaped": len(items) - ret,
                "P_ret_RF": ret / len(items), "P_ret_free": p_free,
                "Delta_P_RF_minus_free": ret / len(items) - p_free,
                "free_retained": sum(x["rf_off_retained_300RF"] for x in items),
                "free_escaped": len(items) - sum(x["rf_off_retained_300RF"] for x in items),
            })
    for (mz, radius), items in sorted(radial_groups.items()):
        p_free = sum(x["rf_off_retained_300RF"] for x in items) / len(items)
        for model in MODEL_ORDER:
            ret = sum(x[f"{model}_status"] == "retained" for x in items)
            radial_cells.append({
                "mz_Th": mz, "r0_m": radius, "r0_mm": radius * 1000,
                "model": model, "n_physical_starts": len(items),
                "rf_retained": ret, "rf_escaped": len(items) - ret,
                "P_ret_RF": ret / len(items), "P_ret_free": p_free,
                "Delta_P_RF_minus_free": ret / len(items) - p_free,
            })

    energy_thresholds = build_threshold_table()
    horizon_rows = build_horizon_table()
    v1_long_rows, v1_audit = read_v1_long_horizon()

    energy_starts = [r for r in matched if float(r["energy_eV"]) > 0]
    radial_starts = [r for r in matched if float(r["energy_eV"]) == 0]
    energy_exact_matches = {
        model: sum(r[f"{model}_status"] == r["rf_off_status_300RF"] for r in energy_starts)
        for model in MODEL_ORDER
    }
    radial_counts = {
        model: {
            "starts": len(radial_starts),
            "rf_retained": sum(r[f"{model}_status"] == "retained" for r in radial_starts),
            "rf_escaped_while_free_retained": sum(
                r[f"{model}_status"] == "escaped" and r["rf_off_status_300RF"] == "retained"
                for r in radial_starts),
        } for model in MODEL_ORDER
    }
    escape_shift_stats = {}
    for model in MODEL_ORDER:
        vals = sorted(float(r["rf_minus_ballistic_period"]) for r in escape_pairs if r["model"] == model)
        escape_shift_stats[model] = {
            "n_both_escaped": len(vals),
            "mean_delay_RF_periods": statistics.mean(vals) if vals else None,
            "median_delay_RF_periods": statistics.median(vals) if vals else None,
            "empirical_95th_percentile_delay_RF_periods_nearest_rank": vals[math.ceil(.95*len(vals))-1] if vals else None,
            "min_delay_RF_periods": vals[0] if vals else None,
            "max_delay_RF_periods": vals[-1] if vals else None,
        }

    summary = {
        "analysis": "analytical free-flight control; no RF integration performed",
        "input_audit": audit,
        "ballistic_classification_counts_by_RF_model_and_300RF_outcome": {
            "|".join(k): v for k, v in sorted(cross.items())
        },
        "three_way_outcome_counts_by_model": dict(outcomes),
        "energy_cells": len(energy_cells),
        "radial_cells": len(radial_cells),
        "energy_launches_matching_ballistic_classification": energy_exact_matches,
        "energy_launches_per_model": len(energy_starts),
        "energy_retention_delta_cells_all_zero": all(abs(float(r["Delta_P_RF_minus_free"])) < 1e-12 for r in energy_cells),
        "radial_launch_counts": radial_counts,
        "matched_RF_escape_time_pairs": len(escape_pairs),
        "escape_time_delay_statistics_for_both_escape_only": escape_shift_stats,
        "V2_90_percent_interval_contains_exact_ballistic_cutoff": {
            "brackets_containing_cutoff": sum(bool(r["ballistic_inside_v2_bracket"]) for r in energy_thresholds),
            "brackets_total": len(energy_thresholds),
        },
        "RF_retained_but_ballistically_lost_300RF": {
            model: outcomes[f"{model}|RF-retained-free-escaped"] for model in MODEL_ORDER
        },
        "RF_escaped_but_ballistically_retained_300RF": {
            model: outcomes[f"{model}|RF-escaped-free-retained"] for model in MODEL_ORDER
        },
        "ballistic_energy_cutoff_scaling": "Ecrit proportional to T^-2 for fixed mass, radius and launch direction",
        "v1_valid_1000RF_controls_reused": v1_audit,
        "no_10000RF_octupole_trajectories_found": True,
    }

    write_csv(OUT / "three_way_matched_starts.csv", matched)
    write_csv(OUT / "ballistic_crossing_times.csv", ballistic)
    write_csv(OUT / "energy_retention_comparison.csv", energy_cells)
    write_csv(OUT / "radial_retention_comparison.csv", radial_cells)
    write_csv(OUT / "escape_time_pairs.csv", escape_pairs)
    write_csv(OUT / "energy_threshold_comparison.csv", energy_thresholds)
    write_csv(OUT / "ballistic_horizon_thresholds.csv", horizon_rows)
    write_csv(OUT / "v1_1000RF_matched_control.csv", v1_long_rows)
    (OUT / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_all(energy_cells, radial_cells, escape_pairs, energy_thresholds, matched)
    return summary


def build_threshold_table() -> list[dict]:
    rows = read_csv(V2 / "energy_threshold_intervals.csv")
    result = []
    horizon_s = 300 / F_RF_HZ
    for x in rows:
        if not math.isclose(float(x["threshold_retention"]), 0.9):
            continue
        intervals = json.loads(x["bracket_intervals_eV"])
        if len(intervals) != 1:
            # Preserve multi-interval results as separate rows if ever observed.
            use_intervals = intervals
        else:
            use_intervals = intervals
        mz = float(x["mz_Th"])
        m = mz * AMU_KG
        for j, (lo, hi) in enumerate(use_intervals, 1):
            r0 = 0.001
            ecrit = 0.5 * m * (APERTURE_M ** 2 - r0 ** 2) / horizon_s ** 2 / EV_J
            result.append({
                "model": x["model"], "mz_Th": mz, "threshold_retention": 0.9,
                "interval_id": j, "v2_lower_eV": lo, "v2_upper_eV": hi,
                "v2_lower_meV": lo * 1000, "v2_upper_meV": hi * 1000,
                "ballistic_Ecrit_eV_300RF": ecrit,
                "ballistic_Ecrit_meV_300RF": ecrit * 1000,
                "ballistic_inside_v2_bracket": lo <= ecrit <= hi,
                "r0_m": r0, "aperture_m": APERTURE_M, "horizon_s": horizon_s,
            })
    return result


def build_horizon_table() -> list[dict]:
    rows = []
    for mz in (50000.0, 100000.0, 150000.0, 200000.0):
        for h in HORIZONS_RF:
            T = h / F_RF_HZ
            m = mz * AMU_KG
            ecrit = 0.5 * m * (APERTURE_M ** 2 - 0.001 ** 2) / T ** 2 / EV_J
            rows.append({
                "mz_Th": mz, "horizon_RF": h, "frequency_Hz": F_RF_HZ,
                "horizon_s": T, "r0_m": 0.001, "aperture_m": APERTURE_M,
                "Ecrit_eV_tangential": ecrit,
                "Ecrit_meV_tangential": ecrit * 1000,
                "scaling_from_300RF": (300 / h) ** 2,
                "V2_primary_RF_results_available_at_horizon": h == 300,
                "V1_selected_zero_energy_RF_control_available": h == 1000 and mz in (50000.0, 100000.0),
            })
    return rows


def read_v1_long_horizon() -> tuple[list[dict], dict]:
    """Reuse actual, successful V1 1000-RF control rows; exclude failed attempts."""
    p = V1 / "mass_velocity_results.csv"
    if not p.exists():
        return [], {"source": str(p), "valid_1000RF_rows": 0, "note": "file absent"}
    all_rows = read_csv(p)
    selected = [r for r in all_rows if int(float(r.get("horizon_RF", r.get("periods", 0)) or 0)) == 1000
                and r.get("status") in ("retained", "escaped")
                and r.get("model") in MODEL_ORDER
                and "unit_error" not in r.get("job_id", "").lower()]
    by_id = {r["job_id"]: r for r in selected}
    result = []
    for r in by_id.values():
        mz = float(r["mz_Th"])
        radius = float(r["r0_m"])
        az = float(r["azimuth_rad"])
        vr = float(r["vr_m_s"])
        vt = float(r["vt_m_s"])
        x0, y0 = radius * math.cos(az), radius * math.sin(az)
        vx = vr * math.cos(az) - vt * math.sin(az)
        vy = vr * math.sin(az) + vt * math.cos(az)
        crossing = first_outward_circle_crossing(x0, y0, vx, vy)
        ballistic_status = retention_at_horizon(crossing, 1000 / F_RF_HZ)
        physical_mass = mz * AMU_KG
        energy = 0.5 * physical_mass * (vr * vr + vt * vt) / EV_J
        result.append({
            "source": "V1 mass_velocity_results.csv; reused, no reintegration",
            "job_id": r["job_id"], "model": r["model"], "mz_Th": mz,
            "mass_kg_macro_recorded": float(r["mass_kg"]), "physical_mass_kg": physical_mass,
            "r0_m": radius, "azimuth_rad": az, "rf_phase_rad": float(r["rf_phase_rad"]),
            "vr_m_s": vr, "vt_m_s": vt, "energy_eV_recomputed_physical": energy,
            "rf_horizon_RF": 1000, "rf_status": r["status"],
            "rf_escape_RF_period": r["escape_period"],
            "ballistic_escape_RF_period": "" if math.isinf(crossing) else crossing * F_RF_HZ,
            "ballistic_status_at_1000RF": ballistic_status,
            "normalization": r["normalization"],
        })
    cp = V1 / "campaign_checkpoint.jsonl"
    checkpoint_attempts = []
    if cp.exists():
        for line in cp.read_text(encoding="utf-8-sig").splitlines():
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(q.get("horizon_RF", 0)) == 1000:
                checkpoint_attempts.append(q)
    invalid = sum(not str(x.get("status", "")).startswith(("retained", "escaped")) for x in checkpoint_attempts)
    audit = {
        "source": str(p.relative_to(PROJECT)),
        "valid_1000RF_rows": len(result),
        "unique_valid_job_ids": len(by_id),
        "RF_statuses": dict(Counter(x["rf_status"] for x in result)),
        "checkpoint_1000RF_attempt_rows": len(checkpoint_attempts),
        "failed_checkpoint_attempt_rows_excluded": invalid,
        "scope": "selected zero-energy radial controls at 50 and 100 kTh, r0=0.5/2 mm; separate from V2 energy grid",
    }
    return result, audit


# Minimal vector scene renderer: vector SVG/PDF and high-resolution PNG without
# adding plotting packages to the audited Python environment.
class Scene:
    def __init__(self, width=1120, height=760, title=""):
        self.w, self.h, self.title = width, height, title
        self.ops = []

    def line(self, x1, y1, x2, y2, color="#222222", width=1, dash=None):
        self.ops.append(("line", x1, y1, x2, y2, color, width, dash))

    def rect(self, x, y, w, h, fill="none", stroke="none", width=1):
        self.ops.append(("rect", x, y, w, h, fill, stroke, width))

    def circle(self, x, y, r, fill="#222222", stroke="none", width=1):
        self.ops.append(("circle", x, y, r, fill, stroke, width))

    def text(self, x, y, txt, size=14, color="#222222", anchor="start", bold=False):
        self.ops.append(("text", x, y, str(txt), size, color, anchor, bold))

    def polyline(self, pts, color="#222222", width=1, dash=None):
        self.ops.append(("polyline", tuple(pts), color, width, dash))

    def save(self, stem: str):
        # SVG
        esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">',
               '<rect width="100%" height="100%" fill="#ffffff"/>']
        if self.title:
            svg.append(f'<text x="{self.w/2}" y="26" text-anchor="middle" dominant-baseline="hanging" font-family="Arial,sans-serif" font-size="19" font-weight="700" fill="#20252b">{esc(self.title)}</text>')
        for op in self.ops:
            if op[0] == "line":
                _,x1,y1,x2,y2,c,w,d=op; ds=f' stroke-dasharray="{d}"' if d else ""
                svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}"{ds}/>')
            elif op[0] == "rect":
                _,x,y,w,h,fill,stroke,lw=op
                svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
            elif op[0] == "circle":
                _,x,y,r,fill,stroke,lw=op
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
            elif op[0] == "text":
                _,x,y,t,size,c,a,b=op
                svg.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{a}" dominant-baseline="hanging" font-family="Arial,sans-serif" font-size="{size}" font-weight="{700 if b else 400}" fill="{c}">{esc(t)}</text>')
            elif op[0] == "polyline":
                _,pts,c,w,d=op; ds=f' stroke-dasharray="{d}"' if d else ""
                p=" ".join(f"{x:.2f},{y:.2f}" for x,y in pts)
                svg.append(f'<polyline points="{p}" fill="none" stroke="{c}" stroke-width="{w}"{ds}/>')
        svg.append("</svg>")
        Path(str(stem)+".svg").write_text("\n".join(svg), encoding="utf-8")

        # PDF (vector)
        c = pdfcanvas.Canvas(str(stem)+".pdf", pagesize=(self.w, self.h), pageCompression=1)
        c.setTitle(self.title or Path(stem).name)
        c.setFillColorRGB(1,1,1); c.rect(0,0,self.w,self.h,fill=1,stroke=0)
        for op in self.ops:
            if op[0] == "line":
                _,x1,y1,x2,y2,col,lw,d=op; _pdf_color(c,col,stroke=True); c.setLineWidth(lw); c.setDash(d if d else [])
                c.line(x1,self.h-y1,x2,self.h-y2); c.setDash([])
            elif op[0] == "rect":
                _,x,y,w,h,fill,stroke,lw=op
                if fill != "none": _pdf_color(c,fill); c.rect(x,self.h-y-h,w,h,fill=1,stroke=0)
                if stroke != "none": _pdf_color(c,stroke,stroke=True); c.setLineWidth(lw); c.rect(x,self.h-y-h,w,h,fill=0,stroke=1)
            elif op[0] == "circle":
                _,x,y,r,fill,stroke,lw=op
                if fill != "none": _pdf_color(c,fill); c.circle(x,self.h-y,r,fill=1,stroke=0)
                if stroke != "none": _pdf_color(c,stroke,stroke=True); c.setLineWidth(lw); c.circle(x,self.h-y,r,fill=0,stroke=1)
            elif op[0] == "text":
                _,x,y,t,size,col,anchor,bold=op; _pdf_color(c,col)
                font="Helvetica-Bold" if bold else "Helvetica"; c.setFont(font,size)
                xx=x
                if anchor=="middle": xx=x-stringWidth(t,font,size)/2
                elif anchor=="end": xx=x-stringWidth(t,font,size)
                c.drawString(xx,self.h-y-size, t)
            elif op[0] == "polyline":
                _,pts,col,lw,d=op; _pdf_color(c,col,stroke=True); c.setLineWidth(lw); c.setDash(d if d else [])
                p=c.beginPath(); p.moveTo(pts[0][0],self.h-pts[0][1])
                for x,y in pts[1:]: p.lineTo(x,self.h-y)
                c.drawPath(p,stroke=1,fill=0); c.setDash([])
        c.save()

        # High-resolution PNG (raster)
        scale=2
        im=Image.new("RGB",(self.w*scale,self.h*scale),"white")
        dr=ImageDraw.Draw(im)
        regular=_font(14*scale,False); boldf=_font(14*scale,True)
        if self.title: dr.text((self.w*scale/2,8*scale),self.title,font=_font(19*scale,True),fill="#20252b",anchor="mt")
        for op in self.ops:
            if op[0]=="line":
                _,x1,y1,x2,y2,col,lw,d=op
                if d:
                    _pil_dashed_line(dr,(x1*scale,y1*scale),(x2*scale,y2*scale),col,lw*scale,d,scale)
                else: dr.line((x1*scale,y1*scale,x2*scale,y2*scale),fill=col,width=max(1,int(lw*scale)))
            elif op[0]=="rect":
                _,x,y,w,h,fill,stroke,lw=op; box=(x*scale,y*scale,(x+w)*scale,(y+h)*scale)
                dr.rectangle(box,fill=None if fill=="none" else fill,outline=None if stroke=="none" else stroke,width=max(1,int(lw*scale)))
            elif op[0]=="circle":
                _,x,y,r,fill,stroke,lw=op; box=((x-r)*scale,(y-r)*scale,(x+r)*scale,(y+r)*scale)
                dr.ellipse(box,fill=None if fill=="none" else fill,outline=None if stroke=="none" else stroke,width=max(1,int(lw*scale)))
            elif op[0]=="text":
                _,x,y,t,size,col,anchor,b=op; font=_font(size*scale,b)
                pil_anchor={"start":"lt","middle":"mt","end":"rt"}[anchor]
                dr.text((x*scale,y*scale),t,font=font,fill=col,anchor=pil_anchor)
            elif op[0]=="polyline":
                _,pts,col,lw,d=op
                if d:
                    for a,b in zip(pts,pts[1:]): _pil_dashed_line(dr,(a[0]*scale,a[1]*scale),(b[0]*scale,b[1]*scale),col,lw*scale,d,scale)
                else: dr.line([(x*scale,y*scale) for x,y in pts],fill=col,width=max(1,int(lw*scale)),joint="curve")
        im.save(str(stem)+".png",dpi=(220,220))


def _font(size, bold=False):
    p=Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    if not p.exists(): p=Path("C:/Windows/Fonts/segoeui.ttf")
    try: return ImageFont.truetype(str(p),int(size))
    except OSError: return ImageFont.load_default()


def _pil_dashed_line(draw, p1, p2, color, width, dash, scale):
    x1,y1=p1; x2,y2=p2; length=math.hypot(x2-x1,y2-y1)
    if length==0: return
    unit=( (x2-x1)/length, (y2-y1)/length ); pat=[float(x)*scale for x in dash]; pos=0.; i=0
    while pos<length:
        end=min(length,pos+pat[i%len(pat)])
        if i%2==0: draw.line((x1+unit[0]*pos,y1+unit[1]*pos,x1+unit[0]*end,y1+unit[1]*end),fill=color,width=max(1,int(width)))
        pos=end; i+=1


def _pdf_color(c, color, stroke=False):
    h=color.lstrip("#"); r,g,b=[int(h[i:i+2],16)/255 for i in (0,2,4)]
    (c.setStrokeColorRGB if stroke else c.setFillColorRGB)(r,g,b)


def axes(scene, x, y, w, h, xlim, ylim, xticks, yticks, xlabel, ylabel, title="", xfmt=None, yfmt=None):
    left=x+62; right=x+w-20; top=y+35; bottom=y+h-52
    X=lambda z:left+(z-xlim[0])/(xlim[1]-xlim[0])*(right-left)
    Y=lambda z:bottom-(z-ylim[0])/(ylim[1]-ylim[0])*(bottom-top)
    scene.line(left,top,left,bottom,"#34404a",1.2); scene.line(left,bottom,right,bottom,"#34404a",1.2)
    for t in yticks:
        yy=Y(t); scene.line(left,yy,right,yy,"#e2e6e9",0.7); scene.line(left-4,yy,left,yy,"#34404a",1)
        lab=yfmt(t) if yfmt else f"{t:g}"; scene.text(left-9,yy-7,lab,11,"#303840","end")
    for t in xticks:
        xx=X(t); scene.line(xx,bottom,xx,bottom+4,"#34404a",1)
        lab=xfmt(t) if xfmt else f"{t:g}"; scene.text(xx,bottom+8,lab,11,"#303840","middle")
    scene.text((left+right)/2,bottom+31,xlabel,12,"#20252b","middle")
    scene.text(left,top-22,ylabel,12,"#20252b","start")
    if title: scene.text((left+right)/2,y+4,title,14,"#20252b","middle",True)
    return X,Y,(left,right,top,bottom)


def plot_all(energy_cells, radial_cells, escape_pairs, thresholds, matched):
    plot_energy(energy_cells)
    plot_thresholds(thresholds)
    plot_radial(radial_cells)
    plot_escape(escape_pairs, matched)
    plot_benefit(energy_cells, radial_cells)


def plot_energy(cells):
    sc=Scene(height=860,title="RF retention and exact ballistic control by transverse energy")
    masses=(50000.,100000.,150000.,200000.)
    for i,mz in enumerate(masses):
        col=i%2; row=i//2; x=22+col*550; y=50+row*375
        X,Y,_=axes(sc,x,y,530,330,(0,10),(0,1),[0,2,4,6,8,10],[0,.5,1],"Initial transverse energy (meV)","Retained fraction",f"m/z = {mz/1000:g} kTh",lambda v:f"{v:g}")
        ecrit=0.5*(mz*AMU_KG)*(APERTURE_M**2-.001**2)/(300/F_RF_HZ)**2/EV_J*1000
        sc.line(X(ecrit),Y(0),X(ecrit),Y(1),"#323a42",1.3,[5,4])
        these=[a for a in cells if float(a["mz_Th"])==mz]
        for a in these:
            ex=float(a["energy_meV"]); vals=[("free",float(a["P_ret_free"])),("standing_equal_peak",float(a["P_ret_RF"])),("rotating",float(a["P_ret_RF"]))]
            # Concentric symbols preserve all three coincident observations without jittering x/y.
            for name,p in vals:
                xx,yy=X(ex),Y(p)
                if name=="free": sc.circle(xx,yy,7,"#ffffff",MODEL_COLOR["free"],1.8)
                elif name=="standing_equal_peak": sc.rect(xx-5,yy-5,10,10,"#ffffff",MODEL_COLOR[name],1.8)
                else: sc.circle(xx,yy,3.8,MODEL_COLOR[name],"#ffffff",0.6)
    ly=812
    for j,(label,color,shape) in enumerate((("Rotating",MODEL_COLOR["rotating"],"circle"),("Standing",MODEL_COLOR["standing_equal_peak"],"square"),("RF off",MODEL_COLOR["free"],"open"),("Exact ballistic Ecrit", "#323a42","line"))):
        xx=280+j*145
        if shape=="circle":sc.circle(xx,ly,4,color,"#ffffff",.5)
        elif shape=="square":sc.rect(xx-4,ly-4,8,8,"#ffffff",color,1.5)
        elif shape=="open":sc.circle(xx,ly,6,"#ffffff",color,1.5)
        else:sc.line(xx-8,ly,xx+8,ly,color,1.3,[5,4])
        sc.text(xx+11,ly-7,label,11,"#303840")
    sc.text(560,840,"16 deterministic starts per nonzero energy shell. Equal results coincide; markers are overlaid without x/y jitter.",11,"#444b52","middle")
    sc.save(str(OUT/"figure_1_energy_retention"))


def plot_thresholds(rows):
    sc=Scene(title="V2 energy brackets and exact RF-off ballistic cutoff")
    X,Y,_=axes(sc,100,70,920,570,(40,210),(0,8.2),[50,100,150,200],[0,1,2,3,4,5,6,7,8],"Mass-to-charge ratio (kTh)","Transverse energy (meV)","300 RF periods; r0 = 1 mm",lambda v:f"{v:g}",lambda v:f"{v:g}")
    # Exact ballistic law: proportional to m/z at fixed charge state, aperture and time.
    linepts=[]
    for mz in (50000,200000):
        ecrit=.5*mz*AMU_KG*(APERTURE_M**2-.001**2)/(300/F_RF_HZ)**2/EV_J*1000
        linepts.append((X(mz/1000),Y(ecrit)))
    sc.polyline(linepts,"#252a30",2.0)
    for r in rows:
        mz=float(r["mz_Th"]); model=r["model"]
        # Tiny horizontal displacement separates coincident RF model intervals.
        xx=X(mz/1000)+( -2.0 if model=="rotating" else 2.0 )
        lo=float(r["v2_lower_meV"]); hi=float(r["v2_upper_meV"])
        color=MODEL_COLOR[model]
        sc.line(xx,Y(lo),xx,Y(hi),color,3)
        sc.line(xx-5,Y(lo),xx+5,Y(lo),color,2); sc.line(xx-5,Y(hi),xx+5,Y(hi),color,2)
        e=float(r["ballistic_Ecrit_meV_300RF"])
        sc.circle(X(mz/1000),Y(e),4,"#252a30","#ffffff",.8)
    lx,ly=300,690
    for j,(lab,c) in enumerate((("Rotating V2 90% bracket",MODEL_COLOR["rotating"]),("Standing V2 90% bracket",MODEL_COLOR["standing_equal_peak"]),("Exact ballistic Ecrit", "#252a30"))):
        xx=lx+j*255
        sc.line(xx,ly,xx+22,ly,c,3);sc.text(xx+29,ly-7,lab,12,"#303840")
    sc.text(560,724,"Both V2 50% and 90% thresholds share these observed intervals; intervals are not point estimates.",11,"#444b52","middle")
    sc.save(str(OUT/"figure_2_energy_thresholds"))


def heat_color(v):
    # Calm sequential scale for 0..1.
    stops=[(0.0,(242,244,246)),(.5,(113,169,204)),(1.0,(23,105,170))]
    for (a,ca),(b,cb) in zip(stops,stops[1:]):
        if v<=b:
            u=max(0,min(1,(v-a)/(b-a))); rgb=tuple(round(ca[i]+(cb[i]-ca[i])*u) for i in range(3)); return "#%02x%02x%02x"%rgb
    return "#1769aa"


def delta_color(v):
    if abs(v)<1e-12:return "#eef0f2"
    # Negative RF-minus-free differences are warm brown; positive would be blue.
    if v<0:
        t=min(1,abs(v)); a=(245,237,224); b=(169,82,35)
    else:
        t=min(1,v); a=(226,240,248); b=(23,105,170)
    rgb=tuple(round(a[i]+(b[i]-a[i])*t) for i in range(3));return "#%02x%02x%02x"%rgb


def plot_radial(cells):
    sc=Scene(height=840,title="Direct mass-radius acceptance points: RF fields versus free flight")
    masses=(50000.,100000.,150000.,200000.)
    panels=[("rotating","Rotating: P_ret"),("standing_equal_peak","Standing: P_ret"),("free","RF off: P_ret"),("delta","Delta P_ret: RF minus free")]
    for i,(model,title) in enumerate(panels):
        col=i%2;row=i//2;x=28+col*545;y=54+row*345
        X,Y,_=axes(sc,x,y,520,310,(0,8.1),(40,210),[0,2,4,6,8],[50,100,150,200],"Initial radius r0 (mm)","Mass-to-charge ratio (kTh)",title,lambda v:f"{v:g}",lambda v:f"{v:g}")
        for d in cells:
            mz=float(d["mz_Th"]);r=float(d["r0_mm"]);xx=X(r);yy=Y(mz/1000)
            n=int(d["n_physical_starts"])
            if model=="free":
                val=float(d["P_ret_free"]); color=heat_color(val); marker="circle"
            elif model=="delta":
                for mdl,offset,marker in (("rotating",-.025,"circle"),("standing_equal_peak",.025,"square")):
                    rr=X(r+offset); val=float(d["Delta_P_RF_minus_free"]) if d["model"]==mdl else None
                    if val is not None: sc.circle(rr,yy,5,delta_color(val),MODEL_COLOR[mdl],1.3) if marker=="circle" else sc.rect(rr-4,yy-4,8,8,delta_color(val),MODEL_COLOR[mdl],1.3)
                continue
            elif d["model"]!=model: continue
            else:
                val=float(d["P_ret_RF"]);color=heat_color(val);marker="circle" if n<=4 else "square"
            if marker=="circle": sc.circle(xx,yy,2.2,color,"#26343e",.7)
            else: sc.rect(xx-2.8,yy-2.8,5.6,5.6,color,"#26343e",.7)
    # Shared retention color scale.
    sc.text(330,733,"P_ret:",11,"#343b42","end",True)
    for i,v in enumerate((0.0,0.5,1.0)):
        xx=350+i*85;sc.rect(xx,731,16,16,heat_color(v),"#26343e",.7);sc.text(xx+23,733,f"{v:g}",11,"#343b42")
    sc.text(710,733,"Delta shade: -1 to 0; symbols: circles / squares identify RF model in Delta panel.",10,"#444b52")
    sc.text(560,780,"Retention panels: circle n=4, square n=24; in Delta panel marks are offset +/-0.07 mm only for display.",11,"#444b52","middle")
    sc.text(560,812,"Zero-speed RF-off starts are stationary; any RF escape is field-induced loss, not confinement gain.",11,"#444b52","middle")
    sc.save(str(OUT/"figure_3_mass_radius_acceptance"))


def plot_escape(pairs, matched):
    sc=Scene(title="RF escape delay relative to exact ballistic crossing")
    censor_counts={}
    for model in MODEL_ORDER:
        censor_counts[model]=sum(float(r["energy_eV"])==0 and r[f"{model}_status"]=="escaped" and r["rf_off_status_300RF"]=="retained" for r in matched)
    for i,model in enumerate(MODEL_ORDER):
        x=40+i*545;y=74
        X,Y,_=axes(sc,x,y,510,530,(0,305),(0,2.6),[0,60,120,180,240,300],[0,.5,1,1.5,2,2.5],"Ballistic first crossing (RF periods)","RF minus ballistic time (RF periods)",MODEL_LABEL[model],lambda v:f"{v:g}",lambda v:f"{v:g}")
        sc.line(X(0),Y(0),X(300),Y(0),"#414951",1.2,[5,4])
        ps=[r for r in pairs if r["model"]==model]
        for r in ps:
            mz=float(r["mz_Th"]); color=MASS_COLORS.get(mz,"#1769aa")
            sc.circle(X(float(r["ballistic_escape_RF_period"])),Y(float(r["rf_minus_ballistic_period"])),3.2,color,"#ffffff",.5)
        sc.text(x+72,607,f"Both escaped: n={len(ps)}; RF escaped/free retained: n={censor_counts[model]}",10,"#303840")
        sc.text(x+72,625,"RF-retained escape times are censored, not set to 300.",10,"#555d65")
    # mass legend
    for i,(mz,c) in enumerate(MASS_COLORS.items()):
        xx=250+i*170;sc.circle(xx,670,4,c,"#ffffff",.5);sc.text(xx+9,663,f"{mz/1000:g} kTh",11,"#333b42")
    sc.text(560,720,"Points show matched cases with finite escape times in both systems; no RF-retained value is imputed.",11,"#444b52","middle")
    sc.save(str(OUT/"figure_4_escape_time_comparison"))


def plot_benefit(energy_cells, radial_cells):
    sc=Scene(title="Measured change in 300-period retention relative to RF-off")
    models=("rotating","standing_equal_peak")
    for j,model in enumerate(models):
        title=MODEL_LABEL[model]
        for rowkind in ("energy","radial"):
            row=0 if rowkind=="energy" else 1
            x=25+j*548;y=55+row*337
            if rowkind=="energy":
                X,Y,_=axes(sc,x,y,525,310,(0,10),(40,210),[0,2,4,6,8,10],[50,100,150,200],"Initial energy (meV), r0 = 1 mm","m/z (kTh)",f"{title}: energy cells",lambda v:f"{v:g}",lambda v:f"{v/1000:g}")
                for a in energy_cells:
                    if a["model"]!=model: continue
                    xx=X(float(a["energy_meV"])); yy=Y(float(a["mz_Th"])/1000); val=float(a["Delta_P_RF_minus_free"])
                    sc.circle(xx,yy,5,delta_color(val),"#3b4248",.8)
            else:
                X,Y,_=axes(sc,x,y,525,310,(0,8.1),(40,210),[0,2,4,6,8],[50,100,150,200],"Initial radius r0 (mm), E = 0","m/z (kTh)",f"{title}: radial cells",lambda v:f"{v:g}",lambda v:f"{v/1000:g}")
                for a in radial_cells:
                    if a["model"]!=model: continue
                    xx=X(float(a["r0_mm"])); yy=Y(float(a["mz_Th"])/1000); val=float(a["Delta_P_RF_minus_free"])
                    rad=7 if int(a["n_physical_starts"])>4 else 5
                    sc.circle(xx,yy,rad,delta_color(val),"#3b4248",.8)
    # Signed legend is categorical and preserves the exact evaluated-point grain.
    lx,ly=290,704
    for i,(val,label) in enumerate(((-1.0,"-1"),(-.5,"-0.5"),(0.0,"0"))):
        xx=lx+i*115;sc.circle(xx,ly,6,delta_color(val),"#3b4248",.8);sc.text(xx+13,ly-7,label,11,"#333b42")
    sc.text(662,697,"Delta P = P_RF - P_free; radial marker size distinguishes n=4 from n=24.",10,"#444b52")
    sc.text(560,744,"All marks are directly evaluated cells; no interpolation. Negative values indicate fewer survivors than free flight.",11,"#444b52","middle")
    sc.save(str(OUT/"figure_5_rf_confinement_benefit"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary=calculate_outputs()
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
