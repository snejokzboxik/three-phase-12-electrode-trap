"""Reanalyse the 11 already solved displaced P4 geometries; performs no FEM solve."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "outputs/octupole_comparison_home/electrode_displacement_sensitivity_v1"
OUT = Path(__file__).resolve().parent
CALC = OLD / "calculations"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def minimum_scaling():
    validation = json.loads((OLD / "processed/effective_minimum_method_validation.json").read_text(encoding="utf-8"))
    h35, h50 = validation["h35_best_radius_mm"], validation["h50_best_radius_mm"]
    rows = []
    for amplitude in (10, 25, 50, 100):
        d = json.loads((CALC / f"radial_out_{amplitude}um_h50/field_diagnostics.json").read_text(encoding="utf-8"))
        m = d["rotating_pseudopotential_minimum"]
        radius = h50 if amplitude == 10 else float(m["radius_mm"])
        if amplitude == 10:
            point = [-0.23656916693688568, -0.37786421916759955]
            e2 = float(m["rotating_field_squared_at_min_V2_m2_per_V2"])
        else:
            point = [float(v) * 1e3 for v in m["position_m"]]
            e2 = float(m["rotating_field_squared_at_min_V2_m2_per_V2"])
        rows.append({"electrode_displacement_um": amplitude, "minimum_radius_mm": radius,
            "minimum_x_mm": point[0], "minimum_y_mm": point[1], "field_squared_per_V2": e2,
            "origin_field_squared_per_V2": float(m["value_at_origin_V2_m2_per_V2"]),
            "minimum_method": "validated h35/h50 multi-method result" if amplitude == 10 else "existing h50 field-analysis minimum"})
    x = np.log([r["electrode_displacement_um"] for r in rows])
    y = np.log([r["minimum_radius_mm"] for r in rows])
    fit = linregress(x, y)
    predicted = [rows[0]["minimum_radius_mm"] * (r["electrode_displacement_um"] / 10) ** (1 / 3) for r in rows]
    for row, cubic in zip(rows, predicted):
        row["r_min_cubic_prediction_mm_from_10um"] = cubic
        row["relative_deviation_from_cubic_prediction"] = row["minimum_radius_mm"] / cubic - 1
    return rows, {"log_log_exponent": float(fit.slope), "log_log_intercept": float(fit.intercept),
        "r_squared": float(fit.rvalue ** 2), "slope_standard_error": float(fit.stderr),
        "n_amplitudes": len(rows), "fit_scope": "four radial-out h50 amplitudes, 10/25/50/100 um",
        "cubic_prediction_reference": "r_min(Delta)=r_min(10 um)*(Delta/10 um)^(1/3)",
        "h35_h50_10um_radii_mm": {"h35": h35, "h50": h50,
            "absolute_difference_mm": abs(h35-h50),
            "relative_to_mean": abs(h35-h50)/((h35+h50)/2)},
        "interpretation": "The four-point exponent is descriptive and does not establish an asymptotic power law."}


def harmonic_scaling():
    rows = read_csv(OLD / "processed/azimuthal_harmonics.csv")
    radial = [r for r in rows if r["resolution"] == "h50" and r["pattern"].startswith("radial_out_")]
    out = []
    by = {(int(r["pattern"].split("_")[-1].replace("um", "")), float(r["radius_mm"]), int(r["azimuthal_order_m"])): float(r["displaced_abs_coefficient_V_per_V"]) for r in radial}
    for radius in (0.5, 1.0, 2.0):
        for mode in (1, 4):
            values = [(a, by[(a, radius, mode)]) for a in (10, 25, 50, 100)]
            slope = linregress(np.log([x[0] for x in values]), np.log([x[1] for x in values]))
            out.append({"radius_mm": radius, "azimuthal_mode_m": mode,
                "amplitude_scaling_exponent_vs_displacement": float(slope.slope),
                "fit_r_squared": float(slope.rvalue ** 2),
                "coefficient_at_10um_V_per_V": values[0][1],
                "coefficient_at_100um_V_per_V": values[-1][1],
                "radial_displacement_samples_um": "10;25;50;100",
                "interpretation": "m=1 is the leading symmetry-breaking dipolar potential term; m=4 is the nominal leading octupolar potential term" if mode == 1 else "m=4 octupolar potential term"})
    radial_orders = []
    for mode in (1, 4):
        values = [(r, by[(10, r, mode)]) for r in (0.5, 1.0, 2.0)]
        slope = linregress(np.log([x[0] for x in values]), np.log([x[1] for x in values]))
        radial_orders.append({"displacement_um": 10, "azimuthal_mode_m": mode,
            "potential_coefficient_radial_exponent": float(slope.slope), "r_squared": float(slope.rvalue ** 2),
            "radius_samples_mm": "0.5;1;2"})
    return out, radial_orders


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    minima, fit = minimum_scaling()
    harmonics, radial_orders = harmonic_scaling()
    write_csv(OUT / "effective_minimum_scaling.csv", minima)
    write_csv(OUT / "harmonic_amplitude_scaling.csv", harmonics)
    write_csv(OUT / "harmonic_radial_order.csv", radial_orders)
    index = json.loads((OLD / "processed/field_analysis_index.json").read_text(encoding="utf-8"))
    pilot_minima=[]
    for case in index["cases"]:
        if case["resolution"]!="h50": continue
        pos=case.get("minimum_position_mm")
        if pos is None: continue
        pilot_minima.append({"pattern":case["pattern"],"resolution":"h50",
            "reported_x_mm":float(pos[0]),"reported_y_mm":float(pos[1]),
            "reported_radius_mm":float(math.hypot(*pos)),
            "scope":"previously reported search point; not a newly established global minimum"})
    write_csv(OUT/"existing_pilot_minima_reference.csv",pilot_minima)
    resolution = json.loads((OLD / "processed/resolution_sensitivity_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((OLD / "processed/effective_minimum_method_validation.json").read_text(encoding="utf-8"))
    summary = {"input": "Only existing FEM P4 solutions and processed V1 pilot CSV/JSON; no FEM solve and no RF integration.",
        "displaced_geometries_reused": 11, "solved_fields_reused": 12,
        "minimum_scaling": fit, "harmonic_displacement_scaling": harmonics,
        "harmonic_radial_order": radial_orders,"existing_pilot_minima_reference":pilot_minima,
        "validated_10um_minimum": {k: validation[k] for k in ("h35_best_radius_mm", "h50_best_radius_mm", "cross_resolution_radius_difference_mm", "cross_resolution_objective_relative_difference", "mirror_degeneracy_note", "validation_status", "interpretation_limit")},
        "field_mesh_sensitivity": resolution,
        "local_minima_and_saddles": {"known_local_minima": "At 10 um radial outward, the previous multi-start validation found the global mirror-related pair near radius 0.44581 mm and a higher local-minimum basin near radius 0.45811 mm along the electrode axis.", "saddle_and_barrier_status": "Not established by the existing files. A local extremum of the quadrature-field norm is an effective-potential stationary point, not an exact RF null. Mesh-piecewise field gradients make unverified local Hessians/barrier heights unreliable."},
        "claim_limit": "The exponent near one third is compatible with balancing a displacement-induced approximately constant dipolar electric field against an ideal octupolar field proportional to r^3; four amplitudes do not prove the asymptotic law or a fabrication tolerance."}
    atomic_json(OUT / "displacement_analysis_summary_v5.json", summary)
    print(json.dumps({"minimum_scaling_exponent": fit["log_log_exponent"], "r_squared": fit["r_squared"],
        "h35_h50_delta_minimum_um": fit["h35_h50_10um_radii_mm"]["absolute_difference_mm"]*1000,
        "mode1_delta_exponents": [x["amplitude_scaling_exponent_vs_displacement"] for x in harmonics if x["azimuthal_mode_m"] == 1],
        "mode4_radial_exponents": [x["potential_coefficient_radial_exponent"] for x in radial_orders if x["azimuthal_mode_m"] == 4]}, indent=2))


if __name__ == "__main__":
    main()
