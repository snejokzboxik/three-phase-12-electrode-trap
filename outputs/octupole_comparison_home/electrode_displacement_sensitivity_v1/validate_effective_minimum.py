"""Independent, multi-method validation of the reported 10-um effective minimum."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

import field_analysis as fa
import solve_displaced_p4 as pilot

OUT = fa.OUT


def evaluate(field, point):
    _, ec, es = field.sample_one(float(point[0]), float(point[1]))
    return float(np.dot(ec, ec) + np.dot(es, es))


def grid_candidates(field, n: int = 81, radius: float = 5e-3):
    axis = np.linspace(-radius, radius, n)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    points = points[np.linalg.norm(points, axis=1) <= radius]
    _, ec, es = field.sample(points)
    values = np.sum(ec * ec, axis=1) + np.sum(es * es, axis=1)
    order = np.argsort(values)
    selected = []
    for idx in order:
        p = points[idx]
        if all(np.linalg.norm(p - old["point_m"]) > 0.25e-3 for old in selected):
            selected.append({"point_m": p, "value": float(values[idx])})
        if len(selected) >= 6:
            break
    return axis, points, values, selected


def optimize(field, start, method):
    bounds = [(-5e-3, 5e-3), (-5e-3, 5e-3)]
    if method == "Nelder-Mead":
        result = minimize(lambda x: evaluate(field, x), start, method=method, bounds=bounds,
                          options={"xatol": 2e-10, "fatol": 1e-18, "maxiter": 1500, "maxfev": 4000})
    elif method == "Powell":
        result = minimize(lambda x: evaluate(field, x), start, method=method, bounds=bounds,
                          options={"xtol": 2e-9, "ftol": 1e-12, "maxiter": 800, "maxfev": 5000})
    else:
        result = minimize(lambda x: evaluate(field, x), start, method="L-BFGS-B", bounds=bounds,
                          options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 1200, "maxfun": 5000, "finite_diff_rel_step": 1e-5})
    return {"method": method, "start_m": list(map(float, start)), "success": bool(result.success), "message": str(result.message), "position_m": list(map(float, result.x)), "radius_mm": float(np.linalg.norm(result.x) * 1e3), "E2_min_V2_m2_per_V2": float(result.fun), "nfev": int(result.nfev)}


def main():
    runtime = pilot.start_or_resume_budget()
    pilot.require_time(runtime, 300.0, "independent validation of the effective-potential minimum")
    started = time.perf_counter()
    results = []
    for resolution, dirname, meshname in (("h35", "radial_out_10um", "displaced_h35.msh"), ("h50", "radial_out_10um_h50", "displaced_h50.msh")):
        pilot.require_time(runtime, 60.0, f"load {resolution} full-domain P4 field")
        directory = OUT / "calculations" / dirname
        field, meta = fa.field_from_coefficients(directory / meshname, directory / "p4_coefficients.npz")
        axis, points, values, starts = grid_candidates(field, n=81)
        grid_step_mm = float((axis[1] - axis[0]) * 1e3)
        opts = []
        methods = ("Nelder-Mead", "Powell", "L-BFGS-B")
        for candidate in starts[:3]:
            for method in methods:
                pilot.require_time(runtime, 30.0, f"{method} minimization on {resolution}")
                opts.append(optimize(field, candidate["point_m"], method))
        origins = [{"point_m": [0.0, 0.0], "E2_V2_m2_per_V2": evaluate(field, [0.0, 0.0])}]
        optimum_values = [r["E2_min_V2_m2_per_V2"] for r in opts]
        best = min(opts, key=lambda x: x["E2_min_V2_m2_per_V2"])
        values_sorted = sorted(values)
        _, ec_min, es_min = field.sample_one(*best["position_m"])
        e2_min = float(np.dot(ec_min, ec_min) + np.dot(es_min, es_min))
        e2_origin = origins[0]["E2_V2_m2_per_V2"]
        pseudo_factor_eV_per_e2 = fa.CHARGE**2 * 250.0**2 / (4 * fa.ION_MASS * (2 * math.pi * 100_000.0)**2 * fa.E_CHARGE)
        results.append({
            "resolution": resolution,
            "field_representation": meta,
            "search_domain": "disk r<=5 mm",
            "grid": {"shape_requested": [81, 81], "disk_points": int(len(points)), "spacing_mm": grid_step_mm, "best_points": [{"position_m": c["point_m"].tolist(), "E2_V2_m2_per_V2": c["value"]} for c in starts], "best_grid_radius_mm": float(np.linalg.norm(starts[0]["point_m"]) * 1e3), "minimum_grid_value": float(values_sorted[0])},
            "optimizer_results": opts,
            "best_optimizer_result": best,
            "value_at_origin_V2_m2_per_V2": origins[0]["E2_V2_m2_per_V2"],
            "minimum_over_all_methods_m": best["position_m"],
            "field_norm_at_best_per_volt_V_m_per_V": float(math.sqrt(e2_min)),
            "field_norm_at_best_for_250V_V_m": float(math.sqrt(e2_min) * 250.0),
            "field_norm_at_origin_per_volt_V_m_per_V": float(math.sqrt(e2_origin)),
            "rotating_pseudopotential_minimum_eV_at_250V_100kHz": float(pseudo_factor_eV_per_e2 * e2_min),
            "rotating_pseudopotential_at_origin_eV_at_250V_100kHz": float(pseudo_factor_eV_per_e2 * e2_origin),
            "optimizer_radius_span_mm_including_secondary_local_minima": float(max(r["radius_mm"] for r in opts) - min(r["radius_mm"] for r in opts)),
            "minimum_search_converged": any(r["success"] for r in opts),
        })
    radii = [r["best_optimizer_result"]["radius_mm"] for r in results]
    positions = [np.array(r["best_optimizer_result"]["position_m"]) for r in results]
    resolution_distance = abs(radii[0] - radii[1])
    objective_rel_difference = abs(results[0]["best_optimizer_result"]["E2_min_V2_m2_per_V2"] - results[1]["best_optimizer_result"]["E2_min_V2_m2_per_V2"]) / max(results[0]["best_optimizer_result"]["E2_min_V2_m2_per_V2"], 1e-30)
    result = {
        "question": "Is the approximately 0.446-mm rotating RF effective-potential minimum after a +10-um radial displacement robust to mesh resolution and optimizer choice?",
        "objective": "|E_cos|^2+|E_sin|^2 per squared electrode volt; proportional to the rotating pseudopotential for fixed q, m, frequency and voltage",
        "methods": "81x81 deterministic Cartesian grid in r<=5 mm, then independent bounded Nelder-Mead, Powell and L-BFGS-B optimizations from three separated grid-minimum starts on each already-solved field",
        "h35_best_radius_mm": radii[0],
        "h50_best_radius_mm": radii[1],
        "cross_resolution_radius_difference_mm": resolution_distance,
        "cross_resolution_objective_relative_difference": objective_rel_difference,
        "all_methods_positions_mm": [{"resolution": row["resolution"], "method": opt["method"], "position_mm": [x * 1e3 for x in opt["position_m"]], "radius_mm": opt["radius_mm"], "E2": opt["E2_min_V2_m2_per_V2"], "success": opt["success"]} for row in results for opt in row["optimizer_results"]],
        "mirror_degeneracy_note": "The one-electrode radial pattern is reflection-symmetric about its radial axis; equal-radius minima on the two sides are physically expected and optimizer signs may differ.",
        "validation_status": "validated" if all(row["minimum_search_converged"] for row in results) and resolution_distance < 0.03 and objective_rel_difference < 0.01 else "unresolved",
        "interpretation_limit": "A nonzero minimum of the 4-component quadrature field norm is an effective-potential minimum, not an exact RF null. It is the lowest location found by the disclosed grid and multi-start methods in r<=5 mm; the finite grid does not prove a mathematical global minimum.",
        "per_resolution_results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    output = OUT / "processed" / "effective_minimum_method_validation.json"
    pilot.atomic_json(output, result)
    pilot.append_checkpoint({"time_utc": pilot.iso(pilot.utc_now()), "stage": "effective_minimum_independent_validation_complete", "status": result["validation_status"], "h35_radius_mm": radii[0], "h50_radius_mm": radii[1]})
    print(json.dumps({k: result[k] for k in ("h35_best_radius_mm", "h50_best_radius_mm", "cross_resolution_radius_difference_mm", "cross_resolution_objective_relative_difference", "validation_status", "wall_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
