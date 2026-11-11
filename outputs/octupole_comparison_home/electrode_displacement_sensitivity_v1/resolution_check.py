"""Limited h35/h50 sensitivity check; N48 is used for nominal fields only."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np

import field_analysis as fa
import solve_displaced_p4 as pilot

ROOT = fa.ROOT
OUT = fa.OUT
N48_35 = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h35_P4/diagnostic_N48.json"
N48_50 = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json"


def metrics(label: str, q35, q50, source35: str, source50: str) -> list[dict]:
    rows = []
    ntheta = 144
    theta = np.arange(ntheta) * (2 * math.pi / ntheta)
    def get_quadratures(obj, xy):
        if hasattr(obj, "field_quadratures"):
            return obj.field_quadratures(xy)
        _, ec, es = obj.sample(xy)
        return ec, es
    for r_mm in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        r = r_mm * 1e-3
        xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        ec35, es35 = get_quadratures(q35, xy)
        ec50, es50 = get_quadratures(q50, xy)
        for model, e35, e50 in (("rotating", np.concatenate((ec35, es35), axis=1), np.concatenate((ec50, es50), axis=1)), ("standing", es35, es50)):
            delta = e50 - e35
            rows.append({
                "comparison": label, "model": model, "radius_mm": r_mm, "azimuth_count": ntheta,
                "E_difference_rms_V_m_per_V": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
                "E_difference_p95_V_m_per_V": float(np.quantile(np.linalg.norm(delta, axis=1), .95)),
                "relative_L2_difference_to_h35": float(np.linalg.norm(delta) / max(np.linalg.norm(e35), np.finfo(float).tiny)),
                "E_h35_rms_V_m_per_V": float(np.sqrt(np.mean(np.sum(e35 * e35, axis=1)))),
                "E_h50_rms_V_m_per_V": float(np.sqrt(np.mean(np.sum(e50 * e50, axis=1)))),
                "source_h35": source35, "source_h50": source50,
            })
    return rows


def main():
    runtime = pilot.start_or_resume_budget()
    pilot.require_time(runtime, 60, "limited h35/h50 field sensitivity analysis")
    p35 = OUT / "calculations" / "radial_out_10um"
    p50 = OUT / "calculations" / "radial_out_10um_h50"
    f35, m35 = fa.field_from_coefficients(p35 / "displaced_h35.msh", p35 / "p4_coefficients.npz")
    f50, m50 = fa.field_from_coefficients(p50 / "displaced_h50.msh", p50 / "p4_coefficients.npz")
    direct_rows = metrics("displaced radial_out_10um full-domain P4", f35, f50, m35["field_sha256"], m50["field_sha256"])

    from three_phase_trap.rf_reconstruction import RFReconstruction
    q35 = RFReconstruction.load(N48_35).unit_quadratures
    q50 = RFReconstruction.load(N48_50).unit_quadratures
    nominal_rows = metrics("nominal validated N48 h35 versus h50", q35, q50, str(N48_35.relative_to(ROOT)), str(N48_50.relative_to(ROOT)))
    rows = direct_rows + nominal_rows
    out = OUT / "processed"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "resolution_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "changed_geometry_comparison": "direct full-domain P4 h35 versus h50 for the same +10 um radial displacement of electrode 0",
        "nominal_comparison": "existing FEM-derived N48 h35 versus h50 on nominal geometry only; N48 is never used to represent a displaced geometry",
        "samples": "same 144 deterministic azimuth points on radii 0.5 to 7 mm; no angular interpolation",
        "max_relative_L2_difference_changed_P4": max(r["relative_L2_difference_to_h35"] for r in direct_rows),
        "max_relative_L2_difference_nominal_N48": max(r["relative_L2_difference_to_h35"] for r in nominal_rows),
        "csv": str((out / "resolution_sensitivity.csv").relative_to(ROOT)),
    }
    pilot.atomic_json(out / "resolution_sensitivity_summary.json", summary)
    pilot.append_checkpoint({"time_utc": pilot.iso(pilot.utc_now()), "stage": "resolution_sensitivity_complete", "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
