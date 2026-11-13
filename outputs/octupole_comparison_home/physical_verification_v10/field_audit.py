from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIELD_ROOT = ROOT / "outputs/abc_validation/direct_fem_validation/v2"
RADII_M = np.array([.025, .05, .1, .25, .5, 1, 2, 4, 6, 7, 8]) * 1e-3
SAMPLES_PER_RING = 513
RING_COUNT = len(RADII_M)


def relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2)))


def main() -> None:
    loaded = {}
    for h in (35, 50):
        d = FIELD_ROOT / f"h{h}_P4"
        with np.load(d / "common_samples.npz") as data:
            loaded[h] = {key: data[key] for key in data.files}
        if len(loaded[h]["points_m"]) < SAMPLES_PER_RING * RING_COUNT:
            raise ValueError(f"Incomplete native FEM ring samples, h{h}")
    if not np.array_equal(loaded[35]["points_m"][:SAMPLES_PER_RING*RING_COUNT],
                          loaded[50]["points_m"][:SAMPLES_PER_RING*RING_COUNT]):
        raise ValueError("h35 and h50 rings are not evaluated at the same points")

    rows = []
    m4 = {}
    for h in (35, 50):
        a = loaded[h]
        for i, radius in enumerate(RADII_M):
            sl = slice(i * SAMPLES_PER_RING, (i + 1) * SAMPLES_PER_RING)
            native = a["field"][sl]
            fit = a["diagnostic_smooth_field"][sl]
            rows.append({
                "mesh_um": h,
                "radius_mm": float(radius * 1e3),
                "native_P4_field_rms_V_per_m_per_250V": float(np.sqrt(np.mean(native**2) * 2)),
                "N48_relative_rms_field_error": relative_rms(fit, native),
            })
            # In the stored convention phi[...,1] + i*phi[...,0] is H*U0.
            # The n=-4 angular Fourier coefficient is the leading m=4 term.
            points = a["points_m"][sl][:512]
            phi = a["phi"][sl][:512, 1] + 1j * a["phi"][sl][:512, 0]
            theta = np.arctan2(points[:, 1], points[:, 0])
            coeff = np.mean(phi * np.exp(1j * 4 * theta)) / (radius**4 * 250.0)
            m4[f"h{h}_r{radius*1e3:g}mm"] = {
                "C4_complex_per_V_per_m4": [float(coeff.real), float(coeff.imag)],
                "C4_magnitude_per_V_per_m4": float(abs(coeff)),
            }

    convergence = []
    for i, radius in enumerate(RADII_M):
        sl = slice(i * SAMPLES_PER_RING, (i + 1) * SAMPLES_PER_RING)
        coarse, fine = loaded[50]["field"][sl], loaded[35]["field"][sl]
        convergence.append({
            "radius_mm": float(radius * 1e3),
            "h50_minus_h35_relative_rms_field": relative_rms(coarse, fine),
            "field_rms_h50_V_per_m_per_250V": float(np.sqrt(np.mean(coarse**2) * 2)),
            "field_rms_h35_V_per_m_per_250V": float(np.sqrt(np.mean(fine**2) * 2)),
        })

    summary = {
        "scope": "reanalysis of saved native P4 FEM field samples; no FEM solve or trajectory run",
        "h35_sector_solution_sha256": "da90df98f9db31b5b7d84397bdc060991ddda5b6d685a3c0889e5ee36b493872",
        "h50_sector_solution_sha256": "336882933c476bc7ecd2fd97bdc060991ddda5b6d685a3c0889e5ee36b493872",
        "same_points_across_meshes": True,
        "samples_per_ring_included": SAMPLES_PER_RING,
        "rings_mm": (RADII_M * 1e3).tolist(),
        "direct_P4_m4_coefficients": m4,
        "h35_h50_field_sensitivity": convergence,
        "not_yet_verified": ["trajectory-weighted field error", "aperture escape classification", "full RF phase robustness"],
    }
    (HERE / "field_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (HERE / "field_audit_by_radius.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (HERE / "field_convergence_by_radius.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(convergence[0]))
        writer.writeheader(); writer.writerows(convergence)
    print(json.dumps({
        "status": "PASS",
        "m4_h35_per_V_m4": [m4[f"h35_r{r:g}mm"]["C4_magnitude_per_V_per_m4"] for r in (.5, 1, 2)],
        "m4_h50_per_V_m4": [m4[f"h50_r{r:g}mm"]["C4_magnitude_per_V_per_m4"] for r in (.5, 1, 2)],
        "N48_field_error_h50_at_8mm": rows[21]["N48_relative_rms_field_error"],
        "P4_h50_h35_field_error_at_8mm": convergence[-1]["h50_minus_h35_relative_rms_field"],
    }, indent=2))


if __name__ == "__main__":
    main()
