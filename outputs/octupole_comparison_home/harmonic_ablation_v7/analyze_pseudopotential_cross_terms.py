"""Decompose the cycle-averaged pseudopotential into mode self/cross terms.

This is a static evaluation of the saved h35 N48 harmonic coefficients. It
performs no RF trajectory integrations and no FEM calculations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run_v7

HERE = Path(__file__).resolve().parent
OUT = HERE / "processed"
E = run_v7.run_v4.E_CHARGE
MASS = run_v7.MASS_MZ_TH * run_v7.run_v4.AMU
RADIUS_M = 8e-3
ANGLE_DEG = np.linspace(0.0, 360.0, 1440, endpoint=False)


def main() -> None:
    q35, _ = run_v7._load_source_fields()
    models, indices = run_v7._make_q_models(q35)
    points = np.column_stack((RADIUS_M*np.cos(np.radians(ANGLE_DEG)),
                              RADIUS_M*np.sin(np.radians(ANGLE_DEG))))
    rows = []
    angular_rows = []
    self_cache = {}
    max_reconstruction_error = 0.0
    for rep in ("m4_m8", "m4_m8_m16", "full_N48_minus_m8", "full_N48"):
        modes = indices[rep]
        for model in ("rotating", "standing"):
            q = models[(rep, model)]
            voltage = run_v7.VOLTAGE[model]
            component_fields = {}
            self_terms = {}
            cross_terms = {}
            for m in modes:
                c = np.zeros_like(q.cosine)
                s = np.zeros_like(q.sine)
                c[m], s[m] = q.cosine[m], q.sine[m]
                from three_phase_trap.rf_reconstruction import HarmonicQuadratures
                qm = HarmonicQuadratures(c, s, q.reference_radius)
                ec, es = qm.field_quadratures(points)
                component_fields[m] = (voltage*ec, voltage*es)
                psi_self = E**2*(np.sum(ec*ec, axis=1)+np.sum(es*es, axis=1))*voltage**2/(4*MASS*run_v7.OMEGA**2*E)
                self_terms[m] = psi_self
                self_cache[(rep, model, m)] = psi_self
                rows.append({"representation": rep, "model": model, "term": f"self_m{m}",
                             "harmonics": str(m), "mean_eV": float(np.mean(psi_self)),
                             "min_eV": float(np.min(psi_self)), "max_eV": float(np.max(psi_self)),
                             "rms_about_mean_eV": float(np.std(psi_self))})
            for i, m in enumerate(modes):
                for n in modes[i+1:]:
                    ecm, esm = component_fields[m]
                    ecn, esn = component_fields[n]
                    cross = E**2*(np.sum(ecm*ecn, axis=1)+np.sum(esm*esn, axis=1))/(2*MASS*run_v7.OMEGA**2*E)
                    cross_terms[(m,n)] = cross
                    rows.append({"representation": rep, "model": model, "term": f"cross_m{m}_m{n}",
                                 "harmonics": f"{m},{n}", "mean_eV": float(np.mean(cross)),
                                 "min_eV": float(np.min(cross)), "max_eV": float(np.max(cross)),
                                 "rms_about_mean_eV": float(np.std(cross))})
            total = q.pseudo(points, E, MASS, run_v7.OMEGA)*voltage**2/E
            self_sum = np.sum(np.asarray(list(self_terms.values())), axis=0)
            cross48 = cross_terms.get((4,8), np.zeros_like(total))
            other_cross = np.sum(np.asarray([v for pair,v in cross_terms.items() if pair != (4,8)]), axis=0) if len(cross_terms) > (1 if (4,8) in cross_terms else 0) else np.zeros_like(total)
            reconstructed = self_sum + sum(cross_terms.values(), np.zeros_like(total))
            max_reconstruction_error = max(max_reconstruction_error, float(np.max(np.abs(reconstructed-total))))
            for i, angle in enumerate(ANGLE_DEG):
                angular_rows.append({"representation": rep, "model": model, "angle_deg": float(angle),
                                     "self_sum_eV": float(self_sum[i]),
                                     "cross_m4_m8_eV": float(cross48[i]),
                                     "other_cross_sum_eV": float(other_cross[i]),
                                     "total_pseudopotential_eV": float(total[i])})
            rows.append({"representation": rep, "model": model, "term": "total",
                         "harmonics": ",".join(map(str, modes)), "mean_eV": float(np.mean(total)),
                         "min_eV": float(np.min(total)), "max_eV": float(np.max(total)),
                         "rms_about_mean_eV": float(np.std(total))})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pseudopotential_mode_cross_terms.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["representation", "model", "term", "harmonics",
                                         "mean_eV", "min_eV", "max_eV", "rms_about_mean_eV"])
        w.writeheader(); w.writerows(rows)
    angular_path = OUT / "pseudopotential_interference_angular.csv"
    with angular_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["representation", "model", "angle_deg", "self_sum_eV",
                                         "cross_m4_m8_eV", "other_cross_sum_eV", "total_pseudopotential_eV"])
        w.writeheader(); w.writerows(angular_rows)

    self_mismatch = {}
    for rep in ("m4_m8", "full_N48"):
        for m in indices[rep]:
            a = self_cache[(rep, "rotating", m)]
            b = self_cache[(rep, "standing", m)]
            self_mismatch[f"{rep}_m{m}"] = float(np.max(np.abs(a-b))/max(np.max(np.abs(a)),1e-30))
    rot_cross48 = next(r for r in rows if r["representation"] == "full_N48" and
                       r["model"] == "rotating" and r["term"] == "cross_m4_m8")
    stand_cross48 = next(r for r in rows if r["representation"] == "full_N48" and
                         r["model"] == "standing" and r["term"] == "cross_m4_m8")
    validation = {"radius_mm": 8.0, "angular_samples": len(ANGLE_DEG),
                  "max_abs_total_reconstruction_error_eV": max_reconstruction_error,
                  "max_relative_single_mode_self_term_mismatch_rotating_vs_standing": max(self_mismatch.values()),
                  "single_mode_self_term_relative_mismatches": self_mismatch,
                  "full_rotating_m4_m8_cross_term_abs_max_eV": max(abs(float(rot_cross48["min_eV"])), abs(float(rot_cross48["max_eV"]))),
                  "full_standing_m4_m8_cross_term_min_eV": float(stand_cross48["min_eV"]),
                  "full_standing_m4_m8_cross_term_max_eV": float(stand_cross48["max_eV"]),
                  "interpretation": "Pseudopotential decomposition into mode self terms and pairwise quadrature dot products; static diagnostic, not a conserved-energy approximation proof."}
    (OUT / "pseudopotential_mode_cross_terms_validation.json").write_text(
        json.dumps(validation, indent=2)+"\n", encoding="utf-8")

    # The m=4 and m=8 self contributions must match under the leading-m4
    # voltage protocol; the between-mode term is the direct interference test.
    print(f"saved {path.relative_to(HERE)} and {angular_path.relative_to(HERE)}; sampled angles={len(ANGLE_DEG)}; r=8 mm")
    print(json.dumps(validation, indent=2))
    for rep in ("m4_m8", "full_N48"):
        print(f"[{rep}]")
        for model in ("rotating", "standing"):
            for r in rows:
                if r["representation"] == rep and r["model"] == model and (
                    r["term"] in ("self_m4", "self_m8", "cross_m4_m8", "total")
                    or (rep == "full_N48" and r["term"] == "cross_m4_m20")
                ):
                    print(model, r["term"], f"mean={r['mean_eV']:.6g}",
                          f"min={r['min_eV']:.6g}", f"max={r['max_eV']:.6g}")


if __name__ == "__main__":
    main()
