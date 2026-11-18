from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v16.rigid import rod_quadrature

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
INPUT = RESULTS / "rod_distribution_coupling.json"
OMEGA = 2 * math.pi * 100_000.0


def main():
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    kappa = float(payload["normalization"]["kappa"])
    gamma = float(payload["normalization"]["gamma_s-1"])
    for row in payload["cases"]:
        length = float(row["length_over_R"])
        kind = row["charge_model"]
        s, q = rod_quadrature(length, kind, 32)
        moments = {n: float(q @ (s ** n)) for n in range(1, 5)}
        inertia = float(row["uniform_mass_inertia_about_com"])
        threshold = OMEGA * 16 * abs(kappa) * moments[4] / inertia
        row["charge_moments_about_mass_com"] = moments
        row["analytic_center_force_any_phase"] = 4 * abs(kappa * moments[3])
        row["analytic_angular_lock_threshold_s-1"] = threshold
        row["angular_torque_balance_possible_at_test_gamma"] = gamma <= threshold
        row["threshold_scope"] = "angular balance with center frozen; for asymmetric charge this is not a full-body equilibrium condition"

    payload["derived_analysis"] = {
        "angular_threshold_formula": "gamma_c = Omega*16*abs(kappa)*mu4/Ibar; for uniform charge mu4=L^4/80 and Ibar=L^2/12, giving 2.4*Omega*abs(kappa)*(L/R)^2",
        "center_force_formula": "at Z=0 in RO, |F_COM|=4*abs(kappa*mu3); it vanishes for even charge profiles",
        "plot_conditions": {"kappa": kappa, "gamma_s-1": gamma, "f_Hz": 100_000, "R_m": .008, "horizon_rf": 120},
    }
    tmp = INPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(INPUT)

    scalar_columns = [k for k, v in payload["cases"][0].items()
                      if not isinstance(v, (dict, list))]
    scalar_columns += ["mu1", "mu2", "mu3", "mu4"]
    with (RESULTS / "rod_distribution_coupling.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=scalar_columns)
        w.writeheader()
        for row in payload["cases"]:
            flat = dict(row)
            for n, val in row["charge_moments_about_mass_com"].items():
                flat[f"mu{n}"] = val
            w.writerow({k: flat.get(k) for k in scalar_columns})

    colors = {"uniform": "#2563eb", "symmetric_quad": "#16a34a", "dipolar": "#dc2626"}
    markers = {-1: "v", 1: "^"}
    fig, ax = plt.subplots(figsize=(8.8, 4.8), layout="constrained")
    for row in payload["cases"]:
        is_off = row["initial_center_x"] != 0
        ax.scatter(row["length_over_R"], row["max_center_radius_sampled"],
                   color=colors[row["charge_model"]], marker=markers[row["direction"]],
                   s=52 if not is_off else 85,
                   facecolors="none" if is_off else colors[row["charge_model"]],
                   edgecolors=colors[row["charge_model"]], linewidths=1.1)
    for kind, color in colors.items():
        ax.scatter([], [], color=color, marker="o", label=kind)
    ax.scatter([], [], color="black", marker="^", label="RO direction +")
    ax.scatter([], [], color="black", marker="v", label="RO direction −")
    ax.scatter([], [], facecolors="none", edgecolors="black", s=85, label="off-centre start")
    ax.set(xlabel=r"rod length $L/R$", ylabel="maximum sampled COM radius / R",
           title=fr"Finite-size translation under RO; 120 RF ($\kappa={kappa}$, $\gamma={gamma:g}\,s^{{-1}}$)"
                 "\nR = 8 mm; f = 100 kHz")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(FIGURES / "rod_distribution_coupling.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"cases": len(payload["cases"]),
                      "all_reached_120rf": all(x["integration_status"] == "FULL_HORIZON" for x in payload["cases"]),
                      "dipolar_center_force_example": next(x["analytic_center_force_any_phase"] for x in payload["cases"] if x["charge_model"] == "dipolar" and x["length_over_R"] == .2),
                      "uniform_L02_threshold_s-1": next(x["analytic_angular_lock_threshold_s-1"] for x in payload["cases"] if x["charge_model"] == "uniform" and x["length_over_R"] == .2),
                      "figure": str(FIGURES / "rod_distribution_coupling.png")}, indent=2))


if __name__ == "__main__":
    main()
