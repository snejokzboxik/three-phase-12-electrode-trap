from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v16.models import Field
from v16.rigid import (
    rod_energy_and_J,
    rod_potential_force_torque,
    rod_quadrature,
    rod_rhs,
)

OUT = ROOT / "results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
T = 2 * math.pi
OMEGA_PHYS = 2 * math.pi * 100_000.0
KAPPA = 0.1044
GAMMA = 3000.0
LAMBDA = GAMMA / OMEGA_PHYS
NODES = 32


def phase_error(a, b):
    return (a - b + math.pi / 4) % (math.pi / 2) - math.pi / 4


def validate_quadrature_and_invariant():
    field = Field("RO", kappa=KAPPA)
    checks = []
    max_qr = 0.0
    max_j_range = 0.0
    for kind in ("uniform", "symmetric_quad", "dipolar"):
        u24 = rod_potential_force_torque(.031, -.027, .43, .91, field, .2, kind, 24)
        u48 = rod_potential_force_torque(.031, -.027, .43, .91, field, .2, kind, 48)
        qr = float(np.max(np.abs(np.asarray(u24) - np.asarray(u48))))
        max_qr = max(max_qr, qr)

        y0 = np.array([.013, -.009, .007, .004, .39, .31])
        sol = solve_ivp(
            lambda t, y: rod_rhs(t, y, field, length_ratio=.2,
                                  charge_model=kind, nodes=NODES),
            (0, 10 * T), y0, method="DOP853", rtol=2e-11, atol=2e-13,
            max_step=T / 100, dense_output=True,
        )
        js = [rod_energy_and_J(sol.sol(t), t, field, length_ratio=.2,
                               charge_model=kind, nodes=NODES)[1]
              for t in np.linspace(0, 10 * T, 101)]
        jrange = float(np.ptp(js))
        max_j_range = max(max_j_range, jrange)
        checks.append({"charge_model": kind, "quadrature_24_vs_48_max_abs": qr,
                       "J_total_range_10rf": jrange})
    if max_qr > 2e-11:
        raise RuntimeError(f"charge quadrature refinement failed: {max_qr:g}")
    if max_j_range > 2e-8:
        raise RuntimeError(f"rigid-body rotating invariant failed: {max_j_range:g}")
    return {"checks": checks, "max_quadrature_difference": max_qr,
            "max_J_total_range_10rf": max_j_range}


def simulate_case(length, kind, direction, center_x=0.0):
    field = Field("RO", kappa=KAPPA, omega_ro=direction)
    s, q = rod_quadrature(length, kind, NODES)
    m4 = float(np.dot(q, s ** 4))
    inertia = length ** 2 / 12.0  # uniform mass distribution about the COM
    spin_target = direction / 4.0
    required_torque = LAMBDA * inertia * spin_target
    torque_amplitude = 4 * KAPPA * m4
    ratio = required_torque / torque_amplitude if torque_amplitude else math.inf
    can_lock = abs(ratio) <= 1.0
    phase = (math.pi - math.asin(max(-1.0, min(1.0, ratio)))) / 4 if can_lock else .35
    x0, y0 = center_x, 0.0
    yinit = np.array([x0, y0, 0.0, 0.0, phase + .08,
                      spin_target + direction * .01])

    def aperture_event(t, z):
        # This bound contains every point on the whole rod, independent of its
        # orientation, and is conservative near the circular aperture.
        return 1.0 - math.hypot(z[0], z[1]) - length / 2
    aperture_event.terminal = True
    aperture_event.direction = -1

    n_periods = 120
    sample_t = np.linspace(0, n_periods * T, n_periods * 8 + 1)
    sol = solve_ivp(
        lambda t, z: rod_rhs(t, z, field, length_ratio=length,
                             charge_model=kind, lambda_translation=LAMBDA,
                             lambda_rotation=LAMBDA, nodes=NODES),
        (0, n_periods * T), yinit, method="DOP853", rtol=3e-10, atol=3e-12,
        max_step=T / 80, t_eval=sample_t, events=aperture_event,
    )
    z = sol.y
    r = np.hypot(z[0], z[1])
    phi = z[4] - direction * sol.t / 4
    tail = sol.t >= max(0.0, sol.t[-1] - 20 * T)
    resultant = np.mean(np.exp(4j * phi[tail])) if np.any(tail) else 0j
    phase_spread = math.sqrt(max(0.0, -2 * math.log(max(abs(resultant), 1e-15)))) / 4
    u0 = rod_potential_force_torque(0.0, 0.0, phase, 0.0, field, length, kind, NODES)
    return {
        "length_over_R": length, "charge_model": kind,
        "direction": direction, "initial_center_x": center_x,
        "gamma_s-1": GAMMA, "kappa": KAPPA,
        "charge_fourth_moment": m4,
        "uniform_mass_inertia_about_com": inertia,
        "predicted_phase_lock_exists_at_center": can_lock,
        "torque_balance_ratio": ratio,
        "center_force_at_initial_center_norm": math.hypot(u0[1], u0[2]),
        "integration_status": "EXIT_APERTURE_BOUND" if sol.status == 1 else "FULL_HORIZON",
        "integrated_rf": float(sol.t[-1] / T),
        "max_center_radius_sampled": float(np.max(r)),
        "max_rod_aperture_bound": float(np.max(r + length / 2)),
        "center_rms_last20rf": float(np.sqrt(np.mean(r[tail] ** 2))) if np.any(tail) else None,
        "phase_spread_last20rf_mod_pi_over_2": phase_spread,
        "mean_spin_last20rf": float(np.mean(z[5, tail])) if np.any(tail) else None,
        "target_spin": spin_target,
        "final_state": z[:, -1].tolist(),
        "samples": int(len(sol.t)),
    }


def main():
    validation = validate_quadrature_and_invariant()
    cases = []
    for length in (.05, .10, .20):
        for kind in ("uniform", "symmetric_quad", "dipolar"):
            for direction in (-1, 1):
                cases.append(simulate_case(length, kind, direction))
    # Off-centre controls isolate translation–rotation coupling at a fixed
    # length: compare symmetric and charge-offset rods with identical starts.
    for kind in ("uniform", "dipolar"):
        for direction in (-1, 1):
            cases.append(simulate_case(.20, kind, direction, center_x=.02))

    payload = {
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "field": "pure rotating octupole RO",
        "normalization": {
            "kappa": KAPPA, "R_m": .008, "f_Hz": 100_000,
            "M_over_Q": "uniform mass rod; total charge/mass normalized to point-particle units",
            "gamma_s-1": GAMMA, "lambda": LAMBDA,
            "mass_distribution": "uniform along rod; Ibar=(L/R)^2/12",
            "charge_profiles": {
                "uniform": "constant density",
                "symmetric_quad": "1 + 0.8*(u^2-1/3), u in [-1,1]",
                "dipolar": "1 + 0.7*u, u in [-1,1]; nonzero charge centroid relative to COM",
            },
            "aperture_test": "center radius + L/(2R) <= 1; conservative whole-segment bound",
        },
        "validation": validation,
        "cases": cases,
        "interpretation_limit": "Ideal 2-D dimensionless rigid-body model; no material-specific mass/charge, gas, polarizability, axial field, or FEM calibration.",
    }
    (OUT / "rod_distribution_coupling.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = OUT / "rod_distribution_coupling.csv"
    columns = [k for k, v in cases[0].items() if not isinstance(v, (list, dict))]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in columns} for row in cases)

    labels = [f"{r['charge_model']}\nL/R={r['length_over_R']:.2f}" for r in cases[:18:2]]
    symmetric = [cases[i]["max_center_radius_sampled"] for i in range(0, 18, 2)]
    asymmetric = [cases[i]["max_center_radius_sampled"] for i in range(4, 18, 6)]
    # Use a clear aggregate scatter: each marker is a simulated direction/model,
    # not a fitted regime boundary.
    fig, ax = plt.subplots(figsize=(8.8, 4.8), layout="constrained")
    colors = {"uniform": "#2563eb", "symmetric_quad": "#16a34a", "dipolar": "#dc2626"}
    markers = {-1: "v", 1: "^"}
    for row in cases:
        ax.scatter(row["length_over_R"], row["max_center_radius_sampled"],
                   color=colors[row["charge_model"]], marker=markers[row["direction"]],
                   s=52 if row["initial_center_x"] == 0 else 85,
                   facecolors="none" if row["initial_center_x"] else colors[row["charge_model"]],
                   edgecolors=colors[row["charge_model"]], linewidths=1.1)
    for kind, color in colors.items():
        ax.scatter([], [], color=color, marker="o", label=kind)
    ax.scatter([], [], color="black", marker="^", label="RO direction +")
    ax.scatter([], [], color="black", marker="v", label="RO direction −")
    ax.scatter([], [], facecolors="none", edgecolors="black", s=85, label="off-centre start")
    ax.set(xlabel=r"rod length $L/R$", ylabel="maximum sampled COM radius / R",
           title="Finite-size translation under RO; 120 RF at each point")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(FIG / "rod_distribution_coupling.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"cases": len(cases), "validation": validation,
                      "output": str(OUT / "rod_distribution_coupling.json"),
                      "full_horizon": sum(r["integration_status"] == "FULL_HORIZON" for r in cases),
                      "exited": sum(r["integration_status"] != "FULL_HORIZON" for r in cases)},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
