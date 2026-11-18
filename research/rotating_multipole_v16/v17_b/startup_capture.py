"""RO3 neutral-dipole spin-up from rest: frequency sweep and 6D checks.

The exact centered reduction is integrated for a deterministic, uniform
orientation grid.  Selected captured starts are then checked in the full
six-state laboratory model with a small COM perturbation.  The frequency
sweep holds dimensionless coefficients fixed; it is not a physical-frequency
calibration of a fixed device.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from v16_1.dynamics import Harmonic, rigid_rhs
from v17_b.model import charge_moments


OUT = Path(__file__).resolve().parent
RES = OUT / "results"
FIG = OUT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
CHECKPOINT = RES / "startup_capture_checkpoint.json"

# Same dimensionless RO3 finite neutral dipole model used in the V17B control.
m = 3
a = 0.1
ell = 2 * a
kappa = 0.03
gamma_trans = 0.05
gamma_rot = 0.02
inertia = ell**2 / 12
omegas = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9,
          1.0, 2.0, 3.0, 3.5, 3.8, 4.0, 4.04]
n_orientation = 360  # 120 distinct relative phases, repeated over 2pi body angle.
qmom = charge_moments([1.0, -1.0], [a, -a], 3)
Cm = qmom.moments[m]
C1 = qmom.moments[1]
assert abs(Cm.imag) < 1e-14 and Cm.real > 0
b = math.comb(m, 2) * kappa * abs(C1)
wf_min_sq = (-gamma_trans**2 + math.sqrt(gamma_trans**4 + 4 * (2 * b) ** 2)) / 2
omega_com_min = m * math.sqrt(wf_min_sq)
omega_rot_max = m * m * kappa * abs(Cm) / (inertia * gamma_rot)


def orientation_sweep(omega: float, nrf: int, nphase: int):
    """Integrate the exact 1-DOF rotating-frame equation for all phases."""
    eta = gamma_rot * inertia * abs(omega) / (m * m * kappa * abs(Cm))
    if eta >= 1:
        return {"omega": omega, "eta": eta, "status": "NO_LOCK_BRANCH"}
    theta_eq = math.pi - math.asin(eta)
    psi0 = 2 * math.pi * np.arange(nphase) / nphase
    theta0 = m * psi0 + np.angle(Cm)
    y0 = np.r_[theta0, np.full(nphase, -omega)]  # spin(0)=0 => theta_dot=-Omega.
    A = m * m * kappa * abs(Cm) / inertia
    period = 2 * math.pi / abs(omega)
    tend = nrf * period
    tail_rf = min(50, max(20, nrf // 20))
    t_eval = np.linspace(tend - tail_rf * period, tend, 1001)

    def rhs(_t, y):
        th, thdot = y[:nphase], y[nphase:]
        return np.r_[thdot, A * np.sin(th) - gamma_rot * (thdot + omega)]

    sol = solve_ivp(rhs, (0.0, tend), y0, method="DOP853", rtol=3e-10,
                    atol=3e-12, max_step=0.8, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    th = sol.y[:nphase]
    thdot = sol.y[nphase:]
    wrapped = np.angle(np.exp(1j * (th - theta_eq)))
    max_phase_error = np.max(np.abs(wrapped), axis=1)
    max_rel_spin = np.max(np.abs(thdot), axis=1) / m
    captured = (max_phase_error < 0.08) & (max_rel_spin < 0.002)
    mean_spin = (np.mean(thdot, axis=1) + omega) / m
    return {
        "omega": omega,
        "spin_target": omega / m,
        "eta_drag_to_lock_threshold": eta,
        "stable_phase_relative_to_field": theta_eq,
        "rf_periods": nrf,
        "orientation_count": nphase,
        "distinct_relative_phase_count": nphase // m,
        "capture_count": int(np.count_nonzero(captured)),
        "orientation_grid_capture_fraction": float(np.mean(captured)),
        "classification": "DETERMINISTIC_UNIFORM_ORIENTATION_GRID_NOT_BINOMIAL",
        "initial_psi_captured_rad": psi0[captured].tolist(),
        "initial_psi_uncaptured_rad": psi0[~captured].tolist(),
        "max_phase_error_captured_rad": float(np.max(max_phase_error[captured])) if np.any(captured) else None,
        "max_rel_spin_error_captured": float(np.max(max_rel_spin[captured])) if np.any(captured) else None,
        "mean_spin_range_uncaptured": [float(np.min(mean_spin[~captured])), float(np.max(mean_spin[~captured]))] if np.any(~captured) else None,
        "nfev": int(sol.nfev),
    }


def full_6d_spinup(omega: float, psi0: float, nrf: int = 1000,
                   com_offset: float = 0.0):
    """Check a from-rest captured phase with a small independent COM offset."""
    harmonic = [Harmonic(m, "RO", kappa, omega, 0.0)]
    period = 2 * math.pi / abs(omega)
    t1 = nrf * period
    y0 = np.array([com_offset, -0.75 * com_offset, 0.25 * com_offset,
                   0.0, psi0, 0.0])

    def rhs(t, y):
        return rigid_rhs(t, y, harmonic, ell, gamma_trans, gamma_rot,
                         inertia, qmom.moments)

    def aperture(_t, y):
        return 1 - a - math.hypot(y[0], y[1])
    aperture.terminal = True
    aperture.direction = -1
    # Retain the full trajectory on a sub-RF sampling grid so reported COM
    # maxima and clearances refer to the complete 1000-RF interval.
    t_eval = np.linspace(0.0, t1, max(2001, int(math.ceil(t1 / 0.25)) + 1))
    sol = solve_ivp(rhs, (0.0, t1), y0, method="DOP853", rtol=2e-9,
                    atol=2e-11, max_step=0.5, events=aperture, t_eval=t_eval)
    if sol.status == 1:
        hit = sol.y_events[0][0]
        return {"omega": omega, "com_offset_parameter": com_offset,
                "status": "ESCAPED_APERTURE", "escape_time": float(sol.t_events[0][0]),
                "com_radius_at_escape": float(math.hypot(hit[0], hit[1])),
                "body_radius_at_escape": float(math.hypot(hit[0], hit[1]) + a),
                "nfev": int(sol.nfev)}
    if not sol.success:
        return {"omega": omega, "com_offset_parameter": com_offset,
                "status": "INTEGRATION_FAILED", "message": sol.message,
                "nfev": int(sol.nfev)}
    t = sol.t
    y = sol.y
    phase = m * y[4] - omega * t + np.angle(Cm)
    phase_rel = np.angle(np.exp(1j * (phase - (math.pi - math.asin(
        gamma_rot * inertia * abs(omega) / (m * m * kappa * abs(Cm)))))))
    com = np.hypot(y[0], y[1])
    spin_error = y[5] - omega / m
    final_mask = t >= t1 - 30 * period
    return {
        "omega": omega,
        "com_offset_parameter": com_offset,
        "initial_state": y0.tolist(),
        "rf_periods": nrf,
        "status": "CAPTURED_6D" if np.max(np.abs(phase_rel[final_mask])) < 0.08 and np.max(np.abs(spin_error[final_mask])) < 0.002 else "NOT_CAPTURED_6D",
        "mean_spin_last30rf": float(np.mean(y[5, final_mask])),
        "target_spin": omega / m,
        "max_abs_phase_error_last30rf": float(np.max(np.abs(phase_rel[final_mask]))),
        "max_spin_error_last30rf": float(np.max(np.abs(spin_error[final_mask]))),
        "max_com_radius_last30rf": float(np.max(com[final_mask])),
        "body_clearance_min_last30rf": float(1 - a - np.max(com[final_mask])),
        "max_com_radius_all": float(np.max(com)),
        "body_clearance_min_all": float(1 - a - np.max(com)),
        "nfev": int(sol.nfev),
        "tolerance": {"rtol": 2e-9, "atol": 2e-11, "max_step": 0.5},
    }


def main():
    if CHECKPOINT.exists():
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        results = checkpoint["frequency_sweep"]
    elif (RES / "startup_capture_results.json").exists():
        previous = json.loads((RES / "startup_capture_results.json").read_text(encoding="utf-8"))
        results = previous["frequency_sweep"]
    else:
        results = []
    done = {float(x["omega"]) for x in results}
    for omega in omegas:
        if omega in done:
            continue
        eta = gamma_rot * inertia * abs(omega) / (m * m * kappa * abs(Cm))
        horizon = 10000 if eta > 0.97 else 4000 if eta > 0.9 else 1500
        row = orientation_sweep(omega, horizon, n_orientation)
        row["joint_local_linear_stability"] = bool(omega > omega_com_min and omega < omega_rot_max)
        results.append(row)
        results.sort(key=lambda x: x["omega"])
        cp = {
            "status": "RUNNING_CHECKPOINT",
            "model": {"field": "pure RO3", "particle": "neutral finite dipole (+q,-q) at (+a,-a)",
                      "a_over_R": a, "ell_over_R": ell, "kappa3": kappa,
                      "gamma_trans": gamma_trans, "gamma_rot": gamma_rot,
                      "inertia": inertia, "C1": float(C1.real), "C3": float(Cm.real)},
            "analytic_dimensionless_joint_band": {"omega_COM_min": omega_com_min,
                "omega_rot_max": omega_rot_max, "COM_stability_condition": "2|b| < (omega/3)*sqrt((omega/3)^2+gamma_trans^2)",
                "rotation_condition": "gamma_rot < 9*kappa*|C3|/(I*|omega|)"},
            "frequency_sweep": results,
        }
        CHECKPOINT.write_text(json.dumps(cp, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"omega": omega, "eta": eta, "capture_fraction": row["orientation_grid_capture_fraction"],
                          "count": row["capture_count"], "rf_periods": horizon}, ensure_ascii=False), flush=True)

    # Full nonlinear checks: a captured low-frequency target from exact COM,
    # followed by progressively larger COM perturbations.
    omega = 0.5
    row = next(x for x in results if math.isclose(x["omega"], omega))
    # Use the center of the broadest sampled capture interval, then perturb
    # the COM progressively to expose sensitivity of startup capture.
    checks = []
    for offset in (0.0, 1e-4, 1e-3, 4e-3):
        checks.append(full_6d_spinup(omega, math.pi / 3, 1000, offset))
    for omega in (1.0, 3.8):
        row = next(x for x in results if math.isclose(x["omega"], omega))
        starts = row.get("initial_psi_captured_rad", [])
        if not starts:
            checks.append({"omega": omega, "status": "NO_GRID_CAPTURED_PHASE"})
            continue
        checks.append(full_6d_spinup(omega, starts[0], 1000, 0.0))

    result = {
        "status": "COMPLETED_DIMENSIONLESS_STARTUP_SWEEP",
        "model": {"field": "pure travelling RO3", "particle": "neutral finite dipole (+q,-q)",
                  "q_ref": qmom.q_ref, "moment_definition": "C_j=sum(q_i/q_ref)*(s_i/R)^j",
                  "a_over_R": a, "ell_over_R": ell, "kappa3": kappa,
                  "gamma_trans": gamma_trans, "gamma_rot": gamma_rot, "inertia": inertia,
                  "C1": float(C1.real), "C3": float(Cm.real), "aperture_radius_over_R": 1.0},
        "analytic_joint_local_band": {"omega_COM_min": omega_com_min, "omega_rot_max": omega_rot_max,
            "spin_target_band": [omega_com_min / m, omega_rot_max / m],
            "note": "Strict local linear stability band in fixed dimensionless coefficients; equality boundaries are marginal."},
        "frequency_sweep": results,
        "full_6d_from_rest_with_com_perturbation": checks,
        "interpretation_limits": [
            "Changing dimensionless omega while holding kappa and dimensionless damping fixed is mathematical continuation, not a fixed-device physical frequency sweep.",
            "Orientation-grid capture fractions are deterministic quadrature estimates, not binomial probabilities or basin volumes.",
            "At the exact COM center the scalar reduction is invariant; selected full-6D cases test finite COM perturbations.",
            "Ideal harmonic field only; no voltage-to-kappa or gas-drag calibration, FEM, noise, or 3D dynamics."
        ],
    }
    outjson = RES / "startup_capture_results.json"
    outjson.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (RES / "startup_capture.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["omega", "spin_target", "eta_drag_to_lock_threshold", "rf_periods", "orientation_count",
                  "distinct_relative_phase_count", "capture_count", "orientation_grid_capture_fraction",
                  "joint_local_linear_stability", "nfev"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})
    CHECKPOINT.unlink(missing_ok=True)

    xs = [r["omega"] for r in results]
    ys = [r["orientation_grid_capture_fraction"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axvspan(omega_com_min, omega_rot_max, color="#d5f5e3", alpha=0.8,
               label="аналитическая совместная локальная устойчивость")
    ax.plot(xs, ys, "o-", color="#1f618d", lw=1.8, ms=5,
            label="захват из spin=0 на равномерной сетке ориентаций")
    ax.axvline(omega_com_min, color="#117864", ls="--", lw=1)
    ax.axvline(omega_rot_max, color="#922b21", ls="--", lw=1)
    ax.set(xlabel=r"безразмерная частота поля $\Omega$", ylabel="доля захваченных ориентаций",
           ylim=(-0.04, 1.04), title="Раскрутка нейтрального конечного диполя в RO3")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "startup_capture_frequency.png", dpi=190)
    plt.close(fig)
    print(json.dumps({"analytic_joint_band": [omega_com_min, omega_rot_max],
                      "sweep": [{"omega": r["omega"], "fraction": r["orientation_grid_capture_fraction"],
                                 "n": r["orientation_count"], "eta": r["eta_drag_to_lock_threshold"]} for r in results],
                      "6d": checks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
