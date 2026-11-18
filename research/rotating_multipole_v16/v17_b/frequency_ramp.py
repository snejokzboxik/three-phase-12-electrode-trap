"""Adiabatic RO3 frequency ramp to steer a finite neutral dipole's spin.

This is a dimensionless control experiment in the ideal harmonic model.  The
full six-state laboratory ODE is integrated with a time-dependent RF phase;
no frequency rescaling for a particular physical device is implied.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from v16_1.dynamics import Harmonic, rigid_rhs
from v17_b.model import charge_moments

OUT = Path(__file__).resolve().parent
RES, FIG = OUT / "results", OUT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

m = 3
a = 0.1
ell = 2 * a
kappa = 0.03
gamma_trans = 0.05
gamma_rot = 0.02
inertia = ell**2 / 12
omega_start = 0.4
omega_target = 3.5
prehold_rf = 1500
ramp_rf_at_start = 1000
final_hold_rf = 500
psi0 = math.pi / 3
com_initial = 1e-4
mom = charge_moments([1.0, -1.0], [a, -a], m)
Cm = mom.moments[m]
A = m * m * kappa * abs(Cm) / inertia

period_start = 2 * math.pi / omega_start
t_pre = prehold_rf * period_start
t_ramp = ramp_rf_at_start * period_start
t_end = t_pre + t_ramp + final_hold_rf * (2 * math.pi / omega_target)
omega_dot = (omega_target - omega_start) / t_ramp


def drive(t: float):
    if t <= t_pre:
        return omega_start * t, omega_start, 0.0
    if t <= t_pre + t_ramp:
        u = t - t_pre
        return omega_start * t_pre + omega_start * u + 0.5 * omega_dot * u * u, omega_start + omega_dot * u, omega_dot
    u = t - t_pre - t_ramp
    phi_ramp = omega_start * t_ramp + 0.5 * omega_dot * t_ramp**2
    phi0 = omega_start * t_pre + phi_ramp
    return phi0 + omega_target * u, omega_target, 0.0


def rhs(t, y):
    phase, _, _ = drive(t)
    # Harmonic's delta enters as exp(+i delta); delta=-phase therefore gives
    # the desired exp(-i Phi(t)) in the travelling field.
    h = Harmonic(m, "RO", kappa, 0.0, -phase)
    return rigid_rhs(t, y, [h], ell, gamma_trans, gamma_rot, inertia, mom.moments)


def aperture(t, y):
    return 1 - a - math.hypot(y[0], y[1])
aperture.terminal = True
aperture.direction = -1


def main():
    # Begin near the stable phase at the start frequency, with zero spin; the
    # finite-dipole COM receives a small position perturbation.
    y0 = np.array([com_initial, 0.0, 0.0, 0.0, psi0, 0.0])
    # Resolve the full COM envelope at <=0.25 dimensionless-time spacing;
    # the ODE max step is 0.5 and dense output fills the sampling grid.
    n_samples = max(10001, int(math.ceil(t_end / 0.25)) + 1)
    t_eval = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", rtol=2e-9,
                    atol=2e-11, max_step=0.5, events=aperture, t_eval=t_eval)
    if not sol.success or sol.status == 1:
        raise RuntimeError(f"frequency-ramp integration failed: {sol.message}; event={sol.t_events[0]}")

    t, y = sol.t, sol.y
    phase_data = np.array([drive(float(q)) for q in t])
    phi, omega, omega_rate = phase_data.T
    theta = m * y[4] - phi + np.angle(Cm)
    eta = gamma_rot * inertia * np.abs(omega) / (m * m * kappa * abs(Cm))
    theta_eq = math.pi - np.arcsin(eta)
    phase_error = np.angle(np.exp(1j * (theta - theta_eq)))
    com = np.hypot(y[0], y[1])
    target_spin = omega / m
    spin_error = y[5] - target_spin
    final_mask = t >= t_end - 30 * (2 * math.pi / omega_target)
    ramp_mask = (t >= t_pre) & (t <= t_pre + t_ramp)
    pre_mask = t <= t_pre
    post_mask = t >= t_pre + t_ramp
    final_phase = phase_error[final_mask]
    final_spin_error = spin_error[final_mask]
    # An unwrapped resonance phase must not cross a neighbouring 2pi branch
    # during the ramp; the error is measured from the instantaneous stable
    # phase, whose value changes smoothly with omega.
    ramp_unwrapped_error = np.unwrap(theta[ramp_mask] - theta_eq[ramp_mask])
    branch_index = np.round((theta[ramp_mask] - theta_eq[ramp_mask]) / (2 * math.pi))
    branch_changes = int(np.count_nonzero(np.diff(branch_index)))
    captured = bool(np.max(np.abs(final_phase)) < 0.08 and np.max(np.abs(final_spin_error)) < 0.002)
    result = {
        "status": "COMPLETED_FULL_6D_FREQUENCY_RAMP",
        "model": {"field": "pure RO3 with chirped phase", "particle": "finite neutral dipole (+q,-q)",
                  "q_ref": mom.q_ref, "a_over_R": a, "ell_over_R": ell,
                  "C1": float(mom.moments[1].real), "C3": float(Cm.real),
                  "kappa3": kappa, "gamma_trans": gamma_trans, "gamma_rot": gamma_rot,
                  "inertia": inertia, "aperture_radius_over_R": 1.0},
        "protocol": {"start_omega": omega_start, "target_omega": omega_target,
                     "start_spin": 0.0, "target_spin": omega_target / m,
                     "prehold_rf_periods": prehold_rf, "linear_ramp_rf_periods_at_start": ramp_rf_at_start,
                     "final_hold_rf_periods": final_hold_rf, "ramp_rate": omega_dot,
                     "initial_state": y0.tolist()},
        "analytic_limits": {"COM_lower_frequency_bound": 0.3887674821559735,
                            "rotational_lock_upper_frequency_bound": m*m*kappa*abs(Cm)/(inertia*gamma_rot),
                            "frequency_status": "both endpoints lie inside the fixed-dimensionless-coefficient joint local-stability band"},
        "integration": {"method": "DOP853", "rtol": 2e-9, "atol": 2e-11,
                        "max_step": 0.5, "nfev": int(sol.nfev), "samples": int(len(t)),
                        "t_final": float(t[-1]), "aperture_event": False},
        "outcome": {"captured_at_target": captured,
                    "mean_spin_last30rf": float(np.mean(y[5, final_mask])),
                    "final_state": y[:, -1].tolist(),
                    "target_spin": omega_target / m,
                    "max_abs_final_phase_error": float(np.max(np.abs(final_phase))),
                    "max_abs_final_spin_error": float(np.max(np.abs(final_spin_error))),
                    "max_com_radius_over_trajectory": float(np.max(com)),
                    "max_com_radius_prehold": float(np.max(com[pre_mask])),
                    "max_com_radius_ramp": float(np.max(com[ramp_mask])),
                    "max_com_radius_final_hold": float(np.max(com[post_mask])),
                    "body_clearance_min": float(1 - a - np.max(com)),
                    "max_abs_resonance_phase_error_during_ramp": float(np.max(np.abs(ramp_unwrapped_error))),
                    "phase_branch_changes_during_ramp": branch_changes,
                    "spin_error_rms_during_ramp": float(np.sqrt(np.mean(spin_error[ramp_mask]**2)))},
        "limits": ["dimensionless mathematical control only; kappa and drag were held fixed while omega changed",
                   "ideal harmonic field; no calibrated electrode voltages, gas model, FEM, noise, or 3D tilt"],
    }
    (RES / "frequency_ramp_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    plot_idx = np.linspace(0, len(t) - 1, min(10001, len(t))).astype(int)
    ax1.plot(t[plot_idx], y[5, plot_idx], lw=1.1, label="spin(t)")
    ax1.plot(t[plot_idx], target_spin[plot_idx], "--", lw=1.1, label="Ω(t)/3 target")
    ax1.axvline(t_pre, color="0.5", ls=":", label="ramp starts/ends")
    ax1.axvline(t_pre + t_ramp, color="0.5", ls=":")
    ax1.set_ylabel("dimensionless angular speed")
    ax1.legend()
    ax2.plot(t[plot_idx], np.maximum(com[plot_idx], 1e-16), color="#117864", lw=1.0)
    ax2.set_ylabel("COM radius / R")
    ax2.set_xlabel("dimensionless time")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-16, max(2e-4, 2 * float(np.max(com))))
    ax2.grid(alpha=0.25)
    fig.suptitle("Full 6D RO3 frequency ramp: capture, then controlled spin-up")
    fig.tight_layout()
    fig.savefig(FIG / "frequency_ramp_6d.png", dpi=190)
    plt.close(fig)
    print(json.dumps({"captured_at_target": captured, "nfev": sol.nfev,
                      "outcome": result["outcome"], "protocol": result["protocol"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
