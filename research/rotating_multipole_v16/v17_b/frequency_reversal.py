"""Test whether a locked RO3 dipole can reverse field rotation in full 6D."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from v16_1.dynamics import Harmonic, rigid_rhs
from v17_b.model import charge_moments, synchronous_branch

OUT = Path(__file__).resolve().parent
RES, FIG = OUT / "results", OUT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

m, a, ell = 3, 0.1, 0.2
kappa, gamma_trans, gamma_rot = 0.03, 0.05, 0.02
inertia = ell * ell / 12
omega_i, omega_f = 0.5, -0.5
final_rf = 1000
com_initial = 0.001
mom = charge_moments([1.0, -1.0], [a, -a], 3)
branch = synchronous_branch(m, kappa, omega_i, mom.moments, gamma_rot, inertia)
psi_i = branch["phase"]
omega_com_min = 0.3887674821559735


def one_ramp(duration: float, *, rtol: float = 2e-9, atol: float = 2e-11,
             max_step: float = 0.5):
    rate = (omega_f - omega_i) / duration
    t_end = duration + final_rf * 2 * math.pi / abs(omega_f)

    def drive(t):
        if t <= duration:
            return omega_i * t + 0.5 * rate * t * t, omega_i + rate * t
        phase_end = 0.5 * (omega_i + omega_f) * duration
        return phase_end + omega_f * (t - duration), omega_f

    def rhs(t, y):
        phase, _ = drive(t)
        h = Harmonic(m, "RO", kappa, 0.0, -phase)
        return rigid_rhs(t, y, [h], ell, gamma_trans, gamma_rot, inertia, mom.moments)

    def aperture(_t, y):
        return 1 - a - math.hypot(y[0], y[1])
    aperture.terminal = True
    aperture.direction = -1

    y0 = np.array([com_initial, 0.0, 0.0, 0.0, psi_i, omega_i / m])
    t_eval = np.linspace(0.0, t_end, max(5001, int(math.ceil(t_end / 0.25)) + 1))
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", rtol=rtol,
                    atol=atol, max_step=max_step, events=aperture, t_eval=t_eval)
    if sol.status == 1:
        hit = sol.y_events[0][0]
        return {"ramp_duration": duration, "ramp_rate": rate, "status": "ESCAPED_APERTURE",
                "escape_time": float(sol.t_events[0][0]),
                "com_radius_at_escape": float(math.hypot(hit[0], hit[1])),
                "body_radius_at_escape": float(math.hypot(hit[0], hit[1]) + a),
                "nfev": int(sol.nfev)}
    if not sol.success:
        return {"ramp_duration": duration, "ramp_rate": rate, "status": "INTEGRATION_FAILED",
                "message": sol.message, "nfev": int(sol.nfev)}

    t, y = sol.t, sol.y
    data = np.array([drive(float(q)) for q in t])
    phi, omega = data.T
    theta = m * y[4] - phi + np.angle(mom.moments[m])
    ramp_rate_series = rate * (t <= duration)
    eta_rate = (gamma_rot * inertia * omega + inertia * ramp_rate_series) / (m * m * kappa * abs(mom.moments[m]))
    theta_eq = np.pi - np.arcsin(np.clip(eta_rate, -1, 1))
    phase_error = np.angle(np.exp(1j * (theta - theta_eq)))
    spin_error = y[5] - omega / m
    com = np.hypot(y[0], y[1])
    final_mask = t >= t_end - 30 * 2 * math.pi / abs(omega_f)
    ramp_mask = t <= duration
    phase_branch = np.round((theta - theta_eq) / (2 * math.pi))
    captured = bool(np.max(np.abs(phase_error[final_mask])) < 0.08 and
                    np.max(np.abs(spin_error[final_mask])) < 0.002)
    unstable_duration = duration * min(1.0, 2 * omega_com_min / abs(omega_i - omega_f))
    return {
        "ramp_duration": duration,
        "ramp_rate": rate,
        "status": "REVERSED_AND_CAPTURED" if captured else "NOT_CAPTURED_AT_TARGET",
        "target_spin": omega_f / m,
        "mean_spin_last30rf": float(np.mean(y[5, final_mask])),
        "max_abs_final_phase_error": float(np.max(np.abs(phase_error[final_mask]))),
        "max_abs_quasistatic_phase_error_during_ramp": float(np.max(np.abs(phase_error[ramp_mask]))),
        "phase_branch_changes_over_protocol": int(np.count_nonzero(np.diff(phase_branch))),
        "max_abs_final_spin_error": float(np.max(np.abs(spin_error[final_mask]))),
        "max_com_radius_entire_sampled_trajectory": float(np.max(com)),
        "body_clearance_min_entire_sampled_trajectory": float(1 - a - np.max(com)),
        "branch_exists_during_ramp": bool(np.max(np.abs(eta_rate[t <= duration])) < 1),
        "time_instantaneous_COM_band_unstable": unstable_duration,
        "nfev": int(sol.nfev),
        "tolerance": {"rtol": rtol, "atol": atol, "max_step": max_step,
                      "output_spacing_max": 0.25},
        "final_state": y[:, -1].tolist(),
        "_plot": {"time": t.tolist(), "spin": y[5].tolist(), "omega": omega.tolist(),
                  "com": com.tolist()},
    }


def main():
    durations = [20.0, 40.0, 80.0]
    rows = []
    for duration in durations:
        row = one_ramp(duration)
        rows.append(row)
        print(json.dumps({"ramp_duration": duration, "status": row["status"],
                          "escape_time": row.get("escape_time"), "nfev": row.get("nfev")},
                         ensure_ascii=False), flush=True)
    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    result = {
        "status": "COMPLETED_FULL_6D_DIRECTION_REVERSAL_SLICE",
        "model": {"field": "pure RO3", "particle": "finite neutral dipole (+q,-q)",
                  "a_over_R": a, "ell_over_R": ell, "q_ref": mom.q_ref,
                  "C1": float(mom.moments[1].real), "C3": float(mom.moments[3].real),
                  "kappa3": kappa, "gamma_trans": gamma_trans, "gamma_rot": gamma_rot,
                  "inertia": inertia},
        "initial_state": {"x_over_R": com_initial, "y_over_R": 0.0,
                          "psi": psi_i, "spin": omega_i / m,
                          "field_omega": omega_i},
        "protocol": {"field_omega_start": omega_i, "field_omega_target": omega_f,
                     "final_hold_rf_periods": final_rf,
                     "local_joint_COM_stability_requires": f"|Omega|>{omega_com_min:.9f}"},
        "rows": public_rows,
        "limits": ["one radial COM perturbation, one initial phase and three linear ramp durations",
                   "fixed dimensionless coefficients; no physical RF recalibration",
                   "ideal harmonics; no FEM or 3D tilt"],
    }
    (RES / "frequency_reversal_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=False)
    for r in rows:
        if "_plot" not in r:
            if r["status"] == "ESCAPED_APERTURE":
                te = r["escape_time"]
                ax1.scatter([te], [0.0], marker="x", s=55, color="#c0392b",
                            label=f"T_ramp={r['ramp_duration']:g}: escaped")
                ax2.scatter([te], [r["com_radius_at_escape"]], marker="x", s=55,
                            color="#c0392b", label=f"T_ramp={r['ramp_duration']:g}: escaped")
            continue
        p = r["_plot"]
        t, spin, target, com = map(np.asarray, (p["time"], p["spin"], p["omega"], p["com"]))
        stride = max(1, len(t) // 5000)
        ax1.plot(t[::stride], spin[::stride], label=f"T_ramp={r['ramp_duration']:g}")
        ax1.plot(t[::stride], target[::stride] / m, ls="--", alpha=0.65,
                 label="Ω(t)/3 target" if r["ramp_duration"] == durations[0] else None)
        ax2.plot(t[::stride], np.maximum(com[::stride], 1e-16), label=f"T_ramp={r['ramp_duration']:g}")
    ax1.axhline(0, color="k", lw=0.7)
    ax1.set_ylabel("spin")
    ax1.legend()
    ax2.set_yscale("log")
    ax2.set_ylim(1e-16, 1)
    ax2.set_ylabel("COM radius / R")
    ax2.set_xlabel("dimensionless time")
    ax2.legend()
    ax2.grid(alpha=0.25)
    fig.suptitle("Full 6D reversal of RO3 field rotation")
    fig.tight_layout()
    fig.savefig(FIG / "frequency_reversal_6d.png", dpi=190)
    plt.close(fig)
    print(json.dumps({"rows": public_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
