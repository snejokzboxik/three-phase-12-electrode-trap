"""Tight-step DOP853 convergence check for the full 6D RO3 chirp."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from v17_b import frequency_ramp as base

OUT = Path(__file__).resolve().parent
RES = OUT / "results"


def main():
    reference = json.loads((RES / "frequency_ramp_results.json").read_text(encoding="utf-8"))
    y0 = np.asarray(reference["protocol"]["initial_state"], float)
    t_end = base.t_end
    t_eval = np.linspace(0.0, t_end, max(10001, int(np.ceil(t_end / 0.25)) + 1))
    sol = solve_ivp(base.rhs, (0.0, t_end), y0, method="DOP853", rtol=3e-10,
                    atol=3e-12, max_step=0.25, events=base.aperture, t_eval=t_eval)
    if not sol.success or sol.status == 1:
        raise RuntimeError(f"refined integration failed: {sol.message}")
    coarse_final = np.asarray(reference["outcome"]["final_state"], float)
    fine_final = sol.y[:, -1]
    phase, omega, _ = np.array([base.drive(float(t)) for t in sol.t]).T
    theta = base.m * sol.y[4] - phase + np.angle(base.Cm)
    eta = base.gamma_rot * base.inertia * np.abs(omega) / (base.m**2 * base.kappa * abs(base.Cm))
    phase_eq = np.pi - np.arcsin(eta)
    phase_err = np.angle(np.exp(1j * (theta - phase_eq)))
    final = sol.t >= t_end - 30 * (2 * np.pi / base.omega_target)
    spin_err = sol.y[5] - omega / base.m
    com = np.hypot(sol.y[0], sol.y[1])
    out = {
        "status": "COMPLETED_DOP853_STEP_REFINEMENT",
        "reference": {"rtol": 2e-9, "atol": 2e-11, "max_step": 0.5},
        "refined": {"rtol": 3e-10, "atol": 3e-12, "max_step": 0.25,
                    "nfev": int(sol.nfev)},
        "max_abs_final_state_difference": float(np.max(np.abs(fine_final - coarse_final))),
        "final_state_difference_components": (fine_final - coarse_final).tolist(),
        "refined_outcome": {
            "mean_spin_last30rf": float(np.mean(sol.y[5, final])),
            "target_spin": base.omega_target / base.m,
            "max_abs_phase_error_last30rf": float(np.max(np.abs(phase_err[final]))),
            "max_abs_spin_error_last30rf": float(np.max(np.abs(spin_err[final]))),
            "max_com_radius_over_trajectory": float(np.max(com)),
            "minimum_body_clearance": float(1 - base.a - np.max(com)),
        },
    }
    (RES / "frequency_ramp_convergence.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
