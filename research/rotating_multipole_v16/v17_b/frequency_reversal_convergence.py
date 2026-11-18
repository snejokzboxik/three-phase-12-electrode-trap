"""Tight-step convergence check for the successful RO3 reversal."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v17_b.frequency_reversal import one_ramp

OUT = Path(__file__).resolve().parent
RES = OUT / "results"


def main():
    reference = json.loads((RES / "frequency_reversal_results.json").read_text(encoding="utf-8"))
    coarse = next(r for r in reference["rows"] if r["ramp_duration"] == 20.0)
    fine = one_ramp(20.0, rtol=3e-10, atol=3e-12, max_step=0.25)
    if fine["status"] != "REVERSED_AND_CAPTURED":
        raise RuntimeError(f"refined reversal changed classification: {fine['status']}")
    final_diff = np.asarray(fine["final_state"]) - np.asarray(coarse["final_state"])
    result = {
        "status": "COMPLETED_DOP853_STEP_REFINEMENT",
        "coarse": {"rtol": 2e-9, "atol": 2e-11, "max_step": 0.5,
                   "classification": coarse["status"]},
        "fine": {"rtol": 3e-10, "atol": 3e-12, "max_step": 0.25,
                 "classification": fine["status"], "nfev": fine["nfev"]},
        "max_abs_final_state_difference": float(np.max(np.abs(final_diff))),
        "final_state_difference_components": final_diff.tolist(),
        "fine_max_com_radius": fine["max_com_radius_entire_sampled_trajectory"],
        "fine_final_phase_error": fine["max_abs_final_phase_error"],
        "fine_final_spin_error": fine["max_abs_final_spin_error"],
        "fine_phase_branch_changes": fine["phase_branch_changes_over_protocol"],
    }
    (RES / "frequency_reversal_convergence.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
