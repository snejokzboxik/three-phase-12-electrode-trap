"""Targeted tighter-tolerance and h50 N48 sensitivity integrations.

These are selected checks of the already-computed V7 cases, not a new scan.
They share the original V7 global deadline and checkpoint journal.
"""
from __future__ import annotations
import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import run_v7

HERE = Path(__file__).resolve().parent

STRICT_CASES = [
    ("early_az22p5_dplus", "full_N48", "standing"),
    ("early_az22p5_dplus", "m4_m8", "standing"),
    ("early_az22p5_dplus", "full_N48_minus_m8", "standing"),
    ("early_az22p5_dplus", "full_N48", "rotating"),
    ("boundary_dminus_escaped", "full_N48", "standing"),
    ("boundary_dminus_escaped", "full_N48_minus_m8", "standing"),
    ("boundary_dminus_escaped", "full_N48", "rotating"),
    ("boundary_dplus_escaped", "full_N48", "standing"),
    ("boundary_dplus_escaped", "full_N48_minus_m8", "standing"),
    ("boundary_dplus_escaped", "full_N48", "rotating"),
]
H50_CASES = [
    ("early_az22p5_dplus", "full_N48", "standing"),
    ("early_az22p5_dplus", "full_N48_minus_m8", "standing"),
    ("early_az22p5_dplus", "full_N48", "rotating"),
    ("boundary_dminus_escaped", "full_N48", "standing"),
    ("boundary_dminus_escaped", "full_N48_minus_m8", "standing"),
    ("boundary_dminus_escaped", "full_N48", "rotating"),
    ("boundary_dplus_escaped", "full_N48", "standing"),
    ("boundary_dplus_escaped", "full_N48_minus_m8", "standing"),
    ("boundary_dplus_escaped", "full_N48", "rotating"),
]
HIGH_ORDER_CASES = [
    (launch, rep, "standing")
    for launch in ("boundary_dminus_escaped", "boundary_dplus_escaped")
    for rep in ("m4_m8_m16_m20", "m4_m8_m20", "full_N48_minus_m20")
]
LEAVE_ONE_OUT_CASES = [
    (launch, f"full_N48_minus_m{m}", "standing")
    for launch in ("boundary_dminus_escaped", "boundary_dplus_escaped")
    for m in (16, 28, 32, 40, 44)
]


def make_case(stage, spec):
    launch, rep, model = spec
    base_stage = "pilot" if launch in run_v7.STAGES["pilot"] else "boundary"
    base_rep = rep if rep in run_v7.CORE_REPRESENTATIONS else "full_N48"
    base = next(j for j in run_v7.make_jobs(base_stage)
                if j["launch_label"] == launch and j["representation"] == base_rep and j["model"] == model)
    job = dict(base)
    job["stage"] = stage
    if stage == "strict_h35":
        job["field_source"] = "h35_N48_polynomial"
        job["field_resolution"] = "h35"
        job["rtol"] = 2e-11
        job["atol"] = [2e-14, 2e-14, 2e-12, 2e-12]
        job["work_atol_eV"] = 2e-12
        job["max_step_RF_fraction"] = 0.01
        job["validation_protocol"] = "DOP853 stricter tolerance and half max-step"
    elif stage == "targeted_high_order":
        selection = run_v7.MODEL_SET[rep]
        job["representation"] = rep
        job["retained_harmonic_orders"] = list(selection) if isinstance(selection, tuple) else selection
        job["field_source"] = "h35_N48_polynomial"
        job["field_resolution"] = "h35"
        job["validation_protocol"] = "targeted high-order ablation at previously escaped boundary starts"
    elif stage == "leave_one_out_high_order":
        selection = run_v7.MODEL_SET[rep]
        job["representation"] = rep
        job["retained_harmonic_orders"] = list(selection) if isinstance(selection, tuple) else selection
        job["field_source"] = "h35_N48_polynomial"
        job["field_resolution"] = "h35"
        job["validation_protocol"] = "one-at-a-time deletion of remaining active high harmonics at escaped boundary starts"
    else:
        job["field_source"] = "h50_N48_polynomial"
        job["field_resolution"] = "h50"
        job["validation_protocol"] = "higher-resolution saved FEM-derived N48 reconstruction"
    payload = {k: job[k] for k in sorted(job) if k != "job_id"}
    job["job_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    return job


def run(stage, cases, field_path):
    deadline = float(json.loads((HERE/"integration_deadline.json").read_text(encoding="utf-8"))["deadline_unix_s"])
    done = run_v7._read_done()
    jobs = [make_case(stage, case) for case in cases]
    todo = [j for j in jobs if j["job_id"] not in done]
    print(f"{stage}: queued={len(todo)} workers=4 global_deadline={deadline:.0f}", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=4, initializer=run_v7._init_worker, initargs=(field_path,)) as pool:
            pending = {}
            k = 0
            while k < len(todo) or pending:
                while k < len(todo) and len(pending) < 4:
                    if time.time()+run_v7.DEADLINE_RESERVE_S >= deadline:
                        print("global deadline reserve reached; no additional validation integrations submitted", flush=True)
                        k = len(todo)
                        break
                    pending[pool.submit(run_v7.integrate_job, todo[k])] = todo[k]
                    k += 1
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future)
                    result = future.result()
                    run_v7._append_result(result)
                    done[result["job_id"]] = result
    run_v7._write_csv(list(run_v7._read_done().values()))
    print(f"{stage}: checkpointed={sum(r['job']['stage']==stage for r in run_v7._read_done().values())}/{len(jobs)}", flush=True)


def main():
    run("strict_h35", STRICT_CASES, run_v7.FIELD35)
    run("resolution_h50", H50_CASES, run_v7.FIELD50)
    run("targeted_high_order", HIGH_ORDER_CASES, run_v7.FIELD35)
    run("leave_one_out_high_order", LEAVE_ONE_OUT_CASES, run_v7.FIELD35)


if __name__ == "__main__":
    main()
