"""Resource-aware checkpointed launcher for independent native-P4 cases.

Each worker loads one saved h35/h50 FEM solution and processes one launch in
both RF configurations, so rotating/standing comparisons remain matched. The
shared index is rebuilt only after the batch to avoid concurrent writes.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "logs"
RUNNER = HERE / "run_native_p4.py"
PYTHONPATH_VALUE = os.environ.get("PYTHONPATH", "")


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_status():
    state = MEMORYSTATUSEX(); state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        return {"available_gib": None, "total_gib": None, "load_percent": None}
    return {"available_gib": state.ullAvailPhys/2**30,
            "total_gib": state.ullTotalPhys/2**30, "load_percent": state.dwMemoryLoad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, choices=(35, 50), default=50)
    ap.add_argument("--launches", nargs="+", default=(
        "negative_retained", "negative_escaped", "positive_retained", "positive_escaped"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--horizon-rf", type=int, default=1000)
    ap.add_argument("--steps-per-rf", type=int, default=200)
    ap.add_argument("--rtol", type=float, default=2e-9)
    ap.add_argument("--chunk-rf", type=int, default=50)
    args = ap.parse_args()
    if os.name != "nt":
        raise RuntimeError("This resource monitor currently targets the Windows workstation")
    OUT.mkdir(parents=True, exist_ok=True)
    memory = memory_status()
    available = memory["available_gib"]
    # A completed h50 worker used approximately 0.95 GiB in the pilot. h35 has
    # roughly twice the elements, so use a conservative 2.25 GiB estimate;
    # reserve at least 5 GiB for the OS and plotting/postprocessing.
    worker_budget_gib = 2.25 if args.mesh == 35 else 1.5
    if available is not None:
        safe_workers = max(1, int(max(0., available-5.0)//worker_budget_gib))
    else:
        safe_workers = 1
    workers = min(args.workers, safe_workers, len(args.launches))
    run = {"mesh_um": args.mesh, "launches": list(args.launches), "requested_workers": args.workers,
           "workers_started": workers, "initial_memory": memory,
           "memory_rule": f"reserve >=5 GiB, budget {worker_budget_gib} GiB per native-P4 process",
           "horizon_rf": args.horizon_rf, "steps_per_rf_period": args.steps_per_rf,
           "rtol": args.rtol, "chunk_rf": args.chunk_rf,
           "status": "running", "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT/"last_batch.json").write_text(json.dumps(run, indent=2)+"\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH_VALUE
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    queue = list(args.launches)
    active = {}
    exit_codes = {}
    while queue or active:
        while queue and len(active) < workers:
            launch = queue.pop(0)
            log = OUT/f"{args.mesh}um_{launch}_{args.horizon_rf}rf.log"
            fh = log.open("a", encoding="utf-8", buffering=1)
            command = [sys.executable, str(RUNNER), "--mesh", str(args.mesh),
                       "--launches", launch, "--models", "rotating", "standing",
                       "--horizon-rf", str(args.horizon_rf), "--steps-per-rf", str(args.steps_per_rf),
                       "--rtol", str(args.rtol), "--chunk-rf", str(args.chunk_rf), "--defer-index"]
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            active[launch] = (proc, fh, log)
            print(json.dumps({"started": launch, "pid": proc.pid, "log": str(log),
                              "memory": memory_status()}), flush=True)
        time.sleep(10)
        for launch, (proc, fh, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            fh.close()
            exit_codes[launch] = code
            print(json.dumps({"finished": launch, "exit_code": code, "log": str(log),
                              "memory": memory_status()}), flush=True)
            del active[launch]
        # Fail fast; any still-running workers complete their current chunk and
        # will be terminated only if a child itself has failed.
        failures = [k for k, v in exit_codes.items() if v != 0]
        if failures:
            for launch, (proc, fh, log) in active.items():
                proc.terminate(); fh.close()
            run.update(status="failed", exit_codes=exit_codes,
                       finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            (OUT/"last_batch.json").write_text(json.dumps(run, indent=2)+"\n", encoding="utf-8")
            raise SystemExit(f"Worker failure: {failures}; checkpoints preserved")
    from run_native_p4 import write_index
    write_index()
    run.update(status="complete", exit_codes=exit_codes,
               final_memory=memory_status(), finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (OUT/"last_batch.json").write_text(json.dumps(run, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
