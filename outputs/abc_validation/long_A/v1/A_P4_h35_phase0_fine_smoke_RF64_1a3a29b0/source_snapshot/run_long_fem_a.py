"""Start native P4 FEM A at t=0 and save a resumable 50-slow-period run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.long_rf_run import RunStore, run_case, sha, verify_completion, write_json, is_running
from build_direct_fem import OUTPUT as FIELD_OUTPUT, load_field, input_signature

REFERENCE = ROOT/"outputs/abc_validation/direct_fem_validation/v2/validation_settings.json"
OUTPUT = ROOT/"outputs/abc_validation/long_A/v1"


def prepare(args):
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    sources = {}
    for name, expected in reference["kernel_sha256"].items():
        path = ROOT/"src/three_phase_trap"/name
        if sha(path) != expected:
            raise ValueError("Verified numerical kernel changed: "+name)
        sources[str(path.relative_to(ROOT)).replace("\\", "/")] = expected
    for relative in ("src/three_phase_trap/long_rf_run.py", "scripts/run_long_fem_a.py", "scripts/build_direct_fem.py"):
        sources[relative] = sha(ROOT/relative)
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    if versions != reference["versions"]:
        raise ValueError(f"Runtime differs from verified reference: {versions}; expected {reference['versions']}")
    directory = FIELD_OUTPUT/f"h{args.mesh}_P4"
    meta = json.loads((directory/"complete.json").read_text(encoding="utf-8"))
    if meta["signature"] != input_signature(args.mesh, 4):
        raise ValueError("Native P4 field does not match the current mesh/source/runtime")
    inputs = {str(REFERENCE.relative_to(ROOT)).replace("\\", "/"): sha(REFERENCE),
              str((directory/"complete.json").relative_to(ROOT)).replace("\\", "/"): sha(directory/"complete.json")}
    for name, expected in meta["artifacts"].items():
        path = directory/name
        if sha(path) != expected:
            raise ValueError("Native P4 field cache checksum mismatch: "+name)
        inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = expected
    for name in ("trap.msh", "sector.msh"):
        path = ROOT/f"outputs/meshes/gmsh_{args.mesh}um"/name
        inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
    ts, p = reference["estimated_slow_period_s"], reference["physical"]
    cycles = 64 if args.smoke_test else math.ceil(args.slow_periods*ts*p["frequency_Hz"])
    settings = dict(schema=1, model="A: direct unsmoothed native FEM P4", start_time_s=0.,
                    mesh_um=args.mesh, FEM_degree=4, phase_index=args.phase_index, phase_rad=p["phase_rad"][args.phase_index],
                    physical=p, initial_state=reference["full_initial_states"][args.phase_index],
                    requested_slow_periods=None if args.smoke_test else args.slow_periods,
                    reference_slow_period_s=ts, RF_periods=cycles, chunk_RF_periods=32 if args.smoke_test else args.chunk_rf_periods,
                    samples_per_RF_period=64, control=reference["controls"][args.level], R_scale_m=reference["R_scale_m"],
                    RF_clock="Exact integer RF cycles; local segment clock with unchanged phi0; absolute sample times from integer RF sample indices",
                    smoke_test=args.smoke_test, versions=versions, source_sha256=sources, input_sha256=inputs,
                    numerical_validation_status="Not yet established for the new long time horizon")
    period_tag = f"{args.slow_periods:g}".replace(".", "p")
    smoke_source = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:8]
    tag = f"A_P4_h{args.mesh}_phase{args.phase_index}_{args.level}_"+("smoke_RF64_"+smoke_source if args.smoke_test else period_tag+"Ts_from_zero")
    tag = args.run_name or tag
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", tag):
        raise ValueError("Run name must be a simple name, without directory separators")
    return settings, OUTPUT/tag


def show_status(directory):
    if not (directory/"settings.json").exists():
        print("[NOT STARTED] No run has been launched in this directory.", flush=True)
        return
    settings = json.loads((directory/"settings.json").read_text(encoding="utf-8"))
    if (directory/"COMPLETED.json").exists():
        marker = json.loads((directory/"COMPLETED.json").read_text(encoding="utf-8"))
        print(f"[COMPLETED] t={marker['end_time_s']:.8f} s; all {marker['segments']} segments saved. Use --verify to check checksums.", flush=True)
        return
    path = directory/"status.json"
    if not path.exists():
        print("[PREPARING] Frozen settings exist; no progress record yet.", flush=True)
        return
    status = json.loads(path.read_text(encoding="utf-8"))
    active = is_running(directory)
    stage = status["status"].upper() if active or status["status"] in ("failed", "interrupted") else "INTERRUPTED"
    print(f"[{stage}] {status['progress_percent']:.2f}%; {status['slow_periods']:.3f} reference slow periods; saved checkpoints {status['committed_segments']}/{status['total_segments']}", flush=True)
    print(f"Last update (UTC): {status['updated_utc']}", flush=True)
    print(f"Resume point: t={status['checkpoint_time_s']:.8f} s; target t={settings['RF_periods']/settings['physical']['frequency_Hz']:.8f} s", flush=True)
    if not active:
        print("No writer is running. Start the same command to resume saved checkpoints.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow-periods", type=float, default=50.)
    parser.add_argument("--mesh", type=int, choices=(35, 50), default=35)
    parser.add_argument("--level", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--phase-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--chunk-rf-periods", type=int, default=2000)
    parser.add_argument("--run-name")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="64 RF periods, two segments and a one-shot comparison; separate directory")
    args = parser.parse_args()
    if not np.isfinite(args.slow_periods) or args.slow_periods <= 0 or args.chunk_rf_periods < 1:
        parser.error("Expected positive duration and checkpoint interval")
    print("[CHECKING] Existing P4 field, frozen numerical sources and physical initial state...", flush=True)
    settings, directory = prepare(args)
    print("Output: "+str(directory), flush=True)
    target = settings["RF_periods"]/settings["physical"]["frequency_Hz"]
    print(f"Plan: t=0 -> {target:.8f} s = {target/settings['reference_slow_period_s']:.8f} reference slow periods; {settings['RF_periods']} RF periods.", flush=True)
    if args.check_only:
        print("[READY] Inputs verified. No trajectory was started; no old trajectory is used.", flush=True)
        return
    if args.status:
        show_status(directory)
        return
    if args.verify:
        if not (directory/"COMPLETED.json").exists():
            raise ValueError("No COMPLETED.json: trajectory has not finished")
        frozen = json.loads((directory/"settings.json").read_text(encoding="utf-8"))
        if frozen != settings:
            raise ValueError("Requested settings differ from the finished run")
        marker = verify_completion(RunStore(directory, frozen))
        print(f"[VERIFIED] {len(marker['artifacts'])} final artifacts and all {marker['segments']} checkpoints passed checksum/state/time checks.", flush=True)
        return
    store = RunStore(directory, settings)
    def field_factory():
        snapshot = directory/"source_snapshot"
        snapshot.mkdir(exist_ok=True)
        manifest = {}
        for name, expected in settings["source_sha256"].items():
            path = ROOT/name
            if sha(path) != expected:
                raise ValueError("Numerical source changed after preflight: "+name)
            shutil.copy2(path, snapshot/path.name)
            manifest[path.name] = expected
        write_json(directory/"source_manifest.json", manifest)
        return load_field(args.mesh, 4, settings["physical"]["U0_V"])
    run_case(store, field_factory, notify=lambda text: print(text, flush=True))
    print("Result: "+str(directory/"trajectory.npz"), flush=True)
    print("Completion marker: "+str(directory/"COMPLETED.json"), flush=True)
    print("The trajectory is computed. Convergence and B/C agreement at this longer duration are a separate check.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Last committed checkpoint is safe. Run the same command to resume.", flush=True)
        sys.exit(130)
    except Exception as exc:
        print("[ERROR] "+str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
