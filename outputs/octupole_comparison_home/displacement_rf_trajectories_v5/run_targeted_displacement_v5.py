"""Checkpointed, matched RF trajectories on the existing h50 full-domain P4 fields."""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time, ctypes, shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "pilot_config_v5.json").read_text(encoding="utf-8"))
V1 = ROOT / "outputs/octupole_comparison_home/electrode_displacement_sensitivity_v1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V1))
sys.path.insert(0, str(V1 / ".deps"))
import field_analysis as fa

E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
FREQ = float(CONFIG["rf"]["frequency_Hz"])
OMEGA = 2 * math.pi * FREQ
PERIOD = 1 / FREQ
MASS = float(CONFIG["ion"]["mz_Th"]) * AMU
QM = E_CHARGE / MASS
VOLTAGE = float(CONFIG["rf"]["rotating_voltage_V"])
APERTURE = float(CONFIG["aperture_radius_mm"]) * 1e-3
HORIZON_RF = int(CONFIG["horizon_RF_periods"])
SEGMENT_RF = int(CONFIG["solver"]["checkpoint_segment_RF"])
SAMPLES_PER_RF = int(CONFIG["solver"]["samples_per_RF"])


def utc_now():
    return datetime.now(timezone.utc)


def atomic_json(path: Path, item: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_checkpoint(row: dict):
    p = HERE / "checkpoints.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"time_utc": utc_now().isoformat(timespec="seconds"), **row}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_global_deadline(min_remaining_s=180):
    budget_path = ROOT / "outputs/octupole_comparison_home/matched_pseudopotential_v5/integration_deadline.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    remaining = float(budget["deadline_unix_s"]) - time.time()
    atomic_json(HERE / "global_deadline_status.json", {
        "started_utc": budget["started_utc"], "deadline_unix_s": budget["deadline_unix_s"],
        "checked_utc": utc_now().isoformat(timespec="seconds"), "remaining_seconds": remaining,
        "minimum_to_start_segment_seconds": min_remaining_s,
        "resources": system_resources(),
    })
    if remaining < min_remaining_s:
        raise TimeoutError(f"Global V5 numerical deadline guard: only {remaining:.1f}s remain.")
    return remaining


def system_resources():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    mem = MemoryStatus()
    mem.dwLength = ctypes.sizeof(mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    proc = ProcessMemoryCountersEx()
    proc.cb = ctypes.sizeof(proc)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    read_process_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    read_process_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCountersEx), ctypes.c_ulong]
    read_process_memory.restype = ctypes.c_int
    if not read_process_memory(get_process(), ctypes.byref(proc), proc.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    disk = shutil.disk_usage(ROOT)
    return {"system_available_RAM_GiB": mem.ullAvailPhys / 2**30,
            "system_memory_load_percent": int(mem.dwMemoryLoad),
            "process_working_set_GiB": proc.WorkingSetSize / 2**30,
            "process_peak_working_set_GiB": proc.PeakWorkingSetSize / 2**30,
            "process_private_bytes_GiB": proc.PrivateUsage / 2**30,
            "free_disk_GiB": disk.free / 2**30}


def physical_initial():
    c = CONFIG["initial_condition"]
    r = float(c["r0_mm"]) * 1e-3
    th = math.radians(float(c["azimuth_deg"]))
    sign = int(c["tangential_direction"])
    if sign not in (-1, 1):
        raise ValueError("Tangential direction must be exactly -1 or +1.")
    speed = math.sqrt(2 * float(c["energy_eV"]) * E_CHARGE / MASS)
    y0 = np.array([r * math.cos(th), r * math.sin(th), -sign * speed * math.sin(th), sign * speed * math.cos(th)], float)
    tangent = np.array([-math.sin(th), math.cos(th)])
    observed_sign = int(np.sign(np.dot(y0[2:], tangent)))
    if observed_sign != sign:
        raise AssertionError(f"Velocity direction audit failed: requested {sign}, state gives {observed_sign}.")
    return y0, {**c, "mz_Th": CONFIG["ion"]["mz_Th"], "charge_state": 1,
                "frequency_Hz": FREQ, "voltage_V": VOLTAGE,
                "x0_m": float(y0[0]), "y0_m": float(y0[1]), "vx0_m_s": float(y0[2]), "vy0_m_s": float(y0[3]),
                "speed_m_s": speed, "verified_tangential_sign": observed_sign}


def load_h50_field(geometry: str):
    if geometry == "nominal":
        fa.NOMINAL_MESH = ROOT / "outputs/meshes/gmsh_50um/trap.msh"
        fa.SECTOR_SOLUTION = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz"
        field, meta = fa.nominal_field()
        meta["resolution"] = "h50"
        meta["representation"] = "validated nominal h50 P4 sector FEM coefficients mapped onto the exact twelve-copy full nominal h50 mesh; not N48"
        return field, meta
    folder = V1 / "calculations" / f"{geometry}_h50"
    if not folder.exists():
        raise FileNotFoundError(f"No existing h50 full-domain field solution for {geometry}: {folder}")
    field, meta = fa.field_from_coefficients(folder / "displaced_h50.msh", folder / "p4_coefficients.npz")
    meta["resolution"] = "h50"
    meta["geometry"] = geometry
    meta["representation"] = "existing full-asymmetry P4 FEM solution on displaced h50 mesh"
    return field, meta


def row_from_state(t, y):
    return [float(t), float(t * FREQ), float(y[0] * 1e3), float(y[1] * 1e3),
            float(math.hypot(y[0], y[1]) * 1e3), float(y[2]), float(y[3])]


def read_existing_samples(trace_dir):
    t, r = [], []
    for path in sorted(trace_dir.glob("segment_*.csv")):
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t.append(float(row["rf_period"]))
                r.append(float(row["r_mm"]))
    return np.asarray(t), np.asarray(r)


def run_case(field, meta, geometry, model, *, tight=False, stop_after_one_segment=False):
    check_global_deadline()
    y_initial, ic = physical_initial()
    key = f"{geometry}_{model}_{'tight' if tight else 'standard'}"
    trace_dir = HERE / "trajectories" / key
    trace_dir.mkdir(parents=True, exist_ok=True)
    progress_path = trace_dir / "progress.json"
    summary_path = trace_dir / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("complete"):
            return old

    rtol = 2e-11 if tight else float(CONFIG["solver"]["rtol"])
    atol = (2e-14, 2e-14, 2e-12, 2e-12) if tight else (2e-13, 2e-13, 2e-11, 2e-11)
    max_step = PERIOD / (100 if tight else round(1 / CONFIG["solver"]["max_step_RF_fraction"]))
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            return progress
        state = np.asarray(progress["state"], float)
        t0 = float(progress["time_s"])
        seg_ix = int(progress["next_segment"])
        nfev = int(progress.get("nfev", 0))
        accumulated_wall = float(progress.get("integration_wall_seconds", 0))
    else:
        state, t0, seg_ix, nfev, accumulated_wall = y_initial.copy(), 0.0, 0, 0, 0.0

    is_rotating = model == "rotating"
    if model not in ("rotating", "standing"):
        raise ValueError(model)
    phase0 = math.radians(float(CONFIG["initial_condition"]["rf_phase_deg"]))

    def rhs(t, y):
        _, ec, es = field.sample_one(float(y[0]), float(y[1]))
        phase = OMEGA * t + phase0
        rf = VOLTAGE * ((ec * math.cos(phase) if is_rotating else 0.0) + es * math.sin(phase))
        return (y[2], y[3], QM * rf[0], QM * rf[1])

    def aperture_event(t, y):
        return APERTURE - math.hypot(y[0], y[1])
    aperture_event.terminal = True
    aperture_event.direction = -1

    horizon_t = HORIZON_RF * PERIOD
    seg_started = time.perf_counter()
    while t0 < horizon_t - 1e-18:
        check_global_deadline()
        t1 = min(horizon_t, t0 + SEGMENT_RF * PERIOD)
        count = int(round((t1 - t0) * FREQ * SAMPLES_PER_RF))
        t_eval = np.linspace(t0, t1, count + 1)
        start = time.perf_counter()
        sol = solve_ivp(rhs, (t0, t1), state, method="DOP853", rtol=rtol, atol=atol,
                        max_step=max_step, t_eval=t_eval, events=aperture_event)
        if not sol.success:
            raise RuntimeError(f"{key}: {sol.message}")
        times, states = sol.t.copy(), sol.y.T.copy()
        escaped = bool(len(sol.t_events[0]))
        escape_time = float(sol.t_events[0][0]) if escaped else None
        if escaped:
            event_state = sol.y_events[0][0]
            if len(times) == 0 or abs(times[-1] - escape_time) > 1e-15:
                times = np.r_[times, escape_time]
                states = np.vstack((states, event_state))
            else:
                states[-1] = event_state
        segment = trace_dir / f"segment_{seg_ix:03d}.csv"
        with segment.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "rf_period", "x_mm", "y_mm", "r_mm", "vx_m_s", "vy_m_s"])
            w.writerows(row_from_state(t, y) for t, y in zip(times, states))
            f.flush()
            os.fsync(f.fileno())
        final_state = states[-1]
        t0, state = float(times[-1]), final_state.copy()
        nfev += int(sol.nfev)
        segment_wall = time.perf_counter() - start
        accumulated_wall += segment_wall
        seg_ix += 1
        complete = escaped or t0 >= horizon_t - 1e-18
        progress = {"key": key, "geometry": geometry, "model": model, "tight": tight,
                    "initial_state": y_initial.tolist(), "initial_conditions": ic,
                    "rtol": rtol, "atol": list(atol), "max_step_s": max_step,
                    "time_s": t0, "rf_period": t0 * FREQ, "state": state.tolist(),
                    "next_segment": seg_ix, "nfev": nfev,
                    "segment_wall_seconds": segment_wall,
                    "integration_wall_seconds": accumulated_wall,
                    "complete": complete,
                    "status": "escaped" if escaped else "retained_through_horizon" if complete else "incomplete_checkpoint",
                    "escape_time_s": escape_time,
                    "escape_time_RF": escape_time * FREQ if escape_time is not None else None,
                    "right_censor_time_RF": None if escaped else HORIZON_RF,
                    "field_representation": meta}
        atomic_json(progress_path, progress)
        append_checkpoint({"key": key, "segment": seg_ix, "rf_period": progress["rf_period"],
                           "status": progress["status"], "segment_wall_seconds": segment_wall,
                           "nfev": nfev})
        print(f"[{key}] checkpoint {progress['rf_period']:.0f}/{HORIZON_RF} RF; {progress['status']}; {segment_wall:.1f}s", flush=True)
        if complete or stop_after_one_segment:
            break

    if stop_after_one_segment:
        return {"key": key, "complete": False, "benchmark_checkpoint": progress,
                "segment_wall_seconds": progress["segment_wall_seconds"]}
    if not progress.get("complete"):
        return progress

    periods, radii_mm = read_existing_samples(trace_dir)
    peaks, _ = find_peaks(radii_mm, prominence=0.05)
    peak_periods = periods[peaks]
    peak_gaps = np.diff(peak_periods)
    summary = {**progress,
        "max_radius_mm_sampled": float(np.max(radii_mm)),
        "rms_radius_mm_sampled": float(np.sqrt(np.mean(radii_mm ** 2))),
        "minimum_radius_mm_sampled": float(np.min(radii_mm)),
        "n_radial_samples": int(len(radii_mm)),
        "radial_peak_prominence_mm": 0.05,
        "radial_oscillation_peak_count": int(len(peaks)),
        "median_radial_peak_spacing_RF": float(np.median(peak_gaps)) if len(peak_gaps) else None,
        "first_radial_peak_RF": float(peak_periods[0]) if len(peak_periods) else None,
        "last_radial_peak_RF": float(peak_periods[-1]) if len(peak_periods) else None,
        "convergence_status": f"{'tight' if tight else 'standard'} DOP853, rtol={rtol:g}, max_step={max_step / PERIOD:g} RF periods",
        "interpretation": "finite-time escape or right-censored 1000-RF survival; no permanent-stability claim"}
    atomic_json(summary_path, summary)
    append_checkpoint({"key": key, "stage": "trajectory_summary_written", "status": summary["status"],
                       "escape_time_RF": summary["escape_time_RF"], "max_radius_mm_sampled": summary["max_radius_mm_sampled"]})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-only", action="store_true", help="continue nominal rotating case through one 100-RF segment")
    ap.add_argument("--tight-validation", action="store_true", help="rerun selected standing boundary cases at tighter tolerances")
    args = ap.parse_args()
    check_global_deadline(600)
    y0, ic = physical_initial()
    # Preserve the V4 input exactly and audit the CW tangential sign.
    source = ROOT / "outputs/octupole_comparison_home/confinement_maps_v4/trajectory_results_v4.csv"
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    matched = [r for r in rows if r.get("model") == "standing" and r.get("energy_eV") == "0.75"
               and r.get("r0_mm") == "1.0" and r.get("azimuth_rad") == str(math.radians(67.5))
               and r.get("phase_rad") == str(math.radians(270.0)) and r.get("direction") == "-1.0"]
    if len(matched) != 1:
        raise RuntimeError(f"Expected one exact V4 reference trajectory, found {len(matched)}")
    v4_reference = matched[0]
    if not np.allclose(y0, [float(v4_reference[k]) for k in ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s")], rtol=0, atol=1e-13):
        raise AssertionError("Reconstructed physical launch differs from the archived V4 initial state.")
    atomic_json(HERE / "initial_condition_audit.json", {
        "status": "passed", "initial_conditions": ic,
        "V4_reference_initial_state": [float(v4_reference[k]) for k in ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s")],
        "max_component_difference": float(np.max(np.abs(y0 - np.array([float(v4_reference[k]) for k in ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s")])))),
        "V4_reference_model": "standing", "V4_reference_escape_time_RF": float(v4_reference["escape_time_RF"]),
        "V4_reference_classification": "rotating retained through 1000 RF; standing escaped",
        "direction": "negative tangential (clockwise)", "direction_sign_audit": "passed"})

    if args.benchmark_only and args.tight_validation:
        raise ValueError("Benchmark-only and tight-validation modes cannot be combined.")
    geometries = (CONFIG["geometries"][:1] if args.benchmark_only else
                  ["nominal", "radial_out_10um", "radial_out_100um"] if args.tight_validation else CONFIG["geometries"])
    result_rows = []
    for geometry in geometries:
        check_global_deadline(600)
        print(f"Loading existing {geometry} h50 P4 field", flush=True)
        load_start = time.perf_counter()
        field, meta = load_h50_field(geometry)
        meta["field_load_wall_seconds"] = time.perf_counter() - load_start
        atomic_json(HERE / "field_loads" / f"{geometry}.json", meta)
        models = ["rotating"] if args.benchmark_only else ["standing"] if args.tight_validation else CONFIG["models"]
        for model in models:
            result = run_case(field, meta, geometry, model, tight=args.tight_validation,
                              stop_after_one_segment=args.benchmark_only)
            result_rows.append(result)
        del field
        if args.benchmark_only:
            atomic_json(HERE / "benchmark_estimate.json", {
                "status": "measured_one_100_RF_segment",
                "geometry": geometry, "model": "rotating", "field_load_seconds": meta["field_load_wall_seconds"],
                "integration_seconds_per_100_RF": result["segment_wall_seconds"],
                "observed_RSS_not_measured": "Windows global memory status is recorded separately; field arrays are loaded one geometry at a time",
                "estimated_standard_batch_integration_seconds": result["segment_wall_seconds"] * 100,
                "resources_after_benchmark": system_resources(),
                "budget_remaining_seconds": check_global_deadline(0),
                "decision": "resume this checkpoint, then proceed if predicted batch time plus field load is less than remaining global budget"})
    if not args.benchmark_only:
        summaries = []
        for p in sorted((HERE / "trajectories").glob("*/summary.json")):
            summaries.append(json.loads(p.read_text(encoding="utf-8")))
        atomic_json(HERE / "matched_displacement_trajectory_summary_v5.json", {
            "protocol": CONFIG, "initial_condition_audit": "initial_condition_audit.json",
            "trajectories": summaries, "summary_count": len(summaries),
            "right_censoring": "each retained trajectory is censored at 1000 RF; it is not an escape at the horizon",
            "sampling_warning": "one selected deterministic initial state; results are examples, not population statistics"})
        if args.tight_validation:
            standard = {r["geometry"]: r for r in summaries if r.get("model") == "standing" and not r.get("tight")}
            tight = {r["geometry"]: r for r in summaries if r.get("model") == "standing" and r.get("tight")}
            comparisons = []
            for geometry, refined in sorted(tight.items()):
                base = standard.get(geometry)
                if base is None:
                    raise RuntimeError(f"Missing standard reference for tight case {geometry}")
                comparisons.append({"geometry": geometry,
                    "standard_status": base["status"], "tight_status": refined["status"],
                    "standard_escape_time_RF": base["escape_time_RF"], "tight_escape_time_RF": refined["escape_time_RF"],
                    "escape_time_difference_RF": (refined["escape_time_RF"] - base["escape_time_RF"]
                        if base["escape_time_RF"] is not None and refined["escape_time_RF"] is not None else None),
                    "same_classification": base["status"] == refined["status"],
                    "standard_max_radius_mm": base["max_radius_mm_sampled"],
                    "tight_max_radius_mm": refined["max_radius_mm_sampled"],
                    "max_radius_difference_mm": refined["max_radius_mm_sampled"] - base["max_radius_mm_sampled"]})
            atomic_json(HERE / "numerical_refinement_comparison_v5.json", {
                "status": "passed" if all(x["same_classification"] for x in comparisons) else "classification_change_requires_review",
                "selected_cases": comparisons,
                "protocol": "same h50 P4 FEM field and same physical initial state; DOP853 rtol 2e-11 and maximum step T/100",
                "interpretation": "agreement validates integration-tolerance sensitivity only; it does not remove spatial FEM discretization uncertainty"})
    print(json.dumps({"mode": "benchmark" if args.benchmark_only else "full_batch",
                      "completed_in_this_process": len(result_rows),
                      "statuses": {r["key"]: r.get("status", "incomplete_benchmark") for r in result_rows}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
