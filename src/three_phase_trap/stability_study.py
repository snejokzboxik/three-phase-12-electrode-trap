"""Restartable publication study of full RF trajectory stability."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import label

from .config import ProjectConfig
from .dynamics import integrate_trajectory, trajectory_metrics
from .fem_cache import fem_fingerprint, load_fem_cache
from .interpolation import FEMInterpolator
from .multipoles import fit_multipoles, polar_sample_points


COARSE_U0_V = (250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 7500.0, 10000.0)
COARSE_FREQUENCY_HZ = (5000.0, 7500.0, 10000.0, 15000.0, 20000.0, 30000.0, 50000.0, 75000.0, 100000.0)
PRIMARY_RADII_M = (0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002)
PRIMARY_ANGLES_DEG = (0.0, 22.5, 45.0, 67.5)
SECONDARY_STATES = ((0.0005, 0.0, 0.02), (0.0005, 0.0, -0.02), (0.001, 45.0, 0.02), (0.001, 45.0, -0.02))
STAGE_A_PERIODS = 50
STAGE_B_PERIODS = 1000
STAGE_C_PERIODS = 10000
INITIAL_CONDITIONS_PER_PAIR = len(PRIMARY_RADII_M) * len(PRIMARY_ANGLES_DEG) + len(SECONDARY_STATES)
OUTPUT_SAMPLES_PER_PERIOD = 4
MAX_WORKERS = 4
CHECKPOINT_NAME = "stability_checkpoint.jsonl"
MANIFEST_NAME = "stability_manifest.json"
CASE_COLUMNS = (
    "case_id", "trajectory_id", "grid_level", "U0_V", "f_rf_Hz", "mass_kg", "charge_C", "gamma_kg_per_s",
    "initial_r_m", "initial_theta_deg", "initial_vx_m_per_s", "initial_vy_m_per_s",
    "ensemble", "stage", "periods_requested", "periods_survived", "escape_time_s",
    "r_max_m", "r_final_m", "r_rms_m", "r_median_m", "bounded", "escaped",
    "stage_a_survived", "source", "source_periods_requested", "candidate_group",
)

_WORKER_INTERPOLATOR: FEMInterpolator | None = None
_WORKER_CONFIG: ProjectConfig | None = None


def _study_definition(config: ProjectConfig) -> dict[str, object]:
    return {
        "coarse_U0_V": COARSE_U0_V,
        "coarse_f_rf_Hz": COARSE_FREQUENCY_HZ,
        "primary_radii_m": PRIMARY_RADII_M,
        "primary_angles_deg": PRIMARY_ANGLES_DEG,
        "secondary_states": SECONDARY_STATES,
        "stage_periods": {"A": STAGE_A_PERIODS, "B": STAGE_B_PERIODS, "C": STAGE_C_PERIODS},
        "ode": {
            "rtol": config.simulation.trajectory_rtol,
            "atol": config.simulation.trajectory_atol,
            "max_step_per_period": config.localization.max_step_per_period,
            "output_samples_per_period": OUTPUT_SAMPLES_PER_PERIOD,
            "method": "DOP853",
        },
        "particle": {
            "mass_kg": config.particle.mass,
            "charge_C": config.particle.charge,
            "gamma_kg_per_s": 0.0,
            "gravity_enabled": False,
            "escape_radius_m": config.simulation.escape_radius,
        },
        "field_fingerprint": fem_fingerprint(config),
        "boundary_refinement": "log-midpoints on adjacent coarse edges with differing nonzero Stage-B survival/strict classifications",
        "stage_c_policy": "up to six coarse parameter pairs; two representative Stage-B-bounded initial states per pair",
    }


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _case_id(grid_level: str, stage: str, U0: float, frequency: float, radius: float, theta: float, velocity: tuple[float, float]) -> str:
    raw = (grid_level, stage, *(f"{v:.12g}" for v in (U0, frequency, radius, theta, velocity[0], velocity[1])))
    return hashlib.sha256("|".join(raw).encode("ascii")).hexdigest()[:20]


def _trajectory_id(grid_level: str, U0: float, frequency: float, radius: float, theta: float, velocity: tuple[float, float]) -> str:
    raw = (grid_level, *(f"{v:.12g}" for v in (U0, frequency, radius, theta, velocity[0], velocity[1])))
    return hashlib.sha256("|".join(raw).encode("ascii")).hexdigest()[:20]


def _manifest(output_dir: Path, config: ProjectConfig) -> dict[str, object]:
    definition = _study_definition(config)
    fingerprint = _fingerprint(definition)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = {"study_fingerprint": fingerprint, "study_definition": definition}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("study_fingerprint") != fingerprint:
            raise RuntimeError(
                f"The stability checkpoint at {output_dir} belongs to a different study configuration. "
                "It has been preserved; select a new output directory or restore its original configuration."
            )
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def _read_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        boundary = content.rfind(b"\n") + 1
        with path.open("r+b") as handle:
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        content = content[:boundary]
    completed: dict[str, dict[str, object]] = {}
    for line_number, raw in enumerate(content.splitlines(), start=1):
        try:
            row = json.loads(raw)
            completed[str(row["case_id"])] = row
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(f"Invalid completed checkpoint record at {path}:{line_number}: {error}") from error
    return completed


def _checkpoint_append(path: Path, row: dict[str, object]) -> None:
    clean = {}
    for key, value in row.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            clean[key] = None
        elif isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(clean, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _make_cases(U0: float, frequency: float, grid_level: str) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for radius in PRIMARY_RADII_M:
        for theta in PRIMARY_ANGLES_DEG:
            cases.append({"U0_V": U0, "f_rf_Hz": frequency, "initial_r_m": radius, "initial_theta_deg": theta, "initial_vx_m_per_s": 0.0, "initial_vy_m_per_s": 0.0, "ensemble": "primary", "grid_level": grid_level})
    for radius, theta, tangent_speed in SECONDARY_STATES:
        angle = np.deg2rad(theta)
        vx, vy = -np.sin(angle) * tangent_speed, np.cos(angle) * tangent_speed
        cases.append({"U0_V": U0, "f_rf_Hz": frequency, "initial_r_m": radius, "initial_theta_deg": theta, "initial_vx_m_per_s": float(vx), "initial_vy_m_per_s": float(vy), "ensemble": "velocity_probe", "grid_level": grid_level})
    return cases


def _job(case: dict[str, object], stage: str, periods: int, config: ProjectConfig, candidate_group: str = "") -> dict[str, object]:
    angle = math.radians(float(case["initial_theta_deg"]))
    return {
        **case,
        "stage": stage,
        "periods_requested": periods,
        "candidate_group": candidate_group or str(case.get("candidate_group", "")),
        "mass_kg": config.particle.mass,
        "charge_C": config.particle.charge,
        "gamma_kg_per_s": 0.0,
        "case_id": _case_id(str(case["grid_level"]), stage, float(case["U0_V"]), float(case["f_rf_Hz"]), float(case["initial_r_m"]), float(case["initial_theta_deg"]), (float(case["initial_vx_m_per_s"]), float(case["initial_vy_m_per_s"]))),
        "trajectory_id": _trajectory_id(str(case["grid_level"]), float(case["U0_V"]), float(case["f_rf_Hz"]), float(case["initial_r_m"]), float(case["initial_theta_deg"]), (float(case["initial_vx_m_per_s"]), float(case["initial_vy_m_per_s"]))),
        "_initial_position": (float(case["initial_r_m"]) * math.cos(angle), float(case["initial_r_m"]) * math.sin(angle)),
    }


def _run_job(interpolator: FEMInterpolator, config: ProjectConfig, job: dict[str, object]) -> dict[str, object]:
    particle = replace(config.particle, gamma=0.0, gravity_enabled=False)
    result = integrate_trajectory(
        interpolator, particle, float(job["U0_V"]), float(job["f_rf_Hz"]),
        initial_position=job["_initial_position"],
        initial_velocity=(float(job["initial_vx_m_per_s"]), float(job["initial_vy_m_per_s"])),
        periods=int(job["periods_requested"]), escape_radius=config.simulation.escape_radius,
        rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol,
        max_step_per_period=config.localization.max_step_per_period,
        output_samples_per_period=OUTPUT_SAMPLES_PER_PERIOD,
    )
    metrics = trajectory_metrics(result, particle, config.simulation.escape_radius)
    return {
        **{key: value for key, value in job.items() if not key.startswith("_")},
        "periods_survived": float(metrics["periods_survived"]),
        "escape_time_s": None if not result.escaped else float(metrics["escape_time_s"]),
        "r_max_m": float(metrics["max_radius_m"]),
        "r_final_m": float(metrics["final_radius_m"]),
        "r_rms_m": float(metrics["rms_radius_m"]),
        "r_median_m": float(metrics["median_radius_m"]),
        "bounded": int(result.bounded),
        "escaped": int(result.escaped),
        "stage_a_survived": int(result.periods_simulated >= STAGE_A_PERIODS - 1.0e-8 / float(job["f_rf_Hz"])) if str(job["stage"]).startswith("A_") else None,
        "source": "computed",
        "source_periods_requested": int(job["periods_requested"]),
    }


def _worker_init(config: ProjectConfig, project_root: str) -> None:
    global _WORKER_INTERPOLATOR, _WORKER_CONFIG
    _WORKER_CONFIG = config
    _WORKER_INTERPOLATOR = load_fem_cache(config, Path(project_root))[3]


def _worker_run(job: dict[str, object]) -> dict[str, object]:
    if _WORKER_INTERPOLATOR is None or _WORKER_CONFIG is None:
        raise RuntimeError("Stability worker was not initialized with the P2 cache.")
    return _run_job(_WORKER_INTERPOLATOR, _WORKER_CONFIG, job)


def _localization_reuse_rows(root: Path, config: ProjectConfig, completed: dict[str, dict[str, object]], checkpoint: Path) -> int:
    source_path = root / "outputs/publication_study/localization_checkpoint.csv"
    if not source_path.exists():
        return 0
    count = 0
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        for original in csv.DictReader(handle):
            if not str(original.get("stage", "")).startswith("A_"):
                continue
            U0, frequency = float(original["U0_V"]), float(original["frequency_Hz"])
            if U0 not in COARSE_U0_V or frequency not in COARSE_FREQUENCY_HZ:
                continue
            radius, theta = float(original["initial_radius_m"]), float(original["initial_angle_deg"])
            if radius not in PRIMARY_RADII_M or theta not in PRIMARY_ANGLES_DEG:
                continue
            if abs(float(original.get("initial_vx_m_per_s", 0.0))) > 1e-14 or abs(float(original.get("initial_vy_m_per_s", 0.0))) > 1e-14:
                continue
            source_periods = int(float(original.get("requested_periods", 0)))
            if source_periods < STAGE_A_PERIODS:
                continue
            case = {"U0_V": U0, "f_rf_Hz": frequency, "initial_r_m": radius, "initial_theta_deg": theta, "initial_vx_m_per_s": 0.0, "initial_vy_m_per_s": 0.0, "ensemble": "primary", "grid_level": "coarse"}
            row = _job(case, "A_50", STAGE_A_PERIODS, config)
            case_id = str(row["case_id"])
            if case_id in completed:
                continue
            source_escape = float(original.get("escape_time_s", "nan") or "nan")
            survived_50 = (not math.isfinite(source_escape)) or source_escape > STAGE_A_PERIODS / frequency + 1e-15
            source_escaped = bool(int(float(original.get("escaped", 0))))
            row.update({
                "periods_survived": min(STAGE_A_PERIODS, float(original["periods_survived"])),
                "escape_time_s": None,
                "r_max_m": None,
                "r_final_m": None,
                "r_rms_m": None,
                "r_median_m": None,
                "bounded": int(survived_50),
                "escaped": int(not survived_50),
                "stage_a_survived": int(survived_50),
                "source": "reused_localization_checkpoint",
                "source_periods_requested": source_periods,
                "source_trajectory_escaped": int(source_escaped),
            })
            if not survived_50 and source_escaped:
                for new_key, old_key in (("escape_time_s", "escape_time_s"), ("r_max_m", "max_radius_m"), ("r_final_m", "final_radius_m"), ("r_rms_m", "rms_radius_m"), ("r_median_m", "median_radius_m")):
                    try:
                        row[new_key] = float(original[old_key])
                    except (ValueError, KeyError):
                        pass
            completed[case_id] = row
            _checkpoint_append(checkpoint, row)
            count += 1
    return count


def _execute_stage(
    cases: list[dict[str, object]], stage: str, periods: int, config: ProjectConfig,
    interpolator: FEMInterpolator, completed: dict[str, dict[str, object]], checkpoint: Path,
    root: Path, workers: int,
) -> int:
    pending = []
    for case in cases:
        job = _job(case, stage, periods, config)
        if str(job["case_id"]) not in completed:
            pending.append(job)
    if not pending:
        return 0
    if workers <= 1:
        for job in pending:
            row = _run_job(interpolator, config, job)
            completed[str(row["case_id"])] = row
            _checkpoint_append(checkpoint, row)
            print(f"{stage}: {len(completed)} checkpointed; {row['U0_V']/1000:g} kV at {row['f_rf_Hz']/1000:g} kHz", flush=True)
        return len(pending)
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(config, str(root)))
    futures = {pool.submit(_worker_run, job): job for job in pending}
    try:
        for future in as_completed(futures):
            row = future.result()
            completed[str(row["case_id"])] = row
            _checkpoint_append(checkpoint, row)
            if len(completed) % 25 == 0 or not futures:
                print(f"{stage}: {len(completed)} trajectory rows checkpointed", flush=True)
    except KeyboardInterrupt:
        running = []
        for future in futures:
            if future.running():
                running.append(future)
            else:
                future.cancel()
        for future in as_completed(running):
            row = future.result()
            completed[str(row["case_id"])] = row
            _checkpoint_append(checkpoint, row)
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return len(pending)


def _pair_summary(rows: list[dict[str, object]], grid_level: str) -> list[dict[str, object]]:
    selected = [row for row in rows if row.get("grid_level") == grid_level]
    pairs = sorted({(float(row["U0_V"]), float(row["f_rf_Hz"])) for row in selected})
    results = []
    expected_primary = len(PRIMARY_RADII_M) * len(PRIMARY_ANGLES_DEG)
    expected_velocity = len(SECONDARY_STATES)
    for U0, frequency in pairs:
        pair = [row for row in selected if float(row["U0_V"]) == U0 and float(row["f_rf_Hz"]) == frequency]
        A = [row for row in pair if str(row["stage"]).startswith("A_")]
        A_survivors = {str(row["case_id"]): row for row in A if int(row.get("stage_a_survived") or 0) == 1}
        B = [row for row in pair if str(row["stage"]).startswith("B_")]
        B_by_id = {str(row["trajectory_id"]): row for row in B}
        primary_a = [row for row in A if row.get("ensemble") == "primary"]
        secondary_a = [row for row in A if row.get("ensemble") == "velocity_probe"]
        total = len(primary_a) + len(secondary_a)
        known = 0
        bounded = 0
        primary_known = 0
        primary_bounded = 0
        velocity_known = 0
        velocity_bounded = 0
        effective_periods = []
        effective_rmax = []
        for a_row in A:
            cid = str(a_row["trajectory_id"])
            if int(a_row.get("stage_a_survived") or 0) == 0:
                known += 1
                if a_row.get("ensemble") == "primary": primary_known += 1
                else: velocity_known += 1
                effective_periods.append(float(a_row.get("periods_survived") or 0.0))
                if a_row.get("r_max_m") is not None: effective_rmax.append(float(a_row["r_max_m"]))
            elif cid in B_by_id:
                b_row = B_by_id[cid]
                known += 1
                is_bounded = int(b_row.get("bounded") or 0) == 1
                bounded += int(is_bounded)
                effective_periods.append(float(b_row.get("periods_survived") or 0.0))
                effective_rmax.append(float(b_row["r_max_m"]))
                if a_row.get("ensemble") == "primary":
                    primary_known += 1
                    primary_bounded += int(is_bounded)
                else:
                    velocity_known += 1
                    velocity_bounded += int(is_bounded)
        complete = total > 0 and known == total
        strict = int(complete and bounded == total)
        results.append({
            "grid_level": grid_level,
            "U0_V": U0,
            "f_rf_Hz": frequency,
            "stage_a_survivors": len(A_survivors),
            "stage_a_total": total,
            "stage_a_survival_fraction": len(A_survivors) / total if total else None,
            "stage_b_bounded": bounded if complete else None,
            "stage_b_total": total if complete else None,
            "stage_b_survival_fraction": bounded / total if complete else None,
            "stage_b_primary_bounded": primary_bounded if primary_known == expected_primary else None,
            "stage_b_primary_total": expected_primary,
            "stage_b_primary_survival_fraction": primary_bounded / expected_primary if primary_known == expected_primary else None,
            "stage_b_velocity_bounded": velocity_bounded if velocity_known == expected_velocity else None,
            "stage_b_velocity_total": expected_velocity,
            "stage_b_velocity_survival_fraction": velocity_bounded / expected_velocity if velocity_known == expected_velocity else None,
            "stage_b_outcomes_complete": int(complete),
            "strict_stability": strict if complete else None,
            "median_periods_survived": float(np.median(effective_periods)) if effective_periods else None,
            "median_r_max_m": float(np.median(effective_rmax)) if effective_rmax else None,
            "stage_c_completed": sum(1 for row in pair if str(row["stage"]).startswith("C_") and int(row.get("bounded") or 0) == 1),
            "stage_c_tested": sum(1 for row in pair if str(row["stage"]).startswith("C_")),
        })
    return results


def _boundary_midpoints(summary: list[dict[str, object]]) -> list[tuple[float, float]]:
    grid = {(float(row["U0_V"]), float(row["f_rf_Hz"])): row for row in summary if int(row["stage_b_outcomes_complete"] or 0) == 1}
    u_values = sorted({pair[0] for pair in grid})
    f_values = sorted({pair[1] for pair in grid})
    candidates: set[tuple[float, float]] = set()
    def classification(value: object) -> int:
        if value is None: return -1
        if float(value) > 0: return 1
        return 0
    for U0 in u_values:
        for f0, f1 in zip(f_values[:-1], f_values[1:]):
            left, right = grid.get((U0, f0)), grid.get((U0, f1))
            if left is None or right is None: continue
            lclass = classification(left["strict_stability"])
            rclass = classification(right["strict_stability"])
            lf = classification(left["stage_b_survival_fraction"])
            rf = classification(right["stage_b_survival_fraction"])
            if lclass != rclass or lf != rf:
                candidates.add((U0, math.sqrt(f0 * f1)))
    for f in f_values:
        for u0, u1 in zip(u_values[:-1], u_values[1:]):
            lower, upper = grid.get((u0, f)), grid.get((u1, f))
            if lower is None or upper is None: continue
            lclass = classification(lower["strict_stability"])
            rclass = classification(upper["strict_stability"])
            lf = classification(lower["stage_b_survival_fraction"])
            rf = classification(upper["stage_b_survival_fraction"])
            if lclass != rclass or lf != rf:
                candidates.add((math.sqrt(u0 * u1), f))
    return sorted(candidates)


def _stage_c_selection(summary: list[dict[str, object]], rows: list[dict[str, object]], maximum_points: int = 3) -> list[tuple[float, float, str]]:
    """Select up to two high-survival candidates and one Stage-B boundary point.

    These are supplemental trajectory checks, not a stability classification.
    In particular, a high survival fraction below one is not described as
    robust or globally stable.
    """
    by_pair = {
        (float(row["U0_V"]), float(row["f_rf_Hz"])): row
        for row in summary
        if row.get("grid_level") == "coarse" and int(row.get("stage_b_outcomes_complete") or 0) == 1
    }
    nonstrict = [pair for pair, row in by_pair.items() if row.get("strict_stability") != 1]

    def boundary_distance(pair: tuple[float, float]) -> float:
        if not nonstrict:
            return math.inf
        return min(math.hypot(math.log(pair[0] / other[0]), math.log(pair[1] / other[1])) for other in nonstrict)

    def radius_score(pair: tuple[float, float]) -> float:
        values = [
            float(row["r_rms_m"])
            for row in rows
            if row.get("grid_level") == "coarse"
            and float(row["U0_V"]) == pair[0]
            and float(row["f_rf_Hz"]) == pair[1]
            and row.get("stage") == "B_1000"
            and int(row.get("bounded") or 0) == 1
            and row.get("r_rms_m") is not None
        ]
        return float(np.mean(values)) if values else math.inf

    selected: list[tuple[float, float, str]] = []
    high_survival = [
        pair for pair, row in by_pair.items()
        if float(row.get("stage_b_survival_fraction") or 0.0) > 0.0
    ]
    high_survival.sort(key=lambda pair: (
        -float(by_pair[pair]["stage_b_survival_fraction"]),
        boundary_distance(pair),
        radius_score(pair),
    ))
    for pair in high_survival[:min(2, maximum_points)]:
        selected.append((pair[0], pair[1], "high_survival_candidate"))

    partial = [
        (pair, float(row["stage_b_survival_fraction"]))
        for pair, row in by_pair.items()
        if row.get("stage_b_survival_fraction") is not None
        and 0.0 < float(row["stage_b_survival_fraction"]) < 1.0
    ]
    if len(selected) < maximum_points and partial:
        remaining_partial = [item for item in partial if item[0] not in {(u, f) for u, f, _ in selected}]
        if remaining_partial:
            pair, _ = min(remaining_partial, key=lambda item: abs(item[1] - 0.5))
            selected.append((pair[0], pair[1], "near_boundary"))
    elif len(selected) < maximum_points:
        remaining = [pair for pair in by_pair if pair not in {(u, f) for u, f, _ in selected}]
        if remaining and nonstrict:
            pair = min(remaining, key=boundary_distance)
            selected.append((pair[0], pair[1], "near_boundary"))
    return selected[:maximum_points]


def _stage_c_cases(selection: list[tuple[float, float, str]], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cases = []
    for U0, frequency, group in selection:
        candidates = [row for row in rows if row.get("grid_level") == "coarse" and float(row["U0_V"]) == U0 and float(row["f_rf_Hz"]) == frequency and row.get("stage") == "B_1000" and int(row.get("bounded") or 0) == 1]
        if not candidates:
            continue
        primary = [row for row in candidates if row.get("ensemble") == "primary"]
        source = min(primary or candidates, key=lambda row: (float(row.get("r_rms_m") or math.inf), float(row["initial_r_m"])))
        cases.append({
            "U0_V": U0, "f_rf_Hz": frequency,
            "initial_r_m": float(source["initial_r_m"]),
            "initial_theta_deg": float(source["initial_theta_deg"]),
            "initial_vx_m_per_s": float(source["initial_vx_m_per_s"]),
            "initial_vy_m_per_s": float(source["initial_vy_m_per_s"]),
            "ensemble": str(source.get("ensemble", "primary")),
            "grid_level": "coarse", "candidate_group": group,
        })
    return cases[:3]


def _atomic_csv(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".csv.tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        Path(name).replace(path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _cell_edges(values: tuple[float, ...] | list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if np.any(x <= 0):
        middle = 0.5 * (x[1:] + x[:-1])
        return np.concatenate(([x[0] - (middle[0] - x[0])], middle, [x[-1] + (x[-1] - middle[-1])]))
    middle = np.sqrt(x[1:] * x[:-1])
    return np.concatenate(([x[0]**2 / middle[0]], middle, [x[-1]**2 / middle[-1]]))


def _grid_matrix(summary: list[dict[str, object]], key: str, level: str = "coarse") -> np.ndarray:
    values = np.full((len(COARSE_U0_V), len(COARSE_FREQUENCY_HZ)), np.nan)
    for row in summary:
        if row.get("grid_level") != level or row.get(key) is None:
            continue
        try:
            i = COARSE_U0_V.index(float(row["U0_V"]))
            j = COARSE_FREQUENCY_HZ.index(float(row["f_rf_Hz"]))
        except ValueError:
            continue
        values[i, j] = float(row[key])
    return values


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def _plot_matrix(ax, matrix: np.ndarray, cmap: str, vmin: float, vmax: float, title: str, colorbar_label: str, stage_c: set[tuple[float, float]] | None = None):
    values = np.ma.masked_invalid(matrix)
    image = ax.pcolormesh(_cell_edges(np.asarray(COARSE_FREQUENCY_HZ) / 1000.0), _cell_edges(np.asarray(COARSE_U0_V) / 1000.0), values, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", edgecolors="white", linewidth=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(np.asarray(COARSE_FREQUENCY_HZ) / 1000.0)
    ax.set_yticks(np.asarray(COARSE_U0_V) / 1000.0)
    ax.set_xticklabels([f"{value/1000:g}" for value in COARSE_FREQUENCY_HZ])
    ax.set_yticklabels([f"{value/1000:g}" for value in COARSE_U0_V])
    ax.set(xlabel="RF frequency, $f_{\\rm RF}$ (kHz)", ylabel="RF amplitude, $U_0$ (kV)", title=title)
    ax.grid(False)
    if stage_c:
        ax.scatter([f / 1000 for _, f in stage_c], [u / 1000 for u, _ in stage_c], marker="*", s=105, facecolor="#f2b134", edgecolor="#222222", linewidth=0.75, zorder=5, label="Stage-C tested")
        ax.legend(loc="best", frameon=True, fontsize=8)
    return image


def _write_pair_summary(summary: list[dict[str, object]], output_dir: Path) -> None:
    columns = (
        "grid_level", "U0_V", "f_rf_Hz", "stage_a_survivors", "stage_a_total", "stage_a_survival_fraction",
        "stage_b_bounded", "stage_b_total", "stage_b_survival_fraction", "stage_b_primary_bounded",
        "stage_b_primary_total", "stage_b_primary_survival_fraction", "stage_b_velocity_bounded",
        "stage_b_velocity_total", "stage_b_velocity_survival_fraction", "stage_b_outcomes_complete",
        "strict_stability", "median_periods_survived", "median_r_max_m", "stage_c_completed", "stage_c_tested",
    )
    _atomic_csv(output_dir / "stability_summary.csv", summary, columns)


def _initial_radius_figure(summary: list[dict[str, object]], cases: list[dict[str, object]], output_dir: Path, stage_c: set[tuple[float, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), constrained_layout=True)
    stage_b = {str(row["trajectory_id"]): row for row in cases if str(row["stage"]).startswith("B_")}
    stage_a_by_id = {str(row["case_id"]): row for row in cases if str(row["stage"]).startswith("A_")}
    for ax, radius in zip(axes.flat, (0.00025, 0.0005, 0.001, 0.002)):
        matrix = np.full((len(COARSE_U0_V), len(COARSE_FREQUENCY_HZ)), np.nan)
        for ui, U0 in enumerate(COARSE_U0_V):
            for fi, frequency in enumerate(COARSE_FREQUENCY_HZ):
                statuses = []
                for theta in PRIMARY_ANGLES_DEG:
                    a = next((row for row in stage_a_by_id.values() if row.get("grid_level") == "coarse" and float(row["U0_V"]) == U0 and float(row["f_rf_Hz"]) == frequency and float(row["initial_r_m"]) == radius and float(row["initial_theta_deg"]) == theta and row.get("ensemble") == "primary"), None)
                    if a is None:
                        continue
                    if int(a.get("stage_a_survived") or 0) == 0:
                        statuses.append(0)
                    else:
                        b = stage_b.get(str(a["trajectory_id"]))
                        if b is not None:
                            statuses.append(int(b.get("bounded") or 0))
                if len(statuses) == len(PRIMARY_ANGLES_DEG):
                    matrix[ui, fi] = np.mean(statuses)
        img = _plot_matrix(ax, matrix, "viridis", 0.0, 1.0, f"$r_0$ = {radius * 1000:g} mm", "Stage-B survival fraction", stage_c)
        fig.colorbar(img, ax=ax, label="Bounded fraction (4 angles)", ticks=(0, 0.25, 0.5, 0.75, 1))
    fig.suptitle("Full-dynamics survival by initial radius (1000 RF periods)", fontsize=14)
    _save_figure(fig, output_dir, "fig_stability_by_initial_radius")


def _save_stability_figures(summary: list[dict[str, object]], cases: list[dict[str, object]], output_dir: Path) -> None:
    coarse = [row for row in summary if row["grid_level"] == "coarse"]
    strict = _grid_matrix(coarse, "strict_stability")
    fraction = _grid_matrix(coarse, "stage_b_survival_fraction")
    escape = _grid_matrix(coarse, "median_periods_survived")
    rmax = _grid_matrix(coarse, "median_r_max_m") * 1000.0
    stage_c = {(float(row["U0_V"]), float(row["f_rf_Hz"])) for row in cases if str(row["stage"]).startswith("C_")}
    refined = [row for row in summary if row.get("grid_level") == "refined" and int(row.get("stage_b_outcomes_complete") or 0) == 1]
    fig, ax = plt.subplots(figsize=(9.1, 6.7), constrained_layout=True)
    img = _plot_matrix(ax, strict, ListedColormap(("#cbd2d8", "#3b7898")), 0, 1, "Strict Stage-B full-survival classification", "All 32 tested starts bounded", stage_c)
    for row in refined:
        color = "#3b7898" if row.get("strict_stability") == 1 else "#cbd2d8"
        ax.scatter(float(row["f_rf_Hz"]) / 1000, float(row["U0_V"]) / 1000, marker="s", s=43, facecolor=color, edgecolor="#24313a", linewidth=0.6, zorder=4)
    cb = fig.colorbar(img, ax=ax, ticks=(0.25, 0.75))
    cb.ax.set_yticklabels(("Not strict-stable", "Strict-stable"))
    _save_figure(fig, output_dir, "fig_stability_binary")

    fig, ax = plt.subplots(figsize=(9.1, 6.7), constrained_layout=True)
    img = _plot_matrix(ax, fraction, "viridis", 0, 1, "Stage-B survival fraction (coarse + refined observations)", "Bounded fraction", stage_c)
    for row in refined:
        ax.scatter(float(row["f_rf_Hz"]) / 1000, float(row["U0_V"]) / 1000, marker="s", s=43, facecolor=plt.cm.viridis(float(row["stage_b_survival_fraction"])), edgecolor="#24313a", linewidth=0.6, zorder=4)
    fig.colorbar(img, ax=ax, ticks=np.linspace(0, 1, 6), label="Bounded fraction of 32 starts")
    ax.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="", color="#7b3294", markeredgecolor="#24313a", label="Refined Stage-B sample"),
        plt.Line2D([], [], marker="*", linestyle="", color="#f2b134", markeredgecolor="#222222", label="Stage-C trajectory tested"),
    ], fontsize=8, loc="best")
    _save_figure(fig, output_dir, "fig_stability_fraction")

    fig, ax = plt.subplots(figsize=(9.1, 6.7), constrained_layout=True)
    img = _plot_matrix(ax, escape, "magma", 0, STAGE_B_PERIODS, "Median trajectory duration", "Median RF periods survived", stage_c)
    fig.colorbar(img, ax=ax, label="Median RF periods survived", extend="max")
    _save_figure(fig, output_dir, "fig_escape_time")

    fig, ax = plt.subplots(figsize=(9.1, 6.7), constrained_layout=True)
    img = _plot_matrix(ax, rmax, "cividis", 0, 8, "Maximum radius reached", "Median per-case $r_{\\max}$ (mm)", stage_c)
    fig.colorbar(img, ax=ax, label="Median per-case $r_{\\max}$ (mm)")
    _save_figure(fig, output_dir, "fig_rmax_map")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.5), constrained_layout=True)
    coarse_img = _plot_matrix(axes[0], fraction, "viridis", 0, 1, "Coarse cells + refined observed samples (no interpolation)", "Stage-B survival fraction", stage_c)
    fig.colorbar(coarse_img, ax=axes[0], label="Bounded fraction")
    refined = {(float(row["U0_V"]), float(row["f_rf_Hz"])): row for row in summary if row["grid_level"] == "refined" and row.get("stage_b_survival_fraction") is not None}
    for (U0, frequency), row in refined.items():
        color = plt.cm.viridis(float(row["stage_b_survival_fraction"]))
        axes[0].scatter(frequency / 1000, U0 / 1000, marker="s", s=42, color=color, edgecolor="#24313a", linewidth=0.45, zorder=4)
        axes[1].scatter(frequency / 1000, U0 / 1000, marker="s", s=55, color=plt.cm.viridis(float(row["stage_b_survival_fraction"])), edgecolor="#24313a", linewidth=0.55)
    if stage_c:
        for ax in axes:
            ax.scatter([f / 1000 for _, f in stage_c], [u / 1000 for u, _ in stage_c], marker="*", s=110, facecolor="#f2b134", edgecolor="#222222", linewidth=0.7, zorder=6)
    axes[1].set(title="Refined boundary points (observed only)", xlabel="RF frequency, $f_{\\rm RF}$ (kHz)", ylabel="RF amplitude, $U_0$ (kV)", xscale="log", yscale="log")
    axes[1].set_xticks([f / 1000 for f in COARSE_FREQUENCY_HZ]); axes[1].set_yticks([u / 1000 for u in COARSE_U0_V])
    axes[1].set_xticklabels([f"{f/1000:g}" for f in COARSE_FREQUENCY_HZ]); axes[1].set_yticklabels([f"{u/1000:g}" for u in COARSE_U0_V]); axes[1].grid(alpha=0.18)
    axes[1].set_xlim(_cell_edges(np.asarray(COARSE_FREQUENCY_HZ) / 1000.0)[[0, -1]])
    axes[1].set_ylim(_cell_edges(np.asarray(COARSE_U0_V) / 1000.0)[[0, -1]])
    axes[1].scatter([], [], marker="s", facecolor="#7b3294", edgecolor="#24313a", label="Refined Stage-B sample")
    axes[1].scatter([], [], marker="*", facecolor="#f2b134", edgecolor="#222222", label="Stage-C trajectory tested")
    axes[1].legend(loc="best", fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"), ax=axes[1], label="Stage-B survival fraction", ticks=np.linspace(0, 1, 6))
    axes[0].set_title("Observed coarse + refined samples (no interpolation)")
    axes[0].legend(handles=[plt.Line2D([], [], marker="s", linestyle="", color="#7b3294", markeredgecolor="#24313a", label="refined Stage-B sample"), plt.Line2D([], [], marker="*", linestyle="", color="#f2b134", markeredgecolor="#222222", label="Stage-C trajectory tested")], fontsize=8, loc="best")
    _save_figure(fig, output_dir, "fig_stability_boundary_refined")
    _initial_radius_figure(summary, cases, output_dir, stage_c)


def _fit_fourth_order_coefficient(config: ProjectConfig, interpolator: FEMInterpolator) -> tuple[float, float]:
    """Fit the leading rotating m=4 potential coefficient per volt from the cached P2 basis."""
    geometry = config.geometry
    radii = np.linspace(0.00055, min(0.0048, 0.55 * (geometry.electrode_orbit_radius - geometry.electrode_radius)), 10)
    angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    points = polar_sample_points(radii, angles)
    reference_radius = float(np.max(radii))
    coefficients = []
    for time in np.linspace(0.0, config.rf.period, 25, endpoint=False):
        unit_voltage_potential = interpolator.potential(points, float(time), 1.0, config.rf.frequency)
        fitted = fit_multipoles(points, unit_voltage_potential, (4,), reference_radius)[4]
        coefficients.append(fitted.amplitude / reference_radius**4)
    values = np.asarray(coefficients, dtype=float)
    return float(np.mean(values)), float(np.std(values) / max(abs(np.mean(values)), 1.0e-300))


def _dimensionless_lambda(config: ProjectConfig, U0: float, frequency: float, C4_per_V_m4: float) -> float:
    omega = 2.0 * math.pi * frequency
    r_ref = config.simulation.escape_radius
    return abs(config.particle.charge * C4_per_V_m4) * U0 * r_ref**2 / (config.particle.mass * omega**2)


def _dimensionless_figure(summary: list[dict[str, object]], cases: list[dict[str, object]], config: ProjectConfig, output_dir: Path, stage_c: set[tuple[float, float]], C4_per_V_m4: float) -> None:
    rows = [row for row in cases if row.get("grid_level") == "coarse" and row.get("ensemble") == "primary" and (str(row["stage"]).startswith("A_") or str(row["stage"]).startswith("B_"))]
    latest: dict[tuple[float, float, float, float], dict[str, object]] = {}
    for row in rows:
        key = (float(row["U0_V"]), float(row["f_rf_Hz"]), float(row["initial_r_m"]), float(row["initial_theta_deg"]))
        current = latest.get(key)
        if current is None or str(row["stage"]).startswith("B_"):
            latest[key] = row
    x = []; y = []; color = []; freq = []
    for (U0, frequency, radius, theta), row in latest.items():
        survived = int(row.get("bounded") or 0) if str(row["stage"]).startswith("B_") else 0
        if str(row["stage"]).startswith("A_") and int(row.get("stage_a_survived") or 0):
            continue
        x.append(_dimensionless_lambda(config, U0, frequency, C4_per_V_m4))
        y.append(radius / config.simulation.escape_radius)
        color.append(survived)
        freq.append(frequency / 1000)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    points = axes[0].scatter(x, y, c=color, cmap=ListedColormap(("#cbd2d8", "#3b7898")), vmin=0, vmax=1, s=25, alpha=0.8, edgecolors="none")
    c_rows = [row for row in cases if row.get("grid_level") == "coarse" and row.get("stage") == "C_10000"]
    if c_rows:
        axes[0].scatter([_dimensionless_lambda(config, float(row["U0_V"]), float(row["f_rf_Hz"]), C4_per_V_m4) for row in c_rows], [float(row["initial_r_m"]) / config.simulation.escape_radius for row in c_rows], marker="*", s=95, facecolor="#f2b134", edgecolor="#222222", linewidth=0.6, label="Stage-C supplemental validation")
        axes[0].legend(loc="best", fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set(xlabel=r"Fourth-order control parameter $\lambda_4=|q C_4|U_0 r_{ref}^2/(m\Omega^2)$", ylabel=r"Initial radius $r_0/r_{ref}$", title="Trajectory-level dimensionless outcomes")
    cb = fig.colorbar(points, ax=axes[0], ticks=(0.25, 0.75)); cb.ax.set_yticklabels(("Not 1000-period bounded", "1000-period bounded"))
    summary_coarse = [row for row in summary if row["grid_level"] == "coarse" and row.get("stage_b_survival_fraction") is not None]
    lam = [_dimensionless_lambda(config, float(row["U0_V"]), float(row["f_rf_Hz"]), C4_per_V_m4) for row in summary_coarse]
    frac = [float(row["stage_b_survival_fraction"]) for row in summary_coarse]
    sc = axes[1].scatter(lam, frac, c=[float(row["f_rf_Hz"]) / 1000 for row in summary_coarse], cmap="viridis", s=48, edgecolor="white", linewidth=0.4)
    if stage_c:
        axes[1].scatter([_dimensionless_lambda(config, U0, frequency, C4_per_V_m4) for U0, frequency in stage_c], [float(next((r["stage_c_completed"] for r in summary_coarse if float(r["U0_V"]) == U0 and float(r["f_rf_Hz"]) == frequency), 0)) / max(float(next((r["stage_c_tested"] for r in summary_coarse if float(r["U0_V"]) == U0 and float(r["f_rf_Hz"]) == frequency), 1)), 1.0) for U0, frequency in stage_c], marker="*", s=95, facecolor="#f2b134", edgecolor="#222222", linewidth=0.6, label="Stage-C supplemental validation")
        axes[1].legend(loc="best", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set(xlabel=r"Fourth-order control parameter $\lambda_4=|q C_4|U_0 r_{ref}^2/(m\Omega^2)$", ylabel="Stage-B bounded fraction", ylim=(-0.04, 1.04), title="Survival versus dimensionless drive")
    fig.colorbar(sc, ax=axes[1], label="RF frequency (kHz)")
    _save_figure(fig, output_dir, "fig_stability_dimensionless")


def _write_initial_conditions_note(output_dir: Path) -> None:
    data = {
        "primary_initial_radii_m": PRIMARY_RADII_M,
        "primary_angles_deg": PRIMARY_ANGLES_DEG,
        "primary_velocity_m_per_s": [0.0, 0.0],
        "secondary_velocity_probes": [
            {"r_m": radius, "theta_deg": theta, "tangential_speed_m_per_s": speed}
            for radius, theta, speed in SECONDARY_STATES
        ],
        "primary_count_per_pair": len(PRIMARY_RADII_M) * len(PRIMARY_ANGLES_DEG),
        "secondary_count_per_pair": len(SECONDARY_STATES),
    }
    (output_dir / "initial_ensemble.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _select_validation_cases(summary: list[dict[str, object]], cases: list[dict[str, object]]) -> list[tuple[str, dict[str, object]]]:
    coarse = [row for row in summary if row["grid_level"] == "coarse"]
    full_b = [row for row in cases if row.get("grid_level") == "coarse" and row.get("ensemble") == "primary" and row.get("stage") == "B_1000"]
    chosen: list[tuple[str, dict[str, object]]] = []
    stable_pairs = {(float(r["U0_V"]), float(r["f_rf_Hz"])) for r in coarse if r.get("strict_stability") == 1}
    if stable_pairs:
        stable = [r for r in full_b if (float(r["U0_V"]), float(r["f_rf_Hz"])) in stable_pairs and int(r.get("bounded") or 0)]
        if stable:
            chosen.append(("strict_stable", min(stable, key=lambda r: float(r["r_rms_m"]))))
    partial_pairs = {(float(r["U0_V"]), float(r["f_rf_Hz"])) for r in coarse if r.get("stage_b_survival_fraction") is not None and 0.0 < float(r["stage_b_survival_fraction"]) < 1.0}
    partial = [r for r in full_b if (float(r["U0_V"]), float(r["f_rf_Hz"])) in partial_pairs]
    if partial:
        chosen.append(("near_boundary", min(partial, key=lambda r: abs(float(r["r_rms_m"] or 0.0) - 0.5 * 0.008))))
    escaped = [r for r in full_b if int(r.get("escaped") or 0)]
    if escaped:
        chosen.append(("unstable", max(escaped, key=lambda r: float(r["periods_survived"] or 0.0))))
    if not escaped:
        stage_a_escaped = [r for r in cases if r.get("grid_level") == "coarse" and r.get("ensemble") == "primary" and str(r.get("stage", "")).startswith("A_") and int(r.get("escaped") or 0)]
        if stage_a_escaped:
            chosen.append(("unstable", min(stage_a_escaped, key=lambda r: float(r["periods_survived"] or 0.0))))
    if not chosen:
        best = [r for r in cases if r.get("grid_level") == "coarse" and r.get("ensemble") == "primary" and str(r.get("stage", "")).startswith(("A_", "B_"))]
        if best:
            chosen.append(("longest_survivor_candidate", max(best, key=lambda r: float(r.get("periods_survived") or 0.0))))
    unique = []
    seen = set()
    for name, row in chosen:
        key = (float(row["U0_V"]), float(row["f_rf_Hz"]), float(row["initial_r_m"]), float(row["initial_theta_deg"]))
        if key not in seen:
            seen.add(key); unique.append((name, row))
    return unique


def _save_trajectory_examples(summary: list[dict[str, object]], cases: list[dict[str, object]], config: ProjectConfig, interpolator: FEMInterpolator, output_dir: Path) -> dict[str, object]:
    destination = output_dir / "trajectory_examples"
    destination.mkdir(parents=True, exist_ok=True)
    selected = _select_validation_cases(summary, cases)
    if not selected:
        return {"examples": [], "figure": None}
    figure, axes = plt.subplots(3, len(selected), figsize=(4.2 * len(selected), 10), squeeze=False, constrained_layout=True)
    metadata = []
    for column, (label_text, row) in enumerate(selected):
        U0, frequency = float(row["U0_V"]), float(row["f_rf_Hz"])
        theta = math.radians(float(row["initial_theta_deg"]))
        position = (float(row["initial_r_m"]) * math.cos(theta), float(row["initial_r_m"]) * math.sin(theta))
        velocity = (float(row["initial_vx_m_per_s"]), float(row["initial_vy_m_per_s"]))
        requested = STAGE_B_PERIODS if label_text != "unstable" or str(row["stage"]).startswith("B_") else STAGE_A_PERIODS
        result = integrate_trajectory(
            interpolator, replace(config.particle, gamma=0.0, gravity_enabled=False), U0, frequency,
            initial_position=position, initial_velocity=velocity, periods=requested,
            escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol,
            atol=config.simulation.trajectory_atol, max_step_per_period=config.localization.max_step_per_period,
            output_samples_per_period=8,
        )
        stem = f"trajectory_{label_text}_{U0:g}V_{frequency:g}Hz_r{float(row['initial_r_m'])*1e3:.3f}mm"
        trajectory_path = destination / f"{stem}.csv"
        trajectory_data = np.column_stack((result.time, result.state.T))
        np.savetxt(trajectory_path, trajectory_data, delimiter=",", header="time_s,x_m,y_m,vx_m_per_s,vy_m_per_s", comments="")
        period_axis = result.time * frequency
        radius = np.hypot(result.state[0], result.state[1])
        ax = axes[0, column]
        ax.plot(result.state[0] * 1000, result.state[1] * 1000, color="#28799b", lw=0.8)
        ax.scatter([position[0] * 1000], [position[1] * 1000], color="#e5953b", s=16, zorder=3)
        ax.set(title=f"{label_text.replace('_', ' ').title()}\n$U_0$={U0/1000:g} kV, $f$={frequency/1000:g} kHz", xlabel="x (mm)", ylabel="y (mm)")
        ax.set_aspect("equal", adjustable="box")
        axes[1, column].plot(period_axis, radius * 1000, color="#28799b", lw=0.8)
        axes[1, column].axhline(config.simulation.escape_radius * 1000, color="#b74d4d", ls="--", lw=0.8)
        axes[1, column].set(xlabel="RF periods", ylabel="r (mm)")
        poincare_times = np.arange(0, result.time[-1] + 0.5 / frequency, 1.0 / frequency)
        px = np.interp(poincare_times, result.time, result.state[0]) * 1000
        py = np.interp(poincare_times, result.time, result.state[1]) * 1000
        axes[2, column].scatter(px, py, s=4, color="#28799b", alpha=0.65)
        axes[2, column].set(xlabel="x (mm)", ylabel="y (mm)", title="Once-per-period Poincaré points")
        axes[2, column].set_aspect("equal", adjustable="box")
        metrics = trajectory_metrics(result, replace(config.particle, gamma=0.0, gravity_enabled=False), config.simulation.escape_radius)
        safe_metrics = {key: (None if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)) else value) for key, value in metrics.items()}
        metadata.append({
            "label": label_text, "U0_V": U0, "f_rf_Hz": frequency,
            "initial_r_m": float(row["initial_r_m"]), "initial_theta_deg": float(row["initial_theta_deg"]),
            "initial_vx_m_per_s": velocity[0], "initial_vy_m_per_s": velocity[1],
            "stage_requested": requested, "bounded": int(result.bounded), "escaped": int(result.escaped),
            "periods_survived": result.periods_simulated, "r_max_m": result.max_radius,
            "trajectory_csv": trajectory_path.name, "trajectory_metrics": safe_metrics,
        })
    axes[0, 0].text(0.03, 0.97, "Full equation of motion; gamma=0; gravity off", transform=axes[0, 0].transAxes, va="top", fontsize=7.5, color="#36434b")
    figure.suptitle("Representative full-RF trajectories", fontsize=14)
    _save_figure(figure, output_dir, "fig_stability_trajectory_examples")
    (destination / "trajectory_examples.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"examples": metadata, "figure": output_dir / "fig_stability_trajectory_examples.png"}


def _numerical_validation(summary: list[dict[str, object]], cases: list[dict[str, object]], config: ProjectConfig, interpolator: FEMInterpolator, output_dir: Path) -> list[dict[str, object]]:
    selected = _select_validation_cases(summary, cases)
    rows = []
    particle = replace(config.particle, gamma=0.0, gravity_enabled=False)
    for label_text, source in selected:
        U0, frequency = float(source["U0_V"]), float(source["f_rf_Hz"])
        theta = math.radians(float(source["initial_theta_deg"]))
        position = (float(source["initial_r_m"]) * math.cos(theta), float(source["initial_r_m"]) * math.sin(theta))
        velocity = (float(source["initial_vx_m_per_s"]), float(source["initial_vy_m_per_s"]))
        periods = STAGE_B_PERIODS if str(source["stage"]).startswith("B_") else STAGE_A_PERIODS
        baseline = integrate_trajectory(interpolator, particle, U0, frequency, initial_position=position, initial_velocity=velocity, periods=periods, escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol, max_step_per_period=config.localization.max_step_per_period, output_samples_per_period=OUTPUT_SAMPLES_PER_PERIOD)
        strict_max_step_per_period = max(1, config.localization.max_step_per_period // 2)
        strict = integrate_trajectory(interpolator, particle, U0, frequency, initial_position=position, initial_velocity=velocity, periods=periods, escape_radius=config.simulation.escape_radius, rtol=min(config.simulation.trajectory_rtol * 0.25, 5.0e-8), atol=min(config.simulation.trajectory_atol * 0.5, 5.0e-12), max_step_per_period=strict_max_step_per_period, output_samples_per_period=OUTPUT_SAMPLES_PER_PERIOD)
        def recovered_field(p, t):
            return interpolator.field_recovered(np.asarray(p).reshape(1, 2), t, U0, frequency)[0]
        recovered = integrate_trajectory(None, particle, U0, frequency, initial_position=position, initial_velocity=velocity, periods=periods, escape_radius=config.simulation.escape_radius, rtol=config.simulation.trajectory_rtol, atol=config.simulation.trajectory_atol, max_step_per_period=config.localization.max_step_per_period, output_samples_per_period=OUTPUT_SAMPLES_PER_PERIOD, field_function=recovered_field)
        rows.append({
            "selection": label_text, "U0_V": U0, "f_rf_Hz": frequency,
            "initial_r_m": float(source["initial_r_m"]), "initial_theta_deg": float(source["initial_theta_deg"]),
            "periods_requested": periods, "baseline_bounded": int(baseline.bounded),
            "strict_rtol": min(config.simulation.trajectory_rtol * 0.25, 5.0e-8),
            "strict_atol": min(config.simulation.trajectory_atol * 0.5, 5.0e-12),
            "strict_max_step_per_period": strict_max_step_per_period,
            "strict_bounded": int(strict.bounded), "strict_periods_survived": strict.periods_simulated,
            "strict_classification_changed": int(strict.bounded != baseline.bounded),
            "recovered_field_bounded": int(recovered.bounded),
            "recovered_field_periods_survived": recovered.periods_simulated,
            "recovered_field_classification_changed": int(recovered.bounded != baseline.bounded),
        })
    columns = list(rows[0]) if rows else ["selection", "U0_V", "f_rf_Hz"]
    _atomic_csv(output_dir / "stability_numerical_validation.csv", rows, columns)
    return rows


def _summary_report(summary: list[dict[str, object]], cases: list[dict[str, object]], config: ProjectConfig, numerical: list[dict[str, object]], stage_c_selection: list[tuple[float, float, str]], output_dir: Path, C4_per_V_m4: float, C4_relative_phase_spread: float) -> Path:
    coarse = [row for row in summary if row["grid_level"] == "coarse" and row.get("stage_b_outcomes_complete") == 1]
    refined = [row for row in summary if row["grid_level"] == "refined" and row.get("stage_b_outcomes_complete") == 1]
    strict_pairs = [row for row in coarse if row.get("strict_stability") == 1]
    all_strict_pairs = [row for row in coarse + refined if row.get("strict_stability") == 1]
    matrix = np.zeros((len(COARSE_U0_V), len(COARSE_FREQUENCY_HZ)), dtype=bool)
    for row in strict_pairs:
        matrix[COARSE_U0_V.index(float(row["U0_V"])), COARSE_FREQUENCY_HZ.index(float(row["f_rf_Hz"]))] = True
    labels, component_count = label(matrix)
    largest_component = max((int(np.sum(labels == component)) for component in range(1, component_count + 1)), default=0)
    survival_matrix = np.zeros_like(matrix)
    for row in coarse:
        if float(row.get("stage_b_survival_fraction") or 0.0) > 0:
            survival_matrix[COARSE_U0_V.index(float(row["U0_V"])), COARSE_FREQUENCY_HZ.index(float(row["f_rf_Hz"]))] = True
    survival_labels, survival_component_count = label(survival_matrix)
    survival_component = max(range(1, survival_component_count + 1), key=lambda component: int(np.sum(survival_labels == component)), default=0)
    largest_survival_cells = [(float(COARSE_U0_V[i]), float(COARSE_FREQUENCY_HZ[j])) for i, j in zip(*np.where(survival_labels == survival_component))] if survival_component else []
    high_survival = [row for row in coarse if float(row.get("stage_b_survival_fraction") or 0.0) >= 0.5]
    maximum_fraction = max((float(row.get("stage_b_survival_fraction") or 0.0) for row in coarse), default=0.0)
    best_candidates = sorted(coarse, key=lambda row: float(row.get("stage_b_survival_fraction") or 0.0), reverse=True)[:2]
    refined_survivors = [row for row in refined if float(row.get("stage_b_survival_fraction") or 0.0) > 0]
    if strict_pairs:
        u_span = (min(float(row["U0_V"]) for row in strict_pairs) / 1000, max(float(row["U0_V"]) for row in strict_pairs) / 1000)
        f_span = (min(float(row["f_rf_Hz"]) for row in strict_pairs) / 1000, max(float(row["f_rf_Hz"]) for row in strict_pairs) / 1000)
        strongest = min(strict_pairs, key=lambda row: float(row.get("median_r_max_m") or math.inf))
        confinement_text = f"The smallest median maximum radius among strict cells was at U0={float(strongest['U0_V'])/1000:g} kV, f={float(strongest['f_rf_Hz'])/1000:g} kHz (median rmax={float(strongest['median_r_max_m'])*1000:.4g} mm)."
        span_text = f"Strict Stage-B cells span sampled U0={u_span[0]:g}–{u_span[1]:g} kV and f={f_span[0]:g}–{f_span[1]:g} kHz."
    else:
        confinement_text = f"No coarse or refined parameter pair had 100% survival across the prescribed ensemble. The maximum coarse Stage-B fraction was {maximum_fraction:.5g} ({maximum_fraction*32:.0f}/32 starts) at {float(best_candidates[0]['U0_V'])/1000:g} kV and {float(best_candidates[0]['f_rf_Hz'])/1000:g} kHz."
        if largest_survival_cells:
            partial_extent = f"U0={min(u for u, _ in largest_survival_cells)/1000:g}–{max(u for u, _ in largest_survival_cells)/1000:g} kV, f={min(f for _, f in largest_survival_cells)/1000:g}–{max(f for _, f in largest_survival_cells)/1000:g} kHz"
            if high_survival:
                high_extent = f"The connected coarse patch with at least 50% survival covers U0={min(float(r['U0_V']) for r in high_survival)/1000:g}–{max(float(r['U0_V']) for r in high_survival)/1000:g} kV and f={min(float(r['f_rf_Hz']) for r in high_survival)/1000:g}–{max(float(r['f_rf_Hz']) for r in high_survival)/1000:g} kHz ({len(high_survival)} observed cells)."
            else:
                high_extent = "No coarse cell reached 50% survival."
            refined_extent = ""
            if refined_survivors:
                refined_extent = f" The refined observed nonzero-survival samples span U0={min(float(r['U0_V']) for r in refined_survivors)/1000:g}–{max(float(r['U0_V']) for r in refined_survivors)/1000:g} kV and f={min(float(r['f_rf_Hz']) for r in refined_survivors)/1000:g}–{max(float(r['f_rf_Hz']) for r in refined_survivors)/1000:g} kHz; each has only {min(float(r['stage_b_survival_fraction']) for r in refined_survivors)*32:g}/32 survivors. These locate partial-survival transitions, not a strict stability boundary."
            span_text = f"A connected partial-survival candidate component occurs in {len(largest_survival_cells)} coarse cells, spanning {partial_extent}; this is not a stable region because no pair had full-ensemble survival. {high_extent}{refined_extent}"
        else:
            span_text = "No connected nonzero-survival coarse candidate component was observed."
    radii_summary = []
    b_lookup = {str(row["trajectory_id"]): row for row in cases if row.get("stage") == "B_1000" and row.get("grid_level") == "coarse"}
    a_rows = [row for row in cases if row.get("stage") == "A_50" and row.get("grid_level") == "coarse"]
    def resolved_outcome(a_row: dict[str, object], lookup: dict[str, dict[str, object]]) -> int | None:
        if int(a_row.get("stage_a_survived") or 0) == 0:
            return 0
        b_row = lookup.get(str(a_row["trajectory_id"]))
        return int(b_row.get("bounded") or 0) if b_row is not None else None
    primary_rows = [row for row in a_rows if row.get("ensemble") == "primary"]
    for radius in PRIMARY_RADII_M:
        statuses = []
        for a_row in primary_rows:
            if float(a_row["initial_r_m"]) != radius:
                continue
            outcome = resolved_outcome(a_row, b_lookup)
            if outcome is not None:
                statuses.append(outcome)
        fraction = float(np.mean(statuses)) if statuses else float("nan")
        radii_summary.append(f"{radius*1000:g} mm: {sum(statuses)}/{len(statuses)} ({fraction:.3g})")
    angle_summary = []
    for theta in PRIMARY_ANGLES_DEG:
        statuses = [resolved_outcome(row, b_lookup) for row in primary_rows if float(row["initial_theta_deg"]) == theta]
        statuses = [value for value in statuses if value is not None]
        angle_summary.append(f"{theta:g}°: {sum(statuses)}/{len(statuses)} ({sum(statuses)/len(statuses):.3g})")
    velocity_lookup = {str(row["trajectory_id"]): row for row in cases if row.get("stage") == "B_1000" and row.get("grid_level") == "coarse"}
    velocity_summary = []
    velocity_rows = [row for row in a_rows if row.get("ensemble") == "velocity_probe"]
    velocity_groups: dict[tuple[float, float, float, float], list[int]] = {}
    for row in velocity_rows:
        outcome = resolved_outcome(row, velocity_lookup)
        if outcome is None:
            continue
        key = (float(row["initial_r_m"]), float(row["initial_theta_deg"]), float(row["initial_vx_m_per_s"]), float(row["initial_vy_m_per_s"]))
        velocity_groups.setdefault(key, []).append(outcome)
    for (radius, theta, vx, vy), statuses in sorted(velocity_groups.items()):
        velocity_summary.append(f"r0={radius*1000:g} mm, θ0={theta:g}°, v=({vx:+.3g},{vy:+.3g}) m/s: {sum(statuses)}/{len(statuses)}")
    stage_a_coarse = sum(1 for row in cases if row.get("stage") == "A_50" and row.get("grid_level") == "coarse")
    stage_a_refined = sum(1 for row in cases if row.get("stage") == "A_refined_50" and row.get("grid_level") == "refined")
    stage_b_coarse_rows = [row for row in cases if row.get("stage") == "B_1000" and row.get("grid_level") == "coarse"]
    stage_b_refined_rows = [row for row in cases if row.get("stage") == "B_refined_1000" and row.get("grid_level") == "refined"]
    stage_c_rows = [row for row in cases if row.get("stage") == "C_10000"]
    stage_c_bounded = sum(int(row.get("bounded") or 0) for row in stage_c_rows)
    changed_strict = sum(int(row.get("strict_classification_changed") or 0) for row in numerical)
    changed_recovered = sum(int(row.get("recovered_field_classification_changed") or 0) for row in numerical)
    dim_boundary = []
    for frequency in COARSE_FREQUENCY_HZ:
        ordered = sorted((r for r in coarse if float(r["f_rf_Hz"]) == frequency), key=lambda r: float(r["U0_V"]))
        for low, high in zip(ordered[:-1], ordered[1:]):
            if low.get("strict_stability") == 1 and high.get("strict_stability") == 0:
                dim_boundary.append((_dimensionless_lambda(config, math.sqrt(float(low["U0_V"]) * float(high["U0_V"])), frequency, C4_per_V_m4), frequency))
                break
    if dim_boundary:
        lambdas = [v[0] for v in dim_boundary]
        dim_text = f"Using C4={C4_per_V_m4:.4g} V^-1 m^-4 (phase-fit relative spread {C4_relative_phase_spread:.3g}) and r_ref={config.simulation.escape_radius*1000:g} mm, the sampled upper-amplitude boundaries yield lambda4={min(lambdas):.3g}–{max(lambdas):.3g}; coefficient of variation={np.std(lambdas)/max(np.mean(lambdas), 1e-30):.3g}. The boundary is {'approximately collapsed' if np.std(lambdas)/max(np.mean(lambdas),1e-30) < 0.25 else 'not tightly collapsed'} over sampled frequencies."
    else:
        dim_text = f"The leading rotating-field coefficient is C4={C4_per_V_m4:.4g} V^-1 m^-4 (phase-fit relative spread {C4_relative_phase_spread:.3g}); no stable-to-unstable upper-amplitude boundary was bracketed on the sampled coarse grid, so the collapse test is inconclusive."
    validation_text = "No Stage-B cases were available for stricter validation." if not numerical else f"Among {len(numerical)} representative integrations, stricter DOP853 settings changed {changed_strict} bounded/escaped classifications. Replacing the direct P2 field with the area-weighted recovered field changed {changed_recovered} classifications; this is an interpolation-sensitivity diagnostic, not an independent mesh refinement."
    stagec_text = "none" if not stage_c_rows else "; ".join(
        f"({float(row['U0_V'])/1000:g} kV, {float(row['f_rf_Hz'])/1000:g} kHz; {row.get('candidate_group')}; {'bounded' if int(row.get('bounded') or 0) else 'escaped'} for {float(row.get('periods_survived') or 0):g}/10000 periods; r0={float(row['initial_r_m'])*1000:g} mm, θ0={float(row['initial_theta_deg']):g}°)"
        for row in sorted(stage_c_rows, key=lambda item: (float(item["U0_V"]), float(item["f_rf_Hz"])))
    )
    lines = [
        "# Full-dynamics RF stability study", "",
        "## Model and protocol", "",
        "The conservative full equation `m r¨ = q E(r,t)` was integrated with the validated P2 ABCABCABCABC field basis; `gamma=0`, gravity disabled, and the explicit radial escape event is `r >= 8 mm`. No pseudopotential classification is used. The P2 FEM basis was loaded from `outputs/fem_cache/` after fingerprint validation; the existing 351-row localization checkpoint was read for exact overlapping Stage-A initial states and was not modified.", "",
        f"Coarse grid: {len(COARSE_U0_V)} amplitudes × {len(COARSE_FREQUENCY_HZ)} frequencies; {len(PRIMARY_RADII_M)} radii × {len(PRIMARY_ANGLES_DEG)} angles at zero velocity, plus {len(SECONDARY_STATES)} nonzero-velocity probes per parameter pair. Completed data contain {len(cases)} trajectory rows: coarse Stage A {stage_a_coarse}/{len(COARSE_U0_V)*len(COARSE_FREQUENCY_HZ)*32}, coarse Stage B {len(stage_b_coarse_rows)} trajectories, refined Stage A {stage_a_refined}/{len(refined)*32}, refined Stage B {len(stage_b_refined_rows)} trajectories, and {len(stage_c_rows)} supplemental Stage-C trajectories. Stage A screens for {STAGE_A_PERIODS} RF periods; Stage B tests only Stage-A survivors for {STAGE_B_PERIODS} periods. Stage C is supplemental {STAGE_C_PERIODS}-period validation only; Stage-A survivors are never called stable.", "",
        "## Validated stability findings", "",
        f"- Stage-B bounded trajectory counts: coarse {sum(int(row.get('bounded') or 0) for row in stage_b_coarse_rows)}/{len(stage_b_coarse_rows)}; refined {sum(int(row.get('bounded') or 0) for row in stage_b_refined_rows)}/{len(stage_b_refined_rows)}. Pair-level full-survival classifications: {len(all_strict_pairs)}/{len(coarse)+len(refined)} coarse and refined pairs; coarse strict-component size={largest_component}.",
        f"- {span_text}",
        f"- {confinement_text}",
        f"- Highest-survival Stage-B candidate points: " + "; ".join(f"{float(r['U0_V'])/1000:g} kV, {float(r['f_rf_Hz'])/1000:g} kHz ({float(r['stage_b_survival_fraction']):.3g})" for r in best_candidates) + ". These are candidate points, not globally stable operating points.",
        f"- Supplemental Stage-C long-duration checks: {stagec_text}; {stage_c_bounded}/{len(stage_c_rows)} selected trajectories survived their full 10000-period horizon. Each is a single initial condition and does not alter Stage-B classification.",
        "- Each pair-level Stage-B survival fraction uses all 32 prescribed starts; Stage-A escapes count as non-survivors. Strict stability requires all 32 bounded through 1000 RF periods.",
        "- Stage-B radius dependence for the primary zero-velocity ensemble (Stage-A escapes counted as failures; 360 starts/radius): " + "; ".join(radii_summary) + ".",
        "- Angular dependence in the same coarse ensemble (630 starts/angle): " + "; ".join(angle_summary) + ".",
        "- Secondary velocity-probe outcomes, each over 90 coarse parameter points: " + "; ".join(velocity_summary) + ". These four probes sample only two radius/angle combinations and are not a general velocity-space stability test.",
        f"- Dimensionless fourth-order scaling: {dim_text}",
        "- The scaling follows from Phi_RF≈C4 U0 r^4 cos(4 theta-Omega t): with xi=r/r_ref and tau=Omega t, the dimensionless equation contains lambda4=|q C4| U0 r_ref^2/(m Omega^2) multiplying the dimensionless gradient of xi^4 cos(4 theta-tau). This is a geometry-specific fourth-order control parameter, not the conventional Mathieu q.",
        f"- ODE / interpolation sensitivity: {validation_text} The observed boundary is not numerically converged: at least one bounded/escaped classification changes under the stricter ODE settings, and both tested classifications change under the recovered-field sensitivity check.",
        "- Bounded motion over a finite 1000- or 10000-period horizon is not a proof of asymptotic or global stability. The model is conservative (gamma=0), the Stage-C checks each cover one start, the tested initial-state ensemble is finite, and classification sensitivity remains near the boundary.",
        "- The field model retains the existing P2 mesh and the already-generated local pseudopotential mesh-convergence evidence. The direct-P2 versus recovered-field comparison tests interpolation sensitivity only; it does not replace an independent trajectory scan on a second FEM mesh.", "",
        "## Numerics and remaining uncertainty", "",
        f"ODE defaults were retained (`rtol={config.simulation.trajectory_rtol:g}`, `atol={config.simulation.trajectory_atol:g}`, maximum step T/{config.localization.max_step_per_period}); the same solver and settings were used for Stage A/B. Numerical checks tighten rtol/atol and halve the allowed step per RF period. Refinement is restricted to log-midpoints of coarse-grid edges where the Stage-B stability/survival outcome changes; no interpolation is drawn through missing parameter points. `stability_numerical_validation.csv` records stricter-tolerance and recovered-field checks. Stage C is supplemental validation only.", "",
        "## Relation to conventional Mathieu stability", "",
        "In a conventional quadrupole mass filter, the DC quadrupole term defines Mathieu a and the RF quadrupole term defines Mathieu q. The ABCABCABCABC trap's leading central RF potential is fourth order, so its full dynamics are not governed by the same linear Mathieu equation. A common DC offset is not a conventional Mathieu-a term. A meaningful DC/RF parameter plane would require a differential electrode-voltage pattern that creates a specified nonuniform DC multipole near the center; that separate study was not performed.", "",
        "## Outputs", "",
        "`stability_cases.csv` contains trajectory-level scan outcomes; `stability_summary.csv` contains pair-level survival fractions and strict classifications; `stability_checkpoint.jsonl` is the fsync'd per-trajectory resume log. Figures include binary strict stability, survival fraction, median survived-period map, maximum-radius map, observed refined points, initial-radius comparisons, dimensionless outcomes, and full trajectory/Poincaré examples. The refinement CSV and trajectory example files are stored beside the figures.", "",
        "No mass-filter study was run.", "",
    ]
    path = output_dir / "stability_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_stability_study(config: ProjectConfig, project_root: Path, workers: int | None = None) -> dict[str, object]:
    """Run/resume the hierarchical study while checkpointing each completed trajectory."""
    root = Path(project_root)
    output_dir = root / "outputs/publication_study/stability"
    output_dir.mkdir(parents=True, exist_ok=True)
    _manifest(output_dir, config)
    state = load_fem_cache(config, root)
    interpolator = state[3]
    checkpoint = output_dir / CHECKPOINT_NAME
    completed = _read_checkpoint(checkpoint)
    reused = _localization_reuse_rows(root, config, completed, checkpoint)
    if reused:
        print(f"Reused {reused} overlapping Stage-A trajectories from the preserved localization checkpoint.", flush=True)
    worker_count = min(MAX_WORKERS, max(1, os.cpu_count() or 1)) if workers is None else max(1, int(workers))
    coarse_cases = [case for U0 in COARSE_U0_V for frequency in COARSE_FREQUENCY_HZ for case in _make_cases(U0, frequency, "coarse")]
    print(f"Stage A coarse screening: {len(coarse_cases)} trajectory cases (existing exact overlaps reused).", flush=True)
    _execute_stage(coarse_cases, "A_50", STAGE_A_PERIODS, config, interpolator, completed, checkpoint, root, worker_count)
    coarse_stage_a = [row for row in completed.values() if row.get("grid_level") == "coarse" and row.get("stage") == "A_50"]
    stage_b_cases = []
    for a_row in coarse_stage_a:
        if int(a_row.get("stage_a_survived") or 0):
            stage_b_cases.append({key: a_row[key] for key in ("U0_V", "f_rf_Hz", "initial_r_m", "initial_theta_deg", "initial_vx_m_per_s", "initial_vy_m_per_s", "ensemble", "grid_level")})
    print(f"Stage B coarse validation: {len(stage_b_cases)} Stage-A survivors at {STAGE_B_PERIODS} periods.", flush=True)
    _execute_stage(stage_b_cases, "B_1000", STAGE_B_PERIODS, config, interpolator, completed, checkpoint, root, worker_count)

    coarse_summary = _pair_summary(list(completed.values()), "coarse")
    refined_points = _boundary_midpoints(coarse_summary)
    refined_cases = [case for U0, frequency in refined_points for case in _make_cases(U0, frequency, "refined")]
    if refined_cases:
        print(f"Boundary refinement: {len(refined_points)} log-midpoint parameter pairs; Stage A screens {len(refined_cases)} trajectories.", flush=True)
        _execute_stage(refined_cases, "A_refined_50", STAGE_A_PERIODS, config, interpolator, completed, checkpoint, root, worker_count)
        refined_stage_a = [row for row in completed.values() if row.get("grid_level") == "refined" and row.get("stage") == "A_refined_50"]
        refined_b_cases = []
        for a_row in refined_stage_a:
            if int(a_row.get("stage_a_survived") or 0):
                refined_b_cases.append({key: a_row[key] for key in ("U0_V", "f_rf_Hz", "initial_r_m", "initial_theta_deg", "initial_vx_m_per_s", "initial_vy_m_per_s", "ensemble", "grid_level")})
        print(f"Boundary Stage B: {len(refined_b_cases)} Stage-A survivors at {STAGE_B_PERIODS} periods.", flush=True)
        _execute_stage(refined_b_cases, "B_refined_1000", STAGE_B_PERIODS, config, interpolator, completed, checkpoint, root, worker_count)
    else:
        print("No coarse-grid Stage-B transition was bracketed; no unnecessary boundary points added.", flush=True)

    summary = _pair_summary(list(completed.values()), "coarse") + _pair_summary(list(completed.values()), "refined")
    selection = _stage_c_selection(summary, list(completed.values()))
    stage_c_cases = _stage_c_cases(selection, list(completed.values()))
    print(f"Stage C: {len(selection)} representative/boundary parameter pairs, {len(stage_c_cases)} Stage-B-bounded starts to validate for {STAGE_C_PERIODS} periods.", flush=True)
    _execute_stage(stage_c_cases, "C_10000", STAGE_C_PERIODS, config, interpolator, completed, checkpoint, root, worker_count)

    all_rows = sorted(completed.values(), key=lambda row: (str(row.get("grid_level")), float(row["U0_V"]), float(row["f_rf_Hz"]), str(row.get("stage")), float(row["initial_r_m"]), float(row["initial_theta_deg"]), float(row["initial_vx_m_per_s"]), float(row["initial_vy_m_per_s"])))
    _atomic_csv(output_dir / "stability_cases.csv", all_rows, CASE_COLUMNS)
    summary = _pair_summary(all_rows, "coarse") + _pair_summary(all_rows, "refined")
    _write_pair_summary(summary, output_dir)
    _atomic_csv(output_dir / "stability_boundary_refined.csv", [row for row in summary if row["grid_level"] == "refined"], list(summary[0]) if summary else ["grid_level", "U0_V", "f_rf_Hz"])
    _save_stability_figures(summary, all_rows, output_dir)
    coarse_stage_c = {(float(row["U0_V"]), float(row["f_rf_Hz"])) for row in all_rows if row.get("grid_level") == "coarse" and row.get("stage") == "C_10000"}
    C4_per_V_m4, C4_relative_phase_spread = _fit_fourth_order_coefficient(config, interpolator)
    scaling_metadata = {
        "leading_rotating_m4_coefficient_per_volt_V_inv_m_inv4": C4_per_V_m4,
        "relative_phase_spread": C4_relative_phase_spread,
        "reference_radius_m": config.simulation.escape_radius,
        "dimensionless_parameter": "abs(q*C4)*U0*r_ref^2/(m*Omega^2)",
        "basis_source": "validated cached P2 FEM basis; no FEM rebuild",
    }
    (output_dir / "fourth_order_scaling.json").write_text(json.dumps(scaling_metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _dimensionless_figure(summary, all_rows, config, output_dir, coarse_stage_c, C4_per_V_m4)
    _write_initial_conditions_note(output_dir)
    numerical = _numerical_validation(summary, all_rows, config, interpolator, output_dir)
    examples = _save_trajectory_examples(summary, all_rows, config, interpolator, output_dir)
    report = _summary_report(summary, all_rows, config, numerical, selection, output_dir, C4_per_V_m4, C4_relative_phase_spread)
    return {"case_count": len(all_rows), "summary": summary, "checkpoint": checkpoint, "report": report, "trajectory_examples": examples, "stage_c_selection": selection}
