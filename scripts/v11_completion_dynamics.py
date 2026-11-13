"""Shared, checkpointed full-RF trajectory cache for the V11 stages."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os

# run_v11_xy sets one BLAS thread per process before NumPy is imported.
from run_v11_xy import CHARGE, FREQ, MASS, RADIUS, fields, sha, write_json
from three_phase_trap import pilot_dynamics
from three_phase_trap.pilot_dynamics import PolynomialRF, integrate_rf
import numpy as np


STATE_COLUMNS = ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s")
VOLTAGE = dict.fromkeys("ABC", 250.0)
ATOL = (2e-13, 2e-13, 2e-11, 2e-11)
SOLVER = {"method": "DOP853", "rtol": 2e-9, "atol_SI": list(ATOL), "max_step_RF": .02}
STRICT = {"method": "DOP853", "rtol": 2e-10,
          "atol_SI": [value / 10 for value in ATOL], "max_step_RF": .01}
_worker_fields = None
KERNEL_HASH = sha(pilot_dynamics.__file__)


@dataclass(frozen=True)
class Launch:
    model: str
    state: tuple[float, float, float, float]
    phase: float
    voltage: float = 250.0

    def initial(self):
        return {"model": self.model, **dict(zip(STATE_COLUMNS, self.state)),
                "phase_rad": math.radians(self.phase), "voltage_V": self.voltage}


def xy_config(phase, hashes, grid=21):
    return {"grid": grid, "phase_deg": float(phase), "protocol": "equal",
            "reference_radius_mm": None, "mesh_um": 50, "horizon_RF": 1000,
            "segment_RF": 100, "frequency_Hz": FREQ, "mass_kg": MASS,
            "charge_C": CHARGE, "aperture_m": RADIUS,
            "voltage_V": VOLTAGE, "field_sha256": hashes, "solver": SOLVER}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_csv(path, rows, columns=None):
    keys = columns if columns is not None else list(dict.fromkeys(
        key for row in rows for key in row))
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def init_fields(mesh, models):
    global _worker_fields
    quadratures, _ = fields(mesh)
    _worker_fields = {name: PolynomialRF(q) for name, q in zip("ABC", quadratures)
                      if name in models}


def read_part(path):
    with np.load(path, allow_pickle=False) as part:
        return json.loads(part["record"].item())


def integrate_launch(launch, directory, horizon, seed, solver):
    directory.mkdir(parents=True, exist_ok=True)
    started_at = 0.0 if seed is None else seed["reached_RF"]
    state = launch.state if seed is None else seed["final_state_SI"]
    row = seed
    start = int(started_at)
    stops = range((start // 100 + 1) * 100, min(horizon, 1000) + 1, 100)
    long_stops = range(max(2000, (start // 1000 + 1) * 1000), horizon + 1, 1000)
    for stop in (*stops, *long_stops):
        try:
            result = integrate_rf(
                _worker_fields[launch.model], state, CHARGE * launch.voltage, MASS,
                FREQ, math.radians(launch.phase), started_at / FREQ, stop / FREQ,
                output_samples_per_period=1, rtol=solver["rtol"], atol=solver["atol_SI"],
                max_steps_per_period=round(1 / solver["max_step_RF"]), radius=RADIUS,
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            diagnostic = {**launch.initial(), "status": "numerical_error",
                          "escape_RF": None, "reached_RF": started_at,
                          "final_state_SI": list(state), "failed_stop_RF": stop,
                          "error": f"{type(error).__name__}: {error}"}
            write_json(directory / f"error_{stop:06d}.json", diagnostic)
            return diagnostic
        escape = None if result.exit_time is None else result.exit_time * FREQ
        state = result.states[-1].tolist()
        row = {**launch.initial(), "status": "retained" if escape is None else "escaped",
               "escape_RF": escape, "reached_RF": stop if escape is None else escape,
               "final_state_SI": state,
               "sampled_max_radius_m": max(
                   float(np.linalg.norm(result.states[:, :2], axis=1).max()),
                   0.0 if row is None else row["sampled_max_radius_m"]),
               "nfev_segment": result.nfev, "wall_s_segment": result.wall_seconds}
        with (directory / "part.tmp").open("wb") as stream:
            np.savez(stream, time_s=result.time, states_SI=result.states,
                     record=json.dumps(row, allow_nan=False))
        os.replace(directory / "part.tmp", directory / f"rf_{stop:06d}.npz")
        if escape is not None:
            break
        started_at = stop
    return row


def usable(row, horizon):
    return row is not None and (row["status"] == "escaped" or row["reached_RF"] == horizon)


class Calculation:
    def __init__(self, cache, prior, hashes, workers):
        self.cache = cache
        self.prior = prior
        self.hashes = hashes
        self.workers = workers

    def run(self, folder, config, launches, *, attributes=None, solver=SOLVER):
        from v11_prior import prior_key

        folder.mkdir(parents=True, exist_ok=True)
        horizon = config["horizon_RF"]
        if horizon < 100 or horizon % (100 if horizon <= 1000 else 1000):
            raise ValueError(f"Unsupported checkpoint horizon: {horizon}")
        mesh = config["mesh_um"]
        config_path = folder / "config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Calculation config changed in {folder}")
        if not config_path.exists():
            write_json(config_path, config)
        attributes = attributes or [{} for _ in launches]
        initial = [{**launch.initial(), **extra} for launch, extra in zip(launches, attributes, strict=True)]
        write_csv(folder / "initial_states.csv", initial)
        rows = [None] * len(initial)
        pending = {}
        seen = set()
        for index, launch in enumerate(launches):
            key = prior_key(mesh, launch.model, launch.phase, launch.state)
            tag = fingerprint({
                "launch": key, "voltage_V": launch.voltage, "solver": solver,
                "field_sha256": self.hashes[mesh][launch.model],
                "frequency_Hz": FREQ, "mass_kg": MASS, "charge_C": CHARGE,
                "aperture_m": RADIUS, "kernel_sha256": KERNEL_HASH, "schema": 1,
            })
            if tag in seen:
                raise ValueError(f"Duplicate initial state in {folder}: {launch}")
            seen.add(tag)
            directory = self.cache / tag[:2] / tag[2:]
            seed = None
            if launch.voltage == 250.0 and solver == SOLVER:
                prior_row = self.prior.get(key)
                if prior_row is not None and prior_row["reached_RF"] <= horizon:
                    seed = prior_row
                else:
                    for _, row in self.prior.segments.get(key, ()):
                        if row["reached_RF"] <= horizon:
                            seed = row
            # A part is published only after its trajectory and record have both been saved.
            parts = sorted(directory.glob("rf_*.npz"))
            used_part = False
            if parts:
                latest = read_part(parts[-1])
                if latest["reached_RF"] > horizon:
                    eligible = [path for path in parts if int(path.stem[3:]) <= horizon]
                    latest = read_part(eligible[-1]) if eligible else None
                if latest is not None and (seed is None or latest["reached_RF"] >= seed["reached_RF"]):
                    seed = latest
                    used_part = True
            if usable(seed, horizon):
                rows[index] = {**seed, "checkpoint_key": tag if used_part else ""}
            else:
                pending[index] = (launch, directory, seed, tag)
        print(f"{folder.name}: {len(rows) - len(pending)}/{len(rows)} reused; "
              f"{len(pending)} pending to {horizon} RF", flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=min(self.workers, len(pending)),
                                     initializer=init_fields, initargs=(mesh, "ABC")) as pool:
                jobs = {pool.submit(integrate_launch, launch, directory, horizon, seed, solver): index
                        for index, (launch, directory, seed, _) in pending.items()}
                try:
                    for completed, future in enumerate(as_completed(jobs), 1):
                        index = jobs[future]
                        rows[index] = {**future.result(), "checkpoint_key": pending[index][3]}
                        if completed % 50 == 0 or completed == len(jobs):
                            print(f"{folder.name}: {completed}/{len(jobs)} computed", flush=True)
                except BaseException:
                    if hasattr(pool, "terminate_workers"):
                        pool.terminate_workers()
                    else:
                        pool.shutdown(wait=True, cancel_futures=True)
                    raise
        results = [{**row, **state} for row, state in zip(rows, initial, strict=True)]
        write_csv(folder / "results.csv", results)
        return results
