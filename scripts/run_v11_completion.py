"""Continue V11 from its saved checkpoints. Use --plan for read-only checks."""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from run_v11_xy import OUT, ROOT, fields, voltages, write_json
from three_phase_trap.long_rf_run import run_lock
from v11_completion_dynamics import (
    Calculation, Launch, SOLVER, STATE_COLUMNS, STRICT, fingerprint, write_csv, xy_config,
)
from v11_prior import load_prior, prior_key
import numpy as np


DIRECTORY = OUT / "v11_completion"
PHASES = (0, 45, 90)
RESOLUTIONS = (21, 41, 61)
HORIZONS = (1000, 3000, 10000, 100000)
STAGES = ("phase90", "boundary_area", "velocity", "long", "fem", "matched")
STAGE_DETAILS = {
    "phase90": "XY с нулевой скоростью, 305 координат, равные 250 В, h50 N48.",
    "boundary_area": "Равные 250 В. Полные дисковые сетки 21/41/61 и измеренные переходы "
                     "на рёбрах исходной сетки с шагом 0.25 мм.",
    "velocity": "Фазы 0/45/90°, старты (0,0) и (0.5,0) мм, сетка vx/vy 9×9 "
                "в диапазоне ±100 м/с, исключена пара (0,0) м/с.",
    "long": "Фиксированная выборка по исходам и рёбрам карты 21×21 до 100000 RF. "
            "Продолжение удержанных стартов с 1000 RF; ранние выходы учтены.",
    "fem": "Репрезентативные чувствительные точки XY-срезов и скоростей, h35/h50 N48, "
           "строгий DOP853 h50 и прямой P4 h35/h50.",
    "matched": "Фазы 0/45/90°, 21×21, согласование среднеквадратичной силы поля на радиусе 4 мм.",
}


def available_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in (
                            "physical", "available", "page", "page_available", "virtual",
                            "virtual_available", "extended_available")]]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Cannot determine available physical memory")
        return status.available
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def worker_count(requested, gib_per_worker):
    cpus = (getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    if os.name == "nt":
        cpus = min(cpus, 61)
    memory = available_memory()
    max_by_memory = 1 if memory is None else int((memory / 2**30 - 1) / gib_per_worker)
    if memory is not None and max_by_memory < 1:
        raise RuntimeError(f"Available RAM is {memory / 2**30:.2f} GiB; "
                           f"need {1 + gib_per_worker:.2f} GiB to start this stage")
    return min(requested or cpus, cpus, max(1, max_by_memory))


def csv_rows(folder):
    with (folder / "results.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def xy_launches(grid, phase, voltage):
    return [Launch(model, (float(x), float(y), 0.0, 0.0), phase, voltage[model])
            for y in np.linspace(-.0075, .0075, grid)
            for x in np.linspace(-.0075, .0075, grid)
            if math.hypot(x, y) < .0075
            for model in "ABC"]


def read_launch(row):
    return Launch(row["model"], tuple(float(row[key]) for key in STATE_COLUMNS),
                  round(math.degrees(float(row["phase_rad"])), 9), float(row["voltage_V"]))


def xy_status(folder):
    return {(row["model"], round(float(row["x0_m"]) * 1000, 9),
             round(float(row["y0_m"]) * 1000, 9)): row["status"] for row in csv_rows(folder)}


def coarse_edges(status):
    for (model, x, y), left in status.items():
        for dx, dy in ((.75, 0), (0, .75)):
            right = status.get((model, round(x + dx, 9), round(y + dy, 9)))
            if right is not None and right != left:
                yield [(model, round(x + dx * i / 3, 9), round(y + dy * i / 3, 9))
                       for i in range(4)]


def boundary(calc, folder, config, source):
    status = xy_status(source)
    edges = list(coarse_edges(status))
    points = sorted({(x, y) for edge in edges for _, x, y in edge[1:-1]})
    launches = [Launch(model, (x / 1000, y / 1000, 0.0, 0.0), config["phase_deg"])
                for x, y in points for model in "ABC"]
    calc.run(folder, {**config, "step_mm": .25, "source_map": relative(source)}, launches)
    status.update(xy_status(folder))
    rows = [{"model": a[0], "x_left_m": a[1] / 1000, "y_left_m": a[2] / 1000,
             "x_right_m": b[1] / 1000, "y_right_m": b[2] / 1000,
             "status_left": status[a], "status_right": status[b], "width_mm": .25}
            for edge in edges for a, b in zip(edge, edge[1:]) if status[a] != status[b]]
    write_csv(folder / "edges.csv", rows)


def fixed_cohort(source):
    status = xy_status(source)
    positions = sorted({(x, y) for _, x, y in status})
    edge_points = {(x, y) for edge in coarse_edges(status) for _, x, y in (edge[0], edge[-1])}
    grouped = {}
    for x, y in positions:
        pattern = "".join(model for model in "ABC" if status[model, x, y] == "retained")
        region = "edge" if (x, y) in edge_points else "interior"
        grouped.setdefault((pattern or "none", region), []).append((x, y))
    for (pattern, region), points in sorted(grouped.items()):
        width = min(2, len(points))
        # Deterministic spread within each class using radius then azimuth.
        ordered = sorted(points, key=lambda xy: (math.hypot(*xy), math.atan2(xy[1], xy[0])))
        for index in np.linspace(0, len(ordered) - 1, width, dtype=int):
            yield ordered[index], f"{region}:{pattern}"


def candidates(maps, phase):
    selected = {}
    for model in "ABC":
        folder = DIRECTORY / "boundary" / f"phase{phase:02d}"
        with (folder / "edges.csv").open(encoding="utf-8", newline="") as stream:
            edges = [row for row in csv.DictReader(stream) if row["model"] == model]
        if edges:
            edge = min(edges, key=lambda row: hashlib_order(row, model, phase))
            for side in ("left", "right"):
                state = (float(edge[f"x_{side}_m"]), float(edge[f"y_{side}_m"]), 0.0, 0.0)
                selected[Launch(model, state, phase)] = "XY transition endpoint"
        velocities = [row for row in csv_rows(maps["velocity"] / f"phase{phase:02d}")
                      if row["model"] == model]
        escaped = [row for row in velocities if row["status"] == "escaped"]
        retained = [row for row in velocities if row["status"] == "retained"]
        if escaped and retained:
            # Prefer the latest escape at a sampled velocity transition.
            by_state = {read_launch(row).state: row["status"] for row in velocities}
            def adjacent(row):
                x, y, vx, vy = read_launch(row).state
                return any(by_state.get((x, y, vx + dx, vy + dy)) == "retained"
                           for dx, dy in ((-25, 0), (25, 0), (0, -25), (0, 25)))
            near = [row for row in escaped if adjacent(row)]
            if near:
                launch = read_launch(max(near, key=lambda row: float(row["escape_RF"])))
                selected[launch] = "velocity transition, latest escaping neighbor"
    return selected


def hashlib_order(row, model, phase):
    return fingerprint([model, phase, row["x_left_m"], row["y_left_m"],
                        row["x_right_m"], row["y_right_m"]])


def relative(path):
    return path.relative_to(ROOT).as_posix()


def git(*arguments, capture=False):
    result = subprocess.run(("git", *arguments), cwd=ROOT, text=True, check=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def check_branch():
    if git("branch", "--show-current", capture=True) != "master":
        raise RuntimeError("V11 expects the master branch")


class Publication:
    def __init__(self, directory):
        self.journal = directory / "checkpoints" / "publication.json"
        self.output = relative(directory)

    def check(self):
        check_branch()
        pending = self.pending()
        staged = git("diff", "--cached", "--name-only", capture=True).splitlines()
        if staged and (pending is None or any(not path.startswith(self.output + "/") for path in staged)):
            raise RuntimeError("Staged changes outside the active V11 stage must be handled before publication")
        git("fetch", "origin", "master")
        remote = git("rev-parse", "FETCH_HEAD", capture=True)
        head = git("rev-parse", "HEAD", capture=True)
        if pending is None:
            if head != remote or git("status", "--porcelain", "--", self.output, capture=True):
                raise RuntimeError("Publication needs master at origin/master and a clean V11 output directory")
        elif head != pending["base"]:
            if (git("rev-parse", "HEAD^", capture=True) != pending["base"]
                    or git("log", "-1", "--format=%s", capture=True) != f"V11: complete {pending['stage']}"):
                raise RuntimeError("Unexpected local commit during V11 publication")
            if remote not in (head, pending["base"]):
                raise RuntimeError("Remote master advanced during V11 publication")
        elif remote != pending["base"]:
            raise RuntimeError("Remote master advanced during V11 publication")

    def pending(self):
        return json.loads(self.journal.read_text(encoding="utf-8")) if self.journal.exists() else None

    def begin(self, stage):
        pending = self.pending()
        if pending is None:
            write_json(self.journal, {"stage": stage, "base": git("rev-parse", "HEAD", capture=True)})
        elif pending["stage"] != stage:
            raise RuntimeError(f"Pending publication is for {pending['stage']}, not {stage}")

    def finish(self, stage):
        self.check()
        pending = self.pending()
        if pending is None or pending["stage"] != stage:
            raise RuntimeError(f"No pending publication for {stage}")
        if git("rev-parse", "HEAD", capture=True) == pending["base"]:
            git("add", "--", self.output)
            git("diff", "--cached", "--check", "--", self.output)
            git("commit", "--only", "-m", f"V11: complete {stage}", "--", self.output)
        git("push", "origin", "HEAD:refs/heads/master")
        self.journal.unlink()


def stage_runner(args, configs, prior, hashes):
    from run_v11_fem_checks import run_checks

    directory = DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    publication = Publication(directory)
    if args.publish:
        publication.check()
    elif publication.pending():
        raise RuntimeError("Resume the pending V11 stage with --publish")
    calc = Calculation(directory / "checkpoints" / "n48", prior, hashes, args.workers)
    maps = {"velocity": directory / "velocity"}
    coarse = {
        phase: OUT / "xy" / fingerprint(configs[phase])[:16] if phase != 90
        else directory / "xy" / f"phase{phase:02d}_21"
        for phase in PHASES
    }
    manifest_path = directory / "report_inputs.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"required_stages": STAGES, "stage_details": STAGE_DETAILS,
                    "maps": [{"stage": "baseline", "kind": "xy", "directory": relative(coarse[p])}
                             for p in (0, 45)]}
    def add_input(stage, kind, folder, **extra):
        entry = {"stage": stage, "kind": kind, "directory": relative(folder), **extra}
        if entry not in manifest["maps"]:
            manifest["maps"].append(entry)

    for stage in STAGES:
        if stage in manifest.get("completed_stages", []):
            if args.publish and (publication.pending() or {}).get("stage") == stage:
                publication.finish(stage)
            if stage == args.through:
                break
            continue
        if args.publish:
            publication.begin(stage)
        write_json(directory / "prior_audit.json", prior.audit)
        print(f"\n--- V11 {stage} ---", flush=True)
        if stage == "phase90":
            config = configs[90]
            calc.run(coarse[90], config, xy_launches(21, 90, config["voltage_V"]))
            add_input(stage, "xy", coarse[90])
        elif stage == "boundary_area":
            for phase in PHASES:
                config = configs[phase]
                folder = directory / "boundary" / f"phase{phase:02d}"
                boundary(calc, folder, config, coarse[phase])
                add_input(stage, "boundary", folder)
            for grid in RESOLUTIONS[1:]:
                for phase in PHASES:
                    config = {**configs[phase], "grid": grid}
                    folder = directory / "xy" / f"phase{phase:02d}_{grid}"
                    calc.run(folder, config, xy_launches(grid, phase, config["voltage_V"]))
                    add_input(stage, "xy", folder)
        elif stage == "velocity":
            origins = [[0.0, 0.0], [.0005, 0.0]]
            speeds = np.linspace(-100, 100, 9)
            for phase in PHASES:
                folder = maps["velocity"] / f"phase{phase:02d}"
                config = {**configs[phase], "grid": 9, "speed_max_m_s": 100.0,
                          "origins_m": origins, "zero_velocity_excluded": True}
                launches = [Launch(m, (x, y, float(vx), float(vy)), phase)
                            for x, y in origins for vy in speeds for vx in speeds
                            if vx != 0 or vy != 0 for m in "ABC"]
                calc.run(folder, config, launches)
                add_input(stage, "velocity", folder)
        elif stage == "long":
            all_cohorts = {phase: list(fixed_cohort(coarse[phase])) for phase in PHASES}
            cohort_file = directory / "long" / "cohort.json"
            cohort_file.parent.mkdir(parents=True, exist_ok=True)
            frozen = [{"phase_deg": phase, "x0_mm": x, "y0_mm": y, "stratum": stratum}
                      for phase, cohort in all_cohorts.items() for (x, y), stratum in cohort]
            if cohort_file.exists():
                if json.loads(cohort_file.read_text(encoding="utf-8")) != frozen:
                    raise ValueError("Long-term cohort differs from the frozen selection")
            else:
                write_json(cohort_file, frozen)
            for phase in PHASES:
                folder = directory / "long" / f"phase{phase:02d}"
                cohort = all_cohorts[phase]
                launches = [Launch(m, (x / 1000, y / 1000, 0.0, 0.0), phase)
                            for (x, y), _ in cohort for m in "ABC"]
                config = {**configs[phase], "horizon_RF": HORIZONS[-1],
                          "known_horizons": list(HORIZONS), "segment_after_1000_RF": 1000,
                          "selection": STAGE_DETAILS[stage]}
                calc.run(folder, config, launches,
                         attributes=[{"stratum": stratum} for _, stratum in cohort for m in "ABC"])
                add_input(stage, "long", folder)
        elif stage == "fem":
            selected = {launch: reason for phase in PHASES for launch, reason in candidates(maps, phase).items()}
            # Include the existing direct C check as a control with both published meshes.
            selected[Launch("C", (.00375, -.006, 0.0, 0.0), 0)] = "published native C control"
            selected = dict({prior_key(50, launch.model, launch.phase, launch.state): (launch, reason)
                             for launch, reason in selected.items()}.values())
            attributes = [{"reason": reason} for reason in selected.values()]
            config = {key: value for key, value in configs[0].items()
                      if key not in ("grid", "phase_deg")}
            for mesh, solver, label in ((50, SOLVER, "N48"), (35, SOLVER, "N48"),
                                        (50, STRICT, "N48-strict")):
                folder = directory / f"fem_{label.lower()}_h{mesh}"
                calc.run(folder, {**config, "mesh_um": mesh, "field_sha256": hashes[mesh],
                                  "solver": solver, "representation": label},
                         list(selected), attributes=attributes, solver=solver)
                add_input(stage, "fem", folder)
            folder = directory / "fem_native"
            run_checks(folder, [{**launch.initial(), "reason": reason}
                                for launch, reason in selected.items()],
                       worker_count(args.native_workers, 2.5), horizon=1000)
            add_input(stage, "fem", folder, allow_numerical_errors=True)
        else:
            matched = voltages(fields(50)[0], "matched", 4.0)
            for phase in PHASES:
                config = {**configs[phase], "protocol": "matched", "reference_radius_mm": 4.0,
                          "voltage_V": matched}
                folder = directory / "matched" / f"phase{phase:02d}"
                calc.run(folder, config, xy_launches(21, phase, matched))
                add_input(stage, "xy", folder)
        write_json(manifest_path, manifest)
        subprocess.run((sys.executable, str(ROOT / "scripts/report_v11_completion.py"),
                        "--run-dir", str(directory), "--require-stage", stage), cwd=ROOT, check=True)
        manifest.setdefault("completed_stages", []).append(stage)
        write_json(manifest_path, manifest)
        if args.publish:
            publication.finish(stage)
        if stage == args.through:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="CPU worker cap (0: automatic)")
    parser.add_argument("--native-workers", type=int, default=0, help="native P4 worker cap (0: automatic)")
    parser.add_argument("--through", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--publish", action="store_true", help="commit and push each verified stage to master")
    args = parser.parse_args()
    if min(args.workers, args.native_workers) < 0:
        parser.error("Worker caps must be nonnegative")
    check_branch()
    from run_v11_fem_checks import preflight

    hashes = {mesh: fields(mesh)[1] for mesh in (35, 50)}
    configs = {phase: xy_config(phase, hashes[50]) for phase in PHASES}
    prior = load_prior(OUT, configs, dry_run=args.plan)
    native_workers = None
    if STAGES.index(args.through) >= STAGES.index("fem"):
        preflight()
        if args.plan:
            native_workers = worker_count(args.native_workers, 2.5)
    args.workers = worker_count(args.workers, .25)
    print(json.dumps({"cpu_workers": args.workers, "native_workers_now": native_workers,
                      "loaded_initial_states": len(prior), "through": args.through,
                      "phase90_grid_states": len(xy_launches(21, 90, configs[90]["voltage_V"])),
                      "fem_fields": hashes, "output": relative(DIRECTORY)}, indent=2), flush=True)
    if args.plan:
        return
    # OS-held lock is released after normal exit or a terminated process.
    lock_dir = DIRECTORY / "checkpoints"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(lock_dir):
        stage_runner(args, configs, prior, hashes)


if __name__ == "__main__":
    main()
