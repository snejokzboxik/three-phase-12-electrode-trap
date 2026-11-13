"""Read and validate reusable V11 results without extracting the archives."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


BASELINE = {
    "xy/4ceec3a67c8fdfcd": (50, 0., 915),
    "xy/7c854b0d1f8f059f": (50, 45., 915),
    "numerical_checks/h35_edge_equal_phase0": (35, 0., 48),
}
PILOT = "xy/5241b905a48074eb"
ARCHIVES = {
    "checkpoints_v11.zip": (
        "d8b021e1924eb781c0925d739942bb07c45a09d1fac378254a143bedaac139c6", 7937,
    ),
    "checkpoints_phase45.zip": (
        "bec76699bd94031308cf99d4620990c250f576c5a76c2bf10cbe8aadc6f76e37", 8086,
    ),
}
STATE_COLUMNS = ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s")
CONTRACT_FIELDS = (
    "protocol", "reference_radius_mm", "mesh_um", "frequency_Hz", "mass_kg",
    "charge_C", "aperture_m", "voltage_V", "field_sha256", "solver",
)


class PriorResults(dict):
    def __init__(self):
        super().__init__()
        self.segments = {}
        self.audit = {"archives": [], "sources": []}


def prior_key(mesh, model, phase_deg, state):
    """Collapse floating-point differences from generating nested XY grids."""
    canonical_state = tuple(round(float(value), 14) or 0.0 for value in state)
    return int(mesh), model, float(phase_deg) or 0.0, canonical_state


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_archives(out, audit):
    files = defaultdict(dict)
    for name, (expected_hash, count) in ARCHIVES.items():
        path = out / name
        if not path.is_file():
            audit.append({"archive": name, "status": "missing"})
            continue
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != expected_hash:
            raise ValueError(f"V11 checkpoint archive SHA-256 mismatch: {path}")
        with ZipFile(path) as archive:
            members = archive.namelist()
            if len(set(members)) != count or len(members) != count:
                raise ValueError(f"Unexpected V11 archive member count: {path}")
            for name_in_zip in members:
                member = PurePosixPath(name_in_zip)
                if len(member.parts) != 5 or member.parts[2] != "checkpoints":
                    raise ValueError(f"Unexpected V11 archive path: {name_in_zip}")
                source = member.parent.parent.parent.as_posix()
                if source == PILOT:
                    continue
                if source not in BASELINE or member in files[source]:
                    raise ValueError(f"Unexpected or duplicate V11 checkpoint: {name_in_zip}")
                files[source][member] = json.loads(archive.read(member.as_posix()))
        audit.append({"archive": name, "status": "verified", "sha256": checksum, "members": count})
    return files


def _typed_row(row):
    if row["status"] == "numerical_error":
        return row
    for column in (*STATE_COLUMNS, "phase_rad", "voltage_V", "sampled_max_radius_m", "wall_s_segment"):
        row[column] = float(row[column])
    reached = float(row["reached_RF"])
    row["reached_RF"] = int(reached) if reached.is_integer() else reached
    row["escape_RF"] = None if row["escape_RF"] in ("", None) else float(row["escape_RF"])
    row["nfev_segment"] = int(row["nfev_segment"])
    if isinstance(row["final_state_SI"], str):
        row["final_state_SI"] = json.loads(row["final_state_SI"])
    return row


def _verify_row(row, config, end, radius):
    model, phase = row["model"], config["phase_deg"]
    initial = tuple(row[column] for column in STATE_COLUMNS)
    final = row["final_state_SI"]
    maximum = row["sampled_max_radius_m"]
    reached, escape = row["reached_RF"], row["escape_RF"]
    final_radius = math.hypot(*final[:2])
    if (model not in "ABC" or len(model) != 1 or len(final) != 4
            or not all(math.isfinite(value) for value in (*initial, *final, maximum, reached))
            or math.hypot(*initial[:2]) >= radius
            or row["phase_rad"] != math.radians(phase) or row["voltage_V"] != config["voltage_V"][model]
            or not final_radius - 1e-8 <= maximum <= radius + 1e-8):
        raise ValueError(f"Invalid V11 trajectory row: {row}")
    if row["status"] == "retained" and reached == end and escape is None and final_radius < radius:
        return prior_key(config["mesh_um"], model, phase, initial)
    if row["status"] == "escaped" and reached == escape and 0 < escape <= end and abs(final_radius - radius) <= 1e-8:
        return prior_key(config["mesh_um"], model, phase, initial)
    raise ValueError(f"Invalid V11 trajectory status or elapsed RF: {row}")


def _load_source(out, source, config, archived, expected_count, radius):
    folder = out / source
    horizon = config["horizon_RF"]
    segment = config.get("segment_RF", 100)
    checkpoints = dict(archived)
    for path in (folder / "checkpoints").glob("*/rf_*.json"):
        name = PurePosixPath(path.relative_to(out).as_posix())
        row = load_json(path)
        if name in checkpoints and checkpoints[name] != row:
            raise ValueError(f"V11 loose checkpoint differs from verified archive: {path}")
        checkpoints[name] = row

    histories = defaultdict(dict)
    for name, raw in checkpoints.items():
        row = _typed_row(raw)
        if row["status"] == "numerical_error":
            continue
        end = int(name.stem.removeprefix("rf_"))
        if end <= 0 or end > horizon or end % segment:
            raise ValueError(f"Invalid V11 checkpoint RF boundary: {name}")
        key = _verify_row(row, config, end, radius)
        if end in histories[key]:
            raise ValueError(f"Duplicate V11 checkpoint for state and RF boundary: {name}")
        histories[key][end] = row

    summary = {}
    path = folder / "results.csv"
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row = _typed_row(raw)
                if row["status"] == "numerical_error":
                    continue
                key = _verify_row(row, config, horizon, radius)
                if key in summary:
                    raise ValueError(f"Duplicate V11 result in {path}: {key}")
                summary[key] = row

    if expected_count and len(summary) != expected_count:
        raise ValueError(f"Expected {expected_count} verified V11 results in {path}, got {len(summary)}")
    if expected_count and archived and set(histories) != set(summary):
        raise ValueError(f"V11 archive and completed table state sets differ: {source}")

    for key, history in histories.items():
        sorted_history = sorted(history.items())
        for end, row in sorted_history[:-1]:
            if row["status"] == "escaped":
                raise ValueError(f"V11 trajectory has checkpoints after escape: {source}, {key}, RF {end}")
        end, last = sorted_history[-1]
        if key in summary and (end == horizon or last["status"] == "escaped") and summary[key] != last:
            raise ValueError(f"V11 result and terminal checkpoint differ: {source}, {key}")
        if key not in summary:
            summary[key] = last
    return summary, histories


def _compatible(out, source, config, expected, h35_hashes):
    family = source.split("/", 1)[0]
    if source in BASELINE and config["mesh_um"] == 35:
        return (config["field_sha256"] == h35_hashes and config["voltage_V"] == expected["voltage_V"]
                and config["horizon_RF"] == 1000)
    if family == "boundary_refinement":
        # The legacy runner inherits dynamics from the h50 XY map referenced here.
        # Manifests written on Windows use backslashes.
        parent = Path(config["source_map"].replace("\\", "/"))
        source_path = out / "xy" / parent.name / "config.json"
        if not source_path.is_file():
            return False
        parent_config = load_json(source_path)
        return (all(config.get(field) == expected[field] for field in ("mesh_um", "voltage_V", "field_sha256"))
                and parent_config["phase_deg"] == config["phase_deg"]
                and all(parent_config.get(field) == expected[field] for field in CONTRACT_FIELDS))
    if family == "velocity":
        return (all(config.get(field) == expected[field] for field in CONTRACT_FIELDS[2:-1])
                and all(config.get(field) == value for field, value in expected["solver"].items()))
    return all(config.get(field) == expected[field] for field in CONTRACT_FIELDS)


def _verify_baseline_states(source, config, rows):
    positions = config.get("positions")
    if positions is None:
        half = (config["grid"] - 1) // 2
        step = 0.0075 / half
        positions = [(i * step, j * step) for i in range(-half, half + 1)
                     for j in range(-half, half + 1) if i*i + j*j < half*half]
    expected = {prior_key(config["mesh_um"], model, config["phase_deg"], (x, y, 0.0, 0.0))
                for x, y in positions for model in "ABC"}
    if rows.keys() != expected:
        raise ValueError(f"Unexpected physical state set in V11 baseline: {source}")


def load_prior(out, phase_configs, *, dry_run=False):
    """Return reusable rows and segment histories keyed independently of horizon.

    phase_configs maps phase degrees to full h50 XY configs. All reads are safe
    during dry-run; nothing is extracted, integrated, or written.
    """
    from run_v11_xy import fields

    out = Path(out)
    prior = PriorResults()
    archives = _read_archives(out, prior.audit["archives"])
    _, h35_hashes = fields(35)
    sources = set()
    for directory in ("xy", "boundary_refinement", "velocity"):
        sources.update(path.parent.relative_to(out).as_posix() for path in (out / directory).glob("*/config.json"))
    for source in (*BASELINE, *sorted(sources - BASELINE.keys() - {PILOT})):
        baseline = BASELINE.get(source)
        path = out / source / "config.json"
        if not path.is_file():
            raise ValueError(f"Missing V11 baseline config: {path}")
        config = load_json(path)
        mesh, phase = config.get("mesh_um"), config.get("phase_deg")
        if baseline and (mesh, phase) != baseline[:2]:
            raise ValueError(f"Unexpected mesh or phase in V11 baseline config: {path}")
        if baseline and mesh == 50 and (config.get("grid"), config.get("horizon_RF"), config.get("segment_RF")) != (21, 1000, 100):
            raise ValueError(f"Unexpected grid, horizon or checkpoint spacing in V11 baseline config: {path}")
        expected = phase_configs.get(phase)
        if expected is None:
            continue

        if not _compatible(out, source, config, expected, h35_hashes):
            if baseline:
                raise ValueError(f"Incompatible V11 baseline config: {path}")
            continue

        rows, histories = _load_source(out, source, config, archives[source], baseline[2] if baseline else None,
                                       expected["aperture_m"])
        if baseline:
            _verify_baseline_states(source, config, rows)
        for key, row in rows.items():
            if key not in prior:
                prior[key] = row
                prior.segments[key] = tuple(sorted(histories[key].items()))
                continue
            saved = prior[key]
            if saved["status"] == "escaped" or row["status"] == "escaped":
                if saved["status"] == row["status"]:
                    continue
                escaped, other = (saved, row) if saved["status"] == "escaped" else (row, saved)
                if other["reached_RF"] >= escaped["reached_RF"]:
                    raise ValueError(f"Conflicting V11 escape histories: {source}, {key}")
            if row["reached_RF"] > saved["reached_RF"]:
                prior[key] = row
                prior.segments[key] = tuple(sorted({**dict(prior.segments[key]), **histories[key]}.items()))
        prior.audit["sources"].append({"path": source, "mesh_um": mesh, "phase_deg": phase,
                                       "verified_states": len(rows), "segment_rows": sum(map(len, histories.values()))})
    return prior
