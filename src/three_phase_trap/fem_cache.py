"""Fingerprint-validated, restart-safe persistence for the P2 FEM basis."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import skfem
from skfem import Basis, ElementTriP2, MeshTri

from .config import ProjectConfig
from .fem import FEMBasisSolutions, solve_basis
from .fem import element_electric_field, solve_instantaneous
from .geometry import build_geometry
from .interpolation import FEMInterpolator
from .mesh import TrapMesh, build_mesh
from .voltages import phase_voltages


CACHE_FORMAT_VERSION = 1
CACHE_DIRECTORY = Path("outputs/fem_cache")
ARRAYS_NAME = "p2_basis.npz"
METADATA_NAME = "metadata.json"
PHASES = ("A", "B", "C", "DC")


class FEMCacheError(RuntimeError):
    """A present FEM cache is invalid or does not match the requested model."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _solver_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    result = {}
    for name in ("fem.py", "mesh.py", "geometry.py"):
        result[name] = hashlib.sha256((package / name).read_bytes()).hexdigest()
    return result


def fem_fingerprint_payload(config: ProjectConfig) -> dict[str, object]:
    """Return only geometry/FEM inputs that determine the cached basis."""
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "geometry": asdict(config.geometry),
        "mesh": asdict(config.mesh),
        "element": {"family": "triangle", "type": "ElementTriP2", "polynomial_order": 2},
        "laplace_form": "integral(dot(grad(u), grad(v)))",
        "outer_boundary": {"type": "Dirichlet", "voltage_V": 0.0, "sampled_circle": True},
        "basis_definitions": {
            "A": {"electrodes_modulo_3": 0, "voltage_V": 1.0, "other_electrodes_V": 0.0},
            "B": {"electrodes_modulo_3": 1, "voltage_V": 1.0, "other_electrodes_V": 0.0},
            "C": {"electrodes_modulo_3": 2, "voltage_V": 1.0, "other_electrodes_V": 0.0},
            "DC": {"all_electrodes_V": 1.0},
        },
        "phase_voltage_law": {"A": "sin(Omega*t)", "B": "sin(Omega*t-2*pi/3)", "C": "sin(Omega*t+2*pi/3)"},
        "solver_source_sha256": _solver_source_hashes(),
        "scikit_fem_version": getattr(skfem, "__version__", "unknown"),
    }


def fem_fingerprint(config: ProjectConfig) -> str:
    return hashlib.sha256(_canonical_json(fem_fingerprint_payload(config))).hexdigest()


def cache_paths(project_root: Path, config: ProjectConfig | None = None) -> tuple[Path, Path]:
    directory = Path(project_root) / CACHE_DIRECTORY
    if config is not None:
        main_metadata = directory / METADATA_NAME
        if main_metadata.is_file():
            try:
                saved_fingerprint = json.loads(main_metadata.read_text(encoding="utf-8")).get("fingerprint")
            except (OSError, json.JSONDecodeError):
                saved_fingerprint = None
            current_fingerprint = fem_fingerprint(config)
            if saved_fingerprint is not None and saved_fingerprint != current_fingerprint:
                directory = directory / "variants" / current_fingerprint[:20]
    return directory / ARRAYS_NAME, directory / METADATA_NAME


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _arrays_for(solutions: FEMBasisSolutions) -> dict[str, np.ndarray]:
    trap_mesh = solutions.trap_mesh
    arrays: dict[str, np.ndarray] = {
        "mesh_points": np.asarray(trap_mesh.points),
        "mesh_triangles": np.asarray(trap_mesh.triangles),
        "outer_nodes": np.asarray(trap_mesh.outer_nodes, dtype=np.int64),
        "dof_coordinates": np.asarray(solutions.basis.doflocs),
        "element_dofs": np.asarray(solutions.basis.element_dofs),
    }
    for index in sorted(trap_mesh.electrode_nodes):
        arrays[f"electrode_nodes_{index}"] = np.asarray(trap_mesh.electrode_nodes[index], dtype=np.int64)
    for phase in ("A", "B", "C"):
        arrays[f"phase_nodes_{phase}"] = np.asarray(trap_mesh.phase_nodes[phase], dtype=np.int64)
    for phase in PHASES:
        arrays[f"potential_{phase}"] = np.asarray(solutions.potentials[phase])
        arrays[f"field_{phase}"] = np.asarray(solutions.fields[phase])
    return arrays


def save_fem_cache(config: ProjectConfig, project_root: Path, solutions: FEMBasisSolutions) -> tuple[Path, Path]:
    """Persist the full mesh and all four solved P2 bases with atomic replacement."""
    project_root = Path(project_root)
    arrays_path, metadata_path = cache_paths(project_root, config)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _arrays_for(solutions)
    fd, temporary_name = tempfile.mkstemp(prefix="p2_basis_", suffix=".npz.tmp", dir=arrays_path.parent)
    try:
        with open(fd, "wb", closefd=True) as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        temporary_arrays = Path(temporary_name)
        temporary_arrays.replace(arrays_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    payload = fem_fingerprint_payload(config)
    metadata = {
        "fingerprint": fem_fingerprint(config),
        "fingerprint_payload": payload,
        "geometry_parameters": asdict(config.geometry),
        "mesh_parameters": asdict(config.mesh),
        "element_type": "ElementTriP2",
        "polynomial_order": 2,
        "basis_definitions": payload["basis_definitions"],
        "phase_voltage_law": payload["phase_voltage_law"],
        "outer_boundary_condition": payload["outer_boundary"],
        "mesh_vertices": int(len(solutions.points)),
        "mesh_triangles": int(len(solutions.triangles)),
        "p2_dofs": int(solutions.basis.N),
        "arrays_file": arrays_path.name,
        "arrays_sha256": _sha256_file(arrays_path),
        "git_commit": _git_commit(project_root),
        "code_version": "repository source hashes; see fingerprint_payload.solver_source_sha256",
        "scikit_fem_version": payload["scikit_fem_version"],
    }
    fd, temporary_name = tempfile.mkstemp(prefix="metadata_", suffix=".json.tmp", dir=metadata_path.parent)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())
        Path(temporary_name).replace(metadata_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return arrays_path, metadata_path


def load_fem_cache(config: ProjectConfig, project_root: Path) -> tuple[object, TrapMesh, FEMBasisSolutions, FEMInterpolator]:
    """Load only an intact cache whose exact FEM fingerprint matches config."""
    arrays_path, metadata_path = cache_paths(project_root, config)
    if not arrays_path.exists() and not metadata_path.exists():
        raise FileNotFoundError(arrays_path)
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise FEMCacheError(f"Incomplete P2 FEM cache under {arrays_path.parent}; refusing an implicit FEM rebuild.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FEMCacheError(f"Could not read FEM cache metadata: {error}") from error
    expected = fem_fingerprint(config)
    if metadata.get("fingerprint") != expected:
        raise FEMCacheError(
            "The cached P2 basis does not match the requested geometry/FEM configuration. "
            "No FEM rebuild was started. Review the configuration and explicitly authorize a new cache if intended."
        )
    if metadata.get("arrays_sha256") != _sha256_file(arrays_path):
        raise FEMCacheError("P2 FEM cache checksum mismatch; refusing to load or rebuild it implicitly.")
    try:
        with np.load(arrays_path, allow_pickle=False) as data:
            points = data["mesh_points"].copy()
            triangles = data["mesh_triangles"].copy()
            electrode_nodes = {i: data[f"electrode_nodes_{i}"].copy() for i in range(12)}
            phase_nodes = {phase: data[f"phase_nodes_{phase}"].copy() for phase in ("A", "B", "C")}
            outer_nodes = data["outer_nodes"].copy()
            potentials = {phase: data[f"potential_{phase}"].copy() for phase in PHASES}
            fields = {phase: data[f"field_{phase}"].copy() for phase in PHASES}
            dof_coordinates = data["dof_coordinates"].copy()
            element_dofs = data["element_dofs"].copy()
    except (OSError, KeyError, ValueError) as error:
        raise FEMCacheError(f"P2 FEM cache arrays are unreadable or incomplete: {error}") from error
    geometry = build_geometry(
        config.geometry.electrode_count, config.geometry.electrode_orbit_radius,
        config.geometry.electrode_radius, config.geometry.outer_domain_radius,
    )
    mesh = MeshTri(points.T, triangles.T, sort_t=False)
    trap_mesh = TrapMesh(mesh, electrode_nodes, phase_nodes, outer_nodes)
    basis = Basis(mesh, ElementTriP2())
    if not np.array_equal(basis.doflocs, dof_coordinates) or not np.array_equal(basis.element_dofs, element_dofs):
        raise FEMCacheError("Reconstructed P2 DOF ordering differs from the serialized cache; refusing an unsafe load.")
    solutions = FEMBasisSolutions(trap_mesh, basis, potentials, fields)
    if int(metadata.get("mesh_vertices", -1)) != len(points) or int(metadata.get("mesh_triangles", -1)) != len(triangles) or int(metadata.get("p2_dofs", -1)) != basis.N:
        raise FEMCacheError("P2 FEM cache dimensions do not match its metadata.")
    return geometry, trap_mesh, solutions, FEMInterpolator(solutions)


def load_or_build_fem_state(config: ProjectConfig, project_root: Path) -> tuple[object, TrapMesh, FEMBasisSolutions, FEMInterpolator]:
    """Load an exact cache, or perform the initial solve once and round-trip it."""
    try:
        return load_fem_cache(config, project_root)
    except FileNotFoundError:
        geometry = build_geometry(
            config.geometry.electrode_count, config.geometry.electrode_orbit_radius,
            config.geometry.electrode_radius, config.geometry.outer_domain_radius,
        )
        trap_mesh = build_mesh(
            geometry, config.mesh.resolution, config.mesh.boundary_points_per_electrode,
            config.mesh.outer_boundary_points, config.mesh.interior_seed,
        )
        solutions = solve_basis(trap_mesh)
        save_fem_cache(config, project_root, solutions)
        state = load_fem_cache(config, project_root)
        validate_cached_basis(config, project_root, state)
        return state


def validate_cached_basis(config: ProjectConfig, project_root: Path, state: tuple[object, TrapMesh, FEMBasisSolutions, FEMInterpolator]) -> dict[str, float | int]:
    """Independently direct-solve RF+DC boundary cases against cached bases."""
    import csv

    _, trap_mesh, solutions, _ = state
    output_dir = Path(project_root) / CACHE_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    biases = (100.0, 500.0, 1000.0)
    phase_fractions = tuple(np.arange(6, dtype=float) / 6.0)
    rows = []
    for Udc in biases:
        for phase_fraction in phase_fractions:
            voltages = phase_voltages(float(phase_fraction / config.rf.frequency), config.rf.U0, config.rf.frequency)
            direct_voltages = {phase: float(value) + Udc for phase, value in voltages.items()}
            direct_potential = solve_instantaneous(trap_mesh, direct_voltages)
            basis_potential = Udc * solutions.potentials["DC"] + sum(float(voltages[p]) * solutions.potentials[p] for p in ("A", "B", "C"))
            potential_error = direct_potential - basis_potential
            direct_field = element_electric_field(trap_mesh, direct_potential, basis=solutions.basis)
            basis_field = Udc * solutions.fields["DC"] + sum(float(voltages[p]) * solutions.fields[p] for p in ("A", "B", "C"))
            field_error = np.linalg.norm(direct_field - basis_field, axis=1)
            rows.append({
                "Udc_V": Udc,
                "phase_fraction_T": phase_fraction,
                "max_potential_error_V": float(np.max(np.abs(potential_error))),
                "rms_potential_error_V": float(np.sqrt(np.mean(potential_error**2))),
                "max_electric_field_error_V_per_m": float(np.max(field_error)),
                "rms_electric_field_error_V_per_m": float(np.sqrt(np.mean(field_error**2))),
            })
    csv_path = output_dir / "direct_fem_cache_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "cases": len(rows),
        "max_potential_error_V": max(row["max_potential_error_V"] for row in rows),
        "rms_potential_error_V": float(np.sqrt(np.mean([row["rms_potential_error_V"] ** 2 for row in rows]))),
        "max_electric_field_error_V_per_m": max(row["max_electric_field_error_V_per_m"] for row in rows),
        "rms_electric_field_error_V_per_m": float(np.sqrt(np.mean([row["rms_electric_field_error_V_per_m"] ** 2 for row in rows]))),
        "previous_validated": {
            "max_potential_error_V": 5.6843418860808015e-12,
            "rms_potential_error_V": 2.1286732816352618e-13,
            "max_electric_field_error_V_per_m": 4.9585001478266494e-08,
            "rms_electric_field_error_V_per_m": 8.354675680215251e-10,
        },
        "cache_fingerprint": fem_fingerprint(config),
    }
    (output_dir / "direct_fem_cache_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
