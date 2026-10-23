"""Verify and package an existing Gmsh FEM run; never solve/rebuild a basis.

Run from the repository root after the explicit ``cli fem`` command:
    python scripts/archive_fem_run.py --config configs/gmsh_70um.toml
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from skfem import asm

from three_phase_trap.config import load_config
from three_phase_trap.fem import laplace_form
from three_phase_trap.fem_cache import CACHE_DIRECTORY, cache_paths, fem_fingerprint, load_fem_cache
from three_phase_trap.voltages import phase_voltages


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def git(*arguments: str) -> str:
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments],
                          cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_config(config_path)
    arrays_path, metadata_path = cache_paths(ROOT, config)
    directory = arrays_path.parent
    direct_cache_directory = ROOT / CACHE_DIRECTORY
    summary = json.loads((direct_cache_directory/"direct_fem_cache_validation_summary.json").read_text(encoding="utf-8"))
    if summary["cache_fingerprint"] != fem_fingerprint(config):
        raise ValueError("Direct-solve validation belongs to a different FEM configuration.")
    print("Loading checksum-validated basis...", flush=True)
    _, mesh, solutions, interpolator = load_fem_cache(config, ROOT)
    basis = solutions.basis
    print("Assembling Laplace matrix for residual checks (no linear solve)...", flush=True)
    stiffness = asm(laplace_form, basis).tocsr()
    groups = {"outer": mesh.outer_nodes, **{f"electrode_{i}": nodes for i,nodes in mesh.electrode_nodes.items()}}
    dofs = {}
    for name,nodes in groups.items():
        mask = np.zeros(mesh.mesh.nvertices, dtype=bool)
        mask[nodes] = True
        facets = np.flatnonzero(np.all(mask[mesh.mesh.facets], axis=0))
        dofs[name] = np.unique(np.concatenate((basis.nodal_dofs[0,nodes], basis.facet_dofs[:,facets].ravel())))
    boundary = np.unique(np.concatenate(list(dofs.values())))
    interior = np.setdiff1d(np.arange(basis.N), boundary)
    results = {}
    for phase in ("A","B","C","DC"):
        potential = solutions.potentials[phase]
        if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(solutions.fields[phase])):
            raise RuntimeError(f"Non-finite basis {phase}")
        expected = np.zeros(basis.N)
        for i in range(12):
            expected[dofs[f"electrode_{i}"]] = float(phase == "DC" or ("A","B","C")[i%3] == phase)
        residual = np.asarray(stiffness @ potential)[interior]
        scale = np.asarray(abs(stiffness) @ abs(potential))[interior]
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(scale),1e-300))
        boundary_error = float(np.max(abs(potential[boundary]-expected[boundary])))
        indices = np.arange(0,len(mesh.triangles),173)
        points = mesh.points[mesh.triangles[indices]].mean(axis=1)
        evaluated = interpolator.field_basis(points,phase)
        gradient_error = float(np.max(np.abs(evaluated-solutions.fields[phase][indices])))
        assert relative < 1e-11, (phase,"residual",relative)
        assert boundary_error < 1e-12, (phase,"boundary",boundary_error)
        assert gradient_error < 1e-8, (phase,"gradient",gradient_error)
        results[phase] = {"potential_min_V":float(potential.min()), "potential_max_V":float(potential.max()),
                          "boundary_max_error_V":boundary_error,
                          "relative_weak_Laplace_residual_l2":relative,
                          "centroid_field_vs_interpolator_max_error_V_per_m":gradient_error}
    dc_error = float(np.max(abs(solutions.potentials["DC"]-sum(solutions.potentials[p] for p in ("A","B","C")))))
    assert dc_error < 1e-10, dc_error
    # Normalize direct-solve discrepancies by the RMS of the corresponding
    # reconstructed field/potential. Gram matrices avoid allocating every
    # full combined field again for each of the validation cases.
    phases = ("A","B","C","DC")
    phi = np.stack([solutions.potentials[p] for p in phases])
    field = np.stack([solutions.fields[p].ravel() for p in phases])
    gram_phi = phi @ phi.T / phi.shape[1]
    gram_field = field @ field.T / len(mesh.triangles)
    superposition = []
    for file, time_column, field_error_column in (
        (direct_cache_directory/"direct_fem_cache_validation.csv","phase_fraction_T","rms_electric_field_error_V_per_m"),
        (config.output_dir(ROOT)/"direct_fem_validation.csv","t_over_T","rms_field_error_V_per_m"),
    ):
        with file.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                fraction = float(row[time_column])
                voltages = phase_voltages(fraction/config.rf.frequency,config.rf.U0,config.rf.frequency)
                weights = np.array([float(voltages[p]) for p in phases[:3]]+[float(row.get("Udc_V",0))])
                relative_phi = float(row["rms_potential_error_V"])/np.sqrt(weights@gram_phi@weights)
                relative_field = float(row[field_error_column])/np.sqrt(weights@gram_field@weights)
                assert relative_phi < 1e-9 and relative_field < 1e-9, (file,row)
                superposition.append({"source":file.name,"fraction_T":fraction,"Udc_V":float(weights[3]),
                                      "relative_rms_potential_error":float(relative_phi),
                                      "relative_rms_field_error":float(relative_field)})
    # Also compare loaded arrays to the NPZ, not just dimensions or checksums.
    with np.load(arrays_path, allow_pickle=False) as saved:
        assert np.array_equal(saved["mesh_points"],solutions.points)
        assert np.array_equal(saved["mesh_triangles"],solutions.triangles)
        assert np.array_equal(saved["dof_coordinates"],basis.doflocs)
        assert np.array_equal(saved["element_dofs"],basis.element_dofs)
        for phase in ("A","B","C","DC"):
            assert np.array_equal(saved[f"potential_{phase}"],solutions.potentials[phase])
            assert np.array_equal(saved[f"field_{phase}"],solutions.fields[phase])
    validation_path = directory / "basis_integrity_checks.json"
    write_json(validation_path,{"round_trip_arrays_equal":True,"laplace_solved":True,
                               "boundary_dofs":len(boundary),"interior_dofs":len(interior),
                               "dc_minus_abc_max_error_V":dc_error,"bases":results,
                               "direct_superposition_cases":superposition,
                               "scope":"Discrete-system residual, boundary conditions, field reconstruction and serialization; not spatial convergence or trajectory validation."})
    shutil.copy2(config_path,directory / "calculation_config.toml")
    source_paths = sorted([p for folder in ("src","configs","scripts","tests","docs")
                           for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in (".py",".toml",".md")]
                          + [ROOT / name for name in ("README.md","pyproject.toml",".gitignore")])
    source_zip = directory / "source_snapshot.zip"
    with zipfile.ZipFile(source_zip,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path,path.relative_to(ROOT).as_posix())
    mesh_dir = (ROOT/config.mesh.mesh_file).parent
    validation_files = [direct_cache_directory/name for name in ("direct_fem_cache_validation.csv","direct_fem_cache_validation_summary.json")]
    fem_output = config.output_dir(ROOT)
    validation_files += [fem_output/name for name in ("direct_fem_validation.csv","fem_summary.json")]
    for path in validation_files:
        if not path.is_file():
            raise FileNotFoundError(f"Complete cli fem validation before archiving: {path}")
    data_paths = [arrays_path,metadata_path,validation_path,directory/"calculation_config.toml",source_zip,
                  *validation_files,*sorted(p for p in mesh_dir.iterdir() if p.is_file())]
    report_path = directory/"FEM_RUN_REPORT.md"
    lines = ["# Gmsh 70 micrometre FEM basis", "", "Solved A, B, C and common DC at 1 V on the saved mesh.",
             "",f"Vertices: {len(mesh.points)}; triangles: {len(mesh.triangles)}; P2 DOFs: {int(basis.N)}.",
             "", "All cached arrays round-trip exactly. Electrode and outer boundary conditions and discrete Laplace residuals pass.",
             f"Direct-solve superposition checks: {len(superposition)}; maximum relative RMS field discrepancy: {max(v['relative_rms_field_error'] for v in superposition):.6e}.",
             "", "These checks do not establish mesh convergence or the validity of the effective-potential trajectory approximation.",
             "", "Portable archive: `gmsh_70um_fem_bundle.zip`. Verify using `bundle_checksum.json`.",
             "The archive includes the exact source snapshot, configuration, mesh and basis files.",
             "", "Inspect `run_manifest.json` for library versions, source hashes, Git revision and uncommitted-change status."]
    context = ROOT / "docs/FEM_GMSH_70UM.md"
    if context.is_file():
        lines += ["", "---", "", context.read_text(encoding="utf-8")]
    report_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    data_paths.append(report_path)
    manifest = {"created_utc":datetime.now(timezone.utc).isoformat(),"git_head":git("rev-parse","HEAD"),
                "working_tree_status":git("status","--short"),"python":platform.python_version(),
                "resolved_configuration":asdict(config),
                "basis_units":{"applied_boundary_voltage_V":1.0,"coordinates":"m","potentials":"V","fields":"V/m"},
                "versions":{name:importlib.metadata.version(name) for name in ("numpy","scipy","scikit-fem","gmsh","meshio","matplotlib")},
                "reproduce":"python -m three_phase_trap.cli fem --config configs/gmsh_70um.toml",
                "source_files_sha256":{p.relative_to(ROOT).as_posix():checksum(p) for p in source_paths},
                "data_files":{p.relative_to(ROOT).as_posix():{"bytes":p.stat().st_size,"sha256":checksum(p)} for p in data_paths}}
    manifest_path = directory / "run_manifest.json"
    write_json(manifest_path,manifest)
    bundle = directory / "gmsh_70um_fem_bundle.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*data_paths,manifest_path]:
            archive.write(path,path.relative_to(ROOT).as_posix())
        # Include source files directly for convenient standalone extraction.
        for path in source_paths:
            archive.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
    write_json(directory / "bundle_checksum.json",{"file":bundle.name,"bytes":bundle.stat().st_size,"sha256":checksum(bundle)})
    print(json.dumps({"basis_integrity":results,"bundle":str(bundle),"bundle_bytes":bundle.stat().st_size}),flush=True)


if __name__ == "__main__":
    main()
