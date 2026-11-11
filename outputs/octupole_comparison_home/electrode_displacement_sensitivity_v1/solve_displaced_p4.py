"""Generate a full asymmetric trap mesh and solve one displaced P4 RF field.

This deliberately does not use cyclic symmetry or N48.  The CLI starts a single
120-minute wall-clock budget on the first numerical geometry run and checkpoints
the mesh and each solver stage so a resource failure is auditable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OUT / ".deps"))
GMSH_LIB = OUT / ".deps" / "gmsh_sdk" / "gmsh-4.15.2-Windows64-sdk" / "lib"
DLL_DIRECTORY_HANDLE = os.add_dll_directory(str(GMSH_LIB)) if GMSH_LIB.is_dir() and hasattr(os, "add_dll_directory") else None
DLL_DEPENDENCY_HANDLE = os.add_dll_directory(str(OUT / ".deps")) if (OUT / ".deps").is_dir() and hasattr(os, "add_dll_directory") else None
CONFIG_PATH = OUT / "pilot_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RHO = complex(-0.5, -math.sqrt(3.0) / 2.0)
PHASE_VALUES = np.array([1.0 + 0.0j, RHO, RHO * RHO], dtype=complex)
COMPUTE_BUDGET_S = 7200


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def runtime_path() -> Path:
    p = OUT / "runtime" / "compute_budget.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def start_or_resume_budget() -> dict:
    path = runtime_path()
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        start = utc_now()
        state = {
            "start_utc": iso(start),
            "hard_deadline_utc": iso(start + timedelta(seconds=COMPUTE_BUDGET_S)),
            "maximum_new_numerical_wall_seconds": COMPUTE_BUDGET_S,
            "scope": "all newly generated displaced meshes, FEM solves, field diagnostics, and new RF trajectories in this pilot",
            "first_operation": "full-domain asymmetric h35 P4 FEM benchmark",
        }
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    return state


def remaining_seconds(state: dict) -> float:
    end = datetime.fromisoformat(state["hard_deadline_utc"].replace("Z", "+00:00"))
    return (end - utc_now()).total_seconds()


def require_time(state: dict, reserve_s: float, label: str) -> None:
    left = remaining_seconds(state)
    if left < reserve_s:
        raise TimeoutError(f"Refusing to start {label}: only {left:.1f} seconds remain before the hard deadline.")


def append_checkpoint(record: dict) -> None:
    path = OUT / "runtime" / "geometry_checkpoints.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def centers_with_pattern(pattern_name: str) -> tuple[np.ndarray, np.ndarray]:
    patterns = CONFIG["initial_patterns"]
    if pattern_name not in patterns:
        raise KeyError(f"Unknown pattern {pattern_name!r}; available: {', '.join(patterns)}")
    item = patterns[pattern_name]
    geom = CONFIG["geometry"]
    n = int(geom["electrode_count"])
    radius = float(geom["orbit_radius_m"])
    theta = np.arange(n, dtype=float) * (2.0 * math.pi / n)
    radial = np.column_stack((np.cos(theta), np.sin(theta)))
    tangent = np.column_stack((-np.sin(theta), np.cos(theta)))
    delta = np.zeros((n, 2), dtype=float)

    for key, value in item.items():
        if key.startswith("electrode_") and key.endswith("_radial_m"):
            idx = int(key.split("_")[1])
            delta[idx] += float(value) * radial[idx]
        elif key.startswith("electrode_") and key.endswith("_tangential_m"):
            idx = int(key.split("_")[1])
            delta[idx] += float(value) * tangent[idx]
        elif key == "all_electrodes_radial_m":
            delta += float(value) * radial
        elif key == "random_radial_seed":
            seed = int(value)
            limit = float(item["random_max_abs_m"])
            radial_offsets = np.random.default_rng(seed).uniform(-limit, limit, n)
            delta += radial_offsets[:, None] * radial
    nominal = radius * radial
    shifted = nominal + delta
    if np.min(np.linalg.norm(np.roll(shifted, -1, axis=0) - shifted, axis=1)) <= 2 * float(geom["electrode_radius_m"]):
        raise ValueError("Displaced electrode circles would overlap.")
    if np.max(np.linalg.norm(shifted, axis=1)) + float(geom["electrode_radius_m"]) >= float(geom["outer_domain_radius_m"]):
        raise ValueError("Displaced electrode would intersect the outer boundary.")
    return shifted, delta


def generate_mesh(pattern_name: str, mesh_dir: Path, state: dict, resolution_level: str = "h35") -> dict:
    require_time(state, 60.0, f"Gmsh mesh generation for {pattern_name}")
    import gmsh

    geom = CONFIG["geometry"]
    meshcfg = dict(CONFIG["mesh"])
    meshcfg["resolution_m"] = {"h35": 35e-6, "h50": 50e-6}[resolution_level]
    centers, offsets = centers_with_pattern(pattern_name)
    a = float(geom["electrode_radius_m"])
    R = float(geom["outer_domain_radius_m"])
    h = float(meshcfg["resolution_m"])
    ho = float(meshcfg["outer_resolution_m"])
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / f"displaced_{resolution_level}.msh"
    recipe = {
        "pattern": pattern_name,
        "resolution_level": resolution_level,
        "centers_m": centers.tolist(),
        "offset_vectors_m": offsets.tolist(),
        "electrode_radius_m": a,
        "outer_domain_radius_m": R,
        "mesh": meshcfg,
        "full_domain_without_periodic_constraints": True,
    }
    atomic_json(mesh_dir / "geometry_recipe.json", recipe)
    append_checkpoint({"time_utc": iso(utc_now()), "pattern": pattern_name, "stage": "mesh_started"})

    if gmsh.isInitialized():
        raise RuntimeError("Gmsh was already initialized. Run a geometry solve in a fresh process.")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", int(meshcfg["gmsh_seed"]))
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", ho)
        gmsh.model.add(f"displaced_{pattern_name}")
        occ = gmsh.model.occ
        outer_disk = occ.addDisk(0.0, 0.0, 0.0, R, R)
        holes = [occ.addDisk(float(x), float(y), 0.0, a, a) for x, y in centers]
        cut, _ = occ.cut([(2, outer_disk)], [(2, tag) for tag in holes], removeObject=True, removeTool=True)
        occ.synchronize()
        if len(cut) != 1 or cut[0][0] != 2:
            raise ValueError(f"Expected one connected vacuum surface after hole subtraction; got {cut}")
        surface = int(cut[0][1])
        boundary = [int(tag) for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False, recursive=False) if dim == 1]
        groups: dict[int, list[int]] = {i: [] for i in range(12)}
        outer_curves = []
        for curve in boundary:
            cx, cy, _ = occ.getCenterOfMass(1, curve)
            cxy = np.array([cx, cy])
            distances = np.linalg.norm(centers - cxy[None, :], axis=1)
            near = int(np.argmin(distances))
            if distances[near] < 0.25 * a:
                groups[near].append(curve)
            else:
                outer_curves.append(curve)
        if any(not groups[i] for i in range(12)) or not outer_curves:
            raise ValueError(f"Failed to classify all circular boundaries: {groups}; outer={outer_curves}")
        gmsh.model.addPhysicalGroup(2, [surface], 100)
        gmsh.model.setPhysicalName(2, 100, "vacuum")
        gmsh.model.addPhysicalGroup(1, outer_curves, 1)
        gmsh.model.setPhysicalName(1, 1, "outer")
        for i, curves in groups.items():
            gmsh.model.addPhysicalGroup(1, curves, 10 + i)
            gmsh.model.setPhysicalName(1, 10 + i, f"electrode_{i}")

        field = gmsh.model.mesh.field
        field.add("MathEval", 1)
        field.setString(1, "F", f"{h:.17g}+({ho:.17g}-{h:.17g})*Min(1,Max(0,(Sqrt(x*x+y*y)-{meshcfg['fine_radius_m']:.17g})/({meshcfg['transition_radius_m']:.17g}-{meshcfg['fine_radius_m']:.17g})))")
        field.add("Distance", 2)
        field.setNumbers(2, "CurvesList", [curve for vals in groups.values() for curve in vals])
        field.setNumber(2, "Sampling", 400)
        field.add("Threshold", 3)
        field.setNumber(3, "InField", 2)
        field.setNumber(3, "SizeMin", h)
        field.setNumber(3, "SizeMax", ho)
        field.setNumber(3, "DistMin", 2 * h)
        field.setNumber(3, "DistMax", 0.002)
        field.add("Min", 4)
        field.setNumbers(4, "FieldsList", [1, 3])
        field.setAsBackgroundMesh(4)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_path))
        gmsh_version = gmsh.__version__
    finally:
        gmsh.finalize()
    metadata = validate_mesh(mesh_path, recipe)
    metadata.update({"gmsh_version": gmsh_version, "mesh_sha256": sha256(mesh_path)})
    atomic_json(mesh_dir / "mesh_diagnostics.json", metadata)
    append_checkpoint({"time_utc": iso(utc_now()), "pattern": pattern_name, "stage": "mesh_validated", "vertices": metadata["vertices"], "triangles": metadata["triangles"], "diagnostics": metadata})
    return metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mesh(path: Path, recipe: dict) -> dict:
    import meshio
    from skfem import MeshTri

    data = meshio.read(path)
    triangles_raw = data.get_cells_type("triangle")
    lines_raw = data.get_cells_type("line")
    tags = data.get_cell_data("gmsh:physical", "line")
    used = np.unique(triangles_raw)
    remap = np.full(len(data.points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = np.asarray(data.points[used, :2], dtype=float)
    triangles = remap[triangles_raw]
    lines = remap[lines_raw]
    mesh = MeshTri(points.T, triangles.T, sort_t=True)
    expected_edges = np.sort(mesh.facets[:, mesh.boundary_facets()].T, axis=1)
    actual_edges = np.sort(lines, axis=1)
    if set(map(tuple, expected_edges)) != set(map(tuple, actual_edges)) or len(expected_edges) != len(actual_edges):
        raise ValueError("Tagged physical lines do not exactly cover the full mesh boundary.")
    field_data = {name: np.asarray(value).tolist() for name, value in data.field_data.items()}
    for i in range(12):
        if f"electrode_{i}" not in field_data or field_data[f"electrode_{i}"][0] != 10 + i:
            raise ValueError(f"Missing physical boundary group for electrode {i}.")
    if "outer" not in field_data or "vacuum" not in field_data:
        raise ValueError("Missing outer or vacuum physical groups.")
    facets_lookup = {tuple(v): i for i, v in enumerate(np.sort(mesh.facets.T, axis=1))}
    line_facets = np.array([facets_lookup[tuple(edge)] for edge in np.sort(lines, axis=1)])
    line_tags = np.asarray(tags)
    tagged_counts = {str(tag): int(np.sum(line_tags == tag)) for tag in [1, *range(10, 22)]}
    if any(tagged_counts[str(10 + i)] == 0 for i in range(12)) or tagged_counts["1"] == 0:
        raise ValueError("One or more tagged electrode/outer boundaries have no line elements.")

    tri_xy = points[triangles]
    edges = tri_xy[:, [1, 2, 0], :] - tri_xy[:, [0, 1, 2], :]
    first = edges[:, 0]
    second = -edges[:, 2]
    twice_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    area = 0.5 * np.abs(twice_area)
    sides = np.stack((np.linalg.norm(edges[:, 0], axis=1), np.linalg.norm(edges[:, 1], axis=1), np.linalg.norm(edges[:, 2], axis=1)), axis=1)
    quality = 4 * np.sqrt(3) * area / np.sum(sides * sides, axis=1)
    euler = int(len(points) - mesh.nfacets + mesh.nelements)
    expected_area = math.pi * (float(CONFIG["geometry"]["outer_domain_radius_m"])**2 - 12 * float(CONFIG["geometry"]["electrode_radius_m"])**2)
    relative_area_error = float(abs(float(area.sum()) / expected_area - 1.0))
    mids = points[mesh.facets].mean(axis=0)
    edge_lengths = np.linalg.norm(points[mesh.facets[0]] - points[mesh.facets[1]], axis=1)
    local = np.linalg.norm(mids, axis=1) <= float(CONFIG["mesh"]["fine_radius_m"])
    if euler != -11 or relative_area_error > 0.005 or np.min(area) <= 0:
        raise ValueError(f"Mesh topology/area validation failed: Euler={euler}; area error={relative_area_error:g}")
    if len(line_facets) != len(expected_edges):
        raise ValueError("Unexpected boundary facet count mismatch.")
    return {
        "path": str(path.relative_to(ROOT)),
        "vertices": int(len(points)),
        "triangles": int(mesh.nelements),
        "mesh_facets": int(mesh.nfacets),
        "euler_characteristic": euler,
        "area_m2": float(area.sum()),
        "relative_area_error": relative_area_error,
        "minimum_triangle_quality": float(quality.min()),
        "edge_length_all_m": {"min": float(edge_lengths.min()), "median": float(np.median(edge_lengths)), "max": float(edge_lengths.max())},
        "edge_length_working_region_m": {"median": float(np.median(edge_lengths[local])), "p95": float(np.quantile(edge_lengths[local], 0.95)), "max": float(edge_lengths[local].max())},
        "boundary_line_counts_by_physical_tag": tagged_counts,
        "boundary_tag_coverage_exact": True,
        "periodic_symmetry_constraints": False,
        "phase_groups": ["ABCABCABCABC"],
        "offset_vectors_m": recipe["offset_vectors_m"],
        "geometry_order": 1,
        "affine_triangles": True,
    }


def rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def solve_mesh(pattern_name: str, mesh_dir: Path, state: dict, resolution_level: str = "h35") -> dict:
    require_time(state, float(CONFIG["budget"]["deadline_guard_before_new_geometry_solve_s"]), f"the {pattern_name} P4 solve")
    import meshio
    import pyamg
    from scipy.sparse.linalg import cg
    from skfem import Basis, ElementTriP4, MeshTri, asm
    from three_phase_trap.fem import laplace_form

    mesh_path = mesh_dir / f"displaced_{resolution_level}.msh"
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    data = meshio.read(mesh_path)
    triangles_raw = data.get_cells_type("triangle")
    lines_raw = data.get_cells_type("line")
    tags = data.get_cell_data("gmsh:physical", "line")
    used = np.unique(triangles_raw)
    remap = np.full(len(data.points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = np.asarray(data.points[used, :2], dtype=float)
    triangles = remap[triangles_raw]
    lines = remap[lines_raw]
    mesh = MeshTri(points.T, triangles.T, sort_t=True)
    facets_lookup = {tuple(v): i for i, v in enumerate(np.sort(mesh.facets.T, axis=1))}
    line_facets = np.array([facets_lookup[tuple(edge)] for edge in np.sort(lines, axis=1)], dtype=int)
    line_tags = np.asarray(tags)
    boundary_groups = {tag: line_facets[line_tags == tag] for tag in [1, *range(10, 22)]}
    basis = Basis(mesh, ElementTriP4(), intorder=int(CONFIG["mesh"]["intorder"]))
    num_dofs = basis.N
    fixed_parts = [basis.get_dofs(facets=boundary_groups[1]).all()]
    electrode_dofs = {}
    for i in range(12):
        dofs = basis.get_dofs(facets=boundary_groups[10 + i]).all()
        electrode_dofs[i] = np.asarray(dofs, dtype=int)
        fixed_parts.append(dofs)
    fixed = np.unique(np.concatenate(fixed_parts))
    free = np.setdiff1d(np.arange(num_dofs, dtype=int), fixed, assume_unique=True)
    boundary_values = np.zeros(num_dofs, dtype=complex)
    for i, dofs in electrode_dofs.items():
        boundary_values[dofs] = PHASE_VALUES[i % 3]

    print(f"[{pattern_name}] P4 basis: {num_dofs:,} dofs; {len(free):,} free; process RSS {rss_bytes()} bytes", flush=True)
    append_checkpoint({"time_utc": iso(utc_now()), "pattern": pattern_name, "stage": "assembly_started", "dofs": int(num_dofs), "free_dofs": int(len(free)), "rss_bytes": rss_bytes()})
    started = time.perf_counter()
    stiffness = asm(laplace_form, basis).tocsr()
    assembly_seconds = time.perf_counter() - started
    rhs_matrix = -np.column_stack(((stiffness @ boundary_values.real)[free], (stiffness @ boundary_values.imag)[free]))
    matrix_started = time.perf_counter()
    A = stiffness[free][:, free].tocsr()
    matrix_reduction_seconds = time.perf_counter() - matrix_started
    nnz = int(A.nnz)
    append_checkpoint({"time_utc": iso(utc_now()), "pattern": pattern_name, "stage": "matrix_assembled", "dofs": int(num_dofs), "free_dofs": int(len(free)), "nnz": nnz, "assembly_seconds": assembly_seconds, "matrix_reduction_seconds": matrix_reduction_seconds, "rss_bytes": rss_bytes()})
    del stiffness, data

    preconditioner_started = time.perf_counter()
    print(f"[{pattern_name}] assembling AMG hierarchy for {len(free):,} free P4 dofs ({nnz:,} nonzeros)", flush=True)
    ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=500)
    M = ml.aspreconditioner(cycle="V")
    hierarchy_seconds = time.perf_counter() - preconditioner_started
    print(f"[{pattern_name}] AMG levels={len(ml.levels)}, setup={hierarchy_seconds:.1f}s, RSS={rss_bytes()} bytes", flush=True)

    rhs_norms, relative_residuals, scaled_inf_residuals, iteration_counts = [], [], [], []
    solutions = []
    solve_started = time.perf_counter()
    abs_row_sum = np.asarray(np.abs(A).sum(axis=1)).ravel()
    scale = max(float(abs_row_sum.max()), np.finfo(float).tiny)
    for part, rhs in zip(("real", "imag"), rhs_matrix.T):
        require_time(state, 30.0, f"{part} component P4 iterative solve")
        count = [0]
        def callback(_):
            count[0] += 1
            if count[0] % 5 == 0:
                print(f"[{pattern_name}] {part} CG iteration {count[0]}", flush=True)
            if remaining_seconds(state) <= 0:
                raise TimeoutError("The hard computation deadline expired during CG.")
        x, info = cg(A, rhs, M=M, rtol=1e-10, atol=0.0, maxiter=150, callback=callback)
        if info != 0:
            raise RuntimeError(f"P4 conjugate gradients did not converge for {part}: info={info}, iterations={count[0]}")
        residual = A @ x - rhs
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
        scaled = float(np.max(np.abs(residual)) / scale)
        if not np.isfinite(relative) or relative > 1e-9 or scaled > 1e-9:
            raise RuntimeError(f"P4 residual check failed for {part}: relative={relative:.3e}, scaled_inf={scaled:.3e}")
        rhs_norms.append(float(np.linalg.norm(rhs)))
        relative_residuals.append(relative)
        scaled_inf_residuals.append(scaled)
        iteration_counts.append(int(count[0]))
        solutions.append(x)
    solve_seconds = time.perf_counter() - solve_started
    full = boundary_values.copy()
    full[free] = solutions[0] + 1j * solutions[1]
    field_path = mesh_dir / "p4_coefficients.npz"
    np.savez_compressed(field_path, coefficients=full, points=points, triangles=triangles)
    result = {
        "pattern": pattern_name,
        "resolution_level": resolution_level,
        "mesh_path": str(mesh_path.relative_to(ROOT)),
        "field_path": str(field_path.relative_to(ROOT)),
        "mesh_sha256": sha256(mesh_path),
        "field_sha256": sha256(field_path),
        "scikit_fem_version": __import__("skfem").__version__,
        "pyamg_version": pyamg.__version__,
        "degree": 4,
        "full_domain_without_symmetry_reduction": True,
        "dofs": int(num_dofs),
        "free_dofs": int(len(free)),
        "matrix_nonzeros": nnz,
        "amg_levels": int(len(ml.levels)),
        "cg_iterations_real_imag": iteration_counts,
        "relative_l2_residual_real_imag": relative_residuals,
        "scaled_free_dof_inf_residual_real_imag": scaled_inf_residuals,
        "rhs_l2_norm_real_imag": rhs_norms,
        "assembly_seconds": float(assembly_seconds),
        "matrix_reduction_seconds": float(matrix_reduction_seconds),
        "amg_setup_seconds": float(hierarchy_seconds),
        "solve_seconds": float(solve_seconds),
        "total_solve_stage_seconds": float(assembly_seconds + matrix_reduction_seconds + hierarchy_seconds + solve_seconds),
        "rss_bytes_after_solve": rss_bytes(),
        "accepted": True,
        "acceptance_basis": "full-domain P4 Galerkin field, explicit boundary coverage and topology checks, two complex components converged to residual criteria",
    }
    atomic_json(mesh_dir / "solver_diagnostics.json", result)
    append_checkpoint({"time_utc": iso(utc_now()), "pattern": pattern_name, "stage": "p4_solved", "result": result})
    print(f"[{pattern_name}] accepted P4 field: elapsed={result['total_solve_stage_seconds']:.1f}s, residuals={relative_residuals}, RSS={rss_bytes()} bytes", flush=True)
    return result


def run(pattern_name: str, resolution_level: str = "h35") -> dict:
    state = start_or_resume_budget()
    require_time(state, float(CONFIG["budget"]["deadline_guard_before_new_geometry_solve_s"]), "the requested displaced geometry")
    if resolution_level not in ("h35", "h50"):
        raise ValueError("resolution_level must be 'h35' or 'h50'.")
    dirname = pattern_name if resolution_level == "h35" else f"{pattern_name}_h50"
    mesh_dir = OUT / "calculations" / dirname
    mesh_path = mesh_dir / f"displaced_{resolution_level}.msh"
    if mesh_path.exists():
        recipe_path = mesh_dir / "geometry_recipe.json"
        if not recipe_path.exists():
            raise FileNotFoundError(f"Mesh exists without a reproducible geometry recipe: {recipe_path}")
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        mesh_meta = validate_mesh(mesh_path, recipe)
        import gmsh
        mesh_meta.update({"gmsh_version": gmsh.__version__, "mesh_sha256": sha256(mesh_path)})
        atomic_json(mesh_dir / "mesh_diagnostics.json", mesh_meta)
        print(f"[{pattern_name}] reusing and revalidating saved mesh ({mesh_meta['triangles']:,} triangles)", flush=True)
    else:
        mesh_meta = generate_mesh(pattern_name, mesh_dir, state, resolution_level)
        print(f"[{pattern_name}] mesh validated: {mesh_meta['vertices']:,} vertices, {mesh_meta['triangles']:,} triangles, Euler {mesh_meta['euler_characteristic']}", flush=True)
    if (mesh_dir / "solver_diagnostics.json").exists() and (mesh_dir / "p4_coefficients.npz").exists():
        existing = json.loads((mesh_dir / "solver_diagnostics.json").read_text(encoding="utf-8"))
        if existing.get("accepted") and existing.get("mesh_sha256") == sha256(mesh_path) and existing.get("field_sha256") == sha256(mesh_dir / "p4_coefficients.npz"):
            print(f"[{pattern_name}] validated P4 solution already checkpointed; reusing it", flush=True)
            return existing
    return solve_mesh(pattern_name, mesh_dir, state, resolution_level)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="radial_out_10um")
    parser.add_argument("--resolution", choices=("h35", "h50"), default="h35")
    args = parser.parse_args()
    try:
        result = run(args.pattern, args.resolution)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        append_checkpoint({"time_utc": iso(utc_now()), "pattern": args.pattern, "resolution_level": args.resolution, "stage": "failed", "error_type": type(exc).__name__, "error": str(exc), "remaining_seconds": remaining_seconds(start_or_resume_budget()), "rss_bytes": rss_bytes()})
        raise


if __name__ == "__main__":
    main()
