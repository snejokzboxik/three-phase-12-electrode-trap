"""Analyze full-domain displaced P4 fields against the validated nominal P4 FEM."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OUT / ".deps"))
import solve_displaced_p4 as pilot

SECTOR_SOLUTION = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz"
NOMINAL_MESH = ROOT / "outputs/meshes/gmsh_35um/trap.msh"
AMU = 1.66053906660e-27
E_CHARGE = 1.602176634e-19
RHO = complex(-0.5, -math.sqrt(3.0) / 2.0)
ION_MASS = 100_000 * AMU
CHARGE = E_CHARGE
OMEGA = 2 * math.pi * 100_000.0
VOLTAGE = 250.0


def exponents(degree: int) -> list[tuple[int, int]]:
    return [(i, total - i) for total in range(degree + 1) for i in range(total, -1, -1)]


def vandermonde(xy: np.ndarray, degree: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return np.stack([xy[..., 0] ** i * xy[..., 1] ** j for i, j in exponents(degree)], axis=-1)


def permutation_transform(permutation: tuple[int, int, int], degree: int) -> np.ndarray:
    nodes = np.array([(i / degree, j / degree) for i, j in exponents(degree)])
    vertices = np.array([[0., 0.], [1., 0.], [0., 1.]])[list(permutation)]
    old = vertices[0] + nodes[:, [0]] * (vertices[1] - vertices[0]) + nodes[:, [1]] * (vertices[2] - vertices[0])
    return np.linalg.solve(vandermonde(nodes, degree), vandermonde(old, degree))


class FullP4Field:
    """Evaluate a complex P4 FEM potential and its native elementwise gradient."""
    def __init__(self, points: np.ndarray, triangles: np.ndarray, coefficients: np.ndarray, degree: int = 4):
        from skfem import MeshTri
        self.points = np.asarray(points, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.coefficients = np.asarray(coefficients, dtype=complex)
        self.degree = int(degree)
        if self.coefficients.shape != (len(self.triangles), len(exponents(self.degree))):
            raise ValueError(f"Bad polynomial coefficient shape: {self.coefficients.shape}")
        self.vertices = self.points[self.triangles]
        jacobian = np.swapaxes(self.vertices[:, 1:] - self.vertices[:, [0]], 1, 2)
        self.inverse = np.linalg.inv(jacobian)
        self.origin = self.vertices[:, 0]
        self.neighbors = np.empty((len(self.triangles), 3), dtype=int)
        mesh = MeshTri(self.points.T, self.triangles.T, sort_t=False)
        for face, local_edge in enumerate((1, 2, 0)):
            facets = mesh.t2f[local_edge]
            first, second = mesh.f2t[:, facets]
            self.neighbors[:, face] = np.where(first == np.arange(len(first)), second, first)
        self.centroids = self.vertices.mean(axis=1)
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.centroids)
        self.last_id: int | None = None
        powers = exponents(self.degree)
        d_powers = exponents(self.degree - 1)
        index = {power: k for k, power in enumerate(d_powers)}
        self.du = np.zeros((len(self.triangles), len(d_powers)), dtype=complex)
        self.dv = np.zeros_like(self.du)
        for source, (i, j) in enumerate(powers):
            if i:
                self.du[:, index[(i - 1, j)]] += i * self.coefficients[:, source]
            if j:
                self.dv[:, index[(i, j - 1)]] += j * self.coefficients[:, source]

    def _coordinates(self, cell: int, x: float, y: float) -> tuple[float, float, float]:
        delta = np.array([x - self.origin[cell, 0], y - self.origin[cell, 1]])
        u, v = self.inverse[cell] @ delta
        return float(u), float(v), float(min(1.0 - u - v, u, v))

    def find(self, x: float, y: float) -> tuple[int, float, float]:
        cell = self.last_id
        if cell is not None and cell >= 0:
            for _ in range(32):
                u, v, minimum = self._coordinates(cell, x, y)
                if minimum >= -2e-12:
                    self.last_id = cell
                    return cell, u, v
                weights = (1.0 - u - v, u, v)
                face = int(np.argmin(weights))
                cell = int(self.neighbors[cell, face])
                if cell < 0:
                    break
        _, candidates = self.tree.query([x, y], k=20)
        for candidate in np.atleast_1d(candidates):
            candidate = int(candidate)
            u, v, minimum = self._coordinates(candidate, x, y)
            if minimum >= -3e-11:
                self.last_id = candidate
                return candidate, u, v
        raise ValueError(f"Point ({x:g}, {y:g}) is outside the FEM vacuum mesh.")

    @staticmethod
    def polynomial(c: np.ndarray, u: float, v: float, degree: int) -> complex:
        return complex(np.dot(c, vandermonde(np.array([u, v]), degree)))

    def sample_one(self, x: float, y: float) -> tuple[complex, np.ndarray, np.ndarray]:
        cell, u, v = self.find(x, y)
        H = self.polynomial(self.coefficients[cell], u, v, self.degree)
        dHu = self.polynomial(self.du[cell], u, v, self.degree - 1)
        dHv = self.polynomial(self.dv[cell], u, v, self.degree - 1)
        dHx = self.inverse[cell, 0, 0] * dHu + self.inverse[cell, 1, 0] * dHv
        dHy = self.inverse[cell, 0, 1] * dHu + self.inverse[cell, 1, 1] * dHv
        e_cos = -np.array([dHx.imag, dHy.imag], dtype=float)
        e_sin = -np.array([dHx.real, dHy.real], dtype=float)
        return H, e_cos, e_sin

    def sample(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(xy, dtype=float).reshape(-1, 2)
        H = np.empty(len(points), dtype=complex)
        ec = np.empty((len(points), 2), dtype=float)
        es = np.empty((len(points), 2), dtype=float)
        for i, (x, y) in enumerate(points):
            H[i], ec[i], es[i] = self.sample_one(float(x), float(y))
        return H, ec, es


def field_from_coefficients(mesh_path: Path, coeff_path: Path) -> tuple[FullP4Field, dict]:
    from skfem import Basis, ElementTriP4, MeshTri
    data = __import__("meshio").read(mesh_path)
    raw_triangles = data.get_cells_type("triangle")
    used = np.unique(raw_triangles)
    remap = np.full(len(data.points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = np.asarray(data.points[used, :2], dtype=float)
    triangles = remap[raw_triangles]
    mesh = MeshTri(points.T, triangles.T, sort_t=True)
    basis = Basis(mesh, ElementTriP4(), intorder=8)
    with np.load(coeff_path, allow_pickle=False) as archive:
        nodal_values = archive["coefficients"]
    if len(nodal_values) != basis.N:
        raise ValueError(f"P4 global coefficient count mismatch: file={len(nodal_values)} basis={basis.N}")

    # Check local reference-node order against the mesh's affine maps before
    # evaluating element polynomials at arbitrary points.
    ref = basis.elem.doflocs
    tri_xy = mesh.p[:, mesh.t]
    expected = tri_xy[:, [0], :] + (tri_xy[:, [1], :] - tri_xy[:, [0], :]) * ref[:, 0][None, :, None] + (tri_xy[:, [2], :] - tri_xy[:, [0], :]) * ref[:, 1][None, :, None]
    actual = basis.doflocs[:, basis.element_dofs]
    orientation_error = float(np.max(np.linalg.norm(expected - actual, axis=0)))
    if orientation_error > 1e-11:
        raise ValueError(f"P4 local node coordinates do not match canonical reference nodes: {orientation_error:g} m")
    nodal = nodal_values[basis.element_dofs].T
    V = vandermonde(ref, 4)
    element_coeff = nodal @ np.linalg.inv(V).T
    if np.max(np.abs(element_coeff @ V.T - nodal)) > 1e-9:
        raise ValueError("Elementwise P4 polynomial failed to reproduce its nodal values.")
    triangles_ordered = mesh.t.T.copy()
    field = FullP4Field(mesh.p.T.copy(), triangles_ordered, element_coeff, 4)
    meta = {"mesh_path": str(mesh_path.relative_to(ROOT)), "mesh_sha256": pilot.sha256(mesh_path), "field_sha256": pilot.sha256(coeff_path), "dofs": int(basis.N), "element_count": int(mesh.nelements), "local_reference_node_max_error_m": orientation_error, "element_nodal_reproduction_max_error": float(np.max(np.abs(element_coeff @ V.T - nodal)))}
    del data, basis, mesh, nodal_values, nodal, element_coeff
    return field, meta


def nominal_field() -> tuple[FullP4Field, dict]:
    import meshio
    data = meshio.read(NOMINAL_MESH)
    raw_triangles = data.get_cells_type("triangle")
    used = np.unique(raw_triangles)
    remap = np.full(len(data.points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = np.asarray(data.points[used, :2], dtype=float)
    triangles = remap[raw_triangles]
    with np.load(SECTOR_SOLUTION, allow_pickle=False) as archive:
        sector_points = archive["points"]
        sector_triangles = archive["triangles"]
        sector_coefficients = archive["coefficients"]
        degree = int(archive["degree"])
    if degree != 4:
        raise ValueError("Nominal source is not a P4 field.")
    sector_vertices = sector_points[sector_triangles]
    centroids = points[triangles].mean(axis=1)
    from scipy.spatial import cKDTree
    tree = cKDTree(sector_vertices.mean(axis=1))
    phi = np.arctan2(centroids[:, 1], centroids[:, 0])
    sector_angle = math.pi / 6
    rotation_ids = np.floor((phi + math.pi / 12) / sector_angle).astype(int) % 12
    coefficients = np.empty((len(triangles), len(exponents(degree))), dtype=complex)
    max_center_error = 0.0
    perms = tuple(itertools.permutations(range(3)))
    for k in range(12):
        selected = np.flatnonzero(rotation_ids == k)
        angle = k * sector_angle
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        local_vertices = points[triangles[selected]] @ rotation
        distances, ids = tree.query(local_vertices.mean(axis=1))
        if distances.size:
            max_center_error = max(max_center_error, float(np.max(distances)))
        if len(distances) and np.max(distances) > 1e-11:
            raise ValueError("Nominal full mesh is not the validated twelve-sector FEM mesh copy.")
        vertex_distances = np.linalg.norm(local_vertices[:, :, None] - sector_vertices[ids, None, :], axis=-1)
        permutation = np.argmin(vertex_distances, axis=2)
        if np.max(np.min(vertex_distances, axis=2)) > 1e-11:
            raise ValueError("Nominal sector/full mesh vertices do not map exactly.")
        for perm in perms:
            mask = np.all(permutation == perm, axis=1)
            if np.any(mask):
                coefficients[selected[mask]] = RHO ** k * (sector_coefficients[ids[mask]] @ permutation_transform(perm, degree).T)
    field = FullP4Field(points, triangles, coefficients, degree)
    meta = {"source_mesh": str(NOMINAL_MESH.relative_to(ROOT)), "source_mesh_sha256": pilot.sha256(NOMINAL_MESH), "source_sector_solution": str(SECTOR_SOLUTION.relative_to(ROOT)), "source_sector_solution_sha256": pilot.sha256(SECTOR_SOLUTION), "degree": degree, "elements": int(len(triangles)), "max_sector_mapping_center_error_m": max_center_error, "representation": "validated nominal h35 P4 sector FEM coefficients mapped onto the exact twelve-copy full nominal mesh; not N48"}
    return field, meta


def finite_difference_gradient_check(field: FullP4Field) -> dict:
    points = np.array([[1.73e-3, -0.81e-3], [2.31e-3, 0.47e-3], [-3.17e-3, 1.21e-3]])
    h = 1e-8
    checks = []
    for point in points:
        center, ec, es = field.sample_one(*point)
        shifted = np.array([point + [h, 0], point - [h, 0], point + [0, h], point - [0, h]])
        values, _, _ = field.sample(shifted)
        grad_cos = np.array([(values[0].imag - values[1].imag) / (2 * h), (values[2].imag - values[3].imag) / (2 * h)])
        grad_sin = np.array([(values[0].real - values[1].real) / (2 * h), (values[2].real - values[3].real) / (2 * h)])
        err_cos = float(np.linalg.norm(grad_cos + ec) / max(np.linalg.norm(ec), 1.0))
        err_sin = float(np.linalg.norm(grad_sin + es) / max(np.linalg.norm(es), 1.0))
        checks.append({"point_m": point.tolist(), "H_real_imag": [float(center.real), float(center.imag)], "cosine_gradient_relative_error": err_cos, "sine_gradient_relative_error": err_sin})
    return {"finite_difference_step_m": h, "checks": checks, "maximum_relative_error": max(max(x["cosine_gradient_relative_error"], x["sine_gradient_relative_error"]) for x in checks), "purpose": "independent finite-difference check of E=-grad(Phi) for both RF quadratures"}


def compare_fields(pattern: str, resolution: str, field: FullP4Field, nominal: FullP4Field) -> tuple[list[dict], list[dict], dict]:
    field_rows: list[dict] = []
    harmonic_rows: list[dict] = []
    radii = np.array([0.5, 1, 2, 3, 4, 5, 6, 7], dtype=float) * 1e-3
    ntheta = 144
    theta = np.arange(ntheta) * 2 * math.pi / ntheta
    for radius in radii:
        xy = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        H, ec, es = field.sample(xy)
        H0, ec0, es0 = nominal.sample(xy)
        delta_stack = np.concatenate((ec - ec0, es - es0), axis=1)
        nominal_stack = np.concatenate((ec0, es0), axis=1)
        rot_pseudo = CHARGE**2 * VOLTAGE**2 / (4 * ION_MASS * OMEGA**2 * E_CHARGE) * (np.sum(ec * ec, axis=1) + np.sum(es * es, axis=1))
        rot_pseudo0 = CHARGE**2 * VOLTAGE**2 / (4 * ION_MASS * OMEGA**2 * E_CHARGE) * (np.sum(ec0 * ec0, axis=1) + np.sum(es0 * es0, axis=1))
        st_pseudo = CHARGE**2 * VOLTAGE**2 / (4 * ION_MASS * OMEGA**2 * E_CHARGE) * np.sum(es * es, axis=1)
        st_pseudo0 = CHARGE**2 * VOLTAGE**2 / (4 * ION_MASS * OMEGA**2 * E_CHARGE) * np.sum(es0 * es0, axis=1)
        field_rows.append({
            "pattern": pattern, "resolution": resolution, "radius_mm": float(radius * 1e3),
            "field_delta_rms_V_m_per_V": float(np.sqrt(np.mean(np.sum(delta_stack * delta_stack, axis=1)))),
            "field_delta_p95_V_m_per_V": float(np.quantile(np.linalg.norm(delta_stack, axis=1), .95)),
            "field_delta_max_V_m_per_V": float(np.max(np.linalg.norm(delta_stack, axis=1))),
            "field_delta_rms_fraction_nominal": float(np.linalg.norm(delta_stack) / max(np.linalg.norm(nominal_stack), np.finfo(float).tiny)),
            "rotating_pseudopotential_nominal_median_eV": float(np.median(rot_pseudo0)),
            "rotating_pseudopotential_displaced_median_eV": float(np.median(rot_pseudo)),
            "rotating_pseudopotential_median_ratio": float(np.median(rot_pseudo) / max(np.median(rot_pseudo0), np.finfo(float).tiny)),
            "standing_pseudopotential_nominal_median_eV": float(np.median(st_pseudo0)),
            "standing_pseudopotential_displaced_median_eV": float(np.median(st_pseudo)),
            "standing_pseudopotential_median_ratio": float(np.median(st_pseudo) / max(np.median(st_pseudo0), np.finfo(float).tiny)),
            "angular_samples_deterministic": ntheta,
        })
        for m in (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8):
            harmonic = np.mean(H * np.exp(-1j * m * theta))
            harmonic0 = np.mean(H0 * np.exp(-1j * m * theta))
            harmonic_rows.append({"pattern": pattern, "resolution": resolution, "radius_mm": float(radius * 1e3), "azimuthal_order_m": m,
                "displaced_abs_coefficient_V_per_V": float(abs(harmonic)), "nominal_abs_coefficient_V_per_V": float(abs(harmonic0)),
                "displaced_coefficient_real": float(harmonic.real), "displaced_coefficient_imag": float(harmonic.imag),
                "symmetry_class": "allowed C12/ABC mode m=-4 mod 12" if (m + 4) % 12 == 0 else "symmetry-breaking on the balanced 12-angle harmonic grid"})

    # Locate the rotating-field pseudopotential minimum on a disclosed bounded
    # search region, then refine it with a local derivative-free optimizer.
    from scipy.optimize import minimize
    grid = np.linspace(-5e-3, 5e-3, 35)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    candidate = np.column_stack((xx.ravel(), yy.ravel()))
    candidate = candidate[np.linalg.norm(candidate, axis=1) <= 5e-3]
    _, egc, egs = field.sample(candidate)
    objective_values = np.sum(egc * egc, axis=1) + np.sum(egs * egs, axis=1)
    start = candidate[int(np.argmin(objective_values))]
    def objective(xy: np.ndarray) -> float:
        _, ecx, esx = field.sample(np.asarray(xy, dtype=float)[None, :])
        return float(np.sum(ecx[0] ** 2) + np.sum(esx[0] ** 2))
    opt = minimize(objective, start, method="Nelder-Mead", options={"xatol": 2e-9, "fatol": 1e-20, "maxiter": 200})
    nominal_obj = objective(np.zeros(2))
    minimum = {"search_region": "disk r<=5 mm", "grid_shape": [35, 35], "grid_start_m": start.tolist(), "optimizer": "Nelder-Mead", "optimizer_success": bool(opt.success), "optimizer_message": str(opt.message), "position_m": opt.x.tolist(), "radius_mm": float(np.linalg.norm(opt.x) * 1e3), "rotating_field_squared_at_min_V2_m2_per_V2": float(opt.fun), "value_at_origin_V2_m2_per_V2": nominal_obj, "interpretation": "numerically located minimum in the bounded working-region search; not automatically a zero-field null"}
    return field_rows, harmonic_rows, minimum


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_one(pattern: str, resolution: str, nominal: FullP4Field, nominal_meta: dict) -> dict:
    directory = OUT / "calculations" / (pattern if resolution == "h35" else f"{pattern}_{resolution}")
    mesh = directory / f"displaced_{resolution}.msh"
    coeff = directory / "p4_coefficients.npz"
    solver = json.loads((directory / "solver_diagnostics.json").read_text(encoding="utf-8"))
    if not solver.get("accepted") or solver.get("field_sha256") != pilot.sha256(coeff):
        raise ValueError(f"Missing or unvalidated P4 solution: {directory}")
    field, field_meta = field_from_coefficients(mesh, coeff)
    start = time.perf_counter()
    fd = finite_difference_gradient_check(field)
    rows, harmonics, minimum = compare_fields(pattern, resolution, field, nominal)
    result = {"pattern": pattern, "resolution": resolution, "solver": solver, "field_representation": field_meta, "nominal_field": nominal_meta, "gradient_check": fd, "field_metrics": rows, "harmonics": harmonics, "rotating_pseudopotential_minimum": minimum, "analysis_seconds": time.perf_counter() - start, "new_displaced_geometry_representation": "full-domain h35/h50 P4 FEM; no N48 symmetry restriction, no spatial harmonic fit"}
    out = directory / "field_diagnostics.json"
    pilot.atomic_json(out, result)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", action="append", required=True, help="repeat for each completed geometry")
    parser.add_argument("--resolution", action="append", required=True, choices=("h35", "h50"), help="pair each --pattern with one resolution")
    args = parser.parse_args()
    if len(args.pattern) != len(args.resolution):
        parser.error("provide one --resolution for every --pattern")
    runtime = pilot.start_or_resume_budget()
    pilot.require_time(runtime, 120.0, "loading the nominal P4 reference for field diagnostics")
    nominal, nominal_meta = nominal_field()
    all_field_rows, all_harmonic_rows = [], []
    summaries = []
    for pattern, resolution in zip(args.pattern, args.resolution):
        pilot.require_time(runtime, 30.0, f"field diagnostics for {pattern}/{resolution}")
        result = analyze_one(pattern, resolution, nominal, nominal_meta)
        summaries.append({"pattern": pattern, "resolution": resolution, "field_diagnostics": str((OUT / "calculations" / (pattern if resolution == "h35" else f"{pattern}_{resolution}") / "field_diagnostics.json").relative_to(ROOT)), "solver_iterations": result["solver"]["cg_iterations_real_imag"], "gradient_check_max_relative_error": result["gradient_check"]["maximum_relative_error"], "minimum_position_mm": [x * 1e3 for x in result["rotating_pseudopotential_minimum"]["position_m"]]})
        all_field_rows.extend(result["field_metrics"])
        all_harmonic_rows.extend(result["harmonics"])
        pilot.append_checkpoint({"time_utc": pilot.iso(pilot.utc_now()), "pattern": pattern, "resolution_level": resolution, "stage": "field_diagnostics_complete", "analysis_seconds": result["analysis_seconds"], "gradient_check_max_relative_error": result["gradient_check"]["maximum_relative_error"]})
        print(json.dumps(summaries[-1], indent=2), flush=True)
    aggregate = {"nominal_field": nominal_meta, "cases": summaries, "field_metrics_csv": str((OUT / "processed" / "field_sensitivity.csv").relative_to(ROOT)), "harmonics_csv": str((OUT / "processed" / "azimuthal_harmonics.csv").relative_to(ROOT))}
    processed = OUT / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    write_csv(processed / "field_sensitivity.csv", all_field_rows)
    write_csv(processed / "azimuthal_harmonics.csv", all_harmonic_rows)
    pilot.atomic_json(processed / "field_analysis_index.json", aggregate)


if __name__ == "__main__":
    main()
