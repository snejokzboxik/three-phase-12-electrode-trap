"""Symmetric Gmsh geometry, tagged mesh import, and mesh-only diagnostics.

A periodic 30-degree sector is meshed once and rotated twelve times. Sector
interfaces are welded interior edges, not FEM boundaries. Geometry is affine;
quadratic solution DOFs are subsequently constructed by scikit-fem.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP2, MeshTri

from .config import MeshConfig, ProjectConfig, validate_config
from .geometry import TrapGeometry, build_geometry
from .mesh import TrapMesh, build_mesh


def mesh_path(config: MeshConfig, root: Path) -> Path:
    return Path(root) / config.mesh_file


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sector_geo(geometry: TrapGeometry, settings: MeshConfig) -> str:
    """Return a self-contained, metre-based Gmsh sector geometry."""
    r, a, R = geometry.electrode_orbit_radius, geometry.electrode_radius, geometry.outer_domain_radius
    # A complete electrode fits inside this sector; otherwise this construction
    # would need to split the hole along its radial interfaces.
    if a >= r * np.sin(np.pi / 12):
        raise ValueError("Electrode intersects the sector seams.")
    h, ho = settings.resolution, settings.outer_resolution
    return f'''// Metres. Sector [-15, +15] degrees; no electrostatic seam BCs.
SetFactory("Built-in");
R = {R:.17g}; r = {r:.17g}; a = {a:.17g};
h = {h:.17g}; ho = {ho:.17g};
Point(1) = {{0, 0, 0, h}};
Point(2) = {{R*Cos(-Pi/12), R*Sin(-Pi/12), 0, ho}};
Point(3) = {{R*Cos(Pi/12), R*Sin(Pi/12), 0, ho}};
Line(1) = {{1, 2}};
Circle(2) = {{2, 1, 3}};
Line(3) = {{1, 3}};
Curve Loop(1) = {{1, 2, -3}};
Point(4) = {{r, 0, 0, h}};
Point(5) = {{r+a, 0, 0, h}};
Point(6) = {{r, a, 0, h}};
Point(7) = {{r-a, 0, 0, h}};
Point(8) = {{r, -a, 0, h}};
Circle(4) = {{5, 4, 6}};
Circle(5) = {{6, 4, 7}};
Circle(6) = {{7, 4, 8}};
Circle(7) = {{8, 4, 5}};
Curve Loop(2) = {{4, 5, 6, 7}};
Plane Surface(1) = {{1, 2}};
Physical Surface("vacuum", 100) = {{1}};
Physical Curve("outer", 1) = {{2}};
Physical Curve("electrode_0", 10) = {{4, 5, 6, 7}};
Periodic Curve {{3}} = {{1}} Rotate {{{{0,0,1}}, {{0,0,0}}, Pi/6}};
Field[1] = MathEval;
Field[1].F = "{h:.17g}+({ho:.17g}-{h:.17g})*Min(1,Max(0,(Sqrt(x*x+y*y)-{settings.fine_radius:.17g})/({settings.transition_radius:.17g}-{settings.fine_radius:.17g})))";
Field[2] = Distance;
Field[2].CurvesList = {{4,5,6,7}};
Field[2].Sampling = 400;
Field[3] = Threshold;
Field[3].InField = 2;
Field[3].SizeMin = h;
Field[3].SizeMax = ho;
Field[3].DistMin = {2*h:.17g};
Field[3].DistMax = 0.002;
Field[4] = Min;
Field[4].FieldsList = {{1,3}};
Background Field = 4;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = ho;
Mesh.MeshSizeFactor = 1;
Mesh.ElementOrder = 1;
Mesh.Algorithm = 6;
Mesh.RandomSeed = {settings.interior_seed};
General.NumThreads = 1;
'''


def load_gmsh_mesh(path: Path) -> TrapMesh:
    """Read physical line tags, keeping them through vertex renumbering."""
    import meshio

    data = meshio.read(path)
    if any(block.type in ("triangle6", "line3") for block in data.cells):
        raise ValueError("Use affine triangle/line geometry; P2 DOFs are built by scikit-fem.")
    triangles = data.get_cells_type("triangle")
    lines = data.get_cells_type("line")
    tags = data.get_cell_data("gmsh:physical", "line")
    if not len(triangles) or not len(lines):
        raise ValueError("MSH must contain triangles and physically tagged boundary lines.")
    used = np.unique(triangles)
    remap = np.full(len(data.points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    if np.any(remap[lines] < 0):
        raise ValueError("Boundary line uses a vertex absent from the triangulation.")
    mesh = MeshTri(data.points[used, :2].T, remap[triangles].T, sort_t=False)
    lines = remap[lines]
    def nodes(name: str) -> np.ndarray:
        if name not in data.field_data:
            raise ValueError(f"Missing physical group {name}.")
        tag, dimension = data.field_data[name]
        selected = lines[tags == tag]
        if dimension != 1 or not len(selected):
            raise ValueError(f"Missing line elements for {name}.")
        return np.unique(selected)
    electrodes = {i: nodes(f"electrode_{i}") for i in range(12)}
    outer = nodes("outer")
    # Compare edges, not only vertices: this also detects unsown sector seams.
    expected = np.sort(mesh.facets[:, mesh.boundary_facets()].T, axis=1)
    actual = np.sort(lines, axis=1)
    if set(map(tuple, expected)) != set(map(tuple, actual)) or len(actual) != len(expected):
        raise ValueError("Tagged boundaries do not exactly cover mesh boundary edges.")
    phases = {p: np.unique(np.concatenate([electrodes[i] for i in range(12) if i % 3 == j]))
              for j, p in enumerate(("A", "B", "C"))}
    return TrapMesh(mesh, electrodes, phases, outer)


def configured_mesh(geometry: TrapGeometry, settings: MeshConfig, root: Path) -> TrapMesh:
    """Use the selected backend without silently replacing a missing Gmsh file."""
    if settings.backend == "delaunay":
        return build_mesh(geometry, settings.resolution, settings.boundary_points_per_electrode,
                          settings.outer_boundary_points, settings.interior_seed)
    path = mesh_path(settings, root)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["geometry"] != _geometry_parameters(geometry) or metadata["mesh_settings"] != asdict(settings):
        raise ValueError("Saved Gmsh mesh settings differ; run the mesh command with this config first.")
    if metadata["msh_sha256"] != sha256(path):
        raise ValueError("Saved Gmsh mesh checksum mismatch.")
    return load_gmsh_mesh(path)


def _geometry_parameters(g: TrapGeometry) -> dict:
    return {"electrode_count": g.num_electrodes, "electrode_orbit_radius": g.electrode_orbit_radius,
            "electrode_radius": g.electrode_radius, "outer_domain_radius": g.outer_domain_radius}


def mesh_diagnostics(mesh: TrapMesh, geometry: TrapGeometry, fine_radius: float) -> dict:
    points, triangles = mesh.points, mesh.triangles
    p = points[triangles]
    v, w = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(v[:, 0]*w[:, 1] - v[:, 1]*w[:, 0])
    lengths = np.linalg.norm(points[mesh.mesh.facets[0]] - points[mesh.mesh.facets[1]], axis=1)
    fine = np.all(np.linalg.norm(points[mesh.mesh.facets], axis=2) <= fine_radius, axis=0)
    sides = np.stack([np.linalg.norm(p[:, 1]-p[:, 0], axis=1),
                      np.linalg.norm(p[:, 2]-p[:, 1], axis=1), np.linalg.norm(p[:, 0]-p[:, 2], axis=1)], axis=1)
    quality = 4*np.sqrt(3)*area / np.sum(sides*sides, axis=1)
    angle = np.pi/6
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tree = cKDTree(points)
    distances, rotated_ids = tree.query(points @ rotation.T)
    triangle_set = set(map(tuple, np.sort(triangles, axis=1)))
    symmetry = all(tuple(t) in triangle_set for t in np.sort(rotated_ids[triangles], axis=1))
    expected_area = np.pi*(geometry.outer_domain_radius**2 - 12*geometry.electrode_radius**2)
    if np.min(area) <= 0 or not symmetry or np.max(distances) > 1e-11:
        raise ValueError("Degenerate or non-symmetric mesh.")
    return {"vertices": len(points), "triangles": len(triangles),
            "p2_dofs": int(Basis(mesh.mesh, ElementTriP2()).N),
            "euler_characteristic": int(len(points)-mesh.mesh.nfacets+len(triangles)),
            "area_m2": float(area.sum()), "relative_area_error": float(abs(area.sum()/expected_area-1)),
            "min_triangle_quality": float(quality.min()),
            "edge_length_all_m": dict(zip(("min", "median", "max"), map(float, np.quantile(lengths, [0,.5,1])))),
            "edge_length_working_region_m": dict(zip(("min", "median", "p95", "max"), map(float, np.quantile(lengths[fine], [0,.5,.95,1])))),
            "rotation_30deg_max_vertex_error_m": float(distances.max()),
            "rotation_30deg_triangle_match": symmetry,
            "electrode_boundary_nodes": {str(i): len(v) for i,v in mesh.electrode_nodes.items()},
            "outer_boundary_nodes": len(mesh.outer_nodes)}


def generate_mesh_artifacts(config: ProjectConfig, root: Path) -> dict:
    """Generate and validate the full mesh, without solving for potentials."""
    import gmsh
    import meshio

    validate_config(config)
    if config.mesh.backend != "gmsh":
        raise ValueError("mesh command requires mesh.backend = 'gmsh'.")
    g = build_geometry(config.geometry.electrode_count, config.geometry.electrode_orbit_radius,
                       config.geometry.electrode_radius, config.geometry.outer_domain_radius)
    path = mesh_path(config.mesh, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    geo_path = path.with_name("sector.geo")
    geo_path.write_text(sector_geo(g, config.mesh), encoding="utf-8")
    if gmsh.isInitialized():
        raise RuntimeError("Gmsh is already initialized; run mesh generation in its own process.")
    gmsh.initialize()
    try:
        gmsh.open(str(geo_path))
        gmsh.model.mesh.generate(2)
        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        sorted_tags = node_tags[order]
        xy = np.asarray(xyz).reshape(-1, 3)[order, :2]
        def ids(tags):
            return np.searchsorted(sorted_tags, tags)
        _, element_nodes = gmsh.model.mesh.getElementsByType(2)
        sector_triangles = ids(element_nodes).reshape(-1, 3)
        def boundary_lines(curves):
            return np.concatenate([ids(gmsh.model.mesh.getElementsByType(1, c)[1]).reshape(-1,2) for c in curves])
        outer = boundary_lines([2])
        electrode = boundary_lines([4,5,6,7])
        gmsh.write(str(path.with_name("sector.msh")))
    finally:
        gmsh.finalize()
    all_points, all_triangles, all_lines, all_tags = [], [], [], []
    for k in range(12):
        angle = k*np.pi/6
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        offset = k*len(xy)
        all_points.append(xy @ rotation.T)
        all_triangles.append(sector_triangles+offset)
        all_lines.extend((outer+offset, electrode+offset))
        all_tags.extend((np.full(len(outer),1), np.full(len(electrode),10+k)))
    # Weld coincident seam vertices with a distance tolerance, not coordinate
    # rounding (rounding can split close points across a decimal bin boundary).
    raw = np.concatenate(all_points)
    parents = np.arange(len(raw))
    def representative(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for left, right in cKDTree(raw).query_pairs(1e-11, output_type="ndarray"):
        lroot, rroot = representative(left), representative(right)
        parents[max(lroot,rroot)] = min(lroot,rroot)
    roots = np.fromiter((representative(i) for i in range(len(raw))), dtype=int, count=len(raw))
    unique, inverse = np.unique(roots, return_inverse=True)
    points = raw[unique]
    triangles = inverse[np.concatenate(all_triangles)]
    lines = inverse[np.concatenate(all_lines)]
    field_data = {"vacuum": np.array([100,2]), "outer": np.array([1,1])}
    field_data.update({f"electrode_{i}": np.array([10+i,1]) for i in range(12)})
    meshio.write(path, meshio.Mesh(np.column_stack((points, np.zeros(len(points)))),
                 [("line",lines),("triangle",triangles)],
                 cell_data={"gmsh:physical": [np.concatenate(all_tags), np.full(len(triangles),100)],
                            "gmsh:geometrical": [np.concatenate(all_tags), np.full(len(triangles),100)]},
                 field_data=field_data), file_format="gmsh22", binary=True)
    mesh = load_gmsh_mesh(path)
    diagnostics = mesh_diagnostics(mesh, g, config.mesh.fine_radius)
    if diagnostics["euler_characteristic"] != -11:
        raise ValueError("Expected one connected disk with twelve electrode holes (Euler = -11).")
    metadata = {"geometry": _geometry_parameters(g), "mesh_settings": asdict(config.mesh),
                "gmsh_version": gmsh.__version__, "meshio_version": meshio.__version__,
                "msh_sha256": sha256(path), "sector_geo_sha256": sha256(geo_path),
                "generator_sha256": sha256(Path(__file__)), "diagnostics": diagnostics,
                "geometry_order": 1, "solution_order": 2,
                "laplace_solved": False, "units": "metres"}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    _preview(mesh, path.with_suffix(".png"))
    return diagnostics


def _preview(mesh: TrapMesh, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(13,6), constrained_layout=True)
    for ax, limit, title in zip(axes, (.026,.001), ("Full mesh", "Central 2 mm")):
        ax.triplot(mesh.points[:,0], mesh.points[:,1], mesh.triangles, lw=.15, color="#30566f")
        ax.set(xlim=(-limit,limit), ylim=(-limit,limit), aspect="equal", title=title, xlabel="x (m)", ylabel="y (m)")
    fig.savefig(path, dpi=180)
    plt.close(fig)
