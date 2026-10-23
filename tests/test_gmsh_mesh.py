from dataclasses import replace

import numpy as np
import pytest
from skfem import Basis, ElementTriP2

pytest.importorskip("gmsh")
pytest.importorskip("meshio")

from three_phase_trap.config import ProjectConfig, MeshConfig
from three_phase_trap.fem import FEMBasisSolutions
from three_phase_trap.geometry import build_geometry
from three_phase_trap.gmsh_mesh import configured_mesh, generate_mesh_artifacts
from three_phase_trap.interpolation import FEMInterpolator


@pytest.fixture(scope="module")
def imported_mesh(tmp_path_factory):
    root = tmp_path_factory.mktemp("gmsh")
    settings = MeshConfig(backend="gmsh", resolution=0.0007, outer_resolution=0.001,
                          mesh_file=str(root / "trap.msh"))
    config = replace(ProjectConfig(), mesh=settings)
    report = generate_mesh_artifacts(config, root)
    geometry = build_geometry()
    return configured_mesh(geometry, settings, root), geometry, report, config, root


def test_tagged_holes_and_welded_symmetric_interfaces(imported_mesh):
    mesh, geometry, report, _, _ = imported_mesh
    assert report["euler_characteristic"] == -11
    assert report["rotation_30deg_triangle_match"]
    assert report["rotation_30deg_max_vertex_error_m"] < 1e-11
    assert report["relative_area_error"] < 0.002
    assert report["min_triangle_quality"] > 0.3
    for electrode in geometry.electrodes:
        nodes = mesh.electrode_nodes[electrode.index]
        assert np.allclose(np.linalg.norm(mesh.points[nodes]-electrode.center,axis=1),
                           geometry.electrode_radius, rtol=0, atol=1e-12)
        assert set(nodes).issubset(set(mesh.phase_nodes[electrode.phase]))
    basis = Basis(mesh.mesh, ElementTriP2())
    for nodes in [mesh.outer_nodes, *mesh.electrode_nodes.values()]:
        mask = np.zeros(mesh.mesh.nvertices, dtype=bool)
        mask[nodes] = True
        facets = np.flatnonzero(np.all(mask[mesh.mesh.facets], axis=0))
        assert len(basis.get_dofs(facets=facets).all()) == 2*len(nodes)


def test_imported_mesh_supports_existing_p2_potential_and_gradient(imported_mesh):
    mesh, _, _, _, _ = imported_mesh
    basis = Basis(mesh.mesh, ElementTriP2())
    x, y = basis.doflocs
    values = (x*x-y*y)/1e-4 + 30*x
    potentials = {name: values for name in ("A","B","C","DC")}
    solutions = FEMBasisSolutions(mesh, basis, potentials,
                                 {name: np.zeros((len(mesh.triangles),2)) for name in potentials})
    interpolation = FEMInterpolator(solutions)
    points = mesh.points[mesh.triangles[::137]].mean(axis=1)
    x, y = points.T
    assert np.allclose(interpolation.potential_basis(points,"A"),
                       (x*x-y*y)/1e-4+30*x, rtol=0, atol=1e-12)
    assert np.allclose(interpolation.field_basis(points,"A"),
                       np.column_stack((-2*x/1e-4-30, 2*y/1e-4)), rtol=0, atol=1e-9)


def test_changed_mesh_settings_are_not_silently_reused(imported_mesh):
    _, geometry, _, config, root = imported_mesh
    with pytest.raises(ValueError, match="settings differ"):
        configured_mesh(geometry, replace(config.mesh,resolution=config.mesh.resolution/2),root)
