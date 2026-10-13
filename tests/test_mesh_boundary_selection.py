import numpy as np

from three_phase_trap.geometry import build_geometry
from three_phase_trap.mesh import build_mesh


def test_electrode_dirichlet_nodes_are_only_sampled_circle_boundary_points():
    geometry = build_geometry()
    boundary_points = 24
    mesh = build_mesh(geometry, resolution=0.002, boundary_points_per_electrode=boundary_points, outer_boundary_points=96)

    for electrode in geometry.electrodes:
        nodes = mesh.electrode_nodes[electrode.index]
        radial_error = np.abs(np.linalg.norm(mesh.points[nodes] - np.asarray(electrode.center), axis=1) - geometry.electrode_radius)
        assert len(nodes) == boundary_points
        assert np.max(radial_error) < 1e-10
