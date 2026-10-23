"""Conforming-ish triangular mesh generation for the circular electrode domain.

The mesh is made from deterministic interior samples, sampled electrode circles,
and a sampled outer circle.  Delaunay triangles whose centroids are inside an
electrode are removed, leaving the electrode circumferences as Dirichlet
boundaries.  Refining ``mesh.resolution`` provides a straightforward sanity
check on the approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay
from skfem import MeshTri

from .geometry import TrapGeometry, electrode_mask, phase_indices


@dataclass
class TrapMesh:
    mesh: MeshTri
    electrode_nodes: dict[int, np.ndarray]
    phase_nodes: dict[str, np.ndarray]
    outer_nodes: np.ndarray

    @property
    def points(self) -> np.ndarray:
        return self.mesh.p.T

    @property
    def triangles(self) -> np.ndarray:
        return self.mesh.t.T


def _unique_rows(points: np.ndarray, decimals: int = 13) -> np.ndarray:
    rounded = np.round(points, decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def build_mesh(geometry: TrapGeometry, resolution: float, boundary_points_per_electrode: int = 18, outer_boundary_points: int = 180, seed: int = 12345) -> TrapMesh:
    """Create a deterministic triangular mesh of free space around the electrodes."""
    if resolution <= 0:
        raise ValueError("Mesh resolution must be positive.")
    R = geometry.outer_domain_radius
    coordinates: list[np.ndarray] = []
    electrode_source: dict[int, list[int]] = {}
    outer_source: list[int] = []

    outer_angles = 2.0 * np.pi * np.arange(outer_boundary_points) / outer_boundary_points
    outer_points = np.column_stack((R * np.cos(outer_angles), R * np.sin(outer_angles)))
    for point in outer_points:
        outer_source.append(len(coordinates))
        coordinates.append(point)

    boundary_count = max(12, int(boundary_points_per_electrode))
    for electrode in geometry.electrodes:
        local_angles = 2.0 * np.pi * np.arange(boundary_count) / boundary_count
        points = np.column_stack((
            electrode.center[0] + geometry.electrode_radius * np.cos(local_angles),
            electrode.center[1] + geometry.electrode_radius * np.sin(local_angles),
        ))
        electrode_source[electrode.index] = []
        for point in points:
            electrode_source[electrode.index].append(len(coordinates))
            coordinates.append(point)

    half_count = int(np.ceil(R / resolution))
    grid_axis = np.arange(-half_count, half_count + 1, dtype=float) * resolution
    gx, gy = np.meshgrid(grid_axis, grid_axis, indexing="xy")
    interior = np.column_stack((gx.ravel(), gy.ravel()))
    in_outer = np.sum(interior**2, axis=1) < (R - 0.5 * resolution) ** 2
    interior = interior[in_outer]
    interior = interior[~electrode_mask(interior, geometry)]
    coordinates.extend(interior)
    points = _unique_rows(np.asarray(coordinates, dtype=float))

    triangulation = Delaunay(points)
    triangles = triangulation.simplices
    centroids = points[triangles].mean(axis=1)
    valid = (np.sum(centroids**2, axis=1) < R**2) & (~electrode_mask(centroids, geometry))
    triangles = triangles[valid]
    if len(triangles) == 0:
        raise RuntimeError("Mesh generation produced no free-space triangles.")

    used = np.unique(triangles)
    old_to_new = -np.ones(len(points), dtype=int)
    old_to_new[used] = np.arange(len(used))
    points = points[used]
    triangles = old_to_new[triangles]
    mesh = MeshTri(points.T, triangles.T)

    centers = np.array([e.center for e in geometry.electrodes])
    electrode_nodes: dict[int, np.ndarray] = {}
    for electrode in geometry.electrodes:
        distance = np.linalg.norm(points - np.asarray(electrode.center), axis=1)
        # Only the explicitly sampled circular-boundary points receive the
        # electrode Dirichlet value. A mesh-spacing-scaled tolerance also
        # captured nearby interior grid points, displacing the effective
        # boundary by O(h) and introducing angular asymmetry.
        electrode_nodes[electrode.index] = np.flatnonzero(np.isclose(distance, geometry.electrode_radius, rtol=0.0, atol=1e-10))
        if len(electrode_nodes[electrode.index]) < 6:
            raise RuntimeError(f"Too few boundary nodes identified for electrode {electrode.index}.")
    outer_distance = np.linalg.norm(points, axis=1)
    outer_nodes = np.flatnonzero(np.isclose(outer_distance, R, rtol=0.0, atol=max(resolution * 0.12, 1e-12)))
    if len(outer_nodes) < 24:
        raise RuntimeError("Too few outer-boundary nodes identified.")

    phase_node_sets = phase_indices(geometry)
    phase_nodes = {
        phase: np.unique(np.concatenate([electrode_nodes[int(i)] for i in indices]))
        for phase, indices in phase_node_sets.items()
    }
    return TrapMesh(mesh, electrode_nodes, phase_nodes, outer_nodes)
