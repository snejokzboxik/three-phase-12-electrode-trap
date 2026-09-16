"""Laplace FEM solves and linear basis-superposition support."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from skfem import Basis, BilinearForm, ElementTriP2, asm, condense, solve
from skfem.helpers import dot, grad

from .mesh import TrapMesh


@BilinearForm
def laplace_form(u, v, w):
    return dot(grad(u), grad(v))


@dataclass
class FEMBasisSolutions:
    trap_mesh: TrapMesh
    basis: Basis
    potentials: dict[str, np.ndarray]
    fields: dict[str, np.ndarray]

    @property
    def points(self) -> np.ndarray:
        return self.trap_mesh.points

    @property
    def triangles(self) -> np.ndarray:
        return self.trap_mesh.triangles


def solve_dirichlet(trap_mesh: TrapMesh, electrode_voltages: dict[int, float]) -> tuple[np.ndarray, Basis]:
    """Solve ∇²Φ=0 with electrode voltages and a grounded far outer boundary."""
    basis = Basis(trap_mesh.mesh, ElementTriP2())
    stiffness = asm(laplace_form, basis)
    rhs = np.zeros(basis.N, dtype=float)
    boundary_dofs = []
    fixed_values = np.zeros(basis.N, dtype=float)
    # Outer boundary: grounded vertices and edge-midpoint DOFs.
    outer_mask = np.zeros(trap_mesh.mesh.nvertices, dtype=bool)
    outer_mask[trap_mesh.outer_nodes] = True
    facet_ids = np.flatnonzero(outer_mask[trap_mesh.mesh.facets[0]] & outer_mask[trap_mesh.mesh.facets[1]])
    outer_dofs = np.concatenate((basis.nodal_dofs[0, trap_mesh.outer_nodes], basis.facet_dofs[:, facet_ids].ravel()))
    boundary_dofs.append(outer_dofs)
    # Electrode boundary: vertices and P2 edge-midpoint DOFs get the same
    # constant electrode voltage.
    for index, voltage in electrode_voltages.items():
        nodes = trap_mesh.electrode_nodes[int(index)]
        node_mask = np.zeros(trap_mesh.mesh.nvertices, dtype=bool)
        node_mask[nodes] = True
        facet_ids = np.flatnonzero(node_mask[trap_mesh.mesh.facets[0]] & node_mask[trap_mesh.mesh.facets[1]])
        dofs = np.concatenate((basis.nodal_dofs[0, nodes], basis.facet_dofs[:, facet_ids].ravel()))
        boundary_dofs.append(dofs)
        fixed_values[dofs] = float(voltage)
    D = np.unique(np.concatenate(boundary_dofs))
    condensed = condense(stiffness, rhs, D=D, x=fixed_values)
    matrix_c, rhs_c, solution, interior = condensed
    solution[interior] = solve(matrix_c, rhs_c)
    return solution, basis


def solve_basis(trap_mesh: TrapMesh) -> FEMBasisSolutions:
    """Calculate the RF phase bases and the common 1-V DC basis once."""
    potentials: dict[str, np.ndarray] = {}
    basis: Basis | None = None
    for phase in ("A", "B", "C"):
        voltages = {index: float(phase == electrode_phase) for index, electrode_phase in _electrode_phase_map(trap_mesh).items()}
        potentials[phase], basis = solve_dirichlet(trap_mesh, voltages)
    dc_voltages = {index: 1.0 for index in trap_mesh.electrode_nodes}
    potentials["DC"], basis = solve_dirichlet(trap_mesh, dc_voltages)
    if basis is None:
        raise RuntimeError("No FEM basis was calculated.")
    fields = {phase: element_electric_field(trap_mesh, values, basis=basis) for phase, values in potentials.items()}
    return FEMBasisSolutions(trap_mesh, basis, potentials, fields)


def p2_standard_element_dofs(trap_mesh: TrapMesh, basis: Basis) -> np.ndarray:
    """Return P2 DOFs in the canonical (v0,v1,v2,e01,e12,e20) order."""
    triangles = trap_mesh.triangles
    points = trap_mesh.points
    local = basis.element_dofs.T
    standard = np.empty_like(local)
    standard[:, :3] = local[:, :3]
    for element, triangle in enumerate(triangles):
        vertices = points[triangle]
        midpoints = np.array(((vertices[0] + vertices[1]) / 2.0, (vertices[1] + vertices[2]) / 2.0, (vertices[2] + vertices[0]) / 2.0))
        edge_coordinates = basis.doflocs[:, local[element, 3:]].T
        distances = np.linalg.norm(midpoints[:, None, :] - edge_coordinates[None, :, :], axis=-1)
        assignment = np.argmin(distances, axis=1)
        if len(np.unique(assignment)) != 3 or np.max(np.min(distances, axis=1)) > 1.0e-10:
            raise RuntimeError("Could not map P2 edge DOFs to canonical triangle edges.")
        standard[element, 3:] = local[element, 3:][assignment]
    return standard


def _electrode_phase_map(trap_mesh: TrapMesh) -> dict[int, str]:
    """Recover phase labels from the node sets, preserving the fixed ABC mapping."""
    # Node sets alone do not contain electrode labels after meshing; the order
    # is intentionally fixed by the geometry's index modulo three.
    return {index: ("A", "B", "C")[index % 3] for index in trap_mesh.electrode_nodes}


def solve_instantaneous(trap_mesh: TrapMesh, phase_voltages: dict[str, float]) -> np.ndarray:
    """Independent direct solve used only for validation."""
    electrode_voltages = {index: phase_voltages[("A", "B", "C")[index % 3]] for index in trap_mesh.electrode_nodes}
    values, _ = solve_dirichlet(trap_mesh, electrode_voltages)
    return values


def superpose_potential(solutions: FEMBasisSolutions, phase_voltages: dict[str, float]) -> np.ndarray:
    return sum(float(phase_voltages[p]) * solutions.potentials[p] for p in ("A", "B", "C"))


def superpose_field(solutions: FEMBasisSolutions, phase_voltages: dict[str, float]) -> np.ndarray:
    return sum(float(phase_voltages[p]) * solutions.fields[p] for p in ("A", "B", "C"))


def element_electric_field(trap_mesh: TrapMesh, nodal_potential: np.ndarray, basis: Basis | None = None) -> np.ndarray:
    """Return P2-gradient electric fields evaluated at triangle centroids."""
    triangles = trap_mesh.triangles
    points = trap_mesh.points
    p = points[triangles]
    if basis is None:
        values = np.asarray(nodal_potential)[triangles]
        gradients = np.array([
            np.linalg.solve(p_i[1:] - p_i[0], (v_i[1:] - v_i[0]))
            for p_i, v_i in zip(p, values)
        ])
        return -gradients
    local_dofs = p2_standard_element_dofs(trap_mesh, basis)
    local_values = np.asarray(nodal_potential)[local_dofs]
    gradients_p1_p2 = np.array([
        np.column_stack((
            np.linalg.solve(p_i[1:] - p_i[0], np.array([1.0, 0.0])),
            np.linalg.solve(p_i[1:] - p_i[0], np.array([0.0, 1.0])),
        ))
        for p_i in p
    ])
    # The first two solves are gradients of the barycentric weights at
    # vertices p1 and p2; the p0 gradient is their negative sum.
    gradients_lambda = np.concatenate(((-gradients_p1_p2[:, [0]] - gradients_p1_p2[:, [1]]), gradients_p1_p2), axis=1)
    l = np.full((len(p), 3), 1.0 / 3.0)
    shape_gradients = np.stack((
        (4 * l[:, 0] - 1)[:, None] * gradients_lambda[:, 0],
        (4 * l[:, 1] - 1)[:, None] * gradients_lambda[:, 1],
        (4 * l[:, 2] - 1)[:, None] * gradients_lambda[:, 2],
        4 * (l[:, 1, None] * gradients_lambda[:, 0] + l[:, 0, None] * gradients_lambda[:, 1]),
        4 * (l[:, 2, None] * gradients_lambda[:, 1] + l[:, 1, None] * gradients_lambda[:, 2]),
        4 * (l[:, 0, None] * gradients_lambda[:, 2] + l[:, 2, None] * gradients_lambda[:, 0]),
    ), axis=1)
    return -np.einsum("ek,ekd->ed", local_values, shape_gradients)
