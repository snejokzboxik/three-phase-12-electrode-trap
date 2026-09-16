"""Laplace FEM solves and linear basis-superposition support."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from skfem import Basis, BilinearForm, ElementTriP1, asm, condense, solve
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
    basis = Basis(trap_mesh.mesh, ElementTriP1())
    stiffness = asm(laplace_form, basis)
    rhs = np.zeros(basis.N, dtype=float)
    boundary_dofs = [trap_mesh.outer_nodes]
    fixed_values = np.zeros(basis.N, dtype=float)
    for index, voltage in electrode_voltages.items():
        nodes = trap_mesh.electrode_nodes[int(index)]
        boundary_dofs.append(nodes)
        fixed_values[nodes] = float(voltage)
    D = np.unique(np.concatenate(boundary_dofs))
    condensed = condense(stiffness, rhs, D=D, x=fixed_values)
    matrix_c, rhs_c, solution, interior = condensed
    solution[interior] = solve(matrix_c, rhs_c)
    return solution, basis


def solve_basis(trap_mesh: TrapMesh) -> FEMBasisSolutions:
    """Calculate the three independent 1-V phase basis solutions exactly once."""
    phase_node_order = {"A": 0, "B": 1, "C": 2}
    potentials: dict[str, np.ndarray] = {}
    basis: Basis | None = None
    for phase in ("A", "B", "C"):
        voltages = {index: float(phase == electrode_phase) for index, electrode_phase in _electrode_phase_map(trap_mesh).items()}
        potentials[phase], basis = solve_dirichlet(trap_mesh, voltages)
    if basis is None:
        raise RuntimeError("No FEM basis was calculated.")
    fields = {phase: element_electric_field(trap_mesh, values) for phase, values in potentials.items()}
    return FEMBasisSolutions(trap_mesh, basis, potentials, fields)


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


def element_electric_field(trap_mesh: TrapMesh, nodal_potential: np.ndarray) -> np.ndarray:
    """Return constant P1 electric-field vectors for every triangular element."""
    triangles = trap_mesh.triangles
    points = trap_mesh.points
    p = points[triangles]
    values = np.asarray(nodal_potential)[triangles]
    # The gradient is constant inside each linear triangle.  The explicit
    # per-element solve keeps the geometry-to-gradient mapping easy to audit.
    gradients = np.array([
        np.linalg.solve((p_i[1:] - p_i[0]).T, (v_i[1:] - v_i[0]))
        for p_i, v_i in zip(p, values)
    ])
    return -gradients
