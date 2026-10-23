import numpy as np
from skfem import Basis, ElementTriP2, MeshTri

from three_phase_trap.fem import FEMBasisSolutions, element_electric_field
from three_phase_trap.interpolation import FEMInterpolator
from three_phase_trap.mesh import TrapMesh


def _potential(xy):
    x, y = np.asarray(xy)
    return 2.3 * x**2 - 1.7 * x * y + 0.8 * y**2 + 0.4 * x - 0.6 * y + 1.2


def _electric_field(xy):
    x, y = np.asarray(xy)
    return np.array([-4.6 * x + 1.7 * y - 0.4, 1.7 * x - 1.6 * y + 0.6])


def test_production_p2_field_is_gradient_of_p2_potential_on_skew_triangle():
    points = np.array([[0.2, 1.4, -0.3], [-0.4, 0.1, 1.2]])
    mesh = MeshTri(points, np.array([[0], [1], [2]]), sort_t=False)
    basis = Basis(mesh, ElementTriP2())
    trap_mesh = TrapMesh(mesh, {}, {}, np.array([], dtype=int))
    nodal_values = _potential(basis.doflocs)
    potentials = {name: nodal_values.copy() for name in ("A", "B", "C", "DC")}
    fields = {name: np.zeros((1, 2)) for name in potentials}
    solutions = FEMBasisSolutions(trap_mesh, basis, potentials, fields)
    interpolator = FEMInterpolator(solutions)

    barycentric = np.array([0.23, 0.31, 0.46])
    query = barycentric @ points.T
    actual_potential = interpolator.potential_basis(query, "A")
    actual_field = interpolator.field_basis(query, "A")
    assert np.isclose(actual_potential, _potential(query), rtol=0.0, atol=1e-13)
    assert np.allclose(actual_field, _electric_field(query), rtol=0.0, atol=1e-12)

    centroid_field = element_electric_field(trap_mesh, nodal_values, basis)[0]
    centroid = points.mean(axis=1)
    assert np.allclose(centroid_field, _electric_field(centroid), rtol=0.0, atol=1e-12)
