from dataclasses import replace

import numpy as np
from skfem import Basis, ElementTriP2, MeshTri

from three_phase_trap.config import ProjectConfig
from three_phase_trap.fem import FEMBasisSolutions
from three_phase_trap.fem_cache import fem_fingerprint, load_fem_cache, save_fem_cache
from three_phase_trap.mesh import TrapMesh


def _small_synthetic_basis():
    mesh = MeshTri(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[0], [1], [2]]), sort_t=False)
    basis = Basis(mesh, ElementTriP2())
    trap_mesh = TrapMesh(
        mesh,
        {index: np.array([], dtype=int) for index in range(12)},
        {phase: np.array([], dtype=int) for phase in ("A", "B", "C")},
        np.array([0, 1], dtype=int),
    )
    potentials = {phase: np.arange(basis.N, dtype=float) + index for index, phase in enumerate(("A", "B", "C", "DC"))}
    fields = {phase: np.full((1, 2), index, dtype=float) for index, phase in enumerate(("A", "B", "C", "DC"))}
    return FEMBasisSolutions(trap_mesh, basis, potentials, fields)


def test_fem_cache_round_trip_preserves_p2_mesh_dofs_and_bases(tmp_path):
    config = ProjectConfig()
    expected = _small_synthetic_basis()
    save_fem_cache(config, tmp_path, expected)
    _, mesh, actual, _ = load_fem_cache(config, tmp_path)
    assert np.array_equal(mesh.points, expected.points)
    assert np.array_equal(mesh.triangles, expected.triangles)
    assert np.array_equal(actual.basis.doflocs, expected.basis.doflocs)
    assert np.array_equal(actual.basis.element_dofs, expected.basis.element_dofs)
    for phase in ("A", "B", "C", "DC"):
        assert np.array_equal(actual.potentials[phase], expected.potentials[phase])
        assert np.array_equal(actual.fields[phase], expected.fields[phase])


def test_fem_fingerprint_changes_when_mesh_configuration_changes():
    config = ProjectConfig()
    changed = replace(config, mesh=replace(config.mesh, resolution=config.mesh.resolution * 0.9))
    assert fem_fingerprint(config) != fem_fingerprint(changed)
