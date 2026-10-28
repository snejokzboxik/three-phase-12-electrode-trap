import numpy as np
import pytest

from three_phase_trap.rf_reconstruction import HarmonicQuadratures, fit_harmonic_quadratures
from three_phase_trap.spatial_convergence import field_quantities, relative_rms, sample_jumps, sample_origin_limits


def test_relative_rms_uses_the_full_vector_and_returns_no_relative_zero_reference():
    reference = np.array([[3., 4.], [0., 5.]])
    actual = reference+np.array([[0., 1.], [2., 0.]])
    assert relative_rms(actual, reference) == pytest.approx(np.sqrt(5/50))
    assert relative_rms(np.zeros(3), np.zeros(3)) is None
    with pytest.raises(ValueError):
        relative_rms([np.nan], [1.])


def test_field_quantities_matches_a_potential_gradient_for_octupole_and_higher_modes():
    c = np.zeros(9, complex)
    c[4], c[8] = 1.2+.3j, .02-.01j
    q = HarmonicQuadratures(c, 1j*c, 2.)
    x = np.array([[.3, -.4], [.6, .2]])
    ec, es = q.field_quadratures(x)
    jc, js = q.jacobian_quadratures(x)
    psi, force = field_quantities(ec, es, jc, js, 2., 3., 5.)
    assert np.allclose(psi, q.pseudo(x, 2., 3., 5.))
    step = 1e-6
    fd = np.column_stack([-(q.pseudo(x+step*np.eye(2)[j], 2., 3., 5.)-q.pseudo(x-step*np.eye(2)[j], 2., 3., 5.))/(2*step) for j in range(2)])
    assert np.allclose(force, fd, rtol=1e-8, atol=1e-14)


def test_same_harmonic_potential_on_nonnested_training_grids_has_same_field():
    c = np.array([0., .1, -.2j, .03, 1.+.2j])
    q = HarmonicQuadratures(c, .7j*c, 1.)
    fits = []
    for count, offset in ((47, .317), (61, .431)):
        angle = (np.arange(count)+offset)*2*np.pi/count
        x = np.column_stack((.7*np.cos(angle), .7*np.sin(angle)))
        pc, ps = q.potential_quadratures(x)
        fits.append(fit_harmonic_quadratures(x, pc, ps, 8, 1.)[0])
    test = np.array([[.02, .03], [.3, -.1]])
    assert relative_rms(np.stack(fits[0].field_quadratures(test)), np.stack(fits[1].field_quadratures(test))) < 1e-12


def test_jumps_of_continuous_p2_potential_are_normal_to_the_edge():
    from skfem import Basis, ElementTriP2, MeshTri
    from three_phase_trap.mesh import TrapMesh
    from three_phase_trap.fem import FEMBasisSolutions, element_electric_field
    from three_phase_trap.interpolation import FEMInterpolator
    mesh = MeshTri.init_tensor(np.array([-.001, 0., .001]), np.array([-.001, 0., .001]))
    basis = Basis(mesh, ElementTriP2())
    trap = TrapMesh(mesh, {}, {}, np.array([], int))
    x = basis.doflocs[0]
    potentials = {"A": abs(x), "B": x*0, "C": x*0, "DC": x*0}
    fields = {p: element_electric_field(trap, v, basis) for p, v in potentials.items()}
    interp = FEMInterpolator(FEMBasisSolutions(trap, basis, potentials, fields))
    data = sample_jumps(interp, 1.)
    assert data["normal_jump_rf_rms"].max() == pytest.approx(np.sqrt(2.))
    assert data["tangential_jump_rf_rms"].max() < 1e-12
    origin = sample_origin_limits(interp, 1.)
    assert len(origin["triangle_ids"]) >= 2
    assert origin["one_sided_rf_rms"].max() == pytest.approx(1/np.sqrt(2))
    assert np.ptp(origin["one_sided_field"][:, 0, 1]) == pytest.approx(2.)
