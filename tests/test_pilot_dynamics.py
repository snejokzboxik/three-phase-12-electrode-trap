from types import SimpleNamespace
import numpy as np
import pytest
from skfem import Basis, ElementTriP2, MeshTri

from three_phase_trap import pilot_dynamics
from three_phase_trap.pilot_dynamics import AffineP2RF, PolynomialRF, integrate_rf, oscillation_metrics
from three_phase_trap.fem import FEMBasisSolutions, element_electric_field
from three_phase_trap.mesh import TrapMesh
from three_phase_trap.interpolation import FEMInterpolator, InterpolationDomainError
from three_phase_trap.rf_reconstruction import HarmonicQuadratures
from three_phase_trap.validation_benchmarks import ideal_field, quadrupole_exact


def test_affine_evaluator_preserves_piecewise_p2_fields_and_triangle_jumps():
    mesh = MeshTri.init_tensor(np.linspace(0., 1., 4), np.linspace(0., 1., 4))
    basis = Basis(mesh, ElementTriP2())
    x, y = basis.doflocs
    trap = TrapMesh(mesh, {}, {}, np.array([], dtype=int))
    potentials = {"A": x**4+.2*x*y, "B": -.3*x*x+y**3, "C": .4*x*y*y-.2*x*y, "DC": x*0}
    fields = {p: element_electric_field(trap, u, basis) for p, u in potentials.items()}
    interp = FEMInterpolator(FEMBasisSolutions(trap, basis, potentials, fields))
    fast = AffineP2RF(interp, 7.)
    rng = np.random.default_rng(7)
    points = rng.uniform(.01, .99, (117, 2))
    values = np.array([fast.quadratures(*p) for p in points])
    ea, eb, ec = (interp.field_basis(points, p) for p in ("A", "B", "C"))
    expected = np.column_stack((np.sqrt(3)*7*(ec-eb)/2, 7*(ea-eb/2-ec/2)))
    assert np.allclose(values, expected, rtol=0., atol=5e-14)
    # A jump must survive the optimization; no averaging across the edge.
    centers = mesh.p[:, mesh.t].mean(axis=1).T
    jumps = []
    for i, neighbors in enumerate(interp._neighbors):
        for j in neighbors:
            if j > i:
                shared = np.intersect1d(mesh.t[:, i], mesh.t[:, j])
                middle = mesh.p[:, shared].mean(axis=1)
                a, b = middle+1e-7*(centers[i]-middle), middle+1e-7*(centers[j]-middle)
                jumps.append(np.linalg.norm(np.array(fast.quadratures(*a))-np.array(fast.quadratures(*b))))
    assert max(jumps) > 1e-5
    with pytest.raises(InterpolationDomainError):
        fast.quadratures(-2., -2.)


def test_scalar_polynomial_evaluator_keeps_all_high_modes():
    rng = np.random.default_rng(9)
    model = HarmonicQuadratures(rng.normal(size=49)+1j*rng.normal(size=49),
                                rng.normal(size=49)+1j*rng.normal(size=49), .008)
    points = rng.uniform(-.005, .005, (31, 2))
    actual = np.array([PolynomialRF(model).quadratures(*p) for p in points])
    ec, es = model.field_quadratures(points)
    assert np.allclose(actual, np.column_stack((ec, es)), rtol=2e-14, atol=1e-10)


def test_rf_integrator_matches_exact_quad_and_restart_preserves_phase():
    q = ideal_field(2, 2.)
    frequency, phase = 20/(2*np.pi), .7
    initial = np.array([.7, .1, .02, -.03])
    T = 1/frequency
    options = dict(max_steps_per_period=100, rtol=2e-11, atol=2e-13, radius=2.)
    result = integrate_rf(PolynomialRF(q), initial, 1., 1., frequency, phase, 0., 4*T, **options)
    truth = quadrupole_exact(result.time, initial, 2., 20., phase)
    assert result.status == "completed"
    assert np.max(abs(result.states-truth)) < 1e-9
    first = integrate_rf(PolynomialRF(q), initial, 1., 1., frequency, phase, 0., 2*T, **options)
    second = integrate_rf(PolynomialRF(q), first.states[-1], 1., 1., frequency, phase, 2*T, 4*T, **options)
    joined = np.vstack((first.states, second.states[1:]))
    assert np.max(abs(joined-result.states)) < 1e-9


def test_rf_escape_and_failed_solver_have_distinct_outcomes(monkeypatch):
    field = PolynomialRF(ideal_field(2, 0.))
    result = integrate_rf(field, [.5, 0., 1., 0.], 0., 1., 4., .3, 0., 1., radius=1.)
    assert result.status == "domain_exit"
    assert result.exit_time == pytest.approx(.5, abs=1e-12)
    assert result.states[-1, 0] == pytest.approx(1., abs=1e-12)
    monkeypatch.setattr(pilot_dynamics, "solve_ivp", lambda *a, **kw: SimpleNamespace(success=False, message="forced"))
    with pytest.raises(RuntimeError, match="forced"):
        integrate_rf(field, [.5, 0., 0., 0.], 0., 1., 4., 0., 0., 1., radius=1.)


def test_period_estimator_requires_interior_cycles_and_refines_peaks():
    period = .023
    t = np.linspace(.00001, 3.25*period, 2001)
    position = np.column_stack((.001*np.cos(2*np.pi*t/period), t*0))
    result = oscillation_metrics(t, position, period)
    assert result["mean_period_s"] == pytest.approx(period, rel=1e-7)
    assert len(result["periods_s"]) == 2
    short = oscillation_metrics(t[:600], position[:600], period)
    assert short["mean_period_s"] is None
