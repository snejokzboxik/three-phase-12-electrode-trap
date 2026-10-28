import itertools
import numpy as np
import pytest
from skfem import MeshTri, Basis, ElementTriP4

from three_phase_trap.direct_fem import (SectorSolution, DirectFEMRF, RHO, vandermonde,
                                        exponents, permutation_transform, canonical_nodal_values)
from three_phase_trap.mesh import TrapMesh


def analytic_field():
    # Twelve sectors, each two radial triangles; polynomial H=(x-i*y)^4.
    angles = (np.arange(13)-.5)*np.pi/6
    points = np.vstack(([[0., 0.]], np.column_stack((np.cos(angles), np.sin(angles)))))
    points = points[:-1]
    triangles = np.array([[0, k+1, (k+1)%12+1] for k in range(12)])
    full = MeshTri(points.T, triangles.T, sort_t=True)
    trap = TrapMesh(full, {}, {}, np.array([], int))
    sector_points = points[[0, 1, 2]]
    sector_tri = np.array([[0, 1, 2]])
    nodes = np.array([(i/4, j/4) for i, j in exponents(4)])
    xy = sector_points[0]+nodes[:, [0]]*(sector_points[1]-sector_points[0])+nodes[:, [1]]*(sector_points[2]-sector_points[0])
    values = (xy[:, 0]-1j*xy[:, 1])**4
    coefficients = np.linalg.solve(vandermonde(nodes, 4), values)[None]
    return DirectFEMRF(SectorSolution(sector_points, sector_tri, coefficients, 4, {}), trap, 1.)


def test_exact_native_quartic_rotation_and_gradient_in_all_sectors():
    field = analytic_field()
    angle = np.linspace(-np.pi, np.pi, 127, endpoint=False)+.0041
    xy = .3*np.column_stack((np.cos(angle), np.sin(angle)))
    potential, ec, es = field.sample(xy)
    z = xy[:, 0]-1j*xy[:, 1]
    expected = z**4
    dx, dy = 4*z**3, -4j*z**3
    assert np.max(abs(potential-expected)) < 1e-14
    assert np.max(abs(ec-np.column_stack((-dx.imag, -dy.imag)))) < 1e-13
    assert np.max(abs(es-np.column_stack((-dx.real, -dy.real)))) < 1e-13
    for p, c, s in zip(xy, ec, es):
        assert np.max(abs(np.array(field.quadratures(*p))-np.r_[c, s])) < 1e-13


def test_native_jacobian_matches_independent_quartic_derivative():
    field = analytic_field()
    x, y, phase = -.13, .21, .47
    z = x-1j*y
    hessian = 12*z*z*np.array([[1., -1j], [-1j, -1.]])
    expected = -hessian.imag*np.cos(phase)-hessian.real*np.sin(phase)
    assert np.max(abs(field.jacobian(x, y, phase)-expected)) < 1e-12


def test_all_local_vertex_permutations_represent_same_polynomial():
    rng = np.random.default_rng(5)
    c = rng.normal(size=15)+1j*rng.normal(size=15)
    nodes = np.array([[.1, .2], [.3, .1], [.2, .6]])
    vertices = np.array([[0., 0.], [1., 0.], [0., 1.]])
    for perm in itertools.permutations(range(3)):
        new = permutation_transform(perm, 4)@c
        v = vertices[list(perm)]
        old = v[0]+nodes[:, [0]]*(v[1]-v[0])+nodes[:, [1]]*(v[2]-v[0])
        assert np.max(abs(vandermonde(nodes, 4)@new-vandermonde(old, 4)@c)) < 1e-12


def test_p4_sorted_mesh_has_consistent_actual_h1_dof_coordinates():
    mesh = MeshTri.init_tensor(np.linspace(-1., 1., 5), np.linspace(-1., 1., 5))
    basis = Basis(mesh, ElementTriP4())
    x, y = basis.doflocs
    values = (x-1j*y)**4
    nodes = canonical_nodal_values(basis, values)
    assert nodes.shape == (mesh.nelements, 15)


def test_cyclic_phase_encodes_original_ABC_voltage_law():
    for chi in (0., .4, 1.7):
        for k in range(12):
            assert (RHO**k*np.exp(1j*chi)).imag == pytest.approx(np.sin(chi-2*np.pi*(k%3)/3))


@pytest.mark.parametrize("method", ["DOP853", "Radau"])
def test_native_audited_integrator_matches_exact_RF_acceleration_and_clock(method):
    from three_phase_trap.direct_fem_dynamics import integrate_native
    field = analytic_field()
    field.gx[:] = 0.
    field.gy[:] = 0.
    field.gx[:, 0] = -1.
    omega, phase, t0, end = 2*np.pi, .37, .25, .75
    initial = np.array([-.1, .13, .6, .02])
    run = integrate_native(field, initial, 1., 1., 1., phase, t0, end,
                           method=method, steps_per_period=40, rtol=2e-10, atol=2e-12, radius=.9)
    t, state = run.run.time, run.run.states
    dt = t-t0
    exact = initial[0]+initial[2]*dt+np.cos(omega*t0+phase)*dt/omega-(np.sin(omega*t+phase)-np.sin(omega*t0+phase))/omega**2
    assert np.max(abs(state[:, 0]-exact)) < 2e-10
    assert run.diagnostics["crossings"] >= 1
    assert np.array_equal(state[0], initial)
    assert run.diagnostics["position_nudges"] == 0


@pytest.mark.parametrize("method", ["DOP853", "Radau"])
def test_native_quartic_dynamics_matches_independent_analytic_rhs(method):
    from scipy.integrate import solve_ivp
    from three_phase_trap.direct_fem_dynamics import integrate_native
    field = analytic_field()
    phase, t0, end = .42, .1, 1.6
    initial = np.array([-.3, .13, .4, -.05])
    run = integrate_native(field, initial, .01, 1., 1., phase, t0, end,
                           method=method, steps_per_period=50, rtol=2e-11, atol=2e-13, radius=.9)
    def rhs(t, y):
        z = y[0]-1j*y[1]
        gradient = np.array([4*z**3, -4j*z**3])
        e = -np.imag(gradient*np.exp(1j*(2*np.pi*t+phase)))
        return np.r_[y[2:], .01*e]
    reference = solve_ivp(rhs, (t0, end), initial, method="DOP853", rtol=4e-14, atol=2e-15,
                          max_step=.005, t_eval=run.run.time)
    assert np.max(abs(run.run.states-reference.y.T)) < 2e-10
def test_global_clock_geometry_envelope():
    from types import SimpleNamespace
    from three_phase_trap.direct_fem_dynamics import clock_membership_tolerance, forward_cell
    base = 256*np.finfo(float).eps
    clock, speed, gradient = .06366144547945267, .21036, 5e4
    envelope = clock_membership_tolerance(clock, speed, gradient, base)
    assert envelope > 6.1e-14  # Reproduced shared-face residual in native P4.
    assert (envelope-base)/gradient < 1e-16
    assert clock_membership_tolerance(clock, 0., gradient, base) == base
    assert clock_membership_tolerance(.23672, speed, gradient, base) >= envelope
    # A local crossing root represented on a large global clock can lie a
    # fraction of one position ULP inside the old triangle. Reproduce that
    # configuration and check the FORWARD owner without changing the state.
    h = 35e-6
    field = SimpleNamespace(
        origin=np.array([[0., 0.], [h, 0.]]),
        gradient=np.array([[[-1., -1.], [1., 0.], [0., 1.]],
                           [[0., -1.], [1., 1.], [-1., 0.]]])/h,
        neighbors=np.array([[1, -1, -1], [-1, 0, -1]]),
        local_coordinates=lambda cell, x, y: (cell, x, y),
        local_quadratures=lambda cell, x, y: (0., 0., 0., 0.),
        jacobian=lambda *args, **kwargs: np.zeros((2, 2)))
    state = np.array([h*(.5-5e-14), h*(.5-5e-14), .21, 0.])
    saved = state.copy()
    assert forward_cell(field, 0, state, clock, 1., 2*np.pi*1e5, 0., base) == 0
    assert forward_cell(field, 0, state, clock, 1., 2*np.pi*1e5, 0., envelope) == 1
    assert np.array_equal(state, saved)

