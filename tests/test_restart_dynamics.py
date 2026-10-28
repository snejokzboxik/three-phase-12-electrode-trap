import numpy as np
import pytest
from scipy.integrate import Radau
from skfem import Basis, ElementTriP2, MeshTri

from three_phase_trap.audited_rf import bernstein_value
from three_phase_trap.fem import FEMBasisSolutions, element_electric_field
from three_phase_trap.interpolation import FEMInterpolator
from three_phase_trap.mesh import TrapMesh
from three_phase_trap.pilot_dynamics import AffineP2RF
from three_phase_trap.restart_dynamics import (
    affine_state_jacobian, integrate_radau, radau_barycentric, scaled_state_distance,
)


def field(discontinuous=False):
    mesh = MeshTri.init_tensor(np.arange(-2., 3.), np.arange(-1., 2.))
    basis = Basis(mesh, ElementTriP2())
    x, y = basis.doflocs
    trap = TrapMesh(mesh, {}, {}, np.array([], int))
    potentials = {"A": abs(x) if discontinuous else -x, "B": x*0, "C": x*0, "DC": x*0}
    fields = {p: element_electric_field(trap, v, basis) for p, v in potentials.items()}
    return AffineP2RF(FEMInterpolator(FEMBasisSolutions(trap, basis, potentials, fields)), 1.)


def test_radau_degree_elevation_matches_actual_cubic_dense_curve():
    solver = Radau(lambda t, y: np.array([y[2], y[3], -.3*y[0], -2*y[1]]),
                   0., [.2, .3, .4, -.1], .2, rtol=1e-8, atol=1e-10)
    solver.step()
    dense = solver.dense_output()
    g = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    origin = np.array([-.5, -.4])
    b = radau_barycentric(dense, origin, g)
    for u in np.linspace(0, 1, 31):
        reference = g@(dense(dense.t_old+u*dense.h)[:2]-origin)
        reference[0] += 1
        assert np.allclose(bernstein_value(b, u), reference, rtol=0., atol=7e-16)


def test_radau_restart_keeps_global_rf_clock_and_analytic_position():
    omega, phase, t0, t1 = 2*np.pi, .37, .25, .75
    y0 = np.array([-.1, .13, .8, .02])
    result = integrate_radau(field(), y0, 1., 1., 1., phase, t0, t1,
                            max_steps_per_period=30, rtol=2e-10, atol=2e-12)
    t, state = result.run.time, result.run.states
    dt = t-t0
    expected = y0[0]+y0[2]*dt+dt*np.cos(omega*t0+phase)/omega-(np.sin(omega*t+phase)-np.sin(omega*t0+phase))/omega**2
    assert np.max(abs(state[:, 0]-expected)) < 2e-10
    assert np.max(abs(state[:, 1]-(y0[1]+y0[3]*dt))) < 1e-12
    assert result.diagnostics["crossings"] >= 1
    assert np.array_equal(state[0], y0)


def test_radau_jump_force_matches_independent_piecewise_analytic_solution():
    from scipy.optimize import brentq
    phase, omega = .4, 2*np.pi
    def left(t):
        return -.5+t+t*np.cos(phase)/omega-(np.sin(omega*t+phase)-np.sin(phase))/omega**2
    crossing = brentq(left, 0., 1.)
    velocity = 1+(np.cos(phase)-np.cos(omega*crossing+phase))/omega
    def right(t):
        dt = t-crossing
        return velocity*dt-dt*np.cos(omega*crossing+phase)/omega+(np.sin(omega*t+phase)-np.sin(omega*crossing+phase))/omega**2
    result = integrate_radau(field(True), [-.5, .13, 1., 0.], 1., 1., 1., phase, 0., 1.,
                            max_steps_per_period=40, rtol=2e-10, atol=2e-12)
    expected = np.where(result.run.time <= crossing, left(result.run.time), right(result.run.time))
    assert np.max(abs(result.run.states[:, 0]-expected)) < 3e-10


def test_phase_space_distance_uses_velocity_as_well_as_position():
    assert scaled_state_distance([0, 0, .2, 0], [0, 0, 0, 0], .001, .2) == pytest.approx(1.)
    with pytest.raises(ValueError):
        scaled_state_distance(np.zeros(4), np.zeros(4), 0., 1.)


def test_radau_affine_jacobian_contains_the_raw_force_derivative():
    # Build a nonconstant curl-free affine force; a constant-field test cannot
    # detect a missing/incorrect Jacobian in the implicit solver.
    raw = field()
    raw.coefficients[0][:] = [1., 2., 3., -.2, 3., -4.]
    raw.coefficients[1][:] = [-.7, 5., 2., .4, 2., -3.]
    from three_phase_trap.audited_rf import _rhs
    cell, qm, omega, phase = 0, 2., 3., .4
    jac = affine_state_jacobian(raw, cell, qm, omega, phase)
    rhs = _rhs(raw, cell, qm, omega, phase)
    state, clock, h = np.array([-.5, -.3, .2, -.1]), .13, 1e-6
    fd = np.column_stack([(rhs(clock, state+h*np.eye(4)[j])-rhs(clock, state-h*np.eye(4)[j]))/(2*h) for j in range(4)])
    assert np.allclose(jac(clock, state), fd, rtol=1e-9, atol=1e-9)
