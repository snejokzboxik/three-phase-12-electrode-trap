import math

import numpy as np
import pytest
from scipy.integrate import DOP853, solve_ivp
from skfem import Basis, ElementTriP2, MeshTri

from three_phase_trap.audited_rf import (
    BoundaryAmbiguity, GeometryAuditError, bernstein_value, dense_barycentric,
    first_exit, integrate_audited, membership_bound,
)
from three_phase_trap.fem import FEMBasisSolutions, element_electric_field
from three_phase_trap.interpolation import FEMInterpolator
from three_phase_trap.mesh import TrapMesh
from three_phase_trap.pilot_dynamics import AffineP2RF


def scalar_bernstein(power):
    return np.array([sum(power[k]*math.comb(j, k)/math.comb(7, k)
                         for k in range(min(j+1, len(power)))) for j in range(8)])


def barycentric_curve(power):
    first = scalar_bernstein(power)
    return np.column_stack((first, first*0+.3, .7-first))


def field(potential=lambda x, y: -x):
    mesh = MeshTri.init_tensor(np.array([-2., -1., 0., 1., 2.]), np.array([-1., 0., 1.]))
    basis = Basis(mesh, ElementTriP2())
    x, y = basis.doflocs
    trap = TrapMesh(mesh, {}, {}, np.array([], dtype=int))
    potentials = {"A": potential(x, y), "B": x*0, "C": x*0, "DC": x*0}
    fields = {p: element_electric_field(trap, v, basis) for p, v in potentials.items()}
    return AffineP2RF(FEMInterpolator(FEMBasisSolutions(trap, basis, potentials, fields)), 1.)


def test_exit_and_return_is_found_even_with_positive_endpoint_signs():
    # Exact path lambda0=(u-.3)(u-.7). Endpoint sign detection misses both roots.
    bary = barycentric_curve([.21, -1., 1.])
    fraction, face, missed = first_exit(bary, 1e-13)
    assert fraction == pytest.approx(.3, abs=1e-13)
    assert face == 0 and missed
    with pytest.raises(GeometryAuditError, match="leaves"):
        membership_bound(bary, 1e-13)


def test_very_narrow_excursion_is_not_replaced_by_uniform_sampling():
    width = 1e-5
    bary = barycentric_curve([.25-width**2, -1., 1.])
    fraction, face, missed = first_exit(bary, 1e-13)
    assert fraction == pytest.approx(.5-width, abs=2e-10)
    assert face == 0 and missed


def test_scipy_endpoint_event_miss_is_reproduced_on_an_exact_newtonian_path():
    # x''=2, x=(t-.3)(t-.7), y=.3. One exactly resolved RK step contains
    # two x=0 roots. This directly exercises the original event mechanism.
    def rhs(t, state):
        return np.array([state[2], state[3], 2., 0.])
    def event(t, state):
        return state[0]
    event.direction, event.terminal = -1, True
    initial = [.21, .3, -1., 0.]
    old = solve_ivp(rhs, (0., 1.), initial, method="DOP853", first_step=1., max_step=1.,
                    rtol=1e-9, atol=1e-12, events=event)
    assert len(old.t) == 2 and len(old.t_events[0]) == 0
    solver = DOP853(rhs, 0., initial, 1., first_step=1., max_step=1., rtol=1e-9, atol=1e-12)
    solver.step()
    gradients = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    b = dense_barycentric(solver.dense_output(), np.zeros(2), gradients)
    fraction, face, missed = first_exit(b, 1e-13)
    assert fraction == pytest.approx(.3, abs=1e-12)
    assert face == 1 and missed


def test_tangency_and_path_on_edge_do_not_create_spurious_crossings():
    tangent = barycentric_curve([.25, -1., 1.])
    assert first_exit(tangent, 1e-13) is None
    assert membership_bound(tangent, 1e-13) >= -1e-13
    on_edge = barycentric_curve([0.])
    assert first_exit(on_edge, 1e-13) is None


def test_actual_dop853_dense_coefficients_match_independent_evaluation():
    solver = DOP853(lambda t, y: np.array([y[2], y[3], -y[0]+np.sin(t), .3*y[1]]),
                    0., [.2, .3, .4, -.1], .2, rtol=1e-8, atol=1e-10)
    solver.step()
    dense = solver.dense_output()
    origin = np.array([-.5, -.4])
    gradients = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    b = dense_barycentric(dense, origin, gradients)
    for u in np.linspace(0, 1, 29):
        expected = gradients@(dense(dense.t_old+u*dense.h)[:2]-origin)
        expected[0] += 1
        assert np.allclose(bernstein_value(b, u), expected, rtol=0., atol=7e-16)


@pytest.mark.parametrize("offset", [0., 1e-8, -1e-8])
def test_vertex_and_near_vertex_ballistic_passages_preserve_state(offset):
    result = integrate_audited(field(), [-.5, -.5+offset, 1., 1.], 0., 1., 1., .3, 0., 1.,
                               max_steps_per_period=3)
    t, states = result.run.time, result.run.states
    assert np.max(abs(states[:, 0]-(-.5+t))) < 1e-12
    assert np.max(abs(states[:, 1]-(-.5+offset+t))) < 1e-12
    assert np.max(abs(states[:, 2:]-1.)) < 1e-14
    assert result.diagnostics["minimum_output_barycentric"] >= -result.diagnostics["barycentric_tolerance"]
    assert all(event["position_jump_m"] == event["velocity_jump_m_per_s"] == 0 for event in result.crossings)
    if offset == 0:
        assert result.diagnostics["vertex_crossings"] >= 1


def test_trajectory_along_an_edge_does_not_chatter():
    result = integrate_audited(field(), [0., -.5, 0., 1.], 0., 1., 1., 0., 0., 1., max_steps_per_period=3)
    assert np.max(abs(result.run.states[:, 0])) < 1e-14
    assert len(result.crossings) < 8


@pytest.mark.parametrize("touch_only", [False, True])
def test_rf_exit_return_and_tangency_match_analytic_sinusoid(touch_only):
    omega = 2*np.pi
    x0 = -1/omega**2 if touch_only else -.02
    result = integrate_audited(field(), [x0, .13, -1/omega, 0.], 1., 1., 1., 0., 0., 1.,
                               max_steps_per_period=3, rtol=2e-10, atol=2e-12)
    expected = x0-np.sin(omega*result.run.time)/omega**2
    assert np.max(abs(result.run.states[:, 0]-expected)) < 2e-10
    seam = [event for event in result.crossings if abs(event["state"][0]) < 1e-10]
    if not touch_only:
        assert len(seam) == 2


def test_discontinuous_force_against_independent_piecewise_solution():
    from scipy.optimize import brentq
    phase, omega = .4, 2*np.pi
    def left(t):
        return -.5+t+t*np.cos(phase)/omega-(np.sin(omega*t+phase)-np.sin(phase))/omega**2
    crossing = brentq(left, 0., 1.)
    velocity = 1+(np.cos(phase)-np.cos(omega*crossing+phase))/omega
    def right(t):
        dt = t-crossing
        return velocity*dt-dt*np.cos(omega*crossing+phase)/omega+(np.sin(omega*t+phase)-np.sin(omega*crossing+phase))/omega**2
    result = integrate_audited(field(lambda x, y: abs(x)), [-.5, .13, 1., 0.], 1., 1., 1., phase, 0., 1., max_steps_per_period=20)
    expected = np.where(result.run.time <= crossing, left(result.run.time), right(result.run.time))
    assert np.max(abs(result.run.states[:, 0]-expected)) < 1e-10


def test_nonunique_boundary_rest_state_fails_instead_of_choosing_a_cell():
    # Phi=-|x| at sin(phase)=1: both x<0 and x>0 accelerate away from the seam.
    with pytest.raises(BoundaryAmbiguity, match="Different forward"):
        integrate_audited(field(lambda x, y: -abs(x)), [0., .13, 0., 0.], 1., 1., 1., np.pi/2, 0., 1.)


def test_boundary_nonuniqueness_can_first_appear_at_fifth_position_derivative():
    # Phi_c=|x|*y, Phi_s=-|x| are C0 piecewise quadratic. At x=y=0,
    # v=(0,1), phase=0, the two smooth continuations agree through r''''.
    # With omega=1, x''=A*(t*cos(t)-sin(t))+higher terms, A=+/-1;
    # x'''''(0)=-2A points into opposite half-planes. Equal low-order jets
    # therefore do not justify an arbitrary choice of the finder cell.
    mesh = MeshTri.init_tensor(np.array([-2., -1., 0., 1., 2.]), np.array([-1., 0., 1.]))
    basis = Basis(mesh, ElementTriP2())
    x, y = basis.doflocs
    trap = TrapMesh(mesh, {}, {}, np.array([], dtype=int))
    pc, ps = abs(x)*y, -abs(x)
    potentials = {"A": ps, "B": -pc/np.sqrt(3), "C": pc/np.sqrt(3), "DC": x*0}
    fields = {p: element_electric_field(trap, v, basis) for p, v in potentials.items()}
    raw = AffineP2RF(FEMInterpolator(FEMBasisSolutions(trap, basis, potentials, fields)), 1.)
    with pytest.raises(BoundaryAmbiguity):
        integrate_audited(raw, [0., 0., 0., 1.], 1., 1., 1/(2*np.pi), 0., 0., 2*np.pi/64)
