import numpy as np
import pytest
from scipy.optimize import brentq
from skfem import MeshTri, Basis, ElementTriP2

from three_phase_trap.fem import FEMBasisSolutions, element_electric_field
from three_phase_trap.mesh import TrapMesh
from three_phase_trap.interpolation import FEMInterpolator
from three_phase_trap.pilot_dynamics import AffineP2RF
from three_phase_trap.event_resolved_rf import integrate_resolved


def field():
    mesh = MeshTri.init_tensor(np.array([-2., -1., 0., 1., 2.]), np.array([-1., 0., 1.]))
    basis = Basis(mesh, ElementTriP2())
    x, y = basis.doflocs
    trap = TrapMesh(mesh, {}, {}, np.array([], dtype=int))
    potentials = {"A": abs(x), "B": x*0, "C": x*0, "DC": x*0}
    fields = {p: element_electric_field(trap, v, basis) for p, v in potentials.items()}
    return AffineP2RF(FEMInterpolator(FEMBasisSolutions(trap, basis, potentials, fields)), 1.)


def test_ballistic_edge_crossings_keep_global_time_and_state_continuous():
    result, crossings = integrate_resolved(field(), [-1.5, .13, 3., .17], 0., 1., 4., .7, 0., 1.)
    assert result.status == "completed" and crossings >= 4
    assert np.allclose(result.states[:, 0], -1.5+3*result.time, rtol=0., atol=1e-10)
    assert np.allclose(result.states[:, 1], .13+.17*result.time, rtol=0., atol=1e-10)


def test_discontinuous_force_matches_independent_piecewise_analytic_solution():
    phase, omega = .4, 2*np.pi
    def left_x(t):
        return -.5+t+t*np.cos(phase)/omega-(np.sin(omega*t+phase)-np.sin(phase))/omega**2
    crossing = brentq(left_x, 0., 1.)
    crossing_velocity = 1+(np.cos(phase)-np.cos(omega*crossing+phase))/omega
    def right_x(t):
        dt = t-crossing
        return crossing_velocity*dt-dt*np.cos(omega*crossing+phase)/omega+(np.sin(omega*t+phase)-np.sin(omega*crossing+phase))/omega**2
    result, crossings = integrate_resolved(field(), [-.5, .13, 1., 0.], 1., 1., 1., phase, 0., 1.,
                                           max_steps_per_period=20, rtol=2e-11, atol=2e-13)
    expected = np.where(result.time <= crossing, left_x(result.time), right_x(result.time))
    assert crossings >= 1
    assert np.max(abs(result.states[:, 0]-expected)) < 1e-9
