from types import SimpleNamespace
import numpy as np
import pytest

from three_phase_trap import averaged_dynamics
from three_phase_trap.averaged_dynamics import AveragedRFModel, integrate_averaged
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction, ReconstructionDomainError


def quadrupole_reconstruction(radius=3.):
    # Phi=Re(z²)cos(t)+Re(-i z²)sin(t); Psi=2 r² at q=m=Omega=1.
    c, s = np.array([0., 0., 1.], complex), np.array([0., 0., -1j])
    return RFReconstruction(HarmonicQuadratures(c, s, 1.), radius, {"fem_fingerprint": "test"})


def test_scaled_si_quadrupole_integrator_against_exact_slow_solution():
    # Strongly different coordinate/velocity/energy scales exercise the SI conversion.
    reconstruction = quadrupole_reconstruction(.008)
    model = AveragedRFModel(reconstruction, 1e-12, 1e-15, 250., 1e5)
    omega_slow = 2*abs(model.charge*model.amplitude)/(model.mass*model.omega)
    R0, V0 = np.array([.001, .0002]), np.array([0., .0001])
    period = 2*np.pi/omega_slow
    result = integrate_averaged(model, R0, V0, 3*period, samples=501, rtol=1e-10, atol=1e-12, max_step=period/80)
    expected = R0*np.cos(omega_slow*result.time[:, None])+V0*np.sin(omega_slow*result.time[:, None])/omega_slow
    assert result.status == "completed"
    assert np.max(np.linalg.norm(result.states[:, :2]-expected, axis=-1)) < 1e-11
    assert np.max(abs(result.energy/result.energy[0]-1)) < 1e-8


def test_mixed_harmonics_force_is_negative_gradient_and_contains_anisotropy():
    reconstruction = quadrupole_reconstruction()
    q = reconstruction.unit_quadratures
    q.cosine = np.pad(q.cosine, (0, 6))
    q.sine = np.pad(q.sine, (0, 6))
    q.cosine[4], q.sine[4] = .3+.2j, -.1+.4j
    q.cosine[8], q.sine[8] = .03-.02j, .04+.01j
    model = AveragedRFModel(reconstruction, -1., 1., 1., 1/(2*np.pi))
    points = np.array([[.5, .1], [-.3, .6], [.2, -.4]])
    h = 1e-6
    fd = np.column_stack([-(model.potential(points+h*np.eye(2)[j])-model.potential(points-h*np.eye(2)[j]))/(2*h) for j in range(2)])
    assert np.allclose(model.force(points), fd, rtol=1e-8, atol=1e-9)
    # A radial reduction would erase this tangential component.
    force = model.force(points)
    assert np.max(abs(points[:, 0]*force[:, 1]-points[:, 1]*force[:, 0])) > .01


def test_amplitude_frequency_scaling_phase_and_serialization(tmp_path):
    reconstruction = quadrupole_reconstruction()
    path = tmp_path/"rf.json"
    reconstruction.save(path)
    restored = RFReconstruction.load(path, "test")
    with pytest.raises(ValueError, match="fingerprint"):
        RFReconstruction.load(path, "wrong")
    p = np.array([[.2, .3], [.1, -.4]])
    base = AveragedRFModel(restored, 2., 3., 4., 5.)
    scaled = AveragedRFModel(restored, 2., 3., 12., 15.)
    assert np.allclose(scaled.potential(p), base.potential(p), rtol=1e-14, atol=0)
    assert np.allclose(scaled.force(p), base.force(p), rtol=1e-14, atol=0)
    assert np.allclose(restored.field(p, .01, 4., 5., .7), restored.field(p, .01+.7/(2*np.pi*5), 4., 5.), atol=1e-14)
    with pytest.raises(ReconstructionDomainError):
        base.force([3.1, 0.])


def test_domain_exit_is_reported_with_event_time_instead_of_completed():
    model = AveragedRFModel(quadrupole_reconstruction(1.), 0., 1., 1., 1.)
    result = integrate_averaged(model, [.5, 0.], [1., 0.], 1., samples=22, max_step=.09)
    assert result.status == "domain_exit"
    assert result.exit_time == pytest.approx(.5, abs=1e-12)
    assert result.time[-1] == result.exit_time
    assert result.states[-1, 0] == pytest.approx(1., abs=1e-12)
    with pytest.raises(ValueError, match="strictly inside"):
        integrate_averaged(model, [1., 0.], [0., 0.], 1.)


def test_failed_solver_is_not_treated_as_success(monkeypatch):
    model = AveragedRFModel(quadrupole_reconstruction(), 1., 1., 1., 1.)
    monkeypatch.setattr(averaged_dynamics, "solve_ivp", lambda *a, **kw: SimpleNamespace(success=False, message="test failure"))
    with pytest.raises(RuntimeError, match="test failure"):
        integrate_averaged(model, [1., 0.], [0., 0.], 1.)
