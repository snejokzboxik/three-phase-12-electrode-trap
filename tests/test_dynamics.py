import numpy as np

from three_phase_trap.config import ParticleConfig
from three_phase_trap.dynamics import integrate_trajectory, particle_rhs
from three_phase_trap.interpolation import InterpolationDomainError


def zero_field(position, time):
    del position, time
    return np.zeros(2)


def test_zero_field_particle_has_predictable_uniform_motion():
    particle = ParticleConfig(mass=2.0, charge=1.0, gamma=0.0, gravity_enabled=False)
    result = integrate_trajectory(None, particle, U0=0.0, frequency=1.0, initial_position=(0.1, -0.2), initial_velocity=(0.3, 0.4), periods=0.5, escape_radius=10.0, field_function=zero_field)
    assert np.allclose(result.state[0, -1], 0.25, atol=1e-9)
    assert np.allclose(result.state[1, -1], 0.0, atol=1e-9)
    assert result.bounded


def test_invalid_interpolation_is_explicit():
    particle = ParticleConfig()
    try:
        particle_rhs(0.0, np.zeros(4), particle, 1.0, 1.0, field_function=lambda p, t: np.array([np.nan, 0.0]))
    except InterpolationDomainError:
        pass
    else:
        raise AssertionError("Non-finite field was not rejected")

