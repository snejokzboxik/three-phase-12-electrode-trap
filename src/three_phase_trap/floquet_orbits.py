"""Shooting and variational equations for periodic orbits of a smooth RF field."""
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import beta


PHASE = math.pi / 2
SYMPLECTIC = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
IDENTITY = np.eye(4)


class ShootingFailure(RuntimeError):
    pass


class SmoothRFOrbit:
    """Position in aperture radii, velocity in radii/RF-period, clock in RF periods."""

    def __init__(self, cosine, sine):
        self.gradients = [
            tuple(complex(c) for c in np.polynomial.polynomial.polyder(coefficients)[::-1])
            for coefficients in (cosine, sine)
        ]

    def flow(self, z0, coupling, periods, *, steps, variational=True, samples=None):
        scale = 4 * math.pi**2 * coupling
        initial = np.r_[z0, IDENTITY.ravel()] if variational else z0

        def rhs(n, y):
            z = complex(y[0], y[1])
            co, si = math.cos(2 * math.pi * n + PHASE), math.sin(2 * math.pi * n + PHASE)
            fields, gradients = [], []
            for coefficients in self.gradients:
                p = dp = 0j
                for coefficient in coefficients:
                    dp = dp * z + p
                    p = p * z + coefficient
                fields.append(p)
                gradients.append(dp)
            g = scale * (co * fields[0] + si * fields[1])
            dz = np.array([y[2], y[3], -g.real, g.imag])
            if not variational:
                return dz
            h = scale * (co * gradients[0] + si * gradients[1])
            tangent = y[4:].reshape(4, 4)
            acceleration = np.array([[-h.real, h.imag], [h.imag, h.real]]) @ tangent[:2]
            return np.r_[dz, tangent[2:].ravel(), acceleration.ravel()]

        def exit_disk(_, y):
            return 1 - y[0]**2 - y[1]**2

        exit_disk.terminal = True
        exit_disk.direction = -1
        grid = np.linspace(0., periods, round(samples * periods) + 1) if samples else [periods]
        sol = solve_ivp(rhs, (0., periods), initial, t_eval=grid, events=exit_disk, method="DOP853",
                        max_step=1 / steps, rtol=2e-12 if steps <= 24 else 3e-14,
                        atol=2e-14 if steps <= 24 else 3e-16)
        if not sol.success or sol.status == 1 or len(sol.t) != len(grid):
            raise ShootingFailure("RF flow left the aperture or failed to reach the section")
        final = sol.y[:, -1]
        return final[:4], final[4:].reshape(4, 4) if variational else None, sol.y[:4].T


def circular_seed(coupling):
    radius = 1 / (8 * np.sqrt(coupling))
    return np.array([radius, 0., 0., np.pi * radius / 2])


def slow_seed(coupling, periods):
    """Radial r^6 period law, with the leading micromotion velocity at phase pi/2."""
    radius = np.sqrt(beta(1 / 6, .5) / (12 * np.pi * coupling * periods))
    return np.array([radius, 0., 0., 8 * np.pi * coupling * radius**3])


def shoot(evaluate, initial, admissible, *, tolerance, iterations=12):
    """Damped Newton without allowing a step towards the trivial central orbit."""
    state = np.array(initial, float)
    residual, jacobian = evaluate(state)
    for _ in range(iterations):
        size = np.linalg.norm(residual, ord=np.inf)
        if size <= tolerance:
            return state
        delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=1e-11)
        for damping in (1., .5, .25, .125, .0625):
            trial = state + damping * delta
            if not admissible(trial):
                continue
            try:
                r, j = evaluate(trial)
            except ShootingFailure:
                continue
            if np.linalg.norm(r, ord=np.inf) < size:
                state, residual, jacobian = trial, r, j
                break
        else:
            raise ShootingFailure(f"Shooting stalled at closure residual {size:.3g}")
    size = np.linalg.norm(residual, ord=np.inf)
    if size <= tolerance:
        return state
    raise ShootingFailure(f"Shooting did not converge; closure residual {size:.3g}")


def spectrum(matrix, *, defect_limit):
    values = np.linalg.eigvals(matrix)
    values = sorted(values, key=lambda z: (abs(z), z.imag, z.real))
    defect = float(np.linalg.norm(matrix.T @ SYMPLECTIC @ matrix - SYMPLECTIC, ord=np.inf))
    rho = float(max(map(abs, values)))
    separation = min(abs(values[i] - values[j]) for i in range(4) for j in range(i))
    if defect > defect_limit:
        label = "numerical_unresolved"
    elif rho > 1.0002:
        label = "linearly_unstable"
    elif max(abs(abs(z) - 1) for z in values) > 2e-6 or separation < 1e-4:
        label = "near_unit_or_degenerate"
    else:
        label = "linearly_elliptic"
    return dict(classification=label, spectral_radius=rho, symplectic_defect=defect,
                multipliers=[[z.real, z.imag] for z in values])


def shortest_period(states, samples, tolerance):
    """Detect a return at a smaller positive integer RF count."""
    for k, state in enumerate(states[samples::samples], 1):
        if np.linalg.norm(state - states[0], ord=np.inf) < tolerance:
            return k
    return None
