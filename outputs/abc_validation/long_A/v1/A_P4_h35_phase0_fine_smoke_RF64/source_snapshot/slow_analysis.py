"""Centered filters and RF-phase samples for vector trajectories."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _series(values, time):
    values, time = np.asarray(values, float), np.asarray(time, float)
    if time.ndim != 1 or values.ndim != 2 or len(values) != len(time) or len(time) < 3:
        raise ValueError("Expected time (N,) and vector samples (N, d)")
    delta = np.diff(time)
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)) or np.any(delta <= 0):
        raise ValueError("Samples must be finite and times strictly increasing")
    if not np.allclose(delta, delta[0], rtol=1e-8, atol=abs(delta[0])*1e-10):
        raise ValueError("Centered filters require a uniform time grid")
    return values, time, float(delta[0])


def centered_period_mean(values, time, samples_per_period):
    """Trapezoidal window average; labels are the window centres.

    samples_per_period is the number of intervals in the chosen window; use
    2*N or 3*N for two/three RF-period windows. Endpoint extrapolation is absent.
    """
    values, time, _ = _series(values, time)
    n = int(samples_per_period)
    if n != samples_per_period or n < 2 or n % 2 or len(time) <= n:
        raise ValueError("Need even window interval count and more than one window")
    increments = (values[1:]+values[:-1])*(np.diff(time)/2)[:, None]
    integral = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(increments, axis=0)))
    mean = (integral[n:]-integral[:-n])/(time[n:]-time[:-n])[:, None]
    return time[n//2:len(time)-n//2], mean


def local_rf_trend(values, time, samples_per_period, window_periods=2):
    """Local quadratic trend plus RF quadratures and their linear drift.

    Fit [1,u,u²,cos(2πu),sin(2πu),u*cos(2πu),u*sin(2πu)], u=(t-tc)/T_RF.
    Return its trend value at the centre; do not fit or shift a reference path.
    This is a filter sensitivity diagnostic, not an exact secular transform.
    """
    values, time, _ = _series(values, time)
    N, periods = int(samples_per_period), int(window_periods)
    n = N*periods
    if N != samples_per_period or periods != window_periods or N < 16 or N % 2 or periods < 1 or len(time) <= n:
        raise ValueError("Need an even RF interval count >=16 and a positive integer window")
    u = (np.arange(n+1)-n/2)/N
    co, si = np.cos(2*np.pi*u), np.sin(2*np.pi*u)
    design = np.column_stack((np.ones(n+1), u, u*u, co, si, u*co, u*si))
    weights = np.linalg.pinv(design)[0]
    trend = np.column_stack([np.convolve(values[:, j], weights[::-1], mode="valid") for j in range(values.shape[1])])
    return time[n//2:len(time)-n//2], trend


def poincare_samples(time, states, frequency, phase=0., target_phase=0.):
    """Sample at Ω*t+phase=target_phase mod 2π using cubic interpolation.

    These are phase-locked full states, not automatically slow coordinates.
    No extrapolation is used; interpolation accuracy must be checked separately.
    """
    time, states = np.asarray(time, float), np.asarray(states, float)
    if time.ndim != 1 or states.ndim != 2 or len(time) != len(states) or len(time) < 4:
        raise ValueError("Expected at least four time/state samples")
    if not np.all(np.isfinite([frequency, phase, target_phase])) or frequency <= 0:
        raise ValueError("Expected positive frequency and finite phases")
    if np.any(np.diff(time) <= 0) or not np.all(np.isfinite(states)) or not np.all(np.isfinite(time)):
        raise ValueError("Expected finite ordered samples")
    omega = 2*np.pi*frequency
    first = int(np.ceil((omega*time[0]+phase-target_phase)/(2*np.pi)-1e-12))
    last = int(np.floor((omega*time[-1]+phase-target_phase)/(2*np.pi)+1e-12))
    sample_time = (target_phase-phase+2*np.pi*np.arange(first, last+1))/omega
    sample_time = np.clip(sample_time, time[0], time[-1])
    return sample_time, CubicSpline(time, states, axis=0, extrapolate=False)(sample_time)
