"""Unaligned position errors and diagnostic turning-line orientation."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks


def position_errors(actual, reference, scale):
    """Fixed-scale errors; radial agreement never substitutes vector agreement."""
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or scale <= 0:
        raise ValueError("Expected equal (N,2) positions and a positive fixed scale")
    delta = (a-b)/scale
    return dict(vector=np.linalg.norm(delta, axis=1), x=delta[:, 0], y=delta[:, 1],
                radial=(np.linalg.norm(a, axis=1)-np.linalg.norm(b, axis=1))/scale)


def turning_orientations(time, position, expected_period):
    """Line angle modulo pi, unwrapped across radial turning maxima.

    Fitting angle against time is an explanatory diagnostic only. Neither the
    physical reference trajectory nor its clock is fitted/rotated/aligned.
    """
    t, xy = np.asarray(time), np.asarray(position)
    radius = np.linalg.norm(xy, axis=1)
    distance = max(1, int(.3*expected_period/np.median(np.diff(t))))
    ids, _ = find_peaks(radius, distance=distance, prominence=.2*np.ptp(radius))
    if len(ids) < 3:
        raise ValueError("Too few interior radial turning points for orientation")
    spline = CubicSpline(t, xy, axis=0)
    times, positions = [], []
    for i in ids:
        result = minimize_scalar(lambda clock: -float(np.linalg.norm(spline(clock))),
                                 bounds=(t[i-1], t[i+1]), method="bounded", options={"xatol": 1e-13})
        if not result.success:
            raise ValueError("Radial turning-point refinement failed")
        times.append(float(result.x))
        positions.append(spline(result.x))
    times, positions = np.array(times), np.array(positions)
    radii = np.linalg.norm(positions, axis=1)
    angles = .5*np.unwrap(2*np.arctan2(positions[:, 1], positions[:, 0]))
    angles -= np.pi*np.rint(angles[0]/np.pi)
    slope, intercept = np.polyfit(times, angles, 1)
    residual = angles-(intercept+slope*times)
    summary = dict(turning_points=len(times), first_time_s=float(times[0]), last_time_s=float(times[-1]),
                   first_orientation_deg=float(np.rad2deg(angles[0])), last_orientation_deg=float(np.rad2deg(angles[-1])),
                   observed_orientation_change_deg=float(np.rad2deg(angles[-1]-angles[0])),
                   fitted_rate_rad_per_s=float(slope), fitted_rate_deg_per_s=float(np.rad2deg(slope)),
                   fitted_intercept_deg=float(np.rad2deg(intercept)),
                   angular_fit_residual_rms_deg=float(np.rad2deg(np.sqrt(np.mean(residual**2)))),
                   radial_turning_min_m=float(radii.min()), radial_turning_max_m=float(radii.max()),
                   radial_turning_span_m=float(np.ptp(radii)),
                   diagnostic_only="Angle fit does not modify any A/B/C comparison")
    return times, positions, angles, summary
