import numpy as np

from three_phase_trap.geometry import build_geometry, phase_for_electrode
from three_phase_trap.voltages import PHASE_OFFSETS, phase_periodicity_error, phase_voltages, phase_voltages_with_dc, voltage_matrix


def test_geometry_has_twelve_equidistant_electrodes():
    geometry = build_geometry()
    assert len(geometry.electrodes) == 12
    angles = np.unwrap(np.array([e.angle for e in geometry.electrodes]))
    assert np.allclose(np.diff(angles), np.pi / 6.0)
    centers = np.array([e.center for e in geometry.electrodes])
    assert np.allclose(np.linalg.norm(centers, axis=1), geometry.electrode_orbit_radius)


def test_phase_mapping_is_exact_abc_repeated():
    assert [phase_for_electrode(i) for i in range(12)] == list("ABCABCABCABC")
    assert [i for i in range(12) if phase_for_electrode(i) == "A"] == [0, 3, 6, 9]
    assert [i for i in range(12) if phase_for_electrode(i) == "B"] == [1, 4, 7, 10]
    assert [i for i in range(12) if phase_for_electrode(i) == "C"] == [2, 5, 8, 11]


def test_rf_phase_offsets_are_120_degrees_apart():
    assert np.isclose((PHASE_OFFSETS["A"] - PHASE_OFFSETS["B"]) % (2 * np.pi), 2 * np.pi / 3)
    assert np.isclose((PHASE_OFFSETS["B"] - PHASE_OFFSETS["C"]) % (2 * np.pi), 2 * np.pi / 3)


def test_three_phase_identity_and_periodicity():
    times = np.linspace(0.0, 2.0e-3, 1001)
    values = voltage_matrix(times, 1000.0, 10000.0)
    assert np.max(np.abs(values.sum(axis=1))) < 1.0e-10
    assert phase_periodicity_error(times, 1000.0, 10000.0) < 1.0e-10


def test_common_dc_bias_preserves_rf_waveforms():
    t = 1.7e-5
    rf = phase_voltages(t, 1000.0, 10000.0)
    biased = phase_voltages_with_dc(t, 300.0, 1000.0, 10000.0)
    assert all(np.isclose(float(biased[p]) - 300.0, float(rf[p])) for p in "ABC")
