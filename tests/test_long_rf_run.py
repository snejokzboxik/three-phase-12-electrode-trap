import json
from types import SimpleNamespace

import numpy as np
import pytest

from three_phase_trap.long_rf_run import RunStore, run_case, run_lock, verify_completion, write_json


def settings():
    return dict(RF_periods=7, chunk_RF_periods=2, samples_per_RF_period=64,
                initial_state=[.001, .0002, .0001, -.00001], phase_rad=.42,
                physical=dict(charge_C=1., mass_kg=1., frequency_Hz=10.),
                control=dict(method="DOP853", steps=100, rtol=2e-11, atol=[1e-12]*4),
                reference_slow_period_s=.2, R_scale_m=.001, mesh_um=35, smoke_test=False)


def exact_states(t, initial, frequency, phase, start=0.):
    omega, amplitude = 2*np.pi*frequency, 1e-4
    dt, angle, first = t-start, omega*t+phase, omega*start+phase
    out = np.tile(initial, (len(t), 1)).astype(float)
    out[:, 0] += initial[2]*dt+amplitude*((np.cos(first)-np.cos(angle))/omega**2-np.sin(first)*dt/omega)
    out[:, 2] += amplitude*(np.sin(angle)-np.sin(first))/omega
    out[:, 1] += initial[3]*dt
    out[0] = initial
    return out


def integrator(calls, interrupt_at=None, status="completed"):
    def solve(field, initial, charge, mass, frequency, phase, start, end, **kwargs):
        calls.append((start, end, np.array(initial).copy(), phase))
        if interrupt_at is not None and len(calls) == interrupt_at:
            raise KeyboardInterrupt()
        t = np.linspace(start, end, round((end-start)*frequency*kwargs["samples_per_period"])+1)
        y = exact_states(t, initial, frequency, phase, start)
        if "progress" in kwargs:
            kwargs["progress"](dict(time_s=(start+end)/2))
        return SimpleNamespace(run=SimpleNamespace(time=t, states=y, status=status, wall_seconds=.001, nfev=10),
                               sample_triangles=np.zeros(len(t), np.int32), crossings=[],
                               diagnostics=dict(position_nudges=0, velocity_nudges=0))
    return solve


def test_interrupt_resume_preserves_full_initial_state_and_absolute_RF_phase(tmp_path):
    s, calls = settings(), []
    store = RunStore(tmp_path, s)
    with pytest.raises(KeyboardInterrupt):
        run_case(store, lambda: None, integrator=integrator(calls, interrupt_at=2), notify=lambda _: None, make_figures=False)
    assert not (tmp_path/"COMPLETED.json").exists()
    assert json.loads((tmp_path/"status.json").read_text())["status"] == "interrupted"
    first, _ = store.scan()
    assert len(first) == 1 and first[0]["RF_bounds"] == [0, 2]
    resumed = []
    run_case(store, lambda: None, integrator=integrator(resumed), notify=lambda _: None, make_figures=False)
    assert len(resumed) == 3
    assert np.array_equal(resumed[0][2], first[0]["final_state"])
    # Local clocks start at zero, with the unchanged phase; integer-cycle
    # boundaries must produce exactly the same physical global RF drive.
    assert all(c[0] == 0. and c[3] == s["phase_rad"] for c in calls+resumed)
    with np.load(tmp_path/"trajectory.npz") as data:
        t, y = data["time_s"], data["states"]
    expected = exact_states(t, np.array(s["initial_state"]), 10., s["phase_rad"])
    assert len(t) == 7*64+1 and t[0] == 0. and t[-1] == .7
    assert np.all(np.diff(t) > 0)
    assert np.array_equal(y[0], s["initial_state"])
    assert np.max(abs(y-expected)) < 1e-17
    assert verify_completion(store)["status"] == "completed"


def test_completed_case_is_verified_without_starting_another_integration(tmp_path):
    store = RunStore(tmp_path, settings())
    run_case(store, lambda: None, integrator=integrator([]), notify=lambda _: None, make_figures=False)
    def forbidden():
        pytest.fail("A completed run must not load a field or integrate again")
    marker = run_case(store, forbidden, notify=lambda _: None, make_figures=False)
    assert marker["status"] == "completed"


def test_corrupted_checkpoint_is_rejected_even_with_final_marker(tmp_path):
    store = RunStore(tmp_path, settings())
    run_case(store, lambda: None, integrator=integrator([]), notify=lambda _: None, make_figures=False)
    store.paths(0)[0].write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify_completion(store)


def test_changed_run_cannot_reuse_existing_checkpoints(tmp_path):
    original = settings()
    RunStore(tmp_path, original).freeze()
    changed = {**original, "phase_rad": .43}
    with pytest.raises(ValueError, match="settings/source"):
        RunStore(tmp_path, changed).freeze()
    assert json.loads((tmp_path/"settings.json").read_text()) == original


def test_os_lock_prevents_two_simultaneous_writers(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already active"):
            with run_lock(tmp_path):
                pytest.fail("Second writer acquired the same output directory")
    with run_lock(tmp_path):
        pass


def test_incomplete_native_result_never_gets_completion_marker(tmp_path):
    store = RunStore(tmp_path, settings())
    with pytest.raises(ValueError, match="did not complete"):
        run_case(store, lambda: None, integrator=integrator([], status="domain_exit"), notify=lambda _: None, make_figures=False)
    assert not (tmp_path/"COMPLETED.json").exists()
    assert not store.paths(0)[1].exists()
    assert json.loads((tmp_path/"status.json").read_text())["status"] == "failed"


def test_segmented_smoke_is_compared_with_one_shot_solution(tmp_path):
    s = {**settings(), "smoke_test": True}
    run_case(RunStore(tmp_path, s), lambda: None, integrator=integrator([]), notify=lambda _: None, make_figures=False)
    result = json.loads((tmp_path/"smoke_comparison.json").read_text())
    assert result["passed"] and result["relative_max"] < 1e-12


def test_transient_windows_reader_does_not_break_atomic_status_write(tmp_path, monkeypatch):
    from pathlib import Path
    original = Path.replace
    attempts = []
    def replace(path, target):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("Temporary reader holds the target")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    write_json(tmp_path/"status.json", {"status": "running"})
    assert len(attempts) == 2
    assert json.loads((tmp_path/"status.json").read_text()) == {"status": "running"}
