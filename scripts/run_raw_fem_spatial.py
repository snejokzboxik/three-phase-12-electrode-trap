"""Diagnostic step 3: separate 70/50/35 um FEMs and common-state trajectories.

Run mesh levels in separate processes to release sparse-solver memory. Old
70 um caches, pilots and archives are strictly read-only. Analysis is in
analyze_raw_fem_spatial.py; numeric inputs are frozen before any new solves.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.config import load_config
from three_phase_trap.fem_cache import load_fem_cache, save_fem_cache, cache_paths, fem_fingerprint
from three_phase_trap.fem import solve_basis, laplace_form
from three_phase_trap.geometry import build_geometry
from three_phase_trap.gmsh_mesh import generate_mesh_artifacts, configured_mesh
from three_phase_trap.rf_reconstruction import fit_harmonic_quadratures, RFReconstruction
from three_phase_trap.validation_field_audit import rf_potential_quadratures
from three_phase_trap.spatial_convergence import sample_raw, sample_smooth, sample_jumps, sample_origin_limits
from three_phase_trap.averaged_dynamics import AveragedRFModel, integrate_averaged
from three_phase_trap.pilot_dynamics import AffineP2RF, PolynomialRF, integrate_rf
from three_phase_trap.audited_rf import integrate_audited
from run_abc_pilot import sha, write

BASE = ROOT/"outputs/abc_validation"
OUTPUT = BASE/"raw_spatial_diagnosis/v2"


def prepare():
    import scipy
    import skfem
    import gmsh
    protocol = json.loads((BASE/"protocol.json").read_text(encoding="utf-8"))
    pilot = json.loads((BASE/"pilot/settings.json").read_text(encoding="utf-8"))
    restart = json.loads((BASE/"raw_restart_diagnosis/v1/settings.json").read_text(encoding="utf-8"))
    inputs = ["outputs/fem_cache/p2_basis.npz", "outputs/fem_cache/metadata.json", "outputs/fem_cache/gmsh_70um_fem_bundle.zip",
              "outputs/abc_validation/protocol.json", "outputs/abc_validation/initial_conditions.json",
              "outputs/abc_validation/model_b/reconstruction.json", "outputs/abc_validation/model_b/settings.json",
              "outputs/abc_validation/raw_restart_diagnosis/v1/settings.json", "outputs/abc_validation/pilot/settings.json",
              "outputs/abc_validation/raw_edge_audit/v2/A_phase0_fine_pilot.npz"]
    for tag in ("A_smooth_phase0_coarse_pilot", "A_smooth_phase0_fine_pilot", "B_phase0_coarse_pilot", "B_phase0_fine_pilot"):
        inputs += [f"outputs/abc_validation/pilot/{tag}{suffix}" for suffix in (".npz", ".json")]
    for control in ("dop200", "dop400"):
        inputs += [f"outputs/abc_validation/raw_restart_diagnosis/v1/rapid_{control}_common{suffix}" for suffix in (".npz", ".json")]
    kernels = ("spatial_convergence.py", "fem.py", "gmsh_mesh.py", "geometry.py", "mesh.py", "fem_cache.py",
               "interpolation.py", "validation_field_audit.py", "rf_reconstruction.py", "averaged_dynamics.py",
               "pilot_dynamics.py", "audited_rf.py", "slow_analysis.py")
    settings = {"diagnostic_step": 3, "meshes_um": [70, 50, 35], "physical": protocol["pilot"],
                "outer_resolution_scales_with_h": True, "geometry_order": 1, "solution_order": 2,
                "training_settings": json.loads((BASE/"model_b/settings.json").read_text(encoding="utf-8")),
                "orders": [40, 48, 56], "selected_order": 48,
                "holdout_radii_m": (np.array([.025, .05, .1, .25, .5, 1, 2, 4, 6, 7, 8])*1e-3).tolist(),
                "holdout_count": 513, "holdout_offset": .739,
                "trajectory_controls": pilot["controls"], "pilot_periods": pilot["RF_periods"]["pilot"],
                "estimated_slow_period_s": pilot["estimated_slow_period_s"], "raw_window": restart["windows"]["rapid"],
                "raw_controls": {k: restart["controls"][k] for k in ("dop200", "dop400")},
                "samples_per_RF_period": 64, "R_scale_m": .001,
                "coordinate_budgets": {"relative_rms": .001, "relative_max": .003},
                "field_comparison_target": .001, "target_scope": "0.1% common-point relative RMS screening, not a bound on trajectory error or FEM truth",
                "input_sha256": {n: sha(ROOT/n) for n in inputs},
                "config_sha256": {str(h): sha(ROOT/f"configs/gmsh_{h}um.toml") for h in (70, 50, 35)},
                "kernel_sha256": {n: sha(ROOT/"src/three_phase_trap"/n) for n in kernels},
                "worker_sha256": {f.__name__: __import__("hashlib").sha256(inspect.getsource(f).encode()).hexdigest()
                                  for f in (field_level, trajectory_level)},
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                             "skfem": skfem.__version__, "gmsh": gmsh.__version__},
                "scope": "Three nonnested affine-geometry P2 meshes, full RF and averaged smooth pilots phase 0; raw local common-state restart only. No global raw-A convergence claim."}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/"settings.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Frozen study changed; preserve v1 and use a new version")
    if not path.exists():
        write(path, settings)
    return settings


def points_on_rings(radii, count, offset):
    angles = (np.arange(count)+offset)*2*np.pi/count
    return np.concatenate([r*np.column_stack((np.cos(angles), np.sin(angles))) for r in radii])


def check_complete(meta_path, settings):
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["settings_sha256"] != sha(OUTPUT/"settings.json"):
        raise ValueError("Settings changed for existing checkpoint")
    for name, value in meta["artifacts"].items():
        if sha(ROOT/name) != value:
            raise ValueError(f"Corrupt checkpoint artifact {name}")
    return meta["status"] == "completed"


def complete(meta_path, files, **extra):
    write(meta_path, {"status": "completed", "settings_sha256": sha(OUTPUT/"settings.json"),
                     "driver_sha256": sha(Path(__file__)),
                     "artifacts": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}, **extra})


def field_level(h, settings):
    from skfem import asm
    directory = OUTPUT/f"h{h}"
    directory.mkdir(exist_ok=True)
    meta_path = directory/"field_complete.json"
    if check_complete(meta_path, settings):
        print(f"h={h}: reusing complete field checkpoint", flush=True)
        return
    started = time.perf_counter()
    config = load_config(ROOT/f"configs/gmsh_{h}um.toml")
    mesh_path = ROOT/config.mesh.mesh_file
    if h == 70 and not mesh_path.with_suffix(".json").exists():
        raise ValueError("The original 70um mesh must already exist")
    if not mesh_path.with_suffix(".json").exists():
        print(f"h={h}: generating separate Gmsh mesh", flush=True)
        generate_mesh_artifacts(config, ROOT)
    else:
        # Validate existing topology/tags/settings/hash without replacing it.
        geometry = build_geometry(config.geometry.electrode_count, config.geometry.electrode_orbit_radius,
                                  config.geometry.electrode_radius, config.geometry.outer_domain_radius)
        trap = configured_mesh(geometry, config.mesh, ROOT)
        del trap
        gc.collect()
    arrays, metadata = cache_paths(ROOT, config)
    if not arrays.exists() or not metadata.exists():
        if h == 70:
            raise ValueError("Never implicitly rebuild the original FEM basis")
        if arrays.exists() != metadata.exists():
            raise ValueError("Incomplete new FEM cache; inspect before retry")
        geometry = build_geometry(config.geometry.electrode_count, config.geometry.electrode_orbit_radius,
                                  config.geometry.electrode_radius, config.geometry.outer_domain_radius)
        trap = configured_mesh(geometry, config.mesh, ROOT)
        print(f"h={h}: solving four P2 Laplace bases, {len(trap.triangles)} triangles", flush=True)
        solutions = solve_basis(trap)
        save_fem_cache(config, ROOT, solutions)
        del solutions, trap
        gc.collect()
    print(f"h={h}: loading and checking saved basis", flush=True)
    _, trap, solutions, interp = load_fem_cache(config, ROOT)
    # Check the discrete PDE and continuity identities separately from mesh error.
    stiffness = asm(laplace_form, solutions.basis)
    boundary = solutions.basis.get_dofs().all()
    free = np.ones(solutions.basis.N, bool)
    free[boundary] = False
    matrix_norm = float(np.asarray(abs(stiffness).sum(axis=1)).max())
    residuals = {p: float(np.max(abs((stiffness@v)[free]))/matrix_norm) for p, v in solutions.potentials.items()}
    dc_identity = float(np.max(abs(sum(solutions.potentials[p] for p in ("A", "B", "C"))-solutions.potentials["DC"])))
    if max(residuals.values()) > 1e-10 or dc_identity > 1e-9:
        raise ValueError("Discrete Laplace basis consistency failed")
    del stiffness
    gc.collect()
    p = settings["physical"]
    omega = 2*np.pi*p["frequency_Hz"]
    ts = settings["training_settings"]
    train = points_on_rings(ts["training_radii_m"], ts["training_angles"], ts["training_angle_offset"])
    pc, ps = rf_potential_quadratures(interp, train, 1.)
    models, conditions, files = {}, {}, [arrays, metadata, mesh_path, mesh_path.with_suffix(".json")]
    fp = fem_fingerprint(config)
    for order in settings["orders"]:
        q, condition = fit_harmonic_quadratures(train, pc, ps, order, ts["reference_radius_m"])
        reconstruction = RFReconstruction(q, ts["computational_domain_radius_m"], {"fem_fingerprint": fp, "basis_sha256": sha(arrays), "order": order, "all_modes_retained": True})
        path = directory/f"reconstruction_N{order}.json"
        reconstruction.save(path)
        models[order], conditions[str(order)] = reconstruction, condition
        files.append(path)
    train_alt = points_on_rings(ts["alternative_training_radii_m"], ts["alternative_training_angles"], ts["alternative_training_angle_offset"])
    ac, ass = rf_potential_quadratures(interp, train_alt, 1.)
    aq, alt_condition = fit_harmonic_quadratures(train_alt, ac, ass, 48, ts["reference_radius_m"])
    alt = RFReconstruction(aq, ts["computational_domain_radius_m"], {"fem_fingerprint": fp, "alternative_training": True})
    alt_path = directory/"alternative_N48.json"
    alt.save(alt_path)
    files.append(alt_path)
    points = points_on_rings(settings["holdout_radii_m"], settings["holdout_count"], settings["holdout_offset"])
    with np.load(BASE/"raw_edge_audit/v2/A_phase0_fine_pilot.npz") as history:
        path_points, path_time = history["states"][::64, :2], history["time_s"][::64]
    queries = np.vstack((points, path_points))
    data = {"points_m": queries, "ring_points_count": len(points), "path_time_s": path_time}
    data.update({"raw_"+k: v for k, v in sample_raw(interp, queries, p["U0_V"], p["charge_C"], p["mass_kg"], omega).items()})
    for label, reconstruction in [("N"+str(n), model) for n, model in models.items()]+[("alternative", alt)]:
        data.update({label+"_"+k: v for k, v in sample_smooth(reconstruction.scaled(p["U0_V"]), queries, p["charge_C"], p["mass_kg"], omega).items()})
    sample_path = directory/"common_samples.npz"
    np.savez_compressed(sample_path, **data)
    jumps_path = directory/"facet_samples.npz"
    np.savez_compressed(jumps_path, **sample_jumps(interp, p["U0_V"]))
    files.extend((sample_path, jumps_path))
    origin_path = directory/"origin_limits.npz"
    np.savez_compressed(origin_path, **sample_origin_limits(interp, p["U0_V"]))
    files.append(origin_path)
    original_fit_error = None
    if h == 70:
        old = RFReconstruction.load(BASE/"model_b/reconstruction.json", fp)
        original_fit_error = float(np.max(abs(old.scaled(p["U0_V"]).field_quadratures(points)[0]-data["N48_field"][:len(points), :, 0])))
        if original_fit_error > 1e-7:
            raise ValueError("Repeated 70um reconstruction differs from original model B")
    complete(meta_path, files, mesh_um=h, fingerprint=fp, basis_sha256=sha(arrays), basis_path=str(arrays.relative_to(ROOT)).replace("\\", "/"),
             triangles=len(trap.triangles), dofs=int(solutions.basis.N), condition=conditions, alternative_condition=alt_condition,
             discrete_scaled_residuals=residuals, DC_superposition_max_V=dc_identity,
             repeated_70um_field_max_difference_V_per_m=original_fit_error, wall_seconds=time.perf_counter()-started)
    print(f"h={h}: field and reconstruction diagnostics complete in {time.perf_counter()-started:.1f}s", flush=True)


def trajectory_level(h, settings):
    directory = OUTPUT/f"h{h}"
    if not check_complete(directory/"field_complete.json", settings):
        raise ValueError("Field stage must be completed first")
    if check_complete(directory/"trajectories_complete.json", settings):
        print(f"h={h}: reusing trajectory checkpoints", flush=True)
        return
    p = settings["physical"]
    ic = json.loads((BASE/"initial_conditions.json").read_text(encoding="utf-8"))["cases"][0]["A_and_A_smooth_full_state"]
    rf = RFReconstruction.load(directory/"reconstruction_N48.json")
    b = AveragedRFModel(rf, p["charge_C"], p["mass_kg"], p["U0_V"], p["frequency_Hz"])
    smooth = PolynomialRF(b.quadratures)
    end = settings["pilot_periods"]/b.frequency
    local = settings["raw_window"]
    local_ic = np.array(local["initial_states"]["common"])
    records = {}
    files = []
    def save(tag, t, state, metadata):
        path = directory/(tag+".npz")
        if not np.all(np.isfinite(state)) or state.shape != (len(t), 4):
            raise ValueError("Invalid trajectory output")
        np.savez_compressed(path, time_s=t, states=state)
        files.append(path)
        records[tag] = metadata
    for level, control in settings["trajectory_controls"].items():
        for model in ("A_smooth", "B"):
            tag = model+"_"+level
            print(f"h={h}: {tag}, full phase-0 pilot", flush=True)
            if h == 70:
                source = BASE/f"pilot/{model}_phase0_{level}_pilot.npz"
                info = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
                if info["status"] != "completed" or info["arrays_sha256"] != sha(source):
                    raise ValueError("Invalid original pilot")
                with np.load(source) as data:
                    save(tag, data["time_s"], data["states"], {"reused": str(source.relative_to(ROOT)), "source_sha256": sha(source)})
                continue
            if model == "B":
                result = integrate_averaged(b, p["slow_position_m"], p["slow_velocity_m_per_s"], end,
                                            samples=settings["pilot_periods"]*64+1, rtol=control["slow_rtol"], atol=control["slow_dimensionless_atol"],
                                            max_step=settings["estimated_slow_period_s"]/control["slow_steps_per_period"])
                if result.status != "completed":
                    raise ValueError("Averaged pilot did not complete")
                save(tag, result.time, result.states, {"status": result.status, "nfev": result.evaluations})
            else:
                result = integrate_rf(smooth, ic, b.charge, b.mass, b.frequency, p["phase_rad"][0], 0., end,
                                      max_steps_per_period=control["RF_steps_per_period"], rtol=control["RF_rtol"], atol=control["RF_atol_SI"])
                if result.status != "completed" or not np.array_equal(result.states[0], ic):
                    raise ValueError("Smooth full-RF pilot invalid")
                save(tag, result.time, result.states, {"status": result.status, "nfev": result.nfev})
    # Smooth common-state restarts use the identical physical RF state/clock.
    for steps in (100, 200):
        result = integrate_rf(smooth, local_ic, b.charge, b.mass, b.frequency, p["phase_rad"][0], local["start_s"], local["end_s"],
                              max_steps_per_period=steps, rtol=2e-11 if steps == 100 else 2e-13,
                              atol=[2e-15, 2e-15, 2e-13, 2e-13] if steps == 100 else [2e-17, 2e-17, 2e-15, 2e-15])
        if result.status != "completed" or not np.array_equal(result.states[0], local_ic):
            raise ValueError("Smooth common-state restart failed")
        save(f"local_smooth_{steps}", result.time, result.states, {"status": result.status, "nfev": result.nfev})
    raw = None
    for label, control in settings["raw_controls"].items():
        print(f"h={h}: raw common-state restart {label}", flush=True)
        if h == 70:
            source = BASE/f"raw_restart_diagnosis/v1/rapid_{label}_common.npz"
            info = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
            if info["status"] != "completed" or sha(source) != info["arrays_sha256"]:
                raise ValueError("Invalid original restart")
            with np.load(source) as data:
                save("local_raw_"+label, data["time_s"], data["states"], {"reused": str(source.relative_to(ROOT)), "source_sha256": sha(source)})
            continue
        if raw is None:
            _, _, _, interp = load_fem_cache(load_config(ROOT/f"configs/gmsh_{h}um.toml"), ROOT)
            raw = AffineP2RF(interp, b.amplitude)
        result = integrate_audited(raw, local_ic, b.charge, b.mass, b.frequency, p["phase_rad"][0], local["start_s"], local["end_s"],
                                   max_steps_per_period=control["steps"], rtol=control["rtol"], atol=control["atol"])
        if result.run.status != "completed" or not np.array_equal(result.run.states[0], local_ic):
            raise ValueError("Raw common-state restart failed")
        save("local_raw_"+label, result.run.time, result.run.states, {"status": result.run.status, "diagnostics": result.diagnostics, "nfev": result.run.nfev})
    complete(directory/"trajectories_complete.json", files, records=records, physical=p,
             full_initial_state_common_to_all_meshes=ic, local_initial_state_common_to_all_meshes=local_ic.tolist(),
             local_start_time_s=local["start_s"], local_end_time_s=local["end_s"])
    print(f"h={h}: trajectories complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "fields", "field-level", "trajectories", "trajectory-level"), default="all")
    parser.add_argument("--mesh", type=int, choices=(70, 50, 35))
    args = parser.parse_args()
    settings = prepare()
    if args.stage.endswith("-level"):
        if args.mesh is None:
            raise ValueError("--mesh is required for a single-level worker")
        (field_level if args.stage == "field-level" else trajectory_level)(args.mesh, settings)
        return
    for stage in ("fields", "trajectories"):
        if args.stage not in ("all", stage):
            continue
        for h in settings["meshes_um"]:
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            subprocess.run([sys.executable, str(Path(__file__)), "--stage", "field-level" if stage == "fields" else "trajectory-level", "--mesh", str(h)],
                           cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
