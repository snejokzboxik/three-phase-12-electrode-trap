"""Build direct RF FEM P2/P4 caches and verify exact cyclic reduction.

These are new direct FEM discretizations of the same boundary problem.
The legacy P2 caches and harmonic B/C remain immutable.
"""
from __future__ import annotations

import argparse
import inspect
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap import direct_fem
from three_phase_trap.direct_fem import solve_sector, SectorSolution, DirectFEMRF
from three_phase_trap.gmsh_mesh import load_gmsh_mesh
from three_phase_trap.rf_reconstruction import RFReconstruction, fit_harmonic_quadratures
from three_phase_trap.spatial_convergence import relative_rms
from run_abc_pilot import sha, write
from run_abc_model_b import ring_points
from analyze_abc_pilot import csv_write

BASE = ROOT/"outputs/abc_validation"
OUTPUT = BASE/"direct_fem_validation/v1"


def input_signature(h, degree):
    directory = ROOT/f"outputs/meshes/gmsh_{h}um"
    return {"mesh_um": h, "degree": degree, "full_mesh_sha256": sha(directory/"trap.msh"),
            "sector_mesh_sha256": sha(directory/"sector.msh"),
            "direct_module_sha256": sha(Path(direct_fem.__file__)),
            "skfem": __import__("skfem").__version__, "rho": [-.5, -np.sqrt(3)/2],
            "intorder": 2*degree, "geometry_order": 1,
            "boundary": "electrode H=1, outer H=0, H(R30*x)=rho*H(x), H(0)=0; direct gradient"}


def load_field(h, degree=4, amplitude=250.):
    d = OUTPUT/f"h{h}_P{degree}"
    meta = json.loads((d/"complete.json").read_text(encoding="utf-8"))
    if meta["signature"] != input_signature(h, degree):
        raise ValueError("Direct FEM cache source/mesh signature mismatch")
    for n, value in meta["artifacts"].items():
        if sha(d/n) != value:
            raise ValueError("Direct FEM artifact checksum mismatch")
    sector = SectorSolution.load(d/"sector_solution.npz")
    trap = load_gmsh_mesh(ROOT/f"outputs/meshes/gmsh_{h}um/trap.msh")
    return DirectFEMRF(sector, trap, amplitude)


def build(h, degree):
    d = OUTPUT/f"h{h}_P{degree}"
    d.mkdir(parents=True, exist_ok=True)
    sig = input_signature(h, degree)
    complete = d/"complete.json"
    if complete.exists():
        meta = json.loads(complete.read_text(encoding="utf-8"))
        if meta["signature"] != sig or not all(sha(d/n) == value for n, value in meta["artifacts"].items()):
            raise ValueError("Existing direct FEM checkpoint differs; preserve and use a new version")
        print(f"h{h} P{degree}: reusing complete native FEM", flush=True)
        return
    print(f"h{h} P{degree}: direct complex variational solve", flush=True)
    started = time.perf_counter()
    native_path = d/"sector_solution.npz"
    sector = solve_sector(ROOT/f"outputs/meshes/gmsh_{h}um/sector.msh", degree)
    sector.save(native_path)
    write(d/"solver_diagnostics.json", sector.diagnostics)
    shutil.copyfile(Path(direct_fem.__file__), d/"direct_fem_source.py")
    trap = load_gmsh_mesh(ROOT/f"outputs/meshes/gmsh_{h}um/trap.msh")
    field = DirectFEMRF(sector, trap, 250.)
    reference_path = BASE/"raw_spatial_diagnosis/v2"/f"h{h}/common_samples.npz"
    with np.load(reference_path) as data:
        points = data["points_m"]
        original_field = data["raw_field"]
        original_phi = data["raw_phi"]
    phi, ec, es = field.sample(points)
    native_fields = np.stack((ec, es), axis=-1)
    native_phi = np.stack((phi.imag, phi.real), axis=-1)
    # Compare to native P2, not to a reconstructed harmonic field.
    if degree == 2:
        max_field = float(np.max(abs(native_fields-original_field)))
        relative_field = relative_rms(native_fields, original_field)
        if max_field > 1e-5 or relative_field > 1e-8:
            raise ValueError(f"Cyclic reduction does not reproduce full P2: {max_field}, {relative_field}")
        write(d/"full_P2_equivalence.json", {"field_max_abs_V_per_m": max_field, "field_relative_rms": relative_field,
                                           "phi_max_abs_V": float(np.max(abs(native_phi-original_phi))),
                                           "source_samples_sha256": sha(reference_path),
                                           "status": "passed", "scope": "same Galerkin space and BC, different floating-point solve; no fit"})
    radii = np.array([.025, .05, .1, .25, .5, 1, 2, 4, 6, 7, 8])*1e-3
    train = ring_points([.0005, .0015, .003, .0045, .006, .0075], 256, .317)
    train_phi = field.sample(train)[0]/250.
    q, condition = fit_harmonic_quadratures(train, train_phi.imag, train_phi.real, 48, .008)
    reconstruction = RFReconstruction(q, .008, {"direct_FEM": True, "mesh_um": h, "degree": degree,
                                              "sector_solution_sha256": sha(native_path), "all_modes_retained": True,
                                              "design_condition": condition, "not_used_for_A_force": True})
    reconstruction.save(d/"diagnostic_N48.json")
    smooth_c, smooth_s = reconstruction.scaled(250.).field_quadratures(points)
    ref_field = np.stack((smooth_c, smooth_s), axis=-1)
    native_psi = 1e-24*np.sum(native_fields**2, axis=(1, 2))/(4e-15*(2*np.pi*1e5)**2)
    np.savez_compressed(d/"common_samples.npz", points_m=points, phi=native_phi, field=native_fields,
                        psi=native_psi, diagnostic_smooth_field=ref_field)
    rows = []
    for i, radius in enumerate(radii):
        cut = slice(i*513, (i+1)*513)
        rows.append({"radius_m": float(radius), "native_fit_field_relative_rms": relative_rms(native_fields[cut], ref_field[cut]),
                     "native_RF_rms_V_per_m": float(np.sqrt(np.mean(np.sum(native_fields[cut]**2, axis=(1, 2)))/2))})
    csv_write(d/"native_reconstruction_discrepancy.csv", rows)
    at_origin = np.flatnonzero(np.any(np.linalg.norm(field.points[field.triangles], axis=-1) < 1e-12, axis=1))
    origin = np.array([field.local_quadratures(*field.local_coordinates(int(i), 0., 0.)) for i in at_origin])
    origin_rms = np.sqrt(np.sum(origin**2, axis=1)/2)
    write(d/"origin_limits.json", {"max_one_sided_RF_rms_V_per_m": float(origin_rms.max()),
                                  "triangle_ids": at_origin.tolist(), "quadratures": origin.tolist()})
    meta = {"signature": sig, "diagnostics": sector.diagnostics, "full_mapping_max_error_m": field.mapping_error_m,
            "wall_seconds": time.perf_counter()-started,
            "artifacts": {p.name: sha(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != "complete.json"}}
    write(complete, meta)
    print(json.dumps({"h": h, "degree": degree, "seconds": meta["wall_seconds"], "origin_RF_rms": float(origin_rms.max()),
                      "relative_field_fit_1mm": rows[5]["native_fit_field_relative_rms"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, choices=(70, 50, 35), default=70)
    parser.add_argument("--degree", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    build(args.mesh, args.degree)


if __name__ == "__main__":
    main()
