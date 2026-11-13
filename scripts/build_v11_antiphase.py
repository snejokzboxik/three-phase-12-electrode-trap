"""Independent P4 electrostatic solution for alternating 12-electrode RF drive.

Run one mesh at a time: python scripts/build_v11_antiphase.py --mesh 50
The 30-degree sector has electrode potential +1 V, grounded outer circle,
and anti-periodic seams. Its twelve copies therefore have + - + - ... BCs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap import direct_fem
from three_phase_trap.direct_fem import SectorSolution, vandermonde, exponents
from three_phase_trap.rf_reconstruction import RFReconstruction, fit_harmonic_quadratures

OUT = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_sector(sector, xy, with_field=False):
    """Evaluate the independent FEM potential, rotating queries into one sector."""
    xy = np.asarray(xy, float)
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    k = np.floor((angle + np.pi / 12) / (np.pi / 6)).astype(int) % 12
    a = k * np.pi / 6
    local = np.column_stack((xy[:, 0]*np.cos(a)+xy[:, 1]*np.sin(a),
                             -xy[:, 0]*np.sin(a)+xy[:, 1]*np.cos(a)))
    finder = mtri.Triangulation(sector.points[:, 0], sector.points[:, 1], sector.triangles).get_trifinder()
    ids = np.asarray(finder(local[:, 0], local[:, 1]), int)
    if np.any(ids < 0):
        raise ValueError("Sampling point outside FEM vacuum")
    vertices = sector.points[sector.triangles[ids]]
    uv = np.linalg.solve(np.swapaxes(vertices[:, 1:]-vertices[:, [0]], 1, 2),
                         (local-vertices[:, 0])[..., None])[..., 0]
    coefficients = sector.coefficients[ids]
    potential = np.sum(coefficients*vandermonde(uv, sector.degree), axis=1)
    if not with_field:
        return ((-1.)**k * potential).real
    powers = exponents(sector.degree)
    du = np.zeros(len(xy), complex)
    dv = np.zeros(len(xy), complex)
    for j, (px, py) in enumerate(powers):
        if px:
            du += coefficients[:, j]*px*uv[:, 0]**(px-1)*uv[:, 1]**py
        if py:
            dv += coefficients[:, j]*py*uv[:, 0]**px*uv[:, 1]**(py-1)
    inverse = np.linalg.inv(np.swapaxes(vertices[:, 1:]-vertices[:, [0]], 1, 2))
    local_gradient = np.einsum("ni,nij->nj", np.column_stack((du, dv)), inverse)
    global_gradient = np.column_stack((local_gradient[:, 0]*np.cos(a)-local_gradient[:, 1]*np.sin(a),
                                        local_gradient[:, 0]*np.sin(a)+local_gradient[:, 1]*np.cos(a)))
    return ((-1.)**k * potential).real, (-((-1.)**k)[:, None]*global_gradient).real


def ring_points(radii, count, offset):
    angles = 2*np.pi*(np.arange(count)+offset)/count
    return np.concatenate([np.column_stack((r*np.cos(angles), r*np.sin(angles))) for r in radii])


def build(mesh):
    folder = OUT / "fields" / f"h{mesh}_P4_C"
    folder.mkdir(parents=True, exist_ok=True)
    mesh_path = ROOT / f"outputs/meshes/gmsh_{mesh}um/sector.msh"
    sector_path = folder / "sector_solution.npz"
    if sector_path.exists():
        sector = SectorSolution.load(sector_path)
    else:
        # This process uses a different cyclic character than the ABC solver.
        # Keep the validated V10 module byte-for-byte unchanged, including its
        # original cache fingerprint; this isolated process never loads ABC FEM.
        direct_fem.RHO = -1.+0j
        sector = direct_fem.solve_sector(mesh_path, degree=4)
        sector.diagnostics["RF_encoding"] = "Phi/U0=sin(chi)*H(x), H(R30*x)=-H(x)"
        sector.diagnostics["cyclic_factor"] = [-1., 0.]
        sector.save(sector_path)
    training = ring_points(np.array([.0005, .0015, .003, .0045, .006, .0075]), 256, .317)
    potential = sample_sector(sector, training)
    fit, condition = fit_harmonic_quadratures(training, np.zeros(len(training)), potential, 48, .008)
    reconstruction = RFReconstruction(fit, .008, {
        "source": "independent anti-periodic P4 FEM, alternating +1/-1 V electrodes",
        "mesh_um": mesh, "sector_solution_sha256": sha(sector_path),
        "fit_condition": condition, "not_analytic_dodecapole": True})
    reconstruction.save(folder / "reconstruction_N48_C.json")
    holdout = ring_points(np.array([.001, .002, .004, .007]), 513, .739)
    actual, actual_field = sample_sector(sector, holdout, with_field=True)
    predicted = fit.potential_quadratures(holdout)[1]
    rms_error = float(np.linalg.norm(actual-predicted)/np.linalg.norm(actual))
    predicted_field = fit.field_quadratures(holdout)[1]
    field_rms_error = float(np.linalg.norm(actual_field-predicted_field)/np.linalg.norm(actual_field))
    modes = {str(n): float(abs(fit.sine[n])) for n in range(1, 49)}
    mode6_fraction = abs(fit.sine[6])/max(abs(fit.sine[1:]))
    result = {"electrode_voltages_V": [1 if i % 2 == 0 else -1 for i in range(12)],
              "boundary": "outer=0 V, electrode_0=+1 V, Phi(R30*x)=-Phi(x)",
              "mesh_sha256": sha(mesh_path), "sector_sha256": sha(sector_path),
              "holdout_relative_rms_potential": rms_error,
              "holdout_relative_rms_field": field_rms_error,
              "leading_m6_fraction_of_largest_mode": float(mode6_fraction),
              "mode_amplitudes_at_8mm_V_per_V": modes,
              "fem_diagnostics": sector.diagnostics}
    (folder / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if rms_error > .01 or field_rms_error > .01 or mode6_fraction < .99:
        raise RuntimeError("C field fails FEM holdout or leading m=6 check")
    print(json.dumps({"mesh_um": mesh, "holdout_rms": rms_error, "field_rms": field_rms_error,
                      "m6_fraction": float(mode6_fraction), "dofs": sector.diagnostics.get("sector_dofs")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, choices=(35, 50), required=True)
    build(parser.parse_args().mesh)
