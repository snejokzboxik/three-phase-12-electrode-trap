"""Extract the RF harmonics directly from saved native P4 solutions.

Potential is sampled on vacuum circles, Fourier projected in angle, and
compared with native P4 gradient quadratures. This is an audit of saved FEM
solutions, not a new mesh solve and not the existing N48 coefficient fit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from three_phase_trap.direct_fem import DirectFEMRF, SectorSolution
from three_phase_trap.gmsh_mesh import load_gmsh_mesh

OUT = HERE / "data" / "p4_harmonics"
FIG = HERE / "figures"
MASS = 100000.0 * 1.6605390666e-27
Q = 1.602176634e-19
OMEGA = 2*math.pi*100000.0
U_ROT = 250.0
U_STAND = math.sqrt(2)*U_ROT
R_AP = 8e-3
ORDERS = (4, 8, 16, 20, 28, 32, 40, 44)
RADII = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 7.0, 8.0])*1e-3
N_ANGLE = 4096


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths(mesh_um):
    mesh = ROOT / f"outputs/meshes/gmsh_{mesh_um}um/trap.msh"
    base = ROOT / f"outputs/abc_validation/direct_fem_validation/v2/h{mesh_um}_P4"
    return mesh, base / "sector_solution.npz", base / "complete.json"


def mode_gradient(a, signed_order, radius, theta):
    m = abs(signed_order)
    if signed_order > 0:
        scalar = m*a*radius**(m-1)*np.exp(1j*(m-1)*theta)
        grad = np.stack([scalar, 1j*scalar], axis=-1)
    else:
        scalar = m*a*radius**(m-1)*np.exp(-1j*(m-1)*theta)
        grad = np.stack([scalar, -1j*scalar], axis=-1)
    return -grad  # E_H = -grad H, per applied electrode volt


def fourier_modes(phi, radius, theta):
    ft = np.fft.fft(phi)/len(phi)
    modes = {}
    for m in ORDERS:
        for signed in (-m, m):
            idx = signed % len(phi)
            h_at_r = ft[idx]
            modes[signed] = h_at_r/(radius**m)
    return modes


def local_coefficient_rows(mesh_um, sample_function):
    rows = []
    for radius in RADII:
        theta = 2*np.pi*np.arange(N_ANGLE)/N_ANGLE
        pts = radius*np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        phi, ec, es = sample_function(pts)
        coeffs = fourier_modes(phi, radius, theta)
        for m in ORDERS:
            for signed in (-m, m):
                rows.append({"mesh_um":mesh_um, "radius_mm":radius*1e3,
                             "order_m":m, "signed_angular_order":signed,
                             "coefficient_real_V_per_V_m_order":float(coeffs[signed].real),
                             "coefficient_imag_V_per_V_m_order":float(coeffs[signed].imag),
                             "coefficient_abs_V_per_V_m_order":float(abs(coeffs[signed])),
                             "field_coefficient_V_per_V_m_order_minus1":float(m*abs(coeffs[signed])*radius**(m-1)),
                             "direct_P4_field_rms_V_per_m_per_V":float(np.sqrt(np.mean(np.sum(ec*ec+es*es,axis=1))))})
    return rows


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    all_rows=[]; cross_rows=[]; mesh_summaries={}
    for mesh_um in (50,35):
        mesh_path, solution_path, complete_path=paths(mesh_um)
        complete=json.loads(complete_path.read_text(encoding="utf-8"))
        if sha(mesh_path)!=complete["signature"]["full_mesh_sha256"] or sha(solution_path)!=complete["artifacts"]["sector_solution.npz"]:
            raise RuntimeError(f"Input hash mismatch for h{mesh_um}")
        sector=SectorSolution.load(solution_path)
        trap=load_gmsh_mesh(mesh_path)
        field=DirectFEMRF(sector,trap,amplitude=1.0)

        # Coefficient extraction over radii from the native P4 potential.
        rows=local_coefficient_rows(mesh_um,field.sample)
        all_rows.extend(rows)

        # Independent full field and mode reconstruction at the physical aperture.
        theta=2*np.pi*np.arange(N_ANGLE)/N_ANGLE
        pts=R_AP*np.stack([np.cos(theta),np.sin(theta)],axis=-1)
        phi,ec,es=field.sample(pts)
        coeffs=fourier_modes(phi,R_AP,theta)
        chosen={}
        for m in ORDERS:
            signed=max((-m,m), key=lambda s: abs(coeffs[s]))
            chosen[m]=(signed,coeffs[signed])
        eh_modes={m:mode_gradient(a,signed,R_AP,theta) for m,(signed,a) in chosen.items()}
        eh_sum=np.sum(np.stack(list(eh_modes.values()),axis=0),axis=0)
        ec_model=eh_sum.imag; es_model=eh_sum.real
        denom=np.sqrt(np.mean(np.sum(ec*ec+es*es,axis=1)))
        residual=np.sqrt(np.mean(np.sum((ec-ec_model)**2+(es-es_model)**2,axis=1)))/denom

        psi_scale=Q/(4*MASS*OMEGA**2) # eV per squared (V/m) for q=+e
        psi_rot_direct=psi_scale*U_ROT**2*np.sum(ec*ec+es*es,axis=1)
        psi_stand_direct=psi_scale*U_STAND**2*np.sum(es*es,axis=1)
        psi_rot_m48=psi_scale*U_ROT**2*np.sum(abs(eh_modes[4]+eh_modes[8])**2,axis=1)
        psi_stand_m48=psi_scale*U_STAND**2*np.sum((eh_modes[4]+eh_modes[8]).real**2,axis=1)
        cross_rot_48=2*psi_scale*U_ROT**2*np.real(np.sum(eh_modes[4]*np.conj(eh_modes[8]),axis=1))
        cross_stand_48=2*psi_scale*U_STAND**2*np.sum(eh_modes[4].real*eh_modes[8].real,axis=1)
        for j,ang in enumerate(theta):
            cross_rows.append({"mesh_um":mesh_um,"radius_mm":8.0,"theta_deg":float(np.degrees(ang)),
                               "psi_rotating_direct_P4_eV":float(psi_rot_direct[j]),
                               "psi_standing_direct_P4_eV":float(psi_stand_direct[j]),
                               "psi_rotating_m4_plus_m8_eV":float(psi_rot_m48[j]),
                               "psi_standing_m4_plus_m8_eV":float(psi_stand_m48[j]),
                               "cross_rotating_m4_m8_eV":float(cross_rot_48[j]),
                               "cross_standing_m4_m8_eV":float(cross_stand_48[j])})
        self4_rot=psi_scale*U_ROT**2*np.sum(abs(eh_modes[4])**2,axis=1)
        self4_stand=psi_scale*U_STAND**2*np.sum(eh_modes[4].real**2,axis=1)
        summary={"mesh_um":mesh_um,"mesh_sha256":sha(mesh_path),
                 "p4_solution_sha256":sha(solution_path),"samples_per_ring":N_ANGLE,
                 "potential_mode_coefficients_at_aperture":{
                     str(m):{"dominant_signed_l":int(chosen[m][0]),
                             "complex_coefficient_V_per_V_m_order":
                                 [float(chosen[m][1].real),float(chosen[m][1].imag)],
                             "magnitude_V_per_V_m_order":float(abs(chosen[m][1])),
                             "opposite_sign_magnitude":float(abs(coeffs[-chosen[m][0]]))}
                     for m in ORDERS},
                 "m4_standing_vs_rotating_self_term_relative_rms_difference":
                     float(np.sqrt(np.mean((self4_stand-self4_rot)**2))/np.mean(self4_rot)),
                 "rotating_m4_m8_cross_abs_max_eV":float(np.max(np.abs(cross_rot_48))),
                 "standing_m4_m8_cross_min_eV":float(np.min(cross_stand_48)),
                 "standing_m4_m8_cross_max_eV":float(np.max(cross_stand_48)),
                 "all_mode_gradient_reconstruction_relative_rms_residual_at_8mm":float(residual),
                 "direct_P4_rotating_pseudopotential_eV_range":
                     [float(np.min(psi_rot_direct)),float(np.max(psi_rot_direct))],
                 "direct_P4_standing_pseudopotential_eV_range":
                     [float(np.min(psi_stand_direct)),float(np.max(psi_stand_direct))],
                 "scope":"native P4 sample and Fourier projection; no N48 fit used to derive these values"}
        mesh_summaries[str(mesh_um)]=summary
        del field, trap, sector

    save_csv(OUT/"native_p4_harmonic_coefficients.csv",all_rows)
    save_csv(OUT/"native_p4_pseudopotential_cross_m4_m8.csv",cross_rows)
    (OUT/"native_p4_harmonic_audit.json").write_text(json.dumps(mesh_summaries,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(mesh_summaries,indent=2),flush=True)


if __name__=="__main__":
    main()
