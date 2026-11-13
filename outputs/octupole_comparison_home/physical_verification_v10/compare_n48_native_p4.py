"""Compare native P4 and V7 N48 RF fields along the same direct-P4 paths.

The files are sampled on already calculated trajectories. This script neither
integrates trajectories nor solves FEM. The stored native P4 field quadratures
are compared with the matching h35/h50 N48 reconstruction at the same points.
"""
from __future__ import annotations

import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=HERE/"data"/"native_p4"
OUT=HERE/"processed"
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"outputs/octupole_comparison_home/confinement_maps_v4"))
from three_phase_trap.direct_fem import DirectFEMRF,SectorSolution
from three_phase_trap.gmsh_mesh import load_gmsh_mesh
import run_v4

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def load_p4(mesh_um):
    meshp=ROOT/f"outputs/meshes/gmsh_{mesh_um}um/trap.msh"
    base=ROOT/f"outputs/abc_validation/direct_fem_validation/v2/h{mesh_um}_P4"
    solp=base/"sector_solution.npz"
    rec=json.loads((base/"complete.json").read_text(encoding="utf-8"))
    if sha(meshp)!=rec["signature"]["full_mesh_sha256"] or sha(solp)!=rec["artifacts"]["sector_solution.npz"]:
        raise RuntimeError(f"Saved P4 input hash mismatch h{mesh_um}")
    return DirectFEMRF(SectorSolution.load(solp),load_gmsh_mesh(meshp),amplitude=1.)

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    rows=[]; summary=[]
    for mesh in (50,35):
        qpath=ROOT/f"outputs/abc_validation/direct_fem_validation/v2/h{mesh}_P4/diagnostic_N48.json"
        _,q=run_v4.load_fields(qpath)
        p4=load_p4(mesh)
        for mp in sorted(DATA.glob(f"{mesh}um_*.json")):
            if ".checkpoint." in mp.name: continue
            meta=json.loads(mp.read_text(encoding="utf-8"))
            if meta.get("horizon_rf",0)<1000: continue
            with np.load(DATA/meta["timeseries_file"],allow_pickle=False) as z:
                t=z["time_s"].astype(float); s=z["states"].astype(float)
            _,ecp,esp=p4.sample(s[:,:2])
            ecn,esn=q.field_quadratures(s[:,:2])
            if meta["model"]=="rotating":
                dp=ecp-ecn; ds=esp-esn
                p_norm=np.sum(ecp*ecp+esp*esp,axis=1)
                n_norm=np.sum(ecn*ecn+esn*esn,axis=1)
            else:
                dp=np.zeros_like(ecp); ds=esp-esn
                p_norm=np.sum(esp*esp,axis=1); n_norm=np.sum(esn*esn,axis=1)
            err=np.sqrt(np.sum(dp*dp+ds*ds,axis=1))
            reference=np.sqrt(p_norm)
            r=np.linalg.norm(s[:,:2],axis=1)
            edges=np.array([0.,1.,2.,4.,6.,7.,8.])*1e-3
            trajectory=mp.stem
            for lo,hi in zip(edges[:-1],edges[1:]):
                mask=(r>=lo)&(r<hi)
                if not np.any(mask): continue
                rms=float(np.sqrt(np.mean(err[mask]**2)))
                ref=float(np.sqrt(np.mean(reference[mask]**2)))
                row={"trajectory":trajectory,"mesh_um":mesh,"model":meta["model"],
                     "launch_label":meta["launch_label"],"status":meta["status"],
                     "escape_time_rf":meta["escape_time_rf"],"r_min_mm":lo*1e3,
                     "r_max_mm":hi*1e3,"n_samples":int(mask.sum()),
                     "field_error_rms_V_m_per_V":rms,
                     "native_P4_field_rms_V_m_per_V":ref,
                     "relative_rms_error":rms/max(ref,1e-300),
                     "mean_instantaneous_relative_error":float(np.mean(err[mask]/np.maximum(reference[mask],1e-300)))}
                rows.append(row)
            rms=float(np.sqrt(np.mean(err**2))); ref=float(np.sqrt(np.mean(reference**2)))
            summary.append({"trajectory":trajectory,"mesh_um":mesh,"model":meta["model"],
                            "launch_label":meta["launch_label"],"status":meta["status"],
                            "escape_time_rf":meta["escape_time_rf"],"n_samples":len(t),
                            "global_relative_rms_field_error":rms/max(ref,1e-300),
                            "global_relative_rms_field_error_percent":100*rms/max(ref,1e-300),
                            "max_radius_mm":float(r.max()),
                            "compared_components":"both RF quadratures (rotating); sine quadrature (standing)",
                            "field_interpretation":"native direct-P4 field along native-P4 path versus matching mesh's V7 N48 polynomial field"})
        del p4
    write(OUT/"native_P4_vs_N48_field_error_by_radius.csv",rows)
    write(OUT/"native_P4_vs_N48_trajectory_field_error.csv",summary)
    obj={"trajectories_compared":len(summary),"mesh_versions_um":[50,35],
         "finite_horizon_trajectories_only":True,"new_FEM_solves":0,"new_integrations":0,
         "units":"relative RMS electric-field amplitude along each saved path",
         "results":summary}
    (OUT/"native_P4_vs_N48_field_error.json").write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"trajectories_compared":len(summary),"results":summary},indent=2),flush=True)

if __name__=="__main__":main()
