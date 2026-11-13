"""Finite-difference and phase-superposition checks on saved P4 FEM fields."""
from __future__ import annotations
import csv,json,math,sys,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/"processed"
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.direct_fem import DirectFEMRF,SectorSolution
from three_phase_trap.gmsh_mesh import load_gmsh_mesh

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run_mesh(mesh_um):
    meshp=ROOT/f"outputs/meshes/gmsh_{mesh_um}um/trap.msh"
    base=ROOT/f"outputs/abc_validation/direct_fem_validation/v2/h{mesh_um}_P4"
    solp=base/"sector_solution.npz";complete=json.loads((base/"complete.json").read_text())
    if sha(meshp)!=complete["signature"]["full_mesh_sha256"] or sha(solp)!=complete["artifacts"]["sector_solution.npz"]:
        raise RuntimeError(f"Source SHA mismatch h{mesh_um}")
    field=DirectFEMRF(SectorSolution.load(solp),load_gmsh_mesh(meshp),amplitude=1.)
    tri=field.points[field.triangles];centers=tri.mean(axis=1);r=np.linalg.norm(centers,axis=1)
    valid=np.flatnonzero((r>.5e-3)&(r<7.5e-3))
    ids=valid[np.unique(np.linspace(0,len(valid)-1,32,dtype=int))]
    records=[]
    for k,tid in enumerate(ids):
        p=centers[tid]; cell=int(field._finder(p[0],p[1]))
        bary=field.gradient[cell]@(p-field.origin[cell]);bary[0]+=1
        margin=float(np.min(bary/np.linalg.norm(field.gradient[cell],axis=1)))
        h=min(1e-8,margin/20)
        pts=np.array([p+[h,0],p-[h,0],p+[0,h],p-[0,h]])
        phi,ec,es=field.sample(p[None,:]);phi=phi[0];ec=ec[0];es=es[0]
        ph,_,_=field.sample(pts)
        grad_fd=np.array([(ph[0]-ph[1])/(2*h),(ph[2]-ph[3])/(2*h)])
        grad_native=-es-1j*ec
        grad_rel=float(np.linalg.norm(grad_fd-grad_native)/np.linalg.norm(grad_native))
        for lam in (0.371,1.913,4.202):
            for model in ("rotating","standing"):
                if model=="rotating":
                    vpot=np.imag(phi*np.exp(1j*lam))
                    vph=np.imag(ph*np.exp(1j*lam))
                    e_native=ec*math.cos(lam)+es*math.sin(lam)
                else:
                    vpot=phi.real*math.sin(lam)
                    vph=ph.real*math.sin(lam)
                    e_native=es*math.sin(lam)
                e_fd=-np.array([(vph[0]-vph[1])/(2*h),(vph[2]-vph[3])/(2*h)])
                er=float(np.linalg.norm(e_fd-e_native)/max(np.linalg.norm(e_native),1e-300))
                records.append({"mesh_um":mesh_um,"sample_id":k,"triangle_id":int(cell),
                                "x_m":float(p[0]),"y_m":float(p[1]),"radius_m":float(r[tid]),
                                "distance_to_nearest_face_m":margin,"finite_difference_step_m":h,
                                "temporal_phase_rad":lam,"model":model,
                                "complex_H_gradient_relative_error":grad_rel,
                                "electric_field_phase_superposition_relative_error":er})
    summary={"mesh_um":mesh_um,"mesh_sha256":sha(meshp),"solution_sha256":sha(solp),
             "samples":len(ids),"phase_samples_rad":[0.371,1.913,4.202],
             "max_complex_gradient_relative_error":max(r["complex_H_gradient_relative_error"] for r in records),
             "max_phase_superposition_relative_error":max(r["electric_field_phase_superposition_relative_error"] for r in records),
             "gradient_test":"central finite difference of the native P4 potential, with both probes kept inside one element",
             "phase_test":"independent evaluation of Im(H exp(i lambda)) and Re(H) sin(lambda), differentiated consistently",
             "validation":"P4 native potential gradient and rotating/standing phase superposition; no harmonic surrogate"}
    return summary,records

def main():
    OUT.mkdir(parents=True,exist_ok=True);summaries={};rows=[]
    for mesh in (50,35):
        summary,r=run_mesh(mesh);summaries[str(mesh)]=summary;rows.extend(r)
    with (OUT/"native_p4_gradient_phase_checks.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/"native_p4_gradient_phase_checks.json").write_text(json.dumps(summaries,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=="__main__":main()
