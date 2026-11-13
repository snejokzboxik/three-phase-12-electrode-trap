"""Derive energy, work, radial and angular-momentum histories from P4 runs."""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_simpson

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=HERE/"data"/"native_p4"
OUT=HERE/"processed"
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.direct_fem import DirectFEMRF,SectorSolution
from three_phase_trap.gmsh_mesh import load_gmsh_mesh

F=100000.0; OMEGA=2*math.pi*F; MASS=1.6605390666e-22; Q=1.602176634e-19
U={"rotating":250.0,"standing":353.5533905932738}
RREF=8e-3; ORDERS=(4,8,16,20,28,32,40,44); N_ANGLE=4096


def cumulative_trapz(y,x):
    return cumulative_simpson(y,x=x,initial=0.)


def extract_coefficients(field):
    theta=2*np.pi*np.arange(N_ANGLE)/N_ANGLE
    pts=RREF*np.stack([np.cos(theta),np.sin(theta)],axis=-1)
    h,_,_=field.sample(pts); ft=np.fft.fft(h)/N_ANGLE
    coeff={}
    for m in ORDERS:
        signed=max((-m,m),key=lambda l:abs(ft[l%N_ANGLE]))
        coeff[m]=(signed,ft[signed%N_ANGLE]/RREF**m)
    return coeff


def mode_h_gradient(a,signed,x,y):
    m=abs(signed)
    if signed>0:
        z=x+1j*y; hm=a*z**m; gx=m*a*z**(m-1); gy=1j*gx
    else:
        z=x-1j*y; hm=a*z**m; gx=m*a*z**(m-1); gy=-1j*gx
    return hm,np.stack([gx,gy],axis=-1)


def write_csv(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_field(mesh_um):
    mesh_path=ROOT/f"outputs/meshes/gmsh_{mesh_um}um/trap.msh"
    solution_path=ROOT/f"outputs/abc_validation/direct_fem_validation/v2/h{mesh_um}_P4/sector_solution.npz"
    return DirectFEMRF(SectorSolution.load(solution_path),load_gmsh_mesh(mesh_path),amplitude=1.)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    recs=[]
    for path in sorted(DATA.glob("*.json")):
        if ".checkpoint." in path.name or path.name=="index.csv": continue
        meta=json.loads(path.read_text(encoding="utf-8"))
        if int(meta.get("horizon_rf",0))<1000: continue
        recs.append((path,meta))
    summaries=[]; histories=[]; mode_work=[]
    fields={}
    for path,meta in recs:
        mesh=int(meta["mesh_um"])
        if mesh not in fields:
            fields[mesh]=load_field(mesh)
        field=fields[mesh]
        with np.load(DATA/meta["timeseries_file"],allow_pickle=False) as z:
            t=z["time_s"].astype(float); s=z["states"].astype(float)
        x,y,vx,vy=s.T; r=np.hypot(x,y); vr=(x*vx+y*vy)/np.maximum(r,1e-300)
        speed2=vx*vx+vy*vy
        kinetic=.5*MASS*speed2/Q
        lam=OMEGA*t+math.radians(float(meta["phase_deg"]))
        h,ec,es=field.sample(s[:,:2])
        amp=U[meta["model"]]
        if meta["model"]=="rotating":
            phase_factor=np.exp(1j*lam)
            phi=amp*np.imag(h*phase_factor)
            efield=amp*(ec*np.cos(lam)[:,None]+es*np.sin(lam)[:,None])
            p_param=amp*OMEGA*np.real(h*phase_factor)
        else:
            phi=amp*np.real(h)*np.sin(lam)
            efield=amp*es*np.sin(lam)[:,None]
            p_param=amp*OMEGA*np.real(h)*np.cos(lam)
        hamiltonian=kinetic+phi
        p_electric=np.sum(efield*s[:,2:],axis=1)
        work_electric=cumulative_trapz(p_electric,t)
        work_param=cumulative_trapz(p_param,t)
        coeffs=extract_coefficients(field)
        p_modal=np.zeros((len(t),len(ORDERS)))
        torque_modal=np.zeros((len(t),len(ORDERS)))
        phi_modal=np.zeros_like(p_modal)
        h_modal_sum=np.zeros(len(t),dtype=complex)
        eh_by_m=[]
        for k,m in enumerate(ORDERS):
            signed,a=coeffs[m]
            hm,grad=mode_h_gradient(a,signed,x,y)
            h_modal_sum+=hm
            eh=-grad
            if meta["model"]=="rotating":
                phm=amp*np.imag(hm*np.exp(1j*lam))
                ecm=amp*(-grad.imag)*np.cos(lam)[:,None]
                esm=amp*(-grad.real)*np.sin(lam)[:,None]
                e_m=ecm+esm
                pp=amp*OMEGA*np.real(hm*np.exp(1j*lam))
            else:
                phm=amp*np.real(hm)*np.sin(lam)
                e_m=amp*eh.real*np.sin(lam)[:,None]
                pp=amp*OMEGA*np.real(hm)*np.cos(lam)
            phi_modal[:,k]=phm
            p_modal[:,k]=np.sum(e_m*s[:,2:],axis=1)
            torque_modal[:,k]=Q*(x*e_m[:,1]-y*e_m[:,0])
            eh_by_m.append(eh)
            mode_work.append({"trajectory":path.stem,"mesh_um":mesh,"model":meta["model"],
                              "launch_label":meta["launch_label"],"mode_m":m,
                              "work_eV_at_end":float(cumulative_trapz(p_modal[:,k],t)[-1]),
                              "max_abs_cumulative_work_eV":float(np.max(np.abs(cumulative_trapz(p_modal[:,k],t))))})
        w_modal=np.stack([cumulative_trapz(p_modal[:,k],t) for k in range(len(ORDERS))],axis=1)
        torque_integral_modal=np.stack([cumulative_trapz(torque_modal[:,k],t)
                                        for k in range(len(ORDERS))],axis=1)
        torque_total=Q*(x*efield[:,1]-y*efield[:,0])
        torque_integral=cumulative_trapz(torque_total,t)
        w_param_modal=np.stack([cumulative_trapz(
            (amp*OMEGA*np.real(mode_h_gradient(coeffs[m][1],coeffs[m][0],x,y)[0]*np.exp(1j*lam))
             if meta["model"]=="rotating" else amp*OMEGA*np.real(mode_h_gradient(coeffs[m][1],coeffs[m][0],x,y)[0])*np.cos(lam)),t)
             for m in ORDERS],axis=1)
        lz=MASS*(x*vy-y*vx)
        hist_rows=[]
        for i in range(len(t)):
            row={"trajectory":path.stem,"mesh_um":mesh,"model":meta["model"],
                 "launch_label":meta["launch_label"],"time_rf":t[i]*F,"time_s":t[i],
                 "x_m":x[i],"y_m":y[i],"vx_m_s":vx[i],"vy_m_s":vy[i],
                 "r_m":r[i],"vr_m_s":vr[i],"kinetic_eV":kinetic[i],"phi_V":phi[i],
                 "H_eV":hamiltonian[i],"Lz_kg_m2_s":lz[i],
                 "Lz_change_from_integrated_RF_torque_kg_m2_s":torque_integral[i],
                 "work_E_eV":work_electric[i],"work_explicit_RF_eV":work_param[i],
                 "sum_modal_work_eV":float(np.sum(w_modal[i])),
                 "sum_modal_explicit_RF_work_eV":float(np.sum(w_param_modal[i]))}
            for k,m in enumerate(ORDERS):
                row[f"torque_m{m}_N_m"]=float(torque_modal[i,k])
                row[f"integrated_torque_m{m}_kg_m2_s"]=float(torque_integral_modal[i,k])
            hist_rows.append(row)
        histories.extend(hist_rows)
        h_err=np.max(abs((hamiltonian-hamiltonian[0])-work_param))
        k_err=np.max(abs((kinetic-kinetic[0])-work_electric))
        lz_err=np.max(abs((lz-lz[0])-torque_integral))
        modal_tau=np.sum(torque_modal,axis=1)
        modal_tau_rel=float(np.sqrt(np.mean((modal_tau-torque_total)**2)) /
                            max(np.sqrt(np.mean(torque_total**2)),1e-300))
        modal_res=np.sqrt(np.mean(abs(h-h_modal_sum)**2))/max(np.sqrt(np.mean(abs(h)**2)),1e-300)
        summaries.append({"trajectory":path.stem,"mesh_um":mesh,"model":meta["model"],
                          "launch_label":meta["launch_label"],"status":meta["status"],
                          "escape_time_rf":meta["escape_time_rf"],"n_samples":len(t),
                          "r_max_sampled_m":float(r.max()),
                          "r_rms_m":float(np.sqrt(np.trapezoid(r*r,x=t)/(t[-1]-t[0]))),
                          "kinetic_energy_initial_eV":float(kinetic[0]),
                          "kinetic_energy_max_eV":float(kinetic.max()),
                          "hamiltonian_initial_eV":float(hamiltonian[0]),
                          "hamiltonian_max_minus_min_eV":float(np.ptp(hamiltonian)),
                          "electric_work_net_eV":float(work_electric[-1]),
                          "explicit_RF_work_net_eV":float(work_param[-1]),
                          "max_energy_balance_residual_eV":float(h_err),
                          "max_kinetic_work_residual_eV":float(k_err),
                          "max_angular_momentum_torque_closure_kg_m2_s":float(lz_err),
                          "modal_torque_reconstruction_relative_rms":modal_tau_rel,
                          "mode_potential_reconstruction_relative_rms":float(modal_res),
                          "total_potential_energy_max_abs_eV":float(np.max(abs(phi))),
                          "relative_phase":"exact V7 convention; initial phase in metadata"})
    write_csv(OUT/"p4_trajectory_energy_diagnostics.csv",summaries)
    write_csv(OUT/"p4_energy_and_motion_timeseries.csv.gz",histories)
    write_csv(OUT/"p4_harmonic_work_by_trajectory.csv",mode_work)
    print(json.dumps({"trajectories_processed":len(recs),"summaries":summaries},indent=2),flush=True)


if __name__=="__main__": main()
