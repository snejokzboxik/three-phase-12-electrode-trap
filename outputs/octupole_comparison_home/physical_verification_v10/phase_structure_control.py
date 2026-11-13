"""Controlled relative-phase experiments on h50-P4-derived m=4 and m=8.

The spatial coefficients are extracted from the saved native P4 FEM solution.
The independent phase control is synthetic: it shifts one spatial harmonic in
time without changing its amplitude. Results therefore test causality but do
not represent the standard three-group ABC drive unless stated otherwise.
"""
from __future__ import annotations

import csv
import heapq
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/"data"/"phase_structure_control"
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.direct_fem import DirectFEMRF,SectorSolution
from three_phase_trap.gmsh_mesh import load_gmsh_mesh

E_CHARGE=1.602176634e-19; MASS=1.6605390666e-22; Q=E_CHARGE
F=100000.0; OMEGA=2*math.pi*F; PERIOD=1/F
U_ROT=250.0; U_STAND=math.sqrt(2)*U_ROT; R_AP=8e-3
ORDERS=(4,8); N_ANGLE=4096; R_EXTRACT=8e-3
HORIZON_RF=1000; SAMPLES_PER_RF=8


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_modes():
    meshp=ROOT/"outputs/meshes/gmsh_50um/trap.msh"
    p4dir=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4"
    solp=p4dir/"sector_solution.npz"; complete=json.loads((p4dir/"complete.json").read_text())
    if sha(meshp)!=complete["signature"]["full_mesh_sha256"] or sha(solp)!=complete["artifacts"]["sector_solution.npz"]:
        raise RuntimeError("h50 FEM input hash mismatch")
    field=DirectFEMRF(SectorSolution.load(solp),load_gmsh_mesh(meshp),amplitude=1.)
    th=2*np.pi*np.arange(N_ANGLE)/N_ANGLE
    pts=R_EXTRACT*np.stack([np.cos(th),np.sin(th)],axis=-1)
    h,_,_=field.sample(pts);ft=np.fft.fft(h)/N_ANGLE
    modes={}
    for m in ORDERS:
        signed=max((-m,m),key=lambda l:abs(ft[l%N_ANGLE]))
        modes[m]={"signed_l":int(signed),"a":complex(ft[signed%N_ANGLE]/R_EXTRACT**m)}
    return modes,{"mesh_sha256":sha(meshp),"p4_sha256":sha(solp),
                  "harmonic_coefficients_extracted_from":"native P4 potential on r=8 mm circle",
                  "modes":{str(m):{"signed_l":v["signed_l"],"a_real":v["a"].real,
                                      "a_imag":v["a"].imag,"a_abs":abs(v["a"])} for m,v in modes.items()}}


def mode_values(a,signed_l,x,y):
    m=abs(signed_l)
    if signed_l>0:
        z=x+1j*y; h=a*z**m
        gx=m*a*z**(m-1); gy=1j*gx
        z2=m*(m-1)*a*z**(m-2)
        hess=np.stack([np.stack([z2,1j*z2],axis=-1),np.stack([1j*z2,-z2],axis=-1)],axis=-2)
    else:
        z=x-1j*y; h=a*z**m
        gx=m*a*z**(m-1); gy=-1j*gx
        z2=m*(m-1)*a*z**(m-2)
        hess=np.stack([np.stack([z2,-1j*z2],axis=-1),np.stack([-1j*z2,-z2],axis=-1)],axis=-2)
    grad=np.stack([gx,gy],axis=-1)
    return h,grad,hess


def initial_for(entry):
    az=math.radians(float(entry["azimuth_deg"])); v=float(62.12417634275831)
    r=1e-3; d=int(entry["direction"])
    return np.array([r*math.cos(az),r*math.sin(az),-d*v*math.sin(az),d*v*math.cos(az)])


def integrate_full(mode_coeff,model,delta_deg,initial,launch,tag):
    delta=math.radians(delta_deg); amp=U_ROT if model=="rotating" else U_STAND
    def rhs(t,s):
        x,y,vx,vy=s; E=np.zeros(2)
        for m in ORDERS:
            val,grad,_=mode_values(mode_coeff[m]["a"],mode_coeff[m]["signed_l"],x,y)
            phase=OMEGA*t+(delta if m==8 else 0.0)
            if model=="rotating": E-=amp*np.imag(grad*np.exp(1j*phase))
            else: E-=amp*np.real(grad)*math.sin(phase)
        return np.array([vx,vy,Q/MASS*E[0],Q/MASS*E[1]])
    def aperture(t,s): return math.hypot(s[0],s[1])-R_AP
    aperture.terminal=True; aperture.direction=1
    horizon=HORIZON_RF/F
    sample_grid=np.linspace(0,horizon,int(HORIZON_RF*SAMPLES_PER_RF)+1)
    sol=solve_ivp(rhs,(0,horizon),initial,method="DOP853",rtol=2e-9,
                  atol=(2e-13,2e-13,2e-11,2e-11),max_step=PERIOD/100,
                  events=aperture,t_eval=sample_grid,dense_output=True)
    if sol.status<0: raise RuntimeError(f"DOP853 failed: {sol.message}")
    escaped=bool(len(sol.t_events[0])); escape_time=float(sol.t_events[0][0]) if escaped else None
    times=sol.t; states=sol.y.T
    if escaped:
        event_state=sol.y_events[0][0]
        times=np.r_[times,escape_time];states=np.vstack([states,event_state])
    radii=np.linalg.norm(states[:,:2],axis=1)
    status="escaped" if escaped else "retained_to_horizon"
    name=f"{model}_delta{delta_deg:g}_{launch}_{tag}"
    OUT.mkdir(parents=True,exist_ok=True)
    dat=OUT/f"{name}.npz"; tmp=dat.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,time_s=times,states=states);f.flush()
    tmp.replace(dat)
    rec={"model":"synthetic m4+m8 full RF","temporal_model":model,"delta_m8_deg":delta_deg,
         "launch":launch,"initial_state_SI":initial.tolist(),"energy_eV":2.0,"radius_mm":1.0,
         "voltage_V":amp,"frequency_Hz":F,"horizon_RF":HORIZON_RF,"aperture_radius_m":R_AP,
         "status":status,"escape_time_s":escape_time,
         "escape_time_rf":None if escape_time is None else escape_time*F,
         "maximum_sampled_radius_m":float(radii.max()),"nfev":int(sol.nfev),
         "rtol":2e-9,"max_step_s":PERIOD/100,"samples_per_RF":SAMPLES_PER_RF,
         "time_dependent_potential":("sum U*Im(H_m*exp(i*(Omega*t+delta_m)))" if model=="rotating"
                                     else "sum U*Re(H_m)*sin(Omega*t+delta_m)"),
         "drive_status":"mathematical per-harmonic phase control; not generated by the standard tied ABC group drive",
         "spatial_source":"m=4 and m=8 Fourier components sampled from saved h50 P4 FEM potential",
         "timeseries_file":dat.name}
    (OUT/f"{name}.json").write_text(json.dumps(rec,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"completed":name,"status":status,"escape_rf":rec["escape_time_rf"],"nfev":sol.nfev}),flush=True)
    return rec


def integrate_average(mode_coeff,model,delta_deg,initial,launch):
    delta=math.radians(delta_deg); amp=U_ROT if model=="rotating" else U_STAND
    psi_scale=Q/(4*MASS*OMEGA**2)
    def rhs(t,s):
        x,y,vx,vy=s
        pieces=[]
        for m in ORDERS:
            _,grad,hess=mode_values(mode_coeff[m]["a"],mode_coeff[m]["signed_l"],x,y)
            eh=-grad; je=-hess
            pieces.append((eh,je))
        e4,j4=pieces[0];e8,j8=pieces[1]
        if model=="rotating":
            e=amp*(e4+e8*np.exp(1j*delta)); j=amp*(j4+j8*np.exp(1j*delta))
            grad_psi=2*psi_scale*np.real(np.einsum("i,ik->k",np.conj(e),j))
        else:
            e1,e2=e4.real,e8.real;j1,j2=j4.real,j8.real
            grad_psi=2*psi_scale*amp**2*(j1.T@e1+j2.T@e2+math.cos(delta)*(j1.T@e2+j2.T@e1))
        return np.array([vx,vy,-Q/MASS*grad_psi[0],-Q/MASS*grad_psi[1]])
    def aperture(t,s): return math.hypot(s[0],s[1])-R_AP
    aperture.terminal=True;aperture.direction=1
    horizon=HORIZON_RF/F; sample_grid=np.linspace(0,horizon,int(HORIZON_RF*SAMPLES_PER_RF)+1)
    sol=solve_ivp(rhs,(0,horizon),initial,method="DOP853",rtol=2e-9,
                  atol=(2e-13,2e-13,2e-11,2e-11),max_step=PERIOD/10,
                  events=aperture,t_eval=sample_grid,dense_output=True)
    if sol.status<0:raise RuntimeError(f"Averaged DOP853 failed: {sol.message}")
    escaped=bool(len(sol.t_events[0])); escape=float(sol.t_events[0][0]) if escaped else None
    t=sol.t; s=sol.y.T
    if escaped:t=np.r_[t,escape];s=np.vstack([s,sol.y_events[0][0]])
    status="escaped" if escaped else "retained_to_horizon"
    radii=np.linalg.norm(s[:,:2],axis=1)
    name=f"average_{model}_delta{delta_deg:g}_{launch}_p4h50"
    OUT.mkdir(parents=True,exist_ok=True)
    dat=OUT/f"{name}.npz";tmp=dat.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:np.savez_compressed(f,time_s=t,states=s);f.flush()
    tmp.replace(dat)
    rec={"model":"leading pseudopotential m4+m8","temporal_model":model,
         "delta_m8_deg":delta_deg,"launch":launch,"initial_state_SI":initial.tolist(),
         "status":status,"escape_time_s":escape,"escape_time_rf":None if escape is None else escape*F,
         "maximum_sampled_radius_m":float(radii.max()),"nfev":int(sol.nfev),
         "rtol":2e-9,"max_step_s":PERIOD/10,"timeseries_file":dat.name,
         "initial_condition_caveat":"same laboratory (x,y,vx,vy) as full RF; not phase-averaged guiding-center data"}
    (OUT/f"{name}.json").write_text(json.dumps(rec,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"completed_average":name,"status":status,"escape_rf":rec["escape_time_rf"],"nfev":sol.nfev}),flush=True)
    return rec


def pseudopotential_rows(mode_coeff):
    psi_scale=Q/(4*MASS*OMEGA**2)
    rows=[];angles=np.linspace(0,360,721)[:-1]
    for rmm in (1,2,4,6,8):
        r=rmm*1e-3
        for model in ("rotating","standing"):
            U=U_ROT if model=="rotating" else U_STAND
            for delta_deg in (0.,90.,180.):
                delta=math.radians(delta_deg)
                for adeg in angles:
                    th=math.radians(adeg);x,y=r*math.cos(th),r*math.sin(th)
                    eh=[]
                    for m in ORDERS:
                        _,grad,_=mode_values(mode_coeff[m]["a"],mode_coeff[m]["signed_l"],x,y)
                        eh.append(-grad)
                    if model=="rotating":
                        # Opposite-helicity m4/m8 fields are orthogonal; this
                        # explicit norm is retained as a numerical verification.
                        e=U*(eh[0]+eh[1]*np.exp(1j*delta))
                        self4=psi_scale*U**2*float(np.sum(abs(eh[0])**2))
                        self8=psi_scale*U**2*float(np.sum(abs(eh[1])**2))
                        cross=psi_scale*U**2*float(2*np.real(np.sum(eh[0]*np.conj(eh[1]*np.exp(1j*delta)))) )
                        psi=psi_scale*float(np.sum(abs(e)**2))
                    else:
                        e1=eh[0].real; e2=eh[1].real
                        self4=psi_scale*U**2*float(np.dot(e1,e1))
                        self8=psi_scale*U**2*float(np.dot(e2,e2))
                        cross=psi_scale*U**2*float(2*math.cos(delta)*np.dot(e1,e2))
                        psi=self4+self8+cross
                    rows.append({"model":model,"radius_mm":rmm,"delta_m8_deg":delta_deg,
                                 "theta_deg":adeg,"self_m4_eV":self4,"self_m8_eV":self8,
                                 "cross_m4_m8_eV":cross,"pseudopotential_eV":psi})
    return rows


def minimax_barrier(mode_coeff,model,delta_deg,start_azimuth_deg,nr,ntheta):
    """Grid min-max path from the launch radius to the 8-mm aperture."""
    rgrid=np.linspace(1e-3,R_AP,nr);theta=2*np.pi*np.arange(ntheta)/ntheta
    rr,tt=np.meshgrid(rgrid,theta,indexing="ij");x=rr*np.cos(tt);y=rr*np.sin(tt)
    eh=[]
    for m in ORDERS:
        _,grad,_=mode_values(mode_coeff[m]["a"],mode_coeff[m]["signed_l"],x,y)
        eh.append(-grad)
    U=U_ROT if model=="rotating" else U_STAND
    delta=math.radians(delta_deg);scale=Q/(4*MASS*OMEGA**2)
    if model=="rotating":
        psi=scale*U**2*np.sum(abs(eh[0]+eh[1]*np.exp(1j*delta))**2,axis=-1)
    else:
        e1,e2=eh[0].real,eh[1].real
        psi=scale*U**2*(np.sum(e1*e1,axis=-1)+np.sum(e2*e2,axis=-1)
                       +2*math.cos(delta)*np.sum(e1*e2,axis=-1))
    i0=0
    j0=int(round((start_azimuth_deg%360)/360*ntheta))%ntheta
    start=i0*ntheta+j0
    flat=psi.ravel();dist=np.full(len(flat),np.inf);dist[start]=flat[start]
    heap=[(float(flat[start]),start)];target=None
    while heap:
        cost,node=heapq.heappop(heap)
        if cost!=dist[node]:continue
        i,j=divmod(node,ntheta)
        if i==nr-1:target=node;break
        neighbors=((i-1,j),(i+1,j),(i,(j-1)%ntheta),(i,(j+1)%ntheta))
        for ii,jj in neighbors:
            if ii<0 or ii>=nr:continue
            nxt=ii*ntheta+jj;new=max(cost,float(flat[nxt]))
            if new<dist[nxt]:dist[nxt]=new;heapq.heappush(heap,(new,nxt))
    if target is None:raise RuntimeError("No path to aperture found on pseudopotential grid")
    return {"model":model,"delta_m8_deg":delta_deg,"start_azimuth_deg":start_azimuth_deg,
            "nr":nr,"ntheta":ntheta,"radial_spacing_mm":(rgrid[1]-rgrid[0])*1e3,
            "angular_spacing_deg":360/ntheta,"psi_start_eV":float(flat[start]),
            "minimax_aperture_energy_eV":float(dist[target]),
            "effective_barrier_above_start_eV":float(dist[target]-flat[start]),
            "initial_kinetic_energy_eV":2.0,
            "available_initial_total_eV":float(2.0+flat[start]),
            "energetically_open_in_pseudopotential":bool(2.0+flat[start]>=dist[target])}


def main():
    mode_coeff,source=load_modes()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"h50_p4_mode_coefficients.json").write_text(json.dumps(source,indent=2)+"\n",encoding="utf-8")
    controls=pseudopotential_rows(mode_coeff)
    csv_write(OUT/"m4_m8_pseudopotential_phase_control.csv",controls)
    config=json.loads((HERE/"config_v10.json").read_text(encoding="utf-8"))
    startmap={v["label"]:v for v in config["initial_conditions_v7"]}
    launches=("early_az22p5_dplus","negative_escaped","positive_escaped")
    records=[]
    average_records=[]
    for launch in launches:
        initial=initial_for(startmap[launch])
        for model in ("rotating","standing"):
            for delta in (0.,90.,180.):
                records.append(integrate_full(mode_coeff,model,delta,initial,launch,"p4h50"))
                average_records.append(integrate_average(mode_coeff,model,delta,initial,launch))
    csv_write(OUT/"m4_m8_phase_control_trajectories.csv",records)
    csv_write(OUT/"m4_m8_averaged_phase_control_trajectories.csv",average_records)
    barriers=[]
    for model in ("rotating","standing"):
        for delta in (0.,90.,180.):
            coarse=minimax_barrier(mode_coeff,model,delta,22.5,141,360)
            barriers.append(coarse)
    # Refine only the reference and zero-cross-term cases so the path metric
    # cannot be mistaken for a fictitious smooth barrier curve.
    barrier_fine=[minimax_barrier(mode_coeff,model,delta,22.5,281,720)
                  for model,delta in (("rotating",0.),("standing",0.),("standing",90.))]
    csv_write(OUT/"m4_m8_pseudopotential_minimax_barriers.csv",barriers+barrier_fine)
    summary={}
    for row in controls:
        key=f"{row['model']}|r={row['radius_mm']}mm|delta={row['delta_m8_deg']}deg"
        summary.setdefault(key,[]).append(row["pseudopotential_eV"])
    extrema={k:{"min_eV":float(min(v)),"max_eV":float(max(v)),"mean_eV":float(np.mean(v)),"range_eV":float(max(v)-min(v))}
             for k,v in summary.items()}
    result={"source":source,"full_RF_cases":records,"pseudopotential_extrema":extrema,
            "minimax_barriers":barriers+barrier_fine,
            "phase_rule":"standing m4×m8 leading cross term scales as cos(delta); rotating cross term vanishes for opposite helicities",
            "normalization":"equal leading m4 pseudopotential, Ustand/Urot=sqrt(2)",
            "warning":"relative phase between harmonics is a synthetic control, not the standard three-group ABC excitation"}
    (OUT/"phase_control_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
