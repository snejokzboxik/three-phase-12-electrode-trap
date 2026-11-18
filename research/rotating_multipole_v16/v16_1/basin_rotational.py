"""Independent scrambled-Sobol rotational basin comparison for RO and SO.

This is a centered-body angular projection, not the requested full 6D basin.
It records checkpoints after each batch and does not continue escaped COM starts.
"""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"
K=.1044; ELL=.2; GAMMA_ROT=.003; N=256; N_RF=1000
SEEDS=(1701,4813); STEPS=(128,256)

def run_batch(kind,seed,n=N,nrf=N_RF,steps_per_rf=128):
    I=ELL**2/12; amp=K*ELL**4/80
    sample=qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(math.log2(n)))
    psi=sample[:,0]*math.pi
    spin=(sample[:,1]-.5)*1.6
    phase0=sample[:,2]*2*math.pi
    h=2*math.pi/steps_per_rf; total=nrf*steps_per_rf; t=0.
    tail_len=30*steps_per_rf; tail=np.empty((tail_len,n)); j=0
    def f(t,psi,spin):
        tau=t+phase0
        if kind=="RO": torque=4*amp*np.sin(4*psi-tau)
        else: torque=4*amp*np.sin(4*psi)*np.cos(tau)
        return spin,torque/I-GAMMA_ROT*spin
    for nstep in range(total):
        p1,s1=f(t,psi,spin)
        p2,s2=f(t+h/2,psi+h*p1/2,spin+h*s1/2)
        p3,s3=f(t+h/2,psi+h*p2/2,spin+h*s2/2)
        p4,s4=f(t+h,psi+h*p3,spin+h*s3)
        psi+=h*(p1+2*p2+2*p3+p4)/6
        spin+=h*(s1+2*s2+2*s3+s4)/6
        t+=h
        if nstep>=total-tail_len:
            tail[j]=spin; j+=1
    mean=tail.mean(axis=0)
    if kind=="RO": labels=np.where(np.abs(mean-.25)<.015,"lock_plus_quarter","other")
    else:
        labels=np.where(np.abs(mean-.25)<.015,"lock_plus_quarter",
                 np.where(np.abs(mean+.25)<.015,"lock_minus_quarter",
                 np.where(np.abs(mean)<.015,"libration","other")))
    vals,counts=np.unique(labels,return_counts=True)
    return {"field":kind,"seed":int(seed),"n":int(n),"horizon_rf":int(nrf),
            "steps_per_rf":int(steps_per_rf),"initial_psi":"uniform [0,pi)",
            "initial_spin":"uniform [-0.8,0.8]", "initial_rf_phase":"uniform [0,2pi)",
            "classification_average_last_rf":30,"lock_mean_spin_tolerance":.015,
            "counts":{str(k):int(v) for k,v in zip(vals,counts)},
            "mean_spins":mean.tolist(),"classifications":labels.tolist()}

def run_full_batch(kind,seed,n=64,nrf=1000,steps_per_rf=128):
    """Vectorized six-state RK4 ensemble with early stop at rod/aperture exit."""
    ell=ELL; I=ell*ell/12; mu2=ell*ell/12; mu4=ell**4/80; amp=K
    q=qmc.Sobol(7,scramble=True,seed=seed).random_base2(int(math.log2(n)))
    angle=q[:,0]*2*math.pi; radius=.15*np.sqrt(q[:,1])
    x=radius*np.cos(angle); y=radius*np.sin(angle)
    vx=(q[:,2]-.5)*.04; vy=(q[:,3]-.5)*.04
    psi=q[:,4]*math.pi; spin=(q[:,5]-.5)*1.6; phase0=q[:,6]*2*math.pi
    state=np.stack([x,y,vx,vy,psi,spin],axis=1)
    h=2*math.pi/steps_per_rf; total=nrf*steps_per_rf; active=np.ones(n,bool)
    escaped=np.zeros(n,bool); maxrad=np.zeros(n)
    tail_n=30*steps_per_rf; tail_spin=np.zeros((tail_n,n)); tail_plus=np.zeros((tail_n,n)); tail_minus=np.zeros((tail_n,n)); j=0
    strobe_spin=np.zeros((30,n)); strobe_plus=np.zeros((30,n)); strobe_minus=np.zeros((30,n))
    capture_plus=np.full(n,np.nan); capture_minus=np.full(n,np.nan)
    run_plus=np.zeros(n,int); run_minus=np.zeros(n,int); strobe_n=0
    def rhs(t,Y):
        x,y,vx,vy,psi,spin=Y.T; Z=x+1j*y; ep=np.exp(1j*psi)
        px=4*Z**3+12*mu2*Z*ep**2
        ppsi=1j*(12*mu2*Z**2*ep**2+4*mu4*ep**4)
        tau=t+phase0
        fac=amp*np.exp(-1j*tau) if kind=="RO" else amp*np.cos(tau)
        gx=np.real(fac*px); gy=np.real(1j*fac*px); gp=np.real(fac*ppsi)
        d=np.stack([vx,vy,-gx-.05*vx,-gy-.05*vy,spin,-gp/I-GAMMA_ROT*spin],axis=1)
        d[~active]=0.
        return d
    for istep in range(total):
        t=istep*h
        k1=rhs(t,state); k2=rhs(t+h/2,state+h*k1/2); k3=rhs(t+h/2,state+h*k2/2); k4=rhs(t+h,state+h*k3)
        state += h*(k1+2*k2+2*k3+k4)/6
        rr=np.hypot(state[:,0],state[:,1]); maxrad=np.maximum(maxrad,rr)
        out=active & (rr+ell/2>=1.)
        escaped |= out; active[out]=False
        if (istep+1)%steps_per_rf==0:
            rf_index=(istep+1)//steps_per_rf
            slot=(rf_index-1)%30; tau=(istep+1)*h+phase0
            strobe_spin[slot]=state[:,5]
            strobe_plus[slot]=state[:,4]-tau/4
            strobe_minus[slot]=state[:,4]+tau/4
            strobe_n+=1
            if strobe_n>=30:
                mean_s=strobe_spin.mean(axis=0)
                span_p=np.ptp(strobe_plus,axis=0); span_m=np.ptp(strobe_minus,axis=0)
                good_p=active&(np.abs(mean_s-.25)<.015)&(span_p<.15)
                good_m=active&(np.abs(mean_s+.25)<.015)&(span_m<.15)
                run_plus=np.where(good_p,run_plus+1,0); run_minus=np.where(good_m,run_minus+1,0)
                first_p=(run_plus==5)&np.isnan(capture_plus)
                first_m=(run_minus==5)&np.isnan(capture_minus)
                capture_plus[first_p]=rf_index-4
                capture_minus[first_m]=rf_index-4
        if istep>=total-tail_n:
            tau=istep*h+phase0
            tail_spin[j]=state[:,5]
            tail_plus[j]=state[:,4]-tau/4
            tail_minus[j]=state[:,4]+tau/4
            j+=1
    mean=tail_spin.mean(axis=0)
    pspan=np.ptp(tail_plus,axis=0); mspan=np.ptp(tail_minus,axis=0)
    cls=np.full(n,"other",dtype=object)
    cls[escaped]="escaped"
    if kind=="RO":
        cap=(np.abs(mean-.25)<.015)&(pspan<.15)&active; cls[cap]="lock_plus_quarter"
    else:
        cp=(np.abs(mean-.25)<.015)&(pspan<.15)&active
        cm=(np.abs(mean+.25)<.015)&(mspan<.15)&active
        cls[cp]="lock_plus_quarter"; cls[cm]="lock_minus_quarter"
        libr=(np.abs(mean)<.015)&active; cls[libr]="libration"
    vals,counts=np.unique(cls,return_counts=True)
    return {"field":kind,"seed":int(seed),"n":int(n),"horizon_rf":int(nrf),
            "steps_per_rf":int(steps_per_rf),"COM_initial":"uniform disk radius 0.15R",
            "velocity_initial":"independent uniform [-0.02,0.02] in normalized units",
            "orientation_initial":"uniform [0,pi)","spin_initial":"uniform [-0.8,0.8]",
            "RF_phase_initial":"uniform [0,2pi)","rod_clearance":"center radius + ell/2 < 1",
            "capture_window_rf":30,"relative_phase_span_rad_tolerance":.15,
            "counts":{str(k):int(v) for k,v in zip(vals,counts)},
            "capture_time_definition":"first 30-RF stroboscopic window meeting spin/phase criteria for five consecutive RF periods",
            "capture_time_rf":{"plus":{"n":int(np.isfinite(capture_plus).sum()),
                "median":float(np.nanmedian(capture_plus)) if np.isfinite(capture_plus).any() else None,
                "min":float(np.nanmin(capture_plus)) if np.isfinite(capture_plus).any() else None,
                "max":float(np.nanmax(capture_plus)) if np.isfinite(capture_plus).any() else None},
                "minus":{"n":int(np.isfinite(capture_minus).sum()),
                "median":float(np.nanmedian(capture_minus)) if np.isfinite(capture_minus).any() else None,
                "min":float(np.nanmin(capture_minus)) if np.isfinite(capture_minus).any() else None,
                "max":float(np.nanmax(capture_minus)) if np.isfinite(capture_minus).any() else None}},
            "max_center_radius_R":float(maxrad.max())}

def save(path,obj): path.write_text(json.dumps(obj,indent=2),encoding="utf-8")

def main():
    ck=OUT/"basin_rotational_checkpoint.json"
    full_only=os.environ.get("V16_1_FULL_ONLY")=="1"
    if full_only and (OUT/"basin_rotational.json").exists():
        state=json.loads((OUT/"basin_rotational.json").read_text(encoding="utf-8"))
        convergence=state.get("time_refinement_subset",[])
    else:
        state={"schema":"v16.1-rotational-basin-v1","parameters":{"kappa4":K,"ell":ELL,
                "gamma_rot":GAMMA_ROT,"N_RF":N_RF,"seeds":list(SEEDS)},"completed":[]}
        convergence=[]
    if not full_only:
        for kind in ("RO","SO"):
            for seed in SEEDS:
                r=run_batch(kind,seed,steps_per_rf=128)
                r.pop("mean_spins"); r.pop("classifications")
                state["completed"].append(r); save(ck,state)
        # Same 64-start scrambled subset, 128 versus 256 steps/RF convergence check.
        for kind in ("RO","SO"):
            a=run_batch(kind,1701,n=64,steps_per_rf=128)
            b=run_batch(kind,1701,n=64,steps_per_rf=256)
            convergence.append({"field":kind,"n":64,"seed":1701,
                "counts_128":a["counts"],"counts_256":b["counts"],
                "same_classification_counts":a["counts"]==b["counts"],
                "same_individual_classifications":a["classifications"]==b["classifications"]})
    full=[]
    for kind in ("RO","SO"):
        for seed in SEEDS:
            r=run_full_batch(kind,seed); full.append(r)
            state["full_6d_completed"]=full; save(ck,state)
    state["full_6d_scope"]="initial COM displacement, velocity, orientation, spin, and RF phase randomized; early stop on body/aperture exit"
    state["time_refinement_subset"]=convergence
    state["scope_limit"]="angular projection has large N; full six-state basin uses smaller N=64 independent starts per field and seed"
    state["full_6d_batches"]=full
    save(OUT/"basin_rotational.json",state)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ck,OUT/"basin_rotational.json"]}
    save(OUT/"basin_manifest.json",{"schema":"v16.1-basin-manifest-v1","sha256":hashes,
         "seeds":list(SEEDS),"checkpoint":"basin_rotational_checkpoint.json"})
    print(json.dumps({"completed":state["completed"],"time_refinement_subset":convergence,
                      "full_6d_batches":full},indent=2))

if __name__=="__main__": main()
