"""Checkpointed V9 dissipative RF trajectories and full-spectrum tangent checks.

Uses only the archived FEM-derived N48 coefficients. Every trajectory is saved
as JSON plus compressed stroboscopic/output samples before its next job starts.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
V4DIR=REPO/"outputs/octupole_comparison_home/confinement_maps_v4"
V6DIR=REPO/"outputs/octupole_comparison_home/chaos_escape_v6"
V8DIR=REPO/"outputs/octupole_comparison_home/chaos_pilot_v8"
sys.path.insert(0,str(V4DIR))
import run_v4

CFG=json.loads((HERE/"config_v9.json").read_text(encoding="utf-8"))
FIELD={"h35":run_v4.FIELD35,"h50":run_v4.FIELD50}
T=1/CFG["ion"]["frequency_Hz"]
RMAX=CFG["aperture_mm"]/1000
MASS=CFG["ion"]["mass_Th"]*run_v4.AMU
CHARGE=run_v4.E_CHARGE
QN={}

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def init_worker(field_name):
    os.environ.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    _,q=run_v4.load_fields(FIELD[field_name])
    QN.clear(); QN[field_name]=q

def field_for(field_name,model):
    from three_phase_trap.rf_reconstruction import HarmonicQuadratures
    q=QN[field_name]
    return q if model=="rotating" else HarmonicQuadratures(np.zeros_like(q.cosine),q.sine.copy(),q.reference_radius)

def make_job(field_name,launch,model,gamma,stage="sweep",horizon=None,rtol=None,max_step=None):
    j={"stage":stage,"field":field_name,"model":model,"normalization":"matched_leading_m4_pseudopotential",
       "voltage_V":CFG["normalization"][model+"_voltage_V"],"frequency_Hz":CFG["ion"]["frequency_Hz"],
       "mass_Th":CFG["ion"]["mass_Th"],"charge_state":1,"energy_eV":CFG["initial_energy_eV"],
       "r0_mm":CFG["initial_radius_mm"],"azimuth_deg":launch["azimuth_deg"],"direction":launch["direction"],
       "phase_deg":0.0,"gamma_s_inv":float(gamma),"horizon_RF":int(horizon or CFG["horizon_RF"]),
       "aperture_mm":CFG["aperture_mm"],"rtol":rtol or CFG["integrator"]["rtol"],
       "atol":CFG["integrator"]["atol"],"max_step_RF_fraction":max_step or CFG["integrator"]["max_step_RF_fraction"],
       "samples_per_RF":CFG["integrator"]["samples_per_RF"],"launch_label":launch["label"]}
    j["job_id"]=hashlib.sha256(json.dumps(j,sort_keys=True,separators=(",",":" )).encode()).hexdigest()[:20]
    return j

def rhs_factory(q,j,with_tangent=False):
    model=j["model"]; vol=j["voltage_V"]; gamma=j["gamma_s_inv"]
    qm=CHARGE/MASS; omega=2*math.pi*j["frequency_Hz"]; invref=1/q.reference_radius
    dc=np.polynomial.polynomial.polyder(q.cosine); ds=np.polynomial.polynomial.polyder(q.sine)
    d2c=np.polynomial.polynomial.polyder(q.cosine,2); d2s=np.polynomial.polynomial.polyder(q.sine,2)
    if model=="standing":
        dc=np.zeros(1,dtype=complex); d2c=np.zeros(1,dtype=complex)
    phase0=math.radians(j["phase_deg"])
    def rhs(t,y):
        x=y[:2]; vx=y[2:4]; z=complex(x[0],x[1])*invref
        lam=omega*t+phase0; co,si=math.cos(lam),math.sin(lam)
        pc=np.polynomial.polynomial.polyval(z,q.cosine).real
        ps=np.polynomial.polynomial.polyval(z,q.sine).real
        dcp=np.polynomial.polynomial.polyval(z,dc)*invref
        dsp=np.polynomial.polynomial.polyval(z,ds)*invref
        e=np.array([-dcp.real*co-dsp.real*si,dcp.imag*co+dsp.imag*si])*vol
        acc=qm*e-gamma*vx
        phi=vol*(pc*co+ps*si)
        work=CHARGE*vol*omega*(-pc*si+ps*co)
        dissip=MASS*gamma*float(vx@vx)
        base=np.r_[vx,acc,work,dissip]
        if not with_tangent: return base
        d2cp=np.polynomial.polynomial.polyval(z,d2c)*invref**2
        d2sp=np.polynomial.polynomial.polyval(z,d2s)*invref**2
        jc=np.array([[-d2cp.real,d2cp.imag],[d2cp.imag,d2cp.real]])
        js=np.array([[-d2sp.real,d2sp.imag],[d2sp.imag,d2sp.real]])
        je=vol*(jc*co+js*si)
        W=y[6:].reshape(4,4)
        A=np.zeros((4,4));A[:2,2:]=np.eye(2);A[2:,:2]=qm*je;A[2:,2:]=-gamma*np.eye(2)
        return np.r_[base,(A@W).ravel()]
    return rhs

def run_one(j,field_name=None,save=True):
    field_name=field_name or j["field"]
    q=field_for(field_name,j["model"])
    init=run_v4.make_initial(j["mass_Th"],j["r0_mm"],j["energy_eV"]*1000,math.radians(j["azimuth_deg"]),0.0,float(j["direction"]),"tangent")
    r0=math.hypot(init[0],init[1]); vt=(-init[2]*init[1]+init[3]*init[0])/r0
    if j["direction"]*vt<=0: raise ValueError("tangential velocity sign check failed")
    h=int(round(1/j["max_step_RF_fraction"])); dur=j["horizon_RF"]*T
    n=j["horizon_RF"]*j["samples_per_RF"]
    y0=np.r_[init,0.0,0.0]
    def aperture(t,y): return RMAX-math.hypot(y[0],y[1])
    aperture.terminal=True;aperture.direction=-1
    started=time.perf_counter()
    sol=solve_ivp(rhs_factory(q,j),(0,dur),y0,method="DOP853",rtol=j["rtol"],atol=j["atol"]+[max(1e-30,1e-6*CHARGE)]*2,
                  max_step=T/h,t_eval=np.linspace(0,dur,n+1),events=aperture)
    if not sol.success: raise RuntimeError(sol.message)
    escaped=bool(len(sol.t_events[0])); tt=sol.t; yy=sol.y.T
    if escaped and (not len(tt) or sol.t_events[0][0]>tt[-1]+1e-15):
        tt=np.r_[tt,sol.t_events[0][0]];yy=np.vstack([yy,sol.y_events[0][0]])
    rr=np.hypot(yy[:,0],yy[:,1]); v2=yy[:,2]**2+yy[:,3]**2
    pc,ps=q.potential_quadratures(yy[:,:2]); lam=2*math.pi*j["frequency_Hz"]*tt
    phi=j["voltage_V"]*(pc*np.cos(lam)+ps*np.sin(lam))
    H=.5*MASS*v2+CHARGE*phi; H0=float(H[0]); W=yy[:,4]; D=yy[:,5]
    balance=H-H0-W+D; scale=max(CHARGE*j["energy_eV"],np.max(np.abs(H-H0)),np.max(np.abs(W)),np.max(D),1e-30)
    escape=sol.t_events[0][0]/T if escaped else None
    result={"job":j,"status":"escaped" if escaped else "retained_to_horizon","escape_time_RF":escape,
       "right_censored_at_RF":None if escaped else j["horizon_RF"],"initial_state_SI":init.tolist(),
       "initial_tangent_sign_check_m_s":float(vt),"max_radius_mm":float(np.max(rr)*1e3),
       "rms_radius_mm":float(np.sqrt(np.trapezoid(rr*rr,tt)/max(tt[-1],1e-30))*1e3),
       "final_radius_mm":float(rr[-1]*1e3),"final_speed_m_s":float(math.sqrt(v2[-1])),
       "initial_kinetic_eV":float(.5*MASS*v2[0]/CHARGE),"final_kinetic_eV":float(.5*MASS*v2[-1]/CHARGE),
       "friction_dissipated_eV":float(D[-1]/CHARGE),"rf_work_eV":float(W[-1]/CHARGE),
       "mechanical_energy_change_eV":float((H[-1]-H0)/CHARGE),"energy_balance_max_abs_eV":float(np.max(np.abs(balance))/CHARGE),
       "energy_balance_max_relative":float(np.max(np.abs(balance))/scale),"friction_dissipation_nonnegative":bool(np.min(D)>=-1e-26),
       "solver_status":sol.message,"nfev":int(sol.nfev),"wall_s":time.perf_counter()-started,
       "trace_file":f"traces/{j['job_id']}.npz"}
    if save:
        np.savez_compressed(HERE/"traces"/(j["job_id"]+".npz"),time_RF=tt*j["frequency_Hz"],state=yy[:,:4],kinetic_eV=.5*MASS*v2/CHARGE,
                            mechanical_energy_eV=H/CHARGE,rf_work_eV=W/CHARGE,friction_dissipated_eV=D/CHARGE)
    return result

def jobs():
    out=[]
    for launch in CFG["launches"]:
        for model in ("rotating","standing"):
            gs=CFG["gamma_s_inv"] if launch["label"]=="deep_0deg" else [0,100,1000,3000]
            for gamma in gs: out.append(make_job("h50",launch,model,gamma))
    return out

def completed_index():
    out={}
    p=HERE/"trajectory_checkpoint.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r=json.loads(line);out[r["job"]["job_id"]]=r
            except Exception: pass
    return out

def save_result(r):
    atomic_json(HERE/"completed"/(r["job"]["job_id"]+".json"),r)
    with (HERE/"trajectory_checkpoint.jsonl").open("a",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(r,separators=(",",":"),allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
    print(f"{r['job']['field']} {r['job']['model']} gamma={r['job']['gamma_s_inv']:g} {r['job']['launch_label']}: {r['status']} t={r['escape_time_RF']} wall={r['wall_s']:.1f}s",flush=True)

def run_sweep():
    start=time.time();deadline=start+10*60
    atomic_json(HERE/"deadline.json",{"start_unix_s":start,"stop_submitting_unix_s":deadline,"budget_min":10,"workers":4})
    (HERE/"traces").mkdir(exist_ok=True)
    done=completed_index(); todo=[j for j in jobs() if j["job_id"] not in done]
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=("h50",)) as pool:
        pending={}; k=0
        while k<len(todo) or pending:
            while k<len(todo) and len(pending)<4:
                if time.time()>=deadline:
                    print("submission deadline reached; draining completed jobs",flush=True);k=len(todo);break
                pending[pool.submit(run_one,todo[k],"h50")]=todo[k];k+=1
            if not pending: break
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in ready: pending.pop(f);save_result(f.result())
    export_csv()

def run_boundary_pair_checks():
    """Finish a balanced four-job check on the positive-velocity boundary."""
    launch=next(x for x in CFG["launches"] if x["label"]=="boundary_positive")
    requested=[make_job("h50",launch,m,g) for m in ("rotating","standing") for g in (0,1000)]
    done=completed_index();todo=[j for j in requested if j["job_id"] not in done]
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=("h50",)) as pool:
        futures=[pool.submit(run_one,j,"h50") for j in todo]
        for f in futures:save_result(f.result())
    export_csv()

def h35_validation():
    init_worker("h35");rows=[]
    cases=[("deep_0deg","rotating",0,1),("early_escape_22p5deg","standing",22.5,1),
           ("boundary_negative","standing",9.613,-1),("boundary_positive","standing",10.2722,1)]
    references=[]
    for source in (V8DIR/"trajectory_checkpoint.jsonl",V6DIR/"trajectory_checkpoint.jsonl"):
        if not source.exists(): continue
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                z=json.loads(line);j=z.get("job",{})
                if j.get("field") in ("h35_N48","h35") or j.get("representation")=="N48": references.append(z)
            except Exception: pass
    for label,model,az,d in cases:
        launch={"label":label,"azimuth_deg":az,"direction":d};j=make_job("h35",launch,model,0,stage="h35_gamma0_check")
        j["job_id"]=hashlib.sha256((j["job_id"]+"h35ref").encode()).hexdigest()[:20]
        old_path=HERE/"h35_validation"/(j["job_id"]+".json")
        if old_path.exists(): r=json.loads(old_path.read_text(encoding="utf-8"))["result"]
        else: r=run_one(j,"h35")
        matches=[]
        for old in references:
            jo=old["job"]
            if jo.get("representation")=="N48" and jo.get("model")==model and jo.get("azimuth_deg")==az and jo.get("phase_deg")==0 and jo.get("direction")==d:
                matches.append(old)
        ref=matches[0] if matches else None
        row={"launch":label,"model":model,"h35_outcome":r["status"],"h35_escape_time_RF":r["escape_time_RF"],
             "v8_reference_outcome":None if ref is None else ref["status"],"v8_escape_time_RF":None if ref is None else ref.get("escape_time_RF"),
             "classification_agrees":None if ref is None else ((r["status"]=="escaped")== (ref["status"]=="escaped")),
             "escape_time_difference_RF":None if not ref or r["escape_time_RF"] is None or ref.get("escape_time_RF") is None else r["escape_time_RF"]-ref["escape_time_RF"]}
        rows.append(row); atomic_json(old_path,{"result":r,"comparison":row})
    atomic_json(HERE/"h35_validation"/"summary.json",rows)

def lyapunov_job(j,field_name="h50",transient_RF=200,checkpoints=(400,700,1000)):
    q=field_for(field_name,j["model"]); init=run_v4.make_initial(j["mass_Th"],j["r0_mm"],j["energy_eV"]*1000,math.radians(j["azimuth_deg"]),0,float(j["direction"]),"tangent")
    # Integrate the transient without tangent vectors, then QR-renormalize the complete 4D tangent basis every 10 RF periods.
    trans=make_job(field_name,{"label":"transient","azimuth_deg":j["azimuth_deg"],"direction":j["direction"]},j["model"],j["gamma_s_inv"],"lyapunov",horizon=transient_RF)
    trans["rtol"]=j["rtol"];trans["max_step_RF_fraction"]=j["max_step_RF_fraction"]
    # Integrate the transient once without tangent vectors.
    y0=np.r_[init,0.,0.]; rhs=rhs_factory(q,trans)
    end=transient_RF*T
    def transient_aperture(tt,yy): return RMAX-math.hypot(yy[0],yy[1])
    transient_aperture.terminal=True;transient_aperture.direction=-1
    tr=solve_ivp(rhs,(0,end),y0,method="DOP853",rtol=j["rtol"],atol=j["atol"]+[CHARGE*1e-6]*2,max_step=T/50,events=transient_aperture)
    if not tr.success: raise RuntimeError(tr.message)
    if len(tr.t_events[0]): return {"status":"escaped_in_transient","escape_time_RF":float(tr.t_events[0][0]/T)}
    state=np.r_[tr.y[:4,-1],0.,0.,np.eye(4).ravel()]
    rhsaug=rhs_factory(q,j,True); times=[]; sums=np.zeros(4); t=end; total_logs=np.zeros(4)
    duration=j["horizon_RF"]*T; checkpoints=set(checkpoints); samples=[]; escaped=False
    while t<duration-1e-14:
        end=min(t+10*T,duration); term=[]
        def aperture(tt,y): return RMAX-math.hypot(y[0],y[1])
        aperture.terminal=True;aperture.direction=-1
        sol=solve_ivp(rhsaug,(t,end),state,method="DOP853",rtol=j["rtol"],atol=j["atol"]+[CHARGE*1e-6]*2+[2e-12]*16,max_step=T/50,events=aperture)
        if not sol.success: raise RuntimeError(sol.message)
        if len(sol.t_events[0]): state=sol.y_events[0][0];t=sol.t_events[0][0];escaped=True
        else: state=sol.y[:,-1];t=end
        W=state[6:].reshape(4,4); Q,R=np.linalg.qr(W); signs=np.sign(np.diag(R));signs[signs==0]=1;Q=Q*signs;R=signs[:,None]*R
        total_logs+=np.log(np.maximum(np.abs(np.diag(R)),1e-300));state[6:]=Q.ravel()
        rf=t/T
        if any(abs(rf-c)<1e-8 for c in checkpoints): samples.append({"end_RF":rf,"lyapunov_s_inv":(total_logs/(t-transient_RF*T)).tolist(),"sum_s_inv":float(total_logs.sum()/(t-transient_RF*T)),"expected_sum_s_inv":-2*j["gamma_s_inv"]})
        if escaped: break
    elapsed=t-transient_RF*T;spec=total_logs/max(elapsed,1e-30)
    return {"job":j,"field":field_name,"status":"escaped_during_tangent" if escaped else "retained_to_horizon","transient_RF":transient_RF,
       "analysis_end_RF":t/T,"lyapunov_spectrum_s_inv":spec.tolist(),"largest_lyapunov_s_inv":float(max(spec)),
       "sum_s_inv":float(sum(spec)),"expected_sum_s_inv":-2*j["gamma_s_inv"],"sum_error_s_inv":float(sum(spec)+2*j["gamma_s_inv"]),
       "convergence_checkpoints":samples}

def export_csv():
    rows=list(completed_index().values());out=HERE/"processed";out.mkdir(exist_ok=True)
    fields=["job_id","field","model","launch_label","gamma_s_inv","azimuth_deg","direction","status","escape_time_RF","right_censored_at_RF","max_radius_mm","rms_radius_mm","final_radius_mm","initial_kinetic_eV","final_kinetic_eV","rf_work_eV","friction_dissipated_eV","mechanical_energy_change_eV","energy_balance_max_relative","trace_file","wall_s"]
    with (out/"trajectory_results_v9.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            j=r["job"];w.writerow({k:r.get(k,j.get(k,"")) for k in fields})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["sweep","boundary-pairs","h35-check","lyapunov"],required=True);p.add_argument("--gamma",type=float,default=1000);p.add_argument("--model",choices=["rotating","standing"],default="rotating");p.add_argument("--launch",default="deep_0deg");p.add_argument("--field",choices=["h35","h50"],default="h50")
    a=p.parse_args()
    (HERE/"completed").mkdir(parents=True,exist_ok=True);(HERE/"traces").mkdir(exist_ok=True)
    if a.stage=="sweep": run_sweep()
    elif a.stage=="boundary-pairs": run_boundary_pair_checks()
    elif a.stage=="h35-check": h35_validation()
    else:
        init_worker(a.field);launch=next(x for x in CFG["launches"] if x["label"]==a.launch)
        j=make_job(a.field,launch,a.model,a.gamma,"lyapunov",horizon=1000);r=lyapunov_job(j,a.field)
        atomic_json(HERE/"lyapunov"/(a.field+"_"+a.model+"_g"+str(int(a.gamma))+"_"+a.launch+".json"),r);print(json.dumps(r,indent=2))
if __name__=="__main__": main()
