"""Budgeted V8 chaos pilot. Reuses the validated V6 integrator/fields."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V6_DIR = REPO / "outputs/octupole_comparison_home/chaos_escape_v6"
sys.path.insert(0, str(V6_DIR))
import run_v6
import phase_space_diagnostics_v6 as v6ps

OUT = HERE
CFG = json.loads((HERE / "config_v8.json").read_text(encoding="utf-8"))

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def job_id(job):
    payload = {k:job[k] for k in sorted(job) if k not in ("job_id", "stage", "sampling_design", "diagnostic_label")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]

def physical_key(job):
    return (job.get("representation"), job.get("model"), round(float(job.get("azimuth_deg",0)),10),
            round(float(job.get("phase_deg",0)),10), int(job.get("direction",1)),
            round(float(job.get("energy_eV",2)),10), round(float(job.get("r0_mm",1)),10),
            round(float(job.get("horizon_RF",1000)),8), float(job.get("voltage_V",0)),
            float(job.get("rtol",2e-9)), round(float(job.get("max_step_RF_fraction",0.02)),10))

def v8_rows():
    p=OUT/"trajectory_checkpoint.jsonl"
    if not p.exists(): return []
    rows=[]
    for line in p.read_text(encoding="utf-8").splitlines():
        try: rows.append(json.loads(line))
        except Exception: pass
    return rows

def v6_rows():
    p=V6_DIR/"trajectory_checkpoint.jsonl"
    rows=[]
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r=json.loads(line)
            if r.get("status") in ("retained","escaped"): rows.append(r)
        except Exception: pass
    return rows

def existing_index():
    idx={}
    for r in v6_rows()+v8_rows():
        j=r.get("job",{})
        if j and r.get("status") in ("retained","escaped"):
            idx[physical_key(j)]=r
    return idx

def make_job(stage, rep, model, az, phase, direction, *, horizon=1000, sample=1, rtol=None, max_step=None):
    j=run_v6._make_job(stage,rep,model,float(az),float(phase),int(direction),sampling="v8_"+stage)
    j["horizon_RF"]=int(horizon); j["samples_per_RF"]=int(sample)
    if rtol is not None: j["rtol"]=float(rtol)
    if max_step is not None: j["max_step_RF_fraction"]=float(max_step)
    j["job_id"]=job_id(j)
    return j

def map_jobs():
    angles=[8.0,9.0,9.5,9.60205,9.62402,10.0,10.26123,10.28320,11.0,12.0]
    phases=[0.0,90.0,180.0,270.0]
    return [make_job("map","N48","standing",a,p,d) for p in phases for a in angles for d in (-1,1)]

def control_jobs():
    # The map's 270-degree section has no V6 counterpart. Other phase/control
    # samples are re-used from V6 wherever all physical parameters match.
    angles=[9.5,10.0,10.5]
    models=[("N48","rotating"),("pure_m4","standing"),("pure_m4","rotating")]
    return [make_job("control",rep,model,a,270.0,d) for rep,model in models for a in angles for d in (-1,1)]

def adaptive_phase_jobs():
    idx=existing_index(); rows=[]
    for r in list(idx.values()):
        j=r.get("job",{})
        if j.get("representation")=="N48" and j.get("model")=="standing" and j.get("phase_deg") in (0.0,90.0,180.0,270.0): rows.append(r)
    # Only refine new half-way RF phases at brackets actually observed in the
    # coarse map. Do not presume a single-valued or monotone boundary.
    groups={}
    for r in rows:
        j=r["job"]; groups.setdefault((float(j["phase_deg"]),int(j["direction"])),[]).append(r)
    todo=[]; seen=set()
    for (phase,direction), group in groups.items():
        group=sorted(group,key=lambda z:float(z["job"]["azimuth_deg"]))
        for left,right in zip(group,group[1:]):
            if left.get("status")==right.get("status"): continue
            a=float(left["job"]["azimuth_deg"]); b=float(right["job"]["azimuth_deg"])
            amid=(a+b)/2
            for p2 in ((phase+45)%360,(phase-45)%360):
                key=(round(amid,8),float(p2),direction)
                if key in seen: continue
                seen.add(key)
                todo.append(make_job("phase_refine","N48","standing",amid,p2,direction))
    return todo[:48]

def sensitivity_jobs():
    # Two established phase-zero transition brackets. Perturb one coordinate
    # at a time in the Euclidean radian metric on (theta, RF phase).
    centers=[("theta_boundary_vtheta_negative",9.6130,-1),
             ("theta_boundary_vtheta_positive",10.2722,1)]
    deltas=[0.2,0.05,0.0125,0.003125]
    jobs=[]; pairs=[]
    for label,theta,direction in centers:
        base=make_job("sensitivity","N48","standing",theta,0.0,direction)
        jobs.append(base); pairs.append({"label":label,"kind":"center","delta_deg":0.0,"job_id":base["job_id"]})
        for eps in deltas:
            for coord in ("azimuth", "phase"):
                for sign in (-1,1):
                    a=theta+sign*eps if coord=="azimuth" else theta
                    p=sign*eps if coord=="phase" else 0.0
                    j=make_job("sensitivity","N48","standing",a,p,direction)
                    jobs.append(j)
                    pairs.append({"label":label,"kind":coord,"delta_deg":eps,"sign":sign,"job_id":j["job_id"]})
    atomic_json(OUT/"sensitivity_pair_design.json",{"metric":"sqrt((delta_theta_rad)^2+(delta_phi_RF_rad)^2)","pairs":pairs})
    return jobs

def tangent_jobs():
    # Three long-lived controls run to 3000 periods; the near-boundary retained
    # launch is run to 1000. Escape terminates its tangent evolution.
    spec=[("N48_rotating_deep_3000","N48","rotating",0.0,1,3000),
          ("N48_standing_deep_3000","N48","standing",0.0,1,3000),
          ("pure_m4_standing_deep_3000","pure_m4","standing",0.0,1,3000),
          ("standing_near_boundary_retained","N48","standing",9.60205,-1,1000)]
    jobs=[]
    for label,rep,model,az,d,h in spec:
        j=make_job("tangent",rep,model,az,0.0,d,horizon=h)
        j["diagnostic_label"]=label; j["job_id"]=job_id(j); jobs.append(j)
    return jobs

def strict_jobs():
    # Resolve two opposite outcomes at the same parameter point across phase;
    # and repeat one phase-zero threshold bracket with the tighter protocol.
    spec=[(9.613037109375,45.0,-1),(9.613037109375,315.0,-1),
          (10.272216796875,45.0,1),(10.272216796875,315.0,1),
          (9.609875,0.0,-1),(9.61300,0.0,-1),
          (10.2597,0.0,1),(10.26123,0.0,1)]
    jobs=[]
    for az,phase,direction in spec:
        j=make_job("strict","N48","standing",az,phase,direction,rtol=2e-10,max_step=0.01)
        j["sampling_design"]="v8_strict_tolerance_half_step";j["job_id"]=job_id(j);jobs.append(j)
    return jobs

def _trajectory_worker(job):
    r=run_v6.integrate_job(job,save_trace=False)
    r["source"]= "new_v8"
    return r

def _tangent_worker(job):
    return v6ps.tangent_job(job)

def run_trajectory_jobs(jobs):
    OUT.mkdir(parents=True,exist_ok=True)
    idx=existing_index(); pending=[]; reused=[]
    for j in jobs:
        key=physical_key(j)
        if key in idx:
            reused.append({"job":j,"source_job_id":idx[key].get("job_id"),"status":idx[key]["status"],"source":"V6" if (V6_DIR/"completed"/(str(idx[key].get("job_id"))+".json")).exists() else "V8"})
        elif not (OUT/"completed"/(j["job_id"]+".json")).exists(): pending.append(j)
    with (OUT/"reused_from_v6.json").open("w",encoding="utf-8") as f: json.dump(reused,f,indent=2)
    start=json.loads((OUT/"deadline.json").read_text(encoding="utf-8"))["start_unix_s"]
    submit_deadline=start+35*60
    with ProcessPoolExecutor(max_workers=4,initializer=run_v6._init_worker) as pool:
        fut={}; k=0
        while k<len(pending) or fut:
            while k<len(pending) and len(fut)<4:
                if time.time()>=submit_deadline:
                    print("budget guard: stopped submitting integrations at 35 minutes",flush=True); k=len(pending); break
                j=pending[k];fut[pool.submit(_trajectory_worker,j)]=j;k+=1
            if not fut: break
            done,_=wait(fut,return_when=FIRST_COMPLETED)
            for f in done:
                fut.pop(f);r=f.result();_save_trajectory(r)
                print(f"{r['job']['stage']} {r['job']['representation']} {r['job']['model']} theta={r['job']['azimuth_deg']:.5f} phase={r['job']['phase_deg']:.1f} sign={r['job']['direction']:+d}: {r['status']} t={r.get('escape_time_RF')} wall={r['wall_s']:.1f}s",flush=True)
    print(f"trajectory batch finished; planned={len(jobs)} reused={len(reused)} newly_done={len(pending)-max(0,k-len(pending))}",flush=True)

def _save_trajectory(r):
    OUT.mkdir(parents=True,exist_ok=True); j=r["job"]
    cp=OUT/"completed"/(r["job_id"]+".json");atomic_json(cp,r)
    with (OUT/"trajectory_checkpoint.jsonl").open("a",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(r,allow_nan=False,separators=(",",":"))+"\n");f.flush();os.fsync(f.fileno())

def run_tangents(jobs):
    (OUT/"tangent_completed").mkdir(exist_ok=True);(OUT/"tangent_traces").mkdir(exist_ok=True)
    todo=[j for j in jobs if not (OUT/"tangent_completed"/(j["job_id"]+".json")).exists()]
    with ProcessPoolExecutor(max_workers=3,initializer=v6ps._init) as pool:
        fut={};k=0
        while k<len(todo) or fut:
            while k<len(todo) and len(fut)<3:
                if time.time()>=json.loads((OUT/"deadline.json").read_text())["start_unix_s"]+35*60:
                    print("budget guard: stopped tangent submissions at 35 minutes",flush=True);k=len(todo);break
                j=todo[k];fut[pool.submit(_tangent_worker,j)]=j;k+=1
            if not fut: break
            done,_=wait(fut,return_when=FIRST_COMPLETED)
            for f in done:
                fut.pop(f);r=f.result(); jid=r["job_id"]
                np=__import__("numpy")
                np.savez_compressed(OUT/"tangent_traces"/(jid+".npz"),strobe=np.asarray(r["strobe"],float),growth=np.asarray(r["growth"],float),hamiltonian_lz=np.asarray(r["hamiltonian_lz_history"],float))
                brief={k:v for k,v in r.items() if k not in ("strobe","growth","hamiltonian_lz_history")}
                atomic_json(OUT/"tangent_completed"/(jid+".json"),brief)
                with (OUT/"tangent_checkpoint.jsonl").open("a",encoding="utf-8",newline="\n") as f:
                    f.write(json.dumps(brief,allow_nan=False,separators=(",",":"))+"\n");f.flush();os.fsync(f.fileno())
                print(f"FTLE {r['job'].get('diagnostic_label')} status={r['status']} horizon={r['job']['horizon_RF']} RF FTLE={r['finite_time_lyapunov_per_RF']:.6g}/RF wall={r['wall_s']:.1f}s",flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["map","controls","phase-refine","sensitivity","strict","tangent","all-map"],required=True);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    deadline=OUT/"deadline.json"
    if not deadline.exists():
        now=time.time();atomic_json(deadline,{"start_unix_s":now,"deadline_unix_s":now+45*60,"integration_submission_stop_unix_s":now+35*60,"budget_min":45,"hard_user_limit_min":50})
    atomic_json(OUT/"config_v8.json",CFG)
    if a.stage=="map": run_trajectory_jobs(map_jobs())
    elif a.stage=="controls": run_trajectory_jobs(control_jobs())
    elif a.stage=="phase-refine": run_trajectory_jobs(adaptive_phase_jobs())
    elif a.stage=="sensitivity": run_trajectory_jobs(sensitivity_jobs())
    elif a.stage=="strict": run_trajectory_jobs(strict_jobs())
    elif a.stage=="tangent": run_tangents(tangent_jobs())
    elif a.stage=="all-map":
        run_trajectory_jobs(map_jobs());run_trajectory_jobs(control_jobs());run_trajectory_jobs(adaptive_phase_jobs());run_trajectory_jobs(sensitivity_jobs())

if __name__=="__main__": main()
