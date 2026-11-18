from __future__ import annotations

import argparse, hashlib, json, math, os, platform, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from v16.models import Field
from v16.integrate import integrate_ensemble

RESULTS=ROOT/"results"; RESULTS.mkdir(exist_ok=True)
JSONL=RESULTS/"search_results.jsonl"; CHECKPOINT=RESULTS/"checkpoint.json"
PROGRESS=RESULTS/"progress.json"; LOG=RESULTS/"runner.log"

def log(msg):
    line=f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
    print(line,flush=True)
    with LOG.open("a",encoding="utf-8") as f:f.write(line+"\n")

def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=_json_default),encoding="utf-8")
    os.replace(tmp,path)

def _json_default(x):
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,)):return float(x)
    if isinstance(x,(np.ndarray,)):return x.tolist()
    if isinstance(x,(complex,)):return {"real":x.real,"imag":x.imag}
    raise TypeError(type(x).__name__)

def valid_results():
    ids=set(); valid=[]
    if not JSONL.exists():return ids,valid
    raw=JSONL.read_bytes(); chunks=raw.splitlines(keepends=True)
    good=[]
    for line in chunks:
        if not line.endswith(b"\n"):break
        try:
            row=json.loads(line)
        except Exception:break
        record_hash=row.get("record_sha256")
        if record_hash:
            payload=dict(row);payload.pop("record_sha256",None)
            check=hashlib.sha256(json.dumps(payload,ensure_ascii=False,default=_json_default,
                    separators=(",",":")).encode("utf-8")).hexdigest()
            if check!=record_hash:
                raise ValueError(f"result checksum mismatch for {row.get('task_id')}")
        if "task_id" in row:
            ids.add(row["task_id"]); valid.append(row);good.append(line)
    # A process interruption can leave a partial final record; discard only that tail.
    if sum(map(len,good))!=len(raw):
        tmp=JSONL.with_suffix(".jsonl.repair")
        tmp.write_bytes(b"".join(good));os.replace(tmp,JSONL)
    return ids,valid

def append_result(row):
    payload=json.dumps(row,ensure_ascii=False,default=_json_default,separators=(",",":"))
    row["record_sha256"]=hashlib.sha256(payload.encode("utf-8")).hexdigest()
    line=(json.dumps(row,ensure_ascii=False,default=_json_default,separators=(",",":"))+"\n").encode("utf-8")
    with JSONL.open("ab",buffering=0) as f:
        f.write(line);os.fsync(f.fileno())
    digest=hashlib.sha256(line).hexdigest()
    atomic_json(CHECKPOINT,{"completed_count":len(completed)+1,"last_task_id":row["task_id"],
                            "last_record_sha256":digest,"updated":time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    atomic_json(PROGRESS,{"status":"RUNNING","completed_count":len(completed)+1,
                          "active_task":row["task_id"],"updated":time.strftime('%Y-%m-%dT%H:%M:%S%z')})

def sobol_initials(n,seed,kappa=0.0,velocity_model="box"):
    m=int(math.ceil(math.log2(n)))
    u=qmc.Sobol(5,scramble=True,seed=seed).random_base2(m)[:n]
    r=.92*np.sqrt(u[:,0]); a=2*math.pi*u[:,1]
    if velocity_model=="box":
        vx=.28*(2*u[:,2]-1);vy=.28*(2*u[:,3]-1)
    elif velocity_model=="quartic_scaled":
        vscale=np.sqrt(abs(kappa))*r*r
        vx=vscale*(2*u[:,2]-1);vy=vscale*(2*u[:,3]-1)
    else:raise ValueError(velocity_model)
    states=np.column_stack((r*np.cos(a),r*np.sin(a),vx,vy))
    phases=2*math.pi*u[:,4]
    return states,phases

def task_id(stage,model,kappa,direction,gamma,seed,velocity_model="box"):
    suffix=":init=quartic_scaled" if velocity_model!="box" else ""
    return f"{stage}:{model}:k={kappa:+.7g}:dir={direction:+g}:gamma={gamma:g}:seed={seed}{suffix}"

def execute_case(*,stage,model,kappa,direction=1,gamma=0.0,seed=1,n=64,
                 horizon=100,steps=96,field_kwargs=None,deadline=None,
                 velocity_model="box"):
    tid=task_id(stage,model,kappa,direction,gamma,seed,velocity_model)
    if tid in completed:return False
    if deadline is not None and time.monotonic()>=deadline:return None
    x,phase=sobol_initials(n,seed,kappa,velocity_model)
    kw=dict(field_kwargs or {})
    if model=="RO":f=Field("RO",kappa=kappa,omega_ro=direction,**kw)
    elif model=="SO":f=Field("SO",kappa_so=kappa,**kw)
    else:raise ValueError(model)
    lam=gamma/(2*math.pi*100_000.0)
    tic=time.perf_counter()
    got=integrate_ensemble(x,f,rf_count=horizon,steps_per_rf=steps,damping=lam,initial_phase=phase)
    elapsed=time.perf_counter()-tic
    exits=got["exit_tau"]/(2*math.pi)
    fast=int(np.sum(np.isfinite(exits)&(exits<5)))
    slow=int(np.sum(np.isfinite(exits)&(exits>=5)))
    bounded=int(np.sum(~np.isfinite(exits)))
    survivors=np.flatnonzero(~np.isfinite(exits))
    keep=survivors[:min(6,len(survivors))]
    row={"task_id":tid,"stage":stage,"model":model,"kappa":kappa,"direction":direction,
         "gamma_s-1":gamma,"lambda":lam,"seed":seed,"n":n,"horizon_rf":horizon,
         "steps_per_rf":steps,"integrator":"vectorized classical RK4",
         "initial_velocity_model":velocity_model,
         "voltage_equivalent_V_abs":abs(kappa)/0.000646003791546284,
         "within_250V_conditional_legacy_c4":abs(kappa)<=0.161500947886571,
         "exit_fast_lt5":fast,"exit_slow":slow,"bounded_to_horizon":bounded,
         "exit_fraction":(fast+slow)/n,"median_exit_rf":float(np.nanmedian(exits)) if np.any(np.isfinite(exits)) else None,
         "median_max_radius":float(np.median(got["max_radius"])),
         "scout_elapsed_s":elapsed,"initial_examples":x[keep].tolist(),
         "initial_phase_examples":phase[keep].tolist(),
         "final_examples":got["state"][keep].tolist(),
         "warning":"BOUNDED_HORIZON is only finite-time survival, not stability."}
    append_result(row);completed.add(tid)
    log(f"done {tid}: bounded={bounded}/{n}, fast_exit={fast}, slow_exit={slow}, {elapsed:.1f}s")
    return row

def acquire_lock():
    if os.name!="nt":return None
    import msvcrt
    f=(RESULTS/"runner.lock").open("a+b")
    f.seek(0,0)
    if f.tell()==0:f.write(b"0");f.flush()
    f.seek(0)
    try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        f.close();raise RuntimeError("another runner already holds results/runner.lock")
    return f

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hours",type=float,default=10.0)
    ap.add_argument("--quick",action="store_true",help="small pilot instead of the adaptive search")
    args=ap.parse_args()
    lock=acquire_lock()
    global completed
    completed,old=valid_results()
    deadline=time.monotonic()+args.hours*3600
    atomic_json(PROGRESS,{"status":"RUNNING","completed_count":len(completed),
                          "start_time":time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                          "hours_cap":args.hours,"host":platform.node(),"logical_cpus":os.cpu_count()})
    log(f"runner started; {len(completed)} completed cases recovered; cap={args.hours:g} h")
    try:
        if args.quick:
            cases=[("RO",.015,1,0,32,20,64),("RO",.1044,-1,0,32,20,64),
                   ("SO",.1044,1,0,32,20,64),("RO",.1044,1,3000,32,30,64)]
            for model,k,d,g,n,h,s in cases:
                if execute_case(stage="pilot",model=model,kappa=k,direction=d,gamma=g,
                                n=n,horizon=h,steps=s,seed=1000+len(completed),deadline=deadline) is None:break
        else:
            # Phase II: space-filling scouts over both signs, field chirality and controls.
            mags=[.005,.015,.03,.06,.1044,.2,.4,.8,1.2]
            seed=2100
            for mag in mags:
                for sign in (-1,1):
                    k=sign*mag
                    for direction in (-1,1):
                        if execute_case(stage="conservative",model="RO",kappa=k,direction=direction,
                                        seed=seed,n=64,horizon=100,steps=96,deadline=deadline) is None:raise TimeoutError
                        seed+=1
                    if execute_case(stage="conservative",model="SO",kappa=k,direction=1,
                                    seed=seed,n=64,horizon=100,steps=96,deadline=deadline) is None:raise TimeoutError
                    seed+=1
            # Independent cold ensemble: velocity is scaled to the local
            # quartic-potential energy, complementing the broad box measure.
            seed=6100
            for mag in [.005,.015,.03,.06,.1044,.15]:
                for sign in (-1,1):
                    k=sign*mag
                    for direction in (-1,1):
                        if execute_case(stage="conservative_cold",model="RO",kappa=k,direction=direction,
                                        seed=seed,n=64,horizon=200,steps=96,deadline=deadline,
                                        velocity_model="quartic_scaled") is None:raise TimeoutError
                        seed+=1
                    if execute_case(stage="conservative_cold",model="SO",kappa=k,direction=1,
                                    seed=seed,n=64,horizon=200,steps=96,deadline=deadline,
                                    velocity_model="quartic_scaled") is None:raise TimeoutError
                    seed+=1
            # Independent scrambled-Sobol confirmation replicates for the
            # low-|kappa| regimes with nonzero finite-time survivors.
            rep_seed=22000
            for mag in [.005,.015,.03]:
                for sign in (-1,1):
                    k=sign*mag
                    settings=[("RO",-1),("RO",1),("SO",1)]
                    for model,direction in settings:
                        for replicate in range(2):
                            if execute_case(stage="conservative_replicate",model=model,kappa=k,direction=direction,
                                    seed=rep_seed,n=64,horizon=300,steps=96,deadline=deadline,
                                    velocity_model="quartic_scaled") is None:raise TimeoutError
                            rep_seed+=1
            # Phase III: lab drag; record only screening statistics. Longer horizon is
            # concentrated where damping can plausibly change the RF-scale motion.
            gammas=[0,30,100,300,1000,3000,10000]
            ks=[-.3,-.1044,-.015,.015,.1044,.3]
            seed=4200;so_seed=8200
            for gamma in gammas:
                h=100 if gamma==0 else (300 if gamma<=300 else 500)
                for k in ks:
                    for direction in (-1,1):
                        if execute_case(stage="dissipative",model="RO",kappa=k,direction=direction,
                                        gamma=gamma,seed=seed,n=32,horizon=h,steps=80,deadline=deadline) is None:raise TimeoutError
                        seed+=1
                    if execute_case(stage="dissipative",model="SO",kappa=k,direction=1,
                                    gamma=gamma,seed=so_seed,n=32,horizon=h,steps=80,deadline=deadline) is None:raise TimeoutError
                    so_seed+=1
            # Paired low-energy drag controls: identical Sobol starts across
            # gamma, handedness, and RO/SO, so drag effects are not confounded
            # with a different initial ensemble.
            paired_k=[-.1044,-.06,-.015,.015,.06,.1044]
            gamma_values=[0,30,100,300,1000,3000,10000]
            for ki,k in enumerate(paired_k):
                for gamma in gamma_values:
                    h=100 if gamma==0 else (300 if gamma<=300 else 500)
                    for mi,(model,direction) in enumerate([("RO",-1),("RO",1),("SO",1)]):
                        seed=12000+ki*3+mi
                        if execute_case(stage="dissipative_paired",model=model,kappa=k,direction=direction,
                                gamma=gamma,seed=seed,n=32,horizon=h,steps=64,deadline=deadline,
                                velocity_model="quartic_scaled") is None:raise TimeoutError
    except TimeoutError:
        log("time cap reached; stopping between tasks with checkpoint intact")
        atomic_json(PROGRESS,{"status":"PAUSED_AT_TIME_CAP","completed_count":len(completed),
                              "updated":time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    except Exception:
        log("runner failed:\n"+traceback.format_exc())
        atomic_json(PROGRESS,{"status":"FAILED","completed_count":len(completed),
                              "error":"see runner.log","updated":time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        raise
    else:
        log("planned search queue complete")
        atomic_json(PROGRESS,{"status":"QUEUE_COMPLETE","completed_count":len(completed),
                              "updated":time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    finally:
        if lock is not None:
            try:
                import msvcrt
                lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
            except Exception:pass

if __name__=="__main__":main()
