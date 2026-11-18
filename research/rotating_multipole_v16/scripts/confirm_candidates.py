from __future__ import annotations

import csv, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16.models import Field, field_rhs
from v16.integrate import integrate_ensemble, largest_lyapunov
from run_search import sobol_initials

OUT=ROOT/"results";POINCARE=OUT/"poincare";POINCARE.mkdir(parents=True,exist_ok=True)
T=2*math.pi;AP=1.0
CONFIRMATIONS=OUT/"candidate_confirmations.json"

def save_payload(payload):
    tmp=CONFIRMATIONS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(CONFIRMATIONS)

def load_rows():
    p=OUT/"search_results.jsonl"
    if not p.exists():return []
    rows=[]
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():rows.append(json.loads(line))
    return rows

def drive_from_row(row,phase0):
    if row["model"]=="RO":
        return Field("RO",kappa=row["kappa"],omega_ro=row["direction"],
                     delta_ro=-row["direction"]*phase0)
    return Field("SO",kappa_so=row["kappa"],delta_so=-phase0)

def dop853_horizon(y0,field,rf0,rf1,rtol=2e-11,atol=2e-13):
    t0=rf0*T;t1=rf1*T
    def event(t,y):return AP*AP-y[0]*y[0]-y[1]*y[1]
    event.terminal=True;event.direction=-1
    sol=solve_ivp(lambda t,y:field_rhs(t,y,field,0.0),(t0,t1),y0,method="DOP853",
                  rtol=rtol,atol=atol,max_step=T/120,events=event,dense_output=True)
    return sol

def main():
    rows=load_rows()
    cands=[x for x in rows if x.get("stage") in {"conservative_cold","conservative_replicate"}
           and x.get("initial_velocity_model")=="quartic_scaled"
           and x.get("bounded_to_horizon",0)>0]
    # Confirm only the energy-scaled ensemble. Collapse independent scrambles
    # to unique physical settings, selecting the longest-horizon replicate
    # with the largest surviving fraction for its high-accuracy seed.
    best={}
    for x in cands:
        key=(x["model"],round(x["kappa"],7),x["direction"])
        score=(x["horizon_rf"],x["bounded_to_horizon"]/x["n"],x["bounded_to_horizon"])
        if key not in best or score>best[key][0]:best[key]=(score,x)
    chosen=sorted((v[1] for v in best.values()),
        key=lambda x:(x.get("within_250V_conditional_legacy_c4",abs(x["kappa"])<=.1615),
                      x["horizon_rf"],x["bounded_to_horizon"]/x["n"]),reverse=True)[:10]
    previous={}
    if CONFIRMATIONS.exists():
        try: previous=json.loads(CONFIRMATIONS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError,OSError): previous={}
    out=list(previous.get("confirmations",[]))
    completed={(x.get("source_task_id"),x.get("initial_index")) for x in out}
    confirmed_1000={(x.get("model"),round(x.get("kappa",0.0),7),x.get("direction"))
                    for x in out if x.get("high_accuracy_status")=="BOUNDED_1000_RF"}
    payload={"created_local":time.strftime('%Y-%m-%d %H:%M:%S %z'),
        "status":"RUNNING","selected_this_pass":len(chosen),
        "completed_this_pass":0,"confirmations":out}
    save_payload(payload)
    for idx,row in enumerate(chosen):
        physical_key=(row["model"],round(row["kappa"],7),row["direction"])
        if physical_key in confirmed_1000:
            continue
        velocity_model=row.get("initial_velocity_model","box")
        init,phases=sobol_initials(row["n"],row["seed"],row["kappa"],velocity_model)
        scout=integrate_ensemble(init,Field(row["model"],kappa=row["kappa"] if row["model"]=="RO" else 0,
             kappa_so=row["kappa"] if row["model"]=="SO" else 0,
             omega_ro=row.get("direction",1)),rf_count=row["horizon_rf"],
             steps_per_rf=row["steps_per_rf"],initial_phase=phases)
        survivor_ids=np.flatnonzero(~np.isfinite(scout["exit_tau"]))
        if not len(survivor_ids):continue
        sid=int(survivor_ids[0]);y0=init[sid];phase0=float(phases[sid]);field=drive_from_row(row,phase0)
        if (row["task_id"],sid) in completed:
            continue
        t0=time.perf_counter()
        sol=dop853_horizon(y0,field,0,300,rtol=2e-11,atol=2e-13)
        reached300=sol.status==0
        exit_rf=None if reached300 else float(sol.t[-1]/T)
        final=sol.y[:,-1].copy();current=sol.t[-1]
        reached1000=False;sol2=None
        if reached300:
            sol2=dop853_horizon(final,field,300,1000,rtol=2e-11,atol=2e-13)
            reached1000=sol2.status==0
            if not reached1000:exit_rf=float(sol2.t[-1]/T)
            end=sol2.t[-1];end_y=sol2.y[:,-1]
        else:end=sol.t[-1];end_y=final
        max_rf=1000 if reached1000 else int(math.floor(end/T))
        strobe=np.arange(0,max_rf+1)*T
        endpoint_tol=1e-7
        if sol2 is not None:
            # Do not sample through the event; keep only times before each solution end.
            parts=[];times=[]
            a=strobe[(strobe<=min(300*T,end)+endpoint_tol)]
            a=a[a<=sol.t[-1]+endpoint_tol]
            if len(a):
                parts.append(sol.sol(np.clip(a,sol.t[0],sol.t[-1])))
                times.extend(a.tolist())
            b=strobe[(strobe>300*T)&(strobe<=sol2.t[-1]+endpoint_tol)]
            if len(b):
                parts.append(sol2.sol(np.clip(b,sol2.t[0],sol2.t[-1])))
                times.extend(b.tolist())
            p=np.concatenate(parts,axis=1) if parts else np.empty((4,0));tgrid=np.array(times)
        else:
            tgrid=strobe[strobe<=end+endpoint_tol]
            p=sol.sol(np.clip(tgrid,sol.t[0],sol.t[-1])) if len(tgrid) else np.empty((4,0))
        filename=(f"candidate_{idx+1:02d}_{row['stage']}_s{row['seed']}_{row['model']}"
                  f"_k{row['kappa']:+.4g}_d{row.get('direction',1):+g}.csv").replace("+","p").replace("-","m")
        if p.shape[1]:
            with (POINCARE/filename).open("w",newline="",encoding="utf-8") as f:
                wr=csv.writer(f);wr.writerow(["rf","x","y","vx","vy"])
                wr.writerows(zip(tgrid/T,*p))
        lle=None
        if reached1000:
            le=largest_lyapunov(y0,field,rf_count=100,rtol=5e-10,atol=5e-12)
            if le.get("status")=="BOUNDED_HORIZON":lle={"100rf":le["exponent"],"cumulative":le["cumulative"].tolist()}
        row_out={"source_task_id":row["task_id"],"model":row["model"],"kappa":row["kappa"],
            "direction":row.get("direction"),"velocity_model":velocity_model,"seed":row["seed"],
            "initial_index":sid,"initial_state":y0.tolist(),"initial_rf_phase":phase0,
            "scout_bounded_count":row["bounded_to_horizon"],"high_accuracy_status":
            ("BOUNDED_1000_RF" if reached1000 else "BOUNDED_300_RF" if reached300 else "EXIT_BEFORE_300_RF"),
            "first_exit_rf":exit_rf,"final_state":end_y.tolist(),"strobe_count":int(p.shape[1]),
            "poincare_csv":str((POINCARE/filename).relative_to(ROOT)) if p.shape[1] else None,
            "largest_lyapunov_100rf":lle,"high_accuracy_rtol":2e-11,"elapsed_s":time.perf_counter()-t0,
            "interpretation":"finite-time confirmation only; positive/negative finite-time exponent alone is not a chaos/stability proof"}
        out.append(row_out)
        completed.add((row["task_id"],sid))
        payload={"created_local":time.strftime('%Y-%m-%d %H:%M:%S %z'),
            "status":"RUNNING","selected_this_pass":len(chosen),
            "completed_this_pass":sum((x.get("source_task_id"),x.get("initial_index")) in completed for x in out),
            "confirmations":out}
        save_payload(payload)
    payload={"created_local":time.strftime('%Y-%m-%d %H:%M:%S %z'),
        "status":"COMPLETED","selected_this_pass":len(chosen),
        "completed_this_pass":sum((x.get("source_task_id"),x.get("initial_index")) in completed for x in out),
        "candidate_count":len(out),"confirmations":out}
    save_payload(payload)
    print(json.dumps({k:v for k,v in payload.items() if k!="confirmations"},indent=2,ensure_ascii=False))

if __name__=="__main__":main()
