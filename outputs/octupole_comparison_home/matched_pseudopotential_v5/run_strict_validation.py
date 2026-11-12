"""Tighter DOP853 checks of three matched rotating-only-survivor cases."""
from __future__ import annotations
import csv, hashlib, json, os, sys, time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"outputs/octupole_comparison_home/confinement_maps_v4"))
import run_v4

def init_worker(out):
    os.environ.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    run_v4.OUT=Path(out)

def selected_jobs():
    p=HERE/"processed/matched_strength_balanced_pairs.csv"
    with p.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    want=[(1.0,180.0,45.0,1.0),(2.0,0.0,22.5,1.0),(3.0,90.0,22.5,1.0)]
    jobs=[]
    for e,phase,az,direction in want:
        matches=[r for r in rows if float(r["energy_eV"])==e and float(r["r0_mm"])==1.0
            and float(r["phase_deg"])==phase and float(r["azimuth_deg"])==az and float(r["direction"])==direction]
        if len(matches)!=1 or matches[0]["difference_class"]!="rotating_only_retained":
            raise ValueError(f"Missing prespecified differing validation case: {(e,phase,az,direction)}")
        r=matches[0]
        for model,voltage in (("rotating",250.0),("standing",353.5533905932738)):
            jobs.append({"model":model,"field":"h35","normalization":"matched_leading_pseudopotential_at_0p5mm",
                "voltage_V":voltage,"frequency_Hz":100000.0,"mz_Th":100000.0,
                "r0_mm":float(r["r0_mm"]),"energy_meV":1000*e,"azimuth_rad":float(r["azimuth_rad"]),
                "phase_rad":float(r["phase_rad"]),"direction":direction,"velocity_kind":"tangent",
                "horizon_RF":1000,"rtol":2e-11,"max_step_div":100,"sample_per_rf":1,
                "stage":"v5_matched_strength_strict_validation","condition_id":r["condition_id"]})
    return jobs

def main():
    jobs=selected_jobs(); journal=HERE/"strict_validation_checkpoint.jsonl"
    latest={}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row=json.loads(line); latest[row["job_id"]]=row
            except Exception: pass
    todo=[j for j in jobs if latest.get(run_v4.task_key(j),{}).get("status") not in ("retained","escaped")]
    budget=json.loads((HERE/"runtime/global_deadline.json").read_text(encoding="utf-8"))
    if time.time()+300>=float(budget["deadline_unix_s"]):
        raise SystemExit("Global deadline guard: no strict-validation jobs submitted.")
    done_now=[]
    run_v4.OUT=HERE
    if todo:
        with ProcessPoolExecutor(max_workers=min(4,len(todo)),initializer=init_worker,initargs=(str(HERE),)) as pool, journal.open("a",encoding="utf-8") as f:
            active={pool.submit(run_v4.worker,j):j for j in todo}
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    job=active.pop(future); row=future.result()
                    row["validation_condition_id"]=job["condition_id"]
                    f.write(json.dumps(row,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
                    done_now.append(row)
    latest.update({r["job_id"]:r for r in done_now})
    result=[]
    for j in jobs:
        r=latest.get(run_v4.task_key(j))
        if r is None: raise RuntimeError(f"Strict result not checkpointed: {run_v4.task_key(j)}")
        result.append({"condition_id":j["condition_id"],"model":j["model"],"energy_eV":j["energy_meV"]/1000,
            "phase_deg":j["phase_rad"]*180/3.141592653589793,"azimuth_deg":j["azimuth_rad"]*180/3.141592653589793,
            "direction":j["direction"],"rtol":j["rtol"],"max_step_RF_fraction":1/j["max_step_div"],
            "status":r["status"],"escape_time_RF":r.get("escape_periods"),"right_censored_at_RF":1000 if r["status"]=="retained" else None,
            "max_radius_mm":r["max_radius_m"]*1000,"nfev":r["nfev"],"solver_job_id":r["job_id"]})
    out=HERE/"processed/strict_validation_v5.csv"
    with out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    standard=list(csv.DictReader((HERE/"processed/matched_strength_balanced_pairs.csv").open(encoding="utf-8-sig",newline="")))
    audit=[]
    for e,phase,az,d in ((1.,180.,45.,1.),(2.,0.,22.5,1.),(3.,90.,22.5,1.)):
        base=next(x for x in standard if float(x["energy_eV"])==e and float(x["r0_mm"])==1
            and float(x["phase_deg"])==phase and float(x["azimuth_deg"])==az and float(x["direction"])==d)
        strict=[x for x in result if x["condition_id"]==base["condition_id"]]
        audit.append({"condition_id":base["condition_id"],"standard_pair_class":base["difference_class"],
            "standard_standing_escape_time_RF":float(base["standing_escape_time_RF"]),
            "strict_rotating_status":next(x["status"] for x in strict if x["model"]=="rotating"),
            "strict_standing_status":next(x["status"] for x in strict if x["model"]=="standing"),
            "strict_standing_escape_time_RF":next(x["escape_time_RF"] for x in strict if x["model"]=="standing"),
            "classification_preserved":next(x["status"] for x in strict if x["model"]=="rotating")=="retained" and next(x["status"] for x in strict if x["model"]=="standing")=="escaped"})
    summary={"protocol":"DOP853 rtol=2e-11, max step T/100, same h35 N48 fields and exact matched starts; 3 prespecified energy/phase cases, both models",
        "completed_strict_trajectories":len(result),"all_selected_classifications_preserved":all(x["classification_preserved"] for x in audit),
        "cases":audit,"right_censoring":"strict survivors are right-censored at 1000 RF; escaped statuses use outward aperture first passage",
        "limitation":"three representative deterministic cases validate numerical robustness, not the entire 192-condition grid"}
    (HERE/"processed/strict_validation_v5.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
