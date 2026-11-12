"""Tighter DOP853 checks of observed azimuth-boundary cases."""
from __future__ import annotations
import csv,json,math,time
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
from pathlib import Path
import run_v6

HERE=Path(__file__).resolve().parent
RTOL_TIGHT=2e-11
ATOL_TIGHT=(2e-14,2e-14,2e-12,2e-12)
STEP_FRAC=0.01
RESERVE_S=75


def _rows():
    out={}
    for line in (HERE/"trajectory_checkpoint.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r=json.loads(line);out[r["job_id"]]=r
        except Exception:pass
    return list(out.values())


def _first_bracket(rep,model,direction):
    g={}
    for r in _rows():
        j=r["job"]
        if j["representation"]==rep and j["model"]==model and int(j["direction"])==direction and float(j["phase_deg"])==0:
            a=float(j["azimuth_deg"])
            if 0<=a<=22.5:g[a]=r["status"]
    p=sorted(g.items())
    return next(((a,b,sa,sb) for (a,sa),(b,sb) in zip(p,p[1:]) if sa!=sb),None)


def choose_jobs():
    jobs=[]
    for rep in ("N48","pure_m4"):
        for direction in (-1,1):
            bracket=_first_bracket(rep,"standing",direction)
            if bracket is None:continue
            lo,hi,slo,shi=bracket
            for az,side in ((lo,"retained_side"),(hi,"escaped_side")):
                if az==0.0:continue
                j=run_v6._make_job("strict_validation",rep,"standing",az,0.0,direction,
                    sampling="tight_boundary_recheck")
                j.update({"rtol":RTOL_TIGHT,"atol":list(ATOL_TIGHT),
                    "max_step_RF_fraction":STEP_FRAC,"samples_per_RF":1,"strict_side":side})
                j["job_id"]=run_v6._job_id(j);jobs.append(j)
    unique={j["job_id"]:j for j in jobs}
    return list(unique.values())


def main():
    jobs=choose_jobs();done=run_v6.read_done();todo=[j for j in jobs if j["job_id"] not in done]
    deadline=json.loads((HERE/"integration_deadline.json").read_text(encoding="utf-8"))["deadline_unix_s"]
    results=[]
    with ProcessPoolExecutor(max_workers=4,initializer=run_v6._init_worker) as pool:
        pending={};idx=0
        while idx<len(todo) or pending:
            while idx<len(todo) and len(pending)<4:
                if time.time()+RESERVE_S>=deadline:
                    print("global V6 deadline reserve reached; strict checks not started",flush=True);idx=len(todo);break
                j=todo[idx];pending[pool.submit(run_v6._worker,j)]=j;idx+=1
            if not pending:break
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in ready:
                pending.pop(f);r=f.result();run_v6._append_result(r);results.append(r)
    all_rows=_rows()
    index={(r["job"]["representation"],r["job"]["model"],round(float(r["job"]["azimuth_deg"]),12),int(r["job"]["direction"]),round(float(r["job"]["phase_deg"]),9)):r for r in all_rows}
    compare=[]
    for j in jobs:
        k=(j["representation"],j["model"],round(float(j["azimuth_deg"]),12),int(j["direction"]),round(float(j["phase_deg"]),9))
        strict=index.get(k)
        if strict is None:continue
        standard=next((r for r in all_rows if r["job"]["representation"]==j["representation"] and r["job"]["model"]==j["model"]
            and float(r["job"]["azimuth_deg"])==float(j["azimuth_deg"]) and int(r["job"]["direction"])==int(j["direction"])
            and float(r["job"]["phase_deg"])==float(j["phase_deg"]) and r["job"].get("rtol")==run_v6.RTOL),None)
        compare.append({"representation":j["representation"],"model":j["model"],"azimuth_deg":j["azimuth_deg"],
            "direction":j["direction"],"standard_status":standard["status"] if standard else "missing",
            "tight_status":strict["status"],"classification_matches":bool(standard and standard["status"]==strict["status"]),
            "standard_escape_time_RF":standard["escape_time_RF"] if standard else None,
            "tight_escape_time_RF":strict["escape_time_RF"],
            "escape_time_delta_RF":None if not standard or standard["escape_time_RF"] is None or strict["escape_time_RF"] is None else strict["escape_time_RF"]-standard["escape_time_RF"],
            "standard_rtol":run_v6.RTOL,"tight_rtol":RTOL_TIGHT,"standard_max_step_RF_fraction":1/run_v6.MAX_STEP_DIV,
            "tight_max_step_RF_fraction":STEP_FRAC,"tight_nfev":strict.get("nfev")})
    out=HERE/"processed";out.mkdir(parents=True,exist_ok=True)
    if compare:
        with (out/"strict_validation_v6.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(compare[0]));w.writeheader();w.writerows(compare)
    summary={"requested_tight_jobs":len(jobs),"completed_tight_jobs":len(compare),
        "classification_agreements":sum(x["classification_matches"] for x in compare),
        "classification_disagreements":[x for x in compare if not x["classification_matches"]],
        "validation_protocol":{"rtol":RTOL_TIGHT,"atol":ATOL_TIGHT,"max_step_RF_fraction":STEP_FRAC},
        "results":compare}
    (out/"strict_validation_v6.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("requested_tight_jobs","completed_tight_jobs","classification_agreements","classification_disagreements")},indent=2))


if __name__=="__main__":main()
