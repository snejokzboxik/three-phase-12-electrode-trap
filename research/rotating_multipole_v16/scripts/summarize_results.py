from __future__ import annotations

import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results"
rows=[json.loads(s) for s in (OUT/"search_results.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]

def write_csv(path,records):
    if not records:return
    keys=list(records[0])
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)

def main():
    stages=defaultdict(lambda:{"tasks":0,"starts":0,"bounded":0,"fast_exit":0,"slow_exit":0})
    cases=[]
    for x in rows:
        a=stages[x["stage"]];a["tasks"]+=1;a["starts"]+=x["n"]
        a["bounded"]+=x["bounded_to_horizon"];a["fast_exit"]+=x["exit_fast_lt5"];a["slow_exit"]+=x["exit_slow"]
        cases.append({"stage":x["stage"],"model":x["model"],"kappa":x["kappa"],"direction":x["direction"],
            "gamma_s-1":x["gamma_s-1"],"velocity_model":x.get("initial_velocity_model","box"),
            "seed":x["seed"],"n":x["n"],"horizon_rf":x["horizon_rf"],
            "bounded":x["bounded_to_horizon"],"bounded_fraction":x["bounded_to_horizon"]/x["n"],
            "fast_exit":x["exit_fast_lt5"],"slow_exit":x["exit_slow"],
            "median_exit_rf":x["median_exit_rf"],"within_250V_conditional_legacy_c4":
             x.get("within_250V_conditional_legacy_c4",abs(x["kappa"])<=.161500947886571)})
    write_csv(OUT/"case_summary.csv",cases)
    stage_rows=[]
    for stage,a in sorted(stages.items()):
        stage_rows.append({"stage":stage,**a,"bounded_fraction":a["bounded"]/a["starts"] if a["starts"] else None})
    write_csv(OUT/"stage_summary.csv",stage_rows)

    reps=defaultdict(list)
    for x in rows:
        if x["stage"]=="conservative_replicate":
            key=(x["model"],x["kappa"],x["direction"])
            reps[key].append(x["bounded_to_horizon"]/x["n"])
    rep_rows=[]
    for (model,k,d),vals in sorted(reps.items()):
        rep_rows.append({"model":model,"kappa":k,"direction":d,"replicates":len(vals),
            "mean_bounded_fraction":float(np.mean(vals)),
            "between_scramble_sd":float(np.std(vals,ddof=1)) if len(vals)>1 else None,
            "fractions":";".join(f"{v:.6g}" for v in vals),"horizon_rf":300,"n_per_scramble":64})
    write_csv(OUT/"independent_replicate_summary.csv",rep_rows)

    paired=defaultdict(list)
    for x in rows:
        if x["stage"]=="dissipative_paired":
            paired[(x["model"],x["kappa"],x["direction"])].append(x)
    paired_rows=[]
    for (model,k,d),items in sorted(paired.items()):
        for x in sorted(items,key=lambda z:z["gamma_s-1"]):
            paired_rows.append({"model":model,"kappa":k,"direction":d,"gamma_s-1":x["gamma_s-1"],
                "seed":x["seed"],"n":x["n"],"horizon_rf":x["horizon_rf"],
                "bounded_fraction":x["bounded_to_horizon"]/x["n"],
                "same_initials_across_gamma":True,"velocity_model":x.get("initial_velocity_model")})
    write_csv(OUT/"paired_damping_summary.csv",paired_rows)
    payload={"result_rows":len(rows),"stages":stage_rows,"independent_replicate_groups":len(rep_rows),
        "paired_damping_cases":len(paired_rows),"files":["case_summary.csv","stage_summary.csv",
        "independent_replicate_summary.csv","paired_damping_summary.csv"],
        "warning":"QMC fractions are conditional on the explicitly defined initial-state distributions; between-scramble SD is the replication diagnostic, not an IID binomial confidence interval."}
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":main()
