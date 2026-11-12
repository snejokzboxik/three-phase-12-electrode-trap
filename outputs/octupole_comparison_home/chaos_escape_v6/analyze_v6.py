"""Post-process the exact V3-V6 launch records without interpolation."""
from __future__ import annotations
import csv, json, math, time
from collections import Counter, defaultdict
from pathlib import Path
import run_v6

HERE=Path(__file__).resolve().parent
OUT=HERE/"processed"


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)


def checkpoint_rows():
    rows={}
    for line in (HERE/"trajectory_checkpoint.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r=json.loads(line);rows[r["job_id"]]=r
        except Exception:pass
    return list(rows.values())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    v6=checkpoint_rows()
    source=[]
    for label,path,azkey in (
        ("V4_equal_peak",run_v6.ROOT/"outputs/octupole_comparison_home/confinement_maps_v4/matched_pairs_equal_peak.csv","azimuth_rad"),
        ("V5_matched_strength",run_v6.V5_CSV,"azimuth_deg")):
        rows=read_csv(path)
        for energy in (2.0,3.0):
            for az in (0.,22.5,45.,67.5):
                if label.startswith("V4"):
                    group=[r for r in rows if float(r["energy_eV"])==energy
                           and math.isclose(math.degrees(float(r[azkey])),az,abs_tol=1e-8)]
                else:
                    group=[r for r in rows if float(r["energy_eV"])==energy
                           and float(r[azkey])==az]
                for model in ("rotating","standing"):
                    statuses=Counter(r[model+"_status"] for r in group)
                    source.append({"source":label,"energy_eV":energy,"r0_mm":1.0,"azimuth_deg":az,
                        "model":model,"n_states":len(group),"radii_mm":"1,2","retained":statuses.get("retained",0),
                        "escaped":statuses.get("escaped",0),"phases_deg":"0,90,180,270",
                        "directions":"-1,+1","normalization":"equal peak" if label.startswith("V4") else "matched leading pseudopotential"})
    write_csv(OUT/"source_grid_audit_v6.csv",source)

    grouped=defaultdict(list)
    direction_audit=[]
    for r in v6:
        j=r["job"]; rep=j["representation"];model=j["model"];d=int(j["direction"])
        s=r["initial_state"];x,y,vx,vy=map(float,s);radius=math.hypot(x,y)
        vt=(-vx*y+vy*x)/radius
        direction_audit.append({"job_id":r["job_id"],"stage":j["stage"],"representation":rep,"model":model,
            "azimuth_deg":j["azimuth_deg"],"phase_deg":j["phase_deg"],"recorded_direction":d,
            "computed_tangential_velocity_m_s":vt,"sign_pass":int(d*vt>0)})
        key=(rep,model,d,round(float(j["phase_deg"]),9))
        grouped[key].append((float(j["azimuth_deg"]),r["status"],j["stage"],r))
    write_csv(OUT/"initial_velocity_direction_audit_v6.csv",direction_audit)

    transitions=[];summary=[]
    for key,items in grouped.items():
        rep,model,d,phase=key
        if phase!=0:continue
        by_az={}
        for az,status,stage,row in items:
            if 0<=az<=22.5: by_az[az]=status
        ordered=sorted(by_az.items())
        for (lo,slo),(hi,shi) in zip(ordered,ordered[1:]):
            if slo!=shi:
                transitions.append({"representation":rep,"model":model,"direction":d,"phase_deg":phase,
                    "lower_azimuth_deg":lo,"lower_status":slo,"upper_azimuth_deg":hi,
                    "upper_status":shi,"bracket_width_deg":hi-lo,
                    "interpretation":"observed neighboring classifications; no continuous boundary asserted"})
        source_for_grid=[z for z in ordered if z[0] in {round(i*22.5/64,10) for i in range(65)}]
        summary.append({"representation":rep,"model":model,"direction":d,"phase_deg":phase,
            "unique_azimuths_0_to_22p5":len(by_az),"balanced_grid_states":len(source_for_grid),
            "grid_retained":sum(st=="retained" for _,st in source_for_grid),
            "grid_escaped":sum(st=="escaped" for _,st in source_for_grid),
            "minimum_sampled_escape_RF":min((row["escape_time_RF"] for az,st,stage,row in items
                if st=="escaped" and 0<=az<=22.5 and row.get("escape_time_RF") is not None),default=None)})
    write_csv(OUT/"observed_transition_brackets_v6.csv",transitions)
    write_csv(OUT/"outcome_summary_by_model_direction_v6.csv",summary)

    v5=read_csv(run_v6.V5_CSV); reuse=[]
    for az in (0.0,22.5):
        for direction in (-1,1):
            for model in ("rotating","standing"):
                old=next(r for r in v5 if float(r["energy_eV"])==2 and float(r["r0_mm"])==1
                    and float(r["azimuth_deg"])==az and float(r["phase_deg"])==0
                    and int(float(r["direction"]))==direction)
                new=next(r for r in v6 if r["job"]["representation"]=="N48" and r["job"]["model"]==model
                    and float(r["job"]["azimuth_deg"])==az and int(r["job"]["direction"])==direction
                    and float(r["job"]["phase_deg"])==0)
                old_t=float(old[model+"_escape_time_RF"]) if old[model+"_escape_time_RF"] else None
                new_t=new["escape_time_RF"]
                reuse.append({"azimuth_deg":az,"direction":direction,"model":model,
                    "V5_status":old[model+"_status"],"V6_status":new["status"],
                    "classification_match":old[model+"_status"]==new["status"],
                    "V5_escape_time_RF":old_t,"V6_escape_time_RF":new_t,
                    "escape_time_difference_RF":None if old_t is None or new_t is None else new_t-old_t,
                    "V5_solver_summary":old[model+"_solver_status"],"V6_solver_summary":new["solver_status"]})
    write_csv(OUT/"known_v5_reproduction_v6.csv",reuse)

    all_pass=all(x["sign_pass"]==1 for x in direction_audit)
    config=json.loads((HERE/"integration_deadline.json").read_text(encoding="utf-8"))
    result={"source_grid_rows":len(source),"v6_checkpointed_unique_jobs":len(v6),
        "v6_direction_audit_rows":len(direction_audit),"all_v6_tangential_signs_pass":all_pass,
        "known_v5_reproduction_rows":len(reuse),"known_v5_all_classifications_match":all(r["classification_match"] for r in reuse),
        "observed_phase_zero_transitions":transitions,"outcome_summary":summary,
        "compute_elapsed_s":time.time()-config["start_unix_s"],"deadline":config,
        "classification_interpretation":"1000-RF survivors are right-censored finite-time observations, not permanent stability"}
    (OUT/"analysis_summary_v6.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if not all_pass or not result["known_v5_all_classifications_match"]:
        raise AssertionError("V6 direction audit or known V5 reproduction failed")
    print(json.dumps({k:result[k] for k in ("v6_checkpointed_unique_jobs","all_v6_tangential_signs_pass",
        "known_v5_all_classifications_match","observed_phase_zero_transitions","outcome_summary")},indent=2))


if __name__=="__main__":main()
