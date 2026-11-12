"""Create transparent processed tables from V6 reuse and V8 checkpoints."""
import csv,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
V6=REPO/"outputs/octupole_comparison_home/chaos_escape_v6"
OUT=HERE/"processed"; OUT.mkdir(exist_ok=True)

def read_jsonl(path):
    if not path.exists(): return []
    out=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        try: out.append(json.loads(line))
        except Exception: pass
    return out

def phys(r):
    j=r.get("job",{})
    return (j.get("representation"),j.get("model"),round(float(j.get("azimuth_deg",0)),9),round(float(j.get("phase_deg",0)),9),int(j.get("direction",1)),float(j.get("energy_eV",2)),float(j.get("r0_mm",1)),int(j.get("horizon_RF",1000)),float(j.get("voltage_V",0)),float(j.get("rtol",2e-9)),round(float(j.get("max_step_RF_fraction",.02)),10))

def collect():
    rows=read_jsonl(V6/"trajectory_checkpoint.jsonl")+read_jsonl(HERE/"trajectory_checkpoint.jsonl")
    unique={}
    for r in rows:
        if r.get("status") not in ("retained","escaped"): continue
        j=r.get("job",{}); key=phys(r)
        if j.get("representation") not in ("N48","pure_m4"): continue
        if key not in unique or r.get("source")=="new_v8": unique[key]=r
    return list(unique.values())

def write_csv(path,rows,fields):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def main():
    allr=collect(); v8_records=read_jsonl(HERE/"trajectory_checkpoint.jsonl"); maprows=[]
    for r in allr:
        j=r["job"]
        if j.get("model")=="standing" and j.get("representation")=="N48" and float(j.get("rtol",2e-9))==2e-9 and round(float(j.get("max_step_RF_fraction",.02)),10)==.02 and 8<=float(j["azimuth_deg"])<=12 and float(j["phase_deg"]) in (0.,45.,90.,135.,180.,225.,270.,315.):
            maprows.append({"representation":j["representation"],"model":j["model"],"theta0_deg":float(j["azimuth_deg"]),"rf_phase_deg":float(j["phase_deg"]),"vtheta_sign":int(j["direction"]),"outcome":r["status"],"escaped":int(r["status"]=="escaped"),"escape_time_RF":r.get("escape_time_RF"),"right_censored_at_RF":r.get("right_censored_at_RF"),"max_radius_mm":r.get("max_radius_mm"),"rms_radius_mm":r.get("rms_radius_mm"),"source":"V8" if r.get("source")=="new_v8" else "V6_reused","source_stage":j.get("stage"),"sampling_design":j.get("sampling_design"),"rtol":j.get("rtol"),"max_step_RF_fraction":j.get("max_step_RF_fraction"),"horizon_RF":j.get("horizon_RF")})
    maprows.sort(key=lambda x:(x["vtheta_sign"],x["rf_phase_deg"],x["theta0_deg"]))
    map_fields=list(maprows[0]) if maprows else ["theta0_deg"]
    write_csv(OUT/"azimuth_phase_map.csv",maprows,map_fields)
    byphase=[]
    for phase in sorted(set(x["rf_phase_deg"] for x in maprows)):
        for d in (-1,1):
            g=[x for x in maprows if x["rf_phase_deg"]==phase and x["vtheta_sign"]==d]
            byphase.append({"phase_deg":phase,"vtheta_sign":d,"n":len(g),"retained":sum(1-x["escaped"] for x in g),"escaped":sum(x["escaped"] for x in g),"minimum_escape_RF":min((x["escape_time_RF"] for x in g if x["escape_time_RF"] is not None),default=None)})
    (OUT/"map_counts.json").write_text(json.dumps(byphase,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    # Control launch conditions, joined with the reused source records.
    controls=[]
    for r in allr:
        j=r["job"]
        if float(j.get("phase_deg",-1))==270 and (j.get("representation"),j.get("model")) in (("N48","rotating"),("pure_m4","standing"),("pure_m4","rotating")):
            controls.append({"representation":j["representation"],"model":j["model"],"theta0_deg":j["azimuth_deg"],"rf_phase_deg":j["phase_deg"],"vtheta_sign":j["direction"],"outcome":r["status"],"escape_time_RF":r.get("escape_time_RF"),"right_censored_at_RF":r.get("right_censored_at_RF"),"source":"V8" if r.get("source")=="new_v8" else "V6_reused"})
    write_csv(OUT/"control_samples.csv",controls,["representation","model","theta0_deg","rf_phase_deg","vtheta_sign","outcome","escape_time_RF","right_censored_at_RF","source"])
    # Paired small perturbations: no exponent fit, only exact counts.
    design=HERE/"sensitivity_pair_design.json"; v8={r["job_id"]:r for r in read_jsonl(HERE/"trajectory_checkpoint.jsonl")}
    sens=[]
    if design.exists():
        d=json.loads(design.read_text(encoding="utf-8"));lookup={}
        for row in d["pairs"]:
            r=v8.get(row["job_id"])
            if r: lookup[row["job_id"]]=r["status"]
        centers={row["label"]:lookup[row["job_id"]] for row in d["pairs"] if row["kind"]=="center" and row["job_id"] in lookup}
        for row in d["pairs"]:
            if row["kind"]=="center" or row["job_id"] not in lookup or row["label"] not in centers: continue
            dist=float(row["delta_deg"])*math.pi/180
            sens.append({"boundary":row["label"],"perturbation_coordinate":row["kind"],"delta_deg":row["delta_deg"],"delta_metric_rad":dist,"sign":row["sign"],"base_outcome":centers[row["label"]],"perturbed_outcome":lookup[row["job_id"]],"different_outcome":int(centers[row["label"]]!=lookup[row["job_id"]])})
    write_csv(OUT/"paired_sensitivity.csv",sens,["boundary","perturbation_coordinate","delta_deg","delta_metric_rad","sign","base_outcome","perturbed_outcome","different_outcome"])
    sensagg=[]
    for coord in sorted(set(x["perturbation_coordinate"] for x in sens)):
        for eps in sorted(set(x["delta_deg"] for x in sens if x["perturbation_coordinate"]==coord)):
            g=[x for x in sens if x["perturbation_coordinate"]==coord and x["delta_deg"]==eps]
            sensagg.append({"coordinate":coord,"delta_deg":eps,"n_pairs":len(g),"n_different":sum(x["different_outcome"] for x in g),"fraction_different":sum(x["different_outcome"] for x in g)/len(g) if g else None})
    write_csv(OUT/"paired_sensitivity_summary.csv",sensagg,["coordinate","delta_deg","n_pairs","n_different","fraction_different"])
    # Tangent growth history from previous V6 runs and new V8 runs.
    tangent_meta=[]
    for base, comp in ((V6/"phase_space", "v6"),(HERE,"v8")):
        completed=base/"completed" if comp=="v6" else HERE/"tangent_completed"
        traces=base/"traces" if comp=="v6" else HERE/"tangent_traces"
        for jf in completed.glob("*.json") if completed.exists() else []:
            try: m=json.loads(jf.read_text(encoding="utf-8"))
            except Exception: continue
            jid=m.get("job_id",jf.stem); npz=traces/(jid+".npz")
            if not npz.exists(): continue
            z=np.load(npz); growth=z["growth"]
            gmax=float(growth[-1,0]) if len(growth) else 0
            row={"source":comp,"job_id":jid,"label":m.get("job",{}).get("diagnostic_label",f"{m.get('job',{}).get('representation')}_{m.get('job',{}).get('model')}_{m.get('job',{}).get('azimuth_deg')}"),"representation":m.get("job",{}).get("representation"),"model":m.get("job",{}).get("model"),"theta0_deg":m.get("job",{}).get("azimuth_deg"),"phase_deg":m.get("job",{}).get("phase_deg"),"direction":m.get("job",{}).get("direction"),"outcome":m.get("status"),"escape_time_RF":m.get("escape_time_RF"),"end_RF":gmax,"ftle_per_RF":m.get("finite_time_lyapunov_per_RF"),"wall_s":m.get("wall_s"),"jacobian_check":m.get("tangent_jacobian_validation_relative_error")}
            for h in [100,300,1000,2000,3000,5000]:
                ix=np.where(growth[:,0]<=h+1e-8)[0]
                if len(ix):
                    t=float(growth[ix[-1],0]); logs=float(growth[ix[-1],1]); row[f"ftle_at_{h}_RF"]=(logs/t if t>0 else None);row[f"end_at_{h}_RF"]=t
                else: row[f"ftle_at_{h}_RF"]=None;row[f"end_at_{h}_RF"]=None
            tangent_meta.append(row)
    write_csv(OUT/"ftle_summary.csv",tangent_meta,["source","job_id","label","representation","model","theta0_deg","phase_deg","direction","outcome","escape_time_RF","end_RF","ftle_per_RF","ftle_at_100_RF","end_at_100_RF","ftle_at_300_RF","end_at_300_RF","ftle_at_1000_RF","end_at_1000_RF","ftle_at_2000_RF","end_at_2000_RF","ftle_at_3000_RF","end_at_3000_RF","ftle_at_5000_RF","end_at_5000_RF","wall_s","jacobian_check"])
    # Strict h35 repeats are compared against the nominal DOP853 classifications.
    def classkey(j): return (j.get("representation"),j.get("model"),round(float(j.get("azimuth_deg",0)),7),round(float(j.get("phase_deg",0)),7),int(j.get("direction",1)),int(j.get("horizon_RF",1000)))
    nominal={classkey(r["job"]):r for r in allr if r.get("job",{}).get("rtol",2e-9)==2e-9 and r.get("job",{}).get("max_step_RF_fraction",.02)==.02}
    strict=[]
    for r in v8_records:
        j=r.get("job",{})
        if j.get("stage")!="strict": continue
        b=nominal.get(classkey(j))
        strict.append({"theta0_deg":j.get("azimuth_deg"),"phase_deg":j.get("phase_deg"),"direction":j.get("direction"),"baseline_status":b.get("status") if b else None,"strict_status":r.get("status"),"baseline_escape_time_RF":b.get("escape_time_RF") if b else None,"strict_escape_time_RF":r.get("escape_time_RF"),"classification_agrees":(b.get("status")==r.get("status")) if b else None,"baseline_rtol":2e-9,"strict_rtol":j.get("rtol"),"baseline_max_step_RF_fraction":.02,"strict_max_step_RF_fraction":j.get("max_step_RF_fraction"),"same_initial_condition":bool(b)})
    (OUT/"strict_solver_comparison.json").write_text(json.dumps(strict,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    h50=[]
    hp=HERE/"h50_validation"/"summary.json"
    if hp.exists():
        hrows=json.loads(hp.read_text(encoding="utf-8"))
        for r in hrows:
            j=r.get("job",{}); az=float(j.get("azimuth_deg",0)); d=int(j.get("direction",1))
            b=next((z for z in allr if z.get("job",{}).get("representation")=="N48" and z.get("job",{}).get("model")=="standing" and abs(float(z["job"].get("azimuth_deg",0))-az)<1e-8 and float(z["job"].get("phase_deg",0))==0 and int(z["job"].get("direction",1))==d and z.get("job",{}).get("rtol",2e-9)==2e-9),None)
            h50.append({"theta0_deg":az,"phase_deg":j.get("phase_deg"),"direction":d,"h35_status":b.get("status") if b else None,"h35_escape_time_RF":b.get("escape_time_RF") if b else None,"h50_status":r.get("status"),"h50_escape_time_RF":r.get("escape_time_RF"),"classification_agrees":bool(b and b.get("status")==r.get("status"))})
    (OUT/"h35_h50_comparison.json").write_text(json.dumps(h50,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"all_unique_trajectory_records":len(allr),"map_points":len(maprows),"phase_counts":byphase,"sensitivity_pairs":len(sens),"tangent_records":len(tangent_meta)},indent=2))

if __name__=="__main__": main()
