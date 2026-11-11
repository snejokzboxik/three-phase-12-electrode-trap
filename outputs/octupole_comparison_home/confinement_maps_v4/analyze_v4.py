"""Collect, validate and adaptively refine the V4 confinement campaign."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, shutil, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
V3 = ROOT / "outputs/octupole_comparison_home/rf_confinement_v1"
sys.path.insert(0, str(OUT))
import run_v4


def fnum(x, default=None):
    try:
        if x is None or str(x).strip() == "": return default
        return float(x)
    except (TypeError, ValueError):
        return default


def basekey(row):
    j = row.get("job", row)
    return tuple(float(j[k]) for k in ("r0_mm", "energy_meV", "azimuth_rad", "phase_rad", "direction")) + (
        str(j.get("velocity_kind", "tangent")), str(j["model"]), str(j.get("normalization", "equal_peak_voltage")),
        float(j.get("voltage_V", 250.)), float(j.get("frequency_Hz", 100000.)), str(j.get("field", "h35")),
        int(j.get("horizon_RF", 1000)))


def trace_path(result, is_v3=False):
    if is_v3:
        p = str(result.get("sample_csv", "")).replace("\\", "/")
        return ROOT / p if p else None
    job_id = result.get("job_id", "")
    return OUT / "trajectories" / (hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16] + ".csv")


def v3_rows():
    rows=[]
    with (V3 / "trajectory_results.csv").open(newline="", encoding="utf-8") as f:
        for x in csv.DictReader(f):
            if not (x["field"] == "h35" and x["normalization"] == "equal_peak_voltage"
                    and float(x["voltage_V"]) == 250. and float(x["frequency_Hz"]) == 100000.
                    and float(x["mz_Th"]) == 100000. and x["velocity_kind"] == "tangent"
                    and int(float(x["horizon_RF"])) == 1000 and 1000. <= float(x["energy_meV"]) <= 3000.
                    and float(x["r0_mm"]) in (1., 2.) and x["status"] in ("retained", "escaped")):
                continue
            j={"model":x["model"],"field":x["field"],"normalization":x["normalization"],
               "voltage_V":float(x["voltage_V"]),"frequency_Hz":float(x["frequency_Hz"]),"mz_Th":float(x["mz_Th"]),
               "r0_mm":float(x["r0_mm"]),"energy_meV":float(x["energy_meV"]),"azimuth_rad":float(x["azimuth_rad"]),
               "phase_rad":float(x["phase_rad"]),"direction":float(x["direction"]),"velocity_kind":x["velocity_kind"],
               "horizon_RF":int(float(x["horizon_RF"])),"rtol":2e-9,"max_step_div":50,"sample_per_rf":8,"stage":"reused_v3"}
            key=basekey({"job":j})
            if any(basekey({"job":r["job"]}) == key and r["source"] == "V3 reused" for r in rows): continue
            init=run_v4.make_initial(j["mz_Th"],j["r0_mm"],j["energy_meV"],j["azimuth_rad"],j["phase_rad"],j["direction"],j["velocity_kind"])
            esc=fnum(x["RF_escape_periods"])
            source_trace=trace_path(x,True)
            archived=OUT/"reused_v3_traces"/Path(source_trace).name if source_trace else None
            if source_trace and Path(source_trace).exists():
                archived.parent.mkdir(parents=True,exist_ok=True)
                # V3 stores longer continuations beside the 1000-RF result
                # under the same path. Keep only samples through the exact
                # requested horizon so later radial diagnostics cannot leak
                # information from a longer integration.
                with Path(source_trace).open(newline="",encoding="utf-8") as fin, archived.open("w",newline="",encoding="utf-8") as fout:
                    reader=csv.DictReader(fin)
                    writer=csv.DictWriter(fout,fieldnames=reader.fieldnames)
                    writer.writeheader()
                    for sample in reader:
                        if fnum(sample.get("rf_period"),math.inf)<=j["horizon_RF"]+1e-9:
                            writer.writerow(sample)
            rows.append({"job_id":x["job_id"],"job":j,"initial_state":init.tolist(),"status":x["status"],
                "escape_periods":esc,"max_radius_mm":fnum(x["max_radius_mm"]),"rms_radius_mm":fnum(x["rms_radius_mm"]),
                "radial_min_mm":fnum(x["radial_min_mm"]),"nfev":int(float(x["nfev"])),"wall_s":fnum(x["wall_s"]),
                "sample_csv":str(archived if archived and archived.exists() else source_trace),"source":"V3 reused","solver":"V3 DOP853 standard"})
    return rows


def v4_rows():
    path=OUT/"trajectory_checkpoint.jsonl"
    rows=[]
    if not path.exists(): return rows
    seen=set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try: x=json.loads(line)
        except json.JSONDecodeError: continue
        if x.get("status") not in ("retained","escaped"): continue
        j=x.get("job",{})
        key=(x.get("job_id"),x.get("status"))
        if key in seen: continue
        seen.add(key)
        rows.append({"job_id":x["job_id"],"job":j,"initial_state":x.get("initial_state"),"status":x["status"],
            "escape_periods":x.get("escape_periods"),"max_radius_mm":fnum(x.get("max_radius_m"),0)*1000,
            "rms_radius_mm":fnum(x.get("rms_radius_m"),0)*1000,"radial_min_mm":fnum(x.get("radial_min_m"),0)*1000,
            "nfev":x.get("nfev"),"wall_s":x.get("wall_s"),"sample_csv":str(trace_path(x)),"source":"V4 calculated",
            "solver":x.get("solver","DOP853 standard")})
    return rows


def radial_diagnostics(path, frequency_hz):
    if not path or not Path(path).exists():
        return {"radial_oscillation_period_RF":None,"radial_oscillation_maxima":None,
                "radial_envelope_min_mm":None,"radial_envelope_max_mm":None}
    with Path(path).open(newline="",encoding="utf-8") as f:
        data=list(csv.DictReader(f))
    if len(data)<10:
        return {"radial_oscillation_period_RF":None,"radial_oscillation_maxima":0,
                "radial_envelope_min_mm":None,"radial_envelope_max_mm":None}
    # Average within RF periods to suppress micromotion without smoothing the
    # slow radial envelope; V4 traces are already stroboscopic at 1 sample/RF.
    buckets=defaultdict(list)
    for x in data:
        k=round(float(x["rf_period"]))
        buckets[k].append(float(x["r_mm"]))
    ix=np.asarray(sorted(buckets),float)
    rr=np.asarray([np.mean(buckets[k]) for k in sorted(buckets)],float)
    peaks,_=find_peaks(rr,prominence=.1,distance=5)
    period=float(np.mean(np.diff(ix[peaks]))) if len(peaks)>1 else None
    return {"radial_oscillation_period_RF":period,"radial_oscillation_maxima":int(len(peaks)),
            "radial_envelope_min_mm":float(np.min(rr)),"radial_envelope_max_mm":float(np.max(rr)),
            "max_radius_mm_sampled":float(np.max([float(x["r_mm"]) for x in data])),
            "rms_radius_mm_sampled":float(np.sqrt(np.mean([float(x["r_mm"])**2 for x in data]))),
            "samples_per_RF_observed":float(np.median([len(buckets[k]) for k in buckets]))}


def collect():
    rows=v3_rows()+v4_rows()
    enriched=[]
    for r in rows:
        j=r["job"]
        if not j: continue
        r={**r,"model":j["model"],"field":j.get("field","h35"),"normalization":j.get("normalization","equal_peak_voltage"),
           "voltage_V":float(j.get("voltage_V",250.)),"frequency_Hz":float(j.get("frequency_Hz",100000.)),
           "mz_Th":float(j.get("mz_Th",100000.)),"r0_mm":float(j["r0_mm"]),"energy_eV":float(j["energy_meV"])/1000.,
           "energy_meV":float(j["energy_meV"]),"azimuth_rad":float(j["azimuth_rad"]),"phase_rad":float(j["phase_rad"]),
           "direction":float(j["direction"]),"velocity_kind":j.get("velocity_kind","tangent"),"horizon_RF":int(j["horizon_RF"]),
           "sample_per_RF_requested":int(j.get("sample_per_rf",1)),
           "escape_time_RF":r.get("escape_periods") if r["status"]=="escaped" else None,
           "escape_time_s":(r.get("escape_periods")/float(j.get("frequency_Hz",100000.))) if r["status"]=="escaped" else None,
           "right_censored_at_RF":int(j["horizon_RF"]) if r["status"]=="retained" else None,
           "retained_1000RF":r["status"]=="retained","sample_count":None,
           "numerical_convergence_status":"tight rerun" if float(j.get("rtol",2e-9))<=2e-11 else "standard; no individual tight rerun"}
        state=r.get("initial_state") or run_v4.make_initial(r["mz_Th"],r["r0_mm"],r["energy_meV"],r["azimuth_rad"],r["phase_rad"],r["direction"],r["velocity_kind"])
        r.update({"x0_m":state[0],"y0_m":state[1],"vx0_m_s":state[2],"vy0_m_s":state[3]})
        r.update(radial_diagnostics(r.get("sample_csv"),r["frequency_Hz"]))
        if r.get("sample_csv") and Path(r["sample_csv"]).exists():
            with Path(r["sample_csv"]).open(newline="",encoding="utf-8") as f:r["sample_count"]=sum(1 for _ in f)-1
        enriched.append(r)
    # Rotating stays at 250 V in the alternate normalization, so its exact
    # physical integration is the equal-peak rotating run. Represent that
    # successful trajectory under the separate protocol as a provenance-marked
    # alias instead of integrating the identical ODE a second time.
    matched_standing=[r for r in enriched if r["normalization"]=="matched_leading_pseudopotential_at_0p5mm"
        and r["model"]=="standing" and r["field"]=="h35" and r["voltage_V"]==250.*math.sqrt(2)]
    aliases=[]
    for stand in matched_standing:
        rot=next((r for r in enriched if r["normalization"]=="equal_peak_voltage" and r["model"]=="rotating"
            and r["field"]=="h35" and r["voltage_V"]==250. and r["frequency_Hz"]==stand["frequency_Hz"]
            and all(r[k]==stand[k] for k in ("r0_mm","energy_meV","azimuth_rad","phase_rad","direction","velocity_kind","horizon_RF"))),None)
        if rot is None:continue
        alias=dict(rot);alias["job"]=dict(rot["job"],normalization="matched_leading_pseudopotential_at_0p5mm",stage="matched_strength_reused_rotating")
        alias["job_id"]="reused-matched-strength|"+rot["job_id"];alias["normalization"]="matched_leading_pseudopotential_at_0p5mm"
        alias["source"]="reused equal-peak rotating integration for matched-strength protocol"
        aliases.append(alias)
    enriched.extend(aliases)
    # Mark standard records whose exact initial condition and protocol were
    # independently repeated at tighter DOP853 settings.
    controls=set()
    for r in enriched:
        if r["numerical_convergence_status"]!="tight rerun":continue
        j=r["job"]
        controls.add((j.get("normalization"),j.get("model"),j.get("field"),float(j["voltage_V"]),float(j["frequency_Hz"]),
            float(j["r0_mm"]),float(j["energy_meV"]),float(j["azimuth_rad"]),float(j["phase_rad"]),float(j["direction"]),j.get("velocity_kind","tangent")))
    for r in enriched:
        j=r["job"]
        key=(j.get("normalization"),j.get("model"),j.get("field"),float(j.get("voltage_V",250.)),float(j.get("frequency_Hz",100000.)),
            float(j["r0_mm"]),float(j["energy_meV"]),float(j["azimuth_rad"]),float(j["phase_rad"]),float(j["direction"]),j.get("velocity_kind","tangent"))
        if r["numerical_convergence_status"]!="tight rerun" and key in controls:r["numerical_convergence_status"]="same status checked at tight tolerance"
    write_csv(OUT/"trajectory_results_v4.csv",enriched)
    checks,normalization=make_validation_tables(enriched)
    write_csv(OUT/"numerical_checks.csv",checks);write_csv(OUT/"normalization_comparison.csv",normalization)
    check_summary={}
    for name in sorted({x["check"] for x in checks}):
        group=[x for x in checks if x["check"]==name]
        check_summary[name]={"n":len(group),"classification_agreements":sum(x["classification_agrees"] for x in group),
            "classification_disagreements":sum(not x["classification_agrees"] for x in group),
            "max_abs_escape_time_delta_RF":max((abs(x["escape_time_delta_RF"]) for x in group if x["escape_time_delta_RF"] is not None),default=None),
            "max_abs_radius_delta_um":max((abs(x["max_radius_delta_um"]) for x in group),default=None)}
    (OUT/"numerical_validation_summary.json").write_text(json.dumps(check_summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    # Only equal-peak h35, standard-tolerance records answer the main protocol.
    main=[r for r in enriched if r["normalization"]=="equal_peak_voltage" and r["field"]=="h35"
          and r["voltage_V"]==250. and r["frequency_Hz"]==100000. and r["horizon_RF"]==1000
          and (int(r.get("sample_per_RF_requested",1))==1 or r["source"]=="V3 reused")
          and r["numerical_convergence_status"]!="tight rerun"]
    # De-duplicate V3 rows that differ only in storage bookkeeping.
    by={}
    for r in main:
        k=(r["model"],r["r0_mm"],r["energy_meV"],r["azimuth_rad"],r["phase_rad"],r["direction"],r["velocity_kind"])
        if k not in by or r["source"]=="V4 calculated":by[k]=r
    main=list(by.values())
    write_csv(OUT/"equal_peak_main_grid.csv",main)
    main_pairs=paired(main)
    matched_rows=[r for r in enriched if r["normalization"]=="matched_leading_pseudopotential_at_0p5mm"
        and r["field"]=="h35" and r["horizon_RF"]==1000 and r["numerical_convergence_status"]!="tight rerun"]
    all_pairs=main_pairs+paired(matched_rows)
    match_summary={"protocol":"matched_leading_pseudopotential_at_0p5mm","rotating_voltage_V":250.,"standing_voltage_V":250.*math.sqrt(2),
        "paired_initial_conditions":len(all_pairs)-len(main_pairs),"difference_classes":{},
        "note":"separate normalization; matching leading term does not make complete multipole pseudopotentials identical"}
    for p in all_pairs[len(main_pairs):]:match_summary["difference_classes"][p["difference_class"]]=match_summary["difference_classes"].get(p["difference_class"],0)+1
    (OUT/"matched_strength_summary.json").write_text(json.dumps(match_summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    write_csv(OUT/"matched_pairs_equal_peak.csv",main_pairs)
    write_csv(OUT/"matched_pairs_all_protocols.csv",all_pairs)
    stats=summarize(enriched,main,main_pairs)
    (OUT/"analysis_summary_v4.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    return enriched,main,all_pairs,stats


def protocol_key(r,field=None,rtol=None):
    j=r["job"]
    return (j.get("normalization"),j["model"],j.get("field","h35"),float(j.get("voltage_V",250.)),float(j.get("frequency_Hz",100000.)),
        float(j["r0_mm"]),float(j["energy_meV"]),float(j["azimuth_rad"]),float(j["phase_rad"]),float(j["direction"]),
        j.get("velocity_kind","tangent"),int(j.get("horizon_RF",1000)))


def make_validation_tables(rows):
    by={}
    for r in rows:by[protocol_key(r)]=r
    checks=[]
    # Same start, voltage and frequency; compare only solver tolerance/step.
    for r in rows:
        j=r["job"]
        if j.get("field")!="h35" or float(j.get("rtol",2e-9))>2e-11:continue
        k=protocol_key(r);base=list(k);base[2]="h35"
        # The standard row has identical key except numerical settings.
        standard=next((x for x in rows if x is not r and protocol_key(x)==k and float(x["job"].get("rtol",2e-9))>1e-10),None)
        if standard is None:continue
        checks.append({"check":"DOP853 tolerance refinement","model":j["model"],"r0_mm":j["r0_mm"],"energy_eV":float(j["energy_meV"])/1000,
            "azimuth_rad":j["azimuth_rad"],"phase_rad":j["phase_rad"],"direction":j["direction"],"field":"h35",
            "standard_status":standard["status"],"refined_status":r["status"],"classification_agrees":standard["status"]==r["status"],
            "standard_escape_RF":standard["escape_periods"],"refined_escape_RF":r["escape_periods"],
            "escape_time_delta_RF":(r["escape_periods"]-standard["escape_periods"]) if r["escape_periods"] is not None and standard["escape_periods"] is not None else None,
            "max_radius_delta_um":(r["max_radius_mm"]-standard["max_radius_mm"])*1000})
    # Same model and launch under the two pre-existing N48 field reconstructions.
    for r in rows:
        j=r["job"]
        if j.get("field")!="h50":continue
        h35=next((x for x in rows if x is not r and x["job"].get("field")=="h35" and x["job"]["model"]==j["model"]
            and x["job"].get("normalization")==j.get("normalization") and float(x["job"].get("voltage_V",250.))==float(j.get("voltage_V",250.))
            and float(x["job"].get("frequency_Hz",100000.))==float(j.get("frequency_Hz",100000.))
            and all(float(x["job"][k])==float(j[k]) for k in ("r0_mm","energy_meV","azimuth_rad","phase_rad","direction"))
            and x["job"].get("velocity_kind","tangent")==j.get("velocity_kind","tangent")
            and int(x["job"].get("horizon_RF",1000))==int(j.get("horizon_RF",1000))
            and float(x["job"].get("rtol",2e-9))>1e-10),None)
        if h35 is None:continue
        checks.append({"check":"h35 versus h50 N48 field sensitivity","model":j["model"],"r0_mm":j["r0_mm"],"energy_eV":float(j["energy_meV"])/1000,
            "azimuth_rad":j["azimuth_rad"],"phase_rad":j["phase_rad"],"direction":j["direction"],"field":"h35/h50",
            "standard_status":h35["status"],"refined_status":r["status"],"classification_agrees":h35["status"]==r["status"],
            "standard_escape_RF":h35["escape_periods"],"refined_escape_RF":r["escape_periods"],
            "escape_time_delta_RF":(r["escape_periods"]-h35["escape_periods"]) if r["escape_periods"] is not None and h35["escape_periods"] is not None else None,
            "max_radius_delta_um":(r["max_radius_mm"]-h35["max_radius_mm"])*1000})
    # Compare saved radial extrema/RMS from stroboscopic sampling with 8 samples
    # per RF period on representative boundary launches.
    for r in rows:
        j=r["job"]
        if int(j.get("sample_per_rf",1))!=8:continue
        base=next((x for x in rows if x is not r and protocol_key(x)==protocol_key(r)
            and int(x["job"].get("sample_per_rf",1))==1),None)
        if base is None:continue
        checks.append({"check":"intra-RF output sampling sensitivity","model":j["model"],"r0_mm":j["r0_mm"],
            "energy_eV":float(j["energy_meV"])/1000,"azimuth_rad":j["azimuth_rad"],"phase_rad":j["phase_rad"],
            "direction":j["direction"],"field":j.get("field","h35"),"standard_status":base["status"],"refined_status":r["status"],
            "classification_agrees":base["status"]==r["status"],"standard_escape_RF":base["escape_periods"],"refined_escape_RF":r["escape_periods"],
            "escape_time_delta_RF":(r["escape_periods"]-base["escape_periods"]) if r["escape_periods"] is not None and base["escape_periods"] is not None else None,
            "max_radius_delta_um":(r["max_radius_mm"]-base["max_radius_mm"])*1000,
            "rms_radius_delta_um":(r["rms_radius_mm"]-base["rms_radius_mm"])*1000,
            "sampling_standard_per_RF":1,"sampling_refined_per_RF":8})
    # Report the alternate leading-term match independently from equal-peak.
    normalization=[]
    for r in rows:
        j=r["job"]
        if j.get("normalization")!="matched_leading_pseudopotential_at_0p5mm":continue
        base=next((x for x in rows if x is not r and x["job"].get("field")=="h35" and x["job"]["model"]==j["model"]
            and x["job"].get("normalization")=="equal_peak_voltage" and float(x["job"].get("voltage_V",250.))==250.
            and float(x["job"].get("frequency_Hz",100000.))==float(j.get("frequency_Hz",100000.))
            and all(float(x["job"][k])==float(j[k]) for k in ("r0_mm","energy_meV","azimuth_rad","phase_rad","direction"))
            and x["job"].get("velocity_kind","tangent")==j.get("velocity_kind","tangent")
            and int(x["job"].get("horizon_RF",1000))==int(j.get("horizon_RF",1000)) and float(x["job"].get("rtol",2e-9))>1e-10),None)
        if base is None:continue
        normalization.append({"comparison":"matched leading pseudopotential versus equal peak","model":j["model"],"r0_mm":j["r0_mm"],
            "energy_eV":float(j["energy_meV"])/1000,"azimuth_rad":j["azimuth_rad"],"phase_rad":j["phase_rad"],"direction":j["direction"],
            "equal_peak_voltage_V":base["voltage_V"],"matched_protocol_voltage_V":r["voltage_V"],"equal_peak_status":base["status"],
            "matched_protocol_status":r["status"],"equal_peak_max_radius_mm":base["max_radius_mm"],"matched_max_radius_mm":r["max_radius_mm"],
            "delta_max_radius_mm":r["max_radius_mm"]-base["max_radius_mm"]})
    return checks,normalization


def paired(rows):
    by=defaultdict(dict)
    for r in rows:
        k=(r["normalization"],r["field"],r["frequency_Hz"],r["r0_mm"],r["energy_meV"],
           r["azimuth_rad"],r["phase_rad"],r["direction"],r["velocity_kind"],r["horizon_RF"])
        by[k][r["model"]]=r
    out=[]
    for k,d in by.items():
        if not {"rotating","standing"}<=set(d):continue
        a,b=d["rotating"],d["standing"]
        out.append({"normalization":k[0],"field":k[1],"frequency_Hz":k[2],"r0_mm":k[3],"energy_eV":k[4]/1000,
            "azimuth_rad":k[5],"phase_rad":k[6],"direction":k[7],"velocity_kind":k[8],"horizon_RF":k[9],
            "rotating_voltage_V":a["voltage_V"],"standing_voltage_V":b["voltage_V"],
            "rotating_status":a["status"],"standing_status":b["status"],"difference_class":pair_class(a["status"],b["status"]),
            "rotating_escape_time_RF":a["escape_time_RF"],"standing_escape_time_RF":b["escape_time_RF"],
            "rotating_right_censored_at_RF":a["right_censored_at_RF"],"standing_right_censored_at_RF":b["right_censored_at_RF"],
            "rotating_max_radius_mm":a["max_radius_mm"],"standing_max_radius_mm":b["max_radius_mm"],
            "rotating_rms_radius_mm":a["rms_radius_mm"],"standing_rms_radius_mm":b["rms_radius_mm"],
            "delta_max_radius_mm":a["max_radius_mm"]-b["max_radius_mm"]})
    return out


def pair_class(rot,st):
    if rot=="retained" and st=="retained":return "both_retained"
    if rot=="retained" and st=="escaped":return "rotating_only_retained"
    if rot=="escaped" and st=="retained":return "standing_only_retained"
    return "both_escaped"


def summarize(allrows,main,pairs):
    eq=[r for r in allrows if r["normalization"]=="equal_peak_voltage" and r["field"]=="h35" and r["voltage_V"]==250.
        and r["frequency_Hz"]==100000. and r["horizon_RF"]==1000 and r["numerical_convergence_status"]!="tight rerun"]
    counts=defaultdict(int)
    for p in pairs:counts[p["difference_class"]]+=1
    shells=defaultdict(lambda:{"rotating_retained":0,"standing_retained":0,"n_matched":0})
    for p in pairs:
        if p["normalization"]!="equal_peak_voltage" or p["field"]!="h35" or p["rotating_voltage_V"]!=250. or p["standing_voltage_V"]!=250.:continue
        s=shells[p["energy_eV"]];s["n_matched"]+=1
        s["rotating_retained"]+=int(p["rotating_status"]=="retained")
        s["standing_retained"]+=int(p["standing_status"]=="retained")
    return {"protocol":"equal_peak_voltage only for the main map; matched strength analyzed separately",
        "horizon_RF":1000,"equal_peak_trajectories_including_controls":len(eq),"main_unique_equal_peak_trajectories":len(main),
        "matched_equal_peak_initial_conditions":len(pairs),"difference_classes":dict(counts),
        "retention_shells_descriptive_counts_not_probabilities":{str(k):v for k,v in sorted(shells.items())},
        "status_definition":"retained through 1000 RF; escaped at first outward crossing of 8 mm",
        "retained_cases_censored":"yes, at 1000 RF; no escape time assigned",
        "new_rf_jobs_completed":sum(r["source"]=="V4 calculated" for r in allrows),
        "source_v3_rows_reused":sum(r["source"]=="V3 reused" for r in allrows),
        "no_fem_rebuild":True}


def write_csv(path,rows):
    if not rows: return
    cols=[]
    for r in rows:
        for k in r:
            if k not in cols:cols.append(k)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader()
        for r in rows:
            w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()})


def make_jobs(rows, round_num):
    # Analyze only standard, equal-peak, h35 records in the primary horizon.
    d=defaultdict(dict)
    for r in rows:
        j=r["job"]
        if (r["normalization"]!="equal_peak_voltage" or r["field"]!="h35" or r["voltage_V"]!=250.
            or r["frequency_Hz"]!=100000. or r["horizon_RF"]!=1000 or r["numerical_convergence_status"]=="tight rerun"
            or (int(j.get("sample_per_rf",1))!=1 and r["source"]!="V3 reused")):
            continue
        lk=(r["r0_mm"],r["azimuth_rad"],r["phase_rad"],r["direction"],r["velocity_kind"])
        d[lk].setdefault(r["model"],{})[r["energy_eV"]]=r
    candidates=set()
    # Midpoints are added only when an actual adjacent pair of calculated
    # energies changes status for either model. Paired model runs are then
    # added at that exact start so the difference remains interpretable.
    for lk,models in d.items():
        all_e=sorted(set().union(*(set(x) for x in models.values())))
        # Refine energy intervals only when the *model-to-model orbital
        # difference itself changes materially across the interval*. A
        # persistent difference at both endpoints is already represented
        # by those measured shells and must not trigger a broad sweep.
        if "rotating" in models and "standing" in models:
            for ea,eb in zip(all_e,all_e[1:]):
                ra=models["rotating"].get(ea);sa=models["standing"].get(ea)
                rb=models["rotating"].get(eb);sb=models["standing"].get(eb)
                if not (ra and sa and rb and sb) or eb-ea<=.125:continue
                max_delta_a=abs(ra["max_radius_mm"]-sa["max_radius_mm"])
                max_delta_b=abs(rb["max_radius_mm"]-sb["max_radius_mm"])
                rms_delta_a=abs(ra["rms_radius_mm"]-sa["rms_radius_mm"])
                rms_delta_b=abs(rb["rms_radius_mm"]-sb["rms_radius_mm"])
                pa=(ra.get("radial_oscillation_period_RF"),sa.get("radial_oscillation_period_RF"))
                pb=(rb.get("radial_oscillation_period_RF"),sb.get("radial_oscillation_period_RF"))
                period_shift=False
                for xa,xb in zip(pa,pb):
                    if xa is not None and xb is not None and abs(xa-xb)/max(xa,xb)>.25:
                        period_shift=True
                ea_times=(ra.get("escape_time_RF"),sa.get("escape_time_RF"))
                eb_times=(rb.get("escape_time_RF"),sb.get("escape_time_RF"))
                escape_shift=any(x is not None and y is not None and
                    abs(x-y)>max(2.,.1*min(x,y)) for x,y in zip(ea_times,eb_times))
                changing_envelope=(abs(max_delta_a-max_delta_b)>=.75 or
                    abs(rms_delta_a-rms_delta_b)>=.5)
                near_aperture=any(x["max_radius_mm"]>=7.2 for x in (ra,sa,rb,sb))
                if (changing_envelope or period_shift or (escape_shift and near_aperture)):
                    candidates.add((lk,(ea+eb)/2,"orbital_dynamics_midpoint"))
        for model,values in models.items():
            es=sorted(values)
            for a,b in zip(es,es[1:]):
                if values[a]["status"]!=values[b]["status"] and b-a>0.125:
                    candidates.add((lk,(a+b)/2,"boundary_midpoint"))
        # Sparse exploration below/above the requested 1--3 eV focus is
        # restricted to launch conditions already showing model disagreement
        # or an envelope within 0.8 mm of the aperture.
        discordant_at={e:("rotating" in models and "standing" in models and e in models["rotating"] and e in models["standing"]
                         and models["rotating"][e]["status"]!=models["standing"][e]["status"]) for e in all_e}
        near_at_3=any(3.0 in values and values[3.0]["max_radius_mm"]>=7.2 for values in models.values())
        if discordant_at.get(1.0,False) and 0.5 not in all_e:
            candidates.add((lk,.5,"limited_low_probe"))
        if (discordant_at.get(3.0,False) or near_at_3) and 3.0 in all_e:
            candidates.add((lk,3.5,"limited_high_probe"))
        discordant_at_35=("rotating" in models and "standing" in models and 3.5 in models["rotating"] and 3.5 in models["standing"]
            and models["rotating"][3.5]["status"]!=models["standing"][3.5]["status"])
        near_at_35=any(3.5 in values and values[3.5]["max_radius_mm"]>=7.5 for values in models.values())
        if (discordant_at_35 or near_at_35) and 3.5 in all_e and 4.0 not in all_e:
            candidates.add((lk,4.0,"limited_outer_boundary_probe"))
    jobs=[];existing=set()
    for lk,energy,stage in candidates:
        r,az,phase,direction,kind=lk
        for model in ("rotating","standing"):
            if any(float(x["r0_mm"])==r and float(x["azimuth_rad"])==az and float(x["phase_rad"])==phase
                and float(x["direction"])==direction and x["velocity_kind"]==kind and float(x["energy_eV"])==energy
                and x["model"]==model and x["field"]=="h35" and x["normalization"]=="equal_peak_voltage"
                and x["numerical_convergence_status"]!="tight rerun" for x in rows): continue
            sig=(r,energy,az,phase,direction,kind,model)
            if sig in existing:continue
            existing.add(sig)
            jobs.append(dict(model=model,field="h35",voltage_V=250.,frequency_Hz=100000.,mz_Th=100000.,r0_mm=r,
                energy_meV=energy*1000.,azimuth_rad=az,phase_rad=phase,direction=direction,velocity_kind=kind,
                horizon_RF=1000,normalization="equal_peak_voltage",stage=f"{stage}_round{round_num}",refinement_depth=round_num))
    return jobs


def followup_jobs(rows,pairs):
    core=[p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["field"]=="h35"
          and p["rotating_status"]!=p["standing_status"]]
    core.sort(key=lambda p:(p["energy_eV"],p["r0_mm"],p["azimuth_rad"],p["phase_rad"],p["direction"]))
    selected=[]
    for radius,energy in ((1.,1.),(1.,2.),(2.,2.)):
        choices=[p for p in core if p["r0_mm"]==radius and p["energy_eV"]==energy]
        if choices:selected.append(choices[0])
    # Reserve one validation slot for the highest-energy discordant start
    # actually measured. This targets the outer confinement edge when the
    # limited 3.5/4.0 eV probes find one, while retaining a 3 eV fallback.
    outer=[p for p in core if p["energy_eV"]>=3.5]
    if outer:
        outer.sort(key=lambda p:(-p["energy_eV"],p["r0_mm"],p["azimuth_rad"],p["phase_rad"],p["direction"]))
        selected.append(outer[0])
    else:
        choices=[p for p in core if p["r0_mm"]==2. and p["energy_eV"]==3.]
        if choices:selected.append(choices[0])
    if len(selected)<4:
        for p in core:
            sig=(p["r0_mm"],p["energy_eV"],p["azimuth_rad"],p["phase_rad"],p["direction"])
            if all((x["r0_mm"],x["energy_eV"],x["azimuth_rad"],x["phase_rad"],x["direction"])!=sig for x in selected):selected.append(p)
            if len(selected)>=4:break
    selected=selected[:4]
    jobs=[]
    for p in selected:
        for model in ("rotating","standing"):
            common=dict(model=model,field="h35",voltage_V=250.,frequency_Hz=100000.,mz_Th=100000.,r0_mm=p["r0_mm"],
                energy_meV=p["energy_eV"]*1000.,azimuth_rad=p["azimuth_rad"],phase_rad=p["phase_rad"],direction=p["direction"],
                velocity_kind=p["velocity_kind"],horizon_RF=1000,normalization="equal_peak_voltage",stage="tight_boundary_validation",
                selected_case_id=f"r{p['r0_mm']:g}_E{p['energy_eV']:g}_{model}")
            tight={**common,"rtol":2e-11,"max_step_div":100}
            jobs.append(tight)
            h50={**common,"field":"h50","stage":"h50_boundary_sensitivity","rtol":2e-9,"max_step_div":50}
            jobs.append(h50)
            sampled={**common,"sample_per_rf":8,"stage":"intra_rf_sampling_sensitivity"}
            jobs.append(sampled)
            matched={**common,"voltage_V":250. if model=="rotating" else 250.*math.sqrt(2),
                "normalization":"matched_leading_pseudopotential_at_0p5mm","stage":"matched_strength_boundary"}
            jobs.append(matched)
    return jobs,selected


def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("collect","refine","followup"));ap.add_argument("--round",type=int,default=1);a=ap.parse_args()
    allrows,mainrows,pairs,stats=collect()
    if a.command=="refine":
        jobs=make_jobs(allrows,a.round)
        p=OUT/f"adaptive_round{a.round}_jobs.json"
        p.write_text(json.dumps(jobs,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps({"round":a.round,"candidate_jobs":len(jobs),"starts":len(jobs)//2,"file":p.name},indent=2))
    elif a.command=="followup":
        jobs,selected=followup_jobs(allrows,pairs)
        p=OUT/"numerical_field_strength_validation_jobs.json"
        p.write_text(json.dumps(jobs,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps({"selected_boundary_starts":len(selected),"jobs":len(jobs),"file":p.name,
            "starts":[{"r0_mm":x["r0_mm"],"energy_eV":x["energy_eV"],"azimuth_rad":x["azimuth_rad"],"phase_rad":x["phase_rad"],"direction":x["direction"]} for x in selected]},indent=2))
    else:
        print(json.dumps(stats,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
