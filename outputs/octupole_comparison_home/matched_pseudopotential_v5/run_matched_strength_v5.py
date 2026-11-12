"""Balanced matched-leading-pseudopotential trajectory study (V5, Stage 2).

The rotating 250-V arm is reused from V4 after exact physical/protocol checks.
Only standing 353.5533905932738-V trajectories are newly integrated. Every
completed or paused trajectory is checkpointed; no FEM solve is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V4 = ROOT / "outputs/octupole_comparison_home/confinement_maps_v4"
OUT = HERE
sys.path.insert(0, str(V4))
import run_v4  # noqa: E402  (reuse the verified DOP853 implementation)

ION_MZ_TH = 100_000.0
FREQUENCY_HZ = 100_000.0
ROTATING_VOLTAGE_V = 250.0
STANDING_VOLTAGE_V = 353.5533905932738
HORIZON_RF = 1_000
RTOL = 2e-9
MAX_STEP_DIV = 50
BUDGET_S = 4 * 60 * 60
WORKERS = 4
NORMALIZATION = "matched_leading_pseudopotential_at_0p5mm"
PHASES = (0.0, math.pi/2, math.pi, 3*math.pi/2)
AZIMUTHS = (0.0, math.pi/8, math.pi/4, 3*math.pi/8)
ENERGIES_EV = (1.0, 2.0, 3.0)
RADII_MM = (1.0, 2.0)
DIRECTIONS = (-1.0, 1.0)


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def portable_path(value: str) -> str:
    p = Path(value)
    if p.is_file():
        try:
            return p.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            pass
    normalized = value.replace("/", "\\")
    marker = "outputs\\octupole_comparison_home\\confinement_maps_v4\\"
    ix = normalized.lower().find(marker.lower())
    return normalized[ix:].replace("\\", "/") if ix >= 0 else value


def condition_id(row):
    e = int(round(float(row["energy_eV"])))
    r = int(round(float(row["r0_mm"])))
    phase = int(round(math.degrees(float(row["phase_rad"]))))
    az = int(round(math.degrees(float(row["azimuth_rad"]))))
    direction = "p" if float(row["direction"]) > 0 else "m"
    return f"E{e}_r{r}_phase{phase}_az{az}_dir{direction}"


def balanced_rows_and_audit():
    rows = read_csv(V4 / "equal_peak_main_grid.csv")
    selected = [r for r in rows if r["model"] == "rotating"
        and r["field"] == "h35" and r["normalization"] == "equal_peak_voltage"
        and float(r["voltage_V"]) == ROTATING_VOLTAGE_V
        and float(r["frequency_Hz"]) == FREQUENCY_HZ and float(r["mz_Th"]) == ION_MZ_TH
        and float(r["r0_mm"]) in RADII_MM and float(r["energy_eV"]) in ENERGIES_EV
        and float(r["phase_rad"]) in PHASES and float(r["azimuth_rad"]) in AZIMUTHS
        and float(r["direction"]) in DIRECTIONS and r["velocity_kind"] == "tangent"
        and int(float(r["horizon_RF"])) == HORIZON_RF]
    if len(selected) != 192 or len({condition_id(r) for r in selected}) != 192:
        raise ValueError(f"V4 rotating source has {len(selected)} rows; expected 192 unique balanced starts")

    cfg = json.loads((V4 / "study_config_v4.json").read_text(encoding="utf-8"))
    standard_rtol = float(cfg["solver"]["standard_rtol"])
    standard_step_div = round(1/float(cfg["solver"]["standard_max_step_RF_fraction"]))
    if standard_rtol != RTOL or standard_step_div != MAX_STEP_DIV:
        raise ValueError("V4 solver defaults differ from the matched-strength protocol")
    source_counts = Counter(r["source"] for r in selected)
    if source_counts != Counter({"V4 calculated": 168, "V3 reused": 24}):
        raise ValueError(f"Unexpected rotating provenance mix: {dict(source_counts)}")
    status_counts = Counter(r["status"] for r in selected)
    if status_counts != Counter({"retained": 192}):
        raise ValueError(f"Unexpected V4 rotating classifications: {dict(status_counts)}")

    audit_rows = []
    initial_rows = []
    rotating_reuse = []
    for r in selected:
        j = json.loads(r["job"])
        if (j["model"] != "rotating" or j["field"] != "h35"
            or float(j["voltage_V"]) != ROTATING_VOLTAGE_V
            or float(j["frequency_Hz"]) != FREQUENCY_HZ or float(j["mz_Th"]) != ION_MZ_TH
            or int(j["horizon_RF"]) != HORIZON_RF or float(j.get("rtol", standard_rtol)) != RTOL
            or int(j.get("max_step_div", standard_step_div)) != MAX_STEP_DIV):
            raise ValueError(f"Rotating source has a mismatched physical/numerical protocol: {r['job_id']}")
        initial = np.asarray(json.loads(r["initial_state"]), float)
        expected = run_v4.make_initial(ION_MZ_TH, float(r["r0_mm"]), float(r["energy_meV"]),
            float(r["azimuth_rad"]), float(r["phase_rad"]), float(r["direction"]), "tangent")
        if not np.allclose(initial, expected, rtol=0, atol=2e-12):
            raise ValueError(f"V4 physical initial state fails reconstruction: {r['job_id']}")
        x,y,vx,vy = map(float, initial)
        rmag = math.hypot(x,y)
        vtan = (-vx*y+vy*x)/rmag
        if float(r["direction"])*vtan <= 0:
            raise ValueError(f"V4 rotating source has wrong tangential sign: {r['job_id']}")
        if not Path(r["sample_csv"]).is_file():
            raise FileNotFoundError(r["sample_csv"])
        phase_deg=math.degrees(float(r["phase_rad"]))
        az_deg=math.degrees(float(r["azimuth_rad"]))
        cid=condition_id(r)
        audit_rows.append({"condition_id":cid,"source":r["source"],"job_id":r["job_id"],
            "status":r["status"],"rtol":float(j.get("rtol",standard_rtol)),
            "max_step_RF_fraction":1.0/int(j.get("max_step_div",standard_step_div)),
            "saved_samples_per_RF":int(j.get("sample_per_rf",1)),
            "rotating_voltage_V":ROTATING_VOLTAGE_V,"frequency_Hz":FREQUENCY_HZ,
            "field":"h35","horizon_RF":HORIZON_RF,"direction_sign_audit":"pass"})
        initial_rows.append({"condition_id":cid,"energy_eV":float(r["energy_eV"]),"r0_mm":float(r["r0_mm"]),
            "phase_rad":float(r["phase_rad"]),"phase_deg":phase_deg,"azimuth_rad":float(r["azimuth_rad"]),
            "azimuth_deg":az_deg,"direction":float(r["direction"]),"velocity_kind":"tangent",
            "x0_m":x,"y0_m":y,"vx0_m_s":vx,"vy0_m_s":vy,"rotating_source":r["source"],
            "rotating_job_id":r["job_id"]})
        rotating_reuse.append({"condition_id":cid,"job_id":r["job_id"],"source":r["source"],
            "solver":r["solver"],"normalization_original":r["normalization"],"field":r["field"],
            "voltage_V":r["voltage_V"],"frequency_Hz":r["frequency_Hz"],"mz_Th":r["mz_Th"],
            "r0_mm":r["r0_mm"],"energy_eV":r["energy_eV"],"phase_rad":r["phase_rad"],
            "azimuth_rad":r["azimuth_rad"],"direction":r["direction"],"status":r["status"],
            "escape_time_RF":r["escape_time_RF"],"right_censored_at_RF":r["right_censored_at_RF"],
            "max_radius_mm_source":r["max_radius_mm"],"rms_radius_mm_source":r["rms_radius_mm"],
            "sampled_trace":portable_path(r["sample_csv"])})

    jobs=[]
    for start in initial_rows:
        job={"model":"standing","field":"h35","normalization":NORMALIZATION,
            "voltage_V":STANDING_VOLTAGE_V,"frequency_Hz":FREQUENCY_HZ,"mz_Th":ION_MZ_TH,
            "r0_mm":start["r0_mm"],"energy_meV":1000*start["energy_eV"],
            "azimuth_rad":start["azimuth_rad"],"phase_rad":start["phase_rad"],
            "direction":start["direction"],"velocity_kind":"tangent","horizon_RF":HORIZON_RF,
            "rtol":RTOL,"max_step_div":MAX_STEP_DIV,"sample_per_rf":1,
            "stage":"v5_balanced_matched_pseudopotential"}
        job["condition_id"]=start["condition_id"]
        jobs.append(job)
    if len({run_v4.task_key(j) for j in jobs}) != 192:
        raise ValueError("Standing 353.553-V jobs do not form 192 unique physical/numerical requests")

    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"checkpoints").mkdir(exist_ok=True)
    (OUT/"trajectories").mkdir(exist_ok=True)
    (OUT/"processed").mkdir(exist_ok=True)
    write_csv(OUT/"balanced_initial_conditions.csv",initial_rows)
    write_csv(OUT/"rotating_v4_reuse_audit.csv",audit_rows)
    write_csv(OUT/"rotating_v4_reused_trajectories.csv",rotating_reuse)
    atomic_json(OUT/"jobs_standing_353553.json",jobs)
    config={"study":"V5 balanced matched-leading-pseudopotential comparison",
        "normalization":"separate matched-leading-pseudopotential protocol at r_ref=0.5 mm; full multipole pseudopotentials need not be identical",
        "field":"verified h35 N48 reconstruction","rotating_V":ROTATING_VOLTAGE_V,"standing_V":STANDING_VOLTAGE_V,
        "frequency_Hz":FREQUENCY_HZ,"ion_mz_Th":ION_MZ_TH,"charge_state":1,"aperture_radius_mm":8,
        "horizon_RF":HORIZON_RF,"energies_eV":list(ENERGIES_EV),"radii_mm":list(RADII_MM),
        "phases_deg":[0,90,180,270],"azimuths_deg":[0,22.5,45,67.5],"tangential_velocity_signs":[-1,1],
        "balanced_initial_conditions":192,"standing_new_integrations":192,
        "rotating_reuse":{"V4_calculated":168,"V3_reused_into_V4":24,"protocol":"DOP853 rtol=2e-9, max step T/50, full 250 V, same h35 field and horizon; only saved trace sampling differs"},
        "solver":{"method":"DOP853","rtol":RTOL,"max_step_RF_fraction":1/MAX_STEP_DIV,"saved_samples_per_RF":1,
                  "aperture_event":"terminal outward crossing of r=8 mm"},
        "resources":{"max_new_numerical_wall_s":BUDGET_S,"workers":WORKERS,"checkpoint":"append-only fsync journal plus atomic 300-RF segments"},
        "direction_audit":"computed initial tangent projection agrees with recorded direction for all 192 reused states",
        "dataset_interpretation":"descriptive deterministic grid; no population probabilities or infinite-time stability claims"}
    atomic_json(OUT/"study_config_v5.json",config)
    atomic_json(OUT/"source_reuse_audit.json",{"source_counts":dict(source_counts),"balanced_rows":len(selected),
        "status_counts":dict(status_counts),"all_initial_state_reconstructions_pass":True,
        "all_tangential_sign_checks_pass":True,"all_source_traces_exist":True,
        "solver_tolerances_match":True,"saved_sampling_rates":dict(Counter(r["saved_samples_per_RF"] for r in audit_rows)),
        "source_files":{"V4_main_grid":(V4/"equal_peak_main_grid.csv").relative_to(ROOT).as_posix(),
                        "V4_study_config":(V4/"study_config_v4.json").relative_to(ROOT).as_posix()}})
    return jobs


def _init_worker(output_path: str):
    os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    run_v4.OUT=Path(output_path)


def _read_journal():
    out={}; p=OUT/"trajectory_checkpoint.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                row=json.loads(line); out[row["job_id"]]=row
            except Exception:
                continue
    return out


def run_pending(workers=WORKERS):
    if not 1 <= workers <= 4:
        raise ValueError("This study is configured for no more than four CPU workers")
    jobs=json.loads((OUT/"jobs_standing_353553.json").read_text(encoding="utf-8"))
    journal=_read_journal()
    todo=[j for j in jobs if journal.get(run_v4.task_key(j),{}).get("status") not in ("retained","escaped")]
    deadline_path=OUT/"integration_deadline.json"
    global_path=OUT/"runtime/global_deadline.json"
    if not deadline_path.exists():
        start=time.time(); deadline=start+BUDGET_S
        budget={"started_unix_s":start,"deadline_unix_s":deadline,"budget_s":BUDGET_S,
            "started_utc":datetime.fromtimestamp(start,timezone.utc).isoformat(),
            "deadline_utc":datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
            "scope":"all V5 new RF integrations and numerical field diagnostics",
            "first_integrations":"standing 353.5533905932738 V balanced grid"}
        atomic_json(global_path,budget)
        atomic_json(deadline_path,{"start_unix_s":start,"deadline_unix_s":deadline,"budget_s":BUDGET_S,
                                   "started_utc":budget["started_utc"]})
    else:
        budget=json.loads(global_path.read_text(encoding="utf-8"))
        deadline=float(budget["deadline_unix_s"])
    if not todo:
        return {"requested":len(jobs),"pending":0,"message":"All standing trajectories already complete"}
    if time.time()+300 >= deadline:
        return {"requested":len(jobs),"pending":len(todo),"stopped":"five-minute deadline guard"}
    OUT.mkdir(parents=True,exist_ok=True)
    journal_path=OUT/"trajectory_checkpoint.jsonl"
    started=time.perf_counter(); completed_this_run=0; failures=[]; next_i=0
    run_v4.OUT=OUT
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,initargs=(str(OUT),)) as pool, journal_path.open("a",encoding="utf-8") as f:
        active={}
        while active or next_i<len(todo):
            while next_i<len(todo) and len(active)<workers:
                if time.time()+300 >= deadline:
                    break
                job=todo[next_i]; next_i+=1
                active[pool.submit(run_v4.worker,job)]=job
            if not active:
                break
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future)
                try:
                    row=future.result()
                except Exception as exc:
                    row={"job_id":run_v4.task_key(job),"job":job,"status":"numerical_failure",
                         "error":repr(exc),"traceback":traceback.format_exc()}
                    failures.append(row["job_id"])
                f.write(json.dumps(row,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
                completed_this_run+=1
            if time.time()+300 >= deadline:
                break
    latest=_read_journal()
    statuses=Counter(latest.get(run_v4.task_key(j),{}).get("status","missing") for j in jobs)
    result={"requested":len(jobs),"journal_rows_written_this_run":completed_this_run,
        "complete_retained_or_escaped":statuses.get("retained",0)+statuses.get("escaped",0),
        "paused":statuses.get("paused",0),"failures":failures,"remaining_deadline_s":max(0,deadline-time.time()),
        "wall_s":time.perf_counter()-started,"workers":workers,"statuses":dict(statuses)}
    log=OUT/"runtime/runtime_log.jsonl";log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a",encoding="utf-8") as f:f.write(json.dumps(result,allow_nan=False)+"\n")
    atomic_json(OUT/"runtime/runtime_status.json",result)
    return result


def trace_stroboscopic(path: Path, terminal_escape=False):
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    by_period={}
    for q in rows:
        t=float(q["rf_period"]); k=round(t)
        if abs(t-k)<2e-8:
            by_period[int(k)]=float(q["r_mm"])
    if not by_period:
        raise ValueError(f"No integer-period samples in {path}")
    ts=np.asarray(sorted(by_period),float)
    rr=np.asarray([by_period[int(t)] for t in ts],float)
    peaks,_=find_peaks(rr,prominence=.1,distance=5)
    period=float(np.mean(np.diff(ts[peaks]))) if len(peaks)>1 else None
    return {"sample_periods":ts,"sample_radius_mm":rr,
        "stroboscopic_rmin_mm":float(np.min(rr)),"stroboscopic_rmax_mm":float(np.max(rr)),
        "stroboscopic_rrms_mm":float(np.sqrt(np.mean(rr*rr))),
        "radial_oscillation_period_RF":period,"radial_oscillation_maxima":int(len(peaks)),
        "integer_period_samples":len(rr),"escape_endpoint_included_in_rmax":bool(terminal_escape)}


def finalise_primary():
    jobs=json.loads((OUT/"jobs_standing_353553.json").read_text(encoding="utf-8"))
    journal=_read_journal()
    incomplete=[j for j in jobs if journal.get(run_v4.task_key(j),{}).get("status") not in ("retained","escaped")]
    if incomplete:
        return {"complete":False,"pending":len(incomplete),"statuses":dict(Counter(journal.get(run_v4.task_key(j),{}).get("status","missing") for j in incomplete))}
    v4rows=read_csv(V4/"equal_peak_main_grid.csv")
    rotating={condition_id(r):r for r in v4rows if r["model"]=="rotating" and r["field"]=="h35"
        and r["normalization"]=="equal_peak_voltage" and float(r["voltage_V"])==ROTATING_VOLTAGE_V
        and float(r["frequency_Hz"])==FREQUENCY_HZ and float(r["mz_Th"])==ION_MZ_TH
        and float(r["r0_mm"]) in RADII_MM and float(r["energy_eV"]) in ENERGIES_EV
        and float(r["phase_rad"]) in PHASES and float(r["azimuth_rad"]) in AZIMUTHS
        and float(r["direction"]) in DIRECTIONS and int(float(r["horizon_RF"]))==HORIZON_RF}
    paired=[]
    for job in jobs:
        key=run_v4.task_key(job); stand=journal[key]; cid=job["condition_id"]; rot=rotating[cid]
        stand_trace=OUT/"trajectories"/(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]+".csv")
        if not stand_trace.exists(): raise FileNotFoundError(stand_trace)
        rot_trace=Path(rot["sample_csv"])
        rot_diag=trace_stroboscopic(rot_trace,rot["status"]=="escaped")
        stand_diag=trace_stroboscopic(stand_trace,stand["status"]=="escaped")
        diff="both_retained"
        if rot["status"]=="retained" and stand["status"]=="escaped":diff="rotating_only_retained"
        elif rot["status"]=="escaped" and stand["status"]=="retained":diff="standing_only_retained"
        elif rot["status"]=="escaped" and stand["status"]=="escaped":diff="both_escaped"
        t_r=float(rot["escape_time_RF"]) if rot["status"]=="escaped" and rot.get("escape_time_RF") else None
        t_s=float(stand["escape_periods"]) if stand["status"]=="escaped" and stand.get("escape_periods") is not None else None
        rmax_r=float(rot["max_radius_mm"])
        rmax_s=float(stand["max_radius_m"])*1000
        row={"condition_id":cid,"normalization":NORMALIZATION,"field":"h35","frequency_Hz":FREQUENCY_HZ,
            "mz_Th":ION_MZ_TH,"charge_state":1,"energy_eV":float(rot["energy_eV"]),"r0_mm":float(rot["r0_mm"]),
            "phase_rad":float(rot["phase_rad"]),"phase_deg":math.degrees(float(rot["phase_rad"])),
            "azimuth_rad":float(rot["azimuth_rad"]),"azimuth_deg":math.degrees(float(rot["azimuth_rad"])),
            "direction":float(rot["direction"]),"velocity_kind":"tangent","horizon_RF":HORIZON_RF,
            "rotating_voltage_V":ROTATING_VOLTAGE_V,"standing_voltage_V":STANDING_VOLTAGE_V,
            "rotating_source":rot["source"],"rotating_status":rot["status"],"standing_status":stand["status"],
            "difference_class":diff,"rotating_escape_time_RF":t_r,"standing_escape_time_RF":t_s,
            "rotating_right_censored_at_RF":HORIZON_RF if rot["status"]=="retained" else None,
            "standing_right_censored_at_RF":HORIZON_RF if stand["status"]=="retained" else None,
            "rotating_first_passage_s":t_r/FREQUENCY_HZ if t_r is not None else None,
            "standing_first_passage_s":t_s/FREQUENCY_HZ if t_s is not None else None,
            "rotating_max_radius_mm":rmax_r,"standing_max_radius_mm":rmax_s,
            "delta_max_radius_mm_rot_minus_stand":rmax_r-rmax_s,
            "rotating_stroboscopic_rms_mm":rot_diag["stroboscopic_rrms_mm"],
            "standing_stroboscopic_rms_mm":stand_diag["stroboscopic_rrms_mm"],
            "delta_stroboscopic_rms_mm_rot_minus_stand":rot_diag["stroboscopic_rrms_mm"]-stand_diag["stroboscopic_rrms_mm"],
            "rotating_stroboscopic_rmin_mm":rot_diag["stroboscopic_rmin_mm"],
            "rotating_stroboscopic_rmax_mm":rot_diag["stroboscopic_rmax_mm"],
            "standing_stroboscopic_rmin_mm":stand_diag["stroboscopic_rmin_mm"],
            "standing_stroboscopic_rmax_mm":stand_diag["stroboscopic_rmax_mm"],
            "rotating_radial_period_RF":rot_diag["radial_oscillation_period_RF"],
            "standing_radial_period_RF":stand_diag["radial_oscillation_period_RF"],
            "rotating_radial_peaks":rot_diag["radial_oscillation_maxima"],"standing_radial_peaks":stand_diag["radial_oscillation_maxima"],
            "rotating_solver_status":"V4 validated; source sampling harmonized at integer RF periods",
            "standing_solver_status":"DOP853 standard; rtol=2e-9; max step=T/50"}
        paired.append(row)
    paired.sort(key=lambda x:(x["energy_eV"],x["r0_mm"],x["phase_deg"],x["azimuth_deg"],x["direction"]))
    write_csv(OUT/"processed/matched_strength_balanced_pairs.csv",paired)
    counts=Counter(r["difference_class"] for r in paired)
    shell={}
    for energy in ENERGIES_EV:
        group=[r for r in paired if r["energy_eV"]==energy]
        shell[str(energy)]={"n":len(group),"rotating_retained":sum(r["rotating_status"]=="retained" for r in group),
            "standing_retained":sum(r["standing_status"]=="retained" for r in group),
            "rotating_only_retained":sum(r["difference_class"]=="rotating_only_retained" for r in group),
            "standing_only_retained":sum(r["difference_class"]=="standing_only_retained" for r in group),
            "both_retained":sum(r["difference_class"]=="both_retained" for r in group),
            "both_escaped":sum(r["difference_class"]=="both_escaped" for r in group)}
    phase_counts={}
    for phase in (0,90,180,270):
        g=[r for r in paired if round(r["phase_deg"])==phase]
        phase_counts[str(phase)]={"n":len(g),"rotating_only_retained":sum(r["difference_class"]=="rotating_only_retained" for r in g),
            "standing_only_retained":sum(r["difference_class"]=="standing_only_retained" for r in g),
            "both_retained":sum(r["difference_class"]=="both_retained" for r in g),"both_escaped":sum(r["difference_class"]=="both_escaped" for r in g)}
    direction_counts={}
    for d in DIRECTIONS:
        g=[r for r in paired if r["direction"]==d]
        direction_counts["positive" if d>0 else "negative"]={"n":len(g),**dict(Counter(r["difference_class"] for r in g))}
    summary={"protocol":NORMALIZATION,"paired_initial_conditions":len(paired),
        "rotating_voltage_V":ROTATING_VOLTAGE_V,"standing_voltage_V":STANDING_VOLTAGE_V,
        "frequency_Hz":FREQUENCY_HZ,"ion_mz_Th":ION_MZ_TH,"horizon_RF":HORIZON_RF,
        "difference_classes":dict(counts),"by_energy_eV":shell,"by_phase_deg":phase_counts,"by_tangent_sign":direction_counts,
        "right_censoring":"survivors are right-censored at 1000 RF; only outward aperture crossings have event times",
        "sampling":"stroboscopic radial RMS/envelopes use one value at each integer RF period for both arms; max radius retains the solver's event-inclusive maximum",
        "interpretation":"descriptive balanced deterministic design; no population probability or permanent stability claim",
        "numerical_protocol":{"field":"h35","method":"DOP853","rtol":RTOL,"max_step_RF_fraction":1/MAX_STEP_DIV,
            "rotating_reused_from":"V4 250 V, same h35 field and physical protocol","standing_new_integrations":len(paired)}}
    atomic_json(OUT/"processed/matched_strength_summary_v5.json",summary)
    atomic_json(OUT/"runtime/final_primary_status.json",{"complete":True,"pairs":len(paired),"classes":dict(counts),
        "finalized_utc":datetime.now(timezone.utc).isoformat()})
    return summary


def main():
    jobs=balanced_rows_and_audit()
    print(json.dumps({"prepared_standing_jobs":len(jobs),"reused_rotating":192,"new_integration_limit_s":BUDGET_S},indent=2))
    if "--prepare-only" in sys.argv:
        return
    run=run_pending(WORKERS)
    print(json.dumps(run,indent=2))
    if run.get("failures"):
        raise SystemExit("One or more integrations failed; inspect the checkpoint journal.")
    if run.get("complete_retained_or_escaped")==192:
        print(json.dumps(finalise_primary(),indent=2))


if __name__ == "__main__":
    main()
