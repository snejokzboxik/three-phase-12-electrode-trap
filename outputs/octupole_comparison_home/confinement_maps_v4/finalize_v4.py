"""Fill the Russian report and build a reproducibility/output audit."""
from __future__ import annotations
import csv, hashlib, json, math, re
from datetime import datetime, timezone
from pathlib import Path

from analyze_v4 import OUT, collect
import render_figures_v4


def read_json(path):return json.loads(path.read_text(encoding="utf-8"))
def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def fmt(x,d=3):return "—" if x is None else f"{float(x):.{d}g}"
def utc(unix):return datetime.fromtimestamp(unix,timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main():
    allrows,main,pairs,stats=collect()
    render_figures_v4.main()
    report=OUT/"ОТЧЁТ_V4_RU.md"
    text=report.read_text(encoding="utf-8")
    classes=stats["difference_classes"]
    n_pairs=stats["matched_equal_peak_initial_conditions"]
    rot_only=classes.get("rotating_only_retained",0);stand_only=classes.get("standing_only_retained",0)
    both_ret=classes.get("both_retained",0);both_escape=classes.get("both_escaped",0)
    rot_ret=rot_only+both_ret;stand_ret=stand_only+both_ret
    if rot_only and not stand_only:
        topline=f"На сетке равного пикового напряжения rotating удержал {rot_ret}/{n_pairs} стартов, standing — {stand_ret}/{n_pairs}; единственная асимметрия исходов — {rot_only} точек в пользу rotating."
        advantage_conclusion=f"Вращающееся поле демонстрирует конечновременное преимущество на проверенной сетке при равных 250 В: {rot_only} рассчитанных matched-стартов удержаны только rotating. Это локальный вывод о данной сетке, а не универсальное преимущество для произвольной фазы, энергии или прибора."
    elif stand_only and not rot_only:
        topline=f"На сетке равного пикового напряжения rotating удержал {rot_ret}/{n_pairs} стартов, standing — {stand_ret}/{n_pairs}; {stand_only} стартов удержаны только standing."
        advantage_conclusion=f"На проверенной сетке общего преимущества rotating по классификации удержания нет: {stand_only} точки благоприятствуют standing, а {rot_only} — rotating."
    elif rot_only or stand_only:
        topline=f"На сетке равного пикового напряжения есть обе асимметрии: {rot_only} старта удержаны только rotating и {stand_only} — только standing."
        advantage_conclusion=f"Различия зависят от старта: {rot_only} точки благоприятствуют rotating и {stand_only} — standing. Сетка демонстрирует фазо-энергетическую неоднородность, а не универсальное преимущество одной модели."
    else:
        topline=f"На сетке равного пикового напряжения статусы совпали для всех {n_pairs} matched стартов; классификационного преимущества rotating не обнаружено."
        advantage_conclusion="На рассчитанной сетке классификационного преимущества rotating не обнаружено; вывод ограничен охваченными стартами и горизонтом."
    reverse=(f"Обратная асимметрия также обнаружена: {stand_only} стартов удержаны только standing." if stand_only
        else "Обратных точек, где только standing удерживает, на данной выборке нет.")
    classnames={"both_retained":"обе удерживают","rotating_only_retained":"только rotating удерживает",
        "standing_only_retained":"только standing удерживает","both_escaped":"обе выходят"}
    class_counts="; ".join(f"**{classnames[k]}** — {classes.get(k,0)}" for k in classnames)
    shells=stats["retention_shells_descriptive_counts_not_probabilities"]
    shell_lines=[]
    for e,s in sorted(((float(k),v) for k,v in shells.items())):
        shell_lines.append(f"- **{e:g} эВ:** rotating {s['rotating_retained']}/{s['n_matched']}, standing {s['standing_retained']}/{s['n_matched']} ({s['n_matched']} matched-старта).")
    if not shell_lines:shell_lines=["- На момент фиксации результатов парные оболочки отсутствуют."]
    escapes=[p for p in pairs if p["normalization"]=="equal_peak_voltage" for m in ("rotating","standing") if p[f"{m}_escape_time_RF"] is not None]
    escape_range=(f"{min(x[f'{m}_escape_time_RF'] for x in pairs for m in ('rotating','standing') if x['normalization']=='equal_peak_voltage' and x[f'{m}_escape_time_RF'] is not None):.3f}–"
                  f"{max(x[f'{m}_escape_time_RF'] for x in pairs for m in ('rotating','standing') if x['normalization']=='equal_peak_voltage' and x[f'{m}_escape_time_RF'] is not None):.3f}") if escapes else "не рассчитан"
    checks=read_json(OUT/"numerical_validation_summary.json")
    tight=checks.get("DOP853 tolerance refinement",{});h50=checks.get("h35 versus h50 N48 field sensitivity",{})
    sampling=checks.get("intra-RF output sampling sensitivity",{})
    detail_checks=read_csv(OUT/"numerical_checks.csv") if (OUT/"numerical_checks.csv").exists() else []
    tight_starts={(x["r0_mm"],x["energy_eV"],x["azimuth_rad"],x["phase_rad"],x["direction"]) for x in detail_checks if x["check"]=="DOP853 tolerance refinement"}
    ncheck=len(tight_starts)
    sample_checks=[x for x in detail_checks if x["check"]=="intra-RF output sampling sensitivity"]
    matched_summary=read_json(OUT/"matched_strength_summary.json")
    matched_classes=matched_summary["difference_classes"]
    nmatched=matched_summary["paired_initial_conditions"]
    if nmatched:
        mparts="; ".join(f"{classnames[k]} — {matched_classes.get(k,0)}" for k in classnames)
        matched_result=f"matched-strength пары дали следующие классы: {mparts}. Эти строки не смешиваются с equal-peak сеткой."
    else:matched_result="matched-strength контрольные запуски отсутствуют; вывод по нему сделать нельзя."
    matched_table=read_csv(OUT/"normalization_comparison.csv") if (OUT/"normalization_comparison.csv").exists() else []
    deltas=[float(x["delta_max_radius_mm"]) for x in matched_table if x.get("delta_max_radius_mm") not in (None,"")]
    median_delta=(f" Медианная разность максимального радиуса (matched minus equal peak, по выбранным matched стартам) — {np_median(deltas):+.3f} мм." if deltas else "")
    v3grid=[r for r in allrows if r["source"]=="V3 reused" and r["energy_eV"] in (1.,2.,3.)]
    newbase=sum(r["source"]=="V4 calculated" and r["job"].get("stage")=="balanced_4x4x2_grid" and r["job"].get("field")=="h35" for r in allrows)
    newtotal=sum(r["source"]=="V4 calculated" for r in allrows)
    journal=OUT/"trajectory_checkpoint.jsonl";failures=[]
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row=json.loads(line)
                if row.get("status") not in ("retained","escaped"):failures.append(row)
            except json.JSONDecodeError:failures.append({"status":"invalid_json"})
    deadline=read_json(OUT/"integration_deadline.json");runs=[]
    for line in (OUT/"runtime_log.jsonl").read_text(encoding="utf-8").splitlines():
        try:runs.append(json.loads(line))
        except json.JSONDecodeError:pass
    end_unix=max([deadline["start_unix_s"]]+[datetime.strptime(x["completed_utc"],"%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() for x in runs])
    elapsed=max(0,end_unix-deadline["start_unix_s"])
    starttxt=deadline.get("started_utc",utc(deadline["start_unix_s"]))
    endtxt=utc(end_unix);deadlinetxt=utc(deadline["deadline_unix_s"])
    for token,value in {
        "{{TOPLINE}}":topline,"{{N_PAIRS}}":str(n_pairs),"{{ADVANTAGE}}":str(rot_only),"{{REVERSE}}":reverse,
        "{{N_MATCHED}}":str(nmatched),"{{MATCHED_RESULT}}":matched_result,"{{N_NEW_BASE}}":str(newbase),
        "{{N_CHECKS}}":str(ncheck),"{{CLASS_COUNTS}}":class_counts,"{{SHELL_COUNTS}}":"\n".join(shell_lines),
        "{{ESCAPE_RANGE}}":escape_range,"{{MATCHED_TEXT}}":matched_result+median_delta,
        "{{TIGHT_AGREE}}":str(tight.get("classification_agreements",0)),"{{TIGHT_N}}":str(tight.get("n",0)),
        "{{TIGHT_DT}}":fmt(tight.get("max_abs_escape_time_delta_RF")),"{{TIGHT_DR}}":fmt(tight.get("max_abs_radius_delta_um")),
        "{{H50_AGREE}}":str(h50.get("classification_agreements",0)),"{{H50_N}}":str(h50.get("n",0)),
        "{{H50_DR}}":fmt(h50.get("max_abs_radius_delta_um")),"{{N_SAMPLE_CHECK}}":str(len(sample_checks)),
        "{{SAMPLE_AGREE}}":str(sampling.get("classification_agreements",0)),
        "{{SAMPLE_DR}}":fmt(sampling.get("max_abs_radius_delta_um")),
        "{{SAMPLE_RMS}}":fmt(max((abs(float(x.get("rms_radius_delta_um",0) or 0)) for x in sample_checks),default=0)),
        "{{START_UTC}}":starttxt,
        "{{END_UTC}}":endtxt,"{{ELAPSED}}":f"{elapsed/60:.1f} min","{{DEADLINE_UTC}}":deadlinetxt,
        "{{N_NEW}}":str(newtotal),"{{N_FAILURES}}":str(len(failures)),"{{ADVANTAGE_CONCLUSION}}":advantage_conclusion
    }.items():text=text.replace(token,value)
    if re.search(r"\{\{[^}]+\}\}",text):raise RuntimeError("Unfilled report tokens remain")
    report.write_text(text,encoding="utf-8")
    config=read_json(OUT/"study_config_v4.json")
    round3_candidates=read_json(OUT/"adaptive_round3_jobs.json") if (OUT/"adaptive_round3_jobs.json").exists() else []
    round3_executed=read_json(OUT/"adaptive_round3_boundary_jobs.json") if (OUT/"adaptive_round3_boundary_jobs.json").exists() else []
    config["execution"]={"started_utc":starttxt,"hard_deadline_utc":deadlinetxt,"last_completed_stage_utc":endtxt,
        "wall_clock_minutes":round(elapsed/60,3),"new_completed_trajectories":newtotal,"new_nonterminal_or_failed_journal_rows":len(failures),
        "round3_candidate_jobs_generated_not_all_submitted":len(round3_candidates),"round3_boundary_jobs_submitted":len(round3_executed),
        "workers":4,"all_new_integrations_stopped_before_deadline":end_unix<deadline["deadline_unix_s"]}
    (OUT/"study_config_v4.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    runtime={"start_utc":starttxt,"deadline_utc":deadlinetxt,"end_utc":endtxt,"elapsed_minutes":round(elapsed/60,3),
        "max_new_integration_minutes":120,"workers":4,"runtime_stages":runs,"new_completed":newtotal,
        "new_grid_integrations":newbase,"v3_reused_rows":len(v3grid),"failed_or_paused_journal_rows":len(failures),
        "round3_candidate_jobs_generated_not_all_submitted":len(round3_candidates),"round3_boundary_jobs_submitted":len(round3_executed),
        "stopped_before_hard_deadline":end_unix<deadline["deadline_unix_s"]}
    (OUT/"runtime_summary_v4.json").write_text(json.dumps(runtime,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    # Final source/output digest inventory. Raw per-RF traces are listed so
    # their provenance remains auditable without hashing the entire repo.
    field_sources=[OUT.parents[1]/"abc_validation/direct_fem_validation/v2/h35_P4/diagnostic_N48.json",
                   OUT.parents[1]/"abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json"]
    v3_sources=[OUT.parents[0]/"rf_confinement_v1"/name for name in ("RF_CONFINEMENT_SEARCH_REPORT_RU.md","study_config_v3.json","run_v3.py","trajectory_results.csv","trajectory_checkpoint.jsonl")]
    audit={"field_sources":[],"v3_sources":[],"v3_trace_sources":[],"v4_artifacts":[],"matplotlib_available":False,"figure_backend":"Pillow + ReportLab",
        "reused_v3_rows":len(v3grid),"new_rf_trajectories":newtotal}
    for p in field_sources:
        if p.exists():audit["field_sources"].append({"path":str(p.relative_to(OUT.parents[2])),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    for p in v3_sources:
        if p.exists():audit["v3_sources"].append({"path":str(p.relative_to(OUT.parents[2])).replace("\\","/"),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    v3_trace_root=OUT.parents[0]/"rf_confinement_v1"/"trajectories"
    v3_trace_files=sorted({v3_trace_root/Path(r["sample_csv"]).name for r in allrows
        if r["source"]=="V3 reused" and r.get("sample_csv")})
    for p in v3_trace_files:
        if p.exists():audit["v3_trace_sources"].append({"path":str(p.relative_to(OUT.parents[2])).replace("\\","/"),
            "bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name=="input_output_audit_v4.json":continue
        if "__pycache__" in p.parts:continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        audit["v4_artifacts"].append({"path":str(p.relative_to(OUT)).replace("\\","/"),"bytes":p.stat().st_size,"sha256":digest})
    (OUT/"input_output_audit_v4.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(report),"equal_peak_pairs":n_pairs,"difference_classes":classes,"matched_strength_pairs":nmatched,
        "new_integrations":newtotal,"wall_minutes":round(elapsed/60,2),"failures":len(failures),"figures":len(list((OUT/"figures").glob("*.pdf")))},indent=2,ensure_ascii=False))


def np_median(values):
    values=sorted(values);n=len(values)
    if not n:return 0.
    return values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2


if __name__=="__main__":main()
