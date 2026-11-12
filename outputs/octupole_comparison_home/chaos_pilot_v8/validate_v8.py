"""Fast reproducibility and data-integrity checks; performs no integrations."""
import csv,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;PRO=HERE/"processed"
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
checks={}
mp=rows(PRO/"azimuth_phase_map.csv");checks["map_has_rows"]=len(mp)>0
checks["map_only_observed_outcomes"]=all(r["outcome"] in ("retained","escaped") for r in mp)
checks["retained_censored_not_escape"]=all((r["escape_time_RF"]=="" and float(r["right_censored_at_RF"])==1000) for r in mp if r["outcome"]=="retained")
checks["escaped_has_first_passage"]=all(float(r["escape_time_RF"])>0 and r["right_censored_at_RF"]=="" for r in mp if r["outcome"]=="escaped")
checks["two_tangential_signs"]=set(int(r["vtheta_sign"]) for r in mp)=={-1,1}
checks["phases_with_actual_samples"]=sorted(set(float(r["rf_phase_deg"]) for r in mp))
checks["samples_have_solver_metadata"]=all(r["rtol"] and r["max_step_RF_fraction"] for r in mp)
sens=rows(PRO/"paired_sensitivity.csv");checks["sensitivity_rows_have_rad_metric"]=all(float(r["delta_metric_rad"])>0 for r in sens)
ft=rows(PRO/"ftle_summary.csv");checks["ftle_has_records"]=len(ft)>0
strict=json.loads((PRO/"strict_solver_comparison.json").read_text(encoding="utf-8"))
paired=[r for r in strict if r.get("same_initial_condition")]
checks["strict_solver_pairs_n"]=len(paired)
checks["all_exact_strict_pairs_agree"]=all(r["classification_agrees"] for r in paired)
h50=json.loads((PRO/"h35_h50_comparison.json").read_text(encoding="utf-8"))
checks["h35_h50_pairs_n"]=len(h50)
checks["h35_h50_disagreement_count"]=sum(not r["classification_agrees"] for r in h50)
for image in (HERE/"figures").glob("Fig*"):
    if image.suffix in (".pdf",".svg",".png"): checks[f"{image.suffix[1:]}_{image.stem}"]=image.stat().st_size>1000
for k,v in checks.items(): print(k,":",v)
fails=[k for k,v in checks.items() if v is False]
if fails: raise SystemExit("FAILED: "+", ".join(fails))
print("Validation checks passed")
