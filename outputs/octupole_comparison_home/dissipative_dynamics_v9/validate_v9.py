"""Fast V9 integrity, energy, censoring, and tangent-divergence checks."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
checks={};details={}

def rows_jsonl(path):
    if not path.exists():return []
    out=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        try:out.append(json.loads(line))
        except Exception:pass
    return out

rows=rows_jsonl(HERE/"trajectory_checkpoint.jsonl")
checks["trajectory_records_exist"]=bool(rows)
checks["checkpoint_job_ids_unique"]=len({r["job"]["job_id"] for r in rows})==len(rows)
checks["friction_energy_nonnegative"]=all(r.get("friction_dissipated_eV",0)>=-1e-12 for r in rows)
checks["friction_dissipation_flag"]=all(r.get("friction_dissipation_nonnegative",False) for r in rows)
checks["energy_balance_relative_below_1e-8"]=all(r.get("energy_balance_max_relative",math.inf)<1e-8 for r in rows)
checks["retained_cases_right_censored"]=all(r.get("escape_time_RF") is None and r.get("right_censored_at_RF")==r["job"]["horizon_RF"] for r in rows if r["status"]=="retained_to_horizon")
checks["escaped_cases_have_first_passage"]=all(r.get("escape_time_RF") is not None and r.get("escape_time_RF")>0 and r.get("right_censored_at_RF") is None for r in rows if r["status"]=="escaped")

# Isolated drag-only analytic regression: v(t)=v0 exp(-gamma t), K(t)-K0=-D(t).
M=100000*1.66053906660e-27; gamma=1000.; v0=62.12417634275831; duration=.01
sol=solve_ivp(lambda t,y:[y[1],-gamma*y[1],M*gamma*y[1]**2],(0,duration),[0.,v0,0.],rtol=1e-12,atol=1e-14)
v_exact=v0*math.exp(-gamma*duration);d_exact=.5*M*v0*v0*(1-math.exp(-2*gamma*duration))
checks["drag_only_velocity_matches_exponential"]=abs(sol.y[1,-1]-v_exact)/v_exact<1e-9
checks["drag_only_dissipation_nonnegative"]=sol.y[2,-1]>=0
checks["drag_only_mechanical_energy_balance"]=abs(.5*M*sol.y[1,-1]**2-.5*M*v0*v0+sol.y[2,-1])<1e-30
details["drag_only"]={"v_numeric":float(sol.y[1,-1]),"v_exact":v_exact,"dissipated_J":float(sol.y[2,-1]),"expected_J":d_exact}

vcheck=HERE/"h35_validation"/"summary.json"
if vcheck.exists():
    h=json.loads(vcheck.read_text(encoding="utf-8"));pairs=[x for x in h if x.get("classification_agrees") is not None]
    checks["h35_gamma0_v8_classifications_agree"]=bool(pairs) and all(x["classification_agrees"] for x in pairs)
    diffs=[abs(x["escape_time_difference_RF"]) for x in pairs if x.get("escape_time_difference_RF") is not None]
    details["h35_gamma0_pairs"]={"n":len(pairs),"max_escape_time_delta_RF":max(diffs) if diffs else None}
else: checks["h35_gamma0_v8_classifications_agree"]=False

lyap=[]
for p in (HERE/"lyapunov").glob("*.json") if (HERE/"lyapunov").exists() else []:
    try:lyap.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception:pass
if lyap:
    checks["full_spectrum_sum_matches_minus_2gamma"]=all(abs(r.get("sum_error_s_inv",math.inf))<max(1e-4,2e-7*max(1,2*r.get("job",{}).get("gamma_s_inv",0))) for r in lyap if r.get("status")!="escaped_in_transient")
    details["lyapunov"]=[{"model":x.get("job",{}).get("model"),"gamma_s_inv":x.get("job",{}).get("gamma_s_inv"),"spectrum":x.get("lyapunov_spectrum_s_inv"),"sum_error_s_inv":x.get("sum_error_s_inv")} for x in lyap]
else:
    checks["full_spectrum_sum_matches_minus_2gamma"]="not_run"

checks["csv_export_exists"]=(HERE/"processed/trajectory_results_v9.csv").exists()
checks["figures_exports_complete"]=all(any((HERE/"figures").glob(f"V9_Fig{i}_*{ext}")) for i in range(1,4) for ext in (".pdf",".svg",".png")) if (HERE/"figures").exists() else "not_run"
expected=len(__import__("run_v9").jobs())
numeric_checks_pass=all((isinstance(v,str) and v=="not_run") or bool(v) for v in checks.values())
payload={"checks":checks,"details":details,"completed_trajectory_count":len(rows),"expected_primary_job_count":expected,
         "coverage_complete":len(rows)==expected,"status":"PASS" if numeric_checks_pass else "FAIL"}
(HERE/"numerical_checks.json").write_text(json.dumps(payload,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)))
if payload["status"]!="PASS":raise SystemExit(1)
