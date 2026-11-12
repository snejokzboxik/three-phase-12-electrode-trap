"""Two h50 N48 checks of the narrow h35 angular outcome window."""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
V7DIR=REPO/"outputs/octupole_comparison_home/harmonic_ablation_v7"
sys.path.insert(0,str(V7DIR)); import run_v7
OUT=HERE/"h50_validation"; OUT.mkdir(parents=True,exist_ok=True)

def main():
    run_v7.OUT=OUT
    run_v7._init_worker(run_v7.FIELD50)
    spec=[("h35_escape_window_lower_edge",10.2597,1),
          ("h35_retained_window_edge",10.26123046875,1),
          ("h35_negative_direction_retained_edge",9.609875,-1),
          ("h35_negative_direction_escape_edge",9.613,-1)]
    out=[]
    for label,theta,direction in spec:
        job={"stage":"V8_h50_sensitivity","launch_label":label,"representation":"full_N48","retained_harmonic_orders":None,"model":"standing","field_source":"h50_N48_polynomial","normalization":"matched_leading_m4_pseudopotential","voltage_V":run_v7.VOLTAGE["standing"],"frequency_Hz":run_v7.FREQUENCY_HZ,"mz_Th":run_v7.MASS_MZ_TH,"charge_state":1,"energy_eV":2.0,"r0_mm":1.0,"azimuth_deg":theta,"phase_deg":0.0,"direction":direction,"velocity_definition":"signed tangential; V6 convention","horizon_RF":1000,"aperture_mm":8.0,"rtol":run_v7.RTOL,"atol":[2e-13,2e-13,2e-11,2e-11],"max_step_RF_fraction":0.02,"samples_per_RF":2}
        job["job_id"]=hashlib.sha256(json.dumps(job,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:20]
        if (OUT/(job["job_id"]+".json")).exists():
            out.append(json.loads((OUT/(job["job_id"]+".json")).read_text(encoding="utf-8")))
            continue
        r=run_v7.integrate_job(job);r["validation_purpose"]="h50 resolution sensitivity of phase-zero h35 narrow outcome window"
        p=OUT/(r["job_id"]+".json");p.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8");out.append(r)
        print(label,r["status"],r["escape_time_RF"],"wall_s",round(r["wall_s"],2),flush=True)
    out=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json")) if p.name!="summary.json"]
    (OUT/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
if __name__=="__main__": main()
