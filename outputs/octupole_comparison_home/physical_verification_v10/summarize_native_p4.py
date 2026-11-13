"""Assemble matched V7-N48 and V10-native-P4 classification comparisons."""
from __future__ import annotations
import csv, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
V7=ROOT/"outputs/octupole_comparison_home/harmonic_ablation_v7/processed/ablation_trajectory_results.csv"
OUT=HERE/"processed"
MAP={"deep_az0_dplus":"deep_az0_dplus",
     "early_az22p5_dplus":"early_az22p5_dplus",
     "negative_retained":"boundary_dminus_retained",
     "negative_escaped":"boundary_dminus_escaped",
     "positive_retained":"boundary_dplus_retained",
     "positive_escaped":"boundary_dplus_escaped"}

def read(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p4=[]
    for path in sorted((HERE/"data"/"native_p4").glob("*.json")):
        if ".checkpoint." in path.name or path.name=="index.csv":continue
        item=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(item,dict) and "mesh_um" in item and "numerics" in item:p4.append(item)
    v7=read(V7)
    n48={}
    for row in v7:
        if row["representation"]!="full_N48" or row["phase_deg"]!="0.0":continue
        if row["stage"] not in ("pilot","boundary","late","resolution_h50"):continue
        mesh=35 if row["field_source"]=="h35_N48_polynomial" else 50 if row["field_source"]=="h50_N48_polynomial" else None
        if mesh is None:continue
        key=(mesh,row["launch_label"],row["model"])
        if key not in n48 or row["stage"] in ("pilot","boundary","resolution_h50"):
            n48[key]=row
    by={}
    for row in p4:
        if int(row["horizon_rf"])!=1000 or float(row["phase_deg"])!=0.0:continue
        if row.get("tag"):continue
        by[(int(row["mesh_um"]),row["launch_label"],row["model"])]=row
    rows=[]
    for (mesh,label,model),row in sorted(by.items()):
        source_label=MAP.get(label)
        if source_label is None:continue
        source=n48.get((mesh,source_label,model))
        rows.append({"launch_label":label,"azimuth_deg":row["azimuth_deg"],
                     "velocity_direction_sign":row["direction"],"rf_phase_deg":row["phase_deg"],
                     "model":model,"mesh_um":mesh,"initial_energy_eV":row["initial_energy_eV"],
                     "voltage_V":row["voltage_V"],"native_P4_status":row["status"],
                     "native_P4_escape_time_RF":row["escape_time_rf"],
                     "native_P4_horizon_RF":row["horizon_rf"],
                     "N48_status":None if source is None else source["status"],
                     "N48_escape_time_RF":None if source is None else source["escape_time_RF"],
                     "N48_field_source":None if source is None else source["field_source"],
                     "N48_rtol":None if source is None else source["rtol"],
                     "comparison_note":"finite-time status only; retained cases are right-censored"})
    write(OUT/"native_P4_vs_N48_matched_outcomes.csv",rows)
    summaries=[]
    for mesh in (35,50):
        for model in ("rotating","standing"):
            vals=[r for r in rows if int(r["mesh_um"])==mesh and r["model"]==model]
            summaries.append({"mesh_um":mesh,"model":model,"matched_starts":len(vals),
                              "retained_to_1000RF":sum(r["native_P4_status"]=="retained_to_horizon" for r in vals),
                              "escaped_by_1000RF":sum(r["native_P4_status"]=="escaped" for r in vals),
                              "N48_rows_matched":sum(r["N48_status"] is not None for r in vals),
                              "classification_disagreements":sum(r["N48_status"] is not None and r["native_P4_status"]!=("retained_to_horizon" if r["N48_status"]=="retained" else "escaped") for r in vals)})
    obj={"matched_records":len(rows),"comparison":"native P4 integration versus corresponding h35/h50 N48 full-field integration",
         "normalization":"rotating 250 V; standing 353.5533905932738 V; leading m=4 only",
         "finite_time_horizon_RF":1000,"new_FEM_solves":0,"new_N48_integrations":0,
         "summary":summaries,"note":"Different exit times near the boundary indicate numerical/field-representation sensitivity and are not permanent-stability claims."}
    (OUT/"native_P4_vs_N48_matched_outcomes.json").write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(obj,indent=2),flush=True)

if __name__=="__main__":main()
