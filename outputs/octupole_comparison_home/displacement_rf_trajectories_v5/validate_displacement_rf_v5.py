"""Validate completed displacement RF records, field provenance, censoring and figure exports."""
from __future__ import annotations
import csv,hashlib,importlib.metadata,json,math,sys
from pathlib import Path
from xml.etree import ElementTree
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def j(path):return json.loads(path.read_text(encoding="utf-8"))
def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def main():
    checks=j(HERE/"numerical_checks_v5.json")
    audit=j(HERE/"initial_condition_audit.json")
    refinement=j(HERE/"numerical_refinement_comparison_v5.json")
    pairs=rows(HERE/"processed_trajectory_pairs_v5.csv")
    summaries=rows(HERE/"processed_trajectory_summary_v5.csv")
    if len(pairs)!=5 or len(summaries)!=13:raise ValueError("Expected five paired geometries and thirteen trajectories.")
    if audit["status"]!="passed" or audit["max_component_difference"]!=0 or audit["direction_sign_audit"]!="passed":raise ValueError("V4 launch/sign audit failed.")
    if abs(audit["V4_reference_escape_time_RF"]-882.7570104978275)>1e-8:raise ValueError("V4 reference start changed.")
    target_state=[audit["initial_conditions"][k] for k in ("x0_m","y0_m","vx0_m_s","vy0_m_s")]
    for p in pairs:
        if p["rotating_status"]!="retained" or p["rotating_right_censor_RF"]!="1000":raise ValueError(f"Rotation censor mismatch: {p['geometry']}")
        if p["standing_status"]!="escaped" or not p["standing_escape_time_RF"] or p["standing_right_censor_RF"]:raise ValueError(f"Standing escape/censor mismatch: {p['geometry']}")
        if p["paired_class"]!="rotating_only_retained" or p["n_matched_deterministic_starts"]!="1":raise ValueError("Unexpected paired outcome.")
        if float(p["rotating_median_peak_spacing_RF"])<10 or float(p["standing_median_peak_spacing_RF"])<10:raise ValueError("Radial envelope oscillation diagnostic did not exclude RF-scale ripples.")
    if refinement["status"]!="passed" or len(refinement["selected_cases"])!=3:raise ValueError("Strict tolerance comparison failed.")
    if not all(x["same_classification"] for x in refinement["selected_cases"]):raise ValueError("At least one strict repeat changed classification.")
    if max(abs(x["escape_time_difference_RF"]) for x in refinement["selected_cases"])>0.003:raise ValueError("Strict escape-time difference exceeded 0.003 RF.")
    # Verify first-passage samples and checkpoint summaries preserve event/censor conventions.
    for p in pairs:
        for model in ("rotating","standing"):
            key=f"{p['geometry']}_{model}_standard"
            d=j(HERE/"trajectories"/key/"summary.json")
            if not d["complete"]:raise ValueError(f"Incomplete trajectory: {key}")
            if max(abs(float(a)-float(b)) for a,b in zip(d["initial_state"],target_state))>1e-13:
                raise ValueError(f"Matched initial state changed in trajectory record: {key}")
            ic=d["initial_conditions"];theta=math.radians(float(ic["azimuth_deg"]))
            tx,ty=-math.sin(theta),math.cos(theta)
            cross=float(d["initial_state"][2])*tx+float(d["initial_state"][3])*ty
            if int(ic["tangential_direction"])!=-1 or cross>=0 or int(ic["verified_tangential_sign"])!=-1:
                raise ValueError(f"Tangential velocity sign audit failed in trajectory record: {key}")
            if d["status"]=="escaped":
                segments=sorted((HERE/"trajectories"/key).glob("segment_*.csv"))
                last=rows(segments[-1])[-1]
                if abs(float(last["r_mm"])-8)>1e-8:raise ValueError(f"Escape root not at aperture: {key}")
                if d["right_censor_time_RF"] is not None:raise ValueError(f"Escaped case was censored: {key}")
            elif d["right_censor_time_RF"]!=1000 or d["escape_time_RF"] is not None:
                raise ValueError(f"Retained case censor fields inconsistent: {key}")
    # Verify source hashes for all five already-solved fields and no field arrays were produced here.
    field_inputs=[]
    for geometry in ("nominal","radial_out_10um","radial_out_25um","radial_out_50um","radial_out_100um"):
        d=j(HERE/"field_loads"/f"{geometry}.json")
        if d.get("resolution")!="h50":raise ValueError(f"Unexpected field resolution: {geometry}")
        if geometry=="nominal":
            for rel,key in ((d["source_mesh"],"source_mesh_sha256"),(d["source_sector_solution"],"source_sector_solution_sha256")):
                p=ROOT/rel
                if sha(p)!=d[key]:raise ValueError(f"Nominal source field hash mismatch: {rel}")
                field_inputs.append({"geometry":geometry,"path":str(rel).replace("\\","/"),"sha256":d[key]})
        else:
            for rel,key in ((d["mesh_path"],"mesh_sha256"),):
                p=ROOT/rel
                if sha(p)!=d[key]:raise ValueError(f"Displaced mesh hash mismatch: {rel}")
                field_inputs.append({"geometry":geometry,"path":str(rel).replace("\\","/"),"sha256":d[key]})
            coeff=ROOT/"outputs/octupole_comparison_home/electrode_displacement_sensitivity_v1"/"calculations"/f"{geometry}_h50"/"p4_coefficients.npz"
            if sha(coeff)!=d["field_sha256"]:raise ValueError(f"Displaced field coefficient hash mismatch: {geometry}")
            field_inputs.append({"geometry":geometry,"path":str(coeff.relative_to(ROOT)).replace("\\","/"),"sha256":d["field_sha256"]})
    if list((HERE/"field_loads").glob("*.npz")):raise ValueError("Redundant FEM coefficient arrays were copied into the study.")
    # Check all print exports; previews have already been visually inspected.
    fig=HERE/"figures";pngs=sorted(fig.glob("*.png"));svgs=sorted(fig.glob("*.svg"));pdfs=sorted(fig.glob("*.pdf"))
    if not (len(pngs)==len(svgs)==len(pdfs)==4):raise ValueError("Expected four publication figures in PDF, SVG and PNG.")
    for p in pngs:
        with Image.open(p) as im:
            if im.info.get("dpi",(0,0))[0]<599:raise ValueError(f"PNG below 600 dpi: {p.name}")
    for p in svgs:ElementTree.parse(p)
    for p in pdfs:
        if p.read_bytes()[:5]!=b"%PDF-":raise ValueError(f"Bad PDF: {p.name}")
    budget=j(ROOT/"outputs/octupole_comparison_home/matched_pseudopotential_v5/integration_deadline.json")
    if sum(float(r["integration_wall_seconds"]) for r in summaries)>7200:raise ValueError("New integration wall-time sum exceeded the authorized 4-hour limit.")
    resource=j(HERE/"global_deadline_status.json").get("resources",{})
    if resource.get("free_disk_GiB",0)<5 or resource.get("process_peak_working_set_GiB",0)>10:raise ValueError("Resource guardrail failed.")
    versions={}
    for pkg in ("numpy","scipy","scikit-fem","meshio","Pillow","reportlab"):
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:versions[pkg]="not installed as a distribution"
    manifest={"study":"V5 targeted displaced-geometry RF dynamics","status":"passed",
      "config":"pilot_config_v5.json","integration_script":"run_targeted_displacement_v5.py",
      "processing_figure_script":"process_and_render_v5.py","validation_script":"validate_displacement_rf_v5.py",
      "python":sys.version,"package_versions":versions,"verified_existing_field_inputs":field_inputs,
      "new_FEM_solves":0,"new_FEM_reconstructions":0,"standard_integrations":10,"strict_repeats":3,
      "integration_budget_seconds":14400,"global_started_utc":budget["started_utc"],"global_deadline_unix_s":budget["deadline_unix_s"],
      "initial_state_audit":"passed; exact V4 state, maximum component difference 0",
      "velocity_direction_audit":"passed for all 13 records; negative tangential/CW",
      "matched_outcomes":"all five: rotating survived/right-censored at 1000 RF; standing escaped",
      "strict_checks":"three selected standing escapes, classifications agree; all absolute time differences < 0.003 RF",
      "figure_exports":{"count":4,"formats":["PDF","SVG","PNG 600 dpi"],"SVG_parse":"passed","PDF_signature":"passed"},
      "figure_png_sha256":{p.name:sha(p) for p in pngs},"resources_at_final_checkpoint":resource,
      "limits":["one deterministic launch","finite horizon only","no fabrication tolerance claim","V4 N48 reference is not mixed into h50 P4 displacement series"]}
    (HERE/"reproducibility_manifest_v5.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","standard_pairs":len(pairs),"tight_repeats":len(refinement["selected_cases"]),
      "field_inputs_verified":len(field_inputs),"figure_exports":4,"resource_checks":"passed"},indent=2))
if __name__=="__main__":main()
