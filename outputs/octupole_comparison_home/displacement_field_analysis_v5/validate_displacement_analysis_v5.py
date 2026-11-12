"""Integrity and numerical-consistency checks for displacement-field analysis V5."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
from xml.etree import ElementTree
from PIL import Image

HERE=Path(__file__).resolve().parent

def readcsv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def main():
    summary=json.loads((HERE/"displacement_analysis_summary_v5.json").read_text(encoding="utf-8"))
    st=json.loads((HERE/"stationary_structure_v5.json").read_text(encoding="utf-8"))
    minima=readcsv(HERE/"effective_minimum_scaling.csv")
    pilot=readcsv(HERE/"existing_pilot_minima_reference.csv")
    if len(minima)!=4 or len(pilot)!=11 or summary["solved_fields_reused"]!=12:
        raise ValueError("Expected four radial scaling points, 11 pilot geometries and 12 existing solved fields.")
    if not 0.30<summary["minimum_scaling"]["log_log_exponent"]<0.35:
        raise ValueError("Descriptive scaling exponent fell outside the prereported cubic-root comparison band.")
    if abs(summary["harmonic_radial_order"][0]["potential_coefficient_radial_exponent"]-1)>0.02:
        raise ValueError("Dipole potential coefficient is inconsistent with the sampled linear radial scaling.")
    if abs(summary["harmonic_radial_order"][1]["potential_coefficient_radial_exponent"]-4)>0.02:
        raise ValueError("Octupole potential coefficient is inconsistent with the sampled fourth-power scaling.")
    if abs(summary["validated_10um_minimum"]["cross_resolution_radius_difference_mm"]-6.090954536352999e-6)>1e-12:
        raise ValueError("h35/h50 minimum-position discrepancy changed from the validated source result.")
    refinements=st["stationary_saddle_refinements"]
    if len(refinements)!=4 or not all(x["root_success"] and x["saddle_signature_seen"] for x in refinements):
        raise ValueError("The h35/h50 finite-difference saddle refinements did not all retain their saddle signature.")
    for pair in ("mirror-minimum connecting ridge","global-to-secondary connecting ridge"):
        a=next(x for x in refinements if x["resolution"]=="h35" and x["pair"]==pair)
        b=next(x for x in refinements if x["resolution"]=="h50" and x["pair"]==pair)
        if abs(a["barrier_above_higher_endpoint_eV"]-b["barrier_above_higher_endpoint_eV"])/a["barrier_above_higher_endpoint_eV"]>0.01:
            raise ValueError(f"h35/h50 saddle barrier differs by more than 1%: {pair}")
        if math.dist(a["stationary_candidate_mm"],b["stationary_candidate_mm"])>0.001:
            raise ValueError(f"h35/h50 saddle locations differ by more than one micrometre: {pair}")
    curves=st["curvature_results"]
    for res in ("h35","h50"):
        q=[x for x in curves if x["resolution"]==res and x["pattern"]=="radial_out_10um" and x["stationary_candidate"]=="global_minimum_branch_1" and x["step_um"] in (25,50,100)]
        if len(q)!=3 or not all(x["positive_curvature_in_both_directions"] for x in q):
            raise ValueError(f"The 10-100 um local-curvature sign is not positive on {res}.")
    if st["FEM_meshes_rebuilt"] or st["RF_integrations_performed"]:
        raise ValueError("Unexpected FEM solve or RF integration was recorded.")
    pngs=list((HERE/"figures").glob("*.png"));svgs=list((HERE/"figures").glob("*.svg"));pdfs=list((HERE/"figures").glob("*.pdf"))
    if not (len(pngs)==len(svgs)==len(pdfs)==3):raise ValueError("Expected three figures in each required format.")
    for p in pngs:
        with Image.open(p) as im:
            if im.info.get("dpi",(0,0))[0]<599:raise ValueError(f"PNG export is below 600 dpi: {p.name}")
    for p in svgs:ElementTree.parse(p)
    for p in pdfs:
        if p.read_bytes()[:5]!=b"%PDF-":raise ValueError(f"Bad PDF signature: {p.name}")
    result={"status":"passed","existing_geometry_count":11,"existing_solved_P4_field_count":12,
        "new_FEM_solves":0,"new_RF_integrations":0,"minimum_scaling_exponent":summary["minimum_scaling"]["log_log_exponent"],
        "harmonic_radial_exponents":{str(x["azimuthal_mode_m"]):x["potential_coefficient_radial_exponent"] for x in summary["harmonic_radial_order"]},
        "h35_h50_minimum_delta_nm":summary["minimum_scaling"]["h35_h50_10um_radii_mm"]["absolute_difference_mm"]*1000,
        "finite_difference_saddles":{"count":len(refinements),"all_hessian_signatures_pass":True,"barrier_mesh_difference_under_1pct":True},
        "curvature_minima_signs":"positive at global minimum for 25/50/100 um stencils on both h35/h50",
        "figure_exports":{"PDF":len(pdfs),"SVG":len(svgs),"PNG_600dpi":len(pngs),"svg_parse":"passed","pdf_signature":"passed"},
        "claim_limits":"descriptive four-point exponent; finite-domain stationary search; local effective-potential barriers do not equal aperture escape thresholds or fabrication tolerances"}
    (HERE/"numerical_verification_v5.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
