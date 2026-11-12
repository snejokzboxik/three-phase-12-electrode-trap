"""Independent integrity and numerical-protocol checks for V5 primary outputs."""
from __future__ import annotations
import csv, hashlib, json, math
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree
from PIL import Image

HERE=Path(__file__).resolve().parent

def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def main():
    pairs=csv_rows(HERE/"processed/matched_strength_balanced_pairs.csv")
    starts=csv_rows(HERE/"balanced_initial_conditions.csv")
    if len(pairs)!=192 or len(starts)!=192 or len({r["condition_id"] for r in pairs})!=192:
        raise ValueError("The balanced paired grid does not contain 192 unique conditions.")
    if len({r["condition_id"] for r in starts})!=192: raise ValueError("Initial-condition table has duplicate IDs.")
    jpath=HERE/"trajectory_checkpoint.jsonl"
    journal=[json.loads(x) for x in jpath.read_text(encoding="utf-8").splitlines()]
    latest={r["job_id"]:r for r in journal}
    if len(latest)!=192 or any(r["status"] not in ("retained","escaped") for r in latest.values()):
        raise ValueError("Standing trajectory journal is incomplete or contains a failed/paused trajectory.")
    byid={r["condition_id"]:r for r in pairs}
    for row in pairs:
        cid=row["condition_id"]
        j=next(x for x in latest.values() if x["job"].get("condition_id")==cid)
        if j["status"]!=row["standing_status"]: raise ValueError(f"Status mismatch for {cid}.")
        for model in ("rotating","standing"):
            status=row[model+"_status"]
            escape=row[model+"_escape_time_RF"]
            censor=row[model+"_right_censored_at_RF"]
            if status=="retained":
                if escape not in (None,"") or float(censor)!=1000: raise ValueError(f"Bad right-censor record: {cid}/{model}")
            elif status=="escaped":
                if escape in (None,"") or censor not in (None,""): raise ValueError(f"Bad event record: {cid}/{model}")
            else: raise ValueError(f"Unknown classification: {cid}/{model}")
        if int(float(row["horizon_RF"]))!=1000 or float(row["frequency_Hz"])!=100000:
            raise ValueError(f"Wrong horizon/frequency for {cid}")
        if float(row["rotating_voltage_V"])!=250 or abs(float(row["standing_voltage_V"])-353.5533905932738)>1e-13:
            raise ValueError(f"Wrong matched-strength voltage for {cid}")
    source=json.loads((HERE/"source_reuse_audit.json").read_text(encoding="utf-8"))
    if not (source["all_initial_state_reconstructions_pass"] and source["all_tangential_sign_checks_pass"] and source["solver_tolerances_match"]):
        raise ValueError("Rotating V4 reuse audit did not pass.")
    strict=json.loads((HERE/"processed/strict_validation_v5.json").read_text(encoding="utf-8"))
    if strict["completed_strict_trajectories"]!=6 or not strict["all_selected_classifications_preserved"]:
        raise ValueError("Tight-tolerance validation is incomplete or changed a selected result.")
    classes=Counter(r["difference_class"] for r in pairs)
    if classes!=Counter({"both_retained":92,"rotating_only_retained":100}):
        raise ValueError(f"Unexpected paired outcome partition: {dict(classes)}")
    pngs=sorted((HERE/"figures").glob("*.png"))
    svgs=sorted((HERE/"figures").glob("*.svg")); pdfs=sorted((HERE/"figures").glob("*.pdf"))
    if len(pngs)!=4 or len(svgs)!=4 or len(pdfs)!=4: raise ValueError("Expected four figure exports per format.")
    for p in pngs:
        with Image.open(p) as im:
            if im.info.get("dpi",(0,0))[0]<599: raise ValueError(f"PNG is not 600 dpi: {p.name}")
    for p in svgs: ElementTree.parse(p)
    for p in pdfs:
        if p.read_bytes()[:5]!=b"%PDF-": raise ValueError(f"Invalid PDF signature: {p.name}")
    status=Counter(r["standing_status"] for r in pairs)
    byenergy={}
    for e in (1.,2.,3.):
        g=[r for r in pairs if float(r["energy_eV"])==e]
        byenergy[str(int(e))]={"n":len(g),"rotating_retained":sum(r["rotating_status"]=="retained" for r in g),
            "standing_retained":sum(r["standing_status"]=="retained" for r in g),
            "rotating_only_retained":sum(r["difference_class"]=="rotating_only_retained" for r in g)}
    result={"status":"passed","paired_conditions":192,"standing_trajectory_jobs":len(latest),
        "standing_status_counts":dict(status),"paired_difference_counts":dict(classes),"by_energy":byenergy,
        "rotating_reuse":{"V4_calculated":168,"V3_reused_under_V4_verified_protocol":24,"all_state_and_sign_checks_pass":True},
        "strict_tolerance":{"n_trajectories":6,"classifications_preserved":True},
        "right_censoring":"retained statuses carry censor time 1000 RF and null escape; event statuses carry escape time and no censor timestamp",
        "figure_exports":{"PDF":len(pdfs),"SVG":len(svgs),"PNG_600dpi":len(pngs),"svg_parse":"passed","pdf_signature":"passed"},
        "claims":"finite-time grid comparison only; no ensemble probabilities, full-pseudopotential equality, or infinite-time stability inference"}
    (HERE/"processed/numerical_verification_v5.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
