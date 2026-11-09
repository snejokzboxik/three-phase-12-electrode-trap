"""Input/output inventory for the V3 confinement study (no RF integrations)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    audit = json.loads((HERE / "physical_model_audit.json").read_text(encoding="utf-8"))
    fields = {}
    for name, item in audit["field_sources"].items():
        path = ROOT / item["path"]
        fields[name] = {"path": str(item["path"]), "exists": path.is_file(),
                        "bytes": path.stat().st_size if path.exists() else None,
                        "sha256": sha256(path) if path.is_file() else None,
                        "expected_sha256": item["sha256"]}
    records = {}
    journal = HERE / "trajectory_checkpoint.jsonl"
    jobs = {}
    failures = 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            failures += 1
            continue
        status = row.get("status", "unknown")
        records[status] = records.get(status, 0) + 1
        if status in ("retained", "escaped"):
            jobs[(row.get("job_id"), row.get("job", {}).get("horizon_RF"))] = status
    raw = list((HERE / "trajectories").glob("*.csv"))
    checkpoints = list((HERE / "checkpoints").glob("*.json"))
    figures = sorted(p.name for p in HERE.glob("figure_*.*") if p.suffix in (".png", ".svg", ".pdf"))
    selected = HERE / "selected_trajectory_samples.csv"
    resume_bundle = HERE / "v3_resume_bundle.zip"
    summary = json.loads((HERE / "analysis_summary.json").read_text(encoding="utf-8"))
    output = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "field_files": fields,
        "journal_lines_by_status": records,
        "unique_successful_job_horizons": len(jobs),
        "journal_parse_failures": failures,
        "auxiliary_failure_rows": max(0, sum(1 for _ in (HERE / "auxiliary_failures.csv").open(encoding="utf-8")) - 1),
        "checkpoint_files": len(checkpoints),
        "checkpoint_bytes": sum(p.stat().st_size for p in checkpoints),
        "raw_trajectory_files_local": len(raw),
        "raw_trajectory_bytes_local": sum(p.stat().st_size for p in raw),
        "selected_trajectory_sample_csv": {"exists": selected.is_file(), "bytes": selected.stat().st_size if selected.exists() else None},
        "resume_bundle": {"exists": resume_bundle.is_file(), "bytes": resume_bundle.stat().st_size if resume_bundle.exists() else None},
        "publication_figures": figures,
        "report_exists": (HERE / "RF_CONFINEMENT_SEARCH_REPORT_RU.md").is_file(),
        "analysis_summary_counts": {k: v for k, v in summary.items() if k != "summaries"},
    }
    (HERE / "input_output_audit.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("journal_lines_by_status", "unique_successful_job_horizons", "checkpoint_files", "raw_trajectory_files_local", "raw_trajectory_bytes_local", "publication_figures", "report_exists")}, indent=2))


if __name__ == "__main__":
    main()
