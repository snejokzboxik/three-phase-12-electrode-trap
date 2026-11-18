from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lyapunov_extension as qr

SOURCE = ROOT / "results" / "lyapunov_extensions.json"
OUTPUT = ROOT / "results" / "lyapunov_10000rf.json"
PROGRESS = ROOT / "results" / "lyapunov_10000rf_progress.json"
qr.TARGET_RF = 10_000
qr.CHECKPOINT = PROGRESS


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = source["records"]
    chosen = max(records, key=lambda x: x["reference_100rf"])
    previous = {}
    if OUTPUT.exists():
        try:
            previous = json.loads(OUTPUT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    record = previous.get("record")
    for checkpoint_path in (PROGRESS, PROGRESS.with_suffix(PROGRESS.suffix + ".tmp")):
        if not checkpoint_path.exists():
            continue
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("records", [])
            if checkpoint and checkpoint[0].get("source_task_id") == chosen["source_task_id"]:
                if record is None or checkpoint[0].get("completed_rf", 0) > record.get("completed_rf", 0):
                    record = checkpoint[0]
        except (OSError, json.JSONDecodeError):
            pass
    if record and record.get("status") == "BOUNDED_10000_RF":
        result = record
    else:
        if record is None:
            record = json.loads(json.dumps(chosen))
            record["status"] = "RUNNING"
            record["target_rf"] = 10_000
            record["reference_1000rf"] = chosen.get("largest_lyapunov_per_tau")
        result = qr.extend(record)
        atomic_json(OUTPUT, {"status": result["status"], "target_rf": 10_000,
                             "source_1000rf_status": chosen["status"], "record": result})
    atomic_json(PROGRESS, {"status": result["status"], "target_rf": 10_000,
                           "updated_local": time.strftime('%Y-%m-%d %H:%M:%S %z'),
                           "records": [result]})
    qr.write_plot([result], filename="lyapunov_10000rf.png")
    if result["status"] == "BOUNDED_10000_RF":
        print(json.dumps({"status": result["status"], "completed_rf": result["completed_rf"],
                          "reference_100rf": result["reference_100rf"],
                          "reference_1000rf": result.get("reference_1000rf"),
                          "lle_100rf": result.get("largest_lyapunov_100rf_recomputed"),
                          "lle_1000rf": result.get("largest_lyapunov_1000rf_recomputed"),
                          "lle_10000rf": result.get("largest_lyapunov_10000rf"),
                          "updated_local": time.strftime('%Y-%m-%d %H:%M:%S %z')}, indent=2))


if __name__ == "__main__":
    main()
