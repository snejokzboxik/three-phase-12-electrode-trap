from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "search_results.jsonl"


def canonical(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def main():
    raw_lines = RESULTS.read_bytes().splitlines(keepends=True)
    output = []
    added = 0
    for number, raw in enumerate(raw_lines, 1):
        if not raw.endswith(b"\n"):
            raise RuntimeError(f"partial line {number}; refusing to rewrite")
        row = json.loads(raw)
        record_hash = row.get("record_sha256")
        if record_hash:
            payload = dict(row)
            payload.pop("record_sha256")
            if hashlib.sha256(canonical(payload)).hexdigest() != record_hash:
                raise RuntimeError(f"existing checksum mismatch at line {number}")
            output.append(raw)
            continue
        digest = hashlib.sha256(canonical(row)).hexdigest()
        row["record_sha256"] = digest
        output.append(canonical(row) + b"\n")
        added += 1

    tmp = RESULTS.with_suffix(".jsonl.rehash.tmp")
    tmp.write_bytes(b"".join(output))
    os.replace(tmp, RESULTS)

    checked = 0
    for number, raw in enumerate(RESULTS.read_bytes().splitlines(), 1):
        row = json.loads(raw)
        digest = row.get("record_sha256")
        if not digest:
            raise RuntimeError(f"missing checksum after rewrite at line {number}")
        payload = dict(row)
        payload.pop("record_sha256")
        if hashlib.sha256(canonical(payload)).hexdigest() != digest:
            raise RuntimeError(f"checksum mismatch after rewrite at line {number}")
        checked += 1
    print(json.dumps({"rows": checked, "checksums_added": added,
                      "status": "VERIFIED" if checked == len(raw_lines) else "COUNT_MISMATCH"}))
    if checked != len(raw_lines):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
