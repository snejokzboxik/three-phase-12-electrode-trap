"""Re-hash all V17B source, report, figure, and result artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
manifest_path = OUT / "MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "PARTIAL"
manifest["created_utc"] = "2026-11-18"
new_work = [
    "RO3 zero-spin startup phase-grid scan through 10000 RF",
    "full 6D frequency ramp from rest, with DOP853 step refinement",
    "full 6D positive-to-negative field rotation reversal and step refinement",
    "one-dimensional full 6D COM perturbation slice at the RO3 startup point",
]
manifest["completed"] = list(dict.fromkeys(manifest.get("completed", []) + new_work))
files = []
for path in sorted(OUT.rglob("*")):
    if (not path.is_file() or path.name == "MANIFEST.json" or
            "__pycache__" in path.parts or ".pytest_cache" in path.parts):
        continue
    data = path.read_bytes()
    files.append({"path": path.relative_to(OUT).as_posix(),
                  "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
manifest["files"] = files
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"files": len(files), "completed": len(manifest["completed"]),
                  "status": manifest["status"]}, ensure_ascii=False))
