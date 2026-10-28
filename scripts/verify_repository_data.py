"""Inventory or verify scientific outputs by SHA-256.

Uses only Python's standard library. Does not solve FEM, integrate, or alter
scientific results. By default verify the published subset; --full-local
also checks the large arrays intentionally retained only on the original PC.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"outputs/REPOSITORY_DATA_MANIFEST.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--full-local", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        result = subprocess.run(["git", "-c", "safe.directory="+str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "outputs"],
                                cwd=ROOT, stdout=subprocess.PIPE, check=True)
        tracked = set(v.decode("utf-8") for v in result.stdout.split(b"\0") if v)
        previous = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"] if MANIFEST.exists() else {}
        local_only = {n for n, v in previous.items() if v.get("availability") == "local_only"}
        names = sorted(tracked | local_only)
        files = {}
        for name in names:
            path = ROOT/name
            if path == MANIFEST:
                continue
            if not path.is_file():
                if name in local_only:
                    files[name] = previous[name]
                continue
            files[name] = dict(bytes=path.stat().st_size, sha256=digest(path),
                               availability="github" if name in tracked else "local_only")
            if len(files) % 100 == 0:
                print(f"Inventoried {len(files)} scientific files", flush=True)
        manifest = dict(schema=2, generated_utc=datetime.now(timezone.utc).isoformat(),
                        repository="https://github.com/snejokzboxik/three-phase-12-electrode-trap",
                        scope="Scientific output inventory; availability distinguishes GitHub files from intentionally omitted local-only arrays; manifest excludes itself",
                        file_count=len(files), bytes=sum(v["bytes"] for v in files.values()), files=files)
        temp = MANIFEST.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        temp.replace(MANIFEST)
        print(json.dumps(dict(status="manifest_written", files=len(files), bytes=manifest["bytes"])), flush=True)
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    verified = 0
    verified_bytes = 0
    omitted = 0
    for index, (name, expected) in enumerate(manifest["files"].items(), 1):
        if expected.get("availability", "github") == "local_only" and not args.full_local:
            omitted += 1
            continue
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Manifest path escapes the repository")
        if not path.is_file():
            failures.append(dict(path=name, reason="missing; restore the original local archive for local_only files"))
        elif path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            failures.append(dict(path=name, reason="size/SHA mismatch"))
        verified += 1
        verified_bytes += expected["bytes"]
        if index % 100 == 0:
            print(f"Verified {index}/{manifest['file_count']} scientific files", flush=True)
    if failures:
        print(json.dumps(dict(status="failed", failures=failures), indent=2), flush=True)
        sys.exit(1)
    print(json.dumps(dict(status="passed", verified_files=verified, bytes=verified_bytes,
                          intentionally_omitted_files=omitted,
                          scope="full_local" if args.full_local else "published_subset")), flush=True)


if __name__ == "__main__":
    main()
