"""Write a source/output checksum manifest for V7."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "reproducibility_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = []
    for directory in (HERE / "completed", HERE / "traces", HERE / "processed", HERE / "figures"):
        paths.extend(p for p in directory.rglob("*") if p.is_file())
    paths.extend(p for p in HERE.iterdir() if p.is_file() and p.name != MANIFEST.name)
    paths.extend([
        ROOT / "outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py",
        ROOT / "outputs/octupole_comparison_home/confinement_maps_v4/run_v4.py",
        ROOT / "src/three_phase_trap/rf_reconstruction.py",
        ROOT / "outputs/abc_validation/direct_fem_validation/v2/h35_P4/diagnostic_N48.json",
        ROOT / "outputs/abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json",
        ROOT / "outputs/octupole_comparison_home/chaos_escape_v6/processed/trajectory_results_v6.csv",
        ROOT / "outputs/octupole_comparison_home/matched_pseudopotential_v5/processed/matched_strength_balanced_pairs.csv",
    ])
    unique = sorted({p.resolve() for p in paths if p.exists()}, key=lambda p: str(p).lower())
    records = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in unique]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    packages = {}
    for package in ("numpy", "scipy", "Pillow", "reportlab"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "study": "V7 causal harmonic ablation and pseudopotential cross-term analysis",
        "git_head_before_v7_commit": commit,
        "python": platform.python_version(),
        "packages": packages,
        "trajectory_count": len(list((HERE / "completed").glob("*.json"))),
        "trajectory_trace_count": len(list((HERE / "traces").glob("*.npz"))),
        "file_count": len(records),
        "files": records,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(records)} checksummed files")


if __name__ == "__main__":
    main()
