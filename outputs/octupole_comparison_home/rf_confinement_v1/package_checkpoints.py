"""Pack lightweight resume artifacts for GitHub; raw RF traces remain local."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "v3_resume_bundle.zip"
FILES = ("trajectory_checkpoint.jsonl", "selected_trajectory_samples.csv", "runtime_log.jsonl", "integration_deadline.json")


def main():
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name in FILES:
            path = HERE / name
            if path.exists():
                archive.write(path, arcname=name)
        for path in sorted((HERE / "checkpoints").glob("*.json")):
            archive.write(path, arcname=f"checkpoints/{path.name}")
    print(f"{OUTPUT.name}: {OUTPUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
