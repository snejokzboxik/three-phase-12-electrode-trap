"""Check figure export completeness, vector containers, and 600-dpi PNGs."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "processed" / "figure_validation.json"
STEMS = [
    "fig01_harmonic_cross_structure",
    "fig02_matched_native_p4_trajectories",
    "fig03_energy_and_rf_work",
    "fig04_mesh_and_rf_phase_validation",
    "fig05_phase_control_causal_check",
    "fig06_radial_velocity_and_angular_momentum",
    "fig07_parameter_robustness_pilot",
]


def main() -> None:
    rows = []
    for stem in STEMS:
        pdf, svg, png = (FIG / f"{stem}.{ext}" for ext in ("pdf", "svg", "png"))
        assert pdf.is_file() and svg.is_file() and png.is_file()
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert "<svg" in svg.read_text(encoding="utf-8", errors="replace")[:10000]
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            width, height = image.size
            assert abs(dpi[0]-600) < 1.0 and abs(dpi[1]-600) < 1.0
        rows.append({"stem": stem, "png_px": [width, height], "png_dpi": list(dpi),
                     "pdf_bytes": pdf.stat().st_size, "svg_bytes": svg.stat().st_size})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "passed", "figure_count": len(rows), "figures": rows},
                              indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "figure_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
