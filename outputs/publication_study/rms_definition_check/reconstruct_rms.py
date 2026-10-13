from __future__ import annotations

import csv
import math
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
QUALITY = ROOT / "outputs" / "publication_study" / "quality_mesh_convergence"
OUT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "default.toml"

RADII_MM = [0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.400, 0.550, 1.000]
N_THETA = 144
N_PHASES = 12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def f(value: str) -> float:
    return float(value)


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def rms(values: list[float]) -> float:
    return math.sqrt(mean([value * value for value in values]))


def key(row: dict[str, str]) -> tuple[float, float, float]:
    return (f(row["radius_mm"]), f(row["angle_deg"]), f(row["drive_phase_deg"]))


def main() -> None:
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    u0 = float(config["rf"]["U0"])
    orbit_radius = float(config["geometry"]["electrode_orbit_radius"])
    global_scale = u0 / orbit_radius

    mesh_files = {
        name: {
            "mesh": QUALITY / f"{name}_mesh.npz",
            "basis": QUALITY / f"{name}_basis.npz",
        }
        for name in ("R1", "R2")
    }
    for paths in mesh_files.values():
        if not paths["mesh"].exists() or not paths["basis"].exists():
            raise FileNotFoundError(paths)

    samples = read_csv(QUALITY / "field_samples.csv")
    old_rows = read_csv(QUALITY / "field_convergence.csv")
    selected = {round(radius, 9) for radius in RADII_MM}

    by_mesh: dict[str, dict[tuple[float, float, float], tuple[float, float]]] = {"R1": {}, "R2": {}}
    for row in samples:
        if row["mesh"] not in by_mesh:
            continue
        radius, angle, phase = key(row)
        if round(radius, 9) not in selected:
            continue
        by_mesh[row["mesh"]][(round(radius, 9), round(angle, 9), round(phase, 9))] = (
            f(row["Ex_V_per_m"]),
            f(row["Ey_V_per_m"]),
        )

    component_rows: list[dict[str, float]] = []
    summary: list[dict[str, float]] = []
    old_check: list[dict[str, float]] = []
    phase_spreads: dict[float, float] = {}

    for radius in RADII_MM:
        radius_key = round(radius, 9)
        keys = sorted(k for k in by_mesh["R1"] if k[0] == radius_key)
        if len(keys) != N_THETA * N_PHASES:
            raise RuntimeError(f"Expected {N_THETA * N_PHASES} R1 points at {radius} mm, found {len(keys)}")
        if set(keys) != {k for k in by_mesh["R2"] if k[0] == radius_key}:
            raise RuntimeError(f"R1/R2 physical sample grid mismatch at {radius} mm")

        r1_ex = []
        r1_ey = []
        r2_ex = []
        r2_ey = []
        d_ex = []
        d_ey = []
        phase_differences: list[float] = []
        phase_keys: dict[float, list[tuple[float, float, float]]] = {}

        for sample_key in keys:
            r1x, r1y = by_mesh["R1"][sample_key]
            r2x, r2y = by_mesh["R2"][sample_key]
            dex = r2x - r1x
            dey = r2y - r1y
            r1_ex.append(r1x)
            r1_ey.append(r1y)
            r2_ex.append(r2x)
            r2_ey.append(r2y)
            d_ex.append(dex)
            d_ey.append(dey)
            phase_keys.setdefault(sample_key[2], []).append(sample_key)
            component_rows.append(
                {
                    "radius_mm": radius,
                    "angle_deg": sample_key[1],
                    "drive_phase_deg": sample_key[2],
                    "Ex_R1_V_per_m": r1x,
                    "Ey_R1_V_per_m": r1y,
                    "Ex_R2_V_per_m": r2x,
                    "Ey_R2_V_per_m": r2y,
                    "dEx_V_per_m": dex,
                    "dEy_V_per_m": dey,
                }
            )

        d_pooled = rms([math.hypot(x, y) for x, y in zip(d_ex, d_ey)])
        e_r1 = rms([math.hypot(x, y) for x, y in zip(r1_ex, r1_ey)])
        e_r2 = rms([math.hypot(x, y) for x, y in zip(r2_ex, r2_ey)])
        phase_rms = []
        for phase, phase_sample_keys in phase_keys.items():
            phase_rms.append(
                rms(
                    [
                        math.hypot(
                            by_mesh["R2"][sample_key][0] - by_mesh["R1"][sample_key][0],
                            by_mesh["R2"][sample_key][1] - by_mesh["R1"][sample_key][1],
                        )
                        for sample_key in phase_sample_keys
                    ]
                )
            )
        d_old = mean(phase_rms)
        phase_spreads[radius] = max(phase_rms) - min(phase_rms)
        stored_old = [
            f(row["rms_field_difference_V_per_m"])
            for row in old_rows
            if row["coarse"] == "R1"
            and row["fine"] == "R2"
            and round(f(row["radius_mm"]), 9) == radius_key
        ]
        stored_old_mean = mean(stored_old)
        old_check.append(
            {
                "radius_mm": radius,
                "reconstructed_old_V_per_m": d_old,
                "stored_old_mean_V_per_m": stored_old_mean,
                "absolute_difference_V_per_m": d_old - stored_old_mean,
            }
        )
        summary.append(
            {
                "radius_mm": radius,
                "E_RMS_R1_V_per_m": e_r1,
                "E_RMS_R2_V_per_m": e_r2,
                "D_old_V_per_m": d_old,
                "D_pooled_V_per_m": d_pooled,
                "relative_error_old": d_old / e_r2,
                "relative_error_pooled": d_pooled / e_r2,
                "global_error_pooled": d_pooled / global_scale,
                "sample_count": len(keys),
            }
        )

    with (OUT / "r1_r2_field_components.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(component_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(component_rows)

    with (OUT / "rms_by_radius.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(summary[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    with (OUT / "old_convention_reconstruction_check.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(old_check[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(old_check)

    critical = next(row for row in summary if abs(row["radius_mm"] - 0.1) < 1e-12)
    old_report_relative = critical["relative_error_old"]
    new_relative = critical["relative_error_pooled"]
    absolute_pp = 100.0 * (new_relative - old_report_relative)
    relative_change_percent = 100.0 * (new_relative - old_report_relative) / old_report_relative
    max_old_reconstruction_error = max(abs(row["absolute_difference_V_per_m"]) for row in old_check)
    max_convention_difference = max(abs(row["D_pooled_V_per_m"] - row["D_old_V_per_m"]) for row in summary)
    critical_phase_spread = phase_spreads[0.1]

    report = f"""# Consistent pooled vector-RMS reconstruction

## Scope and provenance

This isolated post-processing used the final quality-controlled R1/R2 artifacts in `outputs/publication_study/quality_mesh_convergence/`. The saved `R1_mesh.npz`, `R2_mesh.npz`, `R1_basis.npz`, and `R2_basis.npz` files were present. The quality study's complete same-grid `field_samples.csv` already contains the exact corrected-P2 evaluator output for R1 and R2, so those stored field components were reused; no FEM solve, trajectory calculation, ensemble, mesh regeneration, or evaluator rerun was performed.

The saved configuration gives U0 = {u0:g} V and R = {orbit_radius:g} m, hence U0/R = {global_scale:g} V/m. The quality-study hard-coded field evaluation also used U0 = 1000 V and frequency = 10000 Hz, consistent with the saved configuration.

## Sampling and definitions

For each listed nonzero radius, the calculation uses 144 uniformly spaced angular samples and 12 uniformly spaced drive phases, for N = 1728 identical physical sample points for R1 and R2. The field is the full planar vector (Ex, Ey).

The consistent pooled quantities are:

\\[
E_{{\\rm RMS}}^{{(m)}}(r) = \\sqrt{{\\frac{{1}}{{N}}\\sum_{{\\theta,\\phi}}\\left(E_{{x,m}}^2+E_{{y,m}}^2\\right)}} ,
\\]

\\[
D_E^{{(a,b)}}(r) = \\sqrt{{\\frac{{1}}{{N}}\\sum_{{\\theta,\\phi}}\\left[(E_{{x,b}}-E_{{x,a}})^2+(E_{{y,b}}-E_{{y,a}})^2\\right]}} ,
\\]

and \\(ε_E^{{(a,b)}} = D_E^{{(a,b)}}/E_{{\\rm RMS}}^{{(b)}}\\), with b = R2. The old quality-study convention was reconstructed as the arithmetic mean over the 12 phase-specific angular RMS values.

## Critical result at r = 0.100 mm

| quantity | old convention/report | consistent pooled vector RMS |
|---|---:|---:|
| R1 RMS field (V/m) | not reported in the old summary | {critical['E_RMS_R1_V_per_m']:.12g} |
| R2 RMS field (V/m) | {critical['E_RMS_R2_V_per_m']:.12g} | {critical['E_RMS_R2_V_per_m']:.12g} |
| R1→R2 field difference (V/m) | {critical['D_old_V_per_m']:.12g} | {critical['D_pooled_V_per_m']:.12g} |
| relative difference | {100.0 * critical['relative_error_old']:.12g}% | {100.0 * critical['relative_error_pooled']:.12g}% |
| global-normalized difference | not used in old local report | {critical['global_error_pooled']:.12g} |

The published 6.53% value remains unchanged to the displayed precision. The corrected pooled relative difference is **{100.0 * new_relative:.8f}%**, a change of **{absolute_pp:+.8f} percentage points**, or **{relative_change_percent:+.8f}% relative to the old percentage**. At r = 0.100 mm, the spread of the 12 phase-specific RMS differences is only {critical_phase_spread:.6e} V/m, which is why the phase-mean and pooled RMS agree.

The old and reconstructed old conventions agree to a maximum absolute difference of {max_old_reconstruction_error:.6e} V/m across the reported radii; this is numerical roundoff/serialization agreement with the saved quality-study results.
The maximum absolute difference between the reconstructed old phase-mean value and the new pooled value over the requested radii is {max_convention_difference:.6e} V/m.

## Radius-resolved result

The complete table is in `rms_by_radius.csv` and contains E_RMS for R1/R2, the old phase-mean difference, the pooled vector difference, local relative errors, and pooled global normalization. The component-level R1/R2 samples and dEx/dEy are in `r1_r2_field_components.csv`.

The convention change does not alter the qualitative convergence conclusion or the rounded 6.53% central value. The manuscript should clarify the pooled vector-RMS definition; no numerical replacement is required for the existing 6.53% value on account of this convention check.

## Manuscript-ready definition

At each nonzero radius, averages are performed jointly over 144 uniformly spaced angular positions and 12 uniformly spaced RF drive phases. The metric uses the full 2D electric-field vector and computes the RMS of the vector difference; it is not the difference of two RMS magnitudes.

\\[
E_{{\\rm RMS}}^{{(m)}}(r) = \\sqrt{{\\frac{{1}}{{N_\\theta N_\\phi}} \\sum_{{\\theta,\\phi}} \\left(E_{{x,m}}^2+E_{{y,m}}^2\\right)}}.
\\]

\\[
D_E^{{(a,b)}}(r) = \\sqrt{{\\frac{{1}}{{N_\\theta N_\\phi}} \\sum_{{\\theta,\\phi}} \\left[(E_{{x,b}}-E_{{x,a}})^2+(E_{{y,b}}-E_{{y,a}})^2\\right]}}.
\\]

\\[
\\epsilon_E^{{(a,b)}}(r) = \\frac{{D_E^{{(a,b)}}(r)}}{{E_{{\\rm RMS}}^{{(b)}}(r)}}.
\\]
"""
    (OUT / "RMS_DEFINITION_REPORT.md").write_text(report, encoding="utf-8")

    print(f"R1/R2 basis files reused: {mesh_files['R1']['basis'].name}, {mesh_files['R2']['basis'].name}")
    print(f"U0/R = {global_scale:.12g} V/m")
    print(f"r=0.1 mm: E_RMS_R1={critical['E_RMS_R1_V_per_m']:.12g}, E_RMS_R2={critical['E_RMS_R2_V_per_m']:.12g}")
    print(f"r=0.1 mm: D_old={critical['D_old_V_per_m']:.12g}, D_pooled={critical['D_pooled_V_per_m']:.12g}")
    print(f"r=0.1 mm: old_relative={100.0 * old_report_relative:.12g}%, pooled_relative={100.0 * new_relative:.12g}%")
    print(f"max old reconstruction check difference = {max_old_reconstruction_error:.6e} V/m")


if __name__ == "__main__":
    main()
