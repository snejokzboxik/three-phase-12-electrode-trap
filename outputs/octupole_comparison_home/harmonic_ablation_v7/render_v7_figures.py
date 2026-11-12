"""Publication-size V7 field, pseudopotential, trajectory and ablation figures.

Renders checkpointed data and the measured h35/h50 harmonic reconstructions;
no trajectory integration is performed here. PNG is 600 dpi; PDF/SVG are vector.
"""
from __future__ import annotations
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
V6_RENDERER = ROOT / "outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py"
spec = importlib.util.spec_from_file_location("publication_canvas", V6_RENDERER)
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)
pub.FIGURES = OUT
pub.PNG_DPI = 600
Canvas, Axes = pub.FigureCanvas, pub.Axes

sys.path.insert(0, str(HERE))
import run_v7

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#AA4499"
GRAY = "#59636D"
INK = "#202830"
MUTED = "#5D6872"
GRID = "#D9E0E5"
GOLD = "#D69F00"
MODEL_COLOR = {"rotating": BLUE, "standing": ORANGE}
REP_STYLE = {"full_N48": None, "full_N48_minus_m8": (1.8, 1.0),
             "m4_m8": (0.55, 0.8), "pure_m4": (2.2, 0.8),
             "m4_m8_m16": (1.2, 0.65)}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_checkpoints():
    rows = []
    for line in (HERE / "trajectory_checkpoint.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


ROWS = read_checkpoints()


def find(launch, representation, model, stage=None, field_source="h35_N48_polynomial"):
    matches = [r for r in ROWS if r["job"]["launch_label"] == launch
               and r["job"]["representation"] == representation and r["job"]["model"] == model
               and r["job"].get("field_source", "h35_N48_polynomial") == field_source
               and (stage is None or r["job"]["stage"] == stage)]
    if not matches:
        raise KeyError((launch, representation, model, stage, field_source))
    # Prefer the main protocol when multiple validation copies exist.
    matches.sort(key=lambda r: (r["job"]["stage"] not in ("pilot", "boundary"), r["job"]["stage"]))
    return matches[0]


def trace(row):
    with np.load(ROOT / row["trace_file"]) as z:
        return {k: z[k] for k in z.files}


def axis(c, x, y, w, h, xr, yr, xt, yt, xfmt=lambda v:f"{v:g}", yfmt=lambda v:f"{v:g}", ylog=False):
    return Axes(c, x, y, w, h, xr, yr, xt, yt, xfmt=xfmt, yfmt=yfmt, ylog=ylog)


def title(c, x, y, tag, label):
    c.text(x, y, f"{tag}  {label}", size=8.3, weight="bold", ha="left")


def legend_line(c, x, y, color, label, dash=None, text_size=6.1, lw=.55):
    c.line(x, y, x+6, y, color, lw, dash)
    c.text(x+7.5, y, label, size=text_size, ha="left", color=INK)


def field_modes():
    q35, _ = run_v7._load_source_fields()
    modes = [m for m in range(1, len(q35.sine))
             if max(abs(q35.cosine[m]), abs(q35.sine[m])) > max(1e-12, 1e-8*abs(q35.sine[4]))]
    return q35, modes


def figure1():
    c = Canvas("Fig1_harmonic_structure_and_effective_potential", 178, 153, "double_column")
    c.text(89, 7, "Measured harmonic content reshapes the finite-time escape landscape", size=10.0, weight="bold")
    c.text(89, 14, "Matched leading m = 4 pseudopotential · 100 kHz · h35 N48 field; angular extrema sampled directly", size=6.8, color=MUTED)
    q, modes = field_modes()

    # Panel A: radial growth of each measured field harmonic relative to m=4.
    title(c, 11, 23, "A", "Relative field harmonics")
    a = axis(c, 21, 29, 63, 41, (0.5, 8.0), (1e-6, 1.0), (0.5, 2, 4, 6, 8),
             (1e-6, 1e-4, 1e-2, 1), lambda v:f"{v:g}", lambda v:f"{v:.0e}", ylog=True)
    a.xlabel("Radius r (mm)", y=77, size=6.7)
    a.ylabel("|Eₘ| / |E₄|", x=12, size=6.6)
    rmm = np.linspace(0.5, 8.0, 151)
    rho = rmm*1e-3/q.reference_radius
    col = {8:BLUE, 16:ORANGE, 20:GREEN, 28:PURPLE, 32:GRAY, 40:GOLD, 44:"#56B4E9"}
    base4 = math.sqrt(abs(q.cosine[4])**2+abs(q.sine[4])**2)
    for m in modes:
        if m == 4: continue
        amp = math.sqrt(abs(q.cosine[m])**2+abs(q.sine[m])**2)
        ratio = np.maximum(m*amp/(4*base4)*rho**(m-4), 1e-6)
        a.path(list(zip(rmm, ratio)), color=col.get(m, INK), lw=.55)
        if m in (8, 16, 20, 28):
            a.marker(8.0, float(ratio[-1]), "circle", col.get(m, INK), size=.6)
    c.text(23, 83, "At 8 mm: m=8, 65.9%; m=20, 5.15%. Ratios <10⁻⁶ are clipped.", size=5.8, color=MUTED, ha="left")
    # Compact order key.
    for j, m in enumerate((8, 16, 20, 28, 32, 40, 44)):
        xx = 20+(j%4)*18.5; yy = 88+(j//4)*5
        c.line(xx, yy, xx+4.2, yy, col[m], .65)
        c.text(xx+5.2, yy, f"m={m}", size=5.8, ha="left")

    # Panel B: calculated minimum pseudopotential on each radius circle.
    title(c, 94, 23, "B", "Minimum Ψ over azimuth")
    b = axis(c, 105, 29, 61, 41, (0.5, 8.0), (0, 18), (0, 2, 4, 6, 8), (0, 5, 10, 15),
             lambda v:f"{v:g}", lambda v:f"{v:g}")
    b.xlabel("Radius r (mm)", y=77, size=6.7)
    b.ylabel("minθ Ψ (eV)", x=95, size=6.6)
    b.line(0.5, 2, 8.0, 2, GRAY, .35, (1.3, .8))
    p_rows = read_csv(HERE / "processed" / "effective_pseudopotential_radial_minima.csv")
    rep_style = {"full_N48": None, "full_N48_minus_m8": (1.8, .9), "m4_m8": (.55, .8), "pure_m4": (2.2, .8)}
    for rep in ("pure_m4", "m4_m8", "full_N48_minus_m8", "full_N48"):
        for model in ("rotating", "standing"):
            if rep == "pure_m4" and model == "standing":
                continue  # the two measured leading-m4 controls overlay
            rows = [r for r in p_rows if r["representation"] == rep and r["model"] == model]
            points = [(float(r["radius_mm"]), float(r["minimum_pseudopotential_eV"])) for r in rows]
            b.path(points, color=MODEL_COLOR[model], lw=.55, dash=rep_style[rep])
    # Legend, with line style distinguishing harmonic ablation.
    legend_line(c, 108, 83, INK, "pure m=4", rep_style["pure_m4"], 5.7, .55)
    legend_line(c, 142, 83, INK, "m=4+8", rep_style["m4_m8"], 5.7, .55)
    legend_line(c, 108, 88, INK, "full − m=8", rep_style["full_N48_minus_m8"], 5.7, .55)
    legend_line(c, 142, 88, INK, "full N48", None, 5.7, .55)
    legend_line(c, 108, 94, BLUE, "rotating", None, 5.7, .55)
    legend_line(c, 142, 94, ORANGE, "standing", None, 5.7, .55)

    # Panel C: direct azimuthal cuts at the aperture showing the standing-wave valley.
    title(c, 11, 101, "C", "Angular structure at the 8-mm aperture")
    cc = axis(c, 25, 105, 130, 17, (0, 90), (0, 18), (0, 15, 30, 45, 60, 75, 90), (0, 5, 10, 15),
              lambda v:f"{v:g}", lambda v:f"{v:g}")
    cc.xlabel("Azimuth θ (deg)", y=130, size=6.5)
    cc.ylabel("Ψ (eV)", x=14, size=6.4)
    cc.line(0, 2, 90, 2, GRAY, .35, (1.3, .8))
    # Existing angular samples at exactly r=8 mm.
    detail = read_csv(HERE / "processed" / "effective_pseudopotential_angular_samples.csv")
    styles = [("m4_m8", "rotating", BLUE, None), ("m4_m8", "standing", ORANGE, None),
              ("full_N48", "rotating", BLUE, (1.7,.9)), ("full_N48", "standing", ORANGE, (1.7,.9)),
              ("full_N48_minus_m8", "standing", GREEN, (.55,.8))]
    for rep, model, color, dash in styles:
        rr = [r for r in detail if r["field_resolution"] == "h35" and r["representation"] == rep
              and r["model"] == model and float(r["radius_mm"]) == 8.0 and 0 <= float(r["angle_deg"]) <= 90]
        cc.path([(float(r["angle_deg"]), float(r["pseudopotential_eV"])) for r in rr], color=color, lw=.5, dash=dash)
    legend_line(c, 18, 138, BLUE, "rotating m=4+8", None, 5.35, .5)
    legend_line(c, 66, 138, ORANGE, "standing m=4+8", None, 5.35, .5)
    legend_line(c, 119, 138, BLUE, "rotating N48", (1.7,.9), 5.35, .5)
    legend_line(c, 18, 143, ORANGE, "standing N48", (1.7,.9), 5.35, .5)
    legend_line(c, 72, 143, GREEN, "standing N48 − m=8", (.55,.8), 5.35, .5)
    c.text(89, 150, "Dashed horizontal line: initial kinetic energy 2 eV. Ψ is an averaged-field diagnostic, not an exact invariant.", size=5.7, color=MUTED)
    return c.save()


def _strobe_indices(t, end_rf=None):
    idx = np.flatnonzero(np.abs(t-np.rint(t)) < 1e-7)
    if end_rf is not None:
        idx = idx[t[idx] <= end_rf]
    return idx


def figure2():
    c = Canvas("Fig2_matched_trajectory_energy_and_work", 178, 198, "double_column")
    c.text(89, 7, "Matched dynamics reveal how a shallow angular channel becomes accessible", size=10.0, weight="bold")
    c.text(89, 14, "E₀ = 2 eV · r₀ = 1 mm · θ₀ = 22.5° · phase = 0° · +vθ · 250 / 353.553 V", size=6.6, color=MUTED)
    launch = "early_az22p5_dplus"
    cases = {
        "rot": (find(launch, "full_N48", "rotating", "pilot"), BLUE, "rotating, full N48"),
        "full": (find(launch, "full_N48", "standing", "pilot"), ORANGE, "standing, full N48"),
        "no8": (find(launch, "full_N48_minus_m8", "standing", "pilot"), GREEN, "standing, full − m=8"),
        "m48": (find(launch, "m4_m8", "standing", "pilot"), PURPLE, "standing, m=4+8"),
        "m4": (find(launch, "pure_m4", "standing", "pilot"), GRAY, "standing, pure m=4"),
    }
    traces = {k: trace(v[0]) for k, v in cases.items()}
    # Panel A: once-per-period xy section for the first 185 RF periods.
    title(c, 10, 23, "A", "Stroboscopic x–y trajectories")
    a = axis(c, 22, 29, 49, 49, (-8.5,8.5), (-8.5,8.5), (-8,0,8), (-8,0,8), lambda v:f"{v:g}", lambda v:f"{v:g}")
    a.xlabel("x (mm)", y=85, size=6.6); a.ylabel("y (mm)", x=12, size=6.6)
    circ = [(8*math.cos(z),8*math.sin(z)) for z in np.linspace(0,2*math.pi,181)]
    a.path(circ, color=INK, lw=.42, dash=(1.3,.8))
    for key in ("rot","full","no8","m48","m4"):
        row, color, _ = cases[key]; tr = traces[key]; idx = _strobe_indices(tr["time_RF"], 185.4)
        pts = [(tr["state"][i,0]*1e3, tr["state"][i,1]*1e3) for i in idx[::2]]
        if len(pts) > 1: a.path(pts, color=color, lw=.42)
        if pts: a.marker(*pts[0], "circle", color, size=.55)
        if row["status"] == "escaped":
            a.marker(tr["state"][-1,0]*1e3, tr["state"][-1,1]*1e3, "triangle", color, size=.75)
    c.text(47, 91, "8-mm aperture", size=5.5, color=MUTED)

    # Panel B: radial motion over the interval containing the early escape.
    title(c, 89, 23, "B", "Radius and first passage")
    b = axis(c, 101, 29, 65, 49, (0,200), (0,8.5), (0,50,100,150,200), (0,2,4,6,8), lambda v:f"{v:g}", lambda v:f"{v:g}")
    b.xlabel("Time (RF periods)", y=85, size=6.6); b.ylabel("r (mm)", x=92, size=6.6)
    b.line(0,8,200,8,INK,.4,(1.3,.8))
    for key in ("rot","full","no8","m48","m4"):
        row,color,_=cases[key]; tr=traces[key]; keep=tr["time_RF"]<=200.0
        b.path(list(zip(tr["time_RF"][keep],tr["radius_m"][keep]*1e3)),color=color,lw=.48)
    c.text(104, 91, "Survivors are observed to 1000 RF; plot shows first 200 RF only.", size=5.3, color=MUTED)

    # Panel C: kinetic energy, explicitly not identified with the slow energy.
    title(c, 10, 96, "C", "Instantaneous kinetic energy")
    cc = axis(c, 22, 102, 65, 37, (0,200), (0,8), (0,50,100,150,200), (0,2,4,6,8), lambda v:f"{v:g}", lambda v:f"{v:g}")
    cc.xlabel("Time (RF periods)", y=146, size=6.5); cc.ylabel("K (eV)", x=12, size=6.4)
    for key in ("rot","full","no8","m48"):
        tr=traces[key]; keep=tr["time_RF"]<=200.0
        cc.path(list(zip(tr["time_RF"][keep],tr["kinetic_eV"][keep])),color=cases[key][1],lw=.48)
    c.text(23, 151, "Fast micromotion remains in K(t); it is not the secular energy.", size=5.5, color=MUTED, ha="left")

    # Panel D: instantaneous Hamiltonian for the exact full-field pair.
    title(c, 98, 96, "D", "Hamiltonian variation")
    # The laboratory-frame Hamiltonian contains the large instantaneous RF
    # potential (tens to >100 eV near the outer part of the launch orbit). Keep
    # the full observed range visible; a narrow secular-energy scale would
    # silently draw long, misleading strokes outside the panel.
    d = axis(c, 109, 102, 57, 37, (0,200), (-150,150), (0,50,100,150,200), (-100,0,100), lambda v:f"{v:g}", lambda v:f"{v:g}")
    d.xlabel("Time (RF periods)", y=146, size=6.5); d.ylabel("H (eV)", x=99, size=6.4)
    for key in ("rot","full","no8"):
        tr=traces[key]; keep=tr["time_RF"]<=200.0
        d.path(list(zip(tr["time_RF"][keep],tr["hamiltonian_eV"][keep])),color=cases[key][1],lw=.48)
    c.text(110, 151, "Only the pure rotating m=4 control has the exact Jacobi integral.", size=5.5, color=MUTED, ha="left")

    # Panel E: exact integrated kinetic work, separated by spatial harmonic.
    title(c, 10, 157, "E", "Cumulative RF work by spatial mode")
    full = traces["full"]
    f = axis(c, 25, 164, 140, 17, (0,185.5), (-6,6), (0,50,100,150), (-6,0,6), lambda v:f"{v:g}", lambda v:f"{v:g}")
    f.xlabel("Time (RF periods)", y=186, size=6.0); f.ylabel("Wₖ (eV)", x=14, size=6.0)
    orders = full["harmonic_orders"].astype(int); work=full["work_kinetic_by_mode_eV"]
    total_higher=np.sum(work[:,[i for i,m in enumerate(orders) if m not in (4,8)]],axis=1)
    for vals,color,label in ((work[:,list(orders).index(4)],GRAY,"m=4"),
                             (work[:,list(orders).index(8)],PURPLE,"m=8"),
                             (total_higher,GREEN,"m≥16, summed")):
        f.path(list(zip(full["time_RF"],vals)),color=color,lw=.55)
    legend_line(c, 100, 157, GRAY, "m=4", None, 5.9, .55)
    legend_line(c, 122, 157, PURPLE, "m=8", None, 5.9, .55)
    legend_line(c, 144, 157, GREEN, "m≥16", None, 5.9, .55)
    c.text(89, 193, "Work is integrated as an ODE variable; Σₘ Wₘ = ΔK within 10⁻¹¹ eV.", size=5.5, color=MUTED)
    return c.save()


def figure4():
    """Stroboscopic radial phase-space projection for the matched pilot start."""
    c = Canvas("Fig4_stroboscopic_radial_phase_space", 89, 114, "single_column")
    c.text(44.5, 7, "Stroboscopic radial phase-space projection", size=9.0, weight="bold")
    c.text(44.5, 14, "E₀ = 2 eV · r₀ = 1 mm · θ₀ = 22.5° · +vθ · first 200 RF periods", size=6.0, color=MUTED)
    launch = "early_az22p5_dplus"
    cases = [
        ("full_N48", "rotating", BLUE, "rotating full N48"),
        ("full_N48", "standing", ORANGE, "standing full N48"),
        ("full_N48_minus_m8", "standing", GREEN, "standing full − m=8"),
        ("m4_m8", "standing", PURPLE, "standing m=4+8"),
        ("pure_m4", "standing", GRAY, "standing pure m=4"),
    ]
    ax = axis(c, 15, 27, 62, 49, (0, 8.4), (-120, 120), (0, 4, 8), (-100, 0, 100),
              lambda v:f"{v:g}", lambda v:f"{v:g}")
    ax.xlabel("Radius r (mm)", y=84, size=6.4)
    ax.ylabel("Radial velocity vᵣ (m s⁻¹)", x=2.7, size=6.0)
    ax.line(8, -120, 8, 120, INK, .4, (1.2, .8))
    c.text(72.5, 29, "8 mm", size=5.2, color=MUTED, ha="right")
    for rep, model, color, _label in cases:
        row = find(launch, rep, model, "pilot")
        tr = trace(row)
        idx = _strobe_indices(tr["time_RF"], 200.0)[::2]
        for i in idx:
            ax.marker(float(tr["radius_m"][i]*1e3), float(tr["radial_velocity_m_s"][i]),
                      "circle", color, size=.34)
        if row["status"] == "escaped":
            ax.marker(float(tr["radius_m"][-1]*1e3), float(tr["radial_velocity_m_s"][-1]),
                      "triangle", color, size=.78)
    # Three short legend rows keep all deterministic comparisons identifiable.
    legend_line(c, 12, 91, BLUE, "rotating full N48", None, 5.45, .5)
    legend_line(c, 49, 91, ORANGE, "standing full N48", None, 5.45, .5)
    legend_line(c, 12, 97, GREEN, "standing full − m=8", None, 5.45, .5)
    legend_line(c, 49, 97, PURPLE, "standing m=4+8", None, 5.45, .5)
    legend_line(c, 12, 103, GRAY, "standing pure m=4", None, 5.45, .5)
    c.text(44.5, 110, "Markers sample integer RF periods; retained paths are censored at 1000 RF.", size=5.35, color=MUTED)
    return c.save()


def figure5():
    """Direct angular decomposition of averaged self and cross terms."""
    c = Canvas("Fig5_pseudopotential_cross_term_mechanism", 178, 151, "double_column")
    c.text(89, 7, "Coherent m=4–m=8 mixing creates the standing-field escape valley", size=9.6, weight="bold")
    c.text(89, 14, "Direct decomposition of Ψ at r = 8 mm · matched leading m = 4 pseudopotential · h35 N48 coefficients", size=6.3, color=MUTED)
    data = read_csv(HERE / "processed" / "pseudopotential_interference_angular.csv")
    panels = [
        ("A", 23, "m=4+8: rotating", "m4_m8", "rotating", BLUE),
        ("B", 23, "m=4+8: standing", "m4_m8", "standing", ORANGE),
        ("C", 82, "Full N48: rotating", "full_N48", "rotating", BLUE),
        ("D", 82, "Full N48: standing", "full_N48", "standing", ORANGE),
    ]
    for tag, top, label, rep, model, total_color in panels:
        x0 = 11 if tag in ("A", "C") else 96
        title(c, x0, top, tag, label)
        ax = axis(c, x0+10, top+6, 63, 36, (0,90), (-8,18), (0,45,90), (-5,0,5,10,15),
                  lambda v:f"{v:g}", lambda v:f"{v:g}")
        if tag in ("A", "C"):
            ax.ylabel("Contribution to Ψ (eV)", x=x0, size=5.9)
        ax.xlabel("Azimuth θ (deg)", y=top+47, size=6.1)
        rr = [r for r in data if r["representation"] == rep and r["model"] == model
              and 0.0 <= float(r["angle_deg"]) <= 90.0]
        rr.sort(key=lambda r: float(r["angle_deg"]))
        xs = [float(r["angle_deg"]) for r in rr]
        for key, color, dash in (("self_sum_eV", GRAY, (1.2,.8)),
                                 ("cross_m4_m8_eV", GREEN, (1.7,.8)),
                                 ("other_cross_sum_eV", PURPLE, (.55,.7)),
                                 ("total_pseudopotential_eV", total_color, None)):
            ys = [float(r[key]) for r in rr]
            ax.path(list(zip(xs,ys)), color=color, lw=.57, dash=dash)
    legend_line(c, 17, 138, GRAY, "sum of single-mode terms", (1.2,.8), 5.45, .5)
    legend_line(c, 68, 138, GREEN, "cross term (m=4,8)", (1.7,.8), 5.45, .5)
    legend_line(c, 117, 138, PURPLE, "other cross terms", (.55,.7), 5.45, .5)
    c.text(89, 146, "m=4 and m=8 self terms match; their rotating cross term is zero to 1.7×10⁻¹² eV, while the standing term spans ±7.375 eV.", size=5.5, color=MUTED)
    return c.save()


def figure3():
    c = Canvas("Fig3_boundary_sensitivity_and_harmonic_ablation", 178, 157, "double_column")
    c.text(89, 7, "Late escape depends on a specific combination of measured harmonics", size=10.0, weight="bold")
    c.text(89, 14, "Two V6 transition-side launch states · h35/h50 full N48 and one-at-a-time h35 harmonic deletions", size=6.6, color=MUTED)
    launches = [("boundary_dminus_escaped", "−vθ boundary: θ₀ = 9.624°"),
                ("boundary_dplus_escaped", "+vθ boundary: θ₀ = 10.283°")]
    for j,(launch,label) in enumerate(launches):
        x0 = 12+j*87
        title(c, x0, 24, "A" if j==0 else "B", label)
        ax = axis(c, x0+10, 30, 65, 38, (0,1000),(0,8.5),(0,250,500,750,1000),(0,2,4,6,8),lambda v:f"{v:g}",lambda v:f"{v:g}")
        ax.xlabel("Time (RF periods)", y=76, size=6.4)
        if j==0: ax.ylabel("r (mm)", x=x0, size=6.4)
        ax.line(0,8,1000,8,INK,.38,(1.3,.8))
        selected = [
            (find(launch,"full_N48","standing","boundary"), ORANGE, None),
            (find(launch,"full_N48","standing","resolution_h50","h50_N48_polynomial"), ORANGE,(.55,.8)),
            (find(launch,"full_N48_minus_m8","standing","boundary"), GREEN,(1.5,.8)),
            (find(launch,"full_N48","rotating","boundary"), BLUE,None),
        ]
        for row,color,dash in selected:
            tr=trace(row)
            idx=_strobe_indices(tr["time_RF"])
            ax.path([(tr["time_RF"][i],tr["radius_m"][i]*1e3) for i in idx],color=color,lw=.58,dash=dash)
            if row["status"] == "escaped":
                ax.marker(float(row["escape_time_RF"]),8.0,"triangle",color,size=.8)
        if j==0:
            legend_line(c, 17, 83, ORANGE, "standing full N48, h35", None, 5.7, .5)
            legend_line(c, 69, 83, ORANGE, "standing full N48, h50", (.55,.8), 5.7, .5)
            legend_line(c, 124, 83, GREEN, "standing full − m=8", (1.5,.8), 5.7, .5)
            legend_line(c, 17, 88, BLUE, "rotating full N48", None, 5.7, .5)
            c.text(89, 94, "Stroboscopic r at integer RF phase; triangles mark exact first escape. h50 preserves class but shifts late escape by 33–78 RF periods.", size=5.45, color=MUTED, ha="center")

    # Outcome matrix for the deliberately selected two boundary states.
    title(c, 11, 100, "C", "One-at-a-time deletion: exact finite-time outcomes")
    rows = [
        ("full N48", "full_N48"), ("full − m=8", "full_N48_minus_m8"),
        ("full − m=16", "full_N48_minus_m16"), ("full − m=20", "full_N48_minus_m20"),
        ("full − m=28", "full_N48_minus_m28"), ("full − m=32", "full_N48_minus_m32"),
        ("full − m=40", "full_N48_minus_m40"), ("full − m=44", "full_N48_minus_m44"),
        ("m=4+8+20", "m4_m8_m20"), ("m=4+8+16+20", "m4_m8_m16_m20"),
    ]
    xname,xminus,xplus=14,80,127
    c.text(xname,107,"Applied modes",size=5.8,color=MUTED,ha="left")
    c.text(xminus,107,"−vθ launch",size=5.8,color=MUTED)
    c.text(xplus,107,"+vθ launch",size=5.8,color=MUTED)
    for i,(label,rep) in enumerate(rows):
        yy=112+i*3.05
        c.text(xname,yy,label,size=5.5,ha="left")
        for launch,x in (("boundary_dminus_escaped",xminus),("boundary_dplus_escaped",xplus)):
            matches=[r for r in ROWS if r["job"]["launch_label"]==launch and r["job"]["representation"]==rep
                     and r["job"]["model"]=="standing" and r["job"]["field_source"]=="h35_N48_polynomial"]
            if not matches:
                continue
            r=matches[0]
            if r["status"]=="escaped":
                c.marker(x,yy,"circle",PURPLE,size=.78)
                c.text(x+2,yy,f"{float(r['escape_time_RF']):.1f}",size=5.1,ha="left",color=INK)
            else:
                c.marker(x,yy,"square",GREEN,size=.68)
                c.text(x+2,yy,"≥1000",size=5.1,ha="left",color=INK)
    c.marker(22,148,"circle",PURPLE,size=.7); c.text(25,148,"escaped; number = first passage (RF periods)",size=5.6,ha="left")
    c.marker(91,148,"square",GREEN,size=.7); c.text(94,148,"retained; right-censored at 1000 RF",size=5.6,ha="left")
    return c.save()


def main():
    manifests = [figure1(), figure2(), figure3(), figure4(), figure5()]
    manifest = {"figures": manifests, "png_dpi": 600,
                "source": "V7 checkpointed h35/h50 N48 trajectories, work integrals, and measured FEM-derived harmonic coefficients",
                "interpolation": "No trajectory smoothing. Pseudopotential curves are direct evaluations at each listed radius and angular sample; lines only connect computed samples.",
                "scope": "Selected deterministic starts; no probability or infinite-time claim."}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
