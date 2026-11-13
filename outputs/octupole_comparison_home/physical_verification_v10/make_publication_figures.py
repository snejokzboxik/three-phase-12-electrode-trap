"""Build compact publication figures from the archived V10 data only.

Style follows the user's displacement-paper reference: compact white panels,
small panel letters, sparse axes, restrained teal/purple/orange colors.
No interpolated outcome boundaries are drawn.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PROC = HERE / "processed"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

TEAL = "#258f8a"
PURPLE = "#68449a"
ORANGE = "#e77935"
BLUE = "#3976a8"
GRAY = "#55565a"
LIGHT = "#d9d9d9"
COLORS = {"rotating": TEAL, "standing": ORANGE}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.3,
    "axes.linewidth": 0.65, "lines.linewidth": 1.15, "xtick.major.width": 0.65,
    "ytick.major.width": 0.65, "xtick.major.size": 2.7, "ytick.major.size": 2.7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

def panel(ax, letter):
    ax.text(0.015, 0.98, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.2, fontweight="normal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2.5)

def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    svg_path = FIG / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.025)
    # Matplotlib emits harmless path-command spaces at line ends; remove them
    # so generated vector sources pass repository whitespace checks cleanly.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines())+"\n",
                        encoding="utf-8", newline="\n")
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

def load_npz(stem):
    return np.load(DATA / "native_p4" / f"{stem}.npz")

def figure_harmonics():
    coeff = pd.read_csv(DATA / "p4_harmonics" / "native_p4_harmonic_coefficients.csv")
    # Retain the dominant signed component for each positive spatial order.
    coeff = coeff.sort_values("field_coefficient_V_per_V_m_order_minus1", ascending=False)
    coeff = coeff.drop_duplicates(["mesh_um", "radius_mm", "order_m"])
    d = coeff[coeff.mesh_um == 50].pivot(index="radius_mm", columns="order_m",
                                         values="field_coefficient_V_per_V_m_order_minus1")
    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.28), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    for m, color in [(8, PURPLE), (16, BLUE), (20, ORANGE), (28, GRAY), (32, "#7aa6a1"), (40, "#a26c6a"), (44, "#7b7b7b")]:
        if m in d:
            ax.plot(d.index, d[m] / d[4], marker="o", ms=2.6, color=color, label=fr"$m={m}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Radius, $r$ (mm)")
    ax.set_ylabel(r"Field harmonic ratio, $|E_m|/|E_4|$")
    ax.set_xlim(0, 8.2); ax.set_ylim(1e-7, 2)
    ax.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.7, handlelength=1.2)
    panel(ax, "a")

    cross = pd.read_csv(DATA / "p4_harmonics" / "native_p4_pseudopotential_cross_m4_m8.csv")
    d8 = cross[(cross.mesh_um == 50) & (cross.radius_mm == 8)].sort_values("theta_deg")
    ax = axs[1]
    ax.plot(d8.theta_deg, d8.cross_standing_m4_m8_eV, color=ORANGE, label="Standing")
    ax.plot(d8.theta_deg, d8.cross_rotating_m4_m8_eV, color=TEAL, label="Rotating")
    ax.axhline(0, color=GRAY, lw=0.55)
    ax.set_xlabel(r"Azimuth, $\theta$ (deg)")
    ax.set_ylabel(r"$m=4\times8$ cross term (eV)")
    ax.set_xlim(0, 90); ax.set_xticks([0, 22.5, 45, 67.5, 90])
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "b")

    ax = axs[2]
    for key, color, ls, label in [
        ("psi_rotating_m4_plus_m8_eV", TEAL, "-", r"Rotating, $m=4+8$"),
        ("psi_standing_m4_plus_m8_eV", ORANGE, "-", r"Standing, $m=4+8$"),
        ("psi_rotating_direct_P4_eV", TEAL, "--", "Rotating, P4"),
        ("psi_standing_direct_P4_eV", ORANGE, "--", "Standing, P4"),
    ]:
        ax.plot(d8.theta_deg, d8[key], color=color, ls=ls, label=label)
    ax.set_xlabel(r"Azimuth, $\theta$ (deg)")
    ax.set_ylabel(r"Pseudopotential, $\Psi$ (eV)")
    ax.set_xlim(0, 90); ax.set_xticks([0, 22.5, 45, 67.5, 90])
    ax.legend(frameon=False, fontsize=6.4, loc="upper center", ncol=1, handlelength=1.5)
    panel(ax, "c")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.98)
    save(fig, "fig01_harmonic_cross_structure")

def figure_trajectories():
    rot_name = "50um_rotating_early_az22p5_dplus_0deg_1000rf_tight64_early"
    sta_name = "50um_standing_early_az22p5_dplus_0deg_1000rf_tight64_early"
    rot, sta = load_npz(rot_name), load_npz(sta_name)
    rstate, sstate = rot["states"], sta["states"]
    rt, st = rot["time_s"] * 1e5, sta["time_s"] * 1e5  # RF periods at 100 kHz
    max_period = float(st[-1])
    rmask = rt <= max_period
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.5), gridspec_kw={"wspace": 0.38})
    ax = axs[0]
    ax.plot(rstate[rmask, 0] * 1e3, rstate[rmask, 1] * 1e3, color=TEAL, label="Rotating")
    ax.plot(sstate[:, 0] * 1e3, sstate[:, 1] * 1e3, color=ORANGE, label="Standing")
    circle = plt.Circle((0, 0), 8, fill=False, color=GRAY, lw=0.9, ls=(0, (3, 2)))
    ax.add_patch(circle)
    ax.scatter([sstate[-1, 0] * 1e3], [sstate[-1, 1] * 1e3], s=14, marker="x", color=ORANGE, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-8.5, 8.5); ax.set_ylim(-8.5, 8.5)
    ax.set_xlabel(r"$x$ (mm)"); ax.set_ylabel(r"$y$ (mm)")
    panel(ax, "a")

    ax = axs[1]
    ax.plot(rt[rmask], np.hypot(rstate[rmask, 0], rstate[rmask, 1]) * 1e3, color=TEAL, label="Rotating")
    ax.plot(st, np.hypot(sstate[:, 0], sstate[:, 1]) * 1e3, color=ORANGE, label="Standing")
    ax.axhline(8, color=GRAY, lw=0.85, ls=(0, (3, 2)), label="Aperture")
    ax.set_xlabel(r"Time, $t/T_{\rm RF}$")
    ax.set_ylabel(r"Radius, $r$ (mm)")
    ax.set_xlim(0, max_period); ax.set_ylim(0, 8.5)
    panel(ax, "b")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=TEAL, label="Rotating"),
               Line2D([], [], color=ORANGE, label="Standing"),
               Line2D([], [], color=GRAY, ls=(0, (3, 2)), label="8-mm aperture")]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.57, 1.04), columnspacing=1.0, handlelength=1.5)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.2, top=0.88)
    save(fig, "fig02_matched_native_p4_trajectories")

def figure_work_energy():
    f = PROC / "p4_energy_and_motion_timeseries.csv.gz"
    d = pd.read_csv(f)
    tag = "tight64_early"
    d = d[d.trajectory.str.contains(tag)].copy()
    if d.empty:
        raise RuntimeError("Strict dense energy/work data are missing")
    d["time_rf"] = d.time_rf.astype(float)
    d = d[d.time_rf <= 190]
    fig, axs = plt.subplots(2, 1, figsize=(4.2, 3.55), sharex=True,
                            gridspec_kw={"hspace": 0.10, "height_ratios": [1.0, 1.0]})
    for model in ("rotating", "standing"):
        q = d[d.model == model].sort_values("time_rf")
        # Plot stroboscopic samples only: one sample at each integer RF period.
        period = np.arange(0, int(np.floor(q.time_rf.max())) + 1)
        take = np.array([np.argmin(np.abs(q.time_rf.to_numpy() - n)) for n in period])
        q = q.iloc[take]
        c = COLORS[model]
        axs[0].plot(q.time_rf, q.kinetic_eV, color=c, marker=".", ms=2.0,
                    label=model.capitalize())
        axs[1].plot(q.time_rf, q.work_explicit_RF_eV, color=c, marker=".", ms=2.0,
                    label=fr"{model.capitalize()}: $W_{{\rm RF}}$")
        axs[1].plot(q.time_rf, q.H_eV - q.H_eV.iloc[0], color=c, ls="--", alpha=0.85,
                    label=fr"{model.capitalize()}: $H-H_0$")
    axs[0].set_ylabel(r"Stroboscopic $K$ (eV)")
    axs[1].set_ylabel(r"$W_{\rm RF}$ and $H-H_0$ (eV)")
    axs[1].set_xlabel(r"Time, $t/T_{\rm RF}$")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=TEAL, ls="-", label=r"Rotating: $W_{\rm RF}$"),
               Line2D([], [], color=TEAL, ls="--", label=r"Rotating: $H-H_0$"),
               Line2D([], [], color=ORANGE, ls="-", label=r"Standing: $W_{\rm RF}$"),
               Line2D([], [], color=ORANGE, ls="--", label=r"Standing: $H-H_0$")]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.045), fontsize=5.8, columnspacing=0.75,
               handlelength=1.0, handletextpad=0.25)
    for i, ax in enumerate(axs): panel(ax, "ab"[i])
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.14, top=0.87)
    save(fig, "fig03_energy_and_rf_work")

def figure_radial_momentum():
    d = pd.read_csv(PROC / "p4_energy_and_motion_timeseries.csv.gz")
    d = d[d.trajectory.str.contains("tight64_early") & (d.time_rf <= 190)].copy()
    fig, axs = plt.subplots(2, 1, figsize=(4.2, 3.55), sharex=True,
                            gridspec_kw={"hspace": 0.10})
    for model in ("rotating", "standing"):
        q = d[d.model == model].sort_values("time_rf")
        period = np.arange(0, int(np.floor(q.time_rf.max())) + 1)
        take = np.array([np.argmin(np.abs(q.time_rf.to_numpy() - n)) for n in period])
        q = q.iloc[take]
        c = COLORS[model]
        axs[0].plot(q.time_rf, q.vr_m_s, color=c, marker=".", ms=2.0, label=model.capitalize())
        axs[1].plot(q.time_rf, q.Lz_kg_m2_s / 1e-23, color=c, marker=".", ms=2.0,
                    label=model.capitalize())
    axs[0].set_ylabel(r"Stroboscopic $v_r$ (m s$^{-1}$)")
    axs[1].set_ylabel(r"$L_z$ ($10^{-23}$ kg m$^2$ s$^{-1}$)")
    axs[1].set_xlabel(r"Time, $t/T_{\rm RF}$")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=TEAL, label="Rotating"),
                       Line2D([], [], color=ORANGE, label="Standing")],
               frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(0.58,1.04),
               fontsize=6.7,columnspacing=1.0,handlelength=1.3)
    for i, ax in enumerate(axs): panel(ax, "ab"[i])
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.14, top=0.88)
    save(fig, "fig06_radial_velocity_and_angular_momentum")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def figure_mesh_and_phase():
    rows = []
    for mesh in (35, 50):
        for p in (DATA / "native_p4").glob(f"{mesh}um_*_0deg_1000rf.json"):
            try:
                j = read_json(p)
                if j.get("horizon_rf") == 1000 and "tight" not in p.name:
                    rows.append(j)
            except Exception:
                pass
    # Use explicitly curated, matched starts to prevent duplicate or incomplete rows.
    wanted = ["early_az22p5_dplus", "negative_escaped", "positive_retained", "positive_escaped"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.45), gridspec_kw={"wspace": 0.34})
    ax = axs[0]
    label_lookup = {
        "early_az22p5_dplus": r"$22.5^\circ$ (+)",
        "negative_escaped": r"$9.624^\circ$ (−)",
        "positive_retained": r"$10.261^\circ$ (+)",
        "positive_escaped": r"$10.283^\circ$ (+)",
    }
    from matplotlib.lines import Line2D
    for i, launch in enumerate(wanted):
        for mesh in (35, 50):
            for model in ("rotating", "standing"):
                hit = next((q for q in rows if q.get("launch_label") == launch and int(q.get("mesh_um", -1)) == mesh and q.get("model") == model), None)
                if hit is None: continue
                color = COLORS[model]
                escaped = hit.get("status") == "escaped"
                val = float(hit.get("escape_time_rf")) if escaped else 1000.0
                mesh_shift = -0.06 if mesh == 35 else 0.06
                model_shift = -0.15 if model == "rotating" else 0.15
                ax.scatter(i + mesh_shift + model_shift, val, s=24,
                           marker="o" if mesh == 35 else "s",
                           facecolors=color if escaped else "white", edgecolors=color,
                           lw=0.95, zorder=3)
                if not escaped:
                    ax.annotate("", (i+mesh_shift+model_shift, 1040), (i+mesh_shift+model_shift, 1000),
                                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.65})
    ax.axhline(1000, color=LIGHT, lw=0.7, zorder=0)
    ax.set_ylim(0, 1110); ax.set_ylabel(r"First exit time, $t_{\rm esc}/T_{\rm RF}$")
    ax.set_xticks(range(len(wanted)), [label_lookup[k] for k in wanted], fontsize=7.0)
    ax.set_xlabel("Matched initial condition")
    handles = [Line2D([], [], color=TEAL, marker="o", ls="none", label="Rotating, h35"),
               Line2D([], [], color=TEAL, marker="s", ls="none", label="Rotating, h50"),
               Line2D([], [], color=ORANGE, marker="o", ls="none", label="Standing, h35"),
               Line2D([], [], color=ORANGE, marker="s", ls="none", label="Standing, h50"),
               Line2D([], [], color=GRAY, marker="o", mfc="white", ls="none", label="Retained (open)")]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.27, 1.12), fontsize=6.2, columnspacing=0.8, handletextpad=0.3)
    panel(ax, "a")

    ax = axs[1]
    phase_results = {}
    for p in (DATA / "native_p4").glob("50um_*_v8_phase_sensitive_dplus_*_1000rf_phase_probe.json"):
        j = read_json(p)
        if j.get("model") not in COLORS: continue
        phase = round(float(j.get("phase_deg")))
        phase_results[(j["model"], phase)] = j
    for i, phase in enumerate((45, 315)):
        for model, offset in (("rotating", -0.12), ("standing", 0.12)):
            q = phase_results.get((model, phase))
            if not q: continue
            escaped = q.get("status") == "escaped"
            y = float(q["escape_time_rf"]) if escaped else 1000
            c = COLORS[model]
            ax.scatter(i + offset, y, s=42, marker="o" if escaped else "^",
                       facecolors=c if escaped else "white", edgecolors=c, lw=1.1, zorder=3)
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(0, 1080)
    ax.set_xticks([0, 1], ["45°", "315°"])
    ax.set_xlabel(r"Initial RF phase, $\phi_{\rm RF}$")
    ax.set_ylabel(r"First exit time, $t_{\rm esc}/T_{\rm RF}$")
    ax.legend(handles=[Line2D([], [], color=TEAL, marker="^", mfc="white", ls="none", label="Rotating, retained"),
                       Line2D([], [], color=ORANGE, marker="o", ls="none", label="Standing, escaped"),
                       Line2D([], [], color=ORANGE, marker="^", mfc="white", ls="none", label="Standing, retained")],
              frameon=False, fontsize=5.9, loc="lower left", handletextpad=0.3, labelspacing=0.25)
    panel(ax, "b")
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.23, top=0.82)
    save(fig, "fig04_mesh_and_rf_phase_validation")

def figure_phase_control():
    p = DATA / "phase_structure_control"
    full_paths = list(p.glob("standing_delta*_early_az22p5_dplus_p4h50.json"))
    if not full_paths:
        return False
    launches = ["early_az22p5_dplus", "negative_escaped", "positive_escaped"]
    titles = [r"$\theta_0=22.5^\circ$ (+)", r"$\theta_0=9.624^\circ$ (−)", r"$\theta_0=10.283^\circ$ (+)"]
    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.35), sharey=True,
                            gridspec_kw={"wspace": 0.16})
    for ax, launch, title, letter in zip(axs, launches, titles, "abc"):
        full, avg = {}, {}
        for q in p.glob(f"standing_delta*_{launch}_p4h50.json"):
            j=read_json(q); full[float(j["delta_m8_deg"])]=j
        for q in p.glob(f"average_standing_delta*_{launch}_p4h50.json"):
            j=read_json(q); avg[float(j["delta_m8_deg"])]=j
        for deg in (0., 90., 180.):
            for group, data, marker, offset, color in [
                ("Full RF", full, "o", -3.5, TEAL),
                ("Leading pseudopotential", avg, "s", 3.5, PURPLE),
            ]:
                j=data.get(deg)
                if not j: continue
                escaped=j.get("status")=="escaped"
                y=float(j["escape_time_rf"]) if escaped else 1000.
                ax.scatter(deg+offset,y,s=27,marker=marker,
                           facecolors=color if escaped else "white",edgecolors=color,lw=0.9,zorder=3)
        ax.axhline(1000,color=LIGHT,lw=0.65,zorder=0)
        ax.set_title(title, fontsize=7.6, pad=3)
        ax.set_xlim(-18,198); ax.set_ylim(0,1080)
        ax.set_xticks([0,90,180],["0°","90°","180°"])
        ax.set_xlabel(r"Relative phase, $\delta$")
        panel(ax,letter)
        # Keep the panel tag clear of a right-censored point at delta=0.
        ax.texts[-1].set_position((0.13, 0.98))
    axs[0].set_ylabel(r"First exit time, $t_{\rm esc}/T_{\rm RF}$")
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=TEAL,marker="o",ls="none",label="Full RF"),
             Line2D([],[],color=PURPLE,marker="s",ls="none",label="Leading pseudopotential"),
             Line2D([],[],color=GRAY,marker="o",mfc="white",ls="none",label="Retained to horizon")]
    fig.legend(handles=handles,frameon=False,ncol=3,loc="upper center",
               bbox_to_anchor=(0.5,1.13),fontsize=6.2,columnspacing=1.0,handletextpad=0.35)
    fig.text(0.5,0.005,r"Synthetic inter-harmonic phase; standard drive has $\delta=0$",
             ha="center",va="bottom",fontsize=6.4,color=GRAY)
    fig.subplots_adjust(left=0.10,right=0.99,bottom=0.22,top=0.78)
    save(fig, "fig05_phase_control_causal_check")
    return True

def figure_parameter_robustness():
    """Show three preregistered matched-start checks, not a population sample."""
    from matplotlib.lines import Line2D
    labels = [
        "robust_E1_r1_az22p5_dplus",
        "robust_E3_r1_az22p5_dplus",
        "robust_E2_r2_az22p5_dplus",
    ]
    rows = []
    for label in labels:
        for model in ("rotating", "standing"):
            path = DATA / "native_p4" / f"50um_{model}_{label}_0deg_1000rf.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing completed robustness record: {path.name}")
            j = read_json(path)
            rows.append({
                "mesh_um": j["mesh_um"], "model": model, "launch_label": label,
                "energy_eV": j["initial_energy_eV"], "initial_radius_mm": 1e3*j["initial_radius_m"],
                "azimuth_deg": j["azimuth_deg"], "direction": j["direction"],
                "phase_deg": j["phase_deg"], "status": j["status"],
                "escape_time_rf": j["escape_time_rf"], "horizon_rf": j["horizon_rf"],
                "max_radius_mm": 1e3*j["numerics"]["maximum_three_point_dense_radius_m"],
                "rms_radius_mm": 1e3*j["numerics"]["rms_radius_m"],
                "rtol": j["rtol"], "steps_per_rf_period": j["steps_per_rf_period"],
                "timeseries_file": j["timeseries_file"],
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(PROC / "parameter_robustness_summary.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.55))
    x = np.arange(len(labels), dtype=float)
    for model, offset in (("rotating", -0.12), ("standing", 0.12)):
        q = summary[summary.model == model]
        for i, label in enumerate(labels):
            row = q[q.launch_label == label].iloc[0]
            censored = row.status == "retained_to_horizon"
            y = float(row.horizon_rf if censored else row.escape_time_rf)
            color = COLORS[model]
            ax.scatter(i+offset, y, s=32, marker="o",
                       facecolors="white" if censored else color,
                       edgecolors=color, linewidths=0.9, zorder=3)
            if censored:
                ax.annotate("", xy=(i+offset, 1070), xytext=(i+offset, 1005),
                            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8})
    ax.axhline(1000, color=LIGHT, lw=0.7, zorder=0)
    ax.set_xlim(-0.45, 2.45); ax.set_ylim(0, 1120)
    ax.set_xticks(x, ["1 eV\n1 mm", "3 eV\n1 mm", "2 eV\n2 mm"])
    ax.set_xlabel(r"Initial energy and radius, $E_0$, $r_0$")
    ax.set_ylabel(r"First exit time, $t_{\rm esc}/T_{\rm RF}$")
    ax.text(0.99, 0.965, r"$\theta_0=22.5^\circ$, $\phi_{\rm RF}=0^\circ$, $+v_\theta$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=GRAY)
    ax.legend(handles=[
        Line2D([], [], color=TEAL, marker="o", ls="none", label="Rotating"),
        Line2D([], [], color=ORANGE, marker="o", ls="none", label="Standing"),
        Line2D([], [], color=GRAY, marker="o", mfc="white", ls="none", label="Retained to horizon"),
    ], frameon=False, ncol=1, loc="center right", bbox_to_anchor=(1.0, 0.63),
       fontsize=6.5, labelspacing=0.35, handletextpad=0.35)
    panel(ax, "a")
    fig.text(0.5, 0.005, "h50 native P4; three matched starts, not a statistical ensemble",
             ha="center", va="bottom", fontsize=6.4, color=GRAY)
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.24, top=0.98)
    save(fig, "fig07_parameter_robustness_pilot")

def main():
    figure_harmonics()
    figure_trajectories()
    figure_work_energy()
    figure_radial_momentum()
    figure_mesh_and_phase()
    figure_phase_control()
    figure_parameter_robustness()
    print("Wrote figures:", *[p.name for p in sorted(FIG.glob("*.png"))], sep="\n")

if __name__ == "__main__":
    main()
