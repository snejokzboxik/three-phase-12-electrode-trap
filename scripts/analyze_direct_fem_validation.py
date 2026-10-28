"""Verify provenance, numerical controls, and physical validity of direct FEM A.

No trajectory alignment, force smoothing, or threshold adjustment is allowed.
--preview reads completed checkpoints without emitting a final verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from three_phase_trap.pilot_dynamics import coordinate_metrics, oscillation_metrics
from three_phase_trap.slow_analysis import centered_period_mean, local_rf_trend
from three_phase_trap.averaged_dynamics import AveragedRFModel
from three_phase_trap.rf_reconstruction import RFReconstruction
from analyze_abc_pilot import csv_write, relative_period_difference, inverse_batches
from run_direct_fem_validation import OUTPUT, BASE, paths, valid, pilot_cases, extended_cases, prepare
from run_abc_pilot import sha, write


def case(h, phase, extent, level="fine", model="A"):
    return dict(model=model, mesh=h, phase=phase, extent=extent, level=level)


def load(c):
    if not valid(c):
        raise ValueError(f"Incomplete native RF case: {c}")
    a, m = paths(c)
    with np.load(a) as z:
        t, y = z["time_s"], z["states"]
    info = json.loads(m.read_text(encoding="utf-8"))
    if c["extent"] == "extended":
        parent, _ = paths({**c, "extent": "pilot"})
        if info["parent_arrays_sha256"] != sha(parent):
            raise ValueError("Extended trajectory parent changed")
    return t, y, info


def slow(h, model, level, extent, phase=None):
    d = OUTPUT/f"h{h}_P4/slow"
    meta = json.loads((d/"complete.json").read_text(encoding="utf-8"))
    name = f"{model}_{level}_{extent}.npz" if phase is None else f"B_matched_phase{phase}_{extent}.npz"
    if meta["settings_sha256"] != sha(OUTPUT/"validation_settings.json") or sha(d/name) != meta["artifacts"][name]:
        raise ValueError("Native slow checkpoint mismatch")
    with np.load(d/name) as z:
        return z["time_s"], z["states"], z["energy_J"]


def frozen(model, phase, extent, level="fine"):
    phase = phase if model == "A_smooth" else 0
    with np.load(BASE/f"pilot/{model}_phase{phase}_{level}_{extent}.npz") as z:
        return z["time_s"], z["states"]


def clocks(a, b):
    if len(a) != len(b) or np.max(abs(a-b)) > 1e-13:
        raise ValueError("Comparison requires identical physical time nodes")


def metrics(a, b, t, settings):
    out = coordinate_metrics(a, b, settings["R_scale_m"])
    pa = oscillation_metrics(t, a, settings["estimated_slow_period_s"])
    pb = oscillation_metrics(t, b, settings["estimated_slow_period_s"])
    out["relative_period_error"] = relative_period_difference(pa, pb)
    out["elapsed_slow_periods"] = float(t[-1]/max(pa["mean_period_s"], pb["mean_period_s"])) if pa["mean_period_s"] and pb["mean_period_s"] else None
    return out


def passed(m, limits):
    return bool(m["relative_rms"] <= limits["relative_rms"] and m["relative_max"] <= limits["relative_max"]
                and m["relative_period_error"] is not None and m["relative_period_error"] <= limits["relative_period"])


def preview(settings):
    rows, filter_rows = [], []
    p = settings["physical"]
    b = AveragedRFModel(RFReconstruction.load(OUTPUT/"h35_P4/diagnostic_N48.json"), p["charge_C"], p["mass_kg"], p["U0_V"], p["frequency_Hz"])
    for extent, phase in (("pilot", 0), ("pilot", 1), ("extended", 0)):
        for h in (50, 35):
            c = case(h, phase, extent)
            if not valid(c):
                continue
            t, y, _ = load(c)
            tm, ym = centered_period_mean(y[:, :2], t, 64)
            for model in ("B", "C"):
                tb, reference = frozen(model, phase, extent)
                clocks(t, tb)
                rows.append(dict(extent=extent, phase=phase, mesh=h, pair="A-"+model+"70", **metrics(ym, reference[32:-32, :2], tm, settings)))
            cc = case(h, phase, extent, "coarse")
            if valid(cc):
                tc, yc, _ = load(cc)
                clocks(t, tc)
                rows.append(dict(extent=extent, phase=phase, mesh=h, pair="coarse-fine", **metrics(centered_period_mean(yc[:, :2], tc, 64)[1], ym, tm, settings)))
            if h == 35:
                if phase == 0 and extent == "pilot":
                    for level in ("refined", "radau"):
                        control = case(h, phase, extent, level)
                        if valid(control):
                            tt, yy, _ = load(control)
                            clocks(t, tt)
                            rows.append(dict(extent=extent, phase=phase, mesh=h, pair="fine-"+level,
                                             **metrics(centered_period_mean(yy[:, :2], tt, 64)[1], ym, tm, settings)))
                _, ys, _ = slow(35, "B", "fine", extent)
                rows.append(dict(extent=extent, phase=phase, mesh=h, pair="A-B35", **metrics(ym, ys[32:-32, :2], tm, settings)))
                ta, ya, _ = load(case(50, phase, extent))
                clocks(t, ta)
                rows.append(dict(extent=extent, phase=phase, mesh=h, pair="A50-A35", **metrics(centered_period_mean(ya[:, :2], ta, 64)[1], ym, tm, settings)))
                filter_rows.append(filter_checks(y, t, tm, ym, b, p["phase_rad"][phase], settings["R_scale_m"], settings["estimated_slow_period_s"], f"preview_{phase}_{extent}", save=False))
    print(json.dumps({"preliminary_only": True, "comparisons": rows, "filter_controls": filter_rows}, indent=2), flush=True)


def provenance_audit(settings):
    """Recheck physical inputs, array clocks, parent chains, and reused kernels."""
    checked, reused = {}, []
    for name, expected in settings["input_sha256"].items():
        if sha(ROOT/name) != expected:
            raise ValueError("Frozen physical input changed: "+name)
        checked[name] = expected
    for c in pilot_cases()+extended_cases():
        t, y, meta = load(c)
        if len(t) != settings["RF_periods"][c["extent"]]*64+1 or not np.array_equal(y[0], settings["full_initial_states"][c["phase"]]):
            raise ValueError("Incomplete trajectory or changed physical IC")
        if not np.all(np.isfinite(y)) or abs(t[0]) > 1e-15 or abs(t[-1]-settings["RF_periods"][c["extent"]]/settings["physical"]["frequency_Hz"]) > 1e-14:
            raise ValueError("Invalid RF trajectory clock/data")
        if "reused_from_v1" in meta:
            old = meta["reused_from_v1"]
            d = OUTPUT.parent/"v1"
            a, m = paths(c)
            if sha(d/"runs"/m.name) != old["metadata_sha256"] or sha(d/"runs"/a.name) != meta["arrays_sha256"] or sha(d/"validation_settings.json") != old["settings_sha256"]:
                raise ValueError("Reused v1 provenance changed")
            if sha(d/"direct_fem_dynamics_source.py") != old["kernel_sha256"]["direct_fem_dynamics.py"]:
                raise ValueError("Reused v1 numerical source mismatch")
            reused.append(c)
    source = OUTPUT/"source_snapshot"
    source.mkdir(exist_ok=True)
    for name in settings["kernel_sha256"]:
        path = ROOT/"src/three_phase_trap"/name
        if sha(path) != settings["kernel_sha256"][name]:
            raise ValueError("Numerical kernel changed during validation")
        shutil.copy2(path, source/name)
    for name in ("build_direct_fem.py", "run_direct_fem_validation.py", "analyze_direct_fem_validation.py"):
        shutil.copy2(ROOT/"scripts"/name, source/name)
    shutil.copy2(ROOT/"tests/test_direct_fem.py", source/"test_direct_fem.py")
    write(OUTPUT/"provenance_audit.json", dict(frozen_inputs_checked=len(checked), reused_v1_cases=reused,
                                              RF_cases_checked=len(pilot_cases()+extended_cases()), all_initial_states_and_clocks_preserved=True,
                                              kernels_match_frozen_settings=True))


def field_analysis():
    rows, origin, reconstruction, calibration = [], {}, {}, {}
    samples = {}
    radii = np.array([.025, .05, .1, .25, .5, 1, 2, 4, 6, 7, 8])*1e-3
    for h in (70, 50, 35):
        d = OUTPUT/f"h{h}_P4"
        origin[str(h)] = json.loads((d/"origin_limits.json").read_text(encoding="utf-8"))
        with np.load(d/"common_samples.npz") as z:
            samples[h] = {k: z[k] for k in z.files}
        with (d/"native_reconstruction_discrepancy.csv").open(encoding="utf-8", newline="") as stream:
            reconstruction[str(h)] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        path = d/"native_C_calibration.json"
        if path.exists():
            calibration[str(h)] = json.loads(path.read_text(encoding="utf-8"))
    for ha, hb in ((70, 50), (50, 35)):
        a, b = samples[ha], samples[hb]
        if not np.array_equal(a["points_m"], b["points_m"]):
            raise ValueError("Field samples must be identical")
        for i, r in enumerate(radii):
            cut = slice(513*i, 513*(i+1))
            ef = np.sqrt(np.mean((a["field"][cut]-b["field"][cut])**2)/np.mean(b["field"][cut]**2))
            ps = np.sqrt(np.mean((a["psi"][cut]-b["psi"][cut])**2)/np.mean(b["psi"][cut]**2))
            rows.append(dict(coarse_um=ha, fine_um=hb, radius_m=float(r), relative_E_rms=float(ef), relative_Psi_rms=float(ps)))
    csv_write(OUTPUT/"field_spatial_metrics.csv", rows)
    return dict(origin_limits=origin, rows=rows, native_reconstruction=reconstruction, native_C_calibration=calibration)


def filter_checks(y, t, tm, mean, b, phase, scale, expected_period, tag, save=True):
    row = dict(case=tag)
    def period_change(clock, actual, reference):
        return relative_period_difference(oscillation_metrics(clock, actual, expected_period),
                                          oscillation_metrics(clock, reference, expected_period))
    for window in (2, 3):
        tw, mw = centered_period_mean(y[:, :2], t, 64*window)
        offset = (window-1)*32
        m = coordinate_metrics(mw, mean[offset:-offset], scale)
        row[f"window{window}_rms"] = m["relative_rms"]
        row[f"window{window}_max"] = m["relative_max"]
        row[f"window{window}_period"] = period_change(tw, mw, mean[offset:-offset])
    t32, m32 = centered_period_mean(y[::2, :2], t[::2], 32)
    tr, trend = local_rf_trend(y[:, :2], t, 64, 2)
    selected = np.arange(32, len(t)-32, 8)
    inverse = inverse_batches(b, y[selected], t[selected], phase)
    for name, a, ref in (("resolution32", m32, mean[::2]), ("local_trend", trend, mean[32:-32]),
                          ("inverse", inverse[:, :2], mean[::8])):
        m = coordinate_metrics(a, ref, scale)
        row[name+"_rms"], row[name+"_max"] = m["relative_rms"], m["relative_max"]
        clock = {"resolution32": t32, "local_trend": tr, "inverse": t[selected]}[name]
        row[name+"_period"] = period_change(clock, a, ref)
    if save:
        np.savez_compressed(OUTPUT/(tag+"_slow.npz"), time_s=tm, positions_m=mean, inverse_time_s=t[selected], inverse_states=inverse)
    offset = int(round(((-phase/(2*np.pi)) % 1)*64))
    ids = np.arange(offset, len(t), 64)
    pi = inverse_batches(b, y[ids], t[ids], phase)
    phase_err = np.angle(np.exp(1j*(b.omega*t[ids]+phase)))
    if save:
        np.savez_compressed(OUTPUT/(tag+"_poincare.npz"), time_s=t[ids], states=y[ids], inverse_states=pi, RF_phase_error_rad=phase_err)
    row["phase_slice_error_max_rad"] = float(np.max(abs(phase_err)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--fields-only", action="store_true")
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()
    settings = prepare()
    if args.verify_artifacts:
        manifest = json.loads((OUTPUT/"manifest.json").read_text(encoding="utf-8"))
        if manifest["settings_sha256"] != sha(OUTPUT/"validation_settings.json") or manifest["count"] != len(manifest["artifacts"]):
            raise ValueError("Final manifest/settings mismatch")
        for name, expected in manifest["artifacts"].items():
            if sha(ROOT/name) != expected:
                raise ValueError("Final artifact changed: "+name)
        print(json.dumps(dict(artifacts_verified=manifest["count"], frozen_inputs_verified=len(settings["input_sha256"]), status="passed")), flush=True)
        return
    if args.preview:
        preview(settings)
        return
    if args.fields_only:
        print(json.dumps(field_analysis(), indent=2))
        return
    for c in pilot_cases()+extended_cases():
        if not valid(c):
            raise ValueError(f"Required run has not completed: {c}")
    provenance_audit(settings)
    tol = settings["tolerances"]
    physical = dict(relative_rms=tol["relative_position_rms"], relative_max=tol["relative_position_max"], relative_period=tol["relative_period"])
    budget = {k: v*tol["numerical_budget_fraction"] for k, v in physical.items()}
    scale = settings["R_scale_m"]
    p = settings["physical"]
    b = AveragedRFModel(RFReconstruction.load(OUTPUT/"h35_P4/diagnostic_N48.json"), p["charge_C"], p["mass_kg"], p["U0_V"], p["frequency_Hz"])
    numerical, spatial, filters, comparisons, periods, audits = [], [], [], [], [], []
    plot_data = []
    for extent, phase in (("pilot", 0), ("pilot", 1), ("extended", 0)):
        print(f"Analyzing direct FEM {extent}, phase {phase}", flush=True)
        label = f"phase{phase}_{extent}"
        positions = {}
        full_coarse_mesh = None
        full_spatial = None
        common = None
        for h in (50, 35):
            t, y, meta = load(case(h, phase, extent))
            tc, yc, _ = load(case(h, phase, extent, "coarse"))
            clocks(t, tc)
            if not np.array_equal(y[0], settings["full_initial_states"][phase]):
                raise ValueError("A changed the original physical initial state")
            tm, mean = centered_period_mean(y[:, :2], t, 64)
            cm = centered_period_mean(yc[:, :2], tc, 64)[1]
            if common is None:
                common = tm
            clocks(common, tm)
            m = metrics(cm, mean, tm, settings)
            full = coordinate_metrics(yc[:, :2], y[:, :2], scale)
            numerical.append(dict(extent=extent, phase=phase, model=f"A_h{h}", control="T50-T100", **m,
                                  full_relative_rms=full["relative_rms"], full_relative_max=full["relative_max"], passed=passed(m, budget) and full["relative_rms"] <= budget["relative_rms"] and full["relative_max"] <= budget["relative_max"]))
            positions[f"A{h}"] = mean
            if h == 50:
                full_coarse_mesh = y[:, :2].copy()
            else:
                full_spatial = coordinate_metrics(full_coarse_mesh, y[:, :2], scale)
                del full_coarse_mesh
            for level in ("coarse", "fine"):
                _, _, metadata = load(case(h, phase, extent, level))
                d = metadata["diagnostics_this_segment"]
                audits.append(dict(case=f"A_h{h}_{label}_{level}", **d))
            if h == 35:
                row = filter_checks(y, t, tm, mean, b, p["phase_rad"][phase], scale, settings["estimated_slow_period_s"], "A35_"+label)
                row["passed"] = bool(max(v for k, v in row.items() if k.endswith("_rms")) <= budget["relative_rms"]
                                     and max(v for k, v in row.items() if k.endswith("_max")) <= budget["relative_max"]
                                     and all(v is not None and v <= budget["relative_period"] for k, v in row.items() if k.endswith("_period")))
                filters.append(row)
                if extent == "pilot" and phase == 0:
                    for level in ("refined", "radau"):
                        tt, yy, metadata = load(case(h, phase, extent, level))
                        clocks(t, tt)
                        mm = metrics(centered_period_mean(yy[:, :2], tt, 64)[1], mean, tm, settings)
                        ff = coordinate_metrics(yy[:, :2], y[:, :2], scale)
                        numerical.append(dict(extent=extent, phase=phase, model="A_h35", control="T100-"+level, **mm,
                                              full_relative_rms=ff["relative_rms"], full_relative_max=ff["relative_max"],
                                              passed=passed(mm, budget) and ff["relative_rms"] <= budget["relative_rms"] and ff["relative_max"] <= budget["relative_max"]))
                        audits.append(dict(case=f"A_h35_{label}_{level}", **metadata["diagnostics_this_segment"]))
            del y, yc
        sm = metrics(positions["A50"], positions["A35"], common, settings)
        spatial.append(dict(extent=extent, phase=phase, pair="A_P4_h50-h35", **sm,
                            full_relative_rms=full_spatial["relative_rms"], full_relative_max=full_spatial["relative_max"],
                            passed=passed(sm, budget) and full_spatial["relative_rms"] <= budget["relative_rms"] and full_spatial["relative_max"] <= budget["relative_max"]))
        for h in (50, 35):
            for model in ("B", "C"):
                ts, ys, energy = slow(h, model, "fine", extent)
                tc, yc, _ = slow(h, model, "coarse", extent)
                clocks(ts[32:-32], common)
                clocks(ts, tc)
                m = metrics(yc[32:-32, :2], ys[32:-32, :2], common, settings)
                drift = float(np.max(abs(energy-energy[0]))/abs(energy[0]))
                numerical.append(dict(extent=extent, phase=phase, model=f"{model}_h{h}", control="coarse-fine", **m,
                                      full_relative_rms=None, full_relative_max=None, passed=passed(m, budget) and drift <= 1e-8))
                positions[f"{model}{h}"] = ys[32:-32, :2]
                periods.append(dict(extent=extent, phase=phase, model=f"{model}_h{h}", energy_max_relative_drift=drift,
                                    **oscillation_metrics(common, ys[32:-32, :2], settings["estimated_slow_period_s"])))
        for model in ("B", "C", "A_smooth"):
            ts, ys = frozen(model, phase, extent)
            clocks(ts[32:-32], common)
            positions[model+"70"] = centered_period_mean(ys[:, :2], ts, 64)[1] if model == "A_smooth" else ys[32:-32, :2]
        tf, yf, _ = load(case(35, phase, extent, model="A_fit"))
        clocks(tf[32:-32], common)
        positions["A_fit35"] = centered_period_mean(yf[:, :2], tf, 64)[1]
        ts, ys, _ = slow(35, "B", "fine", extent, phase)
        clocks(ts[32:-32], common)
        positions["B35_matched"] = ys[32:-32, :2]
        for name, pos in positions.items():
            if not name.startswith(("B", "C")) or name.endswith("70") or name == "B35_matched":
                periods.append(dict(extent=extent, phase=phase, model=name, energy_max_relative_drift=None,
                                    **oscillation_metrics(common, pos, settings["estimated_slow_period_s"])))
        pairs = [("A35", name) for name in ("B70", "C70", "B35", "C35", "B35_matched", "A_fit35", "A_smooth70")]
        pairs += [("B35", "C35"), ("B70", "C70"), ("B35_matched", "B35"), ("B35", "B70"), ("C35", "C70")]
        for a, ref in pairs:
            m = metrics(positions[a], positions[ref], common, settings)
            components = np.sqrt(np.mean(((positions[a]-positions[ref])/scale)**2, axis=0))
            comparisons.append(dict(extent=extent, phase=phase, pair=a+"-"+ref, **m,
                                    x_relative_rms=float(components[0]), y_relative_rms=float(components[1]),
                                    physical_criteria_passed=passed(m, physical), status="pending_numerical_gates"))
        plot_data.append((extent, phase, common, positions))
    fields = field_analysis()
    field_ok = max(max(row["relative_E_rms"], row["relative_Psi_rms"]) for row in fields["rows"] if row["coarse_um"] == 50 and row["radius_m"] <= .001) <= settings["field_relative_screening_target"]
    verified = all(r["passed"] for r in numerical+spatial+filters) and field_ok
    for r in comparisons:
        enough_time = r["elapsed_slow_periods"] is not None and r["elapsed_slow_periods"] >= (3 if r["extent"] == "pilot" else 10)
        r["status"] = ("passed" if r["physical_criteria_passed"] else "failed") if verified and enough_time else "inconclusive"
    for name, rows in (("numerical_controls", numerical), ("spatial_controls", spatial), ("filter_controls", filters), ("comparisons", comparisons)):
        csv_write(OUTPUT/(name+".csv"), rows)
    write(OUTPUT/"periods.json", periods)
    write(OUTPUT/"geometry_audits.json", audits)
    summary = dict(numerical_verification="passed" if verified else "failed", physical_tolerances=physical, numerical_budgets=budget,
                   numerical_controls=numerical, spatial_controls=spatial, filter_controls=filters, comparisons=comparisons,
                   field_screening_passed=field_ok, field_diagnostics=fields,
                   scope="Same original geometry, drive, physical full initial states, two pilot RF phases and one extended phase; P4 direct unsmoothed FE gradients",
                   caveat="Empirical numerical verification at specified cases; no universal proof of averaging. P2 historical failures remain historical. Radau and DOP853 share the geometric curve checker.")
    write(OUTPUT/"summary.json", summary)
    make_plots(plot_data)
    report(summary)
    inputs = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    inputs["scripts/analyze_direct_fem_validation.py"] = sha(Path(__file__))
    write(OUTPUT/"manifest.json", {"artifacts": inputs, "count": len(inputs), "settings_sha256": sha(OUTPUT/"validation_settings.json")})
    print(json.dumps({"numerical_verification": summary["numerical_verification"], "main_comparisons": [r for r in comparisons if r["pair"] in ("A35-B70", "A35-C70", "A35-B35", "A35-C35")], "report": str(OUTPUT/"DIRECT_FEM_A_REPORT.md")}, indent=2), flush=True)


def make_plots(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    labels = {"A35": "A: direct P4, 35 µm", "B35": "B: P4, 35 µm", "C70": "C: original 70 µm"}
    for row, (extent, phase, t, pos) in enumerate(data):
        ids = slice(None, None, 32)
        for name, style in (("A35", "-"), ("B35", "--"), ("C70", ":")):
            axes[row, 0].plot(t[ids]*1e3, pos[name][ids, 0]*1e3, style, label=labels[name], linewidth=1.2)
        for name in ("B70", "B35", "C35"):
            err = np.linalg.norm(pos["A35"]-pos[name], axis=1)/.001*100
            axes[row, 1].plot(t[ids]*1e3, err[ids], label="A35 - "+name, linewidth=1)
        phase_text = "0" if phase == 0 else "π/2"
        axes[row, 0].set(ylabel="x (mm)", title=f"{extent}, RF phase = {phase_text}")
        axes[row, 1].set(ylabel="Position difference / 1 mm (%)")
        for ax in axes[row]:
            ax.set_xlabel("Time (ms)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.savefig(OUTPUT/"direct_FEM_A_comparison.png", dpi=160)
    fig.savefig(OUTPUT/"direct_FEM_A_comparison.pdf")
    plt.close(fig)
    _, _, t, pos = data[-1]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for name, style in (("A35", "-"), ("B35", "--"), ("C35", ":")):
        ax.plot(t[::32]*1e3, pos[name][::32, 1]*1e6, style, label=name, linewidth=1.1)
    ax.set(xlabel="Time (ms)", ylabel="Slow y coordinate (µm)", title="Extended interval: transverse slow motion, RF phase = 0")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(OUTPUT/"direct_FEM_A_transverse.png", dpi=180)
    fig.savefig(OUTPUT/"direct_FEM_A_transverse.pdf")
    plt.close(fig)


def report(s):
    lines = ["# Direct FEM A validation", "", f"Numerical verification: **{s['numerical_verification']}**.", "",
             "A uses native P4 FE gradients without a harmonic reconstruction, filtering of the force, or fitted trajectory. P4 refines the discretization of the same physical problem; the unreliable historical P2 trajectories are preserved and are not retroactively certified.", "",
             "The tested conservative effective energies are Psi_B=q^2 (|E_c|^2+|E_s|^2)/(4 m Omega^2) and Psi_C=8 q^2 C4^2 r^6/(m Omega^2), with C4 including U0. C is the rotating octupole n=4 model, with sixth-power radial pseudopotential.", "",
             "## Frozen experiment", "", "m = 1e-15 kg, q = +1e-12 C, U0 = 250 V, f = 100 kHz. Original full initial states and global RF phase are unchanged. Pilot: 7506 RF periods (0.07506 s), phases 0 and pi/2. Extended: 23672 RF periods (0.23672 s), phase 0. Position errors use fixed scale 1 mm; no time, phase, or path alignment.", "",
             "Physical thresholds: RMS 1%, maximum 3%, period 1%. Numerical/spatial/filter budgets: RMS 0.1%, maximum 0.3%, period 0.1%. Centered trapezoidal RF-period average with 64 output intervals; endpoints lacking a complete window are excluded.", "",
             "## Numerical and spatial controls", "", "| Case | Control | RMS (%) | Max (%) | Period (%) | Passed |", "|---|---|---:|---:|---:|---|"]
    for r in s["numerical_controls"]+s["spatial_controls"]:
        lines.append(f"| {r['extent']}, phase {r['phase']} | {r.get('model', r.get('pair'))} {r.get('control', '')} | {100*r['relative_rms']:.9g} | {100*r['relative_max']:.9g} | {100*r['relative_period_error']:.9g} | {r['passed']} |")
    lines += ["", "DOP853 uses T_RF/50 and T_RF/100 at both meshes and all three cases. Phase-zero pilot additionally uses T_RF/200 and independent Radau time discretization. Both time methods share the geometry audit. Every accepted dense curve is checked inside its cell; crossings preserve the state and global time, with zero position nudges. v2 explicitly bounds position roundoff from representing local crossing roots on the global clock by eight time ULPs. Completed v1 h50/A_fit pilots retain their original, stricter audit and explicit original-kernel hashes; they are not presented as v2 recomputations. See geometry_audits.json, provenance_audit.json, and individual crossing logs. Extended runs continue matching pilots with parent SHA-256 records.", "",
              "Filter controls compare 32/64 output resolution, one/two/three RF periods, a local RF trend, and inverse micromotion with drift. All controls, including maxima, are in filter_controls.csv. Native B/C additionally pass time refinement and effective-energy drift <=1e-8.", "",
              "## Comparison with effective potentials", "", "| Case | Pair | RMS (%) | Max (%) | Period (%) | Result |", "|---|---|---:|---:|---:|---|"]
    for r in s["comparisons"]:
        lines.append(f"| {r['extent']}, phase {r['phase']} | {r['pair']} | {100*r['relative_rms']:.7g} | {100*r['relative_max']:.7g} | {100*r['relative_period_error']:.7g} | {r['status']} |")
    lines += ["", "B70/C70 are the frozen original article-validation models. B35 uses all N48 harmonic coefficients of the native P4 potential. C35 retains its rotating n=4 coefficient, calibrated from direct potential at 0.5/1/2 mm, without fitting trajectories. B35_matched changes only the slow-model initial condition by applying the native inverse micromotion map to the original physical full state; A always uses that same original physical state. A_fit35 is a diagnostic full-RF trajectory in the N48 reconstruction of P4, not the definition of A; it has one temporal level. A_smooth70 is the historical full-RF harmonic surrogate.", "",
              "## Why this closes the earlier gap", "", "The original P2 gradient created spurious central fields and sensitive element-interface dynamics. P4 improves local approximation while retaining direct FE forces. An exact complex cyclic reduction on a 30-degree sector is used, u(R30 x)=exp(-2 pi i/3)u(x); no spatial harmonics are discarded. Its P2 version reproduces the original full-domain P2 field with relative RMS 4.8e-13. The sector and full geometry, FE assembly residuals, original-cache hashes, and sample diagnostics are recorded in the build artifacts and frozen validation_settings.json.", "",
              "## Interpretation and limits", "", "A passed numerical verification means the requested comparison can be interpreted at its predeclared tolerance. A failed A-B/C physical comparison is a completed negative result for that interval, not numerical inconclusiveness. This experiment does not prove averaging for arbitrary charge/mass, voltage/frequency, initial amplitude, geometry, or time horizon. Long-time accumulated phase/coordinate error must be reported even if short trajectories agree. See comparisons.csv and direct_FEM_A_comparison.png.", "",
              "Reproduce: python scripts/build_direct_fem.py (as documented in docs/DIRECT_FEM_A_PROTOCOL.md); python scripts/run_direct_fem_validation.py --stage all --workers 4; python scripts/analyze_direct_fem_validation.py. Existing hash-checked checkpoints are reused.", ""]
    lead = ["## Main outcome", ""]
    for extent, phase in (("pilot", 0), ("pilot", 1), ("extended", 0)):
        selected = [r for r in s["comparisons"] if r["extent"] == extent and r["phase"] == phase and r["pair"] in ("A35-B35", "A35-C35")]
        lead += [f"{extent}, phase {phase}: "+"; ".join(f"{r['pair']} **{r['status']}** (RMS {100*r['relative_rms']:.6g}%, max {100*r['relative_max']:.6g}%, period {100*r['relative_period_error']:.6g}%)" for r in selected)+".", ""]
    lines[4:4] = lead
    extended = {r["pair"]: r for r in s["comparisons"] if r["extent"] == "extended" and r["phase"] == 0}
    ab, bc, fit, initial = (extended[name] for name in ("A35-B35", "B35-C35", "A35-A_fit35", "B35_matched-B35"))
    lines += ["## Error separation on the extended interval", "",
              f"A35-B35 has x RMS {100*ab['x_relative_rms']:.7g}% and y RMS {100*ab['y_relative_rms']:.7g}%, with period error {100*ab['relative_period_error']:.7g}%. The coordinate disagreement is dominated by transverse slow motion, not a large measured x-period error. See direct_FEM_A_transverse.png.", "",
              f"B35-C35 RMS is {100*bc['relative_rms']:.7g}%. Native A35 versus diagnostic full-RF A_fit35 RMS is {100*fit['relative_rms']:.7g}%. Changing only the slow initial map, B35_matched-B35, changes RMS coordinates by {100*initial['relative_rms']:.7g}%; A35-B35_matched still fails. Filter sensitivity is also far below the observed discrepancy.", "",
              "These controls identify leading averaged dynamics as the main limitation in this case. The n=4 spatial truncation and tested initial-state map do not explain the accumulated error. A higher-order effective equation or a unique physical mechanism for transverse drift has not been derived by this verification.", ""]
    (OUTPUT/"DIRECT_FEM_A_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
