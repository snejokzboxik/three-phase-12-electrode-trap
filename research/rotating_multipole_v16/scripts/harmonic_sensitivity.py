from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v16.models import Field
from v16.integrate import integrate_ensemble
from run_search import sobol_initials

OUT = ROOT / "results"
FIG = ROOT / "figures"
DATA = OUT / "harmonic_sensitivity.jsonl"
PROGRESS = OUT / "harmonic_sensitivity_progress.json"
SUMMARY = OUT / "harmonic_sensitivity_summary.json"
KAPPA = .1044
HORIZON = 300
STEPS = 96
N = 64
PHASES = {8:(0.0, math.pi/4, math.pi/2, 3*math.pi/4),
          10:(0.0, math.pi/4, math.pi/2, 3*math.pi/4)}
EPSILONS = {8:(.03, .1, .3), 10:(.1, .3)}
SEEDS = (6128, 16128)


def canonical_hash(row):
    clean = {k: v for k, v in row.items() if k != "sha256"}
    packed = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def load_rows():
    if not DATA.exists():
        return []
    rows = []
    for number, line in enumerate(DATA.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("sha256") != canonical_hash(row):
            raise RuntimeError(f"checksum mismatch in {DATA.name}, line {number}")
        rows.append(row)
    return rows


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def task_id(seed, order, epsilon, delta):
    return f"RO-k4={KAPPA:g}:m={order}:eps={epsilon:g}:delta={delta:.8f}:seed={seed}"


def build_tasks():
    tasks = []
    for seed in SEEDS:
        tasks.append({"seed": seed, "order": 0, "epsilon": 0.0, "delta": 0.0})
        for order in (8,10):
            for epsilon in EPSILONS[order]:
                for delta in PHASES[order]:
                    tasks.append({"seed": seed, "order": order,
                                  "epsilon": epsilon, "delta": delta})
    return tasks


def execute(task, baseline_mask):
    seed = task["seed"]
    order = task["order"]
    epsilon = task["epsilon"]
    delta = task["delta"]
    init, phases = sobol_initials(N, seed, KAPPA, "quartic_scaled")
    field = Field("RO", kappa=KAPPA, omega_ro=1.0,
                  kappa8=epsilon*KAPPA if order==8 else 0.0, omega8=2.0,
                  delta8=delta if order==8 else 0.0,
                  kappa10=epsilon*KAPPA if order==10 else 0.0, omega10=2.5,
                  delta10=delta if order==10 else 0.0)
    result = integrate_ensemble(init, field, rf_count=HORIZON,
                                steps_per_rf=STEPS, initial_phase=phases)
    survivor_mask = (~np.isfinite(result["exit_tau"])).tolist()
    bounded = int(sum(survivor_mask))
    base_fraction = None
    lost = gained = both = None
    if baseline_mask is not None:
        base = np.asarray(baseline_mask, dtype=bool)
        now = np.asarray(survivor_mask, dtype=bool)
        base_fraction = float(base.mean())
        lost = int(np.count_nonzero(base & ~now))
        gained = int(np.count_nonzero(~base & now))
        both = int(np.count_nonzero(base & now))
    return {
        **task,
        "task_id": task_id(seed, order, epsilon, delta),
        "model": "RO + co-rotating higher Laplace harmonic",
        "kappa4": KAPPA,
        "multipole_order": order,
        "coefficient_ratio": epsilon,
        "relative_phase_rad": delta,
        "harmonic_frequency_over_tau": (2.0 if order==8 else 2.5) if order else None,
        "seed": seed,
        "n": N,
        "initial_velocity_model": "quartic_scaled",
        "horizon_rf": HORIZON,
        "steps_per_rf": STEPS,
        "bounded": bounded,
        "bounded_fraction": bounded/N,
        "median_exit_rf": float(np.nanmedian(result["exit_tau"])/(2*math.pi))
                          if np.any(np.isfinite(result["exit_tau"])) else None,
        "baseline_fraction_same_seed": base_fraction,
        "lost_from_baseline": lost,
        "gained_vs_baseline": gained,
        "survived_both": both,
        "survivor_mask": survivor_mask,
        "max_radius": float(np.max(result["max_radius"])),
        "interpretation": "matched finite-time sensitivity screen; harmonic amplitudes and phases are assumed, not fitted to FEM",
    }


def write_plot(rows):
    FIG.mkdir(exist_ok=True)
    fig, (a8, a10, force_ax) = plt.subplots(1, 3, figsize=(13.8, 4.6), layout="constrained")
    colors = {0.0: "#2563eb", math.pi/4: "#16a34a",
              math.pi/2: "#dc2626", 3*math.pi/4: "#7c3aed"}
    for ax, order in ((a8,8),(a10,10)):
        for seed in SEEDS:
            subset = [r for r in rows if r["seed"] == seed]
            baseline = next((r for r in subset if r["order"] == 0), None)
            if baseline is None:
                continue
            for delta in PHASES[order]:
                pts = sorted((r for r in subset if r["order"]==order and r["epsilon"]>0
                              and r["delta"]==delta),key=lambda x:x["epsilon"])
                xs = [0.0]+[r["epsilon"] for r in pts]
                ys = [baseline["bounded_fraction"]]+[r["bounded_fraction"] for r in pts]
                ax.plot(xs,ys,marker="o",ms=4,lw=1.1,color=colors[delta],
                        alpha=.6 if seed==SEEDS[0] else 1.0,
                        linestyle="-" if seed==SEEDS[0] else "--",
                        label=fr"$\delta_{order}={delta/math.pi:.2g}\pi$, seed {seed}")
        ax.set(xlabel=fr"assumed $|\kappa_{order}/\kappa_4|$",
               ylabel="fraction bounded at 300 RF",title=fr"Order $m={order}$; matched starts")
        ax.set_ylim(-.03,1.03);ax.grid(alpha=.25)
        ax.legend(frameon=False,fontsize=6.2,ncol=2)

    rr=np.linspace(0,1,300)
    for order,epsilons,color,linestyle in ((8,EPSILONS[8],"#2563eb","-"),
                                            (10,EPSILONS[10],"#dc2626","--")):
        for epsilon in epsilons:
            force_ratio=(order/4)*epsilon*rr**(order-4)
            force_ax.plot(rr,force_ratio,color=color,linestyle=linestyle,lw=1.8,
                          label=fr"$m={order}, |\kappa_m/\kappa_4|={epsilon:g}$")
    force_ax.set(xlabel=r"radius $r/R$",ylabel=r"$|\nabla V_m|/|\nabla V_4|$",
                 title="Relative force rises toward aperture")
    force_ax.set_xlim(0,1);force_ax.set_ylim(0,.8);force_ax.grid(alpha=.25)
    force_ax.legend(frameon=False,fontsize=7)
    fig.suptitle(rf"RO harmonic sensitivity: $\kappa_4={KAPPA}$, 300 RF, N={N}; $f=100$ kHz, R=8 mm")
    fig.savefig(FIG / "harmonic_sensitivity.png", dpi=180)
    plt.close(fig)


def main():
    rows = load_rows()
    completed = {r["task_id"] for r in rows}
    masks = {(r["seed"]): r["survivor_mask"] for r in rows if r["epsilon"] == 0}
    tasks = build_tasks()
    atomic_json(PROGRESS, {"status": "RUNNING", "completed": len(completed),
                           "planned": len(tasks), "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    for task in tasks:
        ident = task_id(task["seed"], task["order"], task["epsilon"], task["delta"])
        if ident in completed:
            continue
        baseline = masks.get(task["seed"])
        row = execute(task, baseline)
        row["sha256"] = canonical_hash(row)
        with DATA.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n")
            f.flush(); os.fsync(f.fileno())
        rows.append(row); completed.add(ident)
        if task["order"] == 0:
            masks[task["seed"]] = row["survivor_mask"]
        atomic_json(PROGRESS, {"status": "RUNNING", "completed": len(completed),
                               "planned": len(tasks), "active_task": ident,
                               "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    payload = {
        "status": "COMPLETED",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "cases": len(rows), "seeds": list(SEEDS),
        "epsilon_values": {str(k):list(v) for k,v in EPSILONS.items()},
        "phase_values_rad": {str(k):list(v) for k,v in PHASES.items()}, "n": N, "horizon_rf": HORIZON,
        "steps_per_rf": STEPS,
        "field_model": "V_m=kappa4*Re[z^4 exp(-i*tau)]+kappa_m*Re[z^m exp(-i*(m*tau/4-delta_m))], m=8 or 10",
        "force_ratio": "|grad V_m|/|grad V_4|=(m/4)*|kappa_m/kappa4|*(r/R)^(m-4)",
        "warning": "Order-8 and order-10 amplitudes/phases are assumed perturbations, not measured or FEM-fitted values.",
        "files": ["harmonic_sensitivity.jsonl", "harmonic_sensitivity_summary.json", "../figures/harmonic_sensitivity.png"],
    }
    atomic_json(SUMMARY, payload)
    atomic_json(PROGRESS, {"status": "COMPLETED", "completed": len(completed),
                           "planned": len(tasks), "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    write_plot(rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
