from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SOURCE = OUT / "candidate_confirmations.json"
RESULTS = OUT / "lyapunov_extensions.json"
CHECKPOINT = OUT / "lyapunov_extensions_progress.json"
T = 2 * math.pi
APERTURE = 1.0
TARGET_RF = 1000
BLOCK_RF = 25

import sys
sys.path.insert(0, str(ROOT / "src"))
from v16.models import Field, field_rhs


def atomic_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def make_field(row):
    if row["model"] == "RO":
        return Field("RO", kappa=row["kappa"], omega_ro=row["direction"],
                     delta_ro=-row["direction"] * row["initial_rf_phase"])
    return Field("SO", kappa_so=row["kappa"],
                 delta_so=-row["initial_rf_phase"])


def augment_rhs(field):
    def rhs(t, z):
        y = z[:4]
        mat = z[4:].reshape(4, 4)
        f = field_rhs(t, y, field, 0.0)
        _, _, _, hxx, hxy, hyy = field.evaluate(y[0], y[1], t, hessian=True)
        jac = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                        [-hxx, -hxy, 0, 0], [-hxy, -hyy, 0, 0]], dtype=float)
        return np.r_[f, (jac @ mat).ravel()]
    return rhs


def init_record(row):
    return {
        "source_task_id": row["source_task_id"], "model": row["model"],
        "kappa": row["kappa"], "direction": row.get("direction"),
        "seed": row["seed"], "initial_index": row["initial_index"],
        "initial_state": row["initial_state"],
        "initial_rf_phase": row["initial_rf_phase"], "target_rf": TARGET_RF,
        "completed_rf": 0, "state": row["initial_state"],
        "Q": np.eye(4).tolist(), "log_growth": np.zeros(4).tolist(),
        "cumulative": [], "status": "RUNNING",
        "reference_100rf": row.get("largest_lyapunov_100rf", {}).get("100rf"),
    }


def write_plot(records, filename="lyapunov_extension.png"):
    target = ROOT / "figures" / filename
    target.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), layout="constrained")
    for record in records:
        values = np.asarray(record.get("cumulative", []), dtype=float)
        if not len(values):
            continue
        label = (f"{record['model']} κ={record['kappa']:+g}, "
                 f"dir={record.get('direction')}, λ100={record['reference_100rf']:.4g}")
        ax.plot(np.arange(1, len(values)+1), values, lw=1.1, label=label)
        ax.scatter([min(100, len(values))], [values[min(100, len(values))-1]], s=23)
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="RF periods accumulated", ylabel=r"largest finite-time QR exponent per $\tau$",
           title="Selected survivors: finite-time exponent evolution (not a chaos test)")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def extend(record):
    field = make_field(record)
    rhs = augment_rhs(field)
    state = np.asarray(record["state"], dtype=float)
    q = np.asarray(record["Q"], dtype=float)
    logs = np.asarray(record["log_growth"], dtype=float)
    cumulative = list(record["cumulative"])
    completed = int(record["completed_rf"])

    def exit_event(t, z):
        return APERTURE * APERTURE - z[0] * z[0] - z[1] * z[1]
    exit_event.terminal = True
    exit_event.direction = -1

    started = time.perf_counter()
    while completed < TARGET_RF:
        end_block = min(TARGET_RF, ((completed // BLOCK_RF) + 1) * BLOCK_RF)
        while completed < end_block:
            z0 = np.r_[state, q.ravel()]
            sol = solve_ivp(rhs, (completed * T, (completed + 1) * T), z0,
                            method="DOP853", rtol=5e-10, atol=5e-12,
                            max_step=T / 60, events=exit_event)
            z = sol.y[:, -1]
            state = z[:4]
            if sol.status == 1:
                record["status"] = "EXIT"
                record["exit_tau"] = float(sol.t[-1])
                record["state"] = state.tolist()
                record["completed_rf"] = completed
                record["Q"] = q.tolist()
                record["log_growth"] = logs.tolist()
                record["cumulative"] = cumulative
                record["elapsed_s"] = record.get("elapsed_s", 0.0) + time.perf_counter() - started
                return record
            q, r = np.linalg.qr(z[4:].reshape(4, 4))
            logs += np.log(np.maximum(np.abs(np.diag(r)), 1e-300))
            completed += 1
            cumulative.append(float(np.max(logs / (T * completed))))

        record.update({"completed_rf": completed, "state": state.tolist(),
                       "Q": q.tolist(), "log_growth": logs.tolist(),
                       "cumulative": cumulative,
                       "elapsed_s": record.get("elapsed_s", 0.0) + time.perf_counter() - started})
        record["current_exponent"] = float(np.max(logs / (T * completed)))
        atomic_json(CHECKPOINT, {"status": "RUNNING", "updated_local": time.strftime('%Y-%m-%d %H:%M:%S %z'),
                                 "records": [record]})
        started = time.perf_counter()

    record["status"] = f"BOUNDED_{TARGET_RF}_RF"
    record["spectrum_per_tau"] = (logs / (T * completed)).tolist()
    record["largest_lyapunov_per_tau"] = float(np.max(logs / (T * completed)))
    if TARGET_RF >= 100:
        record["largest_lyapunov_100rf_recomputed"] = float(cumulative[99])
    if TARGET_RF >= 1000:
        record["largest_lyapunov_1000rf_recomputed"] = float(cumulative[999])
    record[f"largest_lyapunov_{TARGET_RF}rf"] = record["largest_lyapunov_per_tau"]
    record["interpretation"] = "finite-time QR exponent; not a chaos proof"
    return record


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    candidates = [x for x in payload["confirmations"]
                  if x.get("high_accuracy_status") == "BOUNDED_1000_RF"
                  and x.get("largest_lyapunov_100rf")]
    if len(candidates) < 2:
        raise RuntimeError("Need at least two 1000-RF candidates with QR estimates")
    ranked = sorted(candidates, key=lambda x: x["largest_lyapunov_100rf"]["100rf"])
    chosen = [ranked[0]]
    if ranked[-1] is not ranked[0]:
        chosen.append(ranked[-1])
    existing = {}
    if RESULTS.exists():
        existing = {x["source_task_id"]: x for x in json.loads(RESULTS.read_text(encoding="utf-8")).get("records", [])}
    checkpoint = {}
    if CHECKPOINT.exists():
        try:
            checkpoint = {x["source_task_id"]: x for x in json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("records", [])}
        except (OSError, json.JSONDecodeError):
            checkpoint = {}
    records = []
    for row in chosen:
        key = row["source_task_id"]
        current = existing.get(key)
        if current and current.get("status") == f"BOUNDED_{TARGET_RF}_RF" or (current and current.get("status") == "EXIT"):
            records.append(current)
            continue
        current = checkpoint.get(key) or init_record(row)
        result = extend(current)
        records.append(result)
        atomic_json(CHECKPOINT, {"status": "RUNNING" if result["status"] == "RUNNING" else "UPDATED",
                                 "updated_local": time.strftime('%Y-%m-%d %H:%M:%S %z'),
                                 "records": [result]})
        atomic_json(RESULTS, {"status": "RUNNING" if result["status"] == "RUNNING" else "PARTIAL",
                              "target_rf": TARGET_RF, "selected_source_extremes": [
                                  {"task_id": x["source_task_id"], "lle_100rf": x["largest_lyapunov_100rf"]["100rf"]}
                                  for x in chosen], "records": records})
    atomic_json(RESULTS, {"status": "COMPLETED", "target_rf": TARGET_RF,
                          "selected_source_extremes": [
                              {"task_id": x["source_task_id"], "lle_100rf": x["largest_lyapunov_100rf"]["100rf"]}
                              for x in chosen], "records": records})
    write_plot(records)
    atomic_json(CHECKPOINT, {"status": "COMPLETED", "target_rf": TARGET_RF,
                             "updated_local": time.strftime('%Y-%m-%d %H:%M:%S %z'),
                             "records": records})
    print(json.dumps({"status": "COMPLETED", "records": [
        {"task_id": x["source_task_id"], "status": x["status"],
         "completed_rf": x["completed_rf"],
         "lle_100rf_reference": x["reference_100rf"],
         "lle_1000rf": x.get("largest_lyapunov_per_tau")} for x in records]}, indent=2))


if __name__ == "__main__":
    main()
