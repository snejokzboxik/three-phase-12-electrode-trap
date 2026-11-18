from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "candidate_confirmations.json"
OUT = ROOT / "figures" / "candidate_poincare_sections.png"


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = [r for r in payload["confirmations"]
            if r.get("high_accuracy_status") == "BOUNDED_1000_RF" and r.get("poincare_csv")]
    rows = sorted(rows, key=lambda r: r["largest_lyapunov_100rf"]["100rf"])
    selected = [rows[0], rows[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.4), layout="constrained")
    for col, row in enumerate(selected):
        path = ROOT / row["poincare_csv"]
        df = pd.read_csv(path)
        ax = axes[0, col]
        ax.scatter(df.x, df.y, s=3, alpha=.38, rasterized=True)
        ax.set(xlabel="x", ylabel="y", title=f"{row['model']}; κ={row['kappa']:+g}; dir={row.get('direction')} — position")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
        ax = axes[1, col]
        ax.scatter(df.x, df.vx, s=3, alpha=.38, rasterized=True)
        ax.set(xlabel="x", ylabel="vx", title="stroboscopic projection")
        ax.grid(alpha=.2)
    fig.suptitle("Two selected survivors: 1000 RF strobe samples; finite horizon only")
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(OUT), "selected": [
        {"model": r["model"], "kappa": r["kappa"], "direction": r.get("direction"),
         "lambda_100rf": r["largest_lyapunov_100rf"]["100rf"], "poincare_csv": r["poincare_csv"]}
        for r in selected]}, indent=2))


if __name__ == "__main__":
    main()
