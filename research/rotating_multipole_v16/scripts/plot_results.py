from __future__ import annotations

import json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results";FIG=ROOT/"figures";FIG.mkdir(exist_ok=True)
rows=[]
for line in (OUT/"search_results.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():rows.append(json.loads(line))

def fraction(x):return x["bounded_to_horizon"]/x["n"]

def main():
    cons=[x for x in rows if x.get("stage") in {"conservative","conservative_cold","conservative_replicate"}]
    fig,axs=plt.subplots(1,3,figsize=(11.2,3.7),sharey=True,layout="constrained")
    panels=[("RO dir +1",[x for x in cons if x["model"]=="RO" and x["direction"]==1]),
            ("RO dir −1",[x for x in cons if x["model"]=="RO" and x["direction"]==-1]),
            ("SO",[x for x in cons if x["model"]=="SO"])]
    cons_style={"conservative":("o","#1d4ed8","broad box, H=100 RF"),
                "conservative_cold":("s","#dc2626","quartic-scaled, H=200 RF"),
                "conservative_replicate":("D","#7c3aed","independent scrambles, H=300 RF")}
    for ax,(title,data) in zip(axs,panels):
        for stage,(marker,color,label) in cons_style.items():
            d=[x for x in data if x.get("stage")==stage]
            if d:ax.scatter([x["kappa"] for x in d],[fraction(x) for x in d],
                s=30,marker=marker,
                facecolors="none" if stage=="conservative" else color,
                edgecolors=color,label=f"{label}, n=64")
        ax.axvspan(-.16150095,.16150095,color="#16a34a",alpha=.07)
        ax.axhline(0,color="black",lw=.6);ax.set_title(title);ax.set_xlabel(r"signed $\kappa$")
        ax.grid(alpha=.2);ax.set_ylim(-.03,1.03)
    axs[0].set_ylabel("fraction surviving stated finite horizon\n(not a stability probability)")
    axs[-1].legend(frameon=False,fontsize=7,loc="upper right")
    fig.suptitle("Conservative ensemble screen; discrete parameter points only")
    fig.savefig(FIG/"conservative_regime_map.png",dpi=180);plt.close(fig)

    diss=[x for x in rows if x.get("stage") in {"dissipative","dissipative_paired"}]
    if diss:
        fig,axs=plt.subplots(1,2,figsize=(9.6,3.7),sharey=True,layout="constrained")
        norm=Normalize(vmin=0,vmax=1)
        for ax,model in zip(axs,["RO","SO"]):
            d=[x for x in diss if x["model"]==model]
            for stage,marker in [("dissipative","o"),("dissipative_paired","D")]:
                q=[x for x in d if x["stage"]==stage]
                if q:
                    ax.scatter([x["kappa"] for x in q],[x["gamma_s-1"] for x in q],
                        c=[fraction(x) for x in q],norm=norm,cmap="viridis",marker=marker,
                        s=48,edgecolors="black",linewidths=.3)
            ax.axvspan(-.16150095,.16150095,color="#16a34a",alpha=.07)
            ax.set_yscale("symlog",linthresh=10);ax.set_xlabel(r"signed $\kappa$");ax.set_title(model)
            ax.grid(alpha=.2)
        fig.colorbar(ScalarMappable(norm=norm,cmap="viridis"),ax=axs,label="bounded fraction")
        handles=[Line2D([0],[0],marker="o",color="black",linestyle="none",label="independent screen"),
                 Line2D([0],[0],marker="D",color="black",linestyle="none",label="paired same-start screen")]
        axs[-1].legend(handles=handles,frameon=False,fontsize=7,loc="upper right")
        axs[0].set_ylabel(r"drag $\gamma$ (s$^{-1}$)")
        fig.suptitle("Dissipative screen: H=100 RF at γ=0, 300 RF for 30–300, 500 RF for 1000–10000; finite survival only")
        fig.savefig(FIG/"dissipative_regime_map.png",dpi=180);plt.close(fig)

    print(f"wrote plots from {len(rows)} result rows to {FIG}")

if __name__=="__main__":main()
