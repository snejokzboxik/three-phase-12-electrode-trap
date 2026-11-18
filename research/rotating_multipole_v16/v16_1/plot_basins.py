"""Plot basin capture fractions and 95% Wilson intervals from saved ensembles."""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/"results"/"basin_rotational.json").read_text(encoding="utf-8"))

def wilson(k,n,z=1.959963984540054):
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

def main():
    ang={"RO +1/4":40,"SO +1/4":19,"SO -1/4":16,"SO total":35}
    full={"RO +1/4":2,"SO +1/4":1,"SO -1/4":4,"SO total":5}
    n_ang=512; n_full=128
    fig,axs=plt.subplots(1,2,figsize=(10.2,4.8),sharey=True)
    colors=["#377eb8","#4daf4a","#e41a1c","#984ea3"]
    for ax,counts,n,title in [(axs[0],ang,n_ang,"Angular projection (n=512/field)"),
                              (axs[1],full,n_full,"Full 6D (n=128/field)")]:
        vals=[]; lower=[]; upper=[]
        for key,k in counts.items():
            lo,hi=wilson(k,n); vals.append(k/n); lower.append(k/n-lo); upper.append(hi-k/n)
        x=np.arange(len(vals))
        ax.bar(x,vals,color=colors,width=.68)
        ax.errorbar(x,vals,yerr=np.array([lower,upper]),fmt="none",ecolor="black",capsize=4,lw=1.1)
        ax.set_xticks(x,list(counts),rotation=25,ha="right")
        ax.set_title(title); ax.set_ylim(0,.18); ax.grid(axis="y",alpha=.25)
        for xx,v,k in zip(x,vals,counts.values()): ax.text(xx,v+.004,f"{k}/{n}",ha="center",fontsize=8)
    axs[0].set_ylabel("captured fraction; 95% Wilson interval")
    fig.suptitle("RO and standing-octupole rotational capture basins")
    fig.tight_layout(); fig.savefig(ROOT/"figures"/"basin_capture_fractions.png",dpi=180); plt.close(fig)

if __name__=="__main__": main()
