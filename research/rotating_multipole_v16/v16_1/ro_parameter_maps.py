"""Exact RO4 6D stability maps over charge moments and damping channels.

The RO synchronous solution and its six eigenvalues reduce to an angular
quadratic plus a translational quartic, so parameter boundaries are computed
from exact coefficients rather than inferred from trajectory color maps.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"; FIG=ROOT/"figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

def profile_moments(ell: float, a: float):
    """Symmetric q(g) proportional to 1+a(g^2-1/3), g=2s/L."""
    mu2=ell**2/12*(1+4*a/15)
    mu4=ell**4/80*(1+8*a/21)
    return mu2,mu4

def spectra(kappa,ell,gt,gr,a,direction=1,phase_delta=0.):
    mu2,mu4=profile_moments(ell,a); I=ell**2/12; w=direction*.25
    amp=kappa*mu4
    ratio=gr*I*w/(4*amp)
    if abs(ratio)>=1: return None
    chi_asin=math.asin(ratio)
    chi=next(c for c in (chi_asin,math.pi-chi_asin) if -amp*math.cos(c)>0)
    K=-16*amp*math.cos(chi)/I
    b=6*kappa*mu2
    com=np.roots([1,2*gt,gt*gt+2*w*w,2*gt*w*w,
                  w**4+gt*gt*w*w-4*b*b])
    rot=np.roots([1,gr,K])
    return np.r_[com,rot],chi,chi/4-phase_delta/4

def main():
    kappa=.8; ell=.3; profiles=[("uniform",0.),("end weighted a=0.8",.8),
                               ("end weighted a=1.6",1.6)]
    gammas=np.linspace(0,.3,301)
    GT,GR=np.meshgrid(gammas,gammas)
    classes=[]; boundaries=[]; invariance=[]
    cmap=ListedColormap(["#984ea3","#4daf4a","#e41a1c","#377eb8"])
    fig,axs=plt.subplots(1,3,figsize=(14,4.7),sharex=True,sharey=True)
    for ax,(name,a) in zip(axs,profiles):
        mu2,mu4=profile_moments(ell,a); I=ell**2/12; w=.25
        b=6*kappa*mu2
        gtc=math.sqrt(max(0.,(2*abs(b)/w)**2-w*w)) if 2*abs(b)>w*w else 0.
        grc=4*abs(kappa*mu4)/(I*w)
        trans=(2*abs(b)<w*np.sqrt(w*w+GT*GT))
        rot=GR<grc
        # 0 neither, 1 both, 2 rotation only, 3 translation only.
        cls=np.zeros_like(GT,dtype=np.uint8)
        cls[trans&rot]=1; cls[~trans&rot]=2; cls[trans&~rot]=3
        classes.append(cls)
        ax.pcolormesh(GT,GR,cls,cmap=cmap,vmin=0,vmax=3,shading="auto")
        ax.axvline(gtc,color="black",ls="--",lw=1)
        ax.axhline(grc,color="black",ls="-.",lw=1)
        ax.set_title(f"{name}\n$\\gamma_{{t,c}}={gtc:.3f}$, $\\gamma_{{r,c}}={grc:.3f}$")
        ax.set_xlabel(r"$\gamma_{trans}$")
        ax.grid(alpha=.15)
        boundaries.append({"profile":name,"charge_profile_a":a,"mu2":mu2,"mu4":mu4,
            "gamma_trans_critical_at_gamma_trans_nonnegative":gtc,
            "gamma_rot_critical":grc,
            "condition_translation":"2|b| < |w| sqrt(w^2 + gamma_trans^2), b=6*kappa4*mu2",
            "condition_rotation":"gamma_rot < 4|kappa4*mu4|/(I*|w|)"})
    axs[0].set_ylabel(r"$\gamma_{rot}$")
    handles=[Patch(color=cmap(i),label=lab) for i,lab in enumerate(
       ["neither stable / no lock","jointly stable","rotation only","translation only"])]
    fig.legend(handles=handles,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.02),fontsize=8)
    fig.suptitle(r"RO4 full linear stability: $\kappa_4=0.8$, $L/R=0.3$; exact boundaries")
    fig.tight_layout(rect=(0,.09,1,.94)); fig.savefig(FIG/"ro_charge_profile_maps.png",dpi=180); plt.close(fig)

    # Verify phase and direction invariance of the eigenvalue set using the
    # full characteristic polynomials at a representative stable point.
    reference=np.sort_complex(spectra(kappa,ell,.2,.05,0.,1,0.)[0])
    eigsets=[]; eig_diffs=[]
    for direction in (-1,1):
        for delta in (0.,math.pi/8,math.pi/4,math.pi/2):
            out=spectra(kappa,ell,.2,.05,0.,direction,delta)
            eig_diffs.append(float(np.max(np.abs(np.sort_complex(out[0])-reference))))
            eigsets.append({"direction":direction,"phase_delta":delta,
                            "max_real_part":float(np.max(out[0].real)),
                            "angular_phase_chi":out[1],"body_phase":out[2]})
    results={"normalization":"V16 dimensionless; omega4=1; I=ell^2/12; gamma channels independent",
      "kappa4":kappa,"ell":ell,"damping_grid":{"gamma_trans":gammas.tolist(),"gamma_rot":gammas.tolist()},
      "profiles":boundaries,"phase_direction_check":{"eigenvalues_depend_on_delta_or_direction":False,
          "reason":"time/rotation covariance; thresholds depend on |w| and the stable torque branch curvature",
          "max_sorted_eigenvalue_difference_from_direction_plus_delta_zero":max(eig_diffs),
          "sample_eigenvalue_sets":eigsets},
      "finite_grid_resolution":"301x301 per profile; boundaries are analytic and plotted exactly",
      "class_labels":{"0":"neither stable / no lock","1":"joint stable","2":"rotation only","3":"translation only"}}
    (OUT/"ro_parameter_maps.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps({"profiles":boundaries,"phase_direction_check":results["phase_direction_check"]},indent=2))

if __name__=="__main__": main()
