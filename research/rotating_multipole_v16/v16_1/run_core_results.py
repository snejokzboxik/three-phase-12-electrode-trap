"""Reproduce the V16.1 analytic-control spectra and core figures."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from v16_1.dynamics import (Harmonic, orbit_monodromy, ro_rotating_jacobian,
                            ro_sync_solution, uniform_rod_moments)
from v16_1.periodic import find_sync_orbits, full_floquet_for_sync
from v16.models import Field

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"; FIG=ROOT/"figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

def plain(v):
    if isinstance(v,complex): return {"real":v.real,"imag":v.imag}
    if isinstance(v,np.ndarray): return plain(v.tolist())
    if isinstance(v,(np.floating,np.integer)): return v.item()
    if isinstance(v,dict): return {str(k):plain(x) for k,x in v.items() if k!="monodromy"}
    if isinstance(v,(list,tuple)): return [plain(x) for x in v]
    return v

def point_qso_monodromy(kappa=.05,gamma=.05):
    f=Field("QSO",alpha=0.,beta=kappa,omega_q=1.)
    def aug(t,z):
        x,y,vx,vy=z[:4]; M=z[4:].reshape(4,4)
        _,_,_,hxx,hxy,hyy=f.evaluate(x,y,t,hessian=True)
        J=np.array([[0,0,1,0],[0,0,0,1],[-hxx,-hxy,-gamma,0],
                    [-hxy,-hyy,0,-gamma]],float)
        return np.r_[vx,vy,-f.evaluate(x,y,t)[1]-gamma*vx,
                     -f.evaluate(x,y,t)[2]-gamma*vy,(J@M).ravel()]
    init=np.r_[np.zeros(4),np.eye(4).ravel()]
    sol=solve_ivp(aug,(0,2*math.pi),init,method="DOP853",rtol=2e-12,
                  atol=2e-14,max_step=2*math.pi/1500)
    M=sol.y[4:,-1].reshape(4,4)
    return {"period":2*math.pi,"multipliers":np.linalg.eigvals(M),
            "integration_success":bool(sol.success)}

def main():
    k4,ell,gt,gr=.1044,.2,.05,.004
    hro=Harmonic(4,"RO",k4,1.,0.)
    ro_solution=ro_sync_solution(k4,ell,gr)
    ro_A=ro_rotating_jacobian(k4,ell,gt,gr,ro_solution["phase_chi"])
    ro_lab=orbit_monodromy([0,0,0,0,ro_solution["phi"],.25],
                           [hro],ell,gt,gr,4*math.pi,steps_per_period=1600)
    ro_eig=np.linalg.eigvals(ro_A)
    ro_full={"solution":ro_solution,"rotating_jacobian_eigenvalues":ro_eig,
             "lab_period":4*math.pi,"lab_multipliers":ro_lab["multipliers"],
             "lab_endpoint":ro_lab["endpoint"]}

    # Standing octupole: shoot both directions from multiple phase seeds.
    hso=Harmonic(4,"SO",k4,1.,0.)
    so_orbits=find_sync_orbits(hso,ell,.003,phase_seeds=8)
    so_records=[]
    for orbit in so_orbits:
        floq=full_floquet_for_sync(hso,ell,gt,.003,orbit)
        so_records.append({"orbit":orbit,"period":floq["period"],
                           "closure_residual":floq["closure_residual"],
                           "multipliers":floq["multipliers"],
                           "max_multiplier_modulus":float(np.max(np.abs(floq["multipliers"])))})

    # Point/rod standing quadrupole libration at psi=0 and rotating QRO locks.
    k2=.05; hqso=Harmonic(2,"SO",k2)
    qso_rod=orbit_monodromy([0,0,0,0,0,0],[hqso],ell,gt,.03,2*math.pi)
    qso_point=point_qso_monodromy(k2,gt)
    hqro=Harmonic(2,"RO",k2)
    qro_orbits=find_sync_orbits(hqro,ell,.03,phase_seeds=8)
    qro_records=[]
    for orbit in qro_orbits:
        floq=full_floquet_for_sync(hqro,ell,gt,.03,orbit)
        qro_records.append({"orbit":orbit,"period":floq["period"],
                            "closure_residual":floq["closure_residual"],
                            "multipliers":floq["multipliers"],
                            "max_multiplier_modulus":float(np.max(np.abs(floq["multipliers"])))})

    # Closed-form thresholds and the independent two-damping map.
    mu=uniform_rod_moments(ell,4); w=.25; b=6*k4*mu[2]
    gamma_rot_crit=2.4*abs(k4)*ell**2
    p=.8*.3**2; gamma_trans_crit=math.sqrt(max(0.,(p/w)**2-w*w))
    gamma_rot_crit_map=2.4*.8*.3**2
    results={
      "metadata":{"version":"V16.1-core","normalization":"tau=Omega*t; R=1; uniform rod; unit total charge",
                  "parameters":{"ell":ell,"kappa4":k4,"gamma_trans":gt,"gamma_rot_RO":gr,
                                "kappa2":k2,"gamma_rot_QRO":.03}},
      "RO_E":{"b":b,"rotational_lock":ro_solution,"rotating_eigenvalues":ro_eig,
              "lab_multipliers":ro_lab["multipliers"],"lab_period":4*math.pi},
      "SO_D":{"gamma_rot":.003,"orbits":so_records},
      "QSO_A_point":{"kappa2":k2,"gamma_trans":gt,"COM_multipliers":qso_point["multipliers"]},
      "QSO_B_rod_libration":{"kappa2":k2,"COM_and_rotation_multipliers":qso_rod["multipliers"]},
      "QRO_C_rod":{"kappa2":k2,"gamma_rot_critical":4*abs(k2),"orbits":qro_records},
      "analytic_boundaries":{"RO_gamma_rot_critical":gamma_rot_crit,
          "RO_COM_no_drag_condition":"abs(kappa4)*ell**2 < (omega4/4)**2",
          "RO_COM_with_drag":"abs(kappa4)*ell**2 < w*sqrt(w**2+gamma_trans**2)",
          "map_example":{"kappa4":.8,"ell":.3,"translation_boundary":gamma_trans_crit,
                         "rotation_boundary":gamma_rot_crit_map}},
      "F_comparison":{"scenario_A_same_RF":"omega2=omega4; frame rates omega2/2 vs omega4/4 differ",
                      "scenario_B_same_frame":"omega4=2*omega2; common rotating frame and H-omega_frame*L invariant when damping=0"}}
    (OUT/"core_stability.json").write_text(json.dumps(plain(results),indent=2),encoding="utf-8")

    # Figure 1: angular locking thresholds at equal dimensionless amplitudes.
    es=np.geomspace(.003,.8,300)
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    for k in (.02,.1,.5): ax.loglog(es,2.4*k*es**2,label=fr"RO octupole, $|\kappa_4|={k}$")
    ax.axhline(.4,color="black",ls="--",label=r"QRO quadrupole, $|\kappa_2|=0.1$ (4|κ₂|)")
    ax.set(xlabel=r"rod aspect ratio $L/R$",ylabel=r"critical $\gamma_{rot}$ (dimensionless)",
           title="Rotational synchronization thresholds")
    ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG/"locking_thresholds.png",dpi=180); plt.close(fig)

    # Figure 2: all six RO modes at the checked point.
    fig,ax=plt.subplots(figsize=(6.6,5.2))
    ax.scatter(ro_eig.real,ro_eig.imag,s=52,c=["#d95f02"]*4+["#1b9e77"]*2)
    ax.axvline(0,color="black",lw=1); ax.axhline(0,color="black",lw=.7)
    ax.set(xlabel="Re(s)",ylabel="Im(s)",title="RO 6D rotating-frame eigenvalues")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/"ro_full_spectrum.png",dpi=180); plt.close(fig)

    # Figure 3: full stability map in (gamma_trans,gamma_rot), not the diagonal.
    gts=np.linspace(0,.24,241); grs=np.linspace(0,.24,241)
    GT,GR=np.meshgrid(gts,grs)
    trans=(p < w*np.sqrt(w*w+GT*GT))
    rot=(GR < gamma_rot_crit_map)
    cls=np.zeros_like(GT,dtype=int)
    cls[trans & rot]=1; cls[~trans & rot]=2; cls[trans & ~rot]=3; cls[~trans & ~rot]=4
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(["#d9d9d9","#4daf4a","#e41a1c","#377eb8","#984ea3"])
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    ax.pcolormesh(GT,GR,cls,cmap=cmap,vmin=0,vmax=4,shading="auto")
    ax.axvline(gamma_trans_crit,color="black",ls="--",lw=1)
    ax.axhline(gamma_rot_crit_map,color="black",ls="-.",lw=1)
    ax.set(xlabel=r"$\gamma_{trans}$",ylabel=r"$\gamma_{rot}$",
           title=r"RO lock and COM stability: $\kappa_4=0.8$, $L/R=0.3$")
    from matplotlib.patches import Patch
    labels=["joint stable","rotation only","translation only","neither stable / no sync"]
    handles=[Patch(color=cmap(i+1),label=labels[i]) for i in range(4)]
    ax.legend(handles=handles,loc="upper left",fontsize=8); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG/"joint_stability_map.png",dpi=180); plt.close(fig)

    # Figure 4: multiplier moduli for the full six-state controls.
    fig,ax=plt.subplots(figsize=(8,4.8))
    series=[]
    series.append(("RO E",np.abs(ro_lab["multipliers"])))
    for rec in so_records: series.append((f"SO D dir {rec['orbit']['direction']} p={rec['orbit']['phase']:.2f}",np.abs(rec["multipliers"])))
    for rec in qro_records: series.append((f"QRO C p={rec['orbit']['phase']:.2f}",np.abs(rec["multipliers"])))
    xpos=0; ticks=[]; names=[]
    for name,vals in series:
        x=np.arange(xpos,xpos+len(vals)); ax.scatter(x,vals,s=28,label=name)
        ticks.append(xpos+2.5); names.append(name); xpos+=len(vals)+1
    ax.axhline(1,color="black",ls="--",lw=1)
    ax.set_xticks(ticks,names,rotation=25,ha="right",fontsize=8)
    ax.set_ylabel("Floquet multiplier modulus"); ax.set_title("RO, SO and rotating quadrupole controls")
    ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIG/"ro_so_qro_floquet.png",dpi=180); plt.close(fig)

    # Size-dependent effective COM quadrupole coefficient.
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    ax.loglog(es,abs(k4)*es**2/2,label=r"$|b|=|\kappa_4|(L/R)^2/2$")
    ax.axhline(w*w/2,color="#d95f02",ls="--",label=r"no-drag marginal $\omega_f^2/2$")
    ax.set(xlabel=r"$L/R$",ylabel="co-rotating quadratic coefficient",
           title="Finite-size center-of-mass term")
    ax.grid(True,which="both",alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/"com_size_scaling.png",dpi=180); plt.close(fig)

    # Hash inputs and outputs (apart from manifest itself) for resumable provenance.
    files=[Path(__file__),ROOT/"dynamics.py",ROOT/"periodic.py",ROOT.parent/"src/v16/rigid.py",
           ROOT.parent/"src/v16/models.py",ROOT.parent/"tests/test_v16_1.py",OUT/"core_stability.json"]
    manifest={"schema":"v16.1-core-manifest-v1","seed_policy":"deterministic, no random seed in analytic run",
              "sha256":{str(p.relative_to(ROOT.parent)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "figures":[p.name for p in sorted(FIG.glob("*.png"))]}
    (OUT/"MANIFEST_v16_1.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(plain(results),indent=2))

if __name__=="__main__": main()
