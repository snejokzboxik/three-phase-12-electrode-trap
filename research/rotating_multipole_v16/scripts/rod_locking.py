from __future__ import annotations

import csv, json, math, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from v16.models import Field
from v16.rigid import rod_rhs, symmetric_rod_lock_prediction

OUT=ROOT/"results"; FIG=ROOT/"figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
OMEGA=2*math.pi*100_000.0

def pdelta(a,b,period=math.pi/2):return (a-b+period/2)%period-period/2

def main():
    kappa=.1044
    gammas=[0,30,100,300,1000,3000,10000]
    lengths=[.005,.01,.025,.05,.10,.20]
    rows=[]
    for L in lengths:
        for gamma in gammas:
            lam=gamma/OMEGA
            p=symmetric_rod_lock_prediction(kappa,L,lam)
            rows.append({"kappa":kappa,"L_over_R":L,"gamma_s-1":gamma,"lambda":lam,
                         "gamma_threshold_s-1":OMEGA*p["threshold_lambda"],
                         "lock_equilibrium_exists":p["can_lock"],
                         "stable_phase_rad":p.get("stable_phase"),
                         "torque_max":p["torque_max"],"required_drag_torque":p["required_drag_torque"],
                         "two_rf_multipliers":json.dumps([{"real":float(z.real),"imag":float(z.imag)}
                             for z in p.get("two_rf_multipliers",[])])})
    with (OUT/"rod_locking_thresholds.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

    confirmations=[]
    field=Field("RO",kappa=kappa)
    L=.2; gamma=3000.; lam=gamma/OMEGA; pred=symmetric_rod_lock_prediction(kappa,L,lam)
    w=.25; phase=pred["stable_phase"]
    # Two nearby starts in one predicted basin. Simulate the full 2-D rod model,
    # not the reduced scalar pendulum used for the bifurcation formula.
    starts=[phase-.06,phase+.06]
    trajectories=[]
    t_end=400*2*math.pi
    t_eval=np.arange(0,t_end+1e-9,4*math.pi)  # head-tail rod strobe, every two RF cycles
    for psi0 in starts:
        y0=np.array([0.,0.,0.,0.,psi0,w+(.015 if psi0>phase else -.015)])
        sol=solve_ivp(lambda t,y:rod_rhs(t,y,field,length_ratio=L,
                    lambda_translation=lam,lambda_rotation=lam,nodes=32),
                    (0,t_end),y0,method="DOP853",rtol=3e-10,atol=3e-12,
                    max_step=2*math.pi/48,t_eval=t_eval)
        ph=sol.y[4]-w*sol.t
        trajectories.append({"start_phase":psi0,"phi":ph,"spin":sol.y[5],
                             "center_radius":np.hypot(sol.y[0],sol.y[1]),"time":sol.t})
    dphi=abs(pdelta(trajectories[0]["phi"][-1],trajectories[1]["phi"][-1]))
    dspin=abs(trajectories[0]["spin"][-1]-trajectories[1]["spin"][-1])
    confirmations.append({"case":"locked full-rod integration","kappa":kappa,"L_over_R":L,
        "gamma_s-1":gamma,"lambda":lam,"rf_horizon":400,"initial_phases":starts,
        "predicted_stable_phase_mod_pi_over_2":phase%(math.pi/2),
        "final_phase_errors": [float(pdelta(q["phi"][-1],phase)) for q in trajectories],
        "between_start_final_phase_distance":float(dphi),"between_start_final_spin_distance":float(dspin),
        "max_center_radius":float(max(np.max(q["center_radius"]) for q in trajectories)),
        "final_spin_values":[float(q["spin"][-1]) for q in trajectories],
        "predicted_two_rf_multipliers":[{"real":float(z.real),"imag":float(z.imag)}
                                         for z in pred["two_rf_multipliers"]],
        "criterion":"same rotating-frame lock reached from two nearby starts; center invariant under symmetric charge distribution"})

    # Analytic nonexistence control, plus full rigid-body phase-slip integration.
    L2=.2; gamma2=10000.; lam2=gamma2/OMEGA
    p2=symmetric_rod_lock_prediction(kappa,L2,lam2)
    y0=np.array([0.,0.,0.,0.,.55,w])
    sol=solve_ivp(lambda t,y:rod_rhs(t,y,field,length_ratio=L2,
                lambda_translation=lam2,lambda_rotation=lam2,nodes=32),
                (0,120*2*math.pi),y0,method="DOP853",rtol=3e-10,atol=3e-12,
                max_step=2*math.pi/48,dense_output=True)
    tplot=np.linspace(0,sol.t[-1],2001); z=sol.sol(tplot)
    phi=z[4]-w*tplot
    confirmations.append({"case":"above-threshold no-lock control","kappa":kappa,"L_over_R":L2,
        "gamma_s-1":gamma2,"lambda":lam2,"rf_horizon":120,
        "predicted_equilibrium_exists":p2["can_lock"],
        "gamma_threshold_s-1":OMEGA*p2["threshold_lambda"],
        "rotating_phase_span_rad":float(np.ptp(phi)),"final_spin":float(z[5,-1]),
        "max_center_radius":float(np.max(np.hypot(z[0],z[1]))),
        "interpretation":"no fixed locked phase exists above the analytic torque-balance threshold; this short run checks the finite-size equations, not a long-time attractor"})

    payload={"timestamp_local":time.strftime('%Y-%m-%d %H:%M:%S %z'),"model":"uniform 2-D rigid charged rod, point-charge quadrature, pure RO",
             "physical_scope":"dimensionless mechanism; only kappa<=0.1615 maps below 250 V under the legacy c4 normalization, but rod mass/charge are not yet tied to a specified material",
             "normalization":"lambda=gamma/(2*pi*100 kHz)","analytic_threshold":"lambda_c=2.4*abs(kappa)*(L/R)^2",
             "confirmations":confirmations}
    (OUT/"rod_lock_validation.json").write_text(json.dumps(payload,indent=2,default=lambda x:complex(x).__repr__() if isinstance(x,complex) else x),encoding="utf-8")

    fig,ax=plt.subplots(figsize=(8.3,5.0),layout="constrained")
    xx=np.logspace(math.log10(min(lengths)*.9),math.log10(max(lengths)*1.1),300)
    ax.loglog(xx,OMEGA*2.4*abs(kappa)*xx**2,color="#172554",lw=2.4,label=r"$γ_c=Ω·2.4|κ|(L/R)^2$")
    colors=plt.cm.viridis(np.linspace(.08,.9,len(gammas)))
    for g,c in zip(gammas,colors):ax.axhline(max(g,0.3),color=c,lw=.8,alpha=.65)
    for g,c in zip(gammas,colors):
        ax.scatter(lengths,[g if g>0 else .3]*len(lengths),s=17,color=c,label=f"{g:g} s$^{{-1}}$")
    ax.set(xlabel=r"rod length $L/R$",ylabel=r"rotational damping $γ$ (s$^{-1}$)",
           title="Analytic torque-balance region for a uniform rod at the trap center")
    ax.text(.006,20000,r"Locked fixed points exist below the threshold curve ($\gamma>0$)."+"\n"+
            "Exact ideal-field result; no FEM or material calibration.",fontsize=8.5)
    ax.grid(True,which="both",alpha=.22);ax.legend(ncol=2,fontsize=7,frameon=False)
    fig.savefig(FIG/"rod_locking_threshold.png",dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.4,4.7),layout="constrained")
    for i,q in enumerate(trajectories):
        err=np.array([pdelta(v,phase) for v in q["phi"]])
        ax.semilogy(q["time"]/(2*math.pi),np.maximum(np.abs(err),1e-12),label=rf"initial $\phi={starts[i]:.3f}$")
    ax.set(xlabel="RF periods",ylabel="distance to locked phase (mod $\\pi/2$)",
           title=fr"Full rod convergence: $L/R={L}$, $\kappa={kappa}$, $\gamma={gamma:g}\,s^{{-1}}$")
    ax.grid(True,alpha=.25);ax.legend(frameon=False)
    fig.savefig(FIG/"rod_locking_convergence.png",dpi=180);plt.close(fig)
    print(json.dumps(payload,indent=2,default=lambda x:complex(x).__repr__() if isinstance(x,complex) else x))

if __name__=="__main__":main()
