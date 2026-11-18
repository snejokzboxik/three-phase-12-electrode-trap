"""Scenario-level dimensional estimates; no new-geometry FEM coefficient assumed."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from v16_1.dynamics import ro_rotating_jacobian, ro_sync_solution

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"; FIG=ROOT/"figures"
E=1.602176634e-19; R_GAS=8.314462618; M_AIR=.028965; TEMP=300.; P0=.2
V_MAX=250.; F_RF=100_000.; OMEGA=2*math.pi*F_RF
# This is only a transfer scenario from the old geometry, not a new FEM result.
C4_REF=.5413395699439

def model(name,L,d,rho_s,charge_e,profile="uniform"):
    mass=rho_s*math.pi*(d/2)**2*L
    charge=charge_e*E; qm=charge/mass
    # Exact solid-cylinder transverse inertia; the axial line-rod limit is L^2/12.
    inertia=mass*(L*L+3*(d/2)**2)/12
    rho_g=P0*M_AIR/(R_GAS*TEMP)
    vbar=math.sqrt(8*R_GAS*TEMP/(math.pi*M_AIR))
    # Scenario free-molecular drag: beta_t=rho_g*vbar*(L*d), zeta_rot=beta_t*L^2/12.
    # Accommodation and cylinder orientation factors are not experimentally calibrated.
    beta=rho_g*vbar*L*d
    gamma_t=beta/mass; gamma_r=(beta*L*L/12)/inertia
    # Symmetric profile q(g) proportional to 1+a(g^2-1/3), g=2s/L.
    a=.8 if profile=="symmetric_end_weighted" else 0.
    mu2_factor=(1/3+4*a/45)/(1/3)
    mu4_factor=(1/5+8*a/105)/(1/5)
    rows={}
    for aperture in (8e-3,50e-6):
        ell=L/aperture
        kappa=qm*C4_REF*V_MAX/(OMEGA**2*aperture**2)
        if profile!="uniform": kappa=kappa # total charge fixed; moments change separately.
        gt=gamma_t/OMEGA; gr=gamma_r/OMEGA
        mu2=ell**2/12*mu2_factor; mu4=ell**4/80*mu4_factor
        Ibar=inertia/(mass*aperture**2)
        # Dimensionless RO threshold is 16*|kappa*mu4|/Ibar at omega4=1;
        # multiply by Omega to recover the physical relaxation rate.
        crit=16*abs(kappa*mu4)/Ibar*OMEGA
        ratio=gr/(crit/OMEGA) if crit else math.inf
        result={"ell":ell,"kappa4_if_c4_ref_transferred":kappa,
                "mu2":mu2,"mu4":mu4,"mu2_factor":mu2_factor,"mu4_factor":mu4_factor,
                "gamma_trans_s-1":gamma_t,"gamma_rot_s-1":gamma_r,
                "gamma_rot_critical_s-1":crit,"rot_drag_to_lock_threshold":ratio,
                "pressure_critical_mbar_linear_drag":(P0/100)*(crit/gamma_r),
                "c4_required_for_kappa4_0p1":.1*OMEGA**2*aperture**2/(qm*V_MAX),
                "kappa4_required_at_current_drag":gamma_r/OMEGA*Ibar/(16*mu4),
                "COM_quadratic_b":6*kappa*mu2,
                "RO_lock_possible_at_reference_drag":bool(ratio<1)}
        if result["RO_lock_possible_at_reference_drag"]:
            sol=ro_sync_solution(kappa,ell,gr,moments=np.array([1,0,mu2,0,mu4]),inertia=Ibar)
            if sol["exists"]:
                A=ro_rotating_jacobian(kappa,ell,gt,gr,sol["phase_chi"],
                                       moments=np.array([1,0,mu2,0,mu4]),inertia=Ibar)
                eig=np.linalg.eigvals(A)
                result["sync_solution"]={"phase":sol["phi"],"critical_gamma_rot_dimless":sol["gamma_rot_critical"],
                                          "eigenvalues_rotating_frame":[[float(z.real),float(z.imag)] for z in eig],
                                          "max_trans_decay_time_s":float(1/(-max(eig[:4].real)*OMEGA)),
                                          "rot_decay_time_s":float(1/(-max(eig[4:].real)*OMEGA))}
        else: result["sync_solution"]={"exists":False}
        rows[f"R_{aperture:g}_m"]=result
    return {"name":name,"geometry":{"length_m":L,"diameter_m":d,"density_kg_m3":rho_s,
                                      "mass_kg":mass,"charge_e":charge_e,"charge_C":charge,
                                      "Q_over_M_C_kg":qm,"moment_of_inertia_kg_m2":inertia,
                                      "profile":profile},
            "environment":{"pressure_mbar":P0/100,"temperature_K":TEMP,"gas":"air",
                            "mean_speed_m_s":vbar,"gas_density_kg_m3":rho_g,
                            "drag_model":"beta_trans=rho_g*vbar*L*d; zeta_rot=beta_trans*L^2/12"},
            "field":{"R_m_options":[8e-3,50e-6],"RF_Hz":F_RF,"Vmax_V":V_MAX,
                     "c4_scenario_only":C4_REF},"apertures":rows}

def main():
    scenarios=[
      model("silica_microrod_uniform",15e-6,4e-6,2200.,1000),
      model("silica_nanostring_uniform",500e-9,50e-9,2200.,10),
      model("silica_microrod_even_nonuniform_charge",15e-6,4e-6,2200.,1000,"symmetric_end_weighted")]
    out={"normalization":"kappa4=(Q/M)c4 V/(Omega^2 R^2); gamma=physical damping/Omega",
         "status":"SCENARIO_ONLY; c4 is not validated for either new geometry",
         "scenarios":scenarios}
    (OUT/"physical_scenarios.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    labels=["micro rod\nR=8 mm","micro rod\nR=50 µm","nano rod\nR=8 mm","nano rod\nR=50 µm"]
    vals=[]; crit=[]; drag=[]
    for idx,name in ((0,"silica_microrod_uniform"),(1,"silica_nanostring_uniform")):
        sc=scenarios[idx]
        for R in ("R_0.008_m","R_5e-05_m"):
            v=sc["apertures"][R]; vals.append(v["kappa4_if_c4_ref_transferred"])
            crit.append(v["gamma_rot_critical_s-1"]); drag.append(v["gamma_rot_s-1"])
    # Reorder scenario-major to the plotted labels.
    x=np.arange(4); fig,ax=plt.subplots(figsize=(8,4.8))
    ax.bar(x,crit,color="#4daf4a",label="RO rotational-lock threshold")
    ax.scatter(x,drag,color="#e41a1c",marker="x",s=75,label="scenario gas rotational damping")
    ax.set_yscale("log"); ax.set_xticks(x,labels); ax.set_ylabel("physical rate (s⁻¹)")
    ax.set_title("Physical comparison at 100 kHz and 250 V\n(c₄ transfer is illustrative; new FEM required)")
    ax.legend(); ax.grid(axis="y",which="both",alpha=.25); fig.tight_layout()
    fig.savefig(FIG/"physical_locking_thresholds.png",dpi=180); plt.close(fig)
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
