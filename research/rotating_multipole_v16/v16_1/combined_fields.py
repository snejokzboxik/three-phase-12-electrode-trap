"""Shooting checks for simultaneous rotating quadrupole and octupole fields."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from .dynamics import Harmonic, potential_jet, rigid_rhs, lab_jacobian

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"

def shoot_lock(harmonics,ell,gamma_rot,target,period,phase_seeds=8):
    I=ell*ell/12; roots=[]
    def orbit(q, save=False):
        psi0,spin0=q
        def rhs(t,y):
            _,g,_=potential_jet(0.,0.,y[0],t,ell,harmonics)
            return [y[1],-g[2]/I-gamma_rot*y[1]]
        sol=solve_ivp(rhs,(0,period),[psi0,spin0],method="DOP853",rtol=5e-10,
                      atol=5e-12,max_step=period/80,dense_output=save)
        residual=np.array([sol.y[0,-1]-psi0-target*period,
                           sol.y[1,-1]-spin0])
        return (sol,residual) if save else residual
    for phase in np.linspace(0,math.pi,phase_seeds,endpoint=False):
        guess=np.array([phase,target])
        opt=root(orbit,guess,tol=1e-9,options={"maxfev":35})
        if not opt.success and np.linalg.norm(orbit(opt.x),np.inf)>1e-9: continue
        res=orbit(opt.x)
        if np.linalg.norm(res,np.inf)>1e-8: continue
        p=float(opt.x[0]%math.pi); s=float(opt.x[1])
        if not any(abs((p-r["phase"]+math.pi/2)%math.pi-math.pi/2)<2e-6 and
                   abs(s-r["spin"])<2e-7 for r in roots):
            roots.append({"phase":p,"spin":s,"direction":1 if target>0 else -1,
                          "target_spin":target,"period":period,
                          "shooting_residual":float(np.linalg.norm(res,np.inf))})
    return roots

def mixed_floquet(harmonics,ell,gamma_trans,gamma_rot,orbit):
    I=ell*ell/12; T=orbit["period"]
    y0=np.array([0.,0.,0.,0.,orbit["phase"],orbit["spin"]])
    def aug(t,z):
        y=z[:6]; M=z[6:].reshape(6,6)
        return np.r_[rigid_rhs(t,y,harmonics,ell,gamma_trans,gamma_rot,I),
                     (lab_jacobian(t,y,harmonics,ell,gamma_trans,gamma_rot,I)@M).ravel()]
    sol=solve_ivp(aug,(0,T),np.r_[y0,np.eye(6).ravel()],method="DOP853",
                  rtol=2e-11,atol=2e-13,max_step=T/500)
    M=sol.y[6:,-1].reshape(6,6); end=sol.y[:6,-1]
    expected=y0.copy(); expected[4]+=orbit["target_spin"]*T
    return {"multipliers":np.linalg.eigvals(M),
            "closure_residual":float(np.linalg.norm(end-expected,np.inf)),
            "max_multiplier_modulus":float(np.max(np.abs(np.linalg.eigvals(M))))}

def enrich(harmonics,ell,gt,gr,targets,period):
    records=[]
    for target in targets:
        for o in shoot_lock(harmonics,ell,gr,target,period):
            floq=mixed_floquet(harmonics,ell,gt,gr,o)
            records.append({"orbit":o,"floquet":floq})
    return records

def main():
    ell=.2; gt=.05; gr=.003
    # Scenario A: equal RF frequencies, different spatial frame rates.
    k2=.005; k4=.1044
    fields_a=[Harmonic(2,"RO",k2,1.,0.),Harmonic(4,"RO",k4,1.,0.)]
    branches_a=[]
    for target in (.5,-.5,.25,-.25):
        period=2*math.pi if abs(target)==.5 else 4*math.pi
        for o in shoot_lock(fields_a,ell,gr,target,period,phase_seeds=4):
            branches_a.append({"orbit":o,"floquet":mixed_floquet(fields_a,ell,gt,gr,o)})

    # Scenario B: omega4=2*omega2; both fields share a frame at w=1/2.
    # This deliberately selects an octupole-dominated angular branch while
    # a 1.2%-amplitude quadrupole cancels part of the COM quadratic tensor.
    k2b=-.12; k4b=10.; d2=math.pi/2; d4=0.
    fields_b=[Harmonic(2,"RO",k2b,1.,d2),Harmonic(4,"RO",k4b,2.,d4)]
    roots_b=shoot_lock(fields_b,ell,gr,.5,2*math.pi,phase_seeds=4)
    branches_b=[]
    for o in roots_b:
        f=mixed_floquet(fields_b,ell,gt,gr,o)
        mu2=ell*ell/12; mu4=ell**4/80; I=ell*ell/12
        c=k2b*np.exp(1j*d2)+6*k4b*mu2*np.exp(1j*(2*o["phase"]+d4))
        U2=-4*k2b*mu2*math.cos(2*o["phase"]+d2)
        U4=-16*k4b*mu4*math.cos(4*o["phase"]+d4)
        K=(U2+U4)/I
        f.update({"effective_COM_quadratic_abs":float(abs(c)),
                  "COM_coefficient_complex":[float(c.real),float(c.imag)],
                  "rotational_stiffness":K,
                  "angularly_stable":K>0,
                  "COM_stability_condition":bool(2*abs(c)<.5*math.sqrt(.5**2+gt**2)),
                  "pure_RO4_COM_coefficient":6*k4b*mu2,
                  "pure_RO4_no_drag_COM_stable":bool(2*abs(6*k4b*mu2)<.5**2)})
        branches_b.append({"orbit":o,"floquet":f})
    out={"scenario_A_same_RF_frequency":{"kappa2":k2,"kappa4":k4,"omega2":1.,"omega4":1.,
          "frame_rates":{"quadrupole":.5,"octupole":.25},
          "common_autonomous_rotating_frame":False,
          "meaning":"periodically forced competition; stable Floquet locks are checked by shooting",
          "branches_by_target_spin":branches_a},
      "scenario_B_same_spatial_frame_rate":{"kappa2":k2b,"kappa4":k4b,"omega2":1.,"omega4":2.,
          "common_frame_rate":.5,"common_frame_conserved_quantity":"H - omega_frame*L_z (zero damping)",
          "gamma_trans":gt,"gamma_rot":gr,"delta2":d2,"delta4":d4,
          "quadrupole_to_octupole_coefficient_ratio":abs(k2b/k4b),
          "branches":branches_b,
          "notes":"Selected phase puts the synchronized body near the stable pure-octupole phase; COM coefficient is the vector sum of Q2 and finite-size RO4 terms."}}
    def plain(x):
        if isinstance(x,complex): return {"real":x.real,"imag":x.imag}
        if isinstance(x,np.ndarray): return plain(x.tolist())
        if isinstance(x,(np.floating,np.integer)): return x.item()
        if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [plain(v) for v in x]
        return x
    out=plain(out)
    (OUT/"combined_fields.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
