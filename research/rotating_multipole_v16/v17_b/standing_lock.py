from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.optimize import root, linear_sum_assignment
from scipy.integrate import solve_ivp
from v16_1.dynamics import Harmonic, rigid_rhs, lab_jacobian, potential_jet
from v17_b.model import charge_moments

OUT=Path(__file__).resolve().parent; RES=OUT/'results'; RES.mkdir(exist_ok=True)
a=.1; ell=2*a; kappa=.03; omega=1.; gt=.05; gr=.02; I=ell**2/12
m=3; w0=omega/m; T=2*math.pi*m/abs(omega)
cm=charge_moments([1.,-1.],[a,-a],m).moments
hs=[Harmonic(m,'SO',kappa,omega,0.)]
results=[]
for sign in (1,-1):
    w=sign*w0
    # Averaged torque amplitude is m*kappa*C_m/2; use the stable branch cos(3phi)<0.
    amp=m*kappa*abs(cm[m])/2
    r=gr*I*w/amp
    if abs(r)>=1: raise RuntimeError('averaged standing-wave branch does not exist at this point')
    phase_eff=math.pi-math.asin(r) if r>=0 else math.pi-math.asin(r)
    # For r<0, pi-asin(r) is in quadrant III and remains the stable branch.
    phi=phase_eff/m
    def residual(z):
        y0=np.array([z[0],z[1]])
        def angle_rhs(t,v):
            _,g,_=potential_jet(0.,0.,v[0],t,ell,hs,cm)
            return np.array([v[1],-g[2]/I-gr*v[1]])
        sol=solve_ivp(angle_rhs,(0,T),y0,
                      method='DOP853',rtol=2e-10,atol=2e-12,max_step=T/800)
        if not sol.success: raise RuntimeError(sol.message)
        end=sol.y[:,-1]
        return np.array([end[0]-y0[0]-2*math.pi*sign,T*(end[1]-y0[1])])
    fit=root(residual,[phi,w],method='hybr',tol=2e-10,options={'maxfev':60})
    y0=np.array([0.,0.,0.,0.,fit.x[0],fit.x[1]])
    # Independent variational monodromy along the full six-dimensional periodic orbit.
    def variational(steps):
        def aug(t,z):
            y=z[:6]; M=z[6:].reshape(6,6)
            return np.r_[rigid_rhs(t,y,hs,ell,gt,gr,I,cm),
                         (lab_jacobian(t,y,hs,ell,gt,gr,I,cm)@M).ravel()]
        out=solve_ivp(aug,(0,T),np.r_[y0,np.eye(6).ravel()],method='DOP853',
                      rtol=3e-11,atol=3e-13,max_step=T/steps)
        if not out.success: raise RuntimeError(out.message)
        return out.y[:6,-1],out.y[6:,-1].reshape(6,6)
    endpoint_coarse,Mcoarse=variational(800)
    endpoint_fine,M=variational(1600)
    mu=np.linalg.eigvals(M); muc=np.linalg.eigvals(Mcoarse)
    ii,jj=linear_sum_assignment(np.abs(mu[:,None]-muc[None,:]))
    expected=y0.copy(); expected[4]+=2*math.pi*sign
    closure=float(np.linalg.norm(endpoint_fine-expected,np.inf))
    exponents=np.log(mu.astype(complex))/T
    results.append({"direction":sign,"shooting_success":bool(fit.success),"shooting_message":fit.message,
      "averaged_phase_guess":phi,"initial_phase":float(fit.x[0]),"initial_spin":float(fit.x[1]),
      "mean_spin_target":w,"period":T,"closure_residual_inf":closure,
      "coarse_fine_multiplier_max_abs_diff":float(np.max(np.abs(mu[ii]-muc[jj]))),
      "multipliers_abs":[float(abs(x)) for x in mu],"multipliers_complex":[[float(x.real),float(x.imag)] for x in mu],
      "floquet_exponents":[[float(x.real),float(x.imag)] for x in exponents],
      "com_multipliers_abs":[float(abs(x)) for x in np.linalg.eigvals(M[:4,:4])],
      "rotation_multipliers_abs":[float(abs(x)) for x in np.linalg.eigvals(M[4:,4:])],
      "linear_block_coupling_inf":float(max(np.max(np.abs(M[:4,4:])),np.max(np.abs(M[4:,:4])))),
      "max_multiplier_abs":float(max(abs(mu))),"stable_6d":bool(max(abs(mu))<1),
      "endpoint":endpoint_fine.tolist(),"integration_success":True})
res={"status":"COMPLETED_FOR_ONE_STANDING_RO3_PARAMETER_POINT","field":"SO3: kappa Re[P3] cos(omega*t)",
 "parameters":{"a_over_R":a,"ell_over_R":ell,"kappa3":kappa,"omega":omega,"gamma_trans":gt,"gamma_rot":gr,"inertia":I,"reference_charge":2.0},
 "averaged_torque_amplitude":amp,"averaged_gamma_rot_critical":amp/(I*abs(w0)),
 "both_directions":results,
 "limitations":["one parameter point only","periodic orbit shooting, not basin maps or continuation","ideal harmonic field, no FEM or physical electrode calibration","COM is at zero by symmetry; Floquet checks perturbations"]}
(RES/'standing_ro3_results.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({"status":res['status'],"avg_gamma_rot_c":res['averaged_gamma_rot_critical'],"results":[{k:v for k,v in q.items() if k in ('direction','shooting_success','initial_phase','initial_spin','closure_residual_inf','max_multiplier_abs','stable_6d','floquet_exponents')} for q in results]},indent=2))
