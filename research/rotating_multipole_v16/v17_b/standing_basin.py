from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from v16_1.dynamics import Harmonic, rigid_rhs, potential_jet

OUT=Path(__file__).resolve().parent; RES=OUT/'results'; RES.mkdir(exist_ok=True)
lockdata=json.loads((RES/'standing_ro3_results.json').read_text(encoding='utf-8'))
a=.1; ell=2*a; kappa=.03; omega=1.; gt=.05; gr=.02; I=ell**2/12; m=3
rfperiod=2*math.pi/omega; nrf=1000; T=3*rfperiod
cm=np.array([0.,a,0.,a**3],dtype=complex)
hs=[Harmonic(3,'SO',kappa,omega,0.)]
seeds=[17017,29011,48391]; per_seed=2
checkpoint=RES/'standing_basin_checkpoint.json'
if checkpoint.exists():
    traj=json.loads(checkpoint.read_text(encoding='utf-8'))['trajectories']
else: traj=[]
completed={r['id'] for r in traj}

def orbit_state(t0,sign):
    r=next(x for x in lockdata['both_directions'] if x['direction']==sign)
    yinit=np.array([r['initial_phase'],r['initial_spin']])
    def f(t,v):
        _,g,_=potential_jet(0.,0.,v[0],t,ell,hs,cm)
        return [v[1],-g[2]/I-gr*v[1]]
    sol=solve_ivp(f,(0,t0),yinit,method='DOP853',rtol=1e-11,atol=1e-13,max_step=.03)
    if not sol.success: raise RuntimeError(sol.message)
    return sol.y[:,-1]

def wilson(k,n,z=1.959963984540054):
    n=max(int(n),1); p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

for sign in (1,-1):
  for seed in seeds:
    for idx in range(per_seed):
      ident=f"{sign:+d}-{seed}-{idx}"
      if ident in completed: continue
      rng=np.random.default_rng(seed*100+idx+100000*(sign<0))
      t0=float(rng.uniform(0,rfperiod)); ph,sp=orbit_state(t0,sign)
      y0=np.array([rng.uniform(-.01,.01),rng.uniform(-.01,.01),rng.uniform(-.005,.005),rng.uniform(-.005,.005),
                   ph+rng.uniform(-.12,.12),sp+rng.uniform(-.02,.02)])
      t1=t0+nrf*rfperiod; tail=t1-5*rfperiod
      t_eval=np.linspace(tail,t1,501)
      def rhs(t,y): return rigid_rhs(t,y,hs,ell,gt,gr,I,cm)
      def aperture(t,y): return (1-a)-math.hypot(y[0],y[1])
      aperture.terminal=True;aperture.direction=-1
      tick=time.time()
      sol=solve_ivp(rhs,(t0,t1),y0,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.4,
                    events=aperture,t_eval=t_eval)
      wall=time.time()-tick
      if sol.status==1:
        row={"id":ident,"direction":sign,"seed":seed,"index":idx,"initial_time":t0,"initial_state":y0.tolist(),
             "status":"ESCAPED_APERTURE","escape_time":float(sol.t[-1]),"nfev":int(sol.nfev),"wall_seconds":wall}
      elif not sol.success or len(sol.t)<10:
        row={"id":ident,"direction":sign,"seed":seed,"index":idx,"initial_time":t0,"initial_state":y0.tolist(),
             "status":"INTEGRATION_FAILED","message":sol.message,"nfev":int(sol.nfev),"wall_seconds":wall}
      else:
        y=sol.y;tt=sol.t; phase=m*y[4]-sign*omega*tt
        slope=float(np.polyfit(tt,phase,1)[0]); wrapped=np.angle(np.exp(1j*(phase-np.mean(phase))))
        pstd=float(np.std(wrapped)); cmax=float(np.max(np.hypot(y[0],y[1]))); ms=float(np.mean(y[5]))
        cap=abs(ms-sign/3)<2e-3 and abs(slope)<2e-3 and pstd<.05 and cmax<.2
        row={"id":ident,"direction":sign,"seed":seed,"index":idx,"initial_time":t0,"initial_state":y0.tolist(),
             "status":"CAPTURED" if cap else "NOT_CAPTURED","mean_spin_last5rf":ms,"phase_slope":slope,
             "wrapped_phase_std":pstd,"com_max_last5rf":cmax,"body_clearance_min_last5rf":1-a-cmax,
             "nfev":int(sol.nfev),"wall_seconds":wall}
      traj.append(row)
      checkpoint.write_text(json.dumps({"parameters":{"a_over_R":a,"ell_over_R":ell,"kappa3":kappa,"omega":omega,
        "gamma_trans":gt,"gamma_rot":gr,"inertia":I,"rf_periods":nrf,"rtol":1e-9,"atol":1e-11,
        "max_step":.4,"seeds":seeds,"per_seed_direction":per_seed},"trajectories":traj},indent=2,ensure_ascii=False),encoding='utf-8')
      print(json.dumps({"id":ident,"status":row['status'],"nfev":row.get('nfev'),"wall_seconds":wall},ensure_ascii=False),flush=True)
summary={"status":"PILOT_1000_RF_BOTH_DIRECTIONS","n":len(traj),"independent_seeds":seeds,"per_seed_direction":per_seed,
 "directions":{str(s):{"n":sum(r['direction']==s for r in traj),"captured":sum(r['direction']==s and r['status']=='CAPTURED' for r in traj),
  "escaped":sum(r['direction']==s and r['status']=='ESCAPED_APERTURE' for r in traj),"capture_wilson95":wilson(sum(r['direction']==s and r['status']=='CAPTURED' for r in traj),sum(r['direction']==s for r in traj))} for s in (1,-1)},
 "checkpoint":str(checkpoint),"trajectories":traj}
(RES/'standing_basin_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='trajectories'},indent=2,ensure_ascii=False))
