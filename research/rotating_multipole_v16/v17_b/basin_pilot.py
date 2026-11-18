from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from v16_1.dynamics import Harmonic, rigid_rhs
from v17_b.model import charge_moments, synchronous_branch

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
RES=OUT/'results'; RES.mkdir(exist_ok=True)
# The pilot uses three independently seeded initial ensembles. Each trajectory
# gets a terminal event at first whole-body aperture loss and is checkpointed.
a=.1; ell=2*a; kappa=.03; omega=1.; gt=.05; gr=.02; I=ell**2/12
cm=charge_moments([1.,-1.],[a,-a],3).moments
lock=synchronous_branch(3,kappa,omega,cm,gr,I); assert lock['exists']
w=omega/3; phi=lock['phase']; hs=[Harmonic(3,'RO',kappa,omega,0.)]
seeds=[17017,29011,48391]; per_seed=2; nrf=1000; rfperiod=2*math.pi/omega
checkpoint=RES/'basin_local_checkpoint.json'
if checkpoint.exists():
    data=json.loads(checkpoint.read_text(encoding='utf-8'))
    results=data['trajectories']
else:
    results=[]
completed={x['id'] for x in results}

def one(seed, idx):
    rng=np.random.default_rng(seed*100+idx)
    t0=float(rng.uniform(0,rfperiod))
    y0=np.array([rng.uniform(-.01,.01),rng.uniform(-.01,.01),
                 rng.uniform(-.005,.005),rng.uniform(-.005,.005),
                 w*t0+phi+rng.uniform(-.12,.12),w+rng.uniform(-.02,.02)])
    t1=t0+nrf*rfperiod
    def rhs(t,y): return rigid_rhs(t,y,hs,ell,gt,gr,I,cm)
    def aperture(t,y): return (1-a)-math.hypot(y[0],y[1])
    aperture.terminal=True; aperture.direction=-1
    tail_start=t1-5*rfperiod
    t_eval=np.linspace(tail_start,t1,501)
    start=time.time()
    sol=solve_ivp(rhs,(t0,t1),y0,method='DOP853',rtol=1e-9,atol=1e-11,
                  max_step=.4,events=aperture,t_eval=t_eval)
    duration=time.time()-start
    if sol.status==1:
        return {"id":f"{seed}-{idx}","seed":seed,"initial_index":idx,"initial_time":t0,
          "initial_state":y0.tolist(),"status":"ESCAPED_APERTURE","escape_time":float(sol.t[-1]),
          "nfev":int(sol.nfev),"wall_seconds":duration}
    if not sol.success or len(sol.t)<10:
        return {"id":f"{seed}-{idx}","seed":seed,"initial_index":idx,"initial_time":t0,
          "initial_state":y0.tolist(),"status":"INTEGRATION_FAILED","message":sol.message,
          "nfev":int(sol.nfev),"wall_seconds":duration}
    y=sol.y; tt=sol.t
    phase=3*y[4]-omega*tt
    slope=float(np.polyfit(tt,phase,1)[0])
    wrapped=np.angle(np.exp(1j*(phase-np.mean(phase))))
    phase_std=float(np.std(wrapped))
    commax=float(np.max(np.hypot(y[0],y[1])))
    mean_spin=float(np.mean(y[5]))
    capture=(abs(mean_spin-w)<2e-3 and abs(slope)<2e-3 and phase_std<.05 and commax<.2)
    return {"id":f"{seed}-{idx}","seed":seed,"initial_index":idx,"initial_time":t0,
      "initial_state":y0.tolist(),"status":"CAPTURED" if capture else "NOT_CAPTURED",
      "mean_spin_last5rf":mean_spin,"lock_phase_slope":slope,"lock_phase_wrapped_std":phase_std,
      "com_max_last5rf":commax,"body_clearance_min_last5rf":float(1-a-commax),
      "nfev":int(sol.nfev),"wall_seconds":duration}

for seed in seeds:
    for idx in range(per_seed):
        ident=f"{seed}-{idx}"
        if ident in completed: continue
        row=one(seed,idx); results.append(row)
        checkpoint.write_text(json.dumps({"parameters":{"a_over_R":a,"ell_over_R":ell,"kappa3":kappa,
          "omega":omega,"gamma_trans":gt,"gamma_rot":gr,"inertia":I,"rf_periods":nrf,
          "rtol":1e-9,"atol":1e-11,"max_step":.4,"seeds":seeds,"per_seed":per_seed},
          "trajectories":results},indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps({"id":ident,"status":row['status'],"nfev":row.get('nfev'),"wall_seconds":row.get('wall_seconds')},ensure_ascii=False),flush=True)
# Ensemble-level Wilson 95% interval for the empirical capture fraction.
n=len(results); k=sum(r['status']=='CAPTURED' for r in results); z=1.959963984540054
p=k/n if n else float('nan'); den=1+z*z/n if n else 1
center=(p+z*z/(2*n))/den if n else float('nan')
half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den if n else float('nan')
summary={"status":"PILOT_1000_RF","n":n,"captured":k,"escaped":sum(r['status']=='ESCAPED_APERTURE' for r in results),
 "not_captured":sum(r['status']=='NOT_CAPTURED' for r in results),"failed":sum(r['status']=='INTEGRATION_FAILED' for r in results),
 "capture_fraction":p,"wilson95":[max(0.,center-half),min(1.,center+half)],"independent_seeds":seeds,
 "checkpoint":str(checkpoint),"trajectories":results}
(RES/'basin_local_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='trajectories'},indent=2,ensure_ascii=False))
