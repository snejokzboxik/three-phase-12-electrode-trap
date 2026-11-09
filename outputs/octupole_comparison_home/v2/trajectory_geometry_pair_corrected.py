import importlib.util,math,csv,json,time,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;V1=P.parent/'v1';spec=importlib.util.spec_from_file_location('base',V1/'run_study.py');base=importlib.util.module_from_spec(spec);sys.modules['base']=base;spec.loader.exec_module(base)
F=base.F;omega=2*math.pi*F;qmacro=base.Q;rot,stand,matched,amp,_=base.fields();M=100000.;R=.00775;az=math.pi/6;phase=math.pi/4;H=300
rows=[];results=[];tstart=time.perf_counter()
for model,field in (('rotating',rot),('standing_equal_peak',stand)):
 mass=qmacro/(base.E_CHARGE/(M*base.AMU));qm=qmacro/mass;s0=np.array([R*math.cos(az),R*math.sin(az),0.,0.])
 def rhs(t,s):
  x,y,vx,vy=s;cx,cy,sx,sy=field.quadratures(x,y);ph=omega*t+phase
  return (vx,vy,qm*(cx*math.cos(ph)+sx*math.sin(ph)),qm*(cy*math.cos(ph)+sy*math.sin(ph)))
 def ev(t,s):return base.APERTURE-math.hypot(s[0],s[1])
 ev.terminal=True;ev.direction=-1
 te=np.linspace(0,H/F,H*12+1)
 ts=time.perf_counter();sol=base.solve_ivp(rhs,(0,H/F),s0,method='DOP853',rtol=2e-11,atol=(2e-14,2e-14,2e-12,2e-12),max_step=1/(F*100),t_eval=te,events=ev)
 if not sol.success:raise RuntimeError(sol.message)
 t=sol.t;y=sol.y;esc=float(sol.t_events[0][0]*F) if len(sol.t_events[0]) else None
 if esc is not None:t=np.append(t,sol.t_events[0][0]);y=np.column_stack((y,sol.y_events[0][0]))
 rr=np.hypot(y[0],y[1]);phi=np.unwrap(np.arctan2(y[1],y[0]));status='escaped' if esc is not None else 'retained'
 res={'model':model,'status':status,'escape_period':esc if esc is not None else '', 'max_radius_m':float(rr.max()),'rms_radius_m':float(np.sqrt(np.mean(rr**2))),'net_angle_rad':float(phi[-1]-phi[0]),'nfev':int(sol.nfev),'wall_s':time.perf_counter()-ts}
 results.append(res)
 for i,tt in enumerate(t):rows.append({'model':model,'rf_period':float(tt*F),'x_mm':float(y[0,i]*1000),'y_mm':float(y[1,i]*1000),'r_mm':float(rr[i]*1000),'phi_rad':float(phi[i]),'vx_m_s':float(y[2,i]),'vy_m_s':float(y[3,i])})
with (P/'trajectory_geometry_pair_corrected_timeseries.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta={'mz_Th':M,'r0_mm':R*1000,'azimuth_deg':30,'rf_phase_deg':45,'energy_eV':0,'horizon_RF':H,'aperture_mm':8,'solver':'DOP853 rtol=2e-11, atol=(2e-14 m,2e-14 m,2e-12 m/s,2e-12 m/s), max_step=T_RF/100; 12 samples/RF; terminal aperture event','normalization':'equal peak U0=250 V; 100 kHz','status_interpretation':'Correct physical q/m macro-scaling copied from run_v2.py and convergence_controls.py; this paired diagnostic launch is outside the regular n=24 ensemble.','wall_s':time.perf_counter()-tstart,'results':results}
(P/'trajectory_geometry_pair_corrected.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');print(json.dumps(meta))
