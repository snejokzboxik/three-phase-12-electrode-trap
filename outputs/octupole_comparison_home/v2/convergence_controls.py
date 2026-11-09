import importlib.util,json,math,sys,time,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;V1=HERE.parent/'v1'
sp=importlib.util.spec_from_file_location('base',V1/'run_study.py');base=importlib.util.module_from_spec(sp);sys.modules['base']=base;sp.loader.exec_module(base)
def run(model,mz,r0,az,phase=0.,energy=0.):
 q=base.Q;mass=q/(base.E_CHARGE/(mz*base.AMU));rot,stand,matched,amp,_=base.fields();field=rot if model=='rotating' else stand
 vt=math.sqrt(2*energy*base.E_CHARGE/(mz*base.AMU))
 s0=np.array([r0*math.cos(az),r0*math.sin(az),-vt*math.sin(az),vt*math.cos(az)]);om=2*math.pi*base.F;qm=q/mass
 def rhs(t,s):
  x,y,vx,vy=s;cx,cy,sx,sy=field.quadratures(x,y);ph=om*t+phase
  return (vx,vy,qm*(cx*math.cos(ph)+sx*math.sin(ph)),qm*(cy*math.cos(ph)+sy*math.sin(ph)))
 def event(t,s):return base.APERTURE-math.hypot(s[0],s[1])
 event.terminal=True;event.direction=-1
 t0=time.perf_counter();times=np.linspace(0,300/base.F,3601)
 sol=base.solve_ivp(rhs,(0,300/base.F),s0,method='DOP853',rtol=2e-11,atol=(2e-14,2e-14,2e-12,2e-12),max_step=1/(base.F*100),t_eval=times,events=event)
 if not sol.success:raise RuntimeError(sol.message)
 rr=np.hypot(sol.y[0],sol.y[1]);esc=float(sol.t_events[0][0]*base.F) if len(sol.t_events[0]) else None
 return {'model':model,'mz_Th':mz,'r0_mm':r0*1e3,'azimuth_rad':az,'rf_phase_rad':phase,'energy_eV':energy,'status':'escaped' if esc is not None else 'retained','escape_period':esc if esc is not None else '',
  'max_radius_m':float(max(rr)),'rms_radius_m':float(np.sqrt(np.mean(rr**2))),'rtol':2e-11,'max_step_RF':.01,'nfev':int(sol.nfev),'wall_s':time.perf_counter()-t0}
def main():
 rows=[];t=time.perf_counter()
 cases=((50000.,.0075,math.pi/4,0.),(50000.,.00725,math.pi/4,0.),
        (100000.,.0075,math.pi/4,0.),(100000.,.007125,0.,math.pi),
        (200000.,.007375,0.,math.pi))
 for mz,r0,az,phase in cases:
  for model in ('rotating','standing_equal_peak'):rows.append(run(model,mz,r0,az,phase))
 energy_edges=((50000.,.0015,.002),(100000.,.003625,.00375),
               (150000.,.005,.00625),(200000.,.00625,.0075))
 for mz,elow,ehigh in energy_edges:
  for energy in (elow,ehigh):
   for model in ('rotating','standing_equal_peak'):rows.append(run(model,mz,.001,0.,0.,energy))
 old={}
 for ln in (HERE/'trajectory_checkpoint.jsonl').read_text(encoding='utf8').splitlines():
  try:
   r=json.loads(ln)
   if r.get('status') in ('retained','escaped') and r.get('energy_eV')==0:old[r['job_id']]=r
  except Exception:pass
 for r in rows:
  key='|'.join(map(str,(r['model'],r['mz_Th'],r['r0_mm']/1000,r['energy_eV'],r['azimuth_rad'],r['rf_phase_rad'],1.,300)))
  o=old.get(key,{});r['main_status']=o.get('status','missing');r['main_escape_period']=o.get('escape_period','');r['classification_agrees']=r['status']==r['main_status']
 with (HERE/'tight_boundary_controls.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 meta={'n_cases':len(rows),'wall_s':time.perf_counter()-t,'settings':'DOP853 rtol=2e-11, atol=(2e-14 m,2e-14 m,2e-12 m/s,2e-12 m/s), max_step=T_RF/100, 12 output samples/RF','results':rows}
 (HERE/'tight_boundary_controls.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8');print(json.dumps(meta))
if __name__=='__main__':main()
