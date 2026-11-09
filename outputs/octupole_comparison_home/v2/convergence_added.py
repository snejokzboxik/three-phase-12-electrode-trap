import importlib.util,math,csv,json,time
from pathlib import Path
p=Path(r'outputs/octupole_comparison_home/v2')
spec=importlib.util.spec_from_file_location('cc',p/'convergence_controls.py'); cc=importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
cases=((50000.,.007125,math.pi/4,0.),(50000.,.007375,math.pi/4,0.),(100000.,.0075,math.pi/4,0.))
main_rows=list(csv.DictReader((p/'trajectory_results.csv').open(encoding='utf8',newline='')))
rows=[];t=time.perf_counter()
for mz,r0,az,phase in cases:
 for model in ('rotating','standing_equal_peak'):
  r=cc.run(model,mz,r0,az,phase,0.)
  key='|'.join(map(str,(model,mz,r0,0.,az,phase,1.,300)))
  matches=[x for x in main_rows if x['model']==model and abs(float(x['mz_Th'])-mz)<1e-7 and abs(float(x['r0_m'])-r0)<1e-12 and abs(float(x['energy_eV']))<1e-14 and abs(float(x['azimuth_rad'])-az)<1e-12 and abs(float(x['rf_phase_rad'])-phase)<1e-12 and abs(float(x['velocity_sign'])-1.)<1e-12]
  if len(matches)!=1: raise RuntimeError(f'Expected one matching main trajectory, got {len(matches)}: {key}')
  old=matches[0]
  r['job_id']=key;r['main_status']=old.get('status','missing');r['main_escape_period']=old.get('escape_period','');r['classification_agrees']=r['status']==r['main_status']
  rows.append(r)
with (p/'tight_boundary_controls_added.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta={'n_cases':len(rows),'wall_s':time.perf_counter()-t,'settings':'DOP853 rtol=2e-11, max_step=T_RF/100; added paired controls for newly enriched radial nodes','results':rows}
(p/'tight_boundary_controls_added.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');print(json.dumps(meta))
