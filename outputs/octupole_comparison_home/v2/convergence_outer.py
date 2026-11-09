import importlib.util,math,csv,json,time
from pathlib import Path
p=Path(r'outputs/octupole_comparison_home/v2')
spec=importlib.util.spec_from_file_location('cc',p/'convergence_controls.py');cc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cc)
main=list(csv.DictReader((p/'trajectory_results.csv').open(encoding='utf8',newline='')))
rows=[];t=time.perf_counter()
for mz in (50000.,100000.,150000.,200000.):
 for model in ('rotating','standing_equal_peak'):
  r=cc.run(model,mz,.00775,math.pi/4,0.,0.)
  matches=[x for x in main if x['model']==model and abs(float(x['mz_Th'])-mz)<1e-7 and abs(float(x['r0_m'])-.00775)<1e-12 and abs(float(x['energy_eV']))<1e-14 and abs(float(x['azimuth_rad'])-math.pi/4)<1e-12 and abs(float(x['rf_phase_rad']))<1e-12]
  if len(matches)!=1:raise RuntimeError(f'missing/duplicate main trajectory for {model} {mz}')
  x=matches[0];r['job_id']=x['job_id'];r['main_status']=x['status'];r['main_escape_period']=x['escape_period'];r['classification_agrees']=r['status']==x['status'];rows.append(r)
with (p/'tight_boundary_controls_outer.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta={'n_cases':len(rows),'wall_s':time.perf_counter()-t,'settings':'DOP853 rtol=2e-11, max_step=T_RF/100; one matched outer transition state per mass and model','all_classifications_agree':all(r['classification_agrees'] for r in rows),'results':rows}
(p/'tight_boundary_controls_outer.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8');print(json.dumps(meta))
