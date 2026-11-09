import importlib.util,math,csv,json,time
from pathlib import Path
p=Path(r'outputs/octupole_comparison_home/v2');spec=importlib.util.spec_from_file_location('cc',p/'convergence_controls.py');cc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cc)
rows=[];t=time.perf_counter()
for model in ('rotating','standing_equal_peak'):
 rows.append(cc.run(model,100000.,.00775,math.pi/6,math.pi/4,0.))
main=json.loads((p/'trajectory_geometry_pair.json').read_text(encoding='utf8'))
for r in rows:
 x=next(q for q in main['results'] if q['model']==r['model']);r['main_status']=x['status'];r['main_escape_period']=x['escape_period'];r['classification_agrees']=r['status']==x['status']
meta={'n_cases':len(rows),'wall_s':time.perf_counter()-t,'settings':'DOP853 rtol=2e-11, max_step=T_RF/100; same diagnostic launch as stored trajectory geometry pair','all_classifications_agree':all(x['classification_agrees'] for x in rows),'results':rows}
(p/'tight_diagnostic_pair.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8');print(json.dumps(meta))
