"""Adaptive paired RF comparison. Full RF N48 trajectories; append-only checkpoints."""
import csv, importlib.util, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
V1=HERE.parent/'v1'
SP=importlib.util.spec_from_file_location('base',V1/'run_study.py')
base=importlib.util.module_from_spec(SP); sys.modules['base']=base; SP.loader.exec_module(base)
JOURNAL=HERE/'trajectory_checkpoint.jsonl'
FREQ=base.F; AP=base.APERTURE

def worker(a):
    wall0=time.perf_counter()
    model,mz,r0,energy,az,phase,sign,horizon=a
    qmacro=base.Q; physical_mass=mz*base.AMU
    mass=qmacro/(base.E_CHARGE/(mz*base.AMU))
    speed=math.sqrt(max(0,2*energy*base.E_CHARGE/physical_mass))
    vr=0.; vt=sign*speed
    rot,stand,matched,amp,_=base.fields()
    field=rot if model=='rotating' else stand
    s0=np.array([r0*math.cos(az),r0*math.sin(az),-vt*math.sin(az),vt*math.cos(az)])
    omega=2*math.pi*FREQ; qm=qmacro/mass
    def rhs(t,s):
        x,y,vx,vy=s; cx,cy,sx,sy=field.quadratures(x,y); p=omega*t+phase
        return (vx,vy,qm*(cx*math.cos(p)+sx*math.sin(p)),qm*(cy*math.cos(p)+sy*math.sin(p)))
    def escape(t,s):return AP-math.hypot(s[0],s[1])
    escape.terminal=True; escape.direction=-1
    # Uniform RF-phase sampling makes the RMS metric time-weighted.
    t_eval=np.linspace(0,horizon/FREQ,horizon*12+1)
    try:
        sol=base.solve_ivp(rhs,(0,horizon/FREQ),s0,method='DOP853',rtol=2e-9,
            atol=(2e-13,2e-13,2e-11,2e-11),max_step=1/(FREQ*50),t_eval=t_eval,events=escape)
        if not sol.success: raise RuntimeError(sol.message)
        rr=np.hypot(sol.y[0],sol.y[1]); ang=np.unwrap(np.arctan2(sol.y[1],sol.y[0]))
        et=float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
        status='escaped' if et is not None else 'retained'
        # Signed net angular sweep; radial range and RMS excursion use uniform-time samples.
        return {'model':model,'mz_Th':mz,'r0_m':r0,'energy_eV':energy,'azimuth_rad':az,
          'rf_phase_rad':phase,'velocity_sign':sign,'speed_m_s':speed,'horizon_RF':horizon,
          'status':status,'escape_period':et*FREQ if et is not None else '',
          'max_radius_m':float(rr.max()),'rms_radius_m':float(np.sqrt(np.mean(rr**2))),
          'mean_radius_m':float(np.mean(rr)),'radial_span_m':float(rr.max()-rr.min()),
          'net_angle_rad':float(ang[-1]-ang[0]),'nfev':int(sol.nfev),
          'wall_s':time.perf_counter()-wall0,'solver':'DOP853 rtol=2e-9, max_step=T/50, 12 samples/RF',
          'normalization':'equal_peak_U0_250V'}
    except Exception as e:
        return {'model':model,'mz_Th':mz,'r0_m':r0,'energy_eV':energy,'azimuth_rad':az,
          'rf_phase_rad':phase,'velocity_sign':sign,'speed_m_s':speed,'horizon_RF':horizon,
          'status':'unresolved:'+type(e).__name__,'error':str(e),'normalization':'equal_peak_U0_250V'}

def key(a):return '|'.join(map(str,a))
def run_stage(stage,workers=4):
    jobs=[]
    if stage=='energy':
      for mz in (50000.,100000.,150000.,200000.):
       for energy in (.003,.0035,.004,.0045,.005):
        for az in (0.,math.pi/2,math.pi,3*math.pi/2):
         for phase in (0.,math.pi):
          for sign in (-1.,1.):
           for model in ('rotating','standing_equal_peak'):
            jobs.append((model,mz,.001,energy,az,phase,sign,300))
    elif stage=='radial_outer_enrich':
      # Expand the two nearest-to-aperture mixed nodes to the established
      # 24-state azimuth/phase grid; baseline states are deduplicated.
      for mz in (50000.,100000.,150000.,200000.):
       for r0 in (.00775,.007875):
        for az in (0.,math.pi/2,math.pi,3*math.pi/2,math.pi/4,3*math.pi/4):
         for phase in (0.,math.pi/2,math.pi,3*math.pi/2):
          for model in ('rotating','standing_equal_peak'):
           jobs.append((model,mz,r0,0.,az,phase,1.,300))
    elif stage=='radial_outer':
      # Extend the zero-velocity radial scan toward the 8-mm aperture;
      # baseline four azimuth/phase starts per model, mass and radius.
      for mz in (50000.,100000.,150000.,200000.):
       for r0 in (.007625,.00775,.007875):
        for az in (0.,math.pi/2):
         for phase in (0.,math.pi):
          for model in ('rotating','standing_equal_peak'):
           jobs.append((model,mz,r0,0.,az,phase,1.,300))
    elif stage=='radial':
      # Coarse direct radial sections at four masses, with zero velocity;
      # only measured points are plotted and transitions can be refined later.
      for mz in (50000.,100000.,150000.,200000.):
       for r0 in (.0005,.001,.002,.004,.0055,.0065,.0075):
        for az in (0.,math.pi/2):
         for phase in (0.,math.pi):
          for model in ('rotating','standing_equal_peak'):
           jobs.append((model,mz,r0,0.,az,phase,1.,300))
    elif stage=='refine':
      # Only bisect actually observed transition intervals, separately per mass/model.
      journal=[]
      if JOURNAL.exists():
       for ln in JOURNAL.read_text(encoding='utf8').splitlines():
        try:
         x=json.loads(ln)
         if x.get('status') in ('retained','escaped') and x.get('r0_m')==.001:journal.append(x)
        except Exception:pass
      by={}
      for x in journal:
       k=(float(x['mz_Th']),x['model'],float(x['energy_eV']))
       by.setdefault(k,[]).append(1 if x['status']=='retained' else 0)
      midpoints=set()
      for mz in (50000.,100000.,150000.,200000.):
       for model in ('rotating','standing_equal_peak'):
        es=sorted(k[2] for k in by if k[:2]==(mz,model))
        for lo,hi in zip(es,es[1:]):
         a=sum(by[(mz,model,lo)])/len(by[(mz,model,lo)])
         b=sum(by[(mz,model,hi)])/len(by[(mz,model,hi)])
         if a!=b and hi-lo>.0002:midpoints.add((mz,(lo+hi)/2))
      # Expand beyond the primary energy window where that mass showed no
      # transition; these are adaptive probes informed by the v1 pilot data.
      for mz,e in ((50000.,.001),(50000.,.0015),(50000.,.002),
                   (150000.,.0075),(150000.,.01),(200000.,.0075),(200000.,.01)):
       midpoints.add((mz,e))
      for mz,energy in sorted(midpoints):
        for az in (0.,math.pi/2,math.pi,3*math.pi/2):
         for phase in (0.,math.pi):
          for sign in (-1.,1.):
           for model in ('rotating','standing_equal_peak'):
            jobs.append((model,mz,.001,energy,az,phase,sign,300))
    elif stage=='radial_refine':
      # Bisect only directly observed radial transitions; enrich the azimuth/RF
      # phase grid only at measured mixed-outcome radii.
      journal=[]
      if JOURNAL.exists():
       for ln in JOURNAL.read_text(encoding='utf8').splitlines():
        try:
         x=json.loads(ln)
         if x.get('status') in ('retained','escaped') and float(x.get('energy_eV',-1))==0:journal.append(x)
        except Exception:pass
      by={}
      for x in journal:
       k=(float(x['mz_Th']),x['model'],float(x['r0_m']))
       by.setdefault(k,[]).append(1 if x['status']=='retained' else 0)
      radii=set();mixed=set()
      for mz in (50000.,100000.,150000.,200000.):
       for model in ('rotating','standing_equal_peak'):
        vals=sorted((k[2],sum(by[k])/len(by[k])) for k in by if k[:2]==(mz,model))
        for (r0,p0),(r1,p1) in zip(vals,vals[1:]):
         if p0!=p1:radii.add((mz,(r0+r1)/2))
        for (r0,p) in vals:
         n=len(by[(mz,model,r0)])
         if 0<p<1 and n<16:mixed.add((mz,r0))
      for mz,r0 in sorted(radii):
       for az in (0.,math.pi/2):
        for phase in (0.,math.pi):
         for model in ('rotating','standing_equal_peak'):
          jobs.append((model,mz,r0,0.,az,phase,1.,300))
      for mz,r0 in sorted(mixed):
       for az in (0.,math.pi/2,math.pi,3*math.pi/2,math.pi/4,3*math.pi/4):
        for phase in (0.,math.pi/2,math.pi,3*math.pi/2):
         for model in ('rotating','standing_equal_peak'):
          jobs.append((model,mz,r0,0.,az,phase,1.,300))
    else: raise ValueError(stage)
    done={}
    if JOURNAL.exists():
      for ln in JOURNAL.read_text(encoding='utf8').splitlines():
       try:
        x=json.loads(ln)
        if x.get('status') in ('retained','escaped'):done[x['job_id']]=x
       except Exception:pass
    todo=[a for a in jobs if key(a) not in done]
    t0=time.perf_counter(); completed=0
    if todo:
      with ProcessPoolExecutor(max_workers=workers) as pool, JOURNAL.open('a',encoding='utf8') as f:
       fs={pool.submit(worker,a):(key(a),a) for a in todo}
       for fut in as_completed(fs):
        jid,a=fs[fut]
        try:r=fut.result()
        except Exception as e:r={'status':'unresolved:'+type(e).__name__,'error':str(e)}
        r['job_id']=jid; f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        if r['status'] in ('retained','escaped'):done[jid]=r
        completed+=1
    rows=list(done.values())
    with (HERE/'trajectory_results.csv').open('w',newline='',encoding='utf8') as f:
      fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    status={'stage':stage,'requested':len(jobs),'valid_completed_total':len(rows),'new_attempts':completed,
      'workers':workers,'wall_s':time.perf_counter()-t0,'throughput_traj_min':completed/max(1e-9,time.perf_counter()-t0)*60,
      'unresolved_total_in_journal':sum(1 for ln in JOURNAL.read_text(encoding='utf8').splitlines() if 'unresolved' in ln),
      'checkpoint':JOURNAL.name,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (HERE/'runtime_log.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
    print(json.dumps(status))

if __name__=='__main__':
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    run_stage(sys.argv[1] if len(sys.argv)>1 else 'energy',int(sys.argv[2]) if len(sys.argv)>2 else 4)
