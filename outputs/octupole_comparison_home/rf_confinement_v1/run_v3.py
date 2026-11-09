"""Reproducible full-RF confinement search using the published P4 N48 fields.

This runner has no dependency on the untracked V1/V2 study scripts.  Its first
trajectory starts the hard 120 minute integration deadline and all completed
segments are appended to an fsync'd checkpoint journal.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.rf_reconstruction import RFReconstruction

OUT = Path(__file__).resolve().parent
FIELD35 = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h35_P4/diagnostic_N48.json"
FIELD50 = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json"
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
APERTURE = 8e-3
MACRO_Q = 1e-12
MACRO_M = 1e-15
B6_MACRO = 22.123924019590998
DEADLINE_S = 120 * 60


def load_fields(path=FIELD35):
    rec = RFReconstruction.load(path)
    q = rec.unit_quadratures
    return rec, q


def field_pair(q, model):
    """Return E_cos,E_sin (V/m per V peak electrode amplitude)."""
    if model == "rotating":
        return q
    if model == "standing":
        from three_phase_trap.rf_reconstruction import HarmonicQuadratures
        return HarmonicQuadratures(np.zeros_like(q.cosine), q.sine.copy(), q.reference_radius)
    raise ValueError(model)


def field_values(q, xy, phase, voltage):
    ec, es = q.field_quadratures(np.asarray(xy, dtype=float))
    return voltage * (ec * math.cos(phase) + es * math.sin(phase))


def audit_fields():
    rec35, q35 = load_fields(FIELD35)
    rec50, q50 = load_fields(FIELD50)
    mz = 100_000.0
    mass = mz * AMU
    qm = E_CHARGE / mass
    omega = 2 * math.pi * 100_000.0
    points = []
    radial_rows = []
    scale = (E_CHARGE**2 / mass) / (MACRO_Q**2 / MACRO_M)
    for label, q in (("h35", q35), ("h50", q50)):
        for radius in (0.5e-3, 1e-3, 2e-3, 2.35e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 7.5e-3):
          for theta in np.arange(12)*np.pi/6:
            xy = np.array([[radius*np.cos(theta), radius*np.sin(theta)]])
            ec, es = q.field_quadratures(xy)
            jc, js = q.jacobian_quadratures(xy)
            ec *= 250.; es *= 250.; jc *= 250.; js *= 250.
            e_rot = float(np.linalg.norm(np.r_[ec[0], es[0]]))
            e_st = float(np.linalg.norm(es[0]))
            grad_bound = max(float(np.linalg.norm(jc[0], 2)), float(np.linalg.norm(js[0], 2)))
            pseudo_rot = E_CHARGE**2 * e_rot**2 / (4 * mass * omega**2)
            pseudo_st = E_CHARGE**2 * e_st**2 / (4 * mass * omega**2)
            amp_bound = qm * e_rot / omega**2
            q_local = qm * grad_bound / omega**2
            radial_rows.append({"field": label, "r_m": radius, "theta_rad": float(theta), "E_rot_quadrature_norm_V_m": e_rot,
                "E_standing_V_m": e_st, "gradient_operator_bound_V_m2": grad_bound,
                "pseudo_rot_eV": pseudo_rot / E_CHARGE, "pseudo_standing_eV": pseudo_st / E_CHARGE,
                "B6_local_rot_J_m6": pseudo_rot / radius**6, "B6_local_standing_J_m6": pseudo_st / radius**6,
                "B6_local_rot_over_scaled_model": (pseudo_rot / radius**6) / (B6_MACRO*scale),
                "micromotion_amplitude_bound_m": amp_bound, "micromotion_over_radius": amp_bound / radius,
                "local_adiabatic_q_bound": q_local})
        points.append({"field": label, "reference_radius_m": q.reference_radius,
                       "valid_radius_m": rec35.valid_radius if label == "h35" else rec50.valid_radius,
                       "harmonic_order": len(q.cosine) - 1,
                       "max_cos_coeff_abs_V_per_V": float(np.max(np.abs(q.cosine))),
                       "max_sin_coeff_abs_V_per_V": float(np.max(np.abs(q.sine)))})
    # Independently check E=-grad(Phi) by central differences on the saved
    # potential quadratures, including the final applied-voltage scale.
    max_fd_rel = 0.
    for q in (q35, q50):
        point = np.array([1.73e-3, -0.81e-3])
        h = 1e-8
        for coeff in (q.cosine, q.sine):
            phi0 = q.potential_quadratures(point[None, :])[0 if coeff is q.cosine else 1][0] * 250.
            # The direct helper is evaluated on four displaced coordinates.
            ps = q.potential_quadratures(np.array([point+[h,0], point-[h,0], point+[0,h], point-[0,h]]))
            which = 0 if coeff is q.cosine else 1
            grad = 250 * np.array([(ps[which][0]-ps[which][1])/(2*h), (ps[which][2]-ps[which][3])/(2*h)])
            ev = q.field_quadratures(point[None, :])[which][0] * 250.
            max_fd_rel = max(max_fd_rel, float(np.linalg.norm(grad + ev) / max(1., np.linalg.norm(ev))))
    pattern = [math.cos(4 * j * math.pi / 6) for j in range(12)]
    b6_phys = B6_MACRO * scale
    audit = {
        "field_sources": {"h35": {"path":str(FIELD35.relative_to(ROOT)),"sha256":hashlib.sha256(FIELD35.read_bytes()).hexdigest()},
                          "h50": {"path":str(FIELD50.relative_to(ROOT)),"sha256":hashlib.sha256(FIELD50.read_bytes()).hexdigest()}},
        "field_unit": "potential coefficient V per 1 V electrode peak; spatial derivative yields V/m/V; multiplied by U0 gives V/m",
        "voltage_scaling": "linear in applied electrode peak voltage U0; pseudopotential scales U0^2/f^2",
        "frequency_convention": "omega=2*pi*f; phase = omega*t + phi; cos/sin quadratures are voltage quadratures",
        "physical_ion": {"mz_Th": mz, "charge_state": 1, "mass_kg": mass, "charge_C": E_CHARGE, "q_over_m_C_kg": qm},
        "macro_reference": {"charge_C": MACRO_Q, "mass_kg": MACRO_M, "q_over_m_C_kg": MACRO_Q/MACRO_M,
            "B6_macro_J_m6": B6_MACRO, "physical_to_macro_q2_over_m": scale,
            "B6_physical_J_m6": b6_phys, "B6_physical_eV_m6": b6_phys/E_CHARGE,
            "U_at_1mm_eV": b6_phys*1e-18/E_CHARGE,
            "turning_radius_for_3p6meV_m": (0.0036*E_CHARGE/b6_phys)**(1/6)},
        "standing_pattern_cos4theta": pattern,
        "standing_quadrature": "the sine quadrature only; cosine quadrature exactly zero; same 12 electrode geometry",
        "rotating_quadrature": "both cosine and sine quadratures; same U0; leading pseudopotential is their squared-field sum",
        "legacy_voltage_audit": {"unit_field_normalization": "diagnostic_N48.json explicitly says potential quadratures at U0=1 V",
            "v2_runner_source": "outputs/octupole_comparison_home/v2/run_v2.py loads base.fields() and evaluates qm*(cx*cos+sx*sin) without multiplying by 250 V",
            "v1_field_source": "outputs/octupole_comparison_home/v1/run_study.py fields() returns unit_quadratures; integrate_rf() applies charge/mass but no RF voltage amplitude",
            "actual_prior_field_scale_V": 1.0,"claimed_prior_field_scale_V": 250.0,
            "prior_pseudopotential_scale_vs_claim": 1/62500,
            "conclusion": "V1/V2 full-RF trajectories were run at 1 V field scale despite 250 V labels; their RF-off-like energy thresholds do not test the declared 250 V physics."},
        "potential_gradient_finite_difference_max_relative_error": max_fd_rel,
        "field_coefficients": points,
        "local_adiabaticity_and_micromotion": radial_rows,
        "interpretation": "Adiabaticity and micromotion metrics are local diagnostic bounds, not proof of pseudopotential validity; full-RF integration remains the classifier.",
        "v2_source_dependency": "V2 run_v2.py imports v1/run_study.py, absent from the published tracked tree. This runner directly imports tracked src/three_phase_trap/rf_reconstruction.py and reads published N48 JSON only.",
    }
    (OUT / "physical_model_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (OUT / "local_adiabaticity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(radial_rows[0])); w.writeheader(); w.writerows(radial_rows)
    return audit


def test_audit():
    """Tiny regression checks; no long trajectory or FEM work."""
    rec, q = load_fields(FIELD35)
    ang = np.arange(12)*np.pi/6
    target = np.cos(4*ang)
    expected = np.resize([1.,-.5,-.5], 12)
    assert np.max(np.abs(target-expected)) < 2e-15
    assert rec.valid_radius == 8e-3
    # Field and potential must be consistent with E=-grad(Phi) (audit also
    # checks this over both phases and the h50 reconstruction).
    x = np.array([1.2e-3, .8e-3]); h=1e-8
    pc, ps = q.potential_quadratures(np.array([x+[h,0],x-[h,0],x+[0,h],x-[0,h]]))
    ec, es = q.field_quadratures(x[None,:])
    for vals, efield in ((pc,ec[0]),(ps,es[0])):
        grad=np.array([(vals[0]-vals[1])/(2*h),(vals[2]-vals[3])/(2*h)])
        assert np.linalg.norm(grad+efield) <= 2e-6*max(1.,np.linalg.norm(efield))
    m=100_000*AMU
    assert abs(E_CHARGE/m-964.8533215665) < 1e-9
    qmacro=MACRO_Q/(E_CHARGE/m)
    assert abs(MACRO_Q/qmacro-E_CHARGE/m) < 1e-12
    # Fast scalar RHS and public vector evaluator must be identical.
    dc=np.polynomial.polynomial.polyder(q.cosine); ds=np.polynomial.polynomial.polyder(q.sine)
    for xy in ([.7e-3,-1.2e-3],[2.8e-3,1.1e-3],[-4.5e-3,.2e-3]):
        z=complex(*xy)/q.reference_radius
        d0=np.polynomial.polynomial.polyval(z,dc)/q.reference_radius
        d1=np.polynomial.polynomial.polyval(z,ds)/q.reference_radius
        for phi in (0.,.731,2.2):
            fast=250*(E_CHARGE/m)*np.array([-d0.real*math.cos(phi)-d1.real*math.sin(phi),
                                             d0.imag*math.cos(phi)+d1.imag*math.sin(phi)])
            ref=(E_CHARGE/m)*field_values(q,np.asarray(xy)[None,:],phi,250.)[0]
            assert np.allclose(fast,ref,rtol=2e-13,atol=1e-10)
    audit=audit_fields()
    assert audit["potential_gradient_finite_difference_max_relative_error"] < 1e-6
    return {"passed": True, "tests": 6, "gradient_max_rel": audit["potential_gradient_finite_difference_max_relative_error"]}


def ballistic_escape_time(initial, radius=APERTURE):
    x,y,vx,vy=map(float, initial)
    c=x*x+y*y-radius*radius
    if c > 1e-12*radius*radius: raise ValueError("initial point outside aperture")
    a=vx*vx+vy*vy
    if a == 0.: return math.inf
    b=2*(x*vx+y*vy)
    disc=b*b-4*a*c
    if disc < 0 and disc > -1e-12*b*b: disc=0.
    if disc < 0: return math.inf
    roots=((-b-math.sqrt(disc))/(2*a),(-b+math.sqrt(disc))/(2*a))
    valid=[t for t in roots if t>=0]
    return min(valid) if valid else math.inf


def make_initial(mz_th, r0_mm, energy_mev, azimuth, phase, direction, kind="tangent"):
    mass=mz_th*AMU
    speed=math.sqrt(2*energy_mev*1e-3*E_CHARGE/mass)
    r=r0_mm*1e-3
    if kind=="tangent": vr,vt=0.,direction*speed
    elif kind=="radial": vr,vt=direction*speed,0.
    elif kind=="oblique": vr,vt=direction*speed/math.sqrt(2),speed/math.sqrt(2)
    else: raise ValueError(kind)
    return np.array([r*math.cos(azimuth),r*math.sin(azimuth),
                     vr*math.cos(azimuth)-vt*math.sin(azimuth),
                     vr*math.sin(azimuth)+vt*math.cos(azimuth)])


def task_key(job):
    vals=[job.get(k,default) for k,default in (("model",None),("field",None),("voltage_V",None),("frequency_Hz",None),
        ("mz_Th",None),("r0_mm",None),("energy_meV",None),("azimuth_rad",None),("phase_rad",None),
        ("direction",None),("velocity_kind",None),("rtol",2e-9),("max_step_div",50))]
    return "|".join(map(str,vals))


def integrate_segment(job, state, t_start, t_stop, rtol=2e-9, max_step_div=50, sample_per_rf=8):
    rec,q=load_fields(FIELD35 if job["field"]=="h35" else FIELD50)
    # Precompute polynomial derivatives once per trajectory. Calling the
    # public vector evaluator at every RHS invocation re-differentiates both
    # 49-term N48 polynomials and dominated the first benchmark.
    if job["model"]=="rotating":
        dc=np.polynomial.polynomial.polyder(q.cosine); ds=np.polynomial.polynomial.polyder(q.sine)
    elif job["model"]=="standing":
        dc=np.zeros(1,dtype=complex); ds=np.polynomial.polynomial.polyder(q.sine)
    else: raise ValueError(job["model"])
    inv_ref=1.0/q.reference_radius
    U=job["voltage_V"]; omega=2*math.pi*job["frequency_Hz"]
    qm=E_CHARGE/(job["mz_Th"]*AMU)
    def rhs(t,s):
        z=complex(s[0],s[1])*inv_ref
        dcos=np.polynomial.polynomial.polyval(z,dc)*inv_ref
        dsin=np.polynomial.polynomial.polyval(z,ds)*inv_ref
        phase=omega*t+job["phase_rad"]; co=math.cos(phase); si=math.sin(phase)
        return (s[2],s[3],qm*U*(-dcos.real*co-dsin.real*si),
                          qm*U*(dcos.imag*co+dsin.imag*si))
    def aperture(t,s): return APERTURE-math.hypot(s[0],s[1])
    aperture.terminal=True; aperture.direction=-1
    f=job["frequency_Hz"]
    n=max(1,int(math.ceil((t_stop-t_start)*f*sample_per_rf)))
    t_eval=np.linspace(t_start,t_stop,n+1)
    sol=solve_ivp(rhs,(t_start,t_stop),state,method="DOP853",rtol=rtol,
        atol=(2e-13,2e-13,2e-11,2e-11) if rtol>1e-10 else (2e-14,2e-14,2e-12,2e-12),
        max_step=1/(f*max_step_div),t_eval=t_eval,events=aperture)
    if not sol.success: raise RuntimeError(sol.message)
    times=sol.t; states=sol.y.T
    escaped=bool(len(sol.t_events[0]))
    if escaped:
        times=np.r_[times,sol.t_events[0][0]]
        states=np.vstack((states,sol.y_events[0][0]))
    radius=np.hypot(states[:,0],states[:,1])
    # Estimate micromotion as short-window departure from RF-period means.
    return {"times":times,"states":states,"status":"escaped" if escaped else "retained",
        "escape_time_s":float(sol.t_events[0][0]) if escaped else None,
        "max_radius_m":float(np.max(radius)),"rms_radius_m":float(np.sqrt(np.mean(radius**2))),
        "radial_min_m":float(np.min(radius)),"radial_span_m":float(np.ptp(radius)),
        "n_sample":int(len(radius)),"sum_r2_m2":float(np.sum(radius**2)),
        "nfev":int(sol.nfev),"final_state":states[-1].tolist(),"final_time_s":float(times[-1])}


def save_samples(key, model, result, frequency):
    import hashlib
    path=OUT/"trajectories"/(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]+".csv")
    path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists()
    fields=["job_id","model","time_s","rf_period","x_mm","y_mm","r_mm","vx_m_s","vy_m_s"]
    with path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if not exists: w.writeheader()
        for t,s in zip(result["times"],result["states"]):
            w.writerow(dict(job_id=key,model=model,time_s=float(t),rf_period=float(t*frequency),x_mm=float(s[0]*1e3),y_mm=float(s[1]*1e3),r_mm=float(math.hypot(s[0],s[1])*1e3),vx_m_s=float(s[2]),vy_m_s=float(s[3])))


def worker(job):
    initial=make_initial(job["mz_Th"],job["r0_mm"],job["energy_meV"],job["azimuth_rad"],job["phase_rad"],job["direction"],job["velocity_kind"])
    tball=ballistic_escape_time(initial)
    import hashlib
    progress=OUT/"checkpoints"/(hashlib.sha256(task_key(job).encode("utf-8")).hexdigest()[:16]+".json")
    start=time.perf_counter(); state=initial; t0=0.; segments=[]; sampled=[]
    progress_source=progress
    if not progress_source.exists() and job.get("rtol",2e-9)==2e-9 and job.get("max_step_div",50)==50:
        legacy_keys=("model","field","voltage_V","frequency_Hz","mz_Th","r0_mm","energy_meV","azimuth_rad","phase_rad","direction","velocity_kind")
        legacy="|".join(str(job[k]) for k in legacy_keys)
        legacy_path=OUT/"checkpoints"/(hashlib.sha256(legacy.encode("utf-8")).hexdigest()[:16]+".json")
        if legacy_path.exists():progress_source=legacy_path
    if progress_source.exists():
        prev=json.loads(progress_source.read_text(encoding="utf-8"))
        state=np.asarray(prev["final_state"],float);t0=float(prev["final_time_s"]);segments=prev.get("segments",[])
        if prev.get("status")=="escaped":
            final_time=segments[-1]["escape_time_s"]
            return {"job_id":task_key(job),"job":job,"initial_state":initial.tolist(),"ballistic_escape_time_s":tball,
                "ballistic_escape_periods":tball*job["frequency_Hz"],"status":"escaped","escape_time_s":final_time,
                "escape_periods":final_time*job["frequency_Hz"],"max_radius_m":max(x["max_radius_m"] for x in segments),
                "rms_radius_m":math.sqrt(sum(x["sum_r2_m2"] for x in segments)/sum(x["n_sample"] for x in segments)),
                "radial_min_m":min(x["radial_min_m"] for x in segments),"radial_span_m":max(x["max_radius_m"] for x in segments)-min(x["radial_min_m"] for x in segments),
                "nfev":sum(x["nfev"] for x in segments),"wall_s":0.,"segments":segments,"final_state":state.tolist(),"final_time_s":t0}
    horizon=job.get("horizon_RF",1000)
    # Progressive physical-time horizons; preserve the absolute RF clock.
    requested=[p for p in (300,1000,3000,10000) if p<=horizon]
    if not requested or requested[-1] < horizon: requested.append(horizon)
    # Short continuation segments make saved states usable after an orderly
    # deadline stop; also retain the requested milestone horizons exactly.
    period_cuts=sorted(set([min(p,horizon) for p in range(300,horizon+1,300)] + requested))
    cuts=[p/job["frequency_Hz"] for p in period_cuts if p/job["frequency_Hz"]>t0+1e-15]
    final=None
    for t1 in cuts:
        deadline_path=OUT/"integration_deadline.json"
        if deadline_path.exists():
            deadline=json.loads(deadline_path.read_text(encoding="utf-8"))["deadline_unix_s"]
            if time.time()+240 >= deadline: break
        res=integrate_segment(job,state,t0,t1,job.get("rtol",2e-9),job.get("max_step_div",50))
        state=res["final_state"]; t0=res["final_time_s"]
        save_samples(task_key(job),job["model"],res,job["frequency_Hz"])
        sampled.append(res["states"] if not sampled else res["states"][1:])
        segments.append({k:res[k] for k in ("status","escape_time_s","max_radius_m","rms_radius_m","radial_min_m","radial_span_m","n_sample","sum_r2_m2","nfev","final_state","final_time_s")})
        # Per-job atomic progress snapshot: the next run can resume at this
        # absolute time and therefore preserves the RF phase exactly.
        import hashlib
        progress.parent.mkdir(parents=True,exist_ok=True)
        tmp=progress.with_suffix(".tmp")
        tmp.write_text(json.dumps({"job":job,"final_state":state,"final_time_s":t0,"segments":segments,"status":res["status"]},allow_nan=False),encoding="utf-8")
        os.replace(tmp,progress)
        final=res
        if res["status"]=="escaped": break
    if final is None:
        status="paused"; escape_time=None
    else:
        status=final["status"] if t0 >= horizon/job["frequency_Hz"] or final["status"]=="escaped" else "paused"
        escape_time=final["escape_time_s"]
    n_total=sum(x["n_sample"] for x in segments)
    max_r=max(x["max_radius_m"] for x in segments);min_r=min(x["radial_min_m"] for x in segments)
    return {"job_id":task_key(job),"job":job,"initial_state":initial.tolist(),"ballistic_escape_time_s":tball,
        "ballistic_escape_periods":tball*job["frequency_Hz"],"status":status,
        "escape_time_s":escape_time,"escape_periods":escape_time*job["frequency_Hz"] if escape_time is not None else None,
        "max_radius_m":max_r,"rms_radius_m":math.sqrt(sum(x["sum_r2_m2"] for x in segments)/n_total),"radial_min_m":min_r,
        "radial_span_m":max_r-min_r,"nfev":sum(x["nfev"] for x in segments),"wall_s":time.perf_counter()-start,
        "segments":segments,"final_state":state,"final_time_s":t0,"solver":"DOP853; max_step=T/50; rtol=2e-9"}


def read_journal():
    path=OUT/"trajectory_checkpoint.jsonl"; rows={}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row=json.loads(line)
                if row.get("status") in ("retained","escaped"): rows[row["job_id"]]=row
            except Exception: pass
    return rows


def run_jobs(jobs, workers=4):
    OUT.mkdir(parents=True,exist_ok=True)
    deadline_file=OUT/"integration_deadline.json"
    if not deadline_file.exists():
        # This is called immediately before first RF integration.
        started=time.time()
        deadline_file.write_text(json.dumps({"start_unix_s":started,"deadline_unix_s":started+DEADLINE_S,
            "budget_s":DEADLINE_S,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(started))},indent=2)+"\n",encoding="utf-8")
    deadline=json.loads(deadline_file.read_text(encoding="utf-8"))["deadline_unix_s"]
    done=read_journal()
    todo=[]
    for j in jobs:
        old=done.get(task_key(j))
        target=j["horizon_RF"]/j["frequency_Hz"]
        if old is None or (old.get("status")!="escaped" and old.get("final_time_s",0.)+1e-15 < target):
            todo.append(j)
    if time.time()+120>deadline:
        raise TimeoutError("Hard deadline guard: no new job submitted with <120 s remaining")
    if not todo: return {"submitted":0,"already_done":len(jobs)}
    attempts=0; t0=time.perf_counter(); journal=OUT/"trajectory_checkpoint.jsonl"
    os.environ.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    with ProcessPoolExecutor(max_workers=workers) as pool, journal.open("a",encoding="utf-8") as f:
        futures={pool.submit(worker,j):j for j in todo}
        for future in as_completed(futures):
            if time.time()+120>deadline:
                for other in futures: other.cancel()
                break
            job=futures[future]
            try: row=future.result()
            except Exception as exc:
                row={"job_id":task_key(job),"job":job,"status":"numerical_failure","error":repr(exc),"traceback":traceback.format_exc()}
            f.write(json.dumps(row,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno()); attempts+=1
    result={"requested":len(jobs),"new_completed":attempts,"existing":len(done),"wall_s":time.perf_counter()-t0,
            "throughput_trajectories_min":attempts/max(1e-12,time.perf_counter()-t0)*60,"workers":workers,
            "seconds_until_deadline":deadline-time.time(),"completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    with (OUT/"runtime_log.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(result,allow_nan=False)+"\n")
    return result


def pilot_jobs():
    # Energies chosen so the RF-off flight is decisively shorter than the
    # 1000-period observation horizon. Matched orientation/phase/direction set.
    settings=[("base_250V_100k",250.,100000.,("h35",))]
    launches=[]
    for r in (1.,2.,3.):
        for az in (0.,math.pi/4):
            for phase in (0.,math.pi/2):
                for direction in (-1.,1.):
                    launches.append((r,az,phase,direction))
    jobs=[]
    for name,u,f,fields in settings:
        energy=10.0
        for r,az,phase,direction in launches:
            for model in ("rotating","standing"):
                jobs.append(dict(model=model,field=fields[0],voltage_V=u,frequency_Hz=f,mz_Th=100000.,r0_mm=r,
                    energy_meV=energy,azimuth_rad=az,phase_rad=phase,direction=direction,velocity_kind="tangent",horizon_RF=1000,
                    normalization="equal_peak_voltage",stage="baseline_pilot"))
    return jobs


def expand_jobs():
    # Compact physical coupling bracket after baseline pilot. Same 10 meV
    # shell and entrance states, with coupling changed by U^2/f^2.
    jobs=[]
    launch=[(0.,0.,1.),(math.pi/4,math.pi/2,-1.),(0.,math.pi/2,-1.),(math.pi/4,0.,1.)]
    # Baseline equal-peak energy/radius boundaries; the 5 meV scan tests the
    # nonmonotonic region adjacent to the V2 transition, while 20 meV explores
    # the predicted finite-depth acceptance at larger radii.
    for r in (.5,1.,2.,3.):
        for energy in (3.75,5.,20.,50.,100.):
            for az,phase,direction in launch:
                for model in ("rotating","standing"):
                    jobs.append(dict(model=model,field="h35",voltage_V=250.,frequency_Hz=100000.,mz_Th=100000.,r0_mm=r,
                        energy_meV=energy,azimuth_rad=az,phase_rad=phase,direction=direction,velocity_kind="tangent",horizon_RF=1000,
                        normalization="equal_peak_voltage",stage="equal_peak_energy_radius"))
    # Extend only the inner two radii into the high-energy regime where
    # pseudopotential predictions approach the finite 8 mm aperture.
    for r in (1.,2.):
        for energy in (200.,400.):
            for az,phase,direction in launch:
                for model in ("rotating","standing"):
                    jobs.append(dict(model=model,field="h35",voltage_V=250.,frequency_Hz=100000.,mz_Th=100000.,r0_mm=r,
                        energy_meV=energy,azimuth_rad=az,phase_rad=phase,direction=direction,velocity_kind="tangent",horizon_RF=1000,
                        normalization="equal_peak_voltage",stage="inner_high_energy_boundary"))
    # Probe the near-aperture/high-energy end where local adiabaticity worsens;
    # these are exploratory full-RF boundary jobs, not pseudopotential claims.
    for r in (1.,2.):
        for energy in (1000.,2000.,3000.):
            for az,phase,direction in launch:
                for model in ("rotating","standing"):
                    jobs.append(dict(model=model,field="h35",voltage_V=250.,frequency_Hz=100000.,mz_Th=100000.,r0_mm=r,
                        energy_meV=energy,azimuth_rad=az,phase_rad=phase,direction=direction,velocity_kind="tangent",horizon_RF=1000,
                        normalization="equal_peak_voltage",stage="near_aperture_boundary"))
    # Radial and oblique launches at the 10 meV shell, with both radial signs.
    for r in (1.,2.):
        for kind in ("radial","oblique"):
            for az,phase in ((0.,0.),(math.pi/4,math.pi/2)):
                for direction in (-1.,1.):
                    for model in ("rotating","standing"):
                        jobs.append(dict(model=model,field="h35",voltage_V=250.,frequency_Hz=100000.,mz_Th=100000.,r0_mm=r,
                            energy_meV=10.,azimuth_rad=az,phase_rad=phase,direction=direction,velocity_kind=kind,horizon_RF=1000,
                            normalization="equal_peak_voltage",stage="velocity_direction"))
    # Coupling bracket around baseline: 250 V/200 kHz (1/4 pseudo strength)
    # and 500 V/100 kHz (4x); deliberately only two matched orientations/radius.
    for label,U,f in (("weak_250V_200k",250.,200000.),("strong_500V_100k",500.,100000.)):
        for r in (1.,2.,3.):
            for az,phase,direction in launch[:2]:
                for model in ("rotating","standing"):
                    jobs.append(dict(model=model,field="h35",voltage_V=U,frequency_Hz=f,mz_Th=100000.,r0_mm=r,
                        energy_meV=10.,azimuth_rad=az,phase_rad=phase,direction=direction,velocity_kind="tangent",horizon_RF=1000,
                        normalization="equal_peak_voltage",stage=label))
    # Separate matched-leading-pseudopotential protocol (standing at sqrt(2)U),
    # using all eight baseline azimuth/phase/direction states at each radius.
    for r in (1.,2.,3.):
        for az in (0.,math.pi/4):
            for phase in (0.,math.pi/2):
                for direction in (-1.,1.):
                    jobs.append(dict(model="standing",field="h35",voltage_V=250.*math.sqrt(2),frequency_Hz=100000.,mz_Th=100000.,r0_mm=r,
                        energy_meV=10.,azimuth_rad=az,phase_rad=phase,direction=direction,velocity_kind="tangent",horizon_RF=1000,
                        normalization="matched_pseudopotential_at_0p5mm",stage="matched_psi"))
    return jobs


def convergence_jobs(existing):
    jobs=[]
    eligible=[x for x in existing if x.get("status") in ("retained","escaped")
        and x.get("job",{}).get("field")=="h35" and x.get("job",{}).get("voltage_V")==250.
        and x.get("job",{}).get("frequency_Hz")==100000. and x.get("job",{}).get("horizon_RF")==1000
        and x.get("job",{}).get("velocity_kind")=="tangent"]
    # Strictly rerun two phase/azimuth high-energy escape cases and their exact
    # rotating/standing counterparts, then a moderate-energy confined pair.
    escapes=[x for x in eligible if x["status"]=="escaped" and x["job"]["energy_meV"]>=1000]
    escapes.sort(key=lambda x:(x["job"]["energy_meV"],x["job"]["r0_mm"],x["job"]["azimuth_rad"]))
    used=set()
    for row in escapes:
        j0=row["job"]
        launch_key=tuple(j0[k] for k in ("r0_mm","energy_meV","azimuth_rad","phase_rad","direction"))
        if launch_key in used:continue
        pair=[x for x in eligible if tuple(x["job"][k] for k in ("r0_mm","energy_meV","azimuth_rad","phase_rad","direction"))==launch_key]
        if {x["job"]["model"] for x in pair}!={"rotating","standing"}:continue
        for x in pair:
            j=dict(x["job"]);j.update(rtol=2e-11,max_step_div=100,horizon_RF=1000,stage="tight_control_high_energy")
            jobs.append(j)
        used.add(launch_key)
        if len(jobs)>=4:break
    if len(jobs)<4:
        for x in eligible:
            j0=x["job"]
            if (j0["energy_meV"],j0["r0_mm"],j0["azimuth_rad"],j0["phase_rad"],j0["direction"],j0["model"]) not in ((10.,2.,0.,0.,1.,"rotating"),(10.,2.,0.,0.,1.,"standing")):continue
            j=dict(j0);j.update(rtol=2e-11,max_step_div=100,horizon_RF=1000,stage="tight_control_moderate")
            jobs.append(j)
    # Keep exactly six distinct controls where available.
    return jobs[:6]


def h50_jobs(existing):
    jobs=[]
    rows=[x for x in existing if x.get("status") in ("retained","escaped") and x.get("job",{}).get("field")=="h35" and x.get("job",{}).get("horizon_RF",0)>=1000]
    targets=((10.,2.,0.,0.,1.),(10.,2.,math.pi/4,math.pi/2,-1.),
             (400.,2.,0.,0.,1.),(3.75,1.,0.,0.,1.))
    for energy,r,az,phase,direction in targets:
        for model in ("rotating","standing"):
            row=next((x for x in rows if x["job"]["model"]==model and x["job"]["voltage_V"]==250.
                and x["job"]["frequency_Hz"]==100000. and x["job"]["r0_mm"]==r
                and x["job"]["energy_meV"]==energy and x["job"]["azimuth_rad"]==az
                and x["job"]["phase_rad"]==phase and x["job"]["direction"]==direction
                and x["job"]["velocity_kind"]=="tangent"),None)
            if row:
                j=dict(row["job"]);j.update(field="h50",stage="h50_field_crosscheck")
                jobs.append(j)
    return jobs


def long_jobs(existing):
    jobs=[]
    # Two matched initial states for both models: one representative central
    # launch and one phase/azimuth/direction perturbation.
    wanted={(2.,0.,0.,1.),(2.,math.pi/4,math.pi/2,-1.)}
    for r,az,phase,direction in sorted(wanted):
        for model in ("rotating","standing"):
            found=next((x for x in existing if x.get("job",{}).get("model")==model
                and x.get("job",{}).get("field")=="h35" and x.get("job",{}).get("voltage_V")==250.
                and x.get("job",{}).get("r0_mm")==r and x.get("job",{}).get("energy_meV")==10.
                and x.get("job",{}).get("azimuth_rad")==az and x.get("job",{}).get("phase_rad")==phase
                and x.get("job",{}).get("direction")==direction),None)
            if found:
                j=dict(found["job"]);j.update(horizon_RF=10000,stage="long_horizon")
                jobs.append(j)
    return jobs


def benchmark6_jobs():
    # Six distinct paired launches also add an azimuth/phase sensitivity check.
    jobs=[]
    for r in (.5,1.,3.):
        for model in ("rotating","standing"):
            jobs.append(dict(model=model,field="h35",voltage_V=250.,frequency_Hz=100000.,mz_Th=100000.,r0_mm=r,
                energy_meV=10.,azimuth_rad=math.pi/2,phase_rad=math.pi/4,direction=1.,velocity_kind="tangent",horizon_RF=1000,
                normalization="equal_peak_voltage",stage="six_worker_benchmark"))
    return jobs


def main():
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("audit","test","pilot","expand","verify","h50","long","benchmark6"));ap.add_argument("--workers",type=int,default=4);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=="audit":
        print(json.dumps(audit_fields(),indent=2));return
    if args.mode=="test": print(json.dumps(test_audit(),indent=2));return
    if args.mode=="pilot": jobs=pilot_jobs()
    elif args.mode=="expand": jobs=expand_jobs()
    elif args.mode=="h50": jobs=h50_jobs(list(read_journal().values()))
    elif args.mode=="long": jobs=long_jobs(list(read_journal().values()))
    elif args.mode=="benchmark6": jobs=benchmark6_jobs()
    else: jobs=convergence_jobs(list(read_journal().values()))
    print(json.dumps(run_jobs(jobs,args.workers),indent=2),flush=True)

if __name__=="__main__":
    main()
