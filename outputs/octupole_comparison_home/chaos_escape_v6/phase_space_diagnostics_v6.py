"""Jacobi-invariant, stroboscopic, and finite-time tangent diagnostics.

Run only after the V6 phase-zero scans and localized refinement. Every new
trajectory uses the shared V6 global deadline and a completed-job checkpoint.
"""
from __future__ import annotations
import csv, json, math, os, sys, time, hashlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run_v6

OUT = HERE / "phase_space"
LENGTH_SCALE = 1e-3
CHUNK_RF = 10
DEADLINE_RESERVE_S = 75
_Q = {}


def _init():
    run_v6._init_worker()
    global _Q, _JAC_ERR
    _Q = {("N48", "rotating"): run_v6._Q_N48,
          ("N48", "standing"): run_v6._Q_N48,
          ("pure_m4", "rotating"): run_v6._Q_PURE["rotating"],
          ("pure_m4", "standing"): run_v6._Q_PURE["standing"]}
    _JAC_ERR = _validate_jacobian()


def _complete_rows():
    p = run_v6.OUT / "trajectory_checkpoint.jsonl"
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try: rows.append(json.loads(line))
        except Exception: pass
    return rows


def _transition_angles(rep="N48", model="standing", direction=1):
    grouped = {}
    for r in _complete_rows():
        j = r.get("job", {})
        if j.get("representation") == rep and j.get("model") == model and int(j.get("direction", 0)) == direction and float(j.get("phase_deg", -999)) == 0:
            a = float(j["azimuth_deg"])
            if 0 <= a <= 22.5:
                grouped[a] = r.get("status")
    points = sorted(grouped.items())
    return [(a, b) for (a, sa), (b, sb) in zip(points, points[1:]) if sa != sb]


def select_jobs():
    # Keep the tangent calculation focused so the primary scan, phase checks,
    # and report remain inside the fixed wall-clock budget.
    selected = [
        ("deep_rotating_jacobi", "pure_m4", "rotating", 0.0),
        ("N48_boundary_standing", "N48", "standing", 10.283203125),
        ("N48_early_escape_rotating", "N48", "rotating", 22.5),
        ("N48_early_escape_standing", "N48", "standing", 22.5),
        ("N48_late_escape_standing", "N48", "standing", 17.2265625),
    ]
    jobs = []
    for label, rep, model, az in selected:
        j = run_v6._make_job("phase_space", rep, model, az, 0.0, 1,
                             sampling="targeted_stroboscopic_and_tangent")
        j["diagnostic_label"] = label
        j["job_id"] = hashlib.sha256(json.dumps({k:v for k,v in j.items() if k not in ("stage", "sampling_design", "diagnostic_label")},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
        jobs.append(j)
    unique = {}
    for j in jobs: unique[j["job_id"]] = j
    return list(unique.values()), {"selected_conditions": selected,
        "selection_reason": "minimum set for Jacobi conservation and matched N48/pure-m4 escape comparison"}


def _acceleration_jacobian(q, model, point, lam, qm, voltage):
    z = complex(point[0], point[1]) / q.reference_radius
    if model == "rotating":
        c2 = np.polynomial.polynomial.polyval(z, np.polynomial.polynomial.polyder(q.cosine, 2)) / q.reference_radius**2
        s2 = np.polynomial.polynomial.polyval(z, np.polynomial.polynomial.polyder(q.sine, 2)) / q.reference_radius**2
    else:
        c2 = 0j
        s2 = np.polynomial.polynomial.polyval(z, np.polynomial.polynomial.polyder(q.sine, 2)) / q.reference_radius**2
    mat_c = np.array([[-c2.real, c2.imag], [c2.imag, c2.real]])
    mat_s = np.array([[-s2.real, s2.imag], [s2.imag, s2.real]])
    return qm * voltage * (mat_c * math.cos(lam) + mat_s * math.sin(lam))


def _validate_jacobian():
    errs=[]
    for (rep, model), q in _Q.items():
        xy=np.array([1.71e-3,-0.62e-3]); h=2e-9; phase=0.731
        c,s=q.field_quadratures(xy[None,:]); field=(c*math.cos(phase)+s*math.sin(phase))[0]
        jc,js=q.jacobian_quadratures(xy[None,:]); jac=(jc*math.cos(phase)+js*math.sin(phase))[0]
        fd=np.zeros((2,2))
        for k in range(2):
            dp=np.zeros(2);dp[k]=h
            cp,sp=q.field_quadratures((xy+dp)[None,:]); cm,sm=q.field_quadratures((xy-dp)[None,:])
            fd[:,k]=((cp*math.cos(phase)+sp*math.sin(phase))-(cm*math.cos(phase)+sm*math.sin(phase)))[0]/(2*h)
        errs.append(float(np.linalg.norm(fd-jac)/max(1.,np.linalg.norm(jac))))
    return max(errs)


def tangent_job(job):
    rep, model = job["representation"], job["model"]
    q = _Q[(rep, model)]
    if model == "rotating":
        dcos=np.polynomial.polynomial.polyder(q.cosine); dsin=np.polynomial.polynomial.polyder(q.sine)
    else:
        dcos=np.zeros(1,dtype=complex); dsin=np.polynomial.polynomial.polyder(q.sine)
    mass=job["mz_Th"]*run_v6.run_v4.AMU; charge=run_v6.run_v4.E_CHARGE; qm=charge/mass
    omega=2*math.pi*job["frequency_Hz"]; voltage=job["voltage_V"]; invref=1/q.reference_radius
    phase0=math.radians(job["phase_deg"]); az=math.radians(job["azimuth_deg"])
    base=run_v6.run_v4.make_initial(job["mz_Th"],job["r0_mm"],job["energy_eV"]*1000,
        az,phase0,float(job["direction"]),"tangent")
    vscale=math.sqrt(2*job["energy_eV"]*charge/mass)
    tangent=np.array([1.,0.61803398875,0.38196601125,-0.27182818285]);tangent/=np.linalg.norm(tangent)
    state=np.r_[base,tangent]
    T=1/job["frequency_Hz"]; horizon=job["horizon_RF"]*T; t=0.
    strobe=[(0.0,*map(float,base))]; growth=[(0.0,0.0)]; logsum=0.; escaped=False

    def rhs(t,y):
        x=y[:2]; vx=y[2:4]; z=complex(x[0],x[1])*invref
        dc=np.polynomial.polynomial.polyval(z,dcos)*invref
        ds=np.polynomial.polynomial.polyval(z,dsin)*invref
        lam=omega*t+phase0; co,si=math.cos(lam),math.sin(lam)
        acc=qm*voltage*np.array([-dc.real*co-ds.real*si,dc.imag*co+ds.imag*si])
        ja=_acceleration_jacobian(q,model,x,lam,qm,voltage)
        w=y[4:]
        wd=np.r_[vscale/LENGTH_SCALE*w[2:], (LENGTH_SCALE/vscale)*(ja@w[:2])]
        return np.r_[vx,acc,wd]
    def aperture(_t,y): return run_v6.APERTURE_M-math.hypot(y[0],y[1])
    aperture.terminal=True;aperture.direction=-1
    started=time.perf_counter()
    while t < horizon - 0.25*T:
        end=min(t+CHUNK_RF*T,horizon)
        t_eval=np.arange(t+T,end+0.25*T,T)
        # arange can return an endpoint a few ulps beyond the integration
        # interval. Clip only that round-off and append the chunk endpoint so
        # the base state and tangent are always advanced to exactly ``end``.
        t_eval=t_eval[t_eval<=end+1e-10*T]
        t_eval=np.minimum(t_eval,end)
        if len(t_eval)==0 or end-t_eval[-1]>1e-9*T: t_eval=np.r_[t_eval,end]
        t_eval=np.unique(t_eval)
        # The augmented state has four dimensionless tangent components in
        # addition to the physical (x, y, vx, vy) state. Give those tangent
        # components their own scalar absolute tolerance; passing the four
        # physical-state tolerances directly would fail to broadcast.
        aug_atol=np.r_[np.asarray(run_v6.ATOL,dtype=float),np.full(4,2e-12)]
        sol=solve_ivp(rhs,(t,end),state,method="DOP853",rtol=run_v6.RTOL,atol=aug_atol,
            max_step=T/run_v6.MAX_STEP_DIV,t_eval=t_eval,events=aperture)
        if not sol.success: raise RuntimeError(sol.message)
        for tt, yy in zip(sol.t,sol.y.T):
            strobe.append((float(tt*job["frequency_Hz"]),*map(float,yy[:4])))
        if len(sol.t_events[0]):
            te=float(sol.t_events[0][0]); ye=sol.y_events[0][0]
            if not strobe or abs(strobe[-1][0]-te*job["frequency_Hz"])>1e-9:
                strobe.append((te*job["frequency_Hz"],*map(float,ye[:4])))
            state=ye;t=te;escaped=True
        else:
            state=sol.y[:,-1];t=float(sol.t[-1]) if len(sol.t) else end
        wn=float(np.linalg.norm(state[4:]))
        if wn==0 or not math.isfinite(wn): raise FloatingPointError("invalid tangent norm")
        logsum+=math.log(wn);state[4:]/=wn
        growth.append((t*job["frequency_Hz"],logsum))
        if escaped: break
    elapsed_rf=t*job["frequency_Hz"]
    elapsed_s=t
    ftle_rf=logsum/max(elapsed_rf,1e-30)
    ftle_s=logsum/max(elapsed_s,1e-30)
    energy_lz=[]
    for row in strobe:
        rf,*sv=row;tt=rf/job["frequency_Hz"];x,y,vx,vy=sv;lam=omega*tt+phase0
        pc,ps=q.potential_quadratures(np.array([[x,y]]))
        phi=voltage*(pc[0]*math.cos(lam)+ps[0]*math.sin(lam))
        ham=.5*mass*(vx*vx+vy*vy)+charge*phi
        lz=mass*(x*vy-y*vx)
        energy_lz.append((float(rf),float(ham/charge),float(lz)))
    hvals=[x[1] for x in energy_lz];lvals=[x[2] for x in energy_lz]
    result={"job":job,"job_id":job["job_id"],"status":"escaped" if escaped else "retained",
        "escape_time_RF":elapsed_rf if escaped else None,"right_censored_at_RF":None if escaped else job["horizon_RF"],
        "finite_time_lyapunov_per_RF":ftle_rf,"finite_time_lyapunov_per_s":ftle_s,
        "tangent_position_scale_m":LENGTH_SCALE,"tangent_velocity_scale_m_s":vscale,
        "tangent_renormalization_interval_RF":CHUNK_RF,"log_growth_sum":logsum,
        "tangent_jacobian_validation_relative_error":_JAC_ERR,
        "instantaneous_H_Lz":{"H_range_eV":[min(hvals),max(hvals)],"H_peak_to_peak_eV":max(hvals)-min(hvals),
            "Lz_range_kg_m2_s":[min(lvals),max(lvals)],"Lz_peak_to_peak_kg_m2_s":max(lvals)-min(lvals),
            "H_start_eV":hvals[0],"H_end_eV":hvals[-1],"Lz_start_kg_m2_s":lvals[0],"Lz_end_kg_m2_s":lvals[-1]},
        "hamiltonian_lz_history":energy_lz,
        "wall_s":time.perf_counter()-started,"strobe":strobe,"growth":growth}
    if rep=="pure_m4" and model=="rotating":
        ks=[(rf,h-(omega/4)*lz/charge,h,lz) for rf,h,lz in energy_lz]
        k0=ks[0][1]; drift=max(abs(x[1]-k0) for x in ks)
        scale=max(abs(k0),job["energy_eV"],1e-30)
        result["jacobi"]={"K_definition":"H - Omega*Lz/4","absolute_max_drift_eV":drift,
            "relative_max_drift_vs_abs_K0":drift/scale,"initial_K_eV":k0,
            "initial_H_eV":ks[0][2],"initial_Lz_kg_m2_s":ks[0][3],
            "samples":[[float(a),float(b),float(c),float(d)] for a,b,c,d in ks],
            "instantaneous_H_range_eV":[min(hvals),max(hvals)],"Lz_range_kg_m2_s":[min(lvals),max(lvals)]}
    return result


def _load_diag_done():
    p=OUT/"phase_space_checkpoint.jsonl"; out={}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r=json.loads(line);out[r["job_id"]]=r
            except Exception: pass
    return out


def _worker(job): return tangent_job(job)


def _save_result(r):
    OUT.mkdir(parents=True,exist_ok=True)
    trace=OUT/"traces"/(r["job_id"]+".npz");trace.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trace,strobe=np.asarray(r["strobe"],float),growth=np.asarray(r["growth"],float),
        hamiltonian_lz=np.asarray(r["hamiltonian_lz_history"],float))
    brief={k:v for k,v in r.items() if k not in ("strobe","growth","hamiltonian_lz_history")}
    (OUT/"completed").mkdir(exist_ok=True)
    tmp=OUT/"completed"/(r["job_id"]+".json.tmp");tmp.write_text(json.dumps(brief,allow_nan=False,indent=2)+"\n",encoding="utf-8");os.replace(tmp,tmp.with_suffix(""))
    with (OUT/"phase_space_checkpoint.jsonl").open("a",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(brief,allow_nan=False,separators=(",",":"))+"\n");f.flush();os.fsync(f.fileno())
    print(f"phase-space {r['job']['representation']} {r['job']['model']} {r['job']['diagnostic_label']} "
          f"az={r['job']['azimuth_deg']:.6g}: {r['status']} FTLE={r['finite_time_lyapunov_per_RF']:.5g}/RF "
          f"wall={r['wall_s']:.1f}s",flush=True)


def run():
    global _JAC_ERR
    _init()
    if _JAC_ERR>1e-6: raise AssertionError(f"field Jacobian check failed: {_JAC_ERR}")
    jobs,selection=select_jobs();done=_load_diag_done()
    todo=[j for j in jobs if j["job_id"] not in done]
    deadline=json.loads((run_v6.OUT/"integration_deadline.json").read_text())["deadline_unix_s"]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"selection.json").write_text(json.dumps({"selection":selection,"jobs":jobs,
        "jacobian_validation_relative_error":_JAC_ERR},indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with ProcessPoolExecutor(max_workers=4,initializer=_init) as pool:
        pending={};idx=0
        while idx<len(todo) or pending:
            while idx<len(todo) and len(pending)<4:
                if time.time()+DEADLINE_RESERVE_S>=deadline:
                    print("global V6 deadline reserve reached; no further phase-space trajectories submitted",flush=True);idx=len(todo);break
                j=todo[idx];pending[pool.submit(_worker,j)]=j;idx+=1
            if not pending:break
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in ready:
                pending.pop(f);r=f.result();_save_result(r)
    print(f"phase-space jobs completed={len(_load_diag_done())} requested={len(jobs)}",flush=True)


if __name__=="__main__": run()
