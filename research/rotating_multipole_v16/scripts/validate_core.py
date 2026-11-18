from __future__ import annotations

import json, math, os, platform, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from v16.models import Field, equilibrium_spectrum
from v16.integrate import integrate_dop853, integrate_ensemble
from v16.rigid import rod_potential_force_torque

OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)
T0=time.perf_counter()

def memory_status():
    if os.name!="nt": return None
    import ctypes,struct
    buf=ctypes.create_string_buffer(64); struct.pack_into('<I',buf,0,64)
    k=ctypes.WinDLL('kernel32'); k.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(ctypes.c_char)]
    k.GlobalMemoryStatusEx.restype=ctypes.c_int
    ok=k.GlobalMemoryStatusEx(buf)
    if not ok: return None
    load,total,avail=struct.unpack_from('<IQQ',buf.raw,4)
    return {"total_gib":total/2**30,"available_gib":avail/2**30,"load_pct":load}

def main():
    # A direct high-accuracy check of J=H-Lz/4 and the autonomous rotating frame.
    f=Field("RO",kappa=.1044,delta_ro=.37)
    y0=np.array([.09,.03,2e-4,-1e-4])
    sol=integrate_dop853(y0,f,rf_count=80,rtol=2e-12,atol=2e-14)
    ts=np.linspace(0,sol.t[-1],801); ys=sol.sol(ts).T
    from v16.models import rotating_invariant
    j=np.array([rotating_invariant(y,t,f) for y,t in zip(ys,ts)])
    # Fixed-step benchmark, short and bounded initial states.
    cloud=np.tile(np.array([.1,.05,1e-4,-1e-4]),(256,1))
    tic=time.perf_counter()
    bench=integrate_ensemble(cloud,Field("RO",kappa=.1044),rf_count=30,steps_per_rf=96)
    bench_s=time.perf_counter()-tic
    # Equilibrium spectrum from the full Coriolis-coupled linearization.
    eq=equilibrium_spectrum(.1044)
    # Smooth-rod center force and multipole torque check.
    u,fx,fy,tau=rod_potential_force_torque(0,0,.31,0.7,Field("RO",kappa=.5),.2,"uniform",32)
    report={"timestamp_local":time.strftime('%Y-%m-%d %H:%M:%S %z'),
            "host":platform.node(),"os":platform.platform(),"python":platform.python_version(),
            "logical_cpus":os.cpu_count(),"memory":memory_status(),
            "integrator":"DOP853 reference; vectorized classical RK4 scout",
            "checks":{"RO_J_max_abs_drift":float(np.max(np.abs(j-j[0]))),
                      "J_relative_range":float(np.ptp(j)/max(abs(j[0]),1e-300)),
                      "ensemble_256x30rf_s":bench_s,
                      "ensemble_bounded_fraction":float(np.mean(np.isnan(bench['exit_tau']))),
                      "stationary_rstar":eq['radius'],
                      "stationary_eigenvalues":[[float(x.real),float(x.imag)] for x in eq['lambda']],
                      "centered_rod_force_norm":float(math.hypot(fx,fy)),
                      "centered_rod_torque":float(tau),
                      "centered_rod_potential":float(u)},
            "elapsed_s":time.perf_counter()-T0}
    (OUT/"validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
