"""Smooth-field benchmarks for the ABC validation, independent of a FEM mesh."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import beta

from .config import ParticleConfig
from .dynamics import integrate_trajectory
from .rf_reconstruction import HarmonicQuadratures
from .micromotion import matched_full_state, recover_slow_state
from .slow_analysis import centered_period_mean



def ideal_field(order, amplitude):
    c=np.zeros(order+1,dtype=complex)
    s=c.copy()
    if order==2:
        c[order],s[order]=amplitude,-1j*amplitude
    elif order==4:
        c[order],s[order]=1j*amplitude,amplitude
    else:
        raise ValueError("Benchmark order must be 2 or 4")
    return HarmonicQuadratures(c,s,1.0)


def quadrupole_exact(time, initial, amplitude, omega, phase, charge=1.0, mass=1.0):
    """Independent rotating-frame constant-coefficient matrix exponential."""
    J=np.array([[0.,-1.],[1.,0.]])
    w=omega/2
    k=2*charge*amplitude/mass
    M=np.block([[np.zeros((2,2)),np.eye(2)],
                [w*w*np.eye(2)-k*np.diag([1.,-1.]),-2*w*J]])
    def rotation(a):
        return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    u0=rotation(-phase/2)@initial[:2]
    du0=rotation(-phase/2)@initial[2:]-w*J@u0
    out=[]
    for t in np.asarray(time):
        state=expm(M*t)@np.r_[u0,du0]
        Q=rotation(w*t+phase/2)
        out.append(np.r_[Q@state[:2],Q@(state[2:]+w*J@state[:2])])
    return np.asarray(out)


def _metrics(actual,reference,scale):
    error=np.linalg.norm(actual-reference,axis=-1)/scale
    return {"relative_rms":float(np.sqrt(np.mean(error**2))),"relative_max":float(error.max())}


def run_benchmarks(output: Path):
    output.mkdir(parents=True,exist_ok=True)
    particle=ParticleConfig(mass=1.,charge=1.,gamma=0.,gravity_enabled=False)
    records=[]
    rng=np.random.default_rng(20261023)
    checks=[]
    for order,base_amplitude in ((2,2.0),(4,5/3)):
        omega0=20.
        base=ideal_field(order,base_amplitude)
        R0=np.array([1.,.1]) if order==2 else np.array([1.,0.])
        V0=np.array([0.,.03]) if order==2 else np.zeros(2)
        scale=float(np.linalg.norm(R0))
        if order==2:
            slow_period=2*np.pi/(2*base_amplitude/omega0)
        else:
            B6=8*base_amplitude**2/omega0**2
            slow_period=4*np.sqrt(1/(2*B6))/scale**2*float(beta(1/6,1/2)/6)
        T0=2*np.pi/omega0
        duration=math.ceil(slow_period/T0)*T0
        points=rng.uniform(-.8,.8,(17,2))
        phases=np.arange(512)*2*np.pi/512
        ec,es=base.field_quadratures(points)
        e=ec[None]*np.cos(phases)[:,None,None]+es[None]*np.sin(phases)[:,None,None]
        numeric=np.mean(np.sum(e*e,axis=-1),axis=0)/(2*omega0**2)
        expected=(2*base_amplitude**2/omega0**2*np.sum(points*points,axis=1)
                  if order==2 else 8*base_amplitude**2/omega0**2*np.sum(points*points,axis=1)**3)
        averaging_error=float(np.max(abs(numeric-expected))/np.max(expected))
        eps=1e-6
        fd=np.column_stack([-(base.pseudo(points+eps*np.eye(2)[j],1,1,omega0)-base.pseudo(points-eps*np.eye(2)[j],1,1,omega0))/(2*eps) for j in range(2)])
        force_error=float(np.max(abs(fd-base.force(points,1,1,omega0)))/np.max(abs(fd)))
        assert averaging_error<1e-12 and force_error<1e-8
        checks.append({"order":order,"fixed_point_time_average_relative_error":averaging_error,
                       "force_vs_energy_gradient_relative_error":force_error})
        def rhs(t,y):
            r=y[:2]
            coefficient=(4*base_amplitude**2/omega0**2 if order==2 else
                         48*base_amplitude**2/omega0**2*np.dot(r,r)**2)
            return np.r_[y[2:],-coefficient*r]
        slow=solve_ivp(rhs,(0,duration),np.r_[R0,V0],rtol=2e-12,atol=2e-14,
                       method="DOP853",max_step=slow_period/150,dense_output=True)
        assert slow.success
        probe=slow.sol(np.linspace(0,duration,1001)).T
        energy=.5*np.sum(probe[:,2:]**2,axis=1)+base.pseudo(probe[:,:2],1,1,omega0)
        energy_drift=float(np.max(abs(energy-energy[0]))/energy[0])
        assert energy_drift<1e-9
        for phase in (0.,np.pi/2):
            last_rms=None
            for multiplier in (1,2,4):
                omega=omega0*multiplier
                amplitude=base_amplitude*multiplier
                model=ideal_field(order,amplitude)
                initial=matched_full_state(model,R0,V0,1,1,omega,phase)
                inverse=recover_slow_state(model,initial,1,1,omega,phase)
                ic_error=float(np.max(abs(inverse-np.r_[R0,V0])))
                assert ic_error<1e-11
                # Scalar polynomial callback avoids generic array polynomial
                # evaluation at every Runge--Kutta stage.
                def rf_field(p,t):
                    angle=omega*t+phase
                    wave=(complex(math.cos(angle),-math.sin(angle)) if order==2 else
                          complex(math.sin(angle),math.cos(angle)))
                    d=order*amplitude*complex(*p)**(order-1)*wave
                    return np.array([-d.real,d.imag])
                runs=[]
                for step_divisor,tol in ((50,2e-9),(100,2e-11)):
                    result=integrate_trajectory(None,particle,amplitude,omega/(2*np.pi),
                         initial_position=tuple(initial[:2]),initial_velocity=tuple(initial[2:]),
                         simulation_duration=duration,escape_radius=100.,rtol=tol,atol=tol*1e-2,
                         max_step_per_period=step_divisor,output_samples_per_period=64.,field_function=rf_field)
                    assert result.termination=="completed" and abs(result.time[-1]-duration)<1e-12
                    runs.append(result)
                coarse,fine=runs
                convergence=_metrics(coarse.state[:2].T,fine.state[:2].T,scale)
                assert convergence["relative_max"]<1e-7
                centres,averaged=centered_period_mean(fine.state[:2].T,fine.time,64)
                reference=slow.sol(centres)[:2].T
                error=_metrics(averaged,reference,scale)
                _,slow_filtered=centered_period_mean(slow.sol(fine.time)[:2].T,fine.time,64)
                filter_bias=_metrics(slow_filtered,reference,scale)
                # Independent leading-micromotion subtraction, evaluated at the
                # smooth reference to avoid circularly fitting the RF trajectory.
                time=fine.time
                R=slow.sol(time)[:2].T
                ec,es=model.field_quadratures(R)
                xi=-(ec*np.cos(omega*time+phase)[:,None]+es*np.sin(omega*time+phase)[:,None])/omega**2
                _,subtracted=centered_period_mean(fine.state[:2].T-xi,time,64)
                subtraction=_metrics(subtracted,reference,scale)
                _,filter_128=centered_period_mean(fine.state[:2].T[::2],time[::2],32)
                filter_sampling=_metrics(filter_128,averaged[::2],scale)
                assert filter_sampling["relative_max"]<1e-5
                exact_error=None
                if order==2:
                    indices=np.linspace(0,len(fine.time)-1,257,dtype=int)
                    exact=quadrupole_exact(fine.time[indices],initial,amplitude,omega,phase)
                    exact_error=_metrics(fine.state[:2,indices].T,exact[:,:2],scale)
                    assert exact_error["relative_max"]<1e-8
                record={"order":order,"phase_rad":float(phase),"frequency_multiplier":multiplier,
                        "omega":omega,"amplitude":amplitude,"mass":1.,"charge":1.,
                        "dimensionless_benchmark":True,"duration":duration,"slow_period_estimate":slow_period,
                        "slow_initial_state":np.r_[R0,V0].tolist(),"full_initial_state":initial.tolist(),
                        "ic_inverse_max_error":ic_error,"ode_step_convergence":convergence,
                        "averaged_vs_slow":error,"filter_bias_on_slow":filter_bias,
                        "micromotion_subtraction_vs_slow":subtraction,
                        "filter_64_vs_32_samples":filter_sampling,"quadrupole_exact_error":exact_error,
                        "slow_energy_relative_drift":energy_drift}
                records.append(record)
                np.savez_compressed(output/f"n{order}_phase{round(phase*1000)}_s{multiplier}.npz",
                                    time=fine.time,state=fine.state,slow_time=centres,rf_period_mean=averaged,slow_reference=reference)
                print(f"Benchmark n={order}, phase={phase:.3f}, s={multiplier}: RMS={error['relative_rms']:.3e}",flush=True)
                if last_rms is not None:
                    assert error["relative_rms"]<last_rms, "High-frequency trend failed"
                last_rms=error["relative_rms"]
    summary={"status":"passed","scope":"Smooth analytic benchmarks, not FEM trajectory validation",
             "normalization_checks":checks,"cases":records}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    _plot_benchmarks(summary,output)
    return summary


def _plot_benchmarks(summary,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,order in zip(axes,(2,4)):
        for phase in (0.,np.pi/2):
            rows=[r for r in summary["cases"] if r["order"]==order and r["phase_rad"]==phase]
            ax.loglog([r["frequency_multiplier"] for r in rows],[r["averaged_vs_slow"]["relative_rms"] for r in rows],"o-",label=f"phase={phase:.2f}")
        ax.set(xlabel="Frequency multiplier (amplitude scaled equally)",ylabel="Relative RMS slow-coordinate difference",title=f"Ideal rotating n={order}")
        ax.grid(True,which="both",alpha=.3)
        ax.legend()
    fig.savefig(output/"frequency_convergence.png",dpi=180)
    plt.close(fig)
