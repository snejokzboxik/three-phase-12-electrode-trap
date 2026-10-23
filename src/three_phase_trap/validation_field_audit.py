"""Single-mesh RF field audit. Fit discrepancies are proxies, not true FEM errors."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .rf_reconstruction import HarmonicQuadratures, fit_harmonic_quadratures


def rf_quadratures(interpolator,points,amplitude):
    a,b,c=(interpolator.field_basis(points,p) for p in ("A","B","C"))
    return np.sqrt(3)*amplitude*(c-b)/2,amplitude*(a-b/2-c/2)


def rf_potential_quadratures(interpolator,points,amplitude):
    a,b,c=(interpolator.potential_basis(points,p) for p in ("A","B","C"))
    return np.sqrt(3)*amplitude*(c-b)/2,amplitude*(a-b/2-c/2)



def element_jacobians(interpolator,element_ids,phase):
    """Exact within-element Jacobian of the affine P2 electric field.

    This does not include interelement jumps, and is not a distributional
    derivative of the full discontinuous FE field.
    """
    ids=np.asarray(element_ids)
    g=interpolator._gradients_lambda[ids]
    h=[4*np.einsum("ni,nj->nij",g[:,j],g[:,j]) for j in range(3)]
    h += [4*(np.einsum("ni,nj->nij",g[:,i],g[:,j])+np.einsum("ni,nj->nij",g[:,j],g[:,i]))
          for i,j in ((0,1),(1,2),(2,0))]
    values=interpolator.solutions.potentials[phase][interpolator._element_dofs[ids]]
    return -np.einsum("nk,nkij->nij",values,np.stack(h,axis=1))


def _local_diagnostics(ec,es,jc,js,charge,mass,omega):
    # Exact maximum amplitude of the two-quadrature vector field.
    amplitude=np.linalg.norm(np.stack((ec,es),axis=-1),ord=2,axis=(-2,-1))
    angle=np.arange(64)*2*np.pi/64
    max_norm=np.zeros(len(ec))
    for phase in angle:
        max_norm=np.maximum(max_norm,np.linalg.norm(jc*np.cos(phase)+js*np.sin(phase),ord=2,axis=(-2,-1)))
    factor=abs(charge)/(mass*omega**2)
    return factor*max_norm,factor*amplitude


def _csv(path,rows):
    import csv
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_field_audit(interpolator,config,output: Path,amplitude=250.,frequency=100000.):
    output.mkdir(parents=True,exist_ok=True)
    charge,mass=config.particle.charge,config.particle.mass
    omega=2*np.pi*frequency
    ref=.008
    training_radii=np.array([.0005,.0015,.003,.0045,.006,.0075])
    training_angles=(np.arange(256)+.317)*2*np.pi/256
    train=np.concatenate([np.column_stack((r*np.cos(training_angles),r*np.sin(training_angles))) for r in training_radii])
    pc,ps=rf_potential_quadratures(interpolator,train,amplitude)
    fits={}
    for order in (8,16,24,32):
        fits[order]=fit_harmonic_quadratures(train,pc,ps,order,ref)
    radii=np.array([.025,.05,.1,.25,.5,1,2,4,6,8])*1e-3
    angles=(np.arange(512)+.173)*2*np.pi/512
    samples=np.concatenate([np.column_stack((r*np.cos(angles),r*np.sin(angles))) for r in radii])
    raw_pc,raw_ps=rf_potential_quadratures(interpolator,samples,amplitude)
    ec,es=rf_quadratures(interpolator,samples,amplitude)
    _,ids=interpolator._query_triangles(samples)
    ja,jb,jd=(element_jacobians(interpolator,ids,p) for p in ("A","B","C"))
    jc=np.sqrt(3)*amplitude*(jd-jb)/2
    js=amplitude*(ja-jb/2-jd/2)
    eta,xi=_local_diagnostics(ec,es,jc,js,charge,mass,omega)
    Q=np.array([[0.,-1.],[1.,0.]])
    rotated_ec,rotated_es=rf_quadratures(interpolator,samples@Q.T,amplitude)
    field_energy=np.sum(ec*ec+es*es,axis=1)
    symmetry_error=np.sum((rotated_ec-ec@Q.T)**2+(rotated_es-es@Q.T)**2,axis=1)
    ring_rows=[]
    spectra=[]
    validation_rows=[]
    best=fits[32][0]
    for ir,r in enumerate(radii):
        cut=slice(ir*len(angles),(ir+1)*len(angles))
        phi4=[]
        modal_power=0.
        forbidden_power=0.
        for n in range(49):
            multiplier=1. if n==0 else 2.
            c=multiplier*np.mean(raw_pc[cut]*np.exp(-1j*n*angles))
            s=multiplier*np.mean(raw_ps[cut]*np.exp(-1j*n*angles))
            magnitude=float(np.sqrt((abs(c)**2+abs(s)**2)/2))
            if n>0:
                modal_power+=magnitude**2
                if n%12 not in (4,8):
                    forbidden_power+=magnitude**2
            spectra.append({"radius_m":float(r),"order":n,"combined_potential_amplitude_V":magnitude,
                            "cosine_coefficient_real":float(c.real),"cosine_coefficient_imag":float(c.imag),
                            "sine_coefficient_real":float(s.real),"sine_coefficient_imag":float(s.imag)})
            if n==4:
                phi4=[magnitude/float(r)**4]
        ring_rows.append({"radius_m":float(r),"rf_vector_rms_V_per_m":float(np.sqrt(np.mean(field_energy[cut])/2)),
                          "rotation_90deg_relative_rms":float(np.sqrt(np.mean(symmetry_error[cut])/np.mean(field_energy[cut]))),
                          "C4_amplitude_from_potential_V_per_m4":phi4[0],
                          "forbidden_potential_mode_fraction":float(np.sqrt(forbidden_power/max(modal_power,1e-300))),
                          "eta_elementwise_cycle_sample_max":float(eta[cut].max()),
                          "micromotion_amplitude_max_m":float(xi[cut].max())})
        for order,(model,condition) in fits.items():
            fit_ec,fit_es=model.field_quadratures(samples[cut])
            fit_pc,fit_ps=model.potential_quadratures(samples[cut])
            model_p,model_s=best.field_quadratures(samples[cut])
            field_error=(np.sum((fit_ec-ec[cut])**2+(fit_es-es[cut])**2,axis=1))
            error_phi=(fit_pc-raw_pc[cut])**2+(fit_ps-raw_ps[cut])**2
            validation_rows.append({"radius_m":float(r),"fit_order":order,"design_condition":condition,
                "potential_relative_rms_discrepancy":float(np.sqrt(np.mean(error_phi)/np.mean(raw_pc[cut]**2+raw_ps[cut]**2))),
                "field_relative_rms_discrepancy":float(np.sqrt(np.mean(field_error)/np.mean(field_energy[cut]))),
                "field_absolute_rms_discrepancy_V_per_m":float(np.sqrt(np.mean(field_error)/2)),
                "field_vs_order32_relative_rms":float(np.sqrt(np.mean(np.sum((fit_ec-model_p)**2+(fit_es-model_s)**2,axis=1))/np.mean(np.sum(model_p**2+model_s**2,axis=1))))})
    # Numerical agreement of the quadrature transformation and direct RF
    # superposition. These are identical fields, not a spatial-accuracy check.
    phase_error=0.
    for phase in (0.,.37,1.9,4.1):
        direct=interpolator.field(samples,phase/omega,amplitude,frequency)
        phase_error=max(phase_error,float(np.max(abs(direct-ec*np.cos(phase)-es*np.sin(phase)))))
    assert phase_error < 1e-7
    # One-sided element evaluations at exactly the same facet midpoint.
    mesh=interpolator.solutions.trap_mesh.mesh
    internal=np.flatnonzero(np.all(mesh.f2t>=0,axis=0))
    midpoint=mesh.p[:,mesh.facets[:,internal]].mean(axis=1).T
    radius_all=np.linalg.norm(midpoint,axis=1)
    # Retain every facet near the centre, where uniform subsampling would leave
    # only a few samples in the most important underresolved region.
    central=internal[radius_all<=.000375]
    outer=internal[(radius_all>.000375)&(radius_all<=ref)][::13]
    internal=np.r_[central,outer]
    midpoint=mesh.p[:,mesh.facets[:,internal]].mean(axis=1).T
    fields=[]
    for side in (0,1):
        a,b,c=(interpolator._field_values(midpoint,mesh.f2t[side,internal],p) for p in ("A","B","C"))
        fields.append((np.sqrt(3)*amplitude*(c-b)/2,amplitude*(a-b/2-c/2)))
    jump=np.sqrt((np.sum((fields[0][0]-fields[1][0])**2+(fields[0][1]-fields[1][1])**2,axis=1))/2)
    reference=np.sqrt((np.sum((fields[0][0]+fields[1][0])**2+(fields[0][1]+fields[1][1])**2,axis=1))/8)
    jump_radius=np.linalg.norm(midpoint,axis=1)
    jump_bins=[]
    for low,high in zip((0.,.075,.15,.375,.75,1.5,3.,5.,7.),(.075,.15,.375,.75,1.5,3.,5.,7.,8.)):
        selected=(jump_radius>=low*1e-3)&(jump_radius<high*1e-3)
        if np.any(selected):
            relative=jump[selected]/np.maximum(reference[selected],1e-30)
            jump_bins.append({"r_min_mm":low,"r_max_mm":high,"sampled_facets":int(selected.sum()),
                              "relative_jump_median":float(np.median(relative)),"relative_jump_p95":float(np.quantile(relative,.95)),
                              "absolute_jump_rms_V_per_m":float(np.sqrt(np.mean(jump[selected]**2)))})
    _csv(output/"jump_radial_bins.csv",jump_bins)
    _csv(output/"interelement_jumps.csv",[{"radius_m":float(r),"x_m":float(p[0]),"y_m":float(p[1]),
        "jump_rf_rms_V_per_m":float(j),"mean_field_rf_rms_V_per_m":float(v),
        "relative_jump":float(j/max(v,1e-30))} for r,p,j,v in zip(jump_radius,midpoint,jump,reference)])
    axis=np.linspace(-ref,ref,81)
    x,y=np.meshgrid(axis,axis)
    mask=(x*x+y*y<=ref**2)&(x*x+y*y>0)
    grid=np.column_stack((x[mask],y[mask]))
    map_ec,map_es=rf_quadratures(interpolator,grid,amplitude)
    _,grid_ids=interpolator._query_triangles(grid)
    ga,gb,gc=(element_jacobians(interpolator,grid_ids,p) for p in ("A","B","C"))
    map_eta,map_xi=_local_diagnostics(map_ec,map_es,np.sqrt(3)*amplitude*(gc-gb)/2,
                                    amplitude*(ga-gb/2-gc/2),charge,mass,omega)
    smooth_ec,smooth_es=best.field_quadratures(grid)
    smooth_jc,smooth_js=best.jacobian_quadratures(grid)
    smooth_eta,smooth_xi=_local_diagnostics(smooth_ec,smooth_es,smooth_jc,smooth_js,charge,mass,omega)
    def as_map(values):
        result=np.full(x.shape,np.nan)
        result[mask]=values
        return result
    np.savez_compressed(output/"diagnostic_maps.npz",axis_m=axis,
                        eta_elementwise=as_map(map_eta),eta_harmonic_proxy=as_map(smooth_eta),
                        xi_elementwise_m=as_map(map_xi),xi_harmonic_proxy_m=as_map(smooth_xi))
    _csv(output/"radial_audit.csv",ring_rows)
    _csv(output/"multipole_spectrum.csv",spectra)
    _csv(output/"harmonic_holdout.csv",validation_rows)
    model_data={"reference_radius_m":ref,"training_radii_m":training_radii.tolist(),
                "training_angles":len(training_angles),"validation_angles":len(angles),
                "training_vs_validation":"different angles; 0.5 and 6 mm appear in both radius lists",
                "fits":{str(n):{"condition":condition,"cosine_real":model.cosine.real.tolist(),
                                "cosine_imag":model.cosine.imag.tolist(),"sine_real":model.sine.real.tolist(),
                                "sine_imag":model.sine.imag.tolist()} for n,(model,condition) in fits.items()}}
    (output/"harmonic_audit_fits.json").write_text(json.dumps(model_data,indent=2)+"\n",encoding="utf-8")
    summary={"status":"audit_complete_spatial_convergence_unverified", "U0_V":amplitude,"frequency_Hz":frequency,
             "mass_kg":mass,"charge_C":charge,"quadrature_reconstruction_max_abs_error_V_per_m":phase_error,
             "radii":ring_rows,"order32_holdout":[r for r in validation_rows if r["fit_order"]==32],
             "candidate_radii_below_1percent_fit_discrepancy_m":[r["radius_m"] for r in validation_rows if r["fit_order"]==32 and r["field_relative_rms_discrepancy"]<=.01],
             "eta_scope":"max over 64 RF phases of the within-element Jacobian; interelement jumps excluded",
             "fit_scope":"Potential-based harmonic reconstruction audit, not independent FEM truth and not completed model B",
             "jump_radial_bins":jump_bins,
             "pilot_1mm_representation_discrepancy":next(r["field_relative_rms_discrepancy"] for r in validation_rows if r["fit_order"]==32 and abs(r["radius_m"]-.001)<1e-12),
             "spatial_accuracy_status":"inconclusive until refined-mesh comparison"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    _plot_audit(output,ring_rows,validation_rows,as_map(map_eta),as_map(smooth_eta),axis)
    return summary


def _plot_audit(output,rings,validation,raw_eta,smooth_eta,axis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    r=np.array([v["radius_m"] for v in rings])
    values=np.array([v["rf_vector_rms_V_per_m"] for v in rings])
    axes[0,0].loglog(r*1e3,values,"o-",label="Raw P2 FEM")
    i=int(np.argmin(abs(r-.001)))
    axes[0,0].loglog(r*1e3,values[i]*(r/r[i])**3,"--",label="r^3 guide at 1 mm")
    axes[0,0].set(xlabel="Radius (mm)",ylabel="RF vector RMS field (V/m)")
    axes[0,0].legend()
    for order in (8,16,24,32):
        rows=[v for v in validation if v["fit_order"]==order]
        axes[0,1].loglog([v["radius_m"]*1e3 for v in rows],[v["field_relative_rms_discrepancy"] for v in rows],"o-",label=f"N={order}")
    axes[0,1].axhline(.01,color="grey",ls="--")
    axes[0,1].set(xlabel="Radius (mm)",ylabel="FEM vs harmonic fit discrepancy (not true error)")
    axes[0,1].legend()
    vmax=float(max(np.nanmax(raw_eta),np.nanmax(smooth_eta)))
    for ax,data,title in zip(axes[1],(raw_eta,smooth_eta),("Elementwise eta (jumps excluded)","Harmonic-fit eta proxy")):
        view=ax.imshow(data,origin="lower",extent=np.array([axis[0],axis[-1],axis[0],axis[-1]])*1e3,
                       norm=LogNorm(vmin=1e-5,vmax=vmax),cmap="viridis")
        ax.set(title=title,xlabel="x (mm)",ylabel="y (mm)")
        fig.colorbar(view,ax=ax)
    fig.savefig(output/"field_audit.png",dpi=180)
    plt.close(fig)
