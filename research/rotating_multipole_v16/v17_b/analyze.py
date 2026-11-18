from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from v16_1.dynamics import Harmonic, potential_jet
from v17_b.model import charge_moments, expanded_polynomial, direct_polynomial, synchronous_branch, rotating_jacobian, lab_monodromy

OUT=Path(__file__).resolve().parent
RES=OUT/'results'; FIG=OUT/'figures'
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

# Reference finite neutral dipole (+q,-q) at (+a,-a), q_ref=2q.
a=0.1; ell=2*a; kappa=0.03; omega=1.0; gamma_t=0.05; gamma_r=0.02
I=ell**2/12
qm=charge_moments([1.,-1.],[a,-a],8)
C=qm.moments
branch=synchronous_branch(3,kappa,omega,C,gamma_r,I)
assert branch['exists'], branch
J, jb=rotating_jacobian(3,kappa,omega,C,gamma_t,gamma_r,I,branch['phase'])
spec=np.linalg.eigvals(J)
mono=lab_monodromy(3,kappa,omega,C,gamma_t,gamma_r,I,branch['phase'])
mu=mono['multipliers']
T=mono['period']
# Compare unordered sets s -> exp(s T) with the independent lab monodromy.
expected=np.exp(spec*T)
ri,ci=linear_sum_assignment(np.abs(expected[:,None]-mu[None,:]))
matcherr=float(np.max(np.abs(expected[ri]-mu[ci])))

# Independent polynomial and finite-difference checks.
rng=np.random.default_rng(17017)
poly_err=0.0
for m in range(1,9):
    Z=complex(rng.normal()*0.07,rng.normal()*0.07); psi=float(rng.uniform(-math.pi,math.pi))
    poly_err=max(poly_err,abs(expanded_polynomial(Z,psi,m,C)-direct_polynomial(Z,psi,m,[1,-1],[a,-a])))
# Potential derivative check for order 3 against centered finite differences of energy.
x,y,psi,tau=0.023,-0.031,0.77,0.42
h=2e-5
hs=[Harmonic(3,'RO',kappa,omega,0.0)]
U,g,H=potential_jet(x,y,psi,tau,1.0,hs,C)
coords=np.array([x,y,psi])
def energy(q):
    uu,_,_=potential_jet(q[0],q[1],q[2],tau,1.0,hs,C); return uu
ng=np.zeros(3); nh=np.zeros((3,3))
for i in range(3):
    ei=np.eye(3)[i]*h
    ng[i]=(energy(coords+ei)-energy(coords-ei))/(2*h)
    for j in range(3):
        ej=np.eye(3)[j]*h
        nh[i,j]=(energy(coords+ei+ej)-energy(coords+ei-ej)-energy(coords-ei+ej)+energy(coords-ei-ej))/(4*h*h)
gerr=float(np.max(np.abs(g-ng)))
herr=float(np.max(np.abs(H-nh)))

# Adjacent controls: m=1,2,3,4,5. A centered lock requires C_(m-1)=0.
orders=[]
for m in range(1,6):
    cprev=C[m-1] if m-1<len(C) else 0j
    cm=C[m] if m<len(C) else 0j
    cquad=C[m-2] if m>=2 else 0j
    orders.append({"m":m,"C_m_minus_1_abs":float(abs(cprev)),"C_m_abs":float(abs(cm)),
                   "C_m_minus_2_abs":float(abs(cquad)),"center_force_zero":bool(abs(cprev)<1e-14),
                   "center_quadratic_localization":bool(abs(cquad)>1e-14)})
fig,ax=plt.subplots(figsize=(7.2,4.8))
xx=np.arange(1,6); width=.34
ax.bar(xx-width/2,[x['center_force_zero'] for x in orders],width,label='zero COM force at center')
ax.bar(xx+width/2,[x['center_quadratic_localization'] for x in orders],width,label='quadratic COM term present')
ax.set(xticks=xx,xlabel='field spatial order m',yticks=[0,1],yticklabels=['no','yes'],title='Neutral symmetric dipole: center-selection rules')
ax.legend(); fig.tight_layout(); fig.savefig(FIG/'multipole_selection_rules.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,4.8))
asweep_lock=np.geomspace(.01,.4,180)
for m in (1,3,5):
    # For this two-charge dipole, C_m=a^m at odd m; I=(2a)^2/12=a^2/3.
    crit=m*m*kappa*asweep_lock**m/(asweep_lock**2/3)
    ax.plot(asweep_lock,crit,label=f'm={m}')
ax.set(xscale='log',yscale='log',xlabel='dipole half-separation a/R',ylabel=r'$\gamma_{rot,c}$ (equal dimensionless $\kappa_m$)',title='RO synchronization thresholds (equal dimensionless coefficients)')
ax.legend(); fig.tight_layout(); fig.savefig(FIG/'rotational_locking_thresholds.png',dpi=180); plt.close(fig)

# Adaptive-ready exact 2D damping classification map for this analytical slice.
gtvals=np.linspace(0.001,0.8,320); grvals=np.linspace(0.001,0.8,320)
w=omega/3; kappa_map=0.25; bmap=3*kappa_map*abs(C[1])
grc_map=27*kappa_map*a
threshold=w*np.sqrt(w*w+gtvals**2)
com_ok=2*bmap < threshold
rot_ok=grvals < grc_map
# labels: 0 neither, 1 rotation only, 2 COM only, 3 joint.
labels=np.where(np.outer(rot_ok,com_ok),3,np.where(np.outer(rot_ok,~com_ok),1,np.where(np.outer(~rot_ok,com_ok),2,0)))
fig,ax=plt.subplots(figsize=(7.2,5.2))
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap=ListedColormap(['#555555','#e67e22','#2874a6','#239b56'])
norm=BoundaryNorm([-0.5,0.5,1.5,2.5,3.5],cmap.N)
pc=ax.pcolormesh(gtvals,grvals,labels,cmap=cmap,norm=norm,shading='auto')
ax.axhline(grc_map,color='white',lw=1.5,ls='--',label=rf'$\gamma_{{rot,c}}={grc_map:.4g}$')
ax.set(xlabel=r'$γ_{trans}$',ylabel=r'$γ_{rot}$',title='RO3 neutral dipole: exact local stability classes')
ax.legend(loc='upper right'); fig.tight_layout(); fig.savefig(FIG/'joint_stability_map.png',dpi=180); plt.close(fig)

# Eigenvalue and size dependence figures.
fig,ax=plt.subplots(figsize=(6.4,5.2));
for name,idxs in [('COM',range(4)),('rotation',range(4,6))]:
    vals=spec[list(idxs)]
    ax.scatter(vals.real,vals.imag,label=name,s=45)
ax.axvline(0,color='k',lw=.8); ax.set(xlabel='Re(s)',ylabel='Im(s)',title='6D co-rotating Jacobian spectrum'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'full_6d_spectrum.png',dpi=180); plt.close(fig)

asweep=np.geomspace(0.003,0.2,250); bvals=3*kappa*asweep; grcrit=27*kappa*asweep
margin=w*w-2*bvals
fig,ax1=plt.subplots(figsize=(7.2,5.2));
ax1.plot(asweep,2*bvals,label=r'$2|b|=6|\kappa_3|a$',color='#21618c')
ax1.axhline(w*w,color='#21618c',ls='--',label=r'$\omega_f^2$ (zero-drag boundary)')
ax1.set_xscale('log'); ax1.set_xlabel(r'dipole half-separation $a/R$'); ax1.set_ylabel('COM restoring split / stability threshold')
ax2=ax1.twinx(); ax2.plot(asweep,grcrit,color='#a93226',label=r'$\gamma_{rot,c}=27|\kappa_3|a$'); ax2.axhline(gamma_r,color='#a93226',ls=':',label=r'fixed $\gamma_{rot}$')
ax2.set_ylabel(r'rotational torque threshold $\gamma_{rot,c}$')
lines=ax1.get_lines()+ax2.get_lines(); ax1.legend(lines,[l.get_label() for l in lines],loc='upper left',fontsize=8)
ax1.set_title('Finite-size localization and spin-lock threshold'); fig.tight_layout(); fig.savefig(FIG/'size_scaling.png',dpi=180); plt.close(fig)

# Summary numerical results.
result={
 "status":"COMPLETED_ANALYTIC_SLICE; broader V17B work PARTIAL",
 "normalization":{"q_ref":qm.q_ref,"moments_definition":"C_j=sum(q_i/q_ref)*(s_i/R)^j","C0":str(C[0]),"charge_sum":0.0},
 "model":{"field":"pure travelling RO3 (hexapole), V=kappa Re[z^3 exp(-i omega t)]","charges":[[1.0,a],[-1.0,-a]],"a_over_R":a,"ell_over_R":ell,"kappa":kappa,"omega":omega,"inertia":I,"gamma_trans":gamma_t,"gamma_rot":gamma_r},
 "moments":[[float(c.real),float(c.imag)] for c in C],
 "exact_sync":{k:(float(v) if isinstance(v,(float,np.floating)) else v) for k,v in branch.items() if not k.startswith('_')},
 "b_abs":jb['b_abs'],"omega_frame":jb['omega_frame'],"jacobian_eigenvalues":[[float(v.real),float(v.imag)] for v in spec],
 "max_real_part":float(max(spec.real)),"monodromy_period":T,"monodromy_endpoint_residual":mono['endpoint_residual'],
 "monodromy_multipliers":[[float(v.real),float(v.imag)] for v in mu],"monodromy_vs_exp_sT_max_matching_error":matcherr,
 "validation":{"multipole_direct_sum_max_abs_error":poly_err,"gradient_fd_max_abs_error":gerr,"hessian_fd_max_abs_error":herr},
 "order_parity_controls":orders,
 "analytic_boundaries":{"rot_lock_gamma_rot_critical":branch['gamma_rot_critical'],"com_boundary":"2|b| < omega_f*sqrt(omega_f^2+gamma_trans^2), for gamma_trans>0","com_zero_drag":"2|b| < omega_f^2","b":jb['b_abs'],"omega_f":w,"gamma_trans_at_COM_boundary":float(math.sqrt(max(0,(2*jb['b_abs']/w)**2-w*w))),"map_example":{"kappa3":kappa_map,"rot_lock_gamma_rot_critical":grc_map,"b":bmap,"gamma_trans_boundary":float(math.sqrt(max(0,(2*bmap/w)**2-w*w))) }},
 "limitations":["2D rigid-body point-charge multipole model only","ideal harmonic field; no electrode geometry, voltage calibration, FEM, gas-specific drag or thermal noise","one 1000-RF small-neighborhood pilot and one broad non-locking start; no basin-size map, RO/SO comparison or adaptive Arnold tongue survey","not a novelty claim"]
}
(RES/'central_ro3_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({"spec":result['jacobian_eigenvalues'],"max_real_part":result['max_real_part'],"period":T,"endpoint_residual":mono['endpoint_residual'],"mono_match_error":matcherr,"validation":result['validation'],"gamma_rot_c":branch['gamma_rot_critical'],"b":jb['b_abs'],"com_boundary_gamma_trans":result['analytic_boundaries']['gamma_trans_at_COM_boundary']},indent=2))
