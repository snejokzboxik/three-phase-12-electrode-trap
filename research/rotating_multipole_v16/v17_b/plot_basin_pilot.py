import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from v16_1.dynamics import Harmonic, potential_jet

OUT=Path(__file__).resolve().parent; RES=OUT/'results'; FIG=OUT/'figures'
ro=json.loads((RES/'basin_local_summary.json').read_text(encoding='utf-8'))
ro_wide=json.loads((RES/'basin_pilot_checkpoint.json').read_text(encoding='utf-8'))
so=json.loads((RES/'standing_basin_summary.json').read_text(encoding='utf-8'))
lock=json.loads((RES/'standing_ro3_results.json').read_text(encoding='utf-8'))
a=.1;ell=.2;k=.03;I=ell**2/12;gr=.02;cm=np.array([0.,a,0.,a**3],complex);hs=[Harmonic(3,'SO',k,1.,0.)]
def so_base(t0,sign):
 r=next(q for q in lock['both_directions'] if q['direction']==sign)
 def f(t,v):
  _,g,_=potential_jet(0.,0.,v[0],t,ell,hs,cm)
  return [v[1],-g[2]/I-gr*v[1]]
 sol=solve_ivp(f,(0,t0),[r['initial_phase'],r['initial_spin']],method='DOP853',rtol=1e-11,atol=1e-13,max_step=.03)
 return sol.y[:,-1]
fig,axs=plt.subplots(1,2,figsize=(11,4.7))
ax=axs[0]
first_ro=ro['trajectories'][0]
phi_ro=first_ro['initial_state'][4]-first_ro['initial_time']/3
for r in ro['trajectories']:
 y=r['initial_state']; t0=r['initial_time']; pe=3*(y[4]-(t0/3+phi_ro))
 ax.scatter((pe+math.pi)%(2*math.pi)-math.pi,y[5]-1/3,c='#239b56',marker='o',s=45,label='RO3 co-rotating' if r is ro['trajectories'][0] else None)
for r in so['trajectories']:
 sign=r['direction']; y=r['initial_state']; t0=r['initial_time']; ph,sp=so_base(t0,sign)
 pe=3*(y[4]-ph)
 color='#2874a6' if sign>0 else '#884ea0'
 ax.scatter((pe+math.pi)%(2*math.pi)-math.pi,y[5]-sp,c=color,marker='s' if sign>0 else '^',s=45,
            label='SO3 +1/3' if sign>0 and r['index']==0 else ('SO3 −1/3' if sign<0 and r['index']==0 else None))
for r in ro_wide['trajectories']:
 y=r['initial_state'];t0=r['initial_time'];pe=3*(y[4]-(t0/3+phi_ro))
 ax.scatter((pe+math.pi)%(2*math.pi)-math.pi,y[5]-1/3,c='#c0392b',marker='x',s=75,label='RO3 broad non-locking' if r is ro_wide['trajectories'][0] else None)
ax.axhline(0,color='0.65',lw=.7);ax.axvline(0,color='0.65',lw=.7)
ax.set(xlabel='initial phase offset from reference orbit (rad)',ylabel='initial spin offset (dimensionless)',title='1000-RF local capture pilot')
ax.legend(fontsize=8)
ax=axs[1]
series=[('RO3 +',6,6),('SO3 +',6,6),('SO3 −',6,6),('RO3 broad',0,1)]
# Wilson intervals; only these compact pilot samples, not a basin-area estimate.
z=1.959963984540054
for i,(name,k,n) in enumerate(series):
 p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 ax.bar(i,p,color=['#239b56','#2874a6','#884ea0','#c0392b'][i],alpha=.85)
 ax.errorbar(i,p,yerr=[[p-max(0,center-half)],[min(1,center+half)-p]],fmt='none',ecolor='black',capsize=4)
ax.set(xticks=range(4),xticklabels=[s[0] for s in series],ylim=(-.05,1.08),ylabel='empirical capture fraction',title='Small neighborhoods: Wilson 95% intervals')
fig.tight_layout();fig.savefig(FIG/'basin_pilot_compare.png',dpi=180);plt.close(fig)
