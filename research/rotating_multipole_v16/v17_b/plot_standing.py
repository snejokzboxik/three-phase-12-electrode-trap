import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from v16_1.dynamics import Harmonic, potential_jet
OUT=Path(__file__).resolve().parent; RES=OUT/'results'; FIG=OUT/'figures'
r=json.loads((RES/'standing_ro3_results.json').read_text(encoding='utf-8'))
a=.1;ell=.2;k=.03;I=ell**2/12;gr=.02;cm=np.array([0.,a,0.,a**3],complex);hs=[Harmonic(3,'SO',k,1.,0.)]
T=6*math.pi;t=np.linspace(0,T,1601)
fig,axs=plt.subplots(1,2,figsize=(10.5,4.4))
for row,color,label in [(r['both_directions'][0],'#2874a6','mean spin +1/3'),(r['both_directions'][1],'#884ea0','mean spin −1/3')]:
 s=row['direction']
 def f(t0,y):
  _,g,_=potential_jet(0.,0.,y[0],t0,ell,hs,cm)
  return [y[1],-g[2]/I-gr*y[1]]
 sol=solve_ivp(f,(0,T),[row['initial_phase'],row['initial_spin']],t_eval=t,method='DOP853',rtol=1e-11,atol=1e-13,max_step=T/1600)
 axs[0].plot(t,sol.y[1]-s/3,color=color,label=label)
axs[0].set(xlabel='dimensionless time τ',ylabel='spin minus mean lock rate',title='Periodic spin modulation')
axs[0].legend()
rows=r['both_directions']; labels=['SO3 +','SO3 −']; vals=[]; colors=[]
for row in rows:
 vals.extend(row['multipliers_abs']); colors.extend(['#2874a6']*6 if row['direction']>0 else ['#884ea0']*6)
axs[1].scatter(np.arange(6),rows[0]['multipliers_abs'],s=55,color='#2874a6',label='positive rotation')
axs[1].scatter(np.arange(6)+.12,rows[1]['multipliers_abs'],s=55,color='#884ea0',marker='s',label='negative rotation')
axs[1].axhline(1,color='k',ls='--',lw=1)
axs[1].set(xlabel='Floquet mode index',ylabel='|μ|',ylim=(0,1.08),title='Full 6D Floquet multiplier moduli')
axs[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(FIG/'standing_so_locking.png',dpi=180);plt.close(fig)
