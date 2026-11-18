import json
from pathlib import Path
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent; RES=OUT/'results'; FIG=OUT/'figures'
local=json.loads((RES/'basin_local_summary.json').read_text(encoding='utf-8'))
wide=json.loads((RES/'basin_pilot_checkpoint.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(figsize=(7.2,5.0))
for r in local['trajectories']:
    y=r['initial_state']; phase0=3*y[4]-r['initial_time']
    ax.scatter((phase0+3.141592653589793)%(2*3.141592653589793)-3.141592653589793,y[5]-1/3,c='#239b56',marker='o',s=54)
for r in wide['trajectories']:
    y=r['initial_state']; phase0=3*y[4]-r['initial_time']
    ax.scatter((phase0+3.141592653589793)%(2*3.141592653589793)-3.141592653589793,y[5]-1/3,c='#c0392b',marker='x',s=70)
ax.axhline(0,color='0.6',lw=.7);ax.axvline(0,color='0.6',lw=.7)
ax.set(xlabel='initial wrapped phase error (rad)',ylabel='initial spin offset from 1/3',title='RO3 basin pilot: green captured (small neighborhood), red non-locking (broad start)')
fig.tight_layout();fig.savefig(FIG/'basin_local_pilot.png',dpi=180);plt.close(fig)
