"""Twelve-electrode discrete phase synthesis and signed-mode alias audit."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent; RES=OUT/'results'; FIG=OUT/'figures'
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
N=12; n=np.arange(N); theta=2*np.pi*n/N
orders=[3,4]
patterns={}
for m in orders:
    v=np.exp(1j*m*theta)
    coeff=np.fft.fft(v)/N
    idx=int(np.argmax(np.abs(coeff)))
    patterns[str(m)]={"dominant_dft_index":idx,"amplitude":float(abs(coeff[idx])),
                      "signed_aliases":"m+12k","examples":[m,m-12],
                      "max_other_dft_amplitude":float(np.max(np.delete(np.abs(coeff),idx)))}
# Explicit waveform-only check. This does not predict the interior electric-field
# coefficient; that requires electrode geometry and FEM.
amp={3:150.,4:100.}; omega={3:1.,4:1.}; delta={3:0.2,4:-0.4}
t=np.linspace(0,2*np.pi,20001)
V=np.zeros((len(t),N))
for m in orders:
    V += amp[m]*np.cos(omega[m]*t[:,None]-m*theta[None,:]+delta[m])
peak=float(np.max(np.abs(V)))
fig,ax=plt.subplots(figsize=(7.2,4.8))
for m in orders:
    vals=np.abs(np.fft.fft(np.exp(1j*m*theta))/N)
    ax.stem(np.arange(N),vals,basefmt=' ',linefmt='C0-' if m==3 else 'C1-',markerfmt='C0o' if m==3 else 'C1s',label=f'm={m}')
ax.set(xlabel='12-electrode DFT bin (mod 12)',ylabel='normalized amplitude',xticks=np.arange(N),title='Spatial aliases on a 12-electrode ring')
ax.legend();fig.tight_layout();fig.savefig(FIG/'electrode_dft_alias.png',dpi=180);plt.close(fig)
res={"status":"ANALYTIC_DFT_AND_WAVEFORM_CHECK_ONLY","N":N,"angles_rad":theta.tolist(),"complex_mode_results":patterns,
 "waveform_example":{"harmonics":[{"m":3,"amplitude_V":amp[3],"omega":omega[3],"delta":delta[3]},{"m":4,"amplitude_V":amp[4],"omega":omega[4],"delta":delta[4]}],
 "sampled_peak_abs_V":peak,"conservative_bound_V":sum(amp.values()),"limit_V":250,"passes_sampled_bound":bool(peak<=250)},
 "limitations":["This is terminal-voltage synthesis only; not a field map or electrode transfer calibration.","Actual multipole coefficients, cross-harmonic contamination and phase lag require the electrode CAD, circuit response and FEM.","Signed aliasing means m=4 on 12 samples is indistinguishable from m=-8 at the electrode samples."]}
(RES/'electrode_dft_results.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({"patterns":patterns,"waveform_peak_V":peak,"sampled_peak_within_250V":peak<=250},indent=2))
