# Consistent pooled vector-RMS reconstruction

## Scope and provenance

This isolated post-processing used the final quality-controlled R1/R2 artifacts in `outputs/publication_study/quality_mesh_convergence/`. The saved `R1_mesh.npz`, `R2_mesh.npz`, `R1_basis.npz`, and `R2_basis.npz` files were present. The quality study's complete same-grid `field_samples.csv` already contains the exact corrected-P2 evaluator output for R1 and R2, so those stored field components were reused; no FEM solve, trajectory calculation, ensemble, mesh regeneration, or evaluator rerun was performed.

The saved configuration gives U0 = 1000 V and R = 0.01 m, hence U0/R = 100000 V/m. The quality-study hard-coded field evaluation also used U0 = 1000 V and frequency = 10000 Hz, consistent with the saved configuration.

## Sampling and definitions

For each listed nonzero radius, the calculation uses 144 uniformly spaced angular samples and 12 uniformly spaced drive phases, for N = 1728 identical physical sample points for R1 and R2. The field is the full planar vector (Ex, Ey).

The consistent pooled quantities are:

\[
E_{\rm RMS}^{(m)}(r) = \sqrt{\frac{1}{N}\sum_{\theta,\phi}\left(E_{x,m}^2+E_{y,m}^2\right)} ,
\]

\[
D_E^{(a,b)}(r) = \sqrt{\frac{1}{N}\sum_{\theta,\phi}\left[(E_{x,b}-E_{x,a})^2+(E_{y,b}-E_{y,a})^2\right]} ,
\]

and \(ε_E^{(a,b)} = D_E^{(a,b)}/E_{\rm RMS}^{(b)}\), with b = R2. The old quality-study convention was reconstructed as the arithmetic mean over the 12 phase-specific angular RMS values.

## Critical result at r = 0.100 mm

| quantity | old convention/report | consistent pooled vector RMS |
|---|---:|---:|
| R1 RMS field (V/m) | not reported in the old summary | 0.542209323181 |
| R2 RMS field (V/m) | 0.532061694618 | 0.532061694618 |
| R1→R2 field difference (V/m) | 0.0347299933932 | 0.0347299933932 |
| relative difference | 6.52743727739% | 6.52743727739% |
| global-normalized difference | not used in old local report | 3.47299933932e-07 |

The published 6.53% value remains unchanged to the displayed precision. The corrected pooled relative difference is **6.52743728%**, a change of **+0.00000000 percentage points**, or **+0.00000000% relative to the old percentage**. At r = 0.100 mm, the spread of the 12 phase-specific RMS differences is only 7.236859e-10 V/m, which is why the phase-mean and pooled RMS agree.

The old and reconstructed old conventions agree to a maximum absolute difference of 3.469447e-18 V/m across the reported radii; this is numerical roundoff/serialization agreement with the saved quality-study results.
The maximum absolute difference between the reconstructed old phase-mean value and the new pooled value over the requested radii is 8.881784e-16 V/m.

## Radius-resolved result

The complete table is in `rms_by_radius.csv` and contains E_RMS for R1/R2, the old phase-mean difference, the pooled vector difference, local relative errors, and pooled global normalization. The component-level R1/R2 samples and dEx/dEy are in `r1_r2_field_components.csv`.

The convention change does not alter the qualitative convergence conclusion or the rounded 6.53% central value. The manuscript should clarify the pooled vector-RMS definition; no numerical replacement is required for the existing 6.53% value on account of this convention check.

## Manuscript-ready definition

At each nonzero radius, averages are performed jointly over 144 uniformly spaced angular positions and 12 uniformly spaced RF drive phases. The metric uses the full 2D electric-field vector and computes the RMS of the vector difference; it is not the difference of two RMS magnitudes.

\[
E_{\rm RMS}^{(m)}(r) = \sqrt{\frac{1}{N_\theta N_\phi} \sum_{\theta,\phi} \left(E_{x,m}^2+E_{y,m}^2\right)}.
\]

\[
D_E^{(a,b)}(r) = \sqrt{\frac{1}{N_\theta N_\phi} \sum_{\theta,\phi} \left[(E_{x,b}-E_{x,a})^2+(E_{y,b}-E_{y,a})^2\right]}.
\]

\[
\epsilon_E^{(a,b)}(r) = \frac{D_E^{(a,b)}(r)}{E_{\rm RMS}^{(b)}(r)}.
\]
