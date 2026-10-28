# Direct FEM A validation

Numerical verification: **passed**.

## Main outcome

pilot, phase 0: A35-B35 **passed** (RMS 0.555056%, max 1.34997%, period 0.000103512%); A35-C35 **passed** (RMS 0.555057%, max 1.34997%, period 0.000105212%).

pilot, phase 1: A35-B35 **passed** (RMS 0.596591%, max 1.4504%, period 8.92513e-05%); A35-C35 **passed** (RMS 0.596591%, max 1.4504%, period 9.09515e-05%).

extended, phase 0: A35-B35 **failed** (RMS 1.74866%, max 4.49955%, period 9.48596e-05%); A35-C35 **failed** (RMS 1.74866%, max 4.49955%, period 9.65598e-05%).

A uses native P4 FE gradients without a harmonic reconstruction, filtering of the force, or fitted trajectory. P4 refines the discretization of the same physical problem; the unreliable historical P2 trajectories are preserved and are not retroactively certified.

The tested conservative effective energies are Psi_B=q^2 (|E_c|^2+|E_s|^2)/(4 m Omega^2) and Psi_C=8 q^2 C4^2 r^6/(m Omega^2), with C4 including U0. C is the rotating octupole n=4 model, with sixth-power radial pseudopotential.

## Frozen experiment

m = 1e-15 kg, q = +1e-12 C, U0 = 250 V, f = 100 kHz. Original full initial states and global RF phase are unchanged. Pilot: 7506 RF periods (0.07506 s), phases 0 and pi/2. Extended: 23672 RF periods (0.23672 s), phase 0. Position errors use fixed scale 1 mm; no time, phase, or path alignment.

Physical thresholds: RMS 1%, maximum 3%, period 1%. Numerical/spatial/filter budgets: RMS 0.1%, maximum 0.3%, period 0.1%. Centered trapezoidal RF-period average with 64 output intervals; endpoints lacking a complete window are excluded.

## Numerical and spatial controls

| Case | Control | RMS (%) | Max (%) | Period (%) | Passed |
|---|---|---:|---:|---:|---|
| pilot, phase 0 | A_h50 T50-T100 | 9.30458246e-09 | 3.05921285e-08 | 2.41244438e-09 | True |
| pilot, phase 0 | A_h35 T50-T100 | 5.28445505e-09 | 1.01830073e-08 | 3.48758101e-10 | True |
| pilot, phase 0 | A_h35 T100-refined | 1.8288424e-09 | 4.45019668e-09 | 1.05174337e-10 | True |
| pilot, phase 0 | A_h35 T100-radau | 3.77835637e-06 | 7.39781293e-06 | 5.90863414e-07 | True |
| pilot, phase 0 | B_h50 coarse-fine | 2.47043504e-10 | 6.36275056e-10 | 4.86938601e-11 | True |
| pilot, phase 0 | C_h50 coarse-fine | 2.46990729e-10 | 6.36795473e-10 | 9.38269519e-11 | True |
| pilot, phase 0 | B_h35 coarse-fine | 2.4774408e-10 | 6.38416567e-10 | 8.14048524e-11 | True |
| pilot, phase 0 | C_h35 coarse-fine | 2.48649733e-10 | 6.40621924e-10 | 1.48100335e-10 | True |
| pilot, phase 1 | A_h50 T50-T100 | 3.72263146e-09 | 8.9471324e-09 | 6.90954552e-10 | True |
| pilot, phase 1 | A_h35 T50-T100 | 7.67996968e-09 | 2.36690485e-08 | 6.7841945e-10 | True |
| pilot, phase 1 | B_h50 coarse-fine | 2.47043504e-10 | 6.36275056e-10 | 4.86938601e-11 | True |
| pilot, phase 1 | C_h50 coarse-fine | 2.46990729e-10 | 6.36795473e-10 | 9.38269519e-11 | True |
| pilot, phase 1 | B_h35 coarse-fine | 2.4774408e-10 | 6.38416567e-10 | 8.14048524e-11 | True |
| pilot, phase 1 | C_h35 coarse-fine | 2.48649733e-10 | 6.40621924e-10 | 1.48100335e-10 | True |
| extended, phase 0 | A_h50 T50-T100 | 1.92860591e-07 | 6.18512039e-07 | 1.29370285e-08 | True |
| extended, phase 0 | A_h35 T50-T100 | 1.04506561e-07 | 3.65428124e-07 | 3.54094916e-09 | True |
| extended, phase 0 | B_h50 coarse-fine | 2.45745282e-09 | 6.35482462e-09 | 2.13045028e-10 | True |
| extended, phase 0 | C_h50 coarse-fine | 2.46700444e-09 | 6.37665232e-09 | 2.52408774e-10 | True |
| extended, phase 0 | B_h35 coarse-fine | 2.46947327e-09 | 6.38376816e-09 | 2.56399743e-10 | True |
| extended, phase 0 | C_h35 coarse-fine | 2.46124168e-09 | 6.36287271e-09 | 2.82347722e-10 | True |
| pilot, phase 0 | A_P4_h50-h35  | 0.0302636157 | 0.0598434247 | 0.00371550028 | True |
| pilot, phase 1 | A_P4_h50-h35  | 0.0323913739 | 0.0586849115 | 0.00372203184 | True |
| extended, phase 0 | A_P4_h50-h35  | 0.0933662737 | 0.186086315 | 0.00371443859 | True |

DOP853 uses T_RF/50 and T_RF/100 at both meshes and all three cases. Phase-zero pilot additionally uses T_RF/200 and independent Radau time discretization. Both time methods share the geometry audit. Every accepted dense curve is checked inside its cell; crossings preserve the state and global time, with zero position nudges. v2 explicitly bounds position roundoff from representing local crossing roots on the global clock by eight time ULPs. Completed v1 h50/A_fit pilots retain their original, stricter audit and explicit original-kernel hashes; they are not presented as v2 recomputations. See geometry_audits.json, provenance_audit.json, and individual crossing logs. Extended runs continue matching pilots with parent SHA-256 records.

Filter controls compare 32/64 output resolution, one/two/three RF periods, a local RF trend, and inverse micromotion with drift. All controls, including maxima, are in filter_controls.csv. Native B/C additionally pass time refinement and effective-energy drift <=1e-8.

## Comparison with effective potentials

| Case | Pair | RMS (%) | Max (%) | Period (%) | Result |
|---|---|---:|---:|---:|---|
| pilot, phase 0 | A35-B70 | 0.5622666 | 1.34997 | 0.01105455 | passed |
| pilot, phase 0 | A35-C70 | 0.5624434 | 1.34997 | 0.01119386 | passed |
| pilot, phase 0 | A35-B35 | 0.5550565 | 1.349971 | 0.0001035117 | passed |
| pilot, phase 0 | A35-C35 | 0.5550565 | 1.349971 | 0.0001052119 | passed |
| pilot, phase 0 | A35-B35_matched | 0.5551569 | 1.350243 | 0.0001030788 | passed |
| pilot, phase 0 | A35-A_fit35 | 5.061987e-07 | 1.126263e-06 | 1.034438e-08 | passed |
| pilot, phase 0 | A35-A_smooth70 | 0.08928483 | 0.176553 | 0.01097083 | passed |
| pilot, phase 0 | B35-C35 | 1.344467e-05 | 2.68415e-05 | 1.700149e-06 | passed |
| pilot, phase 0 | B70-C70 | 0.001100749 | 0.002199503 | 0.0001393306 | passed |
| pilot, phase 0 | B35_matched-B35 | 0.003167116 | 0.003658138 | 4.32895e-07 | passed |
| pilot, phase 0 | B35-B70 | 0.08652733 | 0.1728946 | 0.01095105 | passed |
| pilot, phase 0 | C35-C70 | 0.08761463 | 0.1750672 | 0.01108866 | passed |
| pilot, phase 1 | A35-B70 | 0.6028895 | 1.450385 | 0.01104029 | passed |
| pilot, phase 1 | A35-C70 | 0.6030492 | 1.450385 | 0.0111796 | passed |
| pilot, phase 1 | A35-B35 | 0.5965906 | 1.450395 | 8.925133e-05 | passed |
| pilot, phase 1 | A35-C35 | 0.5965906 | 1.450395 | 9.095147e-05 | passed |
| pilot, phase 1 | A35-B35_matched | 0.5560737 | 1.352459 | 9.59208e-05 | passed |
| pilot, phase 1 | A35-A_fit35 | 5.108204e-07 | 1.145769e-06 | 2.85865e-08 | passed |
| pilot, phase 1 | A35-A_smooth70 | 0.09557305 | 0.1731454 | 0.01094663 | passed |
| pilot, phase 1 | B35-C35 | 1.344467e-05 | 2.68415e-05 | 1.700149e-06 | passed |
| pilot, phase 1 | B70-C70 | 0.001100749 | 0.002199503 | 0.0001393306 | passed |
| pilot, phase 1 | B35_matched-B35 | 0.04063089 | 0.09812752 | 6.669476e-06 | passed |
| pilot, phase 1 | B35-B70 | 0.08652733 | 0.1728946 | 0.01095105 | passed |
| pilot, phase 1 | C35-C70 | 0.08761463 | 0.1750672 | 0.01108866 | passed |
| extended, phase 0 | A35-B70 | 1.77044 | 4.499499 | 0.0110459 | failed |
| extended, phase 0 | A35-C70 | 1.770985 | 4.499496 | 0.01118521 | failed |
| extended, phase 0 | A35-B35 | 1.748659 | 4.499546 | 9.485963e-05 | failed |
| extended, phase 0 | A35-C35 | 1.748659 | 4.499546 | 9.655978e-05 | failed |
| extended, phase 0 | A35-B35_matched | 1.749009 | 4.50046 | 9.442668e-05 | failed |
| extended, phase 0 | A35-A_fit35 | 1.513517e-06 | 3.558477e-06 | 3.725228e-08 | passed |
| extended, phase 0 | A35-A_smooth70 | 0.2754345 | 0.5489628 | 0.0109546 | passed |
| extended, phase 0 | B35-C35 | 4.233877e-05 | 8.46548e-05 | 1.70015e-06 | passed |
| extended, phase 0 | B70-C70 | 0.003468855 | 0.006936984 | 0.0001393306 | passed |
| extended, phase 0 | B35_matched-B35 | 0.003191249 | 0.003684958 | 4.329525e-07 | passed |
| extended, phase 0 | B35-B70 | 0.2726804 | 0.54529 | 0.01095105 | passed |
| extended, phase 0 | C35-C70 | 0.2761069 | 0.5521423 | 0.01108866 | passed |

B70/C70 are the frozen original article-validation models. B35 uses all N48 harmonic coefficients of the native P4 potential. C35 retains its rotating n=4 coefficient, calibrated from direct potential at 0.5/1/2 mm, without fitting trajectories. B35_matched changes only the slow-model initial condition by applying the native inverse micromotion map to the original physical full state; A always uses that same original physical state. A_fit35 is a diagnostic full-RF trajectory in the N48 reconstruction of P4, not the definition of A; it has one temporal level. A_smooth70 is the historical full-RF harmonic surrogate.

## Why this closes the earlier gap

The original P2 gradient created spurious central fields and sensitive element-interface dynamics. P4 improves local approximation while retaining direct FE forces. An exact complex cyclic reduction on a 30-degree sector is used, u(R30 x)=exp(-2 pi i/3)u(x); no spatial harmonics are discarded. Its P2 version reproduces the original full-domain P2 field with relative RMS 4.8e-13. The sector and full geometry, FE assembly residuals, original-cache hashes, and sample diagnostics are recorded in the build artifacts and frozen validation_settings.json.

## Interpretation and limits

A passed numerical verification means the requested comparison can be interpreted at its predeclared tolerance. A failed A-B/C physical comparison is a completed negative result for that interval, not numerical inconclusiveness. This experiment does not prove averaging for arbitrary charge/mass, voltage/frequency, initial amplitude, geometry, or time horizon. Long-time accumulated phase/coordinate error must be reported even if short trajectories agree. See comparisons.csv and direct_FEM_A_comparison.png.

Reproduce: python scripts/build_direct_fem.py (as documented in docs/DIRECT_FEM_A_PROTOCOL.md); python scripts/run_direct_fem_validation.py --stage all --workers 4; python scripts/analyze_direct_fem_validation.py. Existing hash-checked checkpoints are reused.

## Error separation on the extended interval

A35-B35 has x RMS 0.0315337% and y RMS 1.748374%, with period error 9.485963e-05%. The coordinate disagreement is dominated by transverse slow motion, not a large measured x-period error. See direct_FEM_A_transverse.png.

B35-C35 RMS is 4.233877e-05%. Native A35 versus diagnostic full-RF A_fit35 RMS is 1.513517e-06%. Changing only the slow initial map, B35_matched-B35, changes RMS coordinates by 0.003191249%; A35-B35_matched still fails. Filter sensitivity is also far below the observed discrepancy.

These controls identify leading averaged dynamics as the main limitation in this case. The n=4 spatial truncation and tested initial-state map do not explain the accumulated error. A higher-order effective equation or a unique physical mechanism for transverse drift has not been derived by this verification.
