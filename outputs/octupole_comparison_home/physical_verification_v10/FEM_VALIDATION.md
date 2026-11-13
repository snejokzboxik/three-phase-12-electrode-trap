# Direct P4 field validation

## Saved field inputs

This validation loads existing quartic FEM solutions and meshes. It does not rebuild a mesh or run a new FEM solve.

| Resolution | Mesh SHA-256 | P4 solution SHA-256 | Elements |
|---|---|---|---:|
| h35 | `4919f8c75b2a60f516d9bc54af7e29cb7261cdf885c2f8557dbfecb6bc975abf` | `da90df98f9db31b5b7d84397bd9605bc6a4b940d8b30a8f9acb7d19cd8815a3d` | 647,484 |
| h50 | `8ea670e6bf5bada2c86756bc98718063fe38cbfd1692c872c4a4e77f59b1a284` | `336882933c476bc7ecd2fd97bdc060991ddda5b6d685a3c0889e5ee36b493872` | 319,956 |

Hash verification passed against each FEM completion manifest. The FEM solver count for V10 is zero.

## Native element-field evaluation

The new direct integrator evaluates the P4 polynomial potential and its analytic in-element gradient from the saved sector solution. It evaluates the rotating/standing voltage phase superposition at the same physical `(x,y)` and uses a dense first-passage event at `r=8 mm`. The trajectory outputs record element-search/transition diagnostics and the precise FEM input hashes. No field extrapolation beyond the source mesh domain is used to classify the aperture event.

For 32 interior points on each FEM grid, central finite differences of the P4 potential agree with the analytic complex gradient to maximum relative error `4.09e-11` (h50) and `4.51e-11` (h35). A second test independently checks rotating/standing temporal phase superposition and force evaluation; maximum relative errors are `6.37e-11` and `4.68e-11` respectively.

## Direct P4 retention outcomes

For the four h35 V7 boundary starts at 22.5°, 9.624°, 10.261° and 10.283°, rotating P4 retains all to 1000 periods and standing P4 exits all four. For the three late starts the h35 exit times are 892.582, 980.585 and 722.583 periods.

The h50 direct P4 sample includes six V7 starts, two RF phase controls at 10.272215°, and a strict-tolerance repeat of the early 22.5° matched pair. Rotating P4 retains all six standard-phase starts. Standing P4 retains the 0° and 9.60205° starts, and exits at 22.5°, 9.62402°, 10.26123° and 10.28320° at 185.493, 925.409, 815.382 and 800.603 periods, respectively. Retained classifications are right-censored at 1000 periods.

## Mesh and N48 comparison

All four matched late boundary outcomes in h35 and h50 direct P4 have the same class, but exit times differ by 32.8–165.2 periods. The h35 N48 model retains the positive start at 10.26123°, while both h35 and h50 direct P4 classify it as an exit. This is a direct counterexample to treating a near-threshold N48 classification as spatially converged.

The local N48-to-P4 field errors along saved direct P4 paths are small: approximately `3.7–4.1e-9` relative RMS for rotating retained paths and `2.7–4.9e-6` for standing escaped paths. This pathwise comparison is not a bound on long-time trajectory error: a slightly perturbed path may sample a different region near the escape boundary.

## Limited energy/radius pilot

Three additional matched h50 native-P4 launch conditions were evaluated without a new FEM solve: `1 eV, r₀=1 mm`; `3 eV, r₀=1 mm`; and `2 eV, r₀=2 mm`, all at `θ₀=22.5°`, `φ_RF=0°`, positive tangential velocity. Both fields retain the 1-eV case to the 1000-period horizon. At 3 eV the standing trajectory exits at `87.6924` periods while the rotating trajectory survives to the horizon. At 2 eV and 2-mm launch radius, the standing trajectory exits at `41.6100` periods while the rotating trajectory survives. The exact pair records and trajectories are stored under `data/native_p4/` and summarized in `processed/parameter_robustness_summary.csv`.

This is a targeted robustness check at three starts, not a resolved energy/radius boundary or an estimate over an initial-condition distribution. It supports persistence of the sampled difference in two conditions and shows that it is absent at the sampled 1-eV condition.

## What this establishes

For the sampled starts, the direct native P4 FEM calculation supports the finite-time rotating-versus-standing classification difference and closely reproduces the early 22.5° standing escape time across h35/h50 and N48/P4. It does not establish continuous-field convergence at late escape boundaries or a universal confinement advantage.
