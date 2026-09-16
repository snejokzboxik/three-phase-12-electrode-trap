# Common DC-bias comparison

This is an additional comparison; the zero-DC results remain preserved. A fifth FEM basis solution `ΦDC` uses all 12 electrodes at 1 V and the same grounded outer boundary. The full instantaneous field is `E(t)=Udc·EDC+ERF(t)` and is what the trajectory integrator uses. No per-time-step FEM solves are performed.

Quantities are kept distinct: (1) RF pseudopotential `ΨRF=q²⟨|ERF|²⟩/(2mΩ²)`; (2) static electrostatic energy `qΦDC` (saved separately); (3) total instantaneous electric field `Udc·EDC+ERF(t)`; and (4) the optional heuristic `q²⟨|ERF+Udc·EDC|²⟩/(2mΩ²)=ΨRF+q²Udc²|EDC|²/(2mΩ²)`. The latter is explicitly a total-field proxy, not the RF pseudopotential. The RF/DC cross term averages to zero because the three RF sinusoids have zero mean.

The direct-FEM check used 18 independent direct solves: Udc=100, 500, and 1000 V at six RF phases each. Across all solves: max/RMS potential error = 5.68434189e-12 / 2.12867328e-13 V; max/RMS electric-field error = 4.95850015e-08 / 8.35467568e-10 V/m. Per-case results are in `dc_superposition_validation.csv`.

The RF-only pseudopotential is unchanged by common Udc (constructed from the same RF basis). At 1000 V, the proxy increment is 3.16425164e-20 J (3.16425e-05 fJ) for r≤2 mm, only 7.43341e-09 of the central RF-pseudopotential range. Across the wider displayed central grid (to 4.2 mm), its maximum is 3.16414877e-16 J. It is negligible on the central-well scale. The static field and static potential energy are distinct and available in each case directory; maximum |qΦDC| in the displayed 1000-V central grid is 9.98446988e-10 J, mostly an offset rather than a measure of trapping curvature.

## Bias metrics

| Udc (V) | central range r≤1.5 mm (nJ) | angular spread at 2 mm (nJ) | illustrative useful radius (mm) |
|---:|---:|---:|---:|
| 0 | 0.00084171387 | 0.0020955662 | 2.02593 |
| 100 | 0.00084171387 | 0.0020955662 | 2.02593 |
| 300 | 0.00084171387 | 0.0020955662 | 2.02593 |
| 500 | 0.00084171387 | 0.0020955662 | 2.02593 |
| 1000 | 0.00084171388 | 0.0020955662 | 2.02593 |

The RF center remains the same flat-bottom higher-order minimum; common DC does not change its RF multipoles or RF pseudopotential. The added static field changes the exact instantaneous force but makes a negligible contribution to the central time-averaged squared-field proxy for these values. The present numerical evidence does not show a meaningful DC-based expansion of the useful central region or a localization benefit.

## Exploratory trajectory subset

One initial condition was tested for Udc=0, 300, and 1000 V over a maximum of 40 periods (U0=1000 V, f=10000 Hz, r0=0.25 mm, v0=0), with the existing ODE tolerances. All three trajectories escaped: 17.7157, 16.5632, and 16.4942 RF periods, respectively. This small probe suggests slightly earlier escape with DC bias for this one case only; it is not a stability result.

Figures and grids are grouped under `outputs/publication_study/dc_bias/`; each Udc case includes RF-only pseudopotential, total-field proxy, static DC energy/field views, radial cuts, anisotropy, and CSV grids. The concise six-panel `fig_dc_bias_summary` distinguishes these quantities.
