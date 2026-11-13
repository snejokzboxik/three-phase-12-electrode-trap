# Numerical validation summary

## Reference protocols

- DOP853 full RF integrations, +e, `m/z=100000 Th`, `f=100 kHz`.
- Base tolerance `rtol=2e-9`, maximum step `T_RF/200`, 8 saved samples per RF period.
- Aperture event: dense DOP853 output, outward radial crossing, subinterval bracketing and Brent root; no post-event position or velocity nudge.
- State ordering is `(x,y,vx,vy)` in SI units; launch velocity is tangential and its sign is retained in metadata.
- `retained_to_horizon` means survived the requested finite horizon and is right-censored.

## Force and phase checks

The P4 native gradient was compared with central finite differences while both probes remained inside one element. Maximum relative errors: h50 `4.08596e-11`, h35 `4.50579e-11`. Separate rotating and standing phase-superposition checks give `6.37268e-11` and `4.68249e-11`. The input field and launch-state hashes are recorded with each trajectory.

## Tolerance refinement

The strict h50 early-start pair uses `rtol=2e-11`, position absolute tolerance `2e-14 m`, velocity absolute tolerance `2e-12 m/s`, maximum step `T_RF/400`, 64 samples per RF period. The standing P4 first passage is `185.49320025428142 T_RF`; the base integration gives `185.4932002543185 T_RF`. The rotating strict run survives 1000 periods. This establishes convergence for the selected early event, not for the later near-boundary events.

## Energy and angular-momentum bookkeeping

For the strict dense pair, the maximum Hamiltonian-work closure residual is `3.08e-4 eV` (rotating) and `4.55e-4 eV` (standing); maximum kinetic-work closure is `8.1e-5 eV` and `1.27e-4 eV`. Angular-momentum closure against integrated RF torque is `3.21e-28` and `3.04e-28 kg m²/s`. Modal torque reconstruction relative RMS is `4.30e-9` rotating and `3.32e-6` standing. These are based on the dense trajectory plus cumulative Simpson quadrature.

An earlier sparse-output diagnostic with only 8 points per RF cycle had multi-eV work-balance residuals and is not used for quantitative energy-work claims. The final diagnostic script now uses cumulative Simpson quadrature and writes the dense table gzip-compressed.

## Spatial resolution

For h35/h50 P4, classifications of the four sampled late boundary starts agree, but escape times differ by up to about 165 periods. One h35 N48 point disagrees in class with its direct P4 counterpart. Integration-tolerance refinement cannot cure this spatial representation sensitivity. Late transition locations and precise escape-time ordering remain uncertain.

## Reproduction

The compact reproduction sequence is in `reproduce_v10.ps1`. `make_publication_figures.py` rebuilds the figures from the archived CSV/JSON/NPZ data. `run_native_p4.py` and `phase_structure_control.py` reproduce their corresponding numerical stages when the matching saved FEM artifacts and Python dependencies are present. They do not create or regenerate FEM data.
