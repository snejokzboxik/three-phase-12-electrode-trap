# Regression checks

Command: `C:/Python314/python.exe -m pytest -q`

Result on 2026-10-24: **89 passed in 4.68 s**.

Ten tests cover direct FEM polynomial evaluation, vertex permutations,
actual P4 H1 node orientation, field/Jacobian and phase encoding, independent
ODE comparisons, and cell ownership with global-clock rounding. The other
79 project tests also passed. Numerical kernels used for the long RF runs
remain frozen by validation_settings.json. Analysis and documentation edits
do not modify those kernels.
