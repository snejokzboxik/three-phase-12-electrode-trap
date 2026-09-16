# Three-phase 12-electrode trap

This is a standalone scientific Python project for investigating the transverse electric field, multipole content, RF effective potential, and full charged-particle dynamics of a 12-electrode electrodynamic trap. It is intentionally independent of any earlier codebase and does not assume that the `ABCABCABCABC` arrangement produces a stable trap.

## Physical model

Twelve identical circular electrodes lie at

\[
\theta_n = 2\pi n/12,\qquad n=0,\ldots,11,
\]

on a configurable orbit radius. The phase assignment is exactly:

```text
A B C A B C A B C A B C
```

with A = {0, 3, 6, 9}, B = {1, 4, 7, 10}, and C = {2, 5, 8, 11}. The continuous RF voltages are

\[
U_A=U_0\sin(\Omega t),\quad
U_B=U_0\sin(\Omega t-2\pi/3),\quad
U_C=U_0\sin(\Omega t+2\pi/3),
\]

where \(\Omega=2\pi f_{RF}\). No switching-state approximation is used.

## FEM and field reconstruction

The free-space region is triangulated with a sampled circular outer boundary and sampled circular electrode surfaces. A quadratic triangular P2 scikit-fem solve enforces Laplace's equation, with the outer boundary held at 0 V as a finite far-field approximation. Three independent solves set one phase to 1 V and the other two to 0 V. Every time-dependent potential and electric field is then reconstructed by linear superposition. Direct instantaneous FEM solves are run only at seven validation phases. The P2 interpolator evaluates quadratic shape functions and their within-element gradients directly; it does not blur or cosmetically smooth the field.

Near-center potentials are fitted to

\[
\Phi=\sum_m (r/R)^m[a_m\cos(m\theta)+b_m\sin(m\theta)],
\]

without hard-coding a dominant order. The RF effective potential uses the exact three-phase average

\[
\langle |\mathbf E|^2\rangle = \frac{U_0^2}{2}\left(\sum_i|\mathbf E_i|^2-\sum_{i<j}\mathbf E_i\cdot\mathbf E_j\right),
\qquad
\Psi_{RF}=\frac{q^2\langle |\mathbf E|^2\rangle}{2m\Omega^2}.
\]

The factor is normalized so a single field \(E_0\cos\Omega t\) gives the familiar \(q^2E_0^2/(4m\Omega^2)\).

Particle trajectories solve

\[
m\ddot{\mathbf r}=q\mathbf E(\mathbf r,t)-\gamma\dot{\mathbf r}+m\mathbf g,
\]

with `scipy.integrate.solve_ivp`. The field interpolator raises an explicit error for electrode or outside-mesh points; it never silently extrapolates. A trajectory is `bounded` only if it remains below the configured `escape_radius` for the full integration interval.

## Installation

From the project directory:

```bash
python -m pip install -e .
python -m pip install -e ".[test]"
```

The project targets Python 3.12 or newer.

## Tests

```bash
pytest
```

The tests cover geometry, phase mapping, RF identities and periodicity, synthetic multipole recovery, exact RF averaging, mesh interpolation failure behavior, and a predictable zero-field particle trajectory.

## Reproducible commands

```bash
python -m three_phase_trap.cli fem
python -m three_phase_trap.cli animate
python -m three_phase_trap.cli multipoles
python -m three_phase_trap.cli effective-potential
python -m three_phase_trap.cli trajectory
python -m three_phase_trap.cli stability
python -m three_phase_trap.cli convergence
python -m three_phase_trap.cli mass-filter
python -m three_phase_trap.cli publication
python -m three_phase_trap.cli all
```

Use another TOML file with `--config path/to/config.toml`. The all-in-one command runs the basis FEM solve, direct validation, voltage diagnostics, snapshots, GIF animation, optional MP4 animation when ffmpeg is installed, multipole spectrum and rotation, radial scaling, effective potential, conservative and damped trajectory examples, stability scan, mesh convergence, and a generated `final_report.md`. The `publication` command additionally creates P2-based high-resolution figures, a checkpointed staged localization scan (50-period Stage A, 1000-period Stage B survivors, and 10000-period Stage C candidates), finite-flight mass transmission, and a mass-frequency survival map in `outputs/publication_study/`. Stage-A results are screening-only; stability conclusions use Stage B/C trajectories.

## Outputs

Results are written to `outputs/three_phase_12_electrode/`:

```text
phase_voltages.csv / .png
potential_animation.gif / .mp4 when ffmpeg is available
potential_snapshots.png
field_snapshots.png
equipotential_snapshots.png
direct_fem_validation.csv
multipole_spectrum.csv / .png
multipole_rotation.csv / .png
radial_field_scaling.csv / .png
effective_potential.png
effective_potential_radial.csv
trajectory_xy.png
trajectory_x_t.png
trajectory_y_t.png
trajectory_radius_t.png
trajectory_summary.csv
stability_scan.csv
stability_map.png
mesh_convergence.csv / .png
final_report.md
```

## Numerical limitations

The electrode and outer circles are represented by sampled boundaries in a Delaunay mesh, so a resolution study is necessary. P2 gradients are smooth within each element but remain C0 finite-element gradients across element boundaries; interpolation is valid only within the free-space triangulation. The outer grounded circle is finite rather than an exact open boundary. The localization scan and mass-filter map are exploratory ensembles, and trajectory conclusions depend on charge-to-mass ratio, RF frequency, damping, interaction time, tolerances, initial state, and the explicit escape criterion. The quasi-2D filter keeps axial velocity constant and omits axial fringe fields. The generated reports distinguish damping-assisted localization from the `gamma=0` case and do not declare a stable trap or mass filter without convergence evidence.
