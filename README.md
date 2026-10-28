# Three-phase 12-electrode trap

**Код и выбранные данные:** [состав опубликованного набора](docs/GITHUB_DATA.md),
[список крупных файлов, оставленных локально](docs/LOCAL_ONLY_DATA.md).
Новые данные доступны обычным `git clone` / `git pull`, Git LFS не требуется.
`python scripts/verify_repository_data.py` проверяет опубликованный набор
по `outputs/REPOSITORY_DATA_MANIFEST.json`; `--full-local` дополнительно
проверяет исключённые массивы на исходном компьютере.

**Продолжение расчёта Gmsh 70 мкм:** начните с
[контекста для другого чата](docs/FEM_GMSH_70UM.md) и
[отчёта готового FEM-расчёта](outputs/fem_cache/FEM_RUN_REPORT.md).
Там указаны параметры, готовые файлы, загрузка без пересчёта и план проверки траекторий.
Подробный порядок следующего этапа: [план проверки моделей A/B/C](docs/ABC_VALIDATION_PLAN.md).
Выполненные первые проверки: [протокол, гладкие эталоны и аудит поля](outputs/abc_validation/INITIAL_CHECKS_REPORT.md).
Пункт 4 выполнен: [модель B, сила и проверки интегратора](outputs/abc_validation/model_b/MODEL_B_REPORT.md).
Пункт 5 выполнен: [коэффициенты октупольной модели C и сравнение B–C](outputs/abc_validation/model_c/MODEL_C_REPORT.md).
Пункт 6 выполнен: [общие начальные условия и проверка фильтров](outputs/abc_validation/filter_checks/INITIAL_CONDITIONS_REPORT.md).
Пункт 7 рассчитан: [длительный пилот и ограничения вывода](outputs/abc_validation/pilot/PILOT_REPORT.md).
Пункт 8: [контролируемый высокочастотный предел](outputs/abc_validation/frequency_limit/FREQUENCY_LIMIT_REPORT.md)
при одновременном масштабировании частоты и напряжения ×1,2,4.
Дополнительная диагностика исходного A: [аудит переходов между FEM-элементами](docs/RAW_FEM_EDGE_AUDIT.md)
и [отчёт проверок](outputs/abc_validation/raw_edge_audit/v2/EDGE_AUDIT_REPORT.md).
Пункт 2 диагностики: [перезапуски из общего состояния и чувствительность траекторий](outputs/abc_validation/raw_restart_diagnosis/v1/RESTART_DIAGNOSIS_REPORT.md).
Пункт 3 диагностики: [FEM 70/50/35 мкм, поля и траектории](outputs/abc_validation/raw_spatial_diagnosis/v2/SPATIAL_DIAGNOSIS_REPORT.md),
[описание данных и воспроизведения](docs/RAW_FEM_SPATIAL_DIAGNOSIS.md).
**Прямая проверка FEM A завершена 24.10.2026:**
[результат и контекст для продолжения](docs/DIRECT_FEM_A_RESULTS.md),
[полный отчёт P4](outputs/abc_validation/direct_fem_validation/v2/DIRECT_FEM_A_REPORT.md).
Решена та же физическая задача с прямым несглаженным градиентом P4
на сетках 50/35 мкм; временные, межсеточные и фильтрационные контроли пройдены.
A–B/C проходят допуски на 3,25 медленного периода (RMS 0,555–0,597%),
но не на 10,25 (RMS 1,749%, max 4,500%). Это определённый результат проверки.
Исторические несходящиеся траектории P2 не объявляются точными.
Новый эталон P4 запускается отдельно; исходный P2-кэш и обычный CLI сохранены.

Подготовлен [отдельный запуск A с t=0 до 50 медленных периодов](docs/RUN_A_50_PERIODS.md):
`scripts/run_A_50_periods.ps1`, контрольные сохранения, продолжение после остановки,
проверка результата и маркер `COMPLETED.json`. Пользователь завершил этот
расчёт; [анализ 50 периодов](docs/LONG_A_50_RESULTS.md) выполнен.
Наблюдаемое A–B35/C35: RMS 8,5225%, max 22,4521%; линия колебаний поворачивается
примерно на 13° при почти постоянной радиальной амплитуде. Это результат
одной траектории: глобальная сходимость A по сетке/шагу на 50 Ts ещё не проверена.

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

The separate direct-A validation uses P4 FE solutions of this same physical
boundary problem, an exact complex cyclic sector reduction, and native element
gradients. It does not substitute the harmonic reconstruction in A's force.
See [the frozen protocol](docs/DIRECT_FEM_A_PROTOCOL.md) and
[the completed numerical/physical comparison](docs/DIRECT_FEM_A_RESULTS.md).

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
python -m three_phase_trap.cli pseudopotential
python -m three_phase_trap.cli dc-bias
python -m three_phase_trap.cli finalize-dc-bias
python -m three_phase_trap.cli trajectory
python -m three_phase_trap.cli stability
python -m three_phase_trap.cli convergence
python -m three_phase_trap.cli mass-filter
python -m three_phase_trap.cli publication
python -m three_phase_trap.cli all
```

Use another TOML file with `--config path/to/config.toml`. The all-in-one command runs the basis FEM solve, direct validation, voltage diagnostics, snapshots, GIF animation, optional MP4 animation when ffmpeg is installed, multipole spectrum and rotation, radial scaling, effective potential, conservative and damped trajectory examples, stability scan, mesh convergence, and a generated `final_report.md`. The `publication` command additionally creates P2-based high-resolution figures, a checkpointed staged localization scan (50-period Stage A, 1000-period Stage B survivors, and 10000-period Stage C candidates), finite-flight mass transmission, and a mass-frequency survival map in `outputs/publication_study/`. Stage-A results are screening-only; stability conclusions use Stage B/C trajectories.

The `pseudopotential` command creates dense regular-grid pseudopotential maps, a monotonic sixth-root central transform, full and central radial cuts, fixed-radius angular anisotropy plots, and a dedicated central mesh-convergence report. Publication maps use an explicitly labeled area-weighted nodal recovery of the P2 gradient for readability; direct elementwise grids remain available for numerical audit.

The `dc-bias` command adds a common 1-V FEM basis with all 12 electrodes biased together and creates a comparative study at Udc=0, 100, 300, 500, and 1000 V. After an existing run, `finalize-dc-bias` validates linear superposition against independent direct FEM solves at multiple nonzero biases and RF phases, writes the concise differential summary, and separates RF pseudopotential, static DC energy/field, total instantaneous field reconstruction, and the optional total-field proxy. It reads the existing DC-bias grids/trajectory CSVs and does not rerun the main DC study. Outputs are stored in `outputs/publication_study/dc_bias/`. The 40-period trajectory subset is exploratory only; it is not a stability conclusion.

## Symmetric Gmsh mesh

Install optional meshing dependencies: `python -m pip install -e ".[gmsh]"`.
From the repository root run:

```bash
python -m three_phase_trap.cli mesh --config configs/gmsh_70um.toml
```

This creates `outputs/meshes/gmsh_70um/trap.msh`, `trap.json`, the preview
`trap.png`, and reproducible sector files `sector.geo` and `sector.msh`.
All coordinates are in metres. A 30-degree sector with a circular electrode
hole is meshed with matching radial interfaces, rotated twelve times and
welded. Sector interfaces are interior edges with no artificial FEM boundary
conditions. Physical tags identify every electrode and the outer boundary;
untagged external edges (including unwelded seams) are rejected on import.

The target size is 70 micrometres at r <= 8 mm and near electrodes, increasing
continuously towards 350 micrometres outside. Targets are not strict bounds on
individual edges. `trap.json` reports measured edge quantiles, triangle quality,
area error relative to exact circles, 12-fold symmetry, and topology. Circular
CAD boundaries are approximated by straight mesh edges. The imported geometry
is affine; the existing solver constructs quadratic solution DOFs using
`ElementTriP2`. Curved P2 geometry is not used.

The `mesh` command does **not** solve Laplace's equation. A later explicit
field calculation uses exactly this saved mesh:

```bash
python -m three_phase_trap.cli fem --config configs/gmsh_70um.toml
```

The FEM fingerprint includes the MSH checksum and Gmsh version. The default
configuration keeps the old Delaunay generator. Legacy automatic convergence
commands reject Gmsh configurations; generate separate mesh files/configs at
each resolution before comparing them.

## Reusable P2 FEM cache

The validated mesh and four P2 basis solutions (`A`, `B`, `C`, and common `DC`) are cached under `outputs/fem_cache/`. `metadata.json` includes a deterministic fingerprint of geometry, mesh settings, element/solver formulation, boundary conditions, phase protocol, FEM source hashes, dependency version, and the creating Git revision. Cached data include mesh/connectivity, electrode/phase boundary-node sets, P2 DOF coordinates/order, basis potentials, and element fields. Normal trajectory, localization, and mass-filter runs load this cache. A changed FEM fingerprint is never silently substituted; mesh-refinement configurations use separate fingerprinted variant directories. The cache was round-trip loaded and checked against independent direct RF+DC FEM solves before the stability scan.

### Verify and archive a completed Gmsh FEM calculation

After `python -m three_phase_trap.cli fem --config configs/gmsh_70um.toml`
has completed, run:

```bash
python scripts/archive_fem_run.py --config configs/gmsh_70um.toml
```

This reads the existing cache without rebuilding any basis. It checks the
P2 Dirichlet values (including edge DOFs), weak Laplace residuals, saved field
reconstruction, exact array reload, and relative RMS errors in the independent
direct RF/RF+DC solves. It saves `basis_integrity_checks.json`, a copy of the
configuration, library versions and SHA-256 manifests, the actual working-tree
source snapshot (including uncommitted changes), and a portable
`gmsh_70um_fem_bundle.zip` with mesh, basis, sources and validation reports.
The bundle checksum is saved separately as `bundle_checksum.json`.
These checks establish discrete-solve and serialization consistency, not
spatial convergence or the validity of averaged particle dynamics.

## Full-dynamics stability study

Run `python -m three_phase_trap.cli stability-study` to resume the publication stability analysis. It integrates `m r¨ = q E(r,t)` with `gamma=0`, gravity disabled, DOP853, the configured tolerances and maximum RF-period step, and a terminal `r >= 8 mm` escape event. Stage A screens the full 10 × 9 coarse `(U0,f_RF)` grid for 50 periods; Stage B runs 1000 periods only for survivors; Stage C runs 10000 periods on representative robust and near-boundary cases. Each pair includes seven radii × four angles at rest and four nonzero tangential-velocity checks. Log-midpoint parameter refinement is limited to coarse-grid Stage-B transitions. Stage-A survivors are not called stable; strict stability requires all 32 primary and velocity-probe starts to remain bounded for 1000 periods.

Trajectory rows are fsync-checkpointed after each completed case in `outputs/publication_study/stability/stability_checkpoint.jsonl`; the separate existing `outputs/publication_study/localization_checkpoint.csv` is only read for exact overlapping Stage-A cases and is not modified. The finalized datasets are `stability_cases.csv` and `stability_summary.csv`. Publication figures include strict binary stability, continuous survival fraction, escape-duration and maximum-radius maps, refined observed points, initial-radius dependence, fourth-order dimensionless scaling, and representative XY/radius/Poincaré trajectories. No interpolation is drawn across uncomputed parameter points. Stage-C results validate only the listed initial states and do not imply global stability. This command does not run the mass-filter study.

In the completed comparison, common Udc leaves the RF pseudopotential unchanged. At 1000 V, the heuristic proxy increment inside r≤2 mm is about 3.16×10⁻²⁰ J (3.16×10⁻⁵ fJ), negligible relative to the RF-well range; the added field is static and is retained separately in the trajectory force. The one exploratory initial condition escaped after 17.72 periods at 0 V, 16.56 periods at 300 V, and 16.49 periods at 1000 V; these short runs do not establish stability or general localization behavior. See `outputs/publication_study/dc_bias/dc_bias_report.md` for direct-FEM errors and per-bias metrics.

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
