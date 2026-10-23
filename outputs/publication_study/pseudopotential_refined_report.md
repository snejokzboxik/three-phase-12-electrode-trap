# Refined pseudopotential visualization report

The publication maps use dense Cartesian regular grids and an area-weighted nodal recovery of the discontinuous elementwise P2 gradient. The FEM solution and direct elementwise field are unchanged; recovered-grid values are a labeled visualization postprocessing projection.

Grid sizes: full=501x501, central=601x601. The transformed map is [(ΨRF-Ψmin)/(10^-10 J)]^(1/6), a monotonic display coordinate that makes an r^6 well approximately linear in radius.

Central minimum on the recovered grid: 3.58078876e-38 J. At r=3 mm, the coarse-to-fine angular-spread ratio is 0.877606.

## Mesh convergence

| radius | coarse relative spread | fine relative spread | fine/coarse absolute spread |
|---:|---:|---:|---:|
| 1 mm | 2.61824 | 1.0578 | 0.449039 |
| 2 mm | 1.18613 | 0.412735 | 0.401234 |
| 3 mm | 0.692345 | 0.579291 | 0.877606 |

The angular irregularity decreases clearly at 2 mm and modestly at 3 mm as the mesh is refined, but it remains nonzero. The refined figures therefore support a central low/flat-bottom region with residual anisotropy that is still partly mesh-sensitive; they do not by themselves establish a converged physical higher-multipole coefficient.
