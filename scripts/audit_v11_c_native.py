"""Direct native P4 check of a selected early C exit, one mesh per process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap import direct_fem
from three_phase_trap.direct_fem import DirectFEMRF, SectorSolution
from three_phase_trap.gmsh_mesh import load_gmsh_mesh
from three_phase_trap.stability_rf_p4 import NativeEscape

OUT = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11"


def main(mesh):
    # DirectFEMRF rotates the native sector solution through the cyclic factor.
    # Isolate the anti-periodic character in this process; the ABC source file
    # and archived V10 cache remain unchanged.
    direct_fem.RHO = -1.+0j
    sector = SectorSolution.load(OUT / f"fields/h{mesh}_P4_C/sector_solution.npz")
    trap = load_gmsh_mesh(ROOT / f"outputs/meshes/gmsh_{mesh}um/trap.msh")
    field = DirectFEMRF(sector, trap, amplitude=250.)
    initial = np.array([.00375, -.006, 0., 0.])
    escape, state, maximum, nfev = NativeEscape(field, .008).run(
        initial, 1.602176634e-19/(100000*1.66053906660e-27), 100000., 0., 1000,
        steps=50, rtol=2e-9, atol=[2e-13, 2e-13, 2e-11, 2e-11])
    folder = OUT / "numerical_checks/native_C_early_exit"
    folder.mkdir(parents=True, exist_ok=True)
    result = {"mesh_um": mesh, "method": "direct native P4, audited element crossings",
              "initial_state_SI": initial.tolist(), "phase_deg": 0., "voltage_V": 250.,
              "escape_RF": escape, "final_state_SI": state.tolist(),
              "max_radius_m": maximum, "nfev": nfev}
    (folder / f"h{mesh}.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"mesh_um": mesh, "escape_RF": escape, "nfev": nfev}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, choices=(35, 50), required=True)
    main(parser.parse_args().mesh)
