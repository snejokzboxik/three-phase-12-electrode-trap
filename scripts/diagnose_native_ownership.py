"""Local replay of P4 cell-ownership failure; never a validation trajectory."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from build_direct_fem import load_field, OUTPUT
from three_phase_trap.direct_fem_dynamics import integrate_native, local_field
from run_abc_pilot import write


def main():
    field = load_field(35, 4, 250.)
    with np.load(OUTPUT/"runs/A_fit_h35_phase0_fine_pilot.npz") as z:
        t, y = z["time_s"], z["states"]
    i = int(np.searchsorted(t, .060))
    try:
        out = integrate_native(field, y[i], 1e-12, 1e-15, 1e5, 0., t[i], t[-1], progress=lambda v: print(v, flush=True))
        print("Diagnostic replay completed", out.diagnostics)
    except Exception as exc:
        tb = exc.__traceback__
        while tb.tb_next and tb.tb_frame.f_code.co_name != "integrate_native":
            tb = tb.tb_next
        v = tb.tb_frame.f_locals
        result = {"exception": repr(exc), "line": tb.tb_lineno}
        for name in ("cell", "other", "previous", "stop", "fraction", "event", "state", "bary"):
            if name in v:
                value = v[name]
                result[name] = value.tolist() if isinstance(value, np.ndarray) else value
        if "state" in v:
            state = v["state"]
            for cell in (v["cell"], *field.neighbors[v["cell"]]):
                if cell < 0:
                    continue
                g = field.gradient[cell]
                bc = g@(state[:2]-field.origin[cell]); bc[0] += 1
                e = local_field(field, cell, state, 2*np.pi*1e5*v["stop"])[0]
                result["cell_"+str(cell)] = dict(bary=bc.tolist(), velocity=(g@state[2:]).tolist(), acceleration=(g@(1e3*e)).tolist())
        write(OUTPUT/"ownership_failure_replay.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
