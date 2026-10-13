"""Command-line entry point for reproducible trap analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_all, run_effective_potential, run_fem, run_mesh_convergence, run_multipoles, run_radial_scaling, run_rotation, run_stability, run_trajectory, run_visuals


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="FEM and particle-dynamics study of a three-phase 12-electrode trap")
    parser.add_argument("command", choices=("fem", "animate", "multipoles", "effective-potential", "pseudopotential", "dc-bias", "finalize-dc-bias", "combined-effective-potential", "trajectory", "stability", "stability-study", "convergence", "mass-filter", "publication", "all"))
    parser.add_argument("--config", type=Path, default=root / "configs" / "default.toml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "fem":
        result = run_fem(config, root)
        print(f"FEM complete: {result['summary']}")
    elif args.command == "animate":
        result = run_visuals(config, root)
        print(f"Visualization complete: {result['animation']}")
    elif args.command == "multipoles":
        state = run_fem(config, root)
        result = run_multipoles(config, root, (state["geometry"], state["trap_mesh"], state["solutions"], state["interpolator"]))
        rotation = run_rotation(config, root, result, (state["geometry"], state["trap_mesh"], state["solutions"], state["interpolator"]))
        radial = run_radial_scaling(config, root, (state["geometry"], state["trap_mesh"], state["solutions"], state["interpolator"]))
        print(f"Multipole analysis complete: dominant m={result['dominant_order']}; rotation={rotation['fit']['angular_rate']:.6g} rad/s; radial p={radial['exponent']:.6g}")
    elif args.command == "effective-potential":
        state = run_fem(config, root)
        result = run_effective_potential(config, root, (state["geometry"], state["trap_mesh"], state["solutions"], state["interpolator"]))
        print(f"Effective-potential analysis complete: center is {result['classification']}")
    elif args.command == "pseudopotential":
        from .pipeline import make_state
        from .publication import publication_dir, publication_pseudopotential_refined
        state = make_state(config)
        result = publication_pseudopotential_refined(publication_dir(root), state[3], state[0], config)
        print(f"Pseudopotential visualization complete: central minimum={result['central_min_J']:.6e} J; mesh spread ratio coarse-to-fine={result['mesh_convergence']['coarse_to_fine_spread_ratio']:.6g}")
    elif args.command == "dc-bias":
        from .publication import run_dc_bias_comparison
        result = run_dc_bias_comparison(config, root)
        print(f"DC-bias comparison complete: {len(result['cases'])} bias cases; report={result['report']}")
    elif args.command == "finalize-dc-bias":
        from .publication import finalize_dc_bias_outputs
        result = finalize_dc_bias_outputs(config, root)
        print(f"DC-bias validation and figure finalization complete: {result['report']}")
    elif args.command == "combined-effective-potential":
        from .publication import generate_combined_secular_effective_potential
        result = generate_combined_secular_effective_potential(config, root)
        print(f"Combined secular effective-potential figures complete: {result['report']}")
    elif args.command == "trajectory":
        state = run_fem(config, root)
        result = run_trajectory(config, root, (state["geometry"], state["trap_mesh"], state["solutions"], state["interpolator"]))
        print(f"Trajectory analysis complete: base case bounded={result['base'].bounded}, damped case bounded={result['damped'].bounded}")
    elif args.command == "stability":
        state = run_fem(config, root)
        result = run_stability(config, root, (state["geometry"], state["trap_mesh"], state["solutions"], state["interpolator"]))
        print(f"Stability scan complete: {len(result['rows'])} integrations")
    elif args.command == "stability-study":
        from .stability_study import run_stability_study
        result = run_stability_study(config, root)
        print(f"Stability study complete: {result['case_count']} trajectory rows; report={result['report']}")
    elif args.command == "convergence":
        result = run_mesh_convergence(config, root)
        print(f"Mesh-convergence check complete: {len(result['rows'])} resolutions")
    elif args.command == "mass-filter":
        from .pipeline import make_state
        from .publication import run_mass_frequency_map, run_mass_transmission_scan
        from .mass_filter import write_mass_transmission_csv
        state = make_state(config)
        output_dir = root / "outputs" / "publication_study"
        output_dir.mkdir(parents=True, exist_ok=True)
        transmission = run_mass_transmission_scan(state[3], config)
        write_mass_transmission_csv(transmission, output_dir / "mass_transmission.csv")
        print(f"Mass-filter transmission scan complete: {len(transmission)} masses")
    elif args.command == "publication":
        from .publication import run_publication_study
        result = run_publication_study(config, root)
        print(f"Publication study complete. Report: {result['report']}")
    else:
        result = run_all(config, root)
        print(f"Complete analysis finished. Report: {result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
