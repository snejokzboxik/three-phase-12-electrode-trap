"""Configuration dataclasses and TOML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class GeometryConfig:
    electrode_count: int = 12
    electrode_orbit_radius: float = 0.010
    electrode_radius: float = 0.0015
    outer_domain_radius: float = 0.025


@dataclass(frozen=True)
class MeshConfig:
    resolution: float = 0.00065
    boundary_points_per_electrode: int = 18
    outer_boundary_points: int = 180
    interior_seed: int = 12345
    backend: str = "delaunay"
    mesh_file: str = "outputs/meshes/gmsh_70um/trap.msh"
    fine_radius: float = 0.008
    transition_radius: float = 0.014
    outer_resolution: float = 0.00035


@dataclass(frozen=True)
class RFConfig:
    U0: float = 1000.0
    frequency: float = 10000.0

    @property
    def omega(self) -> float:
        return 2.0 * 3.141592653589793 * self.frequency

    @property
    def period(self) -> float:
        return 1.0 / self.frequency


@dataclass(frozen=True)
class ParticleConfig:
    mass: float = 1.0e-15
    charge: float = 1.0e-12
    gamma: float = 0.0
    gravity_enabled: bool = False
    gravity: float = 9.81
    initial_position: tuple[float, float] = (0.001, 0.0)
    initial_velocity: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class SimulationConfig:
    frames_per_period: int = 180
    animation_periods: int = 2
    trajectory_periods: int = 40
    escape_radius: float = 0.008
    grid_size: int = 121
    trajectory_rtol: float = 2.0e-7
    trajectory_atol: float = 1.0e-11


@dataclass(frozen=True)
class StabilityConfig:
    U0_values: tuple[float, ...] = (500.0, 1000.0, 1500.0)
    frequency_values: tuple[float, ...] = (5000.0, 10000.0, 20000.0)
    initial_positions: tuple[tuple[float, float], ...] = ((0.0005, 0.0), (0.001, 0.0))


@dataclass(frozen=True)
class LocalizationConfig:
    U0_values: tuple[float, ...] = (500.0, 1000.0, 2000.0, 3000.0, 5000.0, 7500.0, 10000.0)
    frequency_values: tuple[float, ...] = (5000.0, 10000.0, 20000.0, 30000.0, 50000.0, 75000.0, 100000.0)
    initial_radii: tuple[float, ...] = (0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002)
    initial_angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    transverse_velocity_scales: tuple[float, ...] = (0.0, 0.02)
    stage_a_periods: int = 50
    stage_b_periods: int = 1000
    stage_c_periods: int = 10000
    stage_c_candidates: int = 6
    max_step_per_period: int = 10
    workers: int = 8


@dataclass(frozen=True)
class MassFilterConfig:
    length: float = 0.020
    axial_velocity: float = 2.0
    entrance_aperture: float = 0.008
    exit_aperture: float = 0.008
    reference_frequency: float = 10000.0
    mass_multipliers: tuple[float, ...] = (0.10, 0.15, 0.22, 0.32, 0.46, 0.68, 1.0, 1.47, 2.15, 3.16, 4.64, 6.81, 10.0)
    map_frequencies: tuple[float, ...] = (5000.0, 10000.0, 20000.0, 30000.0, 50000.0, 75000.0, 100000.0)
    ensemble_radii: tuple[float, ...] = (0.0001, 0.0005, 0.001)
    ensemble_angles_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    ensemble_velocity_scales: tuple[float, ...] = (0.0, 0.02)


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "outputs/three_phase_12_electrode"


@dataclass(frozen=True)
class ProjectConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    particle: ParticleConfig = field(default_factory=ParticleConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    mass_filter: MassFilterConfig = field(default_factory=MassFilterConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    def output_dir(self, project_root: Path | None = None) -> Path:
        root = Path.cwd() if project_root is None else Path(project_root)
        return root / self.outputs.directory


def _tuple2(values: list[float] | tuple[float, float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def load_config(path: str | Path) -> ProjectConfig:
    """Load a validated project configuration from a TOML file."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    g = raw.get("geometry", {})
    me = raw.get("mesh", {})
    rf = raw.get("rf", {})
    pa = raw.get("particle", {})
    si = raw.get("simulation", {})
    st = raw.get("stability", {})
    lo = raw.get("localization", {})
    mf = raw.get("mass_filter", {})
    out = raw.get("outputs", {})
    geometry = GeometryConfig(
        electrode_count=int(g.get("electrode_count", 12)),
        electrode_orbit_radius=float(g.get("electrode_orbit_radius", 0.010)),
        electrode_radius=float(g.get("electrode_radius", 0.0015)),
        outer_domain_radius=float(g.get("outer_domain_radius", 0.025)),
    )
    mesh = MeshConfig(
        resolution=float(me.get("resolution", 0.00065)),
        boundary_points_per_electrode=int(me.get("boundary_points_per_electrode", 18)),
        outer_boundary_points=int(me.get("outer_boundary_points", 180)),
        interior_seed=int(me.get("interior_seed", 12345)),
        backend=str(me.get("backend", "delaunay")),
        mesh_file=str(me.get("mesh_file", "outputs/meshes/gmsh_70um/trap.msh")),
        fine_radius=float(me.get("fine_radius", 0.008)),
        transition_radius=float(me.get("transition_radius", 0.014)),
        outer_resolution=float(me.get("outer_resolution", 0.00035)),
    )
    rf_config = RFConfig(float(rf.get("U0", 1000.0)), float(rf.get("frequency", 10000.0)))
    particle = ParticleConfig(
        mass=float(pa.get("mass", 1.0e-15)),
        charge=float(pa.get("charge", 1.0e-12)),
        gamma=float(pa.get("gamma", 0.0)),
        gravity_enabled=bool(pa.get("gravity_enabled", False)),
        gravity=float(pa.get("gravity", 9.81)),
        initial_position=_tuple2(pa.get("initial_position", [0.001, 0.0])),
        initial_velocity=_tuple2(pa.get("initial_velocity", [0.0, 0.0])),
    )
    simulation = SimulationConfig(
        frames_per_period=int(si.get("frames_per_period", 180)),
        animation_periods=int(si.get("animation_periods", 2)),
        trajectory_periods=int(si.get("trajectory_periods", 40)),
        escape_radius=float(si.get("escape_radius", 0.008)),
        grid_size=int(si.get("grid_size", 121)),
        trajectory_rtol=float(si.get("trajectory_rtol", 2.0e-7)),
        trajectory_atol=float(si.get("trajectory_atol", 1.0e-11)),
    )
    stability = StabilityConfig(
        tuple(float(v) for v in st.get("U0_values", [500.0, 1000.0, 1500.0])),
        tuple(float(v) for v in st.get("frequency_values", [5000.0, 10000.0, 20000.0])),
        tuple(_tuple2(v) for v in st.get("initial_positions", [[0.0005, 0.0], [0.001, 0.0]])),
    )
    localization = LocalizationConfig(
        tuple(float(v) for v in lo.get("U0_values", [500, 1000, 2000, 3000, 5000, 7500, 10000])),
        tuple(float(v) for v in lo.get("frequency_values", [5000, 10000, 20000, 30000, 50000, 75000, 100000])),
        tuple(float(v) for v in lo.get("initial_radii", [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002])),
        tuple(float(v) for v in lo.get("initial_angles_deg", [0, 45, 90, 135])),
        tuple(float(v) for v in lo.get("transverse_velocity_scales", [0.0, 0.02])),
        int(lo.get("stage_a_periods", 50)), int(lo.get("stage_b_periods", 1000)), int(lo.get("stage_c_periods", 10000)), int(lo.get("stage_c_candidates", 6)), int(lo.get("max_step_per_period", 10)), int(lo.get("workers", 8)),
    )
    mass_filter = MassFilterConfig(
        length=float(mf.get("length", 0.020)), axial_velocity=float(mf.get("axial_velocity", 2.0)),
        entrance_aperture=float(mf.get("entrance_aperture", 0.008)), exit_aperture=float(mf.get("exit_aperture", 0.008)),
        reference_frequency=float(mf.get("reference_frequency", 10000.0)),
        mass_multipliers=tuple(float(v) for v in mf.get("mass_multipliers", [0.10, 0.15, 0.22, 0.32, 0.46, 0.68, 1.0, 1.47, 2.15, 3.16, 4.64, 6.81, 10.0])),
        map_frequencies=tuple(float(v) for v in mf.get("map_frequencies", [5000, 10000, 20000, 30000, 50000, 75000, 100000])),
        ensemble_radii=tuple(float(v) for v in mf.get("ensemble_radii", [0.0001, 0.0005, 0.001])),
        ensemble_angles_deg=tuple(float(v) for v in mf.get("ensemble_angles_deg", [0, 90, 180, 270])),
        ensemble_velocity_scales=tuple(float(v) for v in mf.get("ensemble_velocity_scales", [0.0, 0.02])),
    )
    result = ProjectConfig(geometry, mesh, rf_config, particle, simulation, stability, localization, mass_filter, OutputConfig(str(out.get("directory", "outputs/three_phase_12_electrode"))))
    validate_config(result)
    return result


def validate_config(config: ProjectConfig) -> None:
    """Raise a clear error for invalid physical or numerical parameters."""
    g = config.geometry
    if config.mesh.backend not in ("delaunay", "gmsh"):
        raise ValueError("mesh.backend must be delaunay or gmsh.")
    if config.mesh.resolution <= 0:
        raise ValueError("Mesh resolution must be positive.")
    if config.mesh.backend == "gmsh":
        if not 0 < config.mesh.fine_radius < config.mesh.transition_radius <= g.outer_domain_radius:
            raise ValueError("Require 0 < fine_radius < transition_radius <= outer_domain_radius.")
        if config.mesh.outer_resolution < config.mesh.resolution:
            raise ValueError("outer_resolution must be at least resolution.")
    if g.electrode_count != 12:
        raise ValueError("This model requires exactly 12 electrodes.")
    if g.electrode_orbit_radius <= g.electrode_radius:
        raise ValueError("Electrode orbit radius must exceed electrode radius.")
    if g.outer_domain_radius <= g.electrode_orbit_radius + g.electrode_radius:
        raise ValueError("Outer domain must be outside every electrode.")
    if config.rf.frequency <= 0 or config.rf.U0 < 0:
        raise ValueError("RF frequency must be positive and U0 must be non-negative.")
    if config.particle.mass <= 0:
        raise ValueError("Particle mass must be positive.")
    if config.simulation.escape_radius >= g.electrode_orbit_radius - g.electrode_radius:
        raise ValueError("escape_radius must stay inside the free central region, below the electrode inner radius.")
    if config.simulation.frames_per_period < 12:
        raise ValueError("At least 12 frames per RF period are required.")
    if config.localization.stage_a_periods < 50 or config.localization.stage_a_periods > 100:
        raise ValueError("Stage A must screen the full grid for 50 to 100 RF periods.")
    if config.localization.stage_b_periods < 1000:
        raise ValueError("Stage B must run for at least 1000 RF periods.")
    if config.localization.stage_c_periods < 10000:
        raise ValueError("Stage C must run for at least 10000 RF periods.")
    if config.localization.workers < 1:
        raise ValueError("Localization workers must be positive.")
