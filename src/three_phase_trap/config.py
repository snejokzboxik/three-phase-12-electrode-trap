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
    result = ProjectConfig(geometry, mesh, rf_config, particle, simulation, stability, OutputConfig(str(out.get("directory", "outputs/three_phase_12_electrode"))))
    validate_config(result)
    return result


def validate_config(config: ProjectConfig) -> None:
    """Raise a clear error for invalid physical or numerical parameters."""
    g = config.geometry
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
