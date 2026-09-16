param(
    [string]$Config = "configs/default.toml"
)

python -m three_phase_trap.cli all --config $Config

