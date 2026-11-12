$ErrorActionPreference = 'Stop'

# Resumable V6 workflow. Completed trajectory IDs are read from the existing
# checkpoint; the original integration_deadline.json remains authoritative.
python run_v6.py --stage validate
python run_v6.py --stage pilot --workers 4
python run_v6.py --stage scan --workers 4
for ($i = 0; $i -lt 8; $i++) {
    python run_v6.py --stage refine --workers 4
    if ($LASTEXITCODE -ne 0) { throw 'Adaptive refinement failed.' }
}
python run_v6.py --stage phase-check --workers 4
python strict_validation_v6.py
python analyze_v6.py
python phase_space_diagnostics_v6.py
python render_v6_figures.py
