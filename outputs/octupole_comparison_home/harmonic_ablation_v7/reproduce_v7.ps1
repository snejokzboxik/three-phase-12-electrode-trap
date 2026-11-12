$ErrorActionPreference = 'Stop'
$studyDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $studyDir
try {
    # Rebuild derived tables and publication figures from the committed,
    # checkpointed trajectory records. This script submits no RF integrations.
    python .\run_v7.py --stage validate
    python .\analyze_v7.py
    python .\analyze_pseudopotential_cross_terms.py
    python .\render_v7_figures.py
}
finally {
    Pop-Location
}
