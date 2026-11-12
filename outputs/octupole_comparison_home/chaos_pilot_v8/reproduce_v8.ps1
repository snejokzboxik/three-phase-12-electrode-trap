$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Here
try {
    python .\analyze_v8.py
    python .\render_v8_figures.py
    python .\validate_v8.py
} finally { Pop-Location }
