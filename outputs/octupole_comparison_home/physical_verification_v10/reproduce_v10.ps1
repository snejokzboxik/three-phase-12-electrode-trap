param(
    [ValidateSet("figures", "checks", "p4", "robustness", "phase-control")]
    [string]$Mode = "figures",
    [int]$MeshUm = 50,
    [string[]]$Launches = @("negative_retained", "negative_escaped", "positive_retained", "positive_escaped")
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$PythonDeps = Join-Path $env:LOCALAPPDATA "Temp\v10deps"
$env:PYTHONPATH = "$PythonDeps;$RepoRoot\src;$PSScriptRoot"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"

function Invoke-Python([string[]]$Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python step failed with exit code $LASTEXITCODE" }
}

Push-Location $RepoRoot
try {
    switch ($Mode) {
        "figures" {
            Invoke-Python @("$PSScriptRoot\make_publication_figures.py")
            Invoke-Python @("$PSScriptRoot\validate_publication_figures.py")
        }
        "checks" {
            Invoke-Python @("$PSScriptRoot\validate_direct_p4_field.py")
            Invoke-Python @("$PSScriptRoot\harmonic_cross_audit_p4.py")
            Invoke-Python @("$PSScriptRoot\summarize_native_p4.py")
            Invoke-Python @("$PSScriptRoot\validate_parameter_robustness.py")
            Invoke-Python @("$PSScriptRoot\validate_publication_figures.py")
            $CompileArgs = @("-m", "py_compile") +
                @(Get-ChildItem $PSScriptRoot -Filter "*.py" | ForEach-Object { $_.FullName })
            Invoke-Python $CompileArgs
        }
        "p4" {
            $P4CommandArgs = @("$PSScriptRoot\run_p4_batch.py", "--mesh", "$MeshUm", "--launches") +
                $Launches + @("--workers", "1", "--horizon-rf", "1000")
            Invoke-Python $P4CommandArgs
        }
        "robustness" {
            $RobustnessLaunches = @("robust_E1_r1_az22p5_dplus", "robust_E3_r1_az22p5_dplus", "robust_E2_r2_az22p5_dplus")
            $P4CommandArgs = @("$PSScriptRoot\run_p4_batch.py", "--mesh", "50", "--launches") +
                $RobustnessLaunches + @("--workers", "1", "--horizon-rf", "1000")
            Invoke-Python $P4CommandArgs
            Invoke-Python @("$PSScriptRoot\validate_parameter_robustness.py")
        }
        "phase-control" {
            Invoke-Python @("$PSScriptRoot\phase_structure_control.py")
        }
    }
}
finally {
    Pop-Location
}
