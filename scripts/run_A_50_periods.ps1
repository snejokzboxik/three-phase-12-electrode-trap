param(
    [switch]$CheckOnly,
    [switch]$Status,
    [switch]$Verify,
    [switch]$SmokeTest,
    [ValidateSet(35, 50)][int]$Mesh = 35,
    [ValidateSet('fine', 'coarse')][string]$Level = 'fine'
)

$ErrorActionPreference = 'Stop'
$selectedModes = @($CheckOnly, $Status, $Verify) | Where-Object { $_ }
if (@($selectedModes).Count -gt 1) {
    throw 'Use only one of -CheckOnly, -Status, -Verify.'
}
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = 'C:/Python314/python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Scientific Python not found: $pythonExecutable"
}
$runArguments = @('-u', 'scripts/run_long_fem_a.py', '--slow-periods', '50', '--mesh', "$Mesh", '--level', $Level)
if ($CheckOnly) { $runArguments += '--check-only' }
if ($Status) { $runArguments += '--status' }
if ($Verify) { $runArguments += '--verify' }
if ($SmokeTest) { $runArguments += '--smoke-test' }

Push-Location -LiteralPath $projectDirectory
try {
    & $pythonExecutable @runArguments
    $runExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
if ($runExitCode -ne 0) {
    Write-Host "Calculation did not finish successfully (exit code $runExitCode). Check status.json/run.log/error.txt in the printed output directory."
}
exit $runExitCode
