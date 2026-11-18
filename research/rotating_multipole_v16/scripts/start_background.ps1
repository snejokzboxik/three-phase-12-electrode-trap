param([double]$Hours = 10)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Results = Join-Path $Root 'results'
New-Item -ItemType Directory -Force -Path $Results | Out-Null
$OutLog = Join-Path $Results 'process_stdout.log'
$ErrLog = Join-Path $Results 'process_stderr.log'
$Python = (Get-Command python).Source
$Runner = Join-Path $PSScriptRoot 'run_search.py'
$Proc = Start-Process -FilePath $Python -ArgumentList @('-u', $Runner, '--hours', "$Hours") `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog
Start-Sleep -Seconds 2
$Proc.Refresh()
if ($Proc.HasExited) { throw "Background runner exited immediately (code $($Proc.ExitCode)); inspect $ErrLog" }
@{ pid = $Proc.Id; started = (Get-Date).ToString('o'); runner = $Runner } |
    ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Results 'process.json')
Write-Output "Started detached runner PID $($Proc.Id). Logs: $OutLog"
