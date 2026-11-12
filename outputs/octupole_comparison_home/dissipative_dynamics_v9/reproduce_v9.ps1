$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here "..\..\..")).Path
Push-Location $repo
try {
    python (Join-Path $here "run_v9.py") --stage sweep
    python (Join-Path $here "run_v9.py") --stage h35-check
    python (Join-Path $here "run_v9.py") --stage lyapunov --field h50 --model rotating --gamma 1000 --launch deep_0deg
    python (Join-Path $here "run_v9.py") --stage lyapunov --field h50 --model standing --gamma 1000 --launch deep_0deg
    python (Join-Path $here "render_v9_figures.py")
    python (Join-Path $here "validate_v9.py")
}
finally { Pop-Location }
