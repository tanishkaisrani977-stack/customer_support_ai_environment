param(
    [ValidateSet("app", "inference", "inference-strict", "check", "tests", "all")]
    [string]$Action = "app"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

switch ($Action) {
    "app" {
        & $python app.py
    }
    "check" {
        & $python -c "import sys, inference;`ntry:`n    inference.print_runtime_check()`nexcept RuntimeError as exc:`n    print(f'[ERROR] {exc}', file=sys.stderr)`n    raise SystemExit(1)"
    }
    "inference" {
        Remove-Item Env:OPENENV_STRICT_INFERENCE -ErrorAction SilentlyContinue
        & $python inference.py
    }
    "inference-strict" {
        $env:OPENENV_STRICT_INFERENCE = "1"
        & $python inference.py
    }
    "tests" {
        & $python -m unittest discover -s tests -v
    }
    "all" {
        & $python -m unittest discover -s tests -v
        & $python inference.py
    }
}
