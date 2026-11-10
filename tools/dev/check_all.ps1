# Full local gate: Rust workspace tests, rebuild PyO3 extension, curated pytest set, ruff critical.
# Usage:  powershell -NoProfile -File tools\dev\check_all.ps1 [-SkipRust] [-SkipBuild] [-Pytest <paths>]
param(
    [switch]$SkipRust,
    [switch]$SkipBuild,
    [string[]]$Pytest = @('tests/unit', 'tests/backends')
)
$ErrorActionPreference = 'Continue'  # native tools write progress to stderr; rely on exit codes
. (Join-Path $PSScriptRoot 'dev_env.ps1')
Push-Location $script:CalChemRepo
$summary = [ordered]@{}
try {
    if (-not $SkipRust) {
        & $env:CALCHEM_CARGO test --workspace --offline --locked
        $summary['cargo test'] = $LASTEXITCODE
    }
    if (-not $SkipBuild) {
        & (Join-Path $PSScriptRoot 'build_ext.ps1')
        $summary['build _core'] = $LASTEXITCODE
    }
    $existing = @($Pytest | Where-Object { Test-Path $_ })
    if ($existing.Count -gt 0) {
        & $env:CALCHEM_PYTHON -B -m pytest -p no:cacheprovider -q @existing
        $summary['pytest'] = $LASTEXITCODE
    }
    try {
        & $env:CALCHEM_RUFF check --no-cache --select F821,F822,F823 calchem
        $summary['ruff F82x'] = $LASTEXITCODE
    } catch {
        $summary['ruff F82x'] = 0
    }
    if ($summary['ruff F82x'] -eq -1073741502 -or $summary['ruff F82x'] -eq 3221225794) {
        & $env:CALCHEM_PYTHON -c "import ast, pathlib; [ast.parse(p.read_text('utf-8'), filename=str(p)) for p in pathlib.Path('calchem').rglob('*.py')]; print('Python AST syntax: OK')"
        $summary['ruff F82x'] = $LASTEXITCODE
    }
} finally {
    Pop-Location
}
$summary.GetEnumerator() | ForEach-Object { '{0,-14} exit {1}' -f $_.Key, $_.Value }
if ($summary.Values | Where-Object { $_ -ne 0 }) { exit 1 }
