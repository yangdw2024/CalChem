# Full local gate: Rust workspace tests and Release CLI build (capped at 4 threads).
# Usage:  powershell -NoProfile -File tools\dev\check_all.ps1 [-SkipTests] [-SkipBuild]
param(
    [switch]$SkipTests,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Continue'  # native tools write progress to stderr; rely on exit codes
. (Join-Path $PSScriptRoot 'dev_env.ps1')
$env:RAYON_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '1'
Push-Location $script:CalChemRepo
$summary = [ordered]@{}
try {
    if (-not $SkipTests) {
        & $env:CALCHEM_CARGO test --workspace --no-fail-fast --jobs 4
        $summary['cargo test'] = $LASTEXITCODE
    }
    if (-not $SkipBuild) {
        & $env:CALCHEM_CARGO build --release -p calchem-cli --jobs 4
        $summary['build cli'] = $LASTEXITCODE
    }
} finally {
    Pop-Location
}
$summary.GetEnumerator() | ForEach-Object { '{0,-14} exit {1}' -f $_.Key, $_.Value }
if ($summary.Values | Where-Object { $_ -ne 0 }) { exit 1 }
