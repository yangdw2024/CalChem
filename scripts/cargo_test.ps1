param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\docs\audits\2026-11-10_round2\probes\env.ps1")

$env:RAYON_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '1'
$env:CARGO_PROFILE_DEV_OPT_LEVEL = '2'
$env:CARGO_PROFILE_TEST_OPT_LEVEL = '2'

if ($CargoArgs.Count -eq 0) {
    cargo test --workspace --no-fail-fast --jobs 4 --offline --locked
} else {
    cargo @CargoArgs
}
exit $LASTEXITCODE
