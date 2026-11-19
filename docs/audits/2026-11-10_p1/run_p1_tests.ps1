$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\2026-11-10_round2\probes\env.ps1"

$env:PYO3_PYTHON = 'D:\YDW\Trae_Gromacs\miniconda3\python.exe'
$env:CARGO_PROFILE_DEV_OPT_LEVEL = '2'
$env:CARGO_PROFILE_TEST_OPT_LEVEL = '2'

Write-Host "=== 1. Running Cargo Workspace Tests ==="
cargo test --workspace --locked
Write-Host "Cargo exit code: $LASTEXITCODE"

Write-Host "=== 2. Building calchem-py in release mode ==="
cargo build --release --locked -p calchem-py
Write-Host "Cargo build exit code: $LASTEXITCODE"

# Copy _core.dll to calchem/_core.pyd
$builtPyd = Get-ChildItem "$($env:CARGO_TARGET_DIR)\release\_core.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $builtPyd) {
    $builtPyd = Get-ChildItem "build_target\release\_core.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if ($builtPyd) {
    Write-Host "Copying $($builtPyd.FullName) to calchem\_core.pyd"
    Copy-Item $builtPyd.FullName "calchem\_core.pyd" -Force
}

Write-Host "=== 3. Running Pytest for native backend tests ==="
& $env:PYO3_PYTHON -B -c "import os,tempfile,pytest; tempfile.tempdir=os.environ['TEMP']; raise SystemExit(pytest.main(['tests/unit/test_native_backend.py','-v','-p','no:cacheprovider']))"
Write-Host "Pytest exit code: $LASTEXITCODE"
