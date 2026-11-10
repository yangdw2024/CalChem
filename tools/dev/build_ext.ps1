# Build calchem-py (release) and install it as calchem\_core.pyd.
# The previous .pyd is copied to build_target\runtime\pyd-backup\ first (never deleted).
$ErrorActionPreference = 'Continue'  # native tools write progress to stderr; rely on exit codes
. (Join-Path $PSScriptRoot 'dev_env.ps1')
Push-Location $script:CalChemRepo
try {
    & $env:CALCHEM_CARGO build --release --offline --locked -p calchem-py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $built = Join-Path $env:CARGO_TARGET_DIR 'release\_core.dll'
    if (-not (Test-Path $built)) { $built = Join-Path $env:CARGO_TARGET_DIR 'release\calchem_py.dll' }
    if (-not (Test-Path $built)) { throw "Built extension not found under $env:CARGO_TARGET_DIR\release" }
    $dest = Join-Path $script:CalChemRepo 'calchem\_core.pyd'
    if (Test-Path $dest) {
        $needCopy = ((Get-FileHash -LiteralPath $built).Hash -ne (Get-FileHash -LiteralPath $dest).Hash)
        if ($needCopy) {
            $bk = Join-Path $script:CalChemRepo ('build_target\runtime\pyd-backup\_core-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.pyd')
            New-Item -ItemType Directory -Force -Path (Split-Path $bk) | Out-Null
            Copy-Item -LiteralPath $dest -Destination $bk
            Copy-Item -LiteralPath $built -Destination $dest -Force
        }
    } else {
        Copy-Item -LiteralPath $built -Destination $dest -Force
    }
    & $env:CALCHEM_PYTHON -B -c "import calchem._core as c; print('installed _core:', sorted(n for n in dir(c) if not n.startswith('_')))"
} finally {
    Pop-Location
}
