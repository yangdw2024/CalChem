$ErrorActionPreference = 'Stop'
$audit = Split-Path -Parent $PSScriptRoot
$repo = (Resolve-Path (Join-Path $audit '..\..\..')).Path
$env:TEMP = Join-Path $audit 'runtime\temp'
$env:TMP = $env:TEMP
$env:CARGO_HOME = 'D:\cargo'
$env:RUSTUP_HOME = 'D:\rustup'
$env:RUSTUP_NO_UPDATE_CHECK = '1'
$env:CARGO_TARGET_DIR = Join-Path $audit 'runtime\rust-target'
$env:PYO3_PYTHON = 'D:\YDW\Trae_Gromacs\miniconda3\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Path $env:TEMP,$env:CARGO_TARGET_DIR -Force | Out-Null
Push-Location $repo
try {
    & 'D:\cargo\bin\cargo.exe' --version | Set-Content (Join-Path $audit 'evidence\rust-toolchain.txt')
    & 'D:\cargo\bin\rustc.exe' --version | Add-Content (Join-Path $audit 'evidence\rust-toolchain.txt')
    & 'D:\cargo\bin\cargo.exe' test --workspace --offline --locked *> (Join-Path $audit 'evidence\rust-tests.log')
    $code = $LASTEXITCODE
    "cargo test --workspace --offline --locked exit: $code" | Set-Content (Join-Path $audit 'evidence\rust-exit.txt')
    Get-Content (Join-Path $audit 'evidence\rust-tests.log') -Tail 100
    exit $code
} finally {
    Pop-Location
}
