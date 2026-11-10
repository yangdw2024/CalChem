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
$results = @()
Push-Location $repo
try {
    & 'D:\cargo\bin\cargo.exe' test --workspace --lib --offline --locked *> (Join-Path $audit 'evidence\rust-lib-tests.log')
    $results += [pscustomobject]@{Check='workspace library tests';ExitCode=$LASTEXITCODE}
    & 'D:\cargo\bin\cargo.exe' test -p calchem-basis -p calchem-scf --offline --locked *> (Join-Path $audit 'evidence\rust-basis-scf-tests.log')
    $results += [pscustomobject]@{Check='basis and SCF integration tests';ExitCode=$LASTEXITCODE}
    & 'D:\cargo\bin\cargo.exe' test -p calchem-int --test boys --test one_electron --test eri_symmetry --offline --locked *> (Join-Path $audit 'evidence\rust-integral-subsets.log')
    $results += [pscustomobject]@{Check='Boys, one-electron and ERI symmetry';ExitCode=$LASTEXITCODE}
    $results | ConvertTo-Json | Set-Content (Join-Path $audit 'evidence\rust-subsets-exit.json') -Encoding utf8
    $results | Format-Table -AutoSize
    foreach ($log in 'rust-lib-tests.log','rust-basis-scf-tests.log','rust-integral-subsets.log') {
        Write-Output "FILE: $log"
        Get-Content (Join-Path $audit "evidence\$log") -Tail 45
    }
} finally {
    Pop-Location
}
