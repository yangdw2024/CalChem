$ErrorActionPreference = 'Stop'
$audit = Split-Path -Parent $PSScriptRoot
$env:TEMP = Join-Path $audit 'runtime\temp'
$env:TMP = $env:TEMP
$deps = Join-Path $audit 'runtime\rust-target\debug\deps'
$arguments = @('--edition=2021',(Join-Path $PSScriptRoot 'inspect_integrals.rs'),'-L',"dependency=$deps",'-o',(Join-Path $audit 'runtime\inspect_integrals.exe'))
foreach ($crate in 'calchem_basis','calchem_int','calchem_model','calchem_scf') {
    $library = Get-ChildItem -LiteralPath $deps -Filter "lib$crate-*.rlib" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $library) { throw "Run the Rust subset checks first; missing library: $crate" }
    $arguments += @('--extern',"$crate=$($library.FullName)")
}
& 'D:\cargo\bin\rustc.exe' @arguments *> (Join-Path $audit 'evidence\inspect-integrals-build.log')
if ($LASTEXITCODE -ne 0) { throw 'Audit probe compilation failed; see evidence log.' }
& (Join-Path $audit 'runtime\inspect_integrals.exe') | Set-Content (Join-Path $audit 'evidence\inspect-integrals.jsonl') -Encoding utf8
Get-Content (Join-Path $audit 'evidence\inspect-integrals.jsonl')
