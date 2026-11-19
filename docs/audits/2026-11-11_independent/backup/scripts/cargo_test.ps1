param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs
)
$ErrorActionPreference = "Stop"
$origTemp = $env:TEMP
$origTmp = $env:TMP
. (Join-Path $PSScriptRoot "..\docs\audits\2026-11-10_round2\probes\env.ps1")
$env:TEMP = $origTemp
$env:TMP = $origTmp

$env:PYO3_PYTHON = 'D:\YDW\Trae_Gromacs\miniconda3\python.exe'
$env:CARGO_PROFILE_DEV_OPT_LEVEL = '2'
$env:CARGO_PROFILE_TEST_OPT_LEVEL = '2'

cargo @CargoArgs
