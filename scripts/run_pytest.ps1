param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

. "$PSScriptRoot\..\docs\audits\2026-11-10_round2\probes\env.ps1"
$condaRoot = 'D:\YDW\Trae_Gromacs\miniconda3'
$env:PATH = "$condaRoot;$condaRoot\Scripts;$condaRoot\Library\bin;$env:PATH"
$env:PYTHONPATH = "E:\YDW\Trae_Gaussian"
$pytestCache = Join-Path $auditRuntime 'pytest-cache'

if ($PytestArgs.Count -eq 0) {
    & $env:PYO3_PYTHON -m pytest -o "cache_dir=$pytestCache" tests/unit -v
} else {
    & $env:PYO3_PYTHON -m pytest -o "cache_dir=$pytestCache" @PytestArgs
}
exit $LASTEXITCODE
