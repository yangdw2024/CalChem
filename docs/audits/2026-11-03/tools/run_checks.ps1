param(
    [string]$CheckScript = 'run_checks.py',
    [string]$LogName = 'python-checks.log'
)
$ErrorActionPreference = 'Stop'
$audit = Split-Path -Parent $PSScriptRoot
$repo = (Resolve-Path (Join-Path $audit '..\..\..')).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = Join-Path $audit 'runtime\temp'
$env:TMP = $env:TEMP
$env:CALCHEM_DATA = Join-Path $audit 'runtime\data'
$env:CALCHEM_RUN_DIR = Join-Path $audit 'runtime\external'
$env:CALCHEM_CONFIG = Join-Path $audit 'runtime\config.json'
$env:NUMBA_CACHE_DIR = Join-Path $audit 'runtime\numba'
$env:MPLCONFIGDIR = Join-Path $audit 'runtime\matplotlib'
$env:XDG_CACHE_HOME = Join-Path $audit 'runtime\cache'
$env:CUPY_CACHE_DIR = Join-Path $audit 'runtime\cupy'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
New-Item -ItemType Directory -Path $env:TEMP,$env:CALCHEM_RUN_DIR,$env:NUMBA_CACHE_DIR,$env:MPLCONFIGDIR,$env:XDG_CACHE_HOME,$env:CUPY_CACHE_DIR -Force | Out-Null
Push-Location $repo
try {
    & 'D:\YDW\Trae_Gromacs\miniconda3\python.exe' -B (Join-Path $PSScriptRoot $CheckScript) *> (Join-Path $audit "evidence\$LogName")
    $code = $LASTEXITCODE
    Get-Content (Join-Path $audit "evidence\$LogName") -Tail 75
    Write-Output "Python audit runner exit: $code (pytest status is recorded separately in checks.json)"
} finally {
    Pop-Location
}
