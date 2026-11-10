# CalChem development environment (dot-source this file):  . .\tools\dev\dev_env.ps1
# Redirects every cache / temp / output path to E: (inside build_target\, already git-ignored)
# and pins the D: toolchains. Nothing is written to C:.
$script:CalChemRepo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$rt = Join-Path $script:CalChemRepo 'build_target\runtime'

$env:CALCHEM_PYTHON = 'D:\YDW\Trae_Gromacs\miniconda3\python.exe'
$env:CALCHEM_CARGO  = 'D:\cargo\bin\cargo.exe'
$env:CALCHEM_RUFF   = 'D:\YDW\Trae_Gromacs\miniconda3\Scripts\ruff.exe'

$env:CARGO_HOME = 'D:\cargo'
$env:RUSTUP_HOME = 'D:\rustup'
$env:RUSTUP_NO_UPDATE_CHECK = '1'
$env:CARGO_TARGET_DIR = Join-Path $script:CalChemRepo 'build_target'
$env:PYO3_PYTHON = $env:CALCHEM_PYTHON

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$env:TEMP = Join-Path $rt 'temp'
$env:TMP = $env:TEMP
$env:CALCHEM_DATA = Join-Path $rt 'data'
$env:CALCHEM_RUN_DIR = Join-Path $rt 'external'
$env:CALCHEM_CONFIG = Join-Path $rt 'config.json'
$env:NUMBA_CACHE_DIR = Join-Path $rt 'numba'
$env:MPLCONFIGDIR = Join-Path $rt 'matplotlib'
$env:XDG_CACHE_HOME = Join-Path $rt 'cache'
$env:CUPY_CACHE_DIR = Join-Path $rt 'cupy'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

New-Item -ItemType Directory -Force -Path $env:TEMP, $env:CALCHEM_DATA, $env:CALCHEM_RUN_DIR, `
    $env:NUMBA_CACHE_DIR, $env:MPLCONFIGDIR, $env:XDG_CACHE_HOME, $env:CUPY_CACHE_DIR | Out-Null
