# CalChem 科研精度与工程验证一键复现脚本
# 所有构建、缓存与临时文件均严格限定在 E 盘，不写入 C 盘

$ErrorActionPreference = "Stop"
Set-Location "E:\YDW\Trae_Gaussian"

. .\tools\dev\dev_env.ps1
$env:PATH = "D:\cargo\bin;$env:PATH"

$CertDir = "E:\YDW\Trae_Gaussian\docs\audits\2026-11-16_final_certification"
New-Item -ItemType Directory -Force -Path $CertDir | Out-Null

function Invoke-CheckedStep {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Host $Label
    $global:LASTEXITCODE = 0
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code $LASTEXITCODE : $Label"
    }
}

Invoke-CheckedStep "[1/5] Rebuild calchem._core.pyd and verify SHA-256 parity..." {
    powershell -NoProfile -File .\tools\dev\build_ext.ps1
}

Invoke-CheckedStep "[2/5] Run full Rust workspace tests (scripts/cargo_test.ps1)..." {
    powershell -NoProfile -File .\scripts\cargo_test.ps1
}

Invoke-CheckedStep "[3/5] Run CUDA kernel & FFI contract tests (calchem-cuda)..." {
    & $env:CALCHEM_CARGO test -p calchem-cuda --offline --locked
}

Invoke-CheckedStep "[4/5] Run Gaussian 16 external reference & CPU/GPU RI-J benchmarks..." {
    & $env:CALCHEM_PYTHON -u "$CertDir\run_external_and_gpu_benchmarks.py"
}

Invoke-CheckedStep "[5/5] Run full Python unit test suite (tests/unit)..." {
    powershell -NoProfile -File .\scripts\run_pytest.ps1 tests/unit
}

Write-Host "All 5 verification stages exited with code 0."
