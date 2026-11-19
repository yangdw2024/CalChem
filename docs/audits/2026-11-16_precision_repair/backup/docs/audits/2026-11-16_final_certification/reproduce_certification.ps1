# CalChem 2026-11-16 最终科研可信度验收一键复现脚本
# 所有构建、缓存与临时文件均严格限定在 E 盘，不写入 C 盘

$ErrorActionPreference = "Stop"
Set-Location "E:\YDW\Trae_Gaussian"

. .\tools\dev\dev_env.ps1
$env:PATH = "D:\cargo\bin;$env:PATH"

$CertDir = "E:\YDW\Trae_Gaussian\docs\audits\2026-11-16_final_certification"
New-Item -ItemType Directory -Force -Path $CertDir | Out-Null

Write-Host "[1/5] Rebuild calchem._core.pyd and verify SHA-256 parity..."
powershell -NoProfile -File .\tools\dev\build_ext.ps1

Write-Host "[2/5] Run full Rust workspace tests (scripts/cargo_test.ps1)..."
powershell -NoProfile -File .\scripts\cargo_test.ps1

Write-Host "[3/5] Run CUDA kernel & FFI contract tests (calchem-cuda)..."
& $env:CALCHEM_CARGO test -p calchem-cuda --offline --locked

Write-Host "[4/5] Run Gaussian 16 external reference & CPU/GPU RI-J benchmarks..."
& $env:CALCHEM_PYTHON -u "$CertDir\run_external_and_gpu_benchmarks.py"

Write-Host "[5/5] Run full Python unit test suite (tests/unit)..."
powershell -NoProfile -File .\scripts\run_pytest.ps1 tests/unit

Write-Host "Certification reproduction completed successfully."
