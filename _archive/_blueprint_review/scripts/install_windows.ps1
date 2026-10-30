<#
.SYNOPSIS
    CalChem Windows 自动安装脚本
.DESCRIPTION
    在 Windows 上自动安装 CalChem 及其依赖
    默认安装到 D 盘
#>

param(
    [string]$InstallPath = "D:\calchem-venv",
    [string]$RepoPath = "D:\YDW\Trae_Gaussian",
    [switch]$WithGPU,
    [switch]$WithDev
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CalChem Installer for Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
Write-Host "[1/6] 检查 Python 环境..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  找到 Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  错误: 未找到 Python，请先安装 Python 3.10+" -ForegroundColor Red
    exit 1
}

# 创建虚拟环境
Write-Host "[2/6] 创建虚拟环境到 $InstallPath..." -ForegroundColor Yellow
if (-not (Test-Path $InstallPath)) {
    python -m venv $InstallPath
    Write-Host "  虚拟环境创建完成" -ForegroundColor Green
} else {
    Write-Host "  虚拟环境已存在，跳过创建" -ForegroundColor Green
}

# 激活虚拟环境
$activateScript = Join-Path $InstallPath "Scripts\Activate.ps1"
. $activateScript
Write-Host "  已激活虚拟环境" -ForegroundColor Green

# 升级 pip
Write-Host "[3/6] 升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "  pip 升级完成" -ForegroundColor Green

# 安装 CalChem
Write-Host "[4/6] 安装 CalChem..." -ForegroundColor Yellow
$installArgs = @("-e", $RepoPath)
if ($WithGPU) {
    $installArgs += ".[gpu]"
} else {
    $installArgs += "."
}
if ($WithDev) {
    if ($installArgs -match "\.") {
        $installArgs = $installArgs -replace "\.", ".[dev]"
    }
}

pip install @installArgs
Write-Host "  CalChem 安装完成" -ForegroundColor Green

# 验证安装
Write-Host "[5/6] 验证安装..." -ForegroundColor Yellow
try {
    $version = python -c "import calchem; print(calchem.__version__)" 2>&1
    Write-Host "  CalChem 版本: $version" -ForegroundColor Green
} catch {
    Write-Host "  警告: 版本验证失败，但安装可能成功" -ForegroundColor Yellow
}

Write-Host "[6/6] 配置环境..." -ForegroundColor Yellow

# 检查 CUDA
if ($WithGPU) {
    try {
        $cudaVersion = nvcc --version 2>&1
        Write-Host "  找到 CUDA: $cudaVersion" -ForegroundColor Green
    } catch {
        Write-Host "  警告: 未找到 CUDA 工具包，GPU 功能可能受限" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "使用方法:" -ForegroundColor White
Write-Host "  激活环境:  $InstallPath\Scripts\Activate.ps1"
Write-Host "  运行计算:  calchem run molecule.xyz --method b3lyp --basis def2-svp"
Write-Host "  查看帮助:  calchem --help"
Write-Host ""
