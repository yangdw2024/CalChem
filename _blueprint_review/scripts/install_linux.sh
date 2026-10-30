#!/bin/bash
# CalChem Linux 自动安装脚本
# 默认安装到 ~/calchem-venv

set -e

INSTALL_PATH="${1:-$HOME/calchem-venv}"
REPO_PATH="${2:-$(pwd)}"
WITH_GPU="${3:-false}"
WITH_DEV="${4:-false}"

echo "========================================"
echo "  CalChem Installer for Linux"
echo "========================================"
echo ""

# 检查 Python
echo "[1/6] 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "  错误: 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi
PYTHON_VERSION=$($PYTHON --version)
echo "  找到 Python: $PYTHON_VERSION"

# 创建虚拟环境
echo "[2/6] 创建虚拟环境到 $INSTALL_PATH..."
if [ ! -d "$INSTALL_PATH" ]; then
    $PYTHON -m venv "$INSTALL_PATH"
    echo "  虚拟环境创建完成"
else
    echo "  虚拟环境已存在，跳过创建"
fi

# 激活虚拟环境
source "$INSTALL_PATH/bin/activate"
echo "  已激活虚拟环境"

# 升级 pip
echo "[3/6] 升级 pip..."
pip install --upgrade pip
echo "  pip 升级完成"

# 安装 CalChem
echo "[4/6] 安装 CalChem..."
INSTALL_TARGET="."
if [ "$WITH_GPU" = "true" ] && [ "$WITH_DEV" = "true" ]; then
    INSTALL_TARGET=".[gpu,dev]"
elif [ "$WITH_GPU" = "true" ]; then
    INSTALL_TARGET=".[gpu]"
elif [ "$WITH_DEV" = "true" ]; then
    INSTALL_TARGET=".[dev]"
fi

pip install -e "$REPO_PATH/$INSTALL_TARGET"
echo "  CalChem 安装完成"

# 验证安装
echo "[5/6] 验证安装..."
if python -c "import calchem; print(calchem.__version__)" 2>/dev/null; then
    echo "  验证通过"
else
    echo "  警告: 版本验证失败，但安装可能成功"
fi

echo "[6/6] 配置环境..."
if [ "$WITH_GPU" = "true" ]; then
    if command -v nvcc &> /dev/null; then
        echo "  找到 CUDA: $(nvcc --version | grep release)"
    else
        echo "  警告: 未找到 CUDA 工具包，GPU 功能可能受限"
    fi
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "使用方法:"
echo "  激活环境:  source $INSTALL_PATH/bin/activate"
echo "  运行计算:  calchem run molecule.xyz --method b3lyp --basis def2-svp"
echo "  查看帮助:  calchem --help"
echo ""
