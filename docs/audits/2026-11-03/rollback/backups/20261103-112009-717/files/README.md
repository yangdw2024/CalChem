# CalChem

GPU-accelerated quantum chemistry software for organic and OPV materials.

CalChem 是一款面向有机光电材料（OPV）的量子化学计算软件，原生支持 GPU 加速，与 GROMACS 工作流深度集成，提供比 Gaussian 更现代、更高效、更易用的体验。

## 核心特性

- **GPU 原生加速**：Fock 矩阵构建、电子积分、DFT 网格全链路 GPU 加速
- **OPV 场景优化**：内置共轭体系默认参数（ωB97X-D/def2-TZVP + SMD 溶剂）
- **GROMACS 一键对接**：RESP/CM5 电荷直接生成 GROMACS 拓扑
- **现代接口**：Python API + YAML 声明式输入 + 结构化 JSON 输出
- **Gaussian 兼容**：直接读取 `.gjf` 输入和 `.fchk` 检查点
- **双平台支持**：Windows + Linux 原生运行

## 安装

### Windows (PowerShell)

```powershell
# 自动安装脚本（安装到 D 盘）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-WebRequest -Uri https://raw.githubusercontent.com/CalChem/CalChem/main/scripts/install_windows.ps1 -OutFile install_calchem.ps1
.\install_calchem.ps1
```

### 手动安装

```bash
# 克隆仓库
git clone https://github.com/CalChem/CalChem.git
cd CalChem

# 创建虚拟环境（推荐 D 盘）
python -m venv D:\calchem-venv
D:\calchem-venv\Scripts\activate

# 安装
pip install -e ".[dev,gpu]"
```

## 快速开始

### 命令行

```bash
# 单点能计算
calchem run molecule.xyz --method b3lyp --basis def2-svp

# 几何优化
calchem opt molecule.xyz --method wb97xd --basis def2-tzvp --maxiter 100

# 显示分子信息
calchem info molecule.gjf
```

### Python API

```python
from calchem import Molecule
from calchem.api import single_point, optimize

# 加载分子
mol = Molecule.from_xyz("""
3
water
O  0.0  0.0  0.0
H  0.0  0.0  0.96
H  0.0  0.96  0.0
""")

# 单点能计算
result = single_point(mol, method="B3LYP", basis="def2-SVP")
print(f"Energy: {result.energy:.6f} Hartree")

# 几何优化
opt_result = optimize(mol, method="B3LYP", basis="def2-SVP")
print(f"Optimized energy: {opt_result.energy:.6f} Hartree")
opt_result.molecule.save_xyz("water_opt.xyz")
```

### YAML 输入

```yaml
molecule:
  file: water.xyz
  charge: 0
  multiplicity: 1

calculation:
  type: optimize
  method: B3LYP
  basis: def2-TZVP
  functional: B3LYP
  max_scf_cycles: 100
  max_opt_cycles: 50
  convergence: tight

output:
  format: json
  file: result.json
  cube_files: [homo, lumo]
```

## 项目结构

```
calchem/
├── mol/          # 分子表示与几何
├── io/           # 输入输出 (xyz/gjf/gro/mol2/hdf5)
├── basis/        # 基组库
├── scf/          # SCF 主循环 (HF/DFT)
├── gpu/          # GPU 加速核心
├── grad/         # 梯度与频率
├── opt/          # 几何优化 (BFGS/Berny)
├── props/        # 性质分析 (电荷/多极矩/NMR)
├── excited/      # 激发态 (TDDFT/UV-Vis)
├── posthf/       # 后 HF (MP2/CCSD(T))
├── solvent/      # 溶剂效应 (PCM/SMD)
├── md/           # MD 集成 (GROMACS)
├── cli/          # 命令行接口
├── api/          # Python 高层 API
└── utils/        # 工具函数
```

## 路线图

- **M1** ✅ 分子输入解析器与结构化输出
- **M2** 🔄 GPU 加速的 SCF 核心 (RHF/UHF/DFT)
- **M3** 几何优化器 (BFGS/Berny)
- **M4** RESP/CM5 电荷拟合，GROMACS 对接
- **M5** TDDFT 激发态与 UV-Vis 光谱

## 许可证

Apache-2.0
