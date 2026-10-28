# tests/ 目录布局（2026-10-28 重整后）

本目录原先只有 3 个正式测试（test_io / test_molecule / test_scf），项目根目录散落着 ~107 个 `test_*.py / debug_*.py / verify_*.py / _*.py` 脚本。本次按用途归到 8 个子目录下，**文件名一律保持原名不重命名**，保留历史可追溯性。

## 目录结构

```
tests/
├── conftest.py          # pytest 夹具（water_mol / methane_mol / GPU 自动 skip）
├── test_io.py           # 原有的正式测试（不动）
├── test_molecule.py    # 原有的正式测试（不动）
├── test_scf.py         # 原有的正式测试（不动）
│
├── unit/                # 4 个  —— 包/基组最小编译级测试
├── integrals/           # 14 个 —— 一/双电子积分、ERI 对称性、JK 矩阵
├── scf/                # 26 个 —— RHF/UHF SCF、DIIS、H2/H2O/原子收敛
├── dft/                # 21 个 —— Lebedev 网格、LDA/PBE/TPSS/B3LYP、Vxc
├── posthf/             #  3 个 —— MP/CCSD(T)/CASSCF + 全量回归主表
├── gui/                #  2 个 —— PyQt6 GUI 冒烟测试（依赖 D:\YDW\CalChem\gui）
├── debug/               # 15 个 —— 历史调试脚本（debug_*.py / verify_*.py / calc_eri_ref.py）
└── scratch/             # 21 个 —— 一次性分析稿（_*.py）+ numpy 原型（test_eig/einsum/...）
```

## 各目录职责

### `unit/`
- `test_imports.py` — calchem 包能否 import
- `test_atoms.py` — 原子基本属性
- `test_basis_norm.py` — 基函数归一化 ∫χ²dV=1
- `test_gto_detail.py` — GTO 角动量/球谐细节

### `integrals/`
ERI 一/双电子积分正确性。包括对 H/He/Be/Ne/O 的逐角动量验证、ERI 对称性、JK 矩阵核验。**这是整个项目数值可信度的根基**。

### `scf/`
SCF 主循环与收敛性。包含 H2/He/H2O 在 STO-3G 下的参考能量比对、DIIS 收敛对比、初猜、Fock 本征值、密度初值、能量公式。

**注意**：`test_h2o_direct.py` 和 `test_quick.py` 中的 `sys.path.insert` 已改为向上两级指向项目根（因为它们现在深了两级）。

### `dft/`
DFT 全栈：Lebedev 网格、径向积分、LDA(VWN5)、PBE/GGA、TPSS、B3LYP、D3 色散、Vxc 符号核验。
历史 debug 痕迹最集中的目录——`test_vxc_sign/direct/detailed`、`test_pbe_gamma_diagnose`、`test_lda_no_diis` 都是当年定位 bug 的现场。

### `posthf/`
- `test_regression.py` — **主回归表**，硬编码 H2/He/Ne/H2O_RHF/H2O_MP2/O2_UHF 参考能量，阈值 <1e-8 Eh
- `test_audit.py` — 全模块横向审计（CASSCF/MP4/CCSDT 跨方法）
- `test_ccsd_t_convergence.py` — (T) 修正项收敛

### `gui/`
依赖外部 `D:\YDW\CalChem\gui`，CI 环境下应 skip。

### `debug/`
**不是 pytest 测试**，是当年调试 ERI/VRR/球谐/SCF 时留下的"现场录像"。保留是为了：
- 复现数值问题时知道当年怎么定位的
- `debug_spherical.py`(55KB) 等长文件本身就是经验档案

不要把这一目录纳入 CI。

### `scratch/`
两类：
1. 一次性分析稿（文件名以 `_` 开头，如 `_decompose_mp3.py`、`_validate_mp3.py`）—— 当年推导 MP2 公式时的草稿
2. numpy 原型（`test_eig/eigh/einsum/matmul/dot/matrix/simple/flush/pieces/steps/numpy_config/ortho_detail`）—— 早期验证 LAPACK 路径的实验

同样不纳入 CI。`test_water.xyz` 是测试数据。

## 运行方式

```powershell
# 用 conda base（已装齐依赖）
D:\YDW\Trae_Gromacs\miniconda3\python.exe -m pytest tests/ -v -m "not slow and not gpu"

# 只跑正式测试（排除 debug/scratch/gui）
D:\YDW\Trae_Gromacs\miniconda3\python.exe -m pytest tests/unit tests/integrals tests/scf tests/dft tests/posthf -v

# 单个脚本直接跑（很多脚本写成 __main__ 形式，不是 pytest 用例）
D:\YDW\Trae_Gromacs\miniconda3\python.exe tests\posthf\test_regression.py
```

## 已知问题（留给未来清理）

1. `debug/` 和 `scratch/` 里的脚本**没有 `test_` 前缀或函数**就不会被 pytest 收集，但有些文件名仍以 `test_` 开头（如 `scratch/test_eig.py`），未来若跑全量 pytest 会把它们也收进去——建议下次批量改名或在 `pyproject.toml` 的 `pytest` 配置里加 `norecursedirs`。
2. `conftest.py` 里的 `GAUSSIAN_TEST_DIR = Path(r"D:\G16W\tests\gjf")` 是硬编码路径，换机器即失效。
3. 项目根不是 git 仓库，CI（`.github/workflows/ci.yml`）不会真跑。

## 本次重整操作记录（2026-10-28）

- 从项目根目录搬入 **107 个 .py 文件 + 1 个 .xyz 文件**
- 根目录保留：`build_calchem_exe.py`、`trae_framework.py`（这两个是构建/框架，不是测试）
- 修正 3 个脚本的 `sys.path.insert`：
  - `tests/scf/test_h2o_direct.py`
  - `tests/scf/test_quick.py`
  - `tests/posthf/test_regression.py`
- 验证：110 个文件 `py_compile` 全过；`import calchem` 正常；`tests/scf/test_quick.py` 实跑 H2/SCF/STO-3G = -1.116714325063 Eh（与参考 -1.11671433 误差 <1e-8）
