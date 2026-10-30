# CalChem — Environment & Constraints

> 接手的 AI 必须知道这些硬约束，否则会做出在作者机器上跑不动的方案。

## 硬件（作者本地机）

| 部件 | 规格 | 影响 |
|---|---|---|
| CPU | AMD Ryzen 7 9700X（8 核 16 线程） | Gaussian 用 `nprocshared=8`；CalChem 也建议 8 线程以内 |
| 内存 | 15.1 GB | **严禁并行跑多个计算任务**；大基组大分子可能 OOM |
| 显卡 | 8 GB（型号未明） | CuPy/GPU 方案要谨慎，不能假设大显存 |
| OS | Windows 11 | 路径分隔符、原生 PowerShell；无 WSL 强约束 |
| 系统盘 | C 盘严格瘦身 | **禁止往 C 盘装任何东西**，所有 Python 环境/包必须装 D 盘 |

## 关键路径（硬编码在项目里，重构时要改成可配置）

| 路径 | 用途 |
|---|---|
| `E:\YDW\Trae_Gaussian\` | 本项目根目录 |
| `D:\G16W\` | Gaussian 16 Rev A.03 安装目录 |
| `D:\YDW\CalChem\gui\` | PyQt6 GUI 源码（独立目录） |
| `D:\YDW\Trae_Gromacs\miniconda3\python.exe` | 作者日常 conda base 环境（已装 numpy/scipy/pyyaml/pytest） |
| `D:\YDW\Trae_Gromacs\miniconda3\envs\` | 其他 conda 环境：molforge-webengine, rest-train, rest-v33, rest-win, ztop |

## 软件环境

- **Python**：实际开发在 3.13（.pyc 文件是 cpython-313），pyproject 声明 3.10+。
- **日常跑 calchem 的解释器**：`D:\YDW\Trae_Gromacs\miniconda3\python.exe`（base 环境）。
- **PyPI 依赖**：numpy, scipy, h5py, PyYAML, click, rich, pydantic, pyscf（未真用）, MDAnalysis, spglib, mendeleev, tables。
- **GPU 依赖**（可选，未装）：cupy-cuda12x。
- **GUI 依赖**：PyQt6, matplotlib, pyqtgraph, PyOpenGL。
- **打包工具**：PyInstaller。

## 作者身份与偏好（影响沟通和方案选择）

- 化学专业一年级研究生，方向：**有机太阳能电池（OSC）/ 端基卤化策略**。
- 即将开题答辩；目标期刊：npj CM / CM / JACS。
- 自述**计算化学基础薄弱、语言表达待提升**——接手的 AI 给代码时要**加详细中文注释**，不要只甩代码。
- 熟练工具：Gaussian, GROMACS, VMD, Tcl 脚本, Multiwfn, RESP/RESP2, sobEDA, IGMH。
- 不希望在 C 盘装任何软件。
- 删除文件必须走回收站，禁止 `Remove-Item` 永久删除（但本项目是重构，移动/删除都是可逆的）。
- 跑计算时 Gaussian 和 ML 训练**严格串行**，不能并行。

## 重构时的禁忌

1. **不要建议往 C 盘装环境/包**。
2. **不要建议并行跑多个 heavy 任务**——内存只有 15 GB。
3. **不要把 GPU 方案当成默认**——8 GB 显存，CuPy 要先做小规模验证。
4. **不要假设 Linux/macOS 环境**——这是 Windows 原生项目，脚本要用 PowerShell 友好。
5. **不要删 `calchem/tests/`**——那是包自带的测试，独立于顶层 `tests/`。
6. **不要破坏 1e-8 Eh 的数值基线**（见 `03_NUMERICAL_BASELINE.md`）。
7. **不要把 GUI 硬塞进本仓库**——GUI 在 `D:\YDW\CalChem\gui\`，是独立项目。

## 给接手 AI 的沟通风格建议

- 代码注释用中文，解释"为什么这样做"而不只是"做了什么"。
- 给方案时给 2-3 个选项 + 推荐理由，不要只给一个。
- 数值结果用表格呈现，对比 CalChem vs Gaussian。
- 涉及学术写作（开题、论文）时遵守 GB/T 7714-2025 引用格式。
- 画图要 Origin 风格（学术风、可编辑）。
