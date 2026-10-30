# CalChem — Master Context for AI Hand-off

> 这份文档是给**下一个接手项目的 AI 模型**看的入口。读完它 + `02_ARCHITECTURE.md` + `03_NUMERICAL_BASELINE.md`，你就应该能开始干活。
> 项目所有者是化学研究生（OSC/有机光伏方向），这是他的**自研 Gaussian 替代引擎**，不是"用 Gaussian 跑算"。

---

## 1. 项目一句话

**CalChem** 是用 Python 从零写的量子化学计算引擎（不是封装 PySCF），目标是面向有机光电材料（OPV），最终替代商用 Gaussian 16。当前状态：**小分子（H2/He/Ne/H2O/CH3）在 STO-3G 下数值已对齐 Gaussian 到 1e-8 Eh**，但 GPU、大基组、大分子、CI 都是空壳或未验证。

## 2. 为什么存在（动机）

- 作者日常用 Gaussian 16（装在 `D:\G16W`），但 Gaussian 闭源、OPV 工作流（RESP→GROMACS）需要多软件拼接。
- 借"重写一遍 HF/DFT/Post-HF"系统夯实计算化学基础（作者自述基础薄弱）。
- 想做一个**带现代 GUI、可 GPU 加速、与 GROMACS 一键对接**的自研工具。

## 3. 当前能力（哪些是真的，哪些是吹的）

| 能力 | 状态 | 证据 |
|---|---|---|
| Molecule / xyz/gjf 解析 | ✅ 真 | `calchem/mol/`, `calchem/io/input.py` |
| GTO 基组库（sto-3g/3-21g/6-31g/def2/cc-pVXZ） | ✅ 真 | `calchem/basis/basis_data.py` (67 KB) |
| 一电子积分（S/T/V） | ✅ 真 | `calchem/integrals/one_electron.py` (30 KB) |
| 双电子 ERI（Obara-Saika VRR） | ✅ 真 | `calchem/integrals/two_electron.py` (23 KB) |
| RI 密度拟合 + Schwarz screening | ✅ 真 | `calchem/integrals/ri.py`, `screening.py` |
| RHF/UHF SCF + DIIS | ✅ 真 | `calchem/scf/hf.py` (22 KB), `scf/diis.py` |
| DFT（LDA/VWN5/PBE/TPSS/B3LYP/PBE0/BLYP） | ✅ 真 | `calchem/dft/functionals.py` (35 KB), `grid.py` |
| D3 色散 | ✅ 真 | `calchem/dft/d3.py` |
| MP2 / MP4 / CCSD / CCSD(T) / CASSCF | ✅ 真 | `calchem/post_hf/` |
| 解析梯度 / BFGS 优化 / 频率 | 🟡 代码在，未充分验证 | `calchem/grad/`, `opt/bfgs.py`, `vib/frequency.py` |
| CIS / TDDFT | 🟡 代码在，未充分验证 | `calchem/excited/` |
| RESP 电荷 / GROMACS 拓扑写出 | 🟡 代码在 | `calchem/md/` |
| PCM / SMD 溶剂化 | 🟡 代码在，未验证 | `calchem/solvent/` |
| ONIOM | 🟡 代码在 | `calchem/oniom/` |
| Python API / CLI | ✅ 真 | `calchem/api/high_level.py`, `calchem/cli/main.py` |
| PyQt6 GUI | ✅ 真（在**另一个目录**） | `D:\YDW\CalChem\gui\`，不在本压缩包内 |
| PyInstaller 打包 | ✅ 真 | `build_calchem_exe.py` |
| **GPU 加速（CuPy）** | ❌ **空壳** | `calchem/gpu/fock.py` 只有 923 字节 |
| **CI（GitHub Actions）** | ❌ **空转** | `.github/workflows/ci.yml` 在，但项目不是 git 仓库 |
| 600+ Gaussian 用例全量回归 | ❌ 仅有声明 | `benchmarks/README.md` 提了，但无结果文件 |

## 4. 已验证的数值正确性（这是安全网，重构时不能破坏）

详见 `03_NUMERICAL_BASELINE.md`。**核心阈值：|E_calchem - E_ref| < 1e-8 Eh**。

目前已知对齐的能量（STO-3G 基组）：

- H2/RHF = -1.11671433
- He/RHF = -2.80778396
- Ne/RHF = -126.60452917
- H2O/RHF = -74.69531918（README_CalChem 写 -74.9629，两版有出入，**以 test_regression.py 内 REFERENCES 为准**）
- H2O/MP2 = -74.71482338
- H2O/PBE = -75.5322
- H2O/B3LYP = -75.5566
- H2O/CCSD(T) = -74.9988
- H2O/MP4 = -75.0180
- CH3/ROHF = -40.5539
- H2/CASSCF(2,2) = -1.1373

## 5. 项目历史（时间线）

- **2026-07-05 ~ 07-21**：17 天密集开发，单人 + Trae IDE 结对。
- 2026-07-05：脚手架、GTO、H2/H2O RHF 跑通。
- 2026-07-06：DIIS、H2O 收敛、梯度/频率/溶剂骨架。
- 2026-07-09：高层 API、ONIOM、IRC、TS 搜索。
- 2026-07-10：DFT 大爆发（网格/LDA/PBE/TPSS/B3LYP/Vxc/RI）、MP4、CASSCF、GUI 升级 ~800 行。
- 2026-07-11：CCSD、direct SCF、全量回归。
- 2026-07-21：`trae_framework.py`（作者自己的 AI 工作流框架）沉淀。
- **2026-07-21 之后代码基本停滞**（2026-09-07 只有 __pycache__ 时间戳变化）。

## 6. 重构的靶子（详见 `04_TECH_DEBT.md`）

按优先级：

1. **项目没 git init**——CI、pre-commit 全是模板，永远不跑。**重构第一步应该是 git init + 首个 commit + 把 CI 真跑起来**。
2. **GPU 是空壳**——README 宣称"GPU 原生加速"，但 `calchem/gpu/fock.py` 只有 923 字节。要么真做，要么改 README 不要吹。
3. **硬编码路径**——`tests/conftest.py` 写死 `D:\G16W\tests\gjf`；`build_calchem_exe.py` 写死 `D:\YDW\CalChem\gui`；很多 test 脚本写死 `d:\YDW\Trae_Gaussian`。
4. **测试组织混乱**——根目录曾散落 107 个 test/debug 脚本（已在 2026-10-28 整理到 `tests/{unit,integrals,scf,dft,posthf,gui,debug,scratch}/`），但 `scratch/` 里还有一堆 `test_*` 文件会被 pytest 误收集。
5. **依赖未锁定**——`pyproject.toml` 里有 `pyscf>=2.5`，但代码实际是从零写的，pyscf 几乎没被用。要么真用、要么删掉。
6. **类型标注缺失**——几乎所有函数没有 type hints；mypy 在 pre-commit 里配了但没跑。
7. **参考能量表两处不一致**——`README_CalChem.md` 和 `tests/posthf/test_regression.py` 里的 H2O/RHF 值对不上，需要统一。
8. **OPV 场景未真跑**——所有验证都是 H2/He/H2O/CH3 这种小分子，作者的真实目标（PM6、ClFFCl 这类共轭大分子）从未在 CalChem 里跑过。

## 7. 不能碰的红线

- **不要破坏已对齐的数值**——每改一个模块，跑 `tests/posthf/test_regression.py`，所有 |ΔE| 必须 < 1e-8。
- **不要把 GPU 模块直接当完成**——它是个占位符，要诚实标注。
- **不要并行跑多个计算任务**——作者硬件受限（见 `05_ENV_AND_CONSTRAINTS.md`），且 Gaussian 16 也在同一台机器上。
- **不要删 `calchem/tests/` 目录**——那是包自带的测试，和顶层 `tests/` 是两套。

## 8. 给接手 AI 的第一周建议

1. 先 `git init`，把当前状态做第一个 commit（打 tag `pre-refactor`）。
2. 跑一次 `tests/posthf/test_regression.py`，确认基线绿。
3. 修 `pyproject.toml`：把 `pytest` 配置里加 `norecursedirs = ["debug" "scratch"]`，把不用的 `pyscf` 依赖删了或注释。
4. 把硬编码路径全部改成相对路径/环境变量。
5. 决定 GPU 模块的去留：要么真写 CuPy 版，要么从 README 里删掉"GPU 原生加速"的宣传。
6. 再开始动架构。

## 9. 关键文件索引

| 想看什么 | 读哪个文件 |
|---|---|
| 项目定位、动机、当前状态 | 你现在读的这份 |
| 专业关键词（给 prompt 用） | `01_KEYWORDS.md` |
| 模块架构、每个文件干什么 | `02_ARCHITECTURE.md` |
| 数值安全网（参考能量表） | `03_NUMERICAL_BASELINE.md` |
| 技术债、重构靶子 | `04_TECH_DEBT.md` |
| 硬件、环境、用户偏好 | `05_ENV_AND_CONSTRAINTS.md` |
| 完整工作流历史 | `PROJECT_WORKFLOW_SUMMARY.md`（项目根） |
| 正式 README | `README.md`, `README_CalChem.md` |
| 依赖清单 | `pyproject.toml` |
| CLI 入口 | `calchem/cli/main.py` |
| 高层 API | `calchem/api/high_level.py` |
| 主回归测试 | `tests/posthf/test_regression.py` |
