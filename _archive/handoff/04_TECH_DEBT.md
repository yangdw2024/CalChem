# CalChem — Tech Debt & Refactor Targets

> 按优先级排序。每一项都给了"现状 / 为什么是问题 / 建议做法"。

## P0 — 重构前必须先做（否则后续工作无保护）

### 1. 项目没有 git 历史
- **现状**：`E:\YDW\Trae_Gaussian\` 下没有 `.git` 目录，但有 `.github/workflows/ci.yml`、`.pre-commit-config.yaml`。
- **问题**：所有 CI/lint/pre-commit 配置都是死的，没有版本回滚，没有安全网。
- **建议**：
  ```powershell
  cd E:\YDW\Trae_Gaussian
  git init
  git add .
  git commit -m "pre-refactor snapshot: working small-molecule baseline"
  git tag pre-refactor
  # 之后再设 remote
  ```

### 2. H2O/RHF 参考值两处不一致
- **现状**：`test_regression.py` 写 -74.69531918，`README_CalChem.md` 写 -74.9629。
- **问题**：以后改代码时不知道哪个是真值，回归可能误报。
- **建议**：用 `D:\G16W` 现跑一遍 H2O/RHF/STO-3G，统一成一个值。

### 3. pytest 会误收集 scratch/ 和 debug/
- **现状**：`tests/scratch/` 里有 `test_eig.py`、`test_einsum.py` 这种文件，pytest 默认会按 `test_*.py` 规则收集。
- **建议**：在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 加：
  ```toml
  norecursedirs = ["debug" "scratch" "__pycache__" ".pytest_cache"]
  ```

## P1 — 架构层面

### 4. GPU 模块是空壳
- **现状**：`calchem/gpu/fock.py` 只有 923 字节，README 却宣称"GPU 原生加速、Fock 矩阵构建、电子积分、DFT 网格全链路 GPU 加速"。
- **问题**：宣传与实现严重不符。作者硬件是 8 GB 显卡，CuPy 也没真装。
- **建议（二选一）**：
  - (a) 诚实路线：把 README 改成"GPU-ready architecture, CPU-only implementation"，删掉 gpu/ 空壳或留 TODO；
  - (b) 真做路线：先在 Fock 组装上做一个 CuPy 版原型，用 H2O/STO-3G 验证数值一致，再逐步推广到 ERI 和 DFT 网格。

### 5. 大量硬编码 Windows 路径
- **现状**：
  - `tests/conftest.py`: `GAUSSIAN_TEST_DIR = Path(r"D:\G16W\tests\gjf")`
  - `build_calchem_exe.py`: `CALCHEM_PATH = r"D:\YDW\Trae_Gaussian"`, `GUI_DIR = r"D:\YDW\CalChem\gui"`
  - 几十个 test 脚本里 `sys.path.insert(0, r"d:\YDW\Trae_Gaussian")`
- **问题**：换机器、换盘符就崩。
- **建议**：
  - 路径统一走 `calchem.utils.config`（已有 ConfigManager）或环境变量 `CALCHEM_ROOT`、`G16_ROOT`；
  - `sys.path.insert` 这种 hack 在 `pip install -e .` 之后完全不需要，删掉即可；
  - 打包脚本用 CLI 参数或 `--add-data` 传入，不要写死。

### 6. 依赖声明与实际不符
- **现状**：`pyproject.toml` 依赖 `pyscf>=2.5`、`MDAnalysis`、`spglib`、`tables`，但代码实际是从零写的，pyscf 几乎没被用。
- **问题**：新环境 `pip install -e .` 会装一堆没用的重依赖。
- **建议**：
  - 用 `grep -r "import pyscf" calchem/` 确认是否真用；
  - 真用就保留，不用就移到 `[project.optional-dependencies] reference = [...]`；
  - MDAnalysis/spglib/tables 同理审查。

### 7. 类型标注缺失
- **现状**：几乎所有函数没有 type hints。
- **问题**：mypy 在 pre-commit 里配了但形同虚设；重构时 IDE 帮不上忙。
- **建议**：不追求 100% 覆盖，先给公开 API（`calchem/api/high_level.py`、`calchem/cli/main.py`、`calchem/mol/molecule.py`）加 type hints。

## P2 — 工程质量

### 8. CI 没真跑过
- **现状**：`.github/workflows/ci.yml` 配了 Windows + Linux × Python 3.10/3.11/3.12 矩阵，但没 git remote，从来没触发过。
- **建议**：`git init` 之后挂一个 GitHub remote（哪怕是 private），让 CI 至少在 PR/push 时跑 `pytest -m "not slow and not gpu"`。

### 9. 测试双轨制
- **现状**：
  - 顶层 `tests/`（pytest 用例 + 大量历史脚本）
  - `calchem/tests/`（包自带的 4 个测试）
- **建议**：合并到顶层 `tests/`，`calchem/tests/` 留个 `__init__.py` re-export，或直接删除（其内容已经被顶层 tests 覆盖）。

### 10. 文档分散且不一致
- **现状**：`README.md`（目标态宣传）、`README_CalChem.md`（打包说明）、`examples/README.md`、`benchmarks/README.md`、`gui_upgrade/安装说明.md`、`PROJECT_WORKFLOW_SUMMARY.md`（逆向出来的真实状态）。
- **问题**：新读者不知道该信哪个。
- **建议**：
  - 保留一个主 README；
  - 把"已验证能量表"集中到 `handoff/03_NUMERICAL_BASELINE.md` 或 `docs/Baselined.md`；
  - 把 GUI 相关文档移到 `D:\YDW\CalChem\gui\`（它本来就在那边）。

## P3 — 功能补全（按作者真实科研需求）

### 11. 从未在 OPV 大分子上跑过
- **现状**：所有验证都是 H2/He/Ne/H2O/CH3。作者真实目标是 PM6、ClFFCl 这种几十个原子的共轭体系。
- **建议**：重构后第一个真实验收是：拿一个 ClFFCl 的 xyz，跑 B3LYP/def2-SVP 单点，看是否收敛、能量是否合理、RESP 电荷能否写出 GROMACS 拓扑。

### 12. 几何优化未真实验证
- **现状**：`opt/bfgs.py` 代码在，但没有"优化 H2O 到参考键长"的回归测试。
- **建议**：加一个 `tests/opt/test_h2o_bfgs.py`，优化后 O-H 键长应对齐 Gaussian 参考。

### 13. RESP/GROMACS 对接未真实验证
- **现状**：`md/gromacs_io.py` 只有 1.6 KB。
- **建议**：重构后跑通"RESP 电荷 → .itp → GROMACS 能 read"这条链路，这是作者做 OPV MD 的核心需求。

## 重构顺序建议

```
Week 1: P0（git + 参考值统一 + pytest norecurse）
Week 2: P1.4-1.6（GPU 去留、路径去硬编码、依赖清理）
Week 3: P1.7（公开 API 类型标注）
Week 4: P2.8-2.10（CI 真跑、测试合并、文档统一）
Week 5+: P3（OPV 大分子真跑、优化回归、RESP/GROMACS）
```
