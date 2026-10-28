# 给 ChatGPT 的主提示词：CalChem 平台架构设计

> 直接复制下面整段（从 `===BEGIN===` 到 `===END===`）粘贴给 ChatGPT。
> 同时把附件 `CalChem_Refactor_Handoff.zip` 给它。
> 目的：让 ChatGPT 作为首席架构师，产出完整的模块划分、接口定义、任务拆解，之后我（Doubao）按它的设计逐个实现。

---

===BEGIN===

# 角色

你是一名资深计算化学软件架构师，有 10 年以上量子化学软件（Gaussian/ORCA/PySCF/ASE 级别）设计经验，同时熟悉 LLM agent 系统。你现在要为一个学生主导的开源项目做**完整的平台架构设计**和**模块化任务拆解**。

我已经把现有源码、文档、战略定位全部打包在附件 `CalChem_Refactor_Handoff.zip` 里。**在回答前，请先解压并阅读以下文件**（按顺序）：

1. `handoff/00_MASTER_CONTEXT.md` — 项目现状、已验证能力、红线
2. `handoff/06_STRATEGY.md` — 战略定位 v2（平台架构、后端适配、路线图）
3. `handoff/02_ARCHITECTURE.md` — 当前已有的 calchem/ 模块结构
4. `handoff/03_NUMERICAL_BASELINE.md` — 数值安全网（不能破坏的参考能量）
5. `handoff/04_TECH_DEBT.md` — 技术债清单
6. `handoff/05_ENV_AND_CONSTRAINTS.md` — 硬件/环境/禁忌
7. `calchem/` 目录下的源码（重点看 `api/high_level.py`、`scf/hf.py`、`scf/dft.py`、`cli/main.py`、`io/input.py`）
8. `pyproject.toml` — 依赖声明

# 项目背景（一句话）

**CalChem 是一个统一的量子化学计算平台**：
- 自带从零实现的原生引擎（RHF/UHF、DFT、MP2/CCSD(T)/CASSCF、TDDFT、优化、频率、RESP），小分子 STO-3G 能量已与 Gaussian 16 对齐到 1e-8 Ha；
- 通过适配层接入 Gaussian、ORCA、REST（作者自研的另一个量子化学程序）、PySCF、xTB 作为可替换后端；
- 上层由 REST-agent（基于 Qwen3-8B 的智能体）用自然语言驱动；
- 所有后端通过统一的 `Calculator` 接口和统一的 JSON `Result` schema 返回结果。

# 硬约束（设计时必须遵守）

- 操作系统：Windows 11，PowerShell 7，Python 3.10+（开发用 3.13）
- 硬件：AMD Ryzen 7 9700X（8 核）、15.1 GB 内存、8 GB 显存
- **严禁并行跑多个 heavy 计算任务**；C 盘禁止装任何东西
- 数值基线：H2/He/Ne/H2O/CH3 在 STO-3G 下的能量 |ΔE| < 1e-8 Ha 不能破坏
- 代码注释用中文，解释"为什么"而不只是"做什么"
- 作者是化学研究生（OSC/OPV 方向），不是全职软件工程师——模块划分要让一个人能分步实现，不能设计成需要 20 人团队的架构
- Gaussian 16 装在 `D:\G16W`；GUI 在另一个目录 `D:\YDW\CalChem\gui`（不在本仓库）
- 不追求在 H2O 单点速度上超过 TeraChem——那是团队战，我们打不过

# 我要你产出的东西（严格按这个格式）

## 第一部分：总体架构

1. 用 Mermaid 画一张**完整架构图**，从用户输入（自然语言 / CLI / Python API）到后端执行（native / gaussian / orca / rest / pyscf / xtb）到结果输出（JSON / GUI / 文件），标出数据流向。
2. 列出分层（L1 后端、L2 统一 API、L3 Agent/Workflow），每层包含哪些模块，模块之间的依赖方向。
3. 明确说明：哪些是**新建**的，哪些是**重构现有代码**的，哪些是**暂不做**的。

## 第二部分：核心接口设计（最重要）

### 2.1 `Calculator` 抽象基类

给出完整的 Python ABC 定义，方法至少包括：
- `single_point(mol, method, basis, **kwargs) -> Result`
- `optimize(mol, method, basis, **kwargs) -> Result`
- `frequency(mol, method, basis, **kwargs) -> Result`
- `energy(mol) -> float`（便捷方法）
- `backend_name`、`supported_methods`、`supported_basis` 等属性
- 工厂方法 `Calculator.backend(name, **kwargs)`

要包含 type hints、docstring（中文）、异常定义（不收敛、方法不支持、基组不支持）。

### 2.2 `Result` 统一 JSON Schema

给出完整的 JSON Schema（或 Python dataclass），字段至少覆盖：
- 元信息：backend、method、basis、molecule、walltime、version
- 能量：scf_energy、mp2_correction、ccsd_correction、total_energy
- SCF：converged、iterations、energy_history
- 轨道：homo_energy、lumo_energy、orbital_energies、occupations
- 性质：dipole、gradients、vibrational_frequencies、normal_modes
- 错误与警告：errors、warnings
- 原始日志路径：raw_log_path

要说明哪些字段是必填、哪些是可选、哪些是某些后端才有的。

### 2.3 `Molecule` 接口

现有 `calchem/mol/molecule.py` 已经有 Molecule 类。请评审：
- 它是否适合作为跨后端的统一分子表示？
- 需要加什么字段（isotopes、bond orders、charges、periodic boundary）？
- 给出改造建议（不要重写，在现有基础上扩展）。

## 第三部分：后端适配器

为每个后端给出一个**完整的适配器骨架类**（继承 Calculator），包括：

| 后端 | 输入文件 | 执行方式 | 输出解析 | 何时使用 |
|---|---|---|---|---|
| NativeCalchem | 直接 Python 对象 | 内存调用 | 直接内存 | 快速迭代、自定义方法、教学 |
| GaussianBackend | 写 .gjf | 调 g16/g16.exe | 解析 .log / .fchk | 成熟方法、高精度对照 |
| ORCABackend | 写 .inp | 调 orca.exe | 解析 .out | 自由 license、DFT |
| RESTBackend | REST 程序接口 | 按 REST schema | 按 REST schema | 用 REST 特色方法 |
| PySCFBackend | Python 直接调 | in-process | 直接内存 | 快速原型、ML 数据集 |
| xTBBackend | 写输入 | 调 xtb | 解析输出 | 粗筛、几何预优化 |

每个适配器给出：
- 类定义骨架（含主要方法签名）
- 输入文件模板（Gaussian gjf 模板、ORCA inp 模板示例）
- 输出解析策略（用什么库正则/解析，或者建议引入 cclib）
- 错误处理（Gaussian 不收敛怎么办、ORCA 路径找不到怎么办）
- **不要现在写完整实现，只写接口骨架和关键决策**

## 第四部分：模块任务拆解（按优先级排）

把整个 Phase 0（接下来 8 周）拆成具体任务，每个任务给出：

```
### Task X.Y: 任务名
- 所属模块: calchem/core/...
- 目标: 一句话
- 前置依赖: Task X.Y / 无
- 具体动作: 3-5 个可执行步骤
- 产出物: 文件路径 + 关键类/函数名
- 验收标准: 怎么算完成（可测试）
- 预估工作量: S/M/L（半天/2天/1周）
- 风险: 可能卡在哪里
```

任务要覆盖：
- 统一 Calculator ABC 和 Result dataclass
- 现有 high_level.py 重构为 NativeCalchem
- GaussianBackend 第一版（单点能即可）
- ORCABackend 第一版
- CLI 改造（`--backend` 参数）
- 统一 Result 的 JSON 序列化
- 基线回归保护（确保重构不破坏 1e-8 能量）
- 配置系统（路径、可执行文件位置、默认参数）
- 日志系统
- 测试策略（pytest 怎么组织跨后端测试）

## 第五部分：目录结构建议

给出 `calchem/` 重构后的目标目录树，标注每个文件/目录的职责。要和现有结构兼容（能渐进迁移，不要一次性大改）。

## 第六部分：依赖决策

对每个第三方库给出建议（引入/不引入/可选），并说明理由：
- `cclib`（解析 Gaussian/ORCA 输出）
- `ase`（分子表示、后端模式）
- `pydantic`（已经在依赖里，用于 Result 校验）
- `rich`（已经在依赖里）
- `click`（已经在依赖里，CLI）
- `hydra` / `omegaconf`（配置）
- `tqdm`（进度条）
- 其他你认为该引入的

## 第七部分：路线图和里程碑

按周给出 8 周 Phase 0 里程碑，每周结束时能跑什么东西（"周五演示"）。

# 输出要求

- 用中文写，代码注释用中文，技术术语保留英文
- 所有代码块要能直接放进项目里跑（不要伪代码）
- 不要写空话（"要加强模块化"这种），要具体到类名、文件路径、函数签名
- 如果你认为我的战略有问题或有更优解，直接指出，不要迎合
- 最后给一个"我建议你砍掉哪些范围"的清单——8 周做不完的就别做

# 附件说明

`CalChem_Refactor_Handoff.zip` 包含：
- `handoff/` — 6 份导航文档（必读）
- `calchem/` — 现有原生引擎源码
- `tests/` — 已整理的 110 个测试（按 unit/integrals/scf/dft/posthf/gui/debug/scratch 分目录）
- `examples/`、`benchmarks/`、`scripts/`、`gui_upgrade/`
- `pyproject.toml`、`README.md`、`README_CalChem.md`
- `PROJECT_WORKFLOW_SUMMARY.md` — 完整工作流逆向

读完附件再开始设计，不要凭空想象。

===END===

---

## 使用方法

1. 把上面 `===BEGIN===` 到 `===END===` 之间的全部内容复制给 ChatGPT；
2. 同时把 `CalChem_Refactor_Handoff.zip` 作为附件发给它；
3. 让它按格式产出架构设计；
4. 它产出后，把结果给我，我按它的设计逐个模块实现。

## 我已经做的准备

- [x] 战略定位 v2 写在 `handoff/06_STRATEGY.md`
- [x] 现有源码全部打包（calchem/ + tests/ + 文档）
- [x] 数值安全网写明（03_NUMERICAL_BASELINE.md）
- [x] 技术债清单写明（04_TECH_DEBT.md）
- [x] 硬件/禁忌写明（05_ENV_AND_CONSTRAINTS.md）
- [x] 本主提示词放在 `handoff/07_MASTER_PROMPT_FOR_ARCHITECT.md`
