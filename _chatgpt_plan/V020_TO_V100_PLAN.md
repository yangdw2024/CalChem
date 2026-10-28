# CalChem v0.2.0 → v1.0 文件级演进计划

日期：2026-10-28。依据本次 `CalChem_v0.1.0_Handoff.zip`、`CALCHEM_MASTER_PROMPT_FOR_CHATGPT.md` 与 `RELEASE_ACCEPTANCE.md`。目标机器为 Windows 11、8 核 CPU、约 15 GB RAM，单次只运行一个 heavy 计算任务，安装与数据目录位于非 C 盘。

## 0. 规划前事实与可信度

| 项目 | 本次可见证据 | 决策 |
|---|---|---|
| H2O/STO-3G RHF 三后端 | 验收文档列 native/Gaussian/ORCA 能量；按文档数值计算差为 4.232×10⁻⁹ 和 9.373×10⁻⁹ Eh | 将其视为本机先前验收记录；本轮无 `.gjf/.log/.out`，不能独立复验 |
| 数值内核 | ZIP 仅有 `source/calchem/core/` 等新平台层，没有 `source/calchem/scf/`、`integrals/`、`dft/` | 本地 Agent 必须先定位完整仓库，不能把当前 ZIP 当完整源码 |
| 验收脚本 | 报告称存在 `tests/acceptance.py`，ZIP 未包含 `tests/` | 必须补全源码快照并重新执行，不可用报告代替测试 |
| 任务队列 | `JobQueue.submit()` 入队，`get_next()` 需手动调用；`TaskService.submit()` 没有消费 worker；GUI 直接 `start_calc_thread()` | v0.2 P0：建立真正串行调度与持久状态 |
| 结果可信度 | native 取 `energy` 缺失默认为 0、`converged` 缺失默认为 True；Gaussian 取首个 SCF Done；ORCA 偶极 magnitude 被伪造成 z 向量 | v0.2 P0：精确定义能量/收敛/偶极，禁止假数据 |
| CLI JSON | `run --json` 先输出 Rich 的 Running 行，再输出 JSON | v0.2 P0：stdout 只能是 JSON，日志去 stderr |
| 打包 | 文档称已生成 one-folder，但 ZIP 内只有 `.spec`，没有发布目录与干净环境运行记录 | v1 gate 需真实发布包 smoke 测试 |

**结论：v0.1.0 可作为基础，但报告中 PASS 的范围不能自动继承为 v1.0 的质量证明。**

## 1. 产品范围与优先级

v1.0 定义为：可靠的 Python API/CLI/PyQt6 工作台，带可复现项目、串行作业调度、native/Gaussian/ORCA 单点和至少受验证的方法的几何优化/频率、PySCF/xTB 可选真实适配、OPV 最小科研工作流、Windows one-folder 发布包。REST 量化程序适配只有在获得真实协议与运行环境时才列为可用；REST-agent 是 L3 调用者，不等同于 RESTBackend。

优先级：

1. **P0 科学与作业正确性：** Result 真值、方法映射、资源/路径、单并发、可复现性、原生回归。
2. **P1 可工作的科研流程：** 真实后端 adapter、优化、频率、GUI 项目/队列/结果、诊断、Windows 发布。
3. **P2 方法与规模扩展：** DFT 格点/泛函、极化/扩散基组、OPV 重组能、GROMACS 接口。
4. **P3 长期研究：** 原生 M06-2X/ωB97X-D、GPU、JAX、周期性体系、自动 agent 规划。P3 不得为了赶 v1.0 宣称完成。

## 2. 固定技术决策

| 决策 | v1.0 选择 | 原因和实施条件 |
|---|---|---|
| 语言/平台 | Python 3.10+，Windows 11 主目标；3.13 开发环境可用 | 兼容原生引擎，CI 对 3.10 与 3.13 至少做纯 Python 验证 |
| API 契约 | Pydantic v2 Request/Result，显式 schema version | 保持现有结构；严禁模糊单位和 0 代替缺失值 |
| 数值计算 | NumPy/SciPy 原生路径，逐模块验证 | 保持已有基线，后端扩展不顺便重写 SCF/积分 |
| CLI | Click/Rich；`--json` 纯 stdout | 现有技术栈，便于脚本与 agent 消费 |
| GUI | PyQt6 + 现有 Matplotlib 视图，后续可换更快 viewer | 避免同时打包两个 Qt binding；GUI 线程只更新 widgets |
| 外部程序 | Gaussian/ORCA/xTB 由配置指定本机 executable；Python subprocess 参数列表 | 不打包独立程序、许可或用户路径；保存真实 stdout/stderr 和输入 |
| 输出解析 | 首选 cclib + 每程序版本 fixture + 正常结束验证；必要字段可保留有测试的专用解析 | cclib 官方测试 ORCA 4.1/4.2/5.0；本机 6.1.1 必须有真实样例验证，不能盲切 |
| 存储 | `.calchem` 目录，JSON 请求/结果/manifest，原始日志与输入按 Job ID 分开 | 可审计与断点恢复；先用原子写入，不急于加数据库 |
| 队列 | 单调度器，单 heavy slot，持久化状态 | UI 线程不阻塞；即使有多个入口也不能并发启动 heavy 任务 |
| 配置 | `AppConfig` 统一读环境/用户配置/项目/CLI，优先级 CLI > project > user > default | 删除 backend 构造器里的机器特定硬编码路径 |
| 发布 | Windows PyInstaller one-folder，独立干净 venv，非 C 盘构建/数据 | 先保证能跑，安装器放在最终 smoke 后；固定依赖并记录 SHA-256 |

### 科学数据约定

- `total_energy_hartree` 是对应 method 的最终能量；`scf_energy_hartree` 只记录 SCF 参考能量。MP2/CCSD(T) 必须从对应总能行或程序结构化数据提取，不能把 SCF Done 当总能。
- ORCA `wB97X-D4` 不等于请求的 `ωB97X-D`；映射必须按真实方法名分别注册，若不存在相同方法则拒绝，不静默替换。
- 偶极只有 magnitude 时存 `dipole_magnitude_debye`，矢量为 null；不能伪造 z 轴分量。
- `normal_termination`、return code、SCF convergence、最终能量字段都要核对。成功状态必须同时满足该任务所需条件。
- 输入 MoleculeSpec 验证元素、坐标个数、每个坐标有限、电子数/自旋一致性。坐标统一 Å，梯度 Eh/Bohr，频率 cm⁻¹，轨道能量 Eh，偶极 Debye。
- v0.1 已证实的 `1e-8 Eh` 阈值仅适用于严格相同的 H2O STO-3G RHF 输入与指定配置；新方法/基组/几何需要各自的参考输入和误差标准，不能普遍套用。

## 3. 分版本阶段与文件级任务

每阶段的“通过”指代码、自动测试、手工/真实运行证据、文档同步齐全。以完整本地仓库为实施目标；本次 handoff 缺失文件先补齐再动数值内核。

### v0.2.0：可信基础与单并发（先做）

**目标：** 修复会产生虚假成功、脏 JSON、任务丢失/并发、不可追溯的 P0 问题。

| ID | 文件/模块 | 明确动作 | 验收 |
|---|---|---|---|
| 2.01 | 仓库根、`docs/BASELINE.md`, `tests/acceptance.py` | 取得完整仓库快照和 commit；确认数值内核、测试、GUI、spec、构建脚本；复制真实 G16/ORCA 输入与日志进有许可的本地 fixtures；保留二进制/许可边界 | 缺失文件清单归零或明确 BLOCKED；执行 v0.1 原验收并存日志 |
| 2.02 | `core/request.py`, `core/result.py`, `core/units.py` | 增加字段/模型验证、有限数、维度、电子数/多重度；区分 dipole vector/magnitude、最终能/SCF 能，增加 method 实际值与 provenance | 错误输入被拒；JSON round-trip；旧 Result 由版本迁移函数读取 |
| 2.03 | `backends/native.py` | 移除 `energy=0.0`/`converged=True` 默认为成功；映射后 HF 总能和修正；open-shell 轨道 alpha/beta 不扁平化 | 缺 energy/convergence 返回失败或明确 null；现有 RHF/MP2 回归通过 |
| 2.04 | `backends/gaussian.py`, `backends/orca.py`, `backends/external.py` | 修正最后有效能量/方法，禁用未校验别名；要求 return code + 正常终止 + 方法终值；日志含 stdout/stderr，超时后进程清理 | 固定 log fixture 的 HF/DFT/MP2/CCSD(T) 分别提取对应值；真实 water 重跑 |
| 2.05 | `jobs/queue.py`, `jobs/models.py`, `api/service.py` | 建立唯一 dispatcher 消费队列，支持事件/结果落盘和失败后释放 heavy slot；CLI 同步路径也必须进同一锁/调度层 | 排队两个任务均被执行且顺序正确；运行数峰值为 1；异常后下一任务继续 |
| 2.06 | `gui/worker.py`, `gui/main_window.py` | UI 提交到 TaskService 队列，worker 只驱动队列；取消仅为请求，实际停止确认后才转 cancelled | 连续点击/CLI+GUI 同时请求不能并发；窗口不冻结；状态与结果一致 |
| 2.07 | `cli.py`, `config/models.py` | `--json` 普通信息走 stderr；后端路径从 AppConfig 注入；data/run/temp 默认为配置的非 C 盘 | stdout 仅 JSON；不同用户路径能运行；诊断列出最终生效配置但不泄密 |
| 2.08 | `tests/`、`pyproject.toml` | 分 unit、numerical、parser、external、GUI；增加真实科学错误的回归；scratch/debug 默认不收集 | Windows CI 无外部软件也能绿；external tests 明确 skip，不伪装 PASS |

**阶段 gate：** 完整本地仓库的原生基线与文档证据一致；相同水分子三后端真实单点串行跑出结果；一个错误后端不能产生 success；CLI JSON 可用 `json.loads` 解析；队列执行两个任务而非丢掉第二个。

### v0.3.0：后端、配置与可观测性

**目标：** 让适配器路径可配置，parser 稳定，PySCF/xTB 真正可用，REST 协议边界清楚。

| ID | 文件/模块 | 动作 | 验收 |
|---|---|---|---|
| 3.01 | `backends/gaussian.py`, `backends/orca.py`, `backends/parsers/` | 引入 cclib adapter，与原有 parser 并行对照真实 G16 A.03/ORCA 6.1.1 输出；按属性选择稳定路径 | 多任务 fixture 的方法/能量/收敛/偶极/轨道一致；解析失败保留原始日志 |
| 3.02 | `backends/pyscf.py`, `pyproject.toml` | optional extra，延迟导入；RHF/UHF/RKS/UKS/MP2 中实际验证的子集接入；结果单位显式转换 | 未装包仍可启动；安装后 H2O 单点与独立参考吻合；capability 不夸大 |
| 3.03 | `backends/xtb.py` | 支持 executable 发现、真实单点与可选优化；保存输入/输出 | xTB 真运行并产结构化结果；缺 binary 有清楚错误 |
| 3.04 | `backends/rest.py`, `backends/rest_schema.py` | 获得 REST 量化程序实际协议、版本和样例后定义 transport；REST-agent 作为 API 调用方另行定义 | 没有真实协议时 REST 显示 disabled；有协议时在独立环境完成一次真实调用 |
| 3.05 | `config/models.py`, `config/loader.py`, `cli.py`, `gui/diagnostics.py` | 配置优先级与 backend doctor、版本检测、日志/路径可写检查 | 改 GUI 配置后 CLI/API 使用同一配置；失败原因可定位 |

**阶段 gate：** 可选后端的真实运行与 unavailable 状态都经过检查；不同程序版本 parser 不依赖单个关键词侥幸匹配。

### v0.4.0：几何优化与解析/数值梯度

**目标：** native 已验证 RHF 小分子优化、Gaussian/ORCA 优化接入共同 Result。

| ID | 文件/模块 | 动作 | 验收 |
|---|---|---|---|
| 4.01 | `grad/hf_grad.py`, `grad/analytical.py`, `api/high_level.py` | 对完整本地仓库已有梯度做有限差分比较，单位和符号检查；只开放已验证方法 | H2O 每坐标解析/中心差分梯度误差在预先设定阈值内，记录步长收敛 |
| 4.02 | `opt/bfgs.py`, `backends/native.py` | 以原生梯度或显式数值 fallback 实现 optimize，记录每步能量/梯度/坐标；max steps 终止可诊断 | H2O 从扰动几何收敛到局部极小值；重复运行结果一致；失败不标 success |
| 4.03 | `backends/gaussian.py`, `backends/orca.py`, `backends/parsers/` | 各自生成 Opt 任务，读最终几何和能量，检查正常终止/优化收敛 | 同一 H2O 初始几何在两程序得到可比较最终结构；日志与步骤可查 |
| 4.04 | `core/result.py`, `gui/main_window.py`, `tests/opt/` | 加 `optimized_molecule`、`optimization_steps`、梯度单位；GUI 展示前后几何 | 字段序列化/重开项目/图形展示一致 |

**阶段 gate：** native RHF/H2O 解析梯度和优化数值可靠；Gaussian/ORCA Opt 真运行；不支持的方法仍显示 disabled。PySCF 优化如需 geomeTRIC/PyBerny，应明确为可选依赖，不得偷用 native 优化名义。

### v0.5.0：频率、Hessian 与 IR

**目标：** 给已验证方法提供正确模态与频率；IR 强度仅在后端确实提供或导数实现后开放。

| ID | 文件/模块 | 动作 | 验收 |
|---|---|---|---|
| 5.01 | `vib/frequency.py`, `grad/`, `backends/native.py` | 对质量加权 Hessian、平移/转动投影、同位素质量、负特征值/虚频做审查；从梯度差分实现稳健路径 | H2O 的非线性振动模式数、正/虚频判定和参考程序相符；H2 边界情况相符 |
| 5.02 | `backends/gaussian.py`, `backends/orca.py`, `backends/parsers/` | 生成 freq 任务并解析最终频率、normal modes、IR intensities（若有） | 真正完成两程序 H2O freq；unit 与模式数正确 |
| 5.03 | `core/result.py`, `gui/results/frequency_tab.py` | 增频率/normal mode/IR 数据和绘图、CSV 导出；缺 IR 时标记未提供 | GUI/JSON/CSV 数据一致，虚频醒目 |
| 5.04 | `tests/vib/`, `tests/backends/` | 平移/转动零模、不同差分步长、同位素替换测试 | 正常分子和近线性边界用例满足先定义的误差标准 |

**阶段 gate：** 频率的数值、单位、模态数和几何对应正确；未有 IR 强度时 UI 不画“模拟实测”的 IR 谱。

### v0.6.0：原生 DFT、基组和对称性质量

**目标：** 现有 LDA/PBE/B3LYP 在受控分子/基组上有可复现的格点收敛与数值基线；扩展方法按证据开放。

| ID | 文件/模块 | 动作 | 验收 |
|---|---|---|---|
| 6.01 | `dft/grid.py`, `dft/functionals.py`, `scf/dft.py` | 检查 Lebedev/Becke 格点、权重、密度/梯度、Vxc、混合泛函 exact exchange；网格粗/中/细收敛 | H2O/小有机分子的总能/梯度随 grid 收敛，记录与独立后端差异 |
| 6.02 | `basis/basis_data.py`, `basis/basis_set.py`, `integrals/` | 验证 6-31G(d), def2-SVP 和至少一个扩散基组的收缩、角动量、球/笛卡尔顺序与积分对称性 | 分子能量和矩阵元素与独立程序在相同 convention 下比较 |
| 6.03 | `mol/symmetry.py`, `core/capabilities.py` | 先实现/验证点群识别元数据；未将对称性实际用于积分缩减前不宣称加速 | 旋转/平移分子识别不变；禁用不支持的周期体系 |
| 6.04 | `dft/functionals.py`, `backends/native.py` | 对 ωB97X-D、M06-2X 各自单列研究任务：公式/参数/色散版本、能量、Vxc、梯度、spin handling、多个分子数值回归 | 只有全部通过才列入 native capability；否则 v1.0 经 Gaussian/ORCA/PySCF 提供并标明 backend |

**阶段 gate：** 原生已有 DFT 和新基组稳定；原生新泛函未验证时如实列为不可用，不阻塞平台 v1.0。DFT 网格不是“有代码就算完成”。

### v0.7.0：完整 GUI 与项目体验

**目标：** 用户可不借助命令行完成分子、设置、运行、查看、保存、重开全过程。

| ID | 文件/模块 | 动作 | 验收 |
|---|---|---|---|
| 7.01 | `gui/main_window.py`, `gui/project_panel.py`, `io/project.py` | 新建/打开/保存 `.calchem`，原子表编辑与几何同步，项目相对路径、原子/电荷/自旋校验 | 新建项目到重开可还原分子、设置、任务及结果 |
| 7.02 | `gui/job_panel.py`, `gui/worker.py`, `jobs/queue.py` | 多作业排队、状态/取消/重试、运行日志实时显示；队列跨应用重启恢复 | 两个任务依次完成；取消排队项不影响当前项；崩溃恢复状态准确 |
| 7.03 | `gui/results/{summary,orbitals,geometry,frequency,log}_tab.py` | 结果 tabs 和缺失数据状态；收敛曲线/轨道能级/频率图；导出 JSON/CSV/XYZ | 用真实 Result 渲染；无轨道/频率数据时不绘假图 |
| 7.04 | `gui/settings.py`, `gui/diagnostics.py`, `config/loader.py` | 路径选择、后端探测、资源预算、日志/版本诊断；非 C 盘数据目录 | 配置改变后实际 backend 使用新路径；错误场景可自助定位 |
| 7.05 | `gui/molecule_view.py` | 修正 `distance < 1.9 Å` 一刀切画键逻辑；用半径/可选用户键表，显示非化学键时标注推断 | 小分子、卤素、长键不误连；旋转缩放、选择原子同步 |

**阶段 gate：** Windows GUI 无 conda 开发环境也可启动；窗口在 heavy 任务运行时响应；从项目导入到结果导出端到端完成。默认界面只显示真实可用能力。

### v0.8.0：OPV dogfooding 与下游集成

**目标：** 一个可重复的科研工作流，不把小分子成功等同于 OPV 大分子可用。

| ID | 文件/模块 | 动作 | 验收 |
|---|---|---|---|
| 8.01 | `workflows/opv.py`, `jobs/workflow.py` | 以依赖图串行执行优化→单点→轨道/偶极；每步记录 method/basis/geometry hash；失败阻断下游 | H2O/小有机分子 workflow 可复现，重启后追溯每步 |
| 8.02 | `workflows/reorganization.py`, `core/result.py` | 按明确的四能量法计算 hole/electron 重组能；记录中性/离子电荷、自旋、几何和垂直/绝热单点 | 四个能量各有独立 Result；正反算公式一致；不能缺项硬算 |
| 8.03 | `md/gromacs_io.py`, `workflows/gromacs.py` | 明确 topology 力场来源、原子类型、键角参数与电荷来源；写出 itp/gro 或相应格式并由 GROMACS 真读入 | GROMACS 预处理成功；电荷和原子数一致；不把 RESP 电荷等同完整力场 |
| 8.04 | `cli.py`, `api/service.py`, `workflows/batch.py` | 批量分子只排队串行跑，支持失败跳过/继续、断点恢复、结果索引 CSV | 10 个小任务按序执行，峰值 heavy 并发 1；中途失败后可恢复 |
| 8.05 | `examples/opv/`, `docs/opv.md` | 给一个许可明确的中型 OPV 示例与资源预算；ClFFCl/PM6 如有实际输入再做 dogfood | 至少一个中型体系的后端可行性报告，不夸称 native 能处理所有大分子 |

**阶段 gate：** 工作流数据链真实可复现，GROMACS 真实接受导出文件；无参数来源时导出器清楚拒绝生成不可用拓扑。

### v0.9.0：稳定性、CI 与发布候选

**目标：** 对所有承诺功能设置质量门，并在 Windows 干净环境验证。

| ID | 文件/模块 | 动作 | 验收 |
|---|---|---|---|
| 9.01 | `pyproject.toml`, `requirements/`, `.github/workflows/ci.yml` | 约束版本、optional extras、Ruff/mypy、Windows CI；默认无外部许可证 | Windows 3.10/3.13 核心测试绿；依赖安装不靠 C 盘缓存 |
| 9.02 | `tests/` | 构建 contract/numerical/parser/external/gui/workflow/release 诸层；外部程序本地串行 gate | 测试分类明确；出错后不被笼统 skip 掩盖 |
| 9.03 | `calchem.spec`, `scripts/build_windows.ps1` | 干净 D/E 盘环境 PyInstaller one-folder；只收单一 Qt binding；显式数据文件/动态 import | 解压到有空格/中文目录、无 conda 开发环境运行 GUI 和 CLI |
| 9.04 | `docs/`, `README.md`, `LICENSES/` | 对齐当前 capability、用户教程、开发说明、外部程序配置与许可提示 | README 不宣称 GPU 或未实现后端；用户可独立完成首次运行 |

**阶段 gate：** Release candidate ZIP、SHA-256、构建日志、CI 状态和缺陷表齐全。

### v1.0：完整验收与发布

**目标：** 交付可安装/解压、可用、可审计的工作台。

1. 重跑所有原生数值回归，固定同一几何/方法/基组；对优化、频率、DFT/基组建立独立参考表与容差。绝不为通过测试改写参考值而无来源。
2. 在作者 Windows 主机真实、串行运行已配置的 Gaussian/ORCA/PySCF/xTB 项目；保存输入、输出、环境和版本。REST 未取得协议则标记 unavailable 并从“已完成”列表移除。
3. 干净 Windows 用户环境解压 one-folder 发布包；设置非 C 盘数据根；运行 native/H2O、导入/保存/重开项目、查看结果/日志、进行一项优化及频率可用流程。
4. 验证无网络、未配置外部程序、超时、输出损坏、SCF 不收敛、磁盘只读、中文/空格路径、取消/重试、应用重启等边界。
5. 完成 `RELEASE_ACCEPTANCE.md` 的逐项 PASS/FAIL/BLOCKED；生成版本信息、变更日志、SHA-256、安装与卸载说明。只发布真实构建产物。

**v1.0 gate：** 公开能力、代码、GUI、CLI、文档与真实验收一致。可选或长期研究功能未通过不应阻塞可靠平台发布，但必须从 v1.0 能力声明中剔除。

## 4. 关键设计细节和资源边界

### 单并发调度不是“禁用运行按钮”

`TaskService.run()`、`submit()`、GUI worker、CLI、workflow/batch 都必须进入同一个调度层。单进程内的 `threading.Lock` 无法阻止另一个 CLI 进程和 GUI 同时启动程序；使用放在统一 data root 的跨进程排他锁或本地单实例 job supervisor。锁持有时包含 PID、开始时间、job ID；陈旧锁只在确认对应进程已退出后回收。任何一次外部执行都受此锁控制。

### 作业状态与持久化

允许：`pending→queued→running→succeeded|failed|not_converged`，取消路径 `queued→cancelled` 或 `running→cancel_requested→cancelled|failed`。重试必须创建新 job ID 并关联 parent_job_id，不把终态 job 直接改回 running。Request、state、Result 和 manifest 先写临时文件，再原子替换；CLI/GUI 重启后读取同一数据。

### 错误和日志

区分 input invalid、unsupported task/method/basis、binary unavailable、timeout、process nonzero、parse incomplete、SCF not converged、optimizer not converged、I/O error。失败 Result 也有 run directory、原始 stdout/stderr、命令参数（脱敏）、版本。大日志用文件流收集而非全部 `capture_output=True` 常驻内存。

### 化学与方法验证

基组/泛函兼容由每个 backend 的精确 capability 声明，不用“外部程序理论上能运行”推断已验收。比较能量需相同几何、charge/spin、method 实际定义、基组（球/笛卡尔/EC P、grid、SCF threshold）和色散修正。优化后比较最终几何/梯度；频率比较模式/质量/单位。原生高阶泛函必须具备能量、势、梯度的数值一致性，否则只标记实验性且不进入默认能力表。

## 5. 风险与处理策略

| 风险 | 影响 | 预防/应对 |
|---|---|---|
| handoff 缺原生内核与 tests | 无法仅凭附件复验现有 PASS | 第一任务盘点完整本地仓库和日志；保留证据清单 |
| ORCA 6.1.1 与 cclib 支持版本差异 | 字段错位或缺失 | 真实 6.1.1 fixture 与手写 parser 对照，按字段渐进迁移；官方当前概述仅列到 ORCA 5.0 |
| post-HF 解析取错能量行 | 科学结论错误 | 方法专属总能 parser；与原始日志/独立程序逐项核对 |
| native 梯度/频率未验证 | 错误几何/虚频 | 有限差分、步长收敛、平移/转动零模、独立后端对照 |
| 大分子/扩散基组资源超限 | OOM、耗时不可控 | 单任务预算、预估 basis 函数数、超时、明确拒绝或选择外部后端 |
| 多进程并发 | 15 GB RAM 溢出或 G16 冲突 | 跨进程重计算锁与单调度器，CI heavy tests 串行 |
| REST 程序与 agent 混淆 | 虚构接口和错误任务调用 | RESTBackend 仅面向量化程序；REST-agent 调 TaskService；等待真实协议 |
| GROMACS topology 参数不全 | 生成不可用或物理错误输入 | 电荷/原子类型/力场参数来源全记录；缺参数则阻止导出 |
| GUI 仅在 conda 环境可用 | 发布包不可运行 | 在干净 Windows 用户、无开发环境的 one-folder 测试 |
| Windows C 盘约束 | 临时目录/安装违约 | 显式配置 venv/build/cache/TEMP/data 在 D/E 盘并验证实际路径 |

## 6. 最终交付清单

1. 版本化完整源码仓库与 changelog；各可用后端真实 capability 表。
2. Windows one-folder ZIP、SHA-256、非 C 盘首次启动向导、安装/升级/卸载说明。
3. `RELEASE_ACCEPTANCE.md`：每项状态、命令、原始输出/日志路径、环境版本；缺失项明确 BLOCKED。
4. 原生数值基线与新方法参考表；外部后端真实日志与 parser fixture 来源。
5. GUI 操作手册、CLI/API 文档、项目格式/schema 版本、OPV/GROMACS 示例的适用范围。

## 官方参考

- cclib 官方 [Overview](https://cclib.github.io/) 和 [Parsed data](https://cclib.github.io/data.html)：各程序版本与属性覆盖。
- PySCF 官方 [Geometry optimization](https://pyscf.org/user/geomopt.html) 和 [DFT](https://pyscf.org/user/dft.html)：优化依赖及泛函能力。
- ORCA 6.1 官方 [Geometry optimizations](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html)：优化/梯度能力边界。
- PyInstaller 官方 [Operating mode](https://pyinstaller.org/en/stable/operating-mode.html) 和 [Usage](https://pyinstaller.org/en/stable/usage.html)：one-folder 与 Windows 构建。
