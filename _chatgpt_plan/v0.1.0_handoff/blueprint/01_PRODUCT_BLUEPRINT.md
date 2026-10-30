# CalChem 完整产品蓝图

版本：1.0  
目标环境：Windows 11、PowerShell 7、Python 3.10+（开发与发布构建优先 Python 3.13）  
产品类型：可本地运行的量子化学工作台，包含 Python API、CLI、桌面 GUI、多计算后端和可复现项目文件。

## 1. 产品目标

CalChem 提供一个面向研究者的统一工作区：建立或导入分子，配置计算任务，选择实际可用的计算后端，串行执行任务，查看结构化结果与原始日志，并把项目、输入、结果和版本信息一起保存。

首个完整产品版本必须做到：

- 继续运行现有原生 HF/DFT/Post-HF 路径，保留 H2、He、Ne、H2O、CH3 的已验证基线。
- 对外提供稳定 `Molecule`、`CalculationRequest`、`Calculator`、`Result` Python API。
- GUI、CLI、Python API 使用相同任务和结果模型。
- 至少实现 native、Gaussian 16、ORCA 三类可验收单点后端；PySCF、xTB 和 REST 按本蓝图的阶段范围接入，未完成时不得展示为可用。
- 以 JSON 结果和原始日志建立可审计链路；后端、版本、方法、基组、单位、输入、耗时、运行目录均可追溯。
- GUI 能导入 XYZ/GJF、编辑分子信息、提交单点/优化/频率任务、查看队列、结果、日志与轨道/频率图表。
- 全部 heavy 任务串行；关闭或崩溃不会悄悄丢失已完成结果。
- 提供可在 Windows 11 解压运行的便携式 one-folder 发布包，以及开发者安装说明。

CalChem 是平台，不把 Gaussian 定义为对手。外部程序的能力、许可证、安装情况和版本都由配置与运行检查明确显示。统一 Result 只统一字段结构和单位，不声称不同方法/程序的科学数值自动等价。

## 2. 明确技术栈

| 领域 | 选择 | 约束 |
|---|---|---|
| 核心语言 | Python 3.10+ | 发布前至少验证 3.10 和开发版 3.13；依赖版本在 lock/约束文件中固定。数值循环保留 numpy/scipy 实现。 |
| 数组与科学计算 | NumPy、SciPy | 原生引擎既有依赖；不得在统一 API 重构时改积分或 SCF 算法。 |
| 数据模型/配置 | Pydantic v2 | `Result`、任务请求、应用配置及 JSON Schema；数组序列化前显式转 Python list。 |
| CLI | Click + Rich | 保持当前 CLI 生态，命令结果可用 `--format json` 输出纯 JSON。 |
| GUI | PyQt6（沿用现有 GUI 技术） | GUI 仍作为独立应用目录/项目，不把 GUI 文件硬塞到量化核心包。只安装/打包一个 Qt binding。 |
| 图表 | Matplotlib | GUI 中画轨道能级、振动谱、收敛曲线；先使用已声明/已有版本，避免额外图表栈。 |
| 日志解析 | cclib 可选依赖 `external` | 作为 Gaussian/ORCA/xTB parser；应用仍需核对收敛/正常终止标志和必需字段。 |
| 测试 | pytest、pytest-cov | 单元、回归、parser fixture、外部软件集成测试分层；external/heavy 标记并默认串行。 |
| 静态检查 | Ruff、mypy（公开 API 优先） | CI 中执行；逐阶段清理，不要求一次性重写全部老代码。 |
| 构建 | PyInstaller one-folder | 先在目标 Windows 平台构建和运行；不使用 one-file 默认临时解包模式。PyInstaller 文档将 one-folder 作为默认模式，并建议先调试 one-folder。 |
| 安装/更新 | 第一版 ZIP 便携包；通过验收后再选 Inno Setup 安装器 | 用户可以指定安装位置；不得安装到 C 盘。Gaussian/ORCA 等商业/独立软件不随 CalChem 分发。 |
| 配置/数据 | JSON 项目文件 + Pydantic 配置 | 项目内保存任务与结果；全局设置使用用户指定的数据根目录。 |

PyQt6 是在附件既有 GUI 上保持一致的默认决策。若许可证审核或现有代码盘点发现 PyQt6 依赖冲突，先评估 PySide6 迁移成本并形成 ADR；不得在同一冻结环境同时打包 PyQt6 与 PySide6。PyInstaller 对多个 Qt binding 的收集有限制，Matplotlib 的 Qt hook 也需显式选定 binding。

桌面计算在 Qt worker thread 中执行，所有 QWidget 更新回到 GUI 主线程；worker 线程数固定为 1，队列调度器控制最多一个 heavy 任务活动。Qt 文档说明 GUI widgets 必须留在 GUI 线程，耗时工作应转到 worker thread。

## 3. 架构和依赖方向

```text
PyQt6 GUI ─┐
Click CLI ─┼──> calchem.api / TaskService ──> Calculator factory ──> Backend adapters
Python API ┘                    │                         ├── native engine
                                │                         ├── Gaussian 16 process
                                │                         ├── ORCA process
                                │                         ├── PySCF in-process
                                │                         ├── xTB process
                                │                         └── REST protocol adapter
                                ├── JobStore / ProjectStore
                                ├── Result schema / JSON serialization
                                └── serial JobQueue
```

### 模块职责

1. `calchem/mol`: 统一有限分子数据，不承担后端选择或作业调度。
2. `calchem/core`: `CalculationRequest`、`Result`、单位、异常、`Calculator` ABC、能力描述。
3. `calchem/backends`: 将统一请求转换为后端专有输入；执行或调用后端；解析输出；构造 Result。
4. 现有 `integrals/`, `scf/`, `dft/`, `post_hf/`, `grad/`, `opt/`, `vib/`, `props/`: 原生计算核心。第一阶段由 adapter 包装，不重写核心算法。
5. `calchem/api`: `TaskService` 与高层方法；保证 CLI/GUI/API 走同一验证、日志、结果存储路径。
6. `calchem/jobs`: 单并发 heavy job 队列、状态转换、取消请求、恢复与事件。
7. `calchem/io`: 输入格式、项目文件、Result JSON 和原始文件归档。
8. GUI：只调用 `TaskService`，展示能力和状态，不自行写 Gaussian/ORCA parser。

### 目标目录

```text
calchem/
  core/{__init__.py,calculator.py,request.py,result.py,capabilities.py,errors.py,units.py}
  backends/{__init__.py,base.py,external.py,native.py,gaussian.py,orca.py,pyscf.py,xtb.py,rest.py}
  jobs/{__init__.py,queue.py,models.py,events.py,store.py}
  api/{__init__.py,service.py,high_level.py}
  io/{input.py,output.py,project.py,checkpoint.py}
  config/{__init__.py,models.py,loader.py}
  mol/  basis/ integrals/ scf/ dft/ post_hf/ grad/ opt/ vib/ props/ ...
  cli/main.py
gui/   # 当前独立 GUI 项目延续；依赖 calchem 的公开 API
tests/
  unit/ integrals/ scf/ dft/ posthf/ api/ backends/ jobs/ gui/ integration/
docs/
```

第一版不移动旧计算模块；只有在迁移成本低且测试证明安全时才调整模块位置。

## 7. 官方技术参考

- [PyInstaller operating mode](https://pyinstaller.org/en/stable/operating-mode.html)：one-folder/one-file 的行为和调试建议。
- [PyInstaller usage](https://pyinstaller.org/en/stable/usage.html)：Windows 构建选项与平台构建说明。
- [PyInstaller Qt hooks](https://pyinstaller.org/en/stable/hooks-config.html)：避免同时收集多个 Qt binding。
- [Qt for Python: QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html)：worker object 与线程亲和性原则；PyQt6 使用相同 Qt 线程模型。
- [Qt for Python: GUI and worker threads](https://doc.qt.io/qtforpython-6/overviews/qtdoc-thread-basics.html)：widgets 留在 GUI 主线程。
- [cclib parsed data](https://cclib.github.io/data.html)：输出属性及不同 parser 的支持情况。
- [Pydantic serialization](https://docs.pydantic.dev/latest/concepts/serialization/) 和 [JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)：Result 序列化与 schema 生成。

## 4. 公共数据契约

### CalculationRequest

字段：`schema_version`、`task` (`single_point|optimize|frequency`)、`molecule`、`backend`、`method`、`basis`、`xc`、`options`、`resource_limits`、`project_id`、`job_id`。字段默认必须显式可追踪；应用配置可补默认值，但生成 request 时应保存最终值。

资源限制字段：`max_workers` 默认 8，`max_memory_mb` 默认由配置给出，`timeout_seconds` 默认按后端配置；全局 `heavy_job_concurrency` 固定为 1。每个后端只接收允许的 options，不允许把任意 route 文本直接拼进进程命令。

### Result

必须有 `schema_version`、`backend`、`backend_version`（未知可 null）、`task`、`method`、`basis`、完整 molecule snapshot、`status`、`started_at`、`walltime_seconds`、`units`、`errors`、`warnings`、`raw_files`。

标准能量采用 Hartree：`scf_energy_hartree`、`mp2_correction_hartree`、`ccsd_correction_hartree`、`ccsd_t_correction_hartree`、`total_energy_hartree`。不存在的量为 null，不能用 0 填充。轨道能量用 Hartree，占据数保留原始数值；偶极为 Debye；梯度为 Eh/Bohr；频率为 cm⁻¹；坐标在跨后端 Result 中统一为 Å。

分子快照：元素符号、坐标、总电荷、多重度、单位；`Molecule` 元数据可另带同位素/连接信息，但只有后端明确支持时才能产生相应结果。失败作业允许有部分结果，状态为 `failed` 或 `not_converged`，同时保留错误和日志路径。JSON schema 版本变更要有迁移说明。

### Calculator

必需操作：`single_point(request)`、`optimize(request)`、`frequency(request)`。便捷 facade 可提供 `energy(mol, method, basis)`。能力通过 `BackendCapabilities` 暴露：方法、基组、任务类型、是否需要外部程序、版本/路径检查结果、线程/内存限制。Factory 名称由显式注册表管理，不靠模糊 import 扫描。

## 5. 后端实现范围和诚实状态

| 后端 | 输入/运行 | 第一版必须输出 | 不满足时行为 |
|---|---|---|---|
| NativeCalchem | Python 调用现有原生方法 | 能量、收敛、方法/基组、轨道若可用、运行时间、警告 | 现有方法不支持则 UnsupportedMethod；不改变 kernel 数值流程 |
| Gaussian 16 | 写 `.gjf` 到独立 run dir；`subprocess.run`，不启 shell | 最终能量、正常终止、SCF 收敛、方法/基组、日志与输入 | 路径不存在、超时、非零退出、未收敛、解析缺字段都区别报告 |
| ORCA | 写 `.inp` 和辅助文件，运行程序 | 与 Gaussian 一样的核心结果字段 | 绝不把进程退出码 0 当作收敛依据 |
| PySCF | 可选 extra，延迟导入 | 将已有对象和方法映射到 Result | 未安装时报告可选依赖缺失；不影响 native 导入 |
| xTB | 外部程序输入与输出，单独能力标记 | 能量、优化几何（对应任务实现后）、耗时和原始文件 | 不支持的属性置 null；不能假装有 HF 轨道结果 |
| REST | 先依据实际程序接口建立 transport/schema | 只有真实 REST 规范确认后实施 | 未取得接口资料时显示未配置/未实现，禁止虚构 endpoint 和 mock 成功 |

cclib 用于 Gaussian/ORCA 输出解析的通用属性。仍需对运行中的程序版本检查其属性支持，并用真实日志校验正常终止/SCF 标志。

## 6. 产品边界

首个完整交付包括：GUI + CLI + Python API；单点、优化、频率基础工作流；native/Gaussian/ORCA 三后端；PySCF/xTB 可选后端按阶段完成；REST adapter 只有在协议获得后才进入“可用”；项目保存、运行目录管理、Result JSON、错误定位、文档、Windows 便携包。

自动择优后端、自然语言 agent、并行重计算、云计算、多人协作、周期性固体、GPU 加速、differentiable 路径、复杂工作流图编辑器不属于第一版验收范围。UI 不得出现已禁用功能的假按钮或夸大状态。
