# CalChem 分阶段执行计划

执行顺序不可跳过。每阶段先完成前置条件，再实现、验收、修复、记录，随后自动进入下一阶段。阶段结束不是询问用户的节点。

## 阶段 A：仓库与数值基线

**目标：** 可回滚、可复现地保住已验证的原生数值。

1. 核对压缩包和仓库状态，确认 `calchem/tests/` 与顶层 `tests/` 均保留。
2. 初始化/检查 Git，创建 `pre-refactor` 基线提交；生成 `docs/BASELINE.md`，记录 Python、依赖、运行命令与环境。
3. 读取 `test_regression.py` 中的实际几何和 `README_CalChem.md` 的基组/设置；对 H2O RHF 值不一致做溯源。只有在同一输入、同一设置下可重复时才更新值。
4. 修复 pytest 收集配置，使 `debug/`、`scratch/` 明确不进入默认 CI；不移动/删除其文件。
5. 全量执行既有正式回归一次并保存机器可读日志。多个 heavy jobs 必须串行。

**阶段验收：** 所有可运行的既有基线已记录；每个数值行注明来源/几何；H2O 差异有证据，不能用猜测结案。

## 阶段 B：核心接口、配置与结果契约

**目标：** 新架构可独立 import、schema 可序列化、无外部程序时也可测试。

1. 新建 `core/errors.py`、`request.py`、`result.py`、`calculator.py`、`capabilities.py`、`units.py`。
2. 定义字段含义、单位、空值、schema 版本；附一份 JSON 示例和生成 JSON Schema 的命令。
3. 定义显式 backend 注册表和能力检查；不支持任务/方法/基组在执行前报错。
4. 扩展既有 Pydantic Config，加入路径、执行超时、内存/核数、运行目录、日志级别、数据目录字段。
5. 为每个配置项规定优先级：CLI 明确传入 > 项目设置 > 用户配置 > 安全默认值；输出中记录最终生效值。

**验收：** Pydantic round-trip、schema 快照兼容、空字段为 null、无 numpy 原生对象泄漏、未知 backend/method/basis 有稳定异常。

## 阶段 C：NativeCalchem 迁移

**目标：** 老入口与统一接口共用同一数值核心。

1. 实现 `backends/native.py`，仅封装现有高层 API。
2. 映射现有 `Results` 和 `SCFResult`，把 numpy 数组显式转 list。
3. `api/high_level.py` 的旧函数暂时保留，改为调用 `TaskService` 或新 backend 的兼容 shim。
4. 参数映射逐项核对，确保 charge、multiplicity、xc、basis、SCF options 不丢。
5. 运行 H2、He、Ne、H2O、CH3 与 post-HF 的既有基线并对比阶段 A 结果。

**验收：** 老入口和新入口返回相同能量/收敛状态；回归满足原有阈值；无更改积分、SCF 或方法实现。

## 阶段 D：Gaussian 与 ORCA 外部后端

**目标：** 外部程序从统一 API 真正运行一次，并返回结构化 Result。

1. 实现公共 `ExternalProgramBackend`：绝对路径、`cwd`、timeout、捕获 stdout/stderr、run directory、command manifest。
2. 实现模板生成器；清洗/引用用户输入，不用 `shell=True`；支持可配置内存和核心数。
3. Gaussian：生成 gjf；检查实际 `g16` 路径；采集 `.log`/必要 `.chk`；解析 normal termination、SCF convergence、能量。
4. ORCA：生成 inp；采集 `.out` 与相关文件；检查正常终止与 SCF 收敛；按 ORCA 参数单位设置资源。
5. cclib 解析并映射字段，解析失败时保留原始输出与诊断尾部文本。
6. 维护真实输出 fixture 和人工检查说明；parser 单测不替代 end-to-end 运行。
7. 对真实 water fixture，使用相同几何、方法、基组依次串行跑 native/Gaussian/ORCA，记录不同程序数值差异及字段差异。

**验收：** 已安装的 Gaussian/ORCA 真正执行；结果中输入、日志、版本、退出状态、收敛状态和总能量齐全。某一外部程序不可用时，adapter 不得报告成已完成，其他阶段照常继续。

## 阶段 E：PySCF、xTB、REST 接入

**目标：** 各可选 backend 的安装/能力边界真实可见。

1. 将 PySCF 移到 `pyscf` extra；保留当前内部需要它的延迟导入代码，增加缺包错误测试。
2. PySCF backend 至少支持其能力表声明的方法单点；不支持的泛函/后处理方法不做隐式替代。
3. xTB backend 检查 binary 与版本，支持单点；优化等按阶段实现并验证；使用独立 run folder。
4. 检查作者真实 REST 程序文档/API/样例。取得协议后创建 `rest/schema.py` 与版本 adapter；没有接口定义时仅完成 disabled adapter 与接入说明，不编造接口。
5. 所有 backend 的能力状态由同一个 capability registry 提供给 CLI 和 GUI。

**验收：** 可选依赖未安装时 core 仍可导入；backend list 明确显示 available/configured/unavailable；所有已宣称任务有真实运行记录。

## 阶段 F：任务服务、作业队列、项目可复现性

**目标：** CLI、GUI、API 通过同一条运行通道和运行目录。

1. 新建 `TaskService`，完成输入验证、能力检查、资源限制、开始时间、异常转 Result、结果保存。
2. 新建 Job 状态机：`created -> queued -> running -> succeeded|failed|not_converged|cancelled`；状态转换合法性写成测试。
3. 单并发重计算。队列支持查看、暂停未开始任务、取消等待任务；运行中的外部程序可请求终止，必须记录是否成功停止。不能宣称原生线程可随时安全中断。
4. 项目保存 `project.json`、输入分子、请求、结果 JSON、日志路径、软件版本和时间戳；重新打开后可查看历史结果。
5. 加运行目录清理工具但默认不自动删原始日志；只允许用户选择具体目录执行清理。

**验收：** 重启应用后历史项目仍可查看；完成任务可从 run manifest 重建输入；崩溃/解析失败不丢 stdout/stderr/log。

## 阶段 G：完整 PyQt6 GUI

**目标：** 完成可实际工作的 Windows 桌面科学工作台。

1. 盘点现有 GUI 的真实文件和入口；复用能工作的组件，统一导入公开 API。
2. 依 `03_UI_DESIGN_SPEC.md` 完成工作区、分子输入、方法/基组/backend 表单、作业队列和结果页。
3. Qt worker thread 执行任务；主线程只更新 widget；全局只允许一个 heavy job running。
4. 分子几何编辑、结果表、收敛曲线、轨道能级、频率谱、原始日志视图连到真实数据；缺失数据展示“该 backend 未提供”。
5. 实现安装/程序发现向导、项目打开/保存、最近项目、配置编辑与诊断报告。
6. 加键盘可达性、清楚焦点、tooltip、缩放布局、错误提示和空状态；中文界面为默认，科学术语/单位统一。

**验收：** 在 Windows 主机上，从新项目打开到运行/查看/保存/重开全过程；大任务期间窗口响应；不会出现第二个 heavy job 与首个并发。

## 阶段 H：OPV 和 GROMACS 工作流

**目标：** 为产品的实际科研场景完成可追踪而不过度承诺的链路。

1. 选择已有合法分子 fixture（可先用水与一个明确许可的公开 OPV 分子），建立 XYZ 导入和元数据保存流程。
2. 串行 workflow：几何优化 -> 单点 -> 轨道/偶极 -> 可选频率 -> RESP 电荷（仅在实现/后端支持）-> GROMACS topology 导出。
3. 每个阶段保存独立请求与结果；前一步失败则下游标记 blocked，不生成假的结果。
4. 用目标 GROMACS 版本读入生成文件，做真实验收；写清力场/原子类型参数来源和适用限制。
5. ClFFCl/PM6 等真实目标 molecule 的执行资源与数值成功率单独报告，不把小分子通过当成大分子通过。

**验收：** 完成至少一个端到端示例，所有中间结构、参数、输出可追溯；GROMACS 真实读入成功，或明确标记某链路仍不支持。

## 阶段 I：冻结、打包、最终质量门

**目标：** 交付可安装/解压运行的版本和完整验收报告。

1. 在非 C 盘创建干净构建 venv，固定依赖；Windows 目标机执行 PyInstaller one-folder build。
2. 用 ZIP 分发 `CalChem/` 应用文件夹和 README；不捆绑 Gaussian/ORCA/xTB 本体，不包含私有密钥和作者临时文件。
3. 首次启动向导配置数据根目录、可执行路径；检测硬件/内存；显示 license / external software 状态。
4. 用干净 Windows 用户目录运行 smoke tests；测试带空格路径、Unicode 路径、无网络、缺少外部后端、日志权限错误、取消等待作业、应用重启。
5. 全量串行运行基线和可运行集成测试；检查打包文件完整性、启动日志、版本信息和卸载/更新说明。
6. 生成 `RELEASE_ACCEPTANCE.md`，附每个命令、实际结果、输出路径、版本、明确阻塞。

**最终 gate：** `05_FINAL_ACCEPTANCE.md` 无“未检查”项；无法完成的外部依赖项标为 blocked，并有错误/设备/路径证据。不能通过删掉验收用例来获得绿色状态。
