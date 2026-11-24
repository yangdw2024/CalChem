# 四引擎标定批重启后审计

2026-11-23，Phoenix 本机时间。Windows 确实重启，原 CalChem 优化已经结束于进程层面。保留下来的标定结果足以定位任务管线、计算口径、CLI 接线和优化器的不同问题。当前应先恢复可观测的小规模核验，再决定大批次的每项任务是否适用。

## 重启与队列

| 证据 | 状态 |
|---|---|
| Windows `LastBootUpTime` | 2026-11-22 17:55:23，UTC−07:00。系统事件同时记录 17:52 的关机与 17:55 的启动 |
| Pueue 4.0.4 | 本机没有守护进程或 6924 监听。用正确配置查询返回连接失败，退出码 1。系统未查到 Pueue 服务或同名自启动计划任务 |
| 159，CalChem | 旧状态 `Paused`，文件最后写入 17:45:13，早于重启。SP 已完成。OPT 自 06:43:26 开始，至约 17:45 暂停约 11 小时。无原计算进程、结束标记、优化结构、检查点或 opt.json。旧进程内存无法恢复 |
| 160 / 161，ORCA 诊断 | 两条均为 `Queued`，无实际启动时间、输出或诊断结论 |
| `qc_ac` 旧批 | 组暂停，共 32 条。123 成功，124 为 `Killed`，其他 30 条排队但未启动。123 实际为 STO-3G、94 原子、334 AO，其 10.73 秒为程序内部时间，Pueue 记录 12.71 秒。不能将这条用于 6-31G(d,p) 对照 |
| `qc_bench` 标定批 | 磁盘组并发为 1，155–158 四个 wrapper 层显示成功。156 的四段实际均 exit=1，158 的频率 exit=101，TD 探针运行成单点。需要分别核验子命令 |

早先会话结束导致守护消失是旧轮推断。本轮证据支持 Windows 重启和守护未随系统恢复。现存 `start-daemon.ps1` 仍使用 `-d`，早先的临时脱离会话启动没有形成重启持久化。`VERIFY_STATE` 第 8 轮记载的原进程续跑选项只适用于重启前。定时核验频率属于调度设置，不能替代本机守护或计算进程的判活。 [证据快照](SNAPSHOT.json) · [旧状态](E:/YDW/Chemdraws/zmy_3/qc_benchmark_suite/VERIFY_STATE.md) · [启动脚本](E:/YDW/Scheduling_System/scripts/start-daemon.ps1)

本轮对队列、现有脚本、原输出和业务源码只读。将关键任务字段、重启事件和原文件 SHA256 保存到快照。原始队列状态含进程环境变量，对外只使用脱敏字段。状态建议为 159/Opt `INTERRUPTED_REBOOT_NO_CHECKPOINT`，159/Freq 与 159/TD `NOT_STARTED_IN_159`，160/161 `QUEUED_DAEMON_OFFLINE`。CalChem 156 遗留的 freq.log 和 tddft.log 是旧尝试的 stored-ERI 拒绝，不能算作 159 执行过后两段。 [CalChem 步骤日志](D:/YDW/Calc/qc_benchmark_runs/pilot/calchem/_steps.txt)

## 咖啡因实测及可比范围

逐元素、逐坐标比较 CalChem XYZ、Gaussian、ORCA 与 REST 的单点输入，24 原子顺序一致，坐标差为 0 Å。基函数数目分别为 Gaussian/CalChem **260**、ORCA **246**、REST **216**，可由 6-31G 的 s/p 基函数 146 加极化壳解释。14 个重原子加 6D 或 5D，10 个 H 在 (d,p) 基组各加三个 p 函数。三家已输出的单点 D3BJ 项约 −0.04587725 Eh，彼此差小于 5 × 10⁻⁹ Eh，核排斥能也与同一输入几何一致。

| 引擎 | 单点，程序内部墙钟 | 优化 | 频率 | 激发态 |
|---|---|---|---|---|
| Gaussian 16 A.03，8 线程 | −680.431066858 Eh，31.5 s，15 步 | 真收敛，163.8 s | 273.4 s，解析二阶导数，0 虚频，最低 24.5617 cm⁻¹ | 192.2 s，优化 chk 的几何和初猜，5 根完整 TDDFT |
| ORCA 6.1.1，8 MPI 进程 | −680.039873676235 Eh，35.839 s，VWN5、5D、RIJCOSX | 真收敛，与 Gaussian 优化重原子 RMSD 0.00115 Å | Opt+Freq 整段 219.204 s，已输出 Hessian，1 个 −12.63 cm⁻¹ 模 | 咖啡因尚未提交 |
| REST 2026.1.1，WSL，设 8 线程 | −680.4431047580 Eh，29.310 s，6-31G(d) 球谐，in-core RI-J/K | geomeTRIC，11 步后收敛，整作业 260.337 s，重原子 RMSD 0.02307 Å | 位移 SCF 已执行，质量加权 Hessian 对角化 panic，退出 101，约 44m15s 后失败 | 旧关键词落回单点，未获得激发态 |
| CalChem v0.2.2 CLI，16 Rayon 线程 | −680.4420146368293 Eh，29.057 s，20 步，Coarse、RI-J、CPU | 159 重启前无完成证据，约 11 小时后暂停并失去原进程 | 159 未开始。当前实现是中心差分梯度，D3BJ 传递待修 | 159 未开始。CLI 固定 TDA singlet，需匹配参考方法 |

Gaussian 优化与频率是两个 Link1 子作业，各自有 elapsed 和正常终止行。163.8 + 273.4 = 437.2 秒，wrapper 约 440 秒。这才是该组合的整段时间。Gaussian 的 5 根激发能为 4.6961、4.9142、5.6901、5.9786、6.0326 eV。CalChem 29 秒、REST 29 秒和 Gaussian 31 秒的单点各用不同方法配置，CalChem 线程预算也不同。此表不能给出公平速度排名。CPU 为 8 核 16 逻辑线程，CPU 时间 / 墙钟可近似表示占用的逻辑核当量，不能直接得到线程扩展效率。 [Gaussian Opt/Freq](D:/YDW/Calc/qc_benchmark_runs/pilot/gaussian/optfreq.log:4060) · [Gaussian TD](D:/YDW/Calc/qc_benchmark_runs/pilot/gaussian/td.log:815) · [ORCA 输出](D:/YDW/Calc/qc_benchmark_runs/pilot/orca/optfreq.out) · [REST 优化](D:/YDW/Calc/qc_benchmark_runs/pilot/rest/GeomeTRIC.log) · [CalChem 单点](D:/YDW/Calc/qc_benchmark_runs/pilot/calchem/sp.json)

## 能量和振动的归因

**ORCA +245.48 kcal/mol。** 手册与原始日志证明 ORCA 默认 B3LYP 用 VWN5，而 Gaussian B3LYP、CalChem `b3lyp` 对应 VWN3/RPA。VWN 是首要可检验解释。当前还混有 5D/6D、网格与 RIJCOSX 的差异。局部均匀电子气几个取样点及相近的弛豫能不能分解整分子的 0.39119 Eh。约 0.043 kcal/mol 的弛豫能差、0.00115 Å RMSD、0.0014 eV 能隙差，只约束当前构型附近有限的量。RI-J 的固定密度变分讨论也不能排除 ORCA 同时近似 J 与 K、网格和基组改变后的净差。 [ORCA 泛函手册](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/DensityFunctionalTheory.html) · [ORCA 基组手册](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/basisset.html)

160 与 161 **都**值得跑。原 B3LYP/RIJCOSX 对 160 的 B3LYP/NoRI 测 J/K 近似组合，160 对 161 的 B3LYP/G/NoRI 在同为 5D 的设置中隔离 VWN。比较前核实网格、D3BJ 参数和旧 GBW 未参与。两组差值出来后再用 Gaussian 显式 `5D` 给 ORCA 建对照，保持旧 6D 参考服务于 CalChem。ORCA 6.1 对 d 函数采用 5D。四方共有的基础对照可先用 6-31G 的 s/p 基组，验收前核对各元素与壳结构。带 d 极化的四方比较需要 CalChem 5D 能力或分成配对参考。REST 已跑的 6-31G(d) 不含 H 的 p 极化，外部基组文件路径能力应单独验证。

ORCA 第 6 模为 −12.6263 cm⁻¹。独立读取 `.hess`，第 15–17 号 H 对原子位移的平方和贡献为 **97.62%**，属于第 1 号 C 与第 2 号 N 相连的甲基，和绕 C–N 轴扭转位移的绝对归一化内积约 0.972。可定位为甲基扭转主导的软模。现结果在无虚频验收中不通过。用密网格、更紧几何收敛和沿此模式双向微扰再优化，区分真实浅鞍点与数值曲率误差。 [ORCA Hessian](D:/YDW/Calc/qc_benchmark_runs/pilot/orca/optfreq.hess:1114) · [Gaussian 格点与低频说明](https://gaussian.com/wp-content/uploads/dl/vib.pdf)

**CalChem −6.87 kcal/mol。** 已确认该单点是 260 AO、B3LYP/RPA-VWN、Coarse、自动 RI-J。色散和坐标无法解释差值。CalChem Coarse 的基础径向数是 H 25、C/N/O 35，角向区域剪枝，最大 110。CalChem UltraFine 基础 200×590、角向不剪枝、流式实现，径向坐标、部分原子权重、有效点数和 Gaussian 网格仍不同。Gaussian 输入未显式指定格点，输出 `IXCGrd=4`，诊断与后续对照应显式锁定网格。格点差是可信的嫌疑，RI-J、剩余积分和求解误差也要隔离。即使密格点、无 RI 得到的总差很小，也要用四格对照看两种误差是否抵消。 [格点源码](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/grid.rs:142) · [Gaussian 单点原始日志](D:/YDW/Calc/qc_benchmark_runs/pilot/gaussian/sp.log:325)

**几何与派生量。** 输入到 Gaussian 优化结构的弛豫能约 3.27 kcal/mol，说明需区分共同输入结构和优化驻点。这个能量差无法推导 TD 根移动了 0.142 eV。未优化几何的 Hessian 可能出现负曲率，虚频也并非必然。比较振动/热化学时验证各自相同模型的驻点和振型，再做共同几何上的梯度/Hessian 对照。比较光谱时先在同一参考坐标做同种垂直激发计算，按跃迁性质和振子强度匹配根，再测试各自优化几何的影响。

## CalChem 问题的代码落点

| 优先级 | 确认的行为 | 处理方法 |
|---|---|---|
| P0 模型一致性 | CLI 算出初始几何 D3 能并暂加到总能量。`Task::Opt` 构造 `OptOptions` 时没传 `dispersion`，其默认是 `None`。优化结果覆盖最终能量，而色散和电子能分量仍来自初始 SCF。通用 `grad` 不加 D3 梯度，`FreqOptions` 无通用色散字段，位移梯度、Hessian 和热化学参考能不含该 D3 | 让 energy、grad、opt、freq 共用明确的模型参数。优化结果在最终几何报告各能量分量。位移点加同模型 D3 梯度。解析梯度用总能中心差分验证，Hessian 的 D3 项用 D3 梯度独立差分，误差与步长留档 |
| P0 TD 模型 | CLI `tda: true` 固定，Gaussian `TD(NStates=5)` 参考是完整响应。选项相同的 TDDFT 才能验收 | CLI 暴露 TDA/完整线性响应、根数与自旋，JSON 写实际方法。首轮可对 Gaussian TDA，完整响应另做对照，比较激发能与振子强度 |
| P0 可观测与恢复 | 优化全程无步级输出，误差返回让历史与几何丢失。CLI 先跑一次基态 SCF，随后优化重做初点；成功报告的 `scf_iterations` 仍是初始值。试探点的 SCF 或基组错误在 `if let Ok` 内被吞掉 | 每步记录梯度与 SCF 次数、线搜索拒绝原因、耗时。每个接受步原子写入检查点，失败结构与收敛结构显式区分。保留返回码。优化阈值、迭代预算及剩余资源可设，验证中断恢复 |
| P1 优化算法 | 笛卡尔 BFGS，默认力 max 10⁻⁴、RMS 5×10⁻⁵ Eh/bohr。每步最多 20 次 Armijo 回溯，可能另有一次救援。试探几何从新 SCF 开始，接受后为了梯度再做 SCF。任务没有日志能确定实际步数、失败数或梯度单价 | 先用固定几何单点、梯度及 2–3 步短优化逐项计时。接受点复用 SCF；相邻几何的轨道用新重叠矩阵投影并验证正交与电子数，再用作初猜。比较信赖域/内坐标与旧 BFGS，按同一终态力与能量验收 |
| P1 资源与结果 | 初次 260 AO 的 stored-ERI 保守估计 9,897,823,920 字节，预算 4,294,967,296 字节。保守拒绝合理，错误类别被标为 UnsupportedBasis。新版已有 SAD、收敛时自适应阻尼、增量 Fock、Vxc 批处理、RI-J 分段计时和可选 GPU 三中心积分。标定 wrapper 没传 `--gpu` | 自动选路受实际整作业内存预算限制，显式 `--stored` 给资源错误。先用 `CALCHEM_PROFILE=1` 量现有路径，记录 GPU 是否调用及回退，避免重做旧路线图已落地的优化。JSON 补 grid、RI、GPU、占据及轨道能级 |
| P2 大体系频率 | 中心差分步长 0.001 bohr，库收紧位移点收敛阈值。24 原子需 144 个位移梯度加 1 个中心梯度。94 原子对应 564+1。112 原子对应 672+1。外层 Rayon 并行会与内层 SCF 共享资源，还会同时持有多个工作区 | (h)、(2h) 及加密网格小体系验证，并测实际位移点开销、最低模稳定性和并发内存。按位移保存可恢复结果，最后组装并验证 Hessian。半数值法可用但代价需实测。先拒绝将优化总耗时反推频率时长 |

Gaussian 在输出中明确向 Hessian 加入 D3BJ 二阶导数，可直接验证 CalChem 频率缺项影响的方向。Python 的 CalChem 优化绑定已经把 dispersion 传入自己的 `OptOptions`，该接线缺陷具体落在当前 Rust CLI。应在实际入口上核对。 [CLI 模型传递](E:/YDW/Trae_Gaussian/crates/calchem-cli/src/main.rs:625) · [优化器搜索](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/lib.rs:476) · [Python 优化绑定](E:/YDW/Trae_Gaussian/crates/calchem-py/src/lib.rs:891) · [频率差分](E:/YDW/Trae_Gaussian/crates/calchem-vib/src/hessian.rs:140) · [Gaussian D3 Hessian 日志](D:/YDW/Calc/qc_benchmark_runs/pilot/gaussian/optfreq.log:4879)

## REST 能力与边界

小体系的单点使用 216 AO、826 个辅助函数，in-core RI-J/K 估计 220.52 MB，正常收敛。C1 是原工作队列的另一个任务，def2-SVP、1000 AO、5124 个辅助函数，日志记录预算可用 9462.40 MB、需要 29319.76 MB，并改走 direct RI-J/K。C1 无收敛产物，不能用咖啡因时间外推。REST 使用 libcint、libxc、BLAS 及 RI-JK，当前优化输出显示 geomeTRIC 的信赖域、拒步和尺度信息，可借鉴的是复用、算法和可观测性。

频率日志第 10 行进了频率入口，并明确写 `Number of displaced SCF jobs: 144`，72 是 Hessian 维度。累计 force 计时 1241.525 秒，不能代替整段约 2655 秒。LAPACK `DSYEVD` 的正 INFO 表示特征求解失败，没有专指 NaN。对应源码对称化后进行质量加权并调用 `DSYEVD('V')`，应检查位移梯度有限性、质量、对称误差、矩阵范围及 BLAS/LAPACK ABI，先重现小规模、保存 72×72 矩阵再独立求本征残差。已运行的失败算例在输入几何上，修好后也应从优化驻点重新做参考。

激发态旧控制文件只写 `job_type="tddft"`，落入默认单点。当前本地源码支持 `[ctrl] job_type="energy"` 加独立 `[tddft]` 段，后者可设置 `tddft_method="lr"` 或 `"tda"`、`tddft_spin="singlet"`、`nroots=5`。已安装的 WSL 二进制要用小分子复核，其正确工作时主任务仍可能写 `Single-point energy`。成功条件是返回完整激发能、振子强度和响应收敛输出，结合退出码判定。 [C1 REST 日志](E:/YDW/Chemdraws/zmy_3/qc_benchmark_suite/runs_ac/C1/rest/run.log:142) · [REST 频率日志](D:/YDW/Calc/qc_benchmark_runs/pilot/rest/freq_probe.log) · [REST TD 控制参数](E:/YDW/REST/knowledge_base/rest_source/src/ctrl_io/tddft_parameters.rs:103) · [LAPACK DSYEVD](https://www.netlib.org/lapack/explore-html/d8/d30/group__heevd_ga25b71a69f9921df0a6050aa5883f54f4.html)

## 后续执行顺序

1. **队列恢复和防重跑。** 从现有磁盘状态备份并校验，恢复守护后核对任务实际状态及分组。159 作为中断历史处理，不将其重新开始当作续跑。把单点、优化、频率、激发态拆为独立任务和输出目录。每段检查退出码与物理产物，避免 wrapper 成功掩盖崩溃。旧 `qc_ac` 的错误参数任务单独保留。验收守护重启恢复与跨会话存活时用空队列或独立组验证。活跃数据、scratch 和检查点放 D 盘。
2. **短诊断与能量基准。** 保留 ORCA 的 160/161 因子对照，加 Gaussian 5D 参考。CalChem 固定咖啡因坐标、260 AO、VWN3、D3BJ、`--direct`，先做 Coarse/Fine × RI-J/四中心 J 共四个单点。对可承受的密格点做一个精确 J 锚点，再决定是否增加 UltraFine。分别记录 ΔRI 和 Δgrid 及交互项。对 0.1 kcal/mol 的阈值，对应约 1.59×10⁻⁴ Eh。相近掺杂、卤代或构象间的相对能也应独立验证。用同机 8 CPU 线程、实际内存预算、显式格点/SCF 残差和可比结果重复计时，GPU 另测。
3. **CalChem 先修正确性与出结果能力。** 对上表 P0 定向修复和测试，先统一 D3BJ 能量与导数、TD 形式、最终结果字段与检查点。固定几何验证能量与梯度差分后，才短跑咖啡因优化看实际耗时和拒步次数。以前准备的 `calchem_v2` 解决了几何传递的一部分，仍会碰到色散接线、TD 形式、失败产物及同名网格不等价。
4. **优化性能再扩样。** 用真实阶段计时决定优先优化每步 SCF、积分导数、Vxc 梯度或回溯次数。相邻几何初猜复用、接受点波函数复用、稳健的信赖半径优先于追加大批作业。先过咖啡因模型和末态梯度核验，再对一例 94 原子跑单点。频率与激发态各自按精度和资源单独放行。以共同几何 TD 与单点基准推进时，不必等待某个引擎的长频率任务成功。

## 性能和成熟度判断

CalChem 的独立 Rust 单点核心可用，色散单点计算、RI-J、精确交换和 GPU 可选路径均有落地证据。咖啡因 29 秒证明成功执行了这一配置。当前基准尚不能证明在 8 线程、相同精度下领先。早期 STO-3G 的 PTQ10、BTP-4F 已有可重复入口和优化实测，但 Gaussian 的计算口径仍有差异。小分子 benchmark 的部分单点更快，乙烯优化和乙炔频率则偏慢。版本、任务种类、耗时定义及物理条件都不宜混用。

对于本次咖啡因标定，Gaussian 已提供完整参照。ORCA 能完成单点、优化和频率，当前泛函/基组口径与软模需复查。REST 单点与优化已出结果，频率成功进入模块但崩溃，激发态缺真实探针产物。CalChem 单点具备直接数值取证，优化、频率、激发态仍有合约与稳定性缺口。94 原子及更大批次没有形成同级别四方验收。

本轮检查的 CLI SHA256 为 `952BC793E6F1AC5EEC0A75983ECBD6284E19C9B00C962E9EF77D75312EC6A862`，仓库 HEAD `9f57bd7`。此前 11 月 18 日性能报告使用不同二进制。标定脚本未保存各作业当时的二进制哈希，本轮源码行为依据当前检出版本，不能代替对旧进程内部变化的观测。 [旧性能报告](E:/YDW/Trae_Gaussian/docs/audits/2026-11-18_same_condition_analysis/FINDINGS.md) · [验收快照](SNAPSHOT.json)
