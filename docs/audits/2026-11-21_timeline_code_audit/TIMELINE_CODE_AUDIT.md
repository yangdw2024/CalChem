# CalChem 全项目时间线代码审核报告

审核日期：2026-11-21（Asia/Shanghai）
工程：E:\YDW\Trae_Gaussian
审核方式：只读。完整 git 历史 + 早期文档 + 六轮审计记录 + 核心 crate 源码抽查 + benchmarks 原始数据核对。未修改任何业务源码。
证据等级标注：confirmed（有源码/原始日志/实测数据直接支撑）、inferred（由证据合理推导）、unknown（证据不足）。

---

## 0. 先纠正上一轮（11/20）审核的两处过时结论

本次按完整时间线重核，发现上一轮有两处判断已被后续事实推翻，必须更正：

1. **“87 个文件、+7220 行未提交，是最大风险”——已不成立。**
   11/19 已有提交 9f57bd7（feat(release): v0.2.2），一次性提交 838 个文件、+496257 行。当前 `git status` 仅剩 1 个未跟踪目录 `benchmarks/`，工作区其余全部干净，且 HEAD 与 origin/master 一致（已推送）。confirmed。

2. **“GPU 快 2.5~4 倍、溶剂化快 13~20 倍均无落盘实测”——部分错误。**
   上一轮未翻到未跟踪的 `benchmarks/` 目录。该目录含 11/19 07:00 完成的 MASSIVE350 对比评测，有 16 个任务的 CalChem 与 Gaussian 16 原始输出（input.out）、result.json 和 run_pair.py。单点能 3.65x/3.83x、自由基 9.88x/4.11x、溶剂化 13.39x/20.05x 均有实测 wall_time 支撑。这些加速倍数现在应判为 confirmed（限于所测小体系）。confirmed。

但本次也发现该评测报告本身存在选择性呈现与精度口径问题（见第 4 节 P1-1/P1-2/P1-3），不能因“有实测”就全盘接受其结论。

---

## 1. 时间线总览

| 阶段 | 时间 | 关键事件 | 证据 |
|---|---|---|---|
| 一、Python 创世纪 | 2026-07 初 ~ 10 月中 | 以 Python 包起步，愿景宏大：GPU 全链路、OPV、GROMACS 对接、Post-HF、Berny、MD、GUI | 文件时间戳 7/5~7/10；README.md；calchem/ 包结构 |
| 二、仓库化与后端抽象 | 10/28 ~ 10/29 | git 初始化；Phase A~I（后端抽象、Gaussian/ORCA 适配、TaskService、CLI、GUI、OPV 工作流、PyInstaller）；固定数值基线；v0.2~v0.6 | git b8397eb~30dc508；docs/BASELINE.md |
| 三、Rust 转向 | 10/30 | 决定重写为纯 Rust；Stage 0~2（workspace、RHF s 型积分、DIIS、Obara-Saika p 型）；归档旧 handoff | git 4da99f0~c9fd62e；_archive/CalChem_可执行重构方案_2026-10-29.md |
| 四、独立审计与去伪 | 11/3 ~ 11/12 | 11/3 首审；11/9 redteam；11/10 P0/P1/round2 + 发布 v0.2.0 纯 Rust + 写边界文档；11/11 独立审计（22 项发现，撤回大体系认证）；11/12 成熟度/贴文复核 | docs/audits/2026-11-03、09_redteam、10_*、11_independent、12_*；git 2724136 |
| 五、精度整改与外部对标 | 11/16 ~ 11/17 | 11/16 认证复核→精度整改（P1 RI-J 逆、P2 网格、P3 脚本）→最终认证→交付复核（G16 对照、有机 RI-J 基准）；11/17 加 def2-jfit、wB97X-D/SMD 原生跑通、v3 全量回归 143+5+120、SHA256 快照 | docs/audits/2026-11-16_*；pytest_final_v3.log；cargo_final_v3.log |
| 六、v0.2.2 发布与实测 | 11/19 | 提交 9f57bd7（838 文件）；统一 CLI、GPU D 轨道 ERI3C、多任务工作流、PyPI/npm 同步；MASSIVE350 对 G16 十六项实测 | git 9f57bd7；benchmarks/ |
| 七、本次审核 | 11/21 | 全时间线严格代码审核 | 本报告 |

提交节奏特征（confirmed）：10/30→11/10 间隔 11 天无提交，11/10→11/19 间隔 9 天无提交，随后均为大批量提交（11/19 单次 838 文件）。属于“长周期积累 + 大爆炸提交”模式，利于里程碑快照，不利于二分定位与逐次评审（见 P2-3）。

---

## 2. 分阶段严格审核

### 阶段一：Python 创世纪（7 月 ~ 10 月中）

做了什么：搭起一个功能面极广的 Python 量化包，calchem/ 下含 scf、integrals（direct_scf/ri/screening）、post_hf（mp2/mp4/ccsd/ccsd_t/ccd/casscf）、opt（berny/bfgs/irc/ts_search）、solvent（pcm/smd）、md、oniom、props、gui 等。README 的路线图 M1~M5 与项目结构即此时愿景。

严格审核发现：
- **Berny 优化器是空壳（confirmed，P2-1）。** calchem/opt/berny.py 的 step() 实为 `step = -0.1 * grad / (norm(grad)+1e-8)`，是固定步长最速下降，与 Berny 冗余内坐标算法毫无关系，仅借用了名字。
- **Post-HF 有实现但未回归验证（inferred）。** mp2.py/ccsd.py/casscf.py 等体量真实（4~19 KB），但 docs/BASELINE.md 明确把 MP2/CCSD(T)/MP4/CASSCF/PBE/B3LYP 参考值列为“待回归验证”。即“写了”不等于“验过”。
- **README 严重超前于实现（confirmed，P1-4）。** “GPU 全链路加速”“def2-TZVP+SMD 默认”“GROMACS 一键对接”“MP2/CCSD(T)”“Berny”等描述，与后续 Rust 引擎的真实能力不符，属愿景文案而非现状。

阶段评价： ambition 很大，但存在“名实不符”的成分。这为 11/11 独立审计集中暴露问题埋下伏笔。

### 阶段二：仓库化与后端抽象（10/28~10/29）

做了什么：git 化；一天内推进 Phase A~I，建立 CalculationRequest→后端 dispatch 的抽象，接 Gaussian 16（已验证）与 ORCA 适配，TaskService+JobQueue+状态机，Click+Rich CLI，PyQt6 GUI，OPV 工作流，PyInstaller 打包。10/28 固定数值基线（H2/He/Ne/H2O RHF/STO-3G 实测对参考值 1e-8 通过）。

严格审核发现：
- 提交 89aa81d 标题“no fake values, real dispatcher, clean CLI JSON”反向证明此前存在假值/假 dispatch，本阶段在做去伪。confirmed（由提交信息推断历史）。
- BASELINE.md 主动溯源并修正了旧 README 中错误的 H2O 参考值（-74.9629→-74.6953，Szabo 几何），体现了早期就有的数值诚实意识。confirmed。

阶段评价：工程骨架快速成型，且已开始自觉清理假值。

### 阶段三：Rust 转向（10/30）

做了什么：Stage 0 workspace 编译通过并保留 Python 基线；Stage 1 RHF s 型积分验证；Stage 1.5 DIIS + 双轨 native 后端；Stage 2 正确 Obara-Saika VRR、p 型积分。归档旧 handoff。

严格审核发现：
- 转向决策有文档（_archive/CalChem_可执行重构方案_2026-10-29.md），且“保留 Python 基线”说明不是推倒重来，而是以可验证基线约束重写。confirmed。
- 此后形成“双引擎并存”格局：legacy Python 全功能（含部分空壳/未验证）+ 新 Rust 干净但面窄。仓库未清晰声明二者关系与默认路径（P2-1）。

阶段评价：这是项目质量的分水岭。从“广而部分虚”转向“窄而可验证”。

### 阶段四：独立审计与去伪（11/3~11/12）——项目最有价值的阶段

11/11 独立审计（FINAL_AUDIT_REPORT.md）是整個项目的质量基石，confirmed：
- 给出 22 项发现 F01~F22，含 P0 级：CUDA FFI 未校验长度/usize→i32 缩窄可越界（F01）；SCF 最终只查 commutator 是否有限、有限大残差也返回 converged（F02）。
- 揭穿假方法：wB97X-D 实为不完整三项交换多项式+PBE 相关，缺标准 B97 自旋相关/衰减交换/原始色散（F03）；SMD Python facade 实为 C-PCM+表面张力×面积（F14）。处置是“原生入口明确拒绝，不再伪装支持”。
- 内存准入 bug（F07/F22）、RI-J 谎称 Cholesky 实为特征分解+固定阈值（F09）、自动辅助基指数裁剪过粗（F10）、Davidson 子空间停滞仍返回成功（F15/F21）等，均逐项修复并写明验收边界。
- **主动撤回“16 GB/2000 原子已认证”**，把历史性能 CSV 全部降级为 UNVERIFIED_HISTORICAL（F20），并明确“不据此判断主观造假”。
- 评级：工程质量 C、计算物理完整性 C、大体系/工业生产准入 D。

11/12 成熟度复核进一步确认：GPU 仅 J 缩并上链，无大分子端到端加速证明。

阶段评价：这一阶段确立了项目最稀缺的资产——诚实的审计文化。敢于自我否定、把营销数字降级、把假方法改成显式拒绝。这比任何功能都更能说明工程成熟度。

### 阶段五：精度整改与外部对标（11/16~11/17）

11/16 认证复核（FINAL_REVIEW.md）又抓出真问题，confirmed：
- 网格不对齐：Gaussian 用 UltraFine，而 CalChem `GridLevel::from_name("ultrafine")` 实际映射到 Fine（映射 bug），两边并非同网格。
- B3LYP/VWN 注释写反（B3LYP 用 VWN_RPA，B3LYP5 才用 VWN5）。
- auto-jfit RI-J 在 H2O/6-31G(d) 偏差 -8.1 mEh（约 5.08 kcal/mol），定为定量使用阻断项。
- GPU 端到端无提速（CPU 126.80s vs GPU 127.65s，0.993x）。

随后精度整改（01_PLAN_STATUS.md）confirmed：
- P1：auto-jfit 指数比改 2.2、扩角动量覆盖，H2O/B3LYP/6-31G(d) 偏差降到 -1.24e-5 Eh（<0.1 mEh），147 辅助函数、截断模 0，三次独立运行一致。
- P2：加入 Lebedev N590 与真正 UltraFine（200×590 流式）；但双侧 PBE0 网格收敛扫描当时未执行。
- P3：复现脚本逐步检查 $LASTEXITCODE，修正 CSV 计数与 VWN 注释。

11/17 进一步（confirmed）：新增 def2-universal-jfit 辅助基；wB97X-D 与 SMD 原生跑通（test_supported_range_separation_runs_natively[wb97x-d]、test_cam_b3lyp_and_wb97xd_single_point_rust、SMD CDS 实现均 PASSED）；v3 全量回归 143 Rust + 5 CUDA + 120 Python 全过；生成全仓 SHA256 快照，binary_match=true。

阶段评价：从“小体系 RHF 对齐”推进到“DFT/Range-separated/溶剂化原生 + 外部 G16 逐项对照”，是实打实的科研精度进步。RHF 最大总能差 5.53e-8 Eh、B3LYP/CAM 达 1e-9 Eh（最终认证报告）。

### 阶段六：v0.2.2 发布与 MASSIVE350 实测（11/19）

做了什么：提交 9f57bd7（838 文件），统一工业 CLI、GPU D 轨道 ERI3C、多任务工作流、PyPI/npm 同步；benchmarks/ 下完成 MASSIVE350 十六项对 G16 实测（同等 4 线程、Pueue 并行度 2）。

严格审核发现（详见第 4 节）：加速数据真实，但报告选择性呈现、溶剂化精度存疑、NH3“极小值”含虚频、内存结论仅覆盖 ≤6 原子小分子。

阶段评价：有了第一份可复核的横向性能实测，是重大进步；但报告的客观性需要再打磨，否则会把项目辛苦建立的诚实文化打折。

---

## 3. 代码质量硬指标（confirmed）

- **15 个 Rust crate 生产代码：panic! = 0，todo!/unimplemented! = 0。** 全部为零，无未完成桩。
- unwrap() 分布：calchem-int 38、calchem-dft 19、calchem-model 16、calchem-scf 4、calchem-cuda 2、calchem-tensor 2，其余 crate 为 0。抽查 calchem-scf 的 4 处全在内联测试模块（diis.rs 磁盘故障注入测试，481~489 行）；eri.rs 生产路径 0 处。判断：unwrap 主要位于内联测试或不可失败操作，生产热路径无裸 unwrap 风险。inferred（基于抽查，未逐行穷尽）。
- 测试规模：143 Rust + 5 CUDA + 120 Python，11/17 v3 全过，退出码均 0。
- 外部数值对照：RHF 最大总能差 5.53e-8 Eh、梯度 1.51e-8 Eh/Bohr；B3LYP/CAM-B3LYP 1e-9 Eh 量级（最终认证）。
- 制品一致性：_core.pyd 与 _core.dll SHA256 一致，快照清单 binary_match=true。
- 工作区：仅 benchmarks/ 未跟踪，其余干净且已推送 origin。

结论：Rust 核心的工程质量是真实过硬的，不是文档堆出来的。

---

## 4. 问题清单（分级，带证据）

### P1（影响科学结论或对外口径，应优先处理）

**P1-1 溶剂化正确性存疑（最高优先科学问题）。**
benchmarks/task_05_solv_acetic_acid/result.json：CalChem 总能 -229.0513 Eh，G16 -229.0650 Eh，差 13.67 mEh = 8.58 kcal/mol；CalChem solv_energy_kcal_mol = +4.744（正值）。乙酸在水中的实验水合自由能约为 -6.7 kcal/mol（负值），SMD 通常也给出负值。+4.74 kcal/mol 的符号与量级均可疑；8.58 kcal/mol 的引擎间差异远超化学精度。task_06 乙醇差 14.75 mEh（9.26 kcal/mol）类似。报告却标“✅ PASS”并称“数值精度高度一致”。
判定：溶剂化是项目的招牌功能（20x 加速），但其正确性尚未被这份基准证明，反而暴露疑点。confirmed（数据本身）+ unknown（符号错误的根因，需实跑核验）。
建议：用同一几何/基组/溶剂，单独核对 CalChem SMD 的静电项与 CDS 项符号和量级，并与 G16 SMD 的 ΔG_solv 直接对比（而非总能差）。

**P1-2 性能报告选择性呈现。**
MASSIVE350 报告“结论”只强调 3.6x~20x，但同一张表里：opt_ethylene 0.47x、freq_acetylene 0.16x、thermo_acetylene 0.16x、freq_ammonia 0.80x、thermo_ammonia 0.67x 均慢于 G16。即“单点/溶剂化快、优化/频率/热力学慢”。“计算速度优势显著”“全面超越”式表述以偏概全。confirmed。
建议：报告应如实分场景给出加速/减速，慢项正是 DLC 与 GPU 梯度的攻关依据。

**P1-3 NH3“极小值”任务含虚频。**
task_11_freq_ammonia：CalChem 频率首项 -1055.6 cm^-1，G16 -1050.2 cm^-1，均为负（虚频）。任务类别为 11_freq_min（极小值），但虚频意味着该几何不在真正的极小值（疑似近平面 D3h 伞形反转鞍点）。报告却称“严格识别简并模式”而忽略虚频。confirmed。
建议：核查 NH3 优化是否收敛到金字塔极小值；freq_min 基准应保证无虚频。

**P1-4 文档与代码漂移。**
README.md 仍描述 7 月 Python 愿景（GPU 全链路、def2-TZVP+SMD 默认、Berny、MP2/CCSD(T)、GROMACS 对接），与 Rust 现状不符；docs/NATIVE_METHOD_BOUNDARIES.md（11/10）仍写“CAM-B3LYP 与 wB97X(-D) 不可用”“标准 SMD 不提供”，但 11/17 已原生实现并测试通过。边界文档滞后于代码。confirmed。
建议：更新 README 与边界文档至 v0.2.2 真实能力，避免对外口径失真。

### P2（工程与遗留项，应排期处理）

**P2-1 双引擎并存且 legacy 含空壳。**
legacy Python 的 Berny 是固定步长最速下降空壳；post_hf（MP2/CCSD/CASSCF）有代码但 BASELINE 标“待回归”。Rust 引擎不含这些。仓库未清晰声明默认路径与 legacy 状态。confirmed（Berny）+ inferred（post_hf 未验证）。
建议：明确标注 legacy Python 为历史参考实现，Berny 空壳要么补真实现要么改名/移除，避免误导。

**P2-2 RI-J 有机体系精度仍未闭合（沿用上轮，依然有效）。**
TASK3_ORGANIC_RIJ_BENCHMARK.csv（11/17）：CH3CN -1.42e-4、C2H3F -1.23e-4、C4H4S -3.16e-4 Eh，IC-2F/STO-3G -7.30e-3 Eh，均超 1e-4 验收线。def2-jfit 已加入但未用它重跑这组对照。confirmed。
建议：def2-jfit vs auto-jfit 重跑同组有机基准，验收 ≤1e-4 Eh。

**P2-3 大爆炸提交 + benchmarks 未跟踪。**
11 天/9 天无提交后单次 838 文件/+496k 行；benchmarks/（含关键实测证据）至今未纳入版本控制。confirmed。
建议：把 benchmarks/ 提交入库；后续按功能小步提交，便于二分与评审。

**P2-4 大体系内存仍未实测。**
MASSIVE350 仅 ≤6 原子；task4（50~100 原子 RSS/VRAM）脚本存在但未运行；726~2000 原子 16 GB 上限仍是 O(N²) 模型推演。“内存≤2.5GB/大体系绝不爆”无大体系证据。confirmed。
建议：跑 task4，落盘峰值 RSS/VRAM，再谈大体系内存。

**P2-5 溶剂化梯度为全差分（性能瓶颈，上轮已指出）。**
calchem-opt/src/lib.rs：SCRF 面用 6N 次完全重收敛 SCF 做中心差分；边界文档亦承认无解析溶剂梯度。溶剂中优化/频率极贵。confirmed。

### 已闭环 / 不构成问题（避免重复审计）

- 11/11 的 F01~F22 已逐项修复并写明边界；SCF 假收敛、CUDA FFI 越界、假 wB97X-D/SMD 等均已处置。
- 11/16 的网格映射、VWN 注释、RI-J 逆病态、复现脚本退出码已整改。
- 上轮“87 文件未提交”已由 11/19 提交解决。
- CPU 积分已支持 S~F 壳层、RI-J 解析库仑梯度已实现并经有限差分验证——外部评价所称“需升级 f 积分/需实现 RI-J 梯度”前提不成立（上轮已核，本轮维持）。

---

## 5. 总体评价与规划建议

### 总体评价

这确实是你目前最成功的项目，判断依据不是功能清单，而是三点硬事实：
1. Rust 核心零 panic/零桩、143+5+120 测试全过、与 G16 小体系对齐到 1e-8~1e-9 Eh——工程质量真实过硬。
2. 11/11 独立审计敢自我否定 22 项、撤回大体系认证、把假方法改成显式拒绝——这种诚实文化在同类项目里极稀缺，是项目最值钱的资产。
3. 从 Python“广而部分虚”到 Rust“窄而可验证”再到外部逐项对标，是一条清晰的成熟度上升曲线。

但“Production-Ready 1.0 / 特定赛道全面超越”仍说过头。准确定位是：**研究开发级引擎，若干小体系基态方法已获独立外部基准支持，单点/溶剂化有实测速度优势，但优化/频率偏慢、RI-J 有机精度未闭合、溶剂化正确性存疑、大体系内存未验证。** 这与项目自己 11/16 认证的定位一致，不应被外部评价或 MASSIVE350 报告的选择性表述抬高。

### 规划建议（按优先级，已结合本轮新证据调整）

1. **入库 + 文档对齐（半天，纯整理）。** 提交 benchmarks/；更新 README 与 NATIVE_METHOD_BOUNDARIES.md 到 v0.2.2 真实能力。收益：保住实测证据、消除对外口径失真。
2. **核验溶剂化正确性（最高科学优先）。** 查清乙酸/水 +4.74 kcal/mol 的符号与 8.58 kcal/mol 引擎差。溶剂化是招牌功能，正确性不成立则 20x 加速无意义。
3. **闭合 RI-J 有机精度。** def2-jfit 重跑 TASK3 同组基准，验收 ≤1e-4 Eh。决定 OPV 筛选卖点是否成立。
4. **修正基准报告客观性 + NH3 极小值几何。** 如实分场景报告快/慢；保证 freq_min 无虚频。
5. **DLC 内坐标优化器。** 直接针对实测的 opt/freq 偏慢（0.16x~0.47x），是外部评价说对的那条。
6. **task4 大体系内存实测。** 把“内存受控”从模型变成数据。
7. **基组数据扩展（def2-SVP/TZVP JSON）+ 球谐 5d/7f、ADIIS/展宽、GPU 梯度与交换 K、SCRF 解析梯度。** 按序推进。
8. **RI-MP2 暂缓，DLPNO-CCSD(T) 不列入。** 对 OPV 单点筛选 DFT+D3 是主力；legacy Python 的 post_hf 可作参考但须先回归验证。

### 一句话结论

项目底子很好、文化更好，值得继续投入；但当前要做的不是按外部评价冲“工业级”，而是先把溶剂化正确性、RI-J 有机精度、文档口径这三件“里子”补实，再谈 DLC 与性能扩展。把诚实文化守住，这个项目才立得住。

---

## 6. 证据索引

- git 历史：git log（24 提交，10/28~11/19）；9f57bd7（838 文件）；当前 status 仅 benchmarks/ 未跟踪。
- 早期：README.md；docs/BASELINE.md；calchem/opt/berny.py；calchem/post_hf/*；_archive/CalChem_可执行重构方案_2026-10-29.md。
- 审计：docs/audits/2026-11-11_independent/FINAL_AUDIT_REPORT.md（F01~F22）；2026-11-16_certification_review/FINAL_REVIEW.md；2026-11-16_precision_repair/01_PLAN_STATUS.md；2026-11-16_final_certification/FINAL_SCIENTIFIC_CERTIFICATION.md。
- 回归：docs/audits/2026-11-16_delivery_check/{cargo_final_v3.log, pytest_final_v3.log, cuda_final_v3.log, SNAPSHOT_MANIFEST_SHA256.json}。
- 实测：benchmarks/MASSIVE350_CALCHEM_VS_GAUSSIAN16_REPORT.md；benchmarks/massive350_benchmark_results.json；benchmarks/massive350_runs/task_05/08/11/result.json。
- 有机 RI-J：docs/audits/2026-11-16_delivery_check/TASK3_ORGANIC_RIJ_BENCHMARK.csv。
- 源码抽查：crates/calchem-scf/src/diis.rs（unwrap 在内联测试）；crates/calchem-int/src/eri.rs（S~F，0 unwrap）；crates/calchem-opt/src/lib.rs（笛卡尔优化、SCRF 全差分）；crates/calchem-solv/src/{smd.rs, solvent.rs}（SMD 原生）。