# CalChem 独立物理与工程审计报告

审计日期：2026-11-11（Asia/Shanghai）。工程：`E:\YDW\Trae_Gaussian`。

## 审计结论与评级

**不通过原交付报告所称的“国际顶尖、工业级、200–2000 原子已认证”验收。** 当前适宜定位为有真实积分、自洽场、梯度和响应算法的研究开发引擎，不能替代经过独立方法认证的生产量化软件。修复后的测试绿色也不构成全部电子结构理论或全部元素/基组认证。

本次独立评级采用 A（生产验证充分）、B（有边界的可靠使用）、C（研究开发、需独立交叉验证）、D（该项验收未通过）：工程质量 **C**，计算物理完整性 **C**，本报告所述大体系/工业生产准入 **D**。这些是基于现有证据的审计判断，不是行业认证分数。小体系可验证功能有实际开发价值；大体系资源管理、方法完整性及外部精度基准仍是阻断项。

本次没有发现核心 SCF 直接按分子名称返回报告能量的实现。发现的是实质性公式/边界错误、测试未覆盖真实分支、用近似模型冒充标准方法、以及缺少可追溯原始测量的交付主张。不能据此推断开发者主观意图。

## 范围、初始状态与工作流

先完整阅读五项 master_mission 交付资产和任务跟踪表，再沿以下调用链审计：

`CalculationRequest → NativeCalchem → PyO3 resolve_scf_kind/run_single_point → RHF/RKS/UHF/UKS → FockBuilder(Stored/Direct/RI-J) → ERI/Schwarz/辅助度量 → XC 网格 → DIIS/最终驻点检查 → 梯度/TDDFT/溶剂`。

独立核验重点覆盖用户指定路径。14 个 workspace crates 的全量回归用于检测回归，不表示逐行证明了整个仓库的每一种后 HF 方法、全部数据表、所有异常驱动环境或全部数值参数。

工作区开始时已有大量 tracked 修改和 untracked 源文件；没有重置、清理、提交或覆盖基线。`01_PLAN_STATUS_RULES.md` 记录规则；`backup/` 保存本次修改前的源码、脚本、测试与交付资产；`evidence/initial-git-status.txt` 和 `preexisting-tracked.diff` 保存起点。新文件和修改的精确行号、差异及 SHA-256 另见变更清单。

## 问题、修复与残余边界

| ID | 等级 | 发现与证据 | 本次处置 | 验收边界 |
|---|---|---|---|---|
| F01 | P0 | CUDA 安全 Rust API 未检查 B/x/J 长度，usize 直接缩窄为 i32；可向 FFI 提供越界裸指针 | 长度、乘法溢出、零维和 i32 范围在 FFI 前验证 | 恶意维度返回 false，J 不改变；非穷尽硬件故障验证 |
| F02 | P0 | RHF/RKS/UHF/UKS 最终只检查 commutator 是否有限；有限大残差也可返回 converged | 最终强制满足阈值；polishing 不达标返回 NotConverged；纯 canonicalization 后再验证 | 仍是显式容差策略，不是自动鉴别格点噪声 |
| F03 | P0 | wB97X-D 实际为不完整三项交换多项式 + PBE 相关，缺少标准 B97 自旋相关、衰减交换与原始色散；wB97X 被错误当成同一模型 | 原生入口拒绝该 legacy 模型，不再列作支持方法 | **未实现标准 wB97X-D**；更新测试验证正确拒绝，并未伪装实现 |
| F04 | P1 | CAM-B3LYP 半局域交换使用 LDA k_F，漏掉 B88 增强因子对衰减与梯度的作用 | 改用 ITYH k_GGA，补齐 rho/gamma 链式偏导；改进大 a 稳定展开 | 独立高精度点值、偏导与现有核梯度差分验证；未完成外部分子级认证 |
| F05 | P1 | erf ERI 的 Some(0) 返回完整库仑积分；巨大 omega 平方可溢出 | omega=0 正确返回零算符；hypot 计算 theta；负值/非有限值明确拒绝 | 低层非法参数为断言；PyO3 标准方法不接受任意 omega |
| F06 | P1 | Direct 中预先按与密度无关阈值丢弃 shell pairs，后续统计未涵盖这些丢项；交换筛选界需更保守 | 引擎保留 shell pairs，按密度筛选完整 quartet；交换使用绝对系数与保守简并因子；去掉未计账的 1e-16 积分硬截断 | 筛选界仍需与所需能量/梯度误差联合控制，逐项阈值不是总能误差认证 |
| F07 | P1 | 大体系内存准入在总估算超限但单线程估算可容纳时仍放行，实际没有降低线程；原预算遗漏较多矩阵 | 超限明确拒绝；扩大状态、DIIS、scratch 的保守估算；分开总内存预算和 Direct 工作缓冲预算 | **不是 OS 内存硬锁**；格点和 RI 度量也不是磁盘流式，拒绝原 16 GB 认证 |
| F08 | P1 | Vxc 对每个 grid chunk 分配 N² 矩阵并用 Rayon reduction，临时矩阵数由任务树决定 | 限定最多 K 个大块结果矩阵，串行复用首块归并；能量使用补偿求和 | 格点坐标仍整体 Vec 驻留；没有实现全程流式格点 |
| F09 | P1 | RI-J 原报告称 Cholesky，实际是特征分解+固定绝对阈值；还有无用大矩阵乘法 | 对角归一化度量+相对谱截断，拒绝显著负谱，取消无用乘法 | 返回截断的广义逆；不是 Cholesky 求解器，误差依赖辅助基组与秩截断 |
| F10 | P1 | 自动辅助基组把指数裁剪到固定范围，指数比 2.7 较粗；LDA RI 测试竟设为 density_fitting=false | 真正打开测试后先复现 1.092e-3 Eh 超差；取消固定指数截断、加密为比值 2.0，保持既定阈值通过 | 自动辅助基仍为经验近似；高角动量/扩散体系没有通用 1e-8 Eh 保证 |
| F11 | P1 | RI 缓存先收集 Vec<Vec> 再拼接，峰值可近两倍缓存；度量分解无单独规模准入 | 单次预分配平坦 B，借用互不重叠行段并行写入；度量工作区准入；尺寸断言 | on-the-fly 分支是 shell block 两遍计算，不是文档声称的 C_aux=16/32 大块 GPU streaming |
| F12 | P1 | RI-J 波函数没有记录拟合来源；解析梯度仍使用精确四中心库仑导数；响应核也使用精确库仑 | 为 Rust/Serde/PyO3 波函数添加 density_fitting 来源字段；四种 RI 梯度和 RI TDDFT 明确拒绝 | **RI-J 解析梯度/一致响应未实现**，需要三中心与度量导数以及一致响应收缩 |
| F13 | P1 | SCRF RKS 缺少 beta K_lr；Direct/RI 标志可能被接受但实际仍分配 stored ERI | 范围分离 SCRF 与 Direct/RI SCRF 显式拒绝 | C-PCM/COSMO 小体系路径仍可使用；没有完整 CAM/SMD 溶液全链路 |
| F14 | P1 | SMD Python facade 直接调用 C-PCM + 表面张力×面积 | SMD.kernel 明确抛出 NotImplementedError；repr 不再触发隐藏计算 | **SMD 未实现**。简单面积项不能标为标准 SMD |
| F15 | P1 | TDA/RPA Davidson 子空间不再增长时，残差不合格也返回成功 | 停滞返回 NotConverged；非有限残差明确拒绝；补充原始 RPA 方程和辛归一化验证 | 小维 dense + iterative 检查不等于全部困难激发态认证 |
| F16 | P1 | CUDA 端未检查每次 memcpy，内核索引含 int 乘积，失败后可能报成功 | size_t 索引、逐步错误检查、RAII 释放、拷回事务提交、分配作用域串行化 | 回退为 CPU 重算；不承诺硬件/驱动卡死可在进程内恢复 |
| F17 | P1 | NVCC 编译失败时可悄悄链接无来源匹配校验的 prebuilt lib | 取消旧 archive 替代，失败只编译 CPU stub；显式 /MD 匹配 Windows CRT | 当前 Windows 构建脚本仍有工具链硬路径，跨平台可移植性有限 |
| F18 | P2 | use_gpu 没有传给 RI 引擎；文档 CALCHEM_DISABLE_GPU 和 GPU 预算开关未落实 | per-engine use_gpu；实现 disable 环境变量与向下收紧预算，保留 5 GiB 硬上限 | 文档“5.0 GB”与源码“5 GiB”并非同一单位；设备 FP64 不意味着逐位等同 CPU |
| F19 | P1 | cargo_test.ps1 无参只调用 cargo，不运行测试；还恢复 C: TEMP；构建 pyd 复制错误有误用旧版本风险 | 无参实际运行全 workspace 测试，保留 E: TEMP，传播退出码；pyd 复制失败中止并校验 DLL/pyd 哈希 | 日志与退出码同时保留，防止绿色误判 |
| F20 | P1 | 交付报告/矩阵/缩放模型与实际不符；CSV 无可验证来源，甚至给 projected 体系能量和精确耗时 | 五份历史报告显式更正，CSV 状态降为 UNVERIFIED_HISTORICAL，备份保留原值 | 不将历史值作为本次实测；不据此判断主观造假 |
| F21 | P1 | 对称特征分解器强制 effective_tol≥1e-11×矩阵范数；Davidson 请求 1e-10 却停在 6.127e-10，旧“子空间耗尽即成功”掩盖了问题 | 将人为精度下限改为机器舍入尺度；同步缩小 Jacobi skip 阈值；新增本征方程独立残差测试 | 未放宽 Davidson 请求容差；TDDFT 4 项专项通过后重新运行全量 |
| F22 | P1 | UHF/UKS Stored 使用完整 N⁴ Tensor4D，却共用更小的 RKS 对称稀疏内存估算；可能错误放行超预算分配 | 增加开壳层 dense stored-ERI 的 N⁴×8 单独准入，保留 Direct 替代路径 | 只验证准入边界，不分配巨型测试张量；总进程内存仍非硬配额 |

## 对五个重点问题的直接答复

### 1. Direct SCF 有界内存是否真实？

局部 Direct Fock 构建使用预分配 K 个闭壳矩阵或 2K 个开壳矩阵，每个显式 Rayon worker task 借用一套缓冲，atomic shell-pair 任务分配不再为每个 quartet 分配 N² Fock。线程调度可迁移任务，所以应称“每个显式 worker task 的缓冲”，而不是 OS 线程唯一池。quartet block、VRR cache、shell density norms 和全局 SCF 状态仍需额外内存。交换 K 的归并已改为复用首个缓冲，取消额外输出矩阵。

但全过程没有统一内存分配器配额。`CALCHEM_MAX_DIRECT_BYTES` 只控制自动 worker 累加器预算，至少一个 worker 的底限、手工 bounded API、多个并行作业和其他模块都不构成全程硬限额。修复后的 `CALCHEM_MAX_MEMORY_BYTES` 是保守准入估算，超限会拒绝；仍不是峰值 RSS 证明。

N=6224 时，一个 dense f64 矩阵为 **309,905,408 bytes = 295.548828 MiB**。16 个闭壳缓冲为 **4.61795 GiB**；4 GiB 默认池最多容纳 **13** 个此类缓冲，不能按“14 个=4.34 GB<4 GiB”放行。6 步 DIIS 保存 6 个 Fock 和 6 个 error，合计 **3.46346 GiB**。主迭代和 polishing 历史可同时存活，再加 S/H/P/C/X、正交化/对角化、FPS/SPF 和重建临时矩阵，不能用单一池数值证明小于 16 GB。

Becke 网格是内存中的 `Vec<GridPoint>`，构建时 per-atom Vec 与最终 Vec 存在过渡峰值；DIIS 历史也是完整内存矩阵。不存在文档所说的格点/DIIS 磁盘分块流式存储。18,000 AO 时仅一个矩阵就约 **2.41399 GiB**，已有求解器需要多个矩阵，不能证明“2000 原子≤8.5 GiB”。

本次没有启动 p4D18 或 2000 原子全 SCF；不以风险性 OOM 试运行替代资源证明。原“16 GB 完全可行”认证撤回。

### 2. erf 两电子积分、CAM 参数和解析梯度

令 p=alpha+beta、q=gamma+delta、rho=pq/(p+q)，theta=omega/sqrt(omega²+rho)。衰减辅助积分使用 `theta^(2m+1) F_m(theta² T)`。现源码通过 Boys 辅助值包含该权重，VRR 中不应再重复乘 theta²。Fock 采用 `alpha K_full + beta K_lr`，其短程权重 alpha、长程 alpha+beta 的组合正确；这与高斯指数的 alpha/beta 是不同符号角色。

本次修复 omega=0、极大 omega 数值边界，并补充 s/p/d/f 角动量及核中心导数恒等式差分交叉验证。`derivatives.rs` 通过升降角动量公式复用同一 EriPairData，没有再次添加 theta²；未发现该链当前仍有报告所说的重复缩放。但既有 H2/6-31G 测试主要是低角动量，不能单独证明全部高角动量正确。

**正确积分并不自动使整个标准泛函正确。** 本次另修 CAM 的 ITYH 半局域项，保护性拒绝不完整的 wB97X-D。CAM 高精度点值来自独立 70 位公式脚本，分子能量仍需同几何、同基组、同格点/约定的外部实现交叉验证。

### 3. RI-J 稳健性、分块与竞争

实际没有 Cholesky，原代码有固定绝对谱阈值保护。现改为归一化度量的相对截断，加入线性相关、基函数尺度改变及不定矩阵测试；这不是自动辅助基组完备性的证明。三中心缓存改为 disjoint mutable slices，Rust 借用规则防止本路径同一区间并行写入；on-the-fly 各 shell-pair block 为局部所有权，最终写入串行进行，未发现有效维度输入下的数据竞争。

辅助度量仍 O(N_aux²) 内存与密集特征分解；不能把“无四中心 ERI 缓存”误写为整体 O(N_aux N_ao) 内存。缓存和两遍 shell-on-the-fly 一致性另有专项测试。

### 4. GPU/CPU 回退

现在 availability、参数/预算失败、CUDA 分配/拷贝/launch/sync 错误会让 GPU API 返回 false；RI 调用方随后完整 CPU 重算。RAII 释放所有成功申请的 device buffers，主机结果只在成功 D2H 后提交；无异步 stream 生命周期外借。新增测试记录实际是否执行 GPU，不能以“可回退测试通过”充作 GPU 内核运行证明。

本机查询可见 RTX 5060 Ti、驱动 591.86、8151 MiB 显存。`evidence/cuda-runtime.log` 实测 **availability=false、kernel executed=false，CPU 回退契约通过**。原 CUDA 精度测试在不可用时直接 return，测试框架依然显示 ok，不能把它计作 GPU 精度认证。有设备不代表 CUDA runtime/driver 组合兼容。无法承诺“任何损坏驱动、设备丢失或 OS 故障下 100% 静默恢复”，源码也未提供隔离子进程/超时杀死机制。

### 5. 平滑器、交换子残差和“格点噪声”

固定数值格点定义确定的离散能量与势，连续积分误差不是允许 SCF 大残差的理由。原来的最终检查漏掉阈值，确实会接受不合格固定点；现必须满足 `max(100*conv_density, 1e-6)` 的 AO Frobenius 残差要求，polishing 内还使用密度变化判据。

这仍不是依据网格误差估计自动分类噪声的算法，亦不是所有几何优化都达到同等精度的证明。阈值与 AO 数/重叠条件数有关；若需要严格梯度和响应，应进一步引入正交基残差、请求级驻点容差、最终本征方程残差及格点加密一致性检查，而不能简单放宽阈值或接受任意有限数。

## 闭环执行结果

最终统计写入 `evidence/acceptance-summary.json`，以下以该证据文件和带 exit.txt 的日志为准。初始基线为 **123 个 Rust tests 通过**；这并未检测出 RI-J 分支开关错误和若干标准方法缺项。

最终源码与扩展的结果如下，所有完成标记 exit code 均为 0：

| 验证项 | 最终结果 | 证据 |
|---|---|---|
| Release 扩展重建、安装与导入 | 成功；安装 pyd 与刚构建 DLL 的 SHA-256 完全一致 | `build-complete.log` / `.exit.txt` |
| 14-crate Rust 全量回归 | **134 passed / 0 failed / 0 ignored**，68 个 suite summaries | `cargo-complete.log` / `.exit.txt` |
| Python 单元全量回归 | **118 passed / 0 failed**，514.13 s | `pytest-complete.log` / `.exit.txt` |
| 用户指定 OPV 专项 | **3 passed / 0 failed**，14.70 s；含小分子计算和大体系元数据检查 | `opv-special.log` / `.exit.txt` |
| 独立积分/RI 探针 | 3 tests passed；192 组中心导数恒等式差分；缓存/on-the-fly J 最大差 2.897e-11 | `integral-independent.log` |
| CAM ITYH 点值与偏导 | 70 位独立公式的 3 组点值及偏导验证通过 | `cam-reference.txt`、Rust XC test |
| CUDA 契约 | 返回不可用，错误参数及 CPU 回退契约通过；**GPU kernel 未执行** | `cuda-runtime.log` |
| 真实 IC-2F Direct SCF | 22 步收敛，94 AO，−870.5416758786621 Eh | `opv-measured.json` |
| 最终源码/二进制漂移检查 | manifest_drift=[]；pyd 与 DLL 哈希 `fd57b59fd35b084d91e7548a7436ea182b9dad35f3afeffe0f34328011f269fc` | `acceptance-summary.json` |

新增 1 项 Python 来源拦截测试，因此由起点 117 项增加到 118 项。Rust 由基线 123 项增加到 134 项。全量通过仅针对**现行、已纠正的科学接口契约**；拒绝未实现功能属于有意的能力收缩，不能解释为全部原有功能均已保留或已实现。

中间失败全部保留：修复初期改变格点求和分组导致有限差分检查出现约 1.5e-9 的偏差，随后用补偿求和恢复原阈值；真实启用 RI-J 后发现原 LDA 超差并修复辅助指数；旧的 wB97X-D/SMD“成功”测试已改为验证明确拒绝。最后一种属于**科学接口契约纠正和能力收缩**，不能称该功能已经实现或零行为变化。

## OPV 数据真实性与性能表

ChemDatas 中实际 IC-2F 为 22 原子 C13H4F2N2O，所查 Gaussian 日志为 wB97XD/6-31G(d,p)，**290** 个基函数。B3LYP/STO-3G 测试与该方法/基组不同，不能声称总能和 HOMO/LUMO 精准对齐。OPV 测试还包括从日志/结构读取 BTP、Y6、PM6 和 p4D18 参数；这些 loader assertions 不表示对这些大分子做了完整原生 SCF。

本次新增真实 IC-2F Direct SCF 探针记录 atom/AO 数、方法、格点、能量、耗时及 Windows 进程峰值工作集；结果保存在 `evidence/opv-measured.json`。该独立探针与测试并行运行时耗时受竞争影响，不能作独占硬件性能认证。峰值工作集包含整个探针进程，不是按公式估计，也不是大体系外推。

最终构建的实测结果为 **94 AO、22 次 SCF、−870.5416758786621 Eh、333.24 s，峰值工作集 131,301,376 bytes（约 125.22 MiB）**。B3LYP/STO-3G、coarse grid、Direct=true、RI-J=false，能量不含额外 D3。二进制哈希见 `opv-binary.json`，必须与最终安装 pyd 一致。

另按实际 ChemDatas 元素组成和 Rust 原生 JSON 基组逐壳计数，得到以下**维度核对，不是大分子 SCF**（`evidence/native-basis-counts.json`）：

| 输入体系 | 原子数 | STO-3G AO | 6-31G(d) AO | 6-31G(d,p) AO |
|---|---:|---:|---:|---:|
| IC-2F | 22 | 94 | 278 | 290 |
| BTP core | 159 | 455 | 1235 | 1505 |
| y6_full（本次 ChemDatas 结构） | 199 | 639 | 1783 | 2065 |
| PM6 monomer | 158 | 510 | 1388 | 1622 |

历史表中的 IC-2F/STO-3G=72、6-31G(d)=226、BTP/STO-3G=540 均与这些实际输入不符；187 原子 Y6 与 199 原子 y6_full 也不能混成同一算例再交叉使用 AO 数、能量和耗时。

历史 CSV 的 projected 体系能量、RAM/VRAM 和精确耗时均没有本次可认证测量链，统一标为 UNVERIFIED_HISTORICAL；原始状态与数值保存在 backup。不能把矩阵内存算术、日志中的 Gaussian 算力或某种稀疏性假设转换为 CalChem 已运行证据。

## 未闭合的生产准入要求

1. 标准 wB97X-D 的完整半局域 XC、自旋相关、原始色散与 wB97X 的独立参数化尚未实现；SMD 同样未实现。
2. RI-J 的解析梯度与一致响应、范围分离/Direct/RI SCRF 没有实现；本次以拒绝错误结果闭合安全边界。
3. 大体系没有全进程配额分配器、磁盘 DIIS、流式网格和可扩展密集对角化；TDDFT 仍预存 ERI，不能继承 Direct SCF 的内存主张。
4. 需要同几何/同基组/同泛函的独立外部基准，以及近简并、弥散、高角动量、卤素/ECP 的系统误差集合；现有宽泛能量断言或自家有限差分不能代替。
5. C-PCM 中 dense boundary inverse、对角 shift 与表面电荷归一化的变分一致性仍需专题外部验证；解析 PCM 梯度当前明确拒绝，现有 FD 是固定 AO 密度偏导，不是完全弛豫 SCRF 梯度。
6. Windows CUDA 构建存在工具链硬路径；无 GPU/错误驱动环境未穷举。CPU fallback 只能对本次实际测试分支负责。
7. 全量编译仍出现既有 PyO3 cfg 和溶剂模块未使用变量告警；本次没有把告警伪装为“零告警编译”，也没有为了消除告警修改数值验收阈值。

**审计裁决：完成本次识别缺陷的修复或保护性隔离、保存回滚与复现资产；不签署原来的全部功能完成、全部理论正确和工业生产能力认证。**

## 复现与回滚

在工程根目录运行：

```powershell
powershell -NoProfile -File tools/dev/build_ext.ps1
powershell -NoProfile -File scripts/cargo_test.ps1
powershell -NoProfile -File scripts/run_pytest.ps1 tests/unit
powershell -NoProfile -File scripts/run_pytest.ps1 tests/unit/test_p6_opv_molecular_validation.py
```

脚本将构建、缓存和临时输出放在 D:/E:；未下载/安装新依赖。自定义 cargo 参数在 PowerShell 7 中使用 `-CargoArgs @('test','-p','calchem-int')`，避免公共参数 `-p` 的歧义。

回滚时仅将 `backup/` 中相应文件复制到同名原路径；先确认工作区后续无新增人工修改，再复原。不要运行 git reset/clean，因为起点本身有大量未提交交付物。新增测试/审计文档可单独保留。旧 pyd 位于 `build_target/runtime/pyd-backup/`，如恢复旧源码必须同步恢复其兼容扩展，或重新构建；不能混用 ABI/字段定义已变的二进制。

公式出处及查询结论见 `02_PRIMARY_RESEARCH.md`。操作与经验记录见 `03_OPERATIONS_AND_LESSONS.md`。
