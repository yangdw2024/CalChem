# 对粘贴执行记录的独立审查

## 结论

记录中的实现方向有实际源码支撑，但它把“存在代码/小样本测试通过”提升成“真实 GPU 与完整内存闭环完成”，证据不足。当前扩展 DLL 与 `_core.pyd` 哈希确实相同，并且与粘贴文本中的 SHA-256 完全一致；不过，未找到针对这批最新源码的原始 Cargo/Pytest 结束日志。能查到的最近完整验收是 Rust 134、Python 118 项，且形成于本批代码文件修改之前。故“119 项全部通过”和“当前源码对应版本验收通过”暂定为**未证实**。

本次仅做审查，未修改源文件，也未运行测试。

## 逐项核验

### 1. CUDA：当前构建身份可核实，硬件执行不能由测试设计单独证明

- 已核实：当前 `calchem/_core.pyd` 和 `build_target/release/_core.dll` 的 SHA-256 都是 `FA41E0DDF29754EF7B0816E5B5FD6BCA6E680E1F5683D55B18CAE0890EA09D49`，与粘贴文本一致。
- 未证实：粘贴文本没有附 5 项 Rust 测试的原始完整输出、运行时设备/驱动信息、kernel 成功返回证据或该二进制加载路径。旧 `cuda-runtime.log` 记录的是 `CUDA available=false, kernel executed=false`，而且早于当前 CUDA 源码，不能用作本次结果，也不能反驳本次结果。
- 当前测试设计的限制：`gpu_result_or_unchanged_cpu_fallback_contract` 在不可用时也通过；Python GPU/CPU 能量比较同样允许静默 CPU fallback。因此它们是回退/契约测试，不会单独保证 `native_kernel_executed=true`。粘贴中报告的真 GPU 结果若成立，需对应可追溯原始运行日志。
- 实现风险：Windows Driver API 初始化失败时 `init()` 没有释放已加载的 `nvcuda.dll`；若反复探测，会重复 `LoadLibraryA` 并覆盖句柄。主上下文 retain 后若后续设置 current context 失败，也没有 release。属于失败路径资源泄漏风险，需改成一次性初始化状态并成对清理。

### 2. RI-J 梯度：公式在度量满秩时合理，截断度量的导数没有被正确覆盖

- `compute_coulomb_gradient` 采用三中心导数项与 `-1/2 x^T (dV) x` 项；在度量可逆、固定辅助基的前提下，该结构符合 RI-J 能量驻值导数。
- 但 `invert_coulomb_metric` 会将低于阈值的度量特征值置零，构造截断广义逆；梯度代码没有保存“是否截断”状态，仍直接套用普通逆矩阵驻值导数。秩截断广义逆对几何的导数还含投影子空间变化项，若截断子空间随坐标变化，当前结果未必是能量导数。
- 因而 H2O 固定密度中心差分和 H2 一键长全梯度通过（即便粘贴声明属实）只覆盖特定基组/几何；不能证明近线性相关度量下的 RI 梯度正确。建议在截断时拒绝解析梯度，或实现并单独验证广义逆导数。
- 测试覆盖口径：当前 Rust 测试确有 H2O 固定密度的 9 个坐标差分；波函数级 RHF/RKS/UHF/UKS 检查集中在 H2 单一键长方向；几何优化示例也是 H2 RHF。它不是“四种方法在 H2O 全 9 自由度梯度均已对照”的证据。

### 3. Becke 流式网格：去掉全网格驻留成立，不代表零额外内存或大体系峰值已验收

- 已核实：`MolecularGrid` 流式对象的 `points` 是空 `Vec`，保留分子副本和轻量任务描述；RKS/UKS 的 Vxc 与 XC 梯度有按任务批量生成/处理路径。对该字段“没有驻留完整 `Vec<GridPoint>`”的描述准确。
- 但每个并行分块仍生成批次点；RKS/UKS 还为并行结果保留稠密 `N_AO × N_AO` 矩阵，随后归并。`CALCHEM_MAX_DIRECT_BYTES` 只约束该 Vxc worker 的粗略矩阵预算，并非全进程硬内存限额；工作区、输出矩阵、SCF、网格任务描述和 allocator 峰值未纳入同一过程级保证。
- 现有对照测试是小分子水的 materialized/streaming 数值对照，并通过 `points.is_empty()` 验证无完整网格向量；它没有直接测量峰值 RSS、超大分子或批次最大容量。因此“流式实现存在”可信，“零内存驻留/大体系内存闭环已证实”不成立。

### 4. 磁盘 DIIS：溢写功能存在，但失败时会破坏 RAM 上界

- 已核实：超过估算阈值或显式设置环境变量时，`DiisState` 尝试把 Fock/误差历史放进环形临时文件；代码内有与内存实现的数值对照测试。
- 高风险：临时文件创建、锁定或任一读写/外推失败时，错误被吞掉，然后 `push_and_extrapolate` 继续把矩阵复制到内存历史。此路径会令“超过 RAM 阈值自动转磁盘”的上界失效，极端情况下重新引入 OOM 风险。磁盘已部分失败时仍可能继续回退 RAM。
- 路径风险：临时文件路径来自 `std::env::temp_dir()`。审计脚本的环境配置将 TEMP/TMP 指向 E 盘，但直接运行库/API 时由宿主默认临时目录决定，Windows 上可能落到 C 盘。应支持显式、可验证的 CalChem scratch 路径，并在不满足路径/磁盘预算时明确报错或执行有界策略。
- 声称的“DIIS 结果误差 < 1e-14”仅由小型合成矩阵测试支持，不能作为大体系收敛、磁盘 I/O 故障或磁盘占用管理的认证。

### 5. 测试与哈希

- 当前安装二进制与构建产物：哈希一致，且匹配粘贴文本中的 `FA41...09D49`。
- 当前关键源码（网格、Vxc、DIIS、RI-J、CUDA）最后修改时间为 2026-11-11 晚间至 2026-11-12 凌晨；已有 `pytest-complete.log` 是 118 passed，Cargo 日志是 134 passed，形成时间和源码快照早于这批修改。旧日志不能覆盖当前版本。
- 粘贴文本里出现的 pytest 运行过程被截断，随后给出的“119 项全部通过”没有对应结束行、exit code 文件、Cargo workspace 汇总、构建脚本结束日志或逐项哈希 manifest。该测试结论在此次审查中记为**未验证**，不是认定失败。
- 现有 `test_gpu_and_density_fitting.py` 本身只允许 CUDA 不可用时通过；H2O RI-J 的端到端验收也不代表 IC-2F/BTP/Y6/PM6 大体系完成 SCF/梯度/优化。

## 审查评级

- 实现进展：有实质代码，不是纯文档宣称。
- 物理实现：RI-J 梯度在满秩度量上的形式有依据；截断度量的梯度边界需保护。
- 工程边界：流式网格和磁盘 DIIS 有助于削减驻留，但尚未构成 16 GB 全进程内存保证；DIIS I/O 故障回退会违背其自身内存目标。
- 验收状态：**条件性通过源码结构审查；GPU 原生运行、当前版本全量测试、RI 截断度量梯度与大体系内存实测均未证实。**

## 当前源码定位

- `crates/calchem-dft/src/grid.rs:428-557`：自动流式选择、任务描述、按批生成。
- `crates/calchem-dft/src/vxc.rs:21-36,138-230,358-420`：并行内存估算、流式 RKS/UKS 累加。
- `crates/calchem-scf/src/diis.rs:28-32,180-214`：临时文件路径、自动阈值和失败时 RAM 回退。
- `crates/calchem-int/src/density_fitting.rs:398-460,1056-1339`：特征值截断逆和 RI-J 梯度。
- `crates/calchem-cuda/src/cuda_kernels.cu:139-181,270-278`：Driver 初始化/上下文 retain 和 cubin/PTX 加载。
- `crates/calchem-cuda/tests/ffi_contract.rs:12-22`、`tests/unit/test_gpu_and_density_fitting.py:5-63`：GPU 不可用时仍允许测试通过。
