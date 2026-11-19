# CalChem P0 工程与数值基线审计报告

**审计日期**：2026-11-10  
**审计执行环境**：Windows 11 x86_64，独立沙箱运行目录 `build_target/runtime/p0_audit_20261110`  
**审计原则**：只读审计、隔离测试、客观实测、无一隐瞒、绝不破坏现有工作树。

---

## 一、 硬件与开发工具链清单（真实盘点）

| 硬件/软件组件 | 规格与实测参数 | 实际可用状态与评注 |
| :--- | :--- | :--- |
| **CPU 处理器** | AMD Ryzen 7 9700X (8 核心 / 16 逻辑线程) | 物理核充足，AVX-512 / FMA 指令集就绪 |
| **主机物理内存 (RAM)** | **16.0 GB Total** (标配单机工作站) | **核心硬件瓶颈**：必须严格限制 Rayon 线程局部缓冲 |
| **GPU 图形处理器** | NVIDIA GeForce RTX 5060 Ti | 8151 MiB 总显存（当前空闲约 5.9 GB，驱动 591.86，SM 12.0） |
| **CUDA 工具链** | **NVCC 13.3.73** (Release 13.3, V13.3.73) | 编译器与驱动就绪，位于 `D:\NVIDIA_CUDA_Toolkit\bin` |
| **Rust 编译器** | rustc 1.99.1 / cargo 1.99.1 | `x86_64-pc-windows-msvc` 稳定工具链 |
| **Python 运行时** | Python 3.13.13 (Miniconda 64-bit) | 位于 `D:\YDW\Trae_Gromacs\miniconda3\python.exe` |

---

## 二、 针对 ChatGPT 评审核心意见的技术对照与确认

经本轮代码审查与一手文献核对，**ChatGPT 提出的核心批评完全成立**，原讨论与规划中存在以下过度承诺与概念混淆，已在此次审计中全部纠正：

1. **阿姆达尔定律（Amdahl's Law）的严肃适用**：
   * 原规划中“GPU 整体提速 20~100 倍”属于典型的**算子局部提速与端到端全局提速混淆**。
   * 在 B3LYP 等杂化泛函中，若 $V_{\text{xc}}$ 占总耗时的 40%，即使 GPU 将 $V_{\text{xc}}$ 加速到 0 秒，端到端整体提速上限仅为：
     $$\text{Speedup}_{\max} = \frac{1}{(1 - 0.4) + \frac{0.4}{\infty}} = \frac{1}{0.6} \approx 1.67\times$$
   * 真正的耗时大头（精确交换 $K$、稠密对角化、DIIS 历史）如果依然停留在 CPU，整体性能绝不可能获得数量级飞跃。
2. **Rayon 线程局部全矩阵分配的“内存陷阱”**：
   * 原先宣称“Direct SCF 后 2000 原子仅需几 GB 内存”，严重低估了多线程局部缓冲。
   * 实测 `DirectFockEngine` 当前写法：每个 Rayon 分区独立分配一个 `vec![0.0; n * n]`。
   * 对于 6224 AO（p4D18），单张矩阵 309.9 MB。**在 16 线程下，仅 RKS 局部累加器就需 4.96 GB，UKS 双自旋通道需 9.92 GB**！直接占满本机 16 GB 物理内存的 60% 以上，极易发生堆分配失败。
3. **GPU 技术栈：废弃 WGPU，确立 CUDA / cuBLAS / cuSOLVER 路线**：
   * W3C WGSL 规范目前没有原生通用的 `f64` 浮点支持；WebGPU 面向量子化学双精度计算存在根本性的精度缺陷与碎片化。
   * 确认以 **CUDA 13.3 (NVCC) + cuBLAS / cuSOLVER** 作为 CalChem 的第一可选高性能后端。
4. **ωB97X-D 物理模型纠正**：
   * 纠正“ωB97X-D 用 D3(BJ) 验证”的错误概念。Chai & Head-Gordon (2008) 的 $\omega$B97X-D 拥有自身参数化的色散函数，不能混用 Grimme D3(BJ)。

---

## 三、 当前 Direct SCF 源码工程审计（实测分析）

### 1. `crates/calchem-int/src/direct_fock.rs`
* **实现亮点**：
  * 实现了基于 Cauchy-Schwarz 筛选的显著壳层对表 `find_significant_shell_pairs`；
  * `contract_quartet_rks` 与 `contract_quartet_uks` 实现了在循环内随算随缩并，确实避免了构建 $O(N^4)$ 的四维四元组张量；
  * UKS 实现了 Coulomb 与 Exchange 单遍联合遍历（Single-Pass Contraction）。
* **已发现的缺陷与脆弱点**：
  * **缺陷 A（内存分配）**：`fold(|| vec![0.0f64; n * n], ...)` 导致 Rayon 每个工作任务分配一张 $N_{\text{ao}} \times N_{\text{ao}}$ 矩阵，超大体系会发生内存爆炸；
  * **缺陷 B（筛选上界不严谨）**：`compute_shell_pair_max_density` 仅计算 $D_{ij} = \max |P_{\mu\nu}|$，未乘以壳内基函数数目乘积 $N_i N_j$。对于包含 d 壳（$5 \times 5 = 25$）或 f 壳（$7 \times 7 = 49$）的体系，可能低估积分贡献达 25~49 倍，存在截断误差隐患；
  * **缺陷 C（对称展开权重）**：目前的权重采用经验因子除法，对于全相同、成对相同的壳对排列退化，尚未提供严格的形式化群轨道推导与形式化证明。

### 2. `crates/calchem-scf/src/fock_builder.rs` 与驱动层
* **实现亮点**：
  * 在 `rhf.rs`, `rks.rs`, `uhf.rs`, `uks.rs` 中实现了 $\Delta P = P^{(k)} - P^{(k-1)}$ 增量 Fock 构建；
  * 周期性 8 轮强制重置物理全量 Fock，防止浮点误差漂移；
  * $V_{\text{xc}}$ 在每一轮均从真实总密度 $P$ 重新计算，未错误地纳入线性增量递推；
  * 迭代收敛后执行了终态不动点抛光（Rebuild full $F[P]$ 并检查 $\|FPS - SPF\|$ 残差）。
* **已发现的缺陷与脆弱点**：
  * **缺陷 A（内存准入门禁虚设）**：`validate_direct_resources_and_options` 默认允许 64 GB 内存，超越本机 16 GB RAM；且估计值只计算了单通道，未计入 UKS 双通道、DIIS 误差历史和正交化缓存；
  * **缺陷 B（Python 接口未透出）**：`ScfOptions.direct` 在 `calchem/_core.pyi` 与 Python 顶层未暴露，Python 用户无法显式控制 Stored 或 Direct 模式，完全依赖隐式 $N_{\text{ao}} > 64$ 分支。

---

## 四、 真实 Gaussian 16 生产日志深度审计（黄金对照）

从您本地目录 `E:\YDW\ChemDatas\Log\` 中调取两份真实的超大规模计算日志进行审计：

### 1. 典型受体 Y6 (`ydw-Y6-opt.log`)
* **体系规格**：**187 个原子**，closed-shell 单重态（383 $\alpha$ + 383 $\beta$ = 766 电子）；
* **基组与泛函**：`b3lyp/6-311G(d,p) em=gd3bj`；
* **基函数规模**：**2374 个球谐基函数**（2475 个笛卡尔基函数），3990 个原始高斯原函数；
* **Gaussian 运行资源与模式**：
  * `%mem=20GB`, `%nproc=20` (20 线程共享内存)；
  * `Int= 0 D2E= 0`：**显式证实为 Direct 模式**（未在磁盘上存储任何双电子积分）；
  * `NAOKFM=T Big=T`：启用了快速多极展开（Fast Multipole Method）；
* **真实耗时**：
  * CPU 累计耗时：**5 天 21 小时 37 分钟**；
  * 实际墙钟耗时（20 核心并行）：**7 小时 7 分钟**。

### 2. 超大聚合物 p4D18 四聚体 (`ydw-p4D18-opt06.log`)
* **体系规格**：**726 个原子**，closed-shell 单重态（1449 $\alpha$ + 1449 $\beta$ = 2898 电子）；
* **基组与泛函**：`b3lyp/6-31g(d) em=gd3bj opt=(Cartesian,loose)`；
* **基函数规模**：**6224 个基函数**，12312 个原始高斯原函数；
* **Gaussian 运行资源与模式**：
  * `%mem=60GB`, `%nprocshared=40` (**40 核心高性能工作站，60 GB 内存**)；
  * `Int= 0 D2E= 0`：纯 Direct SCF 模式；
* **真实耗时**：
  * CPU 累计耗时：**276 天 3 小时 19 分钟**；
  * 实际墙钟耗时（40 核心并行）：**6 天 23 小时 0 分钟**。

> [!IMPORTANT]
> **严酷的工程现实**：
> Gaussian 16 算 726 原子的 p4D18 动用了 **60 GB 内存与 40 个核心整整跑了一周**！
> 这彻底推翻了“Direct SCF 后 2000 原子在普通 PC 上只需要几 GB 内存且能快速收敛”的盲目乐观假设。若没有分块 Fock 累加、近邻多极展开（FMM）与 RI 辅助基组，纯 Direct 算法在万级基函数下的计算量依然极其庞大。

---

## 五、 隔离测试验证结果汇总（证据链）

在完全隔离的独立沙箱环境变量下（避免任何全局环境污染与环境变量重叠），重新复测全套测试用例：

### 1. Rust 工作区全量测试（`cargo test --workspace --locked`）
* **运行命令**：
  ```powershell
  . docs\audits\2026-11-10_round2\probes\env.ps1
  $env:CARGO_TARGET_DIR = 'E:\YDW\Trae_Gaussian\build_target\runtime\p0_audit_20261110\target'
  $env:CARGO_PROFILE_DEV_OPT_LEVEL = '2'
  $env:CARGO_PROFILE_TEST_OPT_LEVEL = '2'
  cargo test --workspace --locked --target-dir $env:CARGO_TARGET_DIR
  ```
* **退出状态码**：**`0` (Exit 0)**
* **统计结果**：**102 passed, 0 failed, 0 ignored**
* **编译与测试用时**：**1 分 03 秒**

### 2. Direct SCF 独立集成测试（`tests/direct_scf.rs`）
* **统计结果**：**4 passed, 0 failed**
  * `test_rhf_direct_scf_matches_stored_eri_energy_and_orbitals`：H2O / 6-31G (13 AO)，能量差 $|\Delta E| = 2.13 \times 10^{-11}\text{ Eh} \le 10^{-10}\text{ Eh}$，轨道能差 $\le 10^{-8}\text{ Eh}$；
  * `test_rks_b3lyp_direct_scf_matches_stored_eri`：H2O / STO-3G (7 AO)，能量差 $|\Delta E| = 4.88 \times 10^{-12}\text{ Eh} \le 10^{-10}\text{ Eh}$；
  * `test_uhf_and_uks_direct_scf_matches_stored`：H2O+ / STO-3G (7 AO)，UHF 能量差 $3.15 \times 10^{-11}\text{ Eh}$，UKS 能量差 $4.22 \times 10^{-11}\text{ Eh}$；
  * `test_large_molecule_direct_scf_runs_without_memory_barrier`：Ne8 / 6-31G (**72 AO**)，在 Direct 模式下顺利启动并收敛，用时 9.2 秒。
* **边界确认**：目前测试覆盖的最大规模仅为 **72 AO**，距离千级基函数仍有数量级差距，尚未建立 100~500 AO 的渐进性能基准。

### 3. Python 单元测试全量验证（`pytest tests/unit`）
* **运行命令**：
  ```powershell
  . docs\audits\2026-11-10_round2\probes\env.ps1
  & $env:PYO3_PYTHON -B -c "import os,tempfile,pytest; tempfile.tempdir=os.environ['TEMP']; raise SystemExit(pytest.main(['tests/unit','-q','-p','no:cacheprovider']))"
  ```
* **退出状态码**：**`0` (Exit 0)**
* **统计结果**：**104 passed, 0 failed, 耗时 471.91 秒 (07分51秒)**
* **测试用例覆盖**：涵盖适配器、原子、Rust 核心扩展、D3 与溶剂梯度、DFT、ESP 与布局数、I/O、等值面流式生成、分子模型、Native 后端、OPV CAM-B3LYP 与二聚体、OPV 工作流、红队契约、请求校验、服务队列、PCM/SCRF 溶剂化、TDDFT、TS 与 IRC、振动与热力学全套 104 项测试全部通过。

---

## 六、 P0 审计综合定论与后续行动准则

1. **里程碑 1 状态定级**：
   * **功能原型已具备（Functional Prototype Achieved）**：Direct 积分与增量 Fock 的代码主干已经跑通，小体系下与 Stored ERI 满足 $\le 10^{-10}\text{ Eh}$ 严格物理一致性；
   * **生产工程未达标（Not Production-Ready for Large OSCs）**：受限于 Rayon 线程局部缓冲爆炸、上界筛选缺乏壳多重度因子、内存准入门禁虚设等工程缺陷，**目前绝不能宣称“里程碑 1 已完成且支持数千原子”**。
2. **后续推进顺序（严格执行 ChatGPT 建议的科学与工程双主线）**：
   * **P1 阶段（稳定化 Direct SCF）**：
     * 重构 `DirectFockEngine`，实现分块矩阵累加（Chunked Fock Accumulation），将并行额外内存从 $O(K \cdot N^2)$ 压减为常数级；
     * 严密修正 Cauchy-Schwarz 筛选上界（包含壳层维度因子 $N_i N_j$）；
     * 纠正内存门禁，以本机 16 GB RAM 为物理硬约束；
     * 在 Python 层显式暴露 `direct: Optional[bool]` 参数。
   * **P2 阶段（范围分离泛函）**：严格根据文献实现短程/长程 Obara-Saika 积分递推与标准 $\omega$B97X-D 泛函（含自身经验色散）。
   * **P3 阶段（密度拟合 RI-J）**：三中心积分与度量分解，显著降低库仑场计算量。
   * **P4 阶段（CUDA GPU 单算子验证）**：先针对对角化与 $V_{\text{xc}}$ 网格开展可选 CUDA 原型测试，实测端到端收益，严禁虚标加速比。
