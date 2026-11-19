# CalChem 纯 Rust + CUDA 量化计算引擎性能瓶颈求助文档

**日期**：2026-11-18  
**目标**：解决 CalChem 原生引擎在中等/大体系有机分子单点能计算中比 Gaussian 16 慢 10–18 倍的底层架构与算法瓶颈，寻求量子化学算法（精确交换 `K`、稀疏 RI-J）与 CUDA 显存/核函数设计方向的专家建议。

---

## 1. 硬件与软件环境

- **操作系统**：Windows（x86_64）
- **CPU**：16 线程（Rayon 并行）
- **GPU**：NVIDIA GeForce RTX 5060 Ti（8151 MiB 显存，Compute Capability `12.0` / `sm_120`）
- **技术栈**：纯 Rust 计算核心（`calchem-model` / `calchem-basis` / `calchem-tensor` / `calchem-int` / `calchem-dft` / `calchem-scf` / `calchem-cuda`），已剥离 Python 依赖，通过纯 Rust CLI `calchem_cli.exe` 直接解析 `.gjf`/`.xyz` 并执行 SCF。
- **CUDA 集成**：通过 `nvcc -arch=sm_120` 编译 cubin，Windows 下经 `nvcuda.dll` Driver API 动态加载并常驻显存。

---

## 2. 实测基准与核心困境

### 基准 1：72 原子杂化泛函 `ydw-PTQ10`（`B3LYP/STO-3G`，`Coarse` 格点，`--rij --gpu`）
- **体系规模**：72 原子，`n_ao = 212`，`n_shells = 142`
- **Gaussian 16 耗时**：**`16.81 s`**
- **CalChem 纯 Rust CLI 耗时**：**`301.42 s`**（收敛于 27 步，`E_tot = -1849.309081084371 Eh`，`gpu_kernel_invocations = 68`，`gpu_fallback_invocations = 0`）
- **精确阶段耗时剖析（`CALCHEM_PROFILE=1`）**：
  - 输入解析 `parse`：`0.0510 s`
  - 基组构建 `basis`：`0.0006 s`
  - Becke 格点 `grid(Coarse)`（`98,014` 点）：**`0.0893 s`**
  - 单电子积分 `1e`：`2.107 s`
  - 引擎初始化 `builder_init`：`2.959 s`
  - 正交化与初猜 `orth_guess`：`0.123 s`
  - 格点 XC 积分 `scf_vxc`：`4.080 s`
  - 对角化 `scf_diag`：`1.849 s`；对易子 `scf_comm`：`0.138 s`
  - **真正卡死的地方**：**`scf_2e = 211.724 s` + `polish = 78.242 s`（合计 `289.97 s`，占总耗时 96.2%）**

### 基准 2：199 原子纯泛函 `ydw-BTP-4F`（`PBE/STO-3G`，`Coarse` 格点，`--rij --gpu`）
- **体系规模**：199 原子，`n_ao = 639`，`n_shells = 419`
- **Gaussian 16 耗时**：**`107.10 s`**
- **CalChem 纯 Rust CLI 表现**：
  - `parse = 0.0559 s`，`basis = 0.0004 s`，**`grid(Coarse) = 0.5496 s`**（`266,181` 点，`streaming=false`）
  - 随后进入 SCF 在 `95 s` 阈值仍未完成（已按“慢于 Gaussian 即暂停”原则于约 `112 s` 终止）。

---

## 3. 已排除的“假瓶颈”

经过本轮定向重构与实测，以下模块已确认**不是**当前性能瓶颈：
1. **Python 胶水层**：已通过 `calchem_cli.exe` 完全去除。
2. **Becke 格点生成与 `Vxc` 积分**：通过预计算成对距离与不对称参数表（`BeckeAtomTable`）及零分配双阶段筛选（`becke_partition_weight_fast`），72 原子格点仅 `0.089 s`，199 原子格点仅 `0.550 s`，`Vxc` 全程仅 `4.08 s`。
3. **小/中体系 GPU RI-J 收缩**：在 `n_ao = 212` 时，三中心张量 `B` 常驻显存，68 次内核调用全部命中 GPU（0 回退），`J` 矩阵本身几乎不耗时。

---

## 4. 当前面临的三大技术“大山”

### 大山 1：杂化泛函精确交换 `K` 的 CPU 四中心积分极端低效（`ydw-PTQ10` 占 96.2% 耗时）
在 `B3LYP`（20% HF 交换）下，`RksFockBuilder::DensityFitting` 用 GPU RI-J 算完 `J` 后，仍调用 CPU `DirectFockEngine::build_k_with_omega_bounded`（`crates/calchem-int/src/direct_fock.rs`）计算 `K`，平均每次调用耗时约 **6.2 秒**（仅 `n_ao = 212`）：
- **无预计算壳对表**：每次四中心积分 `eri_shell_quartet_with_omega_into`（`crates/calchem-int/src/eri.rs`）都在内部重新构造 `EgpPair`（含指数衰减与高斯乘积中心），导致 $O(N_{\text{quartets}})$ 次重复计算 $O(N_{\text{pairs}})$ 数据。
- **无序遍历无法早停**：`sig_pairs` 未按 Schwarz 上界降序排列，无法在 `b_ij * b_kl * d_max < thresh` 时提前 `break` 内层循环。
- **收敛后冗余 `polish`**：`crates/calchem-scf/src/rks.rs` 在 SCF 已满足收敛条件后，仍执行约 10 次额外 `polish` Fock 重建，白白消耗 `78.24 s`。
- **根本算法缺失**：对于 `K` 矩阵，目前完全依赖 CPU 传统 4 中心 Direct SCF，既没有半数值交换（COSX / sn-K），也没有 RI-K 或 GPU 批处理四中心积分（如 rys/MD GPU kernel）。

### 大山 2：大分子纯泛函 RI-J 的稠密 `B[n_aux, n_ao^2]` 显存/内存爆炸（`ydw-BTP-4F` 退回 CPU）
`ydw-BTP-4F` 为纯泛函 `PBE`（无 `K` 项），理论上只算 `J + Vxc`，却在 95s 内未算完：
- `crates/calchem-int/src/density_fitting.rs` 中缓存的三中心张量 `cached_b` 按稠密 `[n_aux, n_ao * n_ao]` 分配。当 `n_ao = 639`（`n_ao^2 = 408,321`）且 `n_aux` 达数千时，稠密 `B` 张量体积超过 `max_df_bytes`（2.5 GiB）及 8 GB GPU 显存上限。
- 一旦 `cached_b` 为 `None`，引擎立即退回**纯 CPU on-the-fly 模式**：每次 SCF 迭代都在 CPU 上**重新计算两遍**所有三中心积分（Pass 1 算 `v_P`，Pass 2 算 `J`），且内层仍使用堆分配的 `shell_3center_block(...)`，GPU 利用率直接归零。
- **压缩难点**：199 原子体系中，超过 90% 的 `(μ, ν)` AO 对在 Schwarz 意义下为零。若改为仅存储显著上三角 AO 对的稀疏格式 `B_packed[n_aux, n_sig_pairs]`，体积可压缩 10–20 倍并全部塞入 8 GB 显存；但如何在 CUDA 上高效完成 `B_packed` 与密度矩阵的收缩（避免非合并访存与散射写回时的原子操作争用）需要谨慎设计。

### 大山 3：辅助基 Metric 分解先做全矩阵特征分解
- `crates/calchem-tensor/src/lib.rs` 中的 `SpdFactor::from_matrix` 在构建 `RiJEngine` 时，会先对完整的 `N_aux × N_aux` 矩阵调用 `selfadjoint_eigendecomposition`（计算全部特征值与特征向量），然后再做 `cholesky`。
- 当分子扩大到 200 原子、`N_aux` 达到数千时，$O(N_{\text{aux}}^3)$ 的全特征分解在初始化阶段即带来数十秒级开销，实际上仅在 Cholesky 分解失败（出现线性相关/秩亏）时才需要特征分解降维。

---

## 5. 向外部专家/社区求助的关键问题

1. **精确交换 `K` 的路线选择**：在 16 核 CPU + 8 GB 显存（RTX 5060 Ti）的消费级工作站上，针对 70–200 原子有机分子的杂化泛函（B3LYP/PBE0/wB97XD），从工程复杂度与提速比来看，应优先实现 **半数值交换（COSX / Chain-of-Spheres）**、**RI-K**，还是 **基于预计算壳对表的 GPU 批处理 ERI**？
2. **稀疏 RI-J 的 CUDA 内存布局**：将 `B` 压缩为 `B_packed[n_aux, n_sig_pairs]` 后，如何在 CUDA 上高效实现 `v_P = sum_idx B_packed[p, idx] * P_packed[idx]`（本质是稠密矩阵-向量乘 GEMV，可直接走 cuBLAS 或共享内存规约）以及反向 `J_packed[idx] = sum_p x_p * B_packed[p, idx]`，在保持 `sm_120` 合并访存的同时不引入显著索引开销？
3. **SCF 收敛后 `polish` 的必要性**：在 `delta_e < 1e-10`、`rms_d < 1e-8` 且 DIIS 误差达标后，是否有必要保留长达 10 步的固定点 `polish` 环，还是只需单次纯对角化即可满足解析梯度精度要求？
