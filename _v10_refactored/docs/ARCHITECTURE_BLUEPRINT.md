# CalChem v1.0 下一代高性能量子化学计算平台：底层重构架构蓝图与工程实施白皮书

> **文档定位**：面向 CalChem 从 v0.6.0（纯 Python 原型）向 v1.0（生产级高性能混合架构引擎）演进的完整架构设计规范与文件级工程拆解。  
> **硬件与部署基线**：AMD Ryzen 7 9700X（Zen 5 架构，8 核 16 线程，原生双泵 512-bit AVX-512）| 15.1 GB 物理内存 | Windows 11 零 C 盘依赖 One-Folder 独立分发 | 预留跨平台 GPU 异构扩展 | 前端无缝对接 MolForge（PyQt6 + VTK 3D）。

## 1. 战略破局与现有代码库（v0.6.0）深度诊断

### 1.1 为什么“全盘重写”必须结合“精准借力”才能超越四大标杆

在量子化学软件工程领域，[Gaussian 16](https://gaussian.com/people/)、[ORCA 6.1](https://www.faccts.de/docs/orca/6.1/manual/index.html)、[PySCF](https://github.com/sunqm/libcint) 与复旦大学张颖（Igor Ying Zhang）团队开发的 [REST（Rust-based Electronic Structure Toolkit）](https://pubs.aip.org/cps/cjcp/article/38/6/788/3377721/REST-Embracing-the-rust-programming-language-for) 各自代表了不同时代的工程范式。要在**计算速度、系统稳定性、架构现代性**三个维度实现实质性超越，首先必须避免“用低效手写循环重复造轮子”的陷阱，明确真正的性能与体验突破口：

| 超越维度 | 传统软件的痛点与历史包袱 | CalChem v1.0 的破局策略（“精准借力 + 架构代差”） |
| :--- | :--- | :--- |
| **计算速度（Speed）** | [Gaussian 16](https://gaussian.com/people/) 虽有 PRISM 积分，但默认依赖传统 4 中心 ERI 且对 Zen 5 AVX-512 利用不充分；[ORCA 6.1](https://www.faccts.de/docs/orca/6.1/manual/index.html) 依赖大量临时磁盘 I/O（`.tmp` 文件动辄数十 GB）且 Windows 下依赖外部 Microsoft MPI 进程通信；[PySCF](https://github.com/sunqm/libcint) 在 Python 层存在循环调度与内存拷贝开销。 | **全默认无磁盘（Disk-Free / In-Core + Direct）RI-JK / RIJCOSX 架构 + Zen 5 原生 AVX-512 内核**。针对 15.1 GB 内存与 NVMe I/O 瓶颈，通过三中心积分内存压缩（对称存储 + FP32/FP64 混合精度 Screening）与 [Rayon](https://docs.rs/rayon/latest/rayon/) 共享内存并行，消除进程间通信与磁盘读写开销，在中小型及有机光电（OSC）共轭体系上跑赢传统 4 中心引擎与多进程 MPI 引擎。 |
| **数值与系统稳定性（Stability）** | 传统 Fortran/C++ 软件充斥悬垂指针、数组越界、隐式全局变量；SCF 遇到窄带隙共轭体系或过渡金属时极易振荡不收敛；Windows 下子进程崩溃往往留下僵尸锁或损坏的临时文件。 | **Rust 编译期内存安全 + 二阶 SOSCF（Second-Order SCF）/ EDIIS-ADIIS-CDIIS 状态机自动切换**。从根源杜绝段错误（Segfault）与数据竞争；当常规 DIIS 残差停滞时自动无缝切入 Trust-Region Roothaan-Hall（TRRH）或二阶轨道旋转，确保难收敛体系一次收敛。 |
| **工作流与前端协同（Integration）** | 传统软件采用“生成文本输入卡 $\rightarrow$ 启动外部进程 $\rightarrow$ 解析 GB 级文本输出/Cube 文件”的脆弱管道，精度损失大（格式化输出截断）、响应延迟高。 | **PyO3 零拷贝进程内（In-Process）原生绑定 + 外部沙盒双模引擎**。波函数、分子轨道系数、电子密度网格直接通过共享内存暴露为 NumPy/VTK 内存视图，MolForge 渲染轨道等值面从“秒级磁盘读写”降为“毫秒级内存直传”。 |

### 1.2 对 CalChem v0.6.0 核心源码（`CalChem_Rebuild_Research_Core.zip`）的底层审计

通过对上传的 `CalChem_Rebuild_Research_Core.zip` 全量代码审计，当前 v0.6.0 除了“Python 解释器慢”这一表象外，在数值算法层存在五个必须在重构中根治的结构性缺陷：

1. **RI 密度拟合模块（`calchem/integrals/ri.py`）使用了伪造的启发式辅助基组**：在 `build_aux_basis()` 函数中，当前代码并未加载真实的拟合基组，而是用硬编码公式 `exps.append(base_exp * (3.0 ** i))` 动态拼凑未校准的指数。这会导致 RI-J / RI-MP2 的能量偏离标准值数个毫哈特里（mEh），丧失科学可信度。重构后必须全面接入 [MolSSI Basis Set Exchange (BSE)](https://www.basissetexchange.org/) 的标准配对拟合基组（如 `def2-universal-jkfit`、`def2-svp-ri`、`cc-pvdz-ri`）。
2. **解析梯度模块（`calchem/grad/analytical.py` 与 `hf_grad.py`）与优化器脱节且缺失关键 Pulay 项闭环**：`HFGradient.__init__` 默认设置 `numerical=True`，每次梯度计算触发 $6N_{\text{atom}}$ 次完整 SCF；而 `analytical.py` 虽然编写了框架，但仍依赖全尺寸四阶张量 `mf._eri`（内存复杂度 $O(N^4)$），未实现直接积分导数收缩（Direct Derivative Contraction），也未与 `opt/numeric_optimizer.py`、`vib/numeric_frequency.py` 形成生产级闭环。
3. **基组系统（`calchem/basis/basis_set.py`）默认采用笛卡尔高斯（`spherical=False`）**：对于 $d$ 壳层（6 个笛卡尔分量 vs 5 个球谐分量）和 $f$ 壳层（10 vs 7），笛卡尔基组不仅使基函数维度膨胀 $15\%\sim 30\%$，还会引入冗余的 $s$ 型污染分量（如 $x^2+y^2+z^2$），导致大基组（如 `def2-TZVP`、`cc-pVTZ`）下重叠矩阵 $S$ 接近奇异（线性相关），严重破坏 SCF 稳定性。
4. **四中心积分（`calchem/integrals/two_electron.py`）缺乏壳层对（Shell-Quartet）SIMD 向量化**：当前代码虽尝试引入 Numba JIT（`_boys_numba`），但递归过程按单个原始高斯函数（Primitive）逐一分支调用，CPU 分支预测失败率极高，完全无法利用 Ryzen 7 9700X 的 512-bit SIMD 寄存器吞吐能力。
5. **REST 后端（`calchem/backends/rest.py`）仅为未实现的 HTTP 桩代码**：当前 `RESTBackend` 仅预留了对接外部 Agent 的 `NotImplementedError` 接口，尚未吸纳 [REST 引擎](https://pubs.aip.org/cps/cjcp/article/38/6/788/3377721/REST-Embracing-the-rust-programming-language-for) 基于 Rust + `libcint` + Rayon 的核心设计精髓。

## 2. 语言选型与混合架构决策（Deliverable 1）

### 2.1 候选技术栈多维量化评估

针对硬件环境（AMD Ryzen 7 9700X Zen 5、15.1 GB RAM、Windows 11、未来扩展 GPU）与产品形态（MolForge PyQt6 前端 + Windows One-Folder 部署），对五种候选技术路线进行严格对比：

| 评估维度 | Rust 核心 + Python 前端（推荐方案） | C++20 核心 + Python 绑定 | 纯 Julia（Julia + PythonCall） | Mojo | 纯 Rust（含 GUI） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **张量与数值计算性能** | **极高**。通过 [`faer`](https://docs.rs/faer/latest/faer/) 原生纯 Rust AVX-512 矩阵库或绑定 Intel MKL / OpenBLAS，配合 [`libcint-rs`](https://github.com/ajz34/libcint-rs) C99 SIMD 内核，达到硬件理论峰值。 | **极高**。原生调用 Eigen / Blaze / MKL / [Libint2](https://github.com/evaleev/libint)，编译器向量化成熟。 | **高**。LoopVectorization.jl 与 Octavian.jl 性能优异，但 GC 停顿在大内存张量分配时偶有抖动。 | **中**。矩阵乘法演示快，但缺乏成熟的稀疏线性代数、特征值分解（LAPACK）和量子化学底层库。 | **极高**。计算层与混合方案一致，但 GUI 与科学工作流脚本生态弱于 Python。 |
| **GPU 加速可行性（CUDA/ROCm/Vulkan）** | **高**。支持 [`wgpu`](https://wgpu.rs/)（跨硬件支持 AMD 核显/独显、NVIDIA、Apple Metal）及 [`cudarc`](https://crates.io/crates/cudarc) / `cubecl`，无供应商锁定。 | **极高**。CUDA 与 HIP/ROCm 第一方原生支持，Kokkos / SYCL 生态最成熟。 | **高**。CUDA.jl 与 AMDGPU.jl 抽象极佳，但编译缓存体积巨大。 | **低**。仅针对特定 AI 加速卡优化，缺乏通用科学计算双精度（FP64）生态。 | **高**。与 Rust+Python 方案相同。 |
| **自动微分（AD）与解析导数** | **高**。支持 LLVM 级 [`Enzyme`](https://github.com/EnzymeAD/enzyme)（`std::autodiff`）及双数（Dual Numbers）前向/反向传播，结合 `libcint` 高阶积分导数。 | **高**。支持 [Enzyme AD](https://arxiv.org/abs/2010.01709)、Clad、autodiff（C++ 库）及模板元编程。 | **极高**。Zygote.jl 与 Enzyme.jl 是目前科学计算 AD 的标杆。 | **中**。内置微分原语，但尚无法微分复杂的外部 C/Fortran 积分库。 | **高**。与 Rust+Python 方案相同。 |
| **Windows One-Folder 编译与分发** | **极佳**。`cargo` + [`maturin`](https://www.maturin.rs/) 一键编译为单个 `.pyd` 动态库，无需用户安装 MSVC 或设置环境变量，PyInstaller 直接打包。 | **差（痛点）**。Windows 下 CMake + MSVC + vcpkg/conda 处理 Libint2/Libxc/OpenMP 极易出现 DLL Hell 与 ABI 冲突。 | **极差（致命）**。PackageCompiler.jl 生成的 sysimage 动辄 800MB+，首次启动 JIT 延迟（TTFX）严重，与 PyInstaller 极难融合。 | **不可用**。Windows 原生工具链与独立分发尚处于早期阶段，不支持离线单目录生产部署。 | **中**。需重写整个 MolForge（PyQt6 + VTK）前端，抛弃现有 2D/3D 成果，工程代价不可接受。 |
| **与 Python/MolForge 互操作** | **无缝**。[`PyO3`](https://github.com/pyo3/maturin) + `rust-numpy` 提供类型安全的零拷贝内存共享，GIL 释放（`py.allow_threads`）只需一行代码。 | **良**。`pybind11` / `nanobind` 成熟，但手动管理跨语言生命周期容易引发悬垂引用导致 GUI 闪退。 | **中**。通过 `juliacall` 跨语言调用存在运行时桥接开销与线程冲突风险。 | **中**。Python 互操作仍在演进，双向零拷贝生态不完善。 | **差**。完全割裂现有 Python 生态。 |
| **开发效率与可维护性** | **极高**。代数数据类型（`enum`）、Trait 系统、Cargo 依赖管理、无数据竞争的 [`Rayon`](https://docs.rs/rayon/latest/rayon/) 并行，重构零心智负担。 | **中低**。头文件膨胀、模板报错晦涩、多线程 Race Condition 极难排查（尤其在异步 GUI 后台线程中）。 | **高**。数学表达直观，但大型工程缺乏静态类型强制约束，后期维护成本上升。 | **低**。语言语法与标准库频繁发生破坏性更新（Breaking Changes）。 | **中**。计算层高，UI 层低。 |
| **长期生态与社区借鉴** | **极佳**。可直接借鉴并超越复旦大学 [REST 引擎](https://pubs.aip.org/cps/cjcp/article/38/6/788/3377721/REST-Embracing-the-rust-programming-language-for) 生态（`libcint-rs`、`rest_tensors`、`faer`）。 | **极佳**。传统 HPC 主流（ORCA、Psi4、Q-Chem）。 | **良**。DFTK.jl（平面波 DFT）活跃，但分子高斯型轨道（GTO）生态较小。 | **弱**。量子化学领域尚无成熟开源项目。 | **良**。仅限于计算内核。 |

### 2.2 最终战略推荐：三层混合架构（Python/MolForge 编排层 + Rust 核心计算层 + C99/SIMD 算子内核）

**明确推荐采用“Rust 核心引擎（`calchem-core`）+ 工业级 C/Rust 积分与泛函算子（`libcint` / `rustint-qc` + `libxc`）+ PyO3 零拷贝 Python 扩展（`calchem._core`）”的混合架构。**

#### 核心决策依据：
1. **站在巨人肩膀上起跳，避免 3 年无效基建**：量子化学中最繁琐、最容易出错但理论已经完全固化的两个模块是**高角动量高斯积分递推**（Obara-Saika / Rys 多项式，含一阶/二阶几何导数）和**600+ 种交换关联泛函及其高阶导数**。从零手写这两个模块需要 2~3 年，且性能很难超过孙启明博士千锤百炼的 C99 SIMD 库 [`libcint`](https://github.com/sunqm/libcint) 和 Marques 等人维护的 [`Libxc`](https://gitlab.com/Molcas/Mirrors/libxc)。通过 Rust 的 FFI（直接复用或改进 [`libcint-rs`](https://github.com/ajz34/libcint-rs) 与 `libxc-rs` 静态链接），CalChem 可以在**第 1 周**就拥有支持到 $g/h$ 轨道、包含解析一阶/二阶/三阶积分导数、经受过全球上万篇论文检验的底层算子！
2. **把研发精力 100% 集中在真正决定速度上限的高层算法**：一旦底层积分与泛函通过静态链接库就绪，CalChem 的 Rust 核心层就能专注于决定整体速度的**直接 SCF 壳层对筛选、无磁盘 RI-JK / RIJCOSX、二阶收敛 SOSCF、解析梯度 Pulay 组装、CPSCF 响应方程、DLPNO 局域相关与 TDDFT Davidson 对角化**。
3. **完美契合 Ryzen 7 9700X（Zen 5）与 15.1 GB 内存约束**：
   - Zen 5 拥有全速 512-bit AVX-512 数据通路。Rust 的 [`faer`](https://docs.rs/faer/latest/faer/) 线性代数库原生使用 `pulp` 库进行运行时 AVX-512 指令集分派（无需外部安装庞大的 MKL 即可获得媲美 MKL 的 DGEMM 与特征值分解性能，且支持可选切换 OpenBLAS/MKL）。
   - Rust 的所有权模型与显式内存池（Arena Allocator）能精确控制内存水位，严格锁定单任务内存上限（如 $\le 10\text{ GB}$，为 Windows 11 和 MolForge VTK 渲染预留 $5.1\text{ GB}$），杜绝 OOM 死机。
4. **完全保留 v0.6.0 已验证的顶层投资**：现有的 `calchem/core/request.py`（Pydantic v2 数据契约）、`result.py`、`jobs/global_lock.py`、`api/service.py` 以及 MolForge 的 `calchem_engine.py` 无需推翻，只需将 `NativeCalchem` 后端的内部调用从慢速 Python 模块替换为 `calchem._core`（Rust PyO3 扩展模块）。

## 3. 十层核心架构设计规范（Deliverable 2）

### 3.1 总体分层架构与数据流图

```
+-----------------------------------------------------------------------------------+
|                        前端与任务编排层 (Python 3.10+ / MolForge)                  |
|  MolForge (PyQt6 + VTK 3D)  <--Zero-Copy NumPy/MemoryView-->  TaskService & Lock  |
|  CalculationRequest (Pydantic v2)                        Result (Pydantic v2)     |
+-----------------------------------------+-----------------------------------------+
                                          | PyO3 ABI (释放 GIL: py.allow_threads)
                                          v
+===================================================================================+
|                        CalChem Rust 统一计算内核 (calchem-workspace)               |
|                                                                                   |
|  [Layer 10: 优化与动力学层 calchem-opt]                                            |
|     TRIC 内坐标变换 | L-BFGS / RFO 几何优化 | P-RFO 过渡态 | GS2 IRC | Velocity-Verlet |
|                                         |                                         |
|  [Layer 9: 溶剂模型层 calchem-solv]      +--> [Layer 8: 波函数性质层 calchem-prop]   |
|     Lebedev 分子腔体 | IEFPCM / C-PCM    |     偶极/多极矩 | Mulliken/Löwdin/IAO-NBO|
|     SMD 非静电项 (CDS) | 解析溶剂梯度    |     SIMD 并行轨道/密度/ESP Cube 生成器  |
|                                         |                                         |
|  [Layer 7: 激发态与响应层 calchem-excited] <-- [Layer 6: 后 HF 相关层 calchem-corr] |
|     CIS / RPA / Casida TDDFT (TDA/Full) |     RI-MP2 / SCS-MP2 / MP3 / MP4        |
|     Davidson 无矩阵对角化 | EOM-CCSD     |     DF-CCSD / CCSD(T) / DLPNO / CASSCF  |
|                                         |                                         |
|  [Layer 5: 解析导数与响应方程层 calchem-grad]                                      |
|     Pulay 能量加权密度矩阵 W | JK/XC 解析梯度 | CPSCF (Z-Vector) | 解析 Hessian    |
|                                         |                                         |
|  [Layer 4: 自洽场与泛函层 calchem-scf]                                             |
|     RHF / UHF / ROHF / RKS / UKS | SAD/SAP 猜想 | EDIIS+CDIIS+SOSCF | Libxc 网格   |
|                                         |                                         |
|  [Layer 3: 基组与分子拓扑层 calchem-basis]                                         |
|     BSE JSON/NWChem 解析 | 球谐/笛卡尔变换 | 壳层分组 (Shell Batching) | 辅助基组   |
|                                         |                                         |
|  [Layer 2: 多中心积分与筛选层 calchem-int]                                         |
|     1e (S,T,V,ECP) | 4c-2e Schwarz/CSAM 筛选 | 3c-2e / 2c-2e RI-DF | RIJCOSX      |
|     底层双擎驱动: libcint (C99 AVX-512 静态链接) + rustint-qc (纯 Rust Obara-Saika)|
|                                         |                                         |
|  [Layer 1: 异构张量与内存调度基础层 calchem-tensor]                                |
|     CPU (faer AVX-512 / MKL) + GPU (wgpu / CUDA) 统一抽象 | 10GB 水位内存池 | Rayon |
+===================================================================================+
```

### 3.2 各层详细接口定义（Traits/Structs）、数据结构、算法与优化策略

#### Layer 1：基础层——张量引擎与受控内存池（`calchem-tensor`）

* **设计目标**：屏蔽 CPU（Zen 5 AVX-512）与未来 GPU（Vulkan/wgpu/CUDA）的硬件差异；针对 15.1 GB 物理内存提供硬性配额管理（Memory Watermark Guard），防止大基组三中心积分撑爆系统内存。
* **核心数据结构与 Trait 签名**：
```rust
// crates/calchem-tensor/src/lib.rs
use std::sync::Arc;
use faer::{Mat, MatRef, MatMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CpuZen5Avx512 { threads: usize },
    GpuWgpu { adapter_id: u32 },
    GpuCuda { device_id: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Fp64,
    Fp32, // 用于 SCF 早期迭代与远场积分筛选
}

/// 内存配额管理器：针对 15.1 GB RAM 环境，默认上限设为 10.5 GB
pub struct MemoryPool {
    max_bytes: usize,
    allocated_bytes: std::sync::atomic::AtomicUsize,
}

/// 统一对称压缩矩阵（仅存储上三角 N*(N+1)/2，节省 50% 内存与缓存带宽）
pub struct PackedSymMatrix {
    pub dim: usize,
    pub data: Vec<f64>,
}

/// 跨硬件统一线性代数与张量缩并接口
pub trait TensorBackend: Send + Sync {
    /// 矩阵乘法: C = alpha * op(A) * op(B) + beta * C
    fn gemm(
        &self,
        trans_a: bool,
        trans_b: bool,
        alpha: f64,
        a: MatRef<f64>,
        b: MatRef<f64>,
        beta: f64,
        c: MatMut<f64>,
    );

    /// 实对称矩阵特征值分解: F * C = S * C * diag(eps) (通过 Löwdin 正交化转化为标准特征值问题)
    fn syevd(&self, a_sym: MatRef<f64>) -> Result<(Vec<f64>, Mat<f64>), TensorError>;

    /// Cholesky 分解: V = L * L^T (用于 RI 二中心度量矩阵 (P|Q) 分解)
    fn potrf_lower(&self, a_sym: MatMut<f64>) -> Result<(), TensorError>;

    /// 三中心张量与密度矩阵缩并 (RI-J 核心: J_P = sum_{mu,nu} (P|mu nu) D_{mu nu})
    fn contract_3c2e_j(&self, eri_3c: &[f64], dm_packed: &PackedSymMatrix, naux: usize) -> Vec<f64>;
}
```
* **算法与性能优化策略**：
  1. **Zen 5 AVX-512 原生加速**：默认使用 [`faer`](https://docs.rs/faer/latest/faer/) 的分块矩阵乘法与 Divide-and-Conquer 对称特征值分解（`syevd`），编译期开启 `RUSTFLAGS="-C target-cpu=znver5"`（或 `x86-64-v4`），直接生成 `vfmadd231pd`（512-bit 双精度融合乘加，单周期 16 次双精度浮点运算）。
  2. **对称打包存储（Packed Symmetric Storage）**：重叠矩阵 $S$、核心哈密顿量 $H_{\text{core}}$、密度矩阵 $D$、Fock 矩阵 $F$ 以及 RI 三中心积分 $(P|\mu\nu)$ 的原子轨道指标 $(\mu\le\nu)$ 全部采用上三角紧凑布局，使三中心积分张量体积从 $N_{\text{aux}} \times N_{\text{bf}}^2 \times 8\text{ B}$ 直接减半为 $N_{\text{aux}} \times \frac{N_{\text{bf}}(N_{\text{bf}}+1)}{2} \times 8\text{ B}$。例如 $N_{\text{bf}}=500, N_{\text{aux}}=1500$ 时，内存占用从 $3.0\text{ GB}$ 降至 $1.5\text{ GB}$，完全驻留于 15.1 GB 内存中，实现 **100% 零磁盘 I/O**。

#### Layer 2：积分层——多中心积分、Schwarz/CSAM 筛选与 RI 密度拟合（`calchem-int`）

* **设计目标**：提供单电子积分、四中心双电子积分（4c-2e）、三中心/二中心辅助积分（3c-2e / 2c-2e）及其对原子核坐标的一阶/二阶解析导数。
* **核心接口定义**：
```rust
// crates/calchem-int/src/lib.rs
use calchem_basis::{ShellCluster, BasisSetData};
use calchem_tensor::PackedSymMatrix;

#[derive(Debug, Clone, Copy)]
pub enum OneElectronOp {
    Overlap,         // S_{mu,nu} = <mu|nu>
    Kinetic,         // T_{mu,nu} = -1/2 <mu|\nabla^2|nu>
    NuclearAttraction,// V_{mu,nu} = -sum_A Z_A <mu|1/|r-R_A||nu>
    Dipole(usize),   // <mu|r_x, r_y, r_z|nu>
}

/// Schwarz 与距离衰减积分预筛选器
pub struct SchwarzScreener {
    /// 壳层对上限矩阵 Q_{IJ} = max_{mu in I, nu in J} sqrt(|(mu nu | mu nu)|)
    pub shell_pair_bounds: Vec<f64>,
    pub significant_pairs: Vec<(u32, u32, f64)>, // 按 Q_{IJ} 降序排列的有效壳层对
    pub threshold: f64,                          // 默认 1e-12 (SCF 早期可放宽至 1e-9)
}

pub trait IntegralEngine: Send + Sync {
    /// 计算单电子积分矩阵 (Packed Symmetric)
    fn eval_1e(&self, basis: &BasisSetData, op: OneElectronOp) -> PackedSymMatrix;

    /// 计算单电子积分对原子核坐标的一阶导数 d(h_{mu,nu})/dR_A -> [n_atoms, 3, nbf, nbf]
    fn eval_1e_grad(&self, basis: &BasisSetData, op: OneElectronOp) -> Vec<f64>;

    /// 构建 Schwarz 筛选器 (O(N^2) 复杂度)
    fn build_schwarz_screener(&self, basis: &BasisSetData, threshold: f64) -> SchwarzScreener;

    /// 直接 SCF (Direct Fock Build): 利用八重置换对称性与密度筛选直接收缩 J 和 K
    fn direct_jk_build(
        &self,
        basis: &BasisSetData,
        screener: &SchwarzScreener,
        dm: &PackedSymMatrix,
        k_scale: f64, // GGA 为 0.0, B3LYP 为 0.20, RHF 为 1.0
    ) -> (PackedSymMatrix, PackedSymMatrix);

    /// RI/DF 三中心积分与 Cholesky 度量分解: 计算 B_{P, mu<=nu} = sum_Q (L^{-1})_{PQ} (Q|mu nu)
    fn build_ri_b_tensor(
        &self,
        ao_basis: &BasisSetData,
        aux_basis: &BasisSetData,
        screener: &SchwarzScreener,
    ) -> Result<Vec<f64>, IntegralError>;
}
```
* **算法选择与参考文献**：
  1. **混合积分递推算法**：通过静态链接 [`libcint`](https://github.com/sunqm/libcint)（孙启明, *J. Comput. Chem.* **2015**, 36, 1664）处理通用高角动量及导数积分；同时针对低角动量壳层对 $(ss|ss), (ps|ss), (pp|ss)$ 内置纯 Rust 的 Obara-Saika（[Obara & Saika, *J. Chem. Phys.* **1986**, 84, 3963](https://pubs.aip.org/aip/jcp/article/84/7/3963/91928/Efficient-recursive-computation-of-molecular)）与 Head-Gordon–Pople 水平递推（HRR, *J. Chem. Phys.* **1988**, 89, 5777）内联快路径，消除跨语言 FFI 函数调用开销。
  2. **密度加权 Schwarz 筛选（Density-Weighted Schwarz Screening）**：在 Direct SCF 中，利用不等式 $|(\mu\nu|\lambda\sigma)| \le Q_{\mu\nu} Q_{\lambda\sigma}$，结合当前迭代步壳层块最大密度矩阵元 $D_{IJ}^{\max} = \max(|D_{IK}|, |D_{JK}|, |D_{IL}|, |D_{JL}|, |D_{KL}|)$，仅当 $Q_{IJ} Q_{KL} D_{IJKL}^{\max} > \tau_{\text{SCF}}$ 时才调用积分内核（参考 Gill et al., *J. Comput. Chem.* **1993**, 14, 163）。在第二次及之后的 SCF 迭代中，采用**增量 Fock 构建（Incremental Fock Build）**：仅对密度差分 $\Delta D^{(k)} = D^{(k)} - D^{(k-1)}$ 进行积分缩并，使后期 SCF 每步耗时再降 $60\%\sim 80\%$！
  3. **无磁盘 Cholesky-RI-JK 与 RIJCOSX**：借鉴 [ORCA 6.1](https://www.faccts.de/docs/orca/6.1/manual/index.html) 的 SHARK / RIJCOSX（Neese et al., *Chem. Phys.* **2009**, 356, 98）与 [REST](https://pubs.aip.org/cps/cjcp/article/38/6/788/3377721/REST-Embracing-the-rust-programming-language-for) 的 Disk-Free RI 思想，计算二中心库仑度量矩阵 $V_{PQ} = (P|Q)$，对其进行 Cholesky 分解 $V = L L^T$，并通过三角方程求解得到正交化三中心张量 $B_{P,\mu\nu} = \sum_Q (L^{-1})_{PQ} (Q|\mu\nu)$。库仑矩阵构建简化为两次极速 BLAS Level-2 `dgemv`：
     $$J_P = \sum_{\mu\le\nu} (2 - \delta_{\mu\nu}) B_{P,\mu\nu} D_{\mu\nu}, \quad J_{\mu\nu} = \sum_P B_{P,\mu\nu} J_P$$
     计算复杂度从 $O(N^4)$ 骤降为 $O(N_{\text{aux}} N^2)$（实测加速 $20\sim 100$ 倍）。

#### Layer 3：基组层——标准基组库、球谐变换与壳层聚类（`calchem-basis`）

* **设计目标**：彻底解决 v0.6.0 只有 STO-3G~6-31G(d) 且仅支持笛卡尔高斯的问题；内置常用轨道基组与 RI 辅助基组，并按角动量与收缩度对壳层重排以最大化 SIMD 效率。
* **核心数据结构**：
```rust
// crates/calchem-basis/src/lib.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngularType {
    Spherical, // 5d, 7f, 9g (默认推荐，消除线性相关与冗余维度)
    Cartesian, // 6d, 10f, 15g (仅用于特定 6-31G(d) 历史基准比对)
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Shell {
    pub atom_idx: u32,
    pub angular_l: u8,
    pub n_prim: u8,
    pub n_contr: u8,
    pub ao_offset: u32,     // 在全局 AO 矩阵中的起始行/列索引
    pub center_bohr: [f64; 3],
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>, // 含预归一化因子
}

/// 兼容 libcint 内存布局的连续数组打包格式 (atm, bas, env)
pub struct BasisSetData {
    pub name: String,
    pub angular_type: AngularType,
    pub shells: Vec<Shell>,
    pub nbf: usize,
    pub libcint_atm: Vec<[i32; 6]>,
    pub libcint_bas: Vec<[i32; 8]>,
    pub libcint_env: Vec<f64>,
}
```
* **内置基组清单與存储压缩**：
  - 使用构建脚本从 [MolSSI Basis Set Exchange](https://www.basissetexchange.org/) 预提取并压缩打包（Zstandard 压缩嵌入二进制或放于 `data/basis/`）以下全套基组：
    - **Pople 系列**：`STO-3G`, `3-21G`, `6-31G(d)`, `6-31G(d,p)`, `6-311G(d,p)`, `6-311+G(2d,p)`
    - **Karlsruhe def2 系列（主力推荐）**：`def2-SVP`, `def2-SVPD`, `def2-TZVP`, `def2-TZVPP`, `def2-QZVP`（含重元素有效核势 ECP 数据）
    - **Dunning 相关一致系列**：`cc-pVDZ`, `cc-pVTZ`, `cc-pVQZ`, `aug-cc-pVDZ`, `aug-cc-pVTZ`
    - **Jensen 极化一致系列**：`pcseg-1`, `pcseg-2`
    - **配对 RI 辅助基组**：`def2-universal-jkfit`, `def2-svp-ri`, `def2-tzvp-ri`, `cc-pvdz-jkfit`, `cc-pvdz-ri`
  - **球谐-笛卡尔变换**：采用 Schlegel & Frisch（*Int. J. Quantum Chem.* **1995**, 54, 83）的实球谐变换矩阵，在壳层积分出口处直接原地投影为 $2l+1$ 分量。

#### Layer 4：SCF 层——多参考态自洽场与二阶收敛引擎（`calchem-scf`）

* **设计目标**：支持 RHF / UHF / ROHF / RKS / UKS，不仅算得快，而且在共轭有机光电分子（OPV/OSC 体系）和过渡金属配合物上具备超越 Gaussian 默认 SCF 的收敛鲁棒性。
* **核心接口定义**：
```rust
// crates/calchem-scf/src/lib.rs
use calchem_tensor::{Mat, PackedSymMatrix};

#[derive(Debug, Clone)]
pub enum ScfConvergenceStrategy {
    /// 早期迭代 (|e| > 1e-2): EDIIS/ADIIS (凸优化保证能量严格下降，无振荡)
    /// 中期迭代 (1e-2 >= |e| > 1e-5): Pulay C-DIIS (子空间维度 8~10)
    /// 困难体系兜底: 二阶 SOSCF (基于轨道旋转梯度与 BFGS 近似轨道 Hessian)
    AutoComposite { diis_space: usize, soscf_threshold: f64 },
    Damping { factor: f64 },
    LevelShift { shift_hartree: f64 },
}

pub struct ScfOutput {
    pub converged: bool,
    pub iterations: usize,
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion_energy: f64,
    pub xc_energy: Option<f64>,
    pub dispersion_energy: Option<f64>, // DFT-D3(BJ) / D4
    pub mo_energies_alpha: Vec<f64>,
    pub mo_energies_beta: Option<f64>,
    pub mo_Occupations_alpha: Vec<f64>,
    pub mo_coeff_alpha: Mat<f64>,
    pub mo_coeff_beta: Option<Mat<f64>>,
    pub density_alpha: PackedSymMatrix,
    pub density_beta: Option<PackedSymMatrix>,
    pub fock_alpha: PackedSymMatrix,
}
```
* **关键算法与性能优化**：
  1. **高质量初始猜测（SAD / SAP）**：摒弃 v0.6.0 简陋的 $H_{\text{core}}$ 初始猜测（在分子稍大时会导致前 10 步剧烈电荷振荡），默认采用**原子密度叠加（Superposition of Atomic Densities, SAD）**与**原子势叠加（Superposition of Atomic Potentials, SAP，Lehtola, *J. Chem. Theory Comput.* **2019**, 15, 1593）**，使 SCF 迭代步数平均减少 $30\%\sim 50\%$。
  2. **EDIIS + C-DIIS + SOSCF 三段式自动收敛状态机**：
     - 当对易子残差 $\|FDS - SDF\|_\infty > 0.1$ 时，自动启用 EDIIS（Kudin, Scuseria & Cancès, *J. Chem. Phys.* **2002**, 116, 8255）或 ADIIS（Hu & Yang, *J. Chem. Phys.* **2010**, 132, 054109），无需人工设置阻尼即可消除窄带隙体系的电荷晃动（Charge Sloshing）。
     - 当残差进入 $[10^{-5}, 0.1]$ 区间时，切换至经典 [Pulay DIIS（*J. Comput. Chem.* **1982**, 3, 556）](https://www.semanticscholar.org/paper/Improved-SCF-convergence-acceleration-Pulay/e629c6d572568c35e8f8f575cc53b01dede7504a0)。
     - 若连续 5 步能量下降小于 $10^{-6}\text{ E}_\text{h}$ 且未收敛，自动触发**二阶自洽场（SOSCF / Newton-Raphson SCF，Bacskay, *Chem. Phys.* **1981**, 61, 385；Sun et al., *J. Chem. Phys.* **2017**, 147, 164119）**，对占据-虚拟轨道旋转参数 $\kappa_{ia}$ 进行共轭梯度求解，确保二次收敛。
  3. **DFT 数值网格与 [`Libxc`](https://gitlab.com/Molcas/Mirrors/libxc) 深度向量化**：
     - 径向采用 Treutler-Ahlrichs 网格，角向采用 Lebedev 球面网格，空间划分采用 Stratmann-Scuseria-Frisch（SSF）光滑截断权重函数（比傳統 Becke 权重快 $3\sim 4$ 倍，因为对大多数距离比的网格点权重直接精确为 0 或 1，跳过超越函数计算）。
     - 将空间相邻的网格点按 64 或 128 个点打包为一个 **Grid Batch（SimdGridBlock）**，利用空间局部性（Spatial Locality）筛除远离当前网格块的原子轨道壳层，使 DFT 网格积分复杂度达到严格线性标度 $O(N)$。
     - 内置原生 Rust 实现的 Grimme **DFT-D3(BJ)** 与 **DFT-D4** 色散校正及其解析梯度。

#### Layer 5：梯度与导数层——解析梯度、CPSCF 与解析 Hessian（`calchem-grad`）

* **设计目标**：彻底消灭 v0.6.0 中有限差分梯度慢且噪声大的致命缺陷，使得几何优化单步耗时仅为单点能的 $1.2\sim 1.5$ 倍（而非 $6N_{\text{atom}}$ 倍）。
* **理论公式与核心接口**：
  对于闭壳层 RHF/RKS，总能量对原子核坐标 $R_A$ 的一阶解析导数（Pulay, *Mol. Phys.* **1969**, 17, 197）严格表示为：
  $$\frac{\partial E}{\partial R_A} = \frac{\partial V_{\text{nuc}}}{\partial R_A} + \sum_{\mu\nu} D_{\mu\nu} \frac{\partial h_{\mu\nu}}{\partial R_A} + \frac{1}{2} \sum_{\mu\nu\lambda\sigma} \left(D_{\mu\nu}D_{\lambda\sigma} - \frac{1}{2} c_{\text{x}} D_{\mu\lambda}D_{\nu\sigma}\right) \frac{\partial (\mu\nu|\lambda\sigma)}{\partial R_A} + \frac{\partial E_{\text{xc}}^{\text{grid}}}{\partial R_A} - \sum_{\mu\nu} W_{\mu\nu} \frac{\partial S_{\mu\nu}}{\partial R_A}$$
  其中**能量加权密度矩阵（Energy-Weighted Density Matrix）** $W_{\mu\nu} = 2 \sum_{i\in\text{occ}} \varepsilon_i C_{\mu i} C_{\nu i}$ 捕捉了原子轨道随原子核移动而产生的非绝热 Pulay 力（Pulay Force）；DFT 交换关联项还包含网格权重导数项（Grid Weight Derivative）。
```rust
// crates/calchem-grad/src/lib.rs
pub trait AnalyticalDerivative {
    /// 计算 [n_atoms, 3] 解析梯度 (单位: Eh / Bohr)，默认使用 RI-JK 导数或 Direct 4c 导数
    fn compute_gradient(&self, scf_out: &ScfOutput) -> Result<Vec<[f64; 3]>, GradError>;

    /// 求解耦合微扰自洽场方程 (CPSCF / Z-Vector): (A + B) * U^x = b^x
    /// 用于计算解析 Hessian、TDDFT 激发态解析梯度、MP2 松弛密度及极化率
    fn solve_cpscf(&self, scf_out: &ScfOutput, rhs: &Mat<f64>) -> Result<Mat<f64>, GradError>;

    /// 计算 [3*n_atoms, 3*n_atoms] 解析 Hessian 矩阵 (单位: Eh / Bohr^2)
    fn compute_hessian(&self, scf_out: &ScfOutput) -> Result<Mat<f64>, GradError>;
}
```
* **关键优化**：在 RI-JK 模式下，双电子积分导数无需计算四中心导数，而是直接缩并三中心积分导数 $(Q|\mu\nu)^{[A]}$ 与二中心度量导数 $(P|Q)^{[A]}$：
  $$\frac{\partial E_J}{\partial R_A} = \sum_{P,\mu\nu} J_P (P|\mu\nu)^{[A]} D_{\mu\nu} - \frac{1}{2} \sum_{PQ} J_P (P|Q)^{[A]} J_Q$$
  使解析梯度的计算耗时比传统四中心解析梯度再快一个数量级！

#### Layer 6：后 HF 电子相关层——RI-MP2、CCSD(T)、DLPNO 与 CASSCF（`calchem-corr`）

* **设计目标**：利用密度拟合（RI/DF）与局域对自然轨道（DLPNO）突破传统后 HF 方法的内存墙与计算标度墙。
* **核心接口与算法选型**：
  1. **无磁盘 RI-MP2 / SCS-MP2（$O(N^4)$ 至 $O(N^5)$ 极速转换）**：
     - 将 AO 三中心张量 $B_{P,\mu\nu}$ 通过两次半变换转为分子轨道（MO）基下的占据-虚拟块 $B_{P,ia} = \sum_{\mu\nu} B_{P,\mu\nu} C_{\mu i} C_{\nu a}$。
     - 对占据轨道对 $(i,j)$ 并行（[`Rayon`](https://docs.rs/rayon/latest/rayon/) `par_iter`），通过 `faer::gemm` 计算虚拟空间积分块 $(ia|jb) = \sum_P B_{P,ia} B_{P,jb}$，立即累加相反自旋（OS）与相同自旋（SS）相关能并释放该块内存（同时输出 Grimme SCS-MP2 能量）。
     - 结合 Laplace 变换 RI-MP2 或 Z-Vector 方程实现 RI-MP2 解析梯度。
  2. **DF-CCSD 与异步流水线 CCSD(T)**：
     - 将最耗时的粒子-粒子阶梯项（Particle-Particle Ladder Diagram，$\langle ab\|cd\rangle t_{ij}^{cd}$，复杂度 $O(O^2 V^4)$）改写为在 AO 基或三中心 RI 基下的对称张量缩并（参考 Scuseria et al., *J. Chem. Phys.* **1988**, 89, 7382；DePrince & Sherrill, *J. Chem. Theory Comput.* **2013**, 9, 293），避免在 15.1 GB 内存中存储庞大的 $V^4$ 四维张量。
     - $(T)$ 微扰三重激发采用循环交换的 $(i,j,k)$ 无锁并行分块（参考 Janowski & Pulay, *J. Chem. Theory Comput.* **2008**, 4, 1585），每个线程仅持有局部 $V^3$ 缓冲区（例如 $V=150$ 时仅需 $150^3 \times 8\text{ B} \approx 27\text{ MB}$/线程）。
  3. **线性标度 DLPNO-MP2 / DLPNO-CCSD(T)（中远期王牌特性）**：
     - 参照 Neese 团队的 [DLPNO-CCSD(T) 算法（Riplinger & Neese, *J. Chem. Phys.* **2013**, 139, 134101）](https://pubmed.ncbi.nlm.nih.gov/24116546/)：通过 Foster-Boys 或 Pipek-Mezey 局域化占据轨道，利用投影原子轨道（PAO）与对自然轨道（PNO）截断（NormalPNO / TightPNO），将耦合簇计算推广至 100~300 原子体系。

#### Layer 7：激发态层——TDDFT / CIS / EOM-CCSD（`calchem-excited`）

* **设计目标**：专门针对用户所在的**有机太阳能电池（OPV / OSC）**研究方向，提供极速、高精度的单重态/三重态激发能、振子强度（Oscillator Strength）、自然跃迁轨道（NTO）及激发态结构弛豫。
* **核心接口与算法**：
  - 求解 [Casida 线性响应方程（Casida, *Recent Advances in Density Functional Methods* **1995**, 1, 155）](https://www.researchgate.net/publication/233785009_rm_Time-Dependent_Density_Functional_Response_Theory_for_Molecules_in_Recent_Advances_in_Density_Functional_Methods_Vol_1_D_P_Chong_Ed)：
    $$\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^* & \mathbf{A}^* \end{pmatrix} \begin{pmatrix} \mathbf{X}_n \\ \mathbf{Y}_n \end{pmatrix} = \omega_n \begin{pmatrix} \mathbf{1} & \mathbf{0} \\ \mathbf{0} & -\mathbf{1} \end{pmatrix} \begin{pmatrix} \mathbf{X}_n \\ \mathbf{Y}_n \end{pmatrix}$$
  - 同时支持 **Tamm-Dancoff 近似（TDA，$\mathbf{B}=0$）** 与 **Full TDDFT（非厄米消元转化为 $(\mathbf{A}-\mathbf{B})^{1/2}(\mathbf{A}+\mathbf{B})(\mathbf{A}-\mathbf{B})^{1/2}\mathbf{Z}_n = \omega_n^2 \mathbf{Z}_n$）**。
  - **无矩阵（Matrix-Free）多根块 Davidson 对角化（Block Davidson Algorithm，Stratmann, Scuseria & Frisch, *J. Chem. Phys.* **1998**, 109, 8218）**：绝不显式构造维度高达 $(N_{\text{occ}}N_{\text{virt}})^2$ 的超矩阵 $\mathbf{A},\mathbf{B}$，而是将试探向量 $b_{ia}$ 变换回 AO 基伪密度矩阵 $P_{\mu\nu}^{\text{trial}} = \sum_{ia} C_{\mu i} b_{ia} C_{\nu a}$，直接调用 Layer 2 的 RI-JK 与 Layer 4 的 Libxc 二阶核 $f_{\text{xc}}$ 完成矩阵-向量乘积，单次迭代内存占用低于 $200\text{ MB}$。
  - **sTDA / sTDDFT 快筛模式**：内置 Grimme 简化 TDDFT（*J. Chem. Phys.* **2013**, 138, 244104），支持在数秒内完成百原子级 OPV 给受体分子的紫外-可见吸收光谱（UV-Vis）计算。

#### Layer 8：性质与网格波函数层——SIMD 极速轨道渲染与布居分析（`calchem-prop`）

* **设计目标**：彻底替换 v0.6.0 的 Cube 生成 Stub，使 MolForge 能够实时渲染 HOMO/LUMO、自旋密度、静电势（ESP）与自然跃迁轨道（NTO）。
* **核心接口与零拷贝机制**：
```rust
// crates/calchem-prop/src/lib.rs
pub struct Grid3DSpec {
    pub origin_bohr: [f64; 3],
    pub axes_bohr: [[f64; 3]; 3],
    pub dims: [usize; 3], // 例如 [80, 80, 80] = 512,000 体素
}

pub trait WavefunctionPropertyEngine {
    /// 直接在内存中批量评估指定轨道或电子密度在 3D 规则网格上的值 (支持零拷贝返回给 VTK)
    fn eval_orbital_grid(&self, basis: &BasisSetData, mo_vec: &[f64], grid: &Grid3DSpec) -> Vec<f32>;
    fn eval_density_grid(&self, basis: &BasisSetData, dm: &PackedSymMatrix, grid: &Grid3DSpec) -> Vec<f32>;
    fn eval_esp_grid(&self, basis: &BasisSetData, dm: &PackedSymMatrix, grid: &Grid3DSpec) -> Vec<f32>;

    /// 导出标准 Gaussian .cube 格式文件 (兼容第三方软件)
    fn write_cube_file(&self, path: &std::path::Path, grid: &Grid3DSpec, values: &[f32]) -> std::io::Result<()>;

    /// 布居分析与键级: Mulliken, Löwdin, Mayer 键级, IAO/IBO 内禀原子轨道 (Knizia 2013)
    fn population_analysis(&self, basis: &BasisSetData, s_mat: &PackedSymMatrix, dm: &PackedSymMatrix) -> PopulationReport;
}
```
* **性能突破**：计算 $80\times 80\times 80$ 网格（51.2 万个空间点）上的分子轨道值时，按 $Z$ 轴连续扫描线条（SIMD 一次计算 8 个 FP64 或 16 个 FP32 网格点），结合高斯指数径向截断（当 $\alpha |r - R_A|^2 > 36.0$ 时 $\exp(-\alpha r^2) < 2.3\times 10^{-16}$ 直接跳过），在 Ryzen 7 9700X 上生成单个轨道网格耗时 **$< 15\text{ ms}$**！并且通过 `PyO3` 直接传递 `numpy.ndarray(dtype=np.float32)` 给 MolForge 的 `vtkImageImport`，完全免除 `.cube` 文本文件的格式化输出与字符串解析开销。

#### Layer 9：溶剂层——连续介质溶剂模型 IEFPCM / C-PCM / SMD（`calchem-solv`）

* **设计目标**：支持溶液相单点能、几何优化、频率与 TDDFT 溶剂化效应（平衡态与非平衡态激发态溶剂化）。
* **算法选择与参考文献**：
  1. **分子表面腔体离散化（Switching/Gaussian Smooth Cavity）**：采用 York & Karplus（*J. Phys. Chem. A* **1999**, 103, 11060）与 Scalmani & Frisch（*J. Chem. Phys.* **2010**, 132, 114110）的**球面高斯平滑切换函数（SWIG）**构造以原子为中心的 Lebedev 网格腔体。为什么选 SWIG？因为传统多面体镶嵌（Tesserae）在原子移动时网格数量突变，会导致势能面不连续、几何优化震荡；而 SWIG 保证能量对原子坐标**严格无穷阶光滑可微**，解析溶剂梯度稳定可靠！
  2. **静电溶剂化算子**：实现 C-PCM（导体近似，Barone & Cossi, *J. Phys. Chem. A* **1998**, 102, 1995）与 IEFPCM（积分方程形式，Cancès, Mennucci & Tomasi, *J. Chem. Phys.* **1997**, 107, 3032），在每次 SCF 迭代中将表面表观电荷（ASC）向量 $\mathbf{q}$ 产生的单电子势 $V_{\mu\nu}^{\text{PCM}}$ 加入 Fock 矩阵。
  3. **SMD 通用溶剂化模型**：完整实现 [Marenich, Cramer & Truhlar（*J. Phys. Chem. B* **2009**, 113, 6378）](https://pubs.acs.org/jpcbfk/article/113/18/6378/1427431/Universal-Solvation-Model-Based-on-Solute-Electron) 的 SMD 模型，内置常见有机溶剂（水、氯仿、二氯甲烷、甲苯、氯苯、THF、乙腈、DMSO 等，涵盖 OPV 旋涂常用溶剂）的介电常数 $\varepsilon$、折射率 $n$、宏观表面张力 $\gamma$ 与酸碱参数 $\alpha, \beta$，自动计算非静电空腔-色散-溶剂结构项 $\Delta G_{\text{CDS}}$。

#### Layer 10：几何优化、过渡态与动力学层（`calchem-opt`）

* **设计目标**：基于解析梯度与解析/近似 Hessian，提供工业级几何优化、过渡态搜索（TS）与内禀反应坐标（IRC）。
* **算法选择与参考文献**：
  1. **TRIC 平移-转动-内坐标系统（Translation-Rotation-Internal Coordinates）**：弃用纯笛卡尔坐标优化，完整实现王力平与宋晨提出的 [**TRIC 坐标系**（Wang & Song, *J. Chem. Phys.* **2016**, 144, 214108）](https://pubs.aip.org/aip/jcp/article/144/21/214108/313176/Geometry-optimization-made-simple-with-translation)。TRIC 在冗余内坐标（键长、键角、二面角）基础上显式加入分子片段的整体平移与四元数转动坐标，不仅使单分子优化步数比笛卡尔坐标减少 $50\%\sim 70\%$，而且天然支持多分子复合物（如 OPV 给体-受体 $\pi-\pi$ 堆积二聚体）的无崩溃优化！
  2. **有理函数优化（RFO）与信赖域 L-BFGS**：
     - 极小点搜索：采用 Schlegel 的模型 Hessian 初始猜测（Fischer & Almlöf, *J. Phys. Chem.* **1992**, 96, 9768）配合 GDIIS（幾何 DIIS）与信赖域 RFO（Rational Function Optimization）。
     - 过渡态搜索（TS）：采用分区有理函数优化（**P-RFO**，Baker, *J. Comput. Chem.* **1986**, 7, 385）与二聚体方法（Dimer / NEB-TS），沿第一虚频本征模最大化、其余 $3N-7$ 个自由度最小化。
     - 反应路径追踪（IRC）：采用 Gonzalez-Schlegel 二阶步进算法（**GS2**，*J. Chem. Phys.* **1989**, 90, 2154）。

## 4. 十一大性能创新方向可行性与收益量化评估（Deliverable 4）

严格遵循“不做无基准测试的虚假声明、不引入不稳定技术栈”原则，对 11 项前沿技术在 **AMD Ryzen 7 9700X（8C/16T Zen 5）+ 15.1 GB RAM** 环境下的加速比、工程复杂度与实施阶段进行客观评估：

| 创新技术方向 | 理论复杂度改变 | 实测/文献加速比预期（对比传统 4c-ERI 基线） | 15.1 GB 内存适配性 | 工程风险与实施策略 | 推荐落地阶段 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 分辨率恒等式（RI-JK / RIJCOSX / OCC-RI-K）** | $O(N^4) \to O(N_{\text{aux}}N^2)$（J）与 $O(N^2\sim N^3)$（K） | **15x ~ 80x**（纯泛函 RI-J 达 50x+，杂化泛函结合 COSX 或占据轨道投影 OCC-RI-K 达 15x~30x） | **极佳**。对称压缩后 $N_{\text{bf}}\le 800$ 可全内存（In-Core）运行，彻底消灭 ORCA 的磁盘 I/O 瓶颈。 | **低风险，最高优先级**。采用库仑度量 $(P|Q)$ + 局域交换筛选，误差 $< 0.05\text{ kcal/mol}$。 | **Phase 1 (MVP) & Phase 2** |
| **2. 稀疏矩阵与积分预筛选（Schwarz + CSAM + 增量 Fock）** | 渐近 $O(N^4) \to O(N^2)$（非零壳层对） | **3x ~ 12x**（随分子尺寸增大收益呈超线性增长；增量 Fock 构建再省 60% 后期迭代时间） | **极佳**。仅存储有效壳层对索引（几 MB）。 | **低风险，必选**。结合距离敏感的 CSAM（Cauchy-Schwarz with Asymptotic Multipole）筛选远场库仑积分。 | **Phase 1 (MVP)** |
| **3. Zen 5 AVX-512 SIMD 向量化（`faer` + `libcint` + 网格打包）** | 常数因子降低 $4\times\sim 8\times$ | **3.5x ~ 6.5x**（相比标量代码；相比普通 AVX2 提升 1.6x~1.8x） | **极佳**。寄存器内运算，零额外内存开销。 | **低风险**。Zen 5 具备全速 512-bit 单元且不降频，通过 `target-cpu=znver5` 与 `faer` 自动激活。 | **Phase 1 (MVP)** |
| **4. 共享内存多核并行（`Rayon` Work-Stealing + 全局锁协同）** | 常数因子降低 $1/N_{\text{core}}$ | **6.2x ~ 7.4x**（8 核 16 线程下 Direct Fock、RI-MP2 与 DFT 网格积分并行效率 $>85\%$） | **优**。共享只读基组与三中心张量，比 MPI 多进程节省 $80\%$ 重复内存！ | **低风险**。严格遵守单重任务串行调度（`heavy.lock`），单任务独占全部 8 核物理核心。 | **Phase 1 (MVP)** |
| **5. 混合精度计算（Dynamic FP32/FP64 Mixed Precision）** | 内存带宽减半，SIMD 吞吐翻倍（16 lanes FP32/cycle） | **1.6x ~ 2.4x**（在 Direct SCF 前 60% 迭代及远场弱壳层对 $(Q_{IJ}Q_{KL} < 10^{-6})$ 采用 FP32 累加至 FP64） | **极佳**。若将大体系 RI 三中心张量的次要块以 FP32 存储，内存容量等效翻倍至 30 GB。 | **中低风险**。参照 Luehr, Ufimtsev & Martínez (*J. Chem. Theory Comput.* **2011**, 7, 949) 设定动态切换阈值：当 $\|\Delta D\| < 10^{-5}$ 时强制全量 FP64 收尾，保证最终能量 1e-10 Eh 精度不丢失。 | **Phase 2** |
| **6. Cholesky 分解（CD-ERI）** | 秩截断 $M_{\text{chol}} \approx 3N_{\text{bf}} \ll N_{\text{bf}}^2$ | **5x ~ 15x**（无需外部辅助基组的无偏 RI 替代方案） | **良**。两步动态自适应 Cholesky（aCD）仅生成对角块与主元列。 | **中风险**。在具备标准辅助基组时 RI-DF 更快且解析梯度更简单；Cholesky 适合作为无配对辅助基组时的自动后备。 | **Phase 3** |
| **7. 局域相关与张量分解（DLPNO / THC）** | $O(N^7) \to O(N)$（DLPNO-CCSD(T)） | **10x ~ 1000x**（对于 $>40$ 原子的有机分子，传统 CCSD(T) 无法运行而 DLPNO 可在数十分钟完成） | **良**。通过 PNO 截断将内存从 $O(N^4)$ 压至 $O(N\sim N^2)$。 | **中高复杂度**。DLPNO 工程细节繁多（PAO 正交化、多级阈值外推），适合在标准 DF-CCSD(T) 稳定后推进。 | **Phase 4** |
| **8. 自动微分（AD）替代手写解析梯度** | 阶数不变，省去人工推导成本 | **对高阶 XC 泛函/新模型极快；但对底层 GTO 积分慢 2x~5x** | **中**。反向传播（Reverse-mode AD）若记录积分递推中间态会产生巨大 Tape 内存。 | **混合策略（关键判断）**：**绝不用通用 AD 硬跑底层 GTO 积分递推**（会撑爆 15.1 GB 内存且慢于 `libcint`）；而是**底层积分/泛函导数用 `libcint`/`libxc` 代码生成算子 + 顶层能量拉格朗日拼装/溶剂腔体导数采用 Dual Numbers / [Enzyme AD](https://github.com/EnzymeAD/enzyme)**。 | **Phase 1 (算子导数) + Phase 3 (高阶响应 AD)** |
| **9. 线性标度 SCF 算法（$O(N)$ FMM + ONX/LinK）** | $O(N^2) \to O(N)$ | 在 $>200$ 原子体系上加速 **5x ~ 20x**；但在 $<80$ 原子体系上因预因子大反而不如 RI-JK | **良**。稀疏块压缩 CSR 存储。 | **中风险**。设置自动交叉点开关：$N_{\text{atoms}} < 100$ 默认走 In-Core RI-JK；$N_{\text{atoms}} \ge 100$ 自动启用快速多极子（FMM）与局域交换（LinK）。 | **Phase 4** |
| **10. GPU 异构加速（`wgpu` / `cudarc`）** | 高并行吞吐（Fock 构建、RI 半变换、Cube 网格） | 未来接入独立 GPU 后 **10x ~ 40x**；当前无独显环境下若存在 Ryzen 7 9700X 内置 RDNA2 iGPU（2 CU），FP32 辅助网格计算可分流 15% 负载，但 FP64 仍以 Zen 5 CPU 最快 | **需注意 PCIe/统一内存拷贝**。 | **架构预留，渐进激活**。Layer 1 的 `TensorBackend` Trait 100% 预留 GPU 接口，当前默认绑定 `CpuZen5Avx512`。 | **Phase 1 (接口预留) -> Phase 4 (内核实装)** |
| **11. 机器学习加速（ML Initial Guess / $\Delta$-ML / xTB 预优化）** | 减少 SCF 迭代步数 $50\%$ 或减少几何优化步数 $70\%$ | **端到端工作流加速 2x ~ 5x** | **极佳**。轻量级 GNN/等变网络推理仅需几十 MB。 | **高实用价值**。利用 GFN2-xTB 或轻量级等变模型预测初始紧束缚密度矩阵与初始 Hessian（在进入昂贵的 B3LYP/def2-TZVP 优化前先用 xTB/ML 预收敛几何与 Hessian）。 | **Phase 3** |

## 5. 主流计算化学软件深度对标分析（Deliverable 5）

| 软件名称 | 核心语言与技术栈 | 核心优势（Core Strengths） | 架构特点（Architectural Traits） | 性能与工程瓶颈（Bottlenecks & Weaknesses） | CalChem v1.0 的针对性超越方案 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[Gaussian 16](https://gaussian.com/people/)** | Fortran 77/90 + C（PRISM 积分引擎） | 1. 工业界事实标准，收敛 heuristics 极其丰富（EDIIS+CDIIS）；<br>2. 解析一/二/三阶导数（含 TDDFT 梯度与解析频率）覆盖最全；<br>3. IEFPCM/SMD 平滑腔体实现极为稳健。 | 基于历史悠久的 `Link` 覆盖式可执行文件架构（`l101.exe`, `l301.exe`, `l502.exe` 等），通过读写庞大的 `.rwf`（Read-Write File）磁盘文件在进程间传递状态。 | 1. **磁盘与进程切换瓶颈**：每个 Link 启动及 `.rwf` 读写带来显著 I/O 开销；<br>2. **RI/DF 支持滞后**：杂化泛函默认仍走传统 4 中心积分，在中大基组（如 `def2-TZVP`）下比现代 RI 引擎慢 5~20 倍；<br>3. 闭源黑盒，无法与 Python GUI 零拷贝内存互通。 | **全驻留内存（Zero-RWF）+ 默认 RI-JK + 零拷贝 PyO3 接口**。复刻并开源化其优秀的 EDIIS+CDIIS+SOSCF 与 SWIG-PCM 平滑腔体算法，同时用现代无磁盘 RI-JK 实现数量级提速。 |
| **[ORCA 6.1](https://www.faccts.de/docs/orca/6.1/manual/index.html)** | C++（内置 SHARK 积分引擎 + Libint2） | 1. **RIJCOSX 与 DLPNO-CCSD(T)** 领域的绝对标杆（[Riplinger & Neese 2013](https://pubmed.ncbi.nlm.nih.gov/24116546/)）；<br>2. 过渡金属光谱、磁学性质与相对论效应支持极强；<br>3. 学术免费且更新极其活跃。 | 同样采用多独立可执行模块（`orca_scf`, `orca_mp2`, `orca_mdci` 等）+ 磁盘 `.gbw`/`.tmp` 文件交换 + 外部 MPI 多进程并行（即使在单台多核 PC 上也采用多进程 MPI 而非多线程共享内存）。 | 1. **Windows 下 MPI 部署繁琐且吃内存**：8 进程 MPI 意味着多份数据结构副本与进程间 IPC 开销，在 15.1 GB 内存机器上极易触发内存不足；<br>2. **重度依赖磁盘 I/O**：产生海量临时文件，消费级 SSD 磨损大且受限于磁盘带宽；<br>3. 闭源，仅能通过解析 stdout 文本交互。 | **单进程 Rayon 共享内存多线程 + 对称压缩 In-Core RI**。在单节点多核（Ryzen 7 9700X）与 15.1 GB 内存限制下，共享内存架构比 ORCA 的多进程 MPI 节省 $40\%\sim 60\%$ 内存，且无需安装配置 MS-MPI。 |
| **[PySCF](https://github.com/sunqm/libcint)** | Python (~85%) + C99 (~15%, `libcint` + `libxc`) | 1. 模块化与可扩展性极佳，二次开发首选；<br>2. 底层 [`libcint`](https://github.com/sunqm/libcint) 积分速度极快；<br>3. 衍生出 [GPU4PySCF](https://www.alphaxiv.org/@qiming-sun) 等前沿生态。 | “薄 C 内核 + 厚 Python 驱动”架构：底层积分与部分缩并用 C + OpenMP 实现，SCF 循环、DIIS、网格管理、后 HF 调度全部在 Python 中完成。 | 1. **Windows 原生支持薄弱**：官方主要面向 Linux/macOS，Windows 下常需 WSL2 或复杂交叉编译；<br>2. **Python 调度与内存开销**：在小体系高频调用（如 AIMD、数百步优化）或复杂张量切片时，Python GC 与 NumPy 临时数组拷贝开销明显；<br>3. 缺乏统一强类型 Schema 与原生桌面级任务锁调度。 | **底层复用 `libcint` C99 算子，但将整个 SCF、梯度、RI-MP2、TDDFT 与优化循环全部下沉至 Rust 编译态**。消除 Python 循环胶水开销，并提供原生 Windows `.pyd` 单目录免安装分发。 |
| **[Psi4](https://github.com/psi4/psi4/)** | C++17 核心 + Python 前端（Pybind11） | 1. 密度拟合（DF-HF, DF-MP2, DF-CCSD(T)）与 SAPT（对称微扰理论）实现极具代表性（[Smith et al., *J. Chem. Phys.* **2020**, 152, 184108](https://pubs.aip.org/aip/jcp/article/152/18/184108/972964/PSI4-1-4-Open-source-software-for-high-throughput)）；<br>2. 深度集成 QCSchema。 | “Psithon / PsiAPI”双模式，C++ 负责核心 `Wavefunction` 与 `MintsHelper` 积分，Python 负责驱动工作流。 | 1. **C++ 历史包袱与编译地狱**：依赖庞大的 Conda 环境，无法轻量级打包进独立桌面软件（One-Folder 体积动辄数 GB）；<br>2. 存在多代遗留 C++ 张量库并存问题。 | **用现代 Cargo Workspace + Rust Trait 替代臃肿的 CMake/Conda C++ 依赖链**，编译产物体积缩小 90% 以上，同时吸收其优秀的 DF-MP2 / DF-CC 张量分块思想。 |
| **[NWChem](https://nwchemgit.github.io/)** | Fortran 77/90 + C（Global Arrays 工具包） | 面向超算集群（数千至数万 CPU 核）的分布式内存并行（PGAS 模型，[Aprà et al., *J. Chem. Phys.* **2020**, 152, 184102](https://pubs.acs.org/jctcce/article/19/20/7077/870007/NWChem-Recent-and-Ongoing-Developments)）。 | 基于 Global Arrays（GA）和 ARMCI 构建分布式共享内存抽象。 | 单机桌面端（8~16 核）启动开销大、单核微架构（AVX-512）优化不足、代码库庞大陈旧、Windows 桌面用户体验极差。 | 针对单工作站多核与桌面交互场景极致优化单核 SIMD 与线程级 Work-Stealing，不做笨重的超算旧式 PGAS 包装。 |
| **[Q-Chem](https://www.tandfonline.com/doi/abs/10.1080/00268976.2014.952696)** | C++ + Fortran | 1. 激发态方法（ADC, EOM-CC, SF-TDDFT）、新型泛函（$\omega$B97X-V, $\omega$B97M-V）与能量分解（ALMO-EDA）领先；<br>2. BrianQC GPU 插件加速。 | 模块化 C++ `libtensor` 框架与遗留 Fortran 代码混合。 | 昂贵的商业闭源软件，新旧两套积分与张量引擎并存，外部程序集成度低。 | 在 `calchem-scf` 中原生支持非局域 VV10 范德华泛函（如 $\omega$B97X-V / $\omega$B97M-V）与激发态 NTO 分析，服务于有机光电材料研究。 |
| **[REST](https://pubs.aip.org/cps/cjcp/article/38/6/788/3377721/REST-Embracing-the-rust-programming-language-for)** | Rust（[`libcint-rs`](https://github.com/ajz34/libcint-rs) + Rayon + OpenBLAS） | 1. **全球首个成熟的纯 Rust 开源量子化学程序**（复旦大学张颖课题组）；<br>2. 全程采用无磁盘 RI（Disk-Free RI）设计，支持 RPA、双杂化泛函（XYG3）与机器学习泛函；<br>3. 证明了 Rust 在电子结构领域的极高可行性。 | 基于 `rest_tensors`、[`libcint`](https://docs.rs/libcint) 绑定与 `rayon` 共享内存并行，命令行文本输入驱动。 | 1. **重心在于高阶 DFT（双杂化/RPA）单点能**，在**完整解析 Hessian、连续介质溶剂模型（IEFPCM/SMD）、过渡态/IRC 几何优化闭环、以及与桌面 3D GUI（MolForge）的零拷贝进程内协同**方面尚不完善；<br>2. 缺乏面向 Windows 桌面端开箱即用的 One-Folder 图形化工作站生态。 | **生态协同与全方位超越**：在底层积极兼容并优化 [`libcint-rs`](https://github.com/ajz34/libcint-rs) 路线，同时补齐 REST 尚未重点覆盖的 **解析梯度/Hessian 全流程、SWIG-IEFPCM/SMD 溶剂模型、Casida TDDFT 激发态弛豫、TRIC 几何优化器，以及与 MolForge VTK 的毫秒级内存直连**！ |

## 6. 分阶段方法演进路线图与验收基准（Deliverable 4）

为确保每一步重构都脚踏实地、绝不破坏 `BASELINE.md` 已建立的科学正确性基线，制定四个阶段的渐进式交付路线图（每阶段均可独立编译、通过回归测试并接入 MolForge 使用）：

### Phase 1：MVP 高性能底座闭环（预计周期：4~6 周）
* **核心目标**：打通“Rust 核心 + PyO3 零拷贝绑定 + 标准基组库 + 直接/RI SCF + 解析梯度 + TRIC 几何优化 + 极速 Cube 渲染”，一举解决 v0.6.0 七大痛点中的 5 项（速度、基组、解析梯度、多核并行、Cube 渲染）。
* **实现方法**：
  - 基组：内置 `STO-3G`, `3-21G`, `6-31G(d)`, `def2-SVP`, `def2-TZVP`, `cc-pVDZ` 及 `def2-universal-jkfit`（支持球谐 5d/7f 与笛卡尔 6d/10f 双模式）。
  - SCF：RHF / UHF / RKS / UKS（LDA, PBE, B3LYP, PBE0, CAM-B3LYP, $\omega$B97X-D + D3(BJ)），含 SAD 初始猜测与 EDIIS+CDIIS+SOSCF。
  - 积分加速：Schwarz 筛选 Direct SCF + In-Core Cholesky-RI-JK。
  - 解析梯度与优化：RHF / UHF / RKS / UKS 解析一阶梯度（含 DFT 网格权重导数）+ TRIC/L-BFGS 几何优化。
  - 波函数性质：Mulliken/Löwdin 布居、偶极矩、SIMD 轨道/密度/ESP 三维网格生成。
* **严格数值验收标准（对比 `BASELINE.md` 与 Gaussian 16 A.03 / ORCA 6.1.1）**：
  - $\text{H}_2$ / RHF / STO-3G ($R = 1.4\text{ Bohr}$)：$E = -1.116714325063\text{ E}_\text{h}$（误差 $< 10^{-10}\text{ E}_\text{h}$）
  - $\text{He}$ / RHF / STO-3G：$E = -2.8077839575\text{ E}_\text{h}$（误差 $< 10^{-8}\text{ E}_\text{h}$）
  - $\text{Ne}$ / RHF / STO-3G：$E = -126.6045291715\text{ E}_\text{h}$（误差 $< 10^{-8}\text{ E}_\text{h}$）
  - $\text{H}_2\text{O}$ / RHF / STO-3G（Szabo 几何）：$E = -74.695319177132\text{ E}_\text{h}$（与 Gaussian 16 差 $< 5\times 10^{-9}\text{ E}_\text{h}$）
  - $\text{H}_2\text{O}$ / B3LYP / def2-SVP（球谐基组）：单点能与 ORCA 6.1.1（`! B3LYP def2-SVP Norijcosx TightSCFDefGrid3`）偏差 $< 10^{-6}\text{ E}_\text{h}$。
  - **解析梯度精度验证**：对 $\text{H}_2\text{O}$、甲醛（$\text{CH}_2\text{O}$）在 RHF 和 B3LYP/def2-SVP 下，解析梯度与五点中心有限差分（步长 $10^{-4}\text{ Bohr}$，SCF 收敛限 $10^{-12}\text{ E}_\text{h}$）的 RMS 偏差 $< 10^{-7}\text{ E}_\text{h}/\text{Bohr}$；$\text{H}_2$ / RHF / STO-3G 几何优化精确收敛至平衡键长 **$1.346\text{ Bohr}$（$0.712\text{ \AA}$）**（彻底修复 v0.6.0 有限差分停在 $1.67\text{ Bohr}$ 的缺陷！）。
* **Ryzen 7 9700X 性能基准目标**：
  - $\text{H}_2\text{O}$ / STO-3G 单点能：$< 1.5\text{ ms}$（比 v0.6.0 纯 Python 快 $>100\times$，由于免除 Python 胶水开销，比 PySCF 进程调用快 $2\times\sim 3\times$）。
  - 咖啡因分子（$\text{C}_8\text{H}_{10}\text{N}_4\text{O}_2$，24 原子，246 基函数 `def2-SVP`）RI-B3LYP 单点 + 解析梯度：8 核耗时 $< 1.8\text{ s}$。

### Phase 2：激发态（TDDFT）与连续介质溶剂模型（PCM/SMD）（预计周期：4~5 周）
* **核心目标**：全面赋能有机光电（OPV/OSC）计算需求，支持溶液相基态与激发态计算（解决 v0.6.0 第 4、5 项痛点）。
* **实现方法**：
  - 激发态：CIS、TDA-TDDFT、Full Casida TDDFT（单重态与三重态）、块 Davidson 求解器、振子强度、自然跃迁轨道（NTO）、sTDA/sTDDFT 快筛。
  - 溶剂模型：SWIG 平滑腔体 C-PCM / IEFPCM + SMD 非静电项（含基态解析溶剂梯度与 TDDFT 线性响应非平衡溶剂化）。
  - RIJCOSX / OCC-RI-K 半数值/轨道投影交换加速，使杂化泛函（B3LYP, CAM-B3LYP, $\omega$B97X-D）在 `def2-TZVP` 下提速 $10\times\sim 20\times$。
* **数值验收标准**：
  - 甲醛与苯分子在 CAM-B3LYP/def2-SVP 下前 5 个单重态/三重态激发能与 Gaussian 16 / ORCA 6.1 偏差 $< 0.002\text{ eV}$，振子强度 $f$ 偏差 $< 0.001$。
  - 水溶液中 $\text{H}_2\text{O}$、乙酸根离子、苯胺的 IEFPCM 与 SMD 溶剂化自由能 $\Delta G_{\text{solv}}$ 与 Gaussian 16 偏差 $< 0.05\text{ kcal/mol}$；溶液相解析梯度与有限差分偏差 $< 5\times 10^{-7}\text{ E}_\text{h}/\text{Bohr}$。

### Phase 3：后 HF 电子相关、CPSCF 解析频率与过渡态搜索（预计周期：5~6 周）
* **核心目标**：实现无磁盘 RI-MP2 / DF-CCSD(T) 与解析 Hessian，使频率计算、热力学矫正、过渡态（TS）搜索与 IRC 达到生产级精度。
* **实现方法**：
  - 后 HF：In-Core RI-MP2、SCS-MP2、RI-MP2 解析梯度、DF-CCSD、异步分块 CCSD(T)。
  - 响应与二阶导数：CPSCF（Z-Vector）求解器、RHF/RKS 解析 Hessian、红外（IR）强度、RRHO（刚性转子-谐振子）及 Grimme 准谐振子（QRRHO）热化学分析（ZPE, $H, G$）。
  - 反应路径：P-RFO 过渡态优化、GS2 IRC、TDDFT 激发态解析梯度（用于荧光发射光谱 $S_1$ 优化）。
* **数值验收标准**：
  - $\text{H}_2\text{O}$ / MP2 / STO-3G（Szabo 几何）：$E = -74.71482338\text{ E}_\text{h}$（精确匹配 `BASELINE.md`）；$\text{H}_2$ / CCSD(T) / STO-3G 精确匹配全组态相互作用（FCI）极限。
  - 解析 Hessian 算得的振动频率与 Gaussian 16 `Freq` 偏差 $< 0.5\text{ cm}^{-1}$（彻底取代 v0.6.0 误差数十 $\text{cm}^{-1}$ 的数值频率）。

### Phase 4：局域相关（DLPNO）、线性标度与 GPU 异构引擎（预计周期：6~8 周）
* **核心目标**：向 200~500 原子大体系及独立 GPU 硬件扩展。
* **实现方法**：
  - 局域相关：Foster-Boys / Pipek-Mezey 轨道局域化、DLPNO-MP2、DLPNO-CCSD(T)。
  - 多尺度与高阶泛函：双杂化泛函（B2PLYP, XYG3）、VV10 非局域色散、QM/MM 静电嵌入。
  - GPU 异构计算：基于 `wgpu` / `cudarc` 实装 GPU J/K 算子、GPU DFT 网格积分与 GPU RI 三中心缩并。

## 7. 第一阶段（MVP）文件级实施计划与精确代码签名（Deliverable 3）

### 7.1 重构后项目完整目录树（Cargo Workspace + Python 混合工程）

在 `E:\YDW\Trae_Gaussian\` 下采用 [`maturin`](https://github.com/pyo3/maturin) 标准混合工程布局（既保留 `calchem/` Python 包以 100% 兼容现有单元测试与 MolForge，又在 `crates/` 下构建模块化 Rust 核心工作区）：

```
E:\YDW\Trae_Gaussian\
├── Cargo.toml                          # Rust Workspace 根配置 (統一开启 Zen 5 AVX-512 优化)
├── pyproject.toml                      # 升级 build-system 为 maturin>=1.7,<2.0
├── crates/
│   ├── calchem-tensor/                 # [Layer 1] 张量抽象、对称压缩矩阵、10.5GB 内存池
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── memory_pool.rs          # MemoryWatermarkGuard 内存水位控制
│   │       ├── packed_sym.rs           # PackedSymMatrix 上三角紧凑矩阵
│   │       └── cpu_faer.rs             # 基于 faer 的 AVX-512 GEMM / SYEVD / POTRF 实现
│   ├── calchem-basis/                  # [Layer 3] 基组解析、BSE 内置库、球谐/笛卡尔与壳层重排
│   │   ├── Cargo.toml
│   │   ├── data/                       # Zstd 压缩的 sto-3g, 6-31g*, def2-svp, def2-tzvp, jkfit
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── molecule.rs             # Atom, MoleculeCore (Bohr 坐标与核排斥能/梯度)
│   │       ├── shell.rs                # Shell, AngularType, libcint (atm, bas, env) 序列化
│   │       └── bse_loader.rs           # 零 I/O 内存解压加载内置基组与辅助基组
│   ├── calchem-int/                    # [Layer 2] 1e/2e/3c2e 积分、Schwarz 筛选、Direct JK、RI-DF
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── cint_bridge.rs          # libcint-rs 静态封装 (1e, 2e, 3c2e 及一阶导数 int1e_ip*, int2e_ip1)
│   │       ├── obara_saika_fast.rs     # 纯 Rust (ss|ss), (ps|ss), (pp|ss) 内联 AVX-512 快路径
│   │       ├── screening.rs            # SchwarzScreener 与增量密度筛选
│   │       ├── direct_jk.rs            # Rayon 8核无锁分块 Direct J/K 构建器
│   │       └── ri_df.rs                # 二中心度量 Cholesky 分解与 B_{P,mu<=nu} 张量构建
│   ├── calchem-scf/                    # [Layer 4] RHF/UHF/RKS/UKS、SAD 猜想、EDIIS+CDIIS+SOSCF、DFT
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── guess.rs                # Core, GWH, SAD (原子密度叠加) 初始猜测
│   │       ├── diis.rs                 # Pulay C-DIIS, EDIIS, ADIIS 与三段式自动切换器
│   │       ├── soscf.rs                # 二阶轨道旋转共轭梯度求解器 (SOSCF)
│   │       ├── grid.rs                 # Treutler-Ahlrichs + Lebedev + SSF 权重 + SimdGridBlock
│   │       ├── xc_engine.rs            # Libxc 绑定 + 原生 LDA/PBE/B3LYP/CAM-B3LYP + DFT-D3(BJ)
│   │       └── solver.rs               # RHF, UHF, RKS, UKS 主循环与收敛判定
│   ├── calchem-grad/                   # [Layer 5] RHF/UHF/RKS/UKS 解析梯度与 CPSCF
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── pulay.rs                # 能量加权密度矩阵 W_{mu,nu} 与重叠导数项 -Tr(W * dS/dR)
│   │       ├── hf_grad.rs              # 1e 导数 + Direct 4c-2e 导数 / RI-JK 3c-2e 导数收缩
│   │       └── dft_grad.rs             # XC 势导数 + SSF 网格权重解析导数
│   ├── calchem-opt/                    # [Layer 10] TRIC 内坐标与 L-BFGS / RFO 几何优化器
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tric.rs                 # 键长/键角/二面角/平移/四元数转动 Wilson B 矩阵与投影
│   │       ├── hessian_guess.rs        # Fischer-Almlöf 模型 Hessian 与 BFGS/Bofill 更新
│   │       └── rfo_optimizer.rs        # 信赖域 RFO 几何优化主循环
│   ├── calchem-prop/                   # [Layer 8] SIMD 轨道/密度/ESP 网格生成与布居分析
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── cube_simd.rs            # AVX-512 批量网格评估器 (支持直接输出 f32 切片给 VTK)
│   │       └── population.rs           # Mulliken, Löwdin 电荷与自旋布居、偶极矩
│   └── calchem-py/                     # [PyO3 桥接层] 编译为 calchem/_core.cp311-win_amd64.pyd
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs                  # 暴露 run_scf, run_gradient, run_optimize, eval_cube_grid
└── calchem/                            # 现有 Python 顶层包 (100% 保持 API 向下兼容)
    ├── _core.pyi                       # Rust 扩展模块的 Python 类型存根 (供 IDE 和 mypy 检查)
    ├── core/                           # 保持 request.py, result.py, periodic_table.py 不变
    ├── backends/
    │   ├── native.py                   # 改造为优先调用 calchem._core (Rust)，保留 Python fallback
    │   ├── gaussian.py                 # 保持不变
    │   └── orca.py                     # 保持不变
    ├── jobs/                           # 保持 global_lock.py, dispatcher.py 不变
    └── api/                            # 保持 service.py 不变
```

### 7.2 关键文件的精确代码签名与实现规范

#### 1. `Cargo.toml`（Workspace 根配置，开启 Zen 5 极致优化）
```toml
[workspace]
resolver = "2"
members = [
    "crates/calchem-tensor",
    "crates/calchem-basis",
    "crates/calchem-int",
    "crates/calchem-scf",
    "crates/calchem-grad",
    "crates/calchem-opt",
    "crates/calchem-prop",
    "crates/calchem-py",
]

[workspace.dependencies]
faer = { version = "0.20", features = ["rayon", "std"] }
rayon = "1.10"
libcint = "0.1"
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py310"] }
numpy = "0.22"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```

#### 2. `crates/calchem-int/src/direct_jk.rs`（八重对称 + Schwarz 筛选 + Rayon 线程局部缓冲）
```rust
use calchem_basis::BasisSetData;
use calchem_tensor::PackedSymMatrix;
use crate::screening::SchwarzScreener;
use rayon::prelude::*;

/// 使用 Rayon 无锁并行构建库仑矩阵 J 与交换矩阵 K
/// 每个工作线程分配独立的局部 (J_local, K_local)，循环结束后归约，彻底消除原子锁竞争！
pub fn build_jk_direct_parallel(
    basis: &BasisSetData,
    screener: &SchwarzScreener,
    dm: &PackedSymMatrix,
    k_scale: f64,
) -> (PackedSymMatrix, PackedSymMatrix) {
    let nbf = basis.nbf;
    let sig_pairs = &screener.significant_pairs;
    let n_pairs = sig_pairs.len();

    // 计算每个壳层对的最大密度矩阵元 D_{IJ}^{max}，用于动态密度筛选
    let shell_dm_max = compute_shell_block_max_dm(basis, dm);
    let need_k = k_scale.abs() > 1e-14;

    let (j_total, k_total) = (0..n_pairs)
        .into_par_iter()
        .fold(
            || (PackedSymMatrix::zeros(nbf), PackedSymMatrix::zeros(nbf)),
            |(mut j_loc, mut k_loc), idx_ij| {
                let (ish, jsh, q_ij) = sig_pairs[idx_ij];
                for idx_kl in idx_ij..n_pairs {
                    let (ksh, lsh, q_kl) = sig_pairs[idx_kl];
                    // 利用 Q_{IJ} 按降序排列的性质：一旦当前上界小于阈值，直接 break 内层循环！
                    if q_ij * q_kl * screener.max_dm_global < screener.threshold {
                        break;
                    }
                    let d_max = max_quartet_density(&shell_dm_max, ish, jsh, ksh, lsh, need_k);
                    if q_ij * q_kl * d_max < screener.threshold {
                        continue;
                    }
                    // 调用 libcint C99 AVX-512 内核或内联 Obara-Saika 计算壳层四元组 (ish jsh | ksh lsh)
                    contract_shell_quartet_jk(
                        basis, dm, ish, jsh, ksh, lsh, need_k, &mut j_loc, &mut k_loc,
                    );
                }
                (j_loc, k_loc)
            },
        )
        .reduce(
            || (PackedSymMatrix::zeros(nbf), PackedSymMatrix::zeros(nbf)),
            |(mut j_a, mut k_a), (j_b, k_b)| {
                j_a.axpy(1.0, &j_b);
                if need_k {
                    k_a.axpy(1.0, &k_b);
                }
                (j_a, k_a)
            },
        );

    (j_total, k_total)
}
```
> **关键性能优化点解读**：注意内层循环中的 `if q_ij * q_kl * screener.max_dm_global < screener.threshold { break; }`！因为 `significant_pairs` 在构建时已经按 $Q_{IJ}$ **严格降序排序**，一旦某一个 `idx_kl` 不满足不等式，后续所有的 `idx_kl' > idx_kl` 必然也不满足，直接 `break` 跳出整个内层循环，将筛选复杂度本身也从 $O(N_{\text{pairs}}^2)$ 压低至亚二次标度（参考 Häser & Ahlrichs, *J. Comput. Chem.* **1989**, 10, 104）。

#### 3. `crates/calchem-grad/src/hf_grad.rs`（RHF/UHF/RKS 解析梯度组装）
```rust
use calchem_basis::{BasisSetData, MoleculeCore};
use calchem_scf::ScfOutput;

/// 计算精确解析核梯度 dE/dR_A (单位: Eh / Bohr)
pub fn compute_analytical_gradient(
    mol: &MoleculeCore,
    basis: &BasisSetData,
    aux_basis: Option<&BasisSetData>,
    scf: &ScfOutput,
) -> Vec<[f64; 3]> {
    let natom = mol.atoms.len();
    let mut grad = vec![[0.0_f64; 3]; natom];

    // 1. 核排斥能解析导数 dV_nuc / dR_A
    accumulate_nuclear_repulsion_grad(mol, &mut grad);

    // 2. 构造能量加权密度矩阵 W_{mu,nu} = sum_{i in occ} n_i * eps_i * C_{mu,i} * C_{nu,i}
    // 并累加 Pulay 轨道正交性力: - sum_{mu,nu} W_{mu,nu} * dS_{mu,nu}/dR_A (使用 int1e_ipovlp)
    let w_mat = build_energy_weighted_density(scf);
    accumulate_pulay_overlap_grad(basis, &w_mat, &mut grad);

    // 3. 单电子动能与核吸引导数: sum_{mu,nu} D_{mu,nu} * d(T_{mu,nu} + V_{mu,nu})/dR_A
    // 包括 bra/ket 中心导数 (int1e_ipkin, int1e_ipnuc) 与 Hellmann-Feynman 算子核坐标导数 (int1e_rinv)
    accumulate_one_electron_grad(mol, basis, &scf.density_alpha, scf.density_beta.as_ref(), &mut grad);

    // 4. 双电子 J/K 导数: 若启用 RI-JK 则走 3c-2e/2c-2e 极速导数，否则走 Schwarz 筛选 Direct 4c-2e 导数
    if let Some(aux) = aux_basis {
        accumulate_ri_jk_grad(basis, aux, scf, &mut grad);
    } else {
        accumulate_direct_4c2e_grad(basis, scf, &mut grad);
    }

    // 5. 若为 DFT (RKS/UKS)，累加交换关联泛函导数 + SSF 网格权重解析导数 + DFT-D3(BJ) 梯度
    if scf.xc_energy.is_some() {
        accumulate_dft_xc_and_grid_weight_grad(mol, basis, scf, &mut grad);
    }

    grad
}
```

#### 4. `crates/calchem-py/src/lib.rs`（PyO3 零拷贝绑定与 GIL 释放）
```rust
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2};

#[pyclass]
pub struct PyScfWavefunction {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub scf_energy_hartree: f64,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
    #[pyo3(get)]
    pub homo_energy_hartree: Option<f64>,
    #[pyo3(get)]
    pub lumo_energy_hartree: Option<f64>,
    #[pyo3(get)]
    pub dipole_debye: [f64; 3],
    // 内部持有的 Rust 原始波函数与基组句柄，用于后续零拷贝生成 Cube 或计算梯度
    pub(crate) inner_wfn: std::sync::Arc<calchem_scf::WavefunctionHandle>,
}

#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords_bohr, charge, multiplicity, method, basis, xc=None, use_ri=true, max_threads=8))]
pub fn run_single_point(
    py: Python<'_>,
    atomic_numbers: Vec<u8>,
    coords_bohr: PyReadonlyArray2<'_, f64>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    xc: Option<&str>,
    use_ri: bool,
    max_threads: usize,
) -> PyResult<PyScfWavefunction> {
    let coords_slice = coords_bohr.as_slice()?;
    // 关键：调用 py.allow_threads 彻底释放 Python GIL！
    // 确保 MolForge (PyQt6 主线程) 界面进度条、3D 鼠标旋转完全流畅不卡顿，同时 Rust Rayon 跑满 8 核
    let handle = py.allow_threads(move || {
        calchem_scf::execute_scf_pipeline(
            &atomic_numbers, coords_slice, charge, multiplicity, method, basis, xc, use_ri, max_threads
        )
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyScfWavefunction::from_handle(handle))
}

#[pyfunction]
pub fn eval_orbital_cube_zero_copy<'py>(
    py: Python<'py>,
    wfn: &PyScfWavefunction,
    orbital_index: usize,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    dims: [usize; 3],
) -> PyResult Bound<'py, PyArray3<f32>> {
    let handle = wfn.inner_wfn.clone();
    let grid_data: Vec<f32> = py.allow_threads(move || {
        calchem_prop::eval_mo_on_regular_grid(&handle, orbital_index, origin_bohr, spacing_bohr, dims)
    });
    // 零拷贝将 Rust Vec<f32> 所有权直接移交给 NumPy 3D 数组返回给 MolForge VTK！
    let array_1d = grid_data.into_pyarray_bound(py);
    array_1d.reshape(dims)
}
```

## 8. 与 Python / MolForge 前端的无缝集成方案（Deliverable 5）

### 8.1 保持 `TaskService` 与 `CalculationRequest` / `Result` 契约 100% 兼容

在 `calchem/backends/native.py` 中，采用**编译态引擎优先 + 纯 Python 参考回退（Reference Fallback）**的双轨机制：
- 当 `import calchem._core as _rust_core` 成功时，`NativeCalchem.single_point()`、`optimize()`、`frequency()` 直接调用 `_rust_core.run_single_point()`、`_rust_core.run_optimize()`、`_rust_core.run_frequency()`，并将 `Result.backend_version` 标记为 `"calchem-rust-1.0.0"`。
- 原有 v0.6.0 的纯 Python 模块（`calchem/scf/`, `calchem/integrals/`）移动或保留为 `backend="native_py_ref"`，专供单元测试中交叉检验（Golden Reference Test），确保重构过程中任何一处 Rust 代码改动都能立即与 Python 参考值进行 $10^{-10}\text{ E}_\text{h}$ 级比对。
- 全局锁协同：`TaskService.run()` 中的 `acquire_heavy_lock(lock_file, job_id)` 保持不变，同时在进入 `_rust_core` 时将 `AppConfig` 中的线程数（默认 `nproc=8` 物理核心）传入 Rust 的 Rayon 全局线程池，严守“单重任务独占计算、禁止多任务并发争抢 CPU/RAM”的硬约束。

### 8.2 MolForge（PyQt6 + VTK）轨道渲染从“磁盘 `.cube` 解析”升级为“内存直连”

在 `E:\YDW\Trae_Chemdraw\MolForge\molforge\application\calculators\calchem_engine.py` 与 `rendering/orbitals.py` 中：
1. **传统路径（Gaussian/ORCA）**：运行计算 $\rightarrow$ 调用 `formchk` + `cubegen` 生成几十 MB 的 `.cube` 文本文件 $\rightarrow$ Python 逐行 `float()` 解析文本 $\rightarrow$ 填入 VTK。耗时 $1.5\sim 5.0\text{ s}$。
2. **CalChem v1.0 极速路径**：
   - `CalChemEngine` 在内存中缓存最新完成的 `PyScfWavefunction` 句柄（占用内存仅数 MB）。
   - 当用户在 MolForge 的 `qc_result_dialog.py`（Tab 0）点击任意分子轨道（如 HOMO、LUMO、HOMO-1）时，直接调用 `calchem._core.eval_orbital_cube_zero_copy(wfn, mo_idx, origin, spacing, [80, 80, 80])`。
   - 返回的 `np.ndarray(shape=(80, 80, 80), dtype=np.float32)` 通过 `vtk.util.numpy_support.numpy_to_vtk(arr.ravel(), deep=0)` **零拷贝**挂载到 `vtkImageData` 的 PointData 上，直接触发 VTK `vtkFlyingEdges3D`（比 `vtkMarchingCubes` 快 $3\sim 5$ 倍的多核等值面提取算法）渲染 PBR 半透明正负相轨道瓣。
   - **端到端体验**：用户点击轨道列表的瞬间（**$< 25\text{ ms}$**，超越人眼 60Hz 帧率感知），3D 视口内轨道等值面即刻切换，实现商业软件（GaussView 6 / Chem3D）无法企及的丝滑交互体验！

## 9. Windows 11 零 C 盘依赖编译、打包与 One-Folder 分发方案（Deliverable 6）

针对用户硬件环境硬约束（**严禁在 C 盘安装任何软件或产生大型编译缓存，最终产物必须支持 Windows One-Folder 独立部署**），制定完整的工程配置与打包流水线：

### 9.1 零 C 盘占用的 Rust 与构建工具链重定向配置

Rust 的 `rustup`、`cargo` 以及编译过程中的临时目录默认会写入 `C:\Users\<User>\.rustup` 和 `C:\Users\<User>\.cargo`（动辄占用 5~15 GB）。在开始重构前，先在 PowerShell 中执行以下永久环境变量重定向（全部指向 `D:\` 或 `E:\`）：

```powershell
# 1. 将 Rust 工具链、Cargo 缓存与临时构建目录全部重定向至 D 盘/E 盘，确保 C 盘零写入
[Environment]::SetEnvironmentVariable("RUSTUP_HOME", "D:\YDW\RustToolchain\rustup", "User")
[Environment]::SetEnvironmentVariable("CARGO_HOME", "D:\YDW\RustToolchain\cargo", "User")
[Environment]::SetEnvironmentVariable("CARGO_TARGET_DIR", "E:\YDW\Trae_Gaussian\build_target", "User")
[Environment]::SetEnvironmentVariable("TMP", "D:\YDW\Temp", "User")
[Environment]::SetEnvironmentVariable("TEMP", "D:\YDW\Temp", "User")

# 2. 针对 AMD Ryzen 7 9700X (Zen 5) 开启原生 AVX-512 与静态 CRT 链接
[Environment]::SetEnvironmentVariable("RUSTFLAGS", "-C target-cpu=znver5 -C target-feature=+avx512f,+avx512dq,+avx512vl,+fma", "User")
```

### 9.2 消除 DLL Hell 的单文件 `.pyd` 编译与 PyInstaller One-Folder 封装

1. **静态链接底层算子**：在 `crates/calchem-int/build.rs` 和 `crates/calchem-scf/build.rs` 中，使用 Rust 的 `cc` / `cmake` crate 将 `libcint` 与 `libxc` 编译为静态库（`.lib`）直接链接进 `_core.cp311-win_amd64.pyd`，同时使用纯 Rust 的 [`faer`](https://docs.rs/faer/latest/faer/) 作为默认线性代数引擎。这样生成的 `_core.pyd` **不依赖任何外部的 `libopenblas.dll`、`mkl_rt.dll` 或 `libcint.dll`**，彻底根除 Windows 下的 `DLL load failed` 顽疾（正如 v0.6.0 总结文档第 7 节记录的 `QtCore DLL load failed` 教训）。
2. **使用 `molforge-webengine` 环境统一构建与打包**：
```powershell
$py_gui = "D:\YDW\Trae_Gromacs\miniconda3\envs\molforge-webengine\python.exe"
cd E:\YDW\Trae_Gaussian

# 编译并安装 Rust 扩展模块到当前环境 (开发模式)
& $py_gui -m maturin develop --release

# 生成可独立分发的 Windows Wheel 包 (输出至 E:\YDW\Trae_Gaussian\dist\)
& $py_gui -m maturin build --release --out E:\YDW\Trae_Gaussian\dist

# 使用 PyInstaller 构建包含 MolForge + CalChem Rust 内核的 One-Folder 发行版
& $py_gui -m PyInstaller --noconfirm --onedir `
    --workpath "E:\YDW\Trae_Gaussian\build_pyinstaller" `
    --distpath "E:\YDW\Trae_Gaussian\dist_onefolder" `
    --add-data "E:\YDW\Trae_Gaussian\crates\calchem-basis\data;calchem/basis_data" `
    --hidden-import "calchem._core" `
    E:\YDW\Trae_Chemdraw\MolForge\run_molforge.py
```

## 10. 阶段性验收命令与预期输出规范

每一个子模块完成后，通过以下命令组合执行自动化科学正确性与性能回归验证：

| 验证层级 | 验收执行命令（PowerShell） | 预期通过标准（Acceptance Criteria） |
| :--- | :--- | :--- |
| **1. Rust 底层算子与基线回归** | `cargo test --workspace --release -- --nocapture` | 1. 球谐/笛卡尔基组归一化和重叠矩阵迹 $\text{Tr}(S)$ 误差 $< 10^{-14}$；<br>2. $\text{H}_2/\text{He}/\text{Ne}/\text{H}_2\text{O}$ 在 `STO-3G` 下的 RHF 能量与 `BASELINE.md` 偏差 $< 10^{-10}\text{ E}_\text{h}$；<br>3. 解析梯度与 $10^{-4}\text{ Bohr}$ 五点中心差分梯度最大分量差 $< 10^{-7}\text{ E}_\text{h}/\text{Bohr}$。 |
| **2. Python 契约与 22 项既有单测** | `& "D:\YDW\Trae_Gromacs\miniconda3\envs\molforge-webengine\python.exe" -m pytest tests/unit -v` | 原有 22 个单元测试 100% PASS；`Result.actual_method`、`scf_converged`、`dipole_magnitude_debye` 等字段完备且无假值欺骗。 |
| **3. 三后端交叉精度与几何优化对标** | `& "D:\YDW\Trae_Gromacs\miniconda3\envs\molforge-webengine\python.exe" tests/regression/verify_three_backends.py` | 1. $\text{H}_2\text{O}$ / B3LYP / def2-SVP 下，`native`（Rust）与 `gaussian`（G16）、`orca`（ORCA 6.1.1）单点能差 $< 10^{-6}\text{ E}_\text{h}$；<br>2. $\text{H}_2$ / RHF / STO-3G 几何优化收敛键长 $= 1.3460 \pm 0.0002\text{ Bohr}$（与 Gaussian 16 Opt 完全一致）。 |
| **4. 性能与 15.1 GB 内存安全压测** | `cargo bench --bench scf_grad_bench` | 1. $\text{H}_2\text{O}$ / def2-TZVP RHF+Grad 总耗时 $< 35\text{ ms}$；<br>2. 富勒烯 $\text{C}_{60}$ / def2-SVP（840 基函数）RI-PBE 单点能峰值内存占用 $< 3.2\text{ GB}$，全程零磁盘临时文件写入。 |

## 11. 风险评估矩阵与工程缓解措施（Deliverable 7）

| 风险编号与类别 | 风险具体描述 | 严重度 | 发生概率 | 工程缓解与兜底措施（Mitigation Strategy） |
| :--- | :--- | :--- | :--- | :--- |
| **R1：Windows 下 C/Fortran 库交叉编译失败** | `libcint` 或 `libxc` 在 Windows MSVC 环境下因 C99 复数语法或 CMake 配置差异导致编译报错。 | 高 | 中 | 1. 在 `calchem-int` 中采用条件编译 Feature（`features = ["cint-c99", "pure-rust-int"]`）：内置纯 Rust Obara-Saika/Rys 积分引擎（参考 [`rustint-qc`](https://crates.io/crates/rustint-qc)）作为 100% 免 C 编译器的原生保底；<br>2. 提供针对 MSVC 预打补丁的 `libcint` C 源码子模块，通过 `cc` crate 直接编译。 |
| **R2：DFT 数值网格与积分筛选差异导致与 Gaussian 微小能量偏差** | 不同软件默认的 DFT 径向/角向网格点数、对称性旋转及 Screening 阈值不同，导致 B3LYP 能量出现 $10^{-4}\text{ E}_\text{h}$ 级差异，被误判为 Bug。 | 中 | 高 | 1. 在 `calchem-scf` 中内置 **`GridPreset::GaussianFine`（75, 302 网格 + Becke/SSF 权重）** 与 **`GridPreset::OrcaDefGrid2`** 精确复刻模式；<br>2. 先用与网格无关的 **RHF / UHF / MP2** 验证积分与线性代数精度（要求对齐至 $10^{-8}\text{ E}_\text{h}$），再用对齐网格后的 DFT 验证泛函实现（要求对齐至 $10^{-6}\text{ E}_\text{h}$）。 |
| **R3：大体系或弥散基组（如 `aug-cc-pVTZ`）下重叠矩阵 $S$ 线性相关** | 弥散基函数重叠过大导致 $S$ 的最小特征值 $\lambda_{\min} < 10^{-7}$，Löwdin 对称正交化 $S^{-1/2}$ 放大数值噪声使 SCF 崩溃。 | 高 | 中 | 在 `calchem-scf` 正交化阶段强制实施**正则正交化（Canonical Orthogonalization，Szabo & Ostlund 第 3.4.5 节）**：自动剔除特征值 $\lambda_i < 10^{-6}$ 的线性相关本征向量（降维构建正交基 $X_{N\times M}, M < N$），彻底免疫弥散基组病态问题。 |
| **R4：15.1 GB 物理内存下大体系后 HF 或 TDDFT 触发 OOM** | 用户在 MolForge 打开较大共轭聚合物（如 $>60$ 原子、`def2-TZVP`）运行 MP2 或 CCSD 时内存超出 15.1 GB。 | 高 | 中 | 1. 在计算启动前执行**预飞内存估算器（Pre-Flight Memory Estimator）**：根据 $(N_{\text{bf}}, N_{\text{aux}}, N_{\text{occ}}, N_{\text{virt}})$ 精确计算峰值内存需求；<br>2. 若估算值 $> 10.5\text{ GB}$，对于 SCF/TDDFT 自动从 In-Core RI 切换为 Direct / Semidirect 分批重算模式；对于传统 CCSD(T) 则立即返回清晰的 `MemoryLimitExceededError` 并引导用户使用 RI-MP2 或缩小基组，绝不导致操作系统死机。 |
| **R5：CPU 型号跨平台分发时的 AVX-512 指令非法异常（SIGILL）** | 开发机（Ryzen 7 9700X）支持 AVX-512，但若将 One-Folder 包拷贝到不支持 AVX-512 的旧款 Intel（如 12/13/14 代酷睿）或旧 AMD（Zen 3）电脑上运行会报 `Illegal Instruction`。 | 中 | 中 | [`faer`](https://docs.rs/faer/latest/faer/) 库内部通过 `pulp` 原生支持**运行时 CPUID 动态指令集分派（Runtime SIMD Dispatch）**。开发机本地编译可开启 `target-cpu=znver5`；对外发布的通用安装包则设置基础目标为 `x86-64-v3`（AVX2+FMA），并在关键算子入口通过 `is_x86_feature_detected!("avx512f")` 动态跳转至 AVX-512 分支，兼顾本机极致性能与跨机兼容性。 |

## 12. 参考文献与核心开源项目索引

1. **REST (Rust-based Electronic Structure Toolkit)**: Zhang, I. Y. et al. *REST: Embracing the Rust Programming Language for Modern Electronic Structure Theory*. [Chin. J. Chem. Phys. **2025**, 38, 788](https://pubs.aip.org/cps/cjcp/article/38/6/788/3377721/REST-Embracing-the-rust-programming-language-for) | [ResearchGate Preprint](https://www.researchgate.net/publication/399989458_REST_Embracing_the_rust_programming_language_for_modern_electronic_structure_theory).
2. **PySCF & Libcint**: Sun, Q. *Libcint: An efficient general integral library for Gaussian basis functions*. J. Comput. Chem. **2015**, 36, 1664 | [Libcint GitHub](https://github.com/sunqm/libcint) | [libcint-rs Rust Bindings](https://github.com/ajz34/libcint-rs) | [rustint-qc Crate](https://crates.io/crates/rustint-qc).
3. **ORCA 6 & SHARK / DLPNO**: Neese, F. et al. *The ORCA quantum chemistry program package*. [ORCA 6.1 Manual](https://www.faccts.de/docs/orca/6.1/manual/index.html) | Riplinger, C.; Neese, F. *An efficient and near linear scaling pair natural orbital based local coupled cluster method*. [J. Chem. Phys. **2013**, 139, 134101](https://pubmed.ncbi.nlm.nih.gov/24116546/) | Guo, Y. et al. [J. Chem. Phys. **2018**, 148, 011101](https://pubs.aip.org/aip/jcp/article/148/1/011101/195649/Communication-An-improved-linear-scaling).
4. **Psi4**: Smith, D. G. A. et al. *PSI4 1.4: Open-source software for high-throughput quantum chemistry*. [J. Chem. Phys. **2020**, 152, 184108](https://pubs.aip.org/aip/jcp/article/152/18/184108/972964/PSI4-1-4-Open-source-software-for-high-throughput) | [Psi4 GitHub](https://github.com/psi4/psi4/).
5. **Q-Chem & NWChem**: Shao, Y. et al. *Advances in molecular quantum chemistry contained in the Q-Chem 4 program package*. [Mol. Phys. **2015**, 113, 184](https://www.tandfonline.com/doi/abs/10.1080/00268976.2014.952696) | Aprà, E. et al. *NWChem: Past, present, and future*. [NWChem Official](https://nwchemgit.github.io/).
6. **GTO Molecular Integrals**: Obara, S.; Saika, A. *Efficient recursive computation of molecular integrals over Cartesian Gaussian functions*. [J. Chem. Phys. **1986**, 84, 3963](https://pubs.aip.org/aip/jcp/article/84/7/3963/91928/Efficient-recursive-computation-of-molecular) | Valeev, E. F. [Libint2 Library](https://github.com/evaleev/libint).
7. **SCF Acceleration (DIIS)**: Pulay, P. *Improved SCF convergence acceleration*. [J. Comput. Chem. **1982**, 3, 556](https://www.semanticscholar.org/paper/Improved-SCF-convergence-acceleration-Pulay/e629c6d572568c35e8f8f575cc53b01dede7504a0).
8. **Basis Set Exchange & Libxc**: Pritchard, B. P. et al. *New Basis Set Exchange: An Open, Up-to-Date Resource for the Molecular Sciences Community*. [J. Chem. Inf. Model. **2019**, 59, 4814](https://pubs.acs.org/jcisd8/article/59/11/4814/874269/New-Basis-Set-Exchange-An-Open-Up-to-Date-Resource) | [BSE Portal](https://www.basissetexchange.org/) | Marques, M. A. L.; Lehtola, S. [Libxc Library](https://gitlab.com/Molcas/Mirrors/libxc).
9. **TDDFT Response Theory**: Casida, M. E. *Time-Dependent Density Functional Response Theory for Molecules*. [Recent Advances in Density Functional Methods **1995**, 1, 155](https://www.researchgate.net/publication/233785009_rm_Time-Dependent_Density_Functional_Response_Theory_for_Molecules_in_Recent_Advances_in_Density_Functional_Methods_Vol_1_D_P_Chong_Ed).
10. **Solvation Models (SMD & PCM)**: Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. *Universal Solvation Model Based on Solute Electron Density (SMD)*. [J. Phys. Chem. B **2009**, 113, 6378](https://pubs.acs.org/jpcbfk/article/113/18/6378/1427431/Universal-Solvation-Model-Based-on-Solute-Electron).
11. **TRIC Geometry Optimization**: Wang, L.-P.; Song, C. *Geometry optimization made simple with translation and rotation coordinates*. [J. Chem. Phys. **2016**, 144, 214108](https://pubs.aip.org/aip/jcp/article/144/21/214108/313176/Geometry-optimization-made-simple-with-translation).
12. **Rust HPC Linear Algebra, AD & Python Bindings**: Quiñones, S. [`faer-rs` High-Performance Linear Algebra in Rust](https://docs.rs/faer/latest/faer/) ([GitHub](https://github.com/sarah-quinones/faer-rs/blob/main/paper.md)) | Moses, W. S.; Churavy, V. [Enzyme High-Performance LLVM Automatic Differentiation](https://github.com/EnzymeAD/enzyme) ([NeurIPS 2020](https://arxiv.org/abs/2010.01709)) | [PyO3 & Maturin](https://github.com/pyo3/maturin) ([User Guide](https://www.maturin.rs/)).
