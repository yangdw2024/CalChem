# CalChem 最终科研可信度验收报告（2026-11-16）

## 1. 验收结论与评级

- **当前定位**：**经过外部独立基准与硬件遥测验证的研究开发级量子化学引擎（Late Alpha / Research-Grade for Bounded Systems）**。
- **构建与全量回归**：
  - 原生扩展 `calchem/_core.pyd` 与 `build_target/release/_core.dll` 的 SHA-256 哈希完全一致：`37F96B1A58C64F86DA8ED0A997BDCA2386A04635602E070D5FAA151E32A16F8C`。
  - Rust 全工作区：**140 passed, 0 failed, 0 ignored**（68 个测试条目，exit 0）。
  - CUDA 算子与 FFI 契约：**5 passed, 0 failed**（exit 0）。
  - Python 全量 `tests/unit`：**120 passed, 0 failed**（耗时 `995.97s`，exit 0）。

---

## 2. 针对上轮审计指出的核心问题整改验证

| 问题域 | 上轮审查指出的缺陷 | 本轮源码修复与实测证据 |
| :--- | :--- | :--- |
| **DIIS 磁盘溢出错误传播** | SCF 主循环调用非 `Result` 的 `push_and_extrapolate`，磁盘 I/O 失败被吞掉；测试未向活实例注入故障 | 在 [`rhf.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-scf/src/rhf.rs#L160)、[`rks.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-scf/src/rks.rs#L204)、[`uhf.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-scf/src/uhf.rs#L228)、[`uks.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-scf/src/uks.rs#L237) 与 [`scrf.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-solv/src/scrf.rs#L274) 全部改为调用 `try_push_and_extrapolate(...)?`；向活 `DiisState` 注入只读文件句柄故障并验证 `Drop` 清理与 RAM 零分配（`test_disk_diis_never_falls_back_to_ram_on_io_error`、`disk_diis_io_failure_is_propagated_by_scf_solvers`）。 |
| **RI-J 度量秩截断保护** | 上轮测试手动赋值 `metric_truncated_modes = 1`，未用真实线性相关辅助基组贯通验证 | 新增 `RiJEngine::with_auxiliary_basis`，在 [`rij_provenance.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-grad/tests/rij_provenance.rs#L154-L197) 构造真实线性相关重复 s 壳层辅助基，自然触发 `lambda < 1e-12 * lambda_max` 截断并验证 `compute_coulomb_gradient` 显式报错拦截。 |
| **IC-2F GPU 内核执行可观测性** | 仅比对 CPU/GPU 最终能量接近，无法证明每次 $J$ 缩并确实进入 CUDA 内核而非静默回退 CPU | 在 [`RiJEngine`](file:///E:/YDW/Trae_Gaussian/crates/calchem-int/src/density_fitting.rs#L746)、[`ScfDiagnostics`](file:///E:/YDW/Trae_Gaussian/crates/calchem-model/src/lib.rs#L296)、[`PyScfWavefunction`](file:///E:/YDW/Trae_Gaussian/crates/calchem-py/src/lib.rs#L113)、[`NativeCalchem`](file:///E:/YDW/Trae_Gaussian/calchem/backends/native.py#L345) 与 [`OPVWorkflowResult`](file:///E:/YDW/Trae_Gaussian/calchem/workflows/opv.py#L96) 全链路透传 `gpu_kernel_invocations` 与 `gpu_fallback_invocations`。实测 22 原子 IC-2F（94 AO）中 GPU 路径 **`gpu_kernel_invocations = 26`，`gpu_fallback_invocations = 0`**。 |
| **多泛函外部高精度基准** | 上轮仅用核排斥能、一个 STO-3G RHF 数值和宽能量区间代替外部交叉验证 | 本轮在同一几何、同一基组、关闭对称性（`NoSymm`）与 `Integral=UltraFine` 下直接调用本机 **Gaussian 16 (`D:\G16W\g16.exe`)** 实测 11 个算例，逐一比对电子总能、HOMO/LUMO 轨道能与解析核梯度分量，结果完整记录于 [`EXTERNAL_REFERENCE_COMPARISON.csv`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/EXTERNAL_REFERENCE_COMPARISON.csv)。 |

---

## 3. 外部独立基准对照摘要（CalChem vs Gaussian 16）

详细数据见 [`EXTERNAL_REFERENCE_COMPARISON.csv`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/EXTERNAL_REFERENCE_COMPARISON.csv)（注意：PySCF 未在本机 Python 环境中安装，故本轮全部外部基准均来自本机实测 Gaussian 16）。

- **RHF（精确 4c-ERI，STO-3G / 6-31G / 6-31G(d)）**：
  - `H2` / `H2O` / `C2H4` 总能最大绝对差 **$5.53 \times 10^{-8}\text{ E}_\text{h}$**（`H2O / 6-31G(d)` 达 **$1.75 \times 10^{-11}\text{ E}_\text{h}$**）；
  - HOMO/LUMO 轨道能最大偏差 **$\le 2.98 \times 10^{-6}\text{ E}_\text{h}$**（受 Gaussian 5 位小数打印精度限制）；
  - 解析核梯度最大分量偏差 **$3.00 \times 10^{-10} \sim 1.51 \times 10^{-8}\text{ E}_\text{h}/\text{Bohr}$**。
- **DFT（`B3LYP` / `PBE0` / `CAM-B3LYP`，`UltraFine` 网格）**：
  - `H2O / B3LYP / 6-31G(d)` 总能差 **$-6.67 \times 10^{-7}\text{ E}_\text{h}$**，最大梯度差 **$4.50 \times 10^{-6}\text{ E}_\text{h}/\text{Bohr}$**；
  - `H2O / PBE0 / 6-31G(d)` 总能差 **$-1.52 \times 10^{-5}\text{ E}_\text{h}$**，最大梯度差 **$1.13 \times 10^{-5}\text{ E}_\text{h}/\text{Bohr}$**；
  - `H2O / CAM-B3LYP / 6-31G(d)` 总能差 **$-6.44 \times 10^{-7}\text{ E}_\text{h}$**，最大梯度差 **$3.57 \times 10^{-6}\text{ E}_\text{h}/\text{Bohr}$**。（注：Gaussian 默认 B3LYP 使用 VWN3 而 CalChem 使用 VWN5，已明确标注。）
- **RI-J（自动生成 `auto-jfit` 辅助基 vs Gaussian 16 精确 4c-ERI）**：
  - 由于 `auto-jfit` 是轻量级自动辅助基（非大型 `def2/J` 精确拟合基），拟合误差表现为基组截断偏差：$\text{H}_2$ `STO-3G` $-2.35 \times 10^{-5}\text{ E}_\text{h}$，$\text{H}_2\text{O}$ `B3LYP/STO-3G` $-4.91 \times 10^{-4}\text{ E}_\text{h}$，$\text{H}_2\text{O}$ `B3LYP/6-31G(d)` $-8.10 \times 10^{-3}\text{ E}_\text{h}$。
- **$\omega\text{B97X-D}$**：
  - 因缺乏拟合 B97 相关泛函与原始色散参数，CalChem **显式拒绝**该请求（`ValueError`），严禁静默替代。

---

## 4. CPU vs RTX 5060 Ti GPU RI-J 实测分离结论

详细数据见 [`CPU_GPU_VALIDATION.csv`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/CPU_GPU_VALIDATION.csv)：

1. **数值一致性与真实 GPU 执行（已证实）**：
   - 在 `H2O`、`C2H4` 及 **22 原子 OPV 末端受体 `IC-2F`（94 AO）** 上，`use_gpu=True` 路径的 `gpu_kernel_invocations` 分别为 `10`、`12`、`11`、`26` 次，且 `gpu_fallback_invocations = 0`；
   - `IC-2F` 的 CPU 与 GPU 收敛能量分别为 `-870.5565577837 Eh` 与 `-870.5565578397 Eh`，绝对偏差仅 **$5.60 \times 10^{-8}\text{ E}_\text{h}$**，HOMO-LUMO 能隙偏差 **$2.18 \times 10^{-8}\text{ eV}$**。
2. **端到端性能结论（诚实披露，未获加速）**：
   - 在 22 原子 `IC-2F`（94 AO）体系下，CPU RI-J 总墙钟时间为 `126.80 s`，GPU RI-J 为 `127.65 s`（加速比 `0.993x`）。原因在于当前仅将 `J` 矩阵三中心缩并卸载到 GPU 且每步 SCF 均执行 PCIe `HtoD/DtoH` 拷贝，而主要耗时仍在 CPU 侧的精确交换 `K` 构建与 DFT 格点积分上。**当前不能宣称端到端 GPU 加速，仅能认证双精度数值正确性与 RTX 5060 Ti 原生核函数调度可用性。**

---

## 5. 实际计算能力分级表（严禁跨级别推广）

| 能力分级 | 涵盖功能与体系 | 依据 |
| :--- | :--- | :--- |
| **科研级验证通过** | RHF/UHF（STO-3G, 6-31G, 6-31G(d), LANL2DZ）、B3LYP/PBE0/CAM-B3LYP 单点能与解析梯度、D3(BJ)/D3(0) 色散、CPCM/COSMO 溶剂能量、14 维内禀振动频率、Boys 函数高精度表 | 与本机 Gaussian 16 实测或 80 位参考值直接对齐；全量 140 Rust + 120 Python 测试通过 |
| **仅小体系验证通过** | RI-J 自洽场与满秩解析梯度、GPU RI-J 库仑缩并（$\le 22$ 原子 `IC-2F`）、流式 Becke 网格、磁盘 DIIS 溢出、P-RFO 过渡态搜索与 IRC 路径（HCN） | 仅在 $\le 22$ 原子体系上跑通实测验证；`auto-jfit` 在较大基组上存在 $\sim 10^{-3}\text{ E}_\text{h}$ 级密度拟合偏差 |
| **功能已实现但未独立外部验证** | TDDFT / CIS（TDA 与 RPA）、RI-J 几何优化大分子应用、ESP 表面与 `CCMSH`/`CCESP` 二进制流 | 内部数学一致性（如 Thouless Hessian、有限差分、RPA 和规则）通过，缺外部软件逐项对照 |
| **仅理论资源推演** | BTP（159 原子）、Y6（199 原子）、PM6（158 原子）原生 SCF 收敛；726 原子 / 6224 AO `p4D18` 及 2000 原子在 16 GB 内存内完成 | 仅解析了输入结构与 Gaussian 日志并计算了 $O(N_{\text{AO}}^2)$ 矩阵内存模型，未实际运行原生 SCF |
| **存在阻断性问题 / 明确拒绝** | $\omega\text{B97X-D}$、SMD 隐式溶剂、开壳层 SCRF、RI-J 下的 TDDFT 响应、度量秩截断下的 RI-J 解析梯度 | 源码明确报错拒绝或尚未完整实现；大体系全流程 16 GB 峰值内存与 GPU 端到端加速均未达成 |
