# CalChem 阶段性科研精度与工程验证报告（2026-11-16 更新版）

本报告记录了针对 [独立复核报告](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_certification_review/FINAL_REVIEW.md) 所提出问题的三阶段整改与全量回归验收结果。所有数值均基于本地实测输出；大体系（726–2000 原子）原生收敛、16 GB 全程内存上限以及 GPU 端到端加速均**尚未实测证明**，不得将本报告外推为通用工业级大体系生产认证。

## 1. 验收结论与构建指纹

- **当前定位**：**经过外部独立基准、网格拆解与硬件遥测验证的研究开发级量子化学引擎（Late Alpha / Research-Grade for Bounded Systems）**。
- **构建与制品指纹**：
  - 原生扩展 `calchem/_core.pyd` 与 `build_target/release/_core.dll` 的 SHA-256 一致：`6ED03FD1E126D480CF265DD518282DCD7F5C510351A30401DCA3473ADAB7CDC8`。
  - 全部构建产物与基准 CSV 哈希记录见 [`MANIFEST_SHA256.json`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/MANIFEST_SHA256.json)。
  - 复现脚本 [`reproduce_certification.ps1`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/reproduce_certification.ps1) 已补全每步子进程 `$LASTEXITCODE` 检查与 SHA-256 清单生成。
- **全量回归（均以退出码 `0` 通过）**：
  - Rust 全工作区：**141 passed, 0 failed, 0 ignored**（69 个测试条目，[`cargo_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/cargo_full.log)）。
  - CUDA 算子与 FFI 契约：**5 passed, 0 failed**（[`cuda_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/cuda_full.log)）。
  - Python 全量 `tests/unit`：**120 passed, 0 failed**（耗时 `1027.68s`，[`pytest_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/pytest_full.log)）。

---

## 2. 本轮三项核心技术整改（P3 / P2 / P1）

| 优先级 | 问题来源 | 整改措施与实测验证 |
| :--- | :--- | :--- |
| **P3：脚本与报告严谨性** | `reproduce_certification.ps1` 未检查每步 `$LASTEXITCODE`；CSV 误标 `B3LYP` 为 `VWN5`；报告将 `12` 个算例误写为 `11` 个 | 重写 [`reproduce_certification.ps1`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/reproduce_certification.ps1) 的 `Invoke-CheckedStep`；修正 XC 注释（`B3LYP` 使用 `VWN_RPA`，`CAM-B3LYP` 与 `B3LYP5` 使用 `VWN5`）；输出 **14 个 Gaussian 数值算例 + 1 个 wB97X-D 拒绝算例**（共 15 行）。 |
| **P2：DFT 网格对齐与 PBE0 差值拆解** | 上轮 Gaussian `UltraFine` vs CalChem `Fine` 并非同网格，PBE0 存在 `1.52e-5 Eh` 能量差与 `1.13e-5 Eh/Bohr` 梯度差 | 将双边网格收敛至无剪枝高密网格（CalChem `GridLevel::UltraFine` 200×590 vs Gaussian 16 `Integral(Grid=200974)` 200×974）。收敛后 **PBE0 梯度最大分量差缩小 26 倍至 `4.352e-7 Eh/Bohr`**；**B3LYP（`1.606e-9 Eh`）与 CAM-B3LYP（`1.661e-9 Eh`）与 Gaussian 达到 $10^{-9}\text{ E}_\text{h}$ 一致**；剩余的 PBE（`-1.815e-5 Eh`）与 PBE0（`-1.459e-5 Eh`）常数偏置源于 CalChem/LibXC 解析常数 $\mu = \beta\pi^2/3, \gamma = (1-\ln 2)/\pi^2$ 与 Gaussian 16 采用的 1996 论文四舍五入常数（$\mu=0.21951, \beta=0.066725, \gamma=0.031091$）之差。 |
| **P1：RI-J 度量逆病态与拟合误差** | `auto-jfit` 在 `H2O/6-31G(d)` 下相对精确库仑存在 `-8.10 mEh` 误差；循环 Jacobi 求逆在病态辅助基下 `max\|V*V^-1 - I\|` 高达 `0.525` | 1. 在 [`calchem-tensor/src/lib.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-tensor/src/lib.rs#L290-L371) 基于 `faer` 实现 `invert_spd_with_rank_info`（Householder EVD 秩检测 + Cholesky 满秩求逆），将 `max\|V*V^-1 - I\|` 从 `0.525` 降至 `3.57e-9 ~ 1.03e-7`；<br>2. 优化 [`AuxiliaryBasisSet::build_auto_jfit`](file:///E:/YDW/Trae_Gaussian/crates/calchem-int/src/density_fitting.rs#L105-L155) 指数比例（`2.3`）与 D/F 角动量窗口，将 `H2O / B3LYP / 6-31G(d)` RI-J 偏差**缩小约 480 倍至 `-1.683e-5 Eh`**（`~0.01 kcal/mol`），`J_RI` vs `J_exact` 矩阵 `max\|dJ\| = 9.275e-5`、`rms\|dJ\| = 1.261e-5`；<br>3. 在 [`calchem-py/src/lib.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-py/src/lib.rs#L810-L820) 中为 `run_gradient(..., density_fitting=True)` 设置适配 RI-J 条件数噪声底的 SCF 阈值（`1e-7 / 1e-7`）。 |

---

## 3. 外部独立基准对照摘要（CalChem vs Gaussian 16）

详细逐项数据见 [`EXTERNAL_REFERENCE_COMPARISON.csv`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/EXTERNAL_REFERENCE_COMPARISON.csv)（共 15 行：14 个实测 Gaussian 16 `D:\G16W\g16.exe` 数值算例 + 1 个显式拒绝条目；本地未安装 PySCF）。

- **RHF（精确 4c-ERI，STO-3G / 6-31G / 6-31G(d)）**：
  - `H2` / `H2O` / `C2H4` 总能最大绝对差 **$5.526 \times 10^{-8}\text{ E}_\text{h}$**（`H2O / 6-31G(d)` 达 **$1.751 \times 10^{-11}\text{ E}_\text{h}$**）；
  - HOMO/LUMO 轨道能最大偏差 **$\le 2.978 \times 10^{-6}\text{ E}_\text{h}$**（受 Gaussian 5 位小数打印精度限制）；
  - 解析核梯度最大分量偏差 **$3.004 \times 10^{-10} \sim 1.514 \times 10^{-8}\text{ E}_\text{h}/\text{Bohr}$**。
- **DFT（`B3LYP` / `PBE` / `PBE0` / `CAM-B3LYP`，CalChem `200×590` vs Gaussian `200×974` 未剪枝高密网格）**：
  - `H2O / B3LYP / STO-3G`：总能差 **$+5.291 \times 10^{-8}\text{ E}_\text{h}$**，最大梯度差 **$3.246 \times 10^{-7}\text{ E}_\text{h}/\text{Bohr}$**；
  - `H2O / B3LYP / 6-31G(d)`：总能差 **$+1.606 \times 10^{-9}\text{ E}_\text{h}$**，最大梯度差 **$1.830 \times 10^{-7}\text{ E}_\text{h}/\text{Bohr}$**；
  - `H2O / CAM-B3LYP / 6-31G(d)`：总能差 **$+1.661 \times 10^{-9}\text{ E}_\text{h}$**，最大梯度差 **$1.626 \times 10^{-7}\text{ E}_\text{h}/\text{Bohr}$**；
  - `H2O / PBE / 6-31G(d)`：总能差 **$-1.815 \times 10^{-5}\text{ E}_\text{h}$**，最大梯度差 **$4.679 \times 10^{-7}\text{ E}_\text{h}/\text{Bohr}$**（能量差反映解析与 1996 四舍五入 PBE 常数之别）；
  - `H2O / PBE0 / 6-31G(d)`：总能差 **$-1.459 \times 10^{-5}\text{ E}_\text{h}$**，最大梯度差 **$4.352 \times 10^{-7}\text{ E}_\text{h}/\text{Bohr}$**。
- **RI-J（改进后的 `auto-jfit` vs Gaussian 16 精确 4c-ERI）**：
  - `H2 / RHF+RI-J / STO-3G`：总能差 **$-5.194 \times 10^{-5}\text{ E}_\text{h}$**，最大梯度差 **$6.255 \times 10^{-5}\text{ E}_\text{h}/\text{Bohr}$**；
  - `H2O / RHF+RI-J / 6-31G(d)`：总能差 **$-1.682 \times 10^{-5}\text{ E}_\text{h}$**，最大梯度差 **$2.327 \times 10^{-5}\text{ E}_\text{h}/\text{Bohr}$**；
  - `H2O / B3LYP+RI-J / STO-3G`：总能差 **$-5.826 \times 10^{-5}\text{ E}_\text{h}$**；
  - `H2O / B3LYP+RI-J / 6-31G(d)`：总能差 **$-1.683 \times 10^{-5}\text{ E}_\text{h}$**。
- **$\omega\text{B97X-D}$**：
  - 因缺乏拟合 B97 相关泛函、短程交换及原始色散参数，CalChem **显式拒绝**该请求（`ValueError`）。

---

## 4. CPU vs RTX 5060 Ti GPU RI-J 实测结论与条件数边界

详细数据见 [`CPU_GPU_VALIDATION.csv`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/CPU_GPU_VALIDATION.csv)：

1. **真实 GPU 内核执行与数值一致性**：
   - `H2O (STO-3G)`、`H2O (6-31G(d))`、`C2H4 (STO-3G)` 与 **22 原子 `IC-2F`（94 AO）** 的 `gpu_kernel_invocations` 分别为 `10`、`12`、`11`、`24` 次，且 `gpu_fallback_invocations = 0`；
   - `IC-2F` 在基准脚本中（21 次迭代）CPU 与 GPU 收敛能量分别为 `-870.5508071411 Eh` 与 `-870.5508078585 Eh`，绝对偏差 **$7.173 \times 10^{-7}\text{ E}_\text{h}$**，HOMO-LUMO 能隙偏差 **$3.375 \times 10^{-7}\text{ eV}$**。
2. **条件数与 IEEE-754 舍入边界（诚实披露）**：
   - 因当前 `auto-jfit` 采用笛卡尔 `D`/`F` 壳层，度量矩阵存在 $\text{cond}(V) \sim 10^6 \text{–} 10^7$（`max|V^-1| ~ 2.2×10^5 – 9.7×10^5`），导致 `J_RI` 与 `E_J` 存在 $\sim 10^{-8}\text{ E}_\text{h}$（小分子）至 $\sim 10^{-7}\text{ E}_\text{h}$（22 原子 `IC-2F`）的浮点舍入噪声底。因此：
     - `IC-2F` 在较松的 `conv_energy=1e-6` 下，CPU 与 GPU 可能相差 1 步迭代（`1.42e-6 Eh`）；收紧至 `conv_energy=1e-8` 时能隙差收敛至 `6.42e-7 eV`，但 GPU 因 FMA 舍入顺序不同在 DIIS 中迭代至 105 步。
     - 对于大体系定量高精度 SCF，后续仍建议引入球谐辅助基（5d/7f）或标准 `def2/J` 收缩辅助基以降低 `cond(V)`。
3. **端到端性能结论（未获加速）**：
   - `IC-2F` 基准中 CPU 耗时 `125.33 s`，GPU 耗时 `130.33 s`（加速比 `0.962x`），因为精确交换 `K` 构建与 DFT 格点积分仍在 CPU 执行，且每步 SCF 存在 PCIe 拷贝。**当前不宣称端到端 GPU 加速。**

---

## 5. 实际计算能力分级表（严禁跨级别推广）

| 能力分级 | 涵盖功能与体系 | 依据 |
| :--- | :--- | :--- |
| **科研级验证通过** | RHF/UHF（STO-3G, 6-31G, 6-31G(d), LANL2DZ）、B3LYP/PBE0/CAM-B3LYP 单点能与解析梯度、D3(BJ)/D3(0) 色散、CPCM/COSMO 溶剂能量、14 维内禀振动频率、Boys 函数高精度表 | 与本机 Gaussian 16 高密网格实测或 80 位参考值直接对齐；全量 141 Rust + 5 CUDA + 120 Python 测试通过 |
| **小体系高精度 RI-J 验证通过** | 满秩 `auto-jfit` RI-J SCF 与解析梯度（$|\Delta E| \le 5.8 \times 10^{-5}\text{ E}_\text{h}$ on $\text{H}_2/\text{H}_2\text{O}$）、RTX 5060 Ti GPU RI-J 库仑缩并（$\le 22$ 原子 `IC-2F`）、流式 Becke 网格、磁盘 DIIS 溢出、P-RFO/IRC | 实测 `H2O/6-31G(d)` 误差 `-1.68e-5 Eh`、梯度 `2.33e-5 Eh/Bohr`；受笛卡尔 `cond(V)~10^6` 限制，暂仅在 $\le 22$ 原子体系实测 |
| **功能已实现但未独立外部验证** | TDDFT / CIS（TDA 与 RPA）、RI-J 几何优化大分子应用、ESP 表面与 `CCMSH`/`CCESP` 二进制流 | 内部数学一致性（Thouless Hessian、有限差分、RPA 和规则）通过，缺外部软件逐项对照 |
| **仅理论资源推演** | BTP（159 原子）、Y6（199 原子）、PM6（158 原子）原生 SCF 收敛；726 原子 / 6224 AO `p4D18` 及 2000 原子在 16 GB 内存内完成 | 仅解析输入结构与 Gaussian 日志并建立 $O(N_{\text{AO}}^2)$ 内存模型，未实际运行原生 SCF |
| **存在阻断性问题 / 明确拒绝** | $\omega\text{B97X-D}$、SMD 隐式溶剂、开壳层 SCRF、RI-J 下的 TDDFT 响应、度量秩截断下的 RI-J 解析梯度 | 源码明确报错拒绝或尚未完整实现；大体系全流程 16 GB 峰值内存与 GPU 端到端加速均未达成 |
