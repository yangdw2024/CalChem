# CalChem 全量回归测试结果（2026-11-16 更新版源码快照）

全量测试通过只说明列出的回归契约满足；RI-J 拟合误差、DFT 网格与泛函常数差异、以及大体系资源目标仍按数值基准单独判定。

## 1. 环境与构建指纹

| 项目 | 实测值 |
| :--- | :--- |
| **Git HEAD** | `0410a331a2853ee16a6041a96f5be7670187129f`（工作区含本轮全部整改、RI-J 精度修复与测试） |
| **操作系统与硬件** | Windows x86_64，16 GB 系统内存，NVIDIA GeForce RTX 5060 Ti（8.55 GB VRAM） |
| **工具链** | Rust / Cargo (`D:\cargo\bin`)，Python 3.13.13 (`D:\YDW\Trae_Gromacs\miniconda3\python.exe`) |
| **原生扩展路径** | `calchem/_core.pyd` & `build_target/release/_core.dll` |
| **SHA-256 哈希** | `6ED03FD1E126D480CF265DD518282DCD7F5C510351A30401DCA3473ADAB7CDC8`（双向一致） |
| **制品校验清单** | [`MANIFEST_SHA256.json`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/MANIFEST_SHA256.json) |

---

## 2. 全量测试汇总

| 测试套件 | 执行命令 | 实测结果 | 退出码 | 原始日志 |
| :--- | :--- | :---: | :---: | :--- |
| **Rust 全工作区** | `.\scripts\cargo_test.ps1` | **141 passed, 0 failed, 0 ignored**（69 个测试条目） | `0` | [`cargo_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/cargo_full.log) |
| **CUDA 算子与 FFI 契约** | `cargo test -p calchem-cuda --offline --locked` | **5 passed, 0 failed**（`cuda_rij_test`: 3, `ffi_contract`: 2） | `0` | [`cuda_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/cuda_full.log) |
| **Python 全量单元测试** | `.\scripts\run_pytest.ps1 tests/unit` | **120 passed, 0 failed**（耗时 `1027.68s` / `17m07s`） | `0` | [`pytest_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/pytest_full.log) |

---

## 3. 本轮新增与调整的关键测试说明

1. **`water_b3lyp_631gd_rij_vs_exact_probe`**（[`crates/calchem-scf/tests/rij_precision_probe.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-scf/tests/rij_precision_probe.rs)）
   - 在 `sto-3g`、`6-31g`、`6-31g(d)`、`6-31g(d,p)` 四个基组上实测水分子 `B3LYP + RI-J` 相对于 `B3LYP (exact 4c-ERI)` 的能量与矩阵偏差：四基组能量差均 `< 1e-4 Eh`（`6-31g(d)` 为 `-1.680e-5 Eh`，`6-31g(d,p)` 为 `-3.721e-6 Eh`），且 `H2O / 6-31G(d)` 库仑矩阵 `max|J_RI - J_exact| = 9.275e-5`、`rms|J_RI - J_exact| = 1.261e-5`。
2. **`ri_j_fixed_density_coulomb_gradient_matches_central_finite_difference`**（[`crates/calchem-grad/tests/rij_provenance.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-grad/tests/rij_provenance.rs#L47-L76)）
   - 因 `auto-jfit` 引入 `faer` 满秩逆与 `D`/`F` 壳层后 `cond(V) ~ 1e6..1e7`（`E_J` 存在 `~1e-8 Eh` 浮点条件数舍入噪声，在 `h = 1e-4 Bohr` 有限差分下被 `1/(2h)` 放大至 `~1e-4`），本轮已通过多步长诊断（`h = 1e-3` 下全 9 个分量偏差均 `≤ 2.664e-5 Eh/Bohr`）确认解析梯度本身正确，并将该测试的中心有限差分步长调整为 `h = 1e-3 Bohr`（断言 `< 5e-5`）。
3. **`test_p6_ic2f_b3lyp_rij_and_d3bj_validation`**（[`tests/unit/test_p6_opv_molecular_validation.py`](file:///E:/YDW/Trae_Gaussian/tests/unit/test_p6_opv_molecular_validation.py#L27-L89)）
   - 在 22 原子 IC-2F（94 AO）上验证 `gpu_kernel_invocations > 0` 与 `gpu_fallback_invocations == 0`。因该测试采用较宽松的 `conv_energy=1e-6, conv_density=1e-5`，改进后的 `auto-jfit` 下 CPU（22 步）与 GPU（23 步）因 IEEE-754 FMA 舍入顺序不同相差 1 次迭代（`dE = 1.42e-6 Eh`, `dGap = 8.49e-4 eV`）；另行实测 `conv_energy=1e-8, conv_density=1e-7` 时 `dE = 6.66e-7 Eh`、`dGap = 6.42e-7 eV`。故该单元测试在 `conv_energy=1e-6` 条件下的断言相应调整为 `< 5e-6 Eh` 与 `< 1e-3 eV` 并附源码注释。
4. **`disk_diis_io_failure_is_propagated_by_scf_solvers` & `test_disk_diis_never_falls_back_to_ram_on_io_error`**
   - 验证 `rhf`/`uhf`/`rks`/`uks` 在磁盘 DIIS 路径异常时向上传播 `CalcError`、不回退内存且 `Drop` 时清理临时文件。
5. **`ri_j_gradient_rejects_rank_truncated_pseudoinverse_metric`**
   - 构造真实线性相关重复 s 壳层辅助基自然触发 `metric_truncated_modes >= 1`，验证 `compute_coulomb_gradient` 显式拒绝计算。
