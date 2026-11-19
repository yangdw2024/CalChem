# CalChem 全量回归测试结果（2026-11-16 最终验收版）

## 1. 环境与构建指纹

| 项目 | 实测值 |
| :--- | :--- |
| **Git HEAD** | `0410a331a2853ee16a6041a96f5be7670187129f`（工作区含本轮全部整改与测试） |
| **操作系统与硬件** | Windows x86_64，16 GB 系统内存，NVIDIA GeForce RTX 5060 Ti（8.55 GB VRAM） |
| **工具链** | Rust / Cargo (`D:\cargo\bin`)，Python 3.13.13 (`D:\YDW\Trae_Gromacs\miniconda3\python.exe`) |
| **原生扩展路径** | `calchem/_core.pyd` & `build_target/release/_core.dll` |
| **SHA-256 哈希** | `37F96B1A58C64F86DA8ED0A997BDCA2386A04635602E070D5FAA151E32A16F8C`（双向一致） |

---

## 2. 全量测试汇总

| 测试套件 | 执行命令 | 实测结果 | 退出码 | 原始日志 |
| :--- | :--- | :---: | :---: | :--- |
| **Rust 全工作区** | `.\scripts\cargo_test.ps1` | **140 passed, 0 failed, 0 ignored**（68 个测试条目） | `0` | [`cargo_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/cargo_full.log) |
| **CUDA 算子与 FFI 契约** | `cargo test -p calchem-cuda --offline --locked` | **5 passed, 0 failed**（`cuda_rij_test`: 3, `ffi_contract`: 2） | `0` | [`cuda_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/cuda_full.log) |
| **Python 全量单元测试** | `.\scripts\run_pytest.ps1 tests/unit` | **120 passed, 0 failed**（耗时 `995.97s` / `16m35s`） | `0` | [`pytest_full.log`](file:///E:/YDW/Trae_Gaussian/docs/audits/2026-11-16_final_certification/pytest_full.log) |

---

## 3. 本轮新增与强化的关键测试项

1. **`disk_diis_io_failure_is_propagated_by_scf_solvers`**（[`crates/calchem-scf/tests/redteam_boundaries.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-scf/tests/redteam_boundaries.rs#L103-L133)）
   - 通过 `ScfOptions { diis_disk: true, diis_dir: Some("NUL:\\...") }` 同时验证 `rhf`、`uhf`、`rks`、`uks` 四个 SCF 主求解器在 DIIS 磁盘溢出路径不可用时均直接向上返回 `DIIS disk spill I/O error`，禁止吞错与静默退回内存。
2. **`test_disk_diis_never_falls_back_to_ram_on_io_error`**（[`crates/calchem-scf/src/diis.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-scf/src/diis.rs#L482-L513)）
   - 向已创建的真实 `DiisState` 实例注入只读文件句柄（模拟磁盘写失败），确认 `try_push_and_extrapolate` 返回 `Err`、RAM `fock_list` / `err_list` 为空，且 `Drop` 时自动删除 E 盘临时文件。
3. **`ri_j_gradient_rejects_rank_truncated_pseudoinverse_metric`**（[`crates/calchem-grad/tests/rij_provenance.rs`](file:///E:/YDW/Trae_Gaussian/crates/calchem-grad/tests/rij_provenance.rs#L154-L197)）
   - 使用 `RiJEngine::with_auxiliary_basis` 构造**真实线性相关**的重复辅助基组（两个完全重合的 s 壳层），自然触发 `invert_coulomb_metric_with_rank_info` 的 $\lambda < 10^{-12}\lambda_{\max}$ 特征值截断（`metric_truncated_modes >= 1`），并验证 `compute_coulomb_gradient` 显式拒绝计算。
4. **`test_p6_ic2f_b3lyp_rij_and_d3bj_validation`**（[`tests/unit/test_p6_opv_molecular_validation.py`](file:///E:/YDW/Trae_Gaussian/tests/unit/test_p6_opv_molecular_validation.py#L27-L87)）
   - 在 22 原子 IC-2F 真实 OPV 受体上同时执行 CPU RI-J 与 GPU RI-J，并通过 `gpu_kernel_invocations > 0` 与 `gpu_fallback_invocations == 0` 实测遥测确认每次 $J$ 矩阵缩并均真正进入 RTX 5060 Ti CUDA 内核。
