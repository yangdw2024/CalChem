# 本次修改文件与行号

只对比本次 backup，不将起点已有改动算成本次贡献。格式化变化也在差异中保留。

| 文件 | 状态 | 修改后的行号起点 |
|---|---|---|
| `calchem/_core.pyi` | modified | 26 |
| `calchem/backends/native.py` | modified | 25, 347, 349, 1268 |
| `calchem/solvent/smd.py` | modified | 3, 24, 38 |
| `crates/calchem-cuda/build.rs` | modified | 16, 32, 37, 41, 51 |
| `crates/calchem-cuda/src/cuda_kernels.cu` | modified | 4, 78, 83, 97, 102, 110, 114, 118, 121, 124, 129, 135, 138 |
| `crates/calchem-cuda/src/lib.rs` | modified | 25, 48, 60, 67, 83, 109 |
| `crates/calchem-dft/src/vxc.rs` | modified | 20, 58, 61, 102, 137, 172, 175, 225, 269 |
| `crates/calchem-dft/src/xc.rs` | modified | 29, 31, 37, 84, 86, 781, 832, 844, 850, 925 |
| `crates/calchem-grad/src/lib.rs` | modified | 253, 289, 300, 370, 444 |
| `crates/calchem-int/src/density_fitting.rs` | modified | 56, 101, 106, 179, 207, 392, 393, 425, 431, 489, 505, 653, 703, 715, 732, 762, 773, 775, 778, 809, 811, 817, 829, 870 |
| `crates/calchem-int/src/direct_fock.rs` | modified | 4, 178, 255, 330, 406, 463, 495, 516, 536, 539, 582, 583, 588, 652, 675, 700, 717, 745, 746, 758, 807, 817, 856, 877, 909, 956, 957, 962 |
| `crates/calchem-int/src/eri.rs` | modified | 44, 88, 98, 314 |
| `crates/calchem-model/src/lib.rs` | modified | 299, 331 |
| `crates/calchem-py/src/lib.rs` | modified | 60, 694, 736, 1496, 1990 |
| `crates/calchem-scf/src/diis.rs` | modified | 20 |
| `crates/calchem-scf/src/fock_builder.rs` | modified | 35, 39, 188, 192 |
| `crates/calchem-scf/src/lib.rs` | modified | 61, 70, 73, 81, 83, 86, 136, 141 |
| `crates/calchem-scf/src/rhf.rs` | modified | 78, 234, 275, 298, 314 |
| `crates/calchem-scf/src/rks.rs` | modified | 120, 270, 319, 344, 360 |
| `crates/calchem-scf/src/uhf.rs` | modified | 127, 145, 302, 341, 351, 362 |
| `crates/calchem-scf/src/uks.rs` | modified | 119, 143, 300, 303, 346, 357, 368 |
| `crates/calchem-scf/tests/density_fitting_scf.rs` | modified | 29, 33, 59, 120 |
| `crates/calchem-scf/tests/redteam_boundaries.rs` | modified | 29, 32, 43, 85 |
| `crates/calchem-solv/src/scrf.rs` | modified | 163, 341, 388, 396, 567 |
| `crates/calchem-tddft/src/davidson.rs` | modified | 2, 170, 234, 432, 498 |
| `crates/calchem-tddft/src/lib.rs` | modified | 473 |
| `crates/calchem-tensor/src/lib.rs` | modified | 179, 199, 366 |
| `docs/audits/master_mission/FUNCTIONAL_CAPABILITY_MATRIX.md` | modified | 2 |
| `docs/audits/master_mission/MASTER_MISSION_FINAL_REPORT.md` | modified | 2 |
| `docs/audits/master_mission/MISSION_TRACKER.md` | modified | 2 |
| `docs/audits/master_mission/PERFORMANCE_BENCHMARKS.csv` | modified | 1 |
| `docs/audits/master_mission/REPRODUCIBILITY_GUIDE.md` | modified | 2 |
| `docs/audits/master_mission/RESOURCE_SCALING_MODEL_200_2000_ATOMS.md` | modified | 2 |
| `scripts/cargo_test.ps1` | modified | 6, 7, 12 |
| `scripts/run_pytest.ps1` | modified | 10, 13, 15 |
| `tests/unit/test_gpu_and_density_fitting.py` | modified | 65 |
| `tests/unit/test_opv_cam_b3lyp_and_dimer.py` | modified | 44 |
| `tests/unit/test_p5_gradients_solvation_tddft.py` | modified | 89, 99, 109 |
| `tests/unit/test_redteam_contracts.py` | modified | 19 |
| `tests/unit/test_solvation_and_scrf.py` | modified | 154 |
| `tools/dev/build_ext.ps1` | modified | 18, 22 |
| `crates/calchem-int/tests/independent_audit.rs` | new | 1 |
| `crates/calchem-cuda/tests/ffi_contract.rs` | new | 1 |
| `crates/calchem-tddft/tests/independent_solver.rs` | new | 1 |
| `crates/calchem-grad/tests/rij_provenance.rs` | new | 1 |

详细差异：`evidence/independent-changes.patch`。原/新哈希：`evidence/changes.json`。源码与实际 pyd 哈希：`evidence/source-binary-sha256.json`。

Rust 新增 density_fitting 字段属于公共结构字面量 API 变化；Python 暴露只读来源字段，未实现的科学路径现明确拒绝。

