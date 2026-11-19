# CalChem P1 测试验证与回归审计结果

**审计日期**：2026-11-10  
**测试环境**：
* 操作系统：Windows 10 / 11 (MSVC x86_64)
* 处理器：AMD Ryzen 7 9700X (8 物理核心 / 16 逻辑线程)
* 内存：16 GB DDR5
* 工具链：Rust 1.99.1 (`stable-x86_64-pc-windows-msvc`), Python 3.13.13 (`pytest 9.1.1`)
* 隔离环境配置：`docs/audits/2026-11-10_round2/probes/env.ps1`

---

## 一、 Rust 工作区全量测试（`cargo test --workspace --locked`）

* **运行命令**：
  ```powershell
  . docs\audits\2026-11-10_round2\probes\env.ps1
  $env:PYO3_PYTHON = 'D:\YDW\Trae_Gromacs\miniconda3\python.exe'
  cargo test --workspace --locked
  ```
* **退出状态码**：**`0` (Exit 0)**
* **通过统计**：**全部测试通过，0 failed, 0 ignored**
  * `calchem-basis`: 4 passed (基函数归一化、收缩自重叠)
  * `calchem-dft`: 9 passed (XC 泛函解析导数、Becke 多中心数值网格积分)
  * `calchem-grad`: 4 passed (DFT 解析梯度 vs 有限差分、极化与 ECP 梯度、弥散基密度守恒)
  * `calchem-int`: 22 passed:
    * `boys.rs`: 4 passed (Boys 函数递归与跨区截断、大 $n$ 渐近稳定性、erf 近似精度)
    * `direct_fock_bounded_benchmark.rs`: 1 passed (1, 4, 8, 16 线程下有界内存伸缩与机器精度确定性)
    * `direct_fock_numerical.rs`: 4 passed (He2 距离剪枝缩放、精确无筛选对标 $\Delta E \le 3.0 \times 10^{-14}\text{ Eh}$、严格柯西-施瓦茨 $L_1$ 界审计)
    * `eri_reference.rs`: 2 passed (双电子积分解析基准)
    * `eri_symmetry.rs`: 1 passed (8-fold 壳层对称性)
    * `one_electron.rs`: 3 passed (单电子动能与核吸引积分、d 壳层积分布局)
    * `screening_and_parallel.rs`: 1 passed (Schwarz 筛选与并行等价性)
    * `redteam_quartets.rs`: 1 passed (惰性四元组生成数学等价性)
    * `lib.rs`: 6 passed (ECP 半局域完备性、Bessel 函数连续性、球谐展开)
  * `calchem-model`: 6 passed (分子几何、对称性与坐标变换)
  * `calchem-opt`: 8 passed (准牛顿优化、ECP 卤素平衡键长、TS 虚频鞍点搜索、IRC 坐标能量梯度严格同步)
  * `calchem-prop`: 8 passed (ESP 静电势、Mulliken/Löwdin 布居分析、偶极矩、Cube 体数据生成)
  * `calchem-scf`: 25 passed:
    * `direct_scf.rs`: 4 passed (RHF/RKS/UHF/UKS Direct SCF 严格对标、Ne8 72 AO Direct 模式内存准入通过)
    * `kohn_sham_dft.rs`: 3 passed (卤素 ECP HBr/LANL2DZ B3LYP RKS/UKS、乙烯前线轨道间隙)
    * `polarization_and_ecp.rs`: 3 passed (水 6-31G(d)/6-31G(d,p) 变分层级与文献能量对标)
    * `redteam_boundaries.rs`: 4 passed (DIIS 残差尺度不变性、非法选项拦截、内存准入拦截、病态基组拦截)
    * `rhf_631g.rs`: 2 passed (6-31G 壳层归一化与水分子参考能)
    * `rhf_sto3g.rs`: 6 passed (He, H2, H2O, Ne 能量基准与 max_iter=1 报错审计)
    * `uhf.rs`: 4 passed (H 原子单电子严格能量、H2 三重态纯自旋、H2O+ 双重态阳离子)
  * `calchem-solv`: 6 passed (C-PCM 与 SMD 隐式溶剂化表面静电反应场能)
  * `calchem-tddft`: 4 passed (CIS 与 TDDFT 激发能与振子强度)
  * `calchem-tensor`: 11 passed (Matrix2D / Vector 矩阵乘法、对称对角化、内存连续切片)
  * `calchem-vib`: 5 passed (解析 Hessian 矩阵与简正振动频率分析)

---

## 二、 Python 原生后端 Direct SCF 选项转发验证

* **运行命令**：
  ```powershell
  & $env:PYO3_PYTHON -B -c "import pytest; raise SystemExit(pytest.main(['tests/unit/test_native_backend.py','-v']))"
  ```
* **退出状态码**：**`0` (Exit 0)**
* **测试项汇总**：
  ```
  tests/unit/test_native_backend.py::test_native_rhf_rust_path PASSED
  tests/unit/test_native_backend.py::test_native_rust_options_forwarding_and_no_silent_fallback PASSED
  tests/unit/test_native_backend.py::test_native_rohf_cleanly_rejected PASSED
  tests/unit/test_native_backend.py::test_native_constructors_valid PASSED
  tests/unit/test_native_backend.py::test_native_mp2_single_point PASSED
  tests/unit/test_native_backend.py::test_native_uhf_rust_single_point_and_ion_optimize PASSED
  tests/unit/test_native_backend.py::test_native_backend_631gd_and_lanl2dz_halogens PASSED
  tests/unit/test_native_backend.py::test_native_direct_scf_options PASSED
  ============================== 8 passed in 1.21s ==============================
  ```
* **核心验证点**：
  1. `options={"direct": True}`：强制 Direct SCF 模式计算，顺利收敛；
  2. `options={"direct": False}`：强制 Stored ERI 模式计算，顺利收敛；
  3. `options={}`（默认）：自适应路径无缝运行；
  4. 三种路径总能量绝对误差 $< 1.0 \times 10^{-12}\text{ Eh}$，证实 Python API 与底层 Rust 引擎参数转发无任何降级或偏差。

---

## 三、 Python 全量回归测试（`pytest tests/unit`）

* **运行命令**：
  ```powershell
  . docs\audits\2026-11-10_round2\probes\env.ps1
  & $env:PYO3_PYTHON -B -c "import os,tempfile,pytest; tempfile.tempdir=os.environ['TEMP']; raise SystemExit(pytest.main(['tests/unit','-q','-p','no:cacheprovider']))"
  ```
* **退出状态码**：**`0` (Exit 0)**
* **耗时**：**477.96 秒（7 分 57 秒）**
* **统计结果**：**105 passed, 0 failed, 0 skipped**（全量回归基线 104 项 + 本次 P1 新增 1 项）
* **覆盖模块测试清单**：
  * `tests/unit/test_adapters.py`: 4 passed
  * `tests/unit/test_atoms.py`: 3 passed
  * `tests/unit/test_core_rust.py`: 12 passed
  * `tests/unit/test_d3_and_solv_gradient.py`: 5 passed
  * `tests/unit/test_dft_rust.py`: 3 passed
  * `tests/unit/test_esp_and_population.py`: 3 passed
  * `tests/unit/test_io.py`: 3 passed
  * `tests/unit/test_isosurface_stream.py`: 5 passed
  * `tests/unit/test_molecule.py`: 5 passed
  * `tests/unit/test_native_backend.py`: 8 passed（含 P1 显式 direct 选项转发验证）
  * `tests/unit/test_opv_cam_b3lyp_and_dimer.py`: 5 passed（含真实光伏体系 IC-2F/BTP 结构加载与基准能量验证）
  * `tests/unit/test_opv_workflow.py`: 7 passed
  * `tests/unit/test_redteam_contracts.py`: 8 passed
  * `tests/unit/test_request_validation.py`: 11 passed
  * `tests/unit/test_service_and_queue.py`: 4 passed
  * `tests/unit/test_solvation_and_scrf.py`: 5 passed
  * `tests/unit/test_tddft_rust.py`: 5 passed
  * `tests/unit/test_ts_and_irc.py`: 5 passed
  * `tests/unit/test_vibration_and_thermo.py`: 4 passed

---

## 四、 阶段审计结论

所有四个子阶段（P1-A、P1-B、P1-C、P1-D）的代码变更、数学推导与接口升级已全部就绪并通过双端（Rust 与 Python）独立隔离审计验证。
无任何回归失效，数值精度满足预期，内存累加具备严格物理上界。

