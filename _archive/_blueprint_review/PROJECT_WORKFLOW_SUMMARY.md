# CalChem (Trae_Gaussian) 项目工作流总结

> 本文档由对 `E:\YDW\Trae_Gaussian\` 目录的**实地逆向勘察**生成，而非依据口头描述。
> 所有结论均来自：源码目录结构、文件名、文件时间戳、README、pyproject、CI 配置、测试脚本中的参考能量、打包脚本、GUI 升级说明。
> 证据不足处显式标注 **【推测】** 或 **【未知】**，不脑补。

---

## 第一部分　Project Overview

| 字段 | 内容 |
|---|---|
| 项目名称 | **CalChem**（仓库目录名 `Trae_Gaussian`；内部代号"自己写一个 Gaussian"） |
| 项目背景 | 作者（化学研究生，OSC/有机光电方向）日常依赖商用 Gaussian 16（`D:\G16W`），但希望有一个**面向 OPV、可 GPU 加速、与 GROMACS 对接、带现代 GUI** 的自研引擎；同时借由"重写一遍 Hartree–Fock/DFT/Post-HF"来系统夯实计算化学基础（与作者自述"基础知识薄弱、即将开题"一致） |
| 项目目标 | 用 Python 从零实现：分子输入 → 基组 → 一/双电子积分 → RHF/UHF/DFT SCF → MP2/MP4/CCSD/CCSD(T)/CASSCF → 解析梯度/几何优化 → 频率 → TDDFT/CIS → RESP 电荷 → GROMACS 拓扑，并配 PyQt6 GUI，最终 PyInstaller 打成 `CalChem.exe` |
| 解决的问题 | (1) 商用 Gaussian 闭源、不可定制、OPV 工作流（RESP→GROMACS）需多软件拼接；(2) 作者本人对量子化学算法的"黑箱化"问题 |
| 目标用户 | 第一用户就是作者本人；远期设想为同方向课题组/OPV 研究者 |
| 最终交付物 | `calchem/` Python 包（已成形）+ `calchem` CLI entry-point + `examples/` 示例 + 与 `D:\YDW\CalChem\gui\` 配套的 GUI + `build_calchem_exe.py` 产出的 `CalChem.exe` |
| 项目起始时间 | 2026-07-05（最早一批代码文件时间戳） |
| 项目代码密集期结束 | 2026-07-21（`trae_framework.py` 最后一次大改）；2026-09-07 仅有 `__pycache__` 时间戳变化，属环境触动非代码推进 |
| 参与角色 | **单人开发**（作者本人），无其他协作者；Trae IDE 作为 AI 结对编程工具；Gaussian 16 作为"数值裁判" |
| 主要工具 | Python 3.13 / numpy / scipy / pytest / PyQt6 / PyInstaller / Trae IDE / Gaussian 16 (D:\G16W) / PySCF（依赖列表中，作为第二参照） |
| 主要约束 | Windows 单机（AMD Ryzen 7 9700X / 15.1 GB RAM / 8 GB 显卡）；CPU 与 Gaussian 任务串行；禁止并行跑多个计算 |
| 最终结果（勘察时实际状态） | 内核模块齐全（mol/io/basis/integrals/scf/dft/post_hf/excited/opt/grad/vib/solvent/md/props），**对 H2/He/Ne/H2O/CH3 等小分子在 STO-3G 下的单点能已与参考值对齐**（README_CalChem 列出 7 个已验证能量）；GUI 已升级一轮（轨道能级图、振动动画、片段库）；**但未初始化 git、CI 未真正跑过、GPU 模块为空壳、大基组/大分子能力未在仓库内验证** |
| 是否达到预期 | **部分达到**：算法教学版内核 + 小分子验证 ✅；生产级 GPU 加速 / 几何优化真实可用 / 600+ Gaussian 用例全量回归 ❌（仅见脚本与声明，无结果文件） |

### 200 字总览
> 项目从 2026-07-05 起步，先用两天把 GTO 基组、一/双电子积分（VRR 递推、球谐化）在 H2/He/Ne 上跑通；7 月 5–6 日完成 RHF SCF + DIIS，在 H2O/STO-3G 上对齐 Gaussian 参考能量；7 月 9 日补齐高层 API、ONIOM、IRC/过渡态搜索；7 月 10 日是爆发日——DFT 网格、Lebedev 求积、LDA/PBE/TPSS/B3LYP 泛函、RI 积分、直接 SCF、MP4、CASSCF、以及 PyQt6 GUI 的一次 ~800 行升级同日落地；7 月 11 日收尾 CCSD/CCSD(T) 与全量回归脚本。整个过程以"写一个模块 → 写一个 debug/verify 脚本 → 与高斯参考值比 1e-8 阈值 → 不通过就改积分/网格/符号"为最小循环，根目录留下的 ~85 个 test/debug 文件就是这个循环的化石。

---

## 第二部分　Stage Mapping（阶段划分）

按文件时间戳与模块依赖还原（非机械套用标准 SDLC）：

| Stage | 阶段名称 | 核心目标 | 主要输入 | 主要活动 | 主要输出 | 负责人 |
|---|---|---|---|---|---|---|
| S0 | 脚手架与技术预研 | 立项目、搭包结构、确认 numpy 数值栈 | 教学目标、硬件约束 | 建 `calchem/` 包、pyproject、跑 numpy 原型（`test_eig/einsum/matmul/dot`） | 可 import 的空包、依赖清单 | 作者 |
| S1 | 基组与 GTO 实现 | 能描述任意基函数 | 原子序数、基组表 | `basis/gto.py`、`basis_data.py`（67KB 基组参数表）、球谐变换、基函数归一化 | GTO/BasisSet 类、sto-3g/3-21g/6-31g/def2/cc-pVXZ 参数库 | 作者 |
| S2 | 一/双电子积分 | 算出 H、S、T、V、ERI 矩阵 | 分子几何 + BasisSet | 解析一电子积分、VRR 双电子递推、球谐积分、密度拟合 RI、积分屏蔽 screening | `one_electron.py`(30KB)、`two_electron.py`(23KB)、`ri.py`、`screening.py` | 作者 |
| S3 | RHF SCF 主循环 | H2O/STO-3G 单点能对齐参考 | 上述积分矩阵 | 初猜（核心哈密顿对角化）、Fock 组装、DIIS、收敛判定、RHF/UHF 分支 | `scf/hf.py`(22KB)、`scf/diis.py`、`scf/guess.py` | 作者 |
| S4 | 数值正确性验证 | 把能量钉到 1e-8 | H2/He/Ne/H2O 参考能量（Szabo + Gaussian） | 写 `test_regression.py`、`verify_eri_*.py`、跑小分子、比对 | 7 个已验证参考能量（写入 README_CalChem） | 作者 |
| S5 | DFT 模块 | 在 SCF 里替换/加入 Vxc | 网格、泛函公式 | Lebedev 球网格、LDA(VWN5)/PBE/TPSS/B3LYP/BLYP/PBE0、D3 色散、Vxc 组装、Fock 组装 | `dft/grid.py`、`dft/functionals.py`(35KB)、`dft/vxc`、`dft/fock.py`、`dft/d3.py` | 作者 |
| S6 | Post-HF | MP2/MP4/CCSD/CCSD(T)/CASSCF | 收敛好的 HF 轨道/积分 | 振幅方程、CCSD(T) 微扰修正、CASSCF 活性空间 | `post_hf/mp2.py`、`mp4.py`、`ccsd.py`、`ccsd_t.py`、`casscf.py`(20KB)、`ccd.py` | 作者 |
| S7 | 梯度/优化/频率 | 能量可用于几何优化 | 解析梯度、Hessian | BFGS、Berny、IRC、TS 搜索、数值频率 | `opt/bfgs.py`、`berny.py`、`irc.py`、`ts_search.py`、`grad/*.py`、`vib/frequency.py` | 作者 |
| S8 | 性质与接口 | 能出电荷、多极矩、cube、RESP、GROMACS | 收敛密度 | 布居分析、RESP、cube 输出、GROMACS 拓扑写出 | `props/`、`md/gromacs_io.py`、`md/ff_fit.py` | 作者 |
| S9 | 高层 API 与 CLI | 让外部能一行调用 | 所有底层模块 | `api/high_level.py`(17KB)、`cli/main.py`（click）、YAML 输入 | `single_point()`、`optimize()`、`frequency()`、`calchem run/opt/info` | 作者 |
| S10 | 激发态 | TDDFT/CIS | HF/DFT 响应 | CIS、TDDFT native、光谱 | `excited/cis.py`、`tddft_native.py`(13KB)、`spectra.py` | 作者 |
| S11 | GUI 升级 | 在已有 PyQt6 GUI 上加轨道图/振动动画/片段库 | `D:\YDW\CalChem\gui\` 旧 GUI | 写 `main_window.py`(39KB)、`result_viewer.py`(37KB)、升级说明文档 | `gui_upgrade/` 补丁包（手动拷贝到 GUI 目录） | 作者 |
| S12 | 打包分发 | 打成 CalChem.exe | GUI 源码 + calchem 包 | PyInstaller 脚本、VC++ 运行时说明 | `build_calchem_exe.py`、`dist/CalChem.exe`【推测产物】 | 作者 |
| S13 | AI 工作流框架 | 给自研工具加配置/日志/资源监控 | Trae 开发经验 | `trae_framework.py`(22KB)，含 ConfigManager/ErrorCategory/ResourceStatus | 统一框架核心 | 作者 |

---

## 第三部分　Step-by-Step 详细工作流

> 粒度原则：**一个实质动作 = 一行**。根目录出现的 debug/verify 文件按其文件名校准顺序。

| Step | Stage | 操作 | 输入 | 具体动作 | 工具 | 负责人 | 输出 | 下一步 | 完成标准 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | S0 | 初始化包骨架 | 项目想法 | 建 `calchem/{mol,io,basis,scf,dft,post_hf,...}/__init__.py` | Trae IDE | 作者 | 空包可 import | 2 | `python -c "import calchem"` 通过 |
| 2 | S0 | 写 pyproject | Python 3.10+ | 声明依赖（numpy/scipy/h5py/click/pydantic/pyscf…）、entry-point `calchem` | 手写 | 作者 | pyproject.toml | 3 | `pip install -e .` 成功 |
| 3 | S0 | numpy 原型验证 | 算法直觉 | 跑 `test_eig/eigh/einsum/matmul/dot/flush/ortho_detail` | numpy | 作者 | 确认 LAPACK/einsum 路径正确 | 4 | 对角化、张量收缩与手算一致 |
| 4 | S1 | 实现原子/分子表示 | 元素周期表数据 | `mol/molecule.py`、`mol/geometry.py`、`mol/symmetry.py`、`io/input.py`（xyz/gjf 解析） | 手写 | 作者 | Molecule 类、from_xyz/gjf | 5 | `examples/water.xyz` 读入正确 |
| 5 | S1 | 实现 GTO 基函数 | 基组参数 | `basis/gto.py`：高斯原型、角动量、球谐归一化；`test_gto_detail.py`、`test_basis_norm.py` | numpy | 作者 | 归一化的 GTO、重叠积分自洽 | 6 | 基函数归一化 ∫χ²dV=1 |
| 6 | S1 | 录入基组参数表 | EMSL Basis Set Exchange | 把 sto-3g/3-21g/6-31g(d)/def2-svp/def2-tzvp/cc-pvdz… 写成 `basis_data.py`(67KB) | 手工+脚本 | 作者 | BasisSet 库 | 7 | H2O/STO-3G 加载出 7 个基函数 |
| 7 | S2 | 一电子积分 | 几何+基组 | `integrals/one_electron.py`(30KB)：重叠 S、动能 T、核吸引 V_N | 解析公式 | 作者 | S/T/V 矩阵 | 8 | H2/STO-3G 与 Szabo 书值一致 |
| 8 | S2 | 双电子 ERI 解析 | 基函数四元组 | VRR（Obara–Saika）递推：`debug_vrr.py`、`debug_vrr_coeff.py`、`verify_eri_analytic.py` | 手写递推 | 作者 | (pq|rs) 张量 | 9 | `test_eri_verify.py`、`test_jk_verify.py` 通过 |
| 9 | S2 | 球谐变换 | 笛卡尔 ERI | `debug_spherical.py`(55KB，最长 debug 文件) 处理 (lm) 组合 | 手写 | 作者 | 球谐基 ERI | 10 | 与笛卡尔结果能量差 <1e-8 |
| 10 | S2 | 原子级 ERI 验证 | H/He/Be/Ne/O | `test_be_atom/he_atom/o_atom/ne_atom/p_1e`、`verify_pp_eri.py`、`verify_d_eri.py` | 逐原子 | 作者 | 各角动量块 ERI 正确性 | 11 | 每个原子误差 <1e-8 |
| 11 | S2 | RI 与屏蔽优化 | ERI 全张量 | `ri.py`(密度拟合)、`screening.py`(Schwarz  screening)、`direct_scf.py` | numpy einsum | 作者 | 内存从 O(N⁸) 降到 O(N⁵) 方案 | 12 | 速度提升且能量不变 |
| 12 | S3 | 初猜 | S/T/V | `scf/guess.py`：核哈密顿对角化得到 P 初猜 | numpy.linalg | 作者 | 初始密度矩阵 P⁰ | 13 | 有负本征值告警处理 |
| 13 | S3 | RHF 主循环 | P⁰、ERI | 组装 F = H + G(P)、对角化 C、构造新 P、能量收敛 | `scf/hf.py`(22KB) | 作者 | 单线程 RHF SCF | 14 | H2/STO-3G 收敛 |
| 14 | S3 | DIIS 加速 | 收敛慢的 SCF | `scf/diis.py`：误差向量、F=PFP 投影；`test_h2o_diis_debug.py` vs `test_h2o_nodiis.py` 对比 | 手写 | 作者 | DIIS SCF | 15 | H2O/STO-3G 迭代数 <15 |
| 15 | S4 | 写回归参考表 | Szabo+Gaussian | `test_regression.py`：硬编码 H2/He/Ne/H2O_RHF/H2O_MP2/O2_UHF 参考能量 | 手写 | 作者 | REFERENCES 字典 | 16 | 每个值来自书/G16，不编造 |
| 16 | S4 | 逐分子跑回归 | RHF 内核 | 跑 `test_h2_sto3g/h2o/h2o_analysis/h2o_bohr/h2o_integrals`、`test_he_eri/o_eri` | pytest | 作者 | 小分子能量 | 17 | |ΔE|<1e-8 Eh |
| 17 | S4 | 失败项 debug | 失败测试 | `debug_h2o.py`、`debug_ne_integrals.py`、`debug_ne_detail.py`、`calc_eri_ref.py` | 逐块打印 | 作者 | 定位错误积分/符号 | 16 | 重跑通过 |
| 18 | S5 | DFT 数值网格 | 原子网格 | `dft/grid.py`(19KB)：Lebedev 球点 + 径向 Becke 分区；`test_lebedev.py`、`test_grid_h/he/phi/phi2` | 数值积分 | 作者 | 原子权重网格 | 19 | ∫1 dV = 原子数 |
| 19 | S5 | LDA/VWN5 试水 | 网格密度 | `test_lda_quick/no_diis/debug/convergence/vs_pbe` | 手写 | 作者 | V_xc(LDA) | 20 | H2O/LDA 收敛 |
| 20 | S5 | PBE/GGA | LDA 基础 | `test_pbe_quick/diagnose/gamma_diagnose/convergence/fix`；`dft/functionals.py`(35KB) | 解析梯度 | 作者 | PBE V_xc | 21 | H2O/PBE=-75.5322 Eh（README 已记录） |
| 21 | S5 | 泛函符号修复 | 跑出来能量差 | `test_vxc_sign.py`、`test_vxc_direct.py`、`test_vxc_detailed.py`、`test_lda_debug.py` | 符号审查 | 作者 | 修正 Vxc 符号约定 | 20 | 能量与参考对齐 |
| 22 | S5 | 高阶泛函 + D3 | PBE | B3LYP、TPSS、BLYP、PBE0、`dft/d3.py`（D3 色散） | 手写 | 作者 | 全泛函表 | 23 | H2O/B3LYP=-75.5566 Eh |
| 23 | S5 | DFT Fock 组装 | Vxc+网格 | `dft/fock.py`、`scf/dft.py`：网格上求 Vxc 贡献 | numpy | 作者 | DFT SCF 闭环 | 24 | 与 RHF 共用同一 DIIS |
| 24 | S6 | MP2 | HF 收敛轨道 | `_decompose_mp3.py`、`_full_mp3.py`、`_analyze_mp3.py`、`_correct_mp3.py`、`_validate_mp3.py`（MP2 被反复分解/校正） | 手写 | 作者 | RMP2/UMP2 关联能 | 25 | H2O/MP2=-74.7148 Eh |
| 25 | S6 | MP4 | MP2 基础 | `post_hf/mp4.py`(16KB) | 手写 | 作者 | MP4 能量 | 26 | H2O/MP4=-75.0180 Eh |
| 26 | S6 | CCSD | MP2 初猜 | `post_hf/ccd.py`、`ccsd.py`(10KB)：振幅 t_ij^ab 迭代 | 向量化 | 作者 | CCSD 能量 | 27 | 收敛 |
| 27 | S6 | CCSD(T) | CCSD 收敛 | `_debug_ccsd.py`、`test_ccsd_t_convergence.py`、`ccsd_t.py`、`ccsd_t` 修正项 | 手写 | 作者 | (T) 微扰修正 | 28 | H2O/CCSD(T)=-74.9988 Eh |
| 28 | S6 | CASSCF | 活性空间 | `casscf.py`(20KB)：Fock 算子 + CI 对角化 | 手写 | 作者 | CASSCF(2,2)/H2=-1.1373 Eh | 29 | 与参考对齐 |
| 29 | S7 | 解析梯度 | HF/DFT 收敛 | `grad/hf_grad.py`、`grad/dft_grad.py`、`grad/analytical.py`(12KB) | 手写 | 作者 | 原子力 | 30 | 数值梯度与解析梯度差 <1e-4 |
| 30 | S7 | 几何优化 | 力 | `opt/bfgs.py`、`optimizer.py`、`berny.py` | BFGS | 作者 | 优化器 | 31 | H2O 键长收敛到参考值 |
| 31 | S7 | 过渡态/IRC | 优化器 | `opt/ts_search.py`(10KB)、`irc.py`(6KB) | 手写 | 作者 | TS 搜索、IRC 路径 | 32 | 【推测】仅 H2/质子转移示例验证 |
| 32 | S7 | 频率 | 优化几何 | `vib/frequency.py`：数值 Hessian、简正模式 | 有限差分 | 作者 | 频率/模式向量 | 33 | 无虚频（极小点） |
| 33 | S8 | 性质 | 收敛密度 | `props/population.py`(Mulliken)、`props/charges.py`(RESP)、`props/multipole_native.py`、`props/cube.py` | 手写 | 作者 | 电荷/多极矩/cube 文件 | 34 | 偶极矩与 Gaussian 对比 |
| 34 | S8 | MD 对接 | RESP 电荷 | `md/gromacs_io.py`、`md/ff_fit.py` | 文本写出 | 作者 | GROMACS .itp/.top | 35 | 能被 GROMACS 读入 |
| 35 | S9 | 高层 API | 所有内核 | `api/high_level.py`(17KB)：single_point/optimize/frequency/resp_charges | 装饰器 | 作者 | Python API | 36 | examples/quick_start.py 跑通 |
| 36 | S9 | CLI | API | `cli/main.py`(9.9KB)：click 命令 `run/opt/info/batch` | click+rich | 作者 | `calchem` 命令 | 37 | `calchem run water.xyz --method b3lyp --basis def2-svp` |
| 37 | S9 | YAML 输入 | CLI | 解析 `water_sp.yaml`、`water_opt.yaml` | pyyaml | 作者 | 批量任务 | 38 | `calchem batch batch.yaml` |
| 38 | S10 | CIS | HF 轨道 | `excited/cis.py` | 对角化 | 作者 | 激发态能量/振子强度 | 39 | 与参考值对齐 |
| 39 | S10 | TDDFT | DFT 轨道 | `excited/tddft_native.py`(13KB)、`spectra.py` | 响应求解 | 作者 | UV-Vis 谱 | 40 | 峰位合理【推测】 |
| 40 | S11 | GUI 升级补丁 | 旧 GUI 在 `D:\YDW\CalChem\gui` | 写 `gui_upgrade/windows/main_window.py`(39KB)、`widgets/result_viewer.py`(37KB)：片段库、轨道能级图、振动动画、耗时/内存显示 | PyQt6+matplotlib | 作者 | 升级说明.md（含备份/拷贝指令） | 41 | 手动 copy 到 GUI 目录 |
| 41 | S11 | 手动部署补丁 | 升级文件 | 按说明 copy 文件、装 matplotlib、`python main.py` | cmd copy | 作者 | 升级后 GUI | 42 | 新功能可见、老功能不坏 |
| 42 | S12 | 打包 EXE | GUI+calchem 包 | `build_calchem_exe.py`：PyInstaller --onefile --windowed，`--add-data "calchem;calchem"`、隐藏导入 numpy/scipy/matplotlib/PyQt6、排除 tkinter | PyInstaller | 作者 | `dist/CalChem.exe` | 43 | 双击能起、能算水 |
| 43 | S13 | 嵌入 AI 框架 | 开发中踩坑经验 | `trae_framework.py`：ConfigManager/ErrorCategory/ResourceStatus/事件总线/审查/纠错/审计 | 手写 | 作者 | 统一框架 | 结束 | 后续项目可复用 |

---

## 第四部分　Input → Process → Output（按 Stage）

### Stage S0–S1：脚手架与基组
- **Input**：作者的教学目标、Python 3.13、numpy/scipy。
- **Process**：建包 → 写 pyproject → 跑 numpy 原型（eig/einsum/matmul）→ 实现 GTO 与球谐 → 录入基组参数表。
- **Output**：可 import 的 `calchem` 包、BasisSet 库、Molecule 类。
- **Next**：进入积分实现。

### Stage S2：一/双电子积分
- **Input**：分子几何（H2/He/H2O）、BasisSet。
- **Process**：解析 1e 积分 → VRR 递推 2e 积分 → 球谐化 → 逐原子/逐角动量验证 → RI+screening 优化。
- **Output**：S/T/V/ERI 矩阵。
- **Next**：SCF 主循环。

### Stage S3–S4：RHF SCF 与数值验证
- **Input**：积分矩阵、参考能量（Szabo + Gaussian）。
- **Process**：核哈密顿初猜 → Fock 组装 → 对角化 → DIIS 加速 → 与 1e-8 阈值比对 → 失败项用 debug 脚本逐块定位。
- **Output**：RHF/UHF SCF、回归测试通过、7 个已验证能量写入 README_CalChem。
- **Next**：DFT。

### Stage S5：DFT
- **Input**：SCF 框架、网格生成算法。
- **Process**：Lebedev 网格 → LDA 试水 → PBE/GGA → 修 Vxc 符号 → B3LYP/TPSS/PBE0/D3 → DFT Fock 组装。
- **Output**：`dft/` 全套模块。
- **Next**：Post-HF。

### Stage S6：Post-HF
- **Input**：收敛的 HF 轨道、积分。
- **Process**：MP2（经历 decompose/full/analyze/correct/validate 五轮）→ MP4 → CCSD → CCSD(T) → CASSCF。
- **Output**：关联能与参考对齐。
- **Next**：梯度/优化。

### Stage S7–S8：梯度、优化、性质、MD
- **Input**：收敛波函数。
- **Process**：解析梯度 vs 数值梯度 → BFGS 优化 → TS/IRC → 频率 → Mulliken/RESP/cube/多极矩 → 写 GROMACS 拓扑。
- **Output**：优化器、频率、性质、GROMACS 对接。
- **Next**：API/CLI。

### Stage S9–S10：API/CLI/激发态
- **Input**：所有内核模块。
- **Process**：高层 API 封装 → click CLI → YAML 批量 → CIS/TDDFT。
- **Output**：`calchem` 命令、Python API、UV-Vis。
- **Next**：GUI。

### Stage S11–S12：GUI 与打包
- **Input**：已存在但独立的 PyQt6 GUI（在 `D:\YDW\CalChem\gui`）。
- **Process**：在 `gui_upgrade/` 写新版 main_window/result_viewer → 写安装说明（含备份命令）→ 手动 copy → PyInstaller 打包。
- **Output**：升级后的 GUI、`CalChem.exe`。

### Stage S13：AI 工作流框架
- **Input**：整个开发过程中的踩坑经验。
- **Process**：抽配置/日志/错误/资源监控为统一框架。
- **Output**：`trae_framework.py`，供后续项目复用。

---

## 第五部分　Artifact Flow

| Artifact | 由哪一步产生 | 输入来源 | 谁创建 | 使用工具 | 下一步如何使用 |
|---|---|---|---|---|---|
| 项目立项想法 | S0 | 作者科研需求 | 作者 | 脑 | 指导模块范围 |
| pyproject.toml | Step 2 | 依赖盘点 | 作者 | 手写 | `pip install -e .` |
| GTO/BasisSet | Step 5–6 | 基组表 | 作者 | numpy | 喂给积分 |
| Molecule/xyz 解析 | Step 4 | examples/*.xyz | 作者 | 手写 | 所有计算入口 |
| S/T/V 矩阵 | Step 7 | 几何+基组 | 作者 | 解析公式 | SCF 初猜与 Fock |
| ERI 张量 | Step 8–10 | 四元基函数 | 作者 | VRR | Fock G(P) |
| 参考能量表 | Step 15 | Szabo 书 + D:\G16W | 作者 | 手工录入 | 回归 1e-8 阈值 |
| RHF 收敛结果 | Step 13–14 | 积分+初猜 | 作者 | DIIS | MP2/CCSD 输入 |
| DFT 网格+Vxc | Step 18–23 | 原子、泛函公式 | 作者 | Lebedev | DFT SCF |
| MP2/CCSD/CCSD(T) 能量 | Step 24–28 | HF 轨道 | 作者 | 向量化 | 回归对比 |
| 解析梯度/Hessian | Step 29–32 | 波函数 | 作者 | 有限差分 | BFGS 优化 |
| cube/RESP/top | Step 33–34 | 密度 | 作者 | 文本写出 | VMD/GROMACS |
| high_level API | Step 35 | 所有内核 | 作者 | 装饰器 | CLI/examples |
| calchem CLI | Step 36–37 | API | 作者 | click | 终端用户入口 |
| TDDFT 激发态 | Step 38–39 | DFT 轨道 | 作者 | 对角化 | UV-Vis 输出 |
| GUI 升级包 | Step 40 | 旧 GUI | 作者 | PyQt6 | 手动 copy 到 D:\YDW\CalChem\gui |
| 安装说明.md | Step 40 | 升级文件 | 作者 | 手写 | 指导部署 |
| CalChem.exe | Step 42 | GUI+内核 | 作者 | PyInstaller | 双击分发 |
| trae_framework.py | Step 43 | 踩坑经验 | 作者 | 手写 | 后续项目复用 |

---

## 第六部分　完整项目流程图（Mermaid）

```mermaid
flowchart TD
    Start([Start: 想做面向 OPV 的自研量化引擎]) --> S0[S0 脚手架+pyproject+numpy 原型]
    S0 --> S1[S1 Molecule + GTO + BasisSet]
    S1 --> S2a[S2 一电子积分 S/T/V]
    S1 --> S2b[S2 二电子 ERI/VRR/球谐]
    S2a --> S3[S3 RHF/UHF SCF + DIIS]
    S2b --> S3
    S3 --> QA1{能量 vs 参考<br/>ΔE < 1e-8 ?}
    QA1 -- No --> DBG1[debug_h2o / debug_ne / debug_vrr<br/>逐块打印定位]
    DBG1 --> S2a
    DBG1 --> S2b
    QA1 -- Yes --> S5[S5 DFT 网格/LDA/PBE/B3LYP/D3]
    S5 --> QA2{Vxc 符号/网格<br/>H2O/PBE 对齐?}
    QA2 -- No --> DBG2[test_vxc_sign / test_pbe_diagnose]
    DBG2 --> S5
    QA2 -- Yes --> S6[S6 MP2→MP4→CCSD→CCSD(T)→CASSCF]
    S6 --> QA3{Post-HF 能量对齐?}
    QA3 -- No --> DBG3[_debug_ccsd / decompose_mp3 系列]
    DBG3 --> S6
    QA3 -- Yes --> S7[S7 解析梯度+BFGS+频率]
    S7 --> S8[S8 性质/RESP/cube/GROMACS 拓扑]
    S8 --> S9[S9 high_level API + CLI + YAML]
    S9 --> S10[S10 CIS / TDDFT]
    S10 --> S11[S11 GUI 升级: 轨道图/振动动画/片段库]
    S11 --> Deploy{手动 copy 升级文件<br/>到 D:/YDW/CalChem/gui ?}
    Deploy -- No --> Manual[按安装说明.md 备份+copy]
    Manual --> Deploy
    Deploy -- Yes --> S12[S12 PyInstaller 打包 CalChem.exe]
    S12 --> S13[S13 沉淀 trae_framework.py]
    S13 --> End([End: 教学版内核+GUI+exe 交付])

    %% 并行分支
    S2a -.并行.-> S2b
    S7 -.可与.-> S10
```

---

## 第七部分　Project Timeline

| 时间 | 阶段 | 事件 | 关键产物 | 结果 |
|---|---|---|---|---|
| 2026-07-05 下午 | S0/S1 | 建包、pyproject、LICENSE、.gitignore、pre-commit；跑 numpy 原型（eig/einsum/matmul） | pyproject.toml、test_eig*.py | 包可 import |
| 2026-07-05 晚 | S2 | 一电子积分、H2 STO-3G 第一次跑通、ERI 初版 | test_h2.py、test_h2_sto3g.py、test_eri*.py | H2 单点出数 |
| 2026-07-05 深夜 | S2 返工 | ERI 球谐、Ne 原子、VRR 系数 debug（debug_spherical.py 55KB） | debug_spherical.py、debug_vrr*.py | VRR 递推跑通 |
| 2026-07-06 早 | S3 | DIIS、H2O SCF（nodiis/direct/diis_debug 三版对比） | test_h2o_nodiis.py、test_h2o_diis_debug.py | H2O/RHF/sto-3G = -74.96 Eh ✅ |
| 2026-07-06 白天 | S7/S8 | 梯度、频率、PCM 溶剂、多极矩 | grad/、vib/、solvent/ | 框架就位 |
| 2026-07-09 | S9/S7 | high_level API、ONIOM、IRC、TS 搜索 | api/high_level.py、oniom/、opt/irc.py、opt/ts_search.py | API 成型 |
| 2026-07-10 下午 | S5 爆发 | DFT 网格、Lebedev、LDA/VWN5/PBE/TPSS/B3LYP/Vxc/RI/screening/direct_scf | dft/*、integrals/ri.py、screening.py | H2O/PBE、H2O/B3LYP 对齐 |
| 2026-07-10 傍晚 | S6 | MP4、CASSCF | post_hf/mp4.py、casscf.py | 关联能扩展 |
| 2026-07-10 夜 | S11 | GUI 升级 ~800 行，含安装说明 | gui_upgrade/main_window.py、result_viewer.py、安装说明.md | GUI 第二版 |
| 2026-07-10 深夜 | S12 | 打包脚本 | build_calchem_exe.py、README_CalChem.md | EXE 流程固化 |
| 2026-07-11 | S6/S2 收尾 | CCSD、direct_scf、test_regression 全量 | post_hf/ccsd.py、integrals/direct_scf.py、test_regression.py | CCSD(T) 对齐 |
| 2026-07-21 | S13 | trae_framework.py 定稿 22KB | trae_framework.py | AI 工作流框架 |
| 2026-09-07 | （非开发） | `__pycache__`/`.pytest_cache`/`.github` 时间戳刷新 | 无代码变化 | 环境被触动，代码未推进 |

> **【未知】**：项目没有 commit 历史（非 git 仓库），以上时间线完全依据文件 mtime 推断；同一天内的先后顺序按文件大小/依赖关系推回。

---

## 第八部分　Roles & Responsibilities

| 角色 | 主要职责 | 负责阶段 | 输入 | 输出 | 与谁交接 |
|---|---|---|---|---|---|
| 作者（PM+架构师+开发+测试+文档） | 全流程 | S0–S13 | 教科书、Gaussian 参考 | 代码、测试、README | 自己 |
| Trae IDE（AI 结对） | 代码生成、原型、补全 | S0–S13 | 作者 prompt | Python 代码 | 作者审阅 |
| Gaussian 16 (`D:\G16W`) | 数值裁判 | S4、S6 回归 | gjf 输入 | out/log 参考能量 | 作者比对 |
| PySCF（依赖） | 第二参照【推测】 | S4 | 同分子 | 能量 | 作者比对 |
| 作者导师/答辩委员会【推测】 | 阶段性方向确认 | 开题前 | 项目进度 | 反馈 | 作者 |

### RACI（简化）

| 活动 | R | A | C | I |
|---|---|---|---|---|
| 模块设计 | 作者 | 作者 | — | — |
| 代码实现 | Trae IDE + 作者 | 作者 | — | — |
| 数值正确性 | 作者 | 作者 | Gaussian 16 | — |
| 文档撰写 | 作者 | 作者 | — | — |
| 打包分发 | 作者 | 作者 | — | 自己 |
| 答辩/方向 | 作者 | 导师 | — | 答辩委员会 |

---

## 第九部分　Tool Stack

| 工具 | 使用阶段 | 用途 | 输入 | 输出 | 频率 |
|---|---|---|---|---|---|
| Python 3.13 | 全程 | 主语言 | — | .py | 每天 |
| numpy / scipy | S2–S10 | 矩阵/线性代数 | 数组 | 矩阵运算 | 每个模块 |
| pytest | S4/S6 | 回归 | test_*.py | 通过/失败 | 每个模块完成后 |
| Trae IDE | 全程 | AI 结对写码 | prompt | .py | 每天 |
| Gaussian 16 (`D:\G16W`) | S4/S6 | 参考能量 | gjf | log | 关键节点 |
| PySCF | 【推测】S4 | 第二参照 | 分子 | 能量 | 偶尔 |
| PyQt6 | S11 | GUI | Python 代码 | 桌面窗口 | 升级时 |
| matplotlib | S11 | 轨道能级图 | 本征值 | 图 | GUI 内 |
| PyInstaller | S12 | 打包 | main.py | CalChem.exe | 发布时 |
| click + rich | S9 | CLI | 命令行 | 输出 | 用户调用时 |
| PyYAML | S9 | 批量输入 | .yaml | 任务列表 | 批量时 |
| h5py / tables | 【声明】 | 检查点 | — | h5 | 未实际使用证据 |
| ruff / mypy | 【模板】 | lint | 代码 | 建议 | pre-commit 未生效 |
| GitHub Actions | 【模板】 | CI | push | 测试 | **未真正运行**（无 .git） |
| VS Code / 文件资源管理器 | 全程 | 编辑/拷贝 | — | — | 每天 |
| cmd `copy` | S11 | 部署 GUI 补丁 | 文件 | 目标目录 | 升级时 |

### Project Tool Stack（流程→工具）

```
Research/参考  → Gaussian 16 (D:\G16W) + Szabo 教材 + PySCF
Design/实现    → Trae IDE + numpy/scipy
数值验证       → pytest + 手写 debug_*.py + verify_*.py
DFT 网格       → Lebedev 求积 + 数值积分
Post-HF        → numpy einsum 向量化
GUI            → PyQt6 + matplotlib
打包           → PyInstaller
分发           → 手动 copy + CalChem.exe
沉淀           → trae_framework.py
```

---

## 第十部分　Dependencies

| Step | 前置依赖 | 类型 | 是否阻塞 | 可并行 |
|---|---|---|---|---|
| Step 7 (1e 积分) | Step 5 (GTO) | Serial | 是 | — |
| Step 8 (2e ERI) | Step 5 | Serial | 是 | 与 Step 7 并行 |
| Step 13 (RHF) | Step 7+8 | Serial | 是 | — |
| Step 14 (DIIS) | Step 13 | Serial | 是 | — |
| Step 16 (回归) | Step 14 | Blocking | 是 | — |
| Step 18 (网格) | 与 RHF 无依赖 | Parallel | 否 | 可在 RHF 阶段并行 |
| Step 24 (MP2) | Step 13 (HF 轨道) | Blocking | 是 | — |
| Step 27 (CCSD(T)) | Step 26 (CCSD) | Serial | 是 | — |
| Step 29 (梯度) | Step 13/23 | Blocking | 是 | 与 Post-HF 并行 |
| Step 30 (BFGS) | Step 29 | Serial | 是 | — |
| Step 35 (API) | 所有底层模块 | Blocking | 是 | — |
| Step 36 (CLI) | Step 35 | Serial | 否 | — |
| Step 40 (GUI 升级) | GUI 旧版存在于 D:\YDW\CalChem\gui | External | 是（硬编码路径） | 与内核收尾并行 |
| Step 42 (打包) | Step 35+40 | Blocking | 是 | — |
| CI 检查 | git 仓库 + remote | External | **当前不满足** | — |

---

## 第十一部分　关键决策点

| Decision | 发生阶段 | 条件 | Option A | Option B | 最终选择 | 原因 |
|---|---|---|---|---|---|---|
| 自己实现还是基于 PySCF 二次开发 | S0 | 教学目标 | 从零写 | 封装 PySCF | **从零写** | 夯实基础、可定制 OPV 工作流（pyproject 仍列 pyscf 为依赖，混用） |
| 用 numpy 还是 GPU(CuPy) 起步 | S0 | 8GB 显卡、CPU 15GB RAM | CPU+numpy 先跑通 | 直接上 CuPy | **CPU 先跑通**，gpu/ 仅留空壳 | 先保正确性，GPU 后置 |
| 参考值来源 | S4 | 需要裁判 | 只信书 | 书+Gaussian 16 | 两者并用 | 书值（Szabo）用于教学，G16 用于真实基准 |
| 单原子验证顺序 | S2 | 哪个最简单 | H→He→Be→Ne→O | 反之 | H/He/Be/Ne/O | 角动量从 s→p→d 递增 |
| Vxc 符号怎么定 | S5 | 能量不对 | 解析推导 | 对比 PBE 参考反推 | 反推 + `test_vxc_sign.py` 钉死 | 实际用对比法 debug |
| GUI 放哪 | S11 | 仓库边界 | 同仓库 | 独立目录 | **独立目录** `D:\YDW\CalChem\gui` | 打包脚本硬编码了此路径 |
| 补丁如何部署 | S11 | 无 git | 自动 pip | 手动 copy | **手动 copy + 备份 .backup** | 权限/路径限制（安装说明原文） |
| 是否初始化 git | 全程 | 协作需要 | 是 | 否 | **否**【现状】 | 【未知】可能未规划协作，或后期补 |
| 是否启用 CI/pre-commit | 全程 | 质量门槛 | 是 | 否 | **模板化但未启用** | 无 .git，CI 永远不会触发 |

---

## 第十二部分　QA 与验收

| QA Step | 检查什么 | 检查方式 | 标准 | 负责人 | 不通过怎么办 |
|---|---|---|---|---|---|
| Q1 基函数归一 | ∫χ²dV | `test_basis_norm.py` | ≈1.0 | 作者 | 修 GTO 归一化系数 |
| Q2 1e 积分 | H2/STO-3G S/T/V | 对照 Szabo 书 | <1e-6 | 作者 | 查解析公式 |
| Q3 ERI | 对已知原子（He/Ne/O） | `verify_eri_analytic/systematic/spherical` | <1e-8 | 作者 | debug_vrr / debug_spherical |
| Q4 RHF 能量 | H2/He/Ne/H2O | `test_regression.py` 参考表 | <1e-8 Eh | 作者 | debug_h2o / debug_ne |
| Q5 SCF 收敛 | 迭代数、能量漂移 | DIIS 收敛日志 | <15 次、梯度<1e-6 | 作者 | 加 DIIS / 换初猜 |
| Q6 DFT 网格 | ∫1 dV | `test_grid_h/he/phi` | 等于原子数 | 作者 | 修 Lebedev 权重 |
| Q7 Vxc 符号 | 能量符号 | `test_vxc_sign/direct/detailed` | 与 PBE 参考一致 | 作者 | 改符号约定 |
| Q8 DFT 能量 | H2O/PBE/B3LYP | README_CalChem 表 | -75.5322 / -75.5566 Eh | 作者 | 修泛函实现 |
| Q9 MP2 关联能 | H2O | REFERENCES | -74.7148 Eh | 作者 | decompose/full/analyze 五连 |
| Q10 CCSD(T) 收敛 | 振幅收敛 | `test_ccsd_t_convergence` | <1e-8 | 作者 | _debug_ccsd |
| Q11 梯度数值一致 | 解析 vs 数值梯度 | 手动对比 | <1e-4 | 作者 | 修解析梯度 |
| Q12 GUI 升级 | 老功能不坏 | 手动跑 main.py | 不崩 | 作者 | 回滚 .backup |
| Q13 EXE 启动 | 双击 | 真实运行 | 能算水 | 作者 | 修 --add-data / hidden-import |
| Q14 CI | push 触发 | GitHub Actions | 3 个 Python 版本 × 2 OS | — | **当前未运行** |

### 最终交付前必须通过的检查
1. 7 个 README_CalChem 列出的参考能量全部 <1e-8 ✅（已记录）
2. `pytest tests/ -v -m "not slow and not gpu"` 绿
3. `python examples/quick_start.py` 跑通
4. `calchem run examples/water.xyz --method b3lyp --basis sto-3g` 出 JSON
5. GUI 升级后老功能不坏
6. CalChem.exe 双击起窗

---

## 第十三部分　Rework Loops

### Rework Loop 1：ERI 球谐与 VRR
- **问题出现在**：S2（Step 8–9）
- **问题**：笛卡尔 ERI 转到球谐基时能量不对；Ne 原子块差很多。
- **证据文件**：`debug_spherical.py`(55KB)、`debug_vrr.py`、`debug_vrr_coeff.py`、`debug_ne_integrals.py`、`debug_ne_detail.py`、`verify_pp_eri*.py`、`verify_d_eri.py`。
- **返回哪一步**：Step 8（ERI 递推系数）。
- **重做什么**：重新核对 Obara–Saika 递推、l/m 组合、归一化因子。
- **完成判据**：H2O/RHF 能量与参考差 <1e-8。

### Rework Loop 2：H2O SCF 不收敛
- **问题出现在**：S3（Step 13–14）
- **证据文件**：`test_h2o_nodiis.py`、`test_h2o_diis_debug.py`、`test_h2o_quick.py`、`test_h2o_direct.py`、`debug_h2o.py`。
- **返回**：Step 14 DIIS。
- **完成判据**：<15 次迭代收敛。

### Rework Loop 3：DFT Vxc 符号/网格
- **问题出现在**：S5。
- **证据文件**：`test_vxc_sign.py`、`test_vxc_direct.py`、`test_vxc_detailed.py`、`test_pbe_gamma_diagnose.py`、`test_pbe_diagnose.py`、`test_lda_debug.py`、`test_lda_no_diis.py`、`test_tpss_fix.py`、`test_grid_phi/phi2/he/h/verify`。
- **返回**：Step 18（网格）/ Step 21（Vxc 符号）。
- **完成判据**：H2O/PBE=-75.5322、B3LYP=-75.5566。

### Rework Loop 4：MP2 公式推导
- **问题出现在**：S6 Step 24。
- **证据文件**（根目录 `_*.py` 前缀明显是临时分析稿）：`_decompose_mp3.py`、`_full_mp3.py`、`_analyze_mp3.py`、`_correct_mp3.py`、`_validate_mp3.py`、`_debug_dft.py`、`_debug_eri.py`。
- **返回**：Step 24。
- **完成判据**：H2O/MP2=-74.7148。

### Rework Loop 5：CCSD(T) 收敛
- **问题出现在**：S6 Step 27。
- **证据文件**：`_debug_ccsd.py`、`test_ccsd_t_convergence.py`。
- **返回**：CCSD 振幅迭代。
- **完成判据**：H2O/CCSD(T)=-74.9988。

### 通用返工回路（模式）

```
写模块 → 跑小分子 → 能量不对
   ↓
写 debug_xxx.py 逐块打印
   ↓
对比参考值定位到积分/网格/符号/初猜
   ↓
修源码
   ↓
重跑 test_xxx.py
   ↓
ΔE<1e-8 → 进下一模块
```

---

## 第十四部分　异常流程

| 异常 | 出现在哪一步 | 原因 | 如何处理 | 最终结果 |
|---|---|---|---|---|
| 打包后缺模块 | Step 42 | PyInstaller 未收集隐藏导入 | 加 `--hidden-import numpy/scipy/matplotlib/PyQt6`、`--add-data` | README_CalChem FAQ 记录 |
| EXE 体积过大 | Step 42 | 含 tkinter 等 | `--exclude-module tkinter` | 已写进打包脚本 |
| EXE 闪退 | Step 42 | 缺 VC++ Redist | 文档提示安装 | README_CalChem FAQ |
| GUI 升级文件权限被拒 | Step 41 | 直接写 `D:\YDW\CalChem\gui` 失败 | 写到 `gui_upgrade/` 再手动 copy | 安装说明.md 诞生 |
| GPU 不可用 | 所有 GPU 测试 | 无 CuPy / 8GB 显存 | conftest.py 自动 skip `gpu` 标记测试 | CI 命令显式 `-m "not gpu"` |
| 大内存任务 | Step 26–28 | 15.1GB RAM | 限制串行、CPU 9700X 8 核 | 与用户偏好"严禁并行 Gaussian/训练"一致 |
| CI 永远不跑 | Step 14 之前 | 未 `git init`、无 remote | **未处理** | `.github/workflows/ci.yml` 形同虚设 |
| 根目录散落 ~85 个 test/debug | 全程 | 没有用 tests/ 统一收纳 | 仅 `tests/` 下有 3 个正式测试，其余在根目录 | 技术债，未清理 |
| 9/7 代码无推进 | 项目后期 | 【未知】可能作者转向开题/其他任务 | — | 项目处于"暂停但可用"状态 |

---

## 第十五部分　Handoff

| From | To | 交接内容 | Artifact | 验收标准 | 风险 |
|---|---|---|---|---|---|
| 算法公式（教科书） | 代码 | 积分/SCF 公式 | 手写笔记【未知】 | 数值对齐 | 记错符号 |
| 代码 | 数值验证 | 模块可调用 | test_*.py | <1e-8 | 测试覆盖不足 |
| 内核 | API | 函数签名 | high_level.py | quick_start 跑通 | API 漂移 |
| API | CLI | 命令参数 | cli/main.py | 命令行出 JSON | 参数不一致 |
| 内核 | GUI | 计算结果 dict | calchem_bridge.py【推测在 GUI 侧】 | GUI 能显示 | 字段缺失 |
| 本仓库 | GUI 目录 | 升级补丁 | gui_upgrade/* | 手动 copy 命令可执行 | 路径硬编码 D:\YDW |
| GUI+内核 | EXE | 全部依赖 | build_calchem_exe.py | 双击运行 | hidden-import 漏 |
| 作者 | 未来的自己/课题组 | 可复用资产 | README + trae_framework | 下一个人能跑 | 根目录混乱 |
| 作者 | 答辩委员会 | 项目意义【推测】 | PPT/口头 | 通过 | 基础薄弱需补课 |

---

## 第十六部分　隐藏工作（容易被忽略但实际存在）

通过文件名与文档反向识别：

1. **根目录 ~85 个 test/debug/verify 脚本**——每个都是一次"写→跑→定位"循环，未被计入正式 `tests/`。
2. **基组参数手工录入**（67KB 的 basis_data.py）——这是体力活，没体现在任何 README。
3. **球谐化的反复调试**（55KB 的 debug_spherical.py）。
4. **Vxc 符号的反向工程**（test_vxc_sign/direct/detailed 三连发）。
5. **MP2 公式拆解**（_decompose/_analyze/_correct/_validate 四件套）。
6. **手动备份 `.backup` 与 copy 命令**（安装说明.md 明示）。
7. **环境维护**：pip 安装 PyInstaller、matplotlib、VC++ 运行时。
8. **参考值的"裁判管理"**：D:\G16W\tests\gjf\ 600+ 用例的存在意味着作者在跑 Gaussian 对照。
9. **文档撰写**：README、README_CalChem、examples/README、benchmarks/README、安装说明。
10. **trae_framework.py 沉淀**（22KB）——把踩过的坑抽象成可复用框架。

---

## 第十七部分　Happy Path（1 分钟主流程）

```
想做 OPV 自研量化引擎
   ↓
搭 calchem 包骨架 + numpy 原型
   ↓
Molecule + GTO + BasisSet
   ↓
一/二电子积分（VRR/球谐）
   ↓
RHF + DIIS → 能量对齐 Gaussian 1e-8
   ↓
DFT（网格+LDA/PBE/B3LYP/D3）
   ↓
MP2/CCSD/CCSD(T)/CASSCF
   ↓
梯度 → BFGS 优化 → 频率
   ↓
RESP/cube/GROMACS 拓扑
   ↓
high_level API + calchem CLI
   ↓
TDDFT
   ↓
PyQt6 GUI（手动 copy 升级包）
   ↓
PyInstaller → CalChem.exe
   ↓
沉淀 trae_framework.py
   ↓
交付
```

---

## 第十八部分　Full Execution Path（含决策/异常/返工）

完整版本已在第六部分 Mermaid 图中画出，包含：
- 3 个 QA 决策点（ΔE<1e-8? / Vxc 对齐? / Post-HF 对齐?）
- 3 条返回路径（ERI 返工、DFT Vxc 返工、CCSD(T) 返工）
- 1 个外部依赖决策（GUI 手动 copy）
- 1 个并行分支（1e 与 2e 积分并行；梯度与 Post-HF 可并行）
- 1 个未解决异常（CI 因非 git 仓库而失效）

---

## 第十九部分　Critical Path（不能跳过的步骤）

```
Step 5  GTO/BasisSet
   ↓
Step 7+8  S/T/V + ERI
   ↓
Step 13+14  RHF + DIIS
   ↓
Step 16+17  回归到 1e-8（含返工）
   ↓
Step 24+27  MP2 → CCSD(T)
   ↓
Step 29+30  梯度 → BFGS
   ↓
Step 35+36  API → CLI
   ↓
Step 40+41  GUI 升级
   ↓
Step 42  CalChem.exe
```

**为什么不能跳**：
- GTO/积分不对，后面所有能量都是错的；
- 不先到 1e-8 就做 Post-HF，关联能根本无法验证；
- 梯度不对，几何优化就是垃圾；
- API/CLI 是唯一让别人用起来的入口；
- EXE 是分发的最终形态。

GPU 加速、TDDFT、ONIOM、TS/IRC、溶剂化在当前阶段**不在关键路径上**，属于可后置扩展。

---

## 第二十部分　可复用资产

| 可复用资产 | 来源 | 用途 | 下次如何使用 |
|---|---|---|---|
| `calchem/` 包骨架 | S0 | 任何量化衍生工具起点 | 直接 import 或 fork |
| 参考能量表（test_regression.py 中 REFERENCES） | S4 | 新模块上线的回归基线 | 每加一个方法就在这加一行 |
| VRR 双电子积分实现 | S2 | 自研积分类项目 | 直接复用，已对 H/He/Ne/O 验证 |
| DIIS 实现 | S3 | 任何 SCF 类迭代 | scf/diis.py |
| Lebedev 网格 + LDA/PBE/B3LYP 泛函 | S5 | DFT 类项目 | dft/ 整目录复用 |
| MP2/CCSD/CCSD(T) 振幅代码 | S6 | 后 HF 起点 | post_hf/ |
| PyInstaller 打包参数 | S12 | PyQt6+numpy 项目打包 | build_calchem_exe.py 模板 |
| GUI 升级说明模板（备份→copy→依赖→故障排除） | S11 | 任何 PyQt6 补丁分发 | gui_upgrade/安装说明.md |
| trae_framework.py | S13 | 配置/日志/资源监控 | 任何 Trae 项目复用 |
| conftest.py 自动 skip GPU 测试 | S4 | 多环境 CI | 直接复用 |
| "写一个 debug_xxx.py 逐块打印"调试法 | 全程 | 数值代码 debug | 作为团队规范 |
| 1e-8 Eh 阈值 | S4 | 数值正确性验收 | 写进任何数值测试 |

---

## 第二十一部分　Standard Operating Procedure（SOP 模板）

### Step 1：新量子化学模块上线 SOP

#### Purpose
保证新模块（方法/泛函/性质）在发布前数值正确、不破坏既有结果。

#### Input
- 算法公式（教科书/论文）
- 一个最小测试分子（H2/He/H2O）
- 参考能量（来自 Szabo 或 D:\G16W 的 gjf）

#### Prerequisite
- 上一个关键路径模块已通过 1e-8 回归
- numpy 可 import
- 参考能量已硬编码进 `REFERENCES`

#### Action
1. 在 `calchem/<module>/` 写实现；
2. 立刻在根目录写 `test_<module>_<case>.py`（先放根目录，跑通后再迁 tests/）；
3. 跑最小分子；
4. 不对就写 `debug_<module>.py`，逐块打印中间矩阵；
5. 对照参考定位错误（符号/归一化/网格/初猜）；
6. 修源码，重跑；
7. 通过后把参考值加进 README_CalChem 的"已验证"表；
8. 加进 `test_regression.py` 的 REFERENCES；
9. 跑全量 `pytest -v -m "not slow and not gpu"`。

#### Tool
numpy / scipy / pytest / Trae IDE / Gaussian 16。

#### Output
- 模块代码、测试、参考值登记、回归绿。

#### QA
- 能量差 <1e-8 Eh；
- 不破坏已有 7 个参考能量。

#### Completion Criteria
- `pytest tests/` 全绿；
- 新参考值写入 README。

#### Next Step
进 API 封装 → CLI 参数。

#### Exception
- 数值不收敛：先加 DIIS / 换初猜 / 降基组；
- GPU 测试失败：conftest 自动 skip，不阻塞；
- 找不到参考值：用 G16 现跑一个 gjf 作裁判。

---

### Step 2：GUI 补丁分发 SOP

#### Purpose
在 GUI 独立目录（`D:\YDW\CalChem\gui`）权限受限时安全升级。

#### Action
1. 在仓库 `gui_upgrade/` 写新版文件；
2. 写 `安装说明.md`，包含：备份原文件为 `.backup`、copy 命令、新增依赖、向后兼容性、故障排除；
3. 让用户手动 copy；
4. 跑 `python main.py` 验证老功能。

#### Output
升级后的 GUI + 可回滚备份。

#### Exception
- 新功能崩：copy `.backup` 回滚。

---

### Step 3：打包 EXE SOP

#### Action
1. 确认 `build_calchem_exe.py` 里的 `CALCHEM_PATH`、`GUI_DIR` 是绝对路径；
2. 清理 `dist/`、`build/`、`.spec`；
3. 跑 `python build_calchem_exe.py`；
4. 在干净机器上双击 EXE，确认能算水。

#### Exception
- 缺模块：补 `--hidden-import`；
- 闪退：装 VC++ Redist。

---

## 第二十二部分　下次项目执行 Checklist

```
☐ 明确项目目标（自研/复用二次开发/纯教学）
☐ 确认硬件预算（CPU/GPU/内存）与串行约束
☐ 先 git init + 远端 + CI 真跑一次（避免本项目的坑）
☐ 写 pyproject 并 pip install -e .
☐ 跑 numpy 原型验证数值栈
☐ 定参考值来源（书 + Gaussian 16/PySCF）并硬编码
☐ 实现 Molecule + IO（xyz/gjf）
☐ 实现 GTO + BasisSet，归一化自检
☐ 一电子积分 → H2/He 对齐
☐ 二电子积分 → VRR → 球谐 → 逐原子验证
☐ RHF + 初猜 + DIIS → H2O/STO-3G <1e-8
☐ 登记参考能量到 README
☐ DFT 网格 → LDA → PBE → 高阶泛函（先修 Vxc 符号）
☐ MP2 → CCSD → CCSD(T) → CASSCF
☐ 解析梯度 vs 数值梯度
☐ BFGS 优化 + 频率
☐ 性质（电荷/cube/多极矩）
☐ RESP + GROMACS 拓扑
☐ high_level API
☐ click CLI + YAML 批量
☐ TDDFT（可后置）
☐ GUI 升级（先写安装说明.md）
☐ PyInstaller 打包
☐ 干净机器双击验证 EXE
☐ pytest 全绿（-m "not slow and not gpu"）
☐ 把根目录散落 test/debug 迁到 tests/（技术债）
☐ 沉淀 trae_framework 类资产
☐ 归档：README 更新、参考表更新、示例跑通
```

---

## 第二十三部分　One-Page Workflow（管理者 3 分钟版）

```
目标：做一个面向 OPV、GPU-ready、带 GUI 的自研量化引擎 CalChem
   ↓
输入：分子几何（xyz/gjf）、基组、泛函/方法参数
   ↓
Stage 1  内核地基      Molecule → GTO → 1e/2e 积分
Stage 2  SCF 闭环      RHF/UHF + DIIS，能量对齐 Gaussian 1e-8
Stage 3  DFT/Post-HF   LDA/PBE/B3LYP/D3 + MP2/CCSD(T)/CASSCF
Stage 4  优化与性质     解析梯度 → BFGS → 频率 → RESP/GROMACS
Stage 5  接口          Python API + calchem CLI + YAML 批量
Stage 6  GUI/分发       PyQt6 升级 → 手动 copy → CalChem.exe
   ↓
QA：每个小分子能量与参考差 <1e-8 Eh；pytest 绿；EXE 双击起
   ↓
交付：calchem 包 + CLI + GUI + EXE + trae_framework
```

- **核心角色**：作者（全栈）+ Trae IDE（结对）+ Gaussian 16（数值裁判）
- **核心工具**：Python/numpy/scipy/pytest/PyQt6/PyInstaller/G16
- **核心产物**：`calchem/` 包、7 个已验证能量、GUI 升级包、`CalChem.exe`
- **关键检查点**：S4 回归、S5 Vxc 符号、S6 CCSD(T) 收敛、S12 EXE 启动
- **关键决策**：从零写而非封装 PySCF；CPU 先跑通再上 GPU；GUI 独立目录+手动 copy
- **核心风险**：CI 未真正启用、根目录技术债、GPU 是空壳、大基组未验证

---

## 第二十四部分　最终项目工作流总结

### 项目一句话总结
> 这个项目通过**单人 + Trae IDE 结对**，在 **~17 天（2026-07-05 ～ 07-21）** 内，以 **"写一个模块 → 写一个 debug 脚本 → 与 Gaussian 16/教科书参考值比 1e-8 Eh → 不通过就逐块定位返工"** 为最小循环，从零搭出了一套覆盖 **分子输入→积分→RHF/DFT→MP2/CCSD(T)/CASSCF→梯度/优化/频率→RESP/GROMACS→API/CLI→PyQt6 GUI→EXE** 的量子化学引擎 CalChem，并把过程中的工程经验沉淀为 `trae_framework.py`。

### 完整主流程
```
Start
 → S0 脚手架 → S1 GTO/BasisSet → S2 1e+2e 积分
 → S3 RHF+DIIS → Q(1e-8)? ──No──> debug/返工 ──┐
                                                  │
 → S5 DFT → Q(Vxc)? ──No──> debug ───────────────┤
 → S6 MP2/CCSD(T) → Q(收敛)? ──No──> debug ─────┤
 → S7 梯度/优化/频率 → S8 性质/MD
 → S9 API/CLI → S10 TDDFT
 → S11 GUI 升级（手动 copy）→ S12 CalChem.exe
 → S13 trae_framework 沉淀
End
```

### 核心输入
- 有机光电分子几何（xyz/gjf）
- 基组（sto-3g/def2/cc-pVXZ…）
- 泛函/方法参数
- 教科书公式 + Gaussian 16 参考能量

### 核心输出
- `calchem/` Python 包
- `calchem` CLI
- 7 个对齐到 1e-8 的参考能量（H2/He/Ne/H2O/CH3/H2）
- PyQt6 GUI（独立目录）
- `CalChem.exe`
- `trae_framework.py`

### 核心角色
作者（全栈单人）、Trae IDE、Gaussian 16（数值裁判）。

### 核心工具
Python 3.13、numpy/scipy、pytest、PyQt6、matplotlib、PyInstaller、G16、Trae IDE。

### 核心 Artifact
pyproject.toml、basis_data.py、one_electron.py、two_electron.py、scf/hf.py、scf/diis.py、dft/functionals.py、post_hf/ccsd.py、test_regression.py、high_level.py、cli/main.py、gui_upgrade/、build_calchem_exe.py、trae_framework.py。

### 核心 Decision
从零写 vs 封装 PySCF（选了从零）、CPU 先跑通 vs 直接 GPU（选 CPU）、GUI 同仓库 vs 独立（选独立）、补丁自动 vs 手动 copy（选手动）。

### 核心 QA
每个小分子能量 |ΔE| < 1e-8 Eh；pytest 绿；EXE 双击起；GUI 老功能不坏。

### 核心 Rework Loop
ERI/VRR 球谐 → H2O DIIS → DFT Vxc 符号 → MP2 公式 → CCSD(T) 收敛，共 5 条已被文件名记录的返工链。

### 核心风险点
1. **项目未 git init**，CI/pre-commit 全是模板，永远不会真跑；
2. 根目录 ~85 个 test/debug 脚本未收纳，技术债重；
3. GPU（CuPy）仅留空壳，README 宣称的"GPU 原生加速"未兑现；
4. 600+ Gaussian 测试用例的全量回归未看到结果文件，仅 H2O/STO-3G 级别被验证；
5. 几何优化/频率/TDDFT 仅在小分子示例验证，OPV 共轭大分子未跑；
6. GUI 路径硬编码 `D:\YDW\CalChem\gui`，换机器即失效。

### 最值得复用的流程
> **"数值代码三问 SOP"**：① 参考值硬编码进 `REFERENCES`；② 每个新模块立刻写一个最小分子测试；③ 不对就写 `debug_xxx.py` 逐块打印中间矩阵，对照参考定位到符号/归一化/网格/初猜，而不是靠猜。
> 这套方法比任何具体模块代码都更可迁移，应作为下一个科研/工程项目的默认开发纪律。

---

## 二十六、自检结果

| 检查项 | 结果 |
|---|---|
| 项目阶段是否遗漏 | 已覆盖 S0–S13 |
| 重要步骤是否遗漏 | 13 个 Stage、43 个 Step 已列 |
| 每步是否有 Input/Output | 是 |
| 关键 Artifact | 已列 18 项 |
| Tool/Role/Dependency/Decision/QA/Rework/Exception/Handoff | 均已成表 |
| 隐藏工作 | 10 项已识别 |
| 流程图与工作流表一致 | 是（Mermaid 对应 3 个 QA 决策） |
| 时间线与 mtime 冲突 | 已注明为 mtime 推断 |
| SOP 是否来自真实流程 | 是，全部来自文件名/README |
| 第三方能否理解 | 可以，每步含工具与完成标准 |
| 标注【推测】/【未知】的地方 | git 历史、PySCF 用途、9/7 后状态、导师交接 |

---

*文档生成时间：2026-10-28*
*依据：对 `E:\YDW\Trae_Gaussian\` 的实地勘察（README、pyproject、源码目录、测试文件、打包脚本、GUI 安装说明、CI 配置、文件 mtime）。*
