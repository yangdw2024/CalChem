# 状态与工作流重建（审计起点）

日期：2026-11-09，Asia/Shanghai。仓库 E:\YDW\Trae_Gaussian；Git HEAD c9fd62e（2026-10-30，master）。本次评估当前工作树。

## 时间线

- 2026-10-28～30：Python 工作站、外部适配器、Rust Stage 0～2、OS 积分和 RHF。
- 2026-11-03：旧审计记录为 Python 19 通过/9 失败，Rust 测试有编译阻断，梯度/优化主要未完成。
- 2026-11-05～06：关键文件时间戳及备份显示大量修复与新模块加入；ARCHITECTURE_PRINCIPLES.md 要求原生计算、无假收敛、修改前备份。
- 2026-11-09 起点：工作树有 60 个已跟踪文件修改（diff 约 +19172/-1655 行），另有多套未跟踪功能和测试。13 个 crate 中新增 dft/tddft/vib/solv 尚未提交。pyproject 与发布 workflow 有近期变动。不存在可据以认定正式发布的已验证发布产物。

## 工作流

用户结构/OPV 内置或 ChemDatas 几何 → MoleculeSpec/CalculationRequest → CalculationService/任务队列 → NativeCalchem 或显式外部 Gaussian/ORCA/REST/PySCF 适配器 → calchem._core（PyO3） → Rust model/basis/int/tensor/scf → grad/opt/prop/dft/tddft/vib/solv → Python 结构化结果及 OPV 分析。

保留的旧 Python 数值实现亦可从 calchem.RHF/RKS 等低层接口访问；不能把整个 Python 包理解为纯胶水。原生 Rust 与外部引擎需要分别确认 actual_method、后端和参数，不能将外部结果算作 Rust 核心验证。

## 当前工具与隔离

- cargo/rustc：D:\cargo\bin；CARGO_HOME=D:\cargo，RUSTUP_HOME=D:\rustup。
- Python：D:\YDW\Trae_Gromacs\miniconda3\python.exe，3.13.13；SciPy/pytest/mpmath 可用，当前解释器未找到 PySCF/maturin。
- 默认 TEMP 在 C 盘，已通过 probes/env.ps1 将本次子进程 TEMP/TMP、数据、缓存改到 E:\YDW\Trae_Gaussian\build_target\runtime\redteam_20261109\。
- 编译使用项目已有 E:\YDW\Trae_Gaussian\build_target；不安装软件，不下载依赖，Cargo 离线且锁定。
- 审计快照含 376 个文件，source_snapshot.zip 为约 1.96 MB；evidence/source_before.json 固定源文件及现有扩展哈希。
- 初始 Python 对本审计 docs 目录直接 mkdir/write_text 报 PermissionError，PowerShell 可正常写入。改用项目既有 build_target/runtime 下的独立目录后 Python 写入和正常 Numba JIT 均恢复；没有改 ACL、关闭防护或修改业务源码。证据由 PowerShell 捕获 stdout。

## 需特别验证的能力边界

- 13 个 crate 包含一个 PyO3 绑定 crate；并非 13 个都不依赖 Python。
- ERI 有真实 OS VRR/HRR、Schwarz 和对称压缩，但 dense、小/大体系切换、梯度路径峰值内存不同。
- 全程同名方法有效性必须以算符及数据来源为准；自洽差分测试不证明标准泛函/色散模型的身份。
- OPV 22/32/60 原子结构、Gaussian 历史能量及基组/泛函设置须从原始文件复核；不把 STO-3G 与 6-31G(d,p) 的能量接近程度当作同口径基准。
