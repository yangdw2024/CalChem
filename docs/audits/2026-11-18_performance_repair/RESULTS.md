# CalChem 原生 RKS 性能修复与验收

2026-11-18，Windows，RTX 5060 Ti。先制定 [解决方案](SOLUTION.md)，再修改源码、编译 release CLI 和 Python 扩展，并用两个真实 OPV 输入运行到收敛。以下用进程墙钟时间评价端到端性能，阶段耗时取自 CLI 内部计时。

## 最终构建的性能验收

| 体系及设置 | CalChem 旧记录 | CalChem 修复后，16 线程 + GPU | 进程峰值工作集 | SCF 步数 | GPU 核调用 / 回退 |
|---|---:|---:|---:|---:|---:|
| PTQ10，72 原子，212 AO，B3LYP/STO-3G | 301.42 s | 18.13 s | 1.083 GiB | 27 | 58 / 0 |
| BTP-4F，199 原子，639 AO，PBE/STO-3G | 约 112 s 时人为中止 | 68.96 s | 3.473 GiB | 54 | 112 / 0 |

PTQ10 较求助文档旧记录加速 16.62 倍，旧版与新版总能量相差 2.18 × 10⁻¹⁰ Eh。BTP-4F 缺少旧版收敛耗时。新版两例的 RI-J 度量截断数都是 0。每个表格数据点均为一次完整运行，工作集按 Windows 进程峰值计数器采样。GPU 峰值显存和系统总提交量未测。

| 阶段 | PTQ10 旧记录 | PTQ10 修复后 | BTP-4F 修复后 |
|---|---:|---:|---:|
| 单电子积分 | 2.107 s | 0.134 s | 2.566 s |
| Fock 构建器初始化 | 2.959 s | 2.222 s | 22.290 s |
| 主 SCF 双电子部分 | 211.724 s | 12.107 s | 4.414 s |
| 主 SCF 交换相关格点积分 | 4.080 s | 2.971 s | 31.085 s |
| 主 SCF 对角化 | 1.849 s | 0.125 s | 5.069 s |
| 最终自洽检查 | 78.242 s | 0.280 s | 1.477 s |

## Gaussian 对照与精度边界

逐行对照原子和三个坐标分量，两对输入的元素顺序一致，坐标最大差均为 0 Å。两份 Gaussian 作业均用 4 线程、2 GB、STO-3G 及 `Int=Coarse`。原记录进程墙钟分别为 16.81 s 和 107.10 s，Gaussian 输出内部 elapsed 分别为 16.5 s 和 106.8 s。Gaussian 的 SCF 分别为 14、20 步。CalChem 使用各自泛函、自动 RI-J、自己的 Coarse 网格，能量和密度阈值为 10⁻¹⁰ Eh、10⁻⁸。

| 体系 | Gaussian 16，4 线程 | CalChem，4 线程 + GPU | CalChem，16 线程 + GPU |
|---|---:|---:|---:|
| PTQ10 | 16.81 s | 35.57 s | 18.13 s |
| BTP-4F | 107.10 s | 141.02 s | 68.96 s |

4 线程下，CalChem 的耗时分别为原 Gaussian 作业的 2.12 倍和 1.32 倍。16 线程下，PTQ10 耗时比原 Gaussian 作业长 7.9%，BTP-4F 的耗时少 35.6%。4 线程运行分别完成 27 步和 56 步，两例均无 GPU 回退。两种线程数下同一个体系的总能量最多相差 6 × 10⁻¹¹ Eh。4 线程 BTP-4F 先前尝试缺少完成产物，表中来自独立重跑 `btp199_release_4_retry`。

| 体系 | CalChem 总能量 (Eh) | Gaussian 总能量 (Eh) | CalChem − Gaussian (Eh) |
|---|---:|---:|---:|
| PTQ10 | −1849.309081084153 | −1849.27237979 | −0.03670129 |
| BTP-4F | −6230.647210389630 | −6230.53246627 | −0.11474412 |

以上能量差尚未拆分到拟合、格点及其他实现差异，速度结果覆盖这两项作业。大基组、解析力以及 726 原子作业的性能需另测。要判断 OPV 相对能量能否用于科研定量，可在固定几何和本机相同网格上先比较精确 J 与 RI-J，再比较不同格点下的精确 J，并保留每个误差分量。

## 源码与工程取舍

| 位置（从仓库根计） | 修改与理由 |
|---|---|
| `crates/calchem-int/src/direct_fock.rs:409`、`:707`，`eri.rs:548` | 壳对按 Schwarz 上界排序，以保守密度上界提前结束遍历，再做逐 quartet 筛选。每次构建预计算原始高斯壳对，跨步缓存普通精确交换积分。默认缓存数据预算 768 MiB，可用 `CALCHEM_MAX_K_CACHE_BYTES=0` 关闭，未命中时直接计算。保留长程积分的独立算符路径。|
| `crates/calchem-int/src/density_fitting.rs:783`、`:975`、`:1533` | 稠密张量较大时，只保存通过原筛选的独立 AO 对。辅助函数的每行连续存储，通过正反两个 FP64 收缩构建 J，密度非对角两向求和，结果对称回填。单测用非对称输入检查权重。|
| `crates/calchem-tensor/src/lib.rs:474` | 对归一化度量用行和上界确定谱截断上界，平移矩阵的 Cholesky 成功后再分解原矩阵。无法确认满秩时走带谱截断的原路径。需要条件数的显式诊断入口保留全谱计算。|
| `crates/calchem-scf/src/rks.rs:344` | 纯物理 Fock 对角化后，密度变化和用新密度重建的交换子同时达标时结束；对 DIIS 外推步骤继续执行规范化和复核。|
| `crates/calchem-tensor/src/lib.rs:88`、`:158` | 中大矩阵乘和对称对角化使用 faer，继续验证本征残差。|
| `crates/calchem-cuda/src/lib.rs:112`、`crates/calchem-cuda/src/cuda_kernels.cu:447`、`:543`、`crates/calchem-cuda/tests/ffi_contract.rs:26` | 随后发现旧设备缓存用地址、尺寸和首尾值识别 B。在同一地址只改张量中部就能误命中。不可变压缩张量现在拥有独立身份，借入的普通切片每次上传，尺寸相同可复用显存分配。Windows 驱动和 CUDA Runtime 两条实现同步修改。本机实际运行的是 Windows 路径。新增回归交错测试原地修改、两份常驻张量和连续正反收缩。|
| `crates/calchem-int/src/one_electron.rs:133`、`boys.rs:116`、`crates/calchem-dft/src/vxc.rs:132` | 核吸引积分按独立壳对并行。Boys 函数常用范围采用近邻节点九阶展开，边界外使用原算法。RKS 交换相关矩阵计算一个三角再对称回填。|

199 原子的三中心张量从每行 408,321 个 AO 对减少到 62,058 个，6,355 行共占 3,155,028,720 字节，压缩 6.58 倍。原 2.5 GiB 默认预算仍装不下，首轮 199 原子试运行在确认其走即时计算路径后主动中止。修复版默认 4 GiB，并受已配置的总预算扣除度量工作空间后余额限制。超预算仍走即时计算路径。PTQ10 对应的张量占 257,286,432 字节。加速保留了原有精确交换算法，无需引入 RI-K 或半数值交换的新增近似。

## 复现与恢复

发布版 CLI 和 Python 扩展均构建成功。已安装 `calchem/_core.pyd` 与 release DLL 的 SHA256 完全相同。Rust 工作区 145 项通过，0 项失败。另将默认忽略的 60 原子性能用例单独执行，23 步收敛至 −3427.9592613174 Eh，50 次核调用，0 回退，测试通过。Python 120 项全部通过，包括 `test_p6_opv_molecular_validation.py` 的 4 项。见 [Rust 日志](rust5.log)、[60 原子日志](opt60.log) 和 [Python 日志](python5.log)。

从项目根目录在 PowerShell 中执行，测试数据位于 `E:/YDW/ChemDatas/Gjf`。用新的输出标签避免覆盖已存计时。构建、测试各步的退出码均应检查。

```powershell
. ./tools/dev/dev_env.ps1
& $env:CALCHEM_CARGO build --release --offline --locked -p calchem-scf --bin calchem_cli
if ($LASTEXITCODE) { throw 'CLI build failed' }
$bench = 'docs/audits/2026-11-18_performance_repair/bench.py'
& $env:CALCHEM_PYTHON -B $bench verify_ptq ydw-PTQ10 b3lyp
if ($LASTEXITCODE) { throw 'PTQ10 failed' }
& $env:CALCHEM_PYTHON -B $bench verify_btp ydw-BTP-4F pbe
if ($LASTEXITCODE) { throw 'BTP-4F failed' }
powershell -NoProfile -File tools/dev/build_ext.ps1
if ($LASTEXITCODE) { throw 'Extension build failed' }
powershell -NoProfile -File scripts/cargo_test.ps1
if ($LASTEXITCODE) { throw 'Rust tests failed' }
powershell -NoProfile -File scripts/run_pytest.ps1 tests/unit
if ($LASTEXITCODE) { throw 'Python tests failed' }
```

同目录的 `ptq72_v4`、`btp199_v4` 日志与 JSON 对应首次 16 线程验收，`*_release_*` 对应 GPU 缓存修正后的发布构建。JSON 含原输入路径、SHA256、可执行文件 SHA256、退出码和计时。添加 `--threads 4` 可复现 4 线程组。四个完成作业、Gaussian 参考文件、回归日志和源文件的 SHA256 收录于 [验收清单](manifest.json)。旧源码、测试及旧 CLI 在 `backup` 中保留原相对路径，对照 `original.jsonl` 校验。旧扩展为 `build_target/runtime/pyd-backup/_core-20261118-101732.pyd`。恢复时将这些文件复制到独立的工程候选副本，再从候选目录重编译并比较结果。

**操作记录。** 中间编译发现本机 faer 的切片构造函数还需要两个矩阵维度泛型，修正后编译成功。CUDA 缓存修复后，Windows 在旧 CLI 仍运行时阻止替换 EXE，保留该作业，待退出后重新链接。最终对照安排在回归退出后顺序执行，避免构建和测试争用性能测量资源。
