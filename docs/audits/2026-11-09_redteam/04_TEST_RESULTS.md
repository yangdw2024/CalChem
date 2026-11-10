# 实测、失败反例及复现

2026-11-09；当前工作树；Windows x86_64；Rust 1.99.1；Python 3.13.13。环境由 probes/env.ps1 固定，Rayon=4、BLAS/OMP=1，所有主动指定的输出/缓存在 E:，工具链 D:。部分审计任务并行运行，下面耗时不构成公平性能对比。

## 1. 门禁

| 检查 | 结果 | 原始记录 |
|---|---|---|
| cargo test --workspace --locked | exit 0；91 passed，0 failed，0 ignored；52 个 lib/integration/doc 测试入口汇总 | evidence/cargo-workspace.log、.exit.txt |
| cargo build --release --locked -p calchem-py | exit 0；4 个 PyO3 cfg warning；原/重建扩展 SHA-256 完全一致 | evidence/cargo-release.log、extension-hashes.txt |
| pytest tests/unit | exit 0；96 passed，420.81 s；正常 JIT | evidence/pytest-normal.log、.exit.txt |
| 当前 CI 的 Ruff E,F,W,I,B | exit 1；848 项；F821/F822/F823=0 | evidence/ruff-ci.json、.exit.txt |
| 其余历史 Python tests/、GUI/external/postHF 整套 | 未宣称全量通过 | 本次范围限制 |

初次目录写权限导致的失败 collection 和中断的 JIT-disabled 诊断均另存，**不算作最终测试失败或成功总数**。详见 05_OPERATIONS_AND_LESSONS.md。

## 2. 独立探针

| 探针 | 结果 | 判定 |
|---|---|---|
| Boys n≤16，323 样本，T 0～10^6，对 70 位独立定义 | max relative 1.14e-15；max absolute 2.84e-16 | 实际低阶积分域优良；不宣称数学全域。 |
| Boys n=128，T=35.000001 | −3.90157 vs 正确 3.35851e-18 | 高阶公共函数失败。 |
| erf，11 样本 | max absolute 8.64e-17 | 此采样范围准确。 |
| RHF(STO-3G/6-31Gdp)、B3LYP/PBE(STO-3G)、UHF 阳离子梯度 | h=1e-4 Bohr 时 L2 相对差约 1.8e-8，绝对分量约 1.2～1.4e-8 Eh/Bohr | 本次全部 <1e-5 相对目标；h=1e-3/3e-4/1e-4 显示差分步长收敛。 |
| 平移/旋转/排列 | HF 能量 ~1e-13 Eh；总力 ~1e-13；DFT 旋转梯度差约 3.4e-6 Eh/Bohr | 正确几何依赖；有限网格残余不可抹去。 |
| 非对称水坐标 ±0.01/0.1/0.2 Å 扫描 | 能量、梯度、偶极连续变化 | 未见按默认测试几何返回答案。 |
| Sparse vs dense，65 AO（跨 64 AO 分支） | J、J−0.1K、J−0.5K、非对称 K max差≤1e-15 | 对称压缩/收缩实证成立；两侧相同 Schwarz 设置。 |
| 自定义 α=1e-4 的归一化单中心电子密度积分 | exact 1，Fine grid 0.008011684 | 固定径向截断使一般弥散密度不可用。 |
| 带水溶剂的 RHF 优化总能量/返回梯度一致性 | max FD差 8.5666e-3 Eh/Bohr | 失败，远超目标。 |
| 普通 H₂ 极小值送入 TS | converged=true；reported imaginary frequency 为正 ~3.78e-5 cm^-1 | 认证语义失败。 |
| cpcm/cosmo 溶剂梯度模型参数 | 输出逐位相同 | 参数被忽略，非模型等价验证。 |
| SCRF max_iter=1 | 返回 converged=false；Native 路由可标 not_converged | 未被当成成功，但 Rust 统一错误契约不一致。 |
| 未知 D3 泛函 | core/wrapper 仍给数值 | 未知模型静默采用 PBE 参数。 |
| SCRF gas_energy 对独立气相解 | 差 1.28708e-4 Eh；ΔG 定义差 0.080766 kcal/mol | 参考语义失败。 |

对应证据：boys-values.csv、boys-comparison.jsonl、independent-numeric.jsonl、core-probe.jsonl、solvent-gas-reference.json。

## 3. 新外部黄金标准

- 7 个水 Gaussian 16/6-31G(d,p)/6D10F 单点；2 个 HBr/LANL2DZ 单点；水的 CIS（再用 Full 统一冻结核）、B3LYP full-TDDFT、HF 频率；另有 Gaussian D3(BJ) 与 PBE/STO-3G follow-up。
- ORCA 新实跑 HF/PBE/B3LYP-D3BJ 的 STO-3G；D3 色散项单独提取。
- 原始文件均新建在独立 job 目录，并检查 returncode 和正常终止；没有复用旧 bench.out。
- 能量、轨道、偶极、激发态、频率、强度及误差详表在 03_REDTEAM_REPORT.md。主要正面证据：水 HF ~3.49e-11 Eh，B3LYP ~1.83e-7 Eh；HBr B3LYP/ECP ~4.18e-8 Eh；TDDFT 与频率一致。主要失败：CAM/ωB97X-D、D3、IR 强度。
- external-alignment.jsonl、external-followup.jsonl、external-extra.jsonl；external_raw.zip 是输入/完整输出的保留副本，external-artifacts.json 是原始文件哈希。
- Gaussian 默认 CIS 冻结 O 1s；统一 Full 后第三根从默认 16.7764 对齐到 16.7760 eV。此前默认值差异不计为 CalChem 错误。
- ORCA 默认 B3LYP 与 Gaussian 型 B3LYP 的变体/色散不同；本轮不将该总能量差直接判为核心 B3LYP 错误。

## 4. OPV 流程

实际重跑当前 benchmark 对应路径（不是全体系 PM6/Y6 SCF）：22 原子 IC-2F、32 原子 Donor-BDD1 各三方法单点；60 原子簇及 32/28 单体的 CAM-D3BJ/STO-3G；氯苯/氯仿溶剂；IC-2F CAM/TDA singlet+triplet。全部 20 条 JSON 记录，计算分段耗时合计 **803.30 s**。exit 0。

60 原子簇总能 −3451.369479363580 Eh，结合能 −49.176801 kcal/mol；IC-2F S1=3.691513 eV，最强 S2=4.727425 eV。它们带有本轮已判失败的理论模型，**仅证明复现和计算通路，不能作为标准 CAM/D3/SMD 结果引用**。

158/159/199 原子的完整 PM6/BTP/Y6 做了来源/组成/原文件哈希核验；没有算其完整 SCF。未做大体系几何优化、四点重组能、counterpoise、CT 态完整 golden 验证。OPV 当前输出参数与旧 Gaussian 参考的方法/基组/介质不同，禁止据其差值评测精度。

另实际调用 align_donor_acceptor，重新计算 Donor-BDD1 / IC-2F 的 B3LYP/STO-3G SCF，81.73 s，exit 0。返回 ΔLUMO=−2.664079 eV、ΔHOMO=−1.565048 eV、estimated Voc=5.076333 V、charge_transfer_feasible=false；它们是片段与 Scharber 启发式输出，不是器件预测。见 opv-alignment.json。

## 5. 包构建反例

使用现成 setuptools 84.0.0、无安装/无网络解析，在独立暂存源码构建：

| 暂存输入 | wheel | 扩展 | WHEEL tag |
|---|---|---|---|
| 干净源码，无旧 binary | calchem_qc-0.2.0-py3-none-any.whl | 无 | Root-Is-Purelib:true；py3-none-any |
| 含当前 Windows .pyd | 相同名字 | calchem/_core.pyd | 同样误标 pure/any |

证据 packaging.jsonl；反例 wheel 留在 build_target/runtime/redteam_20261109/package_probe，不是可发布产物。本轮没有 Maturin 安装或多平台发布验收。

## 6. 复现入口

在 E:\YDW\Trae_Gaussian 中执行；首先 dot-source 环境脚本。不要重新执行快照脚本覆盖基线。

```powershell
. .\docs\audits\2026-11-09_redteam\probes\env.ps1
cargo test --workspace --locked
cargo build --release --locked -p calchem-py
python -B -c "import os,tempfile; tempfile.tempdir=os.environ['TEMP']; import pytest; raise SystemExit(pytest.main(['-p','no:cacheprovider','tests/unit','-q']))"
python -B docs/audits/2026-11-09_redteam/probes/independent_numeric.py
& .\docs\audits\2026-11-09_redteam\probes\run_core_probe.ps1
python -B docs/audits/2026-11-09_redteam/probes/boys_compare.py
```

independent_numeric.py 默认加载本轮 runtime 下已留存的 _core.pyd；boys_compare.py 读取本轮编译同一 boys.rs 产生的 CSV。外部脚本路径依赖本机 Gaussian/ORCA 安装，仅用于本轮复现，不是产品可移植实现。重跑时应另建运行编号保存日志，避免覆盖已有外部证据。

## 7. 完整性与恢复演练

起点清单中的 376 个原文件：0 个修改、0 个缺失。快照 ZIP 的 376 个条目逐一哈希匹配且 CRC 通过。另把 Cargo.toml 与 crates/calchem-dft/src/xc.rs 抽取到独立 E: restore_validation 目录演练恢复，哈希一致，未覆盖源文件。外部原始输入/输出归档含 51 个文件，CRC 通过；报告绝对本地链接无失效项。见 integrity-after.json、archive-and-doc-verification.json。
