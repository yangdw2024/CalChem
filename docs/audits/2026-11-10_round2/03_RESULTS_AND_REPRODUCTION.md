# 最终实测与复现

最终验证对象：2026-11-10 第二轮修复后源码，以及已经复制到 `calchem/_core.pyd` 的独立构建扩展。

扩展 SHA256：

`4e68564103ad866fc14ad71fe057d6860d9b827913e07ef38f5816b70c1c9bbd`

Windows x86_64，Python 3.13.13，Rust 1.99.1；所有运行目录、临时目录、缓存和输出在 E 盘，Cargo/Rust/Python 工具链在 D 盘。本轮没有安装或下载软件依赖。

## 最终验收记录

| 检查 | 结果 | 原始证据 |
|---|---|---|
| Cargo workspace + doctests | **102 passed，0 failed，0 ignored**；56 个 suite 条目含零测试条目 | `evidence/cargo-delivery.log`、`.exit.txt` |
| Python `tests/unit` | **104 passed，484.42 秒，exit 0** | `evidence/pytest-delivery.log`、`.exit.txt` |
| 最终扩展独立数值/异常/GIL | **exit 0** | `evidence/acceptance-delivery.jsonl` |
| ECP 导数、H2 非 TS、部分 IRC 状态 | **exit 0** | `evidence/ecp-ts-delivery.jsonl` |
| Boys 80 位参考 | **exit 0**；2244 点，重点 T35 域最大相对误差 6.58e-16 | `boys-certified.csv`、`boys-certified.jsonl` |
| D3 两体能量/解析梯度 | 8 个新鲜 G16 对照；最大能量差 <5e-11 Eh；最大梯度差 <5e-12 Eh/Bohr | `acceptance-delivery.jsonl`、`external-round2.jsonl`、`external_raw_round2.zip` |
| 独立 Rust 原生探针 | exit 0；65 AO 稀疏/致密差 <1e-15；不同步长溶剂总梯度差约 1.0e-7 Eh/Bohr | `core-closed-loop.log` |
| wheel、sdist 再构建与导入 | 两个 wheel 均 cp310-abi3-win_amd64，非 purelib；He、D3、LRC 拒绝、IRC 契约检查通过 | `package-delivery-contracts.jsonl` |
| Python AST | 语法检查通过 | `ast.log` |
| Ruff F821/F822/F823 | exit 0 | `ruff-critical.log` |
| 回滚恢复演练 | 378 个原始文件恢复至新候选目录，全部 SHA256 一致；无原文件缺失 | `final_manifest.json` |

**测试计数不是代码覆盖率。** 没有运行全风格级 Ruff 作为验收；没有宣称整个仓库所有历史/GUI/外部软件测试都已通过。

## 命令与参数

使用 E 盘环境：

```powershell
Set-Location E:\YDW\Trae_Gaussian
. docs\audits\2026-11-10_round2\probes\env.ps1
```

最终 Cargo 命令使用以下两个环境变量，让 test/dev 保留测试语义和 debug assertions，同时优化数值循环；这一区别明确记录，不冒称未经优化的 debug 耗时：

```powershell
$env:CARGO_PROFILE_DEV_OPT_LEVEL='2'
$env:CARGO_PROFILE_TEST_OPT_LEVEL='2'
cargo test --workspace --locked --target-dir "$auditRuntime\target"
```

曾执行原始无 profile 覆盖的 `cargo test --workspace --locked`；该诊断在并行构建/源文件变化下最后发生 doctest rlib 不一致，**不计为通过**。最终上述独立 target、冻结源码的完整检查才是验收证据。

```powershell
cargo build --release --locked -p calchem-py --target-dir "$auditRuntime\target"
& $env:PYO3_PYTHON -B -c 'import os,tempfile,pytest; tempfile.tempdir=os.environ["TEMP"]; raise SystemExit(pytest.main(["tests/unit","-q","-p","no:cacheprovider"]))'
$env:CALCHEM_AUDIT_EXTENSION="$auditRuntime\delivery_core\_core.pyd"
& $env:PYO3_PYTHON -u -B "$auditRoot\probes\acceptance.py"
& $env:PYO3_PYTHON -u -B "$auditRoot\probes\ecp_and_ts.py"
```

如果重编译了新的扩展，应选择新文件路径并核对 SHA256，不能只改源码后继续加载旧进程中的模块。`acceptance.py` 会输出实际二进制路径和哈希。频率测试会主动收紧底层 SCF 响应容差，所以耗时高于普通单点。

Boys：

```powershell
rustc --edition=2021 -O "$auditRoot\probes\boys_certified.rs" -o "$auditRuntime\boys_certified.exe"
& "$auditRuntime\boys_certified.exe" > "$auditRoot\evidence\boys-certified.csv"
& $env:PYO3_PYTHON -B "$auditRoot\probes\boys_certified_compare.py"
```

最终独立 Rust 驱动在 `build_target/runtime/round2_20261110/rust_probe/`，使用 Cargo 解析同一依赖图。它采用原子坐标、显式基组、独立的有限差分步长，未读取预设分子能量：

```powershell
cargo run --offline --release --manifest-path "$auditRuntime\rust_probe\Cargo.toml" --target-dir "$auditRuntime\probe_target"
```

Gaussian：`probes/external_round2.py` 真实生成输入、运行本机 `D:\G16W\g16.exe`、检查 normal termination，并保留原始输出。用相同几何比较 IR，用固定输入几何比较 D3，不混用优化后和优化前能量。Gaussian 路由/几何/参数在 `external_raw_round2.zip` 内。乙烯禁戒/极弱 IR 模式以绝对误差比较，不对四位小数舍入值做不合理的严格比值要求。

## 独立数值要点

- 水显著 IR：G16 `[7.2374,44.2880,29.9732]` km/mol；最终误差见 JSON。
- 乙烯 12 模式：显著强度与 G16 对齐，禁戒模式接近零；最终最大绝对误差约 4.12e-5 km/mol。
- OH：UHF/UKS 梯度对不同中心差分步长均低于 1e-6 Eh/Bohr 门槛，合力约 1e-15。自旋污染为实测诊断，不宣称自旋投影。
- HBr/LANL2DZ、HI/LANL2DZdp：h=1e-4 时梯度差约 1.16e-9、9.82e-10 Eh/Bohr；合力约 1e-15。
- H2 极小值被 TS 和 IRC 起点认证拒绝；真实 HCN 一阶鞍点、IRC 点的独立 E/g 重算与状态传播通过。
- SCRF 独立气相参考与单独 RHF 值一致，ΔG 恒等式误差约 2.8e-13 kcal/mol。数值梯度对不同 h=3e-4 的总能量差分最大差约 1.0e-7 Eh/Bohr，不将“同一 h 重算误差为零”当作独立证据。
- GIL 心跳有实际计数，两个并发平移计算的能量差约 4.3e-14 Eh；没有据此声称完成所有线程调度/FFI/内存压力认证。

## 打包验证

最终产物在 `artifacts/`。构建源目录未带有效扩展，反而故意放入无效旧 `_core.pyd`。新 setup.py 仍调用 Cargo，生成可导入的二进制 wheel。sdist 中包含 Cargo workspace、Rust 源码和标准 D3 表，不包含 pyd；从该 sdist 再生成 wheel 并重复 He、D3、LRC/IRC 契约检查。

```powershell
$env:CALCHEM_PACKAGE_STAGE='delivery_package_stage'
$env:CALCHEM_PACKAGE_ACCEPTANCE_DIR='package_acceptance_delivery'
& $env:PYO3_PYTHON -B "$auditRoot\probes\package_acceptance.py"
```

构建依赖使用已安装 setuptools 84.0.0，没有 build isolation 下载或 pip 安装。仅验证 Windows/Python 3.13 的实际导入；abi3-py310 是构建契约，其他 Python 和 Linux/macOS 仍需逐平台 CI 安装测试。
