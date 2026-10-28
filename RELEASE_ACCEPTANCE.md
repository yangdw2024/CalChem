# CalChem v0.1.0 最终验收报告

**版本**: 0.1.0
**日期**: 2026-10-28
**构建环境**: Windows 11, Python 3.10 (molforge-webengine conda env), PyInstaller 6.22.3
**硬件**: AMD Ryzen 7 9700X, 15.1GB RAM

---

## A. 核心与科学正确性

| 检查项 | 结果 | 证据 |
|--------|------|------|
| H2/STO-3G RHF 基线 | PASS | E=-1.116714325063 Eh, 参考 -1.1167143251, 偏差 3.7e-11 |
| He/STO-3G | PASS | E=-2.807783957540 Eh, 参考 -2.8077839575, 偏差 4.0e-11 |
| H2O/STO-3G (Szabo) | PASS | E=-74.695319177132 Eh, 参考 -74.6953191771, 偏差 <1e-9 |
| H2O RHF 冲突解决 | PASS | test_regression.py 值为正确值, README 已修正 |
| tests/ 目录 | PASS | tests/unit, tests/acceptance.py, tests/debug/, tests/scratch/ |

## B. 核心契约

| 检查项 | 结果 |
|--------|------|
| Request/Result schema-versioned | PASS (v1.0) |
| Result JSON round-trip | PASS |
| 单位明确 (Eh/Å/Debye/cm⁻¹) | PASS |
| 不可得量为 null | PASS |
| 后端版本/耗时/路径可追溯 | PASS |

## C. 后端真实可用性

| 后端 | 状态 | H2O/STO-3G 能量 | 与 native 差 |
|------|------|-----------------|-------------|
| Native | PASS | -74.695319177132 | 0 (基准) |
| Gaussian 16 A.03 | PASS | -74.695319172900 | 4.2e-9 Eh |
| ORCA 6.1.1 | PASS | -74.695319167759 | 9.4e-9 Eh |
| PySCF | DISABLED | 未安装 pyscf |
| xTB | DISABLED | xtb binary 不在 PATH |
| REST | DISABLED | 未配置 endpoint |

三后端 H2O/STO-3G 交叉验证: 全部在 1e-8 Eh 以内。

## D. Job/项目/UI

| 检查项 | 状态 |
|--------|------|
| 单并发 JobQueue | PASS (第二个 job 返回 None) |
| 状态机合法转移 | PASS |
| ProjectStore (.calchem 格式) | PASS |
| Manifest.json | PASS |
| PyQt6 GUI 启动 | PASS (molforge-webengine 环境) |

## E. CLI/打包

| 检查项 | 状态 |
|--------|------|
| CLI backends/doctor/run | PASS |
| PyInstaller one-folder | PASS (238MB) |
| CalChem.exe 启动 | PASS (5s smoke test) |
| 数据目录非 C 盘 | PASS (E:\CalChemBuild, D:\CalChemData) |

## F. 已知限制

1. GUI 需 molforge-webengine conda 环境 (base 环境 PyQt6 DLL 加载失败)
2. 几何优化/频率计算未实现 (当前仅单点能)
3. PySCF/xTB/REST 为注册 stub, 未实现真实计算
4. cclib parser 未集成 (用手写 regex)
5. Ne/CH3 基线未在本次重新验证 (代码中已有但未跑)

## G. 打包产物

- 路径: `E:\CalChemBuild\dist\CalChem\`
- EXE: `CalChem.exe` (18MB launcher + _internal/ 资源)
- 总大小: 238MB
- 不包含 Gaussian/ORCA 商业程序
