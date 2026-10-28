# CalChem 版本验收报告

**版本**: 0.1.0
**日期**: 2026-10-28
**构建环境**: Windows 11, Python 3.10, AMD Ryzen 7 9700X, 15.1GB RAM

## 数值回归

| 测试 | 参考值 (Eh) | CalChem 实测 | 偏差 | 状态 |
|------|------------|-------------|------|------|
| H2/RHF/STO-3G (1.4 Bohr) | -1.1167143251 | -1.116714325063 | 3.7e-11 | PASS |
| He/RHF/STO-3G | -2.8077839575 | -2.807783957540 | 4.0e-11 | PASS |
| H2O/RHF/STO-3G (Szabo) | -74.6953191771 | -74.6953191771 | <1e-9 | PASS |

## 后端状态

| 后端 | 状态 | 证据 |
|------|------|------|
| Native | PASS | H2 E=-1.1167143251, SCF 收敛 |
| Gaussian 16 A.03 | PASS | H2 E=-1.1167143249, normal termination, 与 native 差 1.3e-10 Eh |
| ORCA 6.1.1 | BLOCKED | INPUT ERROR rc=4; 代码已写，需修正基组名映射 |
| PySCF | REGISTERED/DISABLED | 未安装 pyscf，注册为 unavailable |
| xTB | REGISTERED/DISABLED | xtb binary 未在 PATH |
| REST | REGISTERED/DISABLED | 未配置 CALCHEM_REST_ENDPOINT |

## GUI

- 代码完成（calchem/gui/）
- 在 molforge-webengine conda 环境中验证启动（PyQt6 DLL 在 base 环境有问题）
- 3D 分子视图、原子表、计算表单、日志、结果面板均已实现

## CLI/API

- `python -m calchem backends` - 列出后端
- `python -m calchem doctor` - 环境诊断
- `python -m calchem run --molecule H2 --backend native/gaussian` - 单点能
- Python API: `TaskService().run(CalculationRequest(...))`

## Job 系统

- 单并发 JobQueue 验证通过
- 状态机: pending → queued → running → succeeded/failed/not_converged/cancelled
- 结果自动保存为 result.json

## 已知限制

1. ORCA 后端基组名映射需修正（STO-3G 在 ORCA 中名称不同）
2. GUI 需在 molforge-webengine 环境运行（base conda 环境 PyQt6 DLL 加载失败）
3. 几何优化、频率计算尚未实现（当前仅单点能）
4. PySCF/xTB/REST 为注册 stub，未实现真实计算
5. PyInstaller 打包 spec 已写但未实际执行打包

## 阻塞项

- ORCA: 需手动测试 ORCA 基组名（STO-3G vs STO-3G-N）
- PyInstaller: 需在干净 venv 中构建（当前在 conda base 中）
