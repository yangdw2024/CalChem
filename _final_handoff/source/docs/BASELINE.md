# CalChem 数值基线

> 2026-10-28 在重构前实测固定。任何重构不得使以下数值偏离阈值。

## 环境

- Python: `D:\YDW\Trae_Gromacs\miniconda3\python.exe` (base)
- numpy/scipy 版本见 `pyproject.toml`
- CPU: AMD Ryzen 7 9700X, 串行执行

## 实测基线（STO-3G 基组）

| 分子 | 方法 | 几何 (Bohr) | 实测能量 (Eh) | 参考值 | 容差 | 状态 |
|---|---|---|---|---|---|---|
| H2 | RHF | H-H = 1.4 (Szabo) | -1.1167143251 | -1.11671433 | 1e-8 | PASS |
| He | RHF | (0,0,0) | -2.8077839575 | -2.80778396 | 1e-6 | PASS |
| Ne | RHF | (0,0,0) | -126.6045291715 | -126.60452917 | 1e-6 | PASS |
| H2O | RHF | Szabo 几何* | -74.6953191771 | -74.69531918 | 1e-8 | PASS |

*Szabo 几何：O(0,0,0), H(0,1.432,0), H(1.112,-0.716,0) Bohr。这是非平衡几何，**不要**与平衡水的 RHF/STO-3G 能量混淆。

## H2O RHF 参考值冲突溯源

- `tests/posthf/test_regression.py` 中 H2O_RHF = -74.69531918（Szabo 几何）→ **正确**，实测吻合 2.9e-9。
- `README_CalChem.md` 旧版写 -74.9629 → **错误**，来源不明（可能是不同几何或旧版本 bug）。已修正为 -74.6953（Szabo 几何）。
- 平衡几何水的 RHF/STO-3G 能量不在本基线中；如需对照，应用 Gaussian 跑同一几何获得。

## Post-HF / DFT 参考值（来自 README_CalChem.md，待回归验证）

| 方法 | 分子 | 能量 (Eh) | 备注 |
|---|---|---|---|
| MP2 | H2O/STO-3G | -74.71482338 | test_regression.py |
| PBE | H2O/STO-3G | -75.5322 | 待回归 |
| B3LYP | H2O/STO-3G | -75.5566 | 待回归 |
| CCSD(T) | H2O/STO-3G | -74.9988 | 待回归 |
| MP4 | H2O/STO-3G | -75.0180 | 待回归 |
| ROHF | CH3/STO-3G | -40.5539 | 待回归 |
| CASSCF(2,2) | H2/STO-3G | -1.1373 | 待回归 |

## 回归命令

```powershell
$py = "D:\YDW\Trae_Gromacs\miniconda3\python.exe"
cd E:\YDW\Trae_Gaussian
& $py tests/posthf/test_regression.py
```
