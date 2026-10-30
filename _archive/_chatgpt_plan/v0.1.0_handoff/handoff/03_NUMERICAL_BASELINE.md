# CalChem — Numerical Baseline (Regression Safety Net)

> **重构时这是你的安全网。任何改动都不能让这些能量偏离参考值超过 1e-8 Eh。**

## 硬阈值

```
|E_calchem - E_reference| < 1.0e-8 Hartree
```

这个阈值来自 `tests/posthf/test_regression.py` 自己的实现（见 `test_h2()` 里 `if diff < 1e-8`）。

## 参考能量表（STO-3G 基组）

| 分子 / 方法 | 参考能量 (Eh) | 来源 | 当前状态 |
|---|---|---|---|
| H2 / RHF (1.4 Bohr) | **-1.11671433** | Szabo & Ostlund | ✅ 实测 -1.116714325063 |
| He / RHF | -2.80778396 | Szabo / Gaussian | ✅ |
| Ne / RHF | -126.60452917 | Gaussian | ✅ |
| H2O / RHF | -74.69531918 | test_regression.py | ⚠️ 见下方冲突 |
| H2O / RHF (README_CalChem) | -74.9629 | README_CalChem | ⚠️ 与上不一致 |
| H2O / MP2 | -74.71482338 | test_regression.py | ✅ |
| H2O / PBE | -75.5322 | README_CalChem | ✅ |
| H2O / B3LYP | -75.5566 | README_CalChem | ✅ |
| H2O / MP4 | -75.0180 | README_CalChem | ✅ |
| H2O / CCSD(T) | -74.9988 | README_CalChem | ✅ |
| O2 / UHF | -146.52482528 | test_regression.py | 🟡 |
| CH3 / ROHF | -40.5539 | README_CalChem | 🟡 |
| H2 / CASSCF(2,2) | -1.1373 | README_CalChem | 🟡 |

## ⚠️ 已知不一致（重构时第一步要统一）

**H2O/RHF/STO-3G 的能量在两个地方对不上：**

- `tests/posthf/test_regression.py` 里写的是 **-74.69531918**
- `README_CalChem.md` 里写的是 **-74.9629**

这两个值差了 ~0.27 Hartree，明显不是同一个几何或基组版本。**重构时需要：**
1. 用 Gaussian 16（`D:\G16W`）现跑一个 H2O/RHF/STO-3G 的 gjf，以 G16 输出为唯一真值；
2. 把 test_regression.py 里的 REFERENCES 更新为该值；
3. 把 README_CalChem.md 改成同一个值；
4. 在 commit message 里说明这是"参考值对齐"。

## 怎么跑回归

```powershell
# 用 conda base（作者日常环境）
D:\YDW\Trae_Gromacs\miniconda3\python.exe tests\posthf\test_regression.py
```

预期输出：每个测试打印 `计算值 / 参考值 / 差异 / ✓ 通过`。

## 重构时的最小验证集

改任何模块，至少跑这三个：

1. `tests/scf/test_quick.py` —— H2/RHF/STO-3G，秒级
2. `tests/posthf/test_regression.py` —— 全量参考表
3. `tests/scf/test_h2o_direct.py` —— H2O Direct SCF

如果这三个绿了，改动基本安全。

## 参考裁判

- **Gaussian 16 Rev A.03** 装在 `D:\G16W`（环境变量 `g16root=D:\G16W`）。
- 作者偏好：mem=12GB, nprocshared=8，**严禁并行跑多个 Gaussian 任务**。
- `tests/conftest.py` 里硬编码了 `D:\G16W\tests\gjf` 作为 600+ 测试用例目录（但目前没真用）。
- **Szabo & Ostlund "Modern Quantum Chemistry"** 是 H2/He 参考值的来源。
