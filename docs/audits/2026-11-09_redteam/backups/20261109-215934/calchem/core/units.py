"""CalChem 单位定义与换算。

统一约定：
- 能量: Hartree (Eh)
- 坐标: Å (angstrom) 用于跨后端 Result；后端内部可使用 Bohr
- 偶极: Debye
- 梯度: Eh/Bohr
- 频率: cm⁻¹
- 时间: 秒
"""

from __future__ import annotations

# 换算常数
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_KCAL_MOL = HARTREE_TO_KCAL
HARTREE_TO_KJ = 2625.4996394799
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
DEBYE_TO_EA = 0.2081943  # Debye → e·a0
