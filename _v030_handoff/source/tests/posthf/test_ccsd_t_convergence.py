#!/usr/bin/env python
"""快速测试 CCSD(T) 在 H2O/sto-3g 上的收敛性。"""
import numpy as np
import sys
sys.path.insert(0, r'd:\YDW\Trae_Gaussian')

from calchem.mol.molecule import Molecule
from calchem.post_hf.ccsd_t import CCSDT

# H2O / sto-3g (与 test_audit.py 一致，使用 angstrom)
coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords, units='angstrom')

print("=" * 60)
print("  H2O / sto-3g CCSD(T) 收敛测试")
print("=" * 60)

cc = CCSDT(mol, basis="sto-3g", max_cycles=100, conv_tol=1e-7, verbose=2)
try:
    e_tot = cc.kernel()
    print(f"\n最终结果:")
    print(f"  CCSD 相关能: {cc.e_corr_ccsd:.10f} Eh")
    print(f"  (T) 修正:    {cc.e_corr_pt:.10f} Eh")
    print(f"  CCSD(T)总能量: {e_tot:.10f} Eh")
    print(f"  迭代次数: {cc.n_cycles}")
    print(f"  收敛状态: {'是' if cc.converged else '否'}")
except Exception as exc:
    print(f"\n错误: {exc}")
    import traceback
    traceback.print_exc()
