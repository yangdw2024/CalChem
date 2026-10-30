#!/usr/bin/env python
"""测试 PBE GGA 收敛性。"""
import sys
sys.path.insert(0, r'd:\YDW\Trae_Gaussian')

from calchem.mol.molecule import Molecule
from calchem.scf.dft import RKS

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords, units='angstrom')

print("=" * 60)
print("  H2O / sto-3g PBE 收敛测试")
print("=" * 60)

rks = RKS(mol, basis='sto-3g', xc='pbe', grid_level='coarse', verbose=2, max_cycles=30)
try:
    e = rks.kernel()
    print(f"\nPBE 能量: {e:.10f} Eh")
    print(f"收敛状态: {'是' if rks.converged else '否'}")
except Exception as exc:
    print(f"\n错误: {exc}")
    import traceback
    traceback.print_exc()
