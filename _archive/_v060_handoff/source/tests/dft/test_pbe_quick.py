"""快速测试PBE收敛性"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.dft import RKS

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)

try:
    rks = RKS(mol, basis='sto-3g', xc='pbe', grid_level='coarse', verbose=2)
    e = rks.kernel()
    print(f"\nPBE 收敛能量: {e:.10f} Eh")
except Exception as ex:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {ex}")
