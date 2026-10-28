"""快速测试RHF收敛性"""
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)

try:
    rhf = RHF(mol, basis='sto-3g', verbose=2)
    e = rhf.kernel()
    print(f"\nRHF 收敛能量: {e:.10f} Eh")
except Exception as ex:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {ex}")
