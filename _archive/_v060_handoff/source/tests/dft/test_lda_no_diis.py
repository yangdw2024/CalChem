"""测试LDA禁用DIIS的收敛性"""
from calchem.mol.molecule import Molecule
from calchem.scf.dft import RKS

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)

try:
    rks = RKS(mol, basis='sto-3g', xc='lda', grid_level='coarse', verbose=2, diis=False)
    e = rks.kernel()
    print(f"\nLDA 收敛能量: {e:.10f} Eh")
except Exception as ex:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {ex}")
