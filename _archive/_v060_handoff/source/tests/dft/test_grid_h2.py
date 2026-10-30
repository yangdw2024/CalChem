"""调试H2分子的密度计算和网格积分"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.dft.grid import DFTGrid
from calchem.basis.basis_set import BasisSet

mol = Molecule(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
rhf = RHF(mol, basis='sto-3g', verbose=0)
rhf.kernel()
D = rhf.dm
S = rhf._s
print(f"D shape={D.shape}, tr(D*S)={np.sum(D * S):.6f}")

basis_set = BasisSet('sto-3g', mol.to_bohr())
for level in ['coarse', 'medium']:
    grid = DFTGrid(mol.to_bohr(), basis_set, grid_level=level)
    rho = grid.compute_density(D, basis_set, mol.to_bohr())
    nelec = np.sum(rho * grid.weights)
    print(f"{level}: n_points={grid.n_points}, integral(rho)={nelec:.6f} (expected=2)")
