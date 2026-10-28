"""测试单个基函数的归一化积分"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.dft.grid import DFTGrid
from calchem.basis.basis_set import BasisSet

mol = Molecule(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
basis_set = BasisSet('sto-3g')

for level in ['coarse', 'medium']:
    grid = DFTGrid(mol.to_bohr(), basis_set, grid_level=level)
    all_shells = basis_set.get_all_shells(mol.to_bohr())

    for ishell, shell in enumerate(all_shells):
        origin = shell.origin
        phi = np.zeros((grid.n_points, shell.n_functions))
        for alpha, coef, norm_const in zip(shell.exponents, shell._contract_coeffs, shell.norm_constants):
            for ifunc, comp in enumerate(shell.cartesian_components()):
                l, m, n = comp
                phi_prim = grid._eval_gto(l, m, n, alpha, origin)
                phi[:, ifunc] += coef * norm_const * phi_prim
        for ifunc in range(phi.shape[1]):
            norm = np.sum(phi[:, ifunc] ** 2 * grid.weights)
            print(f"level={level} shell={ishell} func={ifunc} integral(phi^2)={norm:.6f}")
