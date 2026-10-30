"""调试DFT网格维度问题"""

import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.dft import RKS

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)

try:
    rks = RKS(mol, basis='sto-3g', xc='tpss', grid_level='coarse', verbose=2)
    print("构建中...")
    rks.build()
    print("构建完成！")
    
    print(f"基函数数目: {rks._dft_impl._s.shape[0]}")
    print(f"网格点数: {rks._dft_impl._grid.n_points}")
    
    dm = rks._dft_impl.get_init_dm()
    print(f"DM shape: {dm.shape}")
    
    print("计算密度...")
    rho = rks._dft_impl._grid.compute_density(dm, rks._dft_impl.basis_set, rks._dft_impl.mol)
    print(f"rho shape: {rho.shape}")
    
    print("计算密度梯度...")
    from calchem.dft.fock import _compute_density_gradient
    grad_rho = _compute_density_gradient(dm, rks._dft_impl.basis_set, rks._dft_impl.mol, rks._dft_impl._grid)
    print(f"grad_rho shape: {grad_rho.shape}")
    
except Exception as ex:
    import traceback
    traceback.print_exc()
