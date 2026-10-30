#!/usr/bin/env python
"""诊断 PBE vxc_gamma 计算。"""
import sys
sys.path.insert(0, r'd:\YDW\Trae_Gaussian')

import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.dft import RKS
from calchem.dft.fock import _compute_density_gradient

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords, units='angstrom')

rks = RKS(mol, basis='sto-3g', xc='pbe', grid_level='coarse', verbose=0)
rks.build()

from calchem.scf.hf import _symmetric_orthogonalize, _eigh_safe, _matmul
x = _symmetric_orthogonalize(rks._s)
f_ortho = x.T @ rks._hcore @ x
eigval, eigvec_ortho = _eigh_safe(f_ortho)
mo_coeff = x @ eigvec_ortho
dm = _matmul(mo_coeff[:, :5] * 2.0, mo_coeff[:, :5].T)

rho = rks._dft_impl._grid.compute_density(dm, rks._dft_impl.basis_set, rks._dft_impl.mol)
grad_rho = _compute_density_gradient(dm, rks._dft_impl.basis_set, rks._dft_impl.mol, rks._dft_impl._grid)

from calchem.dft.functionals import PBE
pbe = PBE()
exc, vxc_rho, vxc_gamma, vxc_tau = pbe.compute_full(rho, grad_rho)

print(f"vxc_gamma range: [{vxc_gamma.min():.8f}, {vxc_gamma.max():.8f}]")
print(f"vxc_gamma mean: {vxc_gamma.mean():.8f}")
print(f"vxc_rho range: [{vxc_rho.min():.8f}, {vxc_rho.max():.8f}]")

# 检查gamma项对Vxc对角元的贡献量级
weights = rks._dft_impl._grid.weights
all_shells = rks._dft_impl.basis_set.get_all_shells(mol)
shell_range = []
idx = 0
for s in all_shells:
    nf = s.n_functions
    shell_range.append((idx, idx + nf))
    idx += nf

# 近似估算: 仅取第一个s型基函数
phi0 = rks._dft_impl._grid._eval_gto(0, 0, 0, all_shells[0].exponents[0], all_shells[0].origin)
dphi0 = rks._dft_impl._grid._eval_gto_grad(0, 0, 0, all_shells[0].exponents[0], all_shells[0].origin)[1]

dot = np.sum(grad_rho * (2.0 * phi0[:, None] * dphi0), axis=1)
contrib_rho = np.sum(vxc_rho * phi0**2 * weights)
contrib_gamma = np.sum(2.0 * vxc_gamma * dot * weights)
print(f"\n对角元近似贡献:")
print(f"rho项: {contrib_rho:.8f}")
print(f"gamma项: {contrib_gamma:.8f}")
print(f"gamma/rho 比值: {abs(contrib_gamma/contrib_rho)*100:.2f}%")
