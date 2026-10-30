"""详细检查Vxc矩阵计算"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import guess_hcore, nuclear_repulsion
from calchem.dft.grid import DFTGrid
from calchem.dft.fock import compute_vxc, _compute_vxc_block
from calchem.dft.functionals import LDA
from calchem.basis.basis_set import BasisSet

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
basis_set = BasisSet('sto-3g', mol)

from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
S = compute_overlap(basis_set, mol.to_bohr())
H_core = compute_kinetic(basis_set, mol.to_bohr()) + compute_nuclear(basis_set, mol.to_bohr())

D = guess_hcore(H_core, S, mol.nelectron, restricted=True)

func = LDA()
grid = DFTGrid(mol.to_bohr(), basis_set, 'coarse')

rho = grid.compute_density(D, basis_set, mol.to_bohr())
exc, vxc_rho = func.compute(rho)

vxc = compute_vxc(D, basis_set, mol.to_bohr(), func, grid)

print(f"rho min/max: {rho.min():.2e} / {rho.max():.2e}")
print(f"exc min/max: {exc.min():.6f} / {exc.max():.6f}")
print(f"vxc_rho min/max: {vxc_rho.min():.6f} / {vxc_rho.max():.6f}")

print(f"\nE_xc = integral(rho * exc * weights) = {np.sum(rho * exc * grid.weights):.6f}")
print(f"Tr(D * vxc) = {np.einsum('mn,mn->', D, vxc):.6f}")
print(f"integral(rho * vxc_rho) = {np.sum(rho * vxc_rho * grid.weights):.6f}")

print(f"\nExpected: Tr(D*vxc) should equal integral(rho*vxc_rho)")
print(f"Difference: {np.einsum('mn,mn->', D, vxc) - np.sum(rho * vxc_rho * grid.weights):.6e}")

print(f"\nVxc matrix trace: {np.trace(vxc):.6f}")
print(f"Vxc matrix min/max: {vxc.min():.6f} / {vxc.max():.6f}")

all_shells = basis_set.get_all_shells(mol.to_bohr())
shell_range = []
idx = 0
for s in all_shells:
    nf = s.n_functions
    shell_range.append((idx, idx + nf))
    idx += nf

vxc_direct = np.zeros_like(vxc)
for i, shell_i in enumerate(all_shells):
    si, ei = shell_range[i]
    for j, shell_j in enumerate(all_shells):
        sj, ej = shell_range[j]
        block = _compute_vxc_block(shell_i, shell_j, vxc_rho, grid)
        vxc_direct[si:ei, sj:ej] += block

print(f"\nDirect Vxc trace: {np.trace(vxc_direct):.6f}")
print(f"||vxc - vxc_direct|| = {np.linalg.norm(vxc - vxc_direct):.6e}")

print(f"\nExpected relation for LDA:")
print(f"  vxc = (4/3)*exc for homogeneous electron gas")
print(f"  (4/3)*E_xc = {(4/3)*np.sum(rho * exc * grid.weights):.6f}")
print(f"  Tr(D*vxc) / 2 = {np.einsum('mn,mn->', D, vxc) / 2:.6f}")
