"""检查DFT能量计算公式"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import guess_hcore, nuclear_repulsion
from calchem.dft.grid import DFTGrid
from calchem.dft.fock import compute_vxc
from calchem.dft.functionals import LDA
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
basis_set = BasisSet('sto-3g', mol)

S = compute_overlap(basis_set, mol.to_bohr())
H_core = compute_kinetic(basis_set, mol.to_bohr()) + compute_nuclear(basis_set, mol.to_bohr())
eri = compute_eri(basis_set, mol.to_bohr())

D = guess_hcore(H_core, S, mol.nelectron, restricted=True)

func = LDA()
grid = DFTGrid(mol.to_bohr(), basis_set, 'coarse')

rho = grid.compute_density(D, basis_set, mol.to_bohr())
exc, vxc_rho = func.compute(rho)

vxc = compute_vxc(D, basis_set, mol.to_bohr(), func, grid)
J = np.einsum("ls,mnls->mn", D, eri)
F = H_core + J + vxc

# 方法1: 标准DFT能量公式
E_1e = np.einsum("mn,mn->", D, H_core)
E_J = 0.5 * np.einsum("mn,mn->", D, J)
E_xc = np.sum(rho * exc * grid.weights)
E_nn = nuclear_repulsion(mol.to_bohr())
E_total1 = E_1e + E_J + E_xc + E_nn

# 方法2: 使用Fock矩阵的公式
E_total2 = 0.5 * np.einsum("mn,mn->", D, H_core + F) + E_nn

# 验证: vxc = exc + rho * d(exc)/d(rho)
# 所以: integral(rho * vxc) = integral(rho * exc) + integral(rho^2 * d(exc)/d(rho))
integral_rho_vxc = np.sum(rho * vxc_rho * grid.weights)
integral_rho_exc = np.sum(rho * exc * grid.weights)
integral_density_matrix_vxc = np.einsum("mn,mn->", D, vxc)

print("Energy components:")
print(f"  E_1e = {E_1e:.6f}")
print(f"  E_J = {E_J:.6f}")
print(f"  E_xc = {E_xc:.6f}")
print(f"  E_nn = {E_nn:.6f}")
print(f"  E_total1 (E_1e+E_J+E_xc+E_nn) = {E_total1:.6f}")
print(f"  E_total2 (0.5*Tr[D*(H+F)] + E_nn) = {E_total2:.6f}")
print(f"  Difference: {E_total1 - E_total2:.6e}")

print("\nIntegrals:")
print(f"  integral(rho * exc) = {integral_rho_exc:.6f}")
print(f"  integral(rho * vxc) = {integral_rho_vxc:.6f}")
print(f"  Tr(D * vxc) = {integral_density_matrix_vxc:.6f}")

# 检查: vxc = exc + rho * d(exc)/d(rho)
# integral(rho * vxc) = integral(rho * exc) + integral(rho^2 * d(exc)/d(rho))
# Tr(D * vxc) = integral(rho * vxc)
print(f"\nConsistency check:")
print(f"  Tr(D*vxc) should equal integral(rho*vxc): {integral_density_matrix_vxc - integral_rho_vxc:.6e}")

# LDA vxc = (4/3)*exc, 所以 integral(rho * vxc) = (4/3)*integral(rho * exc)
print(f"  LDA: (4/3)*integral(rho*exc) = {(4/3)*integral_rho_exc:.6f}")
print(f"  integral(rho*vxc) = {integral_rho_vxc:.6f}")
print(f"  Difference: {integral_rho_vxc - (4/3)*integral_rho_exc:.6e}")
