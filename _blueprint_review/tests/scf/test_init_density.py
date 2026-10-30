"""检查初始密度矩阵"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF, guess_hcore
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.basis.basis_set import BasisSet

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
basis_set = BasisSet('sto-3g', mol)

S = compute_overlap(basis_set, mol.to_bohr())
T = compute_kinetic(basis_set, mol.to_bohr())
V = compute_nuclear(basis_set, mol.to_bohr())
H_core = T + V

# 方式1: 直接使用H_core本征向量
eigvals, eigvecs = np.linalg.eigh(H_core)
C1 = eigvecs[:, :5]
D1 = 2.0 * C1 @ C1.T

# 方式2: 使用guess_hcore
D2 = guess_hcore(H_core, S, mol.nelectron, restricted=True)

print(f"D1 tr(D*S) = {np.sum(D1 * S):.6f}")
print(f"D2 tr(D*S) = {np.sum(D2 * S):.6f}")
print(f"||D1 - D2|| = {np.linalg.norm(D1 - D2):.6e}")

# 计算初始能量
J = np.einsum("ls,mnls->mn", D1, compute_eri(basis_set, mol.to_bohr()))
E_1e = np.einsum("mn,mn->", D1, H_core)
E_J = 0.5 * np.einsum("mn,mn->", D1, J)
print(f"\nD1: E_1e = {E_1e:.6f}, E_J = {E_J:.6f}, E_core = {E_1e + E_J:.6f}")

J = np.einsum("ls,mnls->mn", D2, compute_eri(basis_set, mol.to_bohr()))
E_1e = np.einsum("mn,mn->", D2, H_core)
E_J = 0.5 * np.einsum("mn,mn->", D2, J)
print(f"D2: E_1e = {E_1e:.6f}, E_J = {E_J:.6f}, E_core = {E_1e + E_J:.6f}")
