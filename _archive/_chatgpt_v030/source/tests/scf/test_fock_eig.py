"""对比RHF和LDA的Fock矩阵特征值"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import guess_hcore
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
vxc = compute_vxc(D, basis_set, mol.to_bohr(), func, grid)

J = np.einsum("ls,mnls->mn", D, eri)
K = np.einsum("ls,mlns->mn", D, eri)

F_rhf = H_core + J - K
F_lda = H_core + J + vxc

print("Fock eigenvalues:")
print(f"{'H_core':<12}: {np.linalg.eigvalsh(H_core)}")
print(f"{'J':<12}: {np.linalg.eigvalsh(J)}")
print(f"{'K':<12}: {np.linalg.eigvalsh(K)}")
print(f"{'Vxc':<12}: {np.linalg.eigvalsh(vxc)}")
print(f"{'F_RHF':<12}: {np.linalg.eigvalsh(F_rhf)}")
print(f"{'F_LDA':<12}: {np.linalg.eigvalsh(F_lda)}")

print("\nTrace comparison:")
print(f"  H_core: {np.trace(H_core):.6f}")
print(f"  J: {np.trace(J):.6f}")
print(f"  K: {np.trace(K):.6f}")
print(f"  -K: {-np.trace(K):.6f}")
print(f"  Vxc: {np.trace(vxc):.6f}")
print(f"  F_RHF: {np.trace(F_rhf):.6f}")
print(f"  F_LDA: {np.trace(F_lda):.6f}")
print(f"  Difference: {np.trace(F_lda) - np.trace(F_rhf):.6f}")

# 检查对角元素
print("\nDiagonal elements comparison:")
print(f"  H_core diag: {np.diag(H_core)}")
print(f"  J diag: {np.diag(J)}")
print(f"  K diag: {np.diag(K)}")
print(f"  Vxc diag: {np.diag(vxc)}")
print(f"  F_RHF diag: {np.diag(F_rhf)}")
print(f"  F_LDA diag: {np.diag(F_lda)}")
