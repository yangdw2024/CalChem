import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from scipy.linalg import eigh

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri

mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")
S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
H = T + V
eri = compute_eri(basis, mol)

eigval, eigvec = eigh(H, S)
print("Hcore 轨道能量:", eigval, flush=True)

nocc = 4
dm = 2.0 * np.einsum("ik,jk->ij", eigvec[:, :nocc], eigvec[:, :nocc])

print(f"\nTr(D @ S) = {np.trace(np.einsum('ik,kj->ij', dm, S)):.6f} (应该是 8)", flush=True)
print(f"D 对角元: {np.diag(dm)}", flush=True)
print(f"D 矩阵:", flush=True)
print(np.array2string(dm, precision=4, suppress_small=True), flush=True)

j = np.einsum("mnls,ls->mn", eri, dm)
k = np.einsum("mlns,ls->mn", eri, dm)
g = j - 0.5 * k

print(f"\nJ 对角元: {np.diag(j)}", flush=True)
print(f"K 对角元: {np.diag(k)}", flush=True)
print(f"G 对角元: {np.diag(g)}", flush=True)

e_1e = np.sum(dm * H)
e_2e = 0.5 * np.sum(dm * g)
print(f"\n单电子能量: {e_1e:.6f}", flush=True)
print(f"双电子能量: {e_2e:.6f}", flush=True)
print(f"sum(dm*j) = {np.sum(dm*j):.6f}", flush=True)
print(f"sum(dm*k) = {np.sum(dm*k):.6f}", flush=True)
print(f"0.5*sum(dm*j) = {0.5*np.sum(dm*j):.6f}", flush=True)
print(f"0.25*sum(dm*k) = {0.25*np.sum(dm*k):.6f}", flush=True)

# 检查 ERI 的对称性
print(f"\nERI 对称性检查:", flush=True)
print(f"  eri[0,0,0,0] = {eri[0,0,0,0]:.6f}", flush=True)
print(f"  eri[0,2,0,2] = {eri[0,2,0,2]:.6f} (应该是正的)", flush=True)
print(f"  eri[2,2,2,2] = {eri[2,2,2,2]:.6f}", flush=True)
