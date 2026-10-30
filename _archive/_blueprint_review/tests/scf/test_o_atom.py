import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from scipy.linalg import eigh

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri

print("测试 O 原子 STO-3G (bohr)", flush=True)
mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")

S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
H = T + V

print(f"基函数数目: {S.shape[0]}", flush=True)
print(f"S 对角元: {np.diag(S)}", flush=True)
print(f"T 迹: {np.trace(T):.6f}", flush=True)
print(f"V 迹: {np.trace(V):.6f}", flush=True)

eigval, eigvec = eigh(H, S)
print(f"\nHcore 轨道能量:", flush=True)
for i, e in enumerate(eigval):
    print(f"  {i+1}: {e:12.6f}", flush=True)

# O 原子有 8 个电子 = 4 个占据轨道 (闭壳层)
nocc = 4
dm = 2.0 * np.einsum("ik,jk->ij", eigvec[:, :nocc], eigvec[:, :nocc])
e_1e = np.sum(dm * H)
print(f"\n单电子能量: {e_1e:.6f}", flush=True)

eri = compute_eri(basis, mol)
j = np.einsum("mnls,ls->mn", eri, dm)
k = np.einsum("mlns,ls->mn", eri, dm)
g = j - 0.5 * k
e_2e = 0.5 * np.sum(dm * g)
print(f"双电子能量: {e_2e:.6f}", flush=True)

e_tot = e_1e + e_2e
print(f"总能量 (一步): {e_tot:.6f}", flush=True)
print(f"O 原子参考值 (STO-3G): ~ -74.8 Hartree?", flush=True)

print(f"\nERI (11|11) = {eri[0,0,0,0]:.6f}", flush=True)
print(f"ERI (22|22) = {eri[1,1,1,1]:.6f}", flush=True)
