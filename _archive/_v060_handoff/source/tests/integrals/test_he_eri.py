import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from scipy.linalg import eigh

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri

mol = Molecule(["He"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")
S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
H = T + V
eri = compute_eri(basis, mol)

print("He 原子 STO-3G 测试", flush=True)
print(f"S = {S[0,0]:.10f}", flush=True)
print(f"T = {T[0,0]:.10f}", flush=True)
print(f"V = {V[0,0]:.10f}", flush=True)
print(f"H = {H[0,0]:.10f}", flush=True)
print(f"ERI (11|11) = {eri[0,0,0,0]:.10f}", flush=True)

eigval, eigvec = eigh(H, S)
print(f"轨道能量: {eigval[0]:.10f}", flush=True)

dm = 2.0 * np.einsum("ik,jk->ij", eigvec[:, :1], eigvec[:, :1])
print(f"D 对角元: {dm[0,0]:.10f}", flush=True)
print(f"Tr(D) = {np.trace(dm @ S):.10f} (应该是 2)", flush=True)

# 手动计算 J 和 K
j = eri[0,0,0,0] * dm[0,0]
k = eri[0,0,0,0] * dm[0,0]  # 对于单基函数，J 和 K 的对角元相等
print(f"J[0,0] = {j:.10f}", flush=True)
print(f"K[0,0] = {k:.10f}", flush=True)
print(f"G[0,0] = J - 0.5K = {j - 0.5*k:.10f}", flush=True)

e_1e = np.sum(dm * H)
e_2e = 0.5 * np.sum(dm * (j - 0.5*k))
e_tot = e_1e + e_2e

print(f"\n单电子能量: {e_1e:.10f}", flush=True)
print(f"双电子能量: {e_2e:.10f}", flush=True)
print(f"总能量 (一步): {e_tot:.10f}", flush=True)
print(f"参考值: ~ -2.8477 Hartree", flush=True)

# 用 einsum 计算
j_full = np.einsum("mnls,ls->mn", eri, dm)
k_full = np.einsum("mlns,ls->mn", eri, dm)
g_full = j_full - 0.5 * k_full
e_2e_full = 0.5 * np.sum(dm * g_full)
print(f"\n用 einsum 计算的双电子能量: {e_2e_full:.10f}", flush=True)
