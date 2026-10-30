import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.hf import RHF

print("=" * 60)
print("测试 Be 原子 STO-3G RHF")
print("参考值: ~ -14.35188 Hartree (闭壳层)")
print("=" * 60)

mol = Molecule(["Be"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")

S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
H = T + V
eri = compute_eri(basis, mol)

print(f"\n基函数数目: {S.shape[0]}")
print(f"电子数: {mol.nelectron}")
print(f"S 对角元: {np.diag(S)}")
print(f"Hcore 迹: {np.trace(H):.6f}")

# 手动计算一步
from scipy.linalg import eigh
eigval, eigvec = eigh(H, S)
print(f"\nHcore 轨道能量:")
for i, e in enumerate(eigval):
    print(f"  {i+1}: {e:12.6f}")

nocc = mol.nelectron // 2
dm = 2.0 * np.einsum("ik,jk->ij", eigvec[:, :nocc], eigvec[:, :nocc])

j = np.einsum("mnls,ls->mn", eri, dm)
k = np.einsum("mlns,ls->mn", eri, dm)
g = j - 0.5 * k

e_1e = np.sum(dm * H)
e_2e = 0.5 * np.sum(dm * g)

print(f"\n单电子能量: {e_1e:.6f}")
print(f"双电子能量: {e_2e:.6f}")
print(f"总能量 (一步): {e_1e + e_2e:.6f}")

print(f"\n一些 ERI 值:")
print(f"  (1s,1s|1s,1s) = {eri[0,0,0,0]:.6f}")
print(f"  (2s,2s|2s,2s) = {eri[1,1,1,1]:.6f}")
print(f"  (1s,1s|2s,2s) = {eri[0,0,1,1]:.6f}")
print(f"  (1s,2s|1s,2s) = {eri[0,1,0,1]:.6f}")

# 运行完整 SCF
print("\n" + "=" * 60)
print("完整 SCF 计算:")
print("=" * 60)

mf = RHF(mol, basis="sto-3g", max_cycles=50, conv_tol=1e-8, diis=False, verbose=2)
e_tot = mf.kernel()

print(f"\n最终能量: {e_tot:.6f} Eh")
print(f"参考值:   -14.35188 Eh")
print(f"误差:     {abs(e_tot + 14.35188):.6f} Eh")
print(f"收敛:     {mf.converged}")
