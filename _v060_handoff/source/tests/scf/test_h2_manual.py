import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from scipy.linalg import eigh

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri

print("=" * 60)
print("手动验证 H2 STO-3G 能量计算")
print("参考值: -1.1167 Hartree (Szabo)")
print("=" * 60)

mol = Molecule(["H", "H"], 
               [[0, 0, 0], [0, 0, 1.4]], 
               units="bohr")
basis = BasisSet("sto-3g")

S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
H = T + V
eri = compute_eri(basis, mol)

print(f"\nS矩阵:")
print(S)
print(f"\nHcore矩阵:")
print(H)
print(f"\nERI (ij|kl):")
print(f"  (11|11) = {eri[0,0,0,0]:.6f}")
print(f"  (22|22) = {eri[1,1,1,1]:.6f}")
print(f"  (11|22) = {eri[0,0,1,1]:.6f}")
print(f"  (12|12) = {eri[0,1,0,1]:.6f}")

# 解广义本征方程
eigval, eigvec = eigh(H, S)
print(f"\nHcore轨道能量: {eigval}")

# 构建密度矩阵 (闭壳层, 2个电子 = 1个占据轨道)
nocc = 1
C = eigvec[:, :nocc]
D = 2.0 * C @ C.T
print(f"\n密度矩阵 D:")
print(D)

# 构建 J 和 K 矩阵
J = np.einsum("ls,mnls->mn", D, eri)
K = np.einsum("ls,mlns->mn", D, eri)
G = J - 0.5 * K
F = H + G

print(f"\nJ矩阵:")
print(J)
print(f"\nK矩阵:")
print(K)
print(f"\nG矩阵 (J - 0.5K):")
print(G)
print(f"\nFock矩阵 F = H + G:")
print(F)

# 计算能量
e_1e = np.sum(D * H)
e_2e = 0.5 * np.sum(D * G)

# 核排斥能
from calchem.scf.hf import nuclear_repulsion
e_nn = nuclear_repulsion(mol)

e_tot = e_1e + e_2e + e_nn

print(f"\n能量分量:")
print(f"  单电子能量 E_1e = {e_1e:.10f}")
print(f"  双电子能量 E_2e = {e_2e:.10f}")
print(f"  核排斥能 V_nn = {e_nn:.10f}")
print(f"  总能量 E_tot = {e_tot:.10f}")

# 用另一种公式验证: E = 0.5 * Tr[D (H + F)]
e_tot2 = 0.5 * np.sum(D * (H + F)) + e_nn
print(f"\n验证 (0.5 * Tr[D(H+F)] + V_nn): {e_tot2:.10f}")

# 用轨道能量验证: E = sum_i e_i - E_2e + V_nn ? 不对
# E = sum_i <i|h|i> + sum_ij (2J_ij - K_ij) + V_nn
# epsilon_i = <i|h|i> + sum_j (2J_ij - K_ij)
# sum_i epsilon_i = sum_i <i|h|i> + sum_i sum_j (2J_ij - K_ij) = E_1e + 2 E_2e
# 所以 E = sum_i epsilon_i - E_2e + V_nn

# 计算 MO 基的 Fock 矩阵
F_mo = C.T @ F @ C
print(f"\nMO基 Fock 矩阵对角元 (轨道能量):")
print(f"  e_1 = {F_mo[0,0]:.10f}")

sum_eps = F_mo[0,0] * 2  # 2个电子，每个轨道的能量是 epsilon，但是是占据轨道的能量
# 不对，sum_i epsilon_i 是对轨道求和，不是对电子求和
# 1个占据轨道，所以 sum_i epsilon_i = F_mo[0,0]

sum_eps = F_mo[0,0]
e_tot3 = sum_eps - e_2e + e_nn
print(f"\n验证 (sum(epsilon_i) - E_2e + V_nn): {e_tot3:.10f}")

print(f"\n参考能量: -1.1167 Hartree")
print(f"误差: {abs(e_tot + 1.1167):.6f}")
