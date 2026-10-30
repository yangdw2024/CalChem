import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear

print("=" * 60)
print("测试 O 原子单电子积分 (s-p 交叉项)")
print("=" * 60)

mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")

S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
H = T + V

print(f"\n基函数: 1s, 2s, 2p_x, 2p_y, 2p_z")
print(f"\n重叠矩阵 S:")
print(S)
print(f"\n动能矩阵 T:")
print(T)
print(f"\n核吸引矩阵 V:")
print(V)
print(f"\n核心哈密顿量 H:")
print(H)

print(f"\n检查 s-p 交叉项 (应该为 0):")
print(f"  S[0,2] (1s, 2p_x) = {S[0,2]:.10e}")
print(f"  S[1,2] (2s, 2p_x) = {S[1,2]:.10e}")
print(f"  T[0,2] (1s, 2p_x) = {T[0,2]:.10e}")
print(f"  T[1,2] (2s, 2p_x) = {T[1,2]:.10e}")
print(f"  V[0,2] (1s, 2p_x) = {V[0,2]:.10e}")
print(f"  V[1,2] (2s, 2p_x) = {V[1,2]:.10e}")

print(f"\n检查 p-p 交叉项:")
print(f"  S[2,3] (2p_x, 2p_y) = {S[2,3]:.10e}")
print(f"  T[2,3] (2p_x, 2p_y) = {T[2,3]:.10e}")
print(f"  V[2,3] (2p_x, 2p_y) = {V[2,3]:.10e}")

print(f"\np 轨道对角元:")
print(f"  T[2,2] (2p_x 动能) = {T[2,2]:.10f}")
print(f"  V[2,2] (2p_x 核吸引) = {V[2,2]:.10f}")
print(f"  H[2,2] (2p_x Hcore) = {H[2,2]:.10f}")

# 与 s 轨道比较
print(f"\ns 轨道对角元:")
print(f"  T[0,0] (1s 动能) = {T[0,0]:.10f}")
print(f"  V[0,0] (1s 核吸引) = {V[0,0]:.10f}")
print(f"  T[1,1] (2s 动能) = {T[1,1]:.10f}")
print(f"  V[1,1] (2s 核吸引) = {V[1,1]:.10f}")

# 维里定理验证: 对于原子, <T> = -<E_tot>, <V> = 2<E_tot> (对于库仑势)
# 不对，维里定理是 2<T> + <V> = 0 对于纯库仑势
# 但这是总能量，不是Hcore
print(f"\n维里定理检查 (对于 Hcore 单电子):")
print(f"  2*T[0,0] + V[0,0] = {2*T[0,0] + V[0,0]:.10f}  (应该接近 0 对于类氢 1s?)")
print(f"  2*T[2,2] + V[2,2] = {2*T[2,2] + V[2,2]:.10f}  (应该接近 0 对于类氢 2p?)")
