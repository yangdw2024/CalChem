import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.two_electron import compute_eri

print("=" * 60)
print("验证 J 和 K 矩阵的对角元关系")
print("对于任意轨道 i, J_ii = K_ii = (ii|ii)")
print("=" * 60)

mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")
eri = compute_eri(basis, mol)

nbf = eri.shape[0]

# 构造一个简单的密度矩阵，只有轨道 i 被占据
for i in range(nbf):
    dm = np.zeros((nbf, nbf))
    dm[i, i] = 2.0  # 2个电子
    
    j = np.einsum("mnls,ls->mn", eri, dm)
    k = np.einsum("mlns,ls->mn", eri, dm)
    
    j_ii = j[i, i]
    k_ii = k[i, i]
    eri_iiii = eri[i, i, i, i]
    
    print(f"\n轨道 {i}:")
    print(f"  J_{i}{i} = {j_ii:.10f}")
    print(f"  K_{i}{i} = {k_ii:.10f}")
    print(f"  ({i}{i}|{i}{i}) = {eri_iiii:.10f}")
    print(f"  J_ii == K_ii? {'✓' if abs(j_ii - k_ii) < 1e-10 else '✗'}")
    print(f"  J_ii == (ii|ii)? {'✓' if abs(j_ii - eri_iiii) < 1e-10 else '✗'}")

print("\n" + "=" * 60)
print("验证非对角元: J_ij 和 K_ij 的关系")
print("对于对角密度矩阵，J_ij = (ii|jj), K_ij = (ij|ij)")
print("=" * 60)

# 检查轨道 2 (2p_x) 和轨道 3 (2p_y)
i, j = 2, 3
dm = np.zeros((nbf, nbf))
dm[i, i] = 2.0  # 只有轨道 i 被占据

j_mat = np.einsum("mnls,ls->mn", eri, dm)
k_mat = np.einsum("mlns,ls->mn", eri, dm)

print(f"\n密度矩阵只有轨道 {i} 占据时:")
print(f"  J[{i},{j}] = {j_mat[i,j]:.10f}")
print(f"  K[{i},{j}] = {k_mat[i,j]:.10f}")
print(f"  ({i}{i}|{j}{j}) = {eri[i,i,j,j]:.10f}")
print(f"  ({i}{j}|{i}{j}) = {eri[i,j,i,j]:.10f}")
print(f"  J_ij == (ii|jj)? {'✓' if abs(j_mat[i,j] - eri[i,i,j,j]) < 1e-10 else '✗'}")
print(f"  K_ij == (ij|ij)? {'✓' if abs(k_mat[i,j] - eri[i,j,i,j]) < 1e-10 else '✗'}")

# 验证更多的置换对称性
print("\n" + "=" * 60)
print("验证 ERI 的置换对称性")
print("=" * 60)

# (ab|cd) = (ba|dc) = (cd|ab) = (dc|ba)
# (ab|cd) = (ac|bd)? 不对，那是物理记号的置换
# 化学记号 (ab|cd) 的置换对称性：
# (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)

# 用 p_x, p_y, p_z 来测试
indices = [2, 3, 4]  # p_x, p_y, p_z

print("\n2p 轨道的 ERI 置换对称性检查:")
a, b, c, d = 2, 3, 2, 3  # (p_x p_y | p_x p_y)

print(f"\n(a={a}, b={b}, c={c}, d={d}):")
print(f"  (ab|cd) = {eri[a,b,c,d]:.10f}")
print(f"  (ba|cd) = {eri[b,a,c,d]:.10f}")
print(f"  (ab|dc) = {eri[a,b,d,c]:.10f}")
print(f"  (ba|dc) = {eri[b,a,d,c]:.10f}")
print(f"  (cd|ab) = {eri[c,d,a,b]:.10f}")
print(f"  (dc|ab) = {eri[d,c,a,b]:.10f}")
print(f"  (cd|ba) = {eri[c,d,b,a]:.10f}")
print(f"  (dc|ba) = {eri[d,c,b,a]:.10f}")

a, b, c, d = 2, 2, 3, 3  # (p_x p_x | p_y p_y)
print(f"\n(a={a}, b={b}, c={c}, d={d}):")
print(f"  (ab|cd) = {eri[a,b,c,d]:.10f}")
print(f"  (ba|cd) = {eri[b,a,c,d]:.10f}")
print(f"  (ab|dc) = {eri[a,b,d,c]:.10f}")
print(f"  (cd|ab) = {eri[c,d,a,b]:.10f}")
