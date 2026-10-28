import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.two_electron import compute_eri

mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")
eri = compute_eri(basis, mol)

print("ERI 索引检查 (p_x 轨道，索引 2):", flush=True)
print(f"  eri[2,2,2,2] = {eri[2,2,2,2]:.6f} (J_22 = K_22 应该等于这个)", flush=True)
print(f"  eri[2,2,2,2] 是 (22|22)", flush=True)

# 对于 2 号基函数 (p_x)，J_22 = sum_{ls} D_{ls} * (22|ls)
# K_22 = sum_{ls} D_{ls} * (2l|2s)
# 如果 D 只有 2,2 元非零（D_22 = 2），那么：
# J_22 = D_22 * (22|22) = 2 * eri[2,2,2,2]
# K_22 = D_22 * (22|22) = 2 * eri[2,2,2,2]
# 所以 J_22 = K_22

print(f"\n假设 D_22 = 2:", flush=True)
print(f"  J_22 = 2 * eri[2,2,2,2] = {2 * eri[2,2,2,2]:.6f}", flush=True)
print(f"  K_22 = 2 * eri[2,2,2,2] = {2 * eri[2,2,2,2]:.6f}", flush=True)

# 现在用 einsum 计算
dm_test = np.zeros((5, 5))
dm_test[2, 2] = 2.0

j_test = np.einsum("mnls,ls->mn", eri, dm_test)
k_test = np.einsum("mlns,ls->mn", eri, dm_test)

print(f"\n用 einsum 计算:", flush=True)
print(f"  J_22 = {j_test[2,2]:.6f}", flush=True)
print(f"  K_22 = {k_test[2,2]:.6f}", flush=True)

if abs(j_test[2,2] - k_test[2,2]) > 1e-6:
    print(f"  J != K，有问题！", flush=True)
    print(f"  检查 K 的 einsum 索引...", flush=True)
    # K_mu_nu = sum_{lambda sigma} D_{lambda sigma} * (mu lambda | nu sigma)
    # 即 K[mu, nu] = sum_{l,s} D[l,s] * eri[mu,l,nu,s]
    k_test2 = np.einsum("mlns,ls->mn", eri, dm_test)  # 这应该是对的
    print(f"  K (mlns,ls->mn) = {k_test2[2,2]:.6f}", flush=True)
    
    # 试试其他索引顺序
    k_test3 = np.einsum("m n l s, l s -> m n", eri, dm_test)
    print(f"  K (mnls,ls->mn) = {k_test3[2,2]:.6f}", flush=True)
    
    k_test4 = np.einsum("l m n s, l s -> m n", eri, dm_test)
    print(f"  K (lmns,ls->mn) = {k_test4[2,2]:.6f}", flush=True)
    
    k_test5 = np.einsum("m l s n, l s -> m n", eri, dm_test)
    print(f"  K (mlsn,ls->mn) = {k_test5[2,2]:.6f}", flush=True)
else:
    print(f"  J = K，正确！", flush=True)

print(f"\n检查 ERI 的置换对称性:", flush=True)
print(f"  eri[2,2,3,3] = {eri[2,2,3,3]:.6f}", flush=True)
print(f"  eri[2,3,2,3] = {eri[2,3,2,3]:.6f}", flush=True)
print(f"  eri[2,3,3,2] = {eri[2,3,3,2]:.6f}", flush=True)
