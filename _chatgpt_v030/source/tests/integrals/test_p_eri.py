import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from calchem.basis.gto import GTOShell
from calchem.integrals.two_electron import _eri_full_os, eri_ssss

print("=" * 60)
print("测试 p 轨道双电子积分 (同中心)")
print("=" * 60)

alpha = 1.0
r = np.array([0.0, 0.0, 0.0])

print(f"\nalpha = {alpha}, 所有中心在原点")
print()

# (ss|ss)
ssss = eri_ssss(alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(ss|ss) = {ssss:.10f}")

# (s p_x | s s) - 应该为 0（奇函数积分）
spx_ss = _eri_full_os(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(s p_x | s s) = {spx_ss:.10e}  (应该接近 0)")

# (p_x s | p_x s)
pxs_pxs = _eri_full_os(1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                        alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(p_x s | p_x s) = {pxs_pxs:.10f}")

# (p_x p_x | s s)
pxpx_ss = _eri_full_os(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                        alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(p_x p_x | s s) = {pxpx_ss:.10f}")

# (p_x p_x | p_x p_x)
pxpx_pxpx = _eri_full_os(1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                          alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(p_x p_x | p_x p_x) = {pxpx_pxpx:.10f}")

# (p_x p_y | p_x p_y)
pxpy_pxpy = _eri_full_os(1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
                          alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(p_x p_y | p_x p_y) = {pxpy_pxpy:.10f}")

# (p_x p_x | p_y p_y)
pxpx_pypy = _eri_full_os(1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                          alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(p_x p_x | p_y p_y) = {pxpx_pypy:.10f}")

# 理论值（对于 alpha=1.0 的同中心情况，可以用解析公式验证）
# (ss|ss) = (7/8) * sqrt(pi/2) ≈ 1.096006
print()
print("理论参考 (alpha=1.0, 同中心):")
print(f"  (ss|ss) 理论 = {7/8 * np.sqrt(np.pi/2):.10f}")
print(f"  (p_x p_x | s s) 理论 = ?")

# 用递推关系验证：对于同中心 s 和 p，
# (p_x p_x | s s) + (p_y p_y | s s) + (p_z p_z | s s) 应该等于某个值

# 验证 J_ii = K_ii for p orbitals
# (p_x p_x | p_x p_x) - (p_x p_x | p_x p_x) = 0  trivially
# 但是 (p_x p_y | p_x p_y) 和 (p_x p_x | p_y p_y) 的关系

print()
print("=" * 60)
print("验证已知关系:")
print("=" * 60)

# 对于同中心的 p 轨道，应该满足:
# (p_x p_x | p_x p_x) = (p_y p_y | p_y p_y) = (p_z p_z | p_z p_z)
# (p_x p_y | p_x p_y) = (p_x p_x | p_y p_y)  (对于库仑积分和交换积分的关系)

# 让我们检查:
# J_xx = (p_x p_x | p_x p_x)
# K_xy = (p_x p_y | p_x p_y)
# J_xy = (p_x p_x | p_y p_y)
# 
# 对于同中心高斯函数，这些积分有已知的比例关系。

J_xx = pxpx_pxpx
K_xy = pxpy_pxpy
J_xy = pxpx_pypy

print(f"J_xx = (p_x p_x | p_x p_x) = {J_xx:.10f}")
print(f"K_xy = (p_x p_y | p_x p_y) = {K_xy:.10f}")
print(f"J_xy = (p_x p_x | p_y p_y) = {J_xy:.10f}")
print()
print(f"J_xx / J_xy = {J_xx/J_xy:.6f}")
print(f"J_xx / K_xy = {J_xx/K_xy:.6f}")

# 理论上，对于 s 型函数，(ss|ss) 是已知的
# 对于 p 型函数，我们可以用递推来验证
# 一个简单的验证: (p_x s | p_x s) 应该等于 (s p_x | s p_x) = (p_x s | p_x s)
# 而且 (p_x s | p_x s) = 1/3 * (p_x p_x | s s) ? 不对

# 让我们用另一种方式验证: 
# (p_x p_x | s s) = (p_y p_y | s s) = (p_z p_z | s s) (球对称)
# 并且我们可以用降低角动量的方式来验证

print()
print("=" * 60)
print("与 H 的 1s 轨道比较 (STO-3G):")
print("=" * 60)

# H 的 STO-3G 指数
# 从 Szabo 书中，STO-3G for H 1s:
# alpha = [0.3425250914, 0.6239137298, 0.1688554040]
# 不对，让我用已知的 (11|11) = 0.7746 来验证
# 但那是收缩后的，这里用的是单指数

# 让我们测试收缩基组
print()
print("测试完整的 O 原子 1s 自相互作用 (STO-3G):")

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.two_electron import compute_eri

mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")
eri = compute_eri(basis, mol)

print(f"ERI 形状: {eri.shape}")
print(f"(1s, 1s | 1s, 1s) = eri[0,0,0,0] = {eri[0,0,0,0]:.10f}")
print(f"(2s, 2s | 2s, 2s) = eri[1,1,1,1] = {eri[1,1,1,1]:.10f}")
print(f"(2p_x, 2p_x | 2p_x, 2p_x) = eri[2,2,2,2] = {eri[2,2,2,2]:.10f}")
print(f"(2p_x, 2p_y | 2p_x, 2p_y) = eri[2,3,2,3] = {eri[2,3,2,3]:.10f}")
print(f"(2p_x, 2p_x | 2p_y, 2p_y) = eri[2,2,3,3] = {eri[2,2,3,3]:.10f}")

# 验证对称性
print()
print("对称性验证:")
print(f"  eri[2,2,2,2] = {eri[2,2,2,2]:.10f}")
print(f"  eri[3,3,3,3] = {eri[3,3,3,3]:.10f}  (p_y 应该等于 p_x)")
print(f"  eri[4,4,4,4] = {eri[4,4,4,4]:.10f}  (p_z 应该等于 p_x)")

print()
print("比例关系:")
print(f"  J_xx / J_xy = {eri[2,2,2,2] / eri[2,2,3,3]:.6f}")
print(f"  J_xx / K_xy = {eri[2,2,2,2] / eri[2,3,2,3]:.6f}")
