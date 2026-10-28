import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.two_electron import compute_eri

print("=" * 60)
print("O 原子双电子积分详细检查")
print("=" * 60)

mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")
eri = compute_eri(basis, mol)

print("基函数索引: 0=1s, 1=2s, 2=2p_x, 3=2p_y, 4=2p_z")
print()

print("s-s 类型积分:")
print(f"  (1s,1s|1s,1s) = {eri[0,0,0,0]:.10f}")
print(f"  (1s,1s|2s,2s) = {eri[0,0,1,1]:.10f}")
print(f"  (2s,2s|2s,2s) = {eri[1,1,1,1]:.10f}")
print(f"  (1s,2s|1s,2s) = {eri[0,1,0,1]:.10f}")

print()
print("s-p 类型积分 (应该都不为零):")
print(f"  (1s,1s|2p_x,2p_x) = {eri[0,0,2,2]:.10f}")
print(f"  (1s,1s|2p_y,2p_y) = {eri[0,0,3,3]:.10f}")
print(f"  (1s,1s|2p_z,2p_z) = {eri[0,0,4,4]:.10f}")
print(f"  (2s,2s|2p_x,2p_x) = {eri[1,1,2,2]:.10f}")
print(f"  (1s,2s|2p_x,2p_x) = {eri[0,1,2,2]:.10f}")

print()
print("s-p 交叉项 (应该为零, 因为奇函数):")
print(f"  (1s,2p_x|1s,1s) = {eri[0,2,0,0]:.10e}")
print(f"  (1s,1s|1s,2p_x) = {eri[0,0,0,2]:.10e}")

print()
print("p-p 类型积分:")
print(f"  (2p_x,2p_x|2p_x,2p_x) = {eri[2,2,2,2]:.10f}")
print(f"  (2p_x,2p_x|2p_y,2p_y) = {eri[2,2,3,3]:.10f}")
print(f"  (2p_x,2p_y|2p_x,2p_y) = {eri[2,3,2,3]:.10f}")
print(f"  (2p_x,2p_x|2p_z,2p_z) = {eri[2,2,4,4]:.10f}")
print(f"  (2p_y,2p_y|2p_z,2p_z) = {eri[3,3,4,4]:.10f}")

# 验证一个重要的关系: 对于同中心的 s 和 p 轨道
# (s s | p_x p_x) + (s s | p_y p_y) + (s s | p_z p_z) 应该等于某个值
# 我们可以用 sum rule 来验证

print()
print("=" * 60)
print("Sum rule 验证")
print("=" * 60)

# 对于 (1s, 1s | 2p, 2p) 的迹
sum_pp = eri[0,0,2,2] + eri[0,0,3,3] + eri[0,0,4,4]
print(f"(1s,1s|2p_x,2p_x) + (1s,1s|2p_y,2p_y) + (1s,1s|2p_z,2p_z) = {sum_pp:.10f}")

# 这个和应该等于 (1s, 1s | r^2 Y_1^m 之和) 的某种形式
# 实际上，对于球对称的情况，sum_m (s s | p_m p_m) 有一个确定的值
# 我们可以用递推关系来验证

# 让我们也检查一下 p-p 积分的和
sum_pppp_diag = eri[2,2,2,2] + eri[3,3,3,3] + eri[4,4,4,4]
sum_pppp_offdiag = eri[2,2,3,3] + eri[2,2,4,4] + eri[3,3,4,4]
sum_pppp_exchange = eri[2,3,2,3] + eri[2,4,2,4] + eri[3,4,3,4]
print(f"\np-p 积分和:")
print(f"  对角项和 (J_xx + J_yy + J_zz) = {sum_pppp_diag:.10f}")
print(f"  非对角库仑和 (J_xy + J_xz + J_yz) = {sum_pppp_offdiag:.10f}")
print(f"  交换项和 (K_xy + K_xz + K_yz) = {sum_pppp_exchange:.10f}")

# 对于同中心的 p 轨道，应该满足一些关系
# 比如: J_xx = J_yy = J_zz, J_xy = J_xz = J_yz, K_xy = K_xz = K_yz
print(f"\n对称性检查:")
print(f"  J_xx == J_yy == J_zz: {abs(eri[2,2,2,2] - eri[3,3,3,3]) < 1e-10 and abs(eri[2,2,2,2] - eri[4,4,4,4]) < 1e-10}")
print(f"  J_xy == J_xz == J_yz: {abs(eri[2,2,3,3] - eri[2,2,4,4]) < 1e-10 and abs(eri[2,2,3,3] - eri[3,3,4,4]) < 1e-10}")
print(f"  K_xy == K_xz == K_yz: {abs(eri[2,3,2,3] - eri[2,4,2,4]) < 1e-10 and abs(eri[2,3,2,3] - eri[3,4,3,4]) < 1e-10}")

# 现在让我们用一个简单的方法来估算这些积分的"正确"值
# 我们可以用 s 轨道积分来推算 p 轨道积分
# 
# 从递推关系，我们知道：
# (p_x p_x | s s) = 1/(2ζ_ab) * (s s | s s) [当 PA=0, PQ=0 时]
# 不对，这是对于相同指数的情况

print()
print("=" * 60)
print("用单指数验证 (alpha=1.0)")
print("=" * 60)

from calchem.integrals.two_electron import eri_ssss, _eri_full_os

alpha = 1.0
r = np.array([0.0, 0.0, 0.0])

ssss = eri_ssss(alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(ss|ss) = {ssss:.10f}")

# 对于同中心、同指数的情况：
# 从 VRR，我们可以推导：
# (p_x s | p_x s) = ?
# (p_x p_x | s s) = ?

# 让我们用递推来验证
# 从 (s s | s s) 开始
# (s p_x | s s) = 0 (奇函数)
# (p_x p_x | s s) = ?

# 用 VRR 增加第一个函数的 x 角动量：
# (p_x p_x | s s) = PA_x * (s p_x | s s) + a_x/(2ζ_ab) * (s-1_x p_x | s s) + b_x/(2ζ_ab) * (s s | s s) + ...
# PA_x = 0, a_x = 0 (a是s), b_x = 1 (b是p_x)
# 所以: (p_x p_x | s s) = 1/(2ζ_ab) * (s s | s s) + ket侧项
# ket侧项: (ζ_cd/ζ) * PQ_x * (s p_x | s s) + ... = 0
# 所以: (p_x p_x | s s) = 1/(2ζ_ab) * (ss|ss) = 1/4 * (ss|ss)

expected = ssss / 4.0
actual = _eri_full_os(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n(p_x p_x | s s):")
print(f"  计算值: {actual:.10f}")
print(f"  期望值 (ss|ss)/4 = {expected:.10f}")
print(f"  比值: {actual/ssss:.6f}  (应该是 0.25)")

# 同样，(p_x s | p_x s) = ?
# 增加 ket 侧的 c 函数
# (p_x s | p_x s) = QC_x * (p_x s | s s) + c_x/(2ζ_cd) * (p_x s | s-1_x s) + d_x/(2ζ_cd) * (p_x s | s s-1_x) + bra侧项
# QC_x = 0, c_x = 0 (c是s), d_x = 0 (d是s)
# bra侧项: (ζ_ab/ζ) * (-PQ_x) * (p_x s | s s) + a_x/(2ζ) * (s s | s s) + b_x/(2ζ) * ...
# PQ_x = 0, b_x = 0 (b是s)
# 所以: (p_x s | p_x s) = a_x/(2ζ) * (s s | s s) = 1/(2*4) * (ss|ss) = (ss|ss)/8

expected = ssss / 8.0
actual = _eri_full_os(1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                      alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n(p_x s | p_x s):")
print(f"  计算值: {actual:.10f}")
print(f"  期望值 (ss|ss)/8 = {expected:.10f}")
print(f"  比值: {actual/ssss:.6f}  (应该是 0.125)")

# 现在，(p_x p_x | p_x p_x) = ?
# 这个更复杂，但我们可以用比例来验证
# 让我们看看 (p_x p_x | p_x p_x) / (ss|ss) 的比值

actual = _eri_full_os(1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                      alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n(p_x p_x | p_x p_x):")
print(f"  计算值: {actual:.10f}")
print(f"  比值 (pxpx|pxpx)/(ss|ss) = {actual/ssss:.6f}")

# 对于同中心同指数的情况，这个比值应该是一个确定的数
# 让我手动推导一下...
# 
# 实际上，我们可以用已知的球谐积分来计算。
# 但这太复杂了。让我们用另一种方法验证。
# 
# 让我们检查 (p_x p_x | p_y p_y) / (ss|ss) 的比值
actual = _eri_full_os(1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                      alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n(p_x p_x | p_y p_y):")
print(f"  计算值: {actual:.10f}")
print(f"  比值 = {actual/ssss:.6f}")

# 以及 (p_x p_y | p_x p_y) / (ss|ss)
actual = _eri_full_os(1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
                      alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n(p_x p_y | p_x p_y):")
print(f"  计算值: {actual:.10f}")
print(f"  比值 = {actual/ssss:.6f}")
