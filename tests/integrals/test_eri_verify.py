import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from calchem.integrals.two_electron import _eri_full_os, eri_ssss

print("=" * 60)
print("验证 Obara-Saika 递推关系 (同中心, alpha=1.0)")
print("=" * 60)

alpha = 1.0
r = np.array([0.0, 0.0, 0.0])
zeta_ab = 2.0  # alpha + alpha = 2
zeta_cd = 2.0
zeta = 4.0

# 基础: (ss|ss)
ssss = eri_ssss(alpha, alpha, alpha, alpha, r, r, r, r)
print(f"(ss|ss) = {ssss:.10f}")

# 验证 1: (p_x s | s s) 应该为 0
val = _eri_full_os(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n(p_x s | s s) = {val:.10e}")
print(f"  期望值: 0 (奇函数积分)")
print(f"  验证: {'✓' if abs(val) < 1e-10 else '✗'}")

# 验证 2: (p_x p_x | s s)
# 用 VRR 从 (s p_x | s s) 递推
# (p_x p_x | s s) = PA_x (s p_x | s s) + a_x/(2ζ_ab) (s-1_x p_x | s s) + b_x/(2ζ_ab) (s p_x-1_x | s s) + ...
# 因为 PA_x = 0, a_x = 0 (s轨道), 所以:
# (p_x p_x | s s) = 1/(2ζ_ab) * (s s | s s) + ket侧项
# ket侧项: (ζ_cd/ζ) PQ_x (s p_x | s s) + ... 都是 0
# 所以: (p_x p_x | s s) = 1/(2ζ_ab) * (ss|ss) ?
# 不对, 等一下, b_x 是 lb 的角动量量子数, 也就是 1
# 所以 term3 = lb_ / (2*zeta_ab) * compute(la_-1, lb_-1, ...)
# 这里 la_-1 = 0 (因为我们从 s 递推到 p_x, 所以初始 la=0, 不对...)
# 
# 让我重新想: 我们要计算 (p_x p_x | s s)
# 可以把第一个函数从 s 增加到 p_x:
# (p_x p_x | s s) = (a+1_x b | c d) 其中 a=s(la=0), b=p_x(lb=1), c=s, d=s
#
# PA_x = 0
# a_x = 0 (a是s轨道)
# b_x = 1 (b是p_x轨道, lb=1)
#
# term1 = PA_x * (s p_x | s s) = 0
# term2 = a_x / (2ζ_ab) * (s-1_x p_x | s s) = 0 (因为 a_x = 0)
# term3 = b_x / (2ζ_ab) * (s s | s s) = 1 / (2ζ_ab) * (ss|ss)
# term4_p1 = (ζ_cd/ζ) * PQ_x * (s p_x | s s) = 0
# term4_p2 = c_x / (2ζ) * ... = 0 (c是s轨道)
# term4_p3 = d_x / (2ζ) * ... = 0 (d是s轨道)
#
# 所以 (p_x p_x | s s) = 1/(2ζ_ab) * (ss|ss) ?

expected = 1.0 / (2.0 * zeta_ab) * ssss
actual = _eri_full_os(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n(p_x p_x | s s):")
print(f"  计算值: {actual:.10f}")
print(f"  期望值 (从递推): {expected:.10f}")
print(f"  验证: {'✓' if abs(actual - expected) < 1e-8 else '✗'}")

# 验证 3: (p_x s | p_x s)
# 这个积分 = (p_x s | p_x s)
# 我们可以把 ket 侧的 c 从 s 增加到 p_x
# (p_x s | p_x s) = (a b | c+1_x d) 其中 a=p_x, b=s, c=s, d=s
#
# QC_x = 0
# c_x = 0 (c是s轨道)
# d_x = 0 (d是s轨道)
#
# term1 = QC_x * (p_x s | s s) = 0
# term2 = c_x / (2ζ_cd) * ... = 0
# term3 = d_x / (2ζ_cd) * ... = 0
# term4_p1 = (ζ_ab/ζ) * (-PQ_x) * (p_x s | s s) = 0
# term4_p2 = a_x / (2ζ) * (s s | s s) = 1 / (2ζ) * (ss|ss)
# term4_p3 = b_x / (2ζ) * ... = 0 (b是s轨道)
#
# 所以 (p_x s | p_x s) = 1/(2ζ) * (ss|ss) ?

expected = 1.0 / (2.0 * zeta) * ssss
actual = _eri_full_os(1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                      alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n(p_x s | p_x s):")
print(f"  计算值: {actual:.10f}")
print(f"  期望值 (从递推): {expected:.10f}")
print(f"  验证: {'✓' if abs(actual - expected) < 1e-8 else '✗'}")

# 验证 4: (p_x p_x | p_x p_x)
# 这个更复杂，让我们用分步递推来验证
# 先算 (p_x p_x | p_x s), 再算 (p_x p_x | p_x p_x)
# 
# 或者，让我们用另一种方式验证：检查角向积分的比例
# 
# 对于同中心的 p 轨道，我们知道：
# (p_x p_x | p_x p_x) + (p_y p_y | p_y p_y) + (p_z p_z | p_z p_z)  有一个值
# (p_x p_x | p_y p_y) + (p_x p_x | p_z p_z) + (p_y p_y | p_z p_z) 有另一个值
#
# 实际上，让我们直接计算已知的积分关系：
# J_xx = (p_x p_x | p_x p_x)
# J_xy = (p_x p_x | p_y p_y)
# K_xy = (p_x p_y | p_x p_y)
# 
# 对于笛卡尔高斯函数，有一个关系：
# J_xx = J_xy + 2 K_xy ? 不对
#
# 让我用球谐函数来思考。对于纯 p 轨道（球谐），有：
# (p_0 p_0 | p_0 p_0) = J_0
# (p_0 p_0 | p_1 p_1) = J_1
# (p_1 p_0 | p_1 p_0) = K_1
# 
# 这些有确定的比例。但笛卡尔 p 轨道和球谐 p 轨道不同。
# 
# 让我们换一种方式：用数值积分来验证？
# 不，那样太慢了。

# 让我计算一些基本的积分，看看它们是否满足自洽性
print(f"\n" + "=" * 60)
print("p 轨道积分自洽性检查")
print("=" * 60)

pxpx_ssss = _eri_full_os(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                         alpha, alpha, alpha, alpha, r, r, r, r)
pxs_pxss = _eri_full_os(1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                        alpha, alpha, alpha, alpha, r, r, r, r)
pxpx_pxpx = _eri_full_os(1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                          alpha, alpha, alpha, alpha, r, r, r, r)
pxpy_pxpy = _eri_full_os(1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
                          alpha, alpha, alpha, alpha, r, r, r, r)
pxpx_pypy = _eri_full_os(1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                          alpha, alpha, alpha, alpha, r, r, r, r)

print(f"(p_x p_x | s s) = {pxpx_ssss:.10f}")
print(f"(p_x s | p_x s) = {pxs_pxss:.10f}")
print(f"(p_x p_x | p_x p_x) = {pxpx_pxpx:.10f}")
print(f"(p_x p_y | p_x p_y) = {pxpy_pxpy:.10f}")
print(f"(p_x p_x | p_y p_y) = {pxpx_pypy:.10f}")

# 验证 5: (p_x p_x | s s) 和 (p_x s | p_x s) 的关系
# 从 VRR, 我们可以从 (p_x s | p_x s) 递推出 (p_x p_x | p_x s) 等等
# 但让我们先验证一个更简单的关系:
# 
# 对于 bra 侧，(p_x p_x | s s) = (s p_x | p_x s)? 不对，对称性不同
# 
# 实际上，(ab|cd) = (ba|dc) 吗？
# (p_x s | p_x s) 和 (s p_x | s p_x) 应该相等
val1 = _eri_full_os(1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                    alpha, alpha, alpha, alpha, r, r, r, r)
val2 = _eri_full_os(0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                    alpha, alpha, alpha, alpha, r, r, r, r)
print(f"\n对称性检查:")
print(f"  (p_x s | p_x s) = {val1:.10f}")
print(f"  (s p_x | s p_x) = {val2:.10f}")
print(f"  相等: {'✓' if abs(val1 - val2) < 1e-10 else '✗'}")

# 验证 6: 更重要的验证 - (p_x p_x | p_x p_x) 应该可以从 (p_x s | p_x p_x) 递推得到
# 让我们用递推公式手动计算 (p_x p_x | p_x p_x)
# 从 (p_x s | p_x p_x) 开始，增加 bra 侧第二个函数的 x 角动量
#
# (p_x p_x | p_x p_x) = (a b+1_x | c d) 其中 a=p_x, b=s, c=p_x, d=p_x
#
# PB_x = 0
# a_x = 1 (a是p_x)
# b_x = 0 (b是s)
#
# 不对，HRR？不，VRR 可以增加任意一侧的任意函数。
# 让我们用 VRR 增加 bra 侧第二个函数的角动量：
# (a b+1_x | c d) = PB_x (ab|cd) 
#   + a_x/(2ζ_ab) (a-1_x b|cd) + b_x/(2ζ_ab) (a b-1_x|cd)
#   + (ζ_cd/ζ) PQ_x (ab|cd)
#   + c_x/(2ζ) (ab|c-1_x d) + d_x/(2ζ) (ab|cd-1_x)
#
# 其中 PB_x = P - B = 0 (同中心)
#
# 所以:
# (p_x p_x | p_x p_x) = 0 * (p_x s | p_x p_x)
#   + 1/(2ζ_ab) * (s s | p_x p_x) + 0
#   + (ζ_cd/ζ) * 0 * (p_x s | p_x p_x)
#   + 1/(2ζ) * (p_x s | s p_x) + 1/(2ζ) * (p_x s | p_x s)
#
# 等等，c_x = 1 (c是p_x), d_x = 1 (d是p_x)
# 所以 term4_p2 = c_x/(2ζ) * (p_x s | s p_x) = 1/(2ζ) * (p_x s | s p_x)
# term4_p3 = d_x/(2ζ) * (p_x s | p_x s) = 1/(2ζ) * (p_x s | p_x s)
#
# 但是 (p_x s | s p_x) = (s p_x | p_x s) = (p_x s | p_x s) 吗？
# 不对，(ab|cd) 的置换对称性是:
# (ab|cd) = (ba|dc) = (cd|ab) = (dc|ba)
# 所以 (p_x s | s p_x) = (s p_x | p_x s) = (p_x s | p_x s)
# 是的，它们是相等的！
#
# 所以 term4_p2 + term4_p3 = 2/(2ζ) * (p_x s | p_x s) = 1/ζ * (p_x s | p_x s)
#
# 还有 term2 = 1/(2ζ_ab) * (s s | p_x p_x)
# 而 (s s | p_x p_x) = (p_x p_x | s s) = pxpx_ssss
#
# 所以:
# (p_x p_x | p_x p_x) = 1/(2ζ_ab) * (p_x p_x | s s) + 1/ζ * (p_x s | p_x s)

expected_pxpx_pxpx = (1.0/(2.0 * zeta_ab)) * pxpx_ssss + (1.0 / zeta) * pxs_pxss
print(f"\n(p_x p_x | p_x p_x) 递推验证:")
print(f"  计算值: {pxpx_pxpx:.10f}")
print(f"  递推值: {expected_pxpx_pxpx:.10f}")
print(f"  差值: {pxpx_pxpx - expected_pxpx_pxpx:.2e}")
print(f"  验证: {'✓' if abs(pxpx_pxpx - expected_pxpx_pxpx) < 1e-8 else '✗'}")

print()
print("如果递推验证通过，说明递推关系本身是自洽的。")
print("问题可能在于积分的绝对值是否正确。")
print()
print("让我们与一个已知的参考比较:")
print("  对于 alpha=1.0 的 (ss|ss) =", ssss)
print("  这个值对吗？让我们用解析公式验证。")
print()
print("解析公式: (ss|ss) = 2 pi^(5/2) / (2*2*sqrt(4))")
print("          = 2 pi^2.5 / 8 = pi^2.5 / 4")
import math
analytical = math.pi**2.5 / 4.0
print(f"  解析值: {analytical:.10f}")
print(f"  计算值: {ssss:.10f}")
print(f"  验证: {'✓' if abs(ssss - analytical) < 1e-8 else '✗'}")
