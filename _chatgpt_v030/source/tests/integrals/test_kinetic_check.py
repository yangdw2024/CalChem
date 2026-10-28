"""
验证s和p轨道的动能积分解析公式。
"""
import sys
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
import math
from calchem.basis.gto import GTOShell, GTOPrimitive
from calchem.integrals.one_electron import _os_recurrence_kinetic, _os_recurrence_overlap

print("=" * 70)
print("s 和 p 轨道动能积分解析验证")
print("=" * 70)
print()

alpha = 1.0
origin = np.zeros(3)

# s 轨道
s_prim = GTOPrimitive(alpha, 0, 0, 0, origin)
s_overlap = _os_recurrence_overlap(0,0,0, 0,0,0, alpha, alpha, origin, origin)
s_kinetic = _os_recurrence_kinetic(0,0,0, 0,0,0, alpha, alpha, origin, origin)

print(f"【s 轨道 (α={alpha})】")
print(f"  未归一化重叠: {s_overlap:.10f}")
print(f"  未归一化动能: {s_kinetic:.10f}")
print(f"  归一化常数: {s_prim.norm:.10f}")
print(f"  归一化后重叠: {s_prim.norm**2 * s_overlap:.10f} (应该是 1.0)")
print(f"  归一化后动能: {s_prim.norm**2 * s_kinetic:.10f}")
print(f"  解析值 (3α/2): {3*alpha/2:.10f}")
print(f"  差值: {abs(s_prim.norm**2 * s_kinetic - 3*alpha/2):.2e}")
print()

# p_x 轨道
p_prim = GTOPrimitive(alpha, 1, 0, 0, origin)
p_overlap = _os_recurrence_overlap(1,0,0, 1,0,0, alpha, alpha, origin, origin)
p_kinetic = _os_recurrence_kinetic(1,0,0, 1,0,0, alpha, alpha, origin, origin)

print(f"【p_x 轨道 (α={alpha})】")
print(f"  未归一化重叠: {p_overlap:.10f}")
print(f"  未归一化动能: {p_kinetic:.10f}")
print(f"  归一化常数: {p_prim.norm:.10f}")
print(f"  归一化后重叠: {p_prim.norm**2 * p_overlap:.10f} (应该是 1.0)")
print(f"  归一化后动能: {p_prim.norm**2 * p_kinetic:.10f}")

# 解析计算归一化 p 轨道动能
# 先算未归一化的积分
# 重叠: ∫ x² exp(-2α r²) dr = π^(3/2) / (2^(7/2) α^(5/2))
p_overlap_analytic = math.pi**1.5 / (2**3.5 * alpha**2.5)
# 动能: 让我们用递推或者已知的关系
# 实际上，对于归一化的 p 轨道，动能应该是多少？
# 让我用 virial 或者其他关系...

# 换一种方式：让我计算 T_p / T_s 的比值
print()
print("【比值】")
t_s_norm = s_prim.norm**2 * s_kinetic
t_p_norm = p_prim.norm**2 * p_kinetic
print(f"  T_p / T_s = {t_p_norm / t_s_norm:.10f}")

# 让我推导一下这个比值应该是多少
# 对于归一化的 s 和 p 轨道，相同 α:
# 我们知道 T_s = 3α/2
# 那 T_p 呢？
# 
# 让我用另一种方法计算。动能也可以写成 <T> = 1/2 <∇φ·∇φ>
# 对于 φ_s = N_s exp(-α r²):
# ∇φ_s = -2α r φ_s
# <∇φ_s·∇φ_s> = 4α² <r²>_s
# 所以 T_s = 1/2 * 4α² <r²>_s = 2α² <r²>_s
# 而我们知道 T_s = 3α/2
# 所以 <r²>_s = (3α/2) / (2α²) = 3/(4α)
# 这是对的，因为对于高斯函数 <r²> = 3/(4α)

# 对于 φ_p = N_p x exp(-α r²):
# ∇φ_p = ?
# ∂φ_p/∂x = N_p [exp(-α r²) + x * (-2α x) exp(-α r²)]
#         = N_p (1 - 2α x²) exp(-α r²)
# ∂φ_p/∂y = N_p x * (-2α y) exp(-α r²)
#         = -2α N_p x y exp(-α r²)
# ∂φ_p/∂z = -2α N_p x z exp(-α r²)

# 所以 |∇φ_p|² = N_p² [(1 - 2α x²)² + (2α x y)² + (2α x z)²] exp(-2α r²)
# = N_p² [1 - 4α x² + 4α² x⁴ + 4α² x² y² + 4α² x² z²] exp(-2α r²)
# = N_p² [1 - 4α x² + 4α² x² (x² + y² + z²)] exp(-2α r²)
# = N_p² [1 - 4α x² + 4α² x² r²] exp(-2α r²)

# 所以 T_p = 1/2 ∫ |∇φ_p|² dr
# = N_p² / 2 ∫ [1 - 4α x² + 4α² x² r²] exp(-2α r²) dr

# 计算这三个积分：
# I0 = ∫ exp(-2α r²) dr = (π/(2α))^(3/2)
# I2 = ∫ x² exp(-2α r²) dr = π^(3/2) / (2^(7/2) α^(5/2))
# I4 = ∫ x² r² exp(-2α r²) dr = ?

# 我们之前算过 ∫ x² r² exp(-2α r²) dr = (5/4) π^(3/2) / (2α)^(7/2)
# 对吗？让我再确认一下：
# x² r² = x⁴ + x² y² + x² z²
# ∫ x⁴ exp(-2α r²) dr = (3/4) π^(3/2) / (2α)^(7/2)  （等一下，单位对吗？）
# 
# 实际上，让我用另一种方式：
# ∫ r² exp(-2α r²) dr = 3 ∫ x² exp(-2α r²) dr = 3 * I2
# 
# 但是我们需要的是 ∫ x² r² exp(...)
# 利用角对称，∫ x² r² = ∫ y² r² = ∫ z² r²
# 所以 ∫ x² r² = (1/3) ∫ r⁴ exp(...)
# 
# 而 ∫ r⁴ exp(-2α r²) dr = 4π ∫_0^∞ r⁴ * r² exp(-2α r²) dr
# = 4π ∫_0^∞ r⁶ exp(-2α r²) dr
# 
# ∫_0^∞ r^(2n+2) exp(-a r²) dr = (2n+1)!! / (2^(n+2) a^(n+1.5)) * √π
# 
# 对于 n=2 (r⁶):
# ∫_0^∞ r⁶ exp(-a r²) dr = 5!! / (2^4 a^3.5) * √π = 15 / (16 a^(7/2)) * √π
# 
# 所以 ∫ r⁴ exp(-2α r²) dr = 4π * 15 / (16 (2α)^(7/2)) * √π
# = 4π * 15 √π / (16 * (2α)^(7/2))
# = (60 π^(3/2)) / (16 * (2α)^(7/2))
# = (15 π^(3/2)) / (4 * (2α)^(7/2))
# 
# 所以 ∫ x² r² exp(-2α r²) dr = (1/3) * 15 π^(3/2) / (4 * (2α)^(7/2))
# = 5 π^(3/2) / (4 * (2α)^(7/2))
# = 5 π^(3/2) / (2² * 2^(7/2) α^(7/2))
# = 5 π^(3/2) / (2^(11/2) α^(7/2))

# 好的，现在计算 T_p:
# T_p = N_p² / 2 * [I0 - 4α I2 + 4α² I4]
# 
# N_p² = 1 / I2 = 2^(7/2) α^(5/2) / π^(3/2)
# 
# 代入：
# T_p = (2^(7/2) α^(5/2) / π^(3/2)) / 2 * [
#     (π/(2α))^(3/2) 
#     - 4α * π^(3/2) / (2^(7/2) α^(5/2))
#     + 4α² * 5 π^(3/2) / (2^(11/2) α^(7/2))
# ]
# 
# 化简每一项：
# 
# 第一项: (π/(2α))^(3/2) = π^(3/2) / (2^(3/2) α^(3/2))
# 第二项: -4α * π^(3/2) / (2^(7/2) α^(5/2)) = -4 π^(3/2) / (2^(7/2) α^(3/2))
# 第三项: 4α² * 5 π^(3/2) / (2^(11/2) α^(7/2)) = 20 π^(3/2) / (2^(11/2) α^(3/2))
# 
# 把三项加起来，都有 π^(3/2) / (2^(3/2) α^(3/2)):
# 第一项系数: 1
# 第二项系数: -4 / 2² = -4/4 = -1  （因为 2^(7/2) = 2^(3/2) * 2²）
# 第三项系数: 20 / 2^(8/2) = 20 / 2⁴ = 20 / 16 = 5/4
# 
# 所以总和 = π^(3/2) / (2^(3/2) α^(3/2)) * [1 - 1 + 5/4]
# = π^(3/2) / (2^(3/2) α^(3/2)) * 5/4
# = 5 π^(3/2) / (2^(3/2) * 4 α^(3/2))
# = 5 π^(3/2) / (2^(11/2) α^(3/2))
# 
# 然后乘以 N_p²/2:
# T_p = (2^(7/2) α^(5/2) / π^(3/2)) / 2 * 5 π^(3/2) / (2^(11/2) α^(3/2))
# = 2^(7/2) α^(5/2) / (2 π^(3/2)) * 5 π^(3/2) / (2^(11/2) α^(3/2))
# = 5 * 2^(7/2) α^(5/2) / (2 * 2^(11/2) α^(3/2))
# = 5 * α^(1) / (2 * 2^(4/2))
# = 5α / (2 * 4)
# = 5α / 8

# 等等，这个结果对吗？T_p = 5α/8？
# 但是 T_s = 3α/2 = 12α/8
# 所以 T_p / T_s = (5α/8) / (12α/8) = 5/12 ≈ 0.416666...

print(f"  解析预期 T_p / T_s = 5/12 ≈ 0.4166666667")
print()

# 等等，这可能不对。让我再想想。
# p轨道有三个分量，px, py, pz。
# 对于原子的p壳层，三个轨道的动能之和应该是多少？
# 
# 实际上，让我用另一个方法验证。
# 考虑一个单电子原子，核电荷 Z，用高斯基组变分。
# 不过这可能太复杂了。
# 
# 让我们先看看代码算出来的是什么。
print(f"代码给出的 T_p = {t_p_norm:.10f}")
print(f"代码给出的 T_s = {t_s_norm:.10f}")
print(f"代码给出的 T_p / T_s = {t_p_norm / t_s_norm:.10f}")
