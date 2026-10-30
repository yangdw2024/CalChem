"""
验证 VRR 修正项的系数是否正确。

关键问题：修正项 -a_i/(2ζ) (a-e_i, c)_{n+1} 在 ζ_ab ≠ ζ_cd 时是否正确？

从解析推导：
(2e_i, 0 | 0, 0)_n = ssss_n/(2ζ_ab) - (ζ_cd/ζ) * ssss_{n+1}/(2ζ_ab)

标准 VRR 给出: ssss_n/(2ζ_ab)
缺少的项: -(ζ_cd/ζ) * ssss_{n+1}/(2ζ_ab) = -ζ_cd * ssss_{n+1} / (2ζ ζ_ab)

代码的修正项: -a_i * ssss_{n+1} / (2ζ) = -1 * ssss_{n+1} / (2ζ)

当 ζ_ab ≠ ζ_cd 时，这两个不一样！
正确的系数应该是: -a_i * ζ_cd / (2ζ ζ_ab)
"""
import numpy as np
import math

from calchem.integrals.two_electron import _ERIRecursionFixed
from calchem.integrals.one_electron import boys_function_vec

# 测试1: ζ_ab = ζ_cd 的情况（同基组同壳层）
print("=" * 60)
print("测试1: ζ_ab = ζ_cd (α=1.0)")
print("=" * 60)
alpha = 1.0
origin = np.zeros(3)
rec = _ERIRecursionFixed(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

# (2,0,0|0,0,0) 应该等于 ssss_0/(2ζ_ab) - (ζ_cd/ζ)*ssss_1/(2ζ_ab)
zeta_ab = 2 * alpha
zeta_cd = 2 * alpha
zeta = zeta_ab + zeta_cd

ssss_0 = rec._get_ssss_n(0)
ssss_1 = rec._get_ssss_n(1)

val_vrr = rec.get(2,0,0, 0,0,0, 0,0,0, 0,0,0)
val_correct = ssss_0/(2*zeta_ab) - (zeta_cd/zeta)*ssss_1/(2*zeta_ab)
val_code_correction = ssss_0/(2*zeta_ab) - ssss_1/(2*zeta)

print(f"  VRR 结果: {val_vrr:.10f}")
print(f"  正确值:   {val_correct:.10f}")
print(f"  代码修正: {val_code_correction:.10f}")
print(f"  VRR==正确? {abs(val_vrr - val_correct) < 1e-10}")
print(f"  VRR==代码修正? {abs(val_vrr - val_code_correction) < 1e-10}")

# 测试2: ζ_ab ≠ ζ_cd 的情况
print("\n" + "=" * 60)
print("测试2: ζ_ab ≠ ζ_cd (α_a=2.0, α_c=1.0)")
print("=" * 60)
alpha_a = 2.0
alpha_c = 1.0
rec2 = _ERIRecursionFixed(alpha_a, alpha_a, alpha_c, alpha_c, origin, origin, origin, origin)

zeta_ab = 2 * alpha_a  # = 4
zeta_cd = 2 * alpha_c  # = 2
zeta = zeta_ab + zeta_cd  # = 6

ssss_0 = rec2._get_ssss_n(0)
ssss_1 = rec2._get_ssss_n(1)

# (2,0,0|0,0,0) 的正确值
# 从解析推导: ssss_n/(2ζ_ab) - (ζ_cd/ζ)*ssss_{n+1}/(2ζ_ab)
val_correct = ssss_0/(2*zeta_ab) - (zeta_cd/zeta)*ssss_1/(2*zeta_ab)

# 代码给出的值
val_vrr = rec2.get(2,0,0, 0,0,0, 0,0,0, 0,0,0)

# 代码的修正项: -1/(2ζ) * ssss_1
val_code = ssss_0/(2*zeta_ab) - ssss_1/(2*zeta)

print(f"  ζ_ab = {zeta_ab}, ζ_cd = {zeta_cd}, ζ = {zeta}")
print(f"  ssss_0 = {ssss_0:.10f}")
print(f"  ssss_1 = {ssss_1:.10f}")
print(f"  VRR 结果: {val_vrr:.10f}")
print(f"  正确值:   {val_correct:.10f}  [系数: ζ_cd/(2ζζ_ab) = {zeta_cd/(2*zeta*zeta_ab):.6f}]")
print(f"  代码修正: {val_code:.10f}  [系数: 1/(2ζ) = {1/(2*zeta):.6f}]")
print(f"  VRR==正确? {abs(val_vrr - val_correct) < 1e-10}")
print(f"  VRR==代码修正? {abs(val_vrr - val_code) < 1e-10}")
print(f"  差值 VRR-正确: {val_vrr - val_correct:.10e}")

# 测试3: 更极端的差异
print("\n" + "=" * 60)
print("测试3: 极端差异 (α_a=10.0, α_c=1.0)")
print("=" * 60)
alpha_a = 10.0
alpha_c = 1.0
rec3 = _ERIRecursionFixed(alpha_a, alpha_a, alpha_c, alpha_c, origin, origin, origin, origin)

zeta_ab = 2 * alpha_a  # = 20
zeta_cd = 2 * alpha_c  # = 2
zeta = zeta_ab + zeta_cd  # = 22

ssss_0 = rec3._get_ssss_n(0)
ssss_1 = rec3._get_ssss_n(1)

val_correct = ssss_0/(2*zeta_ab) - (zeta_cd/zeta)*ssss_1/(2*zeta_ab)
val_vrr = rec3.get(2,0,0, 0,0,0, 0,0,0, 0,0,0)
val_code = ssss_0/(2*zeta_ab) - ssss_1/(2*zeta)

print(f"  ζ_ab = {zeta_ab}, ζ_cd = {zeta_cd}, ζ = {zeta}")
print(f"  VRR 结果: {val_vrr:.10f}")
print(f"  正确值:   {val_correct:.10f}")
print(f"  代码修正: {val_code:.10f}")
print(f"  差值 VRR-正确: {val_vrr - val_correct:.10e}")
print(f"  系数比(代码/正确): {(1/(2*zeta)) / (zeta_cd/(2*zeta*zeta_ab)):.6f}")
print(f"  应该是 ζ_ab/ζ_cd = {zeta_ab/zeta_cd:.6f}")
