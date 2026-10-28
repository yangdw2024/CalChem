"""调试双电子积分VRR递推的对称性问题。"""
import numpy as np

alpha = 1.0
origin = np.zeros(3)

from calchem.integrals.two_electron import _ERIRecursion

rec = _ERIRecursion(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

# 手动检查各阶积分
print('检查 (a 0 | c 0) 各阶积分:')
print()

# ssss
for n in range(6):
    val = rec._get_ssss_n(n)
    print(f'(0,0|0,0)_{n} = {val:.12f}')
print()

# 1阶
print('1阶角动量:')
for n in range(5):
    val = rec._get_n(1,0,0, 0,0,0, 0,0,0, 0,0,0, n)
    print(f'  (1,0|0,0)_{n} = {val:.12f}')
print()

# 2阶
print('2阶角动量:')
for n in range(4):
    v200 = rec._get_n(2,0,0, 0,0,0, 0,0,0, 0,0,0, n)
    v020 = rec._get_n(0,2,0, 0,0,0, 0,0,0, 0,0,0, n)
    v110 = rec._get_n(1,1,0, 0,0,0, 0,0,0, 0,0,0, n)
    v101 = rec._get_n(1,0,1, 0,0,0, 0,0,0, 0,0,0, n)
    print(f'  (2,0|0,0)_{n} = {v200:.12f}')
    print(f'  (0,2|0,0)_{n} = {v020:.12f}')
    print(f'  (1,1|0,0)_{n} = {v110:.12f}')
    print(f'  2,0 == 0,2? {np.isclose(v200, v020)}')
    print()

# 2-2阶
print('2-2阶角动量:')
for n in range(3):
    v20_20 = rec._get_n(2,0,0, 0,0,0, 2,0,0, 0,0,0, n)
    v20_02 = rec._get_n(2,0,0, 0,0,0, 0,2,0, 0,0,0, n)
    v02_20 = rec._get_n(0,2,0, 0,0,0, 2,0,0, 0,0,0, n)
    v02_02 = rec._get_n(0,2,0, 0,0,0, 0,2,0, 0,0,0, n)
    print(f'  n = {n}')
    print(f'    (2,0|2,0) = {v20_20:.12f}')
    print(f'    (2,0|0,2) = {v20_02:.12f}')
    print(f'    (0,2|2,0) = {v02_20:.12f}')
    print(f'    (0,2|0,2) = {v02_02:.12f}')
    print(f'    (2,0|0,2) == (0,2|2,0)? {np.isclose(v20_02, v02_20)}')
    print(f'    (2,0|2,0) == (0,2|0,2)? {np.isclose(v20_20, v02_02)}')
    print()

# 让我们手动检查一下递推的正确性
# 用解析公式验证
print('='*60)
print('解析验证:')
print()

import math

def boys(n, x):
    if x < 1e-10:
        return 1.0 / (2*n + 1)
    result = 0.0
    for k in range(200):
        term = (x**k) / math.factorial(k) / (2*n + 2*k + 1)
        result += term
        if abs(term) < 1e-15:
            break
    return result * math.exp(-x)

zeta_ab = 2 * alpha
zeta_cd = 2 * alpha
zeta = zeta_ab + zeta_cd

# (ss|ss)_n
def ssss_n(n):
    prefactor = 2 * math.pi**2.5 / (zeta_ab * zeta_cd * math.sqrt(zeta))
    return prefactor * boys(n, 0.0)

print('(0,0|0,0)_n 解析值:')
for n in range(6):
    print(f'  n={n}: {ssss_n(n):.12f}')
print()

# 验证 (2,0|0,0)_n 应该等于 ssss_n(n) / (2*zeta_ab)
print('(2,0|0,0)_n:')
for n in range(4):
    val_vrr = rec._get_n(2,0,0, 0,0,0, 0,0,0, 0,0,0, n)
    val_ref = ssss_n(n) / (2 * zeta_ab)
    print(f'  n={n}: VRR={val_vrr:.12f}, 解析={val_ref:.12f}, 比值={val_vrr/val_ref:.12f}')
print()

# 验证 (1,0|1,0)_n
# 解析: (1,0|1,0)_n = ssss_{n+1}(n) / (2*zeta)
print('(1,0|1,0)_n:')
for n in range(4):
    val_vrr = rec._get_n(1,0,0, 0,0,0, 1,0,0, 0,0,0, n)
    val_ref = ssss_n(n+1) / (2 * zeta)
    print(f'  n={n}: VRR={val_vrr:.12f}, 解析={val_ref:.12f}, 比值={val_vrr/val_ref:.12f}')
print()

# 验证 (2,0|2,0)_0
# 解析: S_0/(4ζ_abζ_cd) + S_2/(2ζ²)
print('(2,0|2,0)_0:')
val_vrr = rec._get_n(2,0,0, 0,0,0, 2,0,0, 0,0,0, 0)
val_ref = ssss_n(0) / (4*zeta_ab*zeta_cd) + ssss_n(2) / (2*zeta**2)
print(f'  VRR:   {val_vrr:.12f}')
print(f'  解析:  {val_ref:.12f}')
print(f'  比值:  {val_vrr/val_ref:.12f}')
print()

# 验证 (2,0|0,2)_0
# 解析: S_0/(4ζ_abζ_cd)
print('(2,0|0,2)_0:')
val_vrr = rec._get_n(2,0,0, 0,0,0, 0,2,0, 0,0,0, 0)
val_ref = ssss_n(0) / (4*zeta_ab*zeta_cd)
print(f'  VRR:   {val_vrr:.12f}')
print(f'  解析:  {val_ref:.12f}')
print(f'  比值:  {val_vrr/val_ref:.12f}')
print()

print('(0,2|2,0)_0:')
val_vrr = rec._get_n(0,2,0, 0,0,0, 2,0,0, 0,0,0, 0)
val_ref = ssss_n(0) / (4*zeta_ab*zeta_cd)
print(f'  VRR:   {val_vrr:.12f}')
print(f'  解析:  {val_ref:.12f}')
print(f'  比值:  {val_vrr/val_ref:.12f}')
