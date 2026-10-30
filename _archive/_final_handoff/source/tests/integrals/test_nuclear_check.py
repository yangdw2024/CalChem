"""
验证p轨道的核吸引积分。
"""
import sys
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
import math
from calchem.basis.gto import GTOShell, GTOPrimitive
from calchem.integrals.one_electron import _os_recurrence_nuclear, _os_recurrence_overlap

print("=" * 70)
print("s 和 p 轨道核吸引积分验证")
print("=" * 70)
print()

alpha = 1.0
origin = np.zeros(3)
nuc_pos = np.zeros(3)  # 核在原点
Z = 1.0

# s 轨道
s_prim = GTOPrimitive(alpha, 0, 0, 0, origin)
s_nuc = _os_recurrence_nuclear(0,0,0, 0,0,0, alpha, alpha, origin, origin, nuc_pos, Z)

print(f"【s 轨道 (α={alpha}, Z={Z})】")
print(f"  未归一化核吸引: {s_nuc:.10f}")
print(f"  归一化后核吸引: {s_prim.norm**2 * s_nuc:.10f}")
# 解析: <s|V|s> = -2Z * sqrt(2*alpha/pi)
s_nuc_analytic = -2 * Z * math.sqrt(2 * alpha / math.pi)
print(f"  解析值: {s_nuc_analytic:.10f}")
print(f"  差值: {abs(s_prim.norm**2 * s_nuc - s_nuc_analytic):.2e}")
print()

# p_x 轨道
p_prim = GTOPrimitive(alpha, 1, 0, 0, origin)
p_nuc = _os_recurrence_nuclear(1,0,0, 1,0,0, alpha, alpha, origin, origin, nuc_pos, Z)

print(f"【p_x 轨道 (α={alpha}, Z={Z})】")
print(f"  未归一化核吸引: {p_nuc:.10f}")
print(f"  归一化后核吸引: {p_prim.norm**2 * p_nuc:.10f}")
# 解析: <p|V|p> = (2/3) * <s|V|s>  (同中心同指数)
p_nuc_analytic = (2.0/3.0) * s_nuc_analytic
print(f"  解析预期 (2/3 * <s|V|s>): {p_nuc_analytic:.10f}")
print(f"  差值: {abs(p_prim.norm**2 * p_nuc - p_nuc_analytic):.2e}")
print()

# 比值
s_val = s_prim.norm**2 * s_nuc
p_val = p_prim.norm**2 * p_nuc
print(f"【比值】")
print(f"  <p|V|p> / <s|V|s> = {p_val / s_val:.10f}")
print(f"  预期值 (2/3) = {2/3:.10f}")
print()

# 验证其他p分量
py_nuc = _os_recurrence_nuclear(0,1,0, 0,1,0, alpha, alpha, origin, origin, nuc_pos, Z)
pz_nuc = _os_recurrence_nuclear(0,0,1, 0,0,1, alpha, alpha, origin, origin, nuc_pos, Z)
print(f"【各p分量的核吸引（应该相等）】")
print(f"  p_x: {p_prim.norm**2 * p_nuc:.10f}")
print(f"  p_y: {p_prim.norm**2 * py_nuc:.10f}")
print(f"  p_z: {p_prim.norm**2 * pz_nuc:.10f}")
print(f"  最大差值: {max(abs(p_nuc-py_nuc), abs(p_nuc-pz_nuc)):.2e}")
