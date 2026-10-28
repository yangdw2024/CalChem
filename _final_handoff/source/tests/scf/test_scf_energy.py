"""
验证积分的基本一致性和球对称性，
然后测试 H2、Ne、H2O 的 SCF 能量。
"""
import numpy as np
import math
from calchem.integrals.two_electron import _ERIRecursionFixed

alpha = 1.0
origin = np.zeros(3)
rec = _ERIRecursionFixed(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

print("=" * 70)
print("双电子积分一致性验证（同中心）")
print("=" * 70)

# 1. p 轨道的球对称
print("\n1. p 轨道对角元的球对称:")
val_xx = rec.get(1,0,0, 1,0,0, 1,0,0, 1,0,0)  # (p_x p_x | p_x p_x)
val_yy = rec.get(0,1,0, 0,1,0, 0,1,0, 0,1,0)  # (p_y p_y | p_y p_y)
val_zz = rec.get(0,0,1, 0,0,1, 0,0,1, 0,0,1)  # (p_z p_z | p_z p_z)
print(f"   (p_x p_x | p_x p_x) = {val_xx:.10f}")
print(f"   (p_y p_y | p_y p_y) = {val_yy:.10f}")
print(f"   (p_z p_z | p_z p_z) = {val_zz:.10f}")
print(f"   三者相等? {abs(val_xx-val_yy) < 1e-10 and abs(val_yy-val_zz) < 1e-10}  {'✓' if abs(val_xx-val_yy) < 1e-10 else '✗'}")

# 2. (p_x p_x | p_y p_y) 应该等于 (p_x p_x | p_z p_z)
print("\n2. 交叉对角元的球对称:")
val_xxyy = rec.get(1,0,0, 1,0,0, 0,1,0, 0,1,0)  # (p_x p_x | p_y p_y)
val_xxzz = rec.get(1,0,0, 1,0,0, 0,0,1, 0,0,1)  # (p_x p_x | p_z p_z)
val_yyzz = rec.get(0,1,0, 0,1,0, 0,0,1, 0,0,1)  # (p_y p_y | p_z p_z)
print(f"   (p_x p_x | p_y p_y) = {val_xxyy:.10f}")
print(f"   (p_x p_x | p_z p_z) = {val_xxzz:.10f}")
print(f"   (p_y p_y | p_z p_z) = {val_yyzz:.10f}")
print(f"   三者相等? {abs(val_xxyy-val_xxzz) < 1e-10 and abs(val_xxzz-val_yyzz) < 1e-10}  {'✓' if abs(val_xxyy-val_xxzz) < 1e-10 else '✗'}")

# 3. 球对称总和验证
print("\n3. 球对称总和 Σ_i Σ_j (p_i p_i | p_j p_j):")
sum_total = 0.0
for i in range(3):
    for j in range(3):
        li = [0,0,0]; li[i] = 1
        lj = [0,0,0]; lj[j] = 1
        val = rec.get(li[0],li[1],li[2], li[0],li[1],li[2], lj[0],lj[1],lj[2], lj[0],lj[1],lj[2])
        sum_total += val
print(f"   总和 = {sum_total:.10f}")

# 与 ∂²J/∂β² 比较
beta = 2 * alpha
J = lambda z1, z2: 2 * math.pi**2.5 / (z1 * z2 * math.sqrt(z1 + z2))
h = 1e-4
d2J = (J(beta+h, beta+h) - J(beta+h, beta-h) - J(beta-h, beta+h) + J(beta-h, beta-h)) / (4*h*h)
print(f"   ∂²J/∂β² = {d2J:.10f}")
print(f"   比值 = {sum_total / d2J:.10f}  {'✓' if abs(sum_total/d2J - 1) < 1e-4 else '✗'}")

# 4. (p_x p_y | p_x p_y) - 应该有确定值
print("\n4. 非对角轨道积分:")
val_xyxy = rec.get(1,0,0, 0,1,0, 1,0,0, 0,1,0)  # (p_x p_y | p_x p_y)
print(f"   (p_x p_y | p_x p_y) = {val_xyxy:.10f}")

# 5. 验证 (p_x p_x | p_x p_x) + (p_x p_x | p_y p_y) + (p_x p_x | p_z p_z) 关系
print("\n5. Σ_j (p_x p_x | p_j p_j):")
sum_j = val_xx + val_xxyy + val_xxzz
print(f"   = {val_xx:.10f} + {val_xxyy:.10f} + {val_xxzz:.10f}")
print(f"   = {sum_j:.10f}")

# 这应该等于 (p_x p_x | r2²) = -∂/∂β2 (p_x p_x | ss)
# 让我们用有限差分验证
beta2 = 2 * alpha
h2 = 1e-5
# (p_x p_x | ss) = 1/3 * (-∂J/∂β1)
# 不对，让我们直接想
# Σ_j (p_j p_j | ss) = -∂J/∂β_cd
# 但 (p_x p_x | p_y p_y) 是二阶导数

# 6. d 轨道的球对称验证
print("\n6. d 轨道球对称 Σ_i (d_i s | d_i s)? 不，验证 Σ (d_ij | d_ij)")
# 验证 Σ_i (d_{ii} s | ss) = Σ_i (2e_i, 0 | 0, 0)
# 这应该等于什么？= (r², 0 | 0, 0) = -∂J/∂β_ab = 2.733...
sum_diag_d = 0.0
for i in range(3):
    li = [0,0,0]; li[i] = 2
    val = rec.get(li[0],li[1],li[2], 0,0,0, 0,0,0, 0,0,0)
    sum_diag_d += val
    print(f"   d_{['x','y','z'][i]}²: {val:.10f}")
print(f"   Σ d_ii = {sum_diag_d:.10f}")
dJ = -(J(beta+h, beta) - J(beta-h, beta)) / (2*h)
print(f"   -∂J/∂β_ab = {dJ:.10f}")
print(f"   比值 = {sum_diag_d / dJ:.10f}  {'✓' if abs(sum_diag_d/dJ - 1) < 1e-4 else '✗'}")

print("\n" + "=" * 70)

# ===== 现在测试 SCF 能量 =====
print("\n" + "=" * 70)
print("SCF 能量测试")
print("=" * 70)

from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF, UHF

# H2 测试 (键长 1.4 bohr = 0.7408 Å)
print("\n--- H2 STO-3G ---")
h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], units="bohr")
rhf_h2 = RHF(h2, basis="sto-3g", verbose=0)
e_h2 = rhf_h2.kernel()
print(f"RHF 能量: {e_h2:.8f} Eh")
print(f"参考值:  -1.11671433 Eh (Szabo)")
print(f"差值: {e_h2 - (-1.11671433):.6e}")

# Ne 原子测试
print("\n--- Ne 原子 STO-3G ---")
ne = Molecule(["Ne"], [[0.0, 0.0, 0.0]], units="bohr")
try:
    rhf_ne = RHF(ne, basis="sto-3g", verbose=1)
    e_ne = rhf_ne.kernel()
    print(f"RHF 能量: {e_ne:.8f} Eh")
    print(f"参考值:  -126.596986 Eh (Gaussian)")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

# H2O 测试
print("\n--- H2O STO-3G ---")
h2o = Molecule.from_xyz("""
3
H2O
O  0.000000  0.000000  0.119263
H  0.000000  0.763239 -0.477052
H  0.000000 -0.763239 -0.477052
""")
try:
    rhf_h2o = RHF(h2o, basis="sto-3g", verbose=1)
    e_h2o = rhf_h2o.kernel()
    print(f"RHF 能量: {e_h2o:.8f} Eh")
    print(f"参考值:  -74.942078 Eh (Szabo)")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
