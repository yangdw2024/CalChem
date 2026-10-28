"""
系统验证双电子积分的正确性：
1. 同中心 s, p, d 轨道积分与解析值比较
2. 球对称性验证
3. 置换对称性验证
"""
import numpy as np
import math
from scipy.integrate import quad

alpha = 1.0
beta = 2 * alpha  # ζ_ab = ζ_cd = 2α

from calchem.integrals.two_electron import _ERIRecursionFixed

origin = np.zeros(3)
rec = _ERIRecursionFixed(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

def J(z1, z2):
    """(ss|ss) 积分解析值。"""
    return 2 * math.pi**2.5 / (z1 * z2 * math.sqrt(z1 + z2))

print("=" * 70)
print("双电子积分验证（同中心, α=1.0）")
print("=" * 70)

# ========== 1. (ss|ss) ==========
print("\n1. (ss|ss):")
val = rec.get(0,0,0, 0,0,0, 0,0,0, 0,0,0)
ref = J(beta, beta)
print(f"   VRR: {val:.10f}")
print(f"   解析: {ref:.10f}")
print(f"   比值: {val/ref:.10f}  {'✓' if abs(val/ref - 1) < 1e-10 else '✗'}")

# ========== 2. (p_x p_x | ss) ==========
print("\n2. (p_x p_x | ss):")
h = 1e-5
dJ_dz1 = -(J(beta + h, beta) - J(beta - h, beta)) / (2 * h)
ref_pxss = dJ_dz1 / 3  # 1/3 of r^2 average
val = rec.get(1,0,0, 1,0,0, 0,0,0, 0,0,0)
print(f"   VRR: {val:.10f}")
print(f"   解析: {ref_pxss:.10f}")
print(f"   比值: {val/ref_pxss:.10f}  {'✓' if abs(val/ref_pxss - 1) < 1e-4 else '✗'}")

# ========== 3. 球对称验证 Σ (p_i p_i | ss) ==========
print("\n3. 球对称验证 Σ(p_i p_i | ss):")
val_px = rec.get(1,0,0, 1,0,0, 0,0,0, 0,0,0)
val_py = rec.get(0,1,0, 0,1,0, 0,0,0, 0,0,0)
val_pz = rec.get(0,0,1, 0,0,1, 0,0,0, 0,0,0)
sum_p = val_px + val_py + val_pz
ref_sum = dJ_dz1  # = -∂J/∂β1
print(f"   Σ = {sum_p:.10f}")
print(f"   解析: {ref_sum:.10f}")
print(f"   比值: {sum_p/ref_sum:.10f}  {'✓' if abs(sum_p/ref_sum - 1) < 1e-4 else '✗'}")

# ========== 4. (p_x p_x | p_x p_x) ==========
print("\n4. (p_x p_x | p_x p_x):")
h2 = 1e-4
d2J = (J(beta+h2, beta+h2) - J(beta+h2, beta-h2) - J(beta-h2, beta+h2) + J(beta-h2, beta-h2)) / (4*h2*h2)
# 球对称总和 ΣΣ (p_i p_i | p_j p_j) = ∂²J/∂β1∂β2
# (p_x p_x | p_x p_x) 的解析值需要角向分解
# 让我们用数值积分来算参考值

def radial_integral_asym(n1, n2, l, beta_val):
    def region1(r1):
        inner, _ = quad(lambda r2: r2**(2+n2) * r1**l / r2**(l+1) * math.exp(-beta_val*r2**2), r1, 20)
        return r1**(2+n1) * math.exp(-beta_val*r1**2) * inner
    def region2(r2):
        inner, _ = quad(lambda r1: r1**(2+n1) * r2**l / r1**(l+1) * math.exp(-beta_val*r1**2), r2, 20)
        return r2**(2+n2) * math.exp(-beta_val*r2**2) * inner
    reg1, _ = quad(region1, 0, 20)
    reg2, _ = quad(region2, 0, 20)
    return reg1 + reg2

def Y00(th, ph): return 1/math.sqrt(4*math.pi)
def Y20_real(th, ph): return math.sqrt(5 / (16*math.pi)) * (3*math.cos(th)**2 - 1)
def Y22c_real(th, ph): return math.sqrt(15 / (32*math.pi)) * math.sin(th)**2 * math.cos(2*ph)

def integrate_angular(f):
    nth, nph = 100, 200
    dth = math.pi / nth
    dph = 2*math.pi / nph
    res = 0.0
    for i in range(nth):
        th = (i + 0.5) * dth
        sth = math.sin(th)
        for j in range(nph):
            ph = (j + 0.5) * dph
            res += f(th, ph) * sth * dth * dph
    return res

f_xx = lambda th, ph: math.sin(th)**2 * math.cos(ph)**2
a00 = integrate_angular(lambda th, ph: f_xx(th, ph) * Y00(th, ph))
a20 = integrate_angular(lambda th, ph: f_xx(th, ph) * Y20_real(th, ph))
a22c = integrate_angular(lambda th, ph: f_xx(th, ph) * Y22c_real(th, ph))

R0_pp = radial_integral_asym(2, 2, 0, beta)
R2_pp = radial_integral_asym(2, 2, 2, beta)
l0_contrib = 4 * math.pi * a00**2 * R0_pp
l2_contrib = (4 * math.pi / 5) * (a20**2 + a22c**2) * R2_pp
ref_pxpx = l0_contrib + l2_contrib

val_pxpx = rec.get(1,0,0, 1,0,0, 1,0,0, 1,0,0)
print(f"   VRR: {val_pxpx:.10f}")
print(f"   解析: {ref_pxpx:.10f}")
print(f"   比值: {val_pxpx/ref_pxpx:.10f}  {'✓' if abs(val_pxpx/ref_pxpx - 1) < 1e-3 else '✗'}")

# ========== 5. 球对称验证 ΣΣ (p_i p_i | p_j p_j) ==========
print("\n5. 球对称验证 ΣΣ(p_i p_i | p_j p_j):")
sum_pp = 0.0
for i in range(3):
    for j in range(3):
        li = [0,0,0]; li[i] = 1
        lj = [0,0,0]; lj[j] = 1
        val_ij = rec.get(li[0],li[1],li[2], li[0],li[1],li[2], lj[0],lj[1],lj[2], lj[0],lj[1],lj[2])
        sum_pp += val_ij

ref_total_sph = 16 * math.pi**2 * radial_integral_asym(2, 2, 0, beta)
print(f"   ΣΣ = {sum_pp:.10f}")
print(f"   解析: {ref_total_sph:.10f}")
print(f"   比值: {sum_pp/ref_total_sph:.10f}  {'✓' if abs(sum_pp/ref_total_sph - 1) < 1e-3 else '✗'}")

# ========== 6. d 轨道验证 ==========
print("\n6. d 轨道 (d_x2 s | ss) = (p_x p_x | ss):")
val_dx2_ss = rec.get(2,0,0, 0,0,0, 0,0,0, 0,0,0)
print(f"   VRR (2,0,0|0,0,0): {val_dx2_ss:.10f}")
print(f"   解析 (1/3)(-∂J/∂β): {ref_pxss:.10f}")
print(f"   比值: {val_dx2_ss/ref_pxss:.10f}  {'✓' if abs(val_dx2_ss/ref_pxss - 1) < 1e-4 else '✗'}")

# ========== 7. 置换对称性验证 ==========
print("\n7. 置换对称性验证:")
# 选一个非平凡的情况：两个中心分开
rA = np.array([1.0, 0.0, 0.0])
rB = np.array([0.0, 0.0, 0.0])
rC = np.array([0.5, 0.5, 0.0])
rD = np.array([0.0, 0.3, 0.2])
rec2 = _ERIRecursionFixed(1.2, 0.9, 1.1, 0.8, rA, rB, rC, rD)

# (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = ...
v1 = rec2.get(1,0,0, 0,1,0, 0,0,1, 1,0,0)
v2 = rec2.get(0,1,0, 1,0,0, 0,0,1, 1,0,0)  # swap a,b
v3 = rec2.get(1,0,0, 0,1,0, 1,0,0, 0,0,1)  # swap c,d
v4 = rec2.get(0,0,1, 1,0,0, 1,0,0, 0,1,0)  # swap bra,ket

print(f"   (p_x s, p_y s | p_z s, p_x s) = {v1:.12f}")
print(f"   交换 bra: {v2:.12f}  差: {abs(v1-v2):.2e}  {'✓' if abs(v1-v2) < 1e-10 else '✗'}")
print(f"   交换 ket: {v3:.12f}  差: {abs(v1-v3):.2e}  {'✓' if abs(v1-v3) < 1e-10 else '✗'}")
print(f"   交换 bra/ket: {v4:.12f}  差: {abs(v1-v4):.2e}  {'✓' if abs(v1-v4) < 1e-10 else '✗'}")

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)
