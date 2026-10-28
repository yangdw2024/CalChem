"""
系统验证双电子积分的正确性。
对于同中心的情况，我们可以用已知的解析比值来验证。
"""
import numpy as np
import math
from scipy.integrate import quad

def boys(n, x):
    """Boys function."""
    if x < 1e-10:
        return 1.0 / (2*n + 1)
    result = 0.0
    for k in range(100):
        term = (x**k) / math.factorial(k) / (2*n + 2*k + 1)
        result += term
        if abs(term) < 1e-15:
            break
    return result * math.exp(-x)

def eri_ssss(alpha_a, alpha_b, alpha_c, alpha_d, r_a, r_b, r_c, r_d):
    """(ss|ss) 解析积分."""
    zeta_ab = alpha_a + alpha_b
    zeta_cd = alpha_c + alpha_d
    zeta = zeta_ab + zeta_cd
    
    r_p = (alpha_a * r_a + alpha_b * r_b) / zeta_ab
    r_q = (alpha_c * r_c + alpha_d * r_d) / zeta_cd
    
    r_ab2 = np.sum((r_a - r_b)**2)
    r_cd2 = np.sum((r_c - r_d)**2)
    r_pq2 = np.sum((r_p - r_q)**2)
    
    K_ab = math.exp(-alpha_a * alpha_b / zeta_ab * r_ab2)
    K_cd = math.exp(-alpha_c * alpha_d / zeta_cd * r_cd2)
    
    t = zeta_ab * zeta_cd / zeta * r_pq2
    F0 = boys(0, t)
    
    prefactor = 2 * math.pi**2.5 / (zeta_ab * zeta_cd * math.sqrt(zeta))
    return prefactor * K_ab * K_cd * F0


def coulomb_numerical_radial(f, g):
    """球对称库仑积分的数值计算。
    
    ∫∫ f(r1) g(r2) / |r1-r2| d^3r1 d^3r2
    = 16π² ∫_0^∞ r1² f(r1) [∫_0^{r1} r2² g(r2) / r1 dr2 + ∫_{r1}^∞ r2² g(r2) / r2 dr2] dr1
    """
    def inner(r1):
        part1, _ = quad(lambda r2: r2**2 * g(r2) / r1, 0, r1)
        part2, _ = quad(lambda r2: r2**2 * g(r2) / r2, r1, np.inf)
        return part1 + part2
    
    def integrand(r1):
        return 16 * math.pi**2 * r1**2 * f(r1) * inner(r1)
    
    result, _ = quad(integrand, 0, 50)
    return result


print("=" * 60)
print("双电子积分解验证")
print("=" * 60)
print()

alpha = 1.0
origin = np.zeros(3)

from calchem.integrals.two_electron import _ERIRecursion

rec = _ERIRecursion(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

ssss = rec.get(0,0,0, 0,0,0, 0,0,0, 0,0,0)
ssss_ref = eri_ssss(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

print("1. (ss|ss) 积分:")
print(f"   VRR:      {ssss:.12f}")
print(f"   解析:     {ssss_ref:.12f}")
print(f"   比值:     {ssss/ssss_ref:.12f}")
print()

# 数值验证
f_s = lambda r: math.exp(-2*alpha*r**2)
ssss_num = coulomb_numerical_radial(f_s, f_s)
print(f"   数值积分: {ssss_num:.12f}")
print(f"   VRR/数值: {ssss/ssss_num:.12f}")
print()

# 2. (p_x p_x | p_x p_x) 用数值积分验证
# 未归一化: (p_x p_x) 的电荷密度是 x^2 exp(-2αr^2)
# 球对称平均: (1/3) r² exp(-2αr^2)
# 总库仑积分 = ∫∫ [x1² exp(-2αr1²)] [x2² exp(-2αr2²)] / r12 d^3r1 d^3r2

# 利用角向分解，我们知道:
# (p_x p_x | p_x p_x) = (7/45) * r^4 项 + ...
# 不对，让我直接算

# 我们可以用多种方法验证:
# 方法1: 所有 p 轨道的总和应该等于球对称的 r² 分布的库仑积分
# sum_{ij} (p_i p_i | p_j p_j) = ∫∫ r1² exp(-2αr1²) r2² exp(-2αr2²) / r12 d^3r1 d^3r2

print("2. p-p 总库仑积分 (球对称验证):")

f_r2 = lambda r: r**2 * math.exp(-2*alpha*r**2)
pp_total_num = coulomb_numerical_radial(f_r2, f_r2)
print(f"   数值积分 (Σ_i p_i²|Σ_j p_j²): {pp_total_num:.12f}")

# 计算 VRR 的总和
pp_total_vrr = 0.0
for i in range(3):
    for j in range(3):
        li, mi, ni = 0, 0, 0
        lj, mj, nj = 0, 0, 0
        if i == 0: li = 1
        elif i == 1: mi = 1
        else: ni = 1
        if j == 0: lj = 1
        elif j == 1: mj = 1
        else: nj = 1
        val = rec.get(li, mi, ni, li, mi, ni, lj, mj, nj, lj, mj, nj)
        pp_total_vrr += val

print(f"   VRR 总和:                   {pp_total_vrr:.12f}")
print(f"   比值:                       {pp_total_vrr/pp_total_num:.12f}")
print()

# 3. 验证 (p_x p_x | p_x p_x) / (ss|ss) 的比值
# 对于归一化的基函数，这个比值应该是 11/10 = 1.1
# 对于未归一化的，需要考虑归一化常数

print("3. 归一化 p 轨道积分比值验证:")

from calchem.basis.gto import GTOShell
from calchem.integrals.two_electron import shell_eri

p_shell = GTOShell('p', [alpha], [1.0], origin=origin)
s_shell = GTOShell('s', [alpha], [1.0], origin=origin)

pp_blk = shell_eri(p_shell, p_shell, p_shell, p_shell)
ss_blk = shell_eri(s_shell, s_shell, s_shell, s_shell)

print(f"   归一化 (ss|ss):              {ss_blk[0,0,0,0]:.12f}")
print(f"   归一化 (p_x p_x|p_x p_x):    {pp_blk[0,0,0,0]:.12f}")
print(f"   比值:                        {pp_blk[0,0,0,0]/ss_blk[0,0,0,0]:.12f}")
print(f"   预期值:                      1.1")
print()

# 4. 验证各种 d 轨道积分之间的关系
print("4. d-d 总库仑积分 (球对称验证):")

f_r4 = lambda r: r**4 * math.exp(-2*alpha*r**2)
dd_total_num = coulomb_numerical_radial(f_r4, f_r4)
print(f"   数值积分 (Σ_i d_i²|Σ_j d_j²): {dd_total_num:.12f}")

# 计算 VRR 的总和 (d_i² | d_j²) 的总和
dd_total_vrr = 0.0
d_indices = [(2,0,0), (0,2,0), (0,0,2), (1,1,0), (1,0,1), (0,1,1)]
for i, (li, mi, ni) in enumerate(d_indices):
    for j, (lj, mj, nj) in enumerate(d_indices):
        val = rec.get(li, mi, ni, 0,0,0, lj, mj, nj, 0,0,0)
        dd_total_vrr += val

print(f"   VRR 总和 (d_i² s|d_j² s):    {dd_total_vrr:.12f}")
print(f"   比值:                        {dd_total_vrr/dd_total_num:.12f}")
print()

# 5. 详细打印各种 ERI 分量
print("5. 详细 ERI 分量 (未归一化):")
print()
print(f"   (ss|ss) = {ssss:.10f}")
print()

print("   (p_x s|p_x s) type:")
for i, name in enumerate(['x', 'y', 'z']):
    li, mi, ni = 0, 0, 0
    if i == 0: li = 1
    elif i == 1: mi = 1
    else: ni = 1
    val = rec.get(li, mi, ni, 0,0,0, li, mi, ni, 0,0,0)
    print(f"     (p_{name} s|p_{name} s) = {val:.10f}")
print()

print("   (p_x p_x|p_y p_y) type (通过 HRR 得到):")
for i, name_i in enumerate(['x', 'y', 'z']):
    for j, name_j in enumerate(['x', 'y', 'z']):
        li, mi, ni = 0, 0, 0
        lj, mj, nj = 0, 0, 0
        if i == 0: li = 1
        elif i == 1: mi = 1
        else: ni = 1
        if j == 0: lj = 1
        elif j == 1: mj = 1
        else: nj = 1
        val = rec.get(li, mi, ni, li, mi, ni, lj, mj, nj, lj, mj, nj)
        print(f"     (p_{name_i} p_{name_i}|p_{name_j} p_{name_j}) = {val:.10f}")
print()

print("6. d 轨道 ERI 分量 (未归一化, (d_i² s|d_j² s)):")
d_names = ['d_x2', 'd_y2', 'd_z2', 'd_xy', 'd_xz', 'd_yz']
for i, (li, mi, ni) in enumerate(d_indices):
    for j, (lj, mj, nj) in enumerate(d_indices):
        val = rec.get(li, mi, ni, 0,0,0, lj, mj, nj, 0,0,0)
        print(f"   ({d_names[i]}|{d_names[j]}) = {val:.10f}")
