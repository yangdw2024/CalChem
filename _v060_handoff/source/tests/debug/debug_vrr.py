"""
找出 VRR 的问题。关键发现：径向积分的对称性假设错误。
"""
import numpy as np
import math
from scipy.integrate import quad

alpha = 1.0
beta = 2 * alpha

# 直接数值计算 ∫∫ r1² exp(-βr1²) exp(-βr2²) / r12 d^3r1 d^3r2
# 用球坐标 + 角向积分结果 ∫∫ 1/r12 dΩ1 dΩ2 = 16π² / r_>

def integral_r1_sq(beta):
    """计算 ∫∫ r1² exp(-βr1²-βr2²)/r12 d^3r1 d^3r2"""
    
    def region1(r1):
        # r1 < r2, 1/r_> = 1/r2
        inner, _ = quad(lambda r2: r2**2 / r2 * math.exp(-beta * r2**2), r1, 20)
        return r1**2 * r1**2 * math.exp(-beta * r1**2) * inner
    
    def region2(r2):
        # r1 > r2, 1/r_> = 1/r1
        inner, _ = quad(lambda r1: r1**2 * r1**2 / r1 * math.exp(-beta * r1**2), r2, 20)
        return r2**2 * math.exp(-beta * r2**2) * inner
    
    reg1, _ = quad(region1, 0, 20)
    reg2, _ = quad(region2, 0, 20)
    
    return 16 * math.pi**2 * (reg1 + reg2)

result = integral_r1_sq(beta)
print(f"∫∫ r1² exp(-βr1²-βr2²)/r12 d^3r1 d^3r2 = {result:.15f}")

# 用参数微分法验证
J = lambda b1, b2: 2 * math.pi**2.5 / (b1 * b2 * math.sqrt(b1 + b2))
h = 1e-5
dJ_dbeta1 = -(J(beta + h, beta) - J(beta - h, beta)) / (2 * h)
print(f"-∂J/∂β1 = {dJ_dbeta1:.15f}")
print(f"比值: {result / dJ_dbeta1:.15f}")
print()

# 现在计算 (p_x p_x | ss) = 1/3 * 这个值
print(f"(p_x p_x | ss) 解析值 = {result / 3:.15f}")

# 和 VRR 比较
from calchem.integrals.two_electron import _ERIRecursion
origin = np.zeros(3)
rec = _ERIRecursion(alpha, alpha, alpha, alpha, origin, origin, origin, origin)
vrr_val = rec.get(2,0,0, 0,0,0, 0,0,0, 0,0,0)
print(f"VRR (d_x2 s | ss) = {vrr_val:.15f}")
print(f"比值 VRR/解析 = {vrr_val / (result / 3):.15f}")
print()

# 现在计算 (p_x p_x | p_x p_x) 的解析值
# 用傅里叶变换的方法或者直接积分

# 让我们用球谐展开 + 正确的径向积分来计算
# bra: x², ket: x²
# 都有 l=0 和 l=2 分量

def Y00(th, ph): return 1/math.sqrt(4*math.pi)
def Y20(th, ph): return math.sqrt(5/(16*math.pi)) * (3*math.cos(th)**2 - 1)
def Y22c(th, ph): return math.sqrt(5/(32*math.pi)) * math.sin(th)**2 * math.cos(2*ph)  # 不对，让我用正确的

# 先计算展开系数 a_lm = ∫ (x²/r²) Y_lm dΩ
def integrate_angular(f):
    nth, nph = 200, 400
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

f_xx = lambda th, ph: math.sin(th)**2 * math.cos(ph)**2  # x²/r²

a00 = integrate_angular(lambda th, ph: f_xx(th, ph) * Y00(th, ph))
# 我们需要实球谐函数 Y20, Y22c
def Y20_real(th, ph):
    return math.sqrt(5 / (16*math.pi)) * (3*math.cos(th)**2 - 1)

def Y22c_real(th, ph):
    return math.sqrt(15 / (32*math.pi)) * math.sin(th)**2 * math.cos(2*ph)

def Y22s_real(th, ph):
    return math.sqrt(15 / (32*math.pi)) * math.sin(th)**2 * math.sin(2*ph)

a20 = integrate_angular(lambda th, ph: f_xx(th, ph) * Y20_real(th, ph))
a22c = integrate_angular(lambda th, ph: f_xx(th, ph) * Y22c_real(th, ph))
a22s = integrate_angular(lambda th, ph: f_xx(th, ph) * Y22s_real(th, ph))

print("x²/r² 的实球谐展开系数:")
print(f"  a_00 = {a00:.12f}")
print(f"  a_20 = {a20:.12f}")
print(f"  a_22c = {a22c:.12f}")
print(f"  a_22s = {a22s:.12f}")
print(f"  Σ |a_lm|² = {a00**2 + a20**2 + a22c**2 + a22s**2:.12f}")
print(f"  ∫ f² dΩ = {integrate_angular(lambda th, ph: f_xx(th,ph)**2):.12f}")
print()

# 现在计算径向积分
# 对于每个 l，我们需要计算 ∫_0^∞ ∫_0^∞ r1^(2+n1) r2^(2+n2) r_<^l/r_>^{l+1} exp(-βr1²-βr2²) dr1 dr2
# n1 = n2 = 2 (因为 x² ~ r²)

def radial_integral_asym(n1, n2, l, beta):
    """非对称径向积分，n1 是 bra 的 r 幂次，n2 是 ket 的。
    bra: r^n1, ket: r^n2
    """
    def region1(r1):
        # r1 < r2
        inner, _ = quad(lambda r2: r2**(2+n2) * r1**l / r2**(l+1) * math.exp(-beta*r2**2), r1, 20)
        return r1**(2+n1) * math.exp(-beta*r1**2) * inner
    
    def region2(r2):
        # r1 > r2
        inner, _ = quad(lambda r1: r1**(2+n1) * r2**l / r1**(l+1) * math.exp(-beta*r1**2), r2, 20)
        return r2**(2+n2) * math.exp(-beta*r2**2) * inner
    
    reg1, _ = quad(region1, 0, 20)
    reg2, _ = quad(region2, 0, 20)
    return reg1 + reg2

# 验证球对称情况 n1=n2=0 (ss|ss)
R0_ss = radial_integral_asym(0, 0, 0, beta)
print(f"验证 (ss|ss):")
print(f"  16π² * R_0 = {16*math.pi**2 * R0_ss:.15f}")
print(f"  J(β,β) = {J(beta,beta):.15f}")
print(f"  比值 = {16*math.pi**2 * R0_ss / J(beta,beta):.15f}")
print()

# 现在计算 (p_x p_x | ss)，n1=2, n2=0, l=0
R0_pxss = radial_integral_asym(2, 0, 0, beta)
print(f"(p_x p_x | ss) 的径向积分 R_0 = {R0_pxss:.15f}")
print(f"  角因子 16π²/3 * R_0 = {16*math.pi**2/3 * R0_pxss:.15f}")
print(f"  解析值 (1/3)*(-∂J/∂β1) = {dJ_dbeta1/3:.15f}")
print(f"  比值 = {16*math.pi**2/3 * R0_pxss / (dJ_dbeta1/3):.15f}")
print()

# 好的，现在计算 (p_x p_x | p_x p_x)
# bra: x², ket: x²
# n1 = n2 = 2
# l = 0 和 l = 2 都有贡献
# 
# 角向贡献：
# l=0: (4π/1) * a00 * a00 = 4π * a00²
# l=2: (4π/5) * (a20² + a22c² + a22s²)  （因为 m=0, ±1, ±2 共 5 个分量，但 x² 只有 m=0 和 m=±2c）

R0_pp = radial_integral_asym(2, 2, 0, beta)
R2_pp = radial_integral_asym(2, 2, 2, beta)

print(f"(p_x p_x | p_x p_x):")
print(f"  R_0 = {R0_pp:.15f}")
print(f"  R_2 = {R2_pp:.15f}")

l0_contrib = 4 * math.pi * a00**2 * R0_pp
l2_contrib = (4 * math.pi / 5) * (a20**2 + a22c**2 + a22s**2) * R2_pp

print(f"  l=0 贡献: {l0_contrib:.15f}")
print(f"  l=2 贡献: {l2_contrib:.15f}")
print(f"  总值 = {l0_contrib + l2_contrib:.15f}")
print(f"  VRR 值 = {rec.get(1,0,0, 1,0,0, 1,0,0, 1,0,0):.15f}")
print(f"  比值 VRR/解析 = {rec.get(1,0,0, 1,0,0, 1,0,0, 1,0,0) / (l0_contrib + l2_contrib):.15f}")
print()

# 验证总和 Σ (p_i p_i | p_j p_j) 应该等于 ∂²J/∂β1∂β2
# 总和对应的是 bra=r², ket=r²，都是球对称，所以只有 l=0
# 角因子 = 16π²
# n1 = n2 = 2

R0_total = radial_integral_asym(2, 2, 0, beta)
total_sph = 16 * math.pi**2 * R0_total

# 有限差分二阶导
d2J = (J(beta+h, beta+h) - J(beta+h, beta-h) - J(beta-h, beta+h) + J(beta-h, beta-h)) / (4*h*h)
print(f"Σ (p_i p_i | p_j p_j):")
print(f"  球对称积分 (16π² R_0) = {total_sph:.15f}")
print(f"  ∂²J/∂β1∂β2 (有限差分) = {d2J:.15f}")
print(f"  比值 = {total_sph / d2J:.15f}")
print()

# 现在让我们用 VRR 公式手动验证几个关键积分
# 看看递推是否正确

print("=" * 70)
print("手动 VRR 验证")
print("=" * 70)
print()

zeta_ab = 2 * alpha
zeta_cd = 2 * alpha
zeta = zeta_ab + zeta_cd

# (ssss)_n
def ssss_n(n):
    from calchem.integrals.two_electron import _eri_ssss_n
    return _eri_ssss_n(n, alpha, alpha, alpha, alpha, origin, origin, origin, origin)

# 用 VRR 公式计算 (2,0,0 | 0,0,0)
# 从 (0,0,0 | 0,0,0) 上升 a_x 两次
# 
# 第一次上升：(1,0,0 | 0,0,0) = 0 （正确）
# 第二次上升：(2,0,0 | 0,0,0)_n = PA_x * ... + (ζ_cd/ζ)PQ_x * ... + a_x/(2ζ_ab) (0,0,0|0,0,0)_n + c_x/(2ζ) (1,0,0|-1,0,0)_{n+1}
# = 0 + 0 + 1/(2ζ_ab) * (ssss)_n + 0
# = (ssss)_n / (2ζ_ab)
# 
# 但是等等，这个结果对吗？

manual_200_000 = ssss_n(0) / (2 * zeta_ab)
print(f"(2,0,0 | 0,0,0) VRR公式: {manual_200_000:.15f}")
print(f"(2,0,0 | 0,0,0) VRR实现: {rec.get(2,0,0, 0,0,0, 0,0,0, 0,0,0):.15f}")
print(f"解析值: {result / 3:.15f}")
print(f"VRR公式/解析 = {manual_200_000 / (result / 3):.15f}")
print()

# 现在让我们推导正确的 VRR 公式。
# Obara-Saika VRR 的标准公式（从文献中）：
# 
# (a+1_i, b | c, d) = PA_i (a, b | c, d) + a_i/(2ζ_ab) (a-1_i, b | c, d)
#                    + (ζ_cd/ζ) PQ_i (a, b | c, d)^1 + c_i/(2ζ) (a, b | c-1_i, d)^1
# 
# 其中 (...)^1 表示 Boys 函数阶数 +1。
# 
# 这和我们的公式一致。那为什么结果不对呢？
# 
# 让我检查一下 Obara-Saika 递推的初始条件。
# (0,0,0 | 0,0,0)_n 是什么？
# 应该是 (ss|ss) 但 Boys 函数阶数为 n。
# 
# 公式：
# (ss|ss)_n = 2π^(5/2) / (ζ_ab ζ_cd √(ζ_ab+ζ_cd)) * K_ab * K_cd * F_n(ζ_ab ζ_cd / ζ * PQ²)
# 
# 对于同中心，PQ=0，F_n(0) = 1/(2n+1)。
# 
# 所以 (ssss)_n = (ssss)_0 * F_n(0)/F_0(0) = (ssss)_0 / (2n+1)
# 
# 让我们验证一下：
print("(ssss)_n 验证:")
for n in range(4):
    val = ssss_n(n)
    expected = ssss_n(0) / (2*n + 1)
    print(f"  n={n}: {val:.12f}, 预期 {expected:.12f}, 比值 {val/expected:.12f}")
print()

# 这些是对的。
# 
# 那让我检查一下 VRR 公式的推导。
# 也许我们的 VRR 公式的符号或者系数有误。
# 
# 让我用另一种方法验证 (1,0,0 | 1,0,0)。
# 我们可以用两种方式计算：
# 1. 上升 a_x 再上升 c_x
# 2. 上升 c_x 再上升 a_x
# 结果应该相同（对称性）。

print("(1,0,0 | 1,0,0) 对称性验证:")
# 我们的 VRR 实现给出
print(f"  VRR: {rec.get(1,0,0, 0,0,0, 1,0,0, 0,0,0):.15f}")

# 手动计算
# 从 (0,0,0|0,0,0)_n 开始
# 上升 a_x: (1,0,0|0,0,0)_n = 0
# 然后上升 c_x: (1,0,0|1,0,0)_n = QC_x*(...) + (ζ_ab/ζ)QP_x*(...)^1 
#                             + c_x/(2ζ_cd)*(1,0,0|0,0,0)_n + a_x/(2ζ)*(0,0,0|0,0,0)_{n+1}
# = 0 + 0 + 0 + 1/(2ζ) * (ssss)_{n+1}
# = (ssss)_{n+1} / (2ζ)
# 
# 对于 n=0:
manual_100_100 = ssss_n(1) / (2 * zeta)
print(f"  手动: {manual_100_100:.15f}")
print(f"  比值: {rec.get(1,0,0, 0,0,0, 1,0,0, 0,0,0) / manual_100_100:.15f}")
print()

# 这个是对的。
# 
# 现在让我们计算 (2,0,0 | 0,0,0)，但用上升 a_x 两次。
# 我们已经算过了，得到 (ssss)_n / (2ζ_ab)。
# 
# 但解析结果是 (1/3) * (-∂J/∂β1) = 0.911。
# VRR 给出 1.093。
# 比值是 1.093 / 0.911 = 1.2 = 6/5。
# 
# 等等，让我重新检查解析值。
# (2,0,0 | 0,0,0) 对应的物理积分是什么？
# 
# bra: 角动量 (2,0,0) 中心 A + s 中心 B
# ket: s 中心 C + s 中心 D
# 
# 积分 = ∫∫ x1² exp(-αr1²) exp(-αr1²) / r12 exp(-αr2²) exp(-αr2²) d^3r1 d^3r2
# = ∫∫ x1² exp(-2αr1²) exp(-2αr2²) / r12 d^3r1 d^3r2
# 
# 而 -∂J/∂β1 是 ∫∫ r1² exp(-βr1²-βr2²)/r12 d^3r1 d^3r2，其中 β=2α。
# 
# 所以 (p_x p_x | ss) = (1/3) * (-∂J/∂β1)。
# 
# 我们已经验证了 -∂J/∂β1 = 2.7334（用有限差分）。
# 所以 (p_x p_x | ss) = 0.9111。
# 
# 但 VRR 给出 1.0933，比值 1.2 = 6/5。
# 
# 6/5... 这个数字很有意思。
# 让我想想，Boys 函数 F_n(0) = 1/(2n+1)。
# 
# 等等，也许 VRR 公式中的 Boys 函数阶数不对？
# 让我查一下 Obara-Saika 的原始论文。
# 
# 实际上，我记得 VRR 公式的正确形式是：
# (a+1_i, 0 | c, 0)_n = PA_i (a,c)_n + a_i/(2ζ_ab) (a-1_i, c)_n
#                      + (ζ_cd/ζ) PQ_i [(a,c)_n - (a,c)_{n+1}]
#                      + c_i/(2ζ) [(a,c-1_i)_n - (a,c-1_i)_{n+1}]
# 
# 不对，这是另一种形式。
# 
# 或者也许我们的 VRR 公式是正确的，但需要考虑 HRR 的影响？
# 不，(2,0,0 | 0,0,0) 是 VR 形式（b=0, d=0），不涉及 HRR。
# 
# 让我用另一种完全不同的方法验证：直接用高斯乘积定理和解析公式。
# 
# 对于两个高斯函数的乘积：
# exp(-αa r_A²) exp(-αb r_B²) = K_ab exp(-ζ_ab r_P²)
# 其中 ζ_ab = αa + αb, P = (αa A + αb B)/ζ_ab, K_ab = exp(-αaαb/ζ_ab |A-B|²)
# 
# 对于 d_x2 轨道，x² exp(-αa r_A²) * exp(-αb r_B²) = ?
# = x² K_ab exp(-ζ_ab r_P²)
# = (x_P + P_x)² K_ab exp(-ζ_ab r_P²)  （不对，r_A = r_P + (P - A)?）
# 
# 实际上，平移后的多项式比较复杂。
# 
# 让我换一个思路。也许我们的 VRR 是正确的，但我对积分的物理解释有误。
# 
# (a, b | c, d) 中的 a, b, c, d 是角动量量子数。
# 对应的高斯函数是 x^a_x y^a_y z^a_z exp(-α r²) 吗？
# 还是说它们是归一化的？
# 
# 不，在 Obara-Saika 递推中，高斯函数是未归一化的，形式为：
# φ_a(r - A) = (x - A_x)^{a_x} (y - A_y)^{a_y} (z - A_z)^{a_z} exp(-α |r - A|²)
# 
# 这和我们的 GTOPrimitive 一致吗？
# 让我检查一下。
