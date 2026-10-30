"""
用解析方法验证双电子积分的正确性。
直接计算 (ss|ss), (p_x s|p_x s), (p_x p_x|ss), (p_x p_x|p_x p_x) 等的解析值。
"""
import numpy as np
import math
from scipy.integrate import quad


def boys(n, x):
    if x < 1e-10:
        return 1.0 / (2*n + 1)
    result = 0.0
    for k in range(100):
        term = (x**k) / math.factorial(k) / (2*n + 2*k + 1)
        result += term
        if abs(term) < 1e-15:
            break
    return result * math.exp(-x)


def eri_ssss_n(n, alpha, r_a, r_b, r_c, r_d):
    """计算 (ss|ss)_n。"""
    zeta_ab = alpha + alpha
    zeta_cd = alpha + alpha
    zeta = zeta_ab + zeta_cd
    
    r_p = (alpha * r_a + alpha * r_b) / zeta_ab
    r_q = (alpha * r_c + alpha * r_d) / zeta_cd
    
    r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
    r_cd2 = float(np.dot(r_c - r_d, r_c - r_d))
    r_pq2 = float(np.dot(r_p - r_q, r_p - r_q))
    
    K_ab = math.exp(-alpha * alpha / zeta_ab * r_ab2)
    K_cd = math.exp(-alpha * alpha / zeta_cd * r_cd2)
    
    t = zeta_ab * zeta_cd / zeta * r_pq2
    fn = boys(n, t)
    
    prefactor = 2.0 * math.pi ** 2.5 / (zeta_ab * zeta_cd * math.sqrt(zeta))
    return prefactor * K_ab * K_cd * fn


alpha = 1.0
origin = np.zeros(3)

print("=" * 70)
print("解析验证双电子积分")
print("=" * 70)
print()

# 1. (ss|ss)
ssss = eri_ssss_n(0, alpha, origin, origin, origin, origin)
print(f"1. (ss|ss):")
print(f"   解析值: {ssss:.15f}")
print()

# 2. 用参数微分法计算各种积分
# J(β1, β2) = ∫∫ exp(-β1 r1² - β2 r2²) / r12 d^3r1 d^3r2
# 对于同中心 GTO，bra 侧两个指数都是 α，所以 bra 电荷密度的指数是 2α。
# 即 J(β1, β2) 对应于 bra 电荷密度指数 β1/2 的两个 s GTO，ket 电荷密度指数 β2/2 的两个 s GTO。
# 
# 更直接地：对于 αa=αb=α，bra = exp(-αr²) * exp(-αr²) = exp(-2αr²)
# 所以 J(β) 中 β = 2α。

beta = 2 * alpha

def J_beta(beta1, beta2):
    """J(β1, β2) = ∫∫ exp(-β1 r1² - β2 r2²) / r12 d^3r1 d^3r2"""
    return 2 * math.pi**2.5 / (beta1 * beta2 * math.sqrt(beta1 + beta2))

print(f"2. J(β,β) 其中 β=2α={beta}:")
print(f"   J = {J_beta(beta, beta):.15f}")
print(f"   应等于 (ss|ss) = {ssss:.15f}")
print(f"   差值: {J_beta(beta, beta) - ssss:.2e}")
print()

# 3. -dJ/dβ1 = ∫∫ r1² exp(-β1 r1² - β2 r2²) / r12 d^3r1 d^3r2
def J_dbeta1(beta1, beta2):
    """-∂J/∂β1"""
    return (2 * math.pi**2.5 / (beta1**2 * beta2 * math.sqrt(beta1 + beta2)) 
            + math.pi**2.5 / (beta1 * beta2 * (beta1 + beta2)**1.5))

dJ_dbeta1 = J_dbeta1(beta, beta)
print(f"3. -∂J/∂β1 = ∫∫ r1² exp(-βr1²-βr2²)/r12 d^3r1 d^3r2:")
print(f"   值: {dJ_dbeta1:.15f}")
print(f"   所以 (p_x p_x | ss) 应该是 1/3 * 这个值 = {dJ_dbeta1/3:.15f}")
print()

# 4. ∂²J/∂β1∂β2 = ∫∫ r1² r2² exp(-β1 r1² - β2 r2²) / r12 d^3r1 d^3r2
def J_dbeta1dbeta2(beta1, beta2):
    """∂²J/∂β1∂β2"""
    term1 = 4 * math.pi**2.5 / (beta1**2 * beta2**2 * math.sqrt(beta1 + beta2))
    term2 = 2 * math.pi**2.5 / (beta1**2 * beta2 * (beta1 + beta2)**1.5)
    term3 = 2 * math.pi**2.5 / (beta1 * beta2**2 * (beta1 + beta2)**1.5)
    term4 = 1.5 * math.pi**2.5 / (beta1 * beta2 * (beta1 + beta2)**2.5)
    return term1 + term2 + term3 + term4

d2J_dbeta1dbeta2 = J_dbeta1dbeta2(beta, beta)
print(f"4. ∂²J/∂β1∂β2 = ∫∫ r1² r2² exp(-βr1²-βr2²)/r12 d^3r1 d^3r2:")
print(f"   值: {d2J_dbeta1dbeta2:.15f}")
print(f"   这就是 Σ_{{ij}} (p_i p_i | p_j p_j)")
print()

# 5. 现在用 VRR 计算并比较
from calchem.integrals.two_electron import _ERIRecursion

rec = _ERIRecursion(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

# 计算各种积分
vrr_ss_ss = rec.get(0,0,0, 0,0,0, 0,0,0, 0,0,0)
vrr_pxs_pxs = rec.get(1,0,0, 0,0,0, 1,0,0, 0,0,0)
vrr_pxpx_ss = rec.get(1,0,0, 1,0,0, 0,0,0, 0,0,0)  # (p_x p_x | ss)
vrr_dxxs_ss = rec.get(2,0,0, 0,0,0, 0,0,0, 0,0,0)  # (d_x2 s | ss) - 应该等于 (p_x p_x | ss)
vrr_pxpx_pxpx = rec.get(1,0,0, 1,0,0, 1,0,0, 1,0,0)
vrr_pxpx_pypy = rec.get(1,0,0, 1,0,0, 0,1,0, 0,1,0)

# 总和
vrr_sum = 0.0
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
        vrr_sum += rec.get(li, mi, ni, li, mi, ni, lj, mj, nj, lj, mj, nj)

print("=" * 70)
print("VRR 结果 vs 解析结果")
print("=" * 70)
print()

print(f"(ss|ss):")
print(f"   VRR:      {vrr_ss_ss:.15f}")
print(f"   解析:     {ssss:.15f}")
print(f"   比值:     {vrr_ss_ss/ssss:.15f}")
print()

print(f"(p_x s | p_x s):")
print(f"   VRR:      {vrr_pxs_pxs:.15f}")
# (p_x s | p_x s) = ∫∫ x1 x2 exp(-2αr1²-2αr2²)/r12 d^3r1 d^3r2
# 用参数微分的话，这需要对 β1 和 β2 各求一次导，但只对 x 分量。
# 由于 x² = r²/3 的关系，我们有 ∫∫ x1 x2 / r12 ... = ?
# 实际上，(p_x s | p_x s) 应该等于 (1/3) * 什么？
# 不对，因为 x1 x2 不是球对称的。
# 让我们用 VRR 公式手动验证：
# (1,0,0 | 1,0,0)_0 = (ssss)_1 / (2ζ) （同中心时）
zeta = 4 * alpha
ssss_1 = eri_ssss_n(1, alpha, origin, origin, origin, origin)
pxs_pxs_analytic = ssss_1 / (2 * zeta)
print(f"   解析(VRR公式): {pxs_pxs_analytic:.15f}")
print(f"   比值:     {vrr_pxs_pxs/pxs_pxs_analytic:.15f}")
print()

print(f"(p_x p_x | ss) = (d_x2 s | ss):")
print(f"   VRR (HRR): {vrr_pxpx_ss:.15f}")
print(f"   VRR (VR):  {vrr_dxxs_ss:.15f}")
print(f"   解析(球对称 1/3): {dJ_dbeta1/3:.15f}")
print(f"   比值 VR/解析: {vrr_dxxs_ss/(dJ_dbeta1/3):.15f}")
print()

print(f"(p_x p_x | p_x p_x):")
print(f"   VRR:      {vrr_pxpx_pxpx:.15f}")
# 解析值我们还不知道，先放着
print()

print(f"Σ (p_i p_i | p_j p_j):")
print(f"   VRR:      {vrr_sum:.15f}")
print(f"   解析:     {d2J_dbeta1dbeta2:.15f}")
print(f"   比值:     {vrr_sum/d2J_dbeta1dbeta2:.15f}")
print()

# 6. 现在让我们手动用 VRR 公式重新推导 (2,0,0 | 0,0,0)
# 看看是否和我们的 VRR 实现一致。
print("=" * 70)
print("手动 VRR 验证 (2,0,0 | 0,0,0)")
print("=" * 70)
print()

zeta_ab = 2 * alpha
zeta_cd = 2 * alpha
zeta = zeta_ab + zeta_cd

# (0,0,0 | 0,0,0)_n = ssss_n
# 上升 a_x 到 (1,0,0 | 0,0,0)_n = 0 （奇函数）
# 再上升 a_x 到 (2,0,0 | 0,0,0)_n
# VRR: (a+1_x|c)_n = PA_x (a|c)_n + (ζ_cd/ζ)PQ_x (a|c)_{n+1}
#                    + a_x/(2ζ_ab)(a-1_x|c)_n + c_x/(2ζ)(a|c-1_x)_{n+1}
# PA=0, PQ=0, c_x=0
# 所以 (2,0,0|0,0,0)_n = 0 + 0 + 1/(2ζ_ab) * (0,0,0|0,0,0)_n + 0
# = (ssss)_n / (2ζ_ab)

ssss_0 = eri_ssss_n(0, alpha, origin, origin, origin, origin)
manual_dxx_ss = ssss_0 / (2 * zeta_ab)
print(f"手动计算 (d_x2 s | ss) = (ssss)_0 / (2 ζ_ab):")
print(f"   = {ssss_0:.10f} / (2 * {zeta_ab})")
print(f"   = {manual_dxx_ss:.15f}")
print(f"VRR 实现: {vrr_dxxs_ss:.15f}")
print(f"差值: {manual_dxx_ss - vrr_dxxs_ss:.2e}")
print()

# 但是这个值和球对称预期的 1/3 * (-dJ/dβ) 不一致。
# 让我们验证一下 J(β) 的导数是否正确。
# 用有限差分验证。

print("=" * 70)
print("有限差分验证导数")
print("=" * 70)
print()

h = 1e-5
beta1 = beta
beta2 = beta

# -dJ/dβ1 的有限差分
dJ_fd = -(J_beta(beta1 + h, beta2) - J_beta(beta1 - h, beta2)) / (2 * h)
print(f"-∂J/∂β1 (解析): {J_dbeta1(beta1, beta2):.15f}")
print(f"-∂J/∂β1 (有限差分): {dJ_fd:.15f}")
print(f"差值: {J_dbeta1(beta1, beta2) - dJ_fd:.2e}")
print()

# ∂²J/∂β1∂β2 的有限差分
d2J_fd = (J_beta(beta1 + h, beta2 + h) - J_beta(beta1 + h, beta2 - h) 
          - J_beta(beta1 - h, beta2 + h) + J_beta(beta1 - h, beta2 - h)) / (4 * h * h)
print(f"∂²J/∂β1∂β2 (解析): {J_dbeta1dbeta2(beta1, beta2):.15f}")
print(f"∂²J/∂β1∂β2 (有限差分): {d2J_fd:.15f}")
print(f"差值: {J_dbeta1dbeta2(beta1, beta2) - d2J_fd:.2e}")
print()

# 现在关键问题：(d_x2 s | ss) 应该等于多少？
# 让我们直接数值积分来验证。

print("=" * 70)
print("直接数值积分验证 (p_x p_x | ss)")
print("=" * 70)
print()

# 积分 = ∫∫ x1² exp(-βr1²) exp(-βr2²) / r12 d^3r1 d^3r2
# 用球坐标 + 角向展开
# 
# x1² = r1² * (1/3 + ... l=2 项)
# 但 1/r12 是球对称的，所以 l=2 项积分后为零？
# 不对，1/r12 不是常数，它依赖于两个坐标的相对方向。
# 
# 让我用球谐展开来计算。
# 
# x1²/r1² = sin²θ1 cos²φ1 = Σ_{l=0,2} a_lm Y_lm(Ω1)
# 1/r12 = Σ_{l,m} (4π/(2l+1)) r_<^l/r_>^{l+1} Y_lm(Ω1) Y_lm^*(Ω2)
# 
# 积分 = ∫∫ r1² r2² exp(-βr1²-βr2²) * r1² * [∫∫ (x1²/r1²) / r12 dΩ1 dΩ2] dr1 dr2
# = ∫_0^∞ ∫_0^∞ r1^4 r2² exp(-βr1²-βr2²) * Σ_l (4π/(2l+1)) (r_<^l/r_>^{l+1}) Σ_m |a_lm|² dr1 dr2
# 
# 对于 x²/r²，只有 l=0 和 l=2 有贡献。
# 
# l=0: a_00 = ∫ sin²θ cos²φ * Y00 sinθ dθ dφ = ?
# 我们之前算过 a_00 = √(4π)/3
# Σ_m |a_0m|² = |a_00|² = 4π/9
# 
# l=2: 有 m=0 和 m=±2 等贡献
# 
# 让我直接计算角向积分的总和。
# 实际上更简单的方法：
# ∫∫ x1² / r12 dΩ1 dΩ2 = ?
# 
# 利用 1/r12 的展开和 x1² 的展开：
# ∫∫ x1² / r12 dΩ1 dΩ2 = Σ_{l,m} (4π/(2l+1)) r_<^l/r_>^{l+1} ∫ x1² Y_lm(1) dΩ1 ∫ Y_lm^*(2) dΩ2
# 
# 但 ket 侧是 s 波（球对称），即 ∫ Y_lm(2) dΩ2 只有 l=0, m=0 时有非零值。
# ∫ Y_00(2) dΩ2 = √(4π)
# 
# 所以只有 l=0, m=0 有贡献！
# 
# ∫∫ x1² / r12 dΩ1 dΩ2 = (4π/1) * (r_<^0/r_>^1) * a_00 * √(4π)
# 不对，让我重新来。
# 
# 等一下，ket 侧是球对称的（ss），所以 ket 的角分布是常数。
# 积分 (bra | ket) 中，ket 的角向积分是 ∫ Y_lm(Ω2) dΩ2。
# 对于 l=0, m=0，∫ Y_00 dΩ = √(4π)。
# 对于 l>0，∫ Y_lm dΩ = 0。
# 
# 所以角向积分 = (4π/1) * (1/r_>) * a_00 * √(4π)？
# 不对，公式是：
# ∫∫ f(Ω1) g(Ω2) / r12 dΩ1 dΩ2
# = Σ_{l,m} (4π/(2l+1)) r_<^l/r_>^{l+1} ∫ f(Ω1) Y_lm(Ω1) dΩ1 ∫ g(Ω2) Y_lm^*(Ω2) dΩ2
# 
# 这里 f = x1²/r1², g = 1（球对称）
# ∫ g Y_lm^* dΩ2 = ∫ Y_lm^* dΩ2 = √(4π) δ_{l,0} δ_{m,0}
# 因为只有 Y_00 有非零的积分值，∫ Y_00 dΩ = √(4π)
# 
# 所以只有 l=0, m=0 有贡献：
# = (4π/1) * (1/r_>) * a_00 * √(4π)
# 
# a_00 = ∫ f Y_00 dΩ1 = ∫ (x²/r²) Y_00 dΩ
# 
# 让我计算 a_00：
# x²/r² = sin²θ cos²φ
# a_00 = ∫_0^π ∫_0^{2π} sin²θ cos²φ * (1/√(4π)) * sinθ dφ dθ
# = (1/√(4π)) ∫_0^π sin³θ dθ ∫_0^{2π} cos²φ dφ
# = (1/√(4π)) * (4/3) * π
# = 4π / (3√(4π))
# = √(4π) / 3
# = 2√π / 3
# 
# 所以角向积分 = (4π) * (1/r_>) * (2√π/3) * √(4π)
# = 4π * (1/r_>) * 2√π/3 * 2√π
# = 4π * (1/r_>) * 4π/3
# = 16π² / (3 r_>)
# 
# 所以 (p_x p_x | ss) = ∫_0^∞ ∫_0^∞ r1^4 r2² exp(-βr1²-βr2²) * 16π²/(3 r_>) dr1 dr2
# = (16π²/3) ∫_0^∞ ∫_0^∞ r1^4 r2² / r_> exp(-βr1²-βr2²) dr1 dr2
# 
# 让我们计算这个积分。

def radial_integral_n1_n2(n1, n2, beta, l=0):
    """计算 ∫_0^∞ ∫_0^∞ r1^(2+n1) r2^(2+n2) exp(-βr1²-βr2²) (r_<^l/r_>^{l+1}) dr1 dr2
    不对，让我重新定义。
    这里的径向积分已经包含了 r1² r2²（来自 d^3r），所以：
    总积分 = ∫ r1² dr1 ∫ r2² dr2 [r1^{n1} r2^{n2} ...]
    即被积函数 r 的幂次是 r1^(2+n1) 和 r2^(2+n2)。
    """
    # 区域 1: r1 < r2, r_< = r1, r_> = r2 → r_<^l / r_>^{l+1} = r1^l / r2^{l+1}
    # 区域 2: r1 > r2, r_< = r2, r_> = r1 → r_<^l / r_>^{l+1} = r2^l / r1^{l+1}
    # 
    # 两个区域对称，所以积分 = 2 * ∫_0^∞ r1^(2+n1) r1^l exp(-βr1²) [∫_{r1}^∞ r2^(2+n2) / r2^{l+1} exp(-βr2²) dr2] dr1
    # = 2 * ∫_0^∞ r1^(2+n1+l) exp(-βr1²) [∫_{r1}^∞ r2^(2+n2-l-1) exp(-βr2²) dr2] dr1
    # = 2 * ∫_0^∞ r1^(n1+l+2) exp(-βr1²) [∫_{r1}^∞ r2^(n2-l+1) exp(-βr2²) dr2] dr1
    
    def inner(r1):
        result, _ = quad(lambda r2: r2**(n2 - l + 1) * math.exp(-beta * r2**2), r1, 20)
        return result
    
    def integrand(r1):
        return 2 * r1**(n1 + l + 2) * math.exp(-beta * r1**2) * inner(r1)
    
    result, _ = quad(integrand, 0, 20)
    return result

# (p_x p_x | ss):
# bra: n1 = 2 (x² ~ r²)
# ket: n2 = 0 (常数)
# l = 0 (只有 s 波贡献)
# 角因子 = 16π²/3

R_pxpx_ss = radial_integral_n1_n2(2, 0, beta, 0)
pxpx_ss_numerical = (16 * math.pi**2 / 3) * R_pxpx_ss

print(f"(p_x p_x | ss) 的数值积分:")
print(f"   径向积分 R_0 = {R_pxpx_ss:.15f}")
print(f"   角因子 16π²/3 = {16*math.pi**2/3:.15f}")
print(f"   总值 = {pxpx_ss_numerical:.15f}")
print(f"   VRR 值 = {vrr_dxxs_ss:.15f}")
print(f"   比值 VRR/数值 = {vrr_dxxs_ss / pxpx_ss_numerical:.15f}")
print()

# 现在让我们计算 (p_x p_x | p_x p_x)
# 这需要 l=0 和 l=2 的贡献。
# 
# bra: f = x²/r², ket: g = x²/r²
# 两者都有 l=0 和 l=2 的分量。
# 
# 总积分 = Σ_{l=0,2} (4π/(2l+1)) (Σ_m a_lm b_lm^*) R_l
# 其中 R_l 是径向积分，n1=2, n2=2。
# 
# 对于 l=0:
# a_00 = 2√π/3, b_00 = 2√π/3
# Σ_m a_0m b_0m^* = a_00 * b_00 = 4π/9
# 贡献 = (4π/1) * (4π/9) * R_0 = 16π²/9 * R_0
# 
# 对于 l=2:
# 有 m=0, ±1, ±2 五个分量。
# 但 x²/r² 只有 m=0 和 m=±2 分量。
# a_20 = ?, a_22c = ?
# 
# 让我直接计算 Σ_m |a_2m|² 对于 x²/r²。
# 我们知道 ∫ f² dΩ = Σ_{l,m} |a_lm|²
# ∫ (x²/r²)² dΩ = ∫ sin^4θ cos^4φ sinθ dθ dφ
# = ∫_0^π sin^5θ dθ ∫_0^{2π} cos^4φ dφ
# = 16/15 * 3π/4 = 4π/5
# 
# 而 Σ_m |a_0m|² = 4π/9
# 所以 Σ_m |a_2m|² = 4π/5 - 4π/9 = 4π * (1/5 - 1/9) = 4π * 4/45 = 16π/45
# 
# 因此 l=2 的贡献 = (4π/5) * (16π/45) * R_2？不对。
# 
# 公式是 Σ_{l,m} (4π/(2l+1)) a_lm b_lm^* R_l
# 对于 bra = ket = x²/r²：
# = (4π/1) |a_00|² R_0 + (4π/5) Σ_m |a_2m|² R_2
# = (4π) * (4π/9) * R_0 + (4π/5) * (16π/45) * R_2
# = 16π²/9 * R_0 + 64π²/225 * R_2

R0_pp = radial_integral_n1_n2(2, 2, beta, 0)
R2_pp = radial_integral_n1_n2(2, 2, beta, 2)

pxpx_pxpx_numerical = (16 * math.pi**2 / 9) * R0_pp + (64 * math.pi**2 / 225) * R2_pp

print(f"(p_x p_x | p_x p_x) 的数值积分:")
print(f"   R_0 = {R0_pp:.15f}")
print(f"   R_2 = {R2_pp:.15f}")
print(f"   l=0 贡献: {16*math.pi**2/9 * R0_pp:.15f}")
print(f"   l=2 贡献: {64*math.pi**2/225 * R2_pp:.15f}")
print(f"   总值 = {pxpx_pxpx_numerical:.15f}")
print(f"   VRR 值 = {vrr_pxpx_pxpx:.15f}")
print(f"   比值 VRR/数值 = {vrr_pxpx_pxpx / pxpx_pxpx_numerical:.15f}")
print()

# 现在计算总和 Σ (p_i p_i | p_j p_j)
# 对于总和，f = Σ_i (x_i²/r²) = 1（球对称），所以只有 l=0 贡献。
# a_00 = ∫ 1 * Y00 dΩ = √(4π)
# 角向因子 = (4π) * (√(4π))² = 4π * 4π = 16π²
# 
# 总积分 = 16π² * R_0

total_numerical = 16 * math.pi**2 * R0_pp
print(f"Σ (p_i p_i | p_j p_j) 的数值积分:")
print(f"   16π² * R_0 = {total_numerical:.15f}")
print(f"   ∂²J/∂β1∂β2 = {d2J_dbeta1dbeta2:.15f}")
print(f"   差值 = {total_numerical - d2J_dbeta1dbeta2:.2e}")
print(f"   VRR 值 = {vrr_sum:.15f}")
print(f"   比值 VRR/数值 = {vrr_sum / total_numerical:.15f}")
print()

print("=" * 70)
print("结论")
print("=" * 70)
