import numpy as np
from scipy.integrate import quad

alpha = 1.0

# 直接用三维球坐标数值积分计算 (d_x2|d_y2) 和 (d_x2|d_x2)
# 积分 = ∫∫ x1² y2² exp(-2αr1²) exp(-2αr2²) / |r1-r2| dr1 dr2
# 
# 利用角度积分的对称性，我们可以用勒让德多项式展开。
# 不过更简单的是，我们已经知道 Σ (p_i²|p_j²) 的球对称值了。
# 我们还需要另一个关系来解出 J_para 和 J_perp。
# 
# 实际上，我们可以计算另一个组合：Σ_{ij} (p_i p_j | p_i p_j)
# 这等于 ∫∫ (r1·r2)² exp(-2αr1²) exp(-2αr2²) / |r1-r2| dr1 dr2 （未归一化）
# 
# 而 (r1·r2)² = r1² r2² cos²γ
# cos²γ = (1/3) + (2/3) P_2(cosγ)
# 
# 所以积分 = (1/3) ∫∫ r1² r2² exp(...) / r12 dr1 dr2
#          + (2/3) ∫∫ r1² r2² P_2(cosγ) exp(...) / r12 dr1 dr2
# 
# 第一项就是 (1/3) * J_total_un
# 第二项可以用 l=2 的多极积分来算。

def coulomb_multipole_radial(f_r, g_r, l):
    """计算 l 阶多极的径向积分。
    返回 ∫_0^∞ ∫_0^∞ r1² f_r(r1) r2² g_r(r2) r_<^l / r_>^{l+1} dr1 dr2
    """
    def inner(r1):
        part1, _ = quad(lambda r2: r2**2 * g_r(r2) * (r2**l / r1**(l+1)), 0, r1)
        part2, _ = quad(lambda r2: r2**2 * g_r(r2) * (r1**l / r2**(l+1)), r1, np.inf)
        return part1 + part2
    def integrand(r1):
        return r1**2 * f_r(r1) * inner(r1)
    result, _ = quad(integrand, 0, np.inf)
    return result

f_r2 = lambda r: r**2 * np.exp(-2*alpha * r**2)

# l=0 的径向积分
rad_0 = coulomb_multipole_radial(f_r2, f_r2, 0)
# 角向因子对于 Σ (p_i²|p_j²) 是 (4π) * (4π/3)? 不对，让我重新来。

# 实际上，对于 Σ_i (p_i²|p_j²) = ∫∫ r1² r2² exp(...) / r12 * (角向积分) dr1 dr2
# 角向积分 = ∫ dΩ1 ∫ dΩ2 Σ_i (x_i²/r1²) Σ_j (x_j²/r2²) = ∫ dΩ1 ∫ dΩ2 1 * 1 = 4π * 4π = 16π²
# 不对，Σ_i (p_i²/r²) = Σ_i x_i²/r² = 1
# 所以角向积分 = ∫ dΩ1 ∫ dΩ2 1 / |r1-r2| * 1 * 1？不对。

# 等等，我混淆了。
# ρ_total(r) = Σ_i p_i(r)^2 = Np² r² exp(-2αr²)
# J_total = ∫∫ ρ_total(r1) ρ_total(r2) / |r1-r2| dr1 dr2
# = Np^4 ∫∫ r1² r2² exp(-2αr1²) exp(-2αr2²) / |r1-r2| dr1 dr2
# 
# 这个积分我们已经算过了，就是球对称库仑积分。
# 径向部分：f(r) = r² exp(-2αr²)
# 角向积分：∫∫ 1 / |r1-r2| dΩ1 dΩ2 = 16π² / max(r1,r2) （之前验证过）
# 
# 所以 J_total_un = ∫_0^∞ ∫_0^∞ r1² f(r1) r2² f(r2) * 16π² / max(r1,r2) dr1 dr2
# = 16π² ∫∫ r1^4 r2^4 exp(...) exp(...) / max(r1,r2) dr1 dr2？不对
# 
# 等等，dr = r² dr dΩ
# 所以积分 = ∫_0^∞ r1² dr1 ∫_0^∞ r2² dr2 ∫ dΩ1 ∫ dΩ2 [r1² exp(...)] [r2² exp(...)] / |r1-r2|
# = ∫_0^∞ ∫_0^∞ r1^4 r2^4 exp(...) exp(...) [∫∫ dΩ1 dΩ2 / |r1-r2|] dr1 dr2
# = ∫_0^∞ ∫_0^∞ r1^4 r2^4 exp(...) exp(...) * 16π² / max(r1,r2) dr1 dr2
# 
# 这和我们之前的球对称积分结果一致。

# 现在，我们想知道 (p_x²|p_x²) 和 (p_x²|p_y²)
# 也就是 (d_x²|d_x²) 和 (d_x²|d_y²) （未归一化，即 x1² * y2² 等的积分）

# 我们可以用角向展开的方法。
# 对于固定的 r1, r2，角向积分是：
# I_xx = ∫∫ sin²θ1 cos²φ1 * sin²θ2 cos²φ2 / |r1-r2| dΩ1 dΩ2
# I_xy = ∫∫ sin²θ1 cos²φ1 * sin²θ2 sin²φ2 / |r1-r2| dΩ1 dΩ2
# 
# 利用加法定理：
# 1/|r1-r2| = Σ_l (4π/(2l+1)) r_<^l/r_>^{l+1} Σ_m Y_lm(Ω1) Y_lm*(Ω2)
# 
# 所以 I_xx = Σ_l (4π/(2l+1)) r_<^l/r_>^{l+1} Σ_m [∫ sin²θ cos²φ Y_lm dΩ]^2
# I_xy = Σ_l (4π/(2l+1)) r_<^l/r_>^{l+1} Σ_m [∫ sin²θ cos²φ Y_lm dΩ] [∫ sin²θ sin²φ Y_lm dΩ]
# 
# 我们需要计算这两个角向积分：
# A_lm = ∫ sin²θ cos²φ Y_lm dΩ
# B_lm = ∫ sin²θ sin²φ Y_lm dΩ
# 
# 由于对称性，只有 l=0 和 l=2 时有非零值。
# 而且对于 m，只有 m=0 和 m=±2 时有非零值。

# 让我们直接数值计算 A_lm 和 B_lm。

def angular_integral(func):
    """数值计算 ∫ f(θ,φ) dΩ"""
    N_theta = 100
    N_phi = 200
    theta = np.linspace(0, np.pi, N_theta)
    phi = np.linspace(0, 2*np.pi, N_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    
    theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
    integrand = func(theta_2d, phi_2d) * np.sin(theta_2d)
    return np.sum(integrand) * dtheta * dphi

# 实球谐函数
def Y_lm_real(l, m, theta, phi):
    if m == 0:
        # Y_l0
        from scipy.special import lpmv
        return np.sqrt((2*l+1) / (4*np.pi)) * lpmv(0, l, np.cos(theta))
    elif m > 0:
        # cosine version
        from scipy.special import lpmv
        return np.sqrt((2*l+1) / (2*np.pi)) * lpmv(m, l, np.cos(theta)) * np.cos(m*phi)
    else:
        # sine version
        from scipy.special import lpmv
        m_abs = -m
        return np.sqrt((2*l+1) / (2*np.pi)) * lpmv(m_abs, l, np.cos(theta)) * np.sin(m_abs*phi)

# 计算 A_lm = ∫ sin²θ cos²φ Y_lm dΩ
def f_A(theta, phi):
    return np.sin(theta)**2 * np.cos(phi)**2

def f_B(theta, phi):
    return np.sin(theta)**2 * np.sin(phi)**2

# 不用 scipy.special，直接手动计算 l=0 和 l=2 的实球谐函数值。
# 
# l=0, m=0:
# Y_00 = 1/sqrt(4π)
# 
# l=2, m=0:
# Y_20 = sqrt(5/(4π)) * (3cos²θ - 1)/2 = sqrt(5/(16π)) (3cos²θ - 1)
# 
# l=2, m=2 (cosine):
# Y_22c = sqrt(15/(4π)) * sin²θ cos2φ / 2 = sqrt(15/(16π)) sin²θ cos2φ

def Y00(theta, phi):
    return 1.0 / np.sqrt(4*np.pi)

def Y20(theta, phi):
    return np.sqrt(5/(16*np.pi)) * (3*np.cos(theta)**2 - 1)

def Y22c(theta, phi):
    return np.sqrt(15/(16*np.pi)) * np.sin(theta)**2 * np.cos(2*phi)

# 计算积分
A_00 = angular_integral(lambda th, ph: f_A(th, ph) * Y00(th, ph))
A_20 = angular_integral(lambda th, ph: f_A(th, ph) * Y20(th, ph))
A_22 = angular_integral(lambda th, ph: f_A(th, ph) * Y22c(th, ph))

B_00 = angular_integral(lambda th, ph: f_B(th, ph) * Y00(th, ph))
B_20 = angular_integral(lambda th, ph: f_B(th, ph) * Y20(th, ph))
B_22 = angular_integral(lambda th, ph: f_B(th, ph) * Y22c(th, ph))

print('角向展开系数:')
print(f'  A_00 = {A_00:.10f}')
print(f'  A_20 = {A_20:.10f}')
print(f'  A_22 = {A_22:.10f}')
print(f'  B_00 = {B_00:.10f}')
print(f'  B_20 = {B_20:.10f}')
print(f'  B_22 = {B_22:.10f}')
print()

# 现在计算径向积分
# 径向函数: r² exp(-2αr²) （因为 x² = r² * sin²θ cos²φ）
f_rad = lambda r: r**2 * np.exp(-2*alpha * r**2)

rad_l0 = coulomb_multipole_radial(f_rad, f_rad, 0)
rad_l2 = coulomb_multipole_radial(f_rad, f_rad, 2)

print(f'径向积分 l=0: {rad_l0:.10f}')
print(f'径向积分 l=2: {rad_l2:.10f}')
print()

# 计算 (d_x2|d_x2) 和 (d_x2|d_y2)
# I = Σ_l (4π/(2l+1)) Σ_m A_lm * B_lm * 径向积分_l
# 
# 对于 (d_x2|d_x2): A = A_lm, B = A_lm
# 对于 (d_x2|d_y2): A = A_lm, B = B_lm

I_xx = (4*np.pi / 1) * (A_00**2 * rad_l0) + (4*np.pi / 5) * (A_20**2 + A_22**2) * rad_l2
I_xy = (4*np.pi / 1) * (A_00 * B_00 * rad_l0) + (4*np.pi / 5) * (A_20 * B_20 + A_22 * B_22) * rad_l2

print(f'(d_x2|d_x2) 未归一化: {I_xx:.12f}')
print(f'(d_x2|d_y2) 未归一化: {I_xy:.12f}')
print()

# 验证总和: 3*I_xx + 6*I_xy 应该等于球对称积分的未归一化值
J_total_calc = 3 * I_xx + 6 * I_xy
print(f'3*I_xx + 6*I_xy = {J_total_calc:.12f}')

# 球对称积分（直接计算验证）
def coulomb_spherical_direct(f):
    def inner(r1):
        part1, _ = quad(lambda r2: r2**2 * f(r2) / r1, 0, r1)
        part2, _ = quad(lambda r2: r2**2 * f(r2) / r2, r1, np.inf)
        return part1 + part2
    def integrand(r1):
        return 16 * np.pi**2 * r1**2 * f(r1) * inner(r1)
    result, _ = quad(integrand, 0, np.inf)
    return result

f_r4 = lambda r: r**4 * np.exp(-2*alpha * r**2)  # 不对，让我重新来
# 球对称积分: ∫∫ [r1² exp(...)] [r2² exp(...)] / |r1-r2| dr1 dr2
# 不对，ρ_total(r) = r² exp(-2αr²)
# 积分 = ∫∫ ρ_total(r1) ρ_total(r2) / |r1-r2| dr1 dr2
# = ∫_0^∞ r1² dr1 ∫_0^∞ r2² dr2 ρ(r1) ρ(r2) ∫∫ dΩ1 dΩ2 / |r1-r2|
# = ∫_0^∞ ∫_0^∞ r1² r2² * r1² r2² exp(...) exp(...) * 16π² / max(r1,r2) dr1 dr2
# 不对，ρ(r) 已经是完整的函数了。
# ρ(r) d^3r = ρ(r) r² dr dΩ
# 所以积分 = ∫ ρ(r1) r1² dr1 dΩ1 ∫ ρ(r2) r2² dr2 dΩ2 / |r1-r2|
# = ∫_0^∞ ∫_0^∞ r1² ρ(r1) r2² ρ(r2) [∫∫ dΩ1 dΩ2 / |r1-r2|] dr1 dr2
# = 16π² ∫_0^∞ ∫_0^∞ r1² ρ(r1) r2² ρ(r2) / max(r1,r2) dr1 dr2
# 
# ρ(r) = r² exp(-2αr²)
# 所以 r² ρ(r) = r^4 exp(-2αr²)

J_total_spherical = coulomb_spherical_direct(lambda r: r**2 * np.exp(-2*alpha * r**2))
# 不对，上面的函数参数不对。让我重写。

def coulomb_spherical_direct2(rho):
    def inner(r1):
        part1, _ = quad(lambda r2: r2**2 * rho(r2) / r1, 0, r1)
        part2, _ = quad(lambda r2: r2**2 * rho(r2) / r2, r1, np.inf)
        return part1 + part2
    def integrand(r1):
        return 16 * np.pi**2 * r1**2 * rho(r1) * inner(r1)
    result, _ = quad(integrand, 0, np.inf)
    return result

rho = lambda r: r**2 * np.exp(-2*alpha * r**2)
J_total_spherical = coulomb_spherical_direct2(rho)

print(f'球对称积分 (直接): {J_total_spherical:.12f}')
print(f'比值: {J_total_calc / J_total_spherical:.12f}')
print()

# 和我们的VRR结果比较
from calchem.integrals.two_electron import _ERIRecursion
origin = np.zeros(3)
rec = _ERIRecursion(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

print('我们的 VRR 结果:')
vrr_xx = rec._get_n(2,0,0, 0,0,0, 2,0,0, 0,0,0, 0)
vrr_xy = rec._get_n(2,0,0, 0,0,0, 0,2,0, 0,0,0, 0)
print(f'  (d_x2|d_x2) = {vrr_xx:.12f}')
print(f'  (d_x2|d_y2) = {vrr_xy:.12f}')
print(f'  3*xx + 6*xy = {3*vrr_xx + 6*vrr_xy:.12f}')
print()
print(f'比值 (VRR/参考):')
print(f'  I_xx: {vrr_xx / I_xx:.12f}')
print(f'  I_xy: {vrr_xy / I_xy:.12f}')
