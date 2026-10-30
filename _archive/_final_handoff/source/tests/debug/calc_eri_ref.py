import numpy as np
from scipy.integrate import quad

alpha = 1.0
Ns = (2*alpha/np.pi)**(3/4)
Np = 2 * (2*alpha)**(5/4) / (np.pi**(3/4))

print("=== 归一化常数 ===")
print(f"Ns = {Ns:.10f}")
print(f"Np = {Np:.10f}")
print(f"Ns^2 = {Ns**2:.10f}")
print(f"Np^2 = {Np**2:.10f}")

def coulomb_spherical(f, g):
    """球对称库仑积分 J = ∫∫ f(r1) g(r2) / |r1-r2| dr1 dr2"""
    def inner1(r1):
        result, _ = quad(lambda r2: r2**2 * g(r2), 0, r1)
        return result / r1
    
    def inner2(r1):
        result, _ = quad(lambda r2: r2 * g(r2), r1, np.inf)
        return result
    
    def integrand(r1):
        return 4 * np.pi * r1**2 * f(r1) * (inner1(r1) + inner2(r1))
    
    result, _ = quad(integrand, 0, np.inf)
    return result

f_ss = lambda r: np.exp(-2*alpha * r**2)
g_ss = lambda r: np.exp(-2*alpha * r**2)

J_ss = coulomb_spherical(f_ss, g_ss)
print()
print("=== 球对称库仑积分 ===")
print(f"未归一化 (ss|ss) = {J_ss:.10f}")
print(f"解析值 π^(5/2)/(4α^(5/2)) = {np.pi**(5/2)/4:.10f}")
print(f"归一化 (ss|ss) = {J_ss * Ns**4:.10f}")

g_r2 = lambda r: r**2 * np.exp(-2*alpha * r**2)
J_s_r2 = coulomb_spherical(f_ss, g_r2)
J_s_px2 = J_s_r2 / 3.0
print()
print(f"未归一化 (ss | p_x^2) = {J_s_px2:.10f}")
print(f"归一化 (ss | p_x^2) = {J_s_px2 * Ns**2 * Np**2:.10f}")
print(f"比值 (ss|pp)/(ss|ss) 未归一化 = {J_s_px2 / J_ss:.10f}")
print(f"预期 5/24 = {5/24:.10f}")
print(f"比值 (ss|pp)/(ss|ss) 归一化 = {J_s_px2 * Ns**2 * Np**2 / (J_ss * Ns**4):.10f}")
print(f"预期 5/6 = {5/6:.10f}")

print()
print("=== (pp|pp) 积分 ===")

def multipole_eri(f_r, g_r, l):
    """l阶多极贡献的库仑积分"""
    def inner1(r1):
        result, _ = quad(lambda r2: r2**2 * g_r(r2) * (r2**l / r1**(l+1)) if r2 <= r1 else 0, 0, r1)
        return result
    
    def inner2(r1):
        result, _ = quad(lambda r2: r2**2 * g_r(r2) * (r1**l / r2**(l+1)) if r2 > r1 else 0, r1, np.inf)
        return result
    
    def integrand(r1):
        return 4 * np.pi / (2*l + 1) * r1**2 * f_r(r1) * (inner1(r1) + inner2(r1))
    
    result, _ = quad(integrand, 0, np.inf)
    return result

# 对于 (p_x p_x | p_x p_x)，两侧都是 x² 分布
# 展开成多极：l=0 和 l=2
# 径向函数都是 r² exp(-2αr²)
# 
# 角向展开系数：
# x² = (1/3) r² + ...  l=0 成分
# x² 的 l=2 成分呢？
# 
# 用球谐函数展开：
# x²/r² = sin²θ cos²φ
# = (1 - cos²θ) * (1 + cos2φ)/2
# = 1/2 - cos²θ/2 + (1 - cos²θ)cos2φ/2
# 
# 用 Legendre 多项式：
# P_0 = 1
# P_1 = cosθ
# P_2 = (3cos²θ - 1)/2
# 
# 所以：cos²θ = (2P_2 + 1)/3
# 1 - cos²θ = 2/3 - 2P_2/3
# 
# sin²θ cos²φ = (1/2 - cos²θ/2) + (1 - cos²θ)cos2φ/2
# = 1/2 - (2P_2+1)/6 + (2/3 - 2P_2/3) cos2φ / 2
# = 1/2 - 1/6 - P_2/3 + (1/3 - P_2/3) cos2φ
# = 1/3 - P_2/3 + (1 - P_2) cos2φ / 3
# 
# l=0 成分: 1/3 （因为 P_0=1，系数是 1/3）
# l=2, m=0 成分: ?
# 
# 等等，让我直接计算角向积分来找到展开系数。
# 
# x²/r² = Σ_lm C_lm Y_lm(Ω)
# C_lm = ∫ (x²/r²) Y_lm*(Ω) dΩ
# 
# 对于实球谐函数：
# Y_00 = 1/sqrt(4π)
# Y_20 = sqrt(5/(16π)) (3cos²θ - 1)
# Y_22c = sqrt(15/(16π)) sin²θ cos2φ
# 
# C_00 = ∫ sin²θ cos²φ * 1/sqrt(4π) dΩ
# = 1/sqrt(4π) ∫_0^π ∫_0^{2π} sin²θ cos²φ sinθ dφ dθ
# = 1/sqrt(4π) * (∫_0^π sin³θ dθ) * (∫_0^{2π} cos²φ dφ)
# = 1/sqrt(4π) * (4/3) * π
# = 1/sqrt(4π) * 4π/3
# = 2 sqrt(π)/3
# 
# 所以 l=0 成分的径向密度是：
# ρ_00(r) = C_00 * Y_00 * r² exp(-2αr²) * Np^2
# 不对，我混淆了。
# 
# 实际上，库仑积分的多极展开公式是：
# ∫∫ ρ1(r1) ρ2(r2) / |r1-r2| dr1 dr2 = Σ_lm (4π/(2l+1)) ∫_0^∞ ∫_0^∞ r1² ρ1_lm(r1) r2² ρ2_lm(r2) r_<^l/r_>^{l+1} dr1 dr2
# 
# 其中 ρ_lm(r) 是径向部分，满足 ρ(r,Ω) = Σ_lm ρ_lm(r) Y_lm(Ω)
# 
# 对于 ρ(r) = Np² x² exp(-2αr²) = Np² r² sin²θ cos²φ exp(-2αr²)
# = Np² r² exp(-2αr²) * sin²θ cos²φ
# 
# 展开 sin²θ cos²φ：
# = (1/3) + (-1/6) * (3cos²θ-1) + (1/2) * sin²θ cos2φ
# 验证：
# 1/3 - (3cos²θ-1)/6 + sin²θ cos2φ/2
# = 1/3 - cos²θ/2 + 1/6 + sin²θ cos2φ/2
# = 1/2 - cos²θ/2 + sin²θ cos2φ/2
# = sin²θ/2 + sin²θ cos2φ/2
# = sin²θ (1 + cos2φ)/2
# = sin²θ cos²φ ✓
# 
# 好的！所以：
# sin²θ cos²φ = A_0 * 1 + A_2_0 * (3cos²θ-1) + A_2_2 * sin²θ cos2φ
# 其中：
# A_0 = 1/3
# A_2_0 = -1/6
# A_2_2 = 1/2
# 
# 现在，实球谐函数：
# Y_00 = 1/sqrt(4π)
# Y_20 = sqrt(5/(16π)) (3cos²θ-1) = sqrt(5/(16π)) * P_2(cosθ) * 2? 不对
# P_2 = (3cos²θ-1)/2，所以 3cos²θ-1 = 2 P_2
# Y_20 = sqrt(5/(4π)) P_2 = sqrt(5/(16π)) * 2 P_2 = sqrt(5/(16π)) (3cos²θ-1)
# 所以 (3cos²θ-1) = sqrt(16π/5) Y_20 = 4 sqrt(π/5) Y_20
# 
# Y_22c = sqrt(15/(16π)) sin²θ cos2φ
# 所以 sin²θ cos2φ = sqrt(16π/15) Y_22c = 4 sqrt(π/15) Y_22c
# 
# 因此：
# sin²θ cos²φ = A_0 * 1 + A_2_0 * 4 sqrt(π/5) Y_20 + A_2_2 * 4 sqrt(π/15) Y_22c
# = A_0 * sqrt(4π) Y_00 + ...
# 
# 等等，1 = sqrt(4π) Y_00
# 
# 所以：
# C_00 * Y_00 = A_0 * 1 = A_0 * sqrt(4π) Y_00
# C_00 = A_0 * sqrt(4π) = (1/3) * 2 sqrt(π) = 2 sqrt(π) / 3
# 
# C_20 * Y_20 = A_2_0 * (3cos²θ-1) = A_2_0 * 4 sqrt(π/5) Y_20
# C_20 = A_2_0 * 4 sqrt(π/5) = (-1/6) * 4 sqrt(π/5) = -2/3 * sqrt(π/5)
# 
# C_22 * Y_22 = A_2_2 * sin²θ cos2φ = A_2_2 * 4 sqrt(π/15) Y_22c
# C_22 = A_2_2 * 4 sqrt(π/15) = (1/2) * 4 sqrt(π/15) = 2 sqrt(π/15)
# 
# 所以 ρ(r,Ω) = Np² r² exp(-2αr²) * sin²θ cos²φ
# = Np² r² exp(-2αr²) * [C_00 Y_00 + C_20 Y_20 + C_22 Y_22c]
# 
# 因此径向展开系数：
# ρ_00(r) = Np² * C_00 * r² exp(-2αr²)
# ρ_20(r) = Np² * C_20 * r² exp(-2αr²)
# ρ_22(r) = Np² * C_22 * r² exp(-2αr²)

C00 = 2 * np.sqrt(np.pi) / 3
C20 = -2/3 * np.sqrt(np.pi/5)
C22 = 2 * np.sqrt(np.pi/15)

print()
print("角向展开系数:")
print(f"C00 = {C00:.10f}")
print(f"C20 = {C20:.10f}")
print(f"C22 = {C22:.10f}")

# 验证归一化：∫ sin²θ cos²φ dΩ = 4π/15?
# ∫_0^π sin³θ dθ = 4/3
# ∫_0^{2π} cos²φ dφ = π
# 所以 ∫ sin²θ cos²φ dΩ = 4/3 * π = 4π/3
# 不对，等等，∫ sin²θ cos²φ dΩ = ∫_0^π sin³θ dθ ∫_0^{2π} cos²φ dφ = (4/3)(π) = 4π/3
# 
# 另一方面，Σ_lm |C_lm|² = C00² + C20² + C22²
# = 4π/9 + 4π/(9*5) + 4π/15
# = 4π/9 + 4π/45 + 4π/15
# = 20π/45 + 4π/45 + 12π/45
# = 36π/45 = 4π/5
# 
# 不对，应该是 ∫ |f(Ω)|² dΩ = Σ |C_lm|²
# ∫ sin^4θ cos^4φ dΩ = ?
# 这应该等于 Σ |C_lm|²
# 让我直接数值验证一下... 算了，先继续。

# 对于 (p_x p_x | p_x p_x)，两侧都是相同的分布
# 积分 = Σ_lm (4π/(2l+1)) * (C_lm)^2 * ∫∫ r1² f(r1) r2² f(r2) r_<^l/r_>^{l+1} dr1 dr2
# 其中 f(r) = Np² r² exp(-2αr²)

def radial_multipole_integral(f, l):
    """计算径向多极积分 ∫∫ r1² f(r1) r2² f(r2) r_<^l/r_>^{l+1} dr1 dr2"""
    def inner1(r1):
        result, _ = quad(lambda r2: r2**2 * f(r2) * (r2**l / r1**(l+1)), 0, r1)
        return result
    
    def inner2(r1):
        result, _ = quad(lambda r2: r2**2 * f(r2) * (r1**l / r2**(l+1)), r1, np.inf)
        return result
    
    def integrand(r1):
        return r1**2 * f(r1) * (inner1(r1) + inner2(r1))
    
    result, _ = quad(integrand, 0, np.inf)
    return result

f_px_r = lambda r: Np**2 * r**2 * np.exp(-2*alpha * r**2)
# 等等，不对。ρ_lm(r) 已经包含了 C_lm 和径向部分。
# 让我重新来。
# 
# ρ(r,Ω) = Σ_lm ρ_lm(r) Y_lm(Ω)
# 其中 ρ_lm(r) = C_lm * Np² r² exp(-2αr²)
# 
# 库仑积分 = Σ_lm (4π/(2l+1)) ∫_0^∞ ∫_0^∞ ρ1_lm(r1) ρ2_lm(r2) r_<^l/r_>^{l+1} r1² r2² dr1 dr2
# 
# 因为 ρ1 = ρ2 = ρ_px，所以：

def eri_from_multipoles(C_lm_dict, f_r):
    """从多极展开计算ERI"""
    total = 0.0
    for (l, m), C in C_lm_dict.items():
        radial_int = radial_multipole_integral(f_r, l)
        contribution = (4*np.pi / (2*l + 1)) * C**2 * radial_int
        total += contribution
        print(f"  l={l}, m={m}: C={C:.6f}, 径向积分={radial_int:.6f}, 贡献={contribution:.10f}")
    return total

# ρ(r,Ω) = Np² r² exp(-2αr²) * sin²θ cos²φ
# = Σ_lm [C_lm * Np² r² exp(-2αr²)] * Y_lm
# 所以 ρ_lm(r) = C_lm * Np² r² exp(-2αr²)

C_lm = {
    (0, 0): C00,
    (2, 0): C20,
    (2, 2): C22,
}

f_r = lambda r: Np**2 * r**2 * np.exp(-2*alpha * r**2)

print()
print("=== (p_x p_x | p_x p_x) 多极展开 ===")
eri_pxpx_pxpx = eri_from_multipoles(C_lm, f_r)
print(f"结果: {eri_pxpx_pxpx:.10f}")
print()

# 现在计算 (p_x p_x | p_y p_y)
# ρ_pypy 对应的是 y² = r² sin²θ sin²φ
# 展开 sin²θ sin²φ：
# = sin²θ (1 - cos2φ)/2
# = sin²θ/2 - sin²θ cos2φ/2
# 
# 和之前类似：
# sin²θ/2 = (1 - cos²θ)/2 = 1/2 - cos²θ/2
# = 1/2 - (2P_2+1)/6 = 1/2 - 1/6 - P_2/3 = 1/3 - P_2/3
# = 1/3 - (3cos²θ-1)/6
# 
# 所以 sin²θ sin²φ = 1/3 - (3cos²θ-1)/6 - sin²θ cos2φ/2
# 
# 系数：
# A_0 = 1/3
# A_2_0 = -1/6
# A_2_2 = -1/2
# 
# C00 = A_0 * sqrt(4π) = 2 sqrt(π)/3 （和p_x一样）
# C20 = A_2_0 * 4 sqrt(π/5) = -2/3 sqrt(π/5) （和p_x一样）
# C22 = A_2_2 * 4 sqrt(π/15) = -1/2 * 4 sqrt(π/15) = -2 sqrt(π/15) （符号变了）

C_lm_py = {
    (0, 0): C00,
    (2, 0): C20,
    (2, 2): -C22,
}

print("=== (p_x p_x | p_y p_y) 多极展开 ===")
# 对于不同的分布，积分 = Σ_lm (4π/(2l+1)) C1_lm C2_lm * 径向积分
eri_pxpx_pypy = 0.0
for (l, m), C1 in C_lm.items():
    C2 = C_lm_py[(l, m)]
    radial_int = radial_multipole_integral(f_r, l)
    contribution = (4*np.pi / (2*l + 1)) * C1 * C2 * radial_int
    eri_pxpx_pypy += contribution
    print(f"  l={l}, m={m}: C1={C1:.6f}, C2={C2:.6f}, 贡献={contribution:.10f}")
print(f"结果: {eri_pxpx_pypy:.10f}")

print()
print("=== 汇总 ===")
print(f"(ss|ss) 归一化 = {J_ss * Ns**4:.10f}")
print(f"(p_x p_x | p_x p_x) = {eri_pxpx_pxpx:.10f}")
print(f"(p_x p_x | p_y p_y) = {eri_pxpx_pypy:.10f}")
print()

J_total = 3 * eri_pxpx_pxpx + 6 * eri_pxpx_pypy
print(f"J(p_total|p_total) = 3*(p_xp_x|p_xp_x) + 6*(p_xp_x|p_yp_y) = {J_total:.10f}")
print(f"球对称积分验证值 = 7.6165593779")
print(f"比值 = {J_total / 7.6165593779:.10f}")

print()
print("各个比值:")
print(f"(p_xp_x|p_xp_x) / (ss|ss) = {eri_pxpx_pxpx / (J_ss * Ns**4):.10f}")
print(f"(p_xp_x|p_yp_y) / (ss|ss) = {eri_pxpx_pypy / (J_ss * Ns**4):.10f}")
