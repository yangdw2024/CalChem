import numpy as np
import math
from scipy.integrate import quad

def coulomb_spherical(rho1, rho2):
    def integrand(r1):
        q_in, _ = quad(lambda r2: rho2(r2) * 4 * math.pi * r2**2, 0, r1)
        v_out, _ = quad(lambda r2: rho2(r2) * 4 * math.pi * r2, r1, np.inf)
        v = q_in / r1 + v_out
        return rho1(r1) * v * 4 * math.pi * r1**2
    result, _ = quad(integrand, 0, np.inf)
    return result

alpha = 1.0

# s 轨道归一化
Ns_sq = (2 * alpha / math.pi)**1.5
Ns = math.sqrt(Ns_sq)
rho_s = lambda r: Ns_sq * math.exp(-2 * alpha * r**2)
print(f'Ns^2 = {Ns_sq:.10f}')

# p 轨道归一化： phi_p = Np * x * exp(-alpha r^2)
# ∫ |phi_p|^2 d3r = Np^2 * ∫ x^2 exp(-2α r^2) d3r = 1
# ∫ x^2 exp(-2α r^2) d3r = (1/2) π^(3/2) / (2α)^(5/2)
beta = 2 * alpha
integral_x2 = 0.5 * math.pi**1.5 / beta**2.5
Np_sq = 1.0 / integral_x2
Np = math.sqrt(Np_sq)
print(f'Np^2 = {Np_sq:.10f}')
print(f'Np = {Np:.10f}')

rho_p_total = lambda r: Np_sq * r**2 * math.exp(-2 * alpha * r**2)

print()
print('=== 验证归一化 ===')
s_norm, _ = quad(lambda r: rho_s(r) * 4 * math.pi * r**2, 0, np.inf)
p_norm, _ = quad(lambda r: rho_p_total(r) * 4 * math.pi * r**2, 0, np.inf)
print(f'∫ rho_s d3r = {s_norm:.10f} (预期: 1)')
print(f'∫ rho_p_total d3r = {p_norm:.10f} (预期: 3，因为3个p轨道)')

print()
print('=== 库仑积分 ===')

J_ss = coulomb_spherical(rho_s, rho_s)
print(f'(ss|ss) = {J_ss:.10f}')

J_sp = coulomb_spherical(rho_s, rho_p_total)
J_ss_pp = J_sp / 3.0
print(f'(ss|p_i p_i) = {J_ss_pp:.10f}')
print(f'比值 (ss|pp)/(ss|ss) = {J_ss_pp/J_ss:.10f}')

J_pp = coulomb_spherical(rho_p_total, rho_p_total)
print()
print(f'J(p_total|p_total) = {J_pp:.10f}')

# J(p_total|p_total) = sum_{ij} (p_i p_i | p_j p_j)
# 所以 sum_{ij} (p_i p_i | p_j p_j) = J_pp
# 而 (p_x p_x|p_x p_x) + 2*(p_x p_x|p_y p_y) = ??? 不对
# 实际上 sum_{ij} (p_i p_i|p_j p_j) = 3*(pp|pp)_diag + 6*(pp|pp)_cross
# 因为对角项有3个 (i=j)，非对角项有6个 (i≠j，i和j各有3种选择，3*2=6)
# 所以 J_pp = 3*J_pp_diag + 6*J_pp_cross

from calchem.basis.gto import GTOShell
from calchem.integrals.two_electron import shell_eri

origin = np.zeros(3)
p_shell = GTOShell('p', [alpha], [1.0], origin=origin)
pp_blk = shell_eri(p_shell, p_shell, p_shell, p_shell)

J_pp_diag = pp_blk[0,0,0,0]
J_pp_cross = pp_blk[0,0,1,1]
sum_ij = 3 * J_pp_diag + 6 * J_pp_cross

print()
print('=== 我们的递推结果 ===')
print(f'  (px px|px px) = {J_pp_diag:.10f}')
print(f'  (px px|py py) = {J_pp_cross:.10f}')
print(f'  sum_ij (p_i p_i|p_j p_j) = {sum_ij:.10f}')
print(f'  J(p_total|p_total) 球对称 = {J_pp:.10f}')
print(f'  比值: {sum_ij / J_pp:.10f}')

print()
print('=== 详细对比 ===')
s_shell = GTOShell('s', [alpha], [1.0], origin=origin)
ss_blk = shell_eri(s_shell, s_shell, s_shell, s_shell)
sp_blk = shell_eri(s_shell, s_shell, p_shell, p_shell)

print(f'(ss|ss):')
print(f'  我们的结果: {ss_blk[0,0,0,0]:.10f}')
print(f'  球对称: {J_ss:.10f}')
print(f'  比值: {ss_blk[0,0,0,0]/J_ss:.10f}')

print()
print(f'(ss|pp):')
print(f'  我们的结果: {sp_blk[0,0,0,0]:.10f}')
print(f'  球对称: {J_ss_pp:.10f}')
print(f'  比值: {sp_blk[0,0,0,0]/J_ss_pp:.10f}')
