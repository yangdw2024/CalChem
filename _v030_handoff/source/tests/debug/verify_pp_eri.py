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
Np = (2 * alpha / math.pi)**0.75 * math.sqrt(2 * alpha)
rho_p_total = lambda r: (2*alpha/math.pi)**1.5 * r**2 * math.exp(-2 * alpha * r**2)

print('=== (pp|pp) 总库仑积分 ===')
print()

J_pp_total = coulomb_spherical(rho_p_total, rho_p_total)
print(f'3个p轨道总库仑能 J(p_total|p_total) = {J_pp_total:.10f}')

from calchem.basis.gto import GTOShell
from calchem.integrals.two_electron import shell_eri

origin = np.zeros(3)
p_shell = GTOShell('p', [alpha], [1.0], origin=origin)
pp_blk = shell_eri(p_shell, p_shell, p_shell, p_shell)

print()
print('我们的递推结果:')
print(f'  (px px|px px) = {pp_blk[0,0,0,0]:.10f}')
print(f'  (px px|py py) = {pp_blk[0,0,1,1]:.10f}')
print(f'  (px py|px py) = {pp_blk[0,1,0,1]:.10f}')

total_from_sum = 0.0
for i in range(3):
    for j in range(3):
        total_from_sum += pp_blk[i,i,j,j]
print()
print(f'  sum_{{ij}} (p_i p_i | p_j p_j) = {total_from_sum:.10f}')
print(f'  J(p_total|p_total) 球对称 = {J_pp_total:.10f}')
print(f'  比值: {total_from_sum / J_pp_total:.10f}')

print()
print('=== 验证 p-p 积分的各种比值 ===')
ss = 1.1283791671
print(f'(pp|pp)_diag / (ss|ss) = {pp_blk[0,0,0,0] / ss:.10f}')
print(f'(pp|pp)_cross / (ss|ss) = {pp_blk[0,0,1,1] / ss:.10f}')
print(f'(pp|pp)_exchange / (ss|ss) = {pp_blk[0,1,0,1] / ss:.10f}')

print()
print('预期的关系:')
print('  (px px|py py) + 2*(px py|px py) = (px px|px px) ???')
print(f'  左边: {pp_blk[0,0,1,1] + 2*pp_blk[0,1,0,1]:.10f}')
print(f'  右边: {pp_blk[0,0,0,0]:.10f}')
