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

Ns = (2 * alpha / math.pi)**0.75
rho_s = lambda r: Ns**2 * math.exp(-2 * alpha * r**2)

beta = 2 * alpha
integral_r4 = 3 * math.sqrt(math.pi) / (8 * beta**2.5)
Np_sq = 1.0 / (integral_r4 * (2.0/3.0) * 2 * math.pi)
Np = math.sqrt(Np_sq)
print(f'Ns = {Ns:.10f}')
print(f'Np = {Np:.10f}')

rho_p_total = lambda r: Np**2 * r**2 * math.exp(-2 * alpha * r**2)

J_ss = coulomb_spherical(rho_s, rho_s)
print()
print('=== 球对称库仑积分 (α=1.0) ===')
print(f'(ss|ss) = {J_ss:.10f}')

J_sp = coulomb_spherical(rho_s, rho_p_total)
J_ss_pp = J_sp / 3.0
print(f'(ss|p_i p_i) = {J_ss_pp:.10f}')

print()
print('比值:')
print(f'(ss|pp) / (ss|ss) = {J_ss_pp / J_ss:.10f}')

from calchem.basis.gto import GTOShell
from calchem.integrals.two_electron import shell_eri

origin = np.zeros(3)
s_shell = GTOShell('s', [alpha], [1.0], origin=origin)
p_shell = GTOShell('p', [alpha], [1.0], origin=origin)

ss_blk = shell_eri(s_shell, s_shell, s_shell, s_shell)
sp_blk = shell_eri(s_shell, s_shell, p_shell, p_shell)

print()
print('=== 我们的递推结果 ===')
print(f'(ss|ss) = {ss_blk[0,0,0,0]:.10f}')
print(f'(ss|p_i p_i) = {sp_blk[0,0,0,0]:.10f}')
print(f'(ss|pp) / (ss|ss) = {sp_blk[0,0,0,0] / ss_blk[0,0,0,0]:.10f}')

print()
print('=== 对比 ===')
print(f'(ss|ss) 比值: {ss_blk[0,0,0,0] / J_ss:.10f}')
print(f'(ss|pp) 比值: {sp_blk[0,0,0,0] / J_ss_pp:.10f}')
