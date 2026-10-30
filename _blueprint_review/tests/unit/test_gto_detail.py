"""详细检查GTO数值"""
import numpy as np
import math
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet

mol = Molecule(['H'], [[0.0, 0.0, 0.0]])
basis_set = BasisSet('sto-3g')
shells = basis_set.get_all_shells(mol.to_bohr())

shell = shells[0]
print(f"shell_type={shell.shell_type}")
print(f"exponents={shell.exponents}")
print(f"coefficients={shell.coefficients}")
print(f"_contract_coeffs={shell._contract_coeffs}")
print(f"norm_constants={shell.norm_constants}")

# 手动计算r=0处的phi值
phi0 = 0.0
for alpha, c_raw, c_norm, norm in zip(shell.exponents, shell.coefficients, shell._contract_coeffs, shell.norm_constants):
    phi0_prim = norm  # r=0时exp(-alpha*r^2)=1, dx=dy=dz=0对于s轨道, 但l=m=n=0所以dx^0=1
    phi0 += c_norm * phi0_prim
    print(f"  alpha={alpha:.6f} c_raw={c_raw:.6f} c_norm={c_norm:.6f} norm={norm:.6f} phi0_prim={phi0_prim:.6f}")

print(f"manual phi(0) = {phi0:.6f}")

# 对比: 用原始系数和归一化常数
phi0_raw = 0.0
for alpha, c_raw, norm in zip(shell.exponents, shell.coefficients, shell.norm_constants):
    phi0_raw += c_raw * norm
print(f"raw phi(0) = {phi0_raw:.6f}")

# 计算解析归一化
S = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        S[i,j] = (math.pi/(shell.exponents[i]+shell.exponents[j]))**1.5 * shell.norm_constants[i] * shell.norm_constants[j]
print(f"overlap matrix of primitives:\n{S}")
total = np.dot(shell.coefficients, np.dot(S, shell.coefficients))
print(f"total_overlap with raw coeffs = {total:.8f}")
print(f"normalization_factor = {1.0/math.sqrt(total):.8f}")
