"""测试基函数解析归一化"""
import numpy as np
import math
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap

mol = Molecule(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
basis_set = BasisSet('sto-3g')
S = compute_overlap(basis_set, mol.to_bohr())
print("S matrix:")
print(S)
print("S diag:", np.diag(S))

# 手动验证第一个基函数的归一化
shells = basis_set.get_all_shells(mol.to_bohr())
for i, shell in enumerate(shells):
    alpha = shell.exponents
    c = shell._contract_coeffs
    total = 0.0
    for j in range(len(alpha)):
        for k in range(len(alpha)):
            N_j = (2.0 * alpha[j] / math.pi) ** 0.75
            N_k = (2.0 * alpha[k] / math.pi) ** 0.75
            total += c[j] * c[k] * N_j * N_k * (math.pi / (alpha[j] + alpha[k])) ** 1.5
    print(f"Shell {i} manual norm = {total:.8f}")
    print(f"  exponents={alpha}")
    print(f"  coeffs={c}")
