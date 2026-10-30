import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
print("numpy:", np.__version__)

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.basis.gto import GTOShell
from calchem.integrals.two_electron import eri_ssss, shell_eri

print("测试 ssss 积分...")
r = np.array([0.0, 0.0, 0.0])
r2 = np.array([0.0, 0.0, 1.4])

e = eri_ssss(1.0, 1.0, 1.0, 1.0, r, r, r, r)
print("  (1s 1s | 1s 1s) same center =", e)

e = eri_ssss(1.0, 1.0, 1.0, 1.0, r, r, r2, r2)
print("  (1s 1s | 1s 1s) different center =", e)

print("\n测试 shell_eri (STO-3G H)...")
exps = [3.42525091, 0.62391373, 0.16885540]
coeffs = [0.15432897, 0.53532814, 0.44463454]
s1 = GTOShell('s', exps, coeffs, origin=r)
s2 = GTOShell('s', exps, coeffs, origin=r2)

eri_1111 = shell_eri(s1, s1, s1, s1)[0, 0, 0, 0]
eri_1122 = shell_eri(s1, s1, s2, s2)[0, 0, 0, 0]
eri_1212 = shell_eri(s1, s2, s1, s2)[0, 0, 0, 0]

print(f"  (11|11) = {eri_1111:.6f}  (参考: 0.7746)")
print(f"  (11|22) = {eri_1122:.6f}  (参考: 0.5696)")
print(f"  (12|12) = {eri_1212:.6f}")

print("\nERI 测试通过!")
