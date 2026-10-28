import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF, UHF

print("=" * 70, flush=True)
print("测试 H2 STO-3G RHF (R = 1.4 bohr)", flush=True)
print("参考值 (Szabo & Ostlund): -1.1167143251893824 Hartree", flush=True)
print("=" * 70, flush=True)

mol = Molecule(
    ["H", "H"],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
    units="bohr",
)

mf = RHF(mol, basis="sto-3g", max_cycles=50, conv_tol=1e-8, verbose=2)
e_tot = mf.kernel()

ref = -1.1167143251893824
error = abs(e_tot - ref)

print(flush=True)
print(f"计算能量: {e_tot:.12f} Eh", flush=True)
print(f"参考能量: {ref:.12f} Eh", flush=True)
print(f"误差:     {error:.2e} Eh", flush=True)

if error < 1e-6:
    print("✓ H2 STO-3G RHF 测试通过!", flush=True)
else:
    print("✗ H2 STO-3G RHF 测试失败!", flush=True)

print(flush=True)
print("=" * 70, flush=True)
print("测试 H2 STO-3G UHF (R = 1.4 bohr, 单重态)", flush=True)
print("=" * 70, flush=True)

mol2 = Molecule(
    ["H", "H"],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
    units="bohr",
    multiplicity=1,
)

mf2 = UHF(mol2, basis="sto-3g", max_cycles=50, conv_tol=1e-8, verbose=2)
e_tot2 = mf2.kernel()

error2 = abs(e_tot2 - ref)
print(flush=True)
print(f"UHF 能量: {e_tot2:.12f} Eh", flush=True)
print(f"RHF 参考: {ref:.12f} Eh", flush=True)
print(f"误差:     {error2:.2e} Eh", flush=True)

if error2 < 1e-5:
    print("✓ H2 STO-3G UHF 测试通过!", flush=True)
else:
    print("✗ H2 STO-3G UHF 测试失败!", flush=True)

print(flush=True)
print("=" * 70, flush=True)
print("所有 H2 测试完成!", flush=True)
print("=" * 70, flush=True)
