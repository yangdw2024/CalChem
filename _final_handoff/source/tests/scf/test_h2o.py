import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
import time

from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF

print("=" * 70, flush=True)
print("测试 H2O STO-3G RHF", flush=True)
print("参考值 (标准): ~ -74.94208 Hartree", flush=True)
print("=" * 70, flush=True)

mol = Molecule(
    ["O", "H", "H"],
    [
        [0.0000000, 0.0000000, 0.0000000],
        [0.0000000, -0.7570000, 0.5860000],
        [0.0000000, 0.7570000, 0.5860000],
    ],
    units="angstrom",
)

print(f"水分子 (单位: {mol.units})", flush=True)
print(f"  原子数: {mol.natom}", flush=True)
print(f"  电子数: {mol.nelectron}", flush=True)

t0 = time.time()
mf = RHF(mol, basis="sto-3g", max_cycles=100, conv_tol=1e-8, verbose=2)
e_tot = mf.kernel()
dt = time.time() - t0

ref = -74.94208
error = abs(e_tot - ref)

print(flush=True)
print(f"计算能量: {e_tot:.6f} Eh", flush=True)
print(f"参考能量: {ref:.6f} Eh", flush=True)
print(f"误差:     {error:.6f} Eh", flush=True)
print(f"计算时间: {dt:.2f} 秒", flush=True)
print(f"收敛:     {mf.converged}", flush=True)
print(f"迭代次数: {mf.n_cycles}", flush=True)

if error < 0.1:
    print("✓ H2O STO-3G RHF 测试通过 (在 0.1 Hartree 范围内)!", flush=True)
else:
    print("✗ H2O STO-3G RHF 测试失败!", flush=True)

print(flush=True)
print("分子轨道能量:", flush=True)
for i, e in enumerate(mf.mo_energy):
    occ = "occ" if mf.mo_occ[i] > 0 else "vir"
    print(f"  {i+1:2d}. {e:12.6f}  {occ}", flush=True)

print(flush=True)
print("=" * 70, flush=True)
print("H2O 测试完成!", flush=True)
print("=" * 70, flush=True)
