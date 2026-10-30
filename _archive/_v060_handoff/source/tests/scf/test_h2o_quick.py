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
try:
    mf = RHF(mol, basis="sto-3g", max_cycles=50, conv_tol=1e-6, diis=False, verbose=2)
    e_tot = mf.kernel()
except Exception as e:
    print(f"Exception: {e}", flush=True)
    e_tot = getattr(mf, 'e_tot', 0.0)
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

print(flush=True)
print("=" * 70, flush=True)
