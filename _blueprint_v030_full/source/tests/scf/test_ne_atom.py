import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF

print("=" * 60)
print("测试 Ne 原子 STO-3G RHF")
print("参考值: ~ -126.0 Hartree? (需要确认)")
print("=" * 60)

mol = Molecule(["Ne"], [[0,0,0]], units="bohr")

mf = RHF(mol, basis="sto-3g", max_cycles=50, conv_tol=1e-8, diis=False, verbose=2)
e_tot = mf.kernel()

print(f"\n最终能量: {e_tot:.6f} Eh")
print(f"收敛: {mf.converged}")
print(f"迭代次数: {mf.n_cycles}")

print(f"\n分子轨道能量:")
for i, e in enumerate(mf.mo_energy):
    occ = "occ" if mf.mo_occ[i] > 0 else "vir"
    print(f"  {i+1:2d}. {e:12.6f}  {occ}")

# 计算各能量分量
print(f"\n能量分量:")
print(f"  核排斥能: {mf.e_nn:.6f}")
print(f"  单电子 + 双电子: {mf.e_tot - mf.e_nn:.6f}")
