"""验证 H2O Direct SCF 修复。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.scf.hf import RHF

basis = BasisSet("STO-3G")

# H2O Szabo geometry (Bohr)
mol = Molecule(
    symbols=["O", "H", "H"],
    coordinates=np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.432, 0.0],
        [1.112, -0.716, 0.0],
    ]),
    units="bohr",
)

print("=== H2O Standard SCF ===")
mf_std = RHF(mol, basis)
e_std = mf_std.kernel()

print("\n=== H2O Direct SCF ===")
mf_dir = RHF(mol, basis, direct_scf=True)
e_dir = mf_dir.kernel()

print(f"\n标准SCF: {e_std:.12f}")
print(f"Direct SCF: {e_dir:.12f}")
print(f"差异: {abs(e_dir - e_std):.2e}")

# 对比Fock矩阵
f_std = mf_std.fock
f_dir = mf_dir.fock
print(f"\nFock矩阵最大差异: {np.max(np.abs(f_std - f_dir)):.2e}")

# MP2 测试
print("\n=== MP2 Test ===")
from calchem.post_hf.mp2 import RMP2
try:
    mp2 = RMP2(mf_std)
    e_mp2 = mp2.kernel()
    print(f"MP2 关联能: {mp2.e_corr:.10f}")
    print(f"MP2 总能量: {e_mp2:.10f}")
except Exception as e:
    import traceback
    traceback.print_exc()

# CCSD 测试
print("\n=== CCSD Test ===")
from calchem.post_hf.ccsd import CCSD
try:
    ccsd = CCSD(mf_std, frozen_core=True)
    e_ccsd = ccsd.run()
    print(f"CCSD 总能量: {e_ccsd:.10f}")
except Exception as e:
    import traceback
    traceback.print_exc()
