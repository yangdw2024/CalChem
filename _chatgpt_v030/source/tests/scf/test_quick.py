"""快速验证 Direct SCF 修复。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.scf.hf import RHF

basis = BasisSet("STO-3G")

# H2 at 1.4 Bohr
mol = Molecule(
    symbols=["H", "H"],
    coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
    units="bohr",
)

print("=== H2 Standard SCF ===")
mf_std = RHF(mol, basis)
e_std = mf_std.kernel()

print("\n=== H2 Direct SCF ===")
mf_dir = RHF(mol, basis, direct_scf=True)
e_dir = mf_dir.kernel()

print(f"\n标准SCF: {e_std:.12f}")
print(f"Direct SCF: {e_dir:.12f}")
print(f"差异: {abs(e_dir - e_std):.2e}")

# 对比Fock矩阵
f_std = mf_std.fock
f_dir = mf_dir.fock
print(f"\nFock矩阵差异:")
print(np.max(np.abs(f_std - f_dir)))
