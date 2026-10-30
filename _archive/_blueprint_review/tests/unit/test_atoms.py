"""
用简单原子来验证积分的正确性。
"""
import sys
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.hf import nuclear_repulsion, RHF


def test_he_sto3g():
    """测试He原子 STO-3G。"""
    print("=" * 70)
    print("He 原子 STO-3G RHF")
    print("参考值: ~ -2.8078 Hartree (Szabo 书 p. 195)")
    print("=" * 70)
    print()
    
    mol = Molecule(["He"], [[0.0, 0.0, 0.0]], units="bohr")
    mf = RHF(mol, basis="sto-3g", max_cycles=50, verbose=0)
    e = mf.kernel()
    
    print(f"计算能量: {e:.6f} Eh")
    print(f"参考能量: -2.8078 Eh")
    print(f"误差: {abs(e + 2.8078):.6f} Eh")
    print()
    
    if abs(e + 2.8078) < 0.01:
        print("✓ He 原子测试通过!")
    else:
        print("✗ He 原子测试失败!")
    print()


def test_be_sto3g():
    """测试Be原子 STO-3G (只有s轨道)。"""
    print("=" * 70)
    print("Be 原子 STO-3G RHF")
    print("参考值: ~ -14.3519 Hartree ( approximate )")
    print("=" * 70)
    print()
    
    mol = Molecule(["Be"], [[0.0, 0.0, 0.0]], units="bohr")
    mf = RHF(mol, basis="sto-3g", max_cycles=50, verbose=1)
    try:
        e = mf.kernel()
        print(f"计算能量: {e:.6f} Eh")
        print(f"单电子能量: {mf.e_1e:.6f} Eh")
        print(f"双电子能量: {mf.e_2e:.6f} Eh")
    except Exception as ex:
        print(f"错误: {ex}")
    print()


def test_ne_sto3g():
    """测试Ne原子 STO-3G (有p轨道)。"""
    print("=" * 70)
    print("Ne 原子 STO-3G RHF")
    print("参考值: ~ -127.07 Hartree ( approximate )")
    print("=" * 70)
    print()
    
    mol = Molecule(["Ne"], [[0.0, 0.0, 0.0]], units="bohr")
    mf = RHF(mol, basis="sto-3g", max_cycles=100, verbose=1)
    try:
        e = mf.kernel()
        print(f"计算能量: {e:.6f} Eh")
        print(f"单电子能量: {mf.e_1e:.6f} Eh")
        print(f"双电子能量: {mf.e_2e:.6f} Eh")
    except Exception as ex:
        print(f"错误: {ex}")
    print()


if __name__ == '__main__':
    test_he_sto3g()
    test_be_sto3g()
    test_ne_sto3g()
