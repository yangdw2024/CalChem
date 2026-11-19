import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF, UHF


def test_h2_sto3g_rhf():
    print("=" * 70)
    print("测试 H2 STO-3G RHF (R = 1.4 bohr)")
    print("参考值 (Szabo & Ostlund): -1.1167143251893824 Hartree")
    print("=" * 70)

    mol = Molecule(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        units="bohr",
    )

    mf = RHF(mol, basis="sto-3g", max_cycles=50, conv_tol=1e-8, verbose=2)
    e_tot = mf.kernel()

    ref = -1.1167143251893824
    error = abs(e_tot - ref)

    print()
    print(f"计算能量: {e_tot:.12f} Eh")
    print(f"参考能量: {ref:.12f} Eh")
    print(f"误差:     {error:.2e} Eh")

    if error < 1e-6:
        print("✓ H2 STO-3G RHF 测试通过!")
    else:
        print("✗ H2 STO-3G RHF 测试失败!")

    assert error < 1e-5, f"H2 RHF 能量误差太大: {error}"
    return e_tot


def test_h2_sto3g_uhf():
    print()
    print("=" * 70)
    print("测试 H2 STO-3G UHF (R = 1.4 bohr)")
    print("参考值 (闭壳层单重态，应与 RHF 一致)")
    print("=" * 70)

    mol = Molecule(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        units="bohr",
        multiplicity=1,
    )

    mf = UHF(mol, basis="sto-3g", max_cycles=50, conv_tol=1e-8, verbose=2)
    e_tot = mf.kernel()

    ref_rhf = -1.1167143251893824
    error = abs(e_tot - ref_rhf)

    print()
    print(f"UHF 能量: {e_tot:.12f} Eh")
    print(f"RHF 参考: {ref_rhf:.12f} Eh")
    print(f"误差:     {error:.2e} Eh")

    if error < 1e-5:
        print("✓ H2 STO-3G UHF 测试通过!")
    else:
        print("✗ H2 STO-3G UHF 测试失败!")

    return e_tot


def test_h2o_sto3g_rhf():
    print()
    print("=" * 70)
    print("测试 H2O STO-3G RHF")
    print("参考值 (标准参考): ~ -74.94208 Hartree")
    print("=" * 70)

    mol = Molecule(
        ["O", "H", "H"],
        [
            [0.0000000, 0.0000000, 0.0000000],
            [0.0000000, -0.7570000, 0.5860000],
            [0.0000000, 0.7570000, 0.5860000],
        ],
        units="angstrom",
    )

    mf = RHF(mol, basis="sto-3g", max_cycles=100, conv_tol=1e-8, verbose=2)
    e_tot = mf.kernel()

    ref = -74.94208
    error = abs(e_tot - ref)

    print()
    print(f"计算能量: {e_tot:.12f} Eh")
    print(f"参考能量: {ref:.6f} Eh")
    print(f"误差:     {error:.6f} Eh")

    if error < 0.1:
        print("✓ H2O STO-3G RHF 测试通过 (在 0.1 Hartree 范围内)!")
    else:
        print("✗ H2O STO-3G RHF 测试失败!")

    return e_tot


if __name__ == "__main__":
    test_h2_sto3g_rhf()
    test_h2_sto3g_uhf()
    test_h2o_sto3g_rhf()
    print()
    print("=" * 70)
    print("所有测试完成!")
    print("=" * 70)
