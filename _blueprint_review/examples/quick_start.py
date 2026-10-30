#!/usr/bin/env python
"""
快速入门示例：水分子单点能计算
"""
from calchem import Molecule
from calchem.api import single_point


def main():
    water_xyz = """
3
water
O  0.0  0.0  0.0
H  0.0  0.0  0.96
H  0.0  0.96  0.0
"""

    print("=== CalChem Quick Start ===")
    print()

    mol = Molecule.from_xyz(water_xyz)
    print(f"Molecule: {mol.formula}")
    print(f"Atoms: {mol.natoms}")
    print(f"Charge: {mol.charge}, Multiplicity: {mol.multiplicity}")
    print()

    print("Running B3LYP/STO-3G single point calculation...")
    result = single_point(mol, method="B3LYP", basis="sto-3g")

    print()
    print("=== Results ===")
    print(f"Converged: {result.converged}")
    print(f"SCF cycles: {result.scf_cycles}")
    print(f"Energy: {result.energy:.8f} Hartree")
    print(f"Dipole moment: {result.dipole_moment:.4f} Debye")


if __name__ == "__main__":
    main()
