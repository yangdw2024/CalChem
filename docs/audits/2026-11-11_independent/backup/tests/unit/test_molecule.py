import pytest
import numpy as np
from calchem.mol.molecule import Molecule


def test_molecule_from_xyz_string():
    xyz = """3
water
O  0.0  0.0  0.0
H  0.0  0.0  0.96
H  0.0  0.96  0.0
"""
    mol = Molecule.from_xyz(xyz)
    assert mol.natoms == 3
    assert mol.symbols == ["O", "H", "H"]
    assert mol.charge == 0
    assert mol.multiplicity == 1


def test_molecule_from_xyz_with_charge():
    xyz = """3
water
O  0.0  0.0  0.0
H  0.0  0.0  0.96
H  0.0  0.96  0.0
"""
    mol = Molecule.from_xyz(xyz, charge=1, multiplicity=2)
    assert mol.charge == 1
    assert mol.multiplicity == 2

    mol = Molecule(symbols=["O", "H", "H"],
                   coords=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]],
                   charge=1, multiplicity=2)
    assert mol.charge == 1
    assert mol.multiplicity == 2


def test_molecule_formula():
    mol = Molecule(symbols=["O", "H", "H"],
                   coords=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]])
    assert mol.formula == "H2O"


def test_bond_length():
    mol = Molecule(symbols=["O", "H", "H"],
                   coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.96, 0.0]])
    d = mol.bond_length(0, 1)
    assert abs(d - 0.96) < 1e-6


def test_bond_angle():
    mol = Molecule(symbols=["O", "H", "H"],
                   coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.96, 0.0]])
    angle = mol.bond_angle(1, 0, 2)
    assert abs(angle - 90.0) < 1e-3
