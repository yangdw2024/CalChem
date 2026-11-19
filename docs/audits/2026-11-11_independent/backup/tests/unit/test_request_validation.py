"""Unit tests for MoleculeSpec validation (v0.2.0 P0)."""

import pytest
from calchem.core.request import MoleculeSpec


def test_valid_water():
    mol = MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
        charge=0, multiplicity=1,
    )
    assert mol.natom == 3


def test_unknown_element_rejected():
    with pytest.raises(ValueError, match="Unknown element"):
        MoleculeSpec(
            symbols=["Xx"],
            coordinates_angstrom=[[0, 0, 0]],
        )


def test_nan_coordinate_rejected():
    with pytest.raises(ValueError, match="finite"):
        MoleculeSpec(
            symbols=["H"],
            coordinates_angstrom=[[float("nan"), 0, 0]],
        )


def test_inf_coordinate_rejected():
    with pytest.raises(ValueError, match="finite"):
        MoleculeSpec(
            symbols=["H"],
            coordinates_angstrom=[[float("inf"), 0, 0]],
        )


def test_wrong_coord_dimension():
    with pytest.raises(ValueError, match="x,y,z"):
        MoleculeSpec(
            symbols=["H"],
            coordinates_angstrom=[[0, 0]],
        )


def test_symbols_coords_mismatch():
    with pytest.raises(ValueError, match="count"):
        MoleculeSpec(
            symbols=["H", "H"],
            coordinates_angstrom=[[0, 0, 0]],
        )


def test_negative_electron_charge():
    with pytest.raises(ValueError, match="Negative electron"):
        MoleculeSpec(
            symbols=["H"],
            coordinates_angstrom=[[0, 0, 0]],
            charge=5,
        )


def test_multiplicity_too_high():
    with pytest.raises(ValueError, match="unpaired"):
        MoleculeSpec(
            symbols=["H", "H"],
            coordinates_angstrom=[[0, 0, 0], [0, 0, 1]],
            multiplicity=5,
        )


def test_h_atom_doublet_correct():
    """H atom: 1 electron, multiplicity=2 is correct (1 unpaired)."""
    mol = MoleculeSpec(
        symbols=["H"],
        coordinates_angstrom=[[0, 0, 0]],
        multiplicity=2,
    )
    assert mol.multiplicity == 2


def test_h_atom_singlet_wrong():
    """H atom: 1 electron, multiplicity=1 (0 unpaired) is wrong."""
    with pytest.raises(ValueError, match="odd unpaired"):
        MoleculeSpec(
            symbols=["H"],
            coordinates_angstrom=[[0, 0, 0]],
            multiplicity=1,
        )


def test_element_case_insensitive():
    mol = MoleculeSpec(
        symbols=["o", "h", "h"],
        coordinates_angstrom=[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
    )
    assert mol.symbols == ["O", "H", "H"]
