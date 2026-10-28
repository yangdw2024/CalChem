import numpy as np

from .constants import (
    BOHR_TO_ANGSTROM,
    ANGSTROM_TO_BOHR,
    HARTREE_TO_EV,
    EV_TO_HARTREE,
    HARTREE_TO_KCAL_MOL,
    KCAL_MOL_TO_HARTREE,
)


def bohr_to_angstrom(value):
    return value * BOHR_TO_ANGSTROM


def angstrom_to_bohr(value):
    return value * ANGSTROM_TO_BOHR


def hartree_to_ev(value):
    return value * HARTREE_TO_EV


def ev_to_hartree(value):
    return value * EV_TO_HARTREE


def hartree_to_kcal_mol(value):
    return value * HARTREE_TO_KCAL_MOL


def kcal_mol_to_hartree(value):
    return value * KCAL_MOL_TO_HARTREE


def hartree_to_wavenumber(value):
    return value * 219474.6313702


def wavenumber_to_hartree(value):
    return value / 219474.6313702


def convert(value, from_unit, to_unit):
    converters = {
        ("bohr", "angstrom"): bohr_to_angstrom,
        ("angstrom", "bohr"): angstrom_to_bohr,
        ("hartree", "ev"): hartree_to_ev,
        ("ev", "hartree"): ev_to_hartree,
        ("hartree", "kcal/mol"): hartree_to_kcal_mol,
        ("kcal/mol", "hartree"): kcal_mol_to_hartree,
        ("hartree", "cm^-1"): hartree_to_wavenumber,
        ("cm^-1", "hartree"): wavenumber_to_hartree,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key not in converters:
        raise ValueError(f"Unknown conversion: {from_unit} -> {to_unit}")

    return converters[key](value)
