from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .molecule import Molecule
from ..utils.exceptions import InputError


class Symmetry:
    def __init__(self, mol: Molecule):
        self.mol = mol

    def get_point_group(self, tolerance: float = 1e-3) -> str:
        try:
            import spglib

            lattice = np.eye(3) * 100.0
            positions = self.mol.coordinates / 100.0
            numbers = self.mol.atomic_numbers

            cell = (lattice, positions, numbers)

            dataset = spglib.get_symmetry_dataset(cell, symprec=tolerance)
            if dataset is not None:
                return dataset["international"]

            return "C1"
        except ImportError:
            return "C1 (spglib not available)"

    def get_symmetry_operations(self, tolerance: float = 1e-3) -> Optional[dict]:
        try:
            import spglib

            lattice = np.eye(3) * 100.0
            positions = self.mol.coordinates / 100.0
            numbers = self.mol.atomic_numbers

            cell = (lattice, positions, numbers)

            dataset = spglib.get_symmetry_dataset(cell, symprec=tolerance)
            if dataset is not None:
                return {
                    "rotations": dataset["rotations"],
                    "translations": dataset["translations"],
                    "n_operations": len(dataset["rotations"]),
                }
            return None
        except ImportError:
            return None

    def find_equivalent_atoms(self, tolerance: float = 1e-3) -> Optional[list]:
        try:
            import spglib

            lattice = np.eye(3) * 100.0
            positions = self.mol.coordinates / 100.0
            numbers = self.mol.atomic_numbers

            cell = (lattice, positions, numbers)

            dataset = spglib.get_symmetry_dataset(cell, symprec=tolerance)
            if dataset is not None:
                return dataset["equivalent_atoms"].tolist()
            return None
        except ImportError:
            return None

    def __repr__(self) -> str:
        return f"Symmetry(mol={self.mol})"
