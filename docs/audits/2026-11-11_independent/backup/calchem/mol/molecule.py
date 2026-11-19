from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple, Union

import numpy as np

from ..utils.constants import ATOMIC_NUMBERS, ELEMENTS
from ..utils.exceptions import InputError


class Molecule:
    def __init__(
        self,
        symbols: List[str],
        coordinates: Union[List[List[float]], np.ndarray] = None,
        charge: int = 0,
        multiplicity: int = 1,
        units: str = "angstrom",
        coords: Union[List[List[float]], np.ndarray] = None,
    ):
        if coords is not None:
            coordinates = coords

        if coordinates is None:
            raise InputError("coordinates or coords must be provided")

        if len(symbols) != len(coordinates):
            raise InputError(
                f"Number of symbols ({len(symbols)}) does not match "
                f"number of coordinates ({len(coordinates)})"
            )

        self.symbols = [s.capitalize() for s in symbols]
        self.coordinates = np.array(coordinates, dtype=float)
        self.charge = charge
        self.multiplicity = multiplicity
        self.units = units.lower()

        if self.units not in ("angstrom", "bohr"):
            raise InputError(f"Unknown units: {units}. Use 'angstrom' or 'bohr'.")

        if self.coordinates.shape != (len(symbols), 3):
            raise InputError(
                f"Coordinates shape {self.coordinates.shape} is not (n, 3)"
            )

        for sym in self.symbols:
            if sym not in ATOMIC_NUMBERS:
                raise InputError(f"Unknown element symbol: {sym}")

    @property
    def natom(self) -> int:
        return len(self.symbols)

    @property
    def natoms(self) -> int:
        return self.natom

    @property
    def atomic_numbers(self) -> List[int]:
        return [ATOMIC_NUMBERS[s] for s in self.symbols]

    @property
    def formula(self) -> str:
        counts = Counter(self.symbols)
        hill_order = ["C", "H"]
        others = sorted([s for s in counts.keys() if s not in hill_order])

        formula_parts = []
        for sym in hill_order + others:
            if sym in counts:
                count = counts[sym]
                formula_parts.append(f"{sym}{count if count > 1 else ''}")

        return "".join(formula_parts)

    @property
    def molecular_weight(self) -> float:
        try:
            from mendeleev import element

            return sum(element(s).atomic_weight for s in self.symbols)
        except ImportError:
            return 0.0

    @property
    def nelectron(self) -> int:
        return sum(self.atomic_numbers) - self.charge

    @property
    def na(self) -> int:
        if (self.nelectron + self.multiplicity - 1) % 2 != 0:
            raise InputError(
                f"Invalid multiplicity {self.multiplicity} for "
                f"{self.nelectron} electrons"
            )
        return (self.nelectron + self.multiplicity - 1) // 2

    @property
    def nb(self) -> int:
        return (self.nelectron - self.multiplicity + 1) // 2

    @classmethod
    def from_xyz(cls, xyz_str: str, charge: int = 0, multiplicity: int = 1) -> "Molecule":
        try:
            with open(xyz_str, "r") as f:
                xyz_str = f.read()
        except (OSError, FileNotFoundError):
            pass

        lines = xyz_str.strip().split("\n")
        if not lines:
            raise InputError("Empty XYZ string")

        try:
            natom = int(lines[0].strip())
        except ValueError:
            raise InputError(f"Invalid number of atoms in XYZ: {lines[0]}")

        symbols = []
        coordinates = []
        start_idx = 2

        for i, line in enumerate(lines[start_idx : start_idx + natom]):
            parts = line.strip().split()
            if len(parts) < 4:
                raise InputError(f"Invalid XYZ line {start_idx + i + 1}: {line}")
            symbol = parts[0].capitalize()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            symbols.append(symbol)
            coordinates.append([x, y, z])

        if len(symbols) != natom:
            raise InputError(
                f"Expected {natom} atoms, but found {len(symbols)}"
            )

        return cls(symbols, coordinates, charge=charge, multiplicity=multiplicity)

    @classmethod
    def from_xyz_file(cls, filename: str, charge: int = 0, multiplicity: int = 1) -> "Molecule":
        with open(filename, "r") as f:
            xyz_str = f.read()
        return cls.from_xyz(xyz_str, charge=charge, multiplicity=multiplicity)

    def to_xyz(self, comment: str = "") -> str:
        lines = [str(self.natom), comment]
        for sym, coord in zip(self.symbols, self.coordinates):
            lines.append(f"{sym:2s}  {coord[0]:12.6f}  {coord[1]:12.6f}  {coord[2]:12.6f}")
        return "\n".join(lines)

    def to_xyz_file(self, filename: str, comment: str = "") -> None:
        with open(filename, "w") as f:
            f.write(self.to_xyz(comment))

    def to_bohr(self) -> "Molecule":
        if self.units == "bohr":
            return self.copy()
        from ..utils.units import angstrom_to_bohr

        new_coords = angstrom_to_bohr(self.coordinates)
        return Molecule(
            self.symbols.copy(),
            new_coords,
            charge=self.charge,
            multiplicity=self.multiplicity,
            units="bohr",
        )

    def to_angstrom(self) -> "Molecule":
        if self.units == "angstrom":
            return self.copy()
        from ..utils.units import bohr_to_angstrom

        new_coords = bohr_to_angstrom(self.coordinates)
        return Molecule(
            self.symbols.copy(),
            new_coords,
            charge=self.charge,
            multiplicity=self.multiplicity,
            units="angstrom",
        )

    def copy(self) -> "Molecule":
        return Molecule(
            self.symbols.copy(),
            self.coordinates.copy(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            units=self.units,
        )

    def center_of_mass(self) -> np.ndarray:
        try:
            from mendeleev import element

            masses = np.array([element(s).atomic_weight for s in self.symbols])
            total_mass = masses.sum()
            com = np.sum(self.coordinates * masses[:, np.newaxis], axis=0) / total_mass
            return com
        except ImportError:
            return self.coordinates.mean(axis=0)

    def translate(self, vector: np.ndarray) -> "Molecule":
        new_coords = self.coordinates + np.array(vector)
        return Molecule(
            self.symbols.copy(),
            new_coords,
            charge=self.charge,
            multiplicity=self.multiplicity,
            units=self.units,
        )

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coordinates[i] - self.coordinates[j]))

    def bond_length(self, i: int, j: int) -> float:
        return self.distance(i, j)

    def angle(self, i: int, j: int, k: int) -> float:
        v1 = self.coordinates[i] - self.coordinates[j]
        v2 = self.coordinates[k] - self.coordinates[j]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def bond_angle(self, i: int, j: int, k: int) -> float:
        return np.degrees(self.angle(i, j, k))

    def dihedral(self, i: int, j: int, k: int, l: int) -> float:
        b1 = self.coordinates[j] - self.coordinates[i]
        b2 = self.coordinates[k] - self.coordinates[j]
        b3 = self.coordinates[l] - self.coordinates[k]

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)

        if n1_norm < 1e-10 or n2_norm < 1e-10:
            return 0.0

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        m1 = np.cross(n1, b2 / np.linalg.norm(b2))

        x = np.dot(n1, n2)
        y = np.dot(m1, n2)

        return float(np.arctan2(y, x))

    def __repr__(self) -> str:
        return (
            f"Molecule(formula={self.formula}, natom={self.natom}, "
            f"charge={self.charge}, multiplicity={self.multiplicity}, "
            f"units={self.units})"
        )

    def __str__(self) -> str:
        return self.to_xyz(f"{self.formula} charge={self.charge} mult={self.multiplicity}")

    def __len__(self) -> int:
        return self.natom

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray]:
        return self.symbols[index], self.coordinates[index]
