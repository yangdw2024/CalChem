from __future__ import annotations
from typing import List, Sequence
import numpy as np
from ..core.periodic_table import ATOMIC_NUMBER, ATOMIC_MASS
from ..core.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from ..core.errors import CalChemError

class Molecule:
    def __init__(
        self,
        symbols: Sequence[str],
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        charge: int = 0,
        multiplicity: int = 1,
        units: str = "bohr",
        validate_overlap: bool = True,
    ):
        if not symbols:
            raise CalChemError("Molecule must contain at least one atom.")
        self.symbols: List[str] = []
        self.atomic_numbers: List[int] = []
        for s in symbols:
            cap = s.strip().capitalize()
            if cap not in ATOMIC_NUMBER:
                raise CalChemError(f"Unsupported or unknown element symbol: '{s}'")
            self.symbols.append(cap)
            self.atomic_numbers.append(ATOMIC_NUMBER[cap])

        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (len(self.symbols), 3):
            raise CalChemError(f"Coordinate shape mismatch: expected ({len(self.symbols)}, 3), got {coords.shape}")
        if not np.all(np.isfinite(coords)):
            raise CalChemError("Coordinates must be finite real numbers (NaN/Inf detected).")

        u = units.lower()
        if u in ("angstrom", "ang", "a"):
            self.coordinates = coords * ANGSTROM_TO_BOHR
        elif u in ("bohr", "au", "a.u."):
            self.coordinates = coords.copy()
        else:
            raise CalChemError(f"Unknown coordinate unit '{units}'.")

        self.units = "bohr"
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        if self.multiplicity < 1:
            raise CalChemError(f"Invalid spin multiplicity: {self.multiplicity}")

        total_z = sum(self.atomic_numbers)
        self.nelectron = total_z - self.charge
        if self.nelectron <= 0:
            raise CalChemError(f"Non-positive electron count ({self.nelectron}).")

        n_unpaired = self.multiplicity - 1
        if n_unpaired > self.nelectron or (self.nelectron - n_unpaired) % 2 != 0:
            raise CalChemError(f"Inconsistent electron count ({self.nelectron}) and multiplicity ({self.multiplicity}).")

        if validate_overlap and len(self.symbols) > 1:
            for i in range(len(self.symbols)):
                for j in range(i + 1, len(self.symbols)):
                    dist = float(np.linalg.norm(self.coordinates[i] - self.coordinates[j]))
                    if dist < 1e-3:
                        raise CalChemError(f"Atoms {i} and {j} overlap (d = {dist:.3e} Bohr).")

    @property
    def natom(self) -> int:
        return len(self.symbols)

    def to_bohr(self) -> "Molecule":
        return self
