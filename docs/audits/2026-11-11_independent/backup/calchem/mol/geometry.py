from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .molecule import Molecule
from ..utils.exceptions import InputError


class Geometry:
    @staticmethod
    def zmatrix_to_cartesian(zmatrix: List[str]) -> Tuple[List[str], np.ndarray]:
        symbols = []
        coordinates = []

        for i, line in enumerate(zmatrix):
            parts = line.strip().split()
            if not parts:
                continue

            sym = parts[0].capitalize()
            symbols.append(sym)

            if i == 0:
                coordinates.append([0.0, 0.0, 0.0])
            elif i == 1:
                atom1 = int(parts[1]) - 1
                r = float(parts[2])
                coordinates.append([0.0, 0.0, r])
            elif i == 2:
                atom1 = int(parts[1]) - 1
                r = float(parts[2])
                atom2 = int(parts[3]) - 1
                angle = np.radians(float(parts[4]))

                p1 = np.array(coordinates[atom1])
                p2 = np.array(coordinates[atom2])

                v = p2 - p1
                v = v / np.linalg.norm(v)

                if abs(v[2]) < 0.99:
                    w = np.array([0, 0, 1])
                else:
                    w = np.array([1, 0, 0])

                u = np.cross(v, w)
                u = u / np.linalg.norm(u)

                pos = p1 + r * np.sin(angle) * u + r * np.cos(angle) * v
                coordinates.append(pos.tolist())
            else:
                atom1 = int(parts[1]) - 1
                r = float(parts[2])
                atom2 = int(parts[3]) - 1
                angle = np.radians(float(parts[4]))
                atom3 = int(parts[5]) - 1
                dihedral = np.radians(float(parts[6]))

                p1 = np.array(coordinates[atom1])
                p2 = np.array(coordinates[atom2])
                p3 = np.array(coordinates[atom3])

                v21 = p1 - p2
                v21 = v21 / np.linalg.norm(v21)
                v32 = p3 - p2
                v32 = v32 / np.linalg.norm(v32)

                n = np.cross(v32, v21)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    n = np.array([1, 0, 0])
                else:
                    n = n / n_norm

                m = np.cross(v21, n)

                dx = r * np.sin(angle) * np.cos(dihedral)
                dy = r * np.sin(angle) * np.sin(dihedral)
                dz = r * np.cos(angle)

                pos = p1 + dx * n + dy * m + dz * v21
                coordinates.append(pos.tolist())

        return symbols, np.array(coordinates)

    @staticmethod
    def cartesian_to_zmatrix(
        symbols: List[str], coordinates: np.ndarray
    ) -> List[str]:
        natom = len(symbols)
        zmatrix = []

        coords = np.array(coordinates)

        if natom >= 1:
            zmatrix.append(f"{symbols[0]}")
        if natom >= 2:
            r = np.linalg.norm(coords[1] - coords[0])
            zmatrix.append(f"{symbols[1]} 1 {r:.6f}")
        if natom >= 3:
            r = np.linalg.norm(coords[2] - coords[0])
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(coords[0] - coords[1], coords[2] - coords[1])
                        / (
                            np.linalg.norm(coords[0] - coords[1])
                            * np.linalg.norm(coords[2] - coords[1])
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
            zmatrix.append(f"{symbols[2]} 1 {r:.6f} 2 {angle:.4f}")

        for i in range(3, natom):
            r = np.linalg.norm(coords[i] - coords[i - 1])

            v1 = coords[i - 2] - coords[i - 1]
            v2 = coords[i] - coords[i - 1]
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                        -1.0,
                        1.0,
                    )
                )
            )

            b1 = coords[i - 2] - coords[i - 3]
            b2 = coords[i - 1] - coords[i - 2]
            b3 = coords[i] - coords[i - 1]

            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)

            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)

            if n1_norm < 1e-10 or n2_norm < 1e-10:
                dihedral = 0.0
            else:
                n1 = n1 / n1_norm
                n2 = n2 / n2_norm
                m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                x = np.dot(n1, n2)
                y = np.dot(m1, n2)
                dihedral = np.degrees(np.arctan2(y, x))

            zmatrix.append(
                f"{symbols[i]} {i} {r:.6f} {i-1} {angle:.4f} {i-2} {dihedral:.4f}"
            )

        return zmatrix

    @staticmethod
    def molecule_from_zmatrix(
        zmatrix: List[str], charge: int = 0, multiplicity: int = 1
    ) -> Molecule:
        symbols, coords = Geometry.zmatrix_to_cartesian(zmatrix)
        return Molecule(symbols, coords, charge=charge, multiplicity=multiplicity)

    @staticmethod
    def molecule_to_zmatrix(mol: Molecule) -> List[str]:
        return Geometry.cartesian_to_zmatrix(mol.symbols, mol.coordinates)
