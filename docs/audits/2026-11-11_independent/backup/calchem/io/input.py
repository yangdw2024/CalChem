from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..mol.molecule import Molecule
from ..utils.exceptions import InputError


def read_xyz(filename: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
    try:
        return Molecule.from_xyz_file(filename, charge=charge, multiplicity=multiplicity)
    except OSError:
        return Molecule.from_xyz(filename, charge=charge, multiplicity=multiplicity)


def read_gjf(filename: str) -> Molecule:
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = filename.strip().split("\n")

    charge = 0
    multiplicity = 1
    symbols = []
    coordinates = []
    in_coords = False
    variables = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("%") or line.startswith("#"):
            continue

        if "=" in line and not in_coords:
            parts = line.split("=")
            var_name = parts[0].strip()
            try:
                var_value = float(parts[1].strip())
                variables[var_name] = var_value
            except ValueError:
                pass
            continue

        if not in_coords and len(line.split()) == 2:
            try:
                charge = int(line.split()[0])
                multiplicity = int(line.split()[1])
                in_coords = True
                continue
            except ValueError:
                pass

        if in_coords or (line and line[0].isalpha()):
            parts = line.split()
            if len(parts) >= 1:
                symbol = parts[0].capitalize()
                if symbol in ["R", "A", "D"]:
                    continue
                try:
                    from calchem.utils.constants import ATOMIC_NUMBERS
                    if symbol not in ATOMIC_NUMBERS:
                        if not in_coords:
                            continue
                except ImportError:
                    pass
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        symbols.append(symbol)
                        coordinates.append([x, y, z])
                        in_coords = True
                    except ValueError:
                        symbols.append(symbol)
                        coordinates.append([0.0, 0.0, 0.0])
                        in_coords = True
                else:
                    symbols.append(symbol)
                    coordinates.append([0.0, 0.0, 0.0])
                    in_coords = True

    if not symbols:
        raise InputError(f"No atoms found in Gaussian input file")

    return Molecule(symbols, coordinates, charge=charge, multiplicity=multiplicity)


def read_gro(filename: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise InputError(f"Invalid GRO file: {filename}")

    try:
        natom = int(lines[1].strip())
    except ValueError:
        raise InputError(f"Invalid number of atoms in GRO file: {lines[1]}")

    symbols = []
    coordinates = []

    for i in range(2, 2 + natom):
        if i >= len(lines):
            break
        line = lines[i]
        if len(line) < 44:
            continue

        atom_name = line[10:15].strip()
        symbol = ""
        for c in atom_name:
            if c.isalpha():
                symbol += c
            else:
                break
        symbol = symbol.capitalize()

        try:
            x = float(line[20:28].strip()) * 10.0
            y = float(line[28:36].strip()) * 10.0
            z = float(line[36:44].strip()) * 10.0
            symbols.append(symbol)
            coordinates.append([x, y, z])
        except ValueError:
            continue

    if not symbols:
        raise InputError(f"No atoms found in GRO file: {filename}")

    return Molecule(symbols, coordinates, charge=charge, multiplicity=multiplicity)


def read_mol2(filename: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
    with open(filename, "r") as f:
        lines = f.readlines()

    in_atoms = False
    symbols = []
    coordinates = []
    natom = 0

    for line in lines:
        line = line.strip()
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        elif line.startswith("@<TRIPOS>"):
            in_atoms = False
            continue

        if in_atoms and line:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    atom_name = parts[1]
                    symbol = ""
                    for c in atom_name:
                        if c.isalpha():
                            symbol += c
                        else:
                            break
                    symbol = symbol.capitalize()

                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    symbols.append(symbol)
                    coordinates.append([x, y, z])
                    natom += 1
                except ValueError:
                    continue

    if not symbols:
        raise InputError(f"No atoms found in MOL2 file: {filename}")

    return Molecule(symbols, coordinates, charge=charge, multiplicity=multiplicity)


def parse_gjf_route(route_line: str) -> dict:
    """解析 Gaussian route section，返回方法、基组、任务类型等信息"""
    info = {
        "method": None,
        "basis": None,
        "jobs": [],
        "keywords": [],
        "raw": route_line.strip(),
    }

    route = route_line.strip().lstrip("#").strip()
    route = route.lstrip("PpNn ")

    tokens = []
    for token in route.split():
        if "/" in token and not token.startswith("opt") and not token.startswith("freq"):
            parts = token.split("/")
            if info["method"] is None and len(parts) == 2:
                info["method"] = parts[0].upper()
                info["basis"] = parts[1]
                continue
        tokens.append(token)

    job_keywords = {
        "SP", "ENERGY", "ENERGY=", "OPT", "OPTIMIZATION", "FREQ",
        "FREQUENCY", "TD", "TDDFT", "NMR", "POP", "SCAN", "IRC",
    }

    for token in tokens:
        upper = token.upper()
        keyword = upper.split("=")[0]
        if keyword in job_keywords:
            info["jobs"].append(keyword)
        info["keywords"].append(token)

    if not info["jobs"]:
        info["jobs"].append("SP")

    return info


def read_molecule(filename: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
    ext = Path(filename).suffix.lower()

    readers = {
        ".xyz": read_xyz,
        ".gjf": read_gjf,
        ".com": read_gjf,
        ".gro": read_gro,
        ".mol2": read_mol2,
    }

    if ext not in readers:
        raise InputError(f"Unsupported file format: {ext}")

    return readers[ext](filename, charge=charge, multiplicity=multiplicity)
