from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from ..basis.basis_set import BasisSet
from ..mol.molecule import Molecule

@dataclass
class GridSpec:
    origin_bohr: Tuple[float, float, float]
    spacing_bohr: Tuple[float, float, float]
    dims: Tuple[int, int, int]

def evaluate_orbital_grid(mol: Molecule, basis_set: BasisSet, mo_coeff: np.ndarray, orbital_index: int, grid: GridSpec) -> np.ndarray:
    if mo_coeff is None or mo_coeff.ndim != 2:
        raise ValueError("Wavefunction MO coefficient matrix is missing or invalid.")
    nbf, nmo = mo_coeff.shape
    if orbital_index < 0 or orbital_index >= nmo:
        raise ValueError(f"Orbital index {orbital_index} out of bounds [0, {nmo}).")
    nx, ny, nz = grid.dims
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid grid dimensions: {grid.dims}")
    ox, oy, oz = grid.origin_bohr
    dx, dy, dz = grid.spacing_bohr
    xs = ox + np.arange(nx, dtype=np.float64) * dx
    ys = oy + np.arange(ny, dtype=np.float64) * dy
    zs = oz + np.arange(nz, dtype=np.float64) * dz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1)
    all_shells = basis_set.get_all_shells(mol)
    mo_vec = mo_coeff[:, orbital_index]
    orb_vals = np.zeros((nx, ny, nz), dtype=np.float64)
    bf_idx = 0
    for shell in all_shells:
        dr = pts - shell.origin
        r2 = np.sum(dr * dr, axis=-1)
        for comp_idx, (l, m, n) in enumerate(shell.angular_combinations):
            c_mo = float(mo_vec[bf_idx])
            bf_idx += 1
            if abs(c_mo) < 1e-16:
                continue
            ang_part = np.ones((nx, ny, nz), dtype=np.float64)
            if l > 0: ang_part = ang_part * (dr[..., 0] ** l)
            if m > 0: ang_part = ang_part * (dr[..., 1] ** m)
            if n > 0: ang_part = ang_part * (dr[..., 2] ** n)
            rad_part = np.zeros((nx, ny, nz), dtype=np.float64)
            for p_idx in range(shell.n_primitives):
                alpha = shell.exponents[p_idx]
                coeff = shell._contract_coeffs[p_idx]
                norm_p = shell._primitives[comp_idx][p_idx].norm
                rad_part += (coeff * norm_p) * np.exp(-alpha * r2)
            orb_vals += c_mo * ang_part * rad_part
    return orb_vals.astype(np.float32)
