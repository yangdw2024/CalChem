from __future__ import annotations
import numpy as np
from ..integrals.one_electron import compute_dipole
AU_TO_DEBYE = 2.541746473
def compute_dipole_moment(mol, basis_set, dm: np.ndarray) -> np.ndarray:
    dip_ints = compute_dipole(basis_set, mol)
    elec_dip = -np.einsum("xmn,mn->x", dip_ints, dm)
    nuc_dip = sum(mol.atomic_numbers[i] * mol.coordinates[i] for i in range(mol.natom))
    return (elec_dip + nuc_dip) * AU_TO_DEBYE
def compute_quadrupole_moment(mol, basis_set, dm: np.ndarray) -> np.ndarray:
    return np.zeros((3, 3))
