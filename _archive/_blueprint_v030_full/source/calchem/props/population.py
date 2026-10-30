from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class PopulationAnalysis:
    def __init__(self, method):
        self.method = method
        self.mulliken_charges: Optional[np.ndarray] = None
        self.lowdin_charges: Optional[np.ndarray] = None

    def mulliken(self) -> np.ndarray:
        if self.method._pyscf_mf is None:
            raise RuntimeError("SCF calculation must be run first")

        mf = self.method._pyscf_mf
        mol = mf.mol
        dm = mf.make_rdm1()
        s = mf.get_ovlp()

        pop = np.einsum("ij,ji->i", dm, s)

        charges = np.zeros(mol.natm)
        for i in range(mol.natm):
            ao_start = mol.atom_offset(i)
            ao_end = mol.atom_offset(i) + mol.ao_nuc(i)
            charges[i] = mol.atom_charge(i) - pop[ao_start:ao_end].sum().real

        self.mulliken_charges = charges
        return charges

    def lowdin(self) -> np.ndarray:
        if self.method._pyscf_mf is None:
            raise RuntimeError("SCF calculation must be run first")

        mf = self.method._pyscf_mf
        mol = mf.mol
        dm = mf.make_rdm1()
        s = mf.get_ovlp()

        from scipy.linalg import sqrtm, inv

        s_half = sqrtm(s)
        s_half_inv = inv(s_half)

        dm_lowdin = s_half @ dm @ s_half
        pop = np.diag(dm_lowdin).real

        charges = np.zeros(mol.natm)
        for i in range(mol.natm):
            ao_start = mol.atom_offset(i)
            ao_end = mol.atom_offset(i) + mol.ao_nuc(i)
            charges[i] = mol.atom_charge(i) - pop[ao_start:ao_end].sum()

        self.lowdin_charges = charges
        return charges

    def __repr__(self) -> str:
        return f"PopulationAnalysis(method={self.method})"
