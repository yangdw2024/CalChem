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
        self.mulliken_spin: Optional[np.ndarray] = None
        self.lowdin_spin: Optional[np.ndarray] = None
        self._rust_cache: Optional[dict] = None

    def _try_rust_population(self) -> Optional[dict]:
        if self._rust_cache is not None:
            return self._rust_cache
        try:
            from calchem import _core
        except ImportError:
            return None
        wf = self.method
        if not isinstance(wf, _core.PyScfWavefunction):
            wf = getattr(self.method, "_rust_wf", None)
        if isinstance(wf, _core.PyScfWavefunction):
            self._rust_cache = _core.compute_population_analysis(wf)
            if self._rust_cache.get("mulliken_spin") is not None:
                self.mulliken_spin = np.asarray(self._rust_cache["mulliken_spin"], dtype=np.float64)
            if self._rust_cache.get("lowdin_spin") is not None:
                self.lowdin_spin = np.asarray(self._rust_cache["lowdin_spin"], dtype=np.float64)
            return self._rust_cache
        return None

    def mulliken(self) -> np.ndarray:
        rust_pop = self._try_rust_population()
        if rust_pop is not None:
            self.mulliken_charges = np.asarray(rust_pop["mulliken"], dtype=np.float64)
            return self.mulliken_charges

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
        rust_pop = self._try_rust_population()
        if rust_pop is not None:
            self.lowdin_charges = np.asarray(rust_pop["lowdin"], dtype=np.float64)
            return self.lowdin_charges

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
