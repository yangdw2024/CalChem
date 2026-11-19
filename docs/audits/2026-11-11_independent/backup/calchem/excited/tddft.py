from __future__ import annotations

from typing import Optional, List

import numpy as np

from ..utils.logging_utils import get_logger
from ..utils.exceptions import CalculationError

logger = get_logger(__name__)


class TDDFTMethod:
    def __init__(self, scf_method, nstates: int = 10, singlet: bool = True):
        self.scf_method = scf_method
        self.nstates = nstates
        self.singlet = singlet

        self.excitation_energies: Optional[np.ndarray] = None
        self.oscillator_strengths: Optional[np.ndarray] = None
        self.transition_dipoles: Optional[np.ndarray] = None
        self.converged: bool = False

    def kernel(self) -> np.ndarray:
        if self.scf_method._pyscf_mf is None:
            raise RuntimeError("SCF calculation must be run first")

        from pyscf import tddft

        if hasattr(self.scf_method._pyscf_mf, "xc"):
            td = tddft.TDDFT(self.scf_method._pyscf_mf)
        else:
            td = tddft.TDHF(self.scf_method._pyscf_mf)

        td.nstates = self.nstates
        td.singlet = self.singlet
        td.verbose = 0

        td.kernel()

        self.excitation_energies = td.e
        self.oscillator_strengths = td.oscillator_strength()
        self.converged = td.converged

        if not self.converged:
            raise CalculationError("TDDFT did not converge")

        logger.info(
            f"TDDFT converged. {len(self.excitation_energies)} states. "
            f"Lowest: {self.excitation_energies[0]:.6f} Eh"
        )

        return self.excitation_energies

    def __repr__(self) -> str:
        return f"TDDFTMethod(scf={self.scf_method}, nstates={self.nstates})"
