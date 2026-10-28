from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RESPCharges:
    def __init__(self, method):
        self.method = method
        self.charges: Optional[np.ndarray] = None

    def fit(self) -> np.ndarray:
        logger.warning("RESP charge fitting not fully implemented")
        if self.method._pyscf_mf is not None:
            mol = self.method._pyscf_mf.mol
            self.charges = np.zeros(mol.natm)
        return self.charges if self.charges is not None else np.array([])

    def __repr__(self) -> str:
        return f"RESPCharges(method={self.method})"


class CM5Charges:
    def __init__(self, method):
        self.method = method
        self.charges: Optional[np.ndarray] = None

    def fit(self) -> np.ndarray:
        logger.warning("CM5 charge fitting not fully implemented")
        if self.method._pyscf_mf is not None:
            mol = self.method._pyscf_mf.mol
            self.charges = np.zeros(mol.natm)
        return self.charges if self.charges is not None else np.array([])

    def __repr__(self) -> str:
        return f"CM5Charges(method={self.method})"
