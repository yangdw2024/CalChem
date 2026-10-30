from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class MultipoleMoments:
    def __init__(self, method):
        self.method = method
        self.dipole: Optional[np.ndarray] = None
        self.quadrupole: Optional[np.ndarray] = None

    def calculate(self) -> dict:
        if self.method._pyscf_mf is None:
            raise RuntimeError("SCF calculation must be run first")

        mf = self.method._pyscf_mf
        self.dipole = mf.dip_moment()

        results = {
            "dipole": self.dipole,
            "dipole_magnitude": np.linalg.norm(self.dipole) if self.dipole is not None else 0.0,
        }

        return results

    def __repr__(self) -> str:
        return f"MultipoleMoments(method={self.method})"
