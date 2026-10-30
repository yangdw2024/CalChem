from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class SMD:
    def __init__(self, method, solvent: str = "water"):
        self.method = method
        self.solvent = solvent
        self.e_solv: float = 0.0

    def kernel(self) -> float:
        logger.warning(f"SMD solvent calculation for {self.solvent} not fully implemented")
        self.e_solv = 0.0
        return self.e_solv

    def __repr__(self) -> str:
        return f"SMD(solvent='{self.solvent}')"
