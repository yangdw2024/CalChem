from __future__ import annotations

from typing import Optional

import numpy as np

from .optimizer import Optimizer
from ..mol.molecule import Molecule


class Berny(Optimizer):
    def __init__(self, mol, grad_func, **kwargs):
        super().__init__(mol, grad_func, **kwargs)

    def step(self, coords: np.ndarray) -> np.ndarray:
        grad = self.grad_func(self.mol)[1].flatten()
        step = -0.1 * grad / (np.linalg.norm(grad) + 1e-8)
        return coords + step

    def optimize(self) -> Molecule:
        return super().optimize()
