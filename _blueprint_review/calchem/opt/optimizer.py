from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ..mol.molecule import Molecule
from ..utils.config import get_config
from ..utils.exceptions import ConvergenceError
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class Optimizer:
    def __init__(
        self,
        mol: Molecule,
        grad_func: Callable,
        max_cycles: Optional[int] = None,
        conv_tol: Optional[float] = None,
    ):
        self.mol = mol
        self.grad_func = grad_func
        config = get_config()
        self.max_cycles = max_cycles or config.opt_max_cycles
        self.conv_tol = conv_tol or config.opt_convergence

        self.converged: bool = False
        self.n_cycles: int = 0
        self.energy_history: list = []
        self.final_mol: Optional[Molecule] = None
        self.final_energy: float = 0.0

    def step(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

    def optimize(self) -> Molecule:
        coords = self.mol.coordinates.copy().flatten()

        for cycle in range(self.max_cycles):
            energy, grad = self.grad_func(self.mol)
            self.energy_history.append(energy)
            self.n_cycles = cycle + 1

            grad_norm = np.linalg.norm(grad)

            logger.info(f"Step {cycle + 1}: E = {energy:.10f} Eh, |g| = {grad_norm:.6f}")

            if grad_norm < self.conv_tol:
                self.converged = True
                self.final_energy = energy
                self.final_mol = self.mol.copy()
                logger.info(f"Optimization converged in {cycle + 1} cycles")
                return self.final_mol

            coords = self.step(coords)

            mol_coords = coords.reshape(-1, 3)
            self.mol = Molecule(
                self.mol.symbols,
                mol_coords,
                charge=self.mol.charge,
                multiplicity=self.mol.multiplicity,
                units=self.mol.units,
            )

        energy, grad = self.grad_func(self.mol)
        self.final_energy = energy
        self.final_mol = self.mol.copy()

        if not self.converged:
            raise ConvergenceError(
                f"Optimization did not converge in {self.max_cycles} cycles"
            )

        return self.final_mol

    def __repr__(self) -> str:
        return f"Optimizer(mol={self.mol})"
