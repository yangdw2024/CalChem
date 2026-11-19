from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .optimizer import Optimizer
from ..mol.molecule import Molecule


class BFGS(Optimizer):
    def __init__(self, mol, grad_func, **kwargs):
        super().__init__(mol, grad_func, **kwargs)
        self._last_coords = None
        self._last_grad = None
        self._hessian_inv = None

    def step(self, coords: np.ndarray) -> np.ndarray:
        if self._hessian_inv is None:
            self._hessian_inv = np.eye(len(coords))

        grad = self.grad_func(self.mol)[1].flatten()

        if self._last_coords is not None:
            s = coords - self._last_coords
            y = grad - self._last_grad
            sy = np.dot(s, y)
            if abs(sy) > 1e-10:
                rho = 1.0 / sy
                I = np.eye(len(coords))
                A = I - rho * np.outer(s, y)
                B = I - rho * np.outer(y, s)
                self._hessian_inv = A @ self._hessian_inv @ B + rho * np.outer(s, s)

        direction = -self._hessian_inv @ grad

        step_size = self._line_search(coords, direction, grad)
        new_coords = coords + step_size * direction

        self._last_coords = coords.copy()
        self._last_grad = grad.copy()

        return new_coords

    def _line_search(self, coords, direction, grad):
        alpha = 1.0
        c1 = 1e-4
        max_ls = 20

        e0 = self.grad_func(self.mol)[0]
        g0 = np.dot(grad.flatten(), direction)

        for i in range(max_ls):
            test_coords = coords + alpha * direction
            test_mol = Molecule(
                self.mol.symbols,
                test_coords.reshape(-1, 3),
                charge=self.mol.charge,
                multiplicity=self.mol.multiplicity,
                units=self.mol.units,
            )
            old_mol = self.mol
            self.mol = test_mol
            e_test = self.grad_func(self.mol)[0]
            self.mol = old_mol

            if e_test <= e0 + c1 * alpha * g0:
                return alpha
            alpha *= 0.5

        return alpha

    def optimize(self) -> Molecule:
        def func(coords):
            mol = Molecule(
                self.mol.symbols,
                coords.reshape(-1, 3),
                charge=self.mol.charge,
                multiplicity=self.mol.multiplicity,
                units=self.mol.units,
            )
            energy, grad = self.grad_func(mol)
            return energy, grad.flatten()

        x0 = self.mol.coordinates.flatten()

        result = minimize(
            func,
            x0,
            method="BFGS",
            jac=True,
            options={
                "maxiter": self.max_cycles,
                "gtol": self.conv_tol,
                "disp": False,
            },
        )

        self.converged = result.success
        self.n_cycles = result.nit
        self.final_energy = result.fun

        final_coords = result.x.reshape(-1, 3)
        self.final_mol = Molecule(
            self.mol.symbols,
            final_coords,
            charge=self.mol.charge,
            multiplicity=self.mol.multiplicity,
            units=self.mol.units,
        )

        return self.final_mol
