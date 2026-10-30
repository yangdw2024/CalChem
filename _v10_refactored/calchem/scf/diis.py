"""
SCF 收敛加速方法：DIIS / ADIIS / EDIIS

参考:
- Pulay, Chem. Phys. Lett. 73, 393 (1980)
- Hu et al., J. Chem. Phys. 109, 8434 (1998)
- Kudin et al., J. Chem. Phys. 116, 8255 (2002)
"""

from __future__ import annotations

import numpy as np


class DIIS:
    def __init__(self, max_vec: int = 8):
        self.max_vec = max_vec
        self.e_vecs = []
        self.f_vecs = []
        self._counter = 0

    def update(self, fock: np.ndarray, error: np.ndarray) -> np.ndarray:
        self.e_vecs.append(error.copy())
        self.f_vecs.append(fock.copy())

        if len(self.e_vecs) > self.max_vec:
            self.e_vecs.pop(0)
            self.f_vecs.pop(0)

        n_vec = len(self.e_vecs)
        if n_vec < 2:
            return fock

        b = np.zeros((n_vec + 1, n_vec + 1))
        b[-1, :-1] = 1.0
        b[:-1, -1] = 1.0

        for i in range(n_vec):
            for j in range(n_vec):
                b[i, j] = np.einsum('ij,ij->', self.e_vecs[i], self.e_vecs[j])

        rhs = np.zeros(n_vec + 1)
        rhs[-1] = 1.0

        try:
            coeff = np.linalg.solve(b, rhs)[:-1]
        except np.linalg.LinAlgError:
            coeff = np.zeros(n_vec)
            coeff[-1] = 1.0

        f_new = np.zeros_like(fock)
        for i in range(n_vec):
            f_new += coeff[i] * self.f_vecs[i]

        return f_new


class ADIIS:
    def __init__(self, max_vec: int = 8, mixing: float = 0.0):
        self.max_vec = max_vec
        self.e_vecs = []
        self.f_vecs = []
        self.d_vecs = []
        self.mixing = mixing
        self._counter = 0

    def update(self, fock: np.ndarray, error: np.ndarray,
               density: np.ndarray = None) -> np.ndarray:
        self.e_vecs.append(error.copy())
        self.f_vecs.append(fock.copy())
        if density is not None:
            self.d_vecs.append(density.copy())

        if len(self.e_vecs) > self.max_vec:
            self.e_vecs.pop(0)
            self.f_vecs.pop(0)
            if self.d_vecs:
                self.d_vecs.pop(0)

        n_vec = len(self.e_vecs)
        if n_vec < 2:
            return fock

        b = np.zeros((n_vec + 1, n_vec + 1))
        b[-1, :-1] = 1.0
        b[:-1, -1] = 1.0

        for i in range(n_vec):
            for j in range(n_vec):
                b[i, j] = np.einsum('ij,ij->', self.e_vecs[i], self.e_vecs[j])

                if self.d_vecs:
                    d_i = self.d_vecs[i]
                    d_j = self.d_vecs[j]
                    b[i, j] += self.mixing * np.einsum('ij,ij->', d_i, d_j)

        rhs = np.zeros(n_vec + 1)
        rhs[-1] = 1.0

        try:
            coeff = np.linalg.solve(b, rhs)[:-1]
        except np.linalg.LinAlgError:
            coeff = np.zeros(n_vec)
            coeff[-1] = 1.0

        f_new = np.zeros_like(fock)
        for i in range(n_vec):
            f_new += coeff[i] * self.f_vecs[i]

        return f_new


class EDIIS:
    def __init__(self, max_vec: int = 6):
        self.max_vec = max_vec
        self.f_history = []
        self.d_history = []
        self.e_history = []

    def update(self, fock: np.ndarray, density: np.ndarray,
               energy: float) -> np.ndarray:
        self.f_history.append(fock.copy())
        self.d_history.append(density.copy())
        self.e_history.append(energy)

        if len(self.f_history) > self.max_vec:
            self.f_history.pop(0)
            self.d_history.pop(0)
            self.e_history.pop(0)

        n_vec = len(self.f_history)
        if n_vec < 2:
            return fock

        nbf = fock.shape[0]

        h = np.zeros((n_vec + 1, n_vec + 1))
        s = np.zeros((n_vec + 1, n_vec + 1))

        for i in range(n_vec):
            h[i, i] = self.e_history[i]
            for j in range(n_vec):
                h[i, j] = np.einsum('ij,ij->', self.d_history[i],
                                    (self.f_history[i] + self.f_history[j]) / 2)

        for i in range(n_vec):
            s[i, n_vec] = 1.0
            s[n_vec, i] = 1.0

        s[n_vec, n_vec] = 0.0

        rhs = np.zeros(n_vec + 1)
        rhs[-1] = 1.0

        try:
            coeff = np.linalg.solve(s, rhs)[:-1]
        except np.linalg.LinAlgError:
            coeff = np.zeros(n_vec)
            coeff[-1] = 1.0

        coeff = np.maximum(coeff, 0.0)
        coeff /= np.sum(coeff)

        f_new = np.zeros_like(fock)
        for i in range(n_vec):
            f_new += coeff[i] * self.f_history[i]

        return f_new


class ConvergenceChecker:
    def __init__(self, energy_tol: float = 1e-10,
                 density_tol: float = 1e-8,
                 gradient_tol: float = 1e-6):
        self.energy_tol = energy_tol
        self.density_tol = density_tol
        self.gradient_tol = gradient_tol
        self.prev_energy = None
        self.prev_density = None

    def check(self, energy: float, density: np.ndarray,
              gradient: np.ndarray = None) -> bool:
        if self.prev_energy is None or self.prev_density is None:
            self.prev_energy = energy
            self.prev_density = density.copy()
            return False
        converged = True

        if self.prev_energy is not None:
            de = abs(energy - self.prev_energy)
            if de > self.energy_tol:
                converged = False
                print(f"  Energy change: {de:.2e} > {self.energy_tol:.2e}")

        if self.prev_density is not None:
            dd = np.sqrt(np.sum((density - self.prev_density) ** 2)) / density.size
            if dd > self.density_tol:
                converged = False
                print(f"  Density RMS: {dd:.2e} > {self.density_tol:.2e}")

        if gradient is not None:
            dg = np.sqrt(np.sum(gradient ** 2)) / gradient.size
            if dg > self.gradient_tol:
                converged = False
                print(f"  Gradient RMS: {dg:.2e} > {self.gradient_tol:.2e}")

        self.prev_energy = energy
        self.prev_density = density.copy()

        return converged