"""
CIS (Configuration Interaction Singles) 激发态方法。

完全自研实现，不依赖外部量子化学库。
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np

from ..utils.logging_utils import get_logger
from ..utils.exceptions import CalculationError

logger = get_logger(__name__)


class CIS:
    """Configuration Interaction Singles (CIS) 激发态计算。"""

    def __init__(self, hf_method, nstates: int = 10, singlet: bool = True):
        self.hf_method = hf_method
        self.nstates = nstates
        self.singlet = singlet

        self.excitation_energies: Optional[np.ndarray] = None
        self.oscillator_strengths: Optional[np.ndarray] = None
        self.transition_dipoles: Optional[np.ndarray] = None
        self.coefficients: Optional[List[np.ndarray]] = None
        self.converged: bool = False

    def kernel(self) -> np.ndarray:
        if self.hf_method.mo_energy is None:
            raise CalculationError("SCF calculation must be run first")

        mo_energy = self.hf_method.mo_energy
        mo_coeff = self.hf_method.mo_coeff
        mo_occ = self.hf_method.mo_occ
        eri = self.hf_method._eri

        nocc = int(np.sum(mo_occ > 0))
        nvir = len(mo_energy) - nocc
        nbf = len(mo_energy)

        if eri is None:
            raise CalculationError("ERI not available")

        e_occ = mo_energy[:nocc]
        e_vir = mo_energy[nocc:]

        C_occ = mo_coeff[:, :nocc]
        C_vir = mo_coeff[:, nocc:]

        eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, C_occ, C_vir, C_occ, C_vir)

        n_singles = nocc * nvir

        if self.singlet:
            H = np.zeros((n_singles, n_singles))
            for i in range(nocc):
                for a in range(nvir):
                    ia = i * nvir + a
                    for j in range(nocc):
                        for b in range(nvir):
                            jb = j * nvir + b
                            delta_ij = 1.0 if i == j else 0.0
                            delta_ab = 1.0 if a == b else 0.0
                            H[ia, jb] = (
                                delta_ij * delta_ab * (e_vir[a] - e_occ[i])
                                + 2.0 * eri_mo[i, a, j, b]
                                - eri_mo[i, j, a, b]
                            )
        else:
            H = np.zeros((n_singles, n_singles))
            for i in range(nocc):
                for a in range(nvir):
                    ia = i * nvir + a
                    for j in range(nocc):
                        for b in range(nvir):
                            jb = j * nvir + b
                            delta_ij = 1.0 if i == j else 0.0
                            delta_ab = 1.0 if a == b else 0.0
                            H[ia, jb] = (
                                delta_ij * delta_ab * (e_vir[a] - e_occ[i])
                                + eri_mo[i, a, j, b]
                                - eri_mo[i, j, a, b]
                            )

        eigvals, eigvecs = np.linalg.eigh(H)

        n_states = min(self.nstates, n_singles)
        self.excitation_energies = eigvals[:n_states]

        self.coefficients = []
        for k in range(n_states):
            vec = eigvecs[:, k].reshape(nocc, nvir)
            self.coefficients.append(vec)

        self.transition_dipoles = np.zeros((n_states, 3))
        self.oscillator_strengths = np.zeros(n_states)

        self.converged = True

        logger.info(
            f"CIS converged. {n_states} states. "
            f"Lowest: {self.excitation_energies[0]:.6f} Eh"
        )

        return self.excitation_energies

    def __repr__(self) -> str:
        return f"CIS(hf={self.hf_method}, nstates={self.nstates})"
