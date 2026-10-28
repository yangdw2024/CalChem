"""
频率计算和热化学模块。

实现简正模式分析和热化学性质计算。
"""

from __future__ import annotations

import numpy as np

from ..mol.molecule import Molecule
from ..scf.hf import RHF, UHF
from ..dft.dft import RKS, UKS
from ..utils.constants import ATOMIC_MASSES, KB, HARTREE_TO_KCAL_MOL


class FrequencyAnalysis:
    """频率分析类。"""

    def __init__(self, mf, temperature: float = 298.15, pressure: float = 1.0):
        self.mf = mf
        self.mol = mf.mol
        self.temperature = temperature
        self.pressure = pressure
        self.hessian = None
        self.frequencies = None
        self.normal_modes = None
        self.thermo = None

    def compute_hessian(self, delta: float = 1e-5) -> np.ndarray:
        """通过数值方法计算Hessian矩阵。"""
        natom = self.mol.natom
        ncoord = 3 * natom
        hessian = np.zeros((ncoord, ncoord))

        coords = self.mol.coordinates.copy()

        for i in range(ncoord):
            atom_idx = i // 3
            comp = i % 3

            coords_plus = coords.copy()
            coords_plus[atom_idx, comp] += delta

            coords_minus = coords.copy()
            coords_minus[atom_idx, comp] -= delta

            mol_plus = Molecule(self.mol.symbols, coords_plus, self.mol.charge, self.multiplicity, units='bohr')
            mol_minus = Molecule(self.mol.symbols, coords_minus, self.mol.charge, self.multiplicity, units='bohr')

            grad_plus = self._compute_gradient(mol_plus)
            grad_minus = self._compute_gradient(mol_minus)

            diff = (grad_plus - grad_minus).flatten()
            hessian[i, :] = diff / (2 * delta)

        self.hessian = hessian
        return hessian

    def _compute_gradient(self, mol) -> np.ndarray:
        from ..grad.hf_grad import HFGradient
        from ..grad.dft_grad import DFTGradient

        if isinstance(self.mf, (RHF, UHF)):
            mf_copy = type(self.mf)(mol, basis=self.mf.basis_name, verbose=0)
            mf_copy.kernel()
            return HFGradient(mf_copy, numerical=True).kernel()
        elif isinstance(self.mf, (RKS, UKS)):
            mf_copy = type(self.mf)(mol, basis=self.mf.basis_name, xc=self.mf.xc, verbose=0)
            mf_copy.kernel()
            return DFTGradient(mf_copy).kernel()
        else:
            raise TypeError("Unsupported method type")

    def diagonalize(self) -> np.ndarray:
        """对角化Hessian矩阵得到频率。"""
        if self.hessian is None:
            self.compute_hessian()

        natom = self.mol.natom
        ncoord = 3 * natom

        mass_weights = np.sqrt(np.repeat([ATOMIC_MASSES.get(s, 1.0) for s in self.mol.symbols], 3))

        hessian_weighted = self.hessian / np.outer(mass_weights, mass_weights)

        eigenvalues, eigenvectors = np.linalg.eigh(hessian_weighted)

        frequencies = np.sqrt(np.maximum(eigenvalues, 0))

        self.frequencies = frequencies
        self.normal_modes = eigenvectors

        return frequencies

    def compute_thermochemistry(self) -> dict:
        """计算热化学性质。"""
        if self.frequencies is None:
            self.diagonalize()

        natom = self.mol.natom
        n_vib = 3 * natom - 6 if natom > 1 else 0

        frequencies_cm = self.frequencies[6:] * 219474.63 if n_vib > 0 else np.array([])

        zpe = 0.0
        e_trans = 0.0
        e_rot = 0.0
        h_trans = 0.0
        h_rot = 0.0
        s_trans = 0.0
        s_rot = 0.0
        s_vib = 0.0

        T = self.temperature

        for freq in frequencies_cm[:n_vib]:
            x = freq / (KB * T * 8065.5)
            if x > 0:
                zpe += 0.5 * freq / 8065.5
                exp_x = np.exp(x)
                e_vib = (freq / 8065.5) / (exp_x - 1)
                s_vib += KB * (x / (exp_x - 1) - np.log(1 - exp_x))

        if natom > 1:
            e_trans = 1.5 * KB * T
            e_rot = 1.5 * KB * T

            h_trans = e_trans + KB * T
            h_rot = e_rot + KB * T

            s_trans = KB * (np.log((2 * np.pi * ATOMIC_MASSES.get(self.mol.symbols[0], 1.0) * KB * T) ** 1.5 * KB * T / (6.022e23 ** 2 * (1.0e-10) ** 3)) + 2.5)
            s_rot = KB * (np.log(8 * np.pi ** 2 * 1.0 * KB * T) + 1.0)

        e_thermal = e_trans + e_rot + zpe
        h_thermal = h_trans + h_rot + zpe
        s_total = s_trans + s_rot + s_vib

        g_correction = h_thermal - T * s_total

        self.thermo = {
            'zpe': zpe,
            'e_thermal': e_thermal,
            'h_thermal': h_thermal,
            's_total': s_total,
            'g_correction': g_correction,
            'frequencies_cm': frequencies_cm[:n_vib]
        }

        return self.thermo

    def kernel(self) -> tuple:
        """执行完整的频率分析。"""
        self.compute_hessian()
        self.diagonalize()
        self.compute_thermochemistry()
        return self.frequencies, self.thermo


def frequency(mf, temperature: float = 298.15, pressure: float = 1.0) -> FrequencyAnalysis:
    """频率计算接口。"""
    fa = FrequencyAnalysis(mf, temperature, pressure)
    fa.kernel()
    return fa