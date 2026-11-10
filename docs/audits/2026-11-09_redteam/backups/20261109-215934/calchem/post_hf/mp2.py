"""
MP2 (Møller-Plesset 二阶微扰理论) 模块。

实现 RMP2 (限制性 MP2) 和 UMP2 (非限制性 MP2) 方法。
"""

from __future__ import annotations

import numpy as np

from ..basis.basis_set import BasisSet
from ..integrals.two_electron import compute_eri
from ..mol.molecule import Molecule
from ..scf.hf import RHF, UHF
from ..scf.results import SCFResult


class MP2:
    """Møller-Plesset 二阶微扰理论基类。"""

    def __init__(self, mf):
        self.mf = mf
        self.mol = mf.mol
        self.basis_set = mf.basis_set
        self._eri = None
        self.e_corr = 0.0
        self.e_tot = 0.0

    @property
    def energy(self) -> float:
        return self.e_tot

    @property
    def converged(self) -> bool:
        return getattr(self.mf, "converged", True)

    @property
    def mo_energy(self):
        return getattr(self.mf, "mo_energy", None)

    @property
    def mo_occ(self):
        return getattr(self.mf, "mo_occ", None)

    def _compute_eri(self):
        if self._eri is None:
            self._eri = compute_eri(self.basis_set, self.mol)
        return self._eri


class RMP2(MP2):
    """限制性 MP2 (RMP2)。"""

    def __init__(self, mf: RHF):
        super().__init__(mf)
        if not isinstance(mf, RHF):
            raise TypeError("RMP2 requires an RHF object")

    def kernel(self) -> float:
        mf = self.mf
        if getattr(mf, "mo_coeff", None) is None or not getattr(mf, "converged", False):
            mf.kernel()
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        nocc = mf.mol.nelectron // 2
        nbf = mo_coeff.shape[0]
        nvir = nbf - nocc

        C_occ = mo_coeff[:, :nocc]
        C_vir = mo_coeff[:, nocc:]

        eri = self._compute_eri()

        eri_iajb = np.einsum("pi,qa,rj,sb,pqrs->ijab", C_occ, C_vir, C_occ, C_vir, eri)

        e_i = mo_energy[:nocc]
        e_a = mo_energy[nocc:]

        de = e_i[:, None, None, None] + e_i[None, :, None, None] - e_a[None, None, :, None] - e_a[None, None, None, :]

        t_iajb = eri_iajb / de

        e_mp2 = np.einsum("ijab,ijab->", eri_iajb, 2.0 * t_iajb - t_iajb.swapaxes(2, 3))

        self.e_corr = e_mp2
        self.e_tot = mf.e_tot + e_mp2

        return self.e_tot


class UMP2(MP2):
    """非限制性 MP2 (UMP2)。"""

    def __init__(self, mf: UHF):
        super().__init__(mf)
        if not isinstance(mf, UHF):
            raise TypeError("UMP2 requires a UHF object")

    def kernel(self) -> float:
        mf = self.mf
        if getattr(mf, "mo_coeff", None) is None or not getattr(mf, "converged", False):
            mf.kernel()
        mo_coeff_a = mf.mo_coeff[0]
        mo_coeff_b = mf.mo_coeff[1]
        mo_energy_a = mf.mo_energy[0]
        mo_energy_b = mf.mo_energy[1]

        nocc_a = sum(mf.mo_occ[0] > 0)
        nocc_b = sum(mf.mo_occ[1] > 0)
        nbf = mo_coeff_a.shape[0]

        C_occ_a = mo_coeff_a[:, :nocc_a]
        C_vir_a = mo_coeff_a[:, nocc_a:]
        C_occ_b = mo_coeff_b[:, :nocc_b]
        C_vir_b = mo_coeff_b[:, nocc_b:]

        eri = self._compute_eri()

        e_corr = 0.0

        e_corr += self._compute_mp2_block(C_occ_a, C_vir_a, C_occ_a, C_vir_a, mo_energy_a, mo_energy_a, eri)
        e_corr += self._compute_mp2_block(C_occ_b, C_vir_b, C_occ_b, C_vir_b, mo_energy_b, mo_energy_b, eri)
        e_corr += 2.0 * self._compute_mp2_block(C_occ_a, C_vir_a, C_occ_b, C_vir_b, mo_energy_a, mo_energy_b, eri)

        self.e_corr = e_corr
        self.e_tot = mf.e_tot + e_corr

        return self.e_tot

    def _compute_mp2_block(self, C_occ_i, C_vir_a, C_occ_j, C_vir_b, e_i, e_j, eri):
        nocc_i = C_occ_i.shape[1]
        nvir_a = C_vir_a.shape[1]
        nocc_j = C_occ_j.shape[1]
        nvir_b = C_vir_b.shape[1]

        eri_iajb = np.einsum("pi,qa,rj,sb,pqrs->ijab", C_occ_i, C_vir_a, C_occ_j, C_vir_b, eri)

        e_occ_i = e_i[:nocc_i]
        e_vir_a = e_i[nocc_i:]
        e_occ_j = e_j[:nocc_j]
        e_vir_b = e_j[nocc_j:]

        de = e_occ_i[:, None, None, None] + e_occ_j[None, :, None, None] - e_vir_a[None, None, :, None] - e_vir_b[None, None, None, :]

        t_iajb = eri_iajb / de

        return np.einsum("ijab,ijab->", eri_iajb, 2.0 * t_iajb - t_iajb.swapaxes(2, 3))


def mp2(mf) -> float:
    """MP2 计算接口。"""
    if isinstance(mf, RHF):
        return RMP2(mf).kernel()
    elif isinstance(mf, UHF):
        return UMP2(mf).kernel()
    else:
        raise TypeError("MP2 requires an RHF or UHF object")