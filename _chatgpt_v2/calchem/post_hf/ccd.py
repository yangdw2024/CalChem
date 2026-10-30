"""
CCD (Coupled Cluster Doubles) 耦合簇双激发方法。

CCD 是 CCSD 的简化版本，仅考虑双激发。
对于单激发对相关能贡献较小的体系，CCD 与 CCSD 能量接近。

CCD 振幅方程:
D_ij^ab * t_ij^ab = <ij||ab> + sum_{klcd} t_kl^cd * <ij||kl> * <ab||cd> / 4
                   - sum_{kc} t_ik^ac * <jl||kc>
                   - sum_{kc} t_jk^bc * <il||kc>

参考:
- Cizek, J. Chem. Phys. 45, 4256 (1966)
- Bartlett, J. Phys. Chem. 93, 1697 (1989)
"""

from __future__ import annotations

import numpy as np

from ..basis.basis_set import BasisSet
from ..integrals.two_electron import compute_eri
from ..mol.molecule import Molecule
from ..scf.hf import RHF


class CCD:
    """CCD 耦合簇双激发方法。"""

    def __init__(
        self,
        mol: Molecule,
        basis: str | BasisSet = "sto-3g",
        max_cycles: int = 50,
        conv_tol: float = 1e-7,
        verbose: int = 1,
    ):
        if isinstance(basis, str):
            self.basis_set = BasisSet(basis)
            self.basis_name = basis
        else:
            self.basis_set = basis
            self.basis_name = basis.name

        self.mol = mol.to_bohr()
        self.max_cycles = max_cycles
        self.conv_tol = conv_tol
        self.verbose = verbose

        self.e_hf: float = 0.0
        self.e_corr: float = 0.0
        self.e_tot: float = 0.0
        self.converged: bool = False
        self.n_cycles: int = 0

        self.mo_energy = None
        self.mo_coeff = None
        self.t2 = None

        self._fock_mo = None
        self._eri_mo = None

        self._nocc = 0
        self._nvirt = 0
        self._nbf = 0

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _transform_eri_to_mo(self, eri_ao, mo_coeff):
        """将 ERI 从 AO 基转换到 MO 基。"""
        mo_t = mo_coeff.T
        tmp1 = np.einsum("pj,pqrs->qjrs", mo_t, eri_ao)
        tmp2 = np.einsum("qi,qjrs->ijrs", mo_t, tmp1)
        tmp3 = np.einsum("rk,ijks->ijk", mo_t, tmp2)
        eri_mo = np.einsum("sl,ijkl->ijkl", mo_t, tmp3)
        return eri_mo

    def build(self):
        """构建 HF 参考态并转换积分。"""
        self._print("构建 HF 参考态...", 1)
        rhf = RHF(self.mol, self.basis_name, verbose=self.verbose - 1)
        rhf.kernel()
        self.e_hf = rhf.e_tot
        self.mol_coeff = rhf.mo_coeff
        self.mo_coeff = rhf.mo_coeff
        self.mo_energy = rhf.mo_energy

        self._nbf = self.mo_coeff.shape[0]
        self._nocc = self.mol.nelectron // 2
        self._nvirt = self._nbf - self._nocc

        self._print(f"  基函数: {self._nbf}, 占据: {self._nocc}, 虚轨道: {self._nvirt}", 2)

        self._print("  ERI AO->MO 4-index 转换中...", 1)
        eri_ao = compute_eri(self.basis_set, self.mol)
        self._eri_mo = self._transform_eri_to_mo(eri_ao, self.mo_coeff)
        del eri_ao

        fock_mo = np.einsum("pq,pi,qj->ij", rhf.fock, self.mo_coeff, self.mo_coeff)
        self._fock_mo = fock_mo

    def kernel(self) -> float:
        """运行 CCD 计算。"""
        self.build()

        self._print("\n" + "=" * 60, 1)
        self._print(f"  CCD/{self.basis_name} 计算", 1)
        self._print("=" * 60, 1)
        self._print(f"\n  HF能量: {self.e_hf:.10f} Eh", 1)

        nocc = self._nocc
        nvirt = self._nvirt
        fock = self._fock_mo
        eri = self._eri_mo

        occ_energies = fock.diagonal()[:nocc]
        vir_energies = fock.diagonal()[nocc:]

        d_ijab = occ_energies[:, None, None, None] + occ_energies[None, :, None, None] - \
                 vir_energies[None, None, :, None] - vir_energies[None, None, None, :]

        antisym_eri = eri - eri.swapaxes(2, 3)

        t2 = np.zeros((nocc, nocc, nvirt, nvirt))
        e_corr = 0.0

        for cycle in range(1, self.max_cycles + 1):
            self.n_cycles = cycle

            t2_new = np.zeros_like(t2)

            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvirt):
                        for b in range(nvirt):
                            ia = nocc + a
                            ib = nocc + b

                            r2 = antisym_eri[i, j, ia, ib]

                            for k in range(nocc):
                                for c in range(nvirt):
                                    ic = nocc + c
                                    r2 -= t2[i, k, a, c] * antisym_eri[j, k, ib, ic]
                                    r2 -= t2[j, k, b, c] * antisym_eri[i, k, ia, ic]

                            for k in range(nocc):
                                for l in range(nocc):
                                    for c in range(nvirt):
                                        for d in range(nvirt):
                                            ic = nocc + c
                                            id_ = nocc + d
                                            r2 += 0.25 * t2[i, j, a, c] * t2[k, l, d, b] * antisym_eri[k, l, ic, id_]

                            t2_new[i, j, a, b] = r2 / d_ijab[i, j, a, b]

            dt = np.max(np.abs(t2_new - t2)) if t2.size > 0 else 0.0
            t2 = t2_new

            e_corr = 0.0
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvirt):
                        for b in range(nvirt):
                            ia = nocc + a
                            ib = nocc + b
                            e_corr += 0.25 * t2[i, j, a, b] * antisym_eri[i, j, ia, ib]

            self._print(
                f"  迭代 {cycle:3d}: E_corr = {e_corr:.10f} Eh  |dt| = {dt:.2e}",
                1,
            )

            if dt < self.conv_tol:
                self._print("\n  CCD 收敛 ✓", 1)
                self.converged = True
                break

        self.t2 = t2
        self.e_corr = e_corr
        self.e_tot = self.e_hf + e_corr

        self._print(f"\n  总能量: {self.e_tot:.10f} Eh", 1)
        self._print(f"  相关能: {e_corr:.10f} Eh", 1)
        self._print("=" * 60 + "\n", 1)

        return self.e_tot