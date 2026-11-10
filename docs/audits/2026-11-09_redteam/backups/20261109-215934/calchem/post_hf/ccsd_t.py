"""
CCSD(T) 三重激发修正实现。

实现 CCSD(T) 的 (T) 修正计算，支持冻芯近似。

参考:
- Raghavachari et al., Chem. Phys. Lett. 190, 421 (1992)
- Scuseria & Schaefer, J. Chem. Phys. 94, 7221 (1991)
"""

from __future__ import annotations

import numpy as np

from ..scf.hf import HF
from .ccsd import CCSD


class CCSD_T:
    def __init__(self, hf: HF, frozen_core: bool = True):
        self.hf = hf
        self.frozen_core = frozen_core
        self._ccsd = None
        self._e_t = None
        self.e_corr = 0.0
        self.e_tot = 0.0
        self.converged = False

    @property
    def energy(self) -> float:
        return self.e_tot

    @property
    def e_t(self) -> float | None:
        return self._e_t

    @property
    def mo_energy(self):
        return getattr(self.hf, "mo_energy", None)

    @property
    def mo_occ(self):
        return getattr(self.hf, "mo_occ", None)

    def kernel(self) -> float:
        return self.run()

    def run(self) -> float:
        print(f"\n{'='*60}")
        print(f"  CCSD(T) 计算")
        print(f"{'='*60}")

        self._ccsd = CCSD(self.hf, frozen_core=self.frozen_core)
        e_ccsd = self._ccsd.run()

        if not self._ccsd.converged:
            print("  CCSD 未收敛，跳过 (T) 修正")
            self.converged = False
            self.e_corr = self._ccsd.e_corr
            self.e_tot = e_ccsd
            return e_ccsd

        e_t = self._compute_triples_correction()
        self._e_t = e_t

        e_total = e_ccsd + e_t
        self.converged = True
        self.e_corr = self._ccsd.e_corr + e_t
        self.e_tot = e_total

        print(f"\n  (T) 修正: {e_t:.12f} Eh")
        print(f"  CCSD(T) 总能量: {e_total:.12f} Eh")
        print(f"{'='*60}")

        return e_total

    def _compute_triples_correction(self) -> float:
        nocc = self._ccsd.nocc
        nvirt = self._ccsd.nvirt
        nfrz = self._ccsd._nfrz

        eri = self._ccsd._eri_mo
        t1 = self._ccsd._t1
        t2 = self._ccsd._t2

        f_ii = np.diag(self._ccsd._fock_mo)[:nocc]
        f_ab = np.diag(self._ccsd._fock_mo)[nocc:]

        e_t = 0.0

        for i in range(nfrz, nocc):
            for j in range(i + 1, nocc):
                for k in range(j + 1, nocc):
                    for a in range(nvirt):
                        for b in range(a + 1, nvirt):
                            for c in range(b + 1, nvirt):
                                numerator = self._triples_amplitude(i, j, k, a, b, c,
                                                                    eri, t1, t2)

                                denom = f_ii[i] + f_ii[j] + f_ii[k] - \
                                        f_ab[a] - f_ab[b] - f_ab[c]

                                e_t += numerator ** 2 / denom

        return e_t

    def _triples_amplitude(self, i, j, k, a, b, c, eri, t1, t2):
        val = eri[i, j, a, b] * t2[k, k, c, c]
        val += eri[i, j, a, c] * t2[k, k, b, b]
        val += eri[i, j, b, c] * t2[k, k, a, a]

        val += eri[i, k, a, b] * t2[j, j, c, c]
        val += eri[i, k, a, c] * t2[j, j, b, b]
        val += eri[i, k, b, c] * t2[j, j, a, a]

        val += eri[j, k, a, b] * t2[i, i, c, c]
        val += eri[j, k, a, c] * t2[i, i, b, b]
        val += eri[j, k, b, c] * t2[i, i, a, a]

        val += 0.5 * eri[i, j, a, b] * t1[k, c] * t1[k, c]
        val += 0.5 * eri[i, j, a, c] * t1[k, b] * t1[k, b]
        val += 0.5 * eri[i, j, b, c] * t1[k, a] * t1[k, a]

        return val


CCSDT = CCSD_T
