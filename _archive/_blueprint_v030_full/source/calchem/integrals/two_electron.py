"""
双电子积分引擎。

计算电子排斥积分 (Electron Repulsion Integral, ERI):
(ij|kl) = ∫∫ φ_i(r1) φ_j(r1) 1/|r1-r2| φ_k(r2) φ_l(r2) dr1 dr2

方法: Obara-Saika 递推
- 高斯乘积定理 (Gaussian Product Theorem)
- HRR (水平递推关系) 将角动量从 b 转移到 a, 从 d 转移到 c
- VRR (垂直递推关系) 上升 a 和 c 的角动量, 含 ζ_ab ≠ ζ_cd 修正项
- Boys 函数用于径向积分

性能优化:
- Schwarz 不等式预筛选: |(μν|λσ)| ≤ sqrt[(μν|μν)(λσ|λσ)]
- 壳层四重对称性压缩: (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)
- 积分分块与预缓存
- Numba JIT 编译加速

支持 s, p, d, f 等壳层（笛卡尔形式）。

参考:
- Obara & Saika, J. Chem. Phys. 84, 3963 (1986)
- Helgaker, Jorgensen, & Olsen, "Molecular Electronic-Structure Theory"
- Gill et al., J. Comput. Chem. 14, 163 (1993) — Schwarz screening
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

try:
    from numba import njit, prange, float64, int64
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from ..basis.gto import GTOShell
from .one_electron import boys_function_vec


def _gaussian_product(alpha_a, alpha_b, r_a, r_b):
    gamma = alpha_a + alpha_b
    r_p = (alpha_a * r_a + alpha_b * r_b) / gamma
    r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
    K = math.exp(-alpha_a * alpha_b / gamma * r_ab2)
    return gamma, r_p, K


# =====================================================================
# Numba 加速的核心递推函数
# =====================================================================

if HAS_NUMBA:
    @njit(float64[:](int64, float64), cache=True)
    def _boys_numba(n, T):
        if T < 1e-15:
            if n == 0:
                return np.array([1.0])
            return np.zeros(n + 1)
        
        result = np.zeros(n + 1)
        
        result[n] = 0.5 * math.pow(T, -0.5) * math.exp(-T)
        
        for i in range(n, 0, -1):
            result[i - 1] = (2 * T * result[i] + (2 * i - 1) * 0.5 * result[i]) if T > 0 else 0.0
            if i == 1:
                result[0] = 0.5 * math.erf(math.sqrt(T)) * math.pow(math.pi, -0.5)
        
        return result


# =====================================================================
# Obara-Saika 递推核心
# =====================================================================

class _ERIRecursionFixed:
    def __init__(self, alpha_a, alpha_b, alpha_c, alpha_d, r_a, r_b, r_c, r_d):
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b
        self.alpha_c = alpha_c
        self.alpha_d = alpha_d
        self.r_a = np.array(r_a, dtype=float)
        self.r_b = np.array(r_b, dtype=float)
        self.r_c = np.array(r_c, dtype=float)
        self.r_d = np.array(r_d, dtype=float)

        self.zeta_ab = alpha_a + alpha_b
        self.zeta_cd = alpha_c + alpha_d
        self.zeta = self.zeta_ab + self.zeta_cd

        self.r_p = (alpha_a * self.r_a + alpha_b * self.r_b) / self.zeta_ab
        self.r_q = (alpha_c * self.r_c + alpha_d * self.r_d) / self.zeta_cd

        self.pa = self.r_p - self.r_a
        self.pb = self.r_p - self.r_b
        self.qc = self.r_q - self.r_c
        self.qd = self.r_q - self.r_d
        self.pq = self.r_p - self.r_q

        self.ab = self.r_a - self.r_b
        self.cd = self.r_c - self.r_d

        self._cache = {}
        self._ssss_cache = {}

    def _get_ssss_n(self, n):
        if n not in self._ssss_cache:
            zeta_ab, r_p, K_ab = _gaussian_product(self.alpha_a, self.alpha_b, self.r_a, self.r_b)
            zeta_cd, r_q, K_cd = _gaussian_product(self.alpha_c, self.alpha_d, self.r_c, self.r_d)
            zeta = zeta_ab + zeta_cd
            r_pq2 = float(np.dot(r_p - r_q, r_p - r_q))
            T = zeta_ab * zeta_cd / zeta * r_pq2
            f = boys_function_vec(n, T)[n]
            val = 2.0 * math.pi ** 2.5 / (zeta_ab * zeta_cd * math.sqrt(zeta)) * K_ab * K_cd * f
            self._ssss_cache[n] = val
        return self._ssss_cache[n]

    def _get_n(self, la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd, n):
        key = (la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd, n)
        if key in self._cache:
            return self._cache[key]
        val = self._compute_n(la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd, n)
        self._cache[key] = val
        return val

    def _compute_n(self, la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd, n):
        total = la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd
        if total == 0:
            return self._get_ssss_n(n)

        if lb > 0:
            a_plus = (la + 1, ma, na, lb - 1, mb, nb, lc, mc, nc, ld, md, nd, n)
            a_same = (la, ma, na, lb - 1, mb, nb, lc, mc, nc, ld, md, nd, n)
            return self._get_n(*a_plus) + self.ab[0] * self._get_n(*a_same)
        if mb > 0:
            a_plus = (la, ma + 1, na, lb, mb - 1, nb, lc, mc, nc, ld, md, nd, n)
            a_same = (la, ma, na, lb, mb - 1, nb, lc, mc, nc, ld, md, nd, n)
            return self._get_n(*a_plus) + self.ab[1] * self._get_n(*a_same)
        if nb > 0:
            a_plus = (la, ma, na + 1, lb, mb, nb - 1, lc, mc, nc, ld, md, nd, n)
            a_same = (la, ma, na, lb, mb, nb - 1, lc, mc, nc, ld, md, nd, n)
            return self._get_n(*a_plus) + self.ab[2] * self._get_n(*a_same)

        if ld > 0:
            c_plus = (la, ma, na, lb, mb, nb, lc + 1, mc, nc, ld - 1, md, nd, n)
            c_same = (la, ma, na, lb, mb, nb, lc, mc, nc, ld - 1, md, nd, n)
            return self._get_n(*c_plus) + self.cd[0] * self._get_n(*c_same)
        if md > 0:
            c_plus = (la, ma, na, lb, mb, nb, lc, mc + 1, nc, ld, md - 1, nd, n)
            c_same = (la, ma, na, lb, mb, nb, lc, mc, nc, ld, md - 1, nd, n)
            return self._get_n(*c_plus) + self.cd[1] * self._get_n(*c_same)
        if nd > 0:
            c_plus = (la, ma, na, lb, mb, nb, lc, mc, nc + 1, ld, md, nd - 1, n)
            c_same = (la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd - 1, n)
            return self._get_n(*c_plus) + self.cd[2] * self._get_n(*c_same)

        return self._compute_upward_n(la, ma, na, lc, mc, nc, n)

    def _compute_upward_n(self, la, ma, na, lc, mc, nc, n_target):
        total_target = la + ma + na + lc + mc + nc
        max_n = n_target + total_target + 5

        computed = {}
        for n in range(max_n + 1):
            computed[(0, 0, 0, 0, 0, 0, n)] = self._get_ssss_n(n)

        for total in range(total_target + 1):
            current_states = []
            for state in list(computed.keys()):
                la_, ma_, na_, lc_, mc_, nc_, n_ = state
                if la_ + ma_ + na_ + lc_ + mc_ + nc_ == total and n_ <= max_n:
                    current_states.append(state)

            for state in current_states:
                la_, ma_, na_, lc_, mc_, nc_, n_ = state
                val = computed[state]

                if la_ + 1 <= la:
                    new_state = (la_ + 1, ma_, na_, lc_, mc_, nc_, n_)
                    if new_state not in computed:
                        computed[new_state] = self._vrr_up_a_x_fixed(
                            la_, ma_, na_, lc_, mc_, nc_, n_, val, computed
                        )

                if ma_ + 1 <= ma:
                    new_state = (la_, ma_ + 1, na_, lc_, mc_, nc_, n_)
                    if new_state not in computed:
                        computed[new_state] = self._vrr_up_a_y_fixed(
                            la_, ma_, na_, lc_, mc_, nc_, n_, val, computed
                        )

                if na_ + 1 <= na:
                    new_state = (la_, ma_, na_ + 1, lc_, mc_, nc_, n_)
                    if new_state not in computed:
                        computed[new_state] = self._vrr_up_a_z_fixed(
                            la_, ma_, na_, lc_, mc_, nc_, n_, val, computed
                        )

                if lc_ + 1 <= lc:
                    new_state = (la_, ma_, na_, lc_ + 1, mc_, nc_, n_)
                    if new_state not in computed:
                        computed[new_state] = self._vrr_up_c_x_fixed(
                            la_, ma_, na_, lc_, mc_, nc_, n_, val, computed
                        )

                if mc_ + 1 <= mc:
                    new_state = (la_, ma_, na_, lc_, mc_ + 1, nc_, n_)
                    if new_state not in computed:
                        computed[new_state] = self._vrr_up_c_y_fixed(
                            la_, ma_, na_, lc_, mc_, nc_, n_, val, computed
                        )

                if nc_ + 1 <= nc:
                    new_state = (la_, ma_, na_, lc_, mc_, nc_ + 1, n_)
                    if new_state not in computed:
                        computed[new_state] = self._vrr_up_c_z_fixed(
                            la_, ma_, na_, lc_, mc_, nc_, n_, val, computed
                        )

        target_state = (la, ma, na, lc, mc, nc, n_target)
        if target_state in computed:
            full_key = (la, ma, na, 0, 0, 0, lc, mc, nc, 0, 0, 0, n_target)
            self._cache[full_key] = computed[target_state]
            return computed[target_state]
        raise ValueError(f"Failed to compute ({la},{ma},{na}|{lc},{mc},{nc})_{n_target}")

    def _vrr_up_a_x_fixed(self, la, ma, na, lc, mc, nc, n, val, computed):
        result = self.pa[0] * val

        key_n1 = (la, ma, na, lc, mc, nc, n + 1)
        if key_n1 in computed:
            result -= (self.zeta_cd / self.zeta) * self.pq[0] * computed[key_n1]

        if la > 0:
            lower_n = (la - 1, ma, na, lc, mc, nc, n)
            term = computed.get(lower_n, 0.0)
            result += la / (2.0 * self.zeta_ab) * term
            lower_n1 = (la - 1, ma, na, lc, mc, nc, n + 1)
            term_n1 = computed.get(lower_n1, 0.0)
            result -= la * self.zeta_cd / (2.0 * self.zeta * self.zeta_ab) * term_n1

        if lc > 0:
            lower_n1 = (la, ma, na, lc - 1, mc, nc, n + 1)
            term = computed.get(lower_n1, 0.0)
            result += lc / (2.0 * self.zeta) * term

        return result

    def _vrr_up_a_y_fixed(self, la, ma, na, lc, mc, nc, n, val, computed):
        result = self.pa[1] * val
        key_n1 = (la, ma, na, lc, mc, nc, n + 1)
        if key_n1 in computed:
            result -= (self.zeta_cd / self.zeta) * self.pq[1] * computed[key_n1]
        if ma > 0:
            lower_n = (la, ma - 1, na, lc, mc, nc, n)
            term = computed.get(lower_n, 0.0)
            result += ma / (2.0 * self.zeta_ab) * term
            lower_n1 = (la, ma - 1, na, lc, mc, nc, n + 1)
            term_n1 = computed.get(lower_n1, 0.0)
            result -= ma * self.zeta_cd / (2.0 * self.zeta * self.zeta_ab) * term_n1
        if mc > 0:
            lower_n1 = (la, ma, na, lc, mc - 1, nc, n + 1)
            term = computed.get(lower_n1, 0.0)
            result += mc / (2.0 * self.zeta) * term
        return result

    def _vrr_up_a_z_fixed(self, la, ma, na, lc, mc, nc, n, val, computed):
        result = self.pa[2] * val
        key_n1 = (la, ma, na, lc, mc, nc, n + 1)
        if key_n1 in computed:
            result -= (self.zeta_cd / self.zeta) * self.pq[2] * computed[key_n1]
        if na > 0:
            lower_n = (la, ma, na - 1, lc, mc, nc, n)
            term = computed.get(lower_n, 0.0)
            result += na / (2.0 * self.zeta_ab) * term
            lower_n1 = (la, ma, na - 1, lc, mc, nc, n + 1)
            term_n1 = computed.get(lower_n1, 0.0)
            result -= na * self.zeta_cd / (2.0 * self.zeta * self.zeta_ab) * term_n1
        if nc > 0:
            lower_n1 = (la, ma, na, lc, mc, nc - 1, n + 1)
            term = computed.get(lower_n1, 0.0)
            result += nc / (2.0 * self.zeta) * term
        return result

    def _vrr_up_c_x_fixed(self, la, ma, na, lc, mc, nc, n, val, computed):
        result = self.qc[0] * val

        key_n1 = (la, ma, na, lc, mc, nc, n + 1)
        if key_n1 in computed:
            result += (self.zeta_ab / self.zeta) * self.pq[0] * computed[key_n1]

        if lc > 0:
            lower_n = (la, ma, na, lc - 1, mc, nc, n)
            term = computed.get(lower_n, 0.0)
            result += lc / (2.0 * self.zeta_cd) * term
            lower_n1 = (la, ma, na, lc - 1, mc, nc, n + 1)
            term_n1 = computed.get(lower_n1, 0.0)
            result -= lc * self.zeta_ab / (2.0 * self.zeta * self.zeta_cd) * term_n1

        if la > 0:
            lower_n1 = (la - 1, ma, na, lc, mc, nc, n + 1)
            term = computed.get(lower_n1, 0.0)
            result += la / (2.0 * self.zeta) * term

        return result

    def _vrr_up_c_y_fixed(self, la, ma, na, lc, mc, nc, n, val, computed):
        result = self.qc[1] * val
        key_n1 = (la, ma, na, lc, mc, nc, n + 1)
        if key_n1 in computed:
            result += (self.zeta_ab / self.zeta) * self.pq[1] * computed[key_n1]
        if mc > 0:
            lower_n = (la, ma, na, lc, mc - 1, nc, n)
            term = computed.get(lower_n, 0.0)
            result += mc / (2.0 * self.zeta_cd) * term
            lower_n1 = (la, ma, na, lc, mc - 1, nc, n + 1)
            term_n1 = computed.get(lower_n1, 0.0)
            result -= mc * self.zeta_ab / (2.0 * self.zeta * self.zeta_cd) * term_n1
        if ma > 0:
            lower_n1 = (la, ma - 1, na, lc, mc, nc, n + 1)
            term = computed.get(lower_n1, 0.0)
            result += ma / (2.0 * self.zeta) * term
        return result

    def _vrr_up_c_z_fixed(self, la, ma, na, lc, mc, nc, n, val, computed):
        result = self.qc[2] * val
        key_n1 = (la, ma, na, lc, mc, nc, n + 1)
        if key_n1 in computed:
            result += (self.zeta_ab / self.zeta) * self.pq[2] * computed[key_n1]
        if nc > 0:
            lower_n = (la, ma, na, lc, mc, nc - 1, n)
            term = computed.get(lower_n, 0.0)
            result += nc / (2.0 * self.zeta_cd) * term
            lower_n1 = (la, ma, na, lc, mc, nc - 1, n + 1)
            term_n1 = computed.get(lower_n1, 0.0)
            result -= nc * self.zeta_ab / (2.0 * self.zeta * self.zeta_cd) * term_n1
        if na > 0:
            lower_n1 = (la, ma, na - 1, lc, mc, nc, n + 1)
            term = computed.get(lower_n1, 0.0)
            result += na / (2.0 * self.zeta) * term
        return result

    def get(self, la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd):
        return self._get_n(la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd, 0)


# =====================================================================
# 壳层双电子积分
# =====================================================================

def shell_eri(shell_a: GTOShell, shell_b: GTOShell,
              shell_c: GTOShell, shell_d: GTOShell) -> np.ndarray:
    n_a = shell_a.n_functions
    n_b = shell_b.n_functions
    n_c = shell_c.n_functions
    n_d = shell_d.n_functions

    block = np.zeros((n_a, n_b, n_c, n_d))

    r_a = shell_a.origin
    r_b = shell_b.origin
    r_c = shell_c.origin
    r_d = shell_d.origin

    n_prim_a = shell_a.n_primitives
    n_prim_b = shell_b.n_primitives
    n_prim_c = shell_c.n_primitives
    n_prim_d = shell_d.n_primitives

    cart_a = shell_a.cartesian_components()
    cart_b = shell_b.cartesian_components()
    cart_c = shell_c.cartesian_components()
    cart_d = shell_d.cartesian_components()

    for pa in range(n_prim_a):
        alpha_a = shell_a.exponents[pa]
        coef_a = shell_a._contract_coeffs[pa]
        norm_a = shell_a.norm_constants[pa]
        for pb in range(n_prim_b):
            alpha_b = shell_b.exponents[pb]
            coef_b = shell_b._contract_coeffs[pb]
            norm_b = shell_b.norm_constants[pb]
            for pc in range(n_prim_c):
                alpha_c = shell_c.exponents[pc]
                coef_c = shell_c._contract_coeffs[pc]
                norm_c = shell_c.norm_constants[pc]
                for pd_ in range(n_prim_d):
                    alpha_d = shell_d.exponents[pd_]
                    coef_d = shell_d._contract_coeffs[pd_]
                    norm_d = shell_d.norm_constants[pd_]

                    prefactor = coef_a * coef_b * coef_c * coef_d * norm_a * norm_b * norm_c * norm_d

                    total_l_max = 0
                    for (la, ma, na) in cart_a:
                        for (lb, mb, nb) in cart_b:
                            for (lc, mc, nc) in cart_c:
                                for (ld, md, nd) in cart_d:
                                    total_l_max = max(total_l_max, la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd)

                    if total_l_max == 0:
                        zeta_ab, r_p, K_ab = _gaussian_product(alpha_a, alpha_b, r_a, r_b)
                        zeta_cd, r_q, K_cd = _gaussian_product(alpha_c, alpha_d, r_c, r_d)
                        zeta = zeta_ab + zeta_cd
                        r_pq2 = float(np.dot(r_p - r_q, r_p - r_q))
                        T = zeta_ab * zeta_cd / zeta * r_pq2
                        f0 = boys_function_vec(0, T)[0]
                        val = 2.0 * math.pi ** 2.5 / (zeta_ab * zeta_cd * math.sqrt(zeta)) * K_ab * K_cd * f0
                        block[0, 0, 0, 0] += prefactor * val
                    else:
                        rec = _ERIRecursionFixed(alpha_a, alpha_b, alpha_c, alpha_d, r_a, r_b, r_c, r_d)
                        for i, (la, ma, na) in enumerate(cart_a):
                            for j, (lb, mb, nb) in enumerate(cart_b):
                                for k, (lc, mc, nc) in enumerate(cart_c):
                                    for l_idx, (ld, md, nd) in enumerate(cart_d):
                                        val = rec.get(la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd)
                                        block[i, j, k, l_idx] += prefactor * val

    return block


# =====================================================================
# Schwarz 预筛选
# =====================================================================

def compute_schwarz_bounds(shells: List[GTOShell]) -> np.ndarray:
    nshell = len(shells)
    schwarz = np.zeros((nshell, nshell))

    for i in range(nshell):
        for j in range(i, nshell):
            block = shell_eri(shells[i], shells[j], shells[i], shells[j])
            max_val = np.max(np.abs(block))
            schwarz[i, j] = np.sqrt(max_val)
            schwarz[j, i] = schwarz[i, j]

    return schwarz


def schwarz_filter(schwarz: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
    nshell = schwarz.shape[0]
    shell_pairs = []

    for i in range(nshell):
        for j in range(i, nshell):
            if schwarz[i, j] > threshold:
                shell_pairs.append((i, j))

    quadruplets = []
    for (i, j) in shell_pairs:
        for (k, l) in shell_pairs:
            if (i, j) <= (k, l):
                schwarz_est = schwarz[i, j] * schwarz[k, l]
                if schwarz_est > threshold:
                    quadruplets.append((i, j, k, l))

    return quadruplets


# =====================================================================
# 分子积分接口（带预筛选和对称压缩）
# =====================================================================

def compute_eri(basis_set, mol, threshold: float = 1e-12) -> np.ndarray:
    shells_by_atom, nbf = basis_set.build_molecule_basis(mol)
    all_shells = []
    for atom_shells in shells_by_atom:
        for s in atom_shells:
            all_shells.append(s)
    nshell = len(all_shells)

    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf

    if threshold > 0:
        schwarz = compute_schwarz_bounds(all_shells)
        quadruplets = schwarz_filter(schwarz, threshold)
    else:
        quadruplets = []
        for i in range(nshell):
            for j in range(i, nshell):
                for k in range(nshell):
                    for l in range(k, nshell):
                        if (i, j) <= (k, l):
                            quadruplets.append((i, j, k, l))

    eri = np.zeros((nbf, nbf, nbf, nbf))

    def fill_symmetry(i, j, k, l, block):
        si, ei = shell_range[i]
        sj, ej = shell_range[j]
        sk, ek = shell_range[k]
        sl, el_ = shell_range[l]

        eri[si:ei, sj:ej, sk:ek, sl:el_] = block

        if i != j:
            eri[sj:ej, si:ei, sk:ek, sl:el_] = block.transpose(1, 0, 2, 3)
        if k != l:
            eri[si:ei, sj:ej, sl:el_, sk:ek] = block.transpose(0, 1, 3, 2)
            if i != j:
                eri[sj:ej, si:ei, sl:el_, sk:ek] = block.transpose(1, 0, 3, 2)
        if (i, j) != (k, l):
            eri[sk:ek, sl:el_, si:ei, sj:ej] = block.transpose(2, 3, 0, 1)
            if i != j:
                eri[sk:ek, sl:el_, sj:ej, si:ei] = block.transpose(2, 3, 1, 0)
            if k != l:
                eri[sl:el_, sk:ek, si:ei, sj:ej] = block.transpose(3, 2, 0, 1)
                if i != j:
                    eri[sl:el_, sk:ek, sj:ej, si:ei] = block.transpose(3, 2, 1, 0)

    for i, j, k, l in quadruplets:
        block = shell_eri(all_shells[i], all_shells[j],
                          all_shells[k], all_shells[l])
        fill_symmetry(i, j, k, l, block)

    return eri


# =====================================================================
# 直接 Fock 矩阵构建（用于直接 SCF）
# =====================================================================

def build_fock_direct_rhf(h_core: np.ndarray, dm: np.ndarray,
                          basis_set, mol, threshold: float = 1e-12) -> np.ndarray:
    shells_by_atom, nbf = basis_set.build_molecule_basis(mol)
    all_shells = []
    for atom_shells in shells_by_atom:
        for s in atom_shells:
            all_shells.append(s)
    nshell = len(all_shells)

    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf

    if threshold > 0:
        schwarz = compute_schwarz_bounds(all_shells)
        quadruplets = schwarz_filter(schwarz, threshold)
    else:
        quadruplets = []
        for i in range(nshell):
            for j in range(i, nshell):
                for k in range(nshell):
                    for l in range(k, nshell):
                        if (i, j) <= (k, l):
                            quadruplets.append((i, j, k, l))

    fock = h_core.copy()

    for i, j, k, l in quadruplets:
        si, ei = shell_range[i]
        sj, ej = shell_range[j]
        sk, ek = shell_range[k]
        sl, el_ = shell_range[l]

        block = shell_eri(all_shells[i], all_shells[j],
                          all_shells[k], all_shells[l])

        d_block = dm[sk:ek, sl:el_]
        j_contrib = np.einsum("cd,abcd->ab", d_block, block)
        fock[si:ei, sj:ej] += j_contrib

        if i != j:
            fock[sj:ej, si:ei] += j_contrib.transpose()

        k_block = shell_eri(all_shells[i], all_shells[k],
                            all_shells[j], all_shells[l])
        k_contrib = np.einsum("cd,acbd->ab", d_block, k_block)
        fock[si:ei, sj:ej] -= 0.5 * k_contrib

        if i != j:
            fock[sj:ej, si:ei] -= 0.5 * k_contrib.transpose()

    return fock


def build_fock_rhf(h_core: np.ndarray, dm: np.ndarray, eri: np.ndarray) -> np.ndarray:
    j = np.einsum("ls,mnls->mn", dm, eri)
    k = np.einsum("ls,mlns->mn", dm, eri)
    g = j - 0.5 * k
    return h_core + g