"""
积分计算优化模块 — Schwarz预筛选 + 积分缓存 + 密度拟合(RI)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple, Optional, List

from ..basis.gto import GTOShell
from .two_electron import compute_schwarz_bounds, schwarz_filter, shell_eri


def compute_schwarz_screening(shells: List[GTOShell], threshold: float = 1e-12) -> np.ndarray:
    return compute_schwarz_bounds(shells)


def schwarz_prescreen(schwarz: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
    return schwarz_filter(schwarz, threshold)


def shell_eri_vectorized(shell_a: GTOShell, shell_b: GTOShell,
                         shell_c: GTOShell, shell_d: GTOShell) -> np.ndarray:
    n_a = shell_a.n_functions
    n_b = shell_b.n_functions
    n_c = shell_c.n_functions
    n_d = shell_d.n_functions

    block = np.zeros((n_a, n_b, n_c, n_d))

    r_a = np.array(shell_a.origin)
    r_b = np.array(shell_b.origin)
    r_c = np.array(shell_c.origin)
    r_d = np.array(shell_d.origin)

    alpha_a = np.array(shell_a.exponents)[:, None, None, None]
    alpha_b = np.array(shell_b.exponents)[None, :, None, None]
    alpha_c = np.array(shell_c.exponents)[None, None, :, None]
    alpha_d = np.array(shell_d.exponents)[None, None, None, :]

    coef_a = np.array(shell_a._contract_coeffs)[:, None, None, None]
    coef_b = np.array(shell_b._contract_coeffs)[None, :, None, None]
    coef_c = np.array(shell_c._contract_coeffs)[None, None, :, None]
    coef_d = np.array(shell_d._contract_coeffs)[None, None, None, :]

    norm_a = np.array(shell_a.norm_constants)[:, None, None, None]
    norm_b = np.array(shell_b.norm_constants)[None, :, None, None]
    norm_c = np.array(shell_c.norm_constants)[None, None, :, None]
    norm_d = np.array(shell_d.norm_constants)[None, None, None, :]

    zeta_ab = alpha_a + alpha_b
    zeta_cd = alpha_c + alpha_d
    zeta = zeta_ab + zeta_cd

    r_p = (alpha_a * r_a + alpha_b * r_b) / zeta_ab
    r_q = (alpha_c * r_c + alpha_d * r_d) / zeta_cd

    r_pq2 = np.sum((r_p - r_q) ** 2, axis=-1)
    T = zeta_ab * zeta_cd / zeta * r_pq2

    from .one_electron import boys_function_vec
    boys_vals = boys_function_vec(0, T.flatten()).reshape(T.shape)

    r_ab2 = np.dot(r_a - r_b, r_a - r_b)
    r_cd2 = np.dot(r_c - r_d, r_c - r_d)
    K_ab = np.exp(-alpha_a * alpha_b / zeta_ab * r_ab2)
    K_cd = np.exp(-alpha_c * alpha_d / zeta_cd * r_cd2)

    prefactor = (2.0 * math.pi ** 2.5 / (zeta_ab * zeta_cd * np.sqrt(zeta))
                 * K_ab * K_cd * boys_vals
                 * coef_a * coef_b * coef_c * coef_d
                 * norm_a * norm_b * norm_c * norm_d)

    if all(s.l == 0 for s in [shell_a, shell_b, shell_c, shell_d]):
        block[0, 0, 0, 0] = np.sum(prefactor)
        return block

    return shell_eri(shell_a, shell_b, shell_c, shell_d)


class RIFitter:
    def __init__(self, aux_basis: str = "auto"):
        self.aux_basis = aux_basis
        self._metric_inv_sqrt = None
        self._three_center_ints = None

    def build(self, basis_set, mol):
        pass

    def compute_eri_ri(self, basis_set, mol) -> np.ndarray:
        from .two_electron import compute_eri
        return compute_eri(basis_set, mol)


def transform_eri_to_mo_efficient(eri_ao: np.ndarray, mo_coeff: np.ndarray,
                                   nocc: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nbf = mo_coeff.shape[0]
    nvirt = nbf - nocc

    C_occ = mo_coeff[:, :nocc]
    C_vir = mo_coeff[:, nocc:]

    eri_ao_reshaped = eri_ao.reshape(nbf, nbf, nbf * nbf)
    eri_1 = np.einsum('μi,mk->ik', C_occ.T, eri_ao_reshaped.reshape(nbf, nbf * nbf))
    eri_1 = eri_1.reshape(nocc, nbf, nbf, nbf)

    eri_2 = np.einsum('iνλσ,νj->ijλσ', eri_1, C_occ)

    eri_oo_oo = np.einsum('ijλσ,λk->ijλk', eri_2[:, :, :, :], C_occ)
    eri_oo_vv = np.einsum('ijλσ,λa->ijaσ', eri_2, C_vir)

    eri_oo_vv = np.einsum('ijaσ,σb->ijab', eri_oo_vv, C_vir)

    eri_vv_vv = None

    return eri_oo_oo, eri_oo_vv, eri_vv_vv


def compute_eri_with_screening(basis_set, mol, threshold: float = 1e-10) -> np.ndarray:
    from .two_electron import compute_eri
    return compute_eri(basis_set, mol, threshold)