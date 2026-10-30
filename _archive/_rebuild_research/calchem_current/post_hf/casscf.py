"""
CASSCF (Complete Active Space Self-Consistent Field) 完全活性空间自洽场。

参考:
- Roos, Taylor, Siegbahn, Chem. Phys. 48, 157 (1980)
- Olsen, Int. J. Quantum Chem. 99, 1 (2004)
- Helgaker, Jørgensen, Olsen, Molecular Electronic-Structure Theory (2000)
- Szabo & Ostlund, Modern Quantum Chemistry (1989)
"""

from __future__ import annotations

from typing import Tuple, List
import math
from itertools import combinations

import numpy as np

from ..basis.basis_set import BasisSet
from ..integrals.two_electron import compute_eri
from ..integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from ..mol.molecule import Molecule
from ..scf.hf import RHF, nuclear_repulsion, _symmetric_orthogonalize, _eigh_safe, _matmul


def generate_cas_determinants(n_orb: int, n_elec: int, multiplicity: int = 1) -> np.ndarray:
    n_alpha = (n_elec + multiplicity - 1) // 2
    n_beta = n_elec - n_alpha

    determinants = []
    for alpha_occ in combinations(range(n_orb), n_alpha):
        for beta_occ in combinations(range(n_orb), n_beta):
            det = np.zeros(n_orb, dtype=np.int8)
            for i in alpha_occ:
                det[i] = 1
            for i in beta_occ:
                if det[i] == 1:
                    det[i] = 2
                else:
                    det[i] = -1
            determinants.append(det)

    return np.array(determinants, dtype=np.int8)


def build_ci_hamiltonian(dets: np.ndarray, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    n_det = len(dets)
    n_orb = dets.shape[1]

    H = np.zeros((n_det, n_det))

    for i in range(n_det):
        H[i, i] = _diag_matrix_element(dets[i], h1, h2)
        for j in range(i + 1, n_det):
            H[i, j] = _off_diag_matrix_element(dets[i], dets[j], h1, h2)
            H[j, i] = H[i, j]

    return H


def _diag_matrix_element(det: np.ndarray, h1: np.ndarray, h2: np.ndarray) -> float:
    n_orb = len(det)
    alpha_occ = [k for k in range(n_orb) if det[k] in (1, 2)]
    beta_occ = [k for k in range(n_orb) if det[k] in (-1, 2)]

    e = 0.0

    for k in alpha_occ:
        e += h1[k, k]
    for k in beta_occ:
        e += h1[k, k]

    for k in alpha_occ:
        for l in alpha_occ:
            e += 0.5 * h2[k, l, k, l] - 0.5 * h2[k, l, l, k]
    for k in beta_occ:
        for l in beta_occ:
            e += 0.5 * h2[k, l, k, l] - 0.5 * h2[k, l, l, k]
    for k in alpha_occ:
        for l in beta_occ:
            e += h2[k, l, k, l]

    return e


def _off_diag_matrix_element(det_i: np.ndarray, det_j: np.ndarray,
                              h1: np.ndarray, h2: np.ndarray) -> float:
    n_orb = len(det_i)

    alpha_diff_i = set()
    alpha_diff_j = set()
    beta_diff_i = set()
    beta_diff_j = set()

    for k in range(n_orb):
        di, dj = det_i[k], det_j[k]
        if di in (1, 2) and dj not in (1, 2):
            alpha_diff_i.add(k)
        if dj in (1, 2) and di not in (1, 2):
            alpha_diff_j.add(k)
        if di in (-1, 2) and dj not in (-1, 2):
            beta_diff_i.add(k)
        if dj in (-1, 2) and di not in (-1, 2):
            beta_diff_j.add(k)

    n_alpha_exc = len(alpha_diff_i)
    n_beta_exc = len(beta_diff_i)
    n_exc = n_alpha_exc + n_beta_exc

    if n_exc > 2:
        return 0.0

    if n_exc == 0:
        return _diag_matrix_element(det_i, h1, h2)

    if n_exc == 1:
        return 0.0

    alpha_occ_i = [k for k in range(n_orb) if det_i[k] in (1, 2)]
    alpha_occ_j = [k for k in range(n_orb) if det_j[k] in (1, 2)]
    beta_occ_i = [k for k in range(n_orb) if det_i[k] in (-1, 2)]
    beta_occ_j = [k for k in range(n_orb) if det_j[k] in (-1, 2)]

    alpha_common = set(alpha_occ_i) & set(alpha_occ_j)
    beta_common = set(beta_occ_i) & set(beta_occ_j)

    if n_alpha_exc == 1 and n_beta_exc == 0:
        p = next(iter(alpha_diff_i))
        q = next(iter(alpha_diff_j))
        val = h1[p, q]
        for k in alpha_common:
            val += h2[p, k, q, k] - h2[p, k, k, q]
        for k in beta_common:
            val += h2[p, k, q, k]
        return val

    if n_alpha_exc == 0 and n_beta_exc == 1:
        p = next(iter(beta_diff_i))
        q = next(iter(beta_diff_j))
        val = h1[p, q]
        for k in beta_common:
            val += h2[p, k, q, k] - h2[p, k, k, q]
        for k in alpha_common:
            val += h2[k, p, k, q]
        return val

    if n_alpha_exc == 2 and n_beta_exc == 0:
        p, q = sorted(list(alpha_diff_i))
        r, s = sorted(list(alpha_diff_j))
        return h2[p, r, q, s] - h2[p, s, q, r]

    if n_alpha_exc == 0 and n_beta_exc == 2:
        p, q = sorted(list(beta_diff_i))
        r, s = sorted(list(beta_diff_j))
        return h2[p, r, q, s] - h2[p, s, q, r]

    if n_alpha_exc == 1 and n_beta_exc == 1:
        p = next(iter(alpha_diff_i))
        q = next(iter(beta_diff_i))
        r = next(iter(alpha_diff_j))
        s = next(iter(beta_diff_j))
        return h2[p, r, q, s]

    return 0.0


def _single_excitation_matrix_element(det_i: np.ndarray, det_j: np.ndarray,
                                       spin: str) -> Tuple[float, int, int]:
    """
    计算两个行列式之间的单激发算符矩阵元的相位和轨道指标。

    <I| a_{p,spin}† a_{q,spin} |J> = phase * delta(其他自旋轨道相同)

    返回: (phase, p, q) 
    如果矩阵元为0，返回 (0.0, -1, -1)
    """
    n_orb = len(det_i)

    if spin == 'alpha':
        occ_i = set(k for k in range(n_orb) if det_i[k] in (1, 2))
        occ_j = set(k for k in range(n_orb) if det_j[k] in (1, 2))
        other_i = set(k for k in range(n_orb) if det_i[k] in (-1, 2))
        other_j = set(k for k in range(n_orb) if det_j[k] in (-1, 2))
    else:
        occ_i = set(k for k in range(n_orb) if det_i[k] in (-1, 2))
        occ_j = set(k for k in range(n_orb) if det_j[k] in (-1, 2))
        other_i = set(k for k in range(n_orb) if det_i[k] in (1, 2))
        other_j = set(k for k in range(n_orb) if det_j[k] in (1, 2))

    if other_i != other_j:
        return (0.0, -1, -1)

    diff_i = occ_i - occ_j
    diff_j = occ_j - occ_i

    if len(diff_i) == 0 and len(diff_j) == 0:
        return (0.0, -1, -1)

    if len(diff_i) != 1 or len(diff_j) != 1:
        return (0.0, -1, -1)

    p = next(iter(diff_i))
    q = next(iter(diff_j))

    all_elems = sorted(list(occ_j) + list(other_i))
    n_before_q = sum(1 for x in all_elems if x < q)

    phase = 1 if n_before_q % 2 == 0 else -1

    return (float(phase), p, q)


def compute_1rdm(dets: np.ndarray, ci_coeffs: np.ndarray,
                  n_active: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算活性空间中的1-RDM（自旋分辨）。

    gamma_pq = <Psi| a_p† a_q |Psi> = sum_IJ c_I c_J <I| a_p† a_q |J>

    返回: (dm_a, dm_b) 分别是alpha和beta自旋的1-RDM
    """
    dm_a = np.zeros((n_active, n_active))
    dm_b = np.zeros((n_active, n_active))

    n_det = len(dets)

    for I in range(n_det):
        c_I = ci_coeffs[I]
        det_I = dets[I]

        alpha_occ_I = [k for k in range(n_active) if det_I[k] in (1, 2)]
        beta_occ_I = [k for k in range(n_active) if det_I[k] in (-1, 2)]

        for p in alpha_occ_I:
            dm_a[p, p] += c_I * c_I
        for p in beta_occ_I:
            dm_b[p, p] += c_I * c_I

        for J in range(I + 1, n_det):
            c_J = ci_coeffs[J]
            det_J = dets[J]

            phase_a, p_a, q_a = _single_excitation_matrix_element(det_I, det_J, 'alpha')
            if phase_a != 0.0:
                val = c_I * c_J * phase_a
                dm_a[p_a, q_a] += val
                dm_a[q_a, p_a] += val

            phase_b, p_b, q_b = _single_excitation_matrix_element(det_I, det_J, 'beta')
            if phase_b != 0.0:
                val = c_I * c_J * phase_b
                dm_b[p_b, q_b] += val
                dm_b[q_b, p_b] += val

    return dm_a, dm_b


def mo_to_active(h1_mo: np.ndarray, h2_mo: np.ndarray,
                 n_inactive: int, n_active: int) -> Tuple[np.ndarray, np.ndarray]:
    h1_act = h1_mo[n_inactive:n_inactive + n_active,
                    n_inactive:n_inactive + n_active].copy()
    h2_act = h2_mo[n_inactive:n_inactive + n_active,
                    n_inactive:n_inactive + n_active,
                    n_inactive:n_inactive + n_active,
                    n_inactive:n_inactive + n_active].copy()

    return h1_act, h2_act


class CASSCF:
    def __init__(
        self,
        mol: Molecule,
        basis: str | BasisSet = "sto-3g",
        n_active: int = 2,
        n_active_elec: int = 2,
        max_iter: int = 50,
        verbose: int = 1,
    ):
        if isinstance(basis, str):
            self.basis_set = BasisSet(basis)
            self.basis_name = basis
        else:
            self.basis_set = basis
            self.basis_name = basis.name

        self.mol = mol.to_bohr()
        self.n_active = n_active
        self.n_active_elec = n_active_elec
        self.max_iter = max_iter
        self.verbose = verbose

        self.e_hf: float = 0.0
        self.e_casscf: float = 0.0
        self.e_tot: float = 0.0
        self.e_nn: float = 0.0
        self.converged: bool = False
        self.n_cycles: int = 0

        self.mo_energy = None
        self.mo_coeff = None
        self.ci_coeffs = None
        self.ci_eigenvalues = None

        self._fock_mo = None
        self._eri_mo = None
        self._hcore_mo = None
        self._hcore_ao = None
        self._s_ao = None
        self._eri_ao = None
        self._nocc = 0
        self._nvirt = 0
        self._nbf = 0

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _transform_eri_to_mo(self, eri_ao, mo_coeff):
        tmp1 = np.einsum("pi,pqrs->iqrs", mo_coeff, eri_ao)
        tmp2 = np.einsum("qj,iqrs->ijrs", mo_coeff, tmp1)
        tmp3 = np.einsum("rk,ijrs->ijks", mo_coeff, tmp2)
        eri_mo = np.einsum("sl,ijks->ijkl", mo_coeff, tmp3)
        return eri_mo

    def build(self):
        self._print("构建 HF 参考态...", 1)
        rhf = RHF(self.mol, self.basis_name, verbose=self.verbose - 1)
        rhf.kernel()
        self.e_hf = rhf.e_tot
        self.mo_coeff = rhf.mo_coeff.copy()
        self.mo_energy = rhf.mo_energy.copy()

        self._nbf = self.mo_coeff.shape[0]
        self._nocc = self.mol.nelectron // 2
        self._nvirt = self._nbf - self._nocc

        self._s_ao = rhf._s
        self._hcore_ao = rhf._hcore
        self.e_nn = nuclear_repulsion(self.mol)

        self._print(f"  基函数: {self._nbf}, 占据: {self._nocc}, 虚轨道: {self._nvirt}", 2)

        self._print("  ERI AO->MO 4-index 转换中...", 1)
        self._eri_ao = compute_eri(self.basis_set, self.mol)
        self._eri_mo = self._transform_eri_to_mo(self._eri_ao, self.mo_coeff)

        self._hcore_mo = np.einsum("pq,pi,qj->ij", self._hcore_ao, self.mo_coeff, self.mo_coeff)

        fock_ao = rhf.fock
        self._fock_mo = np.einsum("pq,pi,qj->ij", fock_ao, self.mo_coeff, self.mo_coeff)

    def _compute_total_energy_from_rdm(self, dm_a_mo, dm_b_mo):
        """
        用MO基的1-RDM和2-RDM计算总能量。
        但2-RDM计算复杂，我们直接用CI本征值+非活性能量来计算。
        这里保留函数用于验证。
        """
        dm_tot = dm_a_mo + dm_b_mo

        e_1e = np.einsum("pq,pq->", dm_tot, self._hcore_mo)

        j = np.einsum("ls,pqls->pq", dm_tot, self._eri_mo)
        e_j = 0.5 * np.einsum("pq,pq->", dm_tot, j)

        k_a = np.einsum("ls,plqs->pq", dm_a_mo, self._eri_mo)
        k_b = np.einsum("ls,plqs->pq", dm_b_mo, self._eri_mo)
        e_k = -0.5 * (np.einsum("pq,pq->", dm_a_mo, k_a) + np.einsum("pq,pq->", dm_b_mo, k_b))

        return e_1e + e_j + e_k + self.e_nn

    def _compute_inactive_energy(self, n_inactive: int) -> float:
        """
        计算非活性轨道的能量（双占据闭壳层）。
        E_inactive = 2 sum_i h_ii + sum_ij (2 J_ij - K_ij)
        """
        e = 0.0
        for i in range(n_inactive):
            e += 2.0 * self._hcore_mo[i, i]

        for i in range(n_inactive):
            for j in range(n_inactive):
                e += 2.0 * self._eri_mo[i, j, i, j] - self._eri_mo[i, j, j, i]

        return e

    def _build_generalized_fock_mo(self, dm_a_act: np.ndarray, dm_b_act: np.ndarray,
                                    n_inactive: int) -> np.ndarray:
        """
        构建MO基的广义Fock矩阵（用于CASSCF轨道优化）。

        广义Fock矩阵:
        F_pq = h_pq + sum_rs [D_rs (pq|rs) - d_rs (pr|qs)]

        其中 D_rs = dm_a[rs] + dm_b[rs] (总密度)
        d_rs = dm_a[rs] (alpha自旋密度，对于alpha分量)

        我们构建的是空间轨道的Fock矩阵，使用平均Fock算符。

        分区：非活性(i,j) / 活性(t,u) / 虚轨道(a,b)
        """
        n_active = self.n_active
        nbf = self._nbf

        dm_a_full = np.zeros((nbf, nbf))
        dm_b_full = np.zeros((nbf, nbf))

        for i in range(n_inactive):
            dm_a_full[i, i] = 1.0
            dm_b_full[i, i] = 1.0

        for i in range(n_active):
            for j in range(n_active):
                dm_a_full[n_inactive + i, n_inactive + j] = dm_a_act[i, j]
                dm_b_full[n_inactive + i, n_inactive + j] = dm_b_act[i, j]

        dm_tot = dm_a_full + dm_b_full

        j = np.einsum("rs,pqrs->pq", dm_tot, self._eri_mo)
        k_a = np.einsum("rs,prqs->pq", dm_a_full, self._eri_mo)
        k_b = np.einsum("rs,prqs->pq", dm_b_full, self._eri_mo)

        fock_a = self._hcore_mo + j - k_a
        fock_b = self._hcore_mo + j - k_b

        fock_avg = 0.5 * (fock_a + fock_b)

        return fock_avg

    def _level_shift(self, fock_mo: np.ndarray, n_inactive: int,
                      n_active: int, shift: float = 0.5) -> np.ndarray:
        """
        对非活性-虚轨道和活性-虚轨道之间施加能级移动，
        防止轨道在优化中发生不合理的旋转。
        """
        nbf = self._nbf
        fock_shifted = fock_mo.copy()

        for i in range(n_inactive):
            for a in range(n_inactive + n_active, nbf):
                fock_shifted[i, a] += shift
                fock_shifted[a, i] += shift

        for t in range(n_inactive, n_inactive + n_active):
            for a in range(n_inactive + n_active, nbf):
                fock_shifted[t, a] += shift * 0.5
                fock_shifted[a, t] += shift * 0.5

        return fock_shifted

    def kernel(self) -> float:
        self.build()

        self._print("\n" + "=" * 60, 1)
        self._print(
            f"  CASSCF({self.n_active_elec},{self.n_active})/{self.basis_name} 计算",
            1,
        )
        self._print("=" * 60, 1)
        self._print(f"\n  HF能量: {self.e_hf:.10f} Eh", 1)
        self._print(f"  核排斥能: {self.e_nn:.10f} Eh", 1)
        self._print(f"  活性空间: ({self.n_active_elec}e, {self.n_active}o)", 1)

        n_inactive = self._nocc - self.n_active_elec // 2

        if n_inactive < 0:
            raise ValueError(
                f"活性电子数 ({self.n_active_elec}) 超过 HF 占据数 ({self._nocc})"
            )

        self._print(f"  非活性轨道: {n_inactive}", 2)

        self._print(f"  行列式生成中...", 1)
        dets = generate_cas_determinants(self.n_active, self.n_active_elec)
        n_det = len(dets)
        self._print(f"  行列式数: {n_det}", 1)

        prev_energy = float('inf')
        energy_history = []

        for mscf_iter in range(1, self.max_iter + 1):
            self.n_cycles = mscf_iter

            h1_act, h2_act = mo_to_active(self._hcore_mo, self._eri_mo, n_inactive, self.n_active)

            for i in range(self.n_active):
                for j in range(self.n_active):
                    for k in range(n_inactive):
                        h1_act[i, j] += 2.0 * self._eri_mo[k, n_inactive+i, k, n_inactive+j]
                        h1_act[i, j] -= self._eri_mo[k, n_inactive+i, n_inactive+j, k]

            self._print("  构建 CI 哈密顿矩阵...", 2)
            H_ci = build_ci_hamiltonian(dets, h1_act, h2_act)

            self._print("  对角化 CI 哈密顿矩阵...", 2)
            eigvals, eigvecs = np.linalg.eigh(H_ci)

            ci_coeffs = eigvecs[:, 0]
            self.ci_coeffs = ci_coeffs
            self.ci_eigenvalues = eigvals

            e_ci = eigvals[0]

            e_inactive = self._compute_inactive_energy(n_inactive)
            e_casscf_total = e_ci + e_inactive + self.e_nn

            energy_history.append(e_casscf_total)
            de = abs(e_casscf_total - prev_energy)

            self._print(
                f"  MCSCF 迭代 {mscf_iter:3d}: E = {e_casscf_total:.10f} Eh  dE = {de:.2e}",
                1,
            )

            if de < 1e-8 and mscf_iter > 1:
                self.converged = True
                self._print("\n  CASSCF 收敛 ✓", 1)
                break

            prev_energy = e_casscf_total

            dm_a_act, dm_b_act = compute_1rdm(dets, ci_coeffs, self.n_active)

            self._update_orbitals_natural(dm_a_act, dm_b_act, n_inactive, damping=0.7)

        self.e_casscf = e_casscf_total
        self.e_tot = e_casscf_total

        self._print(f"\n  CASSCF 能量: {self.e_tot:.10f} Eh", 1)
        self._print(f"  相关能修正: {self.e_tot - self.e_hf:.10f} Eh", 1)
        self._print("=" * 60 + "\n", 1)

        return self.e_tot

    def _compute_natural_orbitals(self, dm_a_act: np.ndarray, dm_b_act: np.ndarray,
                                   n_inactive: int) -> np.ndarray:
        """
        计算活性空间的自然轨道（1-RDM的本征态），
        返回从旧活性轨道到新活性轨道的旋转矩阵。

        总1-RDM = dm_a + dm_b
        自然轨道是总1-RDM的本征向量，按占据数降序排列。
        """
        dm_tot_act = dm_a_act + dm_b_act

        occ_num, rot_mat = np.linalg.eigh(dm_tot_act)

        idx = np.argsort(occ_num)[::-1]
        occ_num = occ_num[idx]
        rot_mat = rot_mat[:, idx]

        return rot_mat

    def _update_orbitals_natural(self, dm_a_act: np.ndarray, dm_b_act: np.ndarray,
                                  n_inactive: int, damping: float = 0.5):
        """
        使用自然轨道更新轨道。

        步骤：
        1. 计算活性空间的自然轨道旋转矩阵
        2. 对旋转施加阻尼（与旧轨道混合）
        3. 非活性轨道和虚轨道保持不变
        4. 更新MO系数和积分
        """
        n_active = self.n_active
        nbf = self._nbf

        rot_act = self._compute_natural_orbitals(dm_a_act, dm_b_act, n_inactive)

        if damping > 0:
            identity = np.eye(n_active)
            rot_act = (1 - damping) * rot_act + damping * identity
            for i in range(n_active):
                norm = np.linalg.norm(rot_act[:, i])
                rot_act[:, i] /= norm
            u, _, vt = np.linalg.svd(rot_act)
            rot_act = u @ vt

        full_rot = np.eye(nbf)
        act_slice = slice(n_inactive, n_inactive + n_active)
        full_rot[act_slice, act_slice] = rot_act

        mo_new = self.mo_coeff @ full_rot

        s_mo = self.mo_coeff.T @ self._s_ao @ self.mo_coeff
        s_new = mo_new.T @ self._s_ao @ mo_new
        for i in range(nbf):
            norm = np.sqrt(s_new[i, i])
            mo_new[:, i] /= norm

        self.mo_coeff = mo_new

        self._hcore_mo = np.einsum("pq,pi,qj->ij", self._hcore_ao, self.mo_coeff, self.mo_coeff)
        self._eri_mo = self._transform_eri_to_mo(self._eri_ao, self.mo_coeff)

        fock_mo_diag = np.einsum("pq,pq->p", self._hcore_mo, np.eye(nbf))
        self.mo_energy = fock_mo_diag
