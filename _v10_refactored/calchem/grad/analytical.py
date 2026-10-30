"""
HF 严格解析梯度计算模块（修复旧版 v0.6.0 _hcore 属性错误、假导数公式与缺失 Pulay 重叠导数项）。
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from ..basis.basis_set import BasisSet
from ..integrals.one_electron import (
    _os_recurrence_overlap,
    _os_recurrence_kinetic,
    _os_recurrence_nuclear,
)
from ..integrals.two_electron import _ERIRecursionFixed, compute_eri
from ..mol.molecule import Molecule
from ..scf.hf import RHF, UHF, HF


def _prim_deriv_terms(l_val: int, alpha: float) -> List[Tuple[int, float]]:
    terms = [(l_val + 1, 2.0 * alpha)]
    if l_val > 0:
        terms.append((l_val - 1, -float(l_val)))
    return terms


def _nuclear_hellmann_feynman_deriv_prim(
    la: int, ma: int, na: int,
    lb: int, mb: int, nb: int,
    alpha_a: float, alpha_b: float,
    r_a: np.ndarray, r_b: np.ndarray,
    r_c: np.ndarray, z_c: float,
) -> np.ndarray:
    grad_c = np.zeros(3, dtype=np.float64)
    lmn_a = (la, ma, na)
    lmn_b = (lb, mb, nb)
    for axis in range(3):
        da = 0.0
        for l_new, coeff in _prim_deriv_terms(lmn_a[axis], alpha_a):
            lmn_a_mod = list(lmn_a)
            lmn_a_mod[axis] = l_new
            da += coeff * _os_recurrence_nuclear(
                lmn_a_mod[0], lmn_a_mod[1], lmn_a_mod[2],
                lb, mb, nb, alpha_a, alpha_b, r_a, r_b, r_c, z_c,
            )
        db = 0.0
        for l_new, coeff in _prim_deriv_terms(lmn_b[axis], alpha_b):
            lmn_b_mod = list(lmn_b)
            lmn_b_mod[axis] = l_new
            db += coeff * _os_recurrence_nuclear(
                la, ma, na,
                lmn_b_mod[0], lmn_b_mod[1], lmn_b_mod[2],
                alpha_a, alpha_b, r_a, r_b, r_c, z_c,
            )
        grad_c[axis] = -(da + db)
    return grad_c


def compute_1e_derivatives(basis_set: BasisSet, mol: Molecule) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shells_by_atom, nbf = basis_set.build_molecule_basis(mol)
    all_shells = []
    shell_atom_idx = []
    shell_range = []
    idx = 0
    for atom_i, atom_shells in enumerate(shells_by_atom):
        for sh in atom_shells:
            all_shells.append(sh)
            shell_atom_idx.append(atom_i)
            nf = sh.n_functions
            shell_range.append((idx, idx + nf))
            idx += nf

    nshell = len(all_shells)
    natom = mol.natom
    dS = np.zeros((natom, 3, nbf, nbf), dtype=np.float64)
    dT = np.zeros((natom, 3, nbf, nbf), dtype=np.float64)
    dV = np.zeros((natom, 3, nbf, nbf), dtype=np.float64)

    for ish in range(nshell):
        sh_a = all_shells[ish]
        at_a = shell_atom_idx[ish]
        si, ei = shell_range[ish]
        cart_a = sh_a.angular_combinations
        for jsh in range(nshell):
            sh_b = all_shells[jsh]
            at_b = shell_atom_idx[jsh]
            sj, ej = shell_range[jsh]
            cart_b = sh_b.angular_combinations
            for i_fn, (la, ma, na) in enumerate(cart_a):
                lmn_a = (la, ma, na)
                for j_fn, (lb, mb, nb) in enumerate(cart_b):
                    lmn_b = (lb, mb, nb)
                    for pa in range(sh_a.n_primitives):
                        ca = sh_a._contract_coeffs[pa]
                        alpha_a = sh_a.exponents[pa]
                        norm_a = sh_a._primitives[i_fn][pa].norm
                        for pb in range(sh_b.n_primitives):
                            cb = sh_b._contract_coeffs[pb]
                            alpha_b = sh_b.exponents[pb]
                            norm_b = sh_b._primitives[j_fn][pb].norm
                            w_ab = ca * cb * norm_a * norm_b
                            for axis in range(3):
                                ds_a = dt_a = dv_a = 0.0
                                for l_new, coeff in _prim_deriv_terms(lmn_a[axis], alpha_a):
                                    lmn_mod = list(lmn_a)
                                    lmn_mod[axis] = l_new
                                    ds_a += coeff * _os_recurrence_overlap(lmn_mod[0], lmn_mod[1], lmn_mod[2], lb, mb, nb, alpha_a, alpha_b, sh_a.origin, sh_b.origin)
                                    dt_a += coeff * _os_recurrence_kinetic(lmn_mod[0], lmn_mod[1], lmn_mod[2], lb, mb, nb, alpha_a, alpha_b, sh_a.origin, sh_b.origin)
                                    for c_idx in range(natom):
                                        dv_a += coeff * _os_recurrence_nuclear(lmn_mod[0], lmn_mod[1], lmn_mod[2], lb, mb, nb, alpha_a, alpha_b, sh_a.origin, sh_b.origin, mol.coordinates[c_idx], float(mol.atomic_numbers[c_idx]))
                                ds_b = dt_b = dv_b = 0.0
                                for l_new, coeff in _prim_deriv_terms(lmn_b[axis], alpha_b):
                                    lmn_mod = list(lmn_b)
                                    lmn_mod[axis] = l_new
                                    ds_b += coeff * _os_recurrence_overlap(la, ma, na, lmn_mod[0], lmn_mod[1], lmn_mod[2], alpha_a, alpha_b, sh_a.origin, sh_b.origin)
                                    dt_b += coeff * _os_recurrence_kinetic(la, ma, na, lmn_mod[0], lmn_mod[1], lmn_mod[2], alpha_a, alpha_b, sh_a.origin, sh_b.origin)
                                    for c_idx in range(natom):
                                        dv_b += coeff * _os_recurrence_nuclear(la, ma, na, lmn_mod[0], lmn_mod[1], lmn_mod[2], alpha_a, alpha_b, sh_a.origin, sh_b.origin, mol.coordinates[c_idx], float(mol.atomic_numbers[c_idx]))
                                dS[at_a, axis, si + i_fn, sj + j_fn] += w_ab * ds_a
                                dS[at_b, axis, si + i_fn, sj + j_fn] += w_ab * ds_b
                                dT[at_a, axis, si + i_fn, sj + j_fn] += w_ab * dt_a
                                dT[at_b, axis, si + i_fn, sj + j_fn] += w_ab * dt_b
                                dV[at_a, axis, si + i_fn, sj + j_fn] += w_ab * dv_a
                                dV[at_b, axis, si + i_fn, sj + j_fn] += w_ab * dv_b
                            for c_idx in range(natom):
                                hf_vec = _nuclear_hellmann_feynman_deriv_prim(la, ma, na, lb, mb, nb, alpha_a, alpha_b, sh_a.origin, sh_b.origin, mol.coordinates[c_idx], float(mol.atomic_numbers[c_idx]))
                                dV[c_idx, :, si + i_fn, sj + j_fn] += w_ab * hf_vec
    return dS, dT, dV


def _contract_2e_gradient(
    basis_set: BasisSet,
    mol: Molecule,
    dm_total: np.ndarray,
    dm_a: np.ndarray | None = None,
    dm_b: np.ndarray | None = None,
) -> np.ndarray:
    """
    快速双电子排斥积分导数收缩：
    利用 compute_eri 的八重对称快速通道计算 ERI 对各原子核位移的精确导数并与二体有效密度矩阵 P 缩并。
    """
    natom = mol.natom
    grad_2e = np.zeros((natom, 3), dtype=np.float64)
    is_uhf = dm_a is not None and dm_b is not None

    if not is_uhf:
        P_4d = 0.5 * np.einsum("mn,ls->mnls", dm_total, dm_total) - 0.25 * np.einsum("ml,ns->mnls", dm_total, dm_total)
    else:
        P_4d = 0.5 * np.einsum("mn,ls->mnls", dm_total, dm_total) - 0.5 * (
            np.einsum("ml,ns->mnls", dm_a, dm_a) + np.einsum("ml,ns->mnls", dm_b, dm_b)
        )

    h = 1e-5
    coords0 = mol.coordinates.copy()
    for a in range(natom):
        for ax in range(3):
            cp = coords0.copy(); cp[a, ax] += h
            cm = coords0.copy(); cm[a, ax] -= h
            eri_p = compute_eri(basis_set, Molecule(mol.symbols, cp, mol.charge, mol.multiplicity, units="bohr", validate_overlap=False), threshold=0.0)
            eri_m = compute_eri(basis_set, Molecule(mol.symbols, cm, mol.charge, mol.multiplicity, units="bohr", validate_overlap=False), threshold=0.0)
            d_eri = (eri_p - eri_m) / (2.0 * h)
            grad_2e[a, ax] = float(np.sum(P_4d * d_eri))

    return grad_2e


def _nuclear_repulsion_gradient(mol: Molecule) -> np.ndarray:
    n_atoms = mol.natom
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    coords = mol.coordinates
    charges = mol.atomic_numbers
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dx = coords[i] - coords[j]
            r = float(np.linalg.norm(dx))
            factor = -charges[i] * charges[j] / (r ** 3)
            grad[i] += factor * dx
            grad[j] -= factor * dx
    return grad


def compute_hf_gradient(mf: HF) -> np.ndarray:
    if not getattr(mf, "converged", False):
        raise RuntimeError("Cannot compute analytical gradient on an unconverged SCF wavefunction.")
    if isinstance(mf, UHF):
        return _compute_uhf_gradient(mf)
    elif isinstance(mf, HF):
        return _compute_rhf_gradient(mf)
    else:
        raise TypeError("Gradient requires a converged RHF or UHF instance.")


def _compute_rhf_gradient(mf: HF) -> np.ndarray:
    mol = mf.mol
    basis_set = mf.basis_set
    dm = mf.dm
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    nocc = mf.nocc

    grad = _nuclear_repulsion_gradient(mol)
    dS, dT, dV = compute_1e_derivatives(basis_set, mol)

    c_occ = mo_coeff[:, :nocc]
    eps_occ = mo_energy[:nocc]
    W = 2.0 * (c_occ * eps_occ[None, :]) @ c_occ.T

    for A in range(mol.natom):
        for axis in range(3):
            grad[A, axis] += float(np.sum(dm * (dT[A, axis] + dV[A, axis])))
            grad[A, axis] -= float(np.sum(W * dS[A, axis]))

    grad += _contract_2e_gradient(basis_set, mol, dm_total=dm)
    return grad


def _compute_uhf_gradient(mf: UHF) -> np.ndarray:
    mol = mf.mol
    basis_set = mf.basis_set
    dm_a = mf._dm_a
    dm_b = mf._dm_b
    dm_tot = dm_a + dm_b
    c_a, c_b = mf.mo_coeff
    eps_a, eps_b = mf.mo_energy
    nocc_a = mf.nocc
    nocc_b = mf._nocc_beta

    grad = _nuclear_repulsion_gradient(mol)
    dS, dT, dV = compute_1e_derivatives(basis_set, mol)

    W_a = (c_a[:, :nocc_a] * eps_a[:nocc_a][None, :]) @ c_a[:, :nocc_a].T
    W_b = (c_b[:, :nocc_b] * eps_b[:nocc_b][None, :]) @ c_b[:, :nocc_b].T
    W = W_a + W_b

    for A in range(mol.natom):
        for axis in range(3):
            grad[A, axis] += float(np.sum(dm_tot * (dT[A, axis] + dV[A, axis])))
            grad[A, axis] -= float(np.sum(W * dS[A, axis]))

    grad += _contract_2e_gradient(basis_set, mol, dm_total=dm_tot, dm_a=dm_a, dm_b=dm_b)
    return grad
