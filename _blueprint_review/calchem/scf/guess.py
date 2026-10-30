from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def _eigh_safe(a: np.ndarray, b: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    安全的对称矩阵对角化。

    优先使用 scipy.linalg.eigh（避免 NumPy 2.5 的 eigh bug），
    回退到 np.linalg.eig + 对称化处理。
    """
    try:
        from scipy.linalg import eigh
        if b is not None:
            return eigh(a, b)
        else:
            return eigh(a)
    except ImportError:
        if b is not None:
            eigval, eigvec = np.linalg.eig(np.linalg.solve(b, a))
            idx = np.argsort(eigval.real)
            return eigval[idx].real, eigvec[:, idx].real
        else:
            eigval, eigvec = np.linalg.eig(a)
            idx = np.argsort(eigval.real)
            return eigval[idx].real, eigvec[:, idx].real


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """安全的矩阵乘法，避免 NumPy 2.5 的 BLAS 崩溃问题。"""
    return np.einsum("ik,kj->ij", a, b)


def _symmetric_orthogonalize(s: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """
    对称正交化：计算 S^(-1/2)

    Parameters
    ----------
    s : np.ndarray
        重叠矩阵 (nbf, nbf)
    threshold : float
        线性相关阈值

    Returns
    -------
    np.ndarray
        正交化矩阵 X (nbf, nbf)，满足 X^T S X = I
    """
    eigval, eigvec = _eigh_safe(s)
    eigval = np.where(eigval > threshold, eigval, 1e15)
    inv_sqrt = 1.0 / np.sqrt(eigval)
    s_half = _matmul(eigvec * inv_sqrt[np.newaxis, :], eigvec.T)
    return s_half


def guess_hcore(
    h_core: np.ndarray,
    overlap: np.ndarray,
    nelectron: int,
    restricted: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    核心哈密顿量对角化初猜。

    解 Hcore C = S C ε，然后构建初始密度矩阵。

    Parameters
    ----------
    h_core : np.ndarray
        核心哈密顿量矩阵 (nbf, nbf)
    overlap : np.ndarray
        重叠矩阵 (nbf, nbf)
    nelectron : int
        电子数
    restricted : bool
        是否为限制的（闭壳层）

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        - restricted=True: 密度矩阵 (nbf, nbf)
        - restricted=False: (D_alpha, D_beta)
    """
    nbf = h_core.shape[0]

    x = _symmetric_orthogonalize(overlap)
    h_ortho = _matmul(_matmul(x.T, h_core), x)
    eigval, eigvec_ortho = _eigh_safe(h_ortho)
    mo_coeff = _matmul(x, eigvec_ortho)

    if restricted:
        nocc = nelectron // 2
        dm = 2.0 * _matmul(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
        return dm
    else:
        na = (nelectron + 1) // 2
        nb = nelectron // 2
        dm_a = _matmul(mo_coeff[:, :na], mo_coeff[:, :na].T)
        dm_b = _matmul(mo_coeff[:, :nb], mo_coeff[:, :nb].T)
        return dm_a, dm_b


def guess_sad(
    mol,
    basis_set,
    nelectron: Optional[int] = None,
    restricted: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    SAD (Superposition of Atomic Densities) 初猜。

    将各原子的密度叠加作为初始密度矩阵。
    简化实现：对每个原子单独做一次原子 ROHF 计算，然后将密度矩阵拼接。

    Parameters
    ----------
    mol : Molecule
        分子对象
    basis_set : BasisSet
        基组对象
    nelectron : int, optional
        电子数，默认从 mol 获取
    restricted : bool
        是否为限制的

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        初始密度矩阵
    """
    from ..integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear

    if nelectron is None:
        nelectron = mol.nelectron

    s = compute_overlap(basis_set, mol)
    t = compute_kinetic(basis_set, mol)
    v = compute_nuclear(basis_set, mol)
    h_core = t + v

    all_shells = basis_set.get_all_shells(mol)
    shells_by_atom, nbf = basis_set.build_molecule_basis(mol)

    dm_total = np.zeros((nbf, nbf))

    for iatom in range(mol.natom):
        atom_shells = shells_by_atom[iatom]
        if len(atom_shells) == 0:
            continue

        nbf_atom = sum(sh.n_functions for sh in atom_shells)
        start = 0
        for i in range(iatom):
            for sh in shells_by_atom[i]:
                start += sh.n_functions
        end = start + nbf_atom

        z = mol.atomic_numbers[iatom]
        atom_ne = z

        h_atom = h_core[start:end, start:end]
        s_atom = s[start:end, start:end]

        try:
            x_atom = _symmetric_orthogonalize(s_atom)
            h_ortho = _matmul(_matmul(x_atom.T, h_atom), x_atom)
            eigval, eigvec_ortho = _eigh_safe(h_ortho)
            mo_coeff_atom = _matmul(x_atom, eigvec_ortho)

            nocc_atom = atom_ne // 2
            dm_atom = 2.0 * _matmul(mo_coeff_atom[:, :nocc_atom], mo_coeff_atom[:, :nocc_atom].T)
        except np.linalg.LinAlgError:
            dm_atom = np.eye(nbf_atom) * (atom_ne / nbf_atom)

        dm_total[start:end, start:end] += dm_atom

    if restricted:
        return dm_total
    else:
        na = (nelectron + 1) // 2
        nb = nelectron // 2
        scale_a = na * 2.0 / nelectron if nelectron > 0 else 1.0
        scale_b = nb * 2.0 / nelectron if nelectron > 0 else 1.0
        return dm_total * scale_a / 2.0, dm_total * scale_b / 2.0


def guess_1e(
    h_core: np.ndarray,
    overlap: np.ndarray,
    nelectron: int,
    restricted: bool = True,
):
    """guess_hcore 的别名，用于向后兼容。"""
    return guess_hcore(h_core, overlap, nelectron, restricted)
