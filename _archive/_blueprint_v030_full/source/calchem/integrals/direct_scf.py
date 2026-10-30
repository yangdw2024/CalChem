"""
直接SCF引擎。
"""

from __future__ import annotations

import numpy as np

from ..basis.basis_set import BasisSet
from ..mol.molecule import Molecule
from ..integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from ..integrals.two_electron import shell_eri, compute_schwarz_bounds, schwarz_filter


def build_fock_direct(
    h_core: np.ndarray,
    dm: np.ndarray,
    basis_set: BasisSet,
    mol: Molecule,
    threshold: float = 1e-12,
) -> np.ndarray:
    """直接SCF Fock矩阵构建 (RHF)。

    遍历bra壳层对 (i<=j) 和所有ket壳层对 (k,l)，
    使用Schwarz预筛选跳过可忽略的四重奏。
    J和K都通过对称性scatter到 (j,i) 位置。

    F = H + J - 0.5*K
    J_{ab} = sum_{cd} (ab|cd) D_{cd}
    K_{ab} = sum_{cd} (ac|bd) D_{cd}
    """
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

    schwarz = compute_schwarz_bounds(all_shells) if threshold > 0 else None

    fock = h_core.copy()

    for i in range(nshell):
        for j in range(i, nshell):
            si, ei = shell_range[i]
            sj, ej = shell_range[j]
            j_block_accum = np.zeros((ei - si, ej - sj))
            k_block_accum = np.zeros((ei - si, ej - sj))

            for k in range(nshell):
                for l in range(nshell):
                    # Schwarz 预筛选
                    if schwarz is not None:
                        if schwarz[i, j] * schwarz[k, l] < threshold:
                            continue

                    sk, ek = shell_range[k]
                    sl, el_ = shell_range[l]

                    # J part: (ij|kl)
                    block = shell_eri(all_shells[i], all_shells[j],
                                      all_shells[k], all_shells[l])
                    d_kl = dm[sk:ek, sl:el_]
                    j_block_accum += np.einsum("cd,abcd->ab", d_kl, block)

                    # K part: (ik|jl)
                    k_block = shell_eri(all_shells[i], all_shells[k],
                                        all_shells[j], all_shells[l])
                    k_block_accum += np.einsum("cd,acbd->ab", d_kl, k_block)

            fock[si:ei, sj:ej] += j_block_accum - 0.5 * k_block_accum
            if i != j:
                fock[sj:ej, si:ei] += j_block_accum.T - 0.5 * k_block_accum.T

    return fock


def build_j_direct(
    dm: np.ndarray,
    basis_set: BasisSet,
    mol: Molecule,
    threshold: float = 1e-12,
) -> np.ndarray:
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

    j = np.zeros((nbf, nbf))

    if threshold > 0:
        schwarz = compute_schwarz_bounds(all_shells)
        quadruplets = schwarz_filter(schwarz, threshold)
    else:
        quadruplets = []
        for i in range(nshell):
            for j_ in range(nshell):
                for k in range(nshell):
                    for l in range(nshell):
                        quadruplets.append((i, j_, k, l))

    for i, j_, k, l in quadruplets:
        si, ei = shell_range[i]
        sj, ej = shell_range[j_]
        sk, ek = shell_range[k]
        sl, el_ = shell_range[l]

        block = shell_eri(all_shells[i], all_shells[j_],
                          all_shells[k], all_shells[l])

        d_block = dm[sk:ek, sl:el_]
        j_contrib = np.einsum("cd,abcd->ab", d_block, block)
        j[si:ei, sj:ej] += j_contrib

    return j


def build_k_direct(
    dm: np.ndarray,
    basis_set: BasisSet,
    mol: Molecule,
    threshold: float = 1e-12,
) -> np.ndarray:
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

    k = np.zeros((nbf, nbf))

    if threshold > 0:
        schwarz = compute_schwarz_bounds(all_shells)
        quadruplets = schwarz_filter(schwarz, threshold)
    else:
        quadruplets = []
        for i in range(nshell):
            for j_ in range(nshell):
                for k_ in range(nshell):
                    for l in range(nshell):
                        quadruplets.append((i, j_, k_, l))

    for i, j_, k_, l in quadruplets:
        si, ei = shell_range[i]
        sj, ej = shell_range[j_]
        sk, ek = shell_range[k_]
        sl, el_ = shell_range[l]

        block = shell_eri(all_shells[i], all_shells[k_],
                          all_shells[j_], all_shells[l])

        d_block = dm[sk:ek, sl:el_]
        k_contrib = np.einsum("cd,abcd->ab", d_block, block)
        k[si:ei, sj:ej] += k_contrib

    return k


def build_fock_direct_uhf(
    h_core: np.ndarray,
    dm_a: np.ndarray,
    dm_b: np.ndarray,
    basis_set: BasisSet,
    mol: Molecule,
    threshold: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """直接SCF Fock矩阵构建 (UHF)。

    遍历bra壳层对 (i<=j) 和所有ket壳层对 (k,l)。
    F_a = H + J(D_a+D_b) - K(D_a)
    F_b = H + J(D_a+D_b) - K(D_b)
    """
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

    schwarz = compute_schwarz_bounds(all_shells) if threshold > 0 else None

    f_a = h_core.copy()
    f_b = h_core.copy()
    dm_total = dm_a + dm_b

    for i in range(nshell):
        for j in range(i, nshell):
            si, ei = shell_range[i]
            sj, ej = shell_range[j]
            j_accum = np.zeros((ei - si, ej - sj))
            k_a_accum = np.zeros((ei - si, ej - sj))
            k_b_accum = np.zeros((ei - si, ej - sj))

            for k in range(nshell):
                for l in range(nshell):
                    if schwarz is not None:
                        if schwarz[i, j] * schwarz[k, l] < threshold:
                            continue

                    sk, ek = shell_range[k]
                    sl, el_ = shell_range[l]

                    # J part: (ij|kl), uses total density
                    block = shell_eri(all_shells[i], all_shells[j],
                                      all_shells[k], all_shells[l])
                    d_kl = dm_total[sk:ek, sl:el_]
                    j_accum += np.einsum("cd,abcd->ab", d_kl, block)

                    # K part: (ik|jl), separate for alpha/beta
                    k_block = shell_eri(all_shells[i], all_shells[k],
                                        all_shells[j], all_shells[l])
                    d_a_kl = dm_a[sk:ek, sl:el_]
                    d_b_kl = dm_b[sk:ek, sl:el_]
                    k_a_accum += np.einsum("cd,acbd->ab", d_a_kl, k_block)
                    k_b_accum += np.einsum("cd,acbd->ab", d_b_kl, k_block)

            f_a[si:ei, sj:ej] += j_accum - k_a_accum
            f_b[si:ei, sj:ej] += j_accum - k_b_accum
            if i != j:
                f_a[sj:ej, si:ei] += j_accum.T - k_a_accum.T
                f_b[sj:ej, si:ei] += j_accum.T - k_b_accum.T

    return f_a, f_b