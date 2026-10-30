"""
HF 解析梯度计算模块。

实现 RHF 和 UHF 的解析梯度。
"""

from __future__ import annotations

import numpy as np

from ..basis.basis_set import BasisSet
from ..integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from ..integrals.two_electron import compute_eri
from ..mol.molecule import Molecule
from ..scf.hf import RHF, UHF


def compute_hf_gradient(mf) -> np.ndarray:
    """计算 HF 解析梯度。"""
    if isinstance(mf, RHF):
        return _compute_rhf_gradient(mf)
    elif isinstance(mf, UHF):
        return _compute_uhf_gradient(mf)
    else:
        raise TypeError("Gradient requires an RHF or UHF object")


def _compute_rhf_gradient(mf: RHF) -> np.ndarray:
    mol = mf.mol
    basis_set = mf.basis_set
    dm = mf.dm
    hcore = mf._hcore
    fock = mf.fock
    eri = mf._eri
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    nocc = mol.nelectron // 2

    n_atoms = mol.natom
    grad = np.zeros((n_atoms, 3))

    grad += _nuclear_repulsion_gradient(mol)
    grad += _core_gradient(dm, hcore, mol, basis_set)
    grad += _two_electron_gradient(dm, eri, mol, basis_set)

    return grad


def _compute_uhf_gradient(mf: UHF) -> np.ndarray:
    mol = mf.mol
    basis_set = mf.basis_set
    dm_a = mf.dm[0]
    dm_b = mf.dm[1]
    hcore = mf._hcore
    eri = mf._eri

    n_atoms = mol.natom
    grad = np.zeros((n_atoms, 3))

    grad += _nuclear_repulsion_gradient(mol)
    grad += _core_gradient(dm_a + dm_b, hcore, mol, basis_set)
    grad += _two_electron_gradient_uhf(dm_a, dm_b, eri, mol, basis_set)

    return grad


def _nuclear_repulsion_gradient(mol: Molecule) -> np.ndarray:
    n_atoms = mol.natom
    grad = np.zeros((n_atoms, 3))
    coords = mol.coordinates
    charges = mol.atomic_numbers

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dx = coords[i] - coords[j]
            r = np.linalg.norm(dx)
            r3 = r ** 3
            factor = charges[i] * charges[j] / r3
            grad[i] += factor * dx
            grad[j] -= factor * dx

    return grad


def _core_gradient(dm: np.ndarray, hcore: np.ndarray, mol: Molecule, basis_set: BasisSet) -> np.ndarray:
    n_atoms = mol.natom
    nbf = dm.shape[0]
    grad = np.zeros((n_atoms, 3))

    for A in range(n_atoms):
        d_t = _compute_kinetic_gradient(basis_set, mol, A)
        d_v = _compute_nuclear_gradient(basis_set, mol, A)
        d_hcore = d_t + d_v
        for comp in range(3):
            grad[A, comp] += np.einsum("mn,mn->", dm, d_hcore[comp])

    return grad


def _two_electron_gradient(dm: np.ndarray, eri: np.ndarray, mol: Molecule, basis_set: BasisSet) -> np.ndarray:
    n_atoms = mol.natom
    nbf = dm.shape[0]
    grad = np.zeros((n_atoms, 3))

    for A in range(n_atoms):
        d_eri = _compute_eri_gradient(basis_set, mol, A)
        for comp in range(3):
            j_grad = np.einsum("ls,mnls->mn", dm, d_eri[comp])
            k_grad = np.einsum("ls,m nsl->mn", dm, d_eri[comp].transpose(0, 1, 3, 2))
            grad[A, comp] += 0.5 * np.einsum("mn,mn->", dm, j_grad) - 0.25 * np.einsum("mn,mn->", dm, k_grad)

    return grad


def _two_electron_gradient_uhf(dm_a: np.ndarray, dm_b: np.ndarray, eri: np.ndarray, mol: Molecule, basis_set: BasisSet) -> np.ndarray:
    n_atoms = mol.natom
    grad = np.zeros((n_atoms, 3))

    for A in range(n_atoms):
        d_eri = _compute_eri_gradient(basis_set, mol, A)
        for comp in range(3):
            j_grad = np.einsum("ls,mnls->mn", dm_a + dm_b, d_eri[comp])
            k_grad_a = np.einsum("ls,m nsl->mn", dm_a, d_eri[comp].transpose(0, 1, 3, 2))
            k_grad_b = np.einsum("ls,m nsl->mn", dm_b, d_eri[comp].transpose(0, 1, 3, 2))
            grad[A, comp] += 0.5 * np.einsum("mn,mn->", dm_a + dm_b, j_grad) - 0.5 * np.einsum("mn,mn->", dm_a, k_grad_a) - 0.5 * np.einsum("mn,mn->", dm_b, k_grad_b)

    return grad


def _compute_kinetic_gradient(basis_set: BasisSet, mol: Molecule, atom_idx: int) -> tuple:
    return _compute_one_electron_gradient(basis_set, mol, atom_idx, 'kinetic')


def _compute_nuclear_gradient(basis_set: BasisSet, mol: Molecule, atom_idx: int) -> tuple:
    return _compute_one_electron_gradient(basis_set, mol, atom_idx, 'nuclear')


def _compute_one_electron_gradient(basis_set: BasisSet, mol: Molecule, atom_idx: int, integral_type: str) -> tuple:
    all_shells = basis_set.get_all_shells(mol)
    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf
    nbf = idx
    result_x = np.zeros((nbf, nbf))
    result_y = np.zeros((nbf, nbf))
    result_z = np.zeros((nbf, nbf))

    for i, shell_i in enumerate(all_shells):
        si, ei = shell_range[i]
        for j, shell_j in enumerate(all_shells):
            sj, ej = shell_range[j]
            if integral_type == 'kinetic':
                block = _kinetic_grad_block(shell_i, shell_j, mol, atom_idx)
            else:
                block = _nuclear_grad_block(shell_i, shell_j, mol, atom_idx)
            result_x[si:ei, sj:ej] += block[0]
            result_y[si:ei, sj:ej] += block[1]
            result_z[si:ei, sj:ej] += block[2]

    return result_x, result_y, result_z


def _kinetic_grad_block(shell_a, shell_b, mol, atom_idx):
    return _compute_grad_block(shell_a, shell_b, mol, atom_idx, _kinetic_integral_deriv)


def _nuclear_grad_block(shell_a, shell_b, mol, atom_idx):
    return _compute_grad_block(shell_a, shell_b, mol, atom_idx, _nuclear_integral_deriv)


def _compute_grad_block(shell_a, shell_b, mol, atom_idx, integral_func):
    n_func_a = shell_a.n_functions
    n_func_b = shell_b.n_functions

    result_x = np.zeros((n_func_a, n_func_b))
    result_y = np.zeros((n_func_a, n_func_b))
    result_z = np.zeros((n_func_a, n_func_b))

    cart_a = shell_a.cartesian_components()
    cart_b = shell_b.cartesian_components()
    r_a = shell_a.origin
    r_b = shell_b.origin

    for pa in range(shell_a.n_primitives):
        alpha_a = shell_a.exponents[pa]
        coef_a = shell_a._contract_coeffs[pa]
        norm_a = shell_a.norm_constants[pa]

        for pb in range(shell_b.n_primitives):
            alpha_b = shell_b.exponents[pb]
            coef_b = shell_b._contract_coeffs[pb]
            norm_b = shell_b.norm_constants[pb]

            prefactor = coef_a * coef_b * norm_a * norm_b

            gamma = alpha_a + alpha_b
            r_p = (alpha_a * r_a + alpha_b * r_b) / gamma
            r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
            K = np.exp(-alpha_a * alpha_b / gamma * r_ab2)

            for i, (la, ma, na) in enumerate(cart_a):
                for j, (lb, mb, nb) in enumerate(cart_b):
                    val = integral_func(la, ma, na, lb, mb, nb, alpha_a, alpha_b, r_a, r_b, r_p, gamma, mol, atom_idx)
                    result_x[i, j] += prefactor * K * val[0]
                    result_y[i, j] += prefactor * K * val[1]
                    result_z[i, j] += prefactor * K * val[2]

    return result_x, result_y, result_z


def _kinetic_integral_deriv(la, ma, na, lb, mb, nb, alpha_a, alpha_b, r_a, r_b, r_p, gamma, mol, atom_idx):
    dx_a = r_a[0] - mol.coordinates[atom_idx][0]
    dy_a = r_a[1] - mol.coordinates[atom_idx][1]
    dz_a = r_a[2] - mol.coordinates[atom_idx][2]

    result = np.zeros(3)

    if la > 0:
        result[0] += -2.0 * alpha_a * la * dx_a ** (la - 1) * dy_a ** ma * dz_a ** na
    if ma > 0:
        result[1] += -2.0 * alpha_a * ma * dx_a ** la * dy_a ** (ma - 1) * dz_a ** na
    if na > 0:
        result[2] += -2.0 * alpha_a * na * dx_a ** la * dy_a ** ma * dz_a ** (na - 1)

    if lb > 0:
        result[0] += -2.0 * alpha_b * lb * dx_a ** lb * dy_a ** mb * dz_a ** nb
    if mb > 0:
        result[1] += -2.0 * alpha_b * mb * dx_a ** lb * dy_a ** (mb - 1) * dz_a ** nb
    if nb > 0:
        result[2] += -2.0 * alpha_b * nb * dx_a ** lb * dy_a ** mb * dz_a ** (nb - 1)

    return result


def _nuclear_integral_deriv(la, ma, na, lb, mb, nb, alpha_a, alpha_b, r_a, r_b, r_p, gamma, mol, atom_idx):
    Z = mol.atomic_numbers[atom_idx]
    dx = r_p[0] - mol.coordinates[atom_idx][0]
    dy = r_p[1] - mol.coordinates[atom_idx][1]
    dz = r_p[2] - mol.coordinates[atom_idx][2]
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    r3 = r ** 3

    result = np.zeros(3)

    if r > 1e-10:
        result[0] = Z * dx / r3
        result[1] = Z * dy / r3
        result[2] = Z * dz / r3

    return result


def _compute_eri_gradient(basis_set: BasisSet, mol: Molecule, atom_idx: int) -> tuple:
    all_shells = basis_set.get_all_shells(mol)
    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf
    nbf = idx
    result_x = np.zeros((nbf, nbf, nbf, nbf))
    result_y = np.zeros((nbf, nbf, nbf, nbf))
    result_z = np.zeros((nbf, nbf, nbf, nbf))

    for i, shell_i in enumerate(all_shells):
        si, ei = shell_range[i]
        for j, shell_j in enumerate(all_shells):
            sj, ej = shell_range[j]
            for k, shell_k in enumerate(all_shells):
                sk, ek = shell_range[k]
                for l, shell_l in enumerate(all_shells):
                    sl, el = shell_range[l]
                    block = _eri_grad_block(shell_i, shell_j, shell_k, shell_l, mol, atom_idx)
                    result_x[si:ei, sj:ej, sk:ek, sl:el] += block[0]
                    result_y[si:ei, sj:ej, sk:ek, sl:el] += block[1]
                    result_z[si:ei, sj:ej, sk:ek, sl:el] += block[2]

    return result_x, result_y, result_z


def _eri_grad_block(shell_a, shell_b, shell_c, shell_d, mol, atom_idx):
    n_func_a = shell_a.n_functions
    n_func_b = shell_b.n_functions
    n_func_c = shell_c.n_functions
    n_func_d = shell_d.n_functions

    result_x = np.zeros((n_func_a, n_func_b, n_func_c, n_func_d))
    result_y = np.zeros((n_func_a, n_func_b, n_func_c, n_func_d))
    result_z = np.zeros((n_func_a, n_func_b, n_func_c, n_func_d))

    cart_a = shell_a.cartesian_components()
    cart_b = shell_b.cartesian_components()
    cart_c = shell_c.cartesian_components()
    cart_d = shell_d.cartesian_components()

    r_a = shell_a.origin
    r_b = shell_b.origin
    r_c = shell_c.origin
    r_d = shell_d.origin

    for pa in range(shell_a.n_primitives):
        for pb in range(shell_b.n_primitives):
            for pc in range(shell_c.n_primitives):
                for pd in range(shell_d.n_primitives):
                    alpha_a = shell_a.exponents[pa]
                    alpha_b = shell_b.exponents[pb]
                    alpha_c = shell_c.exponents[pc]
                    alpha_d = shell_d.exponents[pd]

                    coef = (shell_a._contract_coeffs[pa] * shell_b._contract_coeffs[pb] *
                            shell_c._contract_coeffs[pc] * shell_d._contract_coeffs[pd] *
                            shell_a.norm_constants[pa] * shell_b.norm_constants[pb] *
                            shell_c.norm_constants[pc] * shell_d.norm_constants[pd])

                    gamma_ab = alpha_a + alpha_b
                    gamma_cd = alpha_c + alpha_d
                    gamma = gamma_ab + gamma_cd

                    r_p_ab = (alpha_a * r_a + alpha_b * r_b) / gamma_ab
                    r_p_cd = (alpha_c * r_c + alpha_d * r_d) / gamma_cd
                    r_p = (gamma_ab * r_p_ab + gamma_cd * r_p_cd) / gamma

                    r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
                    r_cd2 = float(np.dot(r_c - r_d, r_c - r_d))

                    K_ab = np.exp(-alpha_a * alpha_b / gamma_ab * r_ab2)
                    K_cd = np.exp(-alpha_c * alpha_d / gamma_cd * r_cd2)

                    dx = r_p[0] - mol.coordinates[atom_idx][0]
                    dy = r_p[1] - mol.coordinates[atom_idx][1]
                    dz = r_p[2] - mol.coordinates[atom_idx][2]
                    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    r3 = r ** 3

                    for i, (la, ma, na) in enumerate(cart_a):
                        for j, (lb, mb, nb) in enumerate(cart_b):
                            for k, (lc, mc, nc) in enumerate(cart_c):
                                for l, (ld, md, nd) in enumerate(cart_d):
                                    if r > 1e-10:
                                        result_x[i, j, k, l] += coef * K_ab * K_cd * dx / r3
                                        result_y[i, j, k, l] += coef * K_ab * K_cd * dy / r3
                                        result_z[i, j, k, l] += coef * K_ab * K_cd * dz / r3

    return result_x, result_y, result_z
