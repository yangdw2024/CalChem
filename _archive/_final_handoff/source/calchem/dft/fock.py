"""
DFT Fock 矩阵构建模块。

实现 DFT 的 Fock 矩阵:
F_μν = H_core_μν + J_μν + V_xc_μν
"""

from __future__ import annotations

import math

import numpy as np

from ..basis.gto import GTOShell


def build_fock_dft(
    h_core: np.ndarray,
    dm: np.ndarray,
    eri: np.ndarray,
    vxc: np.ndarray,
) -> np.ndarray:
    j = np.einsum("ls,mnls->mn", dm, eri)
    return h_core + j + vxc


def compute_vxc(
    dm: np.ndarray,
    basis_set,
    mol,
    functional,
    grid=None,
    grid_level: str = "medium",
) -> np.ndarray:
    if grid is None:
        from .grid import DFTGrid
        grid = DFTGrid(mol, basis_set, grid_level)

    rho = grid.compute_density(dm, basis_set, mol)
    grad_rho = _compute_density_gradient(dm, basis_set, mol, grid)
    
    tau = None
    if functional.type == "meta-gga":
        tau = grid.compute_kinetic_density(dm, basis_set, mol)
    
    exc, vxc_r, vxc_gamma, vxc_tau = functional.compute_full(rho, grad_rho, tau=tau)

    nbf = dm.shape[0]
    vxc = np.zeros((nbf, nbf))

    all_shells = basis_set.get_all_shells(mol)
    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf

    for i, shell_i in enumerate(all_shells):
        si, ei = shell_range[i]
        for j, shell_j in enumerate(all_shells):
            sj, ej = shell_range[j]
            block = _compute_vxc_block(
                shell_i, shell_j, vxc_r, grid,
                vxc_gamma=vxc_gamma,
                vxc_tau=vxc_tau,
                grad_rho=grad_rho,
            )
            vxc[si:ei, sj:ej] += block

    return vxc


def _compute_density_gradient(
    dm: np.ndarray,
    basis_set,
    mol,
    grid,
) -> np.ndarray:
    n_points = grid.n_points
    grad_rho = np.zeros((n_points, 3))

    all_shells = basis_set.get_all_shells(mol)
    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf

    for i, shell_i in enumerate(all_shells):
        si, ei = shell_range[i]
        for j, shell_j in enumerate(all_shells):
            sj, ej = shell_range[j]
            block_dm = dm[si:ei, sj:ej]
            block_grad = _compute_grad_block(shell_i, shell_j, grid)
            for comp in range(3):
                grad_rho[:, comp] += np.einsum("ij,ijp->p", block_dm, block_grad[comp])

    return grad_rho


def _compute_grad_block(
    shell_a: GTOShell,
    shell_b: GTOShell,
    grid,
) -> tuple:
    n_func_a = shell_a.n_functions
    n_func_b = shell_b.n_functions
    n_points = grid.n_points

    result_x = np.zeros((n_func_a, n_func_b, n_points))
    result_y = np.zeros((n_func_a, n_func_b, n_points))
    result_z = np.zeros((n_func_a, n_func_b, n_points))

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

            for i, (la, ma, na) in enumerate(cart_a):
                for j, (lb, mb, nb) in enumerate(cart_b):
                    phi_a = grid._eval_gto(la, ma, na, alpha_a, r_a)
                    phi_b = grid._eval_gto(lb, mb, nb, alpha_b, r_b)

                    dphi_a_x = _gto_deriv_x(la, ma, na, alpha_a, r_a, grid.points)
                    dphi_a_y = _gto_deriv_y(la, ma, na, alpha_a, r_a, grid.points)
                    dphi_a_z = _gto_deriv_z(la, ma, na, alpha_a, r_a, grid.points)

                    dphi_b_x = _gto_deriv_x(lb, mb, nb, alpha_b, r_b, grid.points)
                    dphi_b_y = _gto_deriv_y(lb, mb, nb, alpha_b, r_b, grid.points)
                    dphi_b_z = _gto_deriv_z(lb, mb, nb, alpha_b, r_b, grid.points)

                    result_x[i, j] += prefactor * (dphi_a_x * phi_b + phi_a * dphi_b_x)
                    result_y[i, j] += prefactor * (dphi_a_y * phi_b + phi_a * dphi_b_y)
                    result_z[i, j] += prefactor * (dphi_a_z * phi_b + phi_a * dphi_b_z)

    return result_x, result_y, result_z


def _gto_deriv_x(l, m, n, alpha, origin, points):
    dx = points[:, 0] - origin[0]
    dy = points[:, 1] - origin[1]
    dz = points[:, 2] - origin[2]
    r2 = dx ** 2 + dy ** 2 + dz ** 2
    exp_val = np.exp(-alpha * r2)

    if l > 0:
        return l * dx ** (l - 1) * dy ** m * dz ** n * exp_val - 2.0 * alpha * dx ** (l + 1) * dy ** m * dz ** n * exp_val
    else:
        return -2.0 * alpha * dx * dy ** m * dz ** n * exp_val


def _gto_deriv_y(l, m, n, alpha, origin, points):
    dx = points[:, 0] - origin[0]
    dy = points[:, 1] - origin[1]
    dz = points[:, 2] - origin[2]
    r2 = dx ** 2 + dy ** 2 + dz ** 2
    exp_val = np.exp(-alpha * r2)

    if m > 0:
        return dx ** l * m * dy ** (m - 1) * dz ** n * exp_val - 2.0 * alpha * dx ** l * dy ** (m + 1) * dz ** n * exp_val
    else:
        return -2.0 * alpha * dx ** l * dy * dz ** n * exp_val


def _gto_deriv_z(l, m, n, alpha, origin, points):
    dx = points[:, 0] - origin[0]
    dy = points[:, 1] - origin[1]
    dz = points[:, 2] - origin[2]
    r2 = dx ** 2 + dy ** 2 + dz ** 2
    exp_val = np.exp(-alpha * r2)

    if n > 0:
        return dx ** l * dy ** m * n * dz ** (n - 1) * exp_val - 2.0 * alpha * dx ** l * dy ** m * dz ** (n + 1) * exp_val
    else:
        return -2.0 * alpha * dx ** l * dy ** m * dz * exp_val


def _compute_vxc_block(
    shell_a: GTOShell,
    shell_b: GTOShell,
    vxc_r: np.ndarray,
    grid,
    vxc_gamma: np.ndarray = None,
    vxc_tau: np.ndarray = None,
    grad_rho: np.ndarray = None,
) -> np.ndarray:
    n_func_a = shell_a.n_functions
    n_func_b = shell_b.n_functions
    n_points = grid.n_points
    weights = grid.weights

    result = np.zeros((n_func_a, n_func_b))

    cart_a = shell_a.cartesian_components()
    cart_b = shell_b.cartesian_components()
    r_a = shell_a.origin
    r_b = shell_b.origin

    need_grad = (vxc_gamma is not None) or (vxc_tau is not None)

    for pa in range(shell_a.n_primitives):
        alpha_a = shell_a.exponents[pa]
        coef_a = shell_a._contract_coeffs[pa]
        norm_a = shell_a.norm_constants[pa]

        for pb in range(shell_b.n_primitives):
            alpha_b = shell_b.exponents[pb]
            coef_b = shell_b._contract_coeffs[pb]
            norm_b = shell_b.norm_constants[pb]

            prefactor = coef_a * coef_b * norm_a * norm_b

            for i, (la, ma, na) in enumerate(cart_a):
                if need_grad:
                    phi_a, dphi_a = grid._eval_gto_grad(la, ma, na, alpha_a, r_a)
                else:
                    phi_a = grid._eval_gto(la, ma, na, alpha_a, r_a)
                    dphi_a = None
                for j, (lb, mb, nb) in enumerate(cart_b):
                    if need_grad:
                        phi_b, dphi_b = grid._eval_gto_grad(lb, mb, nb, alpha_b, r_b)
                    else:
                        phi_b = grid._eval_gto(lb, mb, nb, alpha_b, r_b)
                        dphi_b = None

                    integrand = phi_a * phi_b * vxc_r * weights
                    result[i, j] += prefactor * np.sum(integrand)

                    if vxc_gamma is not None:
                        grad_phi_pair = phi_a[:, None] * dphi_b + phi_b[:, None] * dphi_a
                        dot = np.sum(grad_rho * grad_phi_pair, axis=1)
                        integrand_g = 2.0 * vxc_gamma * dot * weights
                        result[i, j] += prefactor * np.sum(integrand_g)

                    if vxc_tau is not None:
                        dot_tau = np.sum(dphi_a * dphi_b, axis=1)
                        integrand_t = vxc_tau * dot_tau * weights
                        result[i, j] += prefactor * np.sum(integrand_t)

    return result