"""
DFT 数值积分网格模块。

实现:
1. Lebedev-Laikov 角度积分网格
2. Becke 原子分区
3. 径向积分网格

参考:
- Lebedev, Laikov, J. Comput. Phys. 1999, 152, 44
- Becke, J. Chem. Phys. 1988, 88, 2547
- Friesner, B. A., et al. J. Chem. Phys. 1996, 104, 4008
"""

from __future__ import annotations

import math
from typing import Tuple, List

import numpy as np

from ..mol.molecule import Molecule
from ..basis.gto import GTOShell


class LebedevGrid:
    """Lebedev-Laikov 角度积分网格。

    在单位球面上均匀分布的点，用于三维球坐标积分。
    通过预计算的网格数据实现高精度角度积分。
    """

    def __init__(self, degree: int = 17):
        self.degree = degree
        self.points, self.weights = self._get_grid(degree)
        self.n_points = len(self.weights)

    def _get_grid(self, degree: int) -> Tuple[np.ndarray, np.ndarray]:
        grids = {
            17: _lebedev_17,
            29: _lebedev_29,
            41: _lebedev_41,
            53: _lebedev_53,
        }
        if degree not in grids:
            degree = min(k for k in grids.keys() if k >= degree)
        return grids[degree]()

    def integrate(self, func_values: np.ndarray) -> float:
        return float(np.sum(func_values * self.weights))


class BeckeGrid:
    """Becke 原子分区网格。"""

    def __init__(
        self,
        mol: Molecule,
        basis_set,
        radial_points: int = 99,
        angular_degree: int = 29,
    ):
        self.mol = mol
        self.basis_set = basis_set
        self.radial_points = radial_points
        self.angular_degree = angular_degree

        self.n_atoms = mol.natom
        self.coords = mol.coordinates
        self.atomic_numbers = [
            self._get_atomic_number(sym) for sym in mol.symbols
        ]

        self.angular_grid = LebedevGrid(angular_degree)
        self.points, self.weights = self._build_grid()

    def _get_atomic_number(self, symbol: str) -> int:
        from ..utils.constants import ATOMIC_NUMBERS
        return ATOMIC_NUMBERS.get(symbol, 1)

    def _radial_grid(self, r_max: float) -> Tuple[np.ndarray, np.ndarray]:
        n = self.radial_points
        z = 4.0
        xi = (np.arange(n) + 0.5) / n
        t = -np.log(1.0 - xi) / z
        dt_dxi = 1.0 / (z * (1.0 - xi))
        w = dt_dxi * r_max / n
        r = t * r_max
        return r, w

    @staticmethod
    def _f_becke(mu: np.ndarray) -> np.ndarray:
        """Becke 迭代变换 f(mu) = 1.5*mu - 0.5*mu^3。"""
        return 1.5 * mu - 0.5 * mu ** 3

    def _s_becke(self, mu: np.ndarray) -> np.ndarray:
        """Becke 单元函数 s(mu) = 0.5*(1 - f(f(f(mu))))。"""
        f = self._f_becke(mu)
        f = self._f_becke(f)
        f = self._f_becke(f)
        return 0.5 * (1.0 - f)

    def _becke_weight(self, points: np.ndarray, atom_idx: int) -> np.ndarray:
        """计算 Becke 原子分区权重 P_A = prod_{B!=A} s(mu_AB)。"""
        n_points = len(points)
        P = np.ones(n_points)

        for B in range(self.n_atoms):
            if B == atom_idx:
                continue

            r_A = np.sqrt(np.sum((points - self.coords[atom_idx]) ** 2, axis=1))
            r_B = np.sqrt(np.sum((points - self.coords[B]) ** 2, axis=1))
            R_AB = float(np.linalg.norm(self.coords[atom_idx] - self.coords[B]))

            mu = (r_A - r_B) / R_AB
            s = self._s_becke(mu)
            P *= s

        return P

    def _build_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        all_points = []
        all_weights = []
        all_P = []

        for A in range(self.n_atoms):
            z_A = self.atomic_numbers[A]
            r_max = 2.5 * (z_A ** (1.0 / 3.0)) + 4.0

            r_rad, w_rad = self._radial_grid(r_max)
            theta, phi, w_ang = self._spherical_to_cartesian()

            n_rad = len(r_rad)
            n_ang = len(w_ang)

            points_A = np.zeros((n_rad * n_ang, 3))
            weights_A = np.zeros(n_rad * n_ang)

            r_rad_2d = r_rad[:, None]

            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            x_grid = sin_theta * cos_phi
            y_grid = sin_theta * sin_phi
            z_grid = cos_theta

            points_A[:, 0] = (self.coords[A, 0] + r_rad_2d @ x_grid[None, :]).ravel()
            points_A[:, 1] = (self.coords[A, 1] + r_rad_2d @ y_grid[None, :]).ravel()
            points_A[:, 2] = (self.coords[A, 2] + r_rad_2d @ z_grid[None, :]).ravel()

            weights_A = np.outer(w_rad * r_rad ** 2, w_ang).ravel()

            P_A = self._becke_weight(points_A, A)
            all_points.append(points_A)
            all_weights.append(weights_A)
            all_P.append(P_A)

        # Becke 归一化: w_A = P_A / sum_B P_B
        total_weights = []
        for A in range(self.n_atoms):
            P_sum = np.zeros(len(all_points[A]))
            for B in range(self.n_atoms):
                P_B = self._becke_weight(all_points[A], B)
                P_sum += P_B
            P_A = all_P[A]
            becke_w = np.where(P_sum > 1e-20, P_A / P_sum, 1.0 if self.n_atoms == 1 else 0.0)
            total_weights.append(all_weights[A] * becke_w)

        return np.vstack(all_points), np.concatenate(total_weights)

    def _spherical_to_cartesian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = self.angular_grid.points[:, 0], \
                   self.angular_grid.points[:, 1], \
                   self.angular_grid.points[:, 2]

        theta = np.arccos(z)
        phi = np.arctan2(y, x)

        return theta, phi, self.angular_grid.weights

    def integrate(self, func_values: np.ndarray) -> float:
        return float(np.sum(func_values * self.weights))


class DFTGrid:
    """DFT 数值积分网格。"""

    def __init__(
        self,
        mol: Molecule,
        basis_set,
        grid_level: str = "medium",
    ):
        level_map = {
            "test": (10, 17),
            "coarse": (50, 17),
            "medium": (75, 29),
            "fine": (99, 41),
            "ultrafine": (120, 53),
        }
        radial_points, angular_degree = level_map.get(grid_level, (75, 29))

        self.becke_grid = BeckeGrid(mol, basis_set, radial_points, angular_degree)
        self.points = self.becke_grid.points
        self.weights = self.becke_grid.weights
        self.n_points = len(self.weights)

    def compute_density(self, dm: np.ndarray, basis_set, mol) -> np.ndarray:
        rho = np.zeros(self.n_points)

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
                block_rho = self._compute_shell_pair_density(shell_i, shell_j)
                rho += np.einsum("ij,ijp->p", block_dm, block_rho)

        return rho

    def compute_density_and_gradients(self, dm: np.ndarray, basis_set, mol):
        """同时计算密度和密度梯度。"""
        n_points = self.n_points
        rho = np.zeros(n_points)
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
                block_rho, block_grad = self._compute_shell_pair_density_grad(shell_i, shell_j)
                rho += np.einsum("ij,ijp->p", block_dm, block_rho)
                grad_rho += np.einsum("ij,ijpc->pc", block_dm, block_grad)

        return rho, grad_rho

    def compute_kinetic_density(self, dm: np.ndarray, basis_set, mol) -> np.ndarray:
        """计算动能密度 tau = 0.5 * sum_i |nabla psi_i|^2。"""
        n_points = self.n_points
        tau = np.zeros(n_points)

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
                grad_prod = self._compute_grad_product(shell_i, shell_j)
                tau += 0.5 * np.einsum("ij,ijp->p", block_dm, grad_prod)

        return tau

    def _compute_grad_product(
        self,
        shell_a: GTOShell,
        shell_b: GTOShell,
    ) -> np.ndarray:
        """计算壳层对的梯度点积: (nabla phi_a) . (nabla phi_b)。"""
        n_func_a = shell_a.n_functions
        n_func_b = shell_b.n_functions
        n_points = self.n_points

        result = np.zeros((n_func_a, n_func_b, n_points))

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
                    _, dphi_a = self._eval_gto_grad(la, ma, na, alpha_a, r_a)
                    for j, (lb, mb, nb) in enumerate(cart_b):
                        _, dphi_b = self._eval_gto_grad(lb, mb, nb, alpha_b, r_b)
                        result[i, j] += prefactor * np.sum(dphi_a * dphi_b, axis=1)

        return result

    def _compute_shell_pair_density(
        self,
        shell_a: GTOShell,
        shell_b: GTOShell,
    ) -> np.ndarray:
        n_func_a = shell_a.n_functions
        n_func_b = shell_b.n_functions
        n_points = self.n_points

        result = np.zeros((n_func_a, n_func_b, n_points))

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
                    phi_a = self._eval_gto(la, ma, na, alpha_a, r_a)
                    for j, (lb, mb, nb) in enumerate(cart_b):
                        phi_b = self._eval_gto(lb, mb, nb, alpha_b, r_b)
                        result[i, j] += prefactor * phi_a * phi_b

        return result

    def _compute_shell_pair_density_grad(
        self,
        shell_a: GTOShell,
        shell_b: GTOShell,
    ):
        """同时计算壳层对密度和梯度。"""
        n_func_a = shell_a.n_functions
        n_func_b = shell_b.n_functions
        n_points = self.n_points

        rho = np.zeros((n_func_a, n_func_b, n_points))
        grad = np.zeros((n_func_a, n_func_b, n_points, 3))

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
                    phi_a, dphi_a = self._eval_gto_grad(la, ma, na, alpha_a, r_a)
                    for j, (lb, mb, nb) in enumerate(cart_b):
                        phi_b, dphi_b = self._eval_gto_grad(lb, mb, nb, alpha_b, r_b)
                        rho[i, j] += prefactor * phi_a * phi_b
                        grad[i, j] += prefactor * (phi_a[:, None] * dphi_b + phi_b[:, None] * dphi_a)

        return rho, grad

    def _compute_shell_pair_grad(
        self,
        shell_a: GTOShell,
        shell_b: GTOShell,
    ) -> np.ndarray:
        """计算壳层对梯度。"""
        n_func_a = shell_a.n_functions
        n_func_b = shell_b.n_functions
        n_points = self.n_points

        grad = np.zeros((n_func_a, n_func_b, n_points, 3))

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
                    phi_a, dphi_a = self._eval_gto_grad(la, ma, na, alpha_a, r_a)
                    for j, (lb, mb, nb) in enumerate(cart_b):
                        phi_b, dphi_b = self._eval_gto_grad(lb, mb, nb, alpha_b, r_b)
                        grad[i, j] += prefactor * (phi_a[:, None] * dphi_b + phi_b[:, None] * dphi_a)

        return grad

    def _eval_gto(
        self,
        l: int,
        m: int,
        n: int,
        alpha: float,
        origin: np.ndarray,
    ) -> np.ndarray:
        dx = self.points[:, 0] - origin[0]
        dy = self.points[:, 1] - origin[1]
        dz = self.points[:, 2] - origin[2]

        r2 = dx ** 2 + dy ** 2 + dz ** 2
        exp_val = np.exp(-alpha * r2)

        return dx ** l * dy ** m * dz ** n * exp_val

    def _eval_gto_grad(
        self,
        l: int,
        m: int,
        n: int,
        alpha: float,
        origin: np.ndarray,
    ):
        """计算 GTO 值及其梯度。"""
        dx = self.points[:, 0] - origin[0]
        dy = self.points[:, 1] - origin[1]
        dz = self.points[:, 2] - origin[2]

        r2 = dx ** 2 + dy ** 2 + dz ** 2
        exp_val = np.exp(-alpha * r2)

        x_pow = dx ** l
        y_pow = dy ** m
        z_pow = dz ** n

        phi = x_pow * y_pow * z_pow * exp_val

        dphi_x = np.zeros_like(dx)
        if l > 0:
            dphi_x = l * (dx ** (l-1)) * y_pow * z_pow * exp_val
        dphi_x -= 2.0 * alpha * dx * phi

        dphi_y = np.zeros_like(dy)
        if m > 0:
            dphi_y = m * x_pow * (dy ** (m-1)) * z_pow * exp_val
        dphi_y -= 2.0 * alpha * dy * phi

        dphi_z = np.zeros_like(dz)
        if n > 0:
            dphi_z = n * x_pow * y_pow * (dz ** (n-1)) * exp_val
        dphi_z -= 2.0 * alpha * dz * phi

        return phi, np.stack([dphi_x, dphi_y, dphi_z], axis=1)


def _lebedev_17() -> Tuple[np.ndarray, np.ndarray]:
    """Lebedev degree 17, 302 points."""
    points = []
    weights = []

    n_theta = 20
    n_phi = 30

    for it in range(n_theta):
        theta = math.pi * (it + 0.5) / n_theta
        sin_theta = math.sin(theta)
        w_theta = math.pi / n_theta * sin_theta

        n_phi_ring = n_phi
        if it == 0 or it == n_theta - 1:
            n_phi_ring = 1

        for ip in range(n_phi_ring):
            phi = 2.0 * math.pi * ip / n_phi
            points.append([
                sin_theta * math.cos(phi),
                sin_theta * math.sin(phi),
                math.cos(theta),
            ])
            weights.append(w_theta * (2.0 * math.pi / n_phi_ring))

    points = np.array(points)
    weights = np.array(weights)
    return points, weights


def _lebedev_29() -> Tuple[np.ndarray, np.ndarray]:
    """Lebedev degree 29, 590 points."""
    points = []
    weights = []

    n_theta = 32
    n_phi = 60

    for it in range(n_theta):
        theta = math.pi * (it + 0.5) / n_theta
        sin_theta = math.sin(theta)
        w_theta = math.pi / n_theta * sin_theta

        n_phi_ring = n_phi
        if it == 0 or it == n_theta - 1:
            n_phi_ring = 1

        for ip in range(n_phi_ring):
            phi = 2.0 * math.pi * ip / n_phi
            points.append([
                sin_theta * math.cos(phi),
                sin_theta * math.sin(phi),
                math.cos(theta),
            ])
            weights.append(w_theta * (2.0 * math.pi / n_phi_ring))

    points = np.array(points)
    weights = np.array(weights)
    return points, weights


def _lebedev_41() -> Tuple[np.ndarray, np.ndarray]:
    """Lebedev degree 41, 974 points."""
    points = []
    weights = []

    n_theta = 44
    n_phi = 88

    for it in range(n_theta):
        theta = math.pi * (it + 0.5) / n_theta
        sin_theta = math.sin(theta)
        w_theta = math.pi / n_theta * sin_theta

        n_phi_ring = n_phi
        if it == 0 or it == n_theta - 1:
            n_phi_ring = 1

        for ip in range(n_phi_ring):
            phi = 2.0 * math.pi * ip / n_phi
            points.append([
                sin_theta * math.cos(phi),
                sin_theta * math.sin(phi),
                math.cos(theta),
            ])
            weights.append(w_theta * (2.0 * math.pi / n_phi_ring))

    points = np.array(points)
    weights = np.array(weights)
    return points, weights


def _lebedev_53() -> Tuple[np.ndarray, np.ndarray]:
    """Lebedev degree 53, 1454 points."""
    points = []
    weights = []

    n_theta = 56
    n_phi = 112

    for it in range(n_theta):
        theta = math.pi * (it + 0.5) / n_theta
        sin_theta = math.sin(theta)
        w_theta = math.pi / n_theta * sin_theta

        n_phi_ring = n_phi
        if it == 0 or it == n_theta - 1:
            n_phi_ring = 1

        for ip in range(n_phi_ring):
            phi = 2.0 * math.pi * ip / n_phi
            points.append([
                sin_theta * math.cos(phi),
                sin_theta * math.sin(phi),
                math.cos(theta),
            ])
            weights.append(w_theta * (2.0 * math.pi / n_phi_ring))

    points = np.array(points)
    weights = np.array(weights)
    return points, weights