"""
高斯型轨道 (Gaussian Type Orbital, GTO) 基础模块。

GTO 原语形式: N * x^l * y^m * z^n * exp(-alpha * r^2)
其中 N 是归一化常数，(l,m,n) 是笛卡尔角量子数，alpha 是指数。

壳层 (Shell) 是一组具有相同角动量的 GTO 原语的收缩组合。
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np


ANGULAR_TO_SHELL = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
SHELL_TO_ANGULAR = {v: k for k, v in ANGULAR_TO_SHELL.items()}


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError(f"factorial: n must be >= 0, got {n}")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    result = 1
    while n > 0:
        result *= n
        n -= 2
    return result


def binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


def cartesian_angular_indices(L: int) -> List[Tuple[int, int, int]]:
    combos = []
    for l in range(L, -1, -1):
        for m in range(L - l, -1, -1):
            n = L - l - m
            combos.append((l, m, n))
    return combos


class GTOPrimitive:
    """
    单个高斯型轨道原语。

    形式: N * x^l * y^m * z^n * exp(-alpha * |r - origin|^2)
    """

    def __init__(
        self,
        alpha: float,
        l: int = 0,
        m: int = 0,
        n: int = 0,
        origin: Optional[np.ndarray] = None,
    ):
        self.alpha = float(alpha)
        self.l = int(l)
        self.m = int(m)
        self.n = int(n)
        if origin is None:
            self.origin = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            self.origin = np.array(origin, dtype=float).copy()
        self.norm = self._compute_norm()

    @property
    def angular_momentum(self) -> int:
        return self.l + self.m + self.n

    def _compute_norm(self) -> float:
        l, m, n = self.l, self.m, self.n
        alpha = self.alpha
        radial = (2.0 * alpha / math.pi) ** 0.75
        num = (4.0 * alpha) ** (l + m + n)
        den = double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)
        if den == 0:
            angular = 1.0
        else:
            angular = math.sqrt(num / den)
        return radial * angular

    def value(self, r: np.ndarray) -> float:
        dr = r - self.origin
        r2 = float(np.dot(dr, dr))
        x, y, z = dr[0], dr[1], dr[2]
        return self.norm * (x ** self.l) * (y ** self.m) * (z ** self.n) * math.exp(-self.alpha * r2)

    def __repr__(self) -> str:
        return (
            f"GTOPrimitive(alpha={self.alpha:.6f}, l=({self.l},{self.m},{self.n}), "
            f"norm={self.norm:.6f})"
        )


class GTOShell:
    """
    GTO 壳层：一组同角动量的 GTO 原语的收缩。

    收缩基函数: phi = sum_i c_i * chi_i
    其中 chi_i 是归一化的 GTO 原语，c_i 是收缩系数。
    """

    def __init__(
        self,
        shell_type: str,
        exponents: List[float],
        coefficients: List[float],
        origin: Optional[np.ndarray] = None,
    ):
        self.shell_type = shell_type.lower()
        self.angular = SHELL_TO_ANGULAR[self.shell_type]
        self.exponents = np.array(exponents, dtype=float).copy()
        self.coefficients = np.array(coefficients, dtype=float).copy()
        if origin is None:
            self.origin = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            self.origin = np.array(origin, dtype=float).copy()

        if len(self.exponents) != len(self.coefficients):
            raise ValueError("exponents and coefficients must have same length")

        self.n_primitives = len(self.exponents)

        self.angular_combinations = cartesian_angular_indices(self.angular)

        self.norm_constants = np.zeros(self.n_primitives)
        self._compute_primitive_norms()

        self._contract_coeffs = self._normalize_contraction()

        self._primitives: List[List[GTOPrimitive]] = []
        for (l, m, n) in self.angular_combinations:
            prims = []
            for i in range(self.n_primitives):
                prim = GTOPrimitive(
                    alpha=self.exponents[i],
                    l=l, m=m, n=n,
                    origin=self.origin,
                )
                prims.append(prim)
            self._primitives.append(prims)

    def _compute_primitive_norms(self):
        l, m, n = self.angular_combinations[0]
        for i in range(self.n_primitives):
            p = GTOPrimitive(self.exponents[i], l, m, n, self.origin)
            self.norm_constants[i] = p.norm

    def _normalize_contraction(self) -> np.ndarray:
        n_prim = self.n_primitives
        l, m, n = self.angular_combinations[0]
        S = np.zeros((n_prim, n_prim))
        for i in range(n_prim):
            for j in range(n_prim):
                alpha_p = self.exponents[i] + self.exponents[j]
                integral = (math.pi / alpha_p) ** 1.5
                ang = 1.0
                if l > 0:
                    ang *= double_factorial(2 * l - 1) / (2.0 * alpha_p) ** l
                if m > 0:
                    ang *= double_factorial(2 * m - 1) / (2.0 * alpha_p) ** m
                if n > 0:
                    ang *= double_factorial(2 * n - 1) / (2.0 * alpha_p) ** n
                S[i, j] = self.norm_constants[i] * self.norm_constants[j] * integral * ang
        c = self.coefficients
        self_overlap = float(c @ S @ c)
        norm_factor = 1.0 / math.sqrt(self_overlap)
        return c * norm_factor

    @property
    def n_functions(self) -> int:
        return len(self.angular_combinations)

    @property
    def n_functions_spherical(self) -> int:
        return 2 * self.angular + 1

    def cartesian_components(self) -> List[Tuple[int, int, int]]:
        return list(self.angular_combinations)

    def translate(self, new_origin: np.ndarray) -> "GTOShell":
        new_shell = GTOShell.__new__(GTOShell)
        new_shell.shell_type = self.shell_type
        new_shell.angular = self.angular
        new_shell.exponents = self.exponents.copy()
        new_shell.coefficients = self.coefficients.copy()
        new_shell.origin = np.array(new_origin, dtype=float).copy()
        new_shell.n_primitives = self.n_primitives
        new_shell.angular_combinations = list(self.angular_combinations)
        new_shell.norm_constants = self.norm_constants.copy()
        new_shell._contract_coeffs = self._contract_coeffs.copy()
        new_shell._primitives = []
        for i, (l, m, n) in enumerate(self.angular_combinations):
            prims = []
            for j in range(self.n_primitives):
                p = GTOPrimitive(self.exponents[j], l, m, n, new_shell.origin)
                prims.append(p)
            new_shell._primitives.append(prims)
        return new_shell

    def value(self, r: np.ndarray) -> np.ndarray:
        values = np.zeros(self.n_functions)
        for i, (l, m, n) in enumerate(self.angular_combinations):
            val = 0.0
            for j in range(self.n_primitives):
                val += self._contract_coeffs[j] * self._primitives[i][j].value(r)
            values[i] = val
        return values

    def __repr__(self) -> str:
        return (
            f"GTOShell(type='{self.shell_type}', "
            f"n_prim={self.n_primitives}, n_func={self.n_functions})"
        )
