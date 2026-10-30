"""
单电子积分引擎。

计算以下单电子积分：
1. 重叠积分:       S_ij = ∫ φ_i(r) φ_j(r) dτ
2. 动能积分:       T_ij = -1/2 ∫ φ_i(r) ∇² φ_j(r) dτ
3. 核吸引积分:     V_ij = -Σ_A Z_A ∫ φ_i(r) 1/|r - R_A| φ_j(r) dτ

使用方法:
- 高斯乘积定理 (Gaussian Product Theorem): 两个高斯函数的乘积仍是高斯函数
- Obara-Saika 递推: 用于高角动量函数的积分递推
- 对于 s 和 p 函数，使用直接解析公式
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ..basis.gto import GTOShell, GTOPrimitive, double_factorial, factorial


# ============================================================================
# 辅助函数：不完全伽马函数 F_0(x) = ∫_0^1 exp(-x t^2) dt
# 这是核吸引积分和双电子积分的核心
# ============================================================================

def boys_function(n: int, x: float) -> float:
    if x < 0.0:
        raise ValueError(f"Boys function does not support negative x = {x}")
    if x < 1e-14:
        return 1.0 / (2.0 * n + 1.0)
    if x > 35.0:
        f = math.sqrt(math.pi / (4.0 * x)) * math.erf(math.sqrt(x))
        exp_x = math.exp(-x)
        for k in range(1, n + 1):
            f = ((2.0 * k - 1.0) * f - exp_x) / (2.0 * x)
        return f
    term = 1.0 / (2.0 * n + 1.0)
    total = term
    k = 1
    while term > 1e-16 * total:
        term *= (2.0 * x) / (2.0 * n + 2.0 * k + 1.0)
        total += term
        k += 1
    return math.exp(-x) * total

def boys_function_vec(max_n: int, x: float) -> np.ndarray:
    f = np.zeros(max_n + 1, dtype=np.float64)
    if x < 1e-14:
        for n in range(max_n + 1):
            f[n] = 1.0 / (2.0 * n + 1.0)
        return f
    if x > 35.0:
        f[0] = math.sqrt(math.pi / (4.0 * x)) * math.erf(math.sqrt(x))
        exp_x = math.exp(-x)
        for n in range(1, max_n + 1):
            f[n] = ((2.0 * n - 1.0) * f[n - 1] - exp_x) / (2.0 * x)
        return f
    f[max_n] = boys_function(max_n, x)
    exp_x = math.exp(-x)
    for n in range(max_n, 0, -1):
        f[n - 1] = (2.0 * x * f[n] + exp_x) / (2.0 * n - 1.0)
    return f

# ============================================================================
# 高斯乘积定理
# ============================================================================

def gaussian_product_center(
    alpha_a: float,
    r_a: np.ndarray,
    alpha_b: float,
    r_b: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    高斯乘积定理：两个高斯函数的乘积是一个新的高斯函数。

    exp(-α_a |r - R_a|^2) * exp(-α_b |r - R_b|^2)
    = exp(-α_a α_b / (α_a + α_b) * |R_a - R_b|^2)
      * exp(-(α_a + α_b) |r - P|^2)

    其中 P = (α_a R_a + α_b R_b) / (α_a + α_b) 是乘积高斯的中心。

    Parameters
    ----------
    alpha_a, alpha_b : float
        两个高斯的指数
    r_a, r_b : np.ndarray
        两个高斯的中心

    Returns
    -------
    prefactor : float
        指数 prefactor: exp(-α_a α_b / γ * |R_a - R_b|^2)
    r_p : np.ndarray
        乘积高斯的中心 P
    """
    gamma = alpha_a + alpha_b
    r_ab = r_a - r_b
    r_ab2 = np.dot(r_ab, r_ab)
    prefactor = math.exp(-alpha_a * alpha_b / gamma * r_ab2)
    r_p = (alpha_a * r_a + alpha_b * r_b) / gamma
    return prefactor, r_p


# ============================================================================
# 原语级别的积分：ss 类型
# ============================================================================

def overlap_ss(alpha_a: float, alpha_b: float, r_ab2: float) -> float:
    """
    两个 s 型 GTO 原语的重叠积分 (未归一化)。

    ∫ exp(-α_a r_a^2) exp(-α_b r_b^2) dτ
    = (π/γ)^(3/2) * exp(-α_a α_b / γ * R_ab^2)

    其中 γ = α_a + α_b
    """
    gamma = alpha_a + alpha_b
    return (math.pi / gamma) ** 1.5 * math.exp(-alpha_a * alpha_b / gamma * r_ab2)


def kinetic_ss(alpha_a: float, alpha_b: float, r_ab2: float) -> float:
    """
    两个 s 型 GTO 原语的动能积分 (未归一化)。

    T = -1/2 ∫ φ_a ∇² φ_b dτ

    对于 s 函数:
    ∇² exp(-α r^2) = (4α² r² - 6α) exp(-α r²)

    结果:
    T = α_a α_b / (α_a + α_b) * (3 - 2 α_a α_b / (α_a + α_b) * R_ab²)
        * (π/(α_a+α_b))^(3/2) * exp(-α_a α_b/(α_a+α_b) R_ab²)
    """
    gamma = alpha_a + alpha_b
    s_integral = (math.pi / gamma) ** 1.5 * math.exp(-alpha_a * alpha_b / gamma * r_ab2)

    # 动能积分 = α_a α_b / γ * (3 - 2 α_a α_b / γ * R_ab²) * S
    t_val = alpha_a * alpha_b / gamma * (3.0 - 2.0 * alpha_a * alpha_b / gamma * r_ab2) * s_integral
    return t_val


def nuclear_ss(
    alpha_a: float,
    alpha_b: float,
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_c: np.ndarray,
    z_c: float,
) -> float:
    """
    两个 s 型 GTO 原语的核吸引积分 (未归一化)。

    V = -Z_C ∫ φ_a(r) 1/|r - R_C| φ_b(r) dτ

    使用 Boys 函数:
    V = -Z_C * 2π / γ * exp(-α_a α_b / γ * R_ab²)
        * F_0(γ * |P - R_C|^2)

    其中 P 是乘积高斯中心，γ = α_a + α_b
    """
    gamma = alpha_a + alpha_b
    r_ab = r_a - r_b
    r_ab2 = np.dot(r_ab, r_ab)

    # 乘积高斯中心
    r_p = (alpha_a * r_a + alpha_b * r_b) / gamma
    r_pc = r_p - r_c
    r_pc2 = np.dot(r_pc, r_pc)

    prefactor = -2.0 * math.pi / gamma * math.exp(-alpha_a * alpha_b / gamma * r_ab2)
    f0 = boys_function(0, gamma * r_pc2)

    return z_c * prefactor * f0


# ============================================================================
# 原语级别的积分：一般角动量 (递推方法)
# ============================================================================

def _os_recurrence_overlap(
    la: int, ma: int, na: int,
    lb: int, mb: int, nb: int,
    alpha_a: float, alpha_b: float,
    r_a: np.ndarray, r_b: np.ndarray,
) -> float:
    """
    Obara-Saika 递推计算两个笛卡尔 GTO 原语的重叠积分。

    递推关系 (以 x 分量为例):
    (a+1|b) = Px - Ax)(a|b) + 1/(2γ) [a_x (a-1|b) + b_x (a|b-1)]

    其中 P 是乘积高斯中心，γ = α_a + α_b

    这是一个递归实现，对于低角动量足够高效。
    """
    gamma = alpha_a + alpha_b
    r_ab = r_a - r_b
    r_ab2 = np.dot(r_ab, r_ab)
    prefactor = math.exp(-alpha_a * alpha_b / gamma * r_ab2)
    r_p = (alpha_a * r_a + alpha_b * r_b) / gamma

    def s_1d(la: int, lb: int, pa: float, pb: float) -> float:
        """1D 重叠积分递推。"""
        if la < 0 or lb < 0:
            return 0.0
        if la == 0 and lb == 0:
            return math.sqrt(math.pi / gamma)

        if la > 0:
            # 递推: (la|lb) = (Px - Ax) * (la-1|lb) + (la-1)/2γ * (la-2|lb) + lb/2γ * (la-1|lb-1)
            # 注意: 这里用 la-1 递推
            return (
                (r_p[0] - r_a[0]) * s_1d(la - 1, lb, pa, pb)
                + (la - 1) / (2.0 * gamma) * s_1d(la - 2, lb, pa, pb)
                + lb / (2.0 * gamma) * s_1d(la - 1, lb - 1, pa, pb)
            )
        else:
            return (
                (r_p[0] - r_b[0]) * s_1d(la, lb - 1, pa, pb)
                + la / (2.0 * gamma) * s_1d(la - 1, lb - 1, pa, pb)
                + (lb - 1) / (2.0 * gamma) * s_1d(la, lb - 2, pa, pb)
            )

    # 用不同的方法 - 使用 Hermite 展开
    # 更简单直接的方法：使用 1D 积分的乘积
    def overlap_1d(l: int, m: int, pa: float, pb: float, axis: int) -> float:
        """
        1D 高斯重叠积分。
        ∫ x^l exp(-α_a (x - A)^2) * x^m exp(-α_b (x - B)^2) dx
        = Σ_i Σ_j C(l,i) C(m,j) A^(l-i) B^(m-j) ∫ x^(i+j) exp(-γ (x-P)^2) dx
        其中 C(n,k) 是二项式系数

        但更高效的是使用递推。
        """
        # 平移坐标到 P 点
        # x - A = (x - P) + (P - A)
        # x^l = Σ_i C(l,i) (x-P)^i (P-A)^(l-i)

        # 直接递推方案
        # 使用 (i|j) 表示 ∫ (x-P)^i (x-P)^j exp(-γ (x-P)^2) dx
        # 即 ∫ (x-P)^(i+j) exp(-γ (x-P)^2) dx
        # 对于 i=j=0: sqrt(π/γ)

        # 我们需要的是 (x-A)^l (x-B)^m exp(-γ(x-P)^2) 的积分
        # = Σ_i Σ_j C(l,i) (P-A)^(l-i) * C(m,j) (P-B)^(m-j) * (i+j|0)

        pa_offset = r_p[axis] - r_a[axis]
        pb_offset = r_p[axis] - r_b[axis]

        result = 0.0
        for i in range(l + 1):
            ci = math.comb(l, i) * pa_offset ** (l - i)
            for j in range(m + 1):
                cj = math.comb(m, j) * pb_offset ** (m - j)
                n = i + j
                # ∫ (x-P)^n exp(-γ (x-P)^2) dx
                if n % 2 == 1:
                    integral = 0.0
                else:
                    integral = math.sqrt(math.pi / gamma) * double_factorial(n - 1) / (2.0 * gamma) ** (n // 2)
                result += ci * cj * integral

        return result

    s_x = overlap_1d(la, lb, r_p[0] - r_a[0], r_p[0] - r_b[0], 0)
    s_y = overlap_1d(ma, mb, r_p[1] - r_a[1], r_p[1] - r_b[1], 1)
    s_z = overlap_1d(na, nb, r_p[2] - r_a[2], r_p[2] - r_b[2], 2)

    return prefactor * s_x * s_y * s_z


def _os_recurrence_kinetic(
    la: int, ma: int, na: int,
    lb: int, mb: int, nb: int,
    alpha_a: float, alpha_b: float,
    r_a: np.ndarray, r_b: np.ndarray,
) -> float:
    """
    动能积分递推。

    动能 T = -1/2 ∫ φ_a ∇² φ_b dτ

    可以用分部积分写成:
    T = 1/2 ∫ ∇φ_a · ∇φ_b dτ

    对于笛卡尔 GTO:
    ∂/∂x [x^l exp(-α x²)] = l x^(l-1) exp(-α x²) - 2α x^(l+1) exp(-α x²)

    所以梯度的每个分量都是两个 GTO 的线性组合。
    """
    def grad_component(l: int, alpha: float) -> List[Tuple[int, float]]:
        """返回 d/dx (x^l exp(-αx²)) 的展开: [(l', coeff)]"""
        terms = []
        if l > 0:
            terms.append((l - 1, float(l)))
        terms.append((l + 1, -2.0 * alpha))
        return terms

    # 计算 ∇φ_a · ∇φ_b 的积分
    t_val = 0.0

    # x 分量
    for (la_p, ca) in grad_component(la, alpha_a):
        for (lb_p, cb) in grad_component(lb, alpha_b):
            s = _os_recurrence_overlap(
                la_p, ma, na, lb_p, mb, nb,
                alpha_a, alpha_b, r_a, r_b
            )
            t_val += ca * cb * s

    # y 分量
    for (ma_p, ca) in grad_component(ma, alpha_a):
        for (mb_p, cb) in grad_component(mb, alpha_b):
            s = _os_recurrence_overlap(
                la, ma_p, na, lb, mb_p, nb,
                alpha_a, alpha_b, r_a, r_b
            )
            t_val += ca * cb * s

    # z 分量
    for (na_p, ca) in grad_component(na, alpha_a):
        for (nb_p, cb) in grad_component(nb, alpha_b):
            s = _os_recurrence_overlap(
                la, ma, na_p, lb, mb, nb_p,
                alpha_a, alpha_b, r_a, r_b
            )
            t_val += ca * cb * s

    return 0.5 * t_val


def _os_recurrence_nuclear(
    la: int, ma: int, na: int,
    lb: int, mb: int, nb: int,
    alpha_a: float, alpha_b: float,
    r_a: np.ndarray, r_b: np.ndarray,
    r_c: np.ndarray, z_c: float,
) -> float:
    """
    核吸引积分的 Obara-Saika 递推。

    V_nuc = -Z_C ∫ φ_a(r) 1/|r - R_C| φ_b(r) dτ

    核吸引积分可以看作双电子积分的特例：ket 侧是位于 R_C 的 s 型电荷分布。
    我们使用完整的 Obara-Saika 递推，通过引入辅助积分 V^{(n)} 来处理。

    实际上，更简单的方法是利用双电子积分的代码，将核视为一个
    指数趋于 0 的 s 轨道（即点电荷）。但这里我们用直接递推。
    """
    gamma = alpha_a + alpha_b
    r_ab = r_a - r_b
    r_ab2 = np.dot(r_ab, r_ab)

    r_p = (alpha_a * r_a + alpha_b * r_b) / gamma
    r_pc = r_p - r_c
    r_pc2 = np.dot(r_pc, r_pc)

    t = gamma * r_pc2
    prefactor = -2.0 * z_c * math.pi / gamma * math.exp(-alpha_a * alpha_b / gamma * r_ab2)

    pa = r_p - r_a
    pb = r_p - r_b
    pc = r_p - r_c

    max_n = la + ma + na + lb + mb + nb
    f = boys_function_vec(max_n, t)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def nuclear_aux(la_, ma_, na_, lb_, mb_, nb_, n):
        if la_ == 0 and ma_ == 0 and na_ == 0 and lb_ == 0 and mb_ == 0 and nb_ == 0:
            return prefactor * f[n]

        if la_ > 0:
            v_ab_n = nuclear_aux(la_-1, ma_, na_, lb_, mb_, nb_, n)
            v_ab_n1 = nuclear_aux(la_-1, ma_, na_, lb_, mb_, nb_, n + 1)
            term1 = pa[0] * v_ab_n
            term2 = -pc[0] * v_ab_n1
            term3 = 0.0
            term4 = 0.0
            if la_ >= 2:
                v_a2b_n = nuclear_aux(la_-2, ma_, na_, lb_, mb_, nb_, n)
                v_a2b_n1 = nuclear_aux(la_-2, ma_, na_, lb_, mb_, nb_, n + 1)
                term3 = (la_ - 1) / (2.0 * gamma) * (v_a2b_n - v_a2b_n1)
            if lb_ > 0:
                v_ab1_n = nuclear_aux(la_-1, ma_, na_, lb_-1, mb_, nb_, n)
                v_ab1_n1 = nuclear_aux(la_-1, ma_, na_, lb_-1, mb_, nb_, n + 1)
                term4 = lb_ / (2.0 * gamma) * (v_ab1_n - v_ab1_n1)
            return term1 + term2 + term3 + term4

        elif ma_ > 0:
            v_ab_n = nuclear_aux(la_, ma_-1, na_, lb_, mb_, nb_, n)
            v_ab_n1 = nuclear_aux(la_, ma_-1, na_, lb_, mb_, nb_, n + 1)
            term1 = pa[1] * v_ab_n
            term2 = -pc[1] * v_ab_n1
            term3 = 0.0
            term4 = 0.0
            if ma_ >= 2:
                v_a2b_n = nuclear_aux(la_, ma_-2, na_, lb_, mb_, nb_, n)
                v_a2b_n1 = nuclear_aux(la_, ma_-2, na_, lb_, mb_, nb_, n + 1)
                term3 = (ma_ - 1) / (2.0 * gamma) * (v_a2b_n - v_a2b_n1)
            if mb_ > 0:
                v_ab1_n = nuclear_aux(la_, ma_-1, na_, lb_, mb_-1, nb_, n)
                v_ab1_n1 = nuclear_aux(la_, ma_-1, na_, lb_, mb_-1, nb_, n + 1)
                term4 = mb_ / (2.0 * gamma) * (v_ab1_n - v_ab1_n1)
            return term1 + term2 + term3 + term4

        elif na_ > 0:
            v_ab_n = nuclear_aux(la_, ma_, na_-1, lb_, mb_, nb_, n)
            v_ab_n1 = nuclear_aux(la_, ma_, na_-1, lb_, mb_, nb_, n + 1)
            term1 = pa[2] * v_ab_n
            term2 = -pc[2] * v_ab_n1
            term3 = 0.0
            term4 = 0.0
            if na_ >= 2:
                v_a2b_n = nuclear_aux(la_, ma_, na_-2, lb_, mb_, nb_, n)
                v_a2b_n1 = nuclear_aux(la_, ma_, na_-2, lb_, mb_, nb_, n + 1)
                term3 = (na_ - 1) / (2.0 * gamma) * (v_a2b_n - v_a2b_n1)
            if nb_ > 0:
                v_ab1_n = nuclear_aux(la_, ma_, na_-1, lb_, mb_, nb_-1, n)
                v_ab1_n1 = nuclear_aux(la_, ma_, na_-1, lb_, mb_, nb_-1, n + 1)
                term4 = nb_ / (2.0 * gamma) * (v_ab1_n - v_ab1_n1)
            return term1 + term2 + term3 + term4

        elif lb_ > 0:
            v_ab_n = nuclear_aux(la_, ma_, na_, lb_-1, mb_, nb_, n)
            v_ab_n1 = nuclear_aux(la_, ma_, na_, lb_-1, mb_, nb_, n + 1)
            term1 = pb[0] * v_ab_n
            term2 = -pc[0] * v_ab_n1
            term3 = 0.0
            term4 = 0.0
            if lb_ >= 2:
                v_ab2_n = nuclear_aux(la_, ma_, na_, lb_-2, mb_, nb_, n)
                v_ab2_n1 = nuclear_aux(la_, ma_, na_, lb_-2, mb_, nb_, n + 1)
                term3 = (lb_ - 1) / (2.0 * gamma) * (v_ab2_n - v_ab2_n1)
            if la_ > 0:
                v_a1b_n = nuclear_aux(la_-1, ma_, na_, lb_-1, mb_, nb_, n)
                v_a1b_n1 = nuclear_aux(la_-1, ma_, na_, lb_-1, mb_, nb_, n + 1)
                term4 = la_ / (2.0 * gamma) * (v_a1b_n - v_a1b_n1)
            return term1 + term2 + term3 + term4

        elif mb_ > 0:
            v_ab_n = nuclear_aux(la_, ma_, na_, lb_, mb_-1, nb_, n)
            v_ab_n1 = nuclear_aux(la_, ma_, na_, lb_, mb_-1, nb_, n + 1)
            term1 = pb[1] * v_ab_n
            term2 = -pc[1] * v_ab_n1
            term3 = 0.0
            term4 = 0.0
            if mb_ >= 2:
                v_ab2_n = nuclear_aux(la_, ma_, na_, lb_, mb_-2, nb_, n)
                v_ab2_n1 = nuclear_aux(la_, ma_, na_, lb_, mb_-2, nb_, n + 1)
                term3 = (mb_ - 1) / (2.0 * gamma) * (v_ab2_n - v_ab2_n1)
            if ma_ > 0:
                v_a1b_n = nuclear_aux(la_, ma_-1, na_, lb_, mb_-1, nb_, n)
                v_a1b_n1 = nuclear_aux(la_, ma_-1, na_, lb_, mb_-1, nb_, n + 1)
                term4 = ma_ / (2.0 * gamma) * (v_a1b_n - v_a1b_n1)
            return term1 + term2 + term3 + term4

        elif nb_ > 0:
            v_ab_n = nuclear_aux(la_, ma_, na_, lb_, mb_, nb_-1, n)
            v_ab_n1 = nuclear_aux(la_, ma_, na_, lb_, mb_, nb_-1, n + 1)
            term1 = pb[2] * v_ab_n
            term2 = -pc[2] * v_ab_n1
            term3 = 0.0
            term4 = 0.0
            if nb_ >= 2:
                v_ab2_n = nuclear_aux(la_, ma_, na_, lb_, mb_, nb_-2, n)
                v_ab2_n1 = nuclear_aux(la_, ma_, na_, lb_, mb_, nb_-2, n + 1)
                term3 = (nb_ - 1) / (2.0 * gamma) * (v_ab2_n - v_ab2_n1)
            if na_ > 0:
                v_a1b_n = nuclear_aux(la_, ma_, na_-1, lb_, mb_, nb_-1, n)
                v_a1b_n1 = nuclear_aux(la_, ma_, na_-1, lb_, mb_, nb_-1, n + 1)
                term4 = na_ / (2.0 * gamma) * (v_a1b_n - v_a1b_n1)
            return term1 + term2 + term3 + term4

        return 0.0

    return nuclear_aux(la, ma, na, lb, mb, nb, 0)


# ============================================================================
# 壳层对的单电子积分
# ============================================================================

def shell_overlap(shell_a: GTOShell, shell_b: GTOShell) -> np.ndarray:
    """
    计算两个壳层之间的重叠积分矩阵块。

    对于收缩基函数 φ = Σ_i c_i χ_i（χ_i 是归一化原语），
    重叠积分: S = Σ_i Σ_j c_a_i c_b_j ∫ χ_a_i χ_b_j dτ

    Parameters
    ----------
    shell_a, shell_b : GTOShell
        两个 GTO 壳层

    Returns
    -------
    np.ndarray
        重叠矩阵块，形状为 (n_func_a, n_func_b)
    """
    na = shell_a.n_functions
    nb = shell_b.n_functions
    S = np.zeros((na, nb))

    for i in range(na):
        la, ma, na_lmn = shell_a.angular_combinations[i]
        for j in range(nb):
            lb, mb, nb_lmn = shell_b.angular_combinations[j]

            val = 0.0
            for p in range(shell_a.n_primitives):
                ca = shell_a._contract_coeffs[p]
                alpha_a = shell_a.exponents[p]
                norm_ap = shell_a._primitives[i][p].norm
                for q in range(shell_b.n_primitives):
                    cb = shell_b._contract_coeffs[q]
                    alpha_b = shell_b.exponents[q]
                    norm_bq = shell_b._primitives[j][q].norm

                    s_prim = _os_recurrence_overlap(
                        la, ma, na_lmn,
                        lb, mb, nb_lmn,
                        alpha_a, alpha_b,
                        shell_a.origin, shell_b.origin,
                    )

                    val += ca * cb * norm_ap * norm_bq * s_prim

            S[i, j] = val

    return S


def shell_kinetic(shell_a: GTOShell, shell_b: GTOShell) -> np.ndarray:
    """
    计算两个壳层之间的动能积分矩阵块。
    """
    na = shell_a.n_functions
    nb = shell_b.n_functions
    T = np.zeros((na, nb))

    for i in range(na):
        la, ma, na_lmn = shell_a.angular_combinations[i]
        for j in range(nb):
            lb, mb, nb_lmn = shell_b.angular_combinations[j]

            val = 0.0
            for p in range(shell_a.n_primitives):
                ca = shell_a._contract_coeffs[p]
                alpha_a = shell_a.exponents[p]
                norm_ap = shell_a._primitives[i][p].norm
                for q in range(shell_b.n_primitives):
                    cb = shell_b._contract_coeffs[q]
                    alpha_b = shell_b.exponents[q]
                    norm_bq = shell_b._primitives[j][q].norm

                    t_prim = _os_recurrence_kinetic(
                        la, ma, na_lmn,
                        lb, mb, nb_lmn,
                        alpha_a, alpha_b,
                        shell_a.origin, shell_b.origin,
                    )

                    val += ca * cb * norm_ap * norm_bq * t_prim

            T[i, j] = val

    return T


def shell_nuclear(
    shell_a: GTOShell,
    shell_b: GTOShell,
    atoms: List[Tuple[str, np.ndarray, float]],
) -> np.ndarray:
    """
    计算两个壳层之间的核吸引积分矩阵块。

    Parameters
    ----------
    shell_a, shell_b : GTOShell
        两个 GTO 壳层
    atoms : List[Tuple[str, np.ndarray, float]]
        原子核列表: [(symbol, position, charge)]

    Returns
    -------
    np.ndarray
        核吸引矩阵块，形状为 (n_func_a, n_func_b)
    """
    na = shell_a.n_functions
    nb = shell_b.n_functions
    V = np.zeros((na, nb))

    for i in range(na):
        la, ma, na_lmn = shell_a.angular_combinations[i]
        for j in range(nb):
            lb, mb, nb_lmn = shell_b.angular_combinations[j]

            val = 0.0
            for p in range(shell_a.n_primitives):
                ca = shell_a._contract_coeffs[p]
                alpha_a = shell_a.exponents[p]
                norm_ap = shell_a._primitives[i][p].norm
                for q in range(shell_b.n_primitives):
                    cb = shell_b._contract_coeffs[q]
                    alpha_b = shell_b.exponents[q]
                    norm_bq = shell_b._primitives[j][q].norm

                    # 对每个原子核求和
                    for (_, r_c, z_c) in atoms:
                        v_prim = _os_recurrence_nuclear(
                            la, ma, na_lmn,
                            lb, mb, nb_lmn,
                            alpha_a, alpha_b,
                            shell_a.origin, shell_b.origin,
                            r_c, z_c,
                        )
                        val += ca * cb * norm_ap * norm_bq * v_prim

            V[i, j] = val

    return V


# ============================================================================
# 分子级别的单电子积分
# ============================================================================

def compute_overlap(basis_set, mol) -> np.ndarray:
    """
    计算整个分子的重叠矩阵 S。

    Parameters
    ----------
    basis_set : BasisSet
        基组对象
    mol : Molecule
        分子对象

    Returns
    -------
    np.ndarray
        重叠矩阵，形状为 (nbf, nbf)
    """
    all_shells = basis_set.get_all_shells(mol)
    nshell = len(all_shells)
    shells_by_atom, nbf = basis_set.build_molecule_basis(mol)

    S = np.zeros((nbf, nbf))

    # 计算每个壳层对的积分块
    for i in range(nshell):
        start_i, end_i = basis_set.shell_index_to_bf_range(all_shells, i)
        for j in range(nshell):
            start_j, end_j = basis_set.shell_index_to_bf_range(all_shells, j)
            block = shell_overlap(all_shells[i], all_shells[j])
            S[start_i:end_i, start_j:end_j] = block

    return S


def compute_kinetic(basis_set, mol) -> np.ndarray:
    """
    计算整个分子的动能矩阵 T。
    """
    all_shells = basis_set.get_all_shells(mol)
    nshell = len(all_shells)
    shells_by_atom, nbf = basis_set.build_molecule_basis(mol)

    T = np.zeros((nbf, nbf))

    for i in range(nshell):
        start_i, end_i = basis_set.shell_index_to_bf_range(all_shells, i)
        for j in range(nshell):
            start_j, end_j = basis_set.shell_index_to_bf_range(all_shells, j)
            block = shell_kinetic(all_shells[i], all_shells[j])
            T[start_i:end_i, start_j:end_j] = block

    return T


def compute_nuclear(basis_set, mol) -> np.ndarray:
    """
    计算整个分子的核吸引矩阵 V。
    """
    all_shells = basis_set.get_all_shells(mol)
    nshell = len(all_shells)
    shells_by_atom, nbf = basis_set.build_molecule_basis(mol)

    # 准备原子核信息
    from ..utils.constants import ATOMIC_NUMBERS
    atoms = []
    for i in range(mol.natom):
        z = float(ATOMIC_NUMBERS[mol.symbols[i]])
        atoms.append((mol.symbols[i], mol.coordinates[i], z))

    V = np.zeros((nbf, nbf))

    for i in range(nshell):
        start_i, end_i = basis_set.shell_index_to_bf_range(all_shells, i)
        for j in range(nshell):
            start_j, end_j = basis_set.shell_index_to_bf_range(all_shells, j)
            block = shell_nuclear(all_shells[i], all_shells[j], atoms)
            V[start_i:end_i, start_j:end_j] = block

    return V


def shell_dipole(shell_a: GTOShell, shell_b: GTOShell) -> np.ndarray:
    """
    计算两个壳层之间的偶极积分矩阵块。
    
    偶极积分: μ_d = ∫ φ_a r_d φ_b dτ, d = x, y, z
    
    Parameters
    ----------
    shell_a, shell_b : GTOShell
        两个 GTO 壳层
    
    Returns
    -------
    np.ndarray
        偶极积分张量，形状为 (3, n_func_a, n_func_b)
        [0] = x分量, [1] = y分量, [2] = z分量
    """
    na = shell_a.n_functions
    nb = shell_b.n_functions
    dipole = np.zeros((3, na, nb))
    
    for i in range(na):
        la, ma, na_lmn = shell_a.angular_combinations[i]
        for j in range(nb):
            lb, mb, nb_lmn = shell_b.angular_combinations[j]
                
            for p in range(shell_a.n_primitives):
                ca = shell_a._contract_coeffs[p]
                alpha_a = shell_a.exponents[p]
                norm_ap = shell_a._primitives[i][p].norm
                for q in range(shell_b.n_primitives):
                    cb = shell_b._contract_coeffs[q]
                    alpha_b = shell_b.exponents[q]
                    norm_bq = shell_b._primitives[j][q].norm
                    
                    for d in range(3):
                        d_prim = _dipole_primitive(
                            la, ma, na_lmn,
                            lb, mb, nb_lmn,
                            alpha_a, alpha_b,
                            shell_a.origin, shell_b.origin,
                            d
                        )
                        dipole[d, i, j] += ca * cb * norm_ap * norm_bq * d_prim
    
    return dipole


def _dipole_primitive(la, ma, na, lb, mb, nb, alpha_a, alpha_b, r_a, r_b, direction):
    """
    计算单个原语的偶极积分 <a|r_d|b>。
    
    使用递推方法：r_d = R_d + (r - R)_d
    其中 R 是高斯乘积中心。
    """
    zeta = alpha_a + alpha_b
    r_p = (alpha_a * r_a + alpha_b * r_b) / zeta
    
    r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
    K = math.exp(-alpha_a * alpha_b / zeta * r_ab2)
    
    def get_overlap(l, m, n, l2, m2, n2):
        if l < 0 or m < 0 or n < 0 or l2 < 0 or m2 < 0 or n2 < 0:
            return 0.0
        return _os_recurrence_overlap(
            l, m, n, l2, m2, n2,
            alpha_a, alpha_b, r_a, r_b
        )
    
    s00 = get_overlap(la, ma, na, lb, mb, nb)
    result = r_p[direction] * s00
    
    if direction == 0:
        if la > 0:
            result += la / (2.0 * zeta) * get_overlap(la - 1, ma, na, lb, mb, nb)
        if lb > 0:
            result += lb / (2.0 * zeta) * get_overlap(la, ma, na, lb - 1, mb, nb)
    elif direction == 1:
        if ma > 0:
            result += ma / (2.0 * zeta) * get_overlap(la, ma - 1, na, lb, mb, nb)
        if mb > 0:
            result += mb / (2.0 * zeta) * get_overlap(la, ma, na, lb, mb - 1, nb)
    else:
        if na > 0:
            result += na / (2.0 * zeta) * get_overlap(la, ma, na - 1, lb, mb, nb)
        if nb > 0:
            result += nb / (2.0 * zeta) * get_overlap(la, ma, na, lb, mb, nb - 1)
    
    return result


def compute_dipole(basis_set, mol) -> np.ndarray:
    """
    计算整个分子的偶极积分矩阵。
    
    Parameters
    ----------
    basis_set : BasisSet
        基组
    mol : Molecule
        分子
    
    Returns
    -------
    np.ndarray
        偶极积分张量，形状为 (3, nbf, nbf)
        [0] = x分量, [1] = y分量, [2] = z分量
    """
    shells_by_atom, nbf = basis_set.build_molecule_basis(mol)
    
    all_shells = []
    for atom_shells in shells_by_atom:
        all_shells.extend(atom_shells)
    
    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf
    
    nshell = len(all_shells)
    
    mu = np.zeros((3, nbf, nbf))
    
    for i in range(nshell):
        si, ei = shell_range[i]
        for j in range(nshell):
            sj, ej = shell_range[j]
            
            block = shell_dipole(all_shells[i], all_shells[j])
            mu[:, si:ei, sj:ej] = block
    
    return mu


compute_dipole_integrals = compute_dipole
