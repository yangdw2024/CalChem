"""
DFT 交换关联泛函模块。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class XCFunctional:
    """交换关联泛函基类。"""

    def __init__(self):
        self.name = "base"
        self.type = "lda"
        self.exx_fraction = 0.0

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def compute_full(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (exc, vxc_rho, vxc_gamma, vxc_tau)。
        vxc_gamma = None 表示 LDA；vxc_tau = None 表示 GGA。"""
        exc, vxc = self.compute(rho, grad_rho, lap_rho, tau)
        return exc, vxc, None, None


class LDA(XCFunctional):
    """LDA 泛函 (Slater + VWN5)。"""

    def __init__(self):
        super().__init__()
        self.name = "lda"
        self.type = "lda"

        self.c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)

        self.a = 0.031091
        self.b = 0.21370
        self.c = 7.5957
        self.d = 3.5876
        self.e = 1.6382
        self.f = 0.49294

    def _slater_exchange(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rho_13 = rho ** (1.0 / 3.0)
        exc = self.c_x * rho_13
        vxc = (4.0 / 3.0) * exc
        return exc, vxc

    def _vwn5_correlation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)

        A = 0.0310907
        b = 0.21370
        c = 7.5957
        d = 3.5876
        e = 1.6382
        f = 0.49294

        rs2 = rs ** 2
        rs3 = rs ** 3

        x = rs ** 0.5
        denom1 = x ** 2 + b * x + c
        denom2 = x ** 2 + d

        term1 = np.log(x ** 2 / denom1)
        term2 = 2.0 * b / e * np.arctan(e / x)
        term3 = -(c / e) * np.log(denom1 / denom2)

        ec = A * (term1 + term2 + term3)

        dec_dx = A * (
            2.0 / x - (2.0 * x + b) / denom1
            - 2.0 * b / (x ** 2 + e ** 2)
            - (c / e) * ((2.0 * x + b) / denom1 - (2.0 * x) / denom2)
        )
        dec_drs = dec_dx * (0.5 / x)

        vxc = ec - rs / 3.0 * dec_drs
        return ec, vxc

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        exc_x, vxc_x = self._slater_exchange(rho)
        exc_c, vxc_c = self._vwn5_correlation(rho)
        return exc_x + exc_c, vxc_x + vxc_c


class GGA(XCFunctional):
    """GGA 泛函基类。"""

    def __init__(self):
        super().__init__()
        self.type = "gga"


class PBE(GGA):
    """PBE 泛函 (Perdew-Burke-Ernzerhof)。"""

    def __init__(self):
        super().__init__()
        self.name = "pbe"

        self.k = 0.804
        self.mu = 0.2195149727645171
        self.beta = 0.06672455060314922

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("PBE 需要密度梯度")

        rho_safe = np.maximum(rho, 1e-20)
        s = self._compute_s(rho_safe, grad_rho)

        exc_x, vxc_x = self._pbe_exchange(rho_safe, s)
        exc_c, vxc_c = self._pbe_correlation(rho_safe, s)

        return exc_x + exc_c, vxc_x + vxc_c

    def compute_full(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("PBE 需要密度梯度")

        rho_safe = np.maximum(rho, 1e-20)
        s = self._compute_s(rho_safe, grad_rho)
        kf = (3.0 * math.pi ** 2 * rho_safe) ** (1.0 / 3.0)

        exc_x, vxc_x = self._pbe_exchange(rho_safe, s)
        exc_c, vxc_c = self._pbe_correlation(rho_safe, s)

        exc = exc_x + exc_c
        vxc_rho = vxc_x + vxc_c

        # vxc_gamma = d(rho * exc) / d(gamma)
        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        s2 = s ** 2
        df_ds2 = self.mu / self.k / (1.0 + self.mu * s2 / self.k) ** 2
        vxc_gamma_x = c_x * df_ds2 / (4.0 * kf ** 2 * rho_safe ** (2.0 / 3.0))

        f_c = 1.0 / (1.0 + self.beta * s2)
        ec_pw = exc_c / f_c
        df_c_ds2 = -self.beta * f_c ** 2
        vxc_gamma_c = ec_pw * df_c_ds2 / (4.0 * kf ** 2 * rho_safe)

        vxc_gamma = vxc_gamma_x + vxc_gamma_c
        # 数值保护: 在极低密度区域屏蔽发散的 vxc_gamma
        gamma_mask = rho_safe < 1e-10
        vxc_gamma = np.where(gamma_mask, 0.0, vxc_gamma)
        return exc, vxc_rho, vxc_gamma, None

    def _compute_s(self, rho: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
        kf = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)
        denom = 2.0 * kf * rho
        mask = denom < 1e-20
        s = np.where(mask, 0.0, grad_norm / denom)
        s = np.clip(s, 0.0, 100.0)
        return s

    def _pbe_exchange(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        f_x = 1.0 + self.k - self.k / (1.0 + self.mu * s ** 2 / self.k)
        exc_x = c_x * rho_13 * f_x

        ds2_drho = -8.0 / 9.0 * s ** 2 / rho
        df_ds2 = self.mu / self.k / (1.0 + self.mu * s ** 2 / self.k) ** 2
        df_drho = df_ds2 * ds2_drho

        vxc_x = (4.0 / 3.0) * exc_x + c_x * rho_13 * rho * df_drho
        return exc_x, vxc_x

    def _pbe_correlation(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        ec_pw, dec_pw_drs = self._pw92_correlation(rho)

        s2 = s ** 2
        f_c = 1.0 / (1.0 + self.beta * s2)
        exc_c = ec_pw * f_c

        dec_drho = -rs / (3.0 * rho) * dec_pw_drs
        ds_drho = -4.0 / 9.0 * s / rho
        df_c_ds2 = -self.beta / (1.0 + self.beta * s2) ** 2
        df_c_drho = df_c_ds2 * 2.0 * s * ds_drho

        vxc_c = exc_c + rho * (dec_drho * f_c + ec_pw * df_c_drho)
        return exc_c, vxc_c

    def _pw92_correlation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perdew-Wang 1992 非极化相关能参数化。"""
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)

        a = 0.0310907
        a1 = 0.21370
        b1 = 7.5957
        b2 = 3.5876
        b3 = 1.6382
        b4 = 0.49294

        rs12 = np.sqrt(rs)
        rs32 = rs * rs12
        rs2 = rs ** 2

        q0 = -2.0 * a * (1.0 + a1 * rs)
        q1 = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
        q1p = a * (b1 / rs12 + 2.0 * b2 + 3.0 * b3 * rs12 + 4.0 * b4 * rs)

        ec = q0 * np.log(1.0 + 1.0 / q1)
        dec_drs = -2.0 * a * a1 * np.log(1.0 + 1.0 / q1) - q0 * q1p / (q1 ** 2 + q1)

        return ec, dec_drs


class B88(GGA):
    """Becke 88 交换泛函。"""

    def __init__(self):
        super().__init__()
        self.name = "b88"
        self.c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        self.b = 0.0042

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("B88 需要密度梯度")

        s = self._compute_s(rho, grad_rho)
        exc_x, vxc_x = self._b88_exchange(rho, s)
        return exc_x, vxc_x

    def _compute_s(self, rho: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
        kf = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)
        denom = 2.0 * kf * rho
        mask = denom < 1e-20
        s = np.where(mask, 0.0, grad_norm / denom)
        s = np.clip(s, 0.0, 100.0)
        return s

    def _b88_exchange(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rho_13 = rho ** (1.0 / 3.0)

        s2 = s ** 2
        mask = s2 < 1e-20
        
        s2_safe = np.where(mask, 1.0, s2)
        arg = s2_safe / self.b
        arg = np.where(mask, 0.0, arg)
        exp_val = np.exp(-arg)
        
        f_x = np.empty_like(s2)
        f_x[mask] = 1.0
        f_x[~mask] = 1.0 + self.b / s2_safe[~mask] * (1.0 - exp_val[~mask])
        exc_x = self.c_x * rho_13 * f_x

        ds_drho = -4.0 / 9.0 * s / rho
        
        df_ds2 = np.empty_like(s2)
        df_ds2[mask] = 0.0
        df_ds2[~mask] = self.b / (s2_safe[~mask] ** 2) * (exp_val[~mask] * arg[~mask] + exp_val[~mask] - 1.0)
        df_drho = df_ds2 * 2.0 * s * ds_drho

        vxc_x = (4.0 / 3.0) * exc_x + self.c_x * rho_13 * rho * df_drho
        return exc_x, vxc_x


class LYP(GGA):
    """Lee-Yang-Parr 关联泛函。"""

    def __init__(self):
        super().__init__()
        self.name = "lyp"

        self.a0 = 0.04918
        self.a1 = 0.132
        self.a2 = 0.234
        self.a3 = 0.349
        self.b0 = 0.218
        self.b1 = 0.298
        self.b2 = 0.408
        self.b3 = 0.507

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("LYP 需要密度梯度")

        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)
        t = grad_norm ** 2 / rho ** (8.0 / 3.0)

        ec, vxc = self._lyp_correlation(rho, rs, t)
        return ec, vxc

    def _lyp_correlation(self, rho, rs, t) -> Tuple[np.ndarray, np.ndarray]:
        rs2 = rs ** 2
        rs4 = rs ** 4

        d = self.a0 + self.a1 * rs + self.a2 * rs2 + self.a3 * rs4
        e = self.b0 + self.b1 * rs + self.b2 * rs2 + self.b3 * rs4

        ec = -d / e * rho ** (1.0 / 3.0)

        dec_drho = (1.0 / 3.0) * ec / rho

        vxc = ec + rho * dec_drho
        return ec, vxc


class B3LYP(GGA):
    """B3LYP 混合泛函。

    B3LYP = (1-a0-ax)*LDA_x + ax*B88_x + a0*HF_x + (1-ac)*VWN5_c + ac*LYP_c

    系数:
    - a0 = 0.20 (HF交换)
    - ax = 0.72 (B88交换)
    - ac = 0.81 (LYP关联)
    """

    def __init__(self):
        super().__init__()
        self.name = "b3lyp"

        self.a0 = 0.20
        self.ax = 0.72
        self.ac = 0.81
        self.exx_fraction = self.a0

        self.lda = LDA()
        self.b88 = B88()
        self.lyp = LYP()

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("B3LYP 需要密度梯度")

        exc_x_lda, vxc_x_lda = self.lda._slater_exchange(rho)
        exc_c_vwn, vxc_c_vwn = self.lda._vwn5_correlation(rho)

        exc_x_b88, vxc_x_b88 = self.b88.compute(rho, grad_rho)
        exc_c_lyp, vxc_c_lyp = self.lyp.compute(rho, grad_rho)

        exc_x = (1.0 - self.a0 - self.ax) * exc_x_lda + self.ax * exc_x_b88
        exc_c = (1.0 - self.ac) * exc_c_vwn + self.ac * exc_c_lyp

        vxc_x = (1.0 - self.a0 - self.ax) * vxc_x_lda + self.ax * vxc_x_b88
        vxc_c = (1.0 - self.ac) * vxc_c_vwn + self.ac * vxc_c_lyp

        return exc_x + exc_c, vxc_x + vxc_c


class PW91(GGA):
    """PW91 泛函 (Perdew-Wang 91)。"""

    def __init__(self):
        super().__init__()
        self.name = "pw91"
        self.beta = 0.0042
        self.mu = 0.2195149727645171

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("PW91 需要密度梯度")

        s = self._compute_s(rho, grad_rho)

        exc_x, vxc_x = self._pw91_exchange(rho, s)
        exc_c, vxc_c = self._pw91_correlation(rho, s)

        return exc_x + exc_c, vxc_x + vxc_c

    def _compute_s(self, rho: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
        kf = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)
        denom = 2.0 * kf * rho
        mask = denom < 1e-20
        s = np.where(mask, 0.0, grad_norm / denom)
        s = np.clip(s, 0.0, 100.0)
        return s

    def _pw91_exchange(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        f_x = 1.0 + self.beta / s * (1.0 - np.exp(-s / self.beta))
        exc_x = c_x * rho_13 * f_x

        ds_drho = -4.0 / 9.0 * s / rho
        df_ds = self.beta / s ** 2 * (np.exp(-s / self.beta) * s / self.beta + np.exp(-s / self.beta) - 1.0)
        df_drho = df_ds * ds_drho

        vxc_x = (4.0 / 3.0) * exc_x + c_x * rho_13 * rho * df_drho
        return exc_x, vxc_x

    def _pw91_correlation(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        ec_pw, dec_pw_drs = self._pw92_correlation(rho)

        f_c = 1.0 - np.exp(-s) + s ** 2 / 12.0
        exc_c = ec_pw * f_c

        dec_drho = -rs / (3.0 * rho) * dec_pw_drs
        ds_drho = -4.0 / 9.0 * s / rho
        df_c_ds = np.exp(-s) * s + s / 6.0
        df_c_drho = df_c_ds * ds_drho

        vxc_c = exc_c + rho * (dec_drho * f_c + ec_pw * df_c_drho)
        return exc_c, vxc_c

    def _pw92_correlation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perdew-Wang 1992 非极化相关能参数化。"""
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)

        a = 0.0310907
        a1 = 0.21370
        b1 = 7.5957
        b2 = 3.5876
        b3 = 1.6382
        b4 = 0.49294

        rs12 = np.sqrt(rs)
        rs32 = rs * rs12
        rs2 = rs ** 2

        q0 = -2.0 * a * (1.0 + a1 * rs)
        q1 = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
        q1p = a * (b1 / rs12 + 2.0 * b2 + 3.0 * b3 * rs12 + 4.0 * b4 * rs)

        ec = q0 * np.log(1.0 + 1.0 / q1)
        dec_drs = -2.0 * a * a1 * np.log(1.0 + 1.0 / q1) - q0 * q1p / (q1 ** 2 + q1)

        return ec, dec_drs


class BLYP(GGA):
    """BLYP 泛函 (Becke88 + LYP)。"""

    def __init__(self):
        super().__init__()
        self.name = "blyp"

        self.b88 = B88()
        self.lyp = LYP()

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("BLYP 需要密度梯度")

        exc_x_b88, vxc_x_b88 = self.b88.compute(rho, grad_rho)
        exc_c_lyp, vxc_c_lyp = self.lyp.compute(rho, grad_rho)

        return exc_x_b88 + exc_c_lyp, vxc_x_b88 + vxc_c_lyp


class PBE0(GGA):
    """PBE0 混合泛函 (PBE + 25% HF交换)。"""

    def __init__(self):
        super().__init__()
        self.name = "pbe0"
        self.exx_fraction = 0.25

        self.pbe = PBE()

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("PBE0 需要密度梯度")

        exc_pbe, vxc_pbe = self.pbe.compute(rho, grad_rho)

        return exc_pbe, vxc_pbe


class revPBE(GGA):
    """revPBE 泛函 (revised PBE)。"""

    def __init__(self):
        super().__init__()
        self.name = "revpbe"

        self.k = 1.245
        self.mu = 0.2195149727645171

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("revPBE 需要密度梯度")

        s = self._compute_s(rho, grad_rho)

        exc_x, vxc_x = self._revpbe_exchange(rho, s)
        exc_c, vxc_c = self._pbe_correlation(rho, s)

        return exc_x + exc_c, vxc_x + vxc_c

    def _compute_s(self, rho: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
        kf = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)
        denom = 2.0 * kf * rho
        mask = denom < 1e-20
        s = np.where(mask, 0.0, grad_norm / denom)
        s = np.clip(s, 0.0, 100.0)
        return s

    def _revpbe_exchange(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        f_x = 1.0 + self.k * (1.0 - np.exp(-self.mu * s ** 2 / self.k))
        exc_x = c_x * rho_13 * f_x

        ds2_drho = -8.0 / 9.0 * s ** 2 / rho
        df_ds2 = self.mu / (1.0 + self.mu * s ** 2 / self.k) ** 2
        df_drho = df_ds2 * ds2_drho

        vxc_x = (4.0 / 3.0) * exc_x + c_x * rho_13 * rho * df_drho
        return exc_x, vxc_x

    def _pbe_correlation(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        ec_pw, dec_pw_drs = self._pw92_correlation(rho)

        s2 = s ** 2
        f_c = 1.0 / (1.0 + self.beta * s2)
        exc_c = ec_pw * f_c

        dec_drho = -rs / (3.0 * rho) * dec_pw_drs
        ds_drho = -4.0 / 9.0 * s / rho
        df_c_ds2 = -self.beta / (1.0 + self.beta * s2) ** 2
        df_c_drho = df_c_ds2 * 2.0 * s * ds_drho

        vxc_c = exc_c + rho * (dec_drho * f_c + ec_pw * df_c_drho)
        return exc_c, vxc_c

    def _pw92_correlation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perdew-Wang 1992 非极化相关能参数化。"""
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)

        a = 0.0310907
        a1 = 0.21370
        b1 = 7.5957
        b2 = 3.5876
        b3 = 1.6382
        b4 = 0.49294

        rs12 = np.sqrt(rs)
        rs32 = rs * rs12
        rs2 = rs ** 2

        q0 = -2.0 * a * (1.0 + a1 * rs)
        q1 = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
        q1p = a * (b1 / rs12 + 2.0 * b2 + 3.0 * b3 * rs12 + 4.0 * b4 * rs)

        ec = q0 * np.log(1.0 + 1.0 / q1)
        dec_drs = -2.0 * a * a1 * np.log(1.0 + 1.0 / q1) - q0 * q1p / (q1 ** 2 + q1)

        return ec, dec_drs


class RPBE(GGA):
    """RPBE 泛函 (relativistic PBE)。"""

    def __init__(self):
        super().__init__()
        self.name = "rpbe"

        self.k = 1.0
        self.mu = 0.804

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("RPBE 需要密度梯度")

        s = self._compute_s(rho, grad_rho)

        exc_x, vxc_x = self._rpbe_exchange(rho, s)
        exc_c, vxc_c = self._pbe_correlation(rho, s)

        return exc_x + exc_c, vxc_x + vxc_c

    def _compute_s(self, rho: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
        kf = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)
        denom = 2.0 * kf * rho
        mask = denom < 1e-20
        s = np.where(mask, 0.0, grad_norm / denom)
        s = np.clip(s, 0.0, 100.0)
        return s

    def _rpbe_exchange(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        f_x = 1.0 + self.k - self.k / (1.0 + self.mu * s ** 2 / self.k)
        exc_x = c_x * rho_13 * f_x

        ds2_drho = -8.0 / 9.0 * s ** 2 / rho
        df_ds2 = self.mu / self.k / (1.0 + self.mu * s ** 2 / self.k) ** 2
        df_drho = df_ds2 * ds2_drho

        vxc_x = (4.0 / 3.0) * exc_x + c_x * rho_13 * rho * df_drho
        return exc_x, vxc_x

    def _pbe_correlation(self, rho: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        ec_pw, dec_pw_drs = self._pw92_correlation(rho)

        s2 = s ** 2
        f_c = 1.0 / (1.0 + self.beta * s2)
        exc_c = ec_pw * f_c

        dec_drho = -rs / (3.0 * rho) * dec_pw_drs
        ds_drho = -4.0 / 9.0 * s / rho
        df_c_ds2 = -self.beta / (1.0 + self.beta * s2) ** 2
        df_c_drho = df_c_ds2 * 2.0 * s * ds_drho

        vxc_c = exc_c + rho * (dec_drho * f_c + ec_pw * df_c_drho)
        return exc_c, vxc_c

    def _pw92_correlation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perdew-Wang 1992 非极化相关能参数化。"""
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)

        a = 0.0310907
        a1 = 0.21370
        b1 = 7.5957
        b2 = 3.5876
        b3 = 1.6382
        b4 = 0.49294

        rs12 = np.sqrt(rs)
        rs32 = rs * rs12
        rs2 = rs ** 2

        q0 = -2.0 * a * (1.0 + a1 * rs)
        q1 = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
        q1p = a * (b1 / rs12 + 2.0 * b2 + 3.0 * b3 * rs12 + 4.0 * b4 * rs)

        ec = q0 * np.log(1.0 + 1.0 / q1)
        dec_drs = -2.0 * a * a1 * np.log(1.0 + 1.0 / q1) - q0 * q1p / (q1 ** 2 + q1)

        return ec, dec_drs


class MetaGGA(XCFunctional):
    """meta-GGA 泛函基类, 使用动能密度 tau。"""

    def __init__(self):
        super().__init__()
        self.type = "meta-gga"


class TPSS(MetaGGA):
    """
    TPSS (Tao-Perdew-Staroverov-Scuseria) meta-GGA 泛函。

    参考: Tao, Perdew, Staroverov, Scuseria,
    Phys. Rev. Lett. 91, 146401 (2003)

    TPSS 不含 HF 交换, 是纯 meta-GGA 泛函。
    """

    def __init__(self):
        super().__init__()
        self.name = "tpss"
        self.exx_fraction = 0.0

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if tau is None or grad_rho is None:
            raise ValueError("TPSS 需要动能密度 tau 和密度梯度")

        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)

        # 数值保护: 避免 rho 过小导致除以零
        rho_safe = np.maximum(rho, 1e-20)
        kf = (3.0 * math.pi ** 2 * rho_safe) ** (1.0 / 3.0)

        denom = 2.0 * kf * rho_safe
        mask = denom < 1e-20
        s = np.where(mask, 0.0, grad_norm / denom)
        s = np.clip(s, 0.0, 100.0)

        # alpha: 动能密度修正参数, 限制在合理范围避免发散
        tau_w = 0.25 * grad_norm ** 2 / rho_safe
        tau_unif = 0.25 * kf ** 2 * rho_safe
        alpha_raw = np.where(tau_unif < 1e-20, 0.0, (tau - tau_w) / tau_unif)
        alpha = np.clip(alpha_raw, 0.0, 10.0)

        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho_safe ** (1.0 / 3.0)

        b = 0.40
        c = 2.35204
        e = 1.537
        d = 2.0

        p = 1.0 + b * alpha ** 2 * (c + alpha) / (1.0 + b * alpha * (alpha + c))
        x_tpss = s ** 2 * p

        exc_x = c_x * rho_13 * (1.0 + e * x_tpss) / (1.0 + d * x_tpss)

        rs = (3.0 / (4.0 * math.pi * rho_safe)) ** (1.0 / 3.0)
        ec, vxc_c = self._tpss_correlation(rho_safe, rs, s, alpha)

        exc = exc_x + ec
        vxc = self._compute_vxc_x_tpss(rho_safe, s, alpha, exc_x) + vxc_c

        return exc, vxc

    def _tpss_correlation(self, rho, rs, s, alpha):
        ec_pw, dec_pw_drs = self._pw92_correlation(rho)

        f_c = 1.0 / (1.0 + 0.06672455060314922 * s ** 2)
        exc_c = ec_pw * f_c

        dec_drho = -rs / (3.0 * rho) * dec_pw_drs
        ds_drho = -4.0 / 9.0 * s / rho
        df_c_ds2 = -0.06672455060314922 / (1.0 + 0.06672455060314922 * s ** 2) ** 2
        df_c_drho = df_c_ds2 * 2.0 * s * ds_drho

        vxc_c = exc_c + rho * (dec_drho * f_c + ec_pw * df_c_drho)
        return exc_c, vxc_c

    def _compute_vxc_x_tpss(self, rho, s, alpha, exc_x):
        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        b = 0.40
        c = 2.35204
        e = 1.537
        d = 2.0

        p = 1.0 + b * alpha ** 2 * (c + alpha) / (1.0 + b * alpha * (alpha + c))
        x_tpss = s ** 2 * p

        ds_drho = -4.0 / 9.0 * s / rho
        d_alpha_drho = -alpha / rho

        dp_dalpha = b * alpha * (2.0 * (c + alpha) * (1.0 + b * alpha * (alpha + c)) + alpha ** 2 * (1.0 + 2.0 * b * alpha)) / (1.0 + b * alpha * (alpha + c)) ** 2
        dp_drho = dp_dalpha * d_alpha_drho

        dx_drho = 2.0 * s * ds_drho * p + s ** 2 * dp_drho

        df_dx = (e * (1.0 + d * x_tpss) - d * (1.0 + e * x_tpss)) / (1.0 + d * x_tpss) ** 2
        df_drho = df_dx * dx_drho

        vxc_x = (4.0 / 3.0) * exc_x + c_x * rho_13 * rho * df_drho
        return vxc_x

    def _pw92_correlation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perdew-Wang 1992 非极化相关能参数化。"""
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)

        a = 0.0310907
        a1 = 0.21370
        b1 = 7.5957
        b2 = 3.5876
        b3 = 1.6382
        b4 = 0.49294

        rs12 = np.sqrt(rs)
        rs32 = rs * rs12
        rs2 = rs ** 2

        q0 = -2.0 * a * (1.0 + a1 * rs)
        q1 = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
        q1p = a * (b1 / rs12 + 2.0 * b2 + 3.0 * b3 * rs12 + 4.0 * b4 * rs)

        ec = q0 * np.log(1.0 + 1.0 / q1)
        dec_drs = -2.0 * a * a1 * np.log(1.0 + 1.0 / q1) - q0 * q1p / (q1 ** 2 + q1)

        return ec, dec_drs


class SCAN(MetaGGA):
    """
    SCAN (Strongly Constrained and Appropriately Normed) meta-GGA 泛函。

    参考: Sun, Ruzsinszky, Perdew,
    Phys. Rev. Lett. 115, 036402 (2015)

    满足 17 个已知物理约束条件。
    """

    def __init__(self):
        super().__init__()
        self.name = "scan"
        self.exx_fraction = 0.0

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if tau is None or grad_rho is None:
            raise ValueError("SCAN 需要动能密度 tau 和密度梯度")

        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)
        kf = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
        s = grad_norm / (2.0 * kf * rho)
        alpha = (tau - 0.25 * grad_norm ** 2 / rho) / (0.25 * kf ** 2 * rho)

        a1 = 4.9479
        f_x = a1 * s ** 2 / (1.0 + a1 * s ** 2)

        exc_x = c_x * rho_13 * (1.0 + f_x)

        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        ec, vxc_c = self._scan_correlation(rho, rs, s, alpha)

        exc = exc_x + ec
        vxc = self._compute_vxc_x_scan(rho, s, exc_x) + vxc_c

        return exc, vxc

    def _scan_correlation(self, rho, rs, s, alpha):
        ec_pw, dec_pw_drs = self._pw92_correlation(rho)

        f_c = 1.0 / (1.0 + 0.06672455060314922 * s ** 2)
        exc_c = ec_pw * f_c

        dec_drho = -rs / (3.0 * rho) * dec_pw_drs
        ds_drho = -4.0 / 9.0 * s / rho
        df_c_ds2 = -0.06672455060314922 / (1.0 + 0.06672455060314922 * s ** 2) ** 2
        df_c_drho = df_c_ds2 * 2.0 * s * ds_drho

        vxc_c = exc_c + rho * (dec_drho * f_c + ec_pw * df_c_drho)
        return exc_c, vxc_c

    def _compute_vxc_x_scan(self, rho, s, exc_x):
        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        a1 = 4.9479

        ds_drho = -4.0 / 9.0 * s / rho
        df_ds2 = a1 / (1.0 + a1 * s ** 2) ** 2
        df_drho = df_ds2 * 2.0 * s * ds_drho

        vxc_x = (4.0 / 3.0) * exc_x + c_x * rho_13 * rho * df_drho
        return vxc_x

    def _pw92_correlation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perdew-Wang 1992 非极化相关能参数化。"""
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)

        a = 0.0310907
        a1 = 0.21370
        b1 = 7.5957
        b2 = 3.5876
        b3 = 1.6382
        b4 = 0.49294

        rs12 = np.sqrt(rs)
        rs32 = rs * rs12
        rs2 = rs ** 2

        q0 = -2.0 * a * (1.0 + a1 * rs)
        q1 = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
        q1p = a * (b1 / rs12 + 2.0 * b2 + 3.0 * b3 * rs12 + 4.0 * b4 * rs)

        ec = q0 * np.log(1.0 + 1.0 / q1)
        dec_drs = -2.0 * a * a1 * np.log(1.0 + 1.0 / q1) - q0 * q1p / (q1 ** 2 + q1)

        return ec, dec_drs


class M06L(MetaGGA):
    """
    M06-L (Minnesota 2006 Local) meta-GGA 泛函。

    参考: Zhao, Truhlar,
    J. Chem. Phys. 125, 194101 (2006)

    高精度 meta-GGA, 适合主族和过渡金属化学。
    """

    def __init__(self):
        super().__init__()
        self.name = "m06-l"
        self.exx_fraction = 0.0

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if tau is None or grad_rho is None:
            raise ValueError("M06-L 需要动能密度 tau 和密度梯度")

        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        grad_norm = np.linalg.norm(grad_rho, axis=1) if grad_rho.ndim == 2 else np.abs(grad_rho)
        kf = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
        s = grad_norm / (2.0 * kf * rho)
        alpha = (tau - 0.25 * grad_norm ** 2 / rho) / (0.25 * kf ** 2 * rho)

        exc_x = c_x * rho_13 * (1.0 + 0.8367 * s ** 2 / (1.0 + 0.8367 * s ** 2))

        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        ec, vxc_c = self._m06l_correlation(rho, rs)

        exc = exc_x + ec
        vxc = self._compute_vxc_x_m06l(rho, s, exc_x) + vxc_c

        return exc, vxc

    def _m06l_correlation(self, rho, rs):
        ec_pw, dec_pw_drs = self._pw92_correlation(rho)

        f_c = 1.0 / (1.0 + 0.06672455060314922 * rs ** 2)
        exc_c = ec_pw * f_c

        dec_drho = -rs / (3.0 * rho) * dec_pw_drs
        df_c_drho = -2.0 * 0.06672455060314922 * rs / (1.0 + 0.06672455060314922 * rs ** 2) ** 2 * (-rs / (3.0 * rho))

        vxc_c = exc_c + rho * (dec_drho * f_c + ec_pw * df_c_drho)
        return exc_c, vxc_c

    def _compute_vxc_x_m06l(self, rho, s, exc_x):
        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        a = 0.8367

        ds_drho = -4.0 / 9.0 * s / rho
        df_ds2 = a / (1.0 + a * s ** 2) ** 2
        df_drho = df_ds2 * 2.0 * s * ds_drho

        vxc_x = (4.0 / 3.0) * exc_x + c_x * rho_13 * rho * df_drho
        return vxc_x

    def _pw92_correlation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perdew-Wang 1992 非极化相关能参数化。"""
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)

        a = 0.0310907
        a1 = 0.21370
        b1 = 7.5957
        b2 = 3.5876
        b3 = 1.6382
        b4 = 0.49294

        rs12 = np.sqrt(rs)
        rs32 = rs * rs12
        rs2 = rs ** 2

        q0 = -2.0 * a * (1.0 + a1 * rs)
        q1 = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
        q1p = a * (b1 / rs12 + 2.0 * b2 + 3.0 * b3 * rs12 + 4.0 * b4 * rs)

        ec = q0 * np.log(1.0 + 1.0 / q1)
        dec_drs = -2.0 * a * a1 * np.log(1.0 + 1.0 / q1) - q0 * q1p / (q1 ** 2 + q1)

        return ec, dec_drs


class DoubleHybrid(XCFunctional):
    """双杂化泛函基类: 在 DFT 基础上加入 MP2 风格的二阶相关能。"""

    def __init__(self):
        super().__init__()
        self.type = "double-hybrid"
        self.mp2_fraction = 0.0
        self.ss_mp2_fraction = 0.0
        self.os_mp2_fraction = 0.0


class B2PLYP(DoubleHybrid):
    """B2-PLYP 双杂化泛函。

    E_XC = (1-a0-ax) E_X^LDA + ax E_X^B88 + a0 E_X^HF + (1-b) E_C^LYP + b E_C^MP2
    其中 a0=0.53, ax=0.15, b=0.27.

    参考: Grimme, J. Chem. Phys. 124, 034108 (2006)
    """

    def __init__(self):
        super().__init__()
        self.name = "b2-plyp"
        self.exx_fraction = 0.53
        self.mp2_fraction = 0.27
        self.ax = 0.15

        self.b88 = B88()
        self.lyp = LYP()

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("B2-PLYP 需要密度梯度")

        c_x = -3.0 / 4.0 * (3.0 / math.pi) ** (1.0 / 3.0)
        rho_13 = rho ** (1.0 / 3.0)

        exc_x_lda = c_x * rho_13
        vxc_x_lda = (4.0 / 3.0) * exc_x_lda

        exc_x_b88, vxc_x_b88 = self.b88.compute(rho, grad_rho)
        exc_c_lyp, vxc_c_lyp = self.lyp.compute(rho, grad_rho)

        exc_x = (1.0 - self.exx_fraction - self.ax) * exc_x_lda + self.ax * exc_x_b88
        exc_c = (1.0 - self.mp2_fraction) * exc_c_lyp

        vxc_x = (1.0 - self.exx_fraction - self.ax) * vxc_x_lda + self.ax * vxc_x_b88
        vxc_c = (1.0 - self.mp2_fraction) * vxc_c_lyp

        return exc_x + exc_c, vxc_x + vxc_c


class DSDPBEP86(DoubleHybrid):
    """DSD-PBEP86 双杂化泛函 (自旋分尺度)。

    E_XC = c_x * E_X^HF + (1-c_x) E_X^PBE86 + c_c,ss * E_C^PBE_ss
         + c_c,os * E_C^PBE_os + c_mp2,ss * E_C^MP2_ss
         + c_mp2,os * E_C^MP2_os

    参考: Kozuch & Martin, Phys. Chem. Chem. Phys. 13, 20104 (2011)
    """

    def __init__(self):
        super().__init__()
        self.name = "dsd-pbep86"
        self.exx_fraction = 0.69
        self.ss_mp2_fraction = 0.46
        self.os_mp2_fraction = 0.05

        self.pbe = PBE()

    def compute(
        self,
        rho: np.ndarray,
        grad_rho: np.ndarray = None,
        lap_rho: np.ndarray = None,
        tau: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if grad_rho is None:
            raise ValueError("DSD-PBEP86 需要密度梯度")

        exc_pbe, vxc_pbe = self.pbe.compute(rho, grad_rho)

        exc = (1.0 - self.exx_fraction) * exc_pbe
        vxc = (1.0 - self.exx_fraction) * vxc_pbe

        return exc, vxc


def get_functional(name: str) -> XCFunctional:
    func_map = {
        "lda": LDA(),
        "pbe": PBE(),
        "b3lyp": B3LYP(),
        "svwn": LDA(),
        "b88": B88(),
        "lyp": LYP(),
        "pw91": PW91(),
        "blyp": BLYP(),
        "pbe0": PBE0(),
        "revpbe": revPBE(),
        "rpbe": RPBE(),
        "tpss": TPSS(),
        "scan": SCAN(),
        "m06-l": M06L(),
        "m06l": M06L(),
        "b2-plyp": B2PLYP(),
        "dsd-pbep86": DSDPBEP86(),
    }
    return func_map.get(name.lower(), PBE())