"""
DFT (密度泛函理论) SCF 模块。

提供 RKS (限制性 Kohn-Sham) 和 UKS (非限制性 Kohn-Sham) 方法。
完全自研实现，不依赖 PySCF。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..mol.molecule import Molecule
from ..dft.dft import RKS as DFT_RKS, UKS as DFT_UKS
from ..io.output import Results
from .hf import HFMethod


class DFTMethod(HFMethod):
    """DFT 方法基类。"""

    def __init__(
        self,
        mol: Molecule,
        basis: str = "sto-3g",
        xc: str = "b3lyp",
        max_cycles: Optional[int] = None,
        conv_tol: Optional[float] = None,
        diis: bool = True,
        grid_level: str = "medium",
        verbose: int = 1,
        init_guess: str = "hcore",
        d3: bool = False,
        damping: float = 0.0,
        damping_cycles: int = 5,
        level_shift: float = 0.0,
        diis_start: int = 3,
    ):
        super().__init__(mol, basis=basis, max_cycles=max_cycles, conv_tol=conv_tol, diis=diis)
        self.xc = xc.lower()
        self.grid_level = grid_level
        self.verbose = verbose
        self.init_guess = init_guess
        self.use_d3 = d3
        self.damping = damping
        self.damping_cycles = damping_cycles
        self.level_shift = level_shift
        self.diis_start = diis_start
        self.exc: float = 0.0
        self.e_d3: float = 0.0

    def get_results(self) -> Results:
        results = super().get_results()
        results.method = f"DFT/{self.xc}"
        results.properties["exc"] = self.exc
        return results


class RKS(DFTMethod):
    """限制性 Kohn-Sham (RKS) DFT。"""

    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)
        self._dft_impl = None

    def build(self) -> None:
        """构建积分矩阵和DFT网格。"""
        self._dft_impl = DFT_RKS(
            self.mol,
            basis=self.basis,
            xc=self.xc,
            grid_level=self.grid_level,
            max_cycles=self.max_cycles,
            conv_tol=self.conv_tol,
            diis=self.use_diis,
            verbose=self.verbose,
            init_guess=self.init_guess,
            d3=self.use_d3,
            damping=self.damping,
            damping_cycles=self.damping_cycles,
            level_shift=self.level_shift,
            diis_start=self.diis_start,
        )
        self._dft_impl.build()

    def kernel(self) -> float:
        """运行 RKS SCF 计算。"""
        if self._dft_impl is None:
            self.build()

        e_tot = self._dft_impl.kernel()

        self.e_tot = self._dft_impl.e_tot
        self.e_1e = self._dft_impl.e_1e
        self.e_2e = self._dft_impl.e_2e
        self.exc = getattr(self._dft_impl, 'e_xc', 0.0)
        self.e_d3 = getattr(self._dft_impl, 'e_d3', 0.0)
        self.converged = self._dft_impl.converged
        self.n_cycles = self._dft_impl.n_cycles
        self.mo_energy = self._dft_impl.mo_energy
        self.mo_coeff = self._dft_impl.mo_coeff
        self.mo_occ = self._dft_impl.mo_occ
        self.dm = self._dft_impl.dm
        self.fock = self._dft_impl.fock

        return e_tot

    @property
    def _hcore(self):
        return self._dft_impl._hcore if self._dft_impl else None

    @property
    def _s(self):
        return self._dft_impl._s if self._dft_impl else None

    @property
    def _eri(self):
        return self._dft_impl._eri if self._dft_impl else None


class UKS(DFTMethod):
    """非限制性 Kohn-Sham (UKS) DFT。"""

    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)
        self._dft_impl = None

    def build(self) -> None:
        """构建积分矩阵和DFT网格。"""
        self._dft_impl = DFT_UKS(
            self.mol,
            basis=self.basis,
            xc=self.xc,
            grid_level=self.grid_level,
            max_cycles=self.max_cycles,
            conv_tol=self.conv_tol,
            diis=self.use_diis,
            verbose=self.verbose,
            init_guess=self.init_guess,
            d3=self.use_d3,
            damping=self.damping,
            damping_cycles=self.damping_cycles,
            level_shift=self.level_shift,
            diis_start=self.diis_start,
        )
        self._dft_impl.build()

    def kernel(self) -> float:
        """运行 UKS SCF 计算。"""
        if self._dft_impl is None:
            self.build()

        e_tot = self._dft_impl.kernel()

        self.e_tot = self._dft_impl.e_tot
        self.e_1e = self._dft_impl.e_1e
        self.e_2e = self._dft_impl.e_2e
        self.exc = getattr(self._dft_impl, 'e_xc', 0.0)
        self.e_d3 = getattr(self._dft_impl, 'e_d3', 0.0)
        self.converged = self._dft_impl.converged
        self.n_cycles = self._dft_impl.n_cycles
        self.mo_energy = (self._dft_impl.mo_energy_a, self._dft_impl.mo_energy_b)
        self.mo_coeff = (self._dft_impl.mo_coeff_a, self._dft_impl.mo_coeff_b)
        self.mo_occ = (self._dft_impl.mo_occ_a, self._dft_impl.mo_occ_b)
        self.dm = (self._dft_impl.dm_a, self._dft_impl.dm_b)
        self.fock = (self._dft_impl.fock_a, self._dft_impl.fock_b)

        return e_tot

    @property
    def _hcore(self):
        return self._dft_impl._hcore if self._dft_impl else None

    @property
    def _s(self):
        return self._dft_impl._s if self._dft_impl else None

    @property
    def _eri(self):
        return self._dft_impl._eri if self._dft_impl else None


class ROKS(DFTMethod):
    """限制性开壳层 Kohn-Sham (ROKS) DFT。

    使用 Roothaan 方案, 闭壳层和开壳层轨道共享同一套 MO,
    适用于单电子开壳层体系 (如自由基、阳离子、阴离子)。

    Fock 矩阵构建:
    F_μν = H_μν + J_μν + V_xc_μν - exx_fraction * K_μν

    其中:
    - D_total = D_alpha + D_beta = 2*D_closed + D_open
    - J_μν = Σ_lσ D_lσ (μν|lσ)
    - K_μν = Σ_lσ D_lσ (μl|νσ)
    - V_xc 是交换关联势
    """

    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)
        self._n_closed = 0
        self._n_open = 0
        self._functional = None
        self._grid = None
        self.e_xc = 0.0
        self.e_exx = 0.0

    def _init_grid(self):
        if self._grid is None:
            from ..dft.grid import DFTGrid
            self._grid = DFTGrid(self.mol, self.basis_set, self.grid_level)

    def build(self) -> None:
        from ..integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
        from ..integrals.two_electron import compute_eri
        from ..basis.basis_set import BasisSet
        from .hf import nuclear_repulsion

        if isinstance(self.basis, str):
            self.basis_set = BasisSet(self.basis)
            self.basis_name = self.basis
        elif not hasattr(self, 'basis_set'):
            self.basis_set = self.basis

        self.mol = self.mol.to_bohr()

        self._s = compute_overlap(self.basis_set, self.mol)
        self._hcore = compute_kinetic(self.basis_set, self.mol) + compute_nuclear(self.basis_set, self.mol)
        self._eri = compute_eri(self.basis_set, self.mol)
        self.e_nn = nuclear_repulsion(self.mol)

        from ..dft.functionals import get_functional
        self._functional = get_functional(self.xc)
        self._init_grid()

        nalpha = self.mol.na
        nbeta = self.mol.nb
        self._n_closed = nbeta
        self._n_open = nalpha - nbeta

    def _compute_vxc(self, dm: np.ndarray) -> np.ndarray:
        from ..dft.fock import compute_vxc
        return compute_vxc(dm, self.basis_set, self.mol, self._functional, self._grid)

    def _compute_xc_energy(self, dm: np.ndarray) -> float:
        from ..dft.fock import _compute_density_gradient

        rho = self._grid.compute_density(dm, self.basis_set, self.mol)
        grad_rho = _compute_density_gradient(dm, self.basis_set, self.mol, self._grid)
        
        tau = None
        if hasattr(self._grid, 'compute_kinetic_density'):
            tau = self._grid.compute_kinetic_density(dm, self.basis_set, self.mol)
        
        exc, _ = self._functional.compute(rho, grad_rho, tau=tau)
        return float(np.sum(exc * rho * self._grid.weights))

    def _compute_exx_energy(self, dm_total: np.ndarray, dm_open: np.ndarray) -> float:
        if self._functional.exx_fraction <= 0:
            return 0.0
        dm_closed = 0.5 * (dm_total - dm_open)
        k_closed = np.einsum("ls,mlns->mn", dm_closed, self._eri)
        k_open = np.einsum("ls,mlns->mn", dm_open, self._eri)
        e_exx_closed = -np.einsum("mn,mn->", dm_closed, k_closed)
        e_exx_open = -0.5 * np.einsum("mn,mn->", dm_open, k_open)
        return self._functional.exx_fraction * (e_exx_closed + e_exx_open)

    def _build_fock(self, dm_total: np.ndarray, dm_open: np.ndarray) -> np.ndarray:
        dm_closed = 0.5 * (dm_total - dm_open)
        dm_a = dm_closed + dm_open
        dm_b = dm_closed
        
        j_total = np.einsum("ls,mnls->mn", dm_total, self._eri)
        vxc = self._compute_vxc(dm_total)

        f_a = self._hcore + j_total + vxc
        f_b = self._hcore + j_total + vxc
        
        if self._functional.exx_fraction > 0:
            k_a = np.einsum("ls,mlns->mn", dm_a, self._eri)
            k_b = np.einsum("ls,mlns->mn", dm_b, self._eri)
            f_a -= self._functional.exx_fraction * k_a
            f_b -= self._functional.exx_fraction * k_b

        f_avg = 0.5 * (f_a + f_b)
        
        return f_avg, f_a, f_b

    def _compute_energy(self, dm_total: np.ndarray, dm_open: np.ndarray) -> float:
        e_1e = np.einsum("mn,mn->", dm_total, self._hcore)
        j_total = np.einsum("ls,mnls->mn", dm_total, self._eri)
        e_j = np.einsum("mn,mn->", dm_total, j_total) * 0.5
        e_xc = self._compute_xc_energy(dm_total)
        e_exx = self._compute_exx_energy(dm_total, dm_open)

        return e_1e + e_j + e_xc + e_exx + self.e_nn

    def kernel(self) -> float:
        from ..scf.hf import _symmetric_orthogonalize, _eigh_safe, _matmul
        from ..scf.diis import DIIS

        if self._functional is None:
            self.build()

        nbf = self._s.shape[0]
        n_occ = self._n_closed + self._n_open

        x = _symmetric_orthogonalize(self._s)
        f_ortho = x.T @ self._hcore @ x
        eigval, eigvec_ortho = _eigh_safe(f_ortho)
        mo_coeff = x @ eigvec_ortho

        dm_total = _matmul(mo_coeff[:, :self._n_closed] * 2.0, mo_coeff[:, :self._n_closed].T)
        dm_open = np.zeros_like(dm_total)
        if self._n_open > 0:
            dm_total = dm_total + _matmul(
                mo_coeff[:, self._n_closed:n_occ], mo_coeff[:, self._n_closed:n_occ].T
            )
            dm_open = _matmul(
                mo_coeff[:, self._n_closed:n_occ], mo_coeff[:, self._n_closed:n_occ].T
            )

        fock, f_closed, f_open = self._build_fock(dm_total, dm_open)
        e_prev = self._compute_energy(dm_total, dm_open)

        self._print(f"\n  ROKS/{self.basis_name} ({self.xc}) SCF 计算", 1)
        self._print(f"  初始能量: {e_prev:.10f} Eh", 1)

        diis = DIIS(ndim=6) if self.use_diis else None
        diis_started = False

        for cycle in range(1, self.max_cycles + 1):
            self.n_cycles = cycle

            if diis is not None and cycle >= self.diis_start:
                diis.add(fock, dm_total, self._s)
                if not diis_started and len(diis) >= 2:
                    diis_started = True
                    self._print(f"  DIIS started at cycle {cycle}", 2)
                if diis_started:
                    fock_diis = diis.extrapolate()
                    if fock_diis is not None:
                        fock = fock_diis

            f_ortho = _matmul(_matmul(x.T, fock), x)
            eigval, eigvec_ortho = _eigh_safe(f_ortho)
            mo_coeff = _matmul(x, eigvec_ortho)

            mo_occ = np.zeros(nbf)
            mo_occ[:self._n_closed] = 2.0
            mo_occ[self._n_closed:n_occ] = 1.0

            dm_total_new = _matmul(mo_coeff[:, :self._n_closed] * 2.0, mo_coeff[:, :self._n_closed].T)
            dm_open_new = np.zeros_like(dm_total_new)
            if self._n_open > 0:
                dm_total_new = dm_total_new + _matmul(
                    mo_coeff[:, self._n_closed:n_occ], mo_coeff[:, self._n_closed:n_occ].T
                )
                dm_open_new = _matmul(
                    mo_coeff[:, self._n_closed:n_occ], mo_coeff[:, self._n_closed:n_occ].T
                )

            if self.damping > 0 and cycle <= self.damping_cycles:
                dm_total_new = (1.0 - self.damping) * dm_total_new + self.damping * dm_total
                dm_open_new = (1.0 - self.damping) * dm_open_new + self.damping * dm_open

            fock, f_closed, f_open = self._build_fock(dm_total_new, dm_open_new)
            e_new = self._compute_energy(dm_total_new, dm_open_new)

            de = e_new - e_prev
            ddm = np.sqrt(np.mean((dm_total_new - dm_total) ** 2))

            self._print(
                f"  迭代 {cycle:3d}: E = {e_new:.10f} Eh  "
                f"dE = {de:+.2e}  RMS(D) = {ddm:.2e}",
                1,
            )

            dm_total = dm_total_new
            dm_open = dm_open_new

            if abs(de) < self.conv_tol and ddm < self.conv_tol:
                self.converged = True
                break

            e_prev = e_new

        self.e_tot = e_new
        self.e_1e = np.einsum("mn,mn->", dm_total, self._hcore)
        self.e_xc = self._compute_xc_energy(dm_total)
        self.e_exx = self._compute_exx_energy(dm_total, dm_open)
        self.e_2e = e_new - self.e_nn - self.e_1e
        self.mo_energy = eigval
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.dm = dm_total
        self.fock = fock

        if self.converged:
            self._print("\n  SCF 收敛 ✓", 1)
        else:
            self._print(f"\n  SCF 未收敛 ✗", 1)
        self._print(f"  总能量: {self.e_tot:.10f} Eh", 1)

        return self.e_tot

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def get_results(self):
        from .hf import SCFResult
        return SCFResult(
            energy=self.e_tot, converged=self.converged, n_cycles=self.n_cycles,
            method=f"ROKS/{self.xc}", basis=self.basis_name,
            e_1e=self.e_1e, e_2e=self.e_2e, e_nn=self.e_nn,
            mo_energies=self.mo_energy, mo_occupations=self.mo_occ,
            mo_coefficients=self.mo_coeff, dm=self.dm, fock=self.fock,
        )