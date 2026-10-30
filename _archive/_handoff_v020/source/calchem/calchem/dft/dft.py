"""
DFT (密度泛函理论) SCF 模块。

实现 RKS (限制性 Kohn-Sham) 和 UKS (非限制性 Kohn-Sham) 方法。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..basis.basis_set import BasisSet
from ..integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from ..integrals.two_electron import compute_eri
from ..mol.molecule import Molecule
from ..utils.constants import ATOMIC_NUMBERS
from ..scf.hf import HFMethod, nuclear_repulsion
from ..scf.diis import DIIS
from ..scf.guess import guess_hcore, guess_sad, _symmetric_orthogonalize, _eigh_safe, _matmul
from ..scf.results import SCFResult
from .functionals import get_functional, XCFunctional


class RKS(HFMethod):
    """限制性 Kohn-Sham (RKS) DFT。"""

    def __init__(
        self,
        mol: Molecule,
        basis: str | BasisSet = "sto-3g",
        xc: str = "pbe",
        grid_level: str = "medium",
        max_cycles: int = 50,
        conv_tol: float = 1e-8,
        diis: bool = True,
        verbose: int = 1,
        init_guess: str = "hcore",
        damping: float = 0.0,
        damping_cycles: int = 5,
        level_shift: float = 0.0,
        diis_start: int = 3,
        d3: bool = False,
    ):
        if isinstance(basis, str):
            self.basis_set = BasisSet(basis)
            self.basis_name = basis
        else:
            self.basis_set = basis
            self.basis_name = basis.name

        self.mol = mol.to_bohr()
        self.xc = xc.lower()
        self.grid_level = grid_level
        self.max_cycles = max_cycles
        self.conv_tol = conv_tol
        self.use_diis = diis
        self.verbose = verbose
        self.init_guess = init_guess
        self.damping = damping
        self.damping_cycles = damping_cycles
        self.level_shift = level_shift
        self.diis_start = diis_start
        self.use_d3 = d3

        self.e_tot: float = 0.0
        self.e_1e: float = 0.0
        self.e_2e: float = 0.0
        self.e_nn: float = 0.0
        self.e_xc: float = 0.0
        self.e_d3: float = 0.0
        self.converged: bool = False
        self.n_cycles: int = 0

        self.mo_energy: Optional[np.ndarray] = None
        self.mo_coeff: Optional[np.ndarray] = None
        self.mo_occ: Optional[np.ndarray] = None
        self.dm: Optional[np.ndarray] = None
        self.fock: Optional[np.ndarray] = None
        self.vxc: Optional[np.ndarray] = None

        self._s: Optional[np.ndarray] = None
        self._t: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._hcore: Optional[np.ndarray] = None
        self._eri: Optional[np.ndarray] = None
        self._functional: Optional[XCFunctional] = None
        self._grid = None

        self._pyscf_mol = None
        self._pyscf_mf = None
        self._pyscf_grad = None

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _init_grid(self):
        if self._grid is None:
            from .grid import DFTGrid
            self._grid = DFTGrid(self.mol, self.basis_set, self.grid_level)

    def build(self) -> None:
        self._print("构建积分矩阵...", 1)

        self._s = compute_overlap(self.basis_set, self.mol)
        self._t = compute_kinetic(self.basis_set, self.mol)
        self._v = compute_nuclear(self.basis_set, self.mol)
        self._hcore = self._t + self._v
        self._eri = compute_eri(self.basis_set, self.mol)
        self.e_nn = nuclear_repulsion(self.mol)
        self._functional = get_functional(self.xc)
        self._init_grid()

        nbf = self._s.shape[0]
        self._print(f"  基函数数目: {nbf}", 2)
        self._print(f"  电子数: {self.mol.nelectron}", 2)
        self._print(f"  核排斥能: {self.e_nn:.10f} Eh", 2)
        self._print(f"  交换关联泛函: {self._functional.name}", 2)
        self._print(f"  积分网格点: {self._grid.n_points}", 2)

    def get_init_dm(self) -> np.ndarray:
        if self._hcore is None or self._s is None:
            self.build()

        if self.init_guess == "sad":
            return guess_sad(self.mol, self.basis_set, restricted=True)
        else:
            return guess_hcore(
                self._hcore, self._s, self.mol.nelectron, restricted=True
            )

    def _compute_vxc(self, dm: np.ndarray) -> np.ndarray:
        from .fock import compute_vxc
        return compute_vxc(dm, self.basis_set, self.mol, self._functional, self._grid)

    def _build_fock(self, dm: np.ndarray) -> np.ndarray:
        from .fock import build_fock_dft
        self.vxc = self._compute_vxc(dm)
        fock = build_fock_dft(self._hcore, dm, self._eri, self.vxc)
        if self._functional.exx_fraction > 0:
            k = np.einsum("ls,mlns->mn", dm, self._eri)
            fock -= self._functional.exx_fraction * k
        return fock

    def _compute_xc_energy(self, dm: np.ndarray) -> float:
        from .fock import _compute_density_gradient

        rho = self._grid.compute_density(dm, self.basis_set, self.mol)
        grad_rho = _compute_density_gradient(dm, self.basis_set, self.mol, self._grid)
        
        tau = None
        if self._functional.type == "meta-gga":
            tau = self._grid.compute_kinetic_density(dm, self.basis_set, self.mol)
        
        exc, _ = self._functional.compute(rho, grad_rho, tau=tau)
        return float(np.sum(exc * rho * self._grid.weights))

    def _compute_exx_energy(self, dm: np.ndarray) -> float:
        if self._functional.exx_fraction <= 0:
            return 0.0
        k = np.einsum("ls,mlns->mn", dm, self._eri)
        return -0.5 * self._functional.exx_fraction * np.einsum("mn,mn->", dm, k)

    def _compute_d3_energy(self) -> float:
        """计算D3色散校正能。"""
        if not self.use_d3:
            return 0.0
        from .d3 import compute_d3_energy
        return compute_d3_energy(
            self.mol.symbols,
            self.mol.coordinates,
            functional=self.xc,
        )

    def _compute_energy(self, dm: np.ndarray) -> float:
        e_1e = np.einsum("mn,mn->", dm, self._hcore)
        j = np.einsum("ls,mnls->mn", dm, self._eri)
        e_j = 0.5 * np.einsum("mn,mn->", dm, j)
        e_xc = self._compute_xc_energy(dm)
        e_exx = self._compute_exx_energy(dm)
        e_d3 = self._compute_d3_energy()
        return e_1e + e_j + e_xc + e_exx + e_d3 + self.e_nn

    def kernel(self) -> float:
        if self._hcore is None:
            self.build()

        nbf = self._s.shape[0]
        nelectron = self.mol.nelectron
        nocc = nelectron // 2

        self._print("\n" + "=" * 60, 1)
        self._print(f"  RKS/{self.basis_name} ({self.xc}) SCF 计算", 1)
        self._print("=" * 60, 1)

        dm = self.get_init_dm()
        fock = self._build_fock(dm)
        e_prev = self._compute_energy(dm)

        self._print(f"\n  初始能量: {e_prev:.10f} Eh", 1)
        if self._functional.exx_fraction > 0:
            self._print(f"  精确交换比例: {self._functional.exx_fraction:.3f}", 2)
        if self.damping > 0:
            self._print(f"  阻尼: {self.damping:.2f} ({self.damping_cycles} cycles)", 2)
        if self.level_shift > 0:
            self._print(f"  能级移动: {self.level_shift:.4f} Eh", 2)

        diis = DIIS(ndim=6) if self.use_diis else None
        diis_started = False

        for cycle in range(1, self.max_cycles + 1):
            self.n_cycles = cycle

            if diis is not None and cycle >= self.diis_start:
                diis.add(fock, dm, self._s)
                if not diis_started and len(diis) >= 2:
                    diis_started = True
                    self._print(f"  DIIS started at cycle {cycle}", 2)
                if diis_started:
                    fock_diis = diis.extrapolate()
                    if fock_diis is not None:
                        fock = fock_diis

            if self.level_shift > 0:
                x = _symmetric_orthogonalize(self._s)
                f_ortho = _matmul(_matmul(x.T, fock), x)
                eigval, eigvec_ortho = _eigh_safe(f_ortho)
                mo_coeff = _matmul(x, eigvec_ortho)
                shift = np.zeros_like(eigval)
                shift[nocc:] += self.level_shift
                f_ortho += np.diag(shift)
                eigval, eigvec_ortho = _eigh_safe(f_ortho)
                mo_coeff = _matmul(x, eigvec_ortho)
            else:
                x = _symmetric_orthogonalize(self._s)
                f_ortho = _matmul(_matmul(x.T, fock), x)
                eigval, eigvec_ortho = _eigh_safe(f_ortho)
                mo_coeff = _matmul(x, eigvec_ortho)

            mo_energy = eigval
            mo_occ = np.zeros(nbf)
            mo_occ[:nocc] = 2.0

            dm_new = 2.0 * _matmul(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)

            if self.damping > 0 and cycle <= self.damping_cycles:
                dm_new = (1.0 - self.damping) * dm_new + self.damping * dm

            fock = self._build_fock(dm_new)
            e_new = self._compute_energy(dm_new)

            de = e_new - e_prev
            ddm = np.sqrt(np.mean((dm_new - dm) ** 2))

            self._print(
                f"  迭代 {cycle:3d}: E = {e_new:.10f} Eh  "
                f"dE = {de:+.2e}  RMS(D) = {ddm:.2e}",
                1,
            )

            dm = dm_new

            if abs(de) < self.conv_tol and ddm < self.conv_tol * 10:
                self.converged = True
                break

            e_prev = e_new

        self.e_tot = e_new
        self.e_1e = np.einsum("mn,mn->", dm, self._hcore)
        j = np.einsum("ls,mnls->mn", dm, self._eri)
        self.e_j = 0.5 * np.einsum("mn,mn->", dm, j)
        self.e_xc = self._compute_xc_energy(dm)
        self.e_exx = self._compute_exx_energy(dm)
        self.e_d3 = self._compute_d3_energy()
        self.e_2e = self.e_j + self.e_xc + self.e_exx + self.e_d3
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.dm = dm
        self.fock = fock

        if self.converged:
            self._print("\n  SCF 收敛 ✓", 1)
        else:
            self._print(f"\n  SCF 未收敛 ✗ (达到最大迭代 {self.max_cycles} 次)", 1)

        self._print(f"  总能量: {self.e_tot:.10f} Eh", 1)
        self._print(f"  迭代次数: {self.n_cycles}", 1)
        self._print("=" * 60 + "\n", 1)

        return self.e_tot

    def get_results(self) -> SCFResult:
        return SCFResult(
            energy=self.e_tot,
            converged=self.converged,
            n_cycles=self.n_cycles,
            method=f"RKS({self.xc})",
            basis=self.basis_name,
            e_1e=self.e_1e,
            e_2e=self.e_2e,
            e_nn=self.e_nn,
            mo_energies=self.mo_energy,
            mo_occupations=self.mo_occ,
            mo_coefficients=self.mo_coeff,
            dm=self.dm,
            fock=self.fock,
        )


class UKS(HFMethod):
    """非限制性 Kohn-Sham (UKS) DFT。"""

    def __init__(
        self,
        mol: Molecule,
        basis: str | BasisSet = "sto-3g",
        xc: str = "pbe",
        grid_level: str = "medium",
        max_cycles: int = 50,
        conv_tol: float = 1e-8,
        diis: bool = True,
        verbose: int = 1,
        init_guess: str = "hcore",
        damping: float = 0.0,
        damping_cycles: int = 5,
        level_shift: float = 0.0,
        diis_start: int = 3,
        d3: bool = False,
    ):
        if isinstance(basis, str):
            self.basis_set = BasisSet(basis)
            self.basis_name = basis
        else:
            self.basis_set = basis
            self.basis_name = basis.name

        self.mol = mol.to_bohr()
        self.xc = xc.lower()
        self.grid_level = grid_level
        self.max_cycles = max_cycles
        self.conv_tol = conv_tol
        self.use_diis = diis
        self.verbose = verbose
        self.init_guess = init_guess
        self.damping = damping
        self.damping_cycles = damping_cycles
        self.level_shift = level_shift
        self.diis_start = diis_start
        self.use_d3 = d3

        self.e_tot: float = 0.0
        self.e_1e: float = 0.0
        self.e_2e: float = 0.0
        self.e_nn: float = 0.0
        self.e_xc: float = 0.0
        self.e_d3: float = 0.0
        self.converged: bool = False
        self.n_cycles: int = 0

        self.mo_energy_a: Optional[np.ndarray] = None
        self.mo_energy_b: Optional[np.ndarray] = None
        self.mo_coeff_a: Optional[np.ndarray] = None
        self.mo_coeff_b: Optional[np.ndarray] = None
        self.mo_occ_a: Optional[np.ndarray] = None
        self.mo_occ_b: Optional[np.ndarray] = None
        self.dm_a: Optional[np.ndarray] = None
        self.dm_b: Optional[np.ndarray] = None
        self.fock_a: Optional[np.ndarray] = None
        self.fock_b: Optional[np.ndarray] = None

        self._s: Optional[np.ndarray] = None
        self._t: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._hcore: Optional[np.ndarray] = None
        self._eri: Optional[np.ndarray] = None
        self._functional: Optional[XCFunctional] = None
        self._grid = None

        self._pyscf_mol = None
        self._pyscf_mf = None
        self._pyscf_grad = None

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _init_grid(self):
        if self._grid is None:
            from .grid import DFTGrid
            self._grid = DFTGrid(self.mol, self.basis_set, self.grid_level)

    def build(self) -> None:
        self._print("构建积分矩阵...", 1)

        self._s = compute_overlap(self.basis_set, self.mol)
        self._t = compute_kinetic(self.basis_set, self.mol)
        self._v = compute_nuclear(self.basis_set, self.mol)
        self._hcore = self._t + self._v
        self._eri = compute_eri(self.basis_set, self.mol)
        self.e_nn = nuclear_repulsion(self.mol)
        self._functional = get_functional(self.xc)
        self._init_grid()

    def get_init_dm(self) -> tuple:
        if self._hcore is None or self._s is None:
            self.build()

        if self.init_guess == "sad":
            return guess_sad(self.mol, self.basis_set, restricted=False)
        else:
            return guess_hcore(
                self._hcore, self._s, self.mol.nelectron, restricted=False
            )

    def _compute_vxc(self, dm_a: np.ndarray, dm_b: np.ndarray) -> np.ndarray:
        from .fock import compute_vxc
        dm_total = dm_a + dm_b
        return compute_vxc(dm_total, self.basis_set, self.mol, self._functional, self._grid)

    def _build_fock(self, dm_a: np.ndarray, dm_b: np.ndarray) -> tuple:
        from .fock import build_fock_dft

        dm_total = dm_a + dm_b
        vxc = self._compute_vxc(dm_a, dm_b)

        j = np.einsum("ls,mnls->mn", dm_total, self._eri)

        f_a = self._hcore + j + vxc
        f_b = self._hcore + j + vxc

        if self._functional.exx_fraction > 0:
            k_a = np.einsum("ls,mlns->mn", dm_a, self._eri)
            k_b = np.einsum("ls,mlns->mn", dm_b, self._eri)
            f_a -= self._functional.exx_fraction * k_a
            f_b -= self._functional.exx_fraction * k_b

        return f_a, f_b

    def _compute_xc_energy(self, dm_a: np.ndarray, dm_b: np.ndarray) -> float:
        from .fock import _compute_density_gradient

        dm_total = dm_a + dm_b
        rho = self._grid.compute_density(dm_total, self.basis_set, self.mol)
        grad_rho = _compute_density_gradient(dm_total, self.basis_set, self.mol, self._grid)
        
        tau = None
        if self._functional.type == "meta-gga":
            tau_a = self._grid.compute_kinetic_density(dm_a, self.basis_set, self.mol)
            tau_b = self._grid.compute_kinetic_density(dm_b, self.basis_set, self.mol)
            tau = tau_a + tau_b
        
        exc, _ = self._functional.compute(rho, grad_rho, tau=tau)
        return float(np.sum(exc * rho * self._grid.weights))

    def _compute_exx_energy(self, dm_a: np.ndarray, dm_b: np.ndarray) -> float:
        if self._functional.exx_fraction <= 0:
            return 0.0
        k_a = np.einsum("ls,mlns->mn", dm_a, self._eri)
        k_b = np.einsum("ls,mlns->mn", dm_b, self._eri)
        return -0.5 * self._functional.exx_fraction * (
            np.einsum("mn,mn->", dm_a, k_a) + np.einsum("mn,mn->", dm_b, k_b)
        )

    def _compute_d3_energy(self) -> float:
        """计算D3色散校正能。"""
        if not self.use_d3:
            return 0.0
        from .d3 import compute_d3_energy
        return compute_d3_energy(
            self.mol.symbols,
            self.mol.coordinates,
            functional=self.xc,
        )

    def kernel(self) -> float:
        if self._hcore is None:
            self.build()

        nbf = self._s.shape[0]
        nelectron = self.mol.nelectron
        nocc_a = (nelectron + self.mol.multiplicity - 1) // 2
        nocc_b = nelectron // 2

        self._print("\n" + "=" * 60, 1)
        self._print(f"  UKS/{self.basis_name} ({self.xc}) SCF 计算", 1)
        self._print("=" * 60, 1)

        dm_a, dm_b = self.get_init_dm()
        fock_a, fock_b = self._build_fock(dm_a, dm_b)

        e_1e = 0.5 * np.einsum("mn,mn->", dm_a + dm_b, self._hcore)
        dm_total = dm_a + dm_b
        j = np.einsum("ls,mnls->mn", dm_total, self._eri)
        e_j = 0.5 * np.einsum("mn,mn->", dm_total, j)
        e_xc = self._compute_xc_energy(dm_a, dm_b)
        e_exx = self._compute_exx_energy(dm_a, dm_b)
        e_d3 = self._compute_d3_energy()

        e_prev = e_1e + e_j + e_xc + e_exx + e_d3 + self.e_nn

        self._print(f"\n  初始能量: {e_prev:.10f} Eh", 1)
        if self._functional.exx_fraction > 0:
            self._print(f"  精确交换比例: {self._functional.exx_fraction:.3f}", 2)
        if self.damping > 0:
            self._print(f"  阻尼: {self.damping:.2f} ({self.damping_cycles} cycles)", 2)
        if self.level_shift > 0:
            self._print(f"  能级移动: {self.level_shift:.4f} Eh", 2)

        diis_a = DIIS(ndim=6) if self.use_diis else None
        diis_b = DIIS(ndim=6) if self.use_diis else None
        diis_started = False

        for cycle in range(1, self.max_cycles + 1):
            self.n_cycles = cycle

            if diis_a is not None and cycle >= self.diis_start:
                diis_a.add(fock_a, dm_a, self._s)
                if not diis_started and len(diis_a) >= 2:
                    diis_started = True
                    self._print(f"  DIIS started at cycle {cycle}", 2)
                if diis_started:
                    fock_a_diis = diis_a.extrapolate()
                    if fock_a_diis is not None:
                        fock_a = fock_a_diis

            if diis_b is not None and cycle >= self.diis_start:
                diis_b.add(fock_b, dm_b, self._s)
                if diis_started:
                    fock_b_diis = diis_b.extrapolate()
                    if fock_b_diis is not None:
                        fock_b = fock_b_diis

            x = _symmetric_orthogonalize(self._s)

            if self.level_shift > 0:
                f_ortho_a = _matmul(_matmul(x.T, fock_a), x)
                eigval_a, eigvec_ortho_a = _eigh_safe(f_ortho_a)
                shift_a = np.zeros_like(eigval_a)
                shift_a[nocc_a:] += self.level_shift
                f_ortho_a += np.diag(shift_a)
                eigval_a, eigvec_ortho_a = _eigh_safe(f_ortho_a)
                mo_coeff_a = _matmul(x, eigvec_ortho_a)

                f_ortho_b = _matmul(_matmul(x.T, fock_b), x)
                eigval_b, eigvec_ortho_b = _eigh_safe(f_ortho_b)
                shift_b = np.zeros_like(eigval_b)
                shift_b[nocc_b:] += self.level_shift
                f_ortho_b += np.diag(shift_b)
                eigval_b, eigvec_ortho_b = _eigh_safe(f_ortho_b)
                mo_coeff_b = _matmul(x, eigvec_ortho_b)
            else:
                f_ortho_a = _matmul(_matmul(x.T, fock_a), x)
                eigval_a, eigvec_ortho_a = _eigh_safe(f_ortho_a)
                mo_coeff_a = _matmul(x, eigvec_ortho_a)

                f_ortho_b = _matmul(_matmul(x.T, fock_b), x)
                eigval_b, eigvec_ortho_b = _eigh_safe(f_ortho_b)
                mo_coeff_b = _matmul(x, eigvec_ortho_b)

            mo_occ_a = np.zeros(nbf)
            mo_occ_a[:nocc_a] = 1.0
            mo_occ_b = np.zeros(nbf)
            mo_occ_b[:nocc_b] = 1.0

            dm_a_new = _matmul(mo_coeff_a[:, :nocc_a], mo_coeff_a[:, :nocc_a].T)
            dm_b_new = _matmul(mo_coeff_b[:, :nocc_b], mo_coeff_b[:, :nocc_b].T)

            if self.damping > 0 and cycle <= self.damping_cycles:
                dm_a_new = (1.0 - self.damping) * dm_a_new + self.damping * dm_a
                dm_b_new = (1.0 - self.damping) * dm_b_new + self.damping * dm_b

            fock_a, fock_b = self._build_fock(dm_a_new, dm_b_new)

            e_1e = 0.5 * np.einsum("mn,mn->", dm_a_new + dm_b_new, self._hcore)
            dm_total = dm_a_new + dm_b_new
            j = np.einsum("ls,mnls->mn", dm_total, self._eri)
            e_j = 0.5 * np.einsum("mn,mn->", dm_total, j)
            e_xc = self._compute_xc_energy(dm_a_new, dm_b_new)
            e_exx = self._compute_exx_energy(dm_a_new, dm_b_new)
            e_d3 = self._compute_d3_energy()

            e_new = e_1e + e_j + e_xc + e_exx + e_d3 + self.e_nn

            de = e_new - e_prev
            ddm = np.sqrt(np.mean((dm_a_new - dm_a) ** 2 + (dm_b_new - dm_b) ** 2))

            self._print(
                f"  迭代 {cycle:3d}: E = {e_new:.10f} Eh  "
                f"dE = {de:+.2e}  RMS(D) = {ddm:.2e}",
                1,
            )

            dm_a, dm_b = dm_a_new, dm_b_new

            if abs(de) < self.conv_tol and ddm < self.conv_tol * 10:
                self.converged = True
                break

            e_prev = e_new

        self.e_tot = e_new
        self.e_1e = e_1e
        self.e_j = e_j
        self.e_xc = e_xc
        self.e_exx = e_exx
        self.e_d3 = self._compute_d3_energy()
        self.e_2e = e_j + e_xc + e_exx + self.e_d3
        self.mo_energy_a = eigval_a
        self.mo_energy_b = eigval_b
        self.mo_coeff_a = mo_coeff_a
        self.mo_coeff_b = mo_coeff_b
        self.mo_occ_a = mo_occ_a
        self.mo_occ_b = mo_occ_b
        self.dm_a = dm_a
        self.dm_b = dm_b
        self.fock_a = fock_a
        self.fock_b = fock_b

        if self.converged:
            self._print("\n  SCF 收敛 ✓", 1)
        else:
            self._print(f"\n  SCF 未收敛 ✗ (达到最大迭代 {self.max_cycles} 次)", 1)

        self._print(f"  总能量: {self.e_tot:.10f} Eh", 1)
        self._print(f"  迭代次数: {self.n_cycles}", 1)
        self._print("=" * 60 + "\n", 1)

        return self.e_tot