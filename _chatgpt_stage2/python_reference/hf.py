"""
Hartree-Fock SCF 求解器。

实现 RHF/UHF/ROHF 三种参考态，
支持自适应收敛策略、直接 SCF、RI 加速。
"""

from __future__ import annotations

import numpy as np

from ..basis.basis_set import BasisSet
from ..mol.molecule import Molecule
from ..integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from ..integrals.two_electron import compute_eri
from ..integrals.direct_scf import build_fock_direct, build_fock_direct_uhf
from ..integrals.ri import RIJEngine
from .diis import DIIS, ADIIS, EDIIS, ConvergenceChecker


class HF:
    def __init__(
        self,
        mol: Molecule,
        basis: BasisSet | str,
        method: str = "RHF",
        max_iter: int = 200,
        energy_tol: float = 1e-10,
        density_tol: float = 1e-8,
        gradient_tol: float = 1e-6,
        damping: float = 0.0,
        level_shift: float = 0.0,
        diis: bool = True,
        diis_type: str = "diis",
        direct_scf: bool = False,
        use_ri: bool = False,
        initial_guess: str = "core",
    ):
        self.mol = mol
        self.basis = basis
        self.method = method.upper()
        self.max_iter = max_iter
        self.energy_tol = energy_tol
        self.density_tol = density_tol
        self.gradient_tol = gradient_tol
        self.damping = damping
        self.level_shift = level_shift
        self.diis = diis
        self.diis_type = diis_type.lower()
        self.direct_scf = direct_scf
        self.use_ri = use_ri
        self.initial_guess = initial_guess.lower()

        self._shells = None
        self._shell_range = None
        self._nbf = None
        self._nocc = None
        self._nvirt = None

        self._h_core = None
        self._s = None
        self._s_sqrt = None
        self._fock = None
        self._dm = None
        self._energy = None
        self._mo_coeff = None
        self._mo_energy = None

        self._conv_checker = None
        self._diis_obj = None
        self._ri_engine = None

        self._converged = False

        if isinstance(basis, str):
            self.basis = BasisSet(basis)
        else:
            self.basis = basis
        self._n_iter = 0
        self._history = []

        if hasattr(mol, 'to_bohr'):
            self.mol = mol.to_bohr()
        else:
            self.mol = mol

    @property
    def nbf(self) -> int:
        if self._nbf is None:
            self._setup_basis()
        return self._nbf

    @property
    def nocc(self) -> int:
        if self._nocc is None:
            self._setup_basis()
        return self._nocc

    @property
    def nvirt(self) -> int:
        if self._nvirt is None:
            self._setup_basis()
        return self._nvirt

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def fock(self) -> np.ndarray:
        return self._fock

    @property
    def basis_set(self) -> BasisSet:
        """basis_set 属性别名 (后HF模块通过 mf.basis_set 访问)。"""
        return self.basis

    @property
    def e_tot(self) -> float:
        return self._energy

    @property
    def mo_coeff(self) -> np.ndarray:
        return self._mo_coeff

    @property
    def mo_energy(self) -> np.ndarray:
        return self._mo_energy

    @property
    def mo_occ(self) -> np.ndarray:
        """轨道占据数 (RHF: 2.0/0.0; UHF覆写为 (occ_a, occ_b) 元组)。"""
        if self._mo_coeff is None:
            return None
        occ = np.zeros(self.nbf)
        occ[:self.nocc] = 2.0
        return occ

    @property
    def dm(self) -> np.ndarray:
        return self._dm

    def _setup_basis(self):
        shells_by_atom, nbf = self.basis.build_molecule_basis(self.mol)
        self._shells = []
        for atom_shells in shells_by_atom:
            self._shells.extend(atom_shells)
        self._nbf = nbf

        self._shell_range = []
        idx = 0
        for s in self._shells:
            nf = s.n_functions
            self._shell_range.append((idx, idx + nf))
            idx += nf

        if self.method == "RHF":
            self._nocc = self.mol.nelectron // 2
        elif self.method == "UHF":
            self._nocc = (self.mol.nelectron + self.mol.multiplicity - 1) // 2
            self._nocc_beta = self.mol.nelectron - self._nocc
        elif self.method == "ROHF":
            self._nocc = self.mol.nelectron // 2
        self._nvirt = nbf - self._nocc

    def _build_h_core(self) -> np.ndarray:
        s = compute_overlap(self.basis, self.mol)
        t = compute_kinetic(self.basis, self.mol)
        v = compute_nuclear(self.basis, self.mol)
        self._s = s
        self._h_core = t + v
        return self._h_core

    def _build_s_sqrt(self):
        if self._s is None:
            self._build_h_core()

        eigvals, eigvecs = np.linalg.eigh(self._s)
        eps = 1e-14
        sqrt_vals = np.where(eigvals > eps, 1.0 / np.sqrt(eigvals), 0.0)
        self._s_sqrt = eigvecs @ np.diag(sqrt_vals) @ eigvecs.T

    def _initial_guess(self) -> np.ndarray:
        if self._s is None:
            self._build_h_core()
        if self._s_sqrt is None:
            self._build_s_sqrt()

        if self.initial_guess == "sad":
            return self._guess_sad()
        elif self.initial_guess == "huckel":
            return self._guess_huckel()

        fock_guess = self._h_core.copy()
        fock_orth = self._s_sqrt.T @ fock_guess @ self._s_sqrt
        eigvals, eigvecs = np.linalg.eigh(fock_orth)
        c = self._s_sqrt @ eigvecs
        dm = 2.0 * c[:, :self.nocc] @ c[:, :self.nocc].T
        return dm

    def _guess_sad(self) -> np.ndarray:
        dm = np.zeros((self.nbf, self.nbf))

        shells_by_atom, _ = self.basis.build_molecule_basis(self.mol)

        for i_atom in range(self.mol.natom):
            sym = self.mol.symbols[i_atom]
            atom_shells = shells_by_atom[i_atom]

            z = self.mol.atomic_numbers[i_atom]
            nelec_atom = z
            for j_atom in range(self.mol.natom):
                if j_atom != i_atom:
                    dist = np.linalg.norm(self.mol.coordinates[i_atom] -
                                          self.mol.coordinates[j_atom])
                    nelec_atom -= 0.5 / dist

            nelec_atom = max(0.0, min(float(z), nelec_atom))

            start = 0
            for shell in atom_shells:
                nf = shell.n_functions
                end = start + nf
                dm[start:end, start:end] += (nelec_atom / np.sum([s.n_functions for s in atom_shells])) * np.eye(nf)
                start = end

        return dm

    def _guess_huckel(self) -> np.ndarray:
        if self._s is None:
            self._build_h_core()

        h_core = self._h_core.copy()

        for i in range(self.nbf):
            for j in range(self.nbf):
                if i != j and abs(self._s[i, j]) > 0.1:
                    h_core[i, j] = 0.5 * (h_core[i, i] + h_core[j, j]) * self._s[i, j]

        eigvals, eigvecs = np.linalg.eigh(h_core)
        c = eigvecs
        dm = 2.0 * c[:, :self.nocc] @ c[:, :self.nocc].T
        return dm

    def _build_fock(self, dm: np.ndarray) -> np.ndarray:
        if self.use_ri and self._ri_engine is not None:
            return self._ri_engine.build_fock(self._h_core, dm)

        if self.direct_scf:
            return build_fock_direct(self._h_core, dm, self.basis, self.mol)

        if not hasattr(self, '_eri') or self._eri is None:
            print("  计算双电子积分...")
            self._eri = compute_eri(self.basis, self.mol)

        j = np.einsum('mnls,ls->mn', self._eri, dm)
        k = np.einsum('mlns,ls->mn', self._eri, dm)
        fock = self._h_core + j - 0.5 * k

        if self.level_shift != 0.0 and self._n_iter > 5:
            fock = self._apply_level_shift(fock, dm)

        return fock

    def _apply_level_shift(self, fock: np.ndarray, dm: np.ndarray) -> np.ndarray:
        if self._s is None:
            self._build_h_core()

        fock_orth = self._s_sqrt.T @ fock @ self._s_sqrt
        eigvals, eigvecs = np.linalg.eigh(fock_orth)
        c = self._s_sqrt @ eigvecs

        occ_mask = np.zeros(self.nbf, dtype=bool)
        occ_mask[:self.nocc] = True

        for i in range(self.nbf):
            if not occ_mask[i]:
                eigvals[i] += self.level_shift

        fock_shifted = c @ np.diag(eigvals) @ c.T
        fock_shifted = 0.5 * (fock_shifted + fock_shifted.T)

        return fock_shifted

    def _compute_energy(self, dm: np.ndarray, fock: np.ndarray = None) -> float:
        if fock is None:
            fock = self._fock

        energy = np.einsum('mn,mn->', dm, self._h_core)
        
        if hasattr(self, '_eri') and self._eri is not None:
            j_part = np.einsum('mnls,mn,ls->', self._eri, dm, dm)
            k_part = np.einsum('mlns,mn,ls->', self._eri, dm, dm)
            energy += 0.5 * j_part - 0.25 * k_part
        else:
            energy += 0.5 * np.einsum('mn,mn->', dm, fock - self._h_core)
        
        nuclear_repulsion = 0.0
        for i in range(self.mol.natom):
            for j in range(i + 1, self.mol.natom):
                r = np.linalg.norm(self.mol.coordinates[i] - self.mol.coordinates[j])
                nuclear_repulsion += (self.mol.atomic_numbers[i] *
                                      self.mol.atomic_numbers[j]) / r
        energy += nuclear_repulsion
        return energy

    def _build_density(self, fock: np.ndarray) -> np.ndarray:
        if self._s_sqrt is None:
            self._build_s_sqrt()

        fock_orth = self._s_sqrt.T @ fock @ self._s_sqrt
        eigvals, eigvecs = np.linalg.eigh(fock_orth)
        c = self._s_sqrt @ eigvecs
        dm = 2.0 * c[:, :self.nocc] @ c[:, :self.nocc].T
        return dm, c, eigvals

    def run(self) -> float:
        print(f"\n{'='*60}")
        print(f"  {self.method} SCF 计算")
        print(f"  基组: {self.basis.name}")
        print(f"  原子数: {self.mol.natom}")
        print(f"  电子数: {self.mol.nelectron}")
        print(f"  基函数数: {self.nbf}")
        print(f"  占据轨道: {self.nocc}")
        print(f"{'='*60}")

        self._build_h_core()
        self._build_s_sqrt()

        if self.use_ri:
            print("  初始化 RI-J/K 引擎...")
            self._ri_engine = RIJEngine(self.basis, self.mol)
            self._ri_engine.build()

        self._conv_checker = ConvergenceChecker(
            energy_tol=self.energy_tol,
            density_tol=self.density_tol,
            gradient_tol=self.gradient_tol,
        )

        if self.diis:
            if self.diis_type == "adiis":
                self._diis_obj = ADIIS(max_vec=6)
            elif self.diis_type == "ediis":
                self._diis_obj = EDIIS(max_vec=6)
            else:
                self._diis_obj = DIIS(max_vec=6)

        dm = self._initial_guess()
        self._dm = dm

        adaptive_damping = self.damping
        adaptive_level_shift = self.level_shift

        print("\n  迭代过程:")
        print("  ---------------------------------------------------")
        print(f"  {'迭代':^6} {'能量 (Eh)':^20} {'ΔE':^12} {'密度RMS':^12}")
        print("  ---------------------------------------------------")

        for iteration in range(self.max_iter):
            self._n_iter = iteration + 1

            fock = self._build_fock(dm)
            self._fock = fock

            energy = self._compute_energy(dm, fock)
            self._energy = energy

            dm_new, c, eps = self._build_density(fock)

            delta_dm = np.sqrt(np.sum((dm_new - dm) ** 2)) / dm.size

            if iteration > 0:
                delta_e = abs(energy - self._history[-1]['energy'])
            else:
                delta_e = float('inf')

            print(f"  {iteration+1:6d} {energy:20.12f} {delta_e:12.2e} {delta_dm:12.2e}")

            self._history.append({
                'iteration': iteration + 1,
                'energy': energy,
                'density_rms': delta_dm,
                'fock': fock.copy(),
                'dm': dm.copy(),
            })

            if self.diis and iteration >= 2:
                error = fock @ dm @ self._s - self._s @ dm @ fock
                if isinstance(self._diis_obj, EDIIS):
                    fock = self._diis_obj.update(fock, dm, energy)
                else:
                    fock = self._diis_obj.update(fock, error)
                self._fock = fock

            if self.damping != 0.0 or adaptive_damping != 0.0:
                dm = (1.0 - adaptive_damping) * dm_new + adaptive_damping * dm
            else:
                dm = dm_new

            self._dm = dm

            if iteration >= 10 and delta_e > 1e-5:
                if adaptive_damping < 0.5:
                    adaptive_damping = min(0.5, adaptive_damping + 0.1)
                    print(f"  自适应阻尼增加到: {adaptive_damping:.1f}")

            if iteration >= 20 and delta_e > 1e-4:
                if adaptive_level_shift < 0.3:
                    adaptive_level_shift = min(0.3, adaptive_level_shift + 0.05)
                    self.level_shift = adaptive_level_shift
                    print(f"  自适应 Level Shift 增加到: {adaptive_level_shift:.2f}")

            if iteration >= 5:
                converged = self._conv_checker.check(energy, dm)
                if converged:
                    self._converged = True
                    print(f"\n  SCF 收敛! 迭代次数: {iteration + 1}")
                    break

        if not self._converged:
            print(f"\n  SCF 未收敛! 最大迭代次数: {self.max_iter}")

        final_fock = self._build_fock(dm)
        self._fock = final_fock
        self._energy = self._compute_energy(dm, final_fock)

        _, self._mo_coeff, self._mo_energy = self._build_density(final_fock)

        print(f"\n  {self.method} 总能量: {self._energy:.12f} Eh")
        print(f"{'='*60}")

        return self._energy

    def kernel(self):
        self.run()
        return self._energy

    def get_results(self):
        from ..scf.results import SCFResult
        return SCFResult(
            energy=self._energy,
            converged=self._converged,
            n_cycles=self._n_iter,
            mo_energies=None,
            mo_coefficients=None,
            mo_occupations=None,
            dm=self._dm,
            fock=self._fock,
        )


class UHF(HF):
    def __init__(self, mol: Molecule, basis: BasisSet, **kwargs):
        kwargs['method'] = "UHF"
        super().__init__(mol, basis, **kwargs)
        self._nocc_beta = None
        self._dm_a = None
        self._dm_b = None
        self._fock_a = None
        self._fock_b = None

    @property
    def mo_occ(self) -> tuple:
        """UHF 轨道占据数: 返回 (occ_a, occ_b) 元组。"""
        if self._mo_coeff is None:
            return None
        occ_a = np.zeros(self.nbf)
        occ_a[:self.nocc] = 1.0
        occ_b = np.zeros(self.nbf)
        occ_b[:self._nocc_beta] = 1.0
        return (occ_a, occ_b)

    @property
    def fock(self) -> np.ndarray:
        return self._fock_a

    def _setup_basis(self):
        super()._setup_basis()
        if self.method == "UHF":
            self._nocc_beta = self.mol.nelectron - self._nocc

    def _initial_guess(self) -> tuple[np.ndarray, np.ndarray]:
        if self._s is None:
            self._build_h_core()
        if self._s_sqrt is None:
            self._build_s_sqrt()

        fock_guess = self._h_core.copy()
        fock_orth = self._s_sqrt.T @ fock_guess @ self._s_sqrt
        eigvals, eigvecs = np.linalg.eigh(fock_orth)
        c = self._s_sqrt @ eigvecs

        dm_a = c[:, :self.nocc] @ c[:, :self.nocc].T
        dm_b = c[:, :self._nocc_beta] @ c[:, :self._nocc_beta].T

        return dm_a, dm_b

    def _build_fock(self, dm_a: np.ndarray, dm_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.use_ri and self._ri_engine is not None:
            j_a = self._ri_engine.build_j(dm_a)
            j_b = self._ri_engine.build_j(dm_b)
            k_a = self._ri_engine.build_k(dm_a)
            k_b = self._ri_engine.build_k(dm_b)
            f_a = self._h_core + j_a + j_b - k_a
            f_b = self._h_core + j_a + j_b - k_b
            return f_a, f_b

        if self.direct_scf:
            return build_fock_direct_uhf(self._h_core, dm_a, dm_b, self.basis, self.mol)

        if not hasattr(self, '_eri') or self._eri is None:
            print("  计算双电子积分...")
            self._eri = compute_eri(self.basis, self.mol)

        j_a = np.einsum('mnls,ls->mn', self._eri, dm_a)
        j_b = np.einsum('mnls,ls->mn', self._eri, dm_b)
        k_a = np.einsum('mlns,ls->mn', self._eri, dm_a)
        k_b = np.einsum('mlns,ls->mn', self._eri, dm_b)

        f_a = self._h_core + j_a + j_b - k_a
        f_b = self._h_core + j_a + j_b - k_b

        return f_a, f_b

    def _compute_energy(self, dm_a: np.ndarray, dm_b: np.ndarray,
                        f_a: np.ndarray = None, f_b: np.ndarray = None) -> float:
        if f_a is None:
            f_a = self._fock_a
        if f_b is None:
            f_b = self._fock_b

        energy = 0.5 * np.einsum('mn,mn->', dm_a + dm_b, self._h_core)
        energy += 0.5 * np.einsum('mn,mn->', dm_a, f_a) + 0.5 * np.einsum('mn,mn->', dm_b, f_b)

        nuclear_repulsion = 0.0
        for i in range(self.mol.natom):
            for j in range(i + 1, self.mol.natom):
                r = np.linalg.norm(self.mol.coordinates[i] - self.mol.coordinates[j])
                nuclear_repulsion += (self.mol.atomic_numbers[i] *
                                      self.mol.atomic_numbers[j]) / r
        energy += nuclear_repulsion

        return energy

    def _build_density(self, fock: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._s_sqrt is None:
            self._build_s_sqrt()

        fock_orth = self._s_sqrt.T @ fock @ self._s_sqrt
        eigvals, eigvecs = np.linalg.eigh(fock_orth)
        c = self._s_sqrt @ eigvecs
        dm = c[:, :self.nocc] @ c[:, :self.nocc].T
        return dm, c, eigvals

    def run(self) -> float:
        print(f"\n{'='*60}")
        print(f"  UHF SCF 计算")
        print(f"  基组: {self.basis.name}")
        print(f"  原子数: {self.mol.natom}")
        print(f"  α电子: {self.nocc}, β电子: {self._nocc_beta}")
        print(f"  基函数数: {self.nbf}")
        print(f"{'='*60}")

        self._build_h_core()
        self._build_s_sqrt()

        if self.use_ri:
            print("  初始化 RI-J/K 引擎...")
            self._ri_engine = RIJEngine(self.basis, self.mol)
            self._ri_engine.build()

        dm_a, dm_b = self._initial_guess()
        self._dm_a = dm_a
        self._dm_b = dm_b

        print("\n  迭代过程:")
        print("  ---------------------------------------------------")
        print(f"  {'迭代':^6} {'能量 (Eh)':^20} {'ΔE':^12}")
        print("  ---------------------------------------------------")

        for iteration in range(self.max_iter):
            self._n_iter = iteration + 1

            f_a, f_b = self._build_fock(dm_a, dm_b)
            self._fock_a = f_a
            self._fock_b = f_b

            energy = self._compute_energy(dm_a, dm_b, f_a, f_b)
            self._energy = energy

            dm_a_new, c_a, eps_a = self._build_density(f_a)
            dm_b_new, c_b, eps_b = self._build_density(f_b)

            if iteration > 0:
                delta_e = abs(energy - self._history[-1]['energy'])
            else:
                delta_e = float('inf')

            print(f"  {iteration+1:6d} {energy:20.12f} {delta_e:12.2e}")

            self._history.append({
                'iteration': iteration + 1,
                'energy': energy,
                'fock_a': f_a.copy(),
                'fock_b': f_b.copy(),
                'dm_a': dm_a.copy(),
                'dm_b': dm_b.copy(),
            })

            if self.damping != 0.0:
                dm_a = (1.0 - self.damping) * dm_a_new + self.damping * dm_a
                dm_b = (1.0 - self.damping) * dm_b_new + self.damping * dm_b
            else:
                dm_a = dm_a_new
                dm_b = dm_b_new

            self._dm_a = dm_a
            self._dm_b = dm_b

            if iteration >= 5:
                if delta_e < self.energy_tol:
                    self._converged = True
                    print(f"\n  SCF 收敛! 迭代次数: {iteration + 1}")
                    break

        if not self._converged:
            print(f"\n  SCF 未收敛! 最大迭代次数: {self.max_iter}")

        final_f_a, final_f_b = self._build_fock(dm_a, dm_b)
        self._fock_a = final_f_a
        self._fock_b = final_f_b
        self._energy = self._compute_energy(dm_a, dm_b, final_f_a, final_f_b)

        # 保存 MO 系数和能量供后HF方法使用
        _, c_a, eps_a = self._build_density(final_f_a)
        _, c_b, eps_b = self._build_density(final_f_b)
        self._mo_coeff = (c_a, c_b)
        self._mo_energy = (eps_a, eps_b)

        print(f"\n  UHF 总能量: {self._energy:.12f} Eh")
        print(f"{'='*60}")

        return self._energy

    def kernel(self):
        self.run()
        return self._energy

    def get_results(self):
        from ..scf.results import SCFResult
        return SCFResult(
            energy=self._energy,
            converged=self._converged,
            n_cycles=self._n_iter,
            mo_energies=None,
            mo_coefficients=None,
            mo_occupations=None,
            dm=self._dm_a + self._dm_b,
            fock=self._fock_a,
        )


class RHF(HF):
    pass


class HFMethod(HF):
    pass


def nuclear_repulsion(mol) -> float:
    energy = 0.0
    for i in range(mol.natom):
        for j in range(i + 1, mol.natom):
            r = np.linalg.norm(mol.coordinates[i] - mol.coordinates[j])
            energy += (mol.atomic_numbers[i] * mol.atomic_numbers[j]) / r
    return energy