"""
PCM (Polarizable Continuum Model) 溶剂模型模块。

实现基于IEF-PCM的隐式溶剂化能计算。
"""

from __future__ import annotations

import numpy as np

from ..mol.molecule import Molecule
from ..basis.basis_set import BasisSet


class PCMSolver:
    """IEF-PCM 溶剂模型求解器。"""

    def __init__(self, mol: Molecule, basis_set: BasisSet, solvent: str = "water",
                 cavity_type: str = "gepol", scaling_factor: float = 1.1):
        self.mol = mol
        self.basis_set = basis_set
        self.solvent = solvent.lower()
        self.cavity_type = cavity_type.lower()
        self.scaling_factor = scaling_factor

        self.solvent_params = self._get_solvent_params()
        self.cavity_radii = self._compute_cavity_radii()
        self.epsilon = self.solvent_params['epsilon']
        self.refractive_index = self.solvent_params['refractive_index']

        self._born_radii = None
        self._green_function = None
        self._pcm_matrix = None

    def _get_solvent_params(self) -> dict:
        solvent_data = {
            'water': {'epsilon': 78.3553, 'refractive_index': 1.3330},
            'ethanol': {'epsilon': 24.55, 'refractive_index': 1.3611},
            'methanol': {'epsilon': 32.61, 'refractive_index': 1.3288},
            'acetonitrile': {'epsilon': 37.5, 'refractive_index': 1.3441},
            'chloroform': {'epsilon': 4.806, 'refractive_index': 1.4459},
            'benzene': {'epsilon': 2.2706, 'refractive_index': 1.5011},
            'diethyl_ether': {'epsilon': 4.24, 'refractive_index': 1.3526},
            'hexane': {'epsilon': 1.88, 'refractive_index': 1.3749},
            'acetone': {'epsilon': 20.7, 'refractive_index': 1.3586},
            'dichloromethane': {'epsilon': 8.93, 'refractive_index': 1.4244},
        }
        return solvent_data.get(self.solvent, {'epsilon': 78.3553, 'refractive_index': 1.3330})

    def _compute_cavity_radii(self) -> np.ndarray:
        vdw_radii = {
            'H': 1.2, 'He': 1.4, 'Li': 1.82, 'Be': 1.53, 'B': 1.92,
            'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
            'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
            'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31,
        }
        radii = []
        for symbol in self.mol.symbols:
            radii.append(vdw_radii.get(symbol, 1.70) * self.scaling_factor)
        return np.array(radii)

    def compute_born_radii(self) -> np.ndarray:
        """计算Born半径。"""
        natom = self.mol.natom
        coords = self.mol.coordinates
        born_radii = np.zeros(natom)

        for i in range(natom):
            r_i = self.cavity_radii[i]
            sum_over_j = 0.0
            for j in range(natom):
                if i != j:
                    r_ij = np.linalg.norm(coords[i] - coords[j])
                    r_j = self.cavity_radii[j]
                    if r_ij > r_i + r_j:
                        sum_over_j += r_j ** 2 / (r_ij ** 3)
            born_radii[i] = 1.0 / (1.0 / r_i - sum_over_j)

        self._born_radii = born_radii
        return born_radii

    def build_pcm_matrix(self) -> np.ndarray:
        """构建PCM矩阵。"""
        nbf = self._get_nbf()
        pcm_matrix = np.zeros((nbf, nbf))

        born_radii = self.compute_born_radii()
        coords = self.mol.coordinates

        all_shells = self.basis_set.get_all_shells(self.mol)
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
                block = self._compute_pcm_block(shell_i, shell_j, coords, born_radii)
                pcm_matrix[si:ei, sj:ej] += block

        self._pcm_matrix = pcm_matrix
        return pcm_matrix

    def _get_nbf(self) -> int:
        all_shells = self.basis_set.get_all_shells(self.mol)
        return sum(s.n_functions for s in all_shells)

    def _compute_pcm_block(self, shell_a, shell_b, coords, born_radii):
        n_func_a = shell_a.n_functions
        n_func_b = shell_b.n_functions
        result = np.zeros((n_func_a, n_func_b))

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
                        result[i, j] += prefactor * K * self._pcm_integral(
                            la, ma, na, lb, mb, nb, alpha_a, alpha_b, r_a, r_b, r_p, coords, born_radii
                        )

        return result

    def _pcm_integral(self, la, ma, na, lb, mb, nb, alpha_a, alpha_b, r_a, r_b, r_p, coords, born_radii):
        screening_factor = self._compute_screening(r_p, coords, born_radii)
        return screening_factor

    def _compute_screening(self, r_p, coords, born_radii):
        screening = 0.0
        for i, r_i in enumerate(coords):
            r = np.linalg.norm(r_p - r_i)
            if r > born_radii[i]:
                screening += (1.0 - self.epsilon) / (r ** 3 * self.epsilon)
        return screening

    def compute_solvation_energy(self, dm: np.ndarray) -> float:
        """计算溶剂化能。"""
        if self._pcm_matrix is None:
            self.build_pcm_matrix()

        e_solv = 0.5 * np.einsum("mn,mn->", dm, self._pcm_matrix)
        return e_solv

    def compute_vpc(self, dm: np.ndarray) -> np.ndarray:
        """计算PCM势能矩阵。"""
        if self._pcm_matrix is None:
            self.build_pcm_matrix()

        vpc = np.einsum("ls,mnls->mn", dm, self._pcm_matrix)
        return vpc

    def kernel(self, dm: np.ndarray) -> float:
        """执行PCM计算。"""
        return self.compute_solvation_energy(dm)


class PCM:
    """PCM溶剂模型接口。"""

    def __init__(self, mf, solvent: str = "water", cavity_type: str = "gepol"):
        self.mf = mf
        self.mol = mf.mol
        self.basis_set = mf.basis_set
        self.solvent = solvent
        self.cavity_type = cavity_type
        self.solver = None

    def build(self):
        """构建PCM求解器。"""
        self.solver = PCMSolver(self.mol, self.basis_set, self.solvent, self.cavity_type)
        self.solver.build_pcm_matrix()

    def compute_solvation_energy(self) -> float:
        """计算溶剂化能。"""
        if self.solver is None:
            self.build()
        return self.solver.compute_solvation_energy(self.mf.dm)

    def kernel(self) -> float:
        """执行PCM计算。"""
        return self.compute_solvation_energy()


def pcm(mf, solvent: str = "water") -> PCM:
    """PCM计算接口。"""
    pcm_obj = PCM(mf, solvent)
    pcm_obj.kernel()
    return pcm_obj