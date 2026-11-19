"""
分子多极矩性质计算（自研实现）。

支持：
- 偶极矩 (dipole moment)
- 四极矩 (quadrupole moment)
- Mulliken布居分析
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np

from ..utils.logging_utils import get_logger
from ..utils.constants import ATOMIC_NUMBERS

logger = get_logger(__name__)


class MultipoleMoments:
    """
    分子多极矩计算。
    
    计算分子的偶极矩、四极矩等性质。
    """
    
    def __init__(self, scf_method):
        """
        Parameters
        ----------
        scf_method : RHF, UHF, RKS, UKS
            收敛的SCF方法对象
        """
        self.method = scf_method
        self.dipole: Optional[np.ndarray] = None
        self.quadrupole: Optional[np.ndarray] = None
        self.dipole_magnitude: float = 0.0
    
    def _get_density_matrix(self):
        """获取密度矩阵。"""
        mf = self.method
        if hasattr(mf, 'dm'):
            dm = mf.dm
            if isinstance(dm, tuple):
                return dm[0] + dm[1]
            return dm
        if hasattr(mf, 'mo_coeff') and hasattr(mf, 'mo_occ'):
            c = mf.mo_coeff
            occ = mf.mo_occ
            if isinstance(c, tuple):
                dm_a = c[0] @ np.diag(occ[0]) @ c[0].T
                dm_b = c[1] @ np.diag(occ[1]) @ c[1].T
                return dm_a + dm_b
            return c @ np.diag(occ) @ c.T
        raise ValueError("Cannot get density matrix")
    
    def _get_basis_and_mol(self):
        """获取基组和分子对象。"""
        mf = self.method
        if hasattr(mf, 'basis_set'):
            return mf.basis_set, mf.mol
        raise ValueError("Cannot get basis set")
    
    def calculate_dipole(self) -> np.ndarray:
        """
        计算分子偶极矩。
        
        μ = Σ_A Z_A R_A - Σ_μν D_μν <μ|r|ν>
        
        Returns
        -------
        np.ndarray
            偶极矩向量 (3,)，原子单位
        """
        basis, mol = self._get_basis_and_mol()
        dm = self._get_density_matrix()
        
        from ..integrals.one_electron import compute_dipole
        mu_ao = compute_dipole(basis, mol)
        
        mu_elec = np.zeros(3)
        for d in range(3):
            mu_elec[d] = np.einsum('uv,uv->', dm, mu_ao[d])
        
        mu_nuc = np.zeros(3)
        for i in range(mol.natom):
            z = float(ATOMIC_NUMBERS.get(mol.symbols[i], 0))
            mu_nuc += z * mol.coordinates[i]
        
        self.dipole = mu_nuc - mu_elec
        self.dipole_magnitude = float(np.linalg.norm(self.dipole))
        
        logger.info(f"Dipole moment: {self.dipole_magnitude:.6f} a.u. = {self.dipole_magnitude * 2.541746:.6f} D")
        
        return self.dipole
    
    def calculate_quadrupole(self) -> np.ndarray:
        """
        计算分子四极矩（迹为零的形式）。
        
        Θ_ij = Σ_A Z_A (3 R_Ai R_Aj - R_A^2 δ_ij) 
               - Σ_μν D_μν <μ| 3r_i r_j - r^2 δ_ij |ν>
        
        Returns
        -------
        np.ndarray
            四极矩矩阵 (3, 3)，原子单位
        """
        basis, mol = self._get_basis_and_mol()
        dm = self._get_density_matrix()
        
        theta = np.zeros((3, 3))
        
        shells_by_atom, nbf = basis.build_molecule_basis(mol)
        
        all_shells = []
        for atom_shells in shells_by_atom:
            all_shells.extend(atom_shells)
        
        shell_range = []
        idx = 0
        for s in all_shells:
            nf = s.n_functions
            shell_range.append((idx, idx + nf))
            idx += nf
        
        from ..integrals.one_electron import _os_recurrence_overlap
        
        def quad_integral(l1, m1, n1, l2, m2, n2, alpha_a, alpha_b, r_a, r_b, i, j):
            """计算 <a| 3 r_i r_j - r^2 δ_ij |b>。"""
            zeta = alpha_a + alpha_b
            r_p = (alpha_a * r_a + alpha_b * r_b) / zeta
            
            r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
            K = np.exp(-alpha_a * alpha_b / zeta * r_ab2)
            
            def s(l, m, n, l2, m2, n2):
                if l < 0 or m < 0 or n < 0 or l2 < 0 or m2 < 0 or n2 < 0:
                    return 0.0
                return _os_recurrence_overlap(
                    l, m, n, l2, m2, n2,
                    alpha_a, alpha_b, r_a, r_b
                )
            
            s0 = s(l1, m1, n1, l2, m2, n2)
            
            def r_component(l1, m1, n1, l2, m2, n2, direction):
                result = r_p[direction] * s(l1, m1, n1, l2, m2, n2)
                if direction == 0:
                    if l1 > 0:
                        result += l1 / (2.0 * zeta) * s(l1 - 1, m1, n1, l2, m2, n2)
                    if l2 > 0:
                        result += l2 / (2.0 * zeta) * s(l1, m1, n1, l2 - 1, m2, n2)
                elif direction == 1:
                    if m1 > 0:
                        result += m1 / (2.0 * zeta) * s(l1, m1 - 1, n1, l2, m2, n2)
                    if m2 > 0:
                        result += m2 / (2.0 * zeta) * s(l1, m1, n1, l2, m2 - 1, n2)
                else:
                    if n1 > 0:
                        result += n1 / (2.0 * zeta) * s(l1, m1, n1 - 1, l2, m2, n2)
                    if n2 > 0:
                        result += n2 / (2.0 * zeta) * s(l1, m1, n1, l2, m2, n2 - 1)
                return result
            
            def rr_component(l1, m1, n1, l2, m2, n2, d1, d2):
                """<a| r_d1 r_d2 |b>"""
                result = r_p[d1] * r_component(l1, m1, n1, l2, m2, n2, d2)
                
                if d1 == 0:
                    if l1 > 0:
                        result += l1 / (2.0 * zeta) * r_component(l1 - 1, m1, n1, l2, m2, n2, d2)
                    if l2 > 0:
                        result += l2 / (2.0 * zeta) * r_component(l1, m1, n1, l2 - 1, m2, n2, d2)
                elif d1 == 1:
                    if m1 > 0:
                        result += m1 / (2.0 * zeta) * r_component(l1, m1 - 1, n1, l2, m2, n2, d2)
                    if m2 > 0:
                        result += m2 / (2.0 * zeta) * r_component(l1, m1, n1, l2, m2 - 1, n2, d2)
                else:
                    if n1 > 0:
                        result += n1 / (2.0 * zeta) * r_component(l1, m1, n1 - 1, l2, m2, n2, d2)
                    if n2 > 0:
                        result += n2 / (2.0 * zeta) * r_component(l1, m1, n1, l2, m2, n2 - 1, d2)
                
                if d1 == d2:
                    result += 1.0 / (2.0 * zeta) * s0
                
                return result
            
            r2 = rr_component(l1, m1, n1, l2, m2, n2, 0, 0) + \
                 rr_component(l1, m1, n1, l2, m2, n2, 1, 1) + \
                 rr_component(l1, m1, n1, l2, m2, n2, 2, 2)
            
            val = 3.0 * rr_component(l1, m1, n1, l2, m2, n2, i, j)
            if i == j:
                val -= r2
            
            return val
        
        nshell = len(all_shells)
        theta_ao = np.zeros((3, 3, nbf, nbf))
        
        for i_dir in range(3):
            for j_dir in range(3):
                for si_idx in range(nshell):
                    si, ei = shell_range[si_idx]
                    shell_i = all_shells[si_idx]
                    for sj_idx in range(nshell):
                        sj, ej = shell_range[sj_idx]
                        shell_j = all_shells[sj_idx]
                        
                        for fi in range(shell_i.n_functions):
                            li, mi, ni = shell_i.angular_combinations[fi]
                            for fj in range(shell_j.n_functions):
                                lj, mj, nj = shell_j.angular_combinations[fj]
                                
                                val = 0.0
                                for pi in range(shell_i.n_primitives):
                                    ci = shell_i._contract_coeffs[pi]
                                    ai = shell_i.exponents[pi]
                                    norm_i = shell_i._primitives[fi][pi].norm
                                    for pj in range(shell_j.n_primitives):
                                        cj = shell_j._contract_coeffs[pj]
                                        aj = shell_j.exponents[pj]
                                        norm_j = shell_j._primitives[fj][pj].norm
                                        
                                        val += ci * cj * norm_i * norm_j * quad_integral(
                                            li, mi, ni, lj, mj, nj,
                                            ai, aj,
                                            shell_i.origin, shell_j.origin,
                                            i_dir, j_dir
                                        )
                                
                                theta_ao[i_dir, j_dir, si + fi, sj + fj] = val
        
        theta_elec = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                theta_elec[i, j] = np.einsum('uv,uv->', dm, theta_ao[i, j])
        
        theta_nuc = np.zeros((3, 3))
        for a in range(mol.natom):
            z = float(ATOMIC_NUMBERS.get(mol.symbols[a], 0))
            R = mol.coordinates[a]
            R2 = np.dot(R, R)
            for i in range(3):
                for j in range(3):
                    theta_nuc[i, j] += z * (3 * R[i] * R[j] - R2 * (1.0 if i == j else 0.0))
        
        self.quadrupole = theta_nuc - theta_elec
        return self.quadrupole
    
    def calculate(self) -> Dict[str, Any]:
        """
        计算所有多极矩。
        
        Returns
        -------
        dict
            包含偶极矩、四极矩等结果的字典
        """
        self.calculate_dipole()
        
        results = {
            'dipole': self.dipole,
            'dipole_magnitude': self.dipole_magnitude,
            'dipole_debye': self.dipole_magnitude * 2.541746,
        }
        
        try:
            self.calculate_quadrupole()
            results['quadrupole'] = self.quadrupole
        except Exception as e:
            logger.warning(f"Could not compute quadrupole: {e}")
        
        return results
