"""
TDDFT (Time-Dependent Density Functional Theory) 激发态方法。

实现 TDA (Tamm-Dancoff Approximation) 和 full TDDFT (Casida equation)。
支持 RKS 和 RHF 参考态。

参考:
- Casida, in Recent Advances in Density Functional Methods (1995)
- Bauernschmitt & Ahlrichs, Chem. Phys. Lett. 256, 454 (1996)
- Stratmann et al., J. Chem. Phys. 109, 8218 (1998)
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TDA:
    """
    Tamm-Dancoff Approximation (TDA) 激发态方法。
    
    求解A矩阵的本征值问题: A * X = ω * X
    比full TDDFT更便宜，精度通常接近。
    """
    
    def __init__(self, scf_method, nstates: int = 10, singlet: bool = True):
        """
        Parameters
        ----------
        scf_method : RKS or RHF
            收敛的SCF方法对象
        nstates : int
            计算的激发态数目
        singlet : bool
            是否计算单重态（仅适用于闭壳层）
        """
        self.scf = scf_method
        self.nstates = nstates
        self.singlet = singlet
        
        self.excitation_energies: Optional[np.ndarray] = None
        self.oscillator_strengths: Optional[np.ndarray] = None
        self.transition_dipoles: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.converged: bool = False
    
    def _get_mo_info(self) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
        """获取分子轨道信息。"""
        mf = self.scf
        e = mf.mo_energy
        c = mf.mo_coeff
        nocc = mf.mol.nelectron // 2
        nbf = len(e)
        nvir = nbf - nocc
        return e, c, nocc, nvir, mf._hcore if hasattr(mf, '_hcore') else None
    
    def _get_eri_mo(self, c: np.ndarray, nocc: int, nvir: int) -> np.ndarray:
        """计算MO基的双电子积分 (ia|jb)。"""
        mf = self.scf
        nbf = c.shape[0]
        
        if hasattr(mf, '_eri') and mf._eri is not None:
            eri = mf._eri
        else:
            from ..integrals.two_electron import compute_eri
            eri = compute_eri(mf.basis_set, mf.mol)
        
        c_occ = c[:, :nocc]
        c_vir = c[:, nocc:]
        
        # Transform (mn|ls) -> (ia|jb)
        # First transform ls -> jb
        eri_1 = np.einsum("mnls,lj->mnjs", eri, c_occ)
        eri_2 = np.einsum("mnjs,sb->mnjb", eri_1, c_vir)
        # Then transform mn -> ia
        eri_3 = np.einsum("mnjb,mi->injb", eri_2, c_occ)
        eri_mo = np.einsum("injb,na->iajb", eri_3, c_vir)
        
        return eri_mo
    
    def _build_A_matrix(
        self, e: np.ndarray, nocc: int, nvir: int, eri_mo: np.ndarray
    ) -> np.ndarray:
        """
        构建TDA A矩阵。
        
        A_ia,jb = δ_ij δ_ab (ε_a - ε_i) + 2(ia|jb) - (ij|ab)
        
        对于纯DFT（无精确交换），这是正确的。
        对于混合泛函，需要添加f_xc项和exx修正，这里先简化为TDA形式。
        """
        e_occ = e[:nocc]
        e_vir = e[nocc:]
        
        n_singles = nocc * nvir
        A = np.zeros((n_singles, n_singles))
        
        for i in range(nocc):
            for a in range(nvir):
                ia = i * nvir + a
                for j in range(nocc):
                    for b in range(nvir):
                        jb = j * nvir + b
                        
                        delta_ij = 1.0 if i == j else 0.0
                        delta_ab = 1.0 if a == b else 0.0
                        
                        A[ia, jb] = delta_ij * delta_ab * (e_vir[a] - e_occ[i])
                        A[ia, jb] += 2.0 * eri_mo[i, a, j, b] - eri_mo[i, j, a, b]
        
        return A
    
    def _compute_transition_dipoles(
        self, c: np.ndarray, nocc: int, nvir: int, X: np.ndarray
    ) -> np.ndarray:
        """计算跃迁偶极矩。"""
        mf = self.scf
        from ..integrals.one_electron import compute_dipole
        mu = compute_dipole(mf.basis_set, mf.mol)
        
        c_occ = c[:, :nocc]
        c_vir = c[:, nocc:]
        
        nstates = X.shape[1]
        transition_dipoles = np.zeros((nstates, 3))
        
        for s in range(nstates):
            X_s = X[:, s].reshape(nocc, nvir)
            
            for d in range(3):
                mu_ia = c_occ.T @ mu[d] @ c_vir
                transition_dipoles[s, d] = np.sum(X_s * mu_ia)
        
        return transition_dipoles
    
    def _compute_oscillator_strengths(
        self, energies: np.ndarray, transition_dipoles: np.ndarray
    ) -> np.ndarray:
        """计算振子强度。
        
        f = (2/3) * ω * |μ|^2  (在原子单位中)
        """
        f = np.zeros(len(energies))
        for i in range(len(energies)):
            f[i] = (2.0 / 3.0) * energies[i] * np.sum(transition_dipoles[i] ** 2)
        return f
    
    def kernel(self) -> np.ndarray:
        """
        运行TDA计算。
        
        Returns
        -------
        np.ndarray
            激发能数组 (Eh)
        """
        logger.info("Starting TDA calculation...")
        
        e, c, nocc, nvir, _ = self._get_mo_info()
        
        n_states = min(self.nstates, nocc * nvir)
        if n_states < self.nstates:
            logger.warning(f"Requested {self.nstates} states but only {n_states} available")
        
        logger.info(f"  nocc = {nocc}, nvir = {nvir}, n_singles = {nocc * nvir}")
        
        logger.info("  Transforming ERI to MO basis...")
        eri_mo = self._get_eri_mo(c, nocc, nvir)
        
        logger.info("  Building A matrix...")
        A = self._build_A_matrix(e, nocc, nvir, eri_mo)
        
        logger.info(f"  Diagonalizing A matrix ({A.shape[0]}x{A.shape[0]})...")
        eigvals, eigvecs = np.linalg.eigh(A)
        
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        pos_idx = eigvals > 0
        eigvals = eigvals[pos_idx]
        eigvecs = eigvecs[:, pos_idx]
        
        n_states = min(n_states, len(eigvals))
        self.excitation_energies = eigvals[:n_states]
        self.eigenvectors = eigvecs[:, :n_states]
        
        try:
            logger.info("  Computing transition dipoles...")
            self.transition_dipoles = self._compute_transition_dipoles(
                c, nocc, nvir, self.eigenvectors
            )
            logger.info("  Computing oscillator strengths...")
            self.oscillator_strengths = self._compute_oscillator_strengths(
                self.excitation_energies, self.transition_dipoles
            )
        except Exception as e:
            logger.warning(f"  Could not compute oscillator strengths: {e}")
            self.transition_dipoles = None
            self.oscillator_strengths = None
        
        self.converged = True
        
        logger.info(f"TDA converged. Lowest excitation: {self.excitation_energies[0]:.6f} Eh")
        for i in range(min(n_states, 5)):
            if self.oscillator_strengths is not None and i < len(self.oscillator_strengths):
                logger.info(f"  S{i+1}: {self.excitation_energies[i]:.6f} Eh, f = {self.oscillator_strengths[i]:.6f}")
            else:
                logger.info(f"  S{i+1}: {self.excitation_energies[i]:.6f} Eh")
        
        return self.excitation_energies


class TDDFT(TDA):
    """
    Full TDDFT (Casida equation) 激发态方法。
    
    求解本征值问题:
    ( A  B ) ( X )       ( 1   0 ) ( X )
    ( B  A ) ( Y ) = ω ( 0  -1 ) ( Y )
    
    或者等价的对称形式:
    ( A+B   A-B ) ( X+Y )     ( X+Y )
    ( A-B   A+B ) ( X-Y ) = ω ( X-Y )
    """
    
    def _build_B_matrix(
        self, e: np.ndarray, nocc: int, nvir: int, eri_mo: np.ndarray
    ) -> np.ndarray:
        """
        构建TDDFT B矩阵。
        
        B_ia,jb = 2(ia|bj) - (ib|aj)
        """
        n_singles = nocc * nvir
        B = np.zeros((n_singles, n_singles))
        
        for i in range(nocc):
            for a in range(nvir):
                ia = i * nvir + a
                for j in range(nocc):
                    for b in range(nvir):
                        jb = j * nvir + b
                        
                        # 2(ia|bj) - (ib|aj)
                        B[ia, jb] = 2.0 * eri_mo[i, a, j, b] - eri_mo[i, j, b, a]
        
        return B
    
    def kernel(self) -> np.ndarray:
        """
        运行TDDFT计算。
        
        Returns
        -------
        np.ndarray
            激发能数组 (Eh)
        """
        logger.info("Starting TDDFT calculation...")
        
        e, c, nocc, nvir, _ = self._get_mo_info()
        
        n_states = min(self.nstates, nocc * nvir)
        if n_states < self.nstates:
            logger.warning(f"Requested {self.nstates} states but only {n_states} available")
        
        logger.info(f"  nocc = {nocc}, nvir = {nvir}, n_singles = {nocc * nvir}")
        
        logger.info("  Transforming ERI to MO basis...")
        eri_mo = self._get_eri_mo(c, nocc, nvir)
        
        logger.info("  Building A matrix...")
        A = self._build_A_matrix(e, nocc, nvir, eri_mo)
        
        logger.info("  Building B matrix...")
        B = self._build_B_matrix(e, nocc, nvir, eri_mo)
        
        logger.info("  Forming TDDFT eigenvalue problem...")
        # Use the symmetric form:
        # M1 = A + B, M2 = A - B
        # Then the problem is:
        # ( M1  M2 ) ( X+Y )     ( X+Y )
        # ( M2  M1 ) ( X-Y ) = ω ( X-Y )
        #
        # Which can be solved via:
        # eigvals of (A+B)(A-B) = ω^2
        
        M_plus = A + B
        M_minus = A - B
        
        # Solve (A+B)(A-B) Z = ω^2 Z
        # Alternatively, solve the generalized eigenvalue problem
        # For stability, use:
        # C = M_minus^(1/2) M_plus M_minus^(1/2)
        # But simpler: just compute eigenvalues of M_plus @ M_minus
        
        prod = M_plus @ M_minus
        
        logger.info(f"  Diagonalizing product matrix ({prod.shape[0]}x{prod.shape[0]})...")
        eigvals_sq, eigvecs_prod = np.linalg.eigh(prod)
        
        # ω = sqrt(eigval), but we need to handle signs
        # Actually, use the standard approach:
        # Solve |A - B|^(-1/2) (A+B) |A - B|^(-1/2) symmetrically
        
        # Let's use a more stable approach:
        # Form the 2N x 2N matrix and diagonalize
        N = A.shape[0]
        full_mat = np.zeros((2 * N, 2 * N))
        full_mat[:N, :N] = A
        full_mat[:N, N:] = B
        full_mat[N:, :N] = -B
        full_mat[N:, N:] = -A
        
        # This is not symmetric, use eig instead of eigh
        logger.info("  Solving full TDDFT eigenvalue problem...")
        eigvals_full, eigvecs_full = np.linalg.eig(full_mat)
        
        # Filter positive real eigenvalues
        pos_real = np.isreal(eigvals_full) & (eigvals_full.real > 1e-10)
        energies = eigvals_full[pos_real].real
        vecs = eigvecs_full[:, pos_real].real
        
        # Sort by energy
        idx = energies.argsort()
        energies = energies[idx]
        vecs = vecs[:, idx]
        
        n_states = min(n_states, len(energies))
        self.excitation_energies = energies[:n_states]
        
        # Extract X vector (upper half)
        X_full = vecs[:N, :n_states]
        
        # Normalize X vectors
        for i in range(n_states):
            norm = np.linalg.norm(X_full[:, i])
            if norm > 1e-10:
                X_full[:, i] /= norm
        
        self.eigenvectors = X_full
        
        try:
            logger.info("  Computing transition dipoles...")
            self.transition_dipoles = self._compute_transition_dipoles(
                c, nocc, nvir, self.eigenvectors
            )
            logger.info("  Computing oscillator strengths...")
            self.oscillator_strengths = self._compute_oscillator_strengths(
                self.excitation_energies, self.transition_dipoles
            )
        except Exception as e:
            logger.warning(f"  Could not compute oscillator strengths: {e}")
            self.transition_dipoles = None
            self.oscillator_strengths = None
        
        self.converged = True
        
        logger.info(f"TDDFT converged. Lowest excitation: {self.excitation_energies[0]:.6f} Eh")
        for i in range(min(n_states, 5)):
            if self.oscillator_strengths is not None and i < len(self.oscillator_strengths):
                logger.info(f"  S{i+1}: {self.excitation_energies[i]:.6f} Eh, f = {self.oscillator_strengths[i]:.6f}")
            else:
                logger.info(f"  S{i+1}: {self.excitation_energies[i]:.6f} Eh")
        
        return self.excitation_energies
