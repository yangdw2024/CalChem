"""
RI (Resolution of Identity) 密度拟合模块。

实现 RI-J (库仑拟合) 和 RI-K (交换拟合)，
以及 RI-MP2 加速。

参考:
- Whitten, J. Chem. Phys. 58, 4496 (1973)
- Dunlap et al., J. Chem. Phys. 69, 3928 (1978)
- Weigend & Ahlrichs, PCCP 7, 3297 (2005)
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np

from ..basis.gto import GTOShell
from ..basis.basis_set import BasisSet
from ..mol.molecule import Molecule
from .one_electron import boys_function_vec


def _gaussian_product(alpha_a, alpha_b, r_a, r_b):
    gamma = alpha_a + alpha_b
    r_p = (alpha_a * r_a + alpha_b * r_b) / gamma
    r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
    K = math.exp(-alpha_a * alpha_b / gamma * r_ab2)
    return gamma, r_p, K


def build_aux_basis(orb_basis: BasisSet, mol: Molecule) -> Dict:
    aux_basis_data = {}

    shells_by_atom, nbf = orb_basis.build_molecule_basis(mol)

    for i_atom, atom_shells in enumerate(shells_by_atom):
        sym = mol.symbols[i_atom].capitalize()
        if sym in aux_basis_data:
            continue

        aux_shells = []
        max_ang = 0
        for shell in atom_shells:
            max_ang = max(max_ang, shell.angular)

        for ang in range(max_ang + 1):
            shell_type = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}[min(ang, 4)]
            exps = []
            coeffs = []
            base_exp = 0.5
            for i in range(3):
                exps.append(base_exp * (3.0 ** i))
                coeffs.append(1.0)
            aux_shells.append({
                'shell_type': shell_type,
                'exponents': exps,
                'coefficients': coeffs,
            })

        for ang in range(max_ang + 1, max_ang + 3):
            shell_type = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}[min(ang, 4)]
            exps = []
            coeffs = []
            for i in range(3):
                exps.append((ang + 1) * (2.0 ** i))
                coeffs.append(1.0)
            aux_shells.append({
                'shell_type': shell_type,
                'exponents': exps,
                'coefficients': coeffs,
            })

        aux_basis_data[sym] = aux_shells

    return aux_basis_data


def _get_all_shells(basis: BasisSet, mol: Molecule):
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

    return all_shells, shell_range, nbf


def _build_aux_shells(orb_basis: BasisSet, mol: Molecule):
    from ..basis.gto import GTOShell

    aux_data = build_aux_basis(orb_basis, mol)

    all_shells = []
    for i in range(mol.natom):
        sym = mol.symbols[i].capitalize()
        origin = mol.coordinates[i]
        if sym in aux_data:
            for shell_dict in aux_data[sym]:
                shell = GTOShell(
                    shell_type=shell_dict['shell_type'],
                    exponents=np.array(shell_dict['exponents']),
                    coefficients=np.array(shell_dict['coefficients']),
                    origin=origin,
                )
                all_shells.append(shell)

    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf

    return all_shells, shell_range, idx


def compute_3center_integrals(
    orb_basis: BasisSet,
    mol: Molecule,
    aux_shells_info=None,
) -> np.ndarray:
    orb_all_shells, orb_shell_range, nbf_orb = _get_all_shells(orb_basis, mol)

    if aux_shells_info is None:
        aux_all_shells, aux_shell_range, nbf_aux = _build_aux_shells(orb_basis, mol)
    else:
        aux_all_shells, aux_shell_range, nbf_aux = aux_shells_info

    nshell_orb = len(orb_all_shells)
    nshell_aux = len(aux_all_shells)

    ints = np.zeros((nbf_orb, nbf_orb, nbf_aux))

    for i in range(nshell_orb):
        si, ei = orb_shell_range[i]
        for j in range(i, nshell_orb):
            sj, ej = orb_shell_range[j]
            for k in range(nshell_aux):
                sk, ek = aux_shell_range[k]

                block = _shell_3center(
                    orb_all_shells[i], orb_all_shells[j], aux_all_shells[k]
                )

                ints[si:ei, sj:ej, sk:ek] = block
                ints[sj:ej, si:ei, sk:ek] = block.transpose(1, 0, 2)

    return ints


def _shell_3center(shell_a, shell_b, shell_c) -> np.ndarray:
    n_a = shell_a.n_functions
    n_b = shell_b.n_functions
    n_c = shell_c.n_functions

    block = np.zeros((n_a, n_b, n_c))

    r_a = shell_a.origin
    r_b = shell_b.origin
    r_c = shell_c.origin

    cart_a = shell_a.cartesian_components()
    cart_b = shell_b.cartesian_components()
    cart_c = shell_c.cartesian_components()

    for pa in range(shell_a.n_primitives):
        alpha_a = shell_a.exponents[pa]
        coef_a = shell_a._contract_coeffs[pa]
        norm_a = shell_a.norm_constants[pa]

        for pb in range(shell_b.n_primitives):
            alpha_b = shell_b.exponents[pb]
            coef_b = shell_b._contract_coeffs[pb]
            norm_b = shell_b.norm_constants[pb]

            for pc in range(shell_c.n_primitives):
                alpha_c = shell_c.exponents[pc]
                coef_c = shell_c._contract_coeffs[pc]
                norm_c = shell_c.norm_constants[pc]

                prefactor = coef_a * coef_b * coef_c * norm_a * norm_b * norm_c

                for i, (la, ma, na) in enumerate(cart_a):
                    for j, (lb, mb, nb) in enumerate(cart_b):
                        for k, (lc, mc, nc) in enumerate(cart_c):
                            val = _primitive_3center(
                                la, ma, na, lb, mb, nb, lc, mc, nc,
                                alpha_a, alpha_b, alpha_c,
                                r_a, r_b, r_c
                            )
                            block[i, j, k] += prefactor * val

    return block


def _primitive_3center(la, ma, na, lb, mb, nb, lc, mc, nc,
                       alpha_a, alpha_b, alpha_c,
                       r_a, r_b, r_c):
    zeta_ab = alpha_a + alpha_b
    r_p = (alpha_a * r_a + alpha_b * r_b) / zeta_ab
    r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
    K_ab = math.exp(-alpha_a * alpha_b / zeta_ab * r_ab2)

    zeta = zeta_ab + alpha_c
    r_q = (zeta_ab * r_p + alpha_c * r_c) / zeta
    r_pc2 = float(np.dot(r_p - r_c, r_p - r_c))
    T = zeta_ab * alpha_c / zeta * r_pc2

    n_boys = la + ma + na + lb + mb + nb + lc + mc + nc + 1
    F = boys_function_vec(n_boys - 1, T)

    sss = 2.0 * math.pi ** 2.5 / (zeta_ab * alpha_c * math.sqrt(zeta)) * K_ab * F[0]

    r_pa = r_p - r_a
    r_pb = r_p - r_b
    r_qc = r_q - r_c

    l_max_bra = la + ma + na + lb + mb + nb
    l_max_ket = lc + mc + nc

    n_bra = (l_max_bra + 1) * (l_max_bra + 2) * (l_max_bra + 3) // 6
    n_ket = (l_max_ket + 1) * (l_max_ket + 2) * (l_max_ket + 3) // 6

    table_bra = np.zeros((n_bra,))
    table_bra[0] = sss

    def idx_bra(l, m, n):
        return (l + m + n) * (l + m + n + 1) * (l + m + n + 2) // 6 + \
               (m + n) * (m + n + 1) // 2 + n

    def idx_ket(l, m, n):
        return (l + m + n) * (l + m + n + 1) * (l + m + n + 2) // 6 + \
               (m + n) * (m + n + 1) // 2 + n

    for L in range(0, l_max_bra):
        for l in range(L, -1, -1):
            for m in range(L - l, -1, -1):
                n = L - l - m
                curr = idx_bra(l, m, n)

                if l > 0:
                    next_idx = idx_bra(l + 1, m, n)
                    table_bra[next_idx] += r_pa[0] * table_bra[curr]
                if m > 0:
                    next_idx = idx_bra(l, m + 1, n)
                    table_bra[next_idx] += r_pa[1] * table_bra[curr]
                if n > 0:
                    next_idx = idx_bra(l, m, n + 1)
                    table_bra[next_idx] += r_pa[2] * table_bra[curr]

                if l > 0 and L > 0:
                    next_idx = idx_bra(l + 1, m, n)
                    table_bra[next_idx] += 0.5 / zeta_ab * (l * table_bra[idx_bra(l - 1, m, n)])
                if m > 0 and L > 0:
                    next_idx = idx_bra(l, m + 1, n)
                    table_bra[next_idx] += 0.5 / zeta_ab * (m * table_bra[idx_bra(l, m - 1, n)])
                if n > 0 and L > 0:
                    next_idx = idx_bra(l, m, n + 1)
                    table_bra[next_idx] += 0.5 / zeta_ab * (n * table_bra[idx_bra(l, m, n - 1)])

    total_bra_idx = idx_bra(la + lb, ma + mb, na + nb)
    bra_val = table_bra[total_bra_idx]

    table_ket = np.zeros((n_ket,))
    table_ket[0] = bra_val

    for L in range(0, l_max_ket):
        for l in range(L, -1, -1):
            for m in range(L - l, -1, -1):
                n = L - l - m
                curr = idx_ket(l, m, n)

                if l > 0:
                    next_idx = idx_ket(l + 1, m, n)
                    table_ket[next_idx] += r_qc[0] * table_ket[curr]
                if m > 0:
                    next_idx = idx_ket(l, m + 1, n)
                    table_ket[next_idx] += r_qc[1] * table_ket[curr]
                if n > 0:
                    next_idx = idx_ket(l, m, n + 1)
                    table_ket[next_idx] += r_qc[2] * table_ket[curr]

                if l > 0 and L > 0:
                    next_idx = idx_ket(l + 1, m, n)
                    table_ket[next_idx] += 0.5 / alpha_c * (l * table_ket[idx_ket(l - 1, m, n)])
                if m > 0 and L > 0:
                    next_idx = idx_ket(l, m + 1, n)
                    table_ket[next_idx] += 0.5 / alpha_c * (m * table_ket[idx_ket(l, m - 1, n)])
                if n > 0 and L > 0:
                    next_idx = idx_ket(l, m, n + 1)
                    table_ket[next_idx] += 0.5 / alpha_c * (n * table_ket[idx_ket(l, m, n - 1)])

    return table_ket[idx_ket(lc, mc, nc)]


def compute_2center_integrals(aux_shells_info) -> np.ndarray:
    all_shells, shell_range, nbf = aux_shells_info

    nshell = len(all_shells)

    ints = np.zeros((nbf, nbf))

    for i in range(nshell):
        si, ei = shell_range[i]
        for j in range(i, nshell):
            sj, ej = shell_range[j]

            block = _shell_2center(all_shells[i], all_shells[j])
            ints[si:ei, sj:ej] = block
            ints[sj:ej, si:ei] = block.T

    return ints


def _shell_2center(shell_a, shell_b) -> np.ndarray:
    n_a = shell_a.n_functions
    n_b = shell_b.n_functions

    block = np.zeros((n_a, n_b))

    r_a = shell_a.origin
    r_b = shell_b.origin

    cart_a = shell_a.cartesian_components()
    cart_b = shell_b.cartesian_components()

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
                for j, (lb, mb, nb) in enumerate(cart_b):
                    val = _primitive_2center(
                        la, ma, na, lb, mb, nb,
                        alpha_a, alpha_b, r_a, r_b
                    )
                    block[i, j] += prefactor * val

    return block


def _primitive_2center(la, ma, na, lb, mb, nb,
                       alpha_a, alpha_b, r_a, r_b):
    zeta_ab = alpha_a + alpha_b
    r_p = (alpha_a * r_a + alpha_b * r_b) / zeta_ab
    r_ab2 = float(np.dot(r_a - r_b, r_a - r_b))
    K_ab = math.exp(-alpha_a * alpha_b / zeta_ab * r_ab2)

    T = zeta_ab * r_ab2 / 4.0

    n_boys = la + ma + na + lb + mb + nb + 1
    F = boys_function_vec(n_boys - 1, T)

    sss = 2.0 * math.pi ** 2.5 / (zeta_ab ** 1.5) * K_ab * F[0]

    r_pa = r_p - r_a

    l_max = la + ma + na + lb + mb + nb
    n_cart = (l_max + 1) * (l_max + 2) * (l_max + 3) // 6

    table = np.zeros((n_cart,))
    table[0] = sss

    def idx(l, m, n):
        return (l + m + n) * (l + m + n + 1) * (l + m + n + 2) // 6 + \
               (m + n) * (m + n + 1) // 2 + n

    for L in range(0, l_max):
        for l in range(L, -1, -1):
            for m in range(L - l, -1, -1):
                n = L - l - m
                curr = idx(l, m, n)

                if l > 0:
                    next_idx = idx(l + 1, m, n)
                    table[next_idx] += r_pa[0] * table[curr]
                if m > 0:
                    next_idx = idx(l, m + 1, n)
                    table[next_idx] += r_pa[1] * table[curr]
                if n > 0:
                    next_idx = idx(l, m, n + 1)
                    table[next_idx] += r_pa[2] * table[curr]

                if l > 0 and L > 0:
                    next_idx = idx(l + 1, m, n)
                    table[next_idx] += 0.5 / zeta_ab * (l * table[idx(l - 1, m, n)])
                if m > 0 and L > 0:
                    next_idx = idx(l, m + 1, n)
                    table[next_idx] += 0.5 / zeta_ab * (m * table[idx(l, m - 1, n)])
                if n > 0 and L > 0:
                    next_idx = idx(l, m, n + 1)
                    table[next_idx] += 0.5 / zeta_ab * (n * table[idx(l, m, n - 1)])

    return table[idx(la + lb, ma + mb, na + nb)]


def build_coulomb_ri(dm: np.ndarray, ints_3c: np.ndarray, ints_2c_inv: np.ndarray) -> np.ndarray:
    b = np.einsum("mn,mnp->p", dm, ints_3c)
    c = ints_2c_inv @ b
    J = np.einsum("p,mnp->mn", c, ints_3c)
    return J


def build_exchange_ri(dm: np.ndarray, ints_3c: np.ndarray, ints_2c_inv: np.ndarray,
                      orb_basis: BasisSet, mol: Molecule) -> np.ndarray:
    nbf = dm.shape[0]
    naux = ints_3c.shape[2]

    K = np.zeros((nbf, nbf))

    b = np.einsum("mn,mnp->p", dm, ints_3c)
    c = ints_2c_inv @ b

    shells_by_atom, _ = orb_basis.build_molecule_basis(mol)
    all_shells = []
    for atom_shells in shells_by_atom:
        all_shells.extend(atom_shells)

    shell_range = []
    idx = 0
    for s in all_shells:
        nf = s.n_functions
        shell_range.append((idx, idx + nf))
        idx += nf

    for i in range(len(all_shells)):
        si, ei = shell_range[i]
        for j in range(len(all_shells)):
            sj, ej = shell_range[j]
            k_contrib = np.einsum("mp,np,p->mn",
                                  ints_3c[si:ei, :, :],
                                  ints_3c[sj:ej, :, :],
                                  c)
            K[si:ei, sj:ej] = k_contrib

    return K


class RIJEngine:
    def __init__(self, basis: BasisSet, mol: Molecule):
        self.basis = basis
        self.mol = mol
        self._aux_shells_info = None
        self._ints_3c = None
        self._ints_2c = None
        self._ints_2c_inv = None

    def build(self):
        print("  构建 RI-J 辅助基组...")
        self._aux_shells_info = _build_aux_shells(self.basis, self.mol)

        print("  构建 RI-J 三中心积分...")
        self._ints_3c = compute_3center_integrals(
            self.basis, self.mol, self._aux_shells_info
        )

        print("  构建 RI-J 两中心积分...")
        self._ints_2c = compute_2center_integrals(self._aux_shells_info)

        print("  求解两中心积分矩阵的逆...")
        n_aux = self._ints_2c.shape[0]
        reg = 1e-10 * np.eye(n_aux)
        try:
            self._ints_2c_inv = np.linalg.inv(self._ints_2c + reg)
        except np.linalg.LinAlgError:
            self._ints_2c_inv = np.linalg.pinv(self._ints_2c)

    def build_j(self, dm: np.ndarray) -> np.ndarray:
        if self._ints_3c is None:
            self.build()
        return build_coulomb_ri(dm, self._ints_3c, self._ints_2c_inv)

    def build_k(self, dm: np.ndarray) -> np.ndarray:
        if self._ints_3c is None:
            self.build()
        return build_exchange_ri(dm, self._ints_3c, self._ints_2c_inv, self.basis, self.mol)

    def build_fock(self, h_core: np.ndarray, dm: np.ndarray) -> np.ndarray:
        J = self.build_j(dm)
        K = self.build_k(dm)
        return h_core + J - 0.5 * K


class RIKEngine:
    def __init__(self, basis: BasisSet, mol: Molecule):
        self.basis = basis
        self.mol = mol
        self._rij_engine = RIJEngine(basis, mol)

    def build(self):
        self._rij_engine.build()

    def build_k(self, dm: np.ndarray) -> np.ndarray:
        if self._rij_engine._ints_3c is None:
            self._rij_engine.build()
        return build_exchange_ri(dm, self._rij_engine._ints_3c,
                                 self._rij_engine._ints_2c_inv,
                                 self.basis, self.mol)