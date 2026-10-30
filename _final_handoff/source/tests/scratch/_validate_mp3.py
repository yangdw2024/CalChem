"""
通过 CI 矩阵验证 MP2 和 MP3 公式。

构造包含参考态、单激发、双激发的 CI 矩阵，
然后用 Rayleigh-Schrödinger 微扰论计算
MP2 和 MP3 能量，与我们的实现对比。
"""

import numpy as np
from itertools import combinations
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.integrals.two_electron import compute_eri
from calchem.integrals.one_electron import compute_kinetic, compute_nuclear

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
rhf = RHF(mol, 'sto-3g', verbose=0)
rhf.kernel()
print(f'HF energy: {rhf.e_tot:.10f} Eh')

mo_coeff = rhf.mo_coeff
mo_energy = rhf.mo_energy
nocc = rhf.mol.nelectron // 2
nbf = mo_coeff.shape[0]
nvirt = nbf - nocc
print(f'nocc={nocc}, nvirt={nvirt}, nbf={nbf}')

eri_ao = compute_eri(rhf.basis_set, rhf.mol)

eri_mo = np.einsum('pi,pqrs->iqrs', mo_coeff, eri_ao)
eri_mo = np.einsum('qj,iqrs->ijrs', mo_coeff, eri_mo)
eri_mo = np.einsum('rk,ijrs->ijks', mo_coeff, eri_mo)
eri_mo = np.einsum('sl,ijks->ijkl', mo_coeff, eri_mo)

T_ao = compute_kinetic(rhf.basis_set, rhf.mol)
Vn_ao = compute_nuclear(rhf.basis_set, rhf.mol)
h_ao = T_ao + Vn_ao
h_mo = mo_coeff.T @ h_ao @ mo_coeff

# Spin orbital indexing:
# alpha: 0 .. nbf-1
# beta:  nbf .. 2*nbf-1
occ_alpha = list(range(nocc))
virt_alpha = list(range(nocc, nbf))
occ_beta = list(range(nbf, nbf + nocc))
virt_beta = list(range(nbf + nocc, 2 * nbf))
occ_spin = occ_alpha + occ_beta
virt_spin = virt_alpha + virt_beta

eps_spin = np.zeros(2*nbf)
eps_spin[:nbf] = mo_energy
eps_spin[nbf:] = mo_energy

def h_spin(p, q):
    sp, pp = p // nbf, p % nbf
    sq, pq = q // nbf, q % nbf
    if sp != sq:
        return 0.0
    return h_mo[pp, pq]

def chem_int_spin(p, q, r, s):
    sp, pp = p // nbf, p % nbf
    sq, pq = q // nbf, q % nbf
    sr, pr = r // nbf, r % nbf
    ss, ps = s // nbf, s % nbf
    if sp != sq or sr != ss:
        return 0.0
    return eri_mo[pp, pq, pr, ps]

def antisym_int_spin(i, j, a, b):
    return chem_int_spin(i, a, j, b) - chem_int_spin(i, b, j, a)

def g_phys(p, q, r, s):
    return chem_int_spin(p, r, q, s) - chem_int_spin(p, s, q, r)

def det_energy(occ_list):
    E = 0.0
    for p in occ_list:
        E += h_spin(p, p)
    for idx_p, p in enumerate(occ_list):
            for q in occ_list:
                E += 0.5 * g_phys(p, q, p, q)
    return E

ref_occ = sorted(occ_spin)
E_ref = det_energy(ref_occ)
print(f'Reference det energy: {E_ref:.10f}')
print(f'E_HF from SCF:      {rhf.e_tot:.10f}')
print(f'Diff: {E_ref - rhf.e_tot:.2e}')

# Build list of states: 0 = reference, then singles, then doubles
singles = []
for i in occ_spin:
    for a in virt_spin:
        singles.append((i, a))
print(f'Number of single excitations: {len(singles)}')

doubles = []
for i, j in combinations(sorted(occ_spin), 2):
    for a, b in combinations(sorted(virt_spin), 2):
        doubles.append((i, j, a, b))
print(f'Number of double excitations: {len(doubles)}')

n_s = len(singles)
n_d = len(doubles)
n_states = 1 + n_s + n_d
print(f'Total states: {n_states}')

# Compute determinant occupation lists
def get_occ(state_idx):
    if state_idx == 0:
        return set(ref_occ)
    elif state_idx <= n_s:
        i, a = singles[state_idx - 1]
        occ = set(ref_occ) - {i} | {a}
        return occ
    else:
        i, j, a, b = doubles[state_idx - 1 - n_s]
        occ = set(ref_occ) - {i, j} | {a, b}
        return occ

# Phase function: compute phase of determinant relative to reference
def det_phase(occ_set):
    # Bring determinant to canonical order (sorted)
    occ_list = sorted(occ_set)
    return 1.0  # for now assume positive

# Build H matrix
H = np.zeros((n_states, n_states))

# Diagonal elements
for s in range(n_states):
    occ = get_occ(s)
    H[s, s] = det_energy(sorted(occ))

# Off-diagonal elements using Slater-Condon rules
def slater_condon(occ1, occ2):
    only1 = occ1 - occ2
    only2 = occ2 - occ1
    k = len(only1)
    if k == 0:
        return det_energy(sorted(occ1))
    elif k == 1:
        p = list(only1)[0]
        q = list(only2)[0]
        # <1|H|2> = <p|f|q> + sum_{i in occ1∩occ2} <pi||qi>
        # = <p|h|q> + sum_{i in common} <pi||qi>
        common = occ1 & occ2
        val = h_spin(p, q)
        for i in common:
            val += g_phys(p, i, q, i)
        return val
    elif k == 2:
        p_list = sorted(list(only1))
        q_list = sorted(list(only2))
        p, r = p_list[0], p_list[1]
        q, s = q_list[0], q_list[1]
        # <pr||qs> (physicists notation)
        return g_phys(p, r, q, s)
    else:
        return 0.0

print('Building H matrix...')
for i in range(n_states):
    occ_i = get_occ(i)
    for j in range(i+1, n_states):
        occ_j = get_occ(j)
        val = slater_condon(occ_i, occ_j)
        H[i, j] = val
        H[j, i] = val

print('H matrix built.')

# Diagonalize to get FCI energy
e_fci, v_fci = np.linalg.eigh(H)
print(f'\nFCI ground state energy: {e_fci[0]:.10f}')

# Now compute MP2 and MP3 using RS perturbation theory
# H0 = diag(H) = orbital energies
E0 = H[0, 0]
E1 = H[0, 0] - E0  # which is 0? No, V = H - H0, and H0 = diag(H) diagonal.

# Actually, H0 is the diagonal of H (orbital energy sums)
# V = H - diag(H)
H0_diag = np.diag(H)
V = H - np.diag(H0_diag)

# RS perturbation theory
# E0 = H0_diag[0]
# E1 = V[0,0] = H[0,0] - H0_diag[0] = 0? No, V[0,0] = 0 because we subtracted diagonal.

# Wait, V = H - H0, where H0 has diagonal elements.
# Actually for RS PT:
# H = H0 + V
# H0|n> = E_n^0 |n>
# E_n^0 = H0_diag[n]

E0_0 = H0_diag[0]
E1_0 = V[0, 0]  # which is 0!

# Wait, if H = H - diag, then V[0,0] = 0.
# But in MP theory, E1 = <0|V|0> is not zero.

# Ah, I see the confusion. In standard MP theory:
# H0 = sum_i f_ii a†_i a_i (Fock operator diagonal)
# V = H - H0
# Then <0|H0|0> = sum_i eps_i = E0_0
# <0|V|0> = E_HF - E0_0 = E1_0 (not zero!)

# But in our CI basis (determinants basis),
# H0 is diagonal with diagonal elements E_m^0 = sum_{p in occ_m} eps_p
# Then V = H - H0, and V[0,0] = E_HF - E0_0 != 0.

# Let me recompute H0 properly:
H0_diag = np.zeros(n_states)
for s in range(n_states):
    occ = get_occ(s)
    H0_diag[s] = sum(eps_spin[p] for p in occ)
E0_0 = H0_diag[0]
E1_0 = H[0, 0] - E0_0
print(f'\nE0 (sum of eps): {E0_0:.10f}')
print(f'E1 (<0|V|0>): {E1_0:.10f}')

V = H - np.diag(H0_diag)
print(f'V[0,0] = {V[0,0]:.10f} (should equal E1)')

# Now compute MP2 and MP3
# E2 = sum_m |V_0m|^2 / (E0_0 - E0_m)
E2 = 0.0
for m in range(1, n_states):
    D = E0_0 - H0_diag[m]
    if abs(D) < 1e-12:
        continue
    E2 += V[0, m] ** 2 / D

print(f'\nMP2 (from CI): {E2:.10f}')

# E3 = sum_{m,n} V_0m V_mn V_n0 / ((E0_0 - E0_m)(E0_0 - E0_n)) - E1 * sum_m |V_0m|^2 / (E0_0 - E0_m)^2
term1 = 0.0
term2 = 0.0
for m in range(n_states):
    Dm = E0_0 - H0_diag[m]
    if abs(Dm) < 1e-12:
        continue
    for n in range(n_states):
        Dn = E0_0 - H0_diag[n]
        if abs(Dn) < 1e-12:
            continue
        term1 += V[0, m] * V[m, n] * V[n, 0] / (Dm * Dn)
    term2 += V[0, m] ** 2 / (Dm * Dm)

term2 = E1_0 * term2
E3 = term1 - term2

print(f'MP3 term1: {term1:.10f}')
print(f'MP3 term2: {term2:.10f}')
print(f'MP3 (from CI): {E3:.10f}')

print(f'\nMP2 + MP3: {E2 + E3:.10f}')
print(f'FCI - HF: {e_fci[0] - rhf.e_tot:.10f}')
