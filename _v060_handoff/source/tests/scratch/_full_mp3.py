"""
更详细地分析 MP3 公式，找出正确的空间轨道表达式。
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

mo_coeff = rhf.mo_coeff
mo_energy = rhf.mo_energy
nocc = rhf.mol.nelectron // 2
nbf = mo_coeff.shape[0]
nvirt = nbf - nocc

eri_ao = compute_eri(rhf.basis_set, rhf.mol)
eri_mo = np.einsum('pi,pqrs->iqrs', mo_coeff, eri_ao)
eri_mo = np.einsum('qj,iqrs->ijrs', mo_coeff, eri_mo)
eri_mo = np.einsum('rk,ijrs->ijks', mo_coeff, eri_mo)
eri_mo = np.einsum('sl,ijks->ijkl', mo_coeff, eri_mo)

T_ao = compute_kinetic(rhf.basis_set, rhf.mol)
Vn_ao = compute_nuclear(rhf.basis_set, rhf.mol)
h_ao = T_ao + Vn_ao
h_mo = mo_coeff.T @ h_ao @ mo_coeff

# Spin orbital setup
occ_alpha = list(range(nocc))
virt_alpha = list(range(nocc, nbf))
occ_beta = list(range(nbf, nbf + nocc))
virt_beta = list(range(nbf + nocc, 2 * nbf))
occ_spin = occ_alpha + occ_beta
virt_spin = virt_alpha + virt_beta

eps_spin = np.zeros(2*nbf)
eps_spin[:nbf] = mo_energy
eps_spin[nbf:] = mo_energy

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

def h_spin(p, q):
    sp, pp = p // nbf, p % nbf
    sq, pq = q // nbf, q % nbf
    if sp != sq:
        return 0.0
    return h_mo[pp, pq]

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
E0 = sum(eps_spin[p] for p in ref_occ)
E1 = E_ref - E0

print('=== Spin orbital reference ===')
print(f'E_ref (det): {E_ref:.10f}')
print(f'E0 (sum eps): {E0:.10f}')
print(f'E1: {E1:.10f}')

# Double excitations (unique, i<j, a<b)
doubles = []
for i, j in combinations(sorted(occ_spin), 2):
    for a, b in combinations(sorted(virt_spin), 2):
        doubles.append((i, j, a, b))
n_d = len(doubles)
print(f'Number of double excitations: {n_d}')

H0_d = np.zeros(n_d)
V0 = np.zeros(n_d)
for m, (i, j, a, b) in enumerate(doubles):
    occ_m = set(ref_occ) - {i, j} | {a, b}
    H0_d[m] = sum(eps_spin[p] for p in occ_m)
    V0[m] = antisym_int_spin(i, j, a, b)

D0 = E0 - H0_d

# MP2
E2 = np.sum(V0**2 / D0)
print(f'MP2 (spin orb, unique): {E2:.10f}')

# Build V_mn matrix (only upper triangle, using Slater-Condon)
def diff_orbs(m_idx, n_idx):
    i, j, a, b = doubles[m_idx]
    k, l, c, d = doubles[n_idx]
    occ_m = set(ref_occ) - {i, j} | {a, b}
    occ_n = set(ref_occ) - {k, l} | {c, d}
    only_m = occ_m - occ_n
    only_n = occ_n - occ_m
    return only_m, only_n, len(only_m)

def V_mn_sc(m_idx, n_idx):
    """Slater-Condon matrix element V_mn = <m|V|n>"""
    if m_idx == n_idx:
        occ_m = set(ref_occ) - set(doubles[m_idx][:2]) | set(doubles[m_idx][2:])
        E_m = det_energy(sorted(occ_m))
        H0_m = sum(eps_spin[p] for p in occ_m)
        return E_m - H0_m
    
    only_m, only_n, diff = diff_orbs(m_idx, n_idx)
    
    if diff == 1:
        # Single excitation difference -> Fock matrix element
        p = list(only_m)[0]
        q = list(only_n)[0]
        # <p|f|q> = h_pq + sum_{k in common} <pk||qk>
        common = (set(ref_occ) - set(doubles[m_idx][:2]) | set(doubles[m_idx][2:])) & \
                 (set(ref_occ) - set(doubles[n_idx][:2]) | set(doubles[n_idx][2:]))
        val = h_spin(p, q)
        for k in common:
            val += g_phys(p, k, q, k)
        return val
    elif diff == 2:
        # Double excitation difference -> <pq||rs>
        p_list = sorted(list(only_m))
        q_list = sorted(list(only_n))
        p, r = p_list[0], p_list[1]
        q, s = q_list[0], q_list[1]
        return g_phys(p, r, q, s)
    else:
        return 0.0

print('\nComputing full V_mn...')
V_mat = np.zeros((n_d, n_d))
for m in range(n_d):
    for n in range(m, n_d):
        val = V_mn_sc(m, n)
        V_mat[m, n] = val
        V_mat[n, m] = val

# term1 = sum_{m,n} V0m * Vmn * V0n / (Dm * Dn)
term1 = V0 @ V_mat @ V0 / (D0[:, None] * D0[None, :])
# Wait, that's not right. It should be sum_{m,n} V0m Vmn V0n / (Dm Dn)
term1 = 0.0
for m in range(n_d):
    for n in range(n_d):
        term1 += V0[m] * V_mat[m, n] * V0[n] / (D0[m] * D0[n])

term2 = E1 * np.sum(V0**2 / D0**2)
E3 = term1 - term2

print(f'term1: {term1:.10f}')
print(f'term2: {term2:.10f}')
print(f'E3 = term1 - term2: {E3:.10f}')

# Now let's also include single excitations
print('\n=== Including single excitations ===')
singles = []
for i in occ_spin:
    for a in virt_spin:
        singles.append((i, a))
n_s = len(singles)
print(f'Number of single excitations: {n_s}')

n_total = 1 + n_s + n_d  # 0 = ref, 1..n_s = singles, n_s+1.. = doubles

# Compute V with singles
# V_ref_singles = 0 (Brillouin's theorem)
# V_singles_singles = Fock matrix (diagonal in MO basis -> only diagonal)
# V_singles_doubles = single-double coupling

# Actually for MP3, let's compute term1 with singles included
# But V_0m = 0 for singles, so singles only appear as intermediate states
# term1 = sum_{m,n} V0m Vmn Vn0 / (Dm Dn)
# Since V0m = 0 for singles, m and n must both be doubles
# So singles don't contribute to term1 directly

# But wait, what about term2? term2 only involves V0m, which is zero for singles.
# So singles don't contribute to MP3?

# Actually, I think singles do contribute through E1? No, E1 is just <0|V|0>.

# Let me verify by including singles in the CI matrix
# and computing MP3 via full RS PT.

print('\n=== Full CI (ref + singles + doubles) ===')
print('Building H matrix...')
H_full = np.zeros((n_total, n_total))

# Diagonal
H_full[0, 0] = E_ref

# Singles diagonal
for s, (i, a) in enumerate(singles):
    occ = set(ref_occ) - {i} | {a}
    H_full[1+s, 1+s] = det_energy(sorted(occ))

# Doubles diagonal
for d, (i, j, a, b) in enumerate(doubles):
    occ = set(ref_occ) - {i, j} | {a, b}
    H_full[1+n_s+d, 1+n_s+d] = det_energy(sorted(occ))

# Off-diagonal using Slater-Condon
def get_occ_state(idx):
    if idx == 0:
        return set(ref_occ)
    elif idx <= n_s:
        i, a = singles[idx-1]
        return set(ref_occ) - {i} | {a}
    else:
        i, j, a, b = doubles[idx-1-n_s]
        return set(ref_occ) - {i, j} | {a, b}

def slater_condon(occ1, occ2):
    only1 = occ1 - occ2
    only2 = occ2 - occ1
    k = len(only1)
    if k == 0:
        return det_energy(sorted(occ1))
    elif k == 1:
        p = list(only1)[0]
        q = list(only2)[0]
        common = occ1 & occ2
        val = h_spin(p, q)
        for r in common:
            val += g_phys(p, r, q, r)
        return val
    elif k == 2:
        p_list = sorted(list(only1))
        q_list = sorted(list(only2))
        p, r = p_list[0], p_list[1]
        q, s = q_list[0], q_list[1]
        return g_phys(p, r, q, s)
    else:
        return 0.0

for i in range(n_total):
    occ_i = get_occ_state(i)
    for j in range(i+1, n_total):
        occ_j = get_occ_state(j)
        val = slater_condon(occ_i, occ_j)
        H_full[i, j] = val
        H_full[j, i] = val

print('H matrix built.')

# Diagonalize
e_fci, v_fci = np.linalg.eigh(H_full)
print(f'FCI ground: {e_fci[0]:.10f}')
print(f'Reference:   {E_ref:.10f}')
print(f'E_corr (FCI - ref): {e_fci[0] - E_ref:.10f}')

# Compute MP3 via RS PT from full H matrix
H0_full = np.diag(H_full)
V_full = H_full - np.diag(H0_full)
E0_full = H0_full[0]

# Check Brillouin
max_V0s = max(abs(V_full[0, 1:1+n_s]))
print(f'Max V0s (should be ~0 for Brillouin): {max_V0s:.2e}')

# MP2
E2_full = 0.0
for m in range(1, n_total):
    D = E0_full - H0_full[m]
    if abs(D) < 1e-12:
        continue
    E2_full += V_full[0, m] ** 2 / D
print(f'MP2 (full CI): {E2_full:.10f}')

# MP3
term1_full = 0.0
term2_full = 0.0
E1_full = V_full[0, 0]
print(f'E1_full (V[0,0]): {E1_full:.10f}')

for m in range(n_total):
    Dm = E0_full - H0_full[m]
    if abs(Dm) < 1e-12:
        continue
    for n in range(n_total):
        Dn = E0_full - H0_full[n]
        if abs(Dn) < 1e-12:
            continue
        term1_full += V_full[0, m] * V_full[m, n] * V_full[n, 0] / (Dm * Dn)
    term2_full += V_full[0, m] ** 2 / (Dm * Dm)

term2_full *= E1_full
E3_full = term1_full - term2_full
print(f'\nterm1_full: {term1_full:.10f}')
print(f'term2_full: {term2_full:.10f}')
print(f'E3_full = term1 - term2: {E3_full:.10f}')

print(f'\n=== Summary ===')
print(f'MP2: {E2_full:.10f}')
print(f'MP3: {E3_full:.10f}')
print(f'MP2+MP3: {E2_full + E3_full:.10f}')
print(f'FCI corr: {e_fci[0] - E_ref:.10f}')
