"""
用正确的 H0 定义（轨道能量之和）计算 MP2 和 MP3。
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

print('=== Reference values ===')
print(f'E_ref (det energy): {E_ref:.10f}')
print(f'E0 (sum of eps):    {E0:.10f}')
print(f'E1 = E_ref - E0:   {E1:.10f}')
print(f'E_HF (total):      {rhf.e_tot:.10f}')

# Build states: reference + singles + doubles
states = []
state_type = []  # 'ref', 'single', 'double'

states.append(('ref', None))
state_type.append('ref')

for i in occ_spin:
    for a in virt_spin:
        states.append(('single', (i, a)))
        state_type.append('single')
        
for i, j in combinations(sorted(occ_spin), 2):
    for a, b in combinations(sorted(virt_spin), 2):
        states.append(('double', (i, j, a, b)))
        state_type.append('double')

n_states = len(states)
n_s = sum(1 for t in state_type if t == 'single')
n_d = sum(1 for t in state_type if t == 'double')
print(f'\nNumber of states: {n_states}')
print(f'  singles: {n_s}')
print(f'  doubles: {n_d}')

def get_occ(idx):
    t, data = states[idx]
    if t == 'ref':
        return set(ref_occ)
    elif t == 'single':
        i, a = data
        return set(ref_occ) - {i} | {a}
    else:
        i, j, a, b = data
        return set(ref_occ) - {i, j} | {a, b}

def get_H0(idx):
    occ = get_occ(idx)
    return sum(eps_spin[p] for p in occ)

def slater_condon_H(occ1, occ2):
    """Compute <occ1|H|occ2> using Slater-Condon rules."""
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

# Build H matrix and H0 vector
print('\nBuilding H matrix...')
H = np.zeros((n_states, n_states))
H0 = np.zeros(n_states)

for i in range(n_states):
    occ_i = get_occ(i)
    H0[i] = get_H0(i)
    for j in range(i, n_states):
        occ_j = get_occ(j)
        val = slater_condon_H(occ_i, occ_j)
        H[i, j] = val
        H[j, i] = val

V = H - np.diag(H0)  # V = H - H0

print('H matrix built.')

# Verify: H[0,0] should equal E_ref
print(f'\nVerification:')
print(f'  H[0,0]: {H[0,0]:.10f}')
print(f'  E_ref:  {E_ref:.10f}')
print(f'  V[0,0] = H[0,0] - H0[0]: {V[0,0]:.10f}')
print(f'  E1:     {E1:.10f}')

# Verify Brillouin's theorem
max_V0s = 0.0
for s in range(n_states):
    if state_type[s] == 'single':
        max_V0s = max(max_V0s, abs(V[0, s]))
print(f'  Max V_ref_single (Brillouin): {max_V0s:.2e}')

# Diagonalize for FCI reference
e_fci, v_fci = np.linalg.eigh(H)
print(f'\nFCI ground energy: {e_fci[0]:.10f}')
print(f'Reference energy: {H[0,0]:.10f}')
print(f'FCI correlation:  {e_fci[0] - H[0,0]:.10f}')

# RS perturbation theory
E0_0 = H0[0]  # E0 = H0[0]
E1_0 = V[0, 0]  # E1 = <0|V|0>

print(f'\n=== RS Perturbation Theory ===')
print(f'E0 = H0[0] = {E0_0:.10f}')
print(f'E1 = V[0,0] = {E1_0:.10f}')

# MP2 = sum_m |V_0m|^2 / (E0 - Em0)
E2 = 0.0
for m in range(1, n_states):
    D = E0_0 - H0[m]
    if abs(D) < 1e-12:
        continue
    E2 += V[0, m] ** 2 / D
print(f'E2 (MP2 corr): {E2:.10f}')

# MP3 = sum_{m,n} V_0m V_mn V_n0 / (D_0m D_0n) - E1 * sum_m |V_0m|^2 / D_0m^2
term1 = 0.0
term2 = 0.0
for m in range(n_states):
    Dm = E0_0 - H0[m]
    if abs(Dm) < 1e-12:
        continue
    for n in range(n_states):
        Dn = E0_0 - H0[n]
        if abs(Dn) < 1e-12:
            continue
        term1 += V[0, m] * V[m, n] * V[n, 0] / (Dm * Dn)
    term2 += V[0, m] ** 2 / (Dm * Dm)

term2 *= E1_0
E3 = term1 - term2
print(f'\nterm1: {term1:.10f}')
print(f'term2: {term2:.10f}')
print(f'E3 (MP3 corr): {E3:.10f}')

print(f'\nTotal MP2 corr: {E2:.10f}')
print(f'Total MP3 corr: {E2 + E3:.10f}')
print(f'FCI corr:       {e_fci[0] - H[0,0]:.10f}')

# Now let's compute using only double excitations
# to see if singles contribute to MP3
print('\n=== Doubles-only MP3 ===')
E2_d = 0.0
term1_d = 0.0
term2_d = 0.0
for m in range(n_states):
    if state_type[m] != 'double' and m != 0:
        continue
    Dm = E0_0 - H0[m]
    if abs(Dm) < 1e-12:
        continue
    if m != 0:
        E2_d += V[0, m] ** 2 / D
    
    for n in range(n_states):
        if state_type[n] != 'double' and n != 0:
            continue
        Dn = E0_0 - H0[n]
        if abs(Dn) < 1e-12:
            continue
        term1_d += V[0, m] * V[m, n] * V[n, 0] / (Dm * Dn)
    term2_d += V[0, m] ** 2 / (Dm * Dm)

term2_d *= E1_0
E3_d = term1_d - term2_d
print(f'E2 (doubles only): {E2_d:.10f}')
print(f'E3 (doubles only): {E3_d:.10f}')
print(f'MP2+MP3 (doubles): {E2_d + E3_d:.10f}')

# Now compute the contribution from single excitations as intermediates
print('\n=== Breakdown by intermediate state type ===')
# term1 = V0m Vmn Vn0 / (Dm Dn)
# m and n can be any type
# Let's break it down by m_type and n_type

breakdown = {}
for m in range(n_states):
    Dm = E0_0 - H0[m]
    if abs(Dm) < 1e-12:
        continue
    m_t = state_type[m]
    for n in range(n_states):
        Dn = E0_0 - H0[n]
        if abs(Dn) < 1e-12:
            continue
        n_t = state_type[n]
        val = V[0, m] * V[m, n] * V[n, 0] / (Dm * Dn)
        key = (m_t, n_t)
        if key not in breakdown:
            breakdown[key] = 0.0
        breakdown[key] += val

print('term1 breakdown:')
for key in sorted(breakdown.keys()):
    print(f'  {key[0]} - {key[1]}: {breakdown[key]:.10f}')

print(f'\nTotal term1: {sum(breakdown.values()):.10f}')
