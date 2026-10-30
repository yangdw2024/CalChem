"""
分解 MP3 能量的各个分量，验证公式。
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
print(f'E_ref (det): {E_ref:.10f}')
print(f'E0 (sum eps): {E0:.10f}')
print(f'E1 = E_ref - E0: {E1:.10f}')

# Double excitations (unique, i<j, a<b)
doubles = []
for i, j in combinations(sorted(occ_spin), 2):
    for a, b in combinations(sorted(virt_spin), 2):
        doubles.append((i, j, a, b))
n_d = len(doubles)
print(f'Number of double excitations: {n_d}')

# Compute H0 diagonal and V_0m
H0_d = np.zeros(n_d)
V0 = np.zeros(n_d)
for m, (i, j, a, b) in enumerate(doubles):
    occ_m = set(ref_occ) - {i, j} | {a, b}
    H0_d[m] = sum(eps_spin[p] for p in occ_m)
    V0[m] = antisym_int_spin(i, j, a, b)

D0 = E0 - H0_d  # E0 - Em

# Compute MP2 from doubles (unique pairs, no 1/4 factor)
E2 = np.sum(V0**2 / D0)
print(f'MP2 (unique doubles): {E2:.10f}')
print(f'Expected MP2:          -0.0355028412')

# Hmm, should match? Let's check with 1/4 factor formula
E2_full = 0.0
for i in occ_spin:
    for j in occ_spin:
        for a in virt_spin:
            for b in virt_spin:
                D = eps_spin[i] + eps_spin[j] - eps_spin[a] - eps_spin[b]
                if abs(D) < 1e-12:
                    continue
                val = antisym_int_spin(i, j, a, b)
                E2_full += 0.25 * val * val / D
print(f'MP2 (full 1/4): {E2_full:.10f}')

# Now compute V_mn matrix element between two double excitations
def V_mn(m_idx, n_idx):
    i, j, a, b = doubles[m_idx]
    k, l, c, d = doubles[n_idx]
    occ_m = set(ref_occ) - {i, j} | {a, b}
    occ_n = set(ref_occ) - {k, l} | {c, d}
    only_m = occ_m - occ_n
    only_n = occ_n - occ_m
    diff = len(only_m)
    if diff == 0:
        return det_energy(sorted(occ_m)) - H0_d[m_idx]
    elif diff == 2:
        p_list = sorted(list(only_m))
        q_list = sorted(list(only_n))
        p, r = p_list[0], p_list[1]
        q, s = q_list[0], q_list[1]
        return g_phys(p, r, q, s)
    else:
        return 0.0

print('\nComputing V_mn for all double excitation pairs...')
print('(This may take a moment...)')

# Compute term1 = sum_{m,n} V0m * Vmn * V0n / (Dm * Dn)
term1 = 0.0
for m in range(n_d):
    for n in range(n_d):
        v_mn = V_mn(m, n)
        if abs(v_mn) < 1e-15:
            continue
        term1 += V0[m] * v_mn * V0[n] / (D0[m] * D0[n])

print(f'term1 = {term1:.10f}')

# term2 = E1 * sum_m |V0m|^2 / Dm^2
term2 = E1 * np.sum(V0**2 / D0**2)
print(f'term2 = {term2:.10f}')

E3 = term1 - term2
print(f'E3 = term1 - term2 = {E3:.10f}')

# Now let's decompose term1 into oooo-type and vvvv-type contributions
# oooo-type: same virtuals (a=c, b=d), different occupancies
# vvvv-type: same occupancies (i=k, j=l), different virtuals
# other types: ???

print('\nDecomposing term1...')
term1_oooo = 0.0
term1_vvvv = 0.0
term1_other = 0.0

for m, (i, j, a, b) in enumerate(doubles):
    for n, (k, l, c, d) in enumerate(doubles):
        v_mn = V_mn(m, n)
        if abs(v_mn) < 1e-15:
            continue
        if m == n:
            # diagonal - skip for now? Actually diagonal is Vmm = Emm - Em0
            # This contributes to term1 as well.
            pass
        
        # Check if same virtuals
        if (a == c and b == d):
            term1_oooo += V0[m] * v_mn * V0[n] / (D0[m] * D0[n])
        # Check if same occupancies
        elif (i == k and j == l):
            term1_vvvv += V0[m] * v_mn * V0[n] / (D0[m] * D0[n])
        else:
            term1_other += V0[m] * v_mn * V0[n] / (D0[m] * D0[n])

print(f'term1_oooo (same virtuals): {term1_oooo:.10f}')
print(f'term1_vvvv (same occupancies): {term1_vvvv:.10f}')
print(f'term1_other: {term1_other:.10f}')
print(f'Sum: {term1_oooo + term1_vvvv + term1_other:.10f}')
print(f'term1 (total): {term1:.10f}')
