import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.integrals.two_electron import compute_eri

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
rhf = RHF(mol, 'sto-3g', verbose=0)
rhf.kernel()
print(f'HF energy: {rhf.e_tot:.10f} Eh')

mo_coeff = rhf.mo_coeff
mo_energy = rhf.mo_energy
nocc = mol.nelectron // 2
nbf = mo_coeff.shape[0]
nvirt = nbf - nocc
print(f'nocc={nocc}, nvirt={nvirt}')

eri_ao = compute_eri(rhf.basis_set, mol)
eri_mo = np.einsum('pi,pqrs->iqrs', mo_coeff, eri_ao)
eri_mo = np.einsum('qj,iqrs->ijrs', mo_coeff, eri_mo)
eri_mo = np.einsum('rk,ijrs->ijks', mo_coeff, eri_mo)
eri_mo = np.einsum('sl,ijks->ijkl', mo_coeff, eri_mo)

# Spin orbit indexing:
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

def chem_int(p, q, r, s):
    """Chemists notation (pq|rs) for spin orbitals."""
    sp, pp = p // nbf, p % nbf
    sq, pq = q // nbf, q % nbf
    sr, pr = r // nbf, r % nbf
    ss, ps = s // nbf, s % nbf
    if sp != sq or sr != ss:
        return 0.0
    return eri_mo[pp, pq, pr, ps]

def antisym_int(i, j, a, b):
    """Physicists notation antisymmetrized <ij||ab> = (ia|jb) - (ib|ja)."""
    return chem_int(i, a, j, b) - chem_int(i, b, j, a)

# Test
i = occ_alpha[0]
j = occ_beta[0]
a = virt_alpha[0]
b = virt_beta[0]
print(f'<alpha0,beta0||alpha5,beta5> = {antisym_int(i, j, a, b):.10f}')
print(f'  should be (0,5|0,5) = {eri_mo[0,5,0,5]:.10f}')

# MP2 energy using spin orbitals
e_mp2 = 0.0
for i in occ_spin:
    for j in occ_spin:
        for a in virt_spin:
            for b in virt_spin:
                denom = eps_spin[i] + eps_spin[j] - eps_spin[a] - eps_spin[b]
                if abs(denom) < 1e-12:
                    continue
                val = antisym_int(i, j, a, b)
                e_mp2 += 0.25 * val * val / denom

print(f'MP2 (spin orb, 1/4): {e_mp2:.10f} Eh')
print(f'MP2 (RMP2 module):  -0.0355028412 Eh')

# Now MP3 using spin orbitals
# E_MP3 = sum_{m,n != 0} <0|V|m><m|V|n><n|V|0> / (D_0m * D_0n)
# where m, n are doubly excited determinants
# and <0|V|m> = <ij||ab> for m = (i,j->a,b)

# But wait, we also need E^(1) term:
# E^(3) = E^(1) * sum_m |V_0m|^2 / D_0m^2 - sum_{m,n} V_0m V_mn V_n0 / (D_0m D_0n)
# Actually let me verify the formula...

# Let's construct the list of double excitations
dets = []  # list of (i, j, a, b) tuples
for i in occ_spin:
    for j in occ_spin:
        for a in virt_spin:
            for b in virt_spin:
                # For spin orbitals, i and j are distinct spin-orbital indices
                # But the standard formula sums over all i,j,a,b with 1/4 factor
                dets.append((i, j, a, b))

n_det = len(dets)
print(f'Number of double excitations: {n_det}')

# Actually, the MP2 formula sums over all i,j,a,b with 1/4 factor.
# So dets should be all pairs, and we use the 1/4 factor.

# For MP3, we need V_mn = <m|V|n> = <Phi_m|H|Phi_n> for m != n
# where Phi_m is a doubly excited determinant.

# According to Slater-Condon rules:
# <Phi_pq^rs|H|Phi_tu^vw> = ?
# This depends on how many orbitals differ.

# Let's implement the general formula.
# Two doubly excited determinants:
# |ij^ab> and |kl^cd>
# The matrix element <ij^ab|H|kl^cd> can be computed using Slater-Condon rules.

# Actually, for two Slater determinants that differ by k orbitals:
# k=0: sum_i <i|h|i> + 1/2 sum_ij <ij||ij>  (diagonal)
# k=1: <p|f|q> where p->q is the single excitation connecting them
# k=2: <pr||qs> where pq->rs is the double excitation

# So we need to find how many spin orbitals differ between
# the two doubly excited determinants.

# Reference determinant occupation: all occ_spin
# |ij^ab>: remove i,j from occ, add a,b to occ
# |kl^cd>: remove k,l from occ, add c,d to occ

# Occupied orbitals of |ij^ab>: (occ_spin \ {i,j}) U {a,b}
# Occupied orbitals of |kl^cd>: (occ_spin \ {k,l}) U {c,d}

# Symmetric difference:
# In |ij^ab> but not in |kl^cd>: {i,j} \ {k,l} U {a,b} \ {c,d}
# In |kl^cd> but not in |ij^ab>: {k,l} \ {i,j} U {c,d} \ {a,b}

# Total differing orbitals = 2 * |{i,j} Δ {k,l}| + 2 * |{a,b} Δ {c,d}|? No.

# Let's compute the symmetric difference size.
def num_diff_orbs(i, j, a, b, k, l, c, d):
    """Number of differing spin orbitals between |ij^ab> and |kl^cd>."""
    occ1 = set(occ_spin) - {i, j} | {a, b}
    occ2 = set(occ_spin) - {k, l} | {c, d}
    return len(occ1.symmetric_difference(occ2))

# Test
print()
print('Testing matrix element computation...')

# V_mn = <m|H|n> for doubly excited m, n
# For m=n (diagonal): we don't need it directly
# For m != n: use Slater-Condon

def double_excitation_h_element(i1, j1, a1, b1, i2, j2, a2, b2):
    """
    Compute <i1j1^a1b1|H|i2j2^a2b2> for doubly excited determinants.
    
    Uses Slater-Condon rules.
    Returns the matrix element.
    """
    occ1 = set(occ_spin) - {i1, j1} | {a1, b1}
    occ2 = set(occ_spin) - {i2, j2} | {c2, d2} if False else None
    
    # Let's do it differently.
    # Find the orbitals in occ1 but not occ2 (these are "holes" in occ2)
    # and orbitals in occ2 but not occ1 (these are "particles" in occ2)
    
    only1 = occ1 - occ2
    only2 = occ2 - occ1
    k = len(only1)  # = len(only2)
    
    if k == 0:
        # Same determinant - diagonal element
        # E = sum_{p in occ} h_p + 1/2 sum_{pq in occ} <pq||pq>
        # But we don't need the diagonal for the off-diagonal V_mn
        return None
    
    elif k == 1:
        # Single excitation difference
        p = list(only1)[0]  # occupied in 1, not in 2 (hole in 2)
        q = list(only2)[0]  # occupied in 2, not in 1 (particle in 2)
        # So excitation from determinant 1 to 2 is p -> q
        # <1|H|2> = <p|f|q> = eps_q * delta_{p,q}? No, Fock matrix in MO basis is diagonal.
        # Wait, Fock matrix is diagonal in MO basis, so <p|f|q> = eps_p * delta_{pq}
        # But p != q for k=1 (different orbitals), so <p|f|q> = 0?
        # No! That's only true if p and q are both occupied or both virtual?
        # No, the Fock operator is diagonal in the canonical MO basis: f|p> = eps_p|p>
        # So <p|f|q> = eps_p * delta_{pq} for ANY spin orbitals p, q.
        # Since p != q (they differ), <p|f|q> = 0.
        # But wait, Slater-Condon says <Phi|H|Phi'> = <p|h|q> + sum_{i in occ} <pi||qi> = <p|f|q>
        # And f is diagonal in canonical basis, so <p|f|q> = 0 for p != q.
        return 0.0
    
    elif k == 2:
        # Double excitation difference
        # Two orbitals in only1 (p, q) and two in only2 (r, s)
        # Excitation: p,q -> r,s
        # <1|H|2> = <pr||qs> (antisymmetrized integral, physicists notation)
        
        p_list = sorted(list(only1))
        q_list = sorted(list(only2))
        
        # Wait, we need to be careful about the phase.
        # Let's compute the phase by bringing the determinants to maximal coincidence.
        
        # Actually, for the simple case, let's use:
        # if we have two determinants differing by 2 orbitals:
        # p and q are in det1 but not det2
        # r and s are in det2 but not det1
        # We need to figure out which pairs with which.
        
        # The matrix element is <pr||qs> or <ps||qr> depending on ordering.
        # Let's use the standard formula:
        # <Phi|H|Phi'> = <p_a p_b||q_a q_b>
        # where p_a, p_b are the particles/holes and we match them correctly.
        
        # Actually, the simplest approach: 
        # Let p,q be the orbitals unique to det1
        # Let r,s be the orbitals unique to det2
        # Then <det1|H|det2> = <pr||qs> - <ps||qr> ? No.
        
        # From Szabo & Ostlund, Eq. (2.118):
        # If two determinants differ by two spin orbitals, say Psi has i,j and Phi has a,b
        # Then <Psi|H|Phi> = <ij||ab> (antisymmetrized, physicists notation)
        
        # But here the difference is: det1 has p,q but not r,s
        #                    det2 has r,s but not p,q
        # So the excitation from det1 to det2 is p,q -> r,s
        # And <det1|H|det2> = <pr||qs> ? No, <pq||rs> (physicists notation)
        
        # Wait: <Psi|H|Phi> where Psi = |... p q ...> and Phi = |... r s ...>
        # = <pq||rs> (physicists notation antisymmetrized)
        # = (pr|qs) - (ps|qr) (chemists notation)
        
        p, q = p_list[0], p_list[1]
        r, s = q_list[0], q_list[1]
        
        # Phase: we need to make sure the ordering is correct.
        # Let's just use <pq||rs> = (pr|qs) - (ps|qr)
        val = chem_int(p, r, q, s) - chem_int(p, s, q, r)
        return val
    
    else:
        # Differ by more than 2 orbitals -> zero
        return 0.0

# This is getting complicated. Let me try a different approach.
# Let me just directly compute MP3 using the formula with the full CI matrix.

# Actually, let me simplify. For the purposes of validating the formula,
# let me just compute MP3 using the known formula for spatial orbitals.

# From standard references, MP3 energy (spatial orbitals, closed shell):
# E_MP3 = 1/4 sum_{ijabklcd} t2_ijab * (2(ij|kl)(ab|cd) - (ik|jl)(ac|bd) + ...) * t2_klcd / D_ijab
# Hmm, no.

# Let me try a simpler approach: use the 1/4 factor spin-orbital formula
# and just compute it directly with all i,j,a,b,k,l,c,d.

print()
print('Computing MP3 with spin orbital formula...')
print('(This may take a moment...)')

e_mp3 = 0.0
count = 0

# E^(3) = E^(1) * sum_m |V_0m|^2 / D_0m^2 - sum_{m,n} V_0m V_mn V_n0 / (D_0m D_0n)
# where m = (i,j,a,b), n = (k,l,c,d), with 1/4 factor for each?

# Actually, the sum over all i,j,a,b with 1/4 factor is equivalent to
# summing over unique pairs i<j, a<b.

# Let me compute E^(1) first.
# E^(1) = <0|V|0> = E_HF - E0
# E0 = sum_{i in occ_spin} eps_i = 2 * sum_{i in occ} eps_i (spatial)
E0 = 2.0 * np.sum(mo_energy[:nocc])
E1 = rhf.e_tot - E0
print(f'E0 = {E0:.10f}')
print(f'E1 = E_HF - E0 = {E1:.10f}')

# Term 1: E^(1) * sum_m |V_0m|^2 / D_0m^2
# sum_m |V_0m|^2 / D_0m^2 = 1/4 sum_{ijab} |<ij||ab>|^2 / D_ijab^2
term1_part = 0.0
for i in occ_spin:
    for j in occ_spin:
        for a in virt_spin:
            for b in virt_spin:
                denom = eps_spin[i] + eps_spin[j] - eps_spin[a] - eps_spin[b]
                if abs(denom) < 1e-12:
                    continue
                val = antisym_int(i, j, a, b)
                term1_part += 0.25 * val * val / (denom * denom)
term1 = E1 * term1_part
print(f'Term 1 (E1 * sum |V|^2/D^2): {term1:.10f}')

# Term 2: - sum_{m,n} V_0m V_mn V_n0 / (D_0m D_0n)
# This is more complex because we need V_mn = <m|H|n> for doubly excited m,n.

# Let me think about this differently. 
# Actually, let me check if the E^(1) term is part of MP3 or not.
# In some formulations, MP3 is defined differently.

# Actually wait. I think I confused myself. Let me go back to basics.

# In Møller-Plesset perturbation theory:
# E_MP0 = E0 = <0|H0|0>
# E_MP1 = E0 + E1 = E0 + <0|V|0> = <0|H|0> = E_HF
# E_MP2 = E0 + E1 + E2 = E_HF + E2  (so MP2 correlation = E2)
# E_MP3 = E_HF + E2 + E3 (MP3 correlation = E2 + E3)
# E_MP4 = E_HF + E2 + E3 + E4 (MP4 correlation = E2 + E3 + E4)

# Wait no. Let me check:
# E = sum_{n=0}^{inf} E^(n)
# E_MP2 correlation = E^(2)
# E_MP3 correlation = E^(2) + E^(3)
# E_MP4 correlation = E^(2) + E^(3) + E^(4)

# Actually no! The standard definition is:
# MP2 total energy = E_HF + E2
# MP3 total energy = E_HF + E2 + E3
# MP4 total energy = E_HF + E2 + E3 + E4
# where E2, E3, E4 are the 2nd, 3rd, 4th order corrections.

# And the MP3 correction E3 is what we want to compute.

# From my derivation:
# E^(3) = E^(1) * sum_m |V_0m|^2 / D_0m^2 - sum_{m,n} V_0m V_mn V_n0 / (D_0m D_0n)

# But wait, is E^(1) = <0|V|0> included here?
# E^(1) = <0|V|0> = E_HF - E0

# For H2O/sto-3g, E1 is about:
# E_HF = -74.963
# E0 = 2 * sum(eps_occ)
# Let's compute sum(eps_occ):
print(f'Sum of occ MO energies: {np.sum(mo_energy[:nocc]):.10f}')
print(f'E0 (2*sum): {E0:.10f}')
print(f'E1 = E_HF - E0 = {rhf.e_tot - E0:.10f}')

# E1 is positive?
# E_HF = T + V_ne + V_ee (total energy)
# E0 = 2 * sum(eps_i) = 2 * sum(h_i + sum_j (2J_ij - K_ij))
#    = 2*T + 2*V_ne + 4*J - 2*K
# E_HF = 2*T + 2*V_ne + 2*J - K
# So E_HF - E0 = -2*J + K = -(2J - K) < 0
# Yes, E1 should be negative!

# Let me verify:
print()
print('Verifying E1...')
print(f'E_HF = {rhf.e_tot:.10f}')
print(f'E0 = 2*sum(eps_occ) = {E0:.10f}')
print(f'E1 = E_HF - E0 = {rhf.e_tot - E0:.10f}')

# OK so E1 is negative (which makes sense).

# Now let me compute term2 - but this requires the full CI matrix between doubles.
# This is O(N^8) which is huge. But for H2O/sto-3g with 10 occ spin orb and 4 virt spin orb,
# it might be manageable.

# Number of double excitations (unique pairs i<j, a<b):
# C(10,2) * C(4,2) = 45 * 6 = 270
# That's very manageable!

# Let me construct the unique double excitations properly.
from itertools import combinations

unique_dets = []  # list of (i, j, a, b) where i<j (occ) and a<b (virt)
for i, j in combinations(occ_spin, 2):
    for a, b in combinations(virt_spin, 2):
        unique_dets.append((i, j, a, b))

n_unique = len(unique_dets)
print(f'Number of unique double excitations: {n_unique}')

# Now build the H matrix in the doubles space
# H_mn = <m|H|n>

# First compute the reference energy shift
# For each determinant |m>, the energy relative to E0 is:
# E_m = E0 - eps_i - eps_j + eps_a + eps_b
# And <m|V|m> = <m|H|m> - E_m = (E_HF + ...) - E_m ? No.

# Actually, <m|H|m> is the expectation value of H for determinant m.
# But for the MP3 formula, we need <m|V|n> = <m|H|n> - E_m delta_{mn}
# Wait no, V = H - H0, so <m|V|n> = <m|H|n> - <m|H0|n> = <m|H|n> - E_m delta_{mn}
# Because H0|n> = E_n|n> -> <m|H0|n> = E_n delta_{mn}

# So V_mn = H_mn - E_m delta_{mn}

# For off-diagonal (m != n): V_mn = H_mn
# For diagonal (m = n): V_mm = H_mm - E_m

# But wait, in the RS perturbation formula, the Hamiltonian is H = H0 + V.
# And we have E0 = <0|H0|0>, and V = H - H0.
# The matrix element <m|V|n> is what appears in the formulas.

# Actually I realize there's a simpler way to compute MP3.
# Let me just directly compute it using the CI matrix.

# Step 1: Build H matrix in the space of reference + singles + doubles
# But for MP3 with RHF reference, singles don't contribute (Brillouin's theorem).
# So we only need reference + doubles.

# Let me do this:
# 0: reference
# 1..n_unique: double excitations

# H matrix is (1 + n_unique) x (1 + n_unique)
n_states = 1 + n_unique
H_ci = np.zeros((n_states, n_states))

# Diagonal elements
# H[0,0] = E_HF (energy of reference)
H_ci[0, 0] = rhf.e_tot

# For each double excitation m = |ij^ab>:
# E_m = sum_{p in occ_m} h_p + 1/2 sum_{pq in occ_m} <pq||pq>
# But we can also compute it as:
# E_m = E_HF - eps_i - eps_j + eps_a + eps_b + ... wait, that's not right.
# Actually, the Fock matrix is diagonal, but the total energy difference
# between two determinants is not just orbital energy differences because
# of the 1/2 factor in the electron-electron interaction.

# Hmm, this is getting complicated. Let me use a simpler formula.

# Actually, from Szabo & Ostlund Eq. (6.56), MP3 energy is:
# E_MP3 = sum_{ijab,klcd} t2_ijab * V_ijab,klcd * t2_klcd / D_ijab ?
# No, that doesn't look right either.

# Let me just compute MP3 by diagonalizing the full CI (in doubles space)
# and compare with perturbation theory.
print()
print('Skipping full CI for now...')
print('Let me use the standard spatial-orbital MP3 formula instead.')

# From standard references (e.g., Levine "Quantum Chemistry" or
# Szabo & Ostlund "Modern Quantum Chemistry"):
#
# MP3 energy (closed shell, spatial orbitals):
# E_MP3 = 0.5 * sum_{ijkl,abcd} t2_ijab * [2(ij|kl)(ab|cd) - (ik|jl)(ac|bd)] * t2_klcd / D_ijab
#
# Wait, no. That's not right either.

# Let me try yet another approach. Let me look at what PySCF computes.
# But I can't access PySCF.

# OK, let me use the following approach:
# MP3 = sum_{ijab} t2_ijab * (sum_kl <ij||kl> t2_klab + sum_cd <ab||cd> t2_ijcd - 2 sum_kc <ik||ac> t2_jkcb) / D_ijab
# where <ij||kl> = 2(ij|kl) - (ik|jl) ? No, I'm confusing notations.

# I think the safest approach is to implement MP3 using the spin orbital
# formula with the 1/4 factor, and compute V_mn properly.
# Since we only have 270 determinants, this is trivial.

print()
print('Building CI matrix in doubles space...')

# Let me compute H matrix elements between double excitations.
# H_mn = <m|H|n>

# For diagonal elements:
# E_m = E_HF - eps_i - eps_j + eps_a + eps_b + 
#        (<ij||ij> - <ia||ia> - <ib||ib> - <ja||ja> - <jb||jb> + ...) No, that's wrong.

# Actually, there's a simpler way to compute the energy of a doubly excited determinant.
# The difference in energy between |ij^ab> and |0> is:
# Delta_E = -eps_i - eps_j + eps_a + eps_b + <ij||ij> + <ab||ab> - <ia||ia> - <ib||ib> - <ja||ja> - <jb||jb>
# No, that's also not right.

# Let me just compute it using the formula for the expectation value:
# <Phi|H|Phi> = sum_{p in occ} h_p + 1/2 sum_{pq in occ} <pq||pq>
# where the sums are over spin orbitals.

# But we know E_HF = sum_{i in occ} h_i + 1/2 sum_{ij in occ} <ij||ij>
# For the doubly excited determinant |ij^ab>:
# occ_m = (occ \ {i,j}) U {a,b}
# E_m = sum_{p in occ_m} h_p + 1/2 sum_{pq in occ_m} <pq||pq>

# The difference:
# E_m - E_HF = (h_a + h_b - h_i - h_j) 
#              + 1/2 [sum_{pq in occ_m} <pq||pq> - sum_{pq in occ} <pq||pq>]

# The two-electron part difference:
# We remove all terms involving i or j, and add all terms involving a or b.
# But we have to be careful about double-counting.

# Actually, let me just compute E_m directly.
def det_energy(i, j, a, b):
    """Compute energy of determinant |ij^ab> (doubly excited from reference)."""
    # Occupied spin orbitals
    occ_m = set(occ_spin) - {i, j} | {a, b}
    occ_list = sorted(occ_m)
    
    E = 0.0
    # One-electron part
    for p in occ_list:
        sp, pp = p // nbf, p % nbf
        # h_p in MO basis... wait, h_p is the one-electron integral.
        # But we can get it from Fock matrix: f_pq = h_pq + sum_{k in occ} <pk||qk>
        # So h_p = f_pp - sum_{k in occ} <pk||pk> = eps_p - sum_{k in occ} <pk||pk>
        # Hmm, this is getting messy.
        
    # Actually, let's use a different approach.
    # Let me compute h_pq in MO basis.
    pass

# OK, I'm spending too much time on the derivation.
# Let me just use a well-known formula for MP3.

# From "A review of nonrelativistic ab initio analytic second derivatives..." by Gauss & Stanton,
# or from standard textbooks:
#
# MP3 correlation energy (using spin-adapted spatial orbitals):
# E_MP3 = sum_{ijabklcd} t2_ijab * g_ijkl * g_abcd * t2_klcd / D_ijab ... no.
# 
# Actually, let me look at it from the coupled cluster perspective.
# MP3 = <0| H (T2)^2 |0>_connected / 2?
# No, MP2 = <0|H T2|0>
# MP3 = <0|H T2 T2|0>_L? No.

# OK I found it in my memory:
# The Møller-Plesset energy up to third order:
# E = E_HF + E2 + E3
# where
# E2 = 1/4 sum_{ijab} <ij||ab> t2_ijab   (t2_ijab = <ij||ab> / D_ijab)
# E3 = 1/4 sum_{ijab,klcd} t2_ijab <ij||kl><cd||ab> t2_klcd / D_ijab? No.

# Wait, let me think dimensionally.
# E2 has units of energy (since <ij||ab>^2 / D has units of energy)
# E3 should also have units of energy.
# t2 has units of 1/energy (since t2 = V / D)
# V * t2 * V * t2 / D has units V^3 / D^3? No.

# Actually:
# V_0m has units of energy
# D_0m has units of energy
# So V^3 / D^2 has units of energy. That's E3.

# The formula E3 = sum_{m,n} V_0m V_mn V_n0 / (D_0m D_0n) has units energy^3 / energy^2 = energy.
# And the E^(1) term: E1 * sum |V|^2 / D^2 has units energy * energy^2 / energy^2 = energy.
# So both terms have correct units.

# OK, let me just implement the full CI approach with 270 states.
# It's easy and will give us the correct MP3 energy.

print()
print('Building reference + doubles CI matrix...')

n_states = 1 + n_unique
H_ci = np.zeros((n_states, n_states))

# We need h_pq (one-electron integrals in MO basis)
from calchem.integrals.one_electron import compute_kinetic, compute_nuclear
T_ao = compute_kinetic(rhf.basis_set, mol)
Vn_ao = compute_nuclear(rhf.basis_set, mol)
h_ao = T_ao + Vn_ao
h_mo = mo_coeff.T @ h_ao @ mo_coeff

# One-electron spin orbital
def h_spin(p, q):
    sp, pp = p // nbf, p % nbf
    sq, pq = q // nbf, q % nbf
    if sp != sq:
        return 0.0
    return h_mo[pp, pq]

# Antisymmetrized two-electron integral (physicists notation)
def g_phys(p, q, r, s):
    """<pq||rs> = (pr|qs) - (ps|qr)"""
    return chem_int(p, r, q, s) - chem_int(p, s, q, r)

def det_energy_full(occ_list):
    """Compute full energy of a determinant with given occupied spin orbitals."""
    E = 0.0
    # One-electron
    for p in occ_list:
        E += h_spin(p, p)
    # Two-electron
    for idx_p, p in enumerate(occ_list):
        for idx_q, q in enumerate(occ_list):
            E += 0.5 * g_phys(p, q, p, q)
    return E

# Reference energy
ref_occ = sorted(occ_spin)
E_ref = det_energy_full(ref_occ)
print(f'Reference energy (from det): {E_ref:.10f}')
print(f'E_HF (from SCF):      {rhf.e_tot:.10f}')
print(f'Difference: {E_ref - rhf.e_tot:.2e}')

# They should match! If not, my integrals are wrong.
if abs(E_ref - rhf.e_tot) > 1e-8:
    print('WARNING: Reference energy mismatch!')

# H_ci[0,0]
H_ci[0, 0] = E_ref

# For each double excitation
for m_idx, (i, j, a, b) in enumerate(unique_dets):
    m_state = 1 + m_idx
    # Occupied spin orbitals of det m
    occ_m = sorted(set(occ_spin) - {i, j} | {a, b})
    E_m = det_energy_full(occ_m)
    H_ci[m_state, m_state] = E_m
    
    # Matrix element with reference: <0|H|m> = <ij||ab> (antisym, phys)
    # But with phase factor!
    # What's the phase of |ij^ab> relative to |0>?
    # Standard convention: remove i, j (in order) and add a, b (in order)
    # Phase = (-1)^{pos(i) + pos(j) + pos(a) + pos(b)} ???
    # Actually, the phase depends on the ordering of orbitals.
    
    # For now, let's just use <0|H|m> = <ij||ab> 
    # (which is what appears in MP2 with the 1/4 factor)
    # But wait, the phase matters for the sign.
    
    # Actually, in the spin orbital MP2 formula:
    # E2 = 1/4 sum_{ijab} <ij||ab>^2 / D_ijab
    # This uses the convention where both i,j and a,b are summed over all pairs,
    # and the 1/4 factor accounts for double counting (i<->j and a<->b).
    
    # The matrix element <0|H|ij^ab> = <ij||ab> (up to sign)
    # But when we sum over all i,j,a,b with the 1/4 factor, we get the right answer.
    
    # For the unique determinant basis (i<j, a<b), the matrix element is:
    # <0|H|m> = <ij||ab>  (with appropriate phase)
    # And E2 = sum_m |<0|H|m>|^2 / D_m  (no 1/4 factor needed for unique dets)
    
    # Let me verify this by computing MP2 from the CI matrix.
    
    # First, compute the matrix element between 0 and m
    # Using Slater-Condon: if det m differs from 0 by 2 spin orbitals (i,j -> a,b),
    # then <0|H|m> = <ia||jb> ??? No, <ij||ab>.
    
    # Wait, Szabo & Ostlund: <Psi|H|Phi> where Psi has i,j and Phi has a,b
    # = <ij||ab> (physicists notation, antisymmetrized)
    
    # But the phase depends on the positions of i,j in the determinant.
    # If i and j are the first two orbitals: phase = +1
    # Otherwise, we get a phase factor of (-1)^{...}
    
    # For simplicity, let's assume all matrix elements have positive sign
    # and just verify MP2 matches.
    val = antisym_int(i, j, a, b)
    H_ci[0, m_state] = val
    H_ci[m_state, 0] = val

# Now compute MP2 from CI matrix
# E2 = sum_{m>0} |H_0m|^2 / (E_0 - E_m)
E2_ci = 0.0
for m_state in range(1, n_states):
    D = H_ci[0, 0] - H_ci[m_state, m_state]  # E0 - Em
    if abs(D) < 1e-12:
        continue
    E2_ci += H_ci[0, m_state] ** 2 / D

print(f'MP2 from CI (unique dets): {E2_ci:.10f}')
print(f'MP2 from spin-orbit sum:  {e_mp2:.10f}')
print(f'MP2 from RMP2 module:    -0.0355028412')

# OK this should tell us if the phases/signs are correct.
