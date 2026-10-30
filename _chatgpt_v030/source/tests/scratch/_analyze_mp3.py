"""
分析 MP4 模块中 MP3 公式的问题。
"""

import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.post_hf.mp4 import MP4

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
rhf = RHF(mol, 'sto-3g', verbose=0)
rhf.kernel()
print(f'HF: {rhf.e_tot:.10f}')

mp4 = MP4(rhf, verbose=0)
# We need to compute the internal quantities
# Let's manually do what kernel does, but more

# Replicate the internal computation
mo_coeff = rhf.mo_coeff
mo_energy = rhf.mo_energy
nocc = rhf.mol.nelectron // 2
nbf = mo_coeff.shape[0]
nvirt = nbf - nocc

eri_ao = mp4._compute_eri()
eri_iajb = mp4._transform_eri_iajb(eri_ao, mo_coeff, nocc)
eri_full = mp4._transform_eri_full(eri_ao, mo_coeff)

o = slice(0, nocc)
v = slice(nocc, nocc + nvirt)

eri_oooo = eri_full[o, o, o, o]
eri_ooov = eri_full[o, o, o, v]
eri_oovv = eri_full[o, o, v, v]
eri_ovov = eri_full[o, v, o, v]
eri_ovvv = eri_full[o, v, v, v]
eri_vvvv = eri_full[v, v, v, v]

e_i = mo_energy[:nocc]
e_a = mo_energy[nocc:]

d_ijab = e_i[:, None, None, None] + e_i[None, :, None, None] - \
         e_a[None, None, :, None] - e_a[None, None, None, :]

t2 = eri_iajb / d_ijab

print(f'nocc={nocc}, nvirt={nvirt}')

# MP2 check
e_mp2 = np.einsum('ijab,ijab->', eri_iajb, 2.0 * t2 - t2.swapaxes(2, 3))
print(f'MP2: {e_mp2:.10f}')

# Now compute our current MP3
def mp3_current(t2, eri_oooo, eri_vvvv, eri_oovv):
    nocc = t2.shape[0]
    nvirt = t2.shape[2]
    
    part1 = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    val = 0.0
                    for k in range(nocc):
                        for l in range(nocc):
                            g_ijkl = 2.0 * eri_oooo[i, j, k, l] - eri_oooo[i, k, j, l]
                            val += g_ijkl * t2[k, l, a, b]
                    part1 += t2[i, j, a, b] * val
    
    part2 = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    val = 0.0
                    for c in range(nvirt):
                        for d in range(nvirt):
                            g_abcd = 2.0 * eri_vvvv[a, b, c, d] - eri_vvvv[a, c, b, d]
                            val += g_abcd * t2[i, j, c, d]
                    part2 += t2[i, j, a, b] * val
    
    part3 = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    val = 0.0
                    for k in range(nocc):
                        for c in range(nvirt):
                            g_ikac = 2.0 * eri_oovv[i, k, a, c] - eri_oovv[i, k, c, a]
                            val += g_ikac * t2[k, j, c, b]
                    part3 += t2[i, j, a, b] * val
    part3 *= -2.0
    
    return part1, part2, part3

p1, p2, p3 = mp3_current(t2, eri_oooo, eri_vvvv, eri_oovv)
print(f'\nCurrent MP3:')
print(f'  part1 (oooo): {p1:.10f}')
print(f'  part2 (vvvv): {p2:.10f}')
print(f'  part3 (ovov): {p3:.10f}')
print(f'  total: {p1+p2+p3:.10f}')

# Now let's try different formulas
# Idea: MP3 = sum_ijab t2_ijab * W_ijab / D_ijab?
# Or MP3 = sum_ijab W_ijab * t2_ijab where W includes denominators?

# Let's try: MP3 = sum_ijab (part1+part2+part3)_ijab * t2_ijab / D_ijab
W = np.zeros_like(t2)
# oooo part
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvirt):
            for b in range(nvirt):
                val = 0.0
                for k in range(nocc):
                    for l in range(nocc):
                        g_ijkl = 2.0 * eri_oooo[i, j, k, l] - eri_oooo[i, k, j, l]
                        val += g_ijkl * t2[k, l, a, b]
                W[i, j, a, b] += val

# vvvv part
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvirt):
            for b in range(nvirt):
                val = 0.0
                for c in range(nvirt):
                    for d in range(nvirt):
                        g_abcd = 2.0 * eri_vvvv[a, b, c, d] - eri_vvvv[a, c, b, d]
                        val += g_abcd * t2[i, j, c, d]
                W[i, j, a, b] += val

# ovov part
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvirt):
            for b in range(nvirt):
                val = 0.0
                for k in range(nocc):
                    for c in range(nvirt):
                        g_ikac = 2.0 * eri_oovv[i, k, a, c] - eri_oovv[i, k, c, a]
                        val += g_ikac * t2[k, j, c, b]
                W[i, j, a, b] += -2.0 * val

# Try MP3 = sum (2*W - W.swap) * t2 (like MP2 but with W instead of eri)
e_mp3_v1 = np.einsum('ijab,ijab->', W, 2.0 * t2 - t2.swapaxes(2, 3))
print(f'\nMP3 v1 (W * (2t2 - t2.swap)): {e_mp3_v1:.10f}')

# Try MP3 = sum W * t2 / D
e_mp3_v2 = np.sum(W * t2 / d_ijab)
print(f'MP3 v2 (sum W*t2/D): {e_mp3_v2:.10f}')

# Try MP3 = sum (2*W - W.swap) * t2 / D
e_mp3_v3 = np.einsum('ijab,ijab->', 2.0*W - W.swapaxes(2,3), t2 / d_ijab)
print(f'MP3 v3 (sum (2W-Wswap)*t2/D): {e_mp3_v3:.10f}')

# Try: MP3 = 0.5 * sum (2*W - W.swap) * t2
e_mp3_v4 = 0.5 * np.einsum('ijab,ijab->', W, 2.0*t2 - t2.swapaxes(2,3))
print(f'MP3 v4 (0.5 * W*(2t2-t2swap)): {e_mp3_v4:.10f}')

# From CI, correct MP3 is about -0.0078
print(f'\nExpected MP3 (from CI): ~ -0.00784')

# Let's also check the E1 term
E0 = 2.0 * np.sum(e_i)
E1 = rhf.e_tot - E0
print(f'\nE0 (2*sum eps_occ): {E0:.10f}')
print(f'E1 (E_HF - E0): {E1:.10f}')

# term2 = E1 * sum |V|^2 / D^2
# V = 2*eri - eri.swap? No, V_ijab = 2(ia|jb) - (ib|ja)
V_ijab = 2.0 * eri_iajb - eri_iajb.swapaxes(2, 3)
term2 = E1 * np.sum(V_ijab**2 / d_ijab**2)
print(f'term2 (E1 * sum V^2/D^2): {term2:.10f}')
