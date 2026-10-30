"""验证MP2能量"""

import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.integrals.two_electron import compute_eri

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords).to_bohr()

rhf = RHF(mol, 'sto-3g', verbose=0)
rhf.kernel()

mo_coeff = rhf.mo_coeff
mo_energy = rhf.mo_energy
nocc = mol.nelectron // 2
nbf = mo_coeff.shape[0]
nvirt = nbf - nocc

C_occ = mo_coeff[:, :nocc]
C_vir = mo_coeff[:, nocc:]
eri = compute_eri(rhf.basis_set, mol)

print(f"HF能量: {rhf.e_tot}")
print(f"nocc={nocc}, nvirt={nvirt}")

eri_iajb = np.einsum("pi,qa,rj,sb,pqrs->ijab", C_occ, C_vir, C_occ, C_vir, eri)

e_i = mo_energy[:nocc]
e_a = mo_energy[nocc:]
de = e_i[:, None, None, None] + e_i[None, :, None, None] - e_a[None, None, :, None] - e_a[None, None, None, :]

t_iajb = eri_iajb / de
e_mp2 = np.einsum("ijab,ijab->", eri_iajb, 2.0 * t_iajb - t_iajb.swapaxes(2, 3))

print(f"\nMP2相关能: {e_mp2}")
print(f"MP2总能量: {rhf.e_tot + e_mp2}")

v_oovv = eri_iajb - eri_iajb.swapaxes(2, 3)
e_mp2_v2 = 0.25 * np.einsum('ijab,ijab->', t_iajb, v_oovv)
print(f"\nMP2相关能 (使用v_oovv): {e_mp2_v2}")

print(f"\n对比:")
print(f"  标准MP2: {e_mp2}")
print(f"  v_oovv方式: {e_mp2_v2}")
print(f"  差异: {abs(e_mp2 - e_mp2_v2)}")

from calchem.post_hf.mp2 import RMP2
mp2 = RMP2(rhf)
mp2.kernel()
print(f"\nRMP2模块结果: {mp2.e_corr}")
