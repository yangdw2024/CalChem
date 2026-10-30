"""调试ERI转换"""

import numpy as np
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.two_electron import compute_eri

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords).to_bohr()
basis_set = BasisSet('sto-3g')

eri_ao = compute_eri(basis_set, mol)
nbf = eri_ao.shape[0]

print(f"ERI AO shape: {eri_ao.shape}")
print(f"ERI AO[0,0,0,0]: {eri_ao[0,0,0,0]}")
print(f"ERI AO[0,0,0,1]: {eri_ao[0,0,0,1]}")
print(f"ERI AO[0,1,0,1]: {eri_ao[0,1,0,1]}")
print(f"ERI AO[0,0,1,1]: {eri_ao[0,0,1,1]}")
print(f"ERI AO[0,1,1,0]: {eri_ao[0,1,1,0]}")

print("\n检查对称性:")
print(f"(0,0|0,0) == (0,0||0,0): {abs(eri_ao[0,0,0,0] - eri_ao[0,0,0,0]) < 1e-15}")
print(f"(0,1|0,1) == (0,1||1,0): {abs(eri_ao[0,1,0,1] - eri_ao[0,1,1,0]) < 1e-15}")
print(f"(0,0|1,1) == (1,1|0,0): {abs(eri_ao[0,0,1,1] - eri_ao[1,1,0,0]) < 1e-15}")

from calchem.scf.hf import RHF
rhf = RHF(mol, 'sto-3g', verbose=0)
rhf.kernel()
mo_coeff = rhf.mo_coeff

nocc = mol.nelectron // 2
nvirt = nbf - nocc
o = slice(0, nocc)
v = slice(nocc, nocc + nvirt)

tmp1 = np.einsum("pi,pqrs->iqrs", mo_coeff, eri_ao)
tmp2 = np.einsum("qj,iqrs->ijrs", mo_coeff, tmp1)
tmp3 = np.einsum("rk,ijrs->ijks", mo_coeff, tmp2)
eri_mo = np.einsum("sl,ijks->ijkl", mo_coeff, tmp3)

print("\nMO基ERI:")
print(f"eri_mo[0,0,0,0]: {eri_mo[0,0,0,0]}")
print(f"eri_mo[0,0,nocc,nocc]: {eri_mo[0,0,nocc,nocc]}")

eri_oovv = eri_mo[o, o, v, v]
print(f"\neri_oovv shape: {eri_oovv.shape}")
print(f"eri_oovv[0,0,0,0]: {eri_oovv[0,0,0,0]}")
print(f"eri_oovv[0,0,0,1]: {eri_oovv[0,0,0,1]}")
print(f"eri_oovv[0,0,1,0]: {eri_oovv[0,0,1,0]}")

v_oovv = eri_oovv - eri_oovv.swapaxes(2, 3)
print(f"\nv_oovv[0,0,0,0]: {v_oovv[0,0,0,0]}")
print(f"v_oovv[0,0,0,1]: {v_oovv[0,0,0,1]}")

print("\n检查AO基MO系数:")
print(f"MO系数列范数: {np.linalg.norm(mo_coeff, axis=0)}")
