"""对比HF和DFT的SCF行为"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.scf.dft import RKS
from calchem.basis.basis_set import BasisSet

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
basis_set = BasisSet('sto-3g', mol)

print("=" * 60)
print("RHF SCF")
print("=" * 60)
hf = RHF(mol, basis='sto-3g', verbose=1)
hf.build()
e_hf = hf.kernel()

print("\n" + "=" * 60)
print("LDA SCF")
print("=" * 60)
lda = RKS(mol, basis='sto-3g', xc='lda', grid_level='coarse', verbose=1, diis=True, diis_start=3, max_cycles=30)
lda.build()
e_lda = lda.kernel()

print("\n" + "=" * 60)
print("PBE SCF")
print("=" * 60)
pbe = RKS(mol, basis='sto-3g', xc='pbe', grid_level='coarse', verbose=1, diis=True, diis_start=3, max_cycles=30)
pbe.build()
e_pbe = pbe.kernel()

print(f"\nSummary:")
print(f"  RHF: {e_hf:.10f} Eh")
print(f"  LDA: {e_lda:.10f} Eh")
print(f"  PBE: {e_pbe:.10f} Eh")
