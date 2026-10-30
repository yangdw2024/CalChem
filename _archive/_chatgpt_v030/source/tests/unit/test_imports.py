import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

print("Testing imports step by step...")

print("1. numpy...")
import numpy as np
print("   OK")

print("2. mol.molecule...")
from calchem.mol.molecule import Molecule
print("   OK")

print("3. basis.basis_set...")
from calchem.basis.basis_set import BasisSet
print("   OK")

print("4. integrals...")
from calchem.integrals.one_electron import compute_overlap
from calchem.integrals.two_electron import compute_eri
print("   OK")

print("5. scf.diis...")
from calchem.scf.diis import DIIS
print("   OK")

print("6. scf.guess...")
from calchem.scf.guess import guess_hcore, _symmetric_orthogonalize
print("   OK")

print("7. scf.results...")
from calchem.scf.results import SCFResult
print("   OK")

print("8. scf.hf (RHF)...")
from calchem.scf.hf import RHF
print("   OK")

print("\nAll imports successful!")
