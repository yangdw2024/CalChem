import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

print("Test 1: import...")
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
print("  OK")

print("Test 2: build integrals...")
mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], units="bohr")
basis = BasisSet("sto-3g")
S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
Hcore = T + V
eri = compute_eri(basis, mol)
print(f"  OK, S shape: {S.shape}, ERI shape: {eri.shape}")

print("Test 3: _symmetric_orthogonalize...")
from calchem.scf.guess import _symmetric_orthogonalize
x = _symmetric_orthogonalize(S)
print(f"  OK, X shape: {x.shape}")

print("Test 4: guess_hcore...")
from calchem.scf.guess import guess_hcore
dm = guess_hcore(Hcore, S, 2, restricted=True)
print(f"  OK, D =\n{dm}")

print("Test 5: build_fock_rhf...")
from calchem.scf.hf import build_fock_rhf
fock = build_fock_rhf(Hcore, dm, eri)
print(f"  OK, F =\n{fock}")

print("Test 6: scf_energy_rhf...")
from calchem.scf.hf import scf_energy_rhf, nuclear_repulsion
e_nn = nuclear_repulsion(mol)
e = scf_energy_rhf(Hcore, dm, fock) + e_nn
print(f"  OK, E = {e:.10f}")

print("\nAll tests passed!")
