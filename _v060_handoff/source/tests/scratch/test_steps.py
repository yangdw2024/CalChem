import sys
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

print("Step 1: import numpy...")
import numpy as np
print("  OK")

print("Step 2: import Molecule...")
from calchem.mol.molecule import Molecule
print("  OK")

print("Step 3: import BasisSet...")
from calchem.basis.basis_set import BasisSet
print("  OK")

print("Step 4: import integrals...")
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
print("  OK")

print("Step 5: create molecule...")
mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], units="bohr")
print("  OK")

print("Step 6: create basis...")
basis = BasisSet("sto-3g")
print("  OK")

print("Step 7: compute overlap...")
S = compute_overlap(basis, mol)
print("  OK, S =", S.shape)

print("Step 8: compute kinetic...")
T = compute_kinetic(basis, mol)
print("  OK")

print("Step 9: compute nuclear...")
V = compute_nuclear(basis, mol)
print("  OK")

print("Step 10: compute ERI...")
eri = compute_eri(basis, mol)
print("  OK, ERI shape =", eri.shape)

print("Step 11: import guess...")
from calchem.scf.guess import _symmetric_orthogonalize, guess_hcore
print("  OK")

print("Step 12: orthogonalize...")
x = _symmetric_orthogonalize(S)
print("  OK")

print("Step 13: initial guess...")
dm = guess_hcore(T+V, S, 2, restricted=True)
print("  OK, dm =")
print(dm)

print("Step 14: import RHF...")
from calchem.scf.hf import RHF
print("  OK")

print("Step 15: create RHF object...")
mf = RHF(mol, basis="sto-3g", max_cycles=5, conv_tol=1e-4, verbose=0)
print("  OK")

print("Step 16: build integrals...")
mf.build()
print("  OK")

print("Step 17: run kernel...")
try:
    e = mf.kernel()
    print("  OK, E =", e)
except Exception as ex:
    print("  FAILED:", ex)
    import traceback
    traceback.print_exc()

print("\nAll done!")
