import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from scipy.linalg import eigh

print("Test 1: np.linalg.solve...", flush=True)
a = np.array([[2.0, 1.0], [1.0, 2.0]])
b = np.array([1.0, 2.0])
try:
    x = np.linalg.solve(a, b)
    print(f"  result = {x}", flush=True)
    print("  OK!", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("\nTest 2: DIIS simulation...", flush=True)
from calchem.scf.diis import DIIS
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.hf import build_fock_rhf, _matmul
from calchem.scf.guess import guess_hcore, _symmetric_orthogonalize

mol = Molecule(["O", "H", "H"],
    [[0,0,0],[0,-0.757,0.586],[0,0.757,0.586]],
    units="angstrom")
basis = BasisSet("sto-3g")
S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
Hcore = T + V
eri = compute_eri(basis, mol)
print("  integrals done", flush=True)

dm = guess_hcore(Hcore, S, 10, restricted=True)
fock = build_fock_rhf(Hcore, dm, eri)
print("  initial fock done", flush=True)

diis = DIIS(ndim=6)
print("  DIIS created", flush=True)

print("  adding to DIIS...", flush=True)
diis.add(fock, dm, S)
print("  add 1 done", flush=True)

x = _symmetric_orthogonalize(S)
f_ortho = np.einsum("ik,kj->ij", np.einsum("ik,kj->ij", x.T, fock), x)
eigval, eigvec_ortho = eigh(f_ortho)
mo_coeff = np.einsum("ik,kj->ij", x, eigvec_ortho)
nocc = 5
dm_new = 2.0 * np.einsum("ik,jk->ij", mo_coeff[:, :nocc], mo_coeff[:, :nocc])
fock_new = build_fock_rhf(Hcore, dm_new, eri)
print("  iteration 1 done", flush=True)

print("  adding 2 to DIIS...", flush=True)
diis.add(fock_new, dm_new, S)
print("  add 2 done", flush=True)

print("  extrapolating...", flush=True)
f_ext = diis.extrapolate()
print(f"  extrapolate done, f_ext shape = {f_ext.shape}", flush=True)

print("\nAll DIIS tests passed!", flush=True)
