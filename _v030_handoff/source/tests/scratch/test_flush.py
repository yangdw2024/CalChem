import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

print("Step 1: numpy", flush=True)
import numpy as np

print("Step 2: Molecule", flush=True)
from calchem.mol.molecule import Molecule

print("Step 3: BasisSet", flush=True)
from calchem.basis.basis_set import BasisSet

print("Step 4: one_electron", flush=True)
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear

print("Step 5: two_electron", flush=True)
from calchem.integrals.two_electron import compute_eri

print("Step 6: scf.guess", flush=True)
from calchem.scf.guess import _symmetric_orthogonalize, guess_hcore

print("Step 7: scf.hf", flush=True)
from calchem.scf.hf import build_fock_rhf, scf_energy_rhf, nuclear_repulsion

print("All imports done!", flush=True)
