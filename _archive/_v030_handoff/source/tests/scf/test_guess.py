import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

print("Step 1: test scipy eigh directly...", flush=True)
from scipy.linalg import eigh
S = np.array([[1.0, 0.6593], [0.6593, 1.0]])
eigval, eigvec = eigh(S)
print(f"  eigval = {eigval}", flush=True)
print(f"  OK!", flush=True)

print("Step 2: import guess module...", flush=True)
from calchem.scf.guess import _symmetric_orthogonalize, _eigh_safe
print("  OK!", flush=True)

print("Step 3: test _eigh_safe...", flush=True)
eigval2, eigvec2 = _eigh_safe(S)
print(f"  eigval = {eigval2}", flush=True)
print("  OK!", flush=True)

print("Step 4: test _symmetric_orthogonalize...", flush=True)
x = _symmetric_orthogonalize(S)
print(f"  X = \n{x}", flush=True)
print("  OK!", flush=True)

print("\nAll done!", flush=True)
