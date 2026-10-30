import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

print("Test: np.linalg.eigh...", flush=True)
S = np.array([[1.0, 0.6593], [0.6593, 1.0]])
print(f"S = \n{S}", flush=True)

print("Calling eigh...", flush=True)
eigval, eigvec = np.linalg.eigh(S)
print(f"eigval = {eigval}", flush=True)
print(f"eigvec = \n{eigvec}", flush=True)

print("Computing diag...", flush=True)
d = np.diag(1.0 / np.sqrt(eigval))
print(f"diag = \n{d}", flush=True)

print("Computing X = eigvec @ diag @ eigvec.T...", flush=True)
x = eigvec @ d @ eigvec.T
print(f"X = \n{x}", flush=True)

print("Verifying X^T S X...", flush=True)
print(x.T @ S @ x, flush=True)

print("\nAll done!", flush=True)
