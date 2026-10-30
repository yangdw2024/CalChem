import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from scipy.linalg import eigh

S = np.array([[1.0, 0.6593], [0.6593, 1.0]])
print("S =", S, flush=True)

print("Step A: eigh...", flush=True)
eigval, eigvec = eigh(S)
print(f"  eigval = {eigval}", flush=True)
print(f"  eigvec = \n{eigvec}", flush=True)

print("Step B: sqrt...", flush=True)
inv_sqrt = 1.0 / np.sqrt(eigval)
print(f"  inv_sqrt = {inv_sqrt}", flush=True)

print("Step C: diag...", flush=True)
d = np.diag(inv_sqrt)
print(f"  diag = \n{d}", flush=True)

print("Step D: first matmul...", flush=True)
temp = eigvec @ d
print(f"  temp = \n{temp}", flush=True)

print("Step E: second matmul...", flush=True)
x = temp @ eigvec.T
print(f"  X = \n{x}", flush=True)

print("\nAll done!", flush=True)
