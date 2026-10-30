import sys
import numpy as np
print("numpy:", np.__version__, flush=True)

print("Test 1: 2x2 @ 2x2 matmul...", flush=True)
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
c = a @ b
print(f"  result = \n{c}", flush=True)
print("  OK!", flush=True)

print("Test 2: from scipy.linalg import eigh...", flush=True)
from scipy.linalg import eigh
print("  OK!", flush=True)

print("Test 3: eigh + matmul...", flush=True)
S = np.array([[1.0, 0.6593], [0.6593, 1.0]])
eigval, eigvec = eigh(S)
print(f"  eigh done", flush=True)
d = np.diag(1.0 / np.sqrt(eigval))
print(f"  diag done", flush=True)

print("  doing eigvec @ d...", flush=True)
temp = eigvec @ d
print(f"  temp = \n{temp}", flush=True)

print("  doing temp @ eigvec.T...", flush=True)
x = temp @ eigvec.T
print(f"  X = \n{x}", flush=True)

print("\nAll done!", flush=True)
