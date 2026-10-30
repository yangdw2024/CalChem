import sys
import numpy as np

print("numpy:", np.__version__, flush=True)

print("Test np.linalg.eig (not eigh)...", flush=True)
S = np.array([[1.0, 0.6593], [0.6593, 1.0]])
eigval, eigvec = np.linalg.eig(S)
print(f"eigval = {eigval}", flush=True)
print(f"eigvec = \n{eigvec}", flush=True)

print("\nTest scipy.linalg.eigh...", flush=True)
try:
    from scipy.linalg import eigh
    eigval2, eigvec2 = eigh(S)
    print(f"eigval = {eigval2}", flush=True)
    print(f"eigvec = \n{eigvec2}", flush=True)
except ImportError:
    print("scipy not available", flush=True)

print("\nAll done!", flush=True)
