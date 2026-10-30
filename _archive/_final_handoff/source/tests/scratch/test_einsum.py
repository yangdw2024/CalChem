import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

print("numpy version:", np.__version__)

print("Test 1: simple einsum...")
a = np.random.rand(5, 5)
b = np.random.rand(5, 5, 5, 5)
try:
    c = np.einsum("ls,mnls->mn", a, b)
    print("  OK, shape:", c.shape)
except Exception as e:
    print("  FAILED:", e)

print("Test 2: another einsum...")
try:
    d = np.einsum("ls,mlns->mn", a, b)
    print("  OK, shape:", d.shape)
except Exception as e:
    print("  FAILED:", e)

print("Test 3: small 2x2x2x2...")
eri = np.random.rand(2, 2, 2, 2)
dm = np.random.rand(2, 2)
try:
    j = np.einsum("ls,mnls->mn", dm, eri)
    print("  J OK:", j.shape)
    k = np.einsum("ls,mlns->mn", dm, eri)
    print("  K OK:", k.shape)
except Exception as e:
    print("  FAILED:", e)
    import traceback
    traceback.print_exc()

print("\nAll einsum tests done!")
