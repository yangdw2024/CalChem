import sys
import numpy as np
print("numpy:", np.__version__, flush=True)

print("Test: np.dot with 2x2...", flush=True)
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
try:
    c = np.dot(a, b)
    print(f"  result = \n{c}", flush=True)
    print("  OK!", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("Test: np.matmul...", flush=True)
try:
    c = np.matmul(a, b)
    print(f"  result = \n{c}", flush=True)
    print("  OK!", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("Test: a @ b (operator)...", flush=True)
try:
    c = a @ b
    print(f"  result = \n{c}", flush=True)
    print("  OK!", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("\nDone!", flush=True)
