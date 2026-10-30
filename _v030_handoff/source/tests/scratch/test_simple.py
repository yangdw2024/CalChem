print("Hello from Python!")
import sys
print("Python version:", sys.version)

try:
    import numpy as np
    print("numpy version:", np.__version__)
except Exception as e:
    print("numpy import error:", e)

try:
    sys.path.insert(0, r"d:\YDW\Trae_Gaussian")
    from calchem.mol.molecule import Molecule
    print("Molecule import OK")
except Exception as e:
    print("Molecule import error:", e)
    import traceback
    traceback.print_exc()
