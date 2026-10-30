import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from scipy.linalg import eigh

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear

mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")
S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
H = T + V

print("重叠矩阵 S:")
print(np.array2string(S, precision=4, suppress_small=True))

print("\n动能矩阵 T:")
print(np.array2string(T, precision=4, suppress_small=True))

print("\n核吸引矩阵 V:")
print(np.array2string(V, precision=4, suppress_small=True))

print("\nHcore 矩阵 H:")
print(np.array2string(H, precision=4, suppress_small=True))

eigval, eigvec = eigh(H, S)
print(f"\nHcore 本征值: {eigval}")

# 手动验证第一个本征向量
print(f"\n第一个本征向量: {eigvec[:, 0]}")
# 计算 H * c1
Hc = H @ eigvec[:, 0]
Sc = S @ eigvec[:, 0]
print(f"H*c / S*c = {Hc / Sc}")
