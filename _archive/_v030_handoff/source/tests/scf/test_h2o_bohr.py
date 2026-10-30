import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from scipy.linalg import eigh

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.hf import nuclear_repulsion

print("测试 H2O STO-3G (bohr 坐标)", flush=True)

# H2O 实验键长，转换为 bohr
# O-H 键长 0.9572 Å = 1.809 bohr
# H-O-H 键角 104.5°
r_oh = 0.9572 * 1.88972613  # bohr
angle = 104.5 * np.pi / 180.0

z = r_oh * np.cos(angle / 2)
y = r_oh * np.sin(angle / 2)

mol = Molecule(
    ["O", "H", "H"],
    [
        [0.0, 0.0, 0.0],
        [0.0, -y, z],
        [0.0, y, z],
    ],
    units="bohr",
)

basis = BasisSet("sto-3g")

S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
H = T + V

print(f"基函数数目: {S.shape[0]}", flush=True)
print(f"T 迹: {np.trace(T):.6f}", flush=True)
print(f"V 迹: {np.trace(V):.6f}", flush=True)

eigval, eigvec = eigh(H, S)
print(f"\nHcore 轨道能量:", flush=True)
for i, e in enumerate(eigval):
    print(f"  {i+1}: {e:12.6f}", flush=True)

nocc = 5
dm = 2.0 * np.einsum("ik,jk->ij", eigvec[:, :nocc], eigvec[:, :nocc])
e_1e = np.sum(dm * H)
print(f"\n单电子能量: {e_1e:.6f}", flush=True)

eri = compute_eri(basis, mol)
j = np.einsum("mnls,ls->mn", eri, dm)
k = np.einsum("mlns,ls->mn", eri, dm)
g = j - 0.5 * k
e_2e = 0.5 * np.sum(dm * g)
print(f"双电子能量: {e_2e:.6f}", flush=True)

e_nn = nuclear_repulsion(mol)
print(f"核排斥能: {e_nn:.6f}", flush=True)

e_tot = e_1e + e_2e + e_nn
print(f"总能量 (一步): {e_tot:.6f}", flush=True)
print(f"参考值: ~ -74.94 Hartree", flush=True)
