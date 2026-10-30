import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri

print("开始测试...")

mol = Molecule(
    ["H", "H"],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
    units="bohr",
)

basis = BasisSet("sto-3g")

print("分子创建成功")
print("基组创建成功")

t0 = time.time()
S = compute_overlap(basis, mol)
print(f"重叠矩阵计算完成，耗时 {time.time()-t0:.3f}s")
print("S =")
print(S)

t0 = time.time()
T = compute_kinetic(basis, mol)
print(f"动能矩阵计算完成，耗时 {time.time()-t0:.3f}s")

t0 = time.time()
V = compute_nuclear(basis, mol)
print(f"核吸引矩阵计算完成，耗时 {time.time()-t0:.3f}s")

Hcore = T + V
print("Hcore =")
print(Hcore)

print()
print("计算 ERI...")
t0 = time.time()
eri = compute_eri(basis, mol)
print(f"ERI 计算完成，耗时 {time.time()-t0:.3f}s，形状: {eri.shape}")
print("(11|11) =", eri[0,0,0,0])
print("(11|22) =", eri[0,0,1,1])
print("(12|12) =", eri[0,1,0,1])
