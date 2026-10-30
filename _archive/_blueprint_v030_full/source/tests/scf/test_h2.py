import sys
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import time
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.hf import RHF

print("=" * 60)
print("测试 H2 STO-3G RHF (R = 1.4 bohr)")
print("=" * 60)

mol = Molecule(
    ["H", "H"],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
    units="bohr",
)

print("\n步骤 1: 计算单电子积分...")
t0 = time.time()
basis = BasisSet("sto-3g")
S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
Hcore = T + V
print(f"  完成，耗时 {time.time()-t0:.3f}s")
print(f"  S11 = {S[0,0]:.6f}, S12 = {S[0,1]:.6f}")
print(f"  H11 = {Hcore[0,0]:.6f}, H12 = {Hcore[0,1]:.6f}")

print("\n步骤 2: 计算双电子积分...")
t0 = time.time()
eri = compute_eri(basis, mol)
print(f"  完成，耗时 {time.time()-t0:.3f}s")
print(f"  (11|11) = {eri[0,0,0,0]:.6f}")
print(f"  (11|22) = {eri[0,0,1,1]:.6f}")
print(f"  (12|12) = {eri[0,1,0,1]:.6f}")

print("\n步骤 3: 运行 RHF SCF...")
t0 = time.time()
mf = RHF(mol, basis="sto-3g", max_cycles=50, conv_tol=1e-8, verbose=2)
e_tot = mf.kernel()
print(f"\n  完成，耗时 {time.time()-t0:.3f}s")

ref = -1.1167143251893824
error = abs(e_tot - ref)

print("\n" + "=" * 60)
print("结果:")
print(f"  计算能量: {e_tot:.12f} Eh")
print(f"  参考能量: {ref:.12f} Eh")
print(f"  误差:     {error:.2e} Eh")
print(f"  收敛:     {mf.converged}")
print(f"  迭代次数: {mf.n_cycles}")

if error < 1e-5:
    print("  ✓ 测试通过!")
else:
    print("  ✗ 测试失败!")
print("=" * 60)
