"""
诊断 H2O SCF 能量问题。
检查电子数、轨道能量、能量分解、积分对称性等。
"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF, nuclear_repulsion, build_fock_rhf, scf_energy_rhf
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.basis.basis_set import BasisSet

# H2O
h2o = Molecule.from_xyz("""
3
H2O
O  0.000000  0.000000  0.119263
H  0.000000  0.763239 -0.477052
H  0.000000 -0.763239 -0.477052
""")

print("=" * 60)
print("H2O 基本信息")
print("=" * 60)
print(f"原子: {h2o.symbols}")
print(f"原子序数: {h2o.atomic_numbers}")
print(f"电子数: {h2o.nelectron}")
print(f"坐标 (angstrom): {h2o.coordinates}")
h2o_bohr = h2o.to_bohr()
print(f"坐标 (bohr): {h2o_bohr.coordinates}")

# 核排斥能
e_nn = nuclear_repulsion(h2o_bohr)
print(f"核排斥能 V_nn = {e_nn:.10f} Eh")

# 构建基组
basis = BasisSet("sto-3g")
shells_by_atom, nbf = basis.build_molecule_basis(h2o_bohr)
all_shells = []
for atom_shells in shells_by_atom:
    for s in atom_shells:
        all_shells.append(s)

print(f"\n基函数数: {nbf}")
print(f"壳层数: {len(all_shells)}")
for i, s in enumerate(all_shells):
    print(f"  壳层 {i}: {s.shell_type}, 原点={s.origin}, n_prim={s.n_primitives}")

# 积分
print("\n构建积分...")
S = compute_overlap(basis, h2o_bohr)
T = compute_kinetic(basis, h2o_bohr)
V = compute_nuclear(basis, h2o_bohr)
hcore = T + V
eri = compute_eri(basis, h2o_bohr)

print(f"\n重叠矩阵 S:")
print(S)
print(f"\n动能矩阵 T (对角元):")
print(np.diag(T))
print(f"\n核吸引矩阵 V (对角元):")
print(np.diag(V))
print(f"\nHcore (对角元):")
print(np.diag(hcore))

# ERI 对称性检查
print(f"\nERI 张量形状: {eri.shape}")
print(f"ERI[0,0,0,0] = {eri[0,0,0,0]:.10f}")

# 检查 ERI 对称性 (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)
max_diff = 0.0
for i in range(nbf):
    for j in range(nbf):
        for k in range(nbf):
            for l in range(nbf):
                v1 = eri[i,j,k,l]
                v2 = eri[j,i,k,l]
                v3 = eri[i,j,l,k]
                v4 = eri[k,l,i,j]
                d = max(abs(v1-v2), abs(v1-v3), abs(v1-v4))
                if d > max_diff:
                    max_diff = d
print(f"ERI 对称性最大差值: {max_diff:.2e}")

# SCF 初猜
from calchem.scf.guess import guess_hcore, _symmetric_orthogonalize, _eigh_safe, _matmul
dm = guess_hcore(hcore, S, h2o_bohr.nelectron, restricted=True)
print(f"\n初始密度矩阵:")
print(dm)
print(f"Tr(D*S) = {np.einsum('mn,mn->', dm, S):.10f} (应为 {h2o_bohr.nelectron})")

# Fock 矩阵
fock = build_fock_rhf(hcore, dm, eri)
print(f"\n初始 Fock 矩阵 (对角元):")
print(np.diag(fock))

# 能量分解
e_1e = np.einsum("mn,mn->", dm, hcore)
e_2e = 0.5 * np.einsum("mn,mn->", dm, fock - hcore)
e_total = e_1e + e_2e + e_nn
print(f"\n能量分解 (初始猜测):")
print(f"  E_1e  = {e_1e:.10f}")
print(f"  E_2e  = {e_2e:.10f}")
print(f"  V_nn  = {e_nn:.10f}")
print(f"  E_tot = {e_total:.10f}")

# 检查 J 和 K 矩阵
j = np.einsum("ls,mnls->mn", dm, eri)
k = np.einsum("ls,mlns->mn", dm, eri)
print(f"\nJ 矩阵 (对角元): {np.diag(j)}")
print(f"K 矩阵 (对角元): {np.diag(k)}")
print(f"J - 0.5*K (对角元): {np.diag(j - 0.5*k)}")

# 检查轨道能量
x = _symmetric_orthogonalize(S)
f_ortho = _matmul(_matmul(x.T, fock), x)
eigval, eigvec_ortho = _eigh_safe(f_ortho)
print(f"\n轨道能量: {eigval}")
