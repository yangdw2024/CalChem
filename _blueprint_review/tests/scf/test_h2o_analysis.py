"""
深入分析H2O能量问题。
"""
import sys
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.hf import nuclear_repulsion
from calchem.scf.guess import _symmetric_orthogonalize, _eigh_safe, _matmul

mol = Molecule(
    ["O", "H", "H"],
    [
        [0.0000000, 0.0000000, 0.0000000],
        [0.0000000, -0.7570000, 0.5860000],
        [0.0000000, 0.7570000, 0.5860000],
    ],
    units="angstrom",
).to_bohr()

basis = BasisSet("sto-3g")
s = compute_overlap(basis, mol)
t = compute_kinetic(basis, mol)
v = compute_nuclear(basis, mol)
h = t + v
eri = compute_eri(basis, mol)
e_nn = nuclear_repulsion(mol)

nbf = s.shape[0]
nelectron = mol.nelectron
nocc = nelectron // 2

print("=" * 70)
print("H2O STO-3G 能量分析")
print("=" * 70)
print()

print(f"基函数数目: {nbf}")
print(f"电子数: {nelectron}")
print(f"占据轨道数: {nocc}")
print(f"核排斥能: {e_nn:.6f} Eh")
print()

print("【重叠矩阵对角元（应该都是1.0）】")
for i in range(nbf):
    print(f"  {i}: S[{i},{i}] = {s[i,i]:.10f}")
print()

print("【基函数顺序】")
print("  0: O 1s")
print("  1: O 2s")
print("  2: O 2px")
print("  3: O 2py")
print("  4: O 2pz")
print("  5: H1 1s")
print("  6: H2 1s")
print()

# 做一次收敛的SCF（用阻尼）
print("【SCF 计算（带阻尼）】")
print()

x = _symmetric_orthogonalize(s)
f_ortho = _matmul(_matmul(x.T, h), x)
eigval, eigvec_ortho = _eigh_safe(f_ortho)
mo_coeff = _matmul(x, eigvec_ortho)
dm = 2.0 * _matmul(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)

def build_fock(dm_):
    j = np.einsum("ls,mnls->mn", dm_, eri)
    k = np.einsum("ls,mlns->mn", dm_, eri)
    g = j - 0.5 * k
    return h + g

def scf_energy(dm_, f_):
    e_1e = np.einsum("mn,mn->", dm_, h)
    e_2e = 0.5 * np.einsum("mn,mn->", dm_, f_ - h)
    return e_1e + e_2e, e_1e, e_2e

fock = build_fock(dm)
e_tot, e_1e, e_2e = scf_energy(dm, fock)
e_tot += e_nn

print(f"  初始: E = {e_tot:.6f}  (1e: {e_1e:.6f}, 2e: {e_2e:.6f})")

# 阻尼迭代
for cycle in range(1, 101):
    f_ortho = _matmul(_matmul(x.T, fock), x)
    eigval, eigvec_ortho = _eigh_safe(f_ortho)
    mo_coeff_new = _matmul(x, eigvec_ortho)
    dm_new = 2.0 * _matmul(mo_coeff_new[:, :nocc], mo_coeff_new[:, :nocc].T)
    
    # 自适应阻尼
    if cycle < 10:
        damping = 0.7
    else:
        damping = 0.5
    dm_new = damping * dm + (1 - damping) * dm_new
    
    fock = build_fock(dm_new)
    e_new, e_1e_new, e_2e_new = scf_energy(dm_new, fock)
    e_new += e_nn
    de = e_new - e_tot
    
    if cycle <= 5 or cycle % 10 == 0:
        print(f"  {cycle:3d}: E = {e_new:.6f}  (1e: {e_1e_new:.6f}, 2e: {e_2e_new:.6f})  dE = {de:+.2e}")
    
    dm = dm_new
    e_tot = e_new
    e_1e = e_1e_new
    e_2e = e_2e_new
    
    if abs(de) < 1e-8:
        print(f"  {cycle:3d}: 收敛!")
        break

print()
print(f"最终能量: {e_tot:.6f} Eh")
print(f"  单电子能量: {e_1e:.6f} Eh")
print(f"  双电子能量: {e_2e:.6f} Eh")
print(f"  核排斥能: {e_nn:.6f} Eh")
print()

print("【参考值（典型）】")
print("  总能量: ~ -74.94 Eh")
print("  单电子能量: ~ -123.84 Eh")
print("  双电子能量: ~ +39.71 Eh")
print("  核排斥能: ~ +9.19 Eh")
print()

print(f"总能量差值: {e_tot - (-74.94208):.6f} Eh")
print(f"单电子能量差值: 需要验证")
print(f"双电子能量差值: 需要验证")
print()

# 检查Fock矩阵对角元
print("【Fock矩阵对角元】")
for i in range(nbf):
    print(f"  {i}: F[{i},{i}] = {fock[i,i]:.6f}")
print()

# 检查轨道能量
print("【轨道能量】")
f_ortho_final = _matmul(_matmul(x.T, fock), x)
mo_e, _ = _eigh_safe(f_ortho_final)
for i, e in enumerate(mo_e):
    occ = "occ" if i < nocc else "vir"
    print(f"  {i+1:2d}. {e:12.6f}  {occ}")
