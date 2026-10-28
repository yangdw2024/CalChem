import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.hf import nuclear_repulsion, build_fock_rhf, scf_energy_rhf
from calchem.scf.guess import _symmetric_orthogonalize, guess_hcore

print("手动测试 H2 RHF SCF...")

mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], units="bohr")
basis = BasisSet("sto-3g")

print("计算积分...")
S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
Hcore = T + V
eri = compute_eri(basis, mol)
e_nn = nuclear_repulsion(mol)
print(f"  E_nn = {e_nn:.10f}")

print("初始猜测...")
dm = guess_hcore(Hcore, S, 2, restricted=True)
print(f"  D =\n{dm}")

print("构建 Fock...")
fock = build_fock_rhf(Hcore, dm, eri)
print(f"  F =\n{fock}")

e_elec = scf_energy_rhf(Hcore, dm, fock)
e_tot = e_elec + e_nn
print(f"  E_elec = {e_elec:.10f}")
print(f"  E_tot = {e_tot:.10f}")

x = _symmetric_orthogonalize(S)
nocc = 1

print("\n开始 SCF 迭代...")
e_prev = e_tot
for cycle in range(1, 51):
    f_ortho = x.T @ fock @ x
    eigval, eigvec_ortho = np.linalg.eigh(f_ortho)
    mo_coeff = x @ eigvec_ortho
    dm_new = 2.0 * mo_coeff[:, :nocc] @ mo_coeff[:, :nocc].T

    fock = build_fock_rhf(Hcore, dm_new, eri)
    e_new = scf_energy_rhf(Hcore, dm_new, fock) + e_nn
    de = e_new - e_prev
    ddm = np.sqrt(np.mean((dm_new - dm) ** 2))

    print(f"  Iter {cycle:2d}: E = {e_new:.10f}  dE = {de:+.2e}  RMS(D) = {ddm:.2e}")

    dm = dm_new
    if abs(de) < 1e-8 and ddm < 1e-8:
        print("  收敛!")
        break
    e_prev = e_new

print(f"\n最终能量: {e_new:.12f} Eh")
print(f"参考值:   -1.116714325189 Eh")
print(f"误差:     {abs(e_new + 1.116714325189):.2e} Eh")
