"""PBE诊断：禁用DIIS，检查能量和密度矩阵变化"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.dft import RKS
from calchem.scf.hf import nuclear_repulsion
from calchem.dft.grid import DFTGrid
from calchem.dft.fock import compute_vxc
from calchem.dft.functionals import PBE
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
basis_set = BasisSet('sto-3g', mol)

S = compute_overlap(basis_set, mol)
T = compute_kinetic(basis_set, mol)
V = compute_nuclear(basis_set, mol)
H_core = T + V

eri = compute_eri(basis_set, mol)

nelec = sum(mol.atomic_numbers)
nocc = nelec // 2

func = PBE()
grid = DFTGrid(mol, basis_set, 'coarse')

# 初始密度
eigvals, eigvecs = np.linalg.eigh(H_core)
C = eigvecs[:, :nocc]
D = 2.0 * C @ C.T

print("Iteration    Energy(Eh)      dE         RMS(D)")
print("-" * 50)

E_prev = 0.0
for cycle in range(15):
    vxc = compute_vxc(D, basis_set, mol, func, grid=grid)
    J = np.einsum("ls,mnls->mn", D, eri)
    F = H_core + J + vxc

    F_prime = np.linalg.inv(np.linalg.cholesky(S)).T @ F @ np.linalg.inv(np.linalg.cholesky(S))
    e, C_prime = np.linalg.eigh(F_prime)
    C = np.linalg.inv(np.linalg.cholesky(S)) @ C_prime
    C = C[:, :nocc]
    D_new = 2.0 * C @ C.T

    E_J = 0.5 * np.einsum("mn,mn->", D, J)
    E_Hcore = np.einsum("mn,mn->", D, H_core)
    rho, grad_rho = grid.compute_density_and_gradients(D, basis_set, mol)
    exc, _ = func.compute(rho, grad_rho)
    weights = grid.weights
    E_xc = np.sum(rho * exc * weights)
    E_nuc = nuclear_repulsion(mol)
    E = E_Hcore + E_J + E_xc + E_nuc

    dE = E - E_prev
    rmsd = np.linalg.norm(D_new - D)
    print(f"  {cycle+1:3d}    {E:14.8f}  {dE:10.6f}  {rmsd:.6e}")

    D = 0.5 * D + 0.5 * D_new  # 阻尼
    E_prev = E
