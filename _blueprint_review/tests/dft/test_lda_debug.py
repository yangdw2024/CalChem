"""诊断LDA振荡问题"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import nuclear_repulsion
from calchem.dft.grid import DFTGrid
from calchem.dft.fock import compute_vxc, build_fock_dft
from calchem.dft.functionals import LDA
from calchem.basis.basis_set import BasisSet
from calchem.scf.hf import guess_hcore
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
basis_set = BasisSet('sto-3g', mol)

S = compute_overlap(basis_set, mol.to_bohr())
T = compute_kinetic(basis_set, mol.to_bohr())
V = compute_nuclear(basis_set, mol.to_bohr())
H_core = T + V
eri = compute_eri(basis_set, mol.to_bohr())

nelec = sum(mol.atomic_numbers)
nocc = nelec // 2

func = LDA()
grid = DFTGrid(mol.to_bohr(), basis_set, 'coarse')

L_S = np.linalg.inv(np.linalg.cholesky(S))

D = guess_hcore(H_core, S, mol.nelectron, restricted=True)

print("Iteration    E_tot        E_1e         E_J          E_xc         E_nn")
print("-" * 80)

for cycle in range(10):
    vxc = compute_vxc(D, basis_set, mol.to_bohr(), func, grid)
    J = np.einsum("ls,mnls->mn", D, eri)
    F = H_core + J + vxc

    F_prime = L_S.T @ F @ L_S
    e, C_prime = np.linalg.eigh(F_prime)
    C = L_S @ C_prime[:, :nocc]
    D_new = 2.0 * C @ C.T

    E_1e = np.einsum("mn,mn->", D, H_core)
    E_J = 0.5 * np.einsum("mn,mn->", D, J)
    rho = grid.compute_density(D, basis_set, mol.to_bohr())
    exc, _ = func.compute(rho)
    E_xc = np.sum(rho * exc * grid.weights)
    E_nn = nuclear_repulsion(mol.to_bohr())
    E_tot = E_1e + E_J + E_xc + E_nn

    print(f"  {cycle+1:2d}    {E_tot:12.6f}  {E_1e:12.6f}  {E_J:12.6f}  {E_xc:12.6f}  {E_nn:12.6f}")
    
    D = D_new

print("\nFock matrix symmetry check:")
print(f"  ||F - F^T|| = {np.linalg.norm(F - F.T):.2e}")
print(f"  ||vxc - vxc^T|| = {np.linalg.norm(vxc - vxc.T):.2e}")
print(f"  ||J - J^T|| = {np.linalg.norm(J - J.T):.2e}")
