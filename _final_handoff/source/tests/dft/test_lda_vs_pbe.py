"""比较LDA和PBE（仅vxc_rho）的SCF行为"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import nuclear_repulsion
from calchem.dft.grid import DFTGrid
from calchem.dft.fock import compute_vxc
from calchem.dft.functionals import LDA, PBE
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

# 用L_S = inv(cholesky(S))
L_S = np.linalg.inv(np.linalg.cholesky(S))

def run_scf(func, name, grid, damping=0.5, max_cycle=20):
    eigvals, eigvecs = np.linalg.eigh(H_core)
    C = eigvecs[:, :nocc]
    D = 2.0 * C @ C.T
    E_prev = 0.0
    print(f"\n=== {name} ===")
    for cycle in range(max_cycle):
        vxc = compute_vxc(D, basis_set, mol, func, grid=grid)
        J = np.einsum("ls,mnls->mn", D, eri)
        F = H_core + J + vxc

        F_prime = L_S.T @ F @ L_S
        e, C_prime = np.linalg.eigh(F_prime)
        C = L_S @ C_prime[:, :nocc]
        D_new = 2.0 * C @ C.T

        E_J = 0.5 * np.einsum("mn,mn->", D, J)
        E_Hcore = np.einsum("mn,mn->", D, H_core)
        if func.type == "lda":
            rho = grid.compute_density(D, basis_set, mol)
            exc, _ = func.compute(rho)
        else:
            rho, grad_rho = grid.compute_density_and_gradients(D, basis_set, mol)
            exc, _ = func.compute(rho, grad_rho)
        E_xc = np.sum(rho * exc * grid.weights)
        E = E_Hcore + E_J + E_xc + nuclear_repulsion(mol)

        dE = E - E_prev
        rmsd = np.linalg.norm(D_new - D)
        print(f"  {cycle+1:2d}  E={E:14.8f} dE={dE:10.4f} RMSD={rmsd:.4e}")
        if abs(dE) < 1e-8 and rmsd < 1e-6 and cycle > 0:
            print(f"  {name} CONVERGED")
            break
        D = (1-damping) * D + damping * D_new
        E_prev = E

grid = DFTGrid(mol, basis_set, 'coarse')
run_scf(LDA(), "LDA", grid)
run_scf(PBE(), "PBE (full)", grid)

# 测试 PBE 但强制用 compute() 而不是 compute_full()
class PBE_LDAlike(PBE):
    def compute_full(self, rho, grad_rho=None, lap_rho=None, tau=None):
        exc, vxc = self.compute(rho, grad_rho)
        return exc, vxc, None, None

run_scf(PBE_LDAlike(), "PBE (vxc_rho only)", grid)
