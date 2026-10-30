"""检查Vxc矩阵符号"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import guess_hcore
from calchem.dft.grid import DFTGrid
from calchem.dft.fock import compute_vxc
from calchem.dft.functionals import LDA
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
basis_set = BasisSet('sto-3g', mol)

S = compute_overlap(basis_set, mol.to_bohr())
T = compute_kinetic(basis_set, mol.to_bohr())
V = compute_nuclear(basis_set, mol.to_bohr())
H_core = T + V

D = guess_hcore(H_core, S, mol.nelectron, restricted=True)

func = LDA()
grid = DFTGrid(mol.to_bohr(), basis_set, 'coarse')

rho = grid.compute_density(D, basis_set, mol.to_bohr())
exc, vxc_rho = func.compute(rho)

print(f"rho stats: min={np.min(rho):.6e}, max={np.max(rho):.6e}, mean={np.mean(rho):.6e}")
print(f"exc stats: min={np.min(exc):.6e}, max={np.max(exc):.6e}, mean={np.mean(exc):.6e}")
print(f"vxc_rho stats: min={np.min(vxc_rho):.6e}, max={np.max(vxc_rho):.6e}, mean={np.mean(vxc_rho):.6e}")

vxc = compute_vxc(D, basis_set, mol.to_bohr(), func, grid)
print(f"\nVxc matrix stats:")
print(f"  min={np.min(vxc):.6e}, max={np.max(vxc):.6e}, mean={np.mean(vxc):.6e}")
print(f"  trace={np.trace(vxc):.6f}")
print(f"  eigenvalues: {np.linalg.eigvalsh(vxc)[:3]} ... {np.linalg.eigvalsh(vxc)[-3:]}")

# 对比J矩阵
from calchem.integrals.two_electron import compute_eri
eri = compute_eri(basis_set, mol.to_bohr())
J = np.einsum("ls,mnls->mn", D, eri)
print(f"\nJ matrix stats:")
print(f"  min={np.min(J):.6e}, max={np.max(J):.6e}, mean={np.mean(J):.6e}")
print(f"  trace={np.trace(J):.6f}")

# 对比K矩阵（RHF交换项）
K = np.einsum("ls,mlns->mn", D, eri)
print(f"\nK matrix stats:")
print(f"  min={np.min(K):.6e}, max={np.max(K):.6e}, mean={np.mean(K):.6e}")
print(f"  trace={np.trace(K):.6f}")

print(f"\nFock components (trace):")
print(f"  H_core: {np.trace(H_core):.6f}")
print(f"  J: {np.trace(J):.6f}")
print(f"  -K (RHF): {-np.trace(K):.6f}")
print(f"  Vxc (DFT): {np.trace(vxc):.6f}")
print(f"  F_RHF: {np.trace(H_core + J - K):.6f}")
print(f"  F_DFT: {np.trace(H_core + J + vxc):.6f}")
