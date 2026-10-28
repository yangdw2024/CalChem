"""直接计算Vxc矩阵进行验证"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import guess_hcore
from calchem.dft.grid import DFTGrid
from calchem.dft.functionals import LDA
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear

coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
mol = Molecule(['O', 'H', 'H'], coords)
basis_set = BasisSet('sto-3g', mol)

S = compute_overlap(basis_set, mol.to_bohr())
H_core = compute_kinetic(basis_set, mol.to_bohr()) + compute_nuclear(basis_set, mol.to_bohr())

D = guess_hcore(H_core, S, mol.nelectron, restricted=True)

func = LDA()
grid = DFTGrid(mol.to_bohr(), basis_set, 'coarse')

rho = grid.compute_density(D, basis_set, mol.to_bohr())
exc, vxc_rho = func.compute(rho)

# 直接计算Vxc矩阵: Vxc[μν] = ∫ vxc(r) φ_μ(r) φ_ν(r) dr
all_shells = basis_set.get_all_shells(mol.to_bohr())
shell_range = []
idx = 0
for s in all_shells:
    nf = s.n_functions
    shell_range.append((idx, idx + nf))
    idx += nf

nbf = idx
vxc_direct = np.zeros((nbf, nbf))

for i, shell_i in enumerate(all_shells):
    si, ei = shell_range[i]
    for j, shell_j in enumerate(all_shells):
        sj, ej = shell_range[j]
        
        block = np.zeros((shell_i.n_functions, shell_j.n_functions))
        origin_i = shell_i.origin
        origin_j = shell_j.origin
        
        for p_i, alpha_i in enumerate(shell_i.exponents):
            c_i = shell_i._contract_coeffs[p_i]
            n_i = shell_i.norm_constants[p_i]
            for p_j, alpha_j in enumerate(shell_j.exponents):
                c_j = shell_j._contract_coeffs[p_j]
                n_j = shell_j.norm_constants[p_j]
                prefactor = c_i * c_j * n_i * n_j
                
                for ifunc, comp_i in enumerate(shell_i.cartesian_components()):
                    l_i, m_i, n_i_exp = comp_i
                    phi_i = grid._eval_gto(l_i, m_i, n_i_exp, alpha_i, origin_i)
                    
                    for jfunc, comp_j in enumerate(shell_j.cartesian_components()):
                        l_j, m_j, n_j_exp = comp_j
                        phi_j = grid._eval_gto(l_j, m_j, n_j_exp, alpha_j, origin_j)
                        
                        integrand = phi_i * phi_j * vxc_rho * grid.weights
                        block[ifunc, jfunc] += prefactor * np.sum(integrand)
        
        vxc_direct[si:ei, sj:ej] = block

print(f"Vxc direct trace: {np.trace(vxc_direct):.6f}")
print(f"Vxc direct min/max: {np.min(vxc_direct):.6f} / {np.max(vxc_direct):.6f}")

# 对比compute_vxc结果
from calchem.dft.fock import compute_vxc
vxc_fock = compute_vxc(D, basis_set, mol.to_bohr(), func, grid)
print(f"\nVxc fock trace: {np.trace(vxc_fock):.6f}")
print(f"Vxc fock min/max: {np.min(vxc_fock):.6f} / {np.max(vxc_fock):.6f}")

print(f"\nDifference: ||vxc_direct - vxc_fock|| = {np.linalg.norm(vxc_direct - vxc_fock):.6e}")
