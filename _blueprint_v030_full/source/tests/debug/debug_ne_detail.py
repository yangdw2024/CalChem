"""
Debug Ne SCF - check all integral components.
"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri, shell_eri
from calchem.scf.hf import RHF, nuclear_repulsion

# He atom (s-only) to verify s-only systems work
print("=" * 60)
print("He STO-3G (s-only atom)")
print("=" * 60)
he = Molecule(["He"], [[0.0, 0.0, 0.0]], units="bohr")
rhf_he = RHF(he, basis="sto-3g", verbose=0)
e_he = rhf_he.kernel()
print(f"RHF energy: {e_he:.8f} Eh")
print(f"Reference:  -2.807783 Eh")
print(f"Diff: {e_he - (-2.807783):.6e}")

# Be atom (s-only in STO-3G)
print("\n" + "=" * 60)
print("Be STO-3G (s-only atom)")
print("=" * 60)
be = Molecule(["Be"], [[0.0, 0.0, 0.0]], units="bohr")
try:
    rhf_be = RHF(be, basis="sto-3g", verbose=0)
    e_be = rhf_be.kernel()
    print(f"RHF energy: {e_be:.8f} Eh")
    print(f"Reference:  -14.351880 Eh")
    print(f"Diff: {e_be - (-14.351880):.6e}")
except Exception as e:
    print(f"Error: {e}")

# Ne atom - detailed analysis
print("\n" + "=" * 60)
print("Ne STO-3G - detailed integral analysis")
print("=" * 60)
ne = Molecule(["Ne"], [[0.0, 0.0, 0.0]], units="bohr")
basis = BasisSet("sto-3g")
shells_by_atom, nbf = basis.build_molecule_basis(ne)
all_shells = shells_by_atom[0]

print(f"Number of shells: {len(all_shells)}")
for i, sh in enumerate(all_shells):
    print(f"  Shell {i}: type={sh.shell_type}, angular={sh.angular}, n_prim={sh.n_primitives}")
    print(f"    exponents: {sh.exponents}")
    print(f"    raw coeffs: {sh.coefficients}")
    print(f"    normed coeffs: {sh._contract_coeffs}")
    print(f"    norm_constants: {sh.norm_constants}")

S = compute_overlap(basis, ne)
T = compute_kinetic(basis, ne)
V = compute_nuclear(basis, ne)
H = T + V
ERI = compute_eri(basis, ne)

e_nn = nuclear_repulsion(ne)
print(f"\nNuclear repulsion: {e_nn}")
print(f"Nelectron: {ne.nelectron}")

# Check overlap matrix
print(f"\nOverlap matrix:")
print(f"  S[0,0] (1s,1s) = {S[0,0]:.10f} (should be 1.0)")
print(f"  S[1,1] (2s,2s) = {S[1,1]:.10f} (should be 1.0)")
print(f"  S[2,2] (2px,2px) = {S[2,2]:.10f} (should be 1.0)")
print(f"  S[3,3] (2py,2py) = {S[3,3]:.10f} (should be 1.0)")
print(f"  S[4,4] (2pz,2pz) = {S[4,4]:.10f} (should be 1.0)")
print(f"  S[0,1] (1s,2s) = {S[0,1]:.10f}")
print(f"  S[0,2] (1s,2px) = {S[0,2]:.10f} (should be 0)")
print(f"  S[2,3] (2px,2py) = {S[2,3]:.10f} (should be 0)")

# Check Hcore eigenvalues (orbital energies from initial guess)
from calchem.scf.guess import guess_hcore, _symmetric_orthogonalize, _eigh_safe, _matmul
X = _symmetric_orthogonalize(S)
H_ortho = _matmul(_matmul(X.T, H), X)
eigvals, eigvecs = _eigh_safe(H_ortho)
print(f"\nHcore eigenvalues (orbital energies):")
for i, e in enumerate(eigvals):
    print(f"  orbital {i}: {e:.10f} Eh")

# Check ERI values
print(f"\nERI values:")
print(f"  (1s,1s|1s,1s) = {ERI[0,0,0,0]:.10f}")
print(f"  (2s,2s|2s,2s) = {ERI[1,1,1,1]:.10f}")
print(f"  (1s,2s|1s,2s) = {ERI[0,1,0,1]:.10f}")
print(f"  (1s,1s|2s,2s) = {ERI[0,0,1,1]:.10f}")
print(f"  (2px,2px|2px,2px) = {ERI[2,2,2,2]:.10f}")
print(f"  (2px,2px|2py,2py) = {ERI[2,2,3,3]:.10f}")
print(f"  (2px,2py|2px,2py) = {ERI[2,3,2,3]:.10f}")
print(f"  (1s,2px|1s,2px) = {ERI[0,2,0,2]:.10f}")
print(f"  (1s,1s|2px,2px) = {ERI[0,0,2,2]:.10f}")

# For Ne, the 2p electrons should have:
# (2p_x 2p_x | 2p_x 2p_x) - this is the self-repulsion of a p_x electron
# Reference: for Ne STO-3G, this should be around 0.3-0.5 Eh

# Let's also check the initial density and energy components
dm = guess_hcore(H, S, ne.nelectron, restricted=True)
e_1e = np.einsum("mn,mn->", dm, H)
j_part = np.einsum("ls,mnls->mn", dm, ERI)
k_part = np.einsum("ls,mlns->mn", dm, ERI)
g = j_part - 0.5 * k_part
e_2e = 0.5 * np.einsum("mn,mn->", dm, g)
e_total = e_1e + e_2e + e_nn

print(f"\nEnergy components (from Hcore guess):")
print(f"  E_1e = {e_1e:.10f}")
print(f"  E_2e = {e_2e:.10f}")
print(f"  E_nn = {e_nn:.10f}")
print(f"  E_total = {e_total:.10f}")

# Check density matrix
print(f"\nDensity matrix diagonal:")
print(f"  D[0,0] (1s) = {dm[0,0]:.10f}")
print(f"  D[1,1] (2s) = {dm[1,1]:.10f}")
print(f"  D[2,2] (2px) = {dm[2,2]:.10f}")
print(f"  D[3,3] (2py) = {dm[3,3]:.10f}")
print(f"  D[4,4] (2pz) = {dm[4,4]:.10f}")
print(f"  Tr(D*S) = {np.einsum('mn,mn->', dm, S):.10f} (should be {ne.nelectron//2})")
