"""调试DIIS在H2O上的行为。"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF, nuclear_repulsion, build_fock_rhf, scf_energy_rhf
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.guess import guess_hcore, _symmetric_orthogonalize, _eigh_safe, _matmul
from calchem.scf.diis import DIIS

h2o = Molecule.from_xyz("""
3
H2O
O  0.000000  0.000000  0.119263
H  0.000000  0.763239 -0.477052
H  0.000000 -0.763239 -0.477052
""")

rhf = RHF(h2o, basis="sto-3g", verbose=0)
rhf.build()

hcore = rhf._hcore
S = rhf._s
eri = rhf._eri
e_nn = rhf.e_nn
nbf = S.shape[0]
nelectron = h2o.to_bohr().nelectron
nocc = nelectron // 2

dm = guess_hcore(hcore, S, nelectron, restricted=True)
fock = build_fock_rhf(hcore, dm, eri)
e_prev = scf_energy_rhf(hcore, dm, fock) + e_nn

diis = DIIS(ndim=6)

x = _symmetric_orthogonalize(S)

print(f"{'cycle':>5} {'E':>18} {'dE':>12} {'RMS(D)':>12} {'DIIS?':>6} {'||e||':>12} {'cond':>12}")
print("-" * 85)

for cycle in range(1, 31):
    # DIIS add and extrapolate
    diis.add(fock, dm, S)
    n_diis = len(diis.fock_matrices)

    # Check error norm
    err = diis.error_matrices[-1]
    err_norm = np.linalg.norm(err)

    # Check conditioning of DIIS B matrix
    cond = 0.0
    fock_diis = None
    if n_diis >= 2:
        n = n_diis
        b = np.zeros((n + 1, n + 1))
        b[n, :] = -1.0
        b[:, n] = -1.0
        b[n, n] = 0.0
        for i in range(n):
            for j in range(n):
                b[i, j] = np.vdot(diis.error_matrices[i], diis.error_matrices[j])
        cond = np.linalg.cond(b)
        try:
            rhs = np.zeros(n + 1)
            rhs[n] = -1.0
            coeffs = np.linalg.solve(b, rhs)
            fock_diis = sum(coeffs[i] * diis.fock_matrices[i] for i in range(n))
        except np.linalg.LinAlgError:
            pass

    used_diis = "yes" if fock_diis is not None else "no"
    if fock_diis is not None:
        fock = fock_diis

    # Diagonalize
    f_ortho = _matmul(_matmul(x.T, fock), x)
    eigval, eigvec_ortho = _eigh_safe(f_ortho)
    mo_coeff = _matmul(x, eigvec_ortho)
    dm_new = 2.0 * _matmul(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)

    fock = build_fock_rhf(hcore, dm_new, eri)
    e_new = scf_energy_rhf(hcore, dm_new, fock) + e_nn

    de = e_new - e_prev
    ddm = np.sqrt(np.mean((dm_new - dm) ** 2))

    print(f"{cycle:5d} {e_new:18.10f} {de:+12.2e} {ddm:12.2e} {used_diis:>6} {err_norm:12.2e} {cond:12.2e}")

    dm = dm_new
    e_prev = e_new

    if abs(de) < 1e-8 and ddm < 1e-8:
        print(f"\nSCF converged at cycle {cycle}")
        break
