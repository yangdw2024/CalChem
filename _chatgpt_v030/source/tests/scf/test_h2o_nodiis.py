"""测试H2O在无DIIS、有阻尼的情况下的SCF收敛。"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF, nuclear_repulsion, build_fock_rhf, scf_energy_rhf
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri
from calchem.scf.guess import guess_hcore, _symmetric_orthogonalize, _eigh_safe, _matmul

h2o = Molecule.from_xyz("""
3
H2O
O  0.000000  0.000000  0.119263
H  0.000000  0.763239 -0.477052
H  0.000000 -0.763239 -0.477052
""")

# 无DIIS
print("=" * 60)
print("H2O SCF - 无DIIS")
print("=" * 60)
rhf_nodiis = RHF(h2o, basis="sto-3g", verbose=1, diis=False, max_cycles=100)
try:
    e = rhf_nodiis.kernel()
    print(f"最终能量: {e:.10f}")
except Exception as ex:
    print(f"异常: {ex}")
