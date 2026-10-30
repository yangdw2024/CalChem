import unittest
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.grad.analytical import compute_hf_gradient
from calchem.grad.hf_grad import HFGradient
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.core.units import BOHR_TO_ANGSTROM
from calchem.backends.native import NativeCalchem

class TestNumericsRhfAndGrad(unittest.TestCase):
    def test_rhf_baselines_and_invariants(self):
        mol_h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], units="bohr")
        mf_h2 = RHF(mol_h2, "sto-3g", energy_tol=1e-12, density_tol=1e-10, gradient_tol=1e-9)
        e_h2 = mf_h2.kernel()
        self.assertTrue(mf_h2.converged)
        self.assertLess(abs(e_h2 - (-1.116714325063)), 1e-10)
        self.assertLess(abs(float(np.trace(mf_h2.dm @ mf_h2._s)) - 2.0), 1e-12)
        comm_h2 = mf_h2.fock @ mf_h2.dm @ mf_h2._s - mf_h2._s @ mf_h2.dm @ mf_h2.fock
        self.assertLess(float(np.linalg.norm(comm_h2)), 1e-9)

        mol_he = Molecule(["He"], [[0.0, 0.0, 0.0]], units="bohr")
        mf_he = RHF(mol_he, "sto-3g", energy_tol=1e-12, density_tol=1e-10)
        e_he = mf_he.kernel()
        self.assertTrue(mf_he.converged)
        self.assertLess(abs(e_he - (-2.8077839575)), 1e-8)

        mol_h2o = Molecule(["O", "H", "H"], [[0.0, 0.0, 0.0], [0.0, 1.432, 0.0], [1.112, -0.716, 0.0]], units="bohr")
        mf_h2o = RHF(mol_h2o, "sto-3g", energy_tol=1e-12, density_tol=1e-10, gradient_tol=1e-8)
        e_h2o = mf_h2o.kernel()
        self.assertTrue(mf_h2o.converged)
        self.assertLess(abs(e_h2o - (-74.6953191771)), 1e-6)
        self.assertLess(abs(float(np.trace(mf_h2o.dm @ mf_h2o._s)) - 10.0), 1e-10)

    def test_h2_energy_varies_smoothly_with_bond_length(self):
        energies = []
        for r in [1.0, 1.2, 1.346, 1.4, 1.8, 2.5]:
            mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, r]], units="bohr")
            mf = RHF(mol, "sto-3g", energy_tol=1e-12, density_tol=1e-10)
            energies.append(mf.kernel())
        self.assertLess(energies[2], energies[0])
        self.assertLess(energies[2], energies[1])
        self.assertLess(energies[2], energies[3])
        self.assertLess(energies[3], energies[4])
        self.assertLess(energies[4], energies[5])

    def test_analytical_gradient_matches_finite_difference(self):
        mol_h2o = Molecule(["O", "H", "H"], [[0.0, 0.0, 0.0], [0.0, 1.432, 0.0], [1.112, -0.716, 0.0]], units="bohr")
        mf_h2o = RHF(mol_h2o, "sto-3g", energy_tol=1e-12, density_tol=1e-10, gradient_tol=1e-8)
        mf_h2o.kernel()
        g_ana = compute_hf_gradient(mf_h2o)
        g_fd = HFGradient(mf_h2o, numerical=True, step=1e-5).kernel()
        self.assertLess(float(np.max(np.abs(np.sum(g_ana, axis=0)))), 1e-12)
        self.assertLess(float(np.max(np.abs(g_ana - g_fd))), 1e-7)

    def test_h2_analytical_geometry_optimization_reaches_exact_minimum(self):
        b2a = BOHR_TO_ANGSTROM
        spec = MoleculeSpec(
            symbols=["H", "H"],
            coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 * b2a]],
            charge=0, multiplicity=1,
        )
        backend = NativeCalchem()
        res = backend.optimize(
            CalculationRequest(task="optimize", molecule=spec, backend="native", method="rhf", basis="sto-3g")
        )
        self.assertEqual(res.status, "succeeded")
        coords_bohr = np.array(res.optimized_coordinates_angstrom) / b2a
        r_eq = float(np.linalg.norm(coords_bohr[1] - coords_bohr[0]))
        self.assertLess(abs(r_eq - 1.34592), 2e-4)
