import unittest
import numpy as np
from calchem.core.errors import CalChemError
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.props.cube import GridSpec, evaluate_orbital_grid
from calchem.scf.hf import RHF

class TestCoreAndBasis(unittest.TestCase):
    def test_molecule_validation_rejects_invalid_inputs(self):
        with self.assertRaises(CalChemError):
            Molecule([], [], units="bohr")
        with self.assertRaises(CalChemError):
            Molecule(["Xx"], [[0.0, 0.0, 0.0]], units="bohr")
        with self.assertRaises(CalChemError):
            Molecule(["H"], [[0.0, float("nan"), 0.0]], units="bohr")
        with self.assertRaises(CalChemError):
            Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1e-5]], units="bohr")
        with self.assertRaises(CalChemError):
            Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], charge=0, multiplicity=2, units="bohr")

    def test_basis_set_rejects_unknown_basis_and_element(self):
        with self.assertRaises(ValueError):
            BasisSet("nonexistent-basis-999g")
        bs = BasisSet("sto-3g")
        with self.assertRaises(ValueError):
            bs.get_element_shells("U")

    def test_real_orbital_grid_non_zero_and_normalized(self):
        mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], units="bohr")
        mf = RHF(mol, "sto-3g", energy_tol=1e-12, density_tol=1e-10)
        mf.kernel()
        self.assertTrue(mf.converged)
        spec = GridSpec(origin_bohr=(-4.0, -4.0, -4.0), spacing_bohr=(0.25, 0.25, 0.25), dims=(33, 33, 39))
        grid_homo = evaluate_orbital_grid(mol, mf.basis, mf.mo_coeff, 0, spec)
        grid_lumo = evaluate_orbital_grid(mol, mf.basis, mf.mo_coeff, 1, spec)
        self.assertEqual(grid_homo.shape, (33, 33, 39))
        self.assertGreater(float(np.max(np.abs(grid_homo))), 0.1)
        dv = 0.25 ** 3
        norm_homo = float(np.sum(grid_homo.astype(np.float64) ** 2) * dv)
        self.assertLess(abs(norm_homo - 1.0), 0.02)
        ovlp_01 = float(np.sum(grid_homo.astype(np.float64) * grid_lumo.astype(np.float64)) * dv)
        self.assertLess(abs(ovlp_01), 1e-6)
