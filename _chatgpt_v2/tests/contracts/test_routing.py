import unittest
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.backends.native import NativeCalchem

class TestRoutingAndContracts(unittest.TestCase):
    def test_unsupported_method_and_basis_explicit_failure(self):
        mol = MoleculeSpec(symbols=["H", "H"], coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], charge=0, multiplicity=1)
        backend = NativeCalchem()

        req_bad_method = CalculationRequest(
            task="single_point", molecule=mol, backend="native", method="nonexistent_xc_99", basis="sto-3g"
        )
        res_m = backend.run(req_bad_method)
        self.assertEqual(res_m.status, "failed")
        self.assertIsNot(res_m.scf_converged, True)
        self.assertIsNone(res_m.total_energy_hartree)
        self.assertGreater(len(res_m.errors), 0)

        req_bad_basis = CalculationRequest(
            task="single_point", molecule=mol, backend="native", method="rhf", basis="fake-basis-set"
        )
        res_b = backend.run(req_bad_basis)
        self.assertEqual(res_b.status, "failed")
        self.assertIsNone(res_b.total_energy_hartree)
        self.assertGreater(len(res_b.errors), 0)

    def test_unconverged_scf_status_is_not_converged(self):
        mol = MoleculeSpec(symbols=["O", "H", "H"], coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]])
        backend = NativeCalchem()
        req = CalculationRequest(
            task="single_point", molecule=mol, backend="native", method="rhf", basis="sto-3g",
            options={"max_iter": 1}
        )
        res = backend.run(req)
        self.assertEqual(res.status, "not_converged")
        self.assertFalse(res.scf_converged)
