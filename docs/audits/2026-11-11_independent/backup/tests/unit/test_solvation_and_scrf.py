import math
import pytest
from calchem import _core
from calchem.backends.native import NativeCalchem
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.solvent.pcm import PCMSolver, PCM, pcm
from calchem.solvent.smd import SMD
from calchem.workflows.opv import OPVWorkflow, OPVSolvationResult


def test_rust_core_run_solvation_water():
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 1.4304, -1.1072],
        [0.0, -1.4304, -1.1072],
    ]

    # Water in high-dielectric aqueous medium (eps = 78.36)
    res_water = _core.run_solvation(
        symbols=symbols,
        coords=coords,
        basis_name="sto-3g",
        solvent="water",
        solvent_model="cpcm",
        charge=0,
        multiplicity=1,
        method="b3lyp",
        xc="b3lyp",
    )

    assert res_water.converged
    assert res_water.delta_g_elec_kcal_mol < 0.0
    assert res_water.cavity_surface_area_angstrom2 > 20.0
    assert abs(res_water.total_surface_charge) < 1e-8
    assert res_water.solvent_name == "water"
    assert math.isclose(res_water.dielectric_constant, 78.3553, abs_tol=0.1)

    # Water in medium-dielectric OPV processing solvent (chlorobenzene, eps = 5.62)
    res_cb = _core.run_solvation(
        symbols=symbols,
        coords=coords,
        basis_name="sto-3g",
        solvent="chlorobenzene",
        solvent_model="cpcm",
        charge=0,
        multiplicity=1,
        method="b3lyp",
        xc="b3lyp",
    )

    assert res_cb.converged
    assert res_cb.delta_g_elec_kcal_mol < 0.0
    assert res_cb.solvent_name == "chlorobenzene"
    assert math.isclose(res_cb.dielectric_constant, 5.62, abs_tol=0.1)

    # Polar solvent must produce more negative (stabilizing) electrostatic interaction than nonpolar/organic
    assert res_water.delta_g_elec_kcal_mol < res_cb.delta_g_elec_kcal_mol


def test_rust_core_run_pcm_from_scf_py():
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 1.4304, -1.1072],
        [0.0, -1.4304, -1.1072],
    ]

    # First execute gas-phase single point
    gas_wf = _core.run_single_point(
        symbols=symbols,
        geometry_bohr=coords,
        charge=0,
        multiplicity=1,
        method="rhf",
        basis="sto-3g",
    )
    assert gas_wf.converged

    # Post-SCF PCM evaluation using converged gas wavefunction
    pcm_res = _core.run_pcm_from_scf_py(
        gas_wf,
        solvent="water",
        solvent_model="cpcm",
        scaling_factor=1.20,
        include_cds=True,
    )

    assert pcm_res.converged
    assert pcm_res.delta_g_elec_kcal_mol < 0.0
    assert math.isclose(pcm_res.gas_energy, gas_wf.total_energy_hartree, abs_tol=1e-7)
    assert pcm_res.cavity_surface_area_angstrom2 > 20.0
    charges = pcm_res.surface_charges
    assert len(charges) > 0


def test_native_backend_solvation_task():
    backend = NativeCalchem()
    assert "solvation" in backend.capabilities().tasks

    req = CalculationRequest(
        task="solvation",
        backend="native",
        method="b3lyp",
        basis="sto-3g",
        solvent="water",
        solvent_model="cpcm",
        molecule=MoleculeSpec(
            symbols=["O", "H", "H"],
            coordinates_angstrom=[
                [0.0, 0.0, 0.0],
                [0.0, 0.757, -0.586],
                [0.0, -0.757, -0.586],
            ],
            charge=0,
            multiplicity=1,
        ),
    )

    result = backend.run(req)
    assert result.status == "succeeded"
    assert result.scf_converged
    assert result.solvated_energy_hartree is not None
    assert result.delta_g_solv_kcal_mol is not None
    assert result.delta_g_elec_kcal_mol is not None
    assert result.delta_g_elec_kcal_mol < 0.0
    assert result.solvent_name == "water"
    assert result.solvent_dielectric is not None and result.solvent_dielectric > 70.0


def test_pcm_and_smd_class_facades():
    from calchem.mol.molecule import Molecule

    mol = Molecule(
        symbols=["O", "H", "H"],
        coordinates=[
            [0.0, 0.0, 0.0],
            [0.0, 0.757, -0.586],
            [0.0, -0.757, -0.586],
        ],
        charge=0,
        multiplicity=1,
    )

    # Test PCMSolver
    solver = PCMSolver(mol, basis_set="sto-3g", solvent="chlorobenzene")
    dg_solv = solver.kernel(method="b3lyp", xc="b3lyp")
    assert isinstance(dg_solv, float)
    assert solver.delta_g_elec_kcal_mol < 0.0
    assert solver.cavity_surface_area_angstrom2 > 10.0

    # Test SMD
    smd_calc = SMD(mol, solvent="water")
    e_smd = smd_calc.kernel()
    assert isinstance(e_smd, float)
    assert "SMD" in repr(smd_calc)


def test_opv_workflow_solvation_free_energy():
    class _InMemoryNativeService:
        def __init__(self):
            self.backend = NativeCalchem()

        def run(self, req: CalculationRequest):
            return self.backend.run(req)

    service = _InMemoryNativeService()
    workflow = OPVWorkflow(backend="native", service=service)

    mol = MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.0],
            [0.0, 0.757, -0.586],
            [0.0, -0.757, -0.586],
        ],
        charge=0,
        multiplicity=1,
    )

    res = workflow.solvation_free_energy(
        mol,
        solvent="chlorobenzene",
        method="b3lyp",
        basis="sto-3g",
        name="water_test",
    )

    assert isinstance(res, OPVSolvationResult)
    assert res.solvent == "chlorobenzene"
    assert res.delta_g_elec_kcal_mol < 0.0
    assert res.cavity_surface_area_angstrom2 > 10.0
    assert "CalChem Solvation Free Energy" in res.summary
    assert "Chlorobenzene" in res.summary
