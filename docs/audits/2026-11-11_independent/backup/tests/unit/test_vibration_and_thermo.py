"""Unit tests for pure Rust Cartesian Hessian, vibrational analysis, and RRHO thermochemistry."""

import math
from calchem import _core as rust_core
from calchem.backends.native import NativeCalchem
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.workflows.opv import OPVWorkflow, OPVThermochemistryResult


def test_rust_core_run_frequency_water():
    """Tests rust_core.run_frequency on water with B3LYP/STO-3G."""
    # Near-equilibrium geometry in Bohr: O at origin, H1/H2 in y-z plane
    symbols = ["O", "H", "H"]
    geometry_bohr = [
        [0.0, 0.0, 0.0],
        [0.0, 1.4304, -1.1072],
        [0.0, -1.4304, -1.1072],
    ]

    res = rust_core.run_frequency(
        symbols=symbols,
        geometry_bohr=geometry_bohr,
        charge=0,
        multiplicity=1,
        method="rks",
        basis="sto-3g",
        step_bohr=0.001,
        xc="b3lyp",
        grid_level="medium",
        temperature_k=298.15,
        pressure_pa=101325.0,
        symmetry_number=2,
    )

    # 1. Molecular geometry & energy
    assert len(res.symbols) == 3
    assert res.energy_hartree < -74.0
    assert not res.vibrational.is_linear
    assert res.vibrational.n_vib == 3
    assert len(res.vibrational.modes) == 3

    # 2. Translation & rotation residuals (should be < 1e-3 Hartree/(amu*Bohr^2))
    for res_lam in res.vibrational.trans_rot_residuals:
        assert abs(res_lam) < 1e-3

    # 3. Normal modes & frequencies
    freqs = res.frequencies_cm
    assert len(freqs) == 3
    for f in freqs:
        assert f > 1000.0, f"Water frequency should be > 1000 cm^-1, got {f}"

    # Bending mode is the lowest (~1600 cm^-1), followed by symmetric and asymmetric stretches
    assert 1400.0 < freqs[0] < 1900.0
    assert 3400.0 < freqs[1] < 4800.0
    assert 3500.0 < freqs[2] < 4800.0
    assert freqs[0] < freqs[1] <= freqs[2]

    # IR intensities: all non-negative, with active infrared transitions (> 0.1 km/mol)
    irs = res.ir_intensities_km_mol
    assert len(irs) == 3
    assert all(val >= 0.0 for val in irs)
    assert max(irs) > 2.0

    # Mode structural properties
    bend = res.vibrational.modes[0]
    assert not bend.is_imaginary
    assert 0.9 < bend.reduced_mass_amu < 2.0
    assert bend.force_constant_m_dyn_a > 1.0
    assert len(bend.displacement) == 3

    # 4. RRHO Thermochemistry consistency
    thermo = res.thermo
    assert thermo.temperature_k == 298.15
    assert thermo.pressure_pa == 101325.0
    assert thermo.zpve_hartree > 0.015  # ~0.02 Hartree for water
    assert thermo.zpve_kcal_mol > 10.0

    # H = E_0 + E_thermal + k_B*T > E_0
    assert res.enthalpy_hartree > res.energy_hartree
    # G = H - T*S, and S > 0, so G < H
    assert res.gibbs_free_energy_hartree < res.enthalpy_hartree
    assert thermo.s_cal_mol_k > 30.0  # Gas phase S ~40-50 cal/(mol*K)
    # Cp = Cv + R
    r_cal = 1.98720425864083
    assert abs((thermo.cp_cal_mol_k - thermo.cv_cal_mol_k) - r_cal) < 1e-5

    # 5. Symmetrized Hessian & APT matrices
    h = res.cartesian_hessian
    assert len(h) == 9 and len(h[0]) == 9
    for i in range(9):
        for j in range(9):
            assert abs(h[i][j] - h[j][i]) < 1e-12

    apt = res.apt
    assert len(apt) == 3 and len(apt[0]) == 9


def test_rust_core_linear_molecule_diatomic():
    """Tests Eckart projection for a linear diatomic molecule (3N - 5 = 1 vibrational mode)."""
    symbols = ["H", "F"]
    # H-F along z-axis
    geometry_bohr = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.733],
    ]

    res = rust_core.run_frequency(
        symbols=symbols,
        geometry_bohr=geometry_bohr,
        charge=0,
        multiplicity=1,
        method="rks",
        basis="sto-3g",
        xc="pbe",
        symmetry_number=1,
    )

    assert res.vibrational.is_linear is True
    # 3*2 - 5 = 1 vibrational mode
    assert res.vibrational.n_vib == 1
    assert len(res.vibrational.modes) == 1
    freq = res.frequencies_cm[0]
    # HF stretch is ~3500 - 4500 cm^-1
    assert 3000.0 < freq < 5000.0
    assert res.vibrational.modes[0].reduced_mass_amu < 1.1


def test_native_backend_frequency_task():
    """Tests NativeCalchem backend executing task='frequency'."""
    native = NativeCalchem()
    assert native.capabilities().supports_task("frequency")

    b2a = 0.529177210903
    coords_ang = [
        [0.0, 0.0, 0.0],
        [0.0, 1.4304 * b2a, -1.1072 * b2a],
        [0.0, -1.4304 * b2a, -1.1072 * b2a],
    ]
    mol = MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=coords_ang,
        charge=0,
        multiplicity=1,
    )

    req = CalculationRequest(
        task="frequency",
        molecule=mol,
        backend="native",
        method="b3lyp",
        basis="sto-3g",
        options={
            "temperature_k": 300.0,
            "pressure_pa": 100000.0,
            "sigma": 2,
        },
    )

    res = native.run(req)
    assert res.status == "succeeded"
    assert res.scf_converged is True
    assert res.task == "frequency"
    assert res.frequencies_cm is not None
    assert len(res.frequencies_cm) == 3
    assert len(res.vibrational_modes) == 3

    assert res.zpve_hartree is not None and res.zpve_hartree > 0.015
    assert res.enthalpy_hartree is not None
    assert res.gibbs_free_energy_hartree is not None
    assert res.entropy_cal_mol_k is not None
    assert res.heat_capacity_cp_cal_mol_k is not None

    mode0 = res.vibrational_modes[0]
    assert mode0.mode_index == 1
    assert mode0.frequency_cm > 1000.0
    assert not mode0.is_imaginary
    assert mode0.reduced_mass_amu > 0.0
    assert len(mode0.displacement) == 3


class _InMemoryNativeService:
    def __init__(self):
        self._backend = NativeCalchem()

    def run(self, request):
        return self._backend.run(request)


def test_opv_workflow_thermochemistry():
    """Tests OPVWorkflow.thermochemistry method."""
    service = _InMemoryNativeService()
    opv = OPVWorkflow(service=service, backend="native")

    b2a = 0.529177210903
    coords_ang = [
        [0.0, 0.0, 0.0],
        [0.0, 1.4304 * b2a, -1.1072 * b2a],
        [0.0, -1.4304 * b2a, -1.1072 * b2a],
    ]
    mol = MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=coords_ang,
        charge=0,
        multiplicity=1,
    )

    res = opv.thermochemistry(
        mol,
        method="b3lyp",
        basis="sto-3g",
        temperature_k=298.15,
        pressure_pa=101325.0,
        name="Water_OPV",
    )

    assert isinstance(res, OPVThermochemistryResult)
    assert res.name == "Water_OPV"
    assert res.n_vibrations == 3
    assert res.n_imaginary_modes == 0
    assert not res.is_transition_state
    assert len(res.frequencies_cm) == 3
    assert res.zpve_hartree > 0.015
    assert "CalChem RRHO Thermochemistry" in res.summary
    assert "Local Minimum" in res.summary
