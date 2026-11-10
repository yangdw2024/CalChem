"""Tests for NativeCalchem backend: Rust dual-track, constructor contracts, and option forwarding."""
import pytest
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.backends.native import NativeCalchem, _build_calculator, _spec_to_molecule
from calchem.core.errors import UnsupportedMethodError


_H2_MOL = MoleculeSpec(
    symbols=["H", "H"],
    coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7408480952641999]],  # 1.4 Bohr
    charge=0,
    multiplicity=1,
)

_H_MOL = MoleculeSpec(
    symbols=["H"],
    coordinates_angstrom=[[0.0, 0.0, 0.0]],
    charge=0,
    multiplicity=2,
)


def test_native_rhf_rust_path():
    """RHF/STO-3G must route to Rust, populate HOMO/LUMO, orbitals, and actual_method."""
    backend = NativeCalchem()
    req = CalculationRequest(
        task="single_point",
        molecule=_H2_MOL,
        backend="native",
        method="rhf",
        basis="sto-3g",
    )
    res = backend.single_point(req)
    assert res.status == "succeeded"
    assert "Rust" in (res.actual_method or "")
    assert res.total_energy_hartree == pytest.approx(-1.1167143251, abs=1e-8)
    assert res.homo_energy_hartree is not None
    assert res.lumo_energy_hartree is not None
    assert len(res.orbitals) == 2
    assert res.orbitals[0].occupation == 2.0
    assert res.orbitals[1].occupation == 0.0


def test_native_rust_options_forwarding_and_no_silent_fallback():
    """Q05: When options max_iter=1 cannot converge, it must return not_converged, NOT silently fall back."""
    backend = NativeCalchem()
    req = CalculationRequest(
        task="single_point",
        molecule=_H2_MOL,
        backend="native",
        method="rhf",
        basis="sto-3g",
        options={"max_iter": 1},
    )
    res = backend.single_point(req)
    # With max_iter=1, SCF cannot converge
    assert res.status == "not_converged"
    assert res.scf_converged is False
    assert len(res.errors) > 0


def test_native_rohf_cleanly_rejected():
    """Q04: ROHF raises UnsupportedMethodError cleanly instead of crashing on missing import."""
    mol = _spec_to_molecule(_H_MOL)
    with pytest.raises(UnsupportedMethodError, match="does not support ROHF"):
        _build_calculator(mol, "rohf", "sto-3g", None, {})


def test_native_constructors_valid():
    """Q04: Constructors for RHF, UHF, RKS, UKS, MP2, CCSD, CCD, CASSCF must construct without error."""
    mol_singlet = _spec_to_molecule(_H2_MOL)
    mol_doublet = _spec_to_molecule(_H_MOL)

    calc_rhf = _build_calculator(mol_singlet, "rhf", "sto-3g", None, {})
    assert hasattr(calc_rhf, "kernel")

    calc_uhf = _build_calculator(mol_doublet, "uhf", "sto-3g", None, {})
    assert hasattr(calc_uhf, "kernel")

    calc_rks = _build_calculator(mol_singlet, "rks", "sto-3g", "b3lyp", {})
    assert hasattr(calc_rks, "kernel")

    calc_uks = _build_calculator(mol_doublet, "uks", "sto-3g", "b3lyp", {})
    assert hasattr(calc_uks, "kernel")

    calc_mp2 = _build_calculator(mol_singlet, "mp2", "sto-3g", None, {})
    assert hasattr(calc_mp2, "kernel")

    calc_ccsd = _build_calculator(mol_singlet, "ccsd", "sto-3g", None, {})
    assert hasattr(calc_ccsd, "kernel")

    calc_ccd = _build_calculator(mol_singlet, "ccd", "sto-3g", None, {})
    assert hasattr(calc_ccd, "kernel")

    calc_casscf = _build_calculator(mol_singlet, "casscf", "sto-3g", None, {})
    assert hasattr(calc_casscf, "kernel")


def test_native_mp2_single_point():
    """MP2 single point via NativeCalchem must produce valid SCF energy and correlation energy."""
    backend = NativeCalchem()
    req = CalculationRequest(
        task="single_point",
        molecule=_H2_MOL,
        backend="native",
        method="mp2",
        basis="sto-3g",
    )
    res = backend.single_point(req)
    assert res.status == "succeeded"
    assert res.scf_energy_hartree == pytest.approx(-1.116714325, abs=1e-5)
    assert res.mp2_correction_hartree is not None
    assert res.mp2_correction_hartree < 0.0
    assert res.total_energy_hartree == pytest.approx(res.scf_energy_hartree + res.mp2_correction_hartree)


def test_native_uhf_rust_single_point_and_ion_optimize():
    """Open-shell doublet single-point and radical-ion BFGS must route to the Rust UHF core."""
    backend = NativeCalchem()
    req_sp = CalculationRequest(
        task="single_point",
        molecule=_H_MOL,
        backend="native",
        method="uhf",
        basis="sto-3g",
    )
    res_sp = backend.single_point(req_sp)
    assert res_sp.status == "succeeded"
    assert res_sp.actual_method == "UHF-Rust(sto-3g)"
    assert res_sp.total_energy_hartree == pytest.approx(-0.466581850, abs=1e-8)

    h2_cation = MoleculeSpec(
        symbols=["H", "H"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=1,
        multiplicity=2,
    )
    req_opt = CalculationRequest(
        task="optimize",
        molecule=h2_cation,
        backend="native",
        method="uhf",
        basis="sto-3g",
    )
    res_opt = backend.optimize(req_opt)
    assert res_opt.status == "succeeded"
    assert res_opt.actual_method == "Rust-UHF-BFGS"
    assert res_opt.optimized_coordinates_angstrom is not None
    r_ang = abs(res_opt.optimized_coordinates_angstrom[1][2] - res_opt.optimized_coordinates_angstrom[0][2])
    assert r_ang > 0.95


def test_native_backend_631gd_and_lanl2dz_halogens():
    pytest.importorskip("calchem._core")
    backend = NativeCalchem()

    # 1. 6-31G* polarization alias routed to Rust core
    h2o = MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.757, -0.586], [0.0, -0.757, -0.586]],
    )
    res_pol = backend.single_point(
        CalculationRequest(task="single_point", molecule=h2o, backend="native", method="rhf", basis="6-31G*")
    )
    assert res_pol.status == "succeeded"
    assert res_pol.actual_method == "RHF-Rust(6-31g(d))"
    assert len(res_pol.orbitals) == 19

    # 2. Heavy halogen HI (Z=53) single-point + BFGS optimization in LANL2DZ via Rust ECP engine
    hi = MoleculeSpec(
        symbols=["H", "I"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.50]],
    )
    res_hi_sp = backend.single_point(
        CalculationRequest(task="single_point", molecule=hi, backend="native", method="rhf", basis="LANL2DZdp")
    )
    assert res_hi_sp.status == "succeeded"
    assert res_hi_sp.actual_method == "RHF-Rust(lanl2dzdp)"

    res_hi_opt = backend.optimize(
        CalculationRequest(task="optimize", molecule=hi, backend="native", method="rhf", basis="LANL2DZ")
    )
    assert res_hi_opt.status == "succeeded"
    assert res_hi_opt.actual_method == "Rust-RHF-BFGS"
    r_hi = abs(res_hi_opt.optimized_coordinates_angstrom[1][2] - res_hi_opt.optimized_coordinates_angstrom[0][2])
    assert 1.60 <= r_hi <= 1.68


