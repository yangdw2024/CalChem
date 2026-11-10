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
