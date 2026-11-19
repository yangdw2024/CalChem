"""Unit tests for CalChem OPV (Organic Photovoltaics) workflow."""

import pytest
from calchem.core.request import MoleculeSpec
from calchem.workflows.opv import HARTREE_TO_EV, OPVWorkflow


@pytest.fixture
def h2_spec():
    # Model small molecule
    return MoleculeSpec(
        symbols=["H", "H"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        multiplicity=1,
    )


@pytest.fixture
def h2o_spec():
    return MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
        ],
        charge=0,
        multiplicity=1,
    )


def test_opv_single_point(h2_spec):
    wf = OPVWorkflow(backend="native")
    res = wf.single_point(h2_spec, method="rhf", basis="sto-3g", name="H2_Donor")

    assert res.converged
    assert res.scf_energy_hartree is not None
    assert res.homo_energy_hartree is not None
    assert res.lumo_energy_hartree is not None
    assert res.homo_ev == pytest.approx(res.homo_energy_hartree * HARTREE_TO_EV, rel=1e-6)
    assert res.lumo_ev == pytest.approx(res.lumo_energy_hartree * HARTREE_TO_EV, rel=1e-6)
    assert res.gap_ev == pytest.approx(res.lumo_ev - res.homo_ev, rel=1e-6)
    assert res.gap_ev > 0.0


def test_opv_optimize(h2_spec):
    wf = OPVWorkflow(backend="native")
    res = wf.optimize(h2_spec, method="rhf", basis="sto-3g", name="H2_Opt")

    assert res.converged
    assert res.optimized_geometry is not None
    assert len(res.optimized_geometry) == 2
    # Verify bond length roughly 0.71 Å in STO-3G
    dist = abs(res.optimized_geometry[1][2] - res.optimized_geometry[0][2])
    assert 0.70 < dist < 0.73


def test_opv_align_donor_acceptor(h2_spec, h2o_spec):
    wf = OPVWorkflow(backend="native")
    pair = wf.align_donor_acceptor(
        donor=h2_spec,
        acceptor=h2o_spec,
        method="rhf",
        basis="sto-3g",
        optimize_geometries=False,
    )

    assert pair.donor.converged
    assert pair.acceptor.converged
    assert pair.delta_lumo_ev == pytest.approx(pair.donor.lumo_ev - pair.acceptor.lumo_ev, rel=1e-6)
    assert pair.delta_homo_ev == pytest.approx(pair.donor.homo_ev - pair.acceptor.homo_ev, rel=1e-6)
    assert pair.estimated_voc_volts >= 0.0
    assert "OPV D-A Energy Level Alignment" in pair.summary
    assert "Delta_LUMO" in pair.summary


def test_opv_four_point_reorganization_energy_both_carriers(h2o_spec):
    """Full Nelsen four-point reorganization energy (hole + electron) and exact identities."""
    wf = OPVWorkflow(backend="native")
    reorg = wf.reorganization_energy(
        h2o_spec,
        method="rhf",
        basis="sto-3g",
        carrier="both",
        name="H2O_Model",
    )

    assert reorg.method_neutral == "rhf"
    assert reorg.method_open_shell == "uhf"
    assert reorg.hole is not None
    assert reorg.electron is not None

    # Every relaxation branch from a vertical geometry to a true minimum must be strictly positive
    h = reorg.hole
    assert h.lambda_ion_relax_ev > 0.0
    assert h.lambda_neutral_relax_ev > 0.0
    assert h.lambda_ev == pytest.approx(h.lambda_ion_relax_ev + h.lambda_neutral_relax_ev, rel=1e-10)
    assert reorg.lambda_hole_mev == pytest.approx(h.lambda_ev * 1000.0, rel=1e-10)
    assert h.geometry_rmsd_angstrom > 0.01

    # Exact thermodynamic identities for hole transfer:
    # IP_vert - IP_adiab == lambda_ion_relax, and IP_vert - HEP == lambda_hole
    assert reorg.vertical_ip_ev > reorg.adiabatic_ip_ev > 0.0
    assert (reorg.vertical_ip_ev - reorg.adiabatic_ip_ev) == pytest.approx(h.lambda_ion_relax_ev, abs=1e-8)
    assert (reorg.vertical_ip_ev - reorg.hole_extraction_potential_ev) == pytest.approx(h.lambda_ev, abs=1e-8)

    # Electron transfer branch & identities:
    # EA_adiab - EA_vert == lambda_ion_relax, and EEP - EA_vert == lambda_electron
    el = reorg.electron
    assert el.lambda_ion_relax_ev > 0.0
    assert el.lambda_neutral_relax_ev > 0.0
    assert el.lambda_ev == pytest.approx(el.lambda_ion_relax_ev + el.lambda_neutral_relax_ev, rel=1e-10)
    assert (reorg.adiabatic_ea_ev - reorg.vertical_ea_ev) == pytest.approx(el.lambda_ion_relax_ev, abs=1e-8)
    assert (reorg.electron_extraction_potential_ev - reorg.vertical_ea_ev) == pytest.approx(el.lambda_ev, abs=1e-8)
    assert reorg.fundamental_gap_adiabatic_ev == pytest.approx(
        reorg.adiabatic_ip_ev - reorg.adiabatic_ea_ev, abs=1e-8
    )
    assert "Four-Point Reorganization Energy" in reorg.summary


def test_opv_reorganization_hole_only_631g(h2o_spec):
    """Split-valence 6-31G single-carrier ('hole') four-point reorganization energy."""
    wf = OPVWorkflow(backend="native")
    reorg = wf.reorganization_energy(
        h2o_spec,
        method="rhf",
        basis="6-31g",
        carrier="hole",
        name="H2O_631G",
    )
    assert reorg.hole is not None
    assert reorg.electron is None
    assert reorg.lambda_electron_ev is None
    assert reorg.lambda_hole_ev > 0.05
    assert reorg.hole.lambda_ion_relax_ev > 0.0
    assert reorg.hole.lambda_neutral_relax_ev > 0.0


def test_opv_reorganization_validation_errors(h2o_spec):
    wf = OPVWorkflow(backend="native")
    with pytest.raises(ValueError, match="carrier must be"):
        wf.reorganization_energy(h2o_spec, carrier="exciton")  # type: ignore[arg-type]

    triplet = MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=h2o_spec.coordinates_angstrom,
        charge=0,
        multiplicity=3,
    )
    with pytest.raises(ValueError, match="closed-shell singlet"):
        wf.reorganization_energy(triplet)


def test_opv_halogenated_reorganization_lanl2dz():
    pytest.importorskip("calchem._core")
    wf = OPVWorkflow(backend="native")
    hbr = MoleculeSpec(
        symbols=["H", "Br"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]],
        charge=0,
        multiplicity=1,
    )
    reorg = wf.reorganization_energy(hbr, basis="lanl2dz", carrier="hole")
    assert reorg.hole is not None
    assert reorg.hole.lambda_ev > 0.0
    assert reorg.hole.lambda_ion_relax_ev >= 0.0
    assert reorg.hole.lambda_neutral_relax_ev >= 0.0
    assert reorg.vertical_ip_ev >= reorg.adiabatic_ip_ev - 1e-5
    assert reorg.adiabatic_ip_ev >= reorg.hole_extraction_potential_ev - 1e-5
    assert (reorg.vertical_ip_ev - reorg.hole_extraction_potential_ev) == pytest.approx(
        reorg.hole.lambda_ev, abs=1e-8
    )
    assert 8.0 < reorg.vertical_ip_ev < 15.0

