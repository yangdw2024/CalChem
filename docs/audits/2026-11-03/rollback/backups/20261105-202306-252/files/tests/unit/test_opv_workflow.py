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
