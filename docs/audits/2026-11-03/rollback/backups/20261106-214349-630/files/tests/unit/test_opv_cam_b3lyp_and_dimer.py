"""Unit tests for Long-Range Corrected Functionals (CAM-B3LYP / wB97X-D)
and OPV PM6:Y6 donor-acceptor dimer binding energy in CalChem pure Rust engine.
"""

import pytest

from calchem import _core
from calchem.core.request import MoleculeSpec
from calchem.workflows.opv import (
    HARTREE_TO_EV,
    OPVMolecule,
    OPVWorkflow,
    build_da_stacking_dimer,
    get_pm6_fragment,
    get_y6_fragment,
)


def test_cam_b3lyp_and_wb97xd_single_point_rust():
    """Verify that CAM-B3LYP and wB97X-D pure Rust kernels converge and yield valid frontier orbitals."""
    wf = OPVWorkflow(backend="native")

    h2o = MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
        ],
        charge=0,
        multiplicity=1,
    )

    # CAM-B3LYP
    res_cam = wf.single_point(h2o, method="camb3lyp", basis="sto-3g", name="H2O_CAM")
    assert res_cam.converged
    assert res_cam.scf_energy_hartree is not None
    assert res_cam.scf_energy_hartree < -74.0
    assert res_cam.homo_ev is not None and res_cam.lumo_ev is not None
    assert res_cam.gap_ev is not None and res_cam.gap_ev > 0.0

    # wB97X-D
    res_wb97x = wf.single_point(h2o, method="wb97xd", basis="sto-3g", name="H2O_wB97XD")
    assert res_wb97x.converged
    assert res_wb97x.scf_energy_hartree is not None
    assert res_wb97x.scf_energy_hartree < -74.0
    assert res_wb97x.gap_ev is not None and res_wb97x.gap_ev > 0.0


def test_pm6_and_y6_fragments_and_dimer_builder():
    """Verify geometry generation and properties for PM6 donor, Y6 acceptor, and pi-pi stacking dimer."""
    pm6 = get_pm6_fragment(fluorinated=True)
    assert isinstance(pm6, OPVMolecule)
    assert pm6.name == "PM6_BDT-2F_Core"
    assert len(pm6.symbols) == 18
    assert "F" in pm6.symbols
    assert "S" in pm6.symbols

    y6 = get_y6_fragment(fluorinated=True)
    assert isinstance(y6, OPVMolecule)
    assert y6.name == "Y6_BT-2F_Core"
    assert len(y6.symbols) == 13
    assert "N" in y6.symbols
    assert "F" in y6.symbols

    dimer = build_da_stacking_dimer(pm6, y6, distance_angstrom=3.5, lateral_shift_angstrom=0.5)
    assert len(dimer.symbols) == 18 + 13
    assert len(dimer.coordinates_angstrom) == 31
    assert dimer.charge == 0
    assert dimer.multiplicity == 1

    # Check interplanar distance along z
    z_coords_pm6 = [c[2] for c in dimer.coordinates_angstrom[:18]]
    z_coords_y6 = [c[2] for c in dimer.coordinates_angstrom[18:]]
    assert all(abs(z - (-1.75)) < 1e-6 for z in z_coords_pm6)
    assert all(abs(z - 1.75) < 1e-6 for z in z_coords_y6)


def test_pm6_y6_da_level_alignment_and_voc():
    """Verify donor-acceptor level alignment and Scharber Voc prediction for PM6:Y6."""
    wf = OPVWorkflow(backend="native")
    pm6 = get_pm6_fragment(fluorinated=True)
    y6 = get_y6_fragment(fluorinated=True)

    pair = wf.align_donor_acceptor(
        donor=pm6.to_spec(),
        acceptor=y6.to_spec(),
        method="b3lyp",
        basis="sto-3g",
        optimize_geometries=False,
    )

    assert pair.donor.converged
    assert pair.acceptor.converged

    # Donor (BDT-2F) LUMO is higher than Acceptor (BT-2F) LUMO
    assert pair.delta_lumo_ev > 0.30
    assert pair.charge_transfer_feasible is True
    # Scharber open-circuit voltage is positive and physically realistic
    assert pair.estimated_voc_volts > 0.50
    assert "Exciton Dissociation Feasible: True" in pair.summary


def test_pm6_y6_dimer_binding_energy_with_d3bj():
    """Verify D3(BJ) dispersion binding energy for PM6:Y6 stacking dimer."""
    wf = OPVWorkflow(backend="native")
    pm6 = get_pm6_fragment(fluorinated=True)
    y6 = get_y6_fragment(fluorinated=True)

    dimer_res = wf.dimer_binding_energy(
        donor=pm6,
        acceptor=y6,
        distance_angstrom=3.5,
        method="camb3lyp",
        basis="sto-3g",
        dispersion="d3bj",
        optimize_dimer=False,
        name="PM6:Y6_D3BJ_Stacking",
    )

    assert dimer_res.donor.converged
    assert dimer_res.acceptor.converged
    assert dimer_res.dimer.converged

    # D3(BJ) dispersion interaction energy must be strictly attractive (negative)
    assert dimer_res.dispersion_energy_kcal_mol is not None
    assert dimer_res.dispersion_energy_kcal_mol < -10.0  # Typically -25 to -40 kcal/mol for 31 atoms
    assert dimer_res.binding_energy_kcal_mol < 0.0  # Overall bound state
    assert dimer_res.intermolecular_distance_angstrom == 3.5
    assert "D3 Dispersion Delta_E" in dimer_res.summary


def test_pm6_solvation_in_chlorobenzene():
    """Verify PCM continuum solvation in chlorobenzene for OPV donor fragment."""
    wf = OPVWorkflow(backend="native")
    pm6 = get_pm6_fragment(fluorinated=True)

    solv_res = wf.solvation_free_energy(
        pm6.to_spec(),
        solvent="chlorobenzene",
        method="b3lyp",
        basis="sto-3g",
        name="PM6_BDT-2F_CB",
    )

    assert solv_res.solvated_energy_hartree != 0.0
    assert solv_res.delta_g_solv_kcal_mol != 0.0
    assert solv_res.cavity_surface_area_angstrom2 > 100.0
    assert abs(solv_res.dielectric_constant - 5.62) < 0.1
    assert "Total Delta G_solv" in solv_res.summary
