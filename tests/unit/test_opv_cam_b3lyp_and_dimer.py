"""Unit tests for Long-Range Corrected Functionals (CAM-B3LYP / wB97X-D),
authentic ChemDatas Gaussian (.log / .gjf / .fchk) molecular systems (PM6, Y6, BTP, IC-2F, MD clusters),
D3(BJ) dispersion, and PCM solvation in CalChem pure Rust engine.
"""

from collections import Counter
import pytest

from calchem.core.request import MoleculeSpec
from calchem.workflows.opv import (
    CHEMDATAS_REFERENCE_DATA,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_MOL,
    OPVMolecule,
    OPVWorkflow,
    build_da_stacking_dimer,
    get_pm6_fragment,
    get_real_opv_system,
    get_y6_fragment,
    load_chemdatas_molecule,
)


def test_cam_b3lyp_and_wb97xd_single_point_rust():
    """Verify that Native backend safely rejects CAM-B3LYP and wB97X-D due to missing range-separated ERIs."""
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

    res_cam = wf.single_point(h2o, method="camb3lyp", basis="sto-3g", name="H2O_CAM")
    assert not res_cam.converged
    assert any("camb3lyp" in str(e).lower() for e in res_cam.errors)

    res_wb97x = wf.single_point(h2o, method="wb97xd", basis="sto-3g", name="H2O_wB97XD")
    assert not res_wb97x.converged
    assert any("wb97xd" in str(e).lower() for e in res_wb97x.errors)

    # Verify supported authentic hybrid B3LYP succeeds on H2O
    res_b3lyp = wf.single_point(h2o, method="b3lyp", basis="sto-3g", name="H2O_B3LYP")
    assert res_b3lyp.converged
    assert res_b3lyp.scf_energy_hartree < -74.0
    assert res_b3lyp.homo_ev is not None and res_b3lyp.lumo_ev is not None
    assert res_b3lyp.gap_ev > 0.0

def test_authentic_chemdatas_loaders_log_gjf_fchk():
    """Verify authentic PM6, Y6 (BTP-4F), BTP core, IC-2F terminal, and MD cluster loading from ChemDatas."""
    # 1. IC-2F acceptor terminal unit (22 atoms, C13H4F2N2O) from .log and .fchk
    ic2f = get_real_opv_system("y6_ic_ff")
    assert len(ic2f.symbols) == 22
    assert Counter(ic2f.symbols) == {"C": 13, "H": 4, "F": 2, "N": 2, "O": 1}
    assert ic2f.reference_data is not None
    assert ic2f.reference_data["scf_energy_hartree"] == pytest.approx(-881.833266861, abs=1e-6)

    fchk_path = CHEMDATAS_REFERENCE_DATA["y6_ic_ff"]["fchk_file"]
    ic2f_fchk = load_chemdatas_molecule(fchk_path, name="IC2F_FCHK")
    assert len(ic2f_fchk.symbols) == 22
    assert ic2f_fchk.symbols == ic2f.symbols
    assert ic2f_fchk.reference_data["scf_energy_hartree"] == pytest.approx(-881.833266861, abs=1e-6)
    for c_log, c_fchk in zip(ic2f.coordinates_angstrom, ic2f_fchk.coordinates_angstrom):
        for q in range(3):
            assert abs(c_log[q] - c_fchk[q]) < 1e-4

    # 2. BTP central core unit (159 atoms, C60H90N4S5)
    btp = get_real_opv_system("y6_btp_core")
    assert len(btp.symbols) == 159
    assert Counter(btp.symbols) == {"C": 60, "H": 90, "N": 4, "S": 5}
    assert btp.reference_data["scf_energy_hartree"] == pytest.approx(-4549.93900185, abs=1e-5)

    # 3. Full Y6 (BTP-4F) non-fullerene acceptor (199 atoms, C86H94F4N8O2S5)
    y6_full = get_real_opv_system("y6_full")
    assert len(y6_full.symbols) == 199
    assert Counter(y6_full.symbols) == {"C": 86, "H": 94, "F": 4, "N": 8, "O": 2, "S": 5}
    assert y6_full.reference_data["scf_energy_hartree"] == pytest.approx(-6314.19224815, abs=1e-5)
    assert y6_full.reference_data["gap_ev"] == pytest.approx(2.0283, abs=0.01)

    # 4. Full PM6 polymer repeat monomer (158 atoms, C68H78F2O2S8)
    pm6_mono = get_real_opv_system("pm6_monomer")
    assert len(pm6_mono.symbols) == 158
    assert Counter(pm6_mono.symbols) == {"C": 68, "H": 78, "F": 2, "O": 2, "S": 8}

    # 5. Authentic 60-atom MD-derived PM6:L8-BO interfacial stacking cluster
    cluster = get_real_opv_system("pm6_l8bo_cluster")
    assert len(cluster.symbols) == 60
    assert Counter(cluster.symbols) == {"C": 35, "H": 14, "F": 2, "N": 2, "O": 3, "S": 4}
    assert cluster.reference_data["scf_energy_hartree"] == pytest.approx(-3469.55737880, abs=1e-5)


def test_pm6_and_y6_fragments_and_dimer_builder():
    """Verify authentic PM6 Donor-BDD1 (32 atoms), Y6 IC-2F (22 atoms), and stacking dimer builder."""
    pm6 = get_pm6_fragment(fluorinated=True)
    assert isinstance(pm6, OPVMolecule)
    assert pm6.name == "PM6_Donor_BDD1"
    assert len(pm6.symbols) == 32
    assert Counter(pm6.symbols) == {"C": 18, "H": 8, "O": 2, "S": 4}

    y6 = get_y6_fragment(fluorinated=True)
    assert isinstance(y6, OPVMolecule)
    assert y6.name == "Y6_IC-2F_Terminal"
    assert len(y6.symbols) == 22
    assert Counter(y6.symbols) == {"C": 13, "H": 4, "F": 2, "N": 2, "O": 1}

    dimer = build_da_stacking_dimer(pm6, y6, distance_angstrom=3.5, lateral_shift_angstrom=0.5)
    assert len(dimer.symbols) == 32 + 22
    assert len(dimer.coordinates_angstrom) == 54
    assert dimer.charge == 0
    assert dimer.multiplicity == 1

    # Check interplanar centroid separation along z is 3.5 Å
    z_mean_pm6 = sum(c[2] for c in dimer.coordinates_angstrom[:32]) / 32.0
    z_mean_y6 = sum(c[2] for c in dimer.coordinates_angstrom[32:]) / 22.0
    assert abs(z_mean_pm6 - (-1.75)) < 1e-5
    assert abs(z_mean_y6 - 1.75) < 1e-5


def test_authentic_ic2f_dft_and_solvation():
    """Verify pure Rust CAM-B3LYP + D3(BJ) + PCM solvation on authentic 22-atom IC-2F acceptor unit."""
    wf = OPVWorkflow(backend="native")
    ic2f = get_y6_fragment(variant="ic2f")

    res_cam = wf.single_point(
        ic2f.to_spec(),
        method="b3lyp",
        basis="sto-3g",
        dispersion="d3bj",
        name="Y6_IC2F_B3LYP",
    )
    assert res_cam.converged
    assert res_cam.scf_energy_hartree < -860.0
    assert res_cam.dispersion_energy_hartree is not None
    assert res_cam.dispersion_energy_hartree < -0.02
    assert res_cam.homo_ev is not None and res_cam.lumo_ev is not None
    assert res_cam.gap_ev > 3.5

    # Solvation in chloroform (the processing solvent used in ChemDatas TH cluster calculations)
    solv_res = wf.solvation_free_energy(
        ic2f.to_spec(),
        solvent="chloroform",
        method="b3lyp",
        basis="sto-3g",
        name="Y6_IC2F_CHCl3",
    )
    assert solv_res.solvated_energy_hartree < -870.0
    assert solv_res.delta_g_solv_kcal_mol != 0.0
    assert solv_res.cavity_surface_area_angstrom2 > 180.0
    assert abs(solv_res.dielectric_constant - 4.71) < 0.1


def test_authentic_md_cluster_intermolecular_d3bj_dispersion():
    """Verify intermolecular D3(BJ) dispersion attraction on the authentic 60-atom MD cluster."""
    from calchem import _core

    b2a = 0.529177210903
    cluster = get_real_opv_system("pm6_l8bo_cluster")
    donor = get_real_opv_system("donor_bdd1")
    acceptor = get_real_opv_system("acceptor_nc1")

    def calc_d3(mol: OPVMolecule) -> float:
        coords_bohr = [[c / b2a for c in row] for row in mol.coordinates_angstrom]
        return float(_core.run_d3_dispersion(mol.symbols, coords_bohr, damping="d3bj", functional="b3lyp").energy_hartree)

    e_d3_cluster = calc_d3(cluster)
    e_d3_donor = calc_d3(donor)
    e_d3_acc = calc_d3(acceptor)
    delta_e_disp_kcal = (e_d3_cluster - e_d3_donor - e_d3_acc) * HARTREE_TO_KCAL_MOL

    # Intermolecular pi-pi dispersion stabilization between 32-atom donor-BDD1 and 28-atom Acceptor2-NC1
    assert delta_e_disp_kcal < -10.0
    assert delta_e_disp_kcal > -40.0
