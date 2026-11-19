import numpy as np
import pytest
from calchem.backends.native import NativeCalchem
from calchem.core.request import CalculationRequest, MoleculeSpec

def test_cam_b3lyp_gradient_accuracy_finite_diff():
    backend = NativeCalchem()
    # H2 test molecule
    r0 = 1.40  # Bohr
    bohr_to_ang = 0.529177210903

    def get_mol(r_bohr):
        return MoleculeSpec(
            symbols=["H", "H"],
            coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, r_bohr * bohr_to_ang]],
            charge=0,
            multiplicity=1,
        )

    # Analytical gradient at r0
    req_grad = CalculationRequest(
        task="single_point",
        backend="native",
        molecule=get_mol(r0),
        method="camb3lyp",
        basis="6-31g",
        options={"conv_energy": 1e-11, "conv_density": 1e-9},
    )
    res_grad = backend.single_point(req_grad)
    assert res_grad.scf_converged

    from calchem import _core
    wf = _core.run_single_point(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, r0]],
        0, 1, "camb3lyp", "6-31g",
        conv_energy=1e-11, conv_density=1e-9, xc="cam-b3lyp"
    )
    ana_grad = _core.run_gradient(
        [1, 1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, r0]],
        0, 1, "6-31g",
        "rks",
        xc="cam-b3lyp",
        grid_level="fine",
    )
    ana_gz = float(ana_grad[1, 2])

    # Central finite differences: d = 0.0005 Bohr with matching fine grid
    delta = 0.0005
    wf_plus = _core.run_single_point(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, r0 + delta]],
        0, 1, "rks", "6-31g",
        conv_energy=1e-12, conv_density=1e-10, xc="cam-b3lyp", grid_level="fine"
    )
    wf_minus = _core.run_single_point(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, r0 - delta]],
        0, 1, "rks", "6-31g",
        conv_energy=1e-12, conv_density=1e-10, xc="cam-b3lyp", grid_level="fine"
    )
    fd_gz = (wf_plus.total_energy_hartree - wf_minus.total_energy_hartree) / (2.0 * delta)

    diff = abs(ana_gz - fd_gz)
    print(f"CAM-B3LYP Analytical: {ana_gz:.8f}, Finite Diff: {fd_gz:.8f}, Diff: {diff:.3e}")
    assert diff < 1e-5, f"Gradient must match finite differences to < 1e-5, got {diff:.3e}"


def test_tddft_with_camb3lyp_and_wb97xd():
    from calchem import _core
    symbols = ["H", "H"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.40]]

    # 1. TDDFT with CAM-B3LYP
    res_cam = _core.run_tddft(
        symbols, coords,
        0, 1, "6-31g",
        xc="cam-b3lyp",
        n_states=2,
        tda=True,
    )
    assert len(res_cam.states) >= 2
    for s in res_cam.states:
        assert s.excitation_energy_ev > 0.0
        assert s.excitation_energy_ev < 50.0

    # 2. TDDFT with wB97X-D now runs natively
    res_wb = _core.run_tddft(symbols, coords, 0, 1, "6-31g", xc="wb97x-d", n_states=2, tda=True)
    assert len(res_wb.states) >= 2
    for s in res_wb.states:
        assert 0.0 < s.excitation_energy_ev < 50.0


def test_cpcm_solvation_with_range_separated_hybrids():
    from calchem import _core
    symbols = ["O", "H", "H"]
    coords = [[0.0, 0.0, 0.22], [0.0, 1.43, -0.88], [0.0, -1.43, -0.88]]

    for xc in ["cam-b3lyp", "wb97x-d"]:
        res = _core.run_solvation(
            symbols, coords,
            "sto-3g",
            "water",
            "cpcm",
            0, 1,
            "rks",
            xc=xc,
        )
        assert res.delta_g_elec_kcal_mol < 0.0

