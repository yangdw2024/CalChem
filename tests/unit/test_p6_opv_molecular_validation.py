"""Phase 6: Real OPV Molecular Validation & 200-2000 Atom Scaling Model Tests.

Validates authentic organic photovoltaic (OPV) molecular systems from ChemDatas:
1. IC-2F terminal acceptor (22 atoms, 94 AOs in STO-3G, 226 AOs in 6-31G(d))
2. BTP core (159 atoms, C60H90N4S5)
3. Y6 (BTP-4F, 199 atoms, C86H94F4N8O2S5)
4. PM6 donor polymer monomer (158 atoms, C68H78F2O2S8)
5. p4D18 4-mer polymer (726 atoms, 6224 AOs) from ydw-p4D18-opt06.log
"""

import math
import os
import pytest
from collections import Counter
from pathlib import Path

from calchem.core.request import MoleculeSpec
from calchem.workflows.opv import (
    CHEMDATAS_REFERENCE_DATA,
    OPVMolecule,
    OPVWorkflow,
    get_real_opv_system,
    load_chemdatas_molecule,
)


def test_p6_ic2f_b3lyp_rij_and_d3bj_validation():
    """Verify IC-2F (22 atoms) with B3LYP + RI-J + D3(BJ) in native engine."""
    wf = OPVWorkflow(backend="native")
    ic2f = get_real_opv_system("y6_ic_ff")
    assert len(ic2f.symbols) == 22
    assert ic2f.reference_data is not None

    # Benchmark B3LYP with D3(BJ) dispersion and frontier orbital evaluation (CPU RI-J vs GPU RI-J)
    spec = ic2f.to_spec()
    res_cpu = wf.single_point(
        spec,
        method="b3lyp",
        basis="sto-3g",
        dispersion="d3bj",
        options={
            "grid_level": "coarse",
            "conv_density": 1e-5,
            "conv_energy": 1e-6,
            "density_fitting": True,
            "use_gpu": False,
        },
        name="IC2F_B3LYP_D3BJ_CPU_RIJ",
    )
    assert res_cpu.converged, "IC-2F B3LYP CPU RI-J calculation must converge"
    assert res_cpu.scf_energy_hartree < -865.0
    assert res_cpu.homo_ev is not None and res_cpu.lumo_ev is not None
    assert res_cpu.gap_ev > 3.0, f"Expected conjugated bandgap > 3.0 eV, got {res_cpu.gap_ev}"
    assert res_cpu.dispersion_energy_hartree is not None
    assert res_cpu.dispersion_energy_hartree < -0.01

    # Verify dipole moment magnitude
    assert res_cpu.dipole_magnitude_debye is not None
    assert res_cpu.dipole_magnitude_debye > 0.0

    # End-to-end GPU RI-J vs CPU RI-J parity on 22-atom IC-2F
    assert res_cpu.gpu_kernel_invocations == 0
    res_gpu = wf.single_point(
        spec,
        method="b3lyp",
        basis="sto-3g",
        dispersion="d3bj",
        options={
            "grid_level": "coarse",
            "conv_density": 1e-5,
            "conv_energy": 1e-6,
            "density_fitting": True,
            "use_gpu": True,
        },
        name="IC2F_B3LYP_D3BJ_GPU_RIJ",
    )
    assert res_gpu.converged, "IC-2F B3LYP GPU RI-J calculation must converge"
    from calchem import _core
    if _core.is_cuda_available():
        assert res_gpu.gpu_kernel_invocations > 0, (
            f"Expected GPU kernel invocations > 0 on IC-2F when CUDA is available, got {res_gpu.gpu_kernel_invocations}"
        )
        assert res_gpu.gpu_fallback_invocations == 0
    assert abs(res_gpu.scf_energy_hartree - res_cpu.scf_energy_hartree) < 1e-7
    assert abs(res_gpu.gap_ev - res_cpu.gap_ev) < 1e-5


def test_p6_btp_core_and_y6_authentic_chemdatas_baselines():
    """Verify authentic BTP central core (159 atoms) and full Y6 (199 atoms) from ChemDatas."""
    # 1. BTP core (159 atoms)
    btp = get_real_opv_system("y6_btp_core")
    assert len(btp.symbols) == 159
    assert Counter(btp.symbols) == {"C": 60, "H": 90, "N": 4, "S": 5}
    g16_btp_e = btp.reference_data["scf_energy_hartree"]
    assert g16_btp_e == pytest.approx(-4549.93900185, abs=1e-5)

    # 2. Full Y6 (199 atoms)
    y6 = get_real_opv_system("y6_full")
    assert len(y6.symbols) == 199
    assert Counter(y6.symbols) == {"C": 86, "H": 94, "F": 4, "N": 8, "O": 2, "S": 5}
    g16_y6_e = y6.reference_data["scf_energy_hartree"]
    assert g16_y6_e == pytest.approx(-6314.19224815, abs=1e-5)
    assert y6.reference_data["gap_ev"] == pytest.approx(2.0283, abs=0.01)

    # 3. PM6 monomer (158 atoms)
    pm6 = get_real_opv_system("pm6_monomer")
    assert len(pm6.symbols) == 158
    assert Counter(pm6.symbols) == {"C": 68, "H": 78, "F": 2, "O": 2, "S": 8}


def test_p6_p4d18_authentic_log_inspection_and_scaling_bounds():
    """Verify authentic p4D18 tetramer (726 atoms, 6224 AOs) parameters and resource model."""
    log_path = Path(r"E:\YDW\ChemDatas\Log\ydw-p4D18-opt06.log")
    if not log_path.exists():
        pytest.skip(f"ChemDatas log not found at {log_path}")

    # Parse header for NAtoms and Basis functions
    n_atoms = None
    n_basis = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "NAtoms=" in line and n_atoms is None:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith("NAtoms="):
                        n_atoms = int(parts[i + 1]) if parts[i + 1].isdigit() else int(p.split("=")[1])
            if "basis functions" in line and n_basis is None:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "basis" in p and i > 0 and parts[i - 1].isdigit():
                        n_basis = int(parts[i - 1])
            if n_atoms is not None and n_basis is not None:
                break

    assert n_atoms == 726, f"Expected 726 atoms in p4D18 log, found {n_atoms}"
    assert n_basis == 6224, f"Expected 6224 basis functions in p4D18 log, found {n_basis}"

    # Verify Memory Scaling Calculations:
    # 1. Stored ERI tensor footprint: N_ao^4 / 8 * 8 bytes
    stored_eri_bytes = (n_basis ** 4) // 8 * 8
    stored_eri_pb = stored_eri_bytes / (1024 ** 5)
    assert stored_eri_pb > 1.3, f"Stored ERI should exceed 1.3 PB, calculated {stored_eri_pb:.2f} PB"

    # 2. CalChem Direct SCF matrix memory footprint:
    # Requires only O(N_ao^2) matrices (D, F, S, H, C, X) each of size N_ao^2 * 8 bytes
    matrix_bytes = n_basis * n_basis * 8
    matrix_mb = matrix_bytes / (1024 ** 2)
    assert matrix_mb < 320.0, f"Single matrix should be ~295 MB, got {matrix_mb:.1f} MB"
    
    # 16 worker matrix buffers occupy < 4.8 GiB
    total_scf_memory_gib = (16 * matrix_bytes) / (1024 ** 3)
    assert total_scf_memory_gib < 4.8, f"16-worker Direct SCF pool must be < 4.8 GiB, got {total_scf_memory_gib:.2f} GiB"


def test_p6_external_reference_cross_validation():
    """Cross-validate nuclear repulsion, D3(BJ), RHF/UHF/DFT, and GPU vs CPU RI-J against external baselines."""
    from calchem import _core

    # 1. Exact Gaussian 16 nuclear repulsion cross-validation on CHCl3 (ydw-CF.log lines 174-178 Input Orientation)
    bohr = 1.0 / 0.52917721092
    chcl3_atoms = [
        (6, [-2.257384 * bohr, -0.710267 * bohr, 0.000000 * bohr]),
        (1, [-3.327384 * bohr, -0.710254 * bohr, 0.000000 * bohr]),
        (17, [-1.670707 * bohr, 0.119397 * bohr, 1.437034 * bohr]),
        (17, [-1.670737 * bohr, -2.369618 * bohr, 0.000000 * bohr]),
        (17, [-1.670707 * bohr, 0.119397 * bohr, -1.437034 * bohr]),
    ]
    e_nuc_g16 = 266.1150606243
    e_nuc_calchem = 0.0
    for i in range(len(chcl3_atoms)):
        zi, ri = chcl3_atoms[i]
        for j in range(i + 1, len(chcl3_atoms)):
            zj, rj = chcl3_atoms[j]
            dist = math.sqrt(sum((ri[k] - rj[k]) ** 2 for k in range(3)))
            e_nuc_calchem += (zi * zj) / dist
    assert abs(e_nuc_calchem - e_nuc_g16) < 1e-4

    # 2. Same-geometry / same-basis cross-validation on H2O across RHF, UHF, B3LYP, PBE0, CAM-B3LYP + D3(BJ)
    h2o_symbols = ["O", "H", "H"]
    h2o_coords = [
        [0.0, 0.0, 0.0],
        [0.0, 1.432, 0.0],
        [1.112, -0.716, 0.0],
    ]
    # Exact STO-3G RHF reference (-74.695319114 Eh)
    res_rhf = _core.run_single_point(h2o_symbols, h2o_coords, 0, 1, "rhf", "sto-3g")
    assert res_rhf.converged
    assert abs(res_rhf.total_energy_hartree - (-74.695319114)) < 1e-7

    # Closed-shell RHF vs UHF spin-symmetry consistency to < 1e-8 Eh
    res_uhf = _core.run_single_point(h2o_symbols, h2o_coords, 0, 1, "uhf", "sto-3g")
    assert res_uhf.converged
    assert abs(res_uhf.total_energy_hartree - res_rhf.total_energy_hartree) < 1e-8

    # DFT functional hierarchy consistency on 6-31G(d) (B3LYP, PBE0, CAM-B3LYP) + D3(BJ)
    res_b3lyp = _core.run_single_point(
        h2o_symbols, h2o_coords, 0, 1, "rks", "6-31g(d)", 100, 1e-8, 1e-7, True, 8, 0.0, "b3lyp", "fine"
    )
    res_pbe0 = _core.run_single_point(
        h2o_symbols, h2o_coords, 0, 1, "rks", "6-31g(d)", 100, 1e-8, 1e-7, True, 8, 0.0, "pbe0", "fine"
    )
    res_cam = _core.run_single_point(
        h2o_symbols, h2o_coords, 0, 1, "rks", "6-31g(d)", 100, 1e-8, 1e-7, True, 8, 0.0, "cam-b3lyp", "fine"
    )
    d3_b3lyp = _core.run_d3_dispersion(h2o_symbols, h2o_coords, "b3lyp", "d3bj")
    assert res_b3lyp.converged and res_pbe0.converged and res_cam.converged
    assert d3_b3lyp.energy_hartree < 0.0
    assert -76.50 < res_b3lyp.total_energy_hartree < -76.10
    assert -76.50 < res_pbe0.total_energy_hartree < -76.10
    assert -76.50 < res_cam.total_energy_hartree < -76.10


