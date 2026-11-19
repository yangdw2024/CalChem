"""End-to-end tests for the pure Rust Kohn-Sham DFT engine (`calchem-dft` + `RKS`/`UKS` + `B3LYP`)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from calchem.backends.native import NativeCalchem
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.props.isosurface import IsosurfaceStreamService, decode_ccmsh_packet
from calchem.utils.constants import ANGSTROM_TO_BOHR
from calchem.workflows.opv import OPVWorkflow


def _water_spec() -> MoleculeSpec:
    r = 0.9572
    half = math.radians(104.52 / 2.0)
    hy = r * math.sin(half)
    hz = -r * math.cos(half)
    return MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, hy, hz], [0.0, -hy, hz]],
        charge=0,
        multiplicity=1,
    )


def test_core_dft_functionals_water_benchmark_and_analytical_gradient():
    rust_core = pytest.importorskip("calchem._core")

    supported = set(rust_core.supported_dft_functionals())
    assert {"svwn5", "blyp", "pbe", "b3lyp", "b3lyp5", "pbe0"}.issubset(supported)

    w = _water_spec()
    coords_bohr = [[c * ANGSTROM_TO_BOHR for c in row] for row in w.coordinates_angstrom]
    wf = rust_core.run_single_point(
        w.symbols,
        coords_bohr,
        0,
        1,
        "b3lyp",
        "6-31g(d)",
        grid_level="fine",
    )
    assert wf.converged
    assert wf.xc_functional == "b3lyp"
    # NIST CCCBDB B3LYP/6-31G(d) (Cartesian 6D) reference for H2O is ~ -76.40895 Eh
    assert wf.total_energy_hartree == pytest.approx(-76.40895, abs=1.5e-3)
    assert wf.dipole_magnitude_debye is not None
    assert 1.85 <= wf.dipole_magnitude_debye <= 2.35

    # Distorted H2O analytical gradient vs central finite difference (both closed-shell RKS and open-shell UKS)
    distorted = [
        [0.02 * ANGSTROM_TO_BOHR, -0.03 * ANGSTROM_TO_BOHR, 0.07 * ANGSTROM_TO_BOHR],
        [0.83 * ANGSTROM_TO_BOHR, 0.57 * ANGSTROM_TO_BOHR, -0.36 * ANGSTROM_TO_BOHR],
        [-0.72 * ANGSTROM_TO_BOHR, 0.64 * ANGSTROM_TO_BOHR, -0.40 * ANGSTROM_TO_BOHR],
    ]
    for charge, mult in [(0, 1), (1, 2)]:
        g_ana = rust_core.run_gradient(
            ["O", "H", "H"],
            distorted,
            charge=charge,
            multiplicity=mult,
            basis="6-31g(d)",
            method="b3lyp",
            grid_level="coarse",
        )
        assert g_ana.shape == (3, 3)

        h = 1e-4
        for a_idx, q_idx in [(0, 2), (1, 0), (2, 1)]:
            cp = [list(r) for r in distorted]
            cm = [list(r) for r in distorted]
            cp[a_idx][q_idx] += h
            cm[a_idx][q_idx] -= h
            ep = rust_core.run_single_point(
                ["O", "H", "H"],
                cp,
                charge,
                mult,
                "b3lyp",
                "6-31g(d)",
                conv_energy=1e-12,
                conv_density=1e-10,
                grid_level="coarse",
            ).total_energy_hartree
            em = rust_core.run_single_point(
                ["O", "H", "H"],
                cm,
                charge,
                mult,
                "b3lyp",
                "6-31g(d)",
                conv_energy=1e-12,
                conv_density=1e-10,
                grid_level="coarse",
            ).total_energy_hartree
            fd = (ep - em) / (2.0 * h)
            assert abs(float(g_ana[a_idx, q_idx]) - fd) < 5e-6


def test_native_backend_dft_routing_no_silent_fallback_and_ecp_opt():
    pytest.importorskip("calchem._core")
    backend = NativeCalchem()
    w = _water_spec()

    # 1. Direct functional method="b3lyp" routes to RKS-B3LYP-Rust
    res_b3lyp = backend.single_point(
        CalculationRequest(
            task="single_point",
            molecule=w,
            backend="native",
            method="b3lyp",
            basis="6-31G*",
        )
    )
    assert res_b3lyp.status == "succeeded"
    assert res_b3lyp.actual_method == "RKS-B3LYP-Rust(6-31g(d))"
    assert res_b3lyp.xc == "b3lyp"

    # 2. method="dft", xc="pbe0" routes to RKS-PBE0-Rust
    res_pbe0 = backend.single_point(
        CalculationRequest(
            task="single_point",
            molecule=w,
            backend="native",
            method="dft",
            xc="pbe0",
            basis="6-31g(d)",
        )
    )
    assert res_pbe0.status == "succeeded"
    assert res_pbe0.actual_method == "RKS-PBE0-Rust(6-31g(d))"

    # 3. max_iter=1 must return not_converged without silently falling back to Python
    res_nc = backend.single_point(
        CalculationRequest(
            task="single_point",
            molecule=w,
            backend="native",
            method="b3lyp",
            basis="sto-3g",
            options={"max_iter": 1},
        )
    )
    assert res_nc.status == "not_converged"
    assert res_nc.scf_converged is False
    assert "Rust" in (res_nc.actual_method or "")

    # 4. Halogen ECP HBr geometry optimization with PBE0/LANL2DZ
    hbr = MoleculeSpec(
        symbols=["H", "Br"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.32]],
    )
    res_opt = backend.optimize(
        CalculationRequest(
            task="optimize",
            molecule=hbr,
            backend="native",
            method="pbe0",
            basis="LANL2DZ",
        )
    )
    assert res_opt.status == "succeeded"
    assert res_opt.actual_method == "Rust-RKS-PBE0-BFGS"
    assert res_opt.optimized_coordinates_angstrom is not None
    r_eq = abs(
        res_opt.optimized_coordinates_angstrom[1][2]
        - res_opt.optimized_coordinates_angstrom[0][2]
    )
    assert 1.40 <= r_eq <= 1.50


class _InMemoryNativeService:
    """Lockless in-memory service adapter for unit-testing OPVWorkflow + NativeCalchem."""

    def __init__(self) -> None:
        self._backend = NativeCalchem()

    def run(self, request: CalculationRequest):
        return self._backend.run(request)


def test_opv_workflow_b3lyp_da_pair_reorganization_and_orbital_mesh_stream():
    rust_core = pytest.importorskip("calchem._core")
    wf_engine = OPVWorkflow(service=_InMemoryNativeService(), backend="native")

    donor = MoleculeSpec(
        symbols=["C", "C", "N", "H", "H", "H", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.0],
            [1.34, 0.0, 0.0],
            [-0.72, 1.18, 0.0],
            [-0.54, -0.94, 0.0],
            [1.90, -0.93, 0.0],
            [1.90, 0.93, 0.0],
            [-1.72, 1.12, 0.0],
            [-0.28, 2.08, 0.0],
        ],
    )
    acceptor = MoleculeSpec(
        symbols=["C", "C", "C", "N", "H", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.0],
            [1.34, 0.0, 0.0],
            [-0.74, 1.22, 0.0],
            [-1.33, 2.22, 0.0],
            [-0.55, -0.94, 0.0],
            [1.90, -0.93, 0.0],
            [1.90, 0.93, 0.0],
        ],
    )

    pair = wf_engine.align_donor_acceptor(
        donor,
        acceptor,
        method="b3lyp",
        basis="6-31g(d)",
        optimize_geometries=False,
    )
    assert pair.donor.converged and pair.acceptor.converged
    # B3LYP frontier gaps are quantitatively compressed into ~5.0-8.5 eV (vs ~14-16 eV in RHF)
    assert pair.donor.gap_ev is not None and 5.0 <= pair.donor.gap_ev <= 8.5
    assert pair.acceptor.gap_ev is not None and 5.0 <= pair.acceptor.gap_ev <= 8.0
    assert pair.delta_lumo_ev > 0.5
    assert pair.charge_transfer_feasible is True
    assert pair.estimated_voc_volts > 0.0

    # Four-point reorganization energy on ethylene at B3LYP / 6-31G (pure Rust RKS + UKS BFGS)
    c2h4 = MoleculeSpec(
        symbols=["C", "C", "H", "H", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.67],
            [0.0, 0.0, -0.67],
            [0.0, 0.93, 1.23],
            [0.0, -0.93, 1.23],
            [0.0, 0.93, -1.23],
            [0.0, -0.93, -1.23],
        ],
    )
    reorg = wf_engine.reorganization_energy(
        c2h4,
        method="b3lyp",
        basis="6-31g",
        carrier="hole",
    )
    assert reorg.hole is not None
    assert reorg.lambda_hole_mev is not None and reorg.lambda_hole_mev > 50.0
    assert reorg.vertical_ip_ev is not None and reorg.adiabatic_ip_ev is not None
    assert reorg.vertical_ip_ev >= reorg.adiabatic_ip_ev - 1e-4

    # 3D B3LYP HOMO isosurface extraction & CCMSH\x01 binary streaming
    c2h4_bohr = [[c * ANGSTROM_TO_BOHR for c in row] for row in c2h4.coordinates_angstrom]
    wf_b3lyp = rust_core.run_single_point(
        c2h4.symbols, c2h4_bohr, 0, 1, "b3lyp", "6-31g(d)"
    )
    homo_idx = IsosurfaceStreamService.homo_index(wf_b3lyp)
    cache = IsosurfaceStreamService.cache_orbital_grid(wf_b3lyp, mo_idx=homo_idx, spacing_bohr=0.25)
    packet = cache.to_binary_packet(isovalue=0.05)
    decoded = decode_ccmsh_packet(packet)
    assert decoded.mo_index == homo_idx
    assert decoded.positive.vertices.shape[0] > 50
    assert decoded.negative.vertices.shape[0] > 50
    assert np.allclose(
        np.linalg.norm(decoded.positive.normals, axis=1), 1.0, atol=1e-4
    )
