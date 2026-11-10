"""End-to-end tests for pure Rust Molecular Electrostatic Potential (ESP),
Mulliken/Löwdin population analysis, and `CCESP\\x01` vertex-colored isodensity streaming."""

from __future__ import annotations

import math

import numpy as np
import pytest

from calchem.backends.native import NativeCalchem
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.props import (
    CCESP_HEADER_BYTES,
    CCESP_MAGIC,
    CubeFile,
    IsosurfaceStreamService,
    PopulationAnalysis,
    decode_ccesp_packet,
)
from calchem.utils.constants import ANGSTROM_TO_BOHR
from calchem.workflows.opv import OPVWorkflow


class _InMemoryNativeService:
    def __init__(self) -> None:
        self._backend = NativeCalchem()

    def run(self, request: CalculationRequest):
        return self._backend.run(request)


def _water_origin_spec(charge: int = 0, multiplicity: int = 1) -> MoleculeSpec:
    """Water with oxygen placed directly at `(0, 0, 0)` to test nuclear-cusp grid safety."""
    r = 0.9572
    half = math.radians(104.52 / 2.0)
    hy = r * math.sin(half)
    hz = -r * math.cos(half)
    return MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, hy, hz], [0.0, -hy, hz]],
        charge=charge,
        multiplicity=multiplicity,
    )


def test_population_analysis_neutral_radical_and_ecp_native_result():
    pytest.importorskip("calchem._core")
    backend = NativeCalchem()

    # 1. Closed-shell B3LYP / 6-31G(d) water
    w = _water_origin_spec(0, 1)
    res = backend.single_point(
        CalculationRequest(
            task="single_point",
            molecule=w,
            backend="native",
            method="b3lyp",
            basis="6-31g(d)",
        )
    )
    assert res.status == "succeeded"
    assert res.mulliken_charges is not None and len(res.mulliken_charges) == 3
    assert res.lowdin_charges is not None and len(res.lowdin_charges) == 3
    assert res.mulliken_spin_populations is None
    assert abs(sum(res.mulliken_charges)) < 1e-8
    assert abs(sum(res.lowdin_charges)) < 1e-8
    assert res.mulliken_charges[0] < -0.5
    assert abs(res.mulliken_charges[1] - res.mulliken_charges[2]) < 1e-8
    assert abs(res.lowdin_charges[1] - res.lowdin_charges[2]) < 1e-8

    # Direct PopulationAnalysis wrapper on Rust PyScfWavefunction (zero SciPy/PySCF)
    import calchem._core as rust_core

    w_bohr = [[c * ANGSTROM_TO_BOHR for c in row] for row in w.coordinates_angstrom]
    wf = rust_core.run_single_point(w.symbols, w_bohr, 0, 1, "b3lyp", "6-31g(d)")
    pa = PopulationAnalysis(wf)
    assert np.allclose(pa.mulliken(), res.mulliken_charges, atol=1e-10)
    assert np.allclose(pa.lowdin(), res.lowdin_charges, atol=1e-10)

    # 2. Open-shell UKS-B3LYP / 6-31G(d) H2O+ doublet
    w_cat = _water_origin_spec(1, 2)
    res_cat = backend.single_point(
        CalculationRequest(
            task="single_point",
            molecule=w_cat,
            backend="native",
            method="b3lyp",
            basis="6-31g(d)",
        )
    )
    assert res_cat.status == "succeeded"
    assert res_cat.mulliken_charges is not None
    assert res_cat.mulliken_spin_populations is not None
    assert res_cat.lowdin_spin_populations is not None
    assert abs(sum(res_cat.mulliken_charges) - 1.0) < 1e-8
    assert abs(sum(res_cat.lowdin_charges) - 1.0) < 1e-8
    assert abs(sum(res_cat.mulliken_spin_populations) - 1.0) < 1e-8
    assert abs(sum(res_cat.lowdin_spin_populations) - 1.0) < 1e-8
    assert res_cat.mulliken_spin_populations[0] > 0.80

    # 3. Halogen ECP HBr (PBE0 / LANL2DZ): valence Z_eff(Br) = 7
    hbr = MoleculeSpec(
        symbols=["H", "Br"],
        coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.43]],
    )
    res_hbr = backend.single_point(
        CalculationRequest(
            task="single_point",
            molecule=hbr,
            backend="native",
            method="pbe0",
            basis="lanl2dz",
        )
    )
    assert res_hbr.status == "succeeded"
    assert abs(sum(res_hbr.mulliken_charges)) < 1e-8
    assert abs(sum(res_hbr.lowdin_charges)) < 1e-8
    assert res_hbr.mulliken_charges[0] > 0.05 and res_hbr.mulliken_charges[1] < -0.05


def test_esp_points_grid_cube_and_nuclear_cusp_handling():
    rust_core = pytest.importorskip("calchem._core")

    w = _water_origin_spec(0, 1)
    w_bohr = [[c * ANGSTROM_TO_BOHR for c in row] for row in w.coordinates_angstrom]
    wf = rust_core.run_single_point(w.symbols, w_bohr, 0, 1, "b3lyp", "6-31g(d)")

    # Exact nuclear position [0,0,0] must be rejected by eval_esp_at_points
    with pytest.raises(ValueError, match="nucleus"):
        rust_core.eval_esp_at_points(wf, np.array([[0.0, 0.0, 0.0]], dtype=np.float64))

    # Symmetric 3D grid passing directly through [0, 0, 0] (ix=2, iy=2, iz=2) must stay finite!
    origin = (-1.0, -1.0, -1.0)
    spacing = (0.5, 0.5, 0.5)
    counts = (5, 5, 5)
    esp_grid = rust_core.eval_esp_cube_zero_copy(wf, origin, spacing, counts)
    assert esp_grid.shape == (5, 5, 5)
    assert esp_grid.dtype == np.float32
    assert np.all(np.isfinite(esp_grid))
    assert float(esp_grid[2, 2, 2]) > 1e5  # core-clamped oxygen peak

    # Off-nuclear grid points match eval_esp_at_points to float32 precision
    probe_pts = np.array(
        [
            [origin[0] + 0 * 0.5, origin[1] + 1 * 0.5, origin[2] + 4 * 0.5],  # lone-pair +Z side
            [origin[0] + 4 * 0.5, origin[1] + 3 * 0.5, origin[2] + 0 * 0.5],
        ],
        dtype=np.float64,
    )
    probe_vals = rust_core.eval_esp_at_points(wf, probe_pts)
    assert abs(float(esp_grid[4, 1, 0]) - float(probe_vals[0])) < 1e-6
    assert abs(float(esp_grid[0, 3, 4]) - float(probe_vals[1])) < 1e-6

    # Exterior van der Waals region: +Z oxygen lone-pair side is negative, -Z H2 side is positive
    ext_vals = rust_core.eval_esp_at_points(
        wf,
        np.array([[0.0, 0.0, 2.8], [0.0, 0.0, -3.2]], dtype=np.float64),
    )
    assert float(ext_vals[0]) < -0.04
    assert float(ext_vals[1]) > 0.04

    # CubeFile.from_esp and pure Rust Gaussian .cube export/parse round-trip
    cube = CubeFile.from_esp(wf, spacing_bohr=0.6, margin_bohr=3.5)
    cube_txt = cube.to_cube_string(title="Water B3LYP ESP")
    _, _, _, _, parsed_grid = rust_core.parse_gaussian_cube_str(cube_txt)
    assert parsed_grid.shape == cube.data.shape
    assert np.max(np.abs(parsed_grid - cube.data)) < 5e-5


def test_esp_mapped_surface_ccesp_zero_copy_and_opv_polarity():
    rust_core = pytest.importorskip("calchem._core")

    w = _water_origin_spec(0, 1)
    w_bohr = [[c * ANGSTROM_TO_BOHR for c in row] for row in w.coordinates_angstrom]
    wf = rust_core.run_single_point(w.symbols, w_bohr, 0, 1, "b3lyp", "6-31g(d)")

    surf = IsosurfaceStreamService.esp_surface(
        wf, density_isovalue=0.001, spacing_bohr=0.28, margin_bohr=4.8
    )
    assert surf.units_angstrom is True
    assert surf.mesh.n_vertices > 200
    assert surf.mesh.n_triangles > 400
    assert surf.esp.shape == (surf.mesh.n_vertices,)
    assert surf.esp.dtype == np.float32
    assert -0.10 < surf.esp_min < -0.03
    assert 0.04 < surf.esp_max < 0.12

    packet = surf.to_binary_packet()
    assert packet[:8] == CCESP_MAGIC
    assert len(packet) == CCESP_HEADER_BYTES + (
        surf.mesh.n_vertices * 7 + surf.mesh.n_triangles * 3
    ) * 4

    decoded = decode_ccesp_packet(packet)
    assert decoded.units_angstrom is True
    assert abs(decoded.density_isovalue - 0.001) < 1e-6
    assert np.allclose(decoded.vertices, surf.mesh.vertices, atol=0.0)
    assert np.allclose(decoded.normals, surf.mesh.normals, atol=0.0)
    assert np.allclose(decoded.esp, surf.esp, atol=0.0)
    assert np.array_equal(decoded.faces, surf.mesh.faces)

    # Verify OPVWorkflow.esp_surface_summary on Donor (vinylamine) vs Acceptor (acrylonitrile)
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
    d_esp = wf_engine.esp_surface_summary(
        donor, method="b3lyp", basis="6-31g(d)", spacing_bohr=0.30, name="Vinylamine"
    )
    a_esp = wf_engine.esp_surface_summary(
        acceptor, method="b3lyp", basis="6-31g(d)", spacing_bohr=0.30, name="Acrylonitrile"
    )
    assert d_esp.surface_area_angstrom2 > 60.0
    assert a_esp.surface_area_angstrom2 > 60.0
    # Electron-withdrawing cyano group yields a strongly negative nitrile lone-pair cap AND
    # shifts the pi/C-H positive surface potential higher than the electron-rich donor:
    assert a_esp.esp_min_kcal_mol < -25.0
    assert a_esp.mpi_kcal_mol > 6.0
    assert d_esp.mpi_kcal_mol > 5.0
    assert "CCESP\\x01" in a_esp.summary
