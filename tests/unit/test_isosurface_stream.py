from __future__ import annotations

import numpy as np
import pytest

from calchem import _core
from calchem.mol.molecule import Molecule
from calchem.props import (
    CCMSH_HEADER_BYTES,
    CCMSH_MAGIC,
    CubeFile,
    IsosurfaceStreamService,
    decode_ccmsh_packet,
)


def _assert_watertight_manifold(vertices: np.ndarray, faces: np.ndarray) -> float:
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert vertices.dtype == np.float32
    assert faces.dtype == np.uint32
    assert len(vertices) >= 4 and len(faces) >= 4

    half_edges = set()
    for i0, i1, i2 in faces.tolist():
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            assert u != v
            assert (u, v) not in half_edges, f"duplicate half-edge ({u}, {v})"
            half_edges.add((u, v))

    for u, v in half_edges:
        assert (v, u) in half_edges, f"boundary crack on half-edge ({u}, {v})"

    v0 = vertices[faces[:, 0]].astype(np.float64)
    v1 = vertices[faces[:, 1]].astype(np.float64)
    v2 = vertices[faces[:, 2]].astype(np.float64)
    signed_vol = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)
    assert signed_vol > 0.0, f"expected positive outward CCW volume, got {signed_vol}"
    return signed_vol


def test_homo_lumo_zero_copy_buffers_and_ccmsh_binary_packet():
    mol = Molecule(
        ["O", "H", "H"],
        [
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
        units="angstrom",
    )
    wf = IsosurfaceStreamService.run_wavefunction(mol, basis="6-31g(d)")
    assert wf.converged

    homo_mesh = IsosurfaceStreamService.homo_mesh(wf, isovalue=0.04, spacing_bohr=0.25)
    assert homo_mesh.units_angstrom is True
    assert homo_mesh.mo_index == 4
    assert homo_mesh.spin_channel == 0

    # Zero-copy pointer stability across repeated property reads:
    ptr_a = homo_mesh.positive.vertices.__array_interface__["data"][0]
    ptr_b = homo_mesh.positive.vertices.__array_interface__["data"][0]
    assert ptr_a == ptr_b

    pos_v = homo_mesh.positive.vertices
    pos_n = homo_mesh.positive.normals
    pos_f = homo_mesh.positive.faces
    assert pos_v.flags["C_CONTIGUOUS"]
    assert pos_n.flags["C_CONTIGUOUS"]
    assert pos_f.flags["C_CONTIGUOUS"]

    norms = np.linalg.norm(pos_n.astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    vol_pos = _assert_watertight_manifold(pos_v, pos_f)
    vol_neg = _assert_watertight_manifold(homo_mesh.negative.vertices, homo_mesh.negative.faces)
    assert abs(vol_pos - vol_neg) / vol_pos < 1e-4

    # CCMSH\x01 binary streaming wire round-trip with zero-copy memoryview decoding:
    pkt = homo_mesh.to_binary_packet()
    assert isinstance(pkt, bytes)
    assert pkt[:8] == CCMSH_MAGIC
    assert len(pkt) >= CCMSH_HEADER_BYTES and len(pkt) % 4 == 0

    decoded = decode_ccmsh_packet(memoryview(pkt))
    assert decoded.units_angstrom is True
    assert decoded.mo_index == 4
    assert decoded.spin_channel == 0
    assert np.isclose(decoded.isovalue, 0.04)
    assert decoded.positive.vertices.base is not None  # zero-copy view into packet buffer
    assert np.array_equal(decoded.positive.vertices, pos_v)
    assert np.array_equal(decoded.positive.normals, pos_n)
    assert np.array_equal(decoded.positive.faces, pos_f)
    assert np.array_equal(decoded.negative.vertices, homo_mesh.negative.vertices)
    assert np.array_equal(decoded.negative.faces, homo_mesh.negative.faces)

    # LUMO check:
    lumo_mesh = IsosurfaceStreamService.lumo_mesh(wf, isovalue=0.04, spacing_bohr=0.25)
    assert lumo_mesh.mo_index == 5
    _assert_watertight_manifold(lumo_mesh.positive.vertices, lumo_mesh.positive.faces)
    _assert_watertight_manifold(lumo_mesh.negative.vertices, lumo_mesh.negative.faces)


def test_cached_volumetric_grid_monotonic_slider_reextraction():
    mol = Molecule(
        ["C", "C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.6695],
            [0.0, 0.0, -0.6695],
            [0.0, 0.9289, 1.2321],
            [0.0, -0.9289, 1.2321],
            [0.0, 0.9289, -1.2321],
            [0.0, -0.9289, -1.2321],
        ],
        units="angstrom",
    )
    wf = IsosurfaceStreamService.run_wavefunction(mol, basis="6-31g(d)")
    homo_idx = IsosurfaceStreamService.homo_index(wf)
    cache = IsosurfaceStreamService.cache_orbital_grid(wf, homo_idx, spacing_bohr=0.28)
    assert cache.grid_zyx.dtype == np.float32
    nz, ny, nx = cache.grid_zyx.shape
    assert (nx, ny, nz) == cache.counts_xyz

    m_low = cache.extract_isosurface(isovalue=0.025)
    m_high = cache.extract_isosurface(isovalue=0.070)
    vol_low = _assert_watertight_manifold(m_low.positive.vertices, m_low.positive.faces)
    vol_high = _assert_watertight_manifold(m_high.positive.vertices, m_high.positive.faces)
    assert vol_low > vol_high > 0.0


def test_uhf_radical_cation_spin_density_stream():
    mol_cat = Molecule(
        ["O", "H", "H"],
        [
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
        charge=1,
        multiplicity=2,
        units="angstrom",
    )
    wf_cat = IsosurfaceStreamService.run_wavefunction(mol_cat, basis="6-31g(d)", method="uhf")
    assert wf_cat.is_uhf
    assert wf_cat.spin_density is not None

    cache_spin = IsosurfaceStreamService.cache_density_grid(
        wf_cat, spacing_bohr=0.22, margin_bohr=4.5, spin_density=True
    )
    dv = (
        cache_spin.spacing_bohr[0]
        * cache_spin.spacing_bohr[1]
        * cache_spin.spacing_bohr[2]
    )
    total_unpaired = float(np.sum(cache_spin.grid_zyx, dtype=np.float64) * dv)
    assert abs(total_unpaired - 1.0) < 0.005

    spin_mesh = IsosurfaceStreamService.density_mesh(
        wf_cat, isovalue=0.01, spacing_bohr=0.24, spin_density=True
    )
    assert spin_mesh.spin_channel == 2
    _assert_watertight_manifold(spin_mesh.positive.vertices, spin_mesh.positive.faces)

    # Beta HOMO on UHF wavefunction uses spin_channel == 1:
    beta_homo = IsosurfaceStreamService.homo_mesh(
        wf_cat, isovalue=0.04, spacing_bohr=0.25, spin="beta"
    )
    assert beta_homo.spin_channel == 1
    assert beta_homo.mo_index >= wf_cat.n_ao


def test_cubefile_rust_save_load_and_iodine_ecp(tmp_path):
    hi = Molecule(
        ["I", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.609]],
        units="angstrom",
    )
    wf_hi = IsosurfaceStreamService.run_wavefunction(hi, basis="lanl2dzdp")
    assert wf_hi.converged

    homo_idx = IsosurfaceStreamService.homo_index(wf_hi)
    cube = CubeFile.from_wavefunction(
        wf_hi, mo_index=homo_idx, spacing_bohr=0.28, margin_bohr=4.5
    )
    assert cube.data is not None
    assert cube.data.dtype == np.float32

    cube_path = tmp_path / "hi_homo.cube"
    cube.save(cube_path, title="HI LANL2DZdp HOMO")
    raw_lines = cube_path.read_text(encoding="utf-8").splitlines()
    # Line 6 is the first atom (Iodine: Z = 53, Z_eff = 7.000000 with 46-electron core ECP)
    iodine_cols = raw_lines[6].split()
    assert iodine_cols[0] == "53"
    assert abs(float(iodine_cols[1]) - 7.0) < 1e-6

    reloaded = CubeFile.load(cube_path)
    assert reloaded.mol.symbols == ["I", "H"]
    assert reloaded.data is not None
    assert reloaded.data.shape == cube.data.shape
    assert np.allclose(reloaded.data, cube.data, atol=2e-5)

    mesh = reloaded.extract_isosurface(isovalue=0.03, output_angstrom=True)
    _assert_watertight_manifold(mesh.positive.vertices, mesh.positive.faces)
    _assert_watertight_manifold(mesh.negative.vertices, mesh.negative.faces)


def test_invalid_ccmsh_packet_rejected():
    with pytest.raises(ValueError, match="too short"):
        decode_ccmsh_packet(b"short")
    with pytest.raises(ValueError, match="magic"):
        decode_ccmsh_packet(b"BADMAGIC" + b"\x00" * 56)
