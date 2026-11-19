"""Headless 3D Molecular Orbital & Density Isosurface Streaming Protocol.

All compute-intensive grid evaluations, Consistent Marching Cubes surface extractions,
normal interpolations, and binary packet encodings execute natively in Rust (`calchem._core`).
This module provides zero-copy Python/NumPy buffer views and thin headless orchestration
for VTK (`vtkImageData` / `vtkPolyData`) and MolForge IPC/WebSocket streaming.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from calchem import _core
from calchem.mol.molecule import Molecule

ANGSTROM_TO_BOHR: float = 1.8897261245650618
BOHR_TO_ANGSTROM: float = 1.0 / ANGSTROM_TO_BOHR

CCMSH_MAGIC: bytes = b"CCMSH\x01\x00\x00"
CCMSH_HEADER_BYTES: int = 64
_CCMSH_HEADER_STRUCT = struct.Struct("<8sIiIfIIII24x")


@dataclass(frozen=True)
class DecodedLobeView:
    """Zero-copy NumPy buffer view into a single lobe of a `CCMSH\\x01` binary packet."""

    isovalue: float
    vertices: NDArray[np.float32]  # shape (V, 3)
    normals: NDArray[np.float32]   # shape (V, 3)
    faces: NDArray[np.uint32]      # shape (F, 3)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True)
class DecodedDualMeshPacket:
    """Zero-copy decoded `CCMSH\\x01` dual-phase orbital or spin-density packet."""

    isovalue: float
    units_angstrom: bool
    mo_index: int
    spin_channel: int
    positive: DecodedLobeView
    negative: DecodedLobeView


def decode_ccmsh_packet(
    packet: Union[bytes, bytearray, memoryview],
) -> DecodedDualMeshPacket:
    """Decodes a `CCMSH\\x01` binary wire packet using O(1) zero-copy `np.frombuffer` views."""
    mv = memoryview(packet)
    if mv.nbytes < CCMSH_HEADER_BYTES:
        raise ValueError(
            f"CCMSH packet too short ({mv.nbytes} bytes < {CCMSH_HEADER_BYTES})"
        )

    (
        magic,
        flags,
        mo_index,
        spin_channel,
        isovalue,
        pv,
        pt,
        nv,
        nt,
    ) = _CCMSH_HEADER_STRUCT.unpack_from(mv, 0)

    if magic != CCMSH_MAGIC:
        raise ValueError(f"Invalid CCMSH magic header: {magic!r}")

    expected_bytes = CCMSH_HEADER_BYTES + (pv * 6 + pt * 3 + nv * 6 + nt * 3) * 4
    if mv.nbytes != expected_bytes:
        raise ValueError(
            f"CCMSH packet byte length mismatch: got {mv.nbytes}, expected {expected_bytes}"
        )

    offset = CCMSH_HEADER_BYTES

    pos_v = np.frombuffer(mv, dtype="<f4", count=pv * 3, offset=offset).reshape(pv, 3)
    offset += pv * 12
    pos_n = np.frombuffer(mv, dtype="<f4", count=pv * 3, offset=offset).reshape(pv, 3)
    offset += pv * 12
    pos_f = np.frombuffer(mv, dtype="<u4", count=pt * 3, offset=offset).reshape(pt, 3)
    offset += pt * 12

    neg_v = np.frombuffer(mv, dtype="<f4", count=nv * 3, offset=offset).reshape(nv, 3)
    offset += nv * 12
    neg_n = np.frombuffer(mv, dtype="<f4", count=nv * 3, offset=offset).reshape(nv, 3)
    offset += nv * 12
    neg_f = np.frombuffer(mv, dtype="<u4", count=nt * 3, offset=offset).reshape(nt, 3)

    return DecodedDualMeshPacket(
        isovalue=float(isovalue),
        units_angstrom=bool(flags & 1),
        mo_index=int(mo_index),
        spin_channel=int(spin_channel),
        positive=DecodedLobeView(
            isovalue=float(isovalue),
            vertices=pos_v,
            normals=pos_n,
            faces=pos_f,
        ),
        negative=DecodedLobeView(
            isovalue=-float(isovalue),
            vertices=neg_v,
            normals=neg_n,
            faces=neg_f,
        ),
    )


CCESP_MAGIC: bytes = b"CCESP\x01\x00\x00"
CCESP_HEADER_BYTES: int = 64
_CCESP_HEADER_STRUCT = struct.Struct("<8sIfIIff32x")


@dataclass(frozen=True)
class DecodedEspSurfacePacket:
    """Zero-copy decoded `CCESP\\x01` ESP-mapped isodensity surface (ESP in Hartree / e)."""

    density_isovalue: float
    units_angstrom: bool
    esp_min: float
    esp_max: float
    vertices: NDArray[np.float32]  # shape (V, 3)
    normals: NDArray[np.float32]   # shape (V, 3)
    esp: NDArray[np.float32]       # shape (V,)
    faces: NDArray[np.uint32]      # shape (F, 3)


def decode_ccesp_packet(
    packet: Union[bytes, bytearray, memoryview],
) -> DecodedEspSurfacePacket:
    """Decodes a `CCESP\\x01` binary wire packet using O(1) zero-copy `np.frombuffer` views."""
    mv = memoryview(packet)
    if mv.nbytes < CCESP_HEADER_BYTES:
        raise ValueError(
            f"CCESP packet too short ({mv.nbytes} bytes < {CCESP_HEADER_BYTES})"
        )
    magic, flags, iso, nv, nt, esp_min, esp_max = _CCESP_HEADER_STRUCT.unpack_from(mv, 0)
    if magic != CCESP_MAGIC:
        raise ValueError(f"Invalid CCESP magic header: {magic!r}")
    expected = CCESP_HEADER_BYTES + (nv * 7 + nt * 3) * 4
    if mv.nbytes != expected:
        raise ValueError(
            f"CCESP packet byte length mismatch: got {mv.nbytes}, expected {expected}"
        )
    off = CCESP_HEADER_BYTES
    verts = np.frombuffer(mv, dtype="<f4", count=nv * 3, offset=off).reshape(nv, 3)
    off += nv * 12
    norms = np.frombuffer(mv, dtype="<f4", count=nv * 3, offset=off).reshape(nv, 3)
    off += nv * 12
    esp = np.frombuffer(mv, dtype="<f4", count=nv, offset=off)
    off += nv * 4
    faces = np.frombuffer(mv, dtype="<u4", count=nt * 3, offset=off).reshape(nt, 3)
    return DecodedEspSurfacePacket(
        density_isovalue=float(iso),
        units_angstrom=bool(flags & 1),
        esp_min=float(esp_min),
        esp_max=float(esp_max),
        vertices=verts,
        normals=norms,
        esp=esp,
        faces=faces,
    )


@dataclass
class OrbitalVolumetricCache:
    """Cached `(nz, ny, nx)` `float32` volumetric grid for millisecond interactive isosurface updates."""

    wf: _core.PyScfWavefunction
    grid_zyx: NDArray[np.float32]
    origin_bohr: Tuple[float, float, float]
    spacing_bohr: Tuple[float, float, float]
    counts_xyz: Tuple[int, int, int]
    mo_index: int = -1
    spin_channel: int = 0

    def extract_isosurface(
        self,
        isovalue: float = 0.03,
        output_angstrom: bool = True,
    ) -> _core.PyDualOrbitalMesh:
        """Extracts dual-phase Marching Cubes meshes in Rust from the cached 3D grid."""
        return _core.extract_isosurface_from_grid(
            self.grid_zyx,
            self.origin_bohr,
            self.spacing_bohr,
            isovalue=isovalue,
            output_angstrom=output_angstrom,
            mo_index=self.mo_index,
            spin_channel=self.spin_channel,
        )

    def extract_single_isosurface(
        self,
        isovalue: float = 0.002,
        output_angstrom: bool = True,
    ) -> _core.PyIsosurfaceMesh:
        """Extracts a single-lobe Marching Cubes mesh in Rust from the cached 3D grid."""
        return _core.extract_single_isosurface_from_grid(
            self.grid_zyx,
            self.origin_bohr,
            self.spacing_bohr,
            isovalue=isovalue,
            output_angstrom=output_angstrom,
        )

    def to_binary_packet(
        self,
        isovalue: float = 0.03,
        output_angstrom: bool = True,
    ) -> bytes:
        """Extracts dual-phase meshes and encodes them into a `CCMSH\\x01` binary packet."""
        return self.extract_isosurface(
            isovalue=isovalue,
            output_angstrom=output_angstrom,
        ).to_binary_packet()

    def to_gaussian_cube(self, title: str = "") -> str:
        """Formats the cached volumetric grid into standard Gaussian `.cube` text in Rust."""
        return _core.format_grid_gaussian_cube(
            self.wf.symbols,
            self.wf.geometry_bohr,
            self.grid_zyx,
            self.origin_bohr,
            self.spacing_bohr,
            title=title,
            basis=self.wf.basis_name,
        )


class IsosurfaceStreamService:
    """Headless service bridging CalChem SCF wavefunctions to VTK / MolForge 3D orbital meshes."""

    @staticmethod
    def run_wavefunction(
        mol: Molecule,
        basis: str = "6-31g(d)",
        method: str = "auto",
    ) -> _core.PyScfWavefunction:
        """Runs a native Rust RHF/UHF single-point calculation from a `calchem.mol.Molecule`."""
        coords = np.asarray(mol.coordinates, dtype=np.float64)
        if mol.units == "angstrom":
            coords_bohr = (coords * ANGSTROM_TO_BOHR).tolist()
        else:
            coords_bohr = coords.tolist()
        return _core.run_single_point(
            list(mol.symbols),
            coords_bohr,
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            method=method,
            basis=basis,
        )

    @staticmethod
    def homo_index(
        wf: _core.PyScfWavefunction,
        spin: Literal["alpha", "beta"] = "alpha",
    ) -> int:
        """Returns the column index in `wf.mo_coefficients` for the HOMO (or alpha/beta SOMO)."""
        if wf.is_uhf and spin == "beta":
            if wf.n_beta == 0:
                raise ValueError("UHF wavefunction has zero beta electrons")
            return wf.n_ao + wf.n_beta - 1
        if wf.n_alpha == 0:
            raise ValueError("Wavefunction has zero occupied alpha orbitals")
        return wf.n_alpha - 1

    @staticmethod
    def lumo_index(
        wf: _core.PyScfWavefunction,
        spin: Literal["alpha", "beta"] = "alpha",
    ) -> int:
        """Returns the column index in `wf.mo_coefficients` for the LUMO."""
        if wf.is_uhf and spin == "beta":
            if wf.n_beta >= wf.n_ao:
                raise ValueError("No virtual beta orbitals in minimal basis")
            return wf.n_ao + wf.n_beta
        if wf.n_alpha >= wf.n_ao:
            raise ValueError("No virtual alpha orbitals in minimal basis")
        return wf.n_alpha

    @classmethod
    def cache_orbital_grid(
        cls,
        wf: _core.PyScfWavefunction,
        mo_idx: int,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 4.5,
    ) -> OrbitalVolumetricCache:
        """Evaluates and caches an orbital `(nz, ny, nx)` `float32` grid in Rust."""
        origin, spacing, counts = _core.auto_orbital_grid_spec(
            wf, margin_bohr=margin_bohr, spacing_bohr=spacing_bohr
        )
        grid = _core.eval_orbital_cube_zero_copy(wf, mo_idx, origin, spacing, counts)
        spin_ch = 1 if (wf.is_uhf and mo_idx >= wf.n_ao) else 0
        return OrbitalVolumetricCache(
            wf=wf,
            grid_zyx=grid,
            origin_bohr=(float(origin[0]), float(origin[1]), float(origin[2])),
            spacing_bohr=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            counts_xyz=(int(counts[0]), int(counts[1]), int(counts[2])),
            mo_index=mo_idx,
            spin_channel=spin_ch,
        )

    @classmethod
    def cache_density_grid(
        cls,
        wf: _core.PyScfWavefunction,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 4.5,
        spin_density: bool = False,
    ) -> OrbitalVolumetricCache:
        """Evaluates and caches a total or spin density `(nz, ny, nx)` `float32` grid in Rust."""
        origin, spacing, counts = _core.auto_orbital_grid_spec(
            wf, margin_bohr=margin_bohr, spacing_bohr=spacing_bohr
        )
        grid = _core.eval_density_cube_zero_copy(
            wf, origin, spacing, counts, spin_density=spin_density
        )
        return OrbitalVolumetricCache(
            wf=wf,
            grid_zyx=grid,
            origin_bohr=(float(origin[0]), float(origin[1]), float(origin[2])),
            spacing_bohr=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            counts_xyz=(int(counts[0]), int(counts[1]), int(counts[2])),
            mo_index=-1,
            spin_channel=2 if spin_density else 0,
        )

    @classmethod
    def orbital_mesh(
        cls,
        wf: _core.PyScfWavefunction,
        mo_idx: int,
        isovalue: float = 0.03,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 4.5,
        output_angstrom: bool = True,
    ) -> _core.PyDualOrbitalMesh:
        """Computes a dual-phase orbital isosurface mesh end-to-end in Rust."""
        return _core.extract_orbital_isosurface(
            wf,
            mo_idx,
            isovalue=isovalue,
            spacing_bohr=spacing_bohr,
            margin_bohr=margin_bohr,
            output_angstrom=output_angstrom,
        )

    @classmethod
    def homo_mesh(
        cls,
        wf: _core.PyScfWavefunction,
        isovalue: float = 0.03,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 4.5,
        spin: Literal["alpha", "beta"] = "alpha",
        output_angstrom: bool = True,
    ) -> _core.PyDualOrbitalMesh:
        """Computes the HOMO (or alpha/beta SOMO) dual-phase isosurface mesh in Rust."""
        idx = cls.homo_index(wf, spin=spin)
        return cls.orbital_mesh(
            wf,
            idx,
            isovalue=isovalue,
            spacing_bohr=spacing_bohr,
            margin_bohr=margin_bohr,
            output_angstrom=output_angstrom,
        )

    @classmethod
    def lumo_mesh(
        cls,
        wf: _core.PyScfWavefunction,
        isovalue: float = 0.03,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 4.5,
        spin: Literal["alpha", "beta"] = "alpha",
        output_angstrom: bool = True,
    ) -> _core.PyDualOrbitalMesh:
        """Computes the LUMO dual-phase isosurface mesh in Rust."""
        idx = cls.lumo_index(wf, spin=spin)
        return cls.orbital_mesh(
            wf,
            idx,
            isovalue=isovalue,
            spacing_bohr=spacing_bohr,
            margin_bohr=margin_bohr,
            output_angstrom=output_angstrom,
        )

    @staticmethod
    def density_mesh(
        wf: _core.PyScfWavefunction,
        isovalue: float = 0.002,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 4.5,
        spin_density: bool = False,
        output_angstrom: bool = True,
    ) -> _core.PyDualOrbitalMesh:
        """Computes a total electron density or UHF spin density isosurface mesh in Rust."""
        return _core.extract_density_isosurface(
            wf,
            isovalue=isovalue,
            spacing_bohr=spacing_bohr,
            margin_bohr=margin_bohr,
            spin_density=spin_density,
            output_angstrom=output_angstrom,
        )

    @staticmethod
    def esp_surface(
        wf: _core.PyScfWavefunction,
        density_isovalue: float = 0.001,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 5.0,
        output_angstrom: bool = True,
    ) -> _core.PyEspSurface:
        """Computes the ESP-mapped `rho = density_isovalue` isodensity surface in Rust."""
        return _core.extract_esp_surface(
            wf,
            density_isovalue=density_isovalue,
            spacing_bohr=spacing_bohr,
            margin_bohr=margin_bohr,
            output_angstrom=output_angstrom,
        )

    @staticmethod
    def atomic_charges(wf: _core.PyScfWavefunction) -> dict:
        """Mulliken / Löwdin charges (and spin populations) computed natively in Rust."""
        return _core.compute_population_analysis(wf)
