"""Gaussian `.cube` volumetric file wrapper backed by CalChem's Rust core (`calchem._core`).

Grid arrays (`self.data`) are stored in contiguous `float32` `(nz, ny, nx)` order matching
VTK `vtkImageData` memory layout, while `.cube` text serialization and parsing translate
between `(nz, ny, nx)` memory order and Gaussian's `(nx, ny, nz)` file convention natively in Rust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from calchem import _core
from ..mol.molecule import Molecule

ANGSTROM_TO_BOHR: float = 1.8897261245650618


class CubeFile:
    """Volumetric grid and Gaussian `.cube` serializer/loader."""

    def __init__(
        self,
        mol: Molecule,
        origin: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        spacing: Tuple[float, float, float] = (0.25, 0.25, 0.25),
        n_points: Tuple[int, int, int] = (60, 60, 60),
    ):
        self.mol = mol
        self.spacing = np.asarray(spacing, dtype=np.float64)
        self.n_points = np.asarray(n_points, dtype=int)  # [nx, ny, nz]

        coords = np.asarray(mol.coordinates, dtype=np.float64)
        if mol.units == "angstrom":
            self._coords_bohr = coords * ANGSTROM_TO_BOHR
        else:
            self._coords_bohr = coords.copy()

        if origin is None:
            center = 0.5 * (self._coords_bohr.min(axis=0) + self._coords_bohr.max(axis=0))
            span = (self.n_points - 1).astype(np.float64) * self.spacing
            self.origin = center - 0.5 * span
        else:
            self.origin = np.asarray(origin, dtype=np.float64)

        self.data: Optional[NDArray[np.float32]] = None
        self.basis_name: str = ""

    def generate(
        self,
        wf_or_basis: Union[_core.PyScfWavefunction, str] = "sto-3g",
        mo_index: Optional[int] = 0,
        method: str = "auto",
        spin_density: bool = False,
    ) -> NDArray[np.float32]:
        """Evaluates an orbital or electron/spin density grid in Rust on `(nz, ny, nx)`."""
        if isinstance(wf_or_basis, _core.PyScfWavefunction):
            wf = wf_or_basis
        else:
            basis = wf_or_basis if isinstance(wf_or_basis, str) else getattr(wf_or_basis, "basis", "sto-3g")
            wf = _core.run_single_point(
                list(self.mol.symbols),
                self._coords_bohr.tolist(),
                charge=self.mol.charge,
                multiplicity=self.mol.multiplicity,
                method=method,
                basis=str(basis),
            )

        self.basis_name = wf.basis_name
        origin_t = (float(self.origin[0]), float(self.origin[1]), float(self.origin[2]))
        spacing_t = (float(self.spacing[0]), float(self.spacing[1]), float(self.spacing[2]))
        counts_t = (int(self.n_points[0]), int(self.n_points[1]), int(self.n_points[2]))

        if spin_density or mo_index is None:
            self.data = _core.eval_density_cube_zero_copy(
                wf, origin_t, spacing_t, counts_t, spin_density=spin_density
            )
        else:
            self.data = _core.eval_orbital_cube_zero_copy(
                wf, int(mo_index), origin_t, spacing_t, counts_t
            )
        return self.data

    @classmethod
    def from_wavefunction(
        cls,
        wf: _core.PyScfWavefunction,
        mo_index: Optional[int] = 0,
        spin_density: bool = False,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 4.5,
    ) -> "CubeFile":
        """Constructs an automatically sized `CubeFile` directly from a Rust `PyScfWavefunction`."""
        mol = Molecule(
            list(wf.symbols),
            [list(p) for p in wf.geometry_bohr],
            charge=wf.charge,
            multiplicity=wf.multiplicity,
            units="bohr",
        )
        origin, spacing, counts = _core.auto_orbital_grid_spec(
            wf, margin_bohr=margin_bohr, spacing_bohr=spacing_bohr
        )
        cube = cls(mol, origin=origin, spacing=spacing, n_points=counts)
        cube.generate(wf, mo_index=mo_index, spin_density=spin_density)
        return cube

    def extract_isosurface(
        self,
        isovalue: float = 0.03,
        output_angstrom: bool = True,
    ) -> _core.PyDualOrbitalMesh:
        """Extracts dual-lobe Marching Cubes triangle meshes in Rust from `self.data`."""
        if self.data is None:
            raise RuntimeError("No volumetric data available. Call generate() or load() first.")
        grid_f32 = np.ascontiguousarray(self.data, dtype=np.float32)
        return _core.extract_isosurface_from_grid(
            grid_f32,
            (float(self.origin[0]), float(self.origin[1]), float(self.origin[2])),
            (float(self.spacing[0]), float(self.spacing[1]), float(self.spacing[2])),
            isovalue=isovalue,
            output_angstrom=output_angstrom,
        )

    def to_cube_string(self, title: str = "") -> str:
        """Formats the volumetric grid into a Gaussian `.cube` string in pure Rust."""
        if self.data is None:
            raise RuntimeError("No data to serialize. Call generate() first.")
        grid_f32 = np.ascontiguousarray(self.data, dtype=np.float32)
        expected_zyx = (int(self.n_points[2]), int(self.n_points[1]), int(self.n_points[0]))
        if grid_f32.shape != expected_zyx:
            raise ValueError(
                f"CubeFile.data shape {grid_f32.shape} does not match (nz, ny, nx) = {expected_zyx}"
            )
        return _core.format_grid_gaussian_cube(
            list(self.mol.symbols),
            self._coords_bohr.tolist(),
            grid_f32,
            (float(self.origin[0]), float(self.origin[1]), float(self.origin[2])),
            (float(self.spacing[0]), float(self.spacing[1]), float(self.spacing[2])),
            title=title,
            basis=self.basis_name,
        )

    def save(self, filename: Union[str, Path], title: str = "") -> None:
        """Writes a standard Bohr Gaussian `.cube` file to `filename`."""
        txt = self.to_cube_string(title=title)
        Path(filename).write_text(txt, encoding="utf-8")

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "CubeFile":
        """Loads a Gaussian `.cube` file using the pure Rust parser."""
        raw = Path(filename).read_text(encoding="utf-8")
        symbols, geom_bohr, origin_bohr, spacing_bohr, grid_zyx = _core.parse_gaussian_cube_str(raw)
        mol = Molecule(symbols, [list(p) for p in geom_bohr], units="bohr")
        nz, ny, nx = grid_zyx.shape
        cube = cls(
            mol,
            origin=origin_bohr,
            spacing=spacing_bohr,
            n_points=(nx, ny, nz),
        )
        cube.data = grid_zyx
        return cube

    def __repr__(self) -> str:
        return f"CubeFile(mol={self.mol}, n_points={tuple(int(x) for x in self.n_points)})"
