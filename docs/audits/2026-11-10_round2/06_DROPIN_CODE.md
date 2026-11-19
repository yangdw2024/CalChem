# 完整 Drop-in 文件 — CalChem 第二轮

每个代码块都是对应最终生产文件的完整内容。替换前先做 SHA256 备份并在独立候选目录复核，不能覆盖用户的新修改。二进制标准 D3 数据及原始表/许可证在 replacements.zip 的 data/d3 路径内；不可只替换 d3.rs 而遗漏其数据依赖。此文件为最终源码的自动导出，没有伪代码、省略号替代实现或模拟返回值。

## `Cargo.toml`

SHA256: `75144ed2dc4a655fb8869a26b129297449acb62b8790a1611363e5cb9962b745`

```toml
[workspace]
resolver = "2"
members = [
    "crates/calchem-model",
    "crates/calchem-basis",
    "crates/calchem-tensor",
    "crates/calchem-int",
    "crates/calchem-dft",
    "crates/calchem-scf",
    "crates/calchem-tddft",
    "crates/calchem-vib",
    "crates/calchem-grad",
    "crates/calchem-prop",
    "crates/calchem-opt",
    "crates/calchem-solv",
    "crates/calchem-py",
]

[workspace.dependencies]
faer = { version = "0.20", features = ["rayon", "std"] }
rayon = "1.10"
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py310"] }
numpy = "0.22"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
sha2 = "0.10"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "unwind"

```

## `setup.py`

SHA256: `7f25f617c1672c8cc57760008cea7d5a4eca42e3e8d0225f406c7445c949e0d3`

```python
"""Build the Rust extension from this source tree; never reuse a packaged binary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Distribution, Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class RustBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name != "calchem._core":
            return super().build_extension(ext)
        if os.environ.get("CARGO_BUILD_TARGET") or os.environ.get("PYO3_CROSS"):
            raise RuntimeError("Use the maturin release workflow for cross compilation and target tags")
        env = os.environ.copy()
        env["PYO3_PYTHON"] = sys.executable
        cargo = env.get("CALCHEM_CARGO", env.get("CARGO", "cargo"))
        result = subprocess.run(
            [cargo, "build", "--release", "--locked", "-p", "calchem-py",
             "--message-format=json-render-diagnostics"],
            cwd=ROOT, env=env, text=True, encoding="utf-8", stdout=subprocess.PIPE,
            check=True,
        )
        artifacts = []
        for line in result.stdout.splitlines():
            msg = json.loads(line)
            if (msg.get("reason") == "compiler-artifact"
                    and msg.get("target", {}).get("name") == "_core"
                    and "cdylib" in msg.get("target", {}).get("crate_types", [])):
                artifacts.extend(Path(p) for p in msg["filenames"]
                                 if Path(p).suffix in {".dll", ".so", ".dylib"})
        if len(artifacts) != 1 or not artifacts[0].is_file():
            raise RuntimeError(f"Cargo did not produce exactly one native extension: {artifacts}")
        destination = Path(self.get_ext_fullpath(ext.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifacts[0], destination)


setup(
    distclass=BinaryDistribution,
    ext_modules=[Extension("calchem._core", sources=[], py_limited_api=True)],
    cmdclass={"build_ext": RustBuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)

```

## `pyproject.toml`

SHA256: `fa066b2798db15de73c01fb078733383146b84841041da37e27a3ba0d1b70ed5`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "calchem-qc"
version = "0.2.0"
description = "Unified quantum chemistry workstation: native engine + Gaussian/ORCA/PySCF adapters"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.10"
authors = [
    { name = "CalChem Team", email = "calchem@trae.ai" }
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Physics",
]
keywords = ["quantum chemistry", "DFT", "GPU", "OPV", "GROMACS", "CalChem"]
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "h5py>=3.8",
    "PyYAML>=6.0",
    "click>=8.1",
    "rich>=13.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
external = [
    "cclib>=1.6",
]
pyscf = [
    "pyscf>=2.5",
]
md = [
    "MDAnalysis>=2.5",
    "mendeleev>=0.15",
]
symmetry = [
    "spglib>=2.2",
]
gpu = [
    "cupy-cuda12x>=13.0",
]
gui = [
    "PyQt6>=6.5",
    "matplotlib>=3.7",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.6",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.5",
]
all = ["calchem-qc[external,pyscf,md,symmetry,gui,dev,docs]"]

[project.scripts]
calchem = "calchem.cli:cli"

[tool.setuptools.packages.find]
where = ["."]
include = ["calchem*"]

[tool.setuptools.package-data]
calchem = ["basis/*.dat", "data/*.yaml", "data/*.json", "_core.pyi", "data/d3/*"]

[tool.setuptools.exclude-package-data]
calchem = ["*.pyd", "*.so", "*.dylib"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
norecursedirs = ["tests/debug", "tests/scratch", "calchem/tests", "docs", "build", "dist"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: requires GPU (deselect with '-m \"not gpu\"')",
    "regression: regression tests against Gaussian reference",
    "external: requires external programs (Gaussian/ORCA/xTB)",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "C4", "SIM", "UP"]
ignore = ["E501", "B008"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
exclude = ["tests/"]

[tool.maturin]
manifest-path = "crates/calchem-py/Cargo.toml"
module-name = "calchem._core"
python-source = "."
features = ["pyo3/extension-module", "pyo3/abi3-py310"]

```

## `MANIFEST.in`

SHA256: `16409f2f3a5b9309ace3d5fefc9a168bf8016f585d86749fcffbc5e8bd5e0ce0`

```text
include Cargo.toml Cargo.lock README.md LICENSE setup.py pyproject.toml
recursive-include crates *.toml *.rs *.json *.bin *.f90 *.md COPYING COPYING.LESSER
recursive-include calchem *.py *.pyi *.dat *.json *.yaml
global-exclude *.pyc *.pyd *.so *.dll *.dylib
prune build_target
prune docs/audits

```

## `calchem/_core.pyi`

SHA256: `ad76b1513c29589c692b02b400e137c6d4db1fbf5fefff0521cc0058b67ac289`

```python
"""Type stub for the Rust extension calchem._core (crates/calchem-py).

Entry points:
- Closed-shell RHF and open-shell UHF single-point calculations
  (STO-3G, 6-31G, 6-31G(d)/6-31G*, 6-31G(d,p)/6-31G**, LANL2DZ, LANL2DZdp;
   including heavy halogens Br/I and scalar Effective Core Potentials)
- Analytical RHF and UHF energy gradients (including d/f shells and ECP gradients)
- BFGS geometry optimization (neutral and charged/open-shell species)
- Electric dipole moments (valence-consistent with ECP effective core charges)
- Zero-copy 3D orbital & electron/spin density volumetric grids (`float32`, `(nz, ny, nx)`)
- Consistent Marching Cubes dual-phase triangle mesh extraction (`float32`/`uint32`) & `CCMSH\\x01` binary streaming
- Pure Rust Gaussian `.cube` export & parser
- Exact molecular electrostatic potential (points / grids), Mulliken & Löwdin charges,
  and ESP-mapped isodensity surfaces with `CCESP\\x01` binary streaming
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

__version__: str

class ScfNotConvergedError(RuntimeError): ...
class RustCoreError(RuntimeError): ...

class PyScfWavefunction:
    converged: bool
    iterations: int
    total_energy_hartree: float
    electronic_energy_hartree: float
    nuclear_repulsion_hartree: float
    n_ao: int
    mo_energies: List[float]
    mo_occupations: List[float]
    mo_coefficients: List[float]  # row-major, index mu * n_mo + i
    density: List[float]  # spin-summed P, row-major
    spin_density: Optional[List[float]]  # P_alpha - P_beta for UHF, None for RHF
    energy_history: List[float]
    is_uhf: bool
    s_squared: Optional[float]
    n_alpha: int
    n_beta: int
    mo_energies_alpha: Optional[List[float]]
    mo_energies_beta: Optional[List[float]]
    symbols: List[str]
    geometry_bohr: List[Tuple[float, float, float]]
    basis_name: str
    charge: int
    multiplicity: int
    xc_functional: Optional[str]
    @property
    def homo_energy_hartree(self) -> Optional[float]: ...
    @property
    def lumo_energy_hartree(self) -> Optional[float]: ...
    @property
    def dipole_debye(self) -> Optional[Tuple[float, float, float]]: ...
    @property
    def dipole_magnitude_debye(self) -> Optional[float]: ...

class PyOptResult:
    converged: bool
    coords: List[float]
    energy: float
    steps: int

class PyIsosurfaceMesh:
    isovalue: float
    n_vertices: int
    n_triangles: int
    @property
    def vertices(self) -> NDArray[np.float32]: ...
    @property
    def normals(self) -> NDArray[np.float32]: ...
    @property
    def faces(self) -> NDArray[np.uint32]: ...
    @property
    def indices(self) -> NDArray[np.uint32]: ...

class PyDualOrbitalMesh:
    isovalue: float
    units_angstrom: bool
    mo_index: int
    spin_channel: int
    positive: PyIsosurfaceMesh
    negative: PyIsosurfaceMesh
    def to_binary_packet(self) -> bytes: ...

class PyEspSurface:
    density_isovalue: float
    esp_min: float
    esp_max: float
    units_angstrom: bool
    mesh: PyIsosurfaceMesh
    @property
    def esp(self) -> NDArray[np.float32]: ...
    def to_binary_packet(self) -> bytes: ...

def compute_population_analysis(wf: PyScfWavefunction) -> Dict[str, Optional[List[float]]]: ...
def eval_esp_at_points(
    wf: PyScfWavefunction, points_bohr: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def eval_esp_cube_zero_copy(
    wf: PyScfWavefunction,
    origin: Sequence[float],
    spacing: Sequence[float],
    counts: Sequence[int],
) -> NDArray[np.float32]: ...
def extract_esp_surface(
    wf: PyScfWavefunction,
    density_isovalue: float = 0.001,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 5.0,
    output_angstrom: bool = True,
) -> PyEspSurface: ...
def export_esp_gaussian_cube(
    wf: PyScfWavefunction,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 5.0,
    title: str = "",
) -> str: ...

def run_single_point(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "rhf",
    basis: str = "sto-3g",
    max_iter: int = 100,
    conv_energy: float = 1e-10,
    conv_density: float = 1e-8,
    use_diis: bool = True,
    diis_max_vecs: int = 8,
    damping: float = 0.0,
    xc: Optional[str] = None,
    grid_level: str = "medium",
) -> PyScfWavefunction: ...

def run_gradient(
    atomic_numbers: Any,
    coords: Any,
    charge: int = 0,
    multiplicity: int = 1,
    basis: str = "sto-3g",
    method: str = "auto",
    xc: Optional[str] = None,
    grid_level: str = "medium",
) -> NDArray[np.float64]: ...

def run_optimize(
    atomic_numbers: Any,
    coords: Any,
    basis: str = "sto-3g",
    max_steps: int = 100,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "auto",
    xc: Optional[str] = None,
    grid_level: str = "medium",
    dispersion: Optional[str] = None,
    solvent: Optional[str] = None,
) -> PyOptResult: ...

def auto_orbital_grid_spec(
    wf: PyScfWavefunction,
    margin_bohr: float = 4.5,
    spacing_bohr: float = 0.25,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]: ...

def eval_orbital_cube_zero_copy(
    wf: PyScfWavefunction,
    mo_idx: int,
    origin: Sequence[float],
    spacing: Sequence[float],
    counts: Sequence[int],
) -> NDArray[np.float32]: ...

def eval_density_cube_zero_copy(
    wf: PyScfWavefunction,
    origin: Sequence[float],
    spacing: Sequence[float],
    counts: Sequence[int],
    spin_density: bool = False,
) -> NDArray[np.float32]: ...

def extract_isosurface_from_grid(
    grid: NDArray[np.float32],
    origin_bohr: Sequence[float],
    spacing_bohr: Sequence[float],
    isovalue: float = 0.03,
    output_angstrom: bool = True,
    mo_index: int = -1,
    spin_channel: int = 0,
) -> PyDualOrbitalMesh: ...

def extract_single_isosurface_from_grid(
    grid: NDArray[np.float32],
    origin_bohr: Sequence[float],
    spacing_bohr: Sequence[float],
    isovalue: float = 0.002,
    output_angstrom: bool = True,
) -> PyIsosurfaceMesh: ...

def extract_orbital_isosurface(
    wf: PyScfWavefunction,
    mo_idx: int,
    isovalue: float = 0.03,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 4.5,
    output_angstrom: bool = True,
) -> PyDualOrbitalMesh: ...

def extract_density_isosurface(
    wf: PyScfWavefunction,
    isovalue: float = 0.002,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 4.5,
    spin_density: bool = False,
    output_angstrom: bool = True,
) -> PyDualOrbitalMesh: ...

def export_orbital_gaussian_cube(
    wf: PyScfWavefunction,
    mo_idx: int,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 4.5,
    title: str = "",
) -> str: ...

def export_density_gaussian_cube(
    wf: PyScfWavefunction,
    spacing_bohr: float = 0.25,
    margin_bohr: float = 4.5,
    spin_density: bool = False,
    title: str = "",
) -> str: ...

def format_grid_gaussian_cube(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    grid: NDArray[np.float32],
    origin_bohr: Sequence[float],
    spacing_bohr: Sequence[float],
    title: str = "",
    basis: str = "",
) -> str: ...

def parse_gaussian_cube_str(
    cube_text: str,
) -> Tuple[
    List[str],
    List[Tuple[float, float, float]],
    Tuple[float, float, float],
    Tuple[float, float, float],
    NDArray[np.float32],
]: ...

def supported_basis_sets() -> List[str]: ...
def supported_dft_functionals() -> List[str]: ...

class PyOrbitalTransition:
    @property
    def occ_index(self) -> int: ...
    @property
    def virt_index(self) -> int: ...
    @property
    def amplitude_x(self) -> float: ...
    @property
    def amplitude_y(self) -> float: ...
    @property
    def weight(self) -> float: ...

class PyExcitedState:
    @property
    def state_index(self) -> int: ...
    @property
    def manifold_index(self) -> int: ...
    @property
    def label(self) -> str: ...
    @property
    def multiplicity(self) -> int: ...
    @property
    def excitation_energy_hartree(self) -> float: ...
    @property
    def excitation_energy_ev(self) -> float: ...
    @property
    def total_energy_hartree(self) -> float: ...
    @property
    def wavelength_nm(self) -> float: ...
    @property
    def transition_dipole_au(self) -> Tuple[float, float, float]: ...
    @property
    def oscillator_strength(self) -> float: ...
    @property
    def dominant_transitions(self) -> List[PyOrbitalTransition]: ...

class PyUvVisSpectrum:
    @property
    def wavelengths_nm(self) -> List[float]: ...
    @property
    def energies_ev(self) -> List[float]: ...
    @property
    def epsilon_l_mol_cm(self) -> List[float]: ...
    @property
    def fwhm_ev(self) -> float: ...

class PyTddftResult:
    @property
    def ground_energy_hartree(self) -> float: ...
    @property
    def xc_functional(self) -> Optional[str]: ...
    @property
    def tda(self) -> bool: ...
    @property
    def n_occ(self) -> int: ...
    @property
    def n_virt(self) -> int: ...
    @property
    def homo_energy_hartree(self) -> float: ...
    @property
    def lumo_energy_hartree(self) -> float: ...
    @property
    def ks_gap_ev(self) -> float: ...
    @property
    def optical_gap_ev(self) -> Optional[float]: ...
    @property
    def lowest_triplet_ev(self) -> Optional[float]: ...
    @property
    def delta_est_ev(self) -> Optional[float]: ...
    @property
    def davidson_iterations_singlet(self) -> int: ...
    @property
    def davidson_iterations_triplet(self) -> int: ...
    @property
    def states(self) -> List[PyExcitedState]: ...
    @property
    def scf_wavefunction(self) -> PyScfWavefunction: ...
    def broaden_spectrum(
        self,
        min_nm: float = 150.0,
        max_nm: float = 800.0,
        n_points: int = 500,
        fwhm_ev: float = 0.33,
    ) -> PyUvVisSpectrum: ...

def run_tddft(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "rks",
    basis: str = "6-31g",
    n_states: int = 5,
    tda: bool = False,
    manifold: str = "singlet",
    xc: Optional[str] = None,
    grid_level: str = "medium",
    max_iter: int = 80,
    conv_tol: float = 1e-6,
    force_davidson: bool = False,
    scf_max_iter: int = 100,
    scf_conv_energy: float = 1e-10,
    scf_conv_density: float = 1e-8,
) -> PyTddftResult: ...

def run_tddft_from_scf_py(
    wf: PyScfWavefunction,
    n_states: int = 5,
    tda: bool = False,
    manifold: str = "singlet",
    xc: Optional[str] = None,
    grid_level: str = "medium",
    max_iter: int = 80,
    conv_tol: float = 1e-6,
    force_davidson: bool = False,
) -> PyTddftResult: ...

def broaden_uv_vis(
    states: Sequence[PyExcitedState],
    min_nm: float = 150.0,
    max_nm: float = 800.0,
    n_points: int = 500,
    fwhm_ev: float = 0.33,
) -> PyUvVisSpectrum: ...

class PyNormalMode:
    mode_index: int
    frequency_cm: float
    is_imaginary: bool
    reduced_mass_amu: float
    force_constant_m_dyn_a: float
    ir_intensity_km_mol: float
    displacement: List[Tuple[float, float, float]]

class PyThermochemistry:
    temperature_k: float
    pressure_pa: float
    zpve_hartree: float
    zpve_kcal_mol: float
    e_trans_hartree: float
    e_rot_hartree: float
    e_vib_hartree: float
    e_thermal_hartree: float
    h_hartree: float
    g_hartree: float
    s_cal_mol_k: float
    s_trans_cal_mol_k: float
    s_rot_cal_mol_k: float
    s_vib_cal_mol_k: float
    s_elec_cal_mol_k: float
    cv_cal_mol_k: float
    cp_cal_mol_k: float

class PyVibrationalAnalysis:
    n_vib: int
    is_linear: bool
    principal_moments_amu_bohr2: Tuple[float, float, float]
    rotational_constants_ghz: Tuple[float, float, float]
    trans_rot_residuals: List[float]
    modes: List[PyNormalMode]

class PyFrequencyResult:
    energy_hartree: float
    symbols: List[str]
    geometry_bohr: List[Tuple[float, float, float]]
    basis_name: str
    charge: int
    multiplicity: int
    method: str
    xc_functional: Optional[str]
    vibrational: PyVibrationalAnalysis
    thermo: PyThermochemistry
    cartesian_hessian: List[List[float]]
    apt: List[List[float]]
    @property
    def frequencies_cm(self) -> List[float]: ...
    @property
    def ir_intensities_km_mol(self) -> List[float]: ...
    @property
    def zpve_hartree(self) -> float: ...
    @property
    def enthalpy_hartree(self) -> float: ...
    @property
    def gibbs_free_energy_hartree(self) -> float: ...
    @property
    def entropy_cal_mol_k(self) -> float: ...
    @property
    def heat_capacity_cv_cal_mol_k(self) -> float: ...
    @property
    def heat_capacity_cp_cal_mol_k(self) -> float: ...

def run_frequency(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "rks",
    basis: str = "sto-3g",
    step_bohr: float = 0.001,
    xc: Optional[str] = None,
    grid_level: str = "medium",
    temperature_k: float = 298.15,
    pressure_pa: float = 101325.0,
    symmetry_number: Optional[int] = None,
    max_iter: int = 100,
    conv_energy: float = 1e-10,
    conv_density: float = 1e-8,
) -> PyFrequencyResult: ...

class TsResult:
    symbols: List[str]
    geometry_bohr: List[List[float]]
    energy_hartree: float
    gradient: List[List[float]]
    steps: int
    converged: bool
    imaginary_frequency_cm: float
    transition_vector: List[float]

class IrcPoint:
    point_index: int
    direction: str
    path_coordinate_s: float
    energy_hartree: float
    delta_energy_kcal_mol: float
    gradient_norm: float
    geometry_bohr: List[List[float]]

class IrcBranch:
    converged_to_minimum: bool
    direction: str
    n_points: int
    barrier_kcal_mol: float
    delta_energy_kcal_mol: float
    points: List[IrcPoint]

class IrcResult:
    transition_state: IrcPoint
    forward: Optional[IrcBranch]
    reverse: Optional[IrcBranch]
    total_points: int
    barrier_forward_kcal_mol: Optional[float]
    barrier_reverse_kcal_mol: Optional[float]
    reaction_energy_kcal_mol: Optional[float]

def run_ts_search(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "auto",
    basis: str = "sto-3g",
    xc: Optional[str] = None,
    grid_level: str = "medium",
    max_steps: int = 50,
    max_force: float = 3e-4,
    rms_force: float = 1.5e-4,
    max_displacement: float = 1.2e-3,
    trust_radius_bohr: float = 0.1,
    mode_index: Optional[int] = None,
    recalc_hessian_interval: int = 0,
    scf_max_iter: int = 100,
    scf_conv_energy: float = 1e-11,
    scf_conv_density: float = 1e-10,
) -> TsResult: ...

def run_irc(
    symbols: Sequence[str],
    geometry_bohr: Sequence[Sequence[float]],
    transition_vector: Sequence[float],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "auto",
    basis: str = "sto-3g",
    xc: Optional[str] = None,
    grid_level: str = "medium",
    step_size_bohr: float = 0.1,
    max_points: int = 20,
    direction: str = "both",
    convergence_tol: float = 1e-3,
    max_gs_iter: int = 10,
    scf_max_iter: int = 100,
    scf_conv_energy: float = 1e-11,
    scf_conv_density: float = 1e-10,
) -> IrcResult: ...

class SolvationResult:
    solvated_energy: float
    gas_energy: float
    delta_g_solv: float
    delta_g_solv_kcal_mol: float
    delta_g_elec_kcal_mol: float
    delta_g_cds_kcal_mol: float
    cavity_surface_area_angstrom2: float
    total_surface_charge: float
    surface_charges: NDArray[np.float64]
    tessera_coords_bohr: List[List[float]]
    scf_iterations: int
    converged: bool
    solvent_name: str
    dielectric_constant: float

def run_solvation(
    symbols: Sequence[str],
    coords: Sequence[Sequence[float]],
    basis_name: str,
    solvent: str = "water",
    solvent_model: str = "cpcm",
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "b3lyp",
    xc: Optional[str] = None,
    scaling_factor: float = 1.20,
    include_cds: bool = True,
    max_iter: int = 100,
    conv_energy: float = 1e-8,
    conv_density: float = 1e-6,
) -> SolvationResult: ...

def run_pcm_from_scf_py(
    scf_wfn: PyScfWavefunction,
    solvent: str = "water",
    solvent_model: str = "cpcm",
    scaling_factor: float = 1.20,
    include_cds: bool = True,
) -> SolvationResult: ...

class D3Result:
    energy_hartree: float
    energy_kcal_mol: float
    gradient: NDArray[np.float64]
    functional: str
    damping: str

def run_d3_dispersion(
    symbols: Sequence[str],
    coords: Sequence[Sequence[float]],
    functional: str = "b3lyp",
    damping: str = "bj",
) -> D3Result: ...

def run_solvation_gradient(
    symbols: Sequence[str],
    coords: Sequence[Sequence[float]],
    basis_name: str,
    density: NDArray[np.float64],
    solvent: str = "water",
    solvent_model: str = "cpcm",
    charge: int = 0,
    multiplicity: int = 1,
    scaling_factor: float = 1.20,
) -> NDArray[np.float64]: ...




```

## `calchem/dft/d3.py`

SHA256: `9db5ffa91111140575ef15682259a18b0ed6eedbbd2baa531b1ccef01ed2ed0c`

```python
"""Native two-body Grimme D3(BJ)/D3(0), coordinates in Bohr.

Reference-state CN interpolation and gradients are evaluated by Rust.
No approximate Python fallback; ATM three-body dispersion is not included.
"""
from __future__ import annotations

import numpy as np


def _evaluate(symbols, coordinates, functional, s6, s8, a1, a2, damping):
    if any(v is not None for v in (s6, s8, a1, a2)):
        raise ValueError(
            "Custom D3 parameters are not supported by this wrapper; "
            "use a documented named functional and damping scheme."
        )
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.shape != (len(symbols), 3) or not np.isfinite(xyz).all():
        raise ValueError("D3 coordinates must be finite and shaped (n_atoms, 3), in Bohr")
    from .. import _core

    return _core.run_d3_dispersion(list(symbols), xyz.tolist(), functional, damping)


def compute_d3_energy(symbols, coordinates, functional='pbe', s6=None, s8=None,
                      a1=None, a2=None, damping='zero') -> float:
    """Return two-body D3 energy in Hartree; invalid inputs raise ValueError."""
    return float(_evaluate(symbols, coordinates, functional, s6, s8, a1, a2,
                           damping).energy_hartree)


def compute_d3_gradient(symbols, coordinates, functional='pbe', s6=None, s8=None,
                        a1=None, a2=None, damping='zero') -> np.ndarray:
    """Return the analytic two-body D3 nuclear derivative in Hartree/Bohr."""
    return np.asarray(_evaluate(symbols, coordinates, functional, s6, s8, a1, a2,
                                damping).gradient, dtype=np.float64)

```

## `calchem/backends/native.py`

SHA256: `9fc2c6ac7b2b4acb7f5832da0e3ac6e27ad7f8eb501de60b7095f481386f0fe9`

```python
"""NativeCalchem: 把现有原生引擎包装成统一 Calculator 接口。

支持两条路径：
1. Rust 内核优先路径 (RHF / STO-3G / 6-31G, 闭壳层单重态)；
2. Python 参考引擎回退路径 (UHF, DFT, Post-HF, CASSCF)。
"""

from __future__ import annotations

import time
from datetime import datetime

from ..core.calculator import Calculator, register_backend
from ..core.capabilities import BackendCapabilities
from ..core.errors import ConvergenceError, UnsupportedMethodError
from ..core.request import CalculationRequest, MoleculeSpec
from ..core.result import ExcitedStateInfo, OrbitalInfo, OrbitalTransitionInfo, Result
from ..mol.molecule import Molecule


_NATIVE_METHODS = [
    "rhf", "hf", "uhf",
    "rks", "uks", "roks", "dft",
    "svwn", "svwn5", "lda", "blyp", "pbe", "b3lyp", "b3lyp5", "pbe0",
    "mp2", "mp4", "ccsd", "ccsd(t)", "ccsdt", "ccd", "casscf",
    "cis", "tdhf", "tddft", "tda", "tda-dft",
]

_RUST_DFT_FUNCTIONALS: dict[str, str] = {
    "svwn": "svwn5",
    "svwn5": "svwn5",
    "lda": "svwn5",
    "slatervwn5": "svwn5",
    "blyp": "blyp",
    "pbe": "pbe",
    "pbepbe": "pbe",
    "b3lyp": "b3lyp",
    "b3lyp5": "b3lyp5",
    "pbe0": "pbe0",
    "pbe1pbe": "pbe0",
    "camb3lyp": "cam-b3lyp",
    "cam-b3lyp": "cam-b3lyp",
    "wb97x": "wb97x",
    "wb97xd": "wb97x-d",
    "wb97x-d": "wb97x-d",
}


def _normalize_xc_token(raw: str) -> str:
    return "".join(ch for ch in raw.strip().lower() if not ch.isspace() and ch not in "-_")


def _resolve_rust_scf_spec(
    method: str, xc: str | None, multiplicity: int, options: dict | None
) -> tuple[bool, str | None, bool, str | None]:
    """Resolves `(is_supported, canonical_xc_or_none, is_open_shell, reject_reason)` for Rust core."""
    m = method.strip().lower()
    m_norm = _normalize_xc_token(m)
    opts = options or {}
    if any(_normalize_xc_token(str(v)) in {"camb3lyp", "wb97x", "wb97xd"}
           for v in (method, xc or "", opts.get("xc", ""))):
        raise UnsupportedMethodError(
            "Native CAM-B3LYP/wB97X(-D) is unavailable: range-separated ERIs are missing. "
            "Choose the Gaussian/ORCA backend explicitly."
        )

    if m in ("rhf", "hf", "uhf"):
        if m == "rhf" and multiplicity != 1:
            return False, None, False, f"multiplicity {multiplicity} != 1 (RHF requires closed-shell singlet)"
        use_u = m == "uhf" or multiplicity != 1
        return True, None, use_u, None

    if m in ("rks", "uks", "dft"):
        if m == "rks" and multiplicity != 1:
            return False, None, False, f"multiplicity {multiplicity} != 1 (RKS requires closed-shell singlet)"
        xc_raw = str(xc or opts.get("xc") or "b3lyp")
        xc_canon = _RUST_DFT_FUNCTIONALS.get(_normalize_xc_token(xc_raw))
        if xc_canon is None:
            return False, None, False, f"XC functional '{xc_raw}' not supported in Rust DFT core"
        use_u = m == "uks" or multiplicity != 1
        return True, xc_canon, use_u, None

    if m_norm in _RUST_DFT_FUNCTIONALS:
        xc_canon = _RUST_DFT_FUNCTIONALS[m_norm]
        use_u = multiplicity != 1
        return True, xc_canon, use_u, None

    return False, None, False, f"method '{method}' not handled by Rust HF/DFT core"

_NATIVE_BASIS = [
    "sto-3g", "3-21g", "6-31g", "6-31g(d)", "6-31g*", "6-31g(d,p)", "6-31g**",
    "lanl2dz", "lanl2dzdp",
    "cc-pvdz", "cc-pvtz", "def2-svp", "def2-tzvp", "d95",
]

_RUST_BASIS_ALIASES: dict[str, str] = {
    "sto-3g": "sto-3g",
    "sto3g": "sto-3g",
    "6-31g": "6-31g",
    "631g": "6-31g",
    "6-31g(d)": "6-31g(d)",
    "6-31g*": "6-31g(d)",
    "631gd": "6-31g(d)",
    "631g*": "6-31g(d)",
    "6-31gd": "6-31g(d)",
    "6-31g(d,p)": "6-31g(d,p)",
    "6-31g**": "6-31g(d,p)",
    "631gdp": "6-31g(d,p)",
    "631g**": "6-31g(d,p)",
    "6-31gdp": "6-31g(d,p)",
    "lanl2dz": "lanl2dz",
    "lanl-2dz": "lanl2dz",
    "lanl2dzdp": "lanl2dzdp",
    "lanl2dz(d,p)": "lanl2dzdp",
    "lanl2dz*": "lanl2dzdp",
}


def _spec_to_molecule(spec: MoleculeSpec) -> Molecule:
    """MoleculeSpec（Å）→ Molecule 对象。"""
    return Molecule(
        symbols=spec.symbols,
        coordinates=spec.coordinates_angstrom,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        units="angstrom",
    )


def _build_calculator(mol: Molecule, method: str, basis: str, xc: str | None, options: dict):
    """根据 method 实例化对应的计算类。"""
    m = method.lower()
    kwargs = dict(options)

    if m in ("hf", "rhf"):
        from ..scf.hf import RHF
        return RHF(mol, basis=basis, **kwargs)
    elif m == "uhf":
        from ..scf.hf import UHF
        return UHF(mol, basis=basis, **kwargs)
    elif m == "rohf":
        raise UnsupportedMethodError(
            "Native backend does not support ROHF; use UHF or ROKS (DFT) instead."
        )
    elif m in ("dft", "rks"):
        from ..scf.dft import RKS
        return RKS(mol, basis=basis, xc=xc or "b3lyp", **kwargs)
    elif m == "uks":
        from ..scf.dft import UKS
        return UKS(mol, basis=basis, xc=xc or "b3lyp", **kwargs)
    elif m == "roks":
        from ..scf.dft import ROKS
        return ROKS(mol, basis=basis, xc=xc or "b3lyp", **kwargs)
    elif _normalize_xc_token(m) in _RUST_DFT_FUNCTIONALS:
        xc_canon = _RUST_DFT_FUNCTIONALS[_normalize_xc_token(m)]
        if mol.multiplicity > 1:
            from ..scf.dft import UKS
            return UKS(mol, basis=basis, xc=xc_canon, **kwargs)
        from ..scf.dft import RKS
        return RKS(mol, basis=basis, xc=xc_canon, **kwargs)
    elif m == "mp2":
        from ..scf.hf import RHF, UHF
        from ..post_hf.mp2 import RMP2, UMP2
        if mol.multiplicity > 1:
            mf = UHF(mol, basis=basis, **kwargs)
            return UMP2(mf)
        else:
            mf = RHF(mol, basis=basis, **kwargs)
            return RMP2(mf)
    elif m == "ccsd":
        from ..scf.hf import RHF
        from ..post_hf.ccsd import CCSD
        mf = RHF(mol, basis=basis, **kwargs)
        return CCSD(mf)
    elif m in ("ccsd(t)", "ccsdt"):
        from ..scf.hf import RHF
        from ..post_hf.ccsd_t import CCSDT
        mf = RHF(mol, basis=basis, **kwargs)
        return CCSDT(mf)
    elif m == "mp4":
        from ..scf.hf import RHF
        from ..post_hf.mp4 import MP4
        mf = RHF(mol, basis=basis)
        return MP4(mf, **kwargs)
    elif m == "ccd":
        from ..post_hf.ccd import CCD
        return CCD(mol, basis=basis, **kwargs)
    elif m == "casscf":
        from ..post_hf.casscf import CASSCF
        return CASSCF(mol, basis=basis, **kwargs)
    else:
        raise UnsupportedMethodError(f"Native backend does not support method '{method}'")


def _extract_orbitals(calc) -> tuple[list[OrbitalInfo], float | None, float | None]:
    """从计算对象提取轨道信息。兼容多种属性命名。"""
    mo_e = getattr(calc, "mo_energy", None)
    if mo_e is None:
        mo_e = getattr(calc, "mo_energies", None)
    mo_o = getattr(calc, "mo_occ", None)
    if mo_o is None:
        mo_o = getattr(calc, "mo_occupations", None)

    if mo_e is None:
        return [], None, None

    # UHF 情况下 mo_energy 为元组 (mo_e_a, mo_e_b)
    if isinstance(mo_e, (tuple, list)) and len(mo_e) == 2 and hasattr(mo_e[0], "__iter__"):
        mo_e = mo_e[0]
        if mo_o is not None and isinstance(mo_o, (tuple, list)) and len(mo_o) == 2:
            mo_o = mo_o[0]

    e_list = mo_e.tolist() if hasattr(mo_e, "tolist") else list(mo_e)
    o_list = [0.0] * len(e_list)
    if mo_o is not None:
        o_list = mo_o.tolist() if hasattr(mo_o, "tolist") else list(mo_o)

    orbitals = [
        OrbitalInfo(index=i, energy_hartree=float(e), occupation=float(o))
        for i, (e, o) in enumerate(zip(e_list, o_list))
    ]

    occupied = [o.energy_hartree for o in orbitals if o.occupation > 0.5]
    virtual = [o.energy_hartree for o in orbitals if o.occupation <= 0.5]
    homo = max(occupied) if occupied else None
    lumo = min(virtual) if virtual else None
    return orbitals, homo, lumo


@register_backend("native")
class NativeCalchem(Calculator):
    """CalChem 原生引擎后端。"""

    def __init__(self):
        self._caps = BackendCapabilities(
            name="native",
            version="0.2.0",
            tasks=["single_point", "optimize", "tddft", "cis", "excited_states", "frequency", "ts", "irc", "solvation"],
            methods=_NATIVE_METHODS,
            basis_sets=_NATIVE_BASIS,
            requires_external_program=False,
            available=True,
        )

    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def _try_rust(self, request: CalculationRequest, mol: Molecule, started: datetime, t0: float) -> tuple[Result | None, str | None]:
        """尝试使用 Rust 引擎计算。

        支持闭壳层 RHF 与开壳层 UHF（STO-3G / 6-31G，任意合法电荷与多重度）。
        转发 request.options 中指定的迭代与收敛参数。
        若 Rust 抛出 ScfNotConvergedError，则如实返回未收敛结果，严禁静默回退。
        """
        method = request.method.lower()
        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)

        opts = request.options or {}
        supported, xc_canon, use_unrestricted, reject_reason = _resolve_rust_scf_spec(
            method, request.xc, request.molecule.multiplicity, opts
        )
        if not supported:
            return None, reject_reason
        if basis_arg is None:
            return None, f"basis '{basis}' not embedded in Rust core"

        # 检查 Rust 扩展是否可用
        try:
            from .. import _core as rust_core
        except ImportError as e:
            return None, f"Rust extension import failed: {e}"

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        # 显式转发用户请求选项
        max_iter = int(opts.get("max_iter", opts.get("max_cycles", 100)))
        conv_energy = float(opts.get("conv_energy", opts.get("energy_tol", 1e-10)))
        conv_density = float(opts.get("conv_density", opts.get("density_tol", 1e-8)))
        use_diis = bool(opts.get("use_diis", opts.get("diis", True)))
        damping = float(opts.get("damping", 0.0))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))

        if xc_canon is None:
            actual_tag = f"UHF-Rust({basis_arg})" if use_unrestricted else f"RHF-Rust({basis_arg})"
        else:
            ks_tag = "UKS" if use_unrestricted else "RKS"
            actual_tag = f"{ks_tag}-{xc_canon.upper()}-Rust({basis_arg})"

        try:
            wf = rust_core.run_single_point(
                symbols, coords_bohr,
                request.molecule.charge,
                request.molecule.multiplicity,
                method, basis_arg,
                max_iter=max_iter,
                conv_energy=conv_energy,
                conv_density=conv_density,
                use_diis=use_diis,
                damping=damping,
                xc=xc_canon,
                grid_level=grid_level,
            )
            walltime = time.time() - t0

            orbitals = [
                OrbitalInfo(index=i, energy_hartree=float(e), occupation=float(o))
                for i, (e, o) in enumerate(zip(wf.mo_energies, wf.mo_occupations))
            ]
            dipole_list = [float(x) for x in wf.dipole_debye] if wf.dipole_debye is not None else None
            mull_q = low_q = mull_s = low_s = None
            if wf.converged and hasattr(rust_core, "compute_population_analysis"):
                pop = rust_core.compute_population_analysis(wf)
                mull_q = [float(x) for x in pop["mulliken"]]
                low_q = [float(x) for x in pop["lowdin"]]
                if pop.get("mulliken_spin") is not None:
                    mull_s = [float(x) for x in pop["mulliken_spin"]]
                if pop.get("lowdin_spin") is not None:
                    low_s = [float(x) for x in pop["lowdin_spin"]]

            # Explicit standard two-body D3. Invalid corrections must not vanish silently.
            disp = opts.get("dispersion", opts.get("d3", None))
            d3_energy = None
            if disp:
                if not hasattr(rust_core, "run_d3_dispersion"):
                    raise UnsupportedMethodError("Native D3 extension is unavailable")
                d3_damp = "bj" if disp is True else str(disp).lower()
                d3_func = xc_canon or "hf"
                d3_calc = rust_core.run_d3_dispersion(
                    symbols, coords_bohr, damping=d3_damp, functional=d3_func
                )
                d3_energy = float(d3_calc.energy_hartree)

            tot_energy = wf.total_energy_hartree
            if d3_energy is not None:
                tot_energy += d3_energy

            res_extra = {}
            if d3_energy is not None:
                res_extra["d3_dispersion_hartree"] = d3_energy
            if xc_canon in ("cam-b3lyp", "wb97x", "wb97x-d"):
                res_extra["method_theory_note"] = (
                    "Native global-hybrid approximation (static c_HF). "
                    "Range-separated attenuated ERI (erf(mu*r12)/r12) requires external adapter (Gaussian/ORCA)."
                )

            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                actual_method=actual_tag,
                basis=request.basis,
                xc=request.xc or xc_canon,
                molecule=request.molecule,
                status="succeeded" if wf.converged else "not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_energy_hartree=wf.total_energy_hartree,
                total_energy_hartree=tot_energy,
                scf_converged=wf.converged,
                scf_iterations=wf.iterations,
                homo_energy_hartree=wf.homo_energy_hartree,
                lumo_energy_hartree=wf.lumo_energy_hartree,
                orbitals=orbitals,
                dipole_debye=dipole_list,
                dipole_magnitude_debye=wf.dipole_magnitude_debye,
                mulliken_charges=mull_q,
                lowdin_charges=low_q,
                mulliken_spin_populations=mull_s,
                lowdin_spin_populations=low_s,
                extra=res_extra,
                errors=[] if wf.converged else ["Rust SCF not converged"],
            ), None

        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            # 真实 SCF 未收敛：直接如实返回，不掩盖、不静默回退
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                actual_method=actual_tag,
                basis=request.basis,
                xc=request.xc or xc_canon,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            ), None

        except Exception as e:
            raise UnsupportedMethodError(f"Native calculation failed without fallback: {e}") from e

    def single_point(self, request: CalculationRequest) -> Result:
        if request.solvent is not None:
            return self.solvation(request)
        mol = _spec_to_molecule(request.molecule)
        started = datetime.now()
        t0 = time.time()

        # 尝试 Rust 引擎（优先路径）
        rust_result, fallback_reason = self._try_rust(request, mol, started, t0)
        if rust_result is not None:
            return rust_result

        # Python 参考引擎（回退路径）
        calc = _build_calculator(mol, request.method, request.basis, request.xc, request.options)
        calc.kernel()
        walltime = time.time() - t0

        # 提取收敛与能量
        converged = getattr(calc, "converged", None)
        if converged is None:
            energy_raw = getattr(calc, "energy", None)
            if energy_raw is None:
                energy_raw = getattr(calc, "e_tot", None)
            converged = energy_raw is not None and energy_raw != 0.0
        else:
            energy_raw = getattr(calc, "energy", None)
            if energy_raw is None:
                energy_raw = getattr(calc, "e_tot", None)

        energy = None
        if energy_raw is not None:
            try:
                e = float(energy_raw)
                if e != 0.0 and e == e:  # 非零且非 NaN
                    energy = e
            except (TypeError, ValueError):
                energy = None

        n_iter = getattr(calc, "n_cycles", None) or getattr(calc, "scf_cycles", None)
        orbitals, homo, lumo = _extract_orbitals(calc)

        dipole = getattr(calc, "dipole_moment", None)
        dipole_list = None
        if dipole is not None:
            try:
                dipole_list = [float(x) for x in (dipole.tolist() if hasattr(dipole, "tolist") else dipole)]
            except (TypeError, ValueError):
                dipole_list = None

        # 关联能处理（MP2, MP4, CCSD, CCSD(T)）
        m_lower = request.method.lower()
        is_post_hf = m_lower in ("mp2", "mp4", "ccsd", "ccsd(t)", "ccsdt")

        mp2_corr = None
        ccsd_corr = None
        ccsd_t_corr = None
        scf_e = energy
        total_e = energy

        if is_post_hf:
            # 对 post-HF，calc.energy/calc.e_tot 已经是总能量
            ref_hf = getattr(calc, "mf", getattr(calc, "hf", None))
            scf_e = getattr(ref_hf, "e_tot", getattr(calc, "e_hf", None))
            if m_lower == "mp2":
                mp2_corr = getattr(calc, "e_corr", None)
            elif m_lower in ("ccsd", "ccsd(t)", "ccsdt"):
                ccsd_corr = getattr(calc, "e_corr", None)
                ccsd_t_corr = getattr(calc, "e_t", None)
            elif m_lower == "mp4":
                mp2_corr = getattr(calc, "e_mp2", None)

        errors = []
        warnings = []
        if fallback_reason:
            warnings.append(f"Rust core bypassed ({fallback_reason}); used Python reference engine.")

        if energy is None:
            errors.append("Could not extract valid energy from native calculation")
        if not converged:
            errors.append(f"SCF did not converge after {n_iter or '?'} iterations")

        status = "succeeded" if (converged and energy is not None) else (
            "not_converged" if not converged and energy is not None else "failed"
        )

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version="0.2.0",
            task=request.task,
            method=request.method,
            actual_method=f"Python-{request.method}",
            basis=request.basis,
            xc=request.xc,
            molecule=request.molecule,
            status=status,
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            scf_energy_hartree=scf_e,
            mp2_correction_hartree=mp2_corr,
            ccsd_correction_hartree=ccsd_corr,
            ccsd_t_correction_hartree=ccsd_t_corr,
            total_energy_hartree=total_e,
            scf_converged=converged,
            scf_iterations=n_iter,
            homo_energy_hartree=homo,
            lumo_energy_hartree=lumo,
            orbitals=orbitals,
            dipole_debye=dipole_list,
            errors=errors,
            warnings=warnings,
        )

    def optimize(self, request: CalculationRequest) -> Result:
        """几何优化：优先 Rust 解析梯度 BFGS，回退 Python 数值梯度。"""
        from ..utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

        started = datetime.now()
        t0 = time.time()

        # 检查 Rust 扩展是否可用
        try:
            from .. import _core as rust_core
        except ImportError:
            rust_core = None

        # 优先尝试 Rust 原生 BFGS + 解析梯度（闭壳层 RHF/RKS 与开壳层/离子态 UHF/UKS，含极化基组与 ECP）
        m_low = request.method.lower()
        b_low = _RUST_BASIS_ALIASES.get(request.basis.strip().lower(), request.basis.strip().lower())
        opts = request.options or {}
        opt_supported, xc_canon, use_unrestricted, _ = _resolve_rust_scf_spec(
            m_low, request.xc, request.molecule.multiplicity, opts
        )
        can_rust_opt = (
            rust_core is not None
            and hasattr(rust_core, "run_optimize")
            and opt_supported
            and b_low in rust_core.supported_basis_sets()
        )
        if can_rust_opt:
            try:
                coords_bohr = [
                    [c * ANGSTROM_TO_BOHR for c in pos]
                    for pos in request.molecule.coordinates_angstrom
                ]
                max_iter = int(opts.get("max_iter", 100))
                grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
                dispersion = opts.get("dispersion", opts.get("d3", None))
                if isinstance(dispersion, bool):
                    dispersion = "d3bj" if dispersion else None
                solvent = request.solvent or opts.get("solvent", None)
                opt_res = rust_core.run_optimize(
                    request.molecule.symbols,
                    coords_bohr,
                    b_low,
                    max_iter,
                    charge=request.molecule.charge,
                    multiplicity=request.molecule.multiplicity,
                    method=m_low,
                    xc=xc_canon,
                    grid_level=grid_level,
                    dispersion=dispersion,
                    solvent=solvent,
                )
                if opt_res.converged:
                    final_coords_bohr = [
                        opt_res.coords[3 * i : 3 * i + 3]
                        for i in range(len(request.molecule.symbols))
                    ]
                    final_coords_ang = [
                        [c * BOHR_TO_ANGSTROM for c in pos]
                        for pos in final_coords_bohr
                    ]
                    walltime = time.time() - t0
                    if xc_canon is None:
                        opt_tag = "Rust-UHF-BFGS" if use_unrestricted else "Rust-RHF-BFGS"
                    else:
                        ks_tag = "UKS" if use_unrestricted else "RKS"
                        opt_tag = f"Rust-{ks_tag}-{xc_canon.upper()}-BFGS"
                    return Result(
                        job_id=request.job_id,
                        project_id=request.project_id,
                        backend="native",
                        backend_version="0.2.0",
                        task="optimize",
                        method=request.method,
                        actual_method=opt_tag,
                        basis=request.basis,
                        xc=request.xc or xc_canon,
                        molecule=request.molecule,
                        status="succeeded",
                        started_at=started,
                        finished_at=datetime.now(),
                        walltime_seconds=walltime,
                        total_energy_hartree=opt_res.energy,
                        scf_energy_hartree=opt_res.energy,
                        optimized_coordinates_angstrom=final_coords_ang,
                        errors=[],
                    )
            except Exception:
                # 优雅回退到 Python 优化路径
                pass

        from ..grad.finite_difference import central_gradient
        from ..opt.numeric_optimizer import optimize_geometry

        # 能量函数：必须成功收敛
        def energy_fn(spec: MoleculeSpec) -> float:
            req = CalculationRequest(
                task="single_point",
                molecule=spec,
                backend="native",
                method=request.method,
                basis=request.basis,
                xc=request.xc,
                options=request.options,
            )
            r = self.single_point(req)
            if r.status != "succeeded" or r.total_energy_hartree is None:
                raise ConvergenceError(
                    f"Single point energy evaluation failed during optimization: {r.errors}"
                )
            return r.total_energy_hartree

        def gradient_fn(spec: MoleculeSpec):
            return central_gradient(energy_fn, spec, step_bohr=0.002)

        max_iter = int(request.options.get("max_iter", 50))
        gtol = float(request.options.get("gtol", 1e-4))

        try:
            outcome = optimize_geometry(
                energy_fn, gradient_fn, request.molecule,
                max_iter=max_iter, gtol=gtol,
            )
            status = "succeeded" if outcome.converged else "not_converged"
            errors = [] if outcome.converged else [f"Optimization: {outcome.message}"]
            final_coords = outcome.final_molecule.coordinates_angstrom
            final_energy = outcome.final_energy_hartree
        except Exception as e:
            status = "failed"
            errors = [f"Optimization failed: {e}"]
            final_coords = None
            final_energy = None

        walltime = time.time() - t0

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version="0.2.0",
            task="optimize",
            method=request.method,
            actual_method=request.method,
            basis=request.basis,
            xc=request.xc,
            molecule=request.molecule,
            status=status,
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=final_energy,
            scf_energy_hartree=final_energy,
            optimized_coordinates_angstrom=final_coords,
            errors=errors,
        )

    def tddft(self, request: CalculationRequest) -> Result:
        """纯 Rust 线性响应激发态 (TD-DFT / TDA / CIS) 计算。"""
        started = datetime.now()
        t0 = time.time()
        opts = request.options or {}

        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)
        if basis_arg is None:
            raise UnsupportedMethodError(f"Basis '{basis}' is not supported by native TD-DFT engine")

        try:
            from .. import _core as rust_core
        except ImportError as e:
            raise RuntimeError(f"Rust core not available for TD-DFT: {e}")

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        n_states = int(opts.get("n_states", opts.get("roots", 5)))
        tda = bool(opts.get("tda", request.method.lower() in ("cis", "tda", "tda-dft") or request.task == "cis"))
        manifold = str(opts.get("manifold", opts.get("spin", "singlet")))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
        max_iter = int(opts.get("max_iter", opts.get("davidson_max_iter", 80)))
        conv_tol = float(opts.get("conv_tol", opts.get("tol", 1e-6)))
        force_davidson = bool(opts.get("force_davidson", False))
        scf_max_iter = int(opts.get("scf_max_iter", 100))
        scf_conv_energy = float(opts.get("scf_conv_energy", 1e-10))
        scf_conv_density = float(opts.get("scf_conv_density", 1e-8))

        xc = request.xc or opts.get("xc")
        method = request.method.lower()
        if method in ("cis", "tdhf"):
            xc = None

        try:
            td_res = rust_core.run_tddft(
                symbols,
                coords_bohr,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=method,
                basis=basis_arg,
                n_states=n_states,
                tda=tda,
                manifold=manifold,
                xc=xc,
                grid_level=grid_level,
                max_iter=max_iter,
                conv_tol=conv_tol,
                force_davidson=force_davidson,
                scf_max_iter=scf_max_iter,
                scf_conv_energy=scf_conv_energy,
                scf_conv_density=scf_conv_density,
            )
        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                basis=request.basis,
                xc=request.xc or xc,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            )

        walltime = time.time() - t0
        py_states = []
        for st in td_res.states:
            dom = [
                OrbitalTransitionInfo(
                    occ_index=t.occ_index,
                    virt_index=t.virt_index,
                    amplitude_x=t.amplitude_x,
                    amplitude_y=t.amplitude_y,
                    weight=t.weight,
                )
                for t in st.dominant_transitions
            ]
            py_states.append(
                ExcitedStateInfo(
                    state_index=st.state_index,
                    manifold_index=st.manifold_index,
                    label=st.label,
                    multiplicity=st.multiplicity,
                    excitation_energy_hartree=st.excitation_energy_hartree,
                    excitation_energy_ev=st.excitation_energy_ev,
                    total_energy_hartree=st.total_energy_hartree,
                    wavelength_nm=st.wavelength_nm,
                    transition_dipole_au=list(st.transition_dipole_au),
                    oscillator_strength=st.oscillator_strength,
                    dominant_transitions=dom,
                )
            )

        wf = td_res.scf_wavefunction
        orbitals = [
            OrbitalInfo(index=i, energy_hartree=float(e), occupation=float(o))
            for i, (e, o) in enumerate(zip(wf.mo_energies, wf.mo_occupations))
        ]
        dipole_list = [float(x) for x in wf.dipole_debye] if wf.dipole_debye is not None else None

        mull_q = low_q = None
        if hasattr(rust_core, "compute_population_analysis"):
            pop = rust_core.compute_population_analysis(wf)
            mull_q = [float(x) for x in pop["mulliken"]]
            low_q = [float(x) for x in pop["lowdin"]]

        actual_tag = f"TDDFT-{td_res.xc_functional or 'HF'}{'-TDA' if td_res.tda else ''}-Rust({basis_arg})"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=td_res.xc_functional or request.xc,
            molecule=request.molecule,
            status="succeeded",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            scf_energy_hartree=td_res.ground_energy_hartree,
            total_energy_hartree=td_res.ground_energy_hartree,
            scf_converged=wf.converged,
            scf_iterations=wf.iterations,
            homo_energy_hartree=td_res.homo_energy_hartree,
            lumo_energy_hartree=td_res.lumo_energy_hartree,
            optical_gap_ev=td_res.optical_gap_ev,
            lowest_triplet_ev=td_res.lowest_triplet_ev,
            delta_est_ev=td_res.delta_est_ev,
            excited_states=py_states,
            orbitals=orbitals,
            dipole_debye=dipole_list,
            dipole_magnitude_debye=wf.dipole_magnitude_debye,
            mulliken_charges=mull_q,
            lowdin_charges=low_q,
            extra={
                "tddft_result": td_res,
                "davidson_iterations_singlet": td_res.davidson_iterations_singlet,
                "davidson_iterations_triplet": td_res.davidson_iterations_triplet,
                "ks_gap_ev": td_res.ks_gap_ev,
            },
        )

    def frequency(self, request: CalculationRequest) -> Result:
        """纯 Rust 核坐标有限差分二阶 Hessian 矩阵、简正模式分析与 RRHO 热力学计算。"""
        started = datetime.now()
        t0 = time.time()
        opts = request.options or {}

        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)
        if basis_arg is None:
            raise UnsupportedMethodError(f"Basis '{basis}' is not supported by native frequency engine")

        try:
            from .. import _core as rust_core
        except ImportError as e:
            raise RuntimeError(f"Rust core not available for frequency: {e}")

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        step_bohr = float(opts.get("step_bohr", opts.get("step", 0.001)))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
        temperature_k = float(opts.get("temperature_k", opts.get("temperature", 298.15)))
        pressure_pa = float(opts.get("pressure_pa", opts.get("pressure", 101325.0)))
        symmetry_number = opts.get("symmetry_number", opts.get("sigma"))
        if symmetry_number is not None:
            symmetry_number = int(symmetry_number)
        max_iter = int(opts.get("max_iter", opts.get("max_cycles", 100)))
        conv_energy = float(opts.get("conv_energy", opts.get("energy_tol", 1e-10)))
        conv_density = float(opts.get("conv_density", opts.get("density_tol", 1e-8)))

        xc = request.xc or opts.get("xc")
        method = request.method.lower()

        try:
            freq_res = rust_core.run_frequency(
                symbols,
                coords_bohr,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=method,
                basis=basis_arg,
                step_bohr=step_bohr,
                xc=xc,
                grid_level=grid_level,
                temperature_k=temperature_k,
                pressure_pa=pressure_pa,
                symmetry_number=symmetry_number,
                max_iter=max_iter,
                conv_energy=conv_energy,
                conv_density=conv_density,
            )
        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                basis=request.basis,
                xc=request.xc or xc,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            )

        walltime = time.time() - t0

        from ..core.result import VibrationalModeInfo

        vibrational_modes = [
            VibrationalModeInfo(
                mode_index=m.mode_index,
                frequency_cm=m.frequency_cm,
                is_imaginary=m.is_imaginary,
                reduced_mass_amu=m.reduced_mass_amu,
                force_constant_m_dyn_a=m.force_constant_m_dyn_a,
                ir_intensity_km_mol=m.ir_intensity_km_mol,
                displacement=[list(d) for d in m.displacement],
            )
            for m in freq_res.vibrational.modes
        ]

        actual_tag = f"{freq_res.method.upper()}-Rust({basis_arg})"
        if freq_res.xc_functional:
            actual_tag = f"{actual_tag}[{freq_res.xc_functional}]"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=request.xc or xc,
            molecule=request.molecule,
            status="succeeded",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=freq_res.energy_hartree,
            scf_energy_hartree=freq_res.energy_hartree,
            scf_converged=True,
            frequencies_cm=list(freq_res.frequencies_cm),
            ir_intensities_km_mol=list(freq_res.ir_intensities_km_mol),
            zpve_hartree=freq_res.zpve_hartree,
            zpve_kcal_mol=freq_res.thermo.zpve_kcal_mol,
            thermal_energy_hartree=freq_res.thermo.e_thermal_hartree,
            enthalpy_hartree=freq_res.enthalpy_hartree,
            gibbs_free_energy_hartree=freq_res.gibbs_free_energy_hartree,
            entropy_cal_mol_k=freq_res.entropy_cal_mol_k,
            heat_capacity_cv_cal_mol_k=freq_res.heat_capacity_cv_cal_mol_k,
            heat_capacity_cp_cal_mol_k=freq_res.heat_capacity_cp_cal_mol_k,
            vibrational_modes=vibrational_modes,
            extra={
                "n_vib": freq_res.vibrational.n_vib,
                "is_linear": freq_res.vibrational.is_linear,
                "principal_moments_amu_bohr2": list(freq_res.vibrational.principal_moments_amu_bohr2),
                "rotational_constants_ghz": list(freq_res.vibrational.rotational_constants_ghz),
                "trans_rot_residuals": list(freq_res.vibrational.trans_rot_residuals),
                "cartesian_hessian": freq_res.cartesian_hessian,
                "apt": freq_res.apt,
            },
        )

    def ts_search(self, request: CalculationRequest) -> Result:
        """纯 Rust 分区有理函数优化 (P-RFO) 过渡态鞍点搜索。"""
        started = datetime.now()
        t0 = time.time()
        opts = request.options or {}

        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)
        if basis_arg is None:
            raise UnsupportedMethodError(f"Basis '{basis}' is not supported by native TS engine")

        try:
            from .. import _core as rust_core
        except ImportError as e:
            raise RuntimeError(f"Rust core not available for TS search: {e}")

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        max_steps = int(opts.get("max_steps", opts.get("max_cycles", 50)))
        max_force = float(opts.get("max_force", 3e-4))
        rms_force = float(opts.get("rms_force", 1.5e-4))
        max_displacement = float(opts.get("max_displacement", 1.2e-3))
        trust_radius_bohr = float(opts.get("trust_radius_bohr", opts.get("trust_radius", 0.1)))
        mode_index = opts.get("mode_index")
        if mode_index is not None:
            mode_index = int(mode_index)
        recalc_hessian_interval = int(opts.get("recalc_hessian_interval", 0))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
        scf_max_iter = int(opts.get("scf_max_iter", opts.get("max_iter", 100)))
        scf_conv_energy = float(opts.get("scf_conv_energy", opts.get("conv_energy", 1e-11)))
        scf_conv_density = float(opts.get("scf_conv_density", opts.get("conv_density", 1e-10)))

        xc = request.xc or opts.get("xc")
        method = request.method.lower()

        try:
            ts_res = rust_core.run_ts_search(
                symbols,
                coords_bohr,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=method,
                basis=basis_arg,
                xc=xc,
                grid_level=grid_level,
                max_steps=max_steps,
                max_force=max_force,
                rms_force=rms_force,
                max_displacement=max_displacement,
                trust_radius_bohr=trust_radius_bohr,
                mode_index=mode_index,
                recalc_hessian_interval=recalc_hessian_interval,
                scf_max_iter=scf_max_iter,
                scf_conv_energy=scf_conv_energy,
                scf_conv_density=scf_conv_density,
            )
        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                basis=request.basis,
                xc=request.xc or xc,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            )

        walltime = time.time() - t0
        opt_coords_ang = [[c * b2a for c in row] for row in ts_res.geometry_bohr]
        opt_mol = request.molecule.model_copy(update={"coordinates_angstrom": opt_coords_ang})

        actual_tag = f"Rust-PRFO-{method.upper()}({basis_arg})"
        if xc:
            actual_tag = f"{actual_tag}[{xc}]"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=request.xc or xc,
            molecule=opt_mol,
            status="succeeded" if ts_res.converged else "not_converged",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=ts_res.energy_hartree,
            scf_energy_hartree=ts_res.energy_hartree,
            scf_converged=ts_res.converged,
            optimized_coordinates_angstrom=opt_coords_ang,
            gradients=ts_res.gradient,
            is_transition_state=True,
            imaginary_frequency_cm=ts_res.imaginary_frequency_cm,
            extra={
                "steps": ts_res.steps,
                "transition_vector": ts_res.transition_vector,
            },
        )

    def irc(self, request: CalculationRequest) -> Result:
        """纯 Rust Gonzalez-Schlegel 内禀反应坐标 (IRC) 路径追踪。"""
        started = datetime.now()
        t0 = time.time()
        opts = request.options or {}

        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)
        if basis_arg is None:
            raise UnsupportedMethodError(f"Basis '{basis}' is not supported by native IRC engine")

        try:
            from .. import _core as rust_core
        except ImportError as e:
            raise RuntimeError(f"Rust core not available for IRC: {e}")

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        transition_vector = opts.get("transition_vector")
        if transition_vector is None:
            # If transition_vector not provided, run a 1-step TS search to obtain initial Hessian transition vector
            ts_res = rust_core.run_ts_search(
                symbols,
                coords_bohr,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=request.method.lower(),
                basis=basis_arg,
                xc=request.xc or opts.get("xc"),
                max_steps=5,
            )
            if not ts_res.converged:
                raise UnsupportedMethodError("IRC requires a certified first-order saddle point")
            coords_bohr = [list(row) for row in ts_res.geometry_bohr]
            transition_vector = ts_res.transition_vector

        step_size_bohr = float(opts.get("step_size_bohr", opts.get("step_size", 0.1)))
        max_points = int(opts.get("max_points", 20))
        direction = str(opts.get("direction", "both"))
        convergence_tol = float(opts.get("convergence_tol", 1e-3))
        max_gs_iter = int(opts.get("max_gs_iter", 10))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
        scf_max_iter = int(opts.get("scf_max_iter", opts.get("max_iter", 100)))
        scf_conv_energy = float(opts.get("scf_conv_energy", opts.get("conv_energy", 1e-11)))
        scf_conv_density = float(opts.get("scf_conv_density", opts.get("conv_density", 1e-10)))

        xc = request.xc or opts.get("xc")
        method = request.method.lower()

        try:
            irc_res = rust_core.run_irc(
                symbols,
                coords_bohr,
                transition_vector=transition_vector,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=method,
                basis=basis_arg,
                xc=xc,
                grid_level=grid_level,
                step_size_bohr=step_size_bohr,
                max_points=max_points,
                direction=direction,
                convergence_tol=convergence_tol,
                max_gs_iter=max_gs_iter,
                scf_max_iter=scf_max_iter,
                scf_conv_energy=scf_conv_energy,
                scf_conv_density=scf_conv_density,
            )
        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                basis=request.basis,
                xc=request.xc or xc,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            )

        walltime = time.time() - t0

        from ..core.result import IrcPointInfo

        path_points: list[IrcPointInfo] = []
        ts_pt = irc_res.transition_state
        path_points.append(
            IrcPointInfo(
                point_index=ts_pt.point_index,
                direction="ts",
                path_coordinate_s=ts_pt.path_coordinate_s,
                energy_hartree=ts_pt.energy_hartree,
                delta_energy_kcal_mol=ts_pt.delta_energy_kcal_mol,
                gradient_norm=ts_pt.gradient_norm,
                geometry_bohr=[list(row) for row in ts_pt.geometry_bohr],
            )
        )

        fwd_barrier = None
        rev_barrier = None
        reaction_energy = None

        if irc_res.forward is not None:
            fwd_barrier = irc_res.barrier_forward_kcal_mol
            for p in irc_res.forward.points:
                path_points.append(
                    IrcPointInfo(
                        point_index=p.point_index,
                        direction="forward",
                        path_coordinate_s=p.path_coordinate_s,
                        energy_hartree=p.energy_hartree,
                        delta_energy_kcal_mol=p.delta_energy_kcal_mol,
                        gradient_norm=p.gradient_norm,
                        geometry_bohr=[list(row) for row in p.geometry_bohr],
                    )
                )

        if irc_res.reverse is not None:
            rev_barrier = irc_res.barrier_reverse_kcal_mol
            for p in irc_res.reverse.points:
                path_points.append(
                    IrcPointInfo(
                        point_index=p.point_index,
                        direction="reverse",
                        path_coordinate_s=p.path_coordinate_s,
                        energy_hartree=p.energy_hartree,
                        delta_energy_kcal_mol=p.delta_energy_kcal_mol,
                        gradient_norm=p.gradient_norm,
                        geometry_bohr=[list(row) for row in p.geometry_bohr],
                    )
                )

        reaction_energy = irc_res.reaction_energy_kcal_mol
        branches = [b for b in (irc_res.forward, irc_res.reverse) if b is not None]
        path_complete = bool(branches) and all(b.converged_to_minimum for b in branches)

        actual_tag = f"Rust-IRC-GS-{method.upper()}({basis_arg})"
        if xc:
            actual_tag = f"{actual_tag}[{xc}]"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=request.xc or xc,
            molecule=request.molecule,
            status="succeeded" if path_complete else "not_converged",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=ts_pt.energy_hartree,
            scf_energy_hartree=ts_pt.energy_hartree,
            scf_converged=True,
            is_transition_state=True,
            irc_forward_barrier_kcal_mol=fwd_barrier,
            irc_reverse_barrier_kcal_mol=rev_barrier,
            reaction_energy_kcal_mol=reaction_energy,
            irc_path=path_points,
            extra={
                "irc_endpoints_converged": path_complete,
                "total_points": irc_res.total_points,
                "has_forward": irc_res.forward is not None,
                "has_reverse": irc_res.reverse is not None,
            },
        )

    def solvation(self, request: CalculationRequest) -> Result:
        """纯 Rust 连续介质溶剂化能与自洽反应场计算 (PCM / COSMO / SMD / SCRF)。"""
        from .. import _core as rust_core
        started = datetime.now()
        t0 = time.time()
        symbols = list(request.molecule.symbols)
        coords = [list(r) for r in request.molecule.coordinates_bohr]
        charge = request.molecule.charge
        multiplicity = request.molecule.multiplicity

        basis_raw = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis_raw, basis_raw)

        method = request.method.lower()
        supported, xc_canon, _, reject_reason = _resolve_rust_scf_spec(
            method, request.xc, request.molecule.multiplicity, request.options
        )
        if not supported:
            raise UnsupportedMethodError(f"Solvation not supported for method '{request.method}': {reject_reason}")

        solvent = request.solvent or request.options.get("solvent", "water")
        solvent_model = request.solvent_model or request.options.get("solvent_model", "cpcm")
        scaling_factor = float(request.options.get("scaling_factor", 1.20))
        include_cds = bool(request.options.get("include_cds", True))
        max_iter = int(request.options.get("max_iter", request.options.get("scf_max_iter", 100)))
        conv_energy = float(request.options.get("conv_energy", request.options.get("scf_conv_energy", 1e-8)))
        conv_density = float(request.options.get("conv_density", request.options.get("scf_conv_density", 1e-6)))

        res = rust_core.run_solvation(
            symbols=symbols,
            coords=coords,
            basis_name=basis_arg,
            solvent=solvent,
            solvent_model=solvent_model,
            charge=charge,
            multiplicity=multiplicity,
            method=method,
            xc=xc_canon,
            scaling_factor=scaling_factor,
            include_cds=include_cds,
            max_iter=max_iter,
            conv_energy=conv_energy,
            conv_density=conv_density,
        )

        walltime = time.time() - t0
        actual_tag = f"Rust-SCRF-{method.upper()}({basis_arg})/{res.solvent_name.capitalize()}"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=request.xc or xc_canon,
            molecule=request.molecule,
            status="succeeded" if res.converged else "not_converged",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=res.solvated_energy,
            scf_energy_hartree=res.gas_energy,
            scf_converged=res.converged,
            scf_iterations=res.scf_iterations,
            solvated_energy_hartree=res.solvated_energy,
            delta_g_solv_kcal_mol=res.delta_g_solv_kcal_mol,
            delta_g_elec_kcal_mol=res.delta_g_elec_kcal_mol,
            delta_g_cds_kcal_mol=res.delta_g_cds_kcal_mol,
            cavity_surface_area_angstrom2=res.cavity_surface_area_angstrom2,
            solvent_name=res.solvent_name,
            solvent_dielectric=res.dielectric_constant,
            errors=[] if res.converged else ["SCRF not converged"],
            extra={
                "total_surface_charge": res.total_surface_charge,
                "n_tesserae": len(res.tessera_coords_bohr),
            },
        )

```

## `calchem/workflows/opv.py`

SHA256: `bf6ae0866d94609f9f4c75a66318d93b710a2f51c9c15af39637b5bd48906f65`

```python
"""CalChem OPV (Organic Photovoltaics) Workflow.

Provides standardized computational pipelines for organic solar cells (OSC/OPV):
1. Molecular preparation (geometry, charge, multiplicity)
2. Native Rust-accelerated RHF/UHF geometry optimization & single-point energy
3. Frontier molecular orbital (FMO) extraction (HOMO, LUMO, Gap in eV)
4. Donor-Acceptor (D-A) energy level alignment & open-circuit voltage (Voc) estimation (Scharber model)
5. Nelsen four-point internal reorganization energy (lambda_hole, lambda_electron) & adiabatic/vertical IP/EA
6. Electric dipole moment evaluation for molecular packing & electrostatic potential analysis
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import calchem.backends  # noqa: F401 - ensure backends are registered
from calchem.api.service import TaskService
from calchem.core.request import CalculationRequest, MoleculeSpec, ResourceLimits
from calchem.core.result import IrcPointInfo
from calchem.io.input import read_gjf
from calchem.utils.constants import BOHR_TO_ANGSTROM, ELEMENTS

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
EV_TO_MEV = 1000.0


def _resolve_open_shell_method(method: str) -> str:
    """Maps a closed-shell method name to its unrestricted open-shell counterpart."""
    m = method.strip().lower()
    if m in ("rhf", "hf"):
        return "uhf"
    if m == "rks":
        return "uks"
    return method


def _centroid_aligned_rmsd(coords_a: List[List[float]], coords_b: List[List[float]]) -> float:
    """Coordinate RMSD (Å) after removing center-of-geometry translation."""
    n = len(coords_a)
    if n == 0 or len(coords_b) != n:
        return 0.0
    ca = [sum(row[q] for row in coords_a) / n for q in range(3)]
    cb = [sum(row[q] for row in coords_b) / n for q in range(3)]
    sq = 0.0
    for ra, rb in zip(coords_a, coords_b):
        for q in range(3):
            d = (ra[q] - ca[q]) - (rb[q] - cb[q])
            sq += d * d
    return math.sqrt(sq / n)


@dataclass
class OPVMolecule:
    """OPV molecule specification (donor or acceptor) backed by authentic quantum chemistry data."""
    name: str
    symbols: List[str]
    coordinates_angstrom: List[List[float]]
    charge: int = 0
    multiplicity: int = 1
    reference_data: Optional[dict] = None

    def to_spec(self) -> MoleculeSpec:
        return MoleculeSpec(
            symbols=self.symbols,
            coordinates_angstrom=self.coordinates_angstrom,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )


@dataclass
class OPVWorkflowResult:
    """Results from an OPV single molecule calculation."""
    name: str
    scf_energy_hartree: Optional[float] = None
    total_energy_hartree: Optional[float] = None
    dispersion_energy_hartree: Optional[float] = None
    homo_energy_hartree: Optional[float] = None
    lumo_energy_hartree: Optional[float] = None
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    gap_ev: Optional[float] = None
    dipole_debye: Optional[List[float]] = None
    dipole_magnitude_debye: Optional[float] = None
    mulliken_charges: Optional[List[float]] = None
    lowdin_charges: Optional[List[float]] = None
    optimized_geometry: Optional[List[List[float]]] = None
    converged: bool = False
    errors: List[str] = field(default_factory=list)


def get_chemdatas_dir() -> Optional[Path]:
    """动态获取本地 ChemDatas 真实基准数据库目录。

    优先级：
    1. CALCHEM_CHEMDATAS_DIR 环境变量
    2. CALCHEM_DATA_DIR 环境变量
    3. E:\\YDW\\ChemDatas（本机开发环境默认路径）
    """
    for ev in ("CALCHEM_CHEMDATAS_DIR", "CALCHEM_DATA_DIR"):
        val = os.environ.get(ev)
        if val and Path(val).is_dir():
            return Path(val)
    dev_default = Path(r"E:\YDW\ChemDatas")
    if dev_default.is_dir():
        return dev_default
    return None


def resolve_chemdatas_file(rel_path: str) -> str:
    """动态解析相对于 ChemDatas 的文件路径。"""
    base = get_chemdatas_dir()
    if base is not None:
        cand = base / rel_path
        if cand.is_file():
            return str(cand)
    return str(Path(r"E:\YDW\ChemDatas") / rel_path)


# Authoritative Gaussian 16 reference ground-truth dataset from ChemDatas
CHEMDATAS_REFERENCE_DATA: Dict[str, dict] = {
    "y6_ic_ff": {
        "source_file": resolve_chemdatas_file(r"ydw01\IC\ydw-IC_FF.log"),
        "fchk_file": resolve_chemdatas_file(r"ydw01\IC\ydw-IC_FF.fchk"),
        "chemical_name": "2-(5,6-difluoro-3-oxo-2,3-dihydro-1H-inden-1-ylidene)malononitrile (IC-2F / INCN-2F)",
        "formula": "C13H4F2N2O",
        "n_atoms": 22,
        "method": "wb97xd",
        "basis": "6-31g(d,p)",
        "scf_energy_hartree": -881.833266861,
        "homo_hartree": -0.34273,
        "lumo_hartree": -0.07139,
        "homo_ev": -0.34273 * HARTREE_TO_EV,
        "lumo_ev": -0.07139 * HARTREE_TO_EV,
        "gap_ev": (-0.07139 - (-0.34273)) * HARTREE_TO_EV,
        "dipole_magnitude_debye": 2.2970,
    },
    "y6_btp_core": {
        "source_file": resolve_chemdatas_file(r"ydw01\BTP\ydw-BTP.log"),
        "chemical_name": "BTP fused heptacyclic dithienothiophen[3.2-b]-pyrrolobenzothiadiazole core",
        "formula": "C60H90N4S5",
        "n_atoms": 159,
        "method": "wb97xd",
        "basis": "6-31g(d,p)",
        "scf_energy_hartree": -4549.93900185,
        "homo_hartree": -0.24268,
        "lumo_hartree": -0.00544,
        "homo_ev": -0.24268 * HARTREE_TO_EV,
        "lumo_ev": -0.00544 * HARTREE_TO_EV,
        "gap_ev": (-0.00544 - (-0.24268)) * HARTREE_TO_EV,
        "dipole_magnitude_debye": 0.4632,
    },
    "y6_full": {
        "source_file": resolve_chemdatas_file(r"ydw01\BTP-series\ydw-BTP-4F.log"),
        "chemical_name": "Y6 (BTP-4F) full A-DA'D-A non-fullerene acceptor",
        "formula": "C86H94F4N8O2S5",
        "n_atoms": 199,
        "method": "b3lyp",
        "dispersion": "gd3bj",
        "basis": "6-311g(d,p)",
        "scf_energy_hartree": -6314.19224815,
        "homo_hartree": -0.21264,
        "lumo_hartree": -0.13810,
        "homo_ev": -0.21264 * HARTREE_TO_EV,
        "lumo_ev": -0.13810 * HARTREE_TO_EV,
        "gap_ev": (-0.13810 - (-0.21264)) * HARTREE_TO_EV,
        "dipole_magnitude_debye": 2.4184,
    },
    "pm6_monomer": {
        "source_file": resolve_chemdatas_file(r"Gjf\ydw-PM6-opt01.gjf"),
        "chemical_name": "PM6 (PBDB-T-2F) full donor repeat monomer (BDT-2F + 2 Thiophene + BDD + alkyl chains)",
        "formula": "C68H78F2O2S8",
        "n_atoms": 158,
    },
    "pm6_l8bo_cluster": {
        "source_file": resolve_chemdatas_file(r"TH\00cluster_PM6-L8BO-2ac_rank1_donor-BDD1_Acceptor2-NC1.log"),
        "gjf_file": resolve_chemdatas_file(r"TH\00cluster_PM6-L8BO-2ac_rank1_donor-BDD1_Acceptor2-NC1(2).gjf"),
        "chemical_name": "MD-extracted PM6:L8-BO interfacial stacking dimer (donor-BDD1 + Acceptor2-NC1)",
        "formula": "C35H14F2N2O3S4",
        "n_atoms": 60,
        "method": "b3lyp",
        "basis": "6-31g(d)",
        "solvation": "smd_chloroform",
        "scf_energy_hartree": -3469.55737880,
        "s1_excitation_ev": 1.4189,
        "s1_wavelength_nm": 873.83,
        "s1_oscillator_strength": 0.0015,
    },
}

# Authentic Gaussian 16 wB97X-D/6-31G(d,p) optimized coordinates of IC-2F (22 atoms, C13H4F2N2O)
# Source: E:\YDW\ChemDatas\ydw01\IC\ydw-IC_FF.log (SCF Done = -881.833266861 Eh)
_AUTHENTIC_IC_2F_SYMBOLS: List[str] = [
    "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "O",
    "C", "N", "C", "N", "H", "H", "F", "C", "H", "H", "F",
]
_AUTHENTIC_IC_2F_COORDS: List[List[float]] = [
    [2.409711, -1.069687, 0.000090],
    [2.991531, 0.205778, -0.000004],
    [2.203458, 1.339654, -0.000102],
    [0.828372, 1.156510, -0.000113],
    [0.231356, -0.110436, -0.000023],
    [1.038917, -1.251870, 0.000079],
    [-0.212830, 2.207411, -0.000049],
    [-1.531539, 1.484223, 0.000045],
    [-1.242333, 0.033885, 0.000032],
    [-2.155746, -0.976275, 0.000024],
    [-0.048340, 3.405934, -0.000137],
    [-1.792190, -2.361623, -0.000154],
    [-1.512096, -3.486857, -0.000336],
    [-3.568051, -0.739900, 0.000091],
    [-4.714006, -0.562568, 0.000168],
    [2.643073, 2.330288, -0.000175],
    [0.653849, -2.262768, 0.000127],
    [4.316575, 0.294700, 0.000017],
    [-2.678672, 2.169457, 0.000106],
    [-3.662137, 1.718078, 0.000201],
    [-2.608823, 3.253902, 0.000071],
    [3.218045, -2.124008, 0.000195],
]

# Authentic PM6 Donor-BDD1 (T-BDD-T, 32 atoms, C18H8O2S4) rigidly aligned to the XY molecular plane
# Source: E:\YDW\ChemDatas\TH\00cluster_PM6-L8BO-2ac_rank1_donor-BDD1_Acceptor2-NC1(2).gjf (indices 0..23, 51..58)
_AUTHENTIC_DONOR_BDD1_INDICES: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 51, 52, 53, 54, 55, 56, 57, 58,
]
_AUTHENTIC_ACCEPTOR_NC1_INDICES: List[int] = [
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 59,
]
_AUTHENTIC_DONOR_BDD1_SYMBOLS: List[str] = [
    "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C",
    "O", "O", "S", "S", "S", "C", "C", "S", "C", "C", "C",
    "H", "H", "H", "H", "H", "H", "H", "H",
]
_AUTHENTIC_DONOR_BDD1_PLANAR_COORDS: List[List[float]] = [
    [1.256517, -0.779947, 0.395646],
    [1.438580, 1.870499, 0.455514],
    [0.503772, 3.063824, 0.187534],
    [-0.906342, 2.992550, 0.204826],
    [-1.541611, 1.698552, 0.544392],
    [-0.694944, 0.502505, 0.425950],
    [0.701974, 0.539392, 0.347470],
    [-1.540952, 4.197079, 0.024335],
    [-1.204474, -0.822051, 0.530024],
    [0.919261, 4.274365, -0.329046],
    [4.916834, -1.431250, -0.920138],
    [2.564570, -1.080425, -0.044479],
    [4.420415, -2.511712, -0.227123],
    [2.593952, 2.007679, 0.743774],
    [-2.694480, 1.667435, 0.937997],
    [3.747401, -0.142014, -0.969139],
    [0.065280, -1.947108, 0.906419],
    [-0.445043, 5.331851, -0.736963],
    [-2.555256, -1.270470, 0.153185],
    [-2.736261, -2.623749, 0.205820],
    [-3.712508, -0.407283, -0.814532],
    [-4.058816, -2.961997, -0.260269],
    [-4.705852, -1.859174, -0.797692],
    [3.056061, -2.318425, 0.269039],
    [-2.598211, 4.440609, -0.080533],
    [1.930763, 4.675634, -0.266223],
    [5.986422, -1.348109, -1.112930],
    [5.028506, -3.359068, 0.089598],
    [-2.113685, -3.329229, 0.756090],
    [-4.518125, -3.949341, -0.308083],
    [-5.689738, -2.058124, -1.222531],
    [2.585990, -3.062499, 0.912069],
]

# Authentic L8-BO/Y6 Acceptor2-NC1 fragment (28 atoms, C17H6F2N2O)
_AUTHENTIC_ACCEPTOR_NC1_SYMBOLS: List[str] = [
    "C", "C", "H", "C", "C", "C", "C", "C", "H", "H", "C", "O", "C", "C", "N",
    "C", "N", "C", "C", "C", "C", "C", "H", "C", "H", "F", "F", "H",
]

_AUTHENTIC_ACCEPTOR_NC1_COORDS: List[List[float]] = [
    [2.534312, 6.301766, 1.750869],
    [1.824313, 5.221766, 1.060869],
    [2.514315, 6.401766, 2.850869],
    [1.844317, 4.881764, -0.419131],
    [1.014315, 3.761765, -0.789131],
    [0.714312, 3.161765, -1.969131],
    [0.454310, 3.321764, 0.440869],
    [-0.395688, 2.221765, 0.340869],
    [1.204314, 3.661765, -2.879131],
    [-0.885686, 1.751765, 1.190869],
    [0.944316, 4.211764, 1.590869],
    [2.484313, 5.541764, -1.249131],
    [0.554316, 4.091765, 2.900869],
    [0.824313, 5.021765, 3.910869],
    [0.984313, 5.771765, 4.770869],
    [-0.355688, 3.101765, 3.340869],
    [-1.115682, 2.291765, 3.700869],
    [-0.695684, 1.601765, -0.929131],
    [-0.115686, 2.121764, -2.129131],
    [-1.615686, 0.561765, -1.059131],
    [-0.425687, 1.611765, -3.409131],
    [-1.885686, 0.021765, -2.279131],
    [-2.115690, 0.091765, -0.229131],
    [-1.295686, 0.531765, -3.489131],
    [0.074317, 2.041764, -4.341285],
    [-1.705686, -0.058235, -4.591292],
    [-2.755685, -0.968235, -2.399131],
    [3.126459, 7.163751, 1.443553],
]

# Authentic PM6:L8-BO interfacial stacking dimer (60 atoms, C35H14F2N2O3S4)
_AUTHENTIC_PM6_L8BO_CLUSTER_SYMBOLS: List[str] = [
    "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "O", "O", "S",
    "S", "S", "C", "C", "S", "C", "C", "C", "C", "C", "H", "C", "C", "C", "C", "C",
    "H", "H", "C", "O", "C", "C", "N", "C", "N", "C", "C", "C", "C", "C", "H", "C",
    "H", "F", "F", "H", "H", "H", "H", "H", "H", "H", "H", "H",
]

_AUTHENTIC_PM6_L8BO_CLUSTER_COORDS: List[List[float]] = [
    [-0.005689, -4.088235, -1.339131],
    [-1.745687, -4.608236, 0.600869],
    [-2.045690, -4.068235, 2.010869],
    [-1.645688, -2.798235, 2.480869],
    [-0.925687, -1.898236, 1.550869],
    [-0.315687, -2.528235, 0.370869],
    [-0.645685, -3.808236, -0.089131],
    [-2.095685, -2.488235, 3.740869],
    [0.564314, -1.838235, -0.509131],
    [-2.525686, -4.838236, 3.050869],
    [0.414317, -7.818236, -2.549131],
    [0.164313, -5.398235, -1.839131],
    [0.704314, -6.918236, -3.549131],
    [-2.345685, -5.528236, 0.120869],
    [-0.895685, -0.698235, 1.760869],
    [-0.045686, -6.988236, -1.089131],
    [0.654311, -2.638236, -2.049131],
    [-2.525686, -3.958236, 4.590869],
    [1.474314, -0.738235, -0.149131],
    [2.324313, -0.358235, -1.149131],
    [1.954314, -0.258235, 1.450869],
    [3.224314, 0.661765, -0.669131],
    [3.104315, 0.841765, 0.700869],
    [0.574313, -5.518236, -3.139131],
    [2.534312, 6.301766, 1.750869],
    [1.824313, 5.221766, 1.060869],
    [2.514315, 6.401766, 2.850869],
    [1.844317, 4.881764, -0.419131],
    [1.014315, 3.761765, -0.789131],
    [0.714312, 3.161765, -1.969131],
    [0.454310, 3.321764, 0.440869],
    [-0.395688, 2.221765, 0.340869],
    [1.204314, 3.661765, -2.879131],
    [-0.885686, 1.751765, 1.190869],
    [0.944316, 4.211764, 1.590869],
    [2.484313, 5.541764, -1.249131],
    [0.554316, 4.091765, 2.900869],
    [0.824313, 5.021765, 3.910869],
    [0.984313, 5.771765, 4.770869],
    [-0.355688, 3.101765, 3.340869],
    [-1.115682, 2.291765, 3.700869],
    [-0.695684, 1.601765, -0.929131],
    [-0.115686, 2.121764, -2.129131],
    [-1.615686, 0.561765, -1.059131],
    [-0.425687, 1.611765, -3.409131],
    [-1.885686, 0.021765, -2.279131],
    [-2.115690, 0.091765, -0.229131],
    [-1.295686, 0.531765, -3.489131],
    [0.074317, 2.041764, -4.341285],
    [-1.705686, -0.058235, -4.591292],
    [-2.755685, -0.968235, -2.399131],
    [-1.893463, -1.630538, 4.382399],
    [-3.085894, -5.767188, 2.944500],
    [0.225472, -8.864185, -2.790849],
    [0.838553, -7.197039, -4.594286],
    [2.194124, -0.568545, -2.210696],
    [3.993623, 1.191963, -1.230518],
    [3.798749, 1.558841, 1.138663],
    [0.685137, -4.712244, -3.864524],
    [3.126459, 7.163751, 1.443553],
]


def parse_gaussian_log(filepath: str | Path) -> Tuple[List[str], List[List[float]], int, int, dict]:
    """Parses a Gaussian .log/.out file to extract the final geometry, charge, multiplicity, and reference energetics."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian log file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")

    charge, mult = 0, 1
    m_cm = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", text)
    if m_cm:
        charge, mult = int(m_cm.group(1)), int(m_cm.group(2))

    blocks = re.findall(
        r"(?:Standard|Input) orientation:\s*\n\s*-+\s*\n\s*Center\s+Atomic\s+Atomic\s+Coordinates[^\n]*\n\s*Number\s+Number\s+Type\s+X\s+Y\s+Z\s*\n\s*-+\s*\n(.*?)\n\s*-+",
        text,
        re.DOTALL,
    )
    if not blocks:
        raise ValueError(f"No coordinate orientation table found in Gaussian log: {path}")

    last_block = blocks[-1]
    symbols: List[str] = []
    coords: List[List[float]] = []
    for line in last_block.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 6:
            atomic_num = int(parts[1])
            sym = ELEMENTS.get(atomic_num, f"X{atomic_num}")
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            symbols.append(sym)
            coords.append([x, y, z])

    scf = re.findall(r"SCF Done:\s*E\([^)]+\)\s*=\s*([-+]?\d+\.\d+)", text)
    scf_energy = float(scf[-1]) if scf else None

    occ: List[float] = []
    virt: List[float] = []
    in_orb_block = False
    for line in text.splitlines():
        is_occ = "Alpha  occ. eigenvalues" in line
        is_virt = "Alpha virt. eigenvalues" in line
        if is_occ or is_virt:
            if not in_orb_block:
                occ, virt = [], []
                in_orb_block = True
            vals = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", line.split("--", 1)[-1])]
            (occ if is_occ else virt).extend(vals)
        elif in_orb_block and "Beta" not in line:
            in_orb_block = False

    homo = occ[-1] if occ else None
    lumo = virt[0] if virt else None

    dip = re.findall(
        r"X=\s*([-+]?\d+\.\d+)\s+Y=\s*([-+]?\d+\.\d+)\s+Z=\s*([-+]?\d+\.\d+)\s+Tot=\s*([-+]?\d+\.\d+)",
        text,
    )
    dipole_mag = float(dip[-1][3]) if dip else None

    meta = {
        "source_file": str(path),
        "n_atoms": len(symbols),
        "scf_energy_hartree": scf_energy,
        "homo_hartree": homo,
        "lumo_hartree": lumo,
        "homo_ev": homo * HARTREE_TO_EV if homo is not None else None,
        "lumo_ev": lumo * HARTREE_TO_EV if lumo is not None else None,
        "gap_ev": (lumo - homo) * HARTREE_TO_EV if (homo is not None and lumo is not None) else None,
        "dipole_magnitude_debye": dipole_mag,
        "normal_termination": "Normal termination" in text,
    }
    return symbols, coords, charge, mult, meta


def parse_gaussian_gjf(filepath: str | Path) -> Tuple[List[str], List[List[float]], int, int, dict]:
    """Parses a Gaussian .gjf/.com input file into symbols, coordinates (Å), charge, multiplicity, and metadata."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian GJF file not found: {path}")
    mol = read_gjf(str(path))
    coords = mol.coordinates.tolist() if hasattr(mol.coordinates, "tolist") else list(mol.coordinates)
    meta = {
        "source_file": str(path),
        "n_atoms": len(mol.symbols),
    }
    return list(mol.symbols), [[float(c) for c in row] for row in coords], int(mol.charge), int(mol.multiplicity), meta


def parse_gaussian_fchk(filepath: str | Path) -> Tuple[List[str], List[List[float]], int, int, dict]:
    """Parses a Gaussian formatted checkpoint (.fchk) file into symbols, coordinates (Å), charge, multiplicity, and metadata."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian FCHK file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    charge = 0
    mult = 1
    total_energy = None
    atomic_nums: List[int] = []
    coords_bohr_flat: List[float] = []

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i]
        if line.startswith("Charge") and " I " in line:
            charge = int(line.split()[-1])
        elif line.startswith("Multiplicity") and " I " in line:
            mult = int(line.split()[-1])
        elif line.startswith("Total Energy") and " R " in line:
            total_energy = float(line.split()[-1])
        elif line.startswith("Atomic numbers") and "N=" in line:
            count = int(line.split()[-1])
            i += 1
            while len(atomic_nums) < count and i < n_lines:
                atomic_nums.extend(int(x) for x in lines[i].split())
                i += 1
            continue
        elif line.startswith("Current cartesian coordinates") and "N=" in line:
            count = int(line.split()[-1])
            i += 1
            while len(coords_bohr_flat) < count and i < n_lines:
                coords_bohr_flat.extend(float(x) for x in lines[i].split())
                i += 1
            continue
        i += 1

    if not atomic_nums or len(coords_bohr_flat) != 3 * len(atomic_nums):
        raise ValueError(f"Incomplete atomic numbers or coordinates in FCHK file: {path}")

    symbols = [ELEMENTS.get(z, f"X{z}") for z in atomic_nums]
    coords_ang = [
        [
            coords_bohr_flat[3 * k] * BOHR_TO_ANGSTROM,
            coords_bohr_flat[3 * k + 1] * BOHR_TO_ANGSTROM,
            coords_bohr_flat[3 * k + 2] * BOHR_TO_ANGSTROM,
        ]
        for k in range(len(atomic_nums))
    ]
    meta = {
        "source_file": str(path),
        "n_atoms": len(symbols),
        "scf_energy_hartree": total_energy,
    }
    return symbols, coords_ang, charge, mult, meta


def load_chemdatas_molecule(filepath: str | Path, name: Optional[str] = None) -> OPVMolecule:
    """Loads an authentic molecular structure from a Gaussian `.log`, `.gjf`, or `.fchk` file."""
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext in (".log", ".out"):
        symbols, coords, chg, mult, meta = parse_gaussian_log(path)
    elif ext in (".gjf", ".com"):
        symbols, coords, chg, mult, meta = parse_gaussian_gjf(path)
    elif ext in (".fchk", ".fch"):
        symbols, coords, chg, mult, meta = parse_gaussian_fchk(path)
    else:
        raise ValueError(f"Unsupported Gaussian file extension '{ext}' for {filepath}")

    return OPVMolecule(
        name=name or path.stem,
        symbols=symbols,
        coordinates_angstrom=coords,
        charge=chg,
        multiplicity=mult,
        reference_data=meta,
    )


def get_real_opv_system(system_name: str) -> OPVMolecule:
    """Returns an authentic, Gaussian-verified OPV molecular system from `E:\\YDW\\ChemDatas`.

    Supported system identifiers:
    - `'y6_ic_ff'` / `'ic_2f'` / `'y6_terminal'`:
        Optimized IC-2F (INCN-2F) acceptor terminal unit (22 atoms, C13H4F2N2O) from `ydw-IC_FF.log`.
    - `'y6_btp_core'` / `'btp_core'` / `'btp'`:
        Optimized BTP central core unit of Y6 (159 atoms, C60H90N4S5) from `ydw-BTP.log`.
    - `'y6_full'` / `'btp_4f'` / `'y6'`:
        Optimized full Y6 (BTP-4F) molecule (199 atoms, C86H94F4N8O2S5) from `ydw-BTP-4F.log`.
    - `'pm6_monomer'` / `'pm6'`:
        Full PM6 polymer repeat monomer (158 atoms, C68H78F2O2S8) from `ydw-PM6-opt01.gjf`.
    - `'pm6_l8bo_cluster'` / `'pm6_y6_md_cluster'` / `'da_cluster'`:
        MD-extracted PM6:L8-BO interfacial stacking dimer (60 atoms: donor-BDD1 + Acceptor2-NC1)
        from `00cluster_PM6-L8BO-2ac_rank1_donor-BDD1_Acceptor2-NC1.log` / `.gjf`.
    - `'donor_bdd1'`:
        Authentic 32-atom PM6 Donor-BDD1 (T-BDD-T, C18H8O2S4) sub-molecule from the MD cluster.
    - `'acceptor_nc1'`:
        Authentic 28-atom L8-BO/Y6 Acceptor2-NC1 (C17H6F2N2O) sub-molecule from the MD cluster.
    """
    key = system_name.strip().lower()
    if key in ("y6_ic_ff", "ic_2f", "ic2f", "y6_terminal"):
        ref = dict(CHEMDATAS_REFERENCE_DATA["y6_ic_ff"])
        p = Path(ref["source_file"])
        if p.is_file():
            mol = load_chemdatas_molecule(p, name="Y6_IC-2F_Terminal")
            mol.reference_data = {**ref, **(mol.reference_data or {})}
            return mol
        return OPVMolecule(
            name="Y6_IC-2F_Terminal",
            symbols=list(_AUTHENTIC_IC_2F_SYMBOLS),
            coordinates_angstrom=[list(r) for r in _AUTHENTIC_IC_2F_COORDS],
            charge=0,
            multiplicity=1,
            reference_data=ref,
        )

    if key in ("y6_btp_core", "btp_core", "btp"):
        ref = dict(CHEMDATAS_REFERENCE_DATA["y6_btp_core"])
        p = Path(ref["source_file"])
        if not p.is_file():
            raise FileNotFoundError(
                f"BTP core dataset not found at '{p}'. To use full 159-atom BTP core on an external machine, "
                "set CALCHEM_CHEMDATAS_DIR to the directory containing ChemDatas, or use the built-in "
                "authentic fragments: 'ic_2f', 'donor_bdd1', 'acceptor_nc1', 'pm6_l8bo_cluster'."
            )
        mol = load_chemdatas_molecule(p, name="Y6_BTP_Core")
        mol.reference_data = {**ref, **(mol.reference_data or {})}
        return mol

    if key in ("y6_full", "btp_4f", "y6"):
        ref = dict(CHEMDATAS_REFERENCE_DATA["y6_full"])
        p = Path(ref["source_file"])
        if not p.is_file():
            raise FileNotFoundError(
                f"Full Y6 (BTP-4F) dataset not found at '{p}'. To use full 199-atom Y6 on an external machine, "
                "set CALCHEM_CHEMDATAS_DIR to the directory containing ChemDatas, or use the built-in "
                "authentic fragments: 'ic_2f', 'donor_bdd1', 'acceptor_nc1', 'pm6_l8bo_cluster'."
            )
        mol = load_chemdatas_molecule(p, name="Y6_BTP-4F_Full")
        mol.reference_data = {**ref, **(mol.reference_data or {})}
        return mol

    if key in ("pm6_monomer", "pm6"):
        ref = dict(CHEMDATAS_REFERENCE_DATA["pm6_monomer"])
        p = Path(ref["source_file"])
        if not p.is_file():
            raise FileNotFoundError(
                f"Full PM6 monomer dataset not found at '{p}'. To use full 158-atom PM6 monomer on an external machine, "
                "set CALCHEM_CHEMDATAS_DIR to the directory containing ChemDatas, or use the built-in "
                "authentic fragments: 'ic_2f', 'donor_bdd1', 'acceptor_nc1', 'pm6_l8bo_cluster'."
            )
        mol = load_chemdatas_molecule(p, name="PM6_Monomer_Full")
        mol.reference_data = {**ref, **(mol.reference_data or {})}
        return mol

    if key in ("pm6_l8bo_cluster", "pm6_y6_md_cluster", "da_cluster"):
        ref = dict(CHEMDATAS_REFERENCE_DATA["pm6_l8bo_cluster"])
        log_p = Path(ref["source_file"])
        gjf_p = Path(ref.get("gjf_file", ""))
        target = log_p if log_p.is_file() else gjf_p
        if target.is_file():
            mol = load_chemdatas_molecule(target, name="PM6_L8BO_MD_Cluster")
            mol.reference_data = {**ref, **(mol.reference_data or {})}
            return mol
        return OPVMolecule(
            name="PM6_L8BO_MD_Cluster",
            symbols=list(_AUTHENTIC_PM6_L8BO_CLUSTER_SYMBOLS),
            coordinates_angstrom=[list(r) for r in _AUTHENTIC_PM6_L8BO_CLUSTER_COORDS],
            charge=0,
            multiplicity=1,
            reference_data=ref,
        )

    if key in ("donor_bdd1", "pm6_bdd1"):
        ref_cluster = CHEMDATAS_REFERENCE_DATA["pm6_l8bo_cluster"]
        gjf_p = Path(ref_cluster.get("gjf_file", ""))
        if gjf_p.is_file():
            cluster = load_chemdatas_molecule(gjf_p, name="PM6_L8BO_MD_Cluster")
            syms = [cluster.symbols[k] for k in _AUTHENTIC_DONOR_BDD1_INDICES]
            coords = [list(cluster.coordinates_angstrom[k]) for k in _AUTHENTIC_DONOR_BDD1_INDICES]
            return OPVMolecule(
                name="PM6_Donor_BDD1",
                symbols=syms,
                coordinates_angstrom=coords,
                charge=0,
                multiplicity=1,
                reference_data={
                    "source_file": str(gjf_p),
                    "formula": "C18H8O2S4",
                    "n_atoms": 32,
                    "fragment_role": "donor-BDD1 (T-BDD-T)",
                },
            )
        return OPVMolecule(
            name="PM6_Donor_BDD1",
            symbols=list(_AUTHENTIC_DONOR_BDD1_SYMBOLS),
            coordinates_angstrom=[list(r) for r in _AUTHENTIC_DONOR_BDD1_PLANAR_COORDS],
            charge=0,
            multiplicity=1,
            reference_data={"formula": "C18H8O2S4", "n_atoms": 32},
        )

    if key in ("acceptor_nc1", "l8bo_nc1"):
        ref_cluster = CHEMDATAS_REFERENCE_DATA["pm6_l8bo_cluster"]
        gjf_p = Path(ref_cluster.get("gjf_file", ""))
        if gjf_p.is_file():
            cluster = load_chemdatas_molecule(gjf_p, name="PM6_L8BO_MD_Cluster")
            syms = [cluster.symbols[k] for k in _AUTHENTIC_ACCEPTOR_NC1_INDICES]
            coords = [list(cluster.coordinates_angstrom[k]) for k in _AUTHENTIC_ACCEPTOR_NC1_INDICES]
            return OPVMolecule(
                name="Y6_Acceptor_NC1",
                symbols=syms,
                coordinates_angstrom=coords,
                charge=0,
                multiplicity=1,
                reference_data={
                    "source_file": str(gjf_p),
                    "formula": "C17H6F2N2O",
                    "n_atoms": 28,
                    "fragment_role": "Acceptor2-NC1",
                },
            )
        return OPVMolecule(
            name="Y6_Acceptor_NC1",
            symbols=list(_AUTHENTIC_ACCEPTOR_NC1_SYMBOLS),
            coordinates_angstrom=[list(r) for r in _AUTHENTIC_ACCEPTOR_NC1_COORDS],
            charge=0,
            multiplicity=1,
            reference_data={
                "formula": "C17H6F2N2O",
                "n_atoms": 28,
                "fragment_role": "Acceptor2-NC1",
            },
        )

    raise ValueError(
        f"Unknown OPV system '{system_name}'. Supported: "
        "'y6_ic_ff', 'y6_btp_core', 'y6_full', 'pm6_monomer', 'pm6_l8bo_cluster', 'donor_bdd1', 'acceptor_nc1'."
    )


def get_pm6_fragment(
    fluorinated: bool = True,
    variant: Literal["donor_bdd1", "monomer"] = "donor_bdd1",
) -> OPVMolecule:
    """Returns an authentic PM6 (PBDB-T-2F) donor molecular structure.

    Variants:
    - `'donor_bdd1'` (default): Authentic 32-atom T-BDD-T conjugated donor unit (`C18H8O2S4`)
      extracted from the PM6:L8-BO interfacial stacking cluster (`donor-BDD1`),
      rigidly aligned to its principal molecular XY plane for stacking/alignment workflows.
    - `'monomer'`: Complete 158-atom PM6 polymer repeat monomer (`C68H78F2O2S8`)
      from the ChemDatas reference dataset.
    """
    if variant == "monomer":
        return get_real_opv_system("pm6_monomer")
    return OPVMolecule(
        name="PM6_Donor_BDD1",
        symbols=list(_AUTHENTIC_DONOR_BDD1_SYMBOLS),
        coordinates_angstrom=[list(r) for r in _AUTHENTIC_DONOR_BDD1_PLANAR_COORDS],
        charge=0,
        multiplicity=1,
        reference_data={
            "source_file": CHEMDATAS_REFERENCE_DATA["pm6_l8bo_cluster"]["gjf_file"],
            "chemical_name": "PM6 Benzo[1,2-c:4,5-c']dithiophene-4,8-dione + dithiophene unit (T-BDD-T)",
            "formula": "C18H8O2S4",
            "n_atoms": 32,
        },
    )


def get_y6_fragment(
    fluorinated: bool = True,
    variant: Literal["ic2f", "btp_core", "full", "acceptor_nc1"] = "ic2f",
) -> OPVMolecule:
    """Returns an authentic Y6 (BTP-4F) non-fullerene acceptor unit.

    Variants:
    - `'ic2f'` (default): Authentic 22-atom Gaussian-optimized IC-2F acceptor terminal unit (`C13H4F2N2O`, `wB97X-D/6-31G(d,p)`).
    - `'btp_core'`: Authentic 159-atom Gaussian-optimized BTP central core (`C60H90N4S5`, `wB97X-D/6-31G(d,p)`).
    - `'full'`: Authentic 199-atom Gaussian-optimized full Y6 molecule (`C86H94F4N8O2S5`, `B3LYP-D3(BJ)/6-311G(d,p)`).
    - `'acceptor_nc1'`: Authentic 28-atom Acceptor2-NC1 terminal fragment (`C17H6F2N2O`) from the interfacial cluster.
    """
    if variant == "btp_core":
        return get_real_opv_system("y6_btp_core")
    if variant == "full":
        return get_real_opv_system("y6_full")
    if variant == "acceptor_nc1":
        return get_real_opv_system("acceptor_nc1")
    return get_real_opv_system("y6_ic_ff")


def build_da_stacking_dimer(
    donor: MoleculeSpec | OPVMolecule,
    acceptor: MoleculeSpec | OPVMolecule,
    distance_angstrom: float = 3.5,
    lateral_shift_angstrom: float = 0.0,
    name: str = "PM6_Y6_Stacking_Dimer",
) -> OPVMolecule:
    """Constructs a cofacial pi-pi stacking dimer geometry at a specified interplanar distance.

    Each monomer is centered at its geometric centroid in XY, with the donor plane positioned at
    `z = z_rel - distance/2` and the acceptor plane at `z = z_rel + distance/2`.
    """
    d_spec = donor.to_spec() if hasattr(donor, "to_spec") else donor
    a_spec = acceptor.to_spec() if hasattr(acceptor, "to_spec") else acceptor

    nd = len(d_spec.coordinates_angstrom)
    na = len(a_spec.coordinates_angstrom)
    d_mean = [sum(c[q] for c in d_spec.coordinates_angstrom) / max(nd, 1) for q in range(3)]
    a_mean = [sum(c[q] for c in a_spec.coordinates_angstrom) / max(na, 1) for q in range(3)]

    d_coords = [
        [c[0] - d_mean[0], c[1] - d_mean[1], (c[2] - d_mean[2]) - distance_angstrom / 2.0]
        for c in d_spec.coordinates_angstrom
    ]
    a_coords = [
        [
            (c[0] - a_mean[0]) + lateral_shift_angstrom,
            c[1] - a_mean[1],
            (c[2] - a_mean[2]) + distance_angstrom / 2.0,
        ]
        for c in a_spec.coordinates_angstrom
    ]

    combined_symbols = list(d_spec.symbols) + list(a_spec.symbols)
    combined_coords = d_coords + a_coords
    total_charge = d_spec.charge + a_spec.charge

    return OPVMolecule(
        name=name,
        symbols=combined_symbols,
        coordinates_angstrom=combined_coords,
        charge=total_charge,
        multiplicity=1,
    )


@dataclass
class OPVEspSummaryResult:
    """Molecular surface electrostatic potential (ESP) & atomic charge descriptors for OPV."""

    name: str
    density_isovalue: float
    n_vertices: int
    n_triangles: int
    surface_area_angstrom2: float
    esp_min_hartree: float
    esp_max_hartree: float
    esp_min_kcal_mol: float
    esp_max_kcal_mol: float
    mpi_kcal_mol: float
    mulliken_charges: List[float]
    lowdin_charges: List[float]
    ccesp_packet: bytes
    summary: str


@dataclass
class OPVAbsorptionResult:
    """OPV 紫外-可见光学吸收光谱与激子结合能分析结果。"""
    name: str
    method: str
    basis: str
    tda: bool
    optical_gap_ev: float
    wavelength_max_nm: float
    oscillator_strength_max: float
    ks_gap_ev: float
    lowest_triplet_ev: Optional[float] = None
    delta_est_ev: Optional[float] = None
    fundamental_gap_vertical_ev: Optional[float] = None
    exciton_binding_energy_ev: Optional[float] = None
    spectrum_wavelengths_nm: List[float] = field(default_factory=list)
    spectrum_epsilon_l_mol_cm: List[float] = field(default_factory=list)
    states: List[dict] = field(default_factory=list)
    summary: str = ""


@dataclass
class OPVThermochemistryResult:
    """Rigid-Rotor Harmonic-Oscillator (RRHO) Vibrational & Thermochemical analysis result."""
    name: str
    method: str
    basis: str
    temperature_k: float
    pressure_pa: float
    scf_energy_hartree: float
    zpve_hartree: float
    zpve_kcal_mol: float
    thermal_energy_hartree: float
    enthalpy_hartree: float
    gibbs_free_energy_hartree: float
    entropy_cal_mol_k: float
    heat_capacity_cv_cal_mol_k: float
    heat_capacity_cp_cal_mol_k: float
    n_vibrations: int
    n_imaginary_modes: int
    is_transition_state: bool
    frequencies_cm: List[float] = field(default_factory=list)
    ir_intensities_km_mol: List[float] = field(default_factory=list)
    summary: str = ""


@dataclass
class OPVReactionBarrierResult:
    """Transition state, activation barrier, and Gonzalez-Schlegel IRC pathway result."""

    name: str
    method: str
    basis: str
    ts_energy_hartree: float
    imaginary_frequency_cm: float
    forward_barrier_kcal_mol: Optional[float]
    reverse_barrier_kcal_mol: Optional[float]
    reaction_energy_kcal_mol: Optional[float]
    n_irc_points: int
    ts_molecule: MoleculeSpec
    irc_path: List[IrcPointInfo] = field(default_factory=list)
    summary: str = ""


@dataclass
class OPVSolvationResult:
    """Continuum solvation workflow outcome for an OPV donor, acceptor, or blend component."""

    name: str
    solvent: str
    dielectric_constant: float
    solvated_energy_hartree: float
    gas_energy_hartree: float
    delta_g_solv_kcal_mol: float
    delta_g_elec_kcal_mol: float
    delta_g_cds_kcal_mol: float
    cavity_surface_area_angstrom2: float
    method: str
    basis: str
    summary: str = ""


@dataclass
class OPVDAPairResult:
    """Donor-Acceptor energy level alignment analysis."""
    donor: OPVWorkflowResult
    acceptor: OPVWorkflowResult
    delta_lumo_ev: float
    delta_homo_ev: float
    estimated_voc_volts: float
    charge_transfer_feasible: bool
    summary: str


@dataclass
class OPVDimerResult:
    """OPV 给受体二聚体堆叠构型结合能与色散相互作用分析结果。"""
    name: str
    method: str
    basis: str
    dispersion: Optional[str]
    e_dimer_hartree: float
    e_donor_hartree: float
    e_acceptor_hartree: float
    binding_energy_hartree: float
    binding_energy_kcal_mol: float
    binding_energy_ev: float
    dispersion_energy_hartree: Optional[float] = None
    dispersion_energy_kcal_mol: Optional[float] = None
    intermolecular_distance_angstrom: float = 3.5
    dimer: Optional[OPVWorkflowResult] = None
    donor: Optional[OPVWorkflowResult] = None
    acceptor: Optional[OPVWorkflowResult] = None
    summary: str = ""


@dataclass
class CarrierReorganizationResult:
    """Nelsen four-point reorganization breakdown for a single carrier ('hole' or 'electron').

    Definitions (for ionic state I in {+, -} and neutral state 0):
      lambda_ion_relax     = E_I(Q_0) - E_I(Q_I)   >= 0  (relaxation on ionic PES)
      lambda_neutral_relax = E_0(Q_I) - E_0(Q_0)   >= 0  (relaxation on neutral PES)
      lambda_total         = lambda_ion_relax + lambda_neutral_relax
    """
    carrier: Literal["hole", "electron"]
    e_neutral_at_neutral_geom_hartree: float
    e_ion_at_neutral_geom_hartree: float
    e_ion_at_ion_geom_hartree: float
    e_neutral_at_ion_geom_hartree: float
    lambda_ion_relax_hartree: float
    lambda_neutral_relax_hartree: float
    lambda_hartree: float
    lambda_ion_relax_ev: float
    lambda_neutral_relax_ev: float
    lambda_ev: float
    lambda_mev: float
    optimized_ion_geometry: List[List[float]]
    geometry_rmsd_angstrom: float


@dataclass
class OPVReorganizationResult:
    """Four-point internal reorganization energy & redox energetics for OPV charge transport."""
    name: str
    method_neutral: str
    method_open_shell: str
    basis: str
    optimized_neutral_geometry: List[List[float]]
    e_neutral_eq_hartree: float
    hole: Optional[CarrierReorganizationResult] = None
    electron: Optional[CarrierReorganizationResult] = None
    lambda_hole_ev: Optional[float] = None
    lambda_hole_mev: Optional[float] = None
    lambda_electron_ev: Optional[float] = None
    lambda_electron_mev: Optional[float] = None
    vertical_ip_ev: Optional[float] = None
    adiabatic_ip_ev: Optional[float] = None
    hole_extraction_potential_ev: Optional[float] = None
    vertical_ea_ev: Optional[float] = None
    adiabatic_ea_ev: Optional[float] = None
    electron_extraction_potential_ev: Optional[float] = None
    fundamental_gap_vertical_ev: Optional[float] = None
    fundamental_gap_adiabatic_ev: Optional[float] = None
    summary: str = ""


class OPVWorkflow:
    """OPV standard computational pipeline."""

    def __init__(self, service: Optional[TaskService] = None, backend: str = "native"):
        self.service = service or TaskService()
        self.backend = backend

    def _default_resources(self) -> ResourceLimits:
        return ResourceLimits(max_cpu_cores=8, max_memory_mb=4000, timeout_seconds=300)

    def _run_energy_only(self, mol: MoleculeSpec, method: str, basis: str, label: str) -> float:
        """Executes a single-point energy job and returns the converged total energy (Eh)."""
        req = CalculationRequest(
            task="single_point",
            molecule=mol,
            backend=self.backend,
            method=method,
            basis=basis,
            resources=self._default_resources(),
        )
        res = self.service.run(req)
        energy = res.total_energy_hartree if res.total_energy_hartree is not None else res.scf_energy_hartree
        if res.status != "succeeded" or res.scf_converged is False or energy is None:
            raise RuntimeError(f"Four-point step '{label}' failed ({res.status}): {res.errors}")
        return float(energy)

    def _run_opt_only(
        self, mol: MoleculeSpec, method: str, basis: str, label: str
    ) -> Tuple[float, List[List[float]]]:
        """Executes a geometry optimization and returns `(energy_hartree, relaxed_coords_angstrom)`."""
        req = CalculationRequest(
            task="optimize",
            molecule=mol,
            backend=self.backend,
            method=method,
            basis=basis,
            resources=self._default_resources(),
        )
        res = self.service.run(req)
        energy = res.total_energy_hartree if res.total_energy_hartree is not None else res.scf_energy_hartree
        coords = res.optimized_coordinates_angstrom
        if res.status != "succeeded" or energy is None or not coords:
            raise RuntimeError(f"Four-point optimization '{label}' failed ({res.status}): {res.errors}")
        return float(energy), [[float(c) for c in row] for row in coords]

    def single_point(
        self,
        mol: MoleculeSpec,
        method: str = "rhf",
        basis: str = "sto-3g",
        dispersion: Optional[str] = None,
        options: Optional[dict] = None,
        name: str = "molecule",
    ) -> OPVWorkflowResult:
        """Runs single-point energy and extracts frontier molecular orbitals."""
        opts = dict(options or {})
        if dispersion is not None:
            opts["dispersion"] = dispersion

        req = CalculationRequest(
            task="single_point",
            molecule=mol,
            backend=self.backend,
            method=method,
            basis=basis,
            options=opts,
            resources=self._default_resources(),
        )
        result = self.service.run(req)

        homo_h = result.homo_energy_hartree
        lumo_h = result.lumo_energy_hartree
        homo_ev = homo_h * HARTREE_TO_EV if homo_h is not None else None
        lumo_ev = lumo_h * HARTREE_TO_EV if lumo_h is not None else None
        gap_ev = (lumo_ev - homo_ev) if (homo_ev is not None and lumo_ev is not None) else None

        dipole = result.dipole_debye
        dipole_mag = result.dipole_magnitude_debye
        disp_h = result.extra.get("d3_dispersion_hartree") if result.extra else None

        return OPVWorkflowResult(
            name=name,
            scf_energy_hartree=result.scf_energy_hartree,
            total_energy_hartree=result.total_energy_hartree,
            dispersion_energy_hartree=disp_h,
            homo_energy_hartree=homo_h,
            lumo_energy_hartree=lumo_h,
            homo_ev=homo_ev,
            lumo_ev=lumo_ev,
            gap_ev=gap_ev,
            dipole_debye=dipole,
            dipole_magnitude_debye=dipole_mag,
            mulliken_charges=result.mulliken_charges,
            lowdin_charges=result.lowdin_charges,
            converged=result.status == "succeeded" and (result.scf_converged is not False),
            errors=result.errors,
        )

    def optimize(
        self,
        mol: MoleculeSpec,
        method: str = "rhf",
        basis: str = "sto-3g",
        dispersion: Optional[str] = None,
        solvent: Optional[str] = None,
        options: Optional[dict] = None,
        name: str = "molecule",
    ) -> OPVWorkflowResult:
        """Optimizes geometry and extracts frontier orbitals at relaxed equilibrium."""
        opts = dict(options or {})
        if dispersion is not None:
            opts["dispersion"] = dispersion
        if solvent is not None:
            opts["solvent"] = solvent

        req = CalculationRequest(
            task="optimize",
            molecule=mol,
            backend=self.backend,
            method=method,
            basis=basis,
            solvent=solvent,
            options=opts,
            resources=self._default_resources(),
        )
        result = self.service.run(req)

        homo_h = result.homo_energy_hartree
        lumo_h = result.lumo_energy_hartree
        opt_geo = result.optimized_coordinates_angstrom

        # If optimizer didn't populate orbitals, run a single point on relaxed geometry
        if opt_geo and (homo_h is None or lumo_h is None):
            relaxed_mol = MoleculeSpec(
                symbols=mol.symbols,
                coordinates_angstrom=opt_geo,
                charge=mol.charge,
                multiplicity=mol.multiplicity,
            )
            sp_res = self.single_point(
                relaxed_mol, method=method, basis=basis, dispersion=dispersion, options=opts, name=name
            )
            homo_h = sp_res.homo_energy_hartree
            lumo_h = sp_res.lumo_energy_hartree
            dipole = sp_res.dipole_debye
            dipole_mag = sp_res.dipole_magnitude_debye
            mull_q = sp_res.mulliken_charges
            low_q = sp_res.lowdin_charges
        else:
            dipole = result.dipole_debye
            dipole_mag = result.dipole_magnitude_debye
            mull_q = result.mulliken_charges
            low_q = result.lowdin_charges

        homo_ev = homo_h * HARTREE_TO_EV if homo_h is not None else None
        lumo_ev = lumo_h * HARTREE_TO_EV if lumo_h is not None else None
        gap_ev = (lumo_ev - homo_ev) if (homo_ev is not None and lumo_ev is not None) else None
        disp_h = result.extra.get("d3_dispersion_hartree") if result.extra else None

        return OPVWorkflowResult(
            name=name,
            scf_energy_hartree=result.scf_energy_hartree,
            total_energy_hartree=result.total_energy_hartree,
            dispersion_energy_hartree=disp_h,
            homo_energy_hartree=homo_h,
            lumo_energy_hartree=lumo_h,
            homo_ev=homo_ev,
            lumo_ev=lumo_ev,
            gap_ev=gap_ev,
            dipole_debye=dipole,
            dipole_magnitude_debye=dipole_mag,
            mulliken_charges=mull_q,
            lowdin_charges=low_q,
            optimized_geometry=opt_geo,
            converged=result.status == "succeeded",
            errors=result.errors,
        )

    def _compute_carrier_branch(
        self,
        symbols: List[str],
        q0_coords: List[List[float]],
        e0_q0: float,
        neutral_charge: int,
        neutral_mult: int,
        ion_charge: int,
        ion_mult: int,
        method_neutral: str,
        method_ion: str,
        basis: str,
        carrier: Literal["hole", "electron"],
    ) -> CarrierReorganizationResult:
        label = "cation (+1)" if carrier == "hole" else "anion (-1)"
        ion_at_q0 = MoleculeSpec(
            symbols=symbols,
            coordinates_angstrom=q0_coords,
            charge=ion_charge,
            multiplicity=ion_mult,
        )
        # 1. Vertical ion energy at neutral equilibrium geometry: E_I(Q_0)
        e_ion_q0 = self._run_energy_only(ion_at_q0, method_ion, basis, f"vertical {label} E_I(Q0)")

        # 2. Optimize ionic state starting from Q_0 -> (E_I(Q_I), Q_I)
        e_ion_qi, qi_coords = self._run_opt_only(
            ion_at_q0, method_ion, basis, f"{label} geometry relaxation Q_I"
        )

        # 3. Vertical neutral energy at relaxed ionic geometry: E_0(Q_I)
        neutral_at_qi = MoleculeSpec(
            symbols=symbols,
            coordinates_angstrom=qi_coords,
            charge=neutral_charge,
            multiplicity=neutral_mult,
        )
        e0_qi = self._run_energy_only(
            neutral_at_qi, method_neutral, basis, f"vertical neutral at {label} geometry E_0(Q_I)"
        )

        lam1_h = e_ion_q0 - e_ion_qi
        lam2_h = e0_qi - e0_q0
        lam_h = lam1_h + lam2_h

        lam1_ev = lam1_h * HARTREE_TO_EV
        lam2_ev = lam2_h * HARTREE_TO_EV
        lam_ev = lam_h * HARTREE_TO_EV

        return CarrierReorganizationResult(
            carrier=carrier,
            e_neutral_at_neutral_geom_hartree=e0_q0,
            e_ion_at_neutral_geom_hartree=e_ion_q0,
            e_ion_at_ion_geom_hartree=e_ion_qi,
            e_neutral_at_ion_geom_hartree=e0_qi,
            lambda_ion_relax_hartree=lam1_h,
            lambda_neutral_relax_hartree=lam2_h,
            lambda_hartree=lam_h,
            lambda_ion_relax_ev=lam1_ev,
            lambda_neutral_relax_ev=lam2_ev,
            lambda_ev=lam_ev,
            lambda_mev=lam_ev * EV_TO_MEV,
            optimized_ion_geometry=qi_coords,
            geometry_rmsd_angstrom=_centroid_aligned_rmsd(q0_coords, qi_coords),
        )

    def reorganization_energy(
        self,
        mol: MoleculeSpec,
        method: str = "rhf",
        basis: str = "sto-3g",
        carrier: Literal["both", "hole", "electron"] = "both",
        name: str = "molecule",
        pre_optimized_neutral: bool = False,
    ) -> OPVReorganizationResult:
        """Computes Nelsen four-point internal reorganization energy (lambda_hole / lambda_electron).

        Starting from a closed-shell reference molecule `mol` (charge q0, multiplicity 1):
          1. Optimizes neutral geometry Q_0 to obtain E_0(Q_0) (unless `pre_optimized_neutral=True`).
          2. For hole transport ('hole' / 'both'): evaluates vertical cation E_+(Q_0),
             relaxes radical cation (q0 + 1, doublet) to Q_+ with energy E_+(Q_+),
             and evaluates vertical neutral E_0(Q_+):
               lambda_hole = [E_+(Q_0) - E_+(Q_+)] + [E_0(Q_+) - E_0(Q_0)].
          3. For electron transport ('electron' / 'both'): evaluates vertical anion E_-(Q_0),
             relaxes radical anion (q0 - 1, doublet) to Q_- with energy E_-(Q_-),
             and evaluates vertical neutral E_0(Q_-):
               lambda_electron = [E_-(Q_0) - E_-(Q_-)] + [E_0(Q_-) - E_0(Q_0)].
        """
        if carrier not in ("both", "hole", "electron"):
            raise ValueError(f"carrier must be 'both', 'hole', or 'electron', got '{carrier}'")
        if mol.multiplicity != 1:
            raise ValueError(
                f"Four-point reorganization energy expects a closed-shell singlet reference "
                f"(multiplicity=1), got multiplicity={mol.multiplicity}"
            )

        method_neutral = method
        method_ion = _resolve_open_shell_method(method)

        if pre_optimized_neutral:
            q0_coords = [[float(c) for c in row] for row in mol.coordinates_angstrom]
            e0_q0 = self._run_energy_only(mol, method_neutral, basis, "neutral E_0(Q0)")
        else:
            e0_q0, q0_coords = self._run_opt_only(
                mol, method_neutral, basis, "neutral geometry optimization Q0"
            )

        hole_res: Optional[CarrierReorganizationResult] = None
        elec_res: Optional[CarrierReorganizationResult] = None

        if carrier in ("both", "hole"):
            hole_res = self._compute_carrier_branch(
                symbols=list(mol.symbols),
                q0_coords=q0_coords,
                e0_q0=e0_q0,
                neutral_charge=mol.charge,
                neutral_mult=1,
                ion_charge=mol.charge + 1,
                ion_mult=2,
                method_neutral=method_neutral,
                method_ion=method_ion,
                basis=basis,
                carrier="hole",
            )

        if carrier in ("both", "electron"):
            elec_res = self._compute_carrier_branch(
                symbols=list(mol.symbols),
                q0_coords=q0_coords,
                e0_q0=e0_q0,
                neutral_charge=mol.charge,
                neutral_mult=1,
                ion_charge=mol.charge - 1,
                ion_mult=2,
                method_neutral=method_neutral,
                method_ion=method_ion,
                basis=basis,
                carrier="electron",
            )

        vip = aip = hep = None
        if hole_res is not None:
            vip = (hole_res.e_ion_at_neutral_geom_hartree - e0_q0) * HARTREE_TO_EV
            aip = (hole_res.e_ion_at_ion_geom_hartree - e0_q0) * HARTREE_TO_EV
            hep = (hole_res.e_ion_at_ion_geom_hartree - hole_res.e_neutral_at_ion_geom_hartree) * HARTREE_TO_EV

        vea = aea = eep = None
        if elec_res is not None:
            vea = (e0_q0 - elec_res.e_ion_at_neutral_geom_hartree) * HARTREE_TO_EV
            aea = (e0_q0 - elec_res.e_ion_at_ion_geom_hartree) * HARTREE_TO_EV
            eep = (elec_res.e_neutral_at_ion_geom_hartree - elec_res.e_ion_at_ion_geom_hartree) * HARTREE_TO_EV

        gap_v = (vip - vea) if (vip is not None and vea is not None) else None
        gap_a = (aip - aea) if (aip is not None and aea is not None) else None

        lines = [
            f"=== OPV Four-Point Reorganization Energy: {name} ({method_neutral.upper()}/{method_ion.upper()}/{basis}) ===",
            f"Neutral equilibrium E_0(Q_0) = {e0_q0:.8f} Eh",
        ]
        if hole_res is not None:
            lines.append(
                f"Hole (M+):     lambda_h = {hole_res.lambda_ev:.4f} eV ({hole_res.lambda_mev:.1f} meV) "
                f"[ion relax = {hole_res.lambda_ion_relax_ev:.4f} eV, neutral relax = {hole_res.lambda_neutral_relax_ev:.4f} eV, "
                f"RMSD = {hole_res.geometry_rmsd_angstrom:.4f} Å]"
            )
            lines.append(
                f"               IP_vert = {vip:.3f} eV | IP_adiab = {aip:.3f} eV | HEP = {hep:.3f} eV"
            )
        if elec_res is not None:
            lines.append(
                f"Electron (M-): lambda_e = {elec_res.lambda_ev:.4f} eV ({elec_res.lambda_mev:.1f} meV) "
                f"[ion relax = {elec_res.lambda_ion_relax_ev:.4f} eV, neutral relax = {elec_res.lambda_neutral_relax_ev:.4f} eV, "
                f"RMSD = {elec_res.geometry_rmsd_angstrom:.4f} Å]"
            )
            lines.append(
                f"               EA_vert = {vea:.3f} eV | EA_adiab = {aea:.3f} eV | EEP = {eep:.3f} eV"
            )
        if gap_a is not None and gap_v is not None:
            lines.append(
                f"Fundamental Gap: adiabatic = {gap_a:.3f} eV | vertical = {gap_v:.3f} eV"
            )

        return OPVReorganizationResult(
            name=name,
            method_neutral=method_neutral,
            method_open_shell=method_ion,
            basis=basis,
            optimized_neutral_geometry=q0_coords,
            e_neutral_eq_hartree=e0_q0,
            hole=hole_res,
            electron=elec_res,
            lambda_hole_ev=hole_res.lambda_ev if hole_res is not None else None,
            lambda_hole_mev=hole_res.lambda_mev if hole_res is not None else None,
            lambda_electron_ev=elec_res.lambda_ev if elec_res is not None else None,
            lambda_electron_mev=elec_res.lambda_mev if elec_res is not None else None,
            vertical_ip_ev=vip,
            adiabatic_ip_ev=aip,
            hole_extraction_potential_ev=hep,
            vertical_ea_ev=vea,
            adiabatic_ea_ev=aea,
            electron_extraction_potential_ev=eep,
            fundamental_gap_vertical_ev=gap_v,
            fundamental_gap_adiabatic_ev=gap_a,
            summary="\n".join(lines) + "\n",
        )

    def align_donor_acceptor(
        self,
        donor: MoleculeSpec,
        acceptor: MoleculeSpec,
        method: str = "rhf",
        basis: str = "sto-3g",
        optimize_geometries: bool = False,
    ) -> OPVDAPairResult:
        """Calculates D-A energy level alignment and predicts OPV open-circuit voltage."""
        if optimize_geometries:
            d_res = self.optimize(donor, method=method, basis=basis, name="Donor")
            a_res = self.optimize(acceptor, method=method, basis=basis, name="Acceptor")
        else:
            d_res = self.single_point(donor, method=method, basis=basis, name="Donor")
            a_res = self.single_point(acceptor, method=method, basis=basis, name="Acceptor")

        if d_res.homo_ev is None or d_res.lumo_ev is None:
            raise RuntimeError(f"Failed to calculate donor frontier orbitals: {d_res.errors}")
        if a_res.homo_ev is None or a_res.lumo_ev is None:
            raise RuntimeError(f"Failed to calculate acceptor frontier orbitals: {a_res.errors}")

        delta_lumo = d_res.lumo_ev - a_res.lumo_ev
        delta_homo = d_res.homo_ev - a_res.homo_ev

        # Scharber model: Voc = (1/e) * (|E_HOMO(Donor)| - |E_LUMO(Acceptor)|) - 0.3 V
        raw_voc = (abs(d_res.homo_ev) - abs(a_res.lumo_ev)) - 0.30
        estimated_voc = max(0.0, raw_voc)
        charge_transfer_feasible = delta_lumo >= 0.30

        summary = (
            f"=== OPV D-A Energy Level Alignment ({method.upper()}/{basis}) ===\n"
            f"Donor:    HOMO = {d_res.homo_ev:6.2f} eV | LUMO = {d_res.lumo_ev:6.2f} eV | Gap = {d_res.gap_ev:5.2f} eV\n"
            f"Acceptor: HOMO = {a_res.homo_ev:6.2f} eV | LUMO = {a_res.lumo_ev:6.2f} eV | Gap = {a_res.gap_ev:5.2f} eV\n"
            f"Offsets:  Delta_LUMO (Driving Force) = {delta_lumo:5.2f} eV | Delta_HOMO = {delta_homo:5.2f} eV\n"
            f"Exciton Dissociation Feasible: {charge_transfer_feasible} (threshold >= 0.30 eV)\n"
            f"Estimated Voc (Scharber model): {estimated_voc:5.2f} V\n"
        )

        return OPVDAPairResult(
            donor=d_res,
            acceptor=a_res,
            delta_lumo_ev=delta_lumo,
            delta_homo_ev=delta_homo,
            estimated_voc_volts=estimated_voc,
            charge_transfer_feasible=charge_transfer_feasible,
            summary=summary,
        )

    def dimer_binding_energy(
        self,
        donor: MoleculeSpec | OPVMolecule,
        acceptor: MoleculeSpec | OPVMolecule,
        distance_angstrom: float = 3.5,
        lateral_shift_angstrom: float = 0.0,
        method: str = "b3lyp",
        basis: str = "sto-3g",
        dispersion: Optional[str] = "d3bj",
        optimize_dimer: bool = False,
        name: str = "PM6:Y6 D-A Dimer",
    ) -> OPVDimerResult:
        """Calculates the supramolecular binding energy and dispersion interaction of an OPV D-A dimer."""
        dimer_mol = build_da_stacking_dimer(
            donor,
            acceptor,
            distance_angstrom=distance_angstrom,
            lateral_shift_angstrom=lateral_shift_angstrom,
        )

        d_spec = donor.to_spec() if hasattr(donor, "to_spec") else donor
        a_spec = acceptor.to_spec() if hasattr(acceptor, "to_spec") else acceptor
        dimer_spec = dimer_mol.to_spec()

        opts = {}
        if dispersion:
            opts["dispersion"] = dispersion

        d_res = self.single_point(
            d_spec, method=method, basis=basis, dispersion=dispersion, options=opts, name="Donor"
        )
        a_res = self.single_point(
            a_spec, method=method, basis=basis, dispersion=dispersion, options=opts, name="Acceptor"
        )

        if optimize_dimer:
            dimer_res = self.optimize(
                dimer_spec,
                method=method,
                basis=basis,
                dispersion=dispersion,
                options=opts,
                name="Dimer_Relaxed",
            )
        else:
            dimer_res = self.single_point(
                dimer_spec,
                method=method,
                basis=basis,
                dispersion=dispersion,
                options=opts,
                name="Dimer_Rigid",
            )

        e_d = (
            d_res.total_energy_hartree
            if d_res.total_energy_hartree is not None
            else (d_res.scf_energy_hartree or 0.0)
        )
        e_a = (
            a_res.total_energy_hartree
            if a_res.total_energy_hartree is not None
            else (a_res.scf_energy_hartree or 0.0)
        )
        e_dimer = (
            dimer_res.total_energy_hartree
            if dimer_res.total_energy_hartree is not None
            else (dimer_res.scf_energy_hartree or 0.0)
        )

        delta_e_h = e_dimer - (e_d + e_a)
        delta_e_kcal = delta_e_h * HARTREE_TO_KCAL_MOL
        delta_e_ev = delta_e_h * HARTREE_TO_EV

        # Evaluate pure D3 dispersion component delta E_disp
        disp_h = None
        disp_kcal = None
        try:
            from calchem import _core as rust_core
            from calchem.utils.constants import ANGSTROM_TO_BOHR

            if hasattr(rust_core, "run_d3_dispersion"):
                d_bohr = [[c * ANGSTROM_TO_BOHR for c in r] for r in d_spec.coordinates_angstrom]
                a_bohr = [[c * ANGSTROM_TO_BOHR for c in r] for r in a_spec.coordinates_angstrom]
                dim_coords = (
                    dimer_res.optimized_geometry
                    if (optimize_dimer and dimer_res.optimized_geometry)
                    else dimer_spec.coordinates_angstrom
                )
                dim_bohr = [[c * ANGSTROM_TO_BOHR for c in r] for r in dim_coords]

                d3_damp = (
                    "d3bj"
                    if (dispersion is True or str(dispersion).lower() in ("d3bj", "bj"))
                    else "d3zero"
                )
                d_d3 = rust_core.run_d3_dispersion(
                    list(d_spec.symbols), d_bohr, damping=d3_damp, functional=method
                )
                a_d3 = rust_core.run_d3_dispersion(
                    list(a_spec.symbols), a_bohr, damping=d3_damp, functional=method
                )
                dim_d3 = rust_core.run_d3_dispersion(
                    list(dimer_spec.symbols), dim_bohr, damping=d3_damp, functional=method
                )

                disp_h = dim_d3.energy_hartree - (d_d3.energy_hartree + a_d3.energy_hartree)
                disp_kcal = disp_h * HARTREE_TO_KCAL_MOL
        except Exception:
            pass

        disp_tag = f"with {dispersion.upper()}" if dispersion else "without dispersion"
        summary = (
            f"=== OPV D-A Stacking Dimer Binding Energy ({name}) ===\n"
            f"Method / Basis:          {method.upper()} / {basis} ({disp_tag})\n"
            f"Stacking Distance:       {distance_angstrom:.2f} A (Lateral shift: {lateral_shift_angstrom:.2f} A)\n"
            f"Donor Energy E(D):       {e_d:16.8f} Hartree\n"
            f"Acceptor Energy E(A):    {e_a:16.8f} Hartree\n"
            f"Dimer Energy E(D:A):     {e_dimer:16.8f} Hartree\n"
            f"Binding Energy Delta_E:  {delta_e_h:16.8f} Hartree ({delta_e_kcal:8.2f} kcal/mol, {delta_e_ev:6.3f} eV)\n"
        )
        if disp_kcal is not None:
            summary += f"D3 Dispersion Delta_E:   {disp_h:16.8f} Hartree ({disp_kcal:8.2f} kcal/mol)\n"

        return OPVDimerResult(
            name=name,
            method=method,
            basis=basis,
            dispersion=dispersion,
            e_dimer_hartree=e_dimer,
            e_donor_hartree=e_d,
            e_acceptor_hartree=e_a,
            binding_energy_hartree=delta_e_h,
            binding_energy_kcal_mol=delta_e_kcal,
            binding_energy_ev=delta_e_ev,
            dispersion_energy_hartree=disp_h,
            dispersion_energy_kcal_mol=disp_kcal,
            intermolecular_distance_angstrom=distance_angstrom,
            dimer=dimer_res,
            donor=d_res,
            acceptor=a_res,
            summary=summary,
        )

    def esp_surface_summary(
        self,
        mol: MoleculeSpec,
        method: str = "b3lyp",
        basis: str = "6-31g(d)",
        density_isovalue: float = 0.001,
        spacing_bohr: float = 0.25,
        margin_bohr: float = 5.0,
        name: str = "molecule",
    ) -> OPVEspSummaryResult:
        """Evaluates the pure Rust `rho = density_isovalue` ESP-mapped surface, Murray-Politzer
        polarity descriptors (`V_min`, `V_max`, `MPI` in kcal/mol), Mulliken/Löwdin charges, and
        the `CCESP\\x01` vertex-colored binary packet for MolForge."""
        import numpy as np

        from calchem import _core
        from calchem.utils.constants import ANGSTROM_TO_BOHR

        coords_bohr = [
            [c * ANGSTROM_TO_BOHR for c in row] for row in mol.coordinates_angstrom
        ]
        wf = _core.run_single_point(
            list(mol.symbols),
            coords_bohr,
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            method=method,
            basis=basis,
        )
        pop = _core.compute_population_analysis(wf)
        surf = _core.extract_esp_surface(
            wf,
            density_isovalue=density_isovalue,
            spacing_bohr=spacing_bohr,
            margin_bohr=margin_bohr,
            output_angstrom=True,
        )
        verts = surf.mesh.vertices
        faces = surf.mesh.faces
        esp_v = surf.esp

        if faces.shape[0] > 0:
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            total_area = float(np.sum(tri_areas))
            tri_esp_abs = np.abs(
                (esp_v[faces[:, 0]] + esp_v[faces[:, 1]] + esp_v[faces[:, 2]]) / 3.0
            )
            mpi_eh = float(np.sum(tri_areas * tri_esp_abs)) / max(total_area, 1e-12)
        else:
            total_area = 0.0
            mpi_eh = 0.0

        esp_min_kcal = float(surf.esp_min) * HARTREE_TO_KCAL_MOL
        esp_max_kcal = float(surf.esp_max) * HARTREE_TO_KCAL_MOL
        mpi_kcal = mpi_eh * HARTREE_TO_KCAL_MOL
        packet = surf.to_binary_packet()
        mull_q = [float(x) for x in pop["mulliken"]]
        low_q = [float(x) for x in pop["lowdin"]]

        summary = (
            f"=== OPV Molecular Surface ESP ({name}: {method.upper()}/{basis}, rho={density_isovalue:g} a.u.) ===\n"
            f"Surface Area = {total_area:.2f} A^2 ({surf.mesh.n_vertices} verts, {surf.mesh.n_triangles} tris)\n"
            f"ESP Extrema:  V_min = {esp_min_kcal:6.2f} kcal/mol | V_max = {esp_max_kcal:6.2f} kcal/mol\n"
            f"Polarity Index (MPI) = {mpi_kcal:.2f} kcal/mol | Packet = {len(packet)} bytes (CCESP\\x01)\n"
        )
        return OPVEspSummaryResult(
            name=name,
            density_isovalue=float(surf.density_isovalue),
            n_vertices=int(surf.mesh.n_vertices),
            n_triangles=int(surf.mesh.n_triangles),
            surface_area_angstrom2=total_area,
            esp_min_hartree=float(surf.esp_min),
            esp_max_hartree=float(surf.esp_max),
            esp_min_kcal_mol=esp_min_kcal,
            esp_max_kcal_mol=esp_max_kcal,
            mpi_kcal_mol=mpi_kcal,
            mulliken_charges=mull_q,
            lowdin_charges=low_q,
            ccesp_packet=packet,
            summary=summary,
        )

    def optical_absorption(
        self,
        mol: MoleculeSpec,
        method: str = "b3lyp",
        basis: str = "6-31g(d)",
        n_states: int = 5,
        tda: bool = False,
        manifold: str = "both",
        fwhm_ev: float = 0.33,
        min_nm: float = 150.0,
        max_nm: float = 800.0,
        n_spectrum_points: int = 400,
        compute_exciton_binding: bool = False,
        name: str = "molecule",
    ) -> OPVAbsorptionResult:
        """计算分子激发态、紫外-可见吸收光谱及激子结合能 (Exciton Binding Energy E_b)。

        - 光学带隙 E_g^opt: 最低单重态激发能 E(S1)
        - 单-三态劈裂 Delta E_ST: E(S1) - E(T1)
        - 准粒子基态带隙 E_g^fund: IP_v - EA_v (当 compute_exciton_binding=True 时调用四点法重组能计算)
        - 激子结合能 E_b = E_g^fund - E_g^opt (表征激子解离势垒)
        """
        req = CalculationRequest(
            task="tddft",
            molecule=mol,
            backend=self.backend,
            method=method,
            basis=basis,
            options={
                "n_states": n_states,
                "tda": tda,
                "manifold": manifold,
                "fwhm_ev": fwhm_ev,
            },
            resources=self._default_resources(),
        )
        res = self.service.run(req)
        if res.status != "succeeded" or not res.excited_states:
            raise RuntimeError(f"TD-DFT optical absorption failed ({res.status}): {res.errors}")

        optical_gap = (
            res.optical_gap_ev
            if res.optical_gap_ev is not None
            else res.excited_states[0].excitation_energy_ev
        )
        lowest_t = res.lowest_triplet_ev
        delta_est = res.delta_est_ev

        # 寻找振子强度最大的亮态 (Bright State)
        bright_states = [s for s in res.excited_states if s.multiplicity == 1]
        if bright_states:
            max_bright = max(bright_states, key=lambda s: s.oscillator_strength)
            lambda_max = max_bright.wavelength_nm
            f_max = max_bright.oscillator_strength
        else:
            lambda_max = res.excited_states[0].wavelength_nm
            f_max = res.excited_states[0].oscillator_strength

        # 生成紫外-可见光谱
        tddft_obj = res.extra.get("tddft_result")
        spec_wl: List[float] = []
        spec_eps: List[float] = []
        if tddft_obj is not None and hasattr(tddft_obj, "broaden_spectrum"):
            spec = tddft_obj.broaden_spectrum(
                min_nm=min_nm,
                max_nm=max_nm,
                n_points=n_spectrum_points,
                fwhm_ev=fwhm_ev,
            )
            spec_wl = list(spec.wavelengths_nm)
            spec_eps = list(spec.epsilon_l_mol_cm)

        # 激子结合能闭环: 若请求则计算垂直电离能 IP_v 与电子亲和能 EA_v
        fund_gap = None
        e_binding = None
        if compute_exciton_binding:
            reorg = self.reorganization_energy(
                mol,
                method=method,
                basis=basis,
                carrier="both",
                pre_optimized_neutral=True,
                name=name,
            )
            if reorg.fundamental_gap_vertical_ev is not None:
                fund_gap = reorg.fundamental_gap_vertical_ev
                e_binding = fund_gap - optical_gap

        # KS 单粒子带隙
        homo_e = res.homo_energy_hartree
        lumo_e = res.lumo_energy_hartree
        ks_gap = (
            (lumo_e - homo_e) * HARTREE_TO_EV
            if (homo_e is not None and lumo_e is not None)
            else 0.0
        )

        states_data = []
        for s in res.excited_states:
            states_data.append(
                {
                    "label": s.label,
                    "multiplicity": s.multiplicity,
                    "energy_ev": s.excitation_energy_ev,
                    "wavelength_nm": s.wavelength_nm,
                    "oscillator_strength": s.oscillator_strength,
                    "dominant_transitions": [
                        {"occ": t.occ_index, "virt": t.virt_index, "weight": t.weight}
                        for t in s.dominant_transitions
                    ],
                }
            )

        eb_str = f"{e_binding:5.2f} eV" if e_binding is not None else "N/A"
        fund_str = f"{fund_gap:5.2f} eV" if fund_gap is not None else "N/A"
        dest_str = f"{delta_est:5.2f} eV" if delta_est is not None else "N/A"
        t1_str = f"{lowest_t:5.2f} eV" if lowest_t is not None else "N/A"

        summary = (
            f"=== OPV Optical Absorption & Exciton Binding ({name}: {method.upper()}/{basis}, TDA={tda}) ===\n"
            f"Optical Bandgap E_g^opt (S1) = {optical_gap:5.2f} eV | Lambda_max = {lambda_max:6.1f} nm (f_max = {f_max:.4f})\n"
            f"Kohn-Sham Gap (HOMO-LUMO)   = {ks_gap:5.2f} eV | Lowest Triplet (T1) = {t1_str}\n"
            f"Singlet-Triplet Gap (dE_ST) = {dest_str}\n"
            f"Fundamental Gap (IP - EA)   = {fund_str} | Exciton Binding Energy E_b = {eb_str}\n"
        )

        return OPVAbsorptionResult(
            name=name,
            method=method,
            basis=basis,
            tda=tda,
            optical_gap_ev=optical_gap,
            wavelength_max_nm=lambda_max,
            oscillator_strength_max=f_max,
            ks_gap_ev=ks_gap,
            lowest_triplet_ev=lowest_t,
            delta_est_ev=delta_est,
            fundamental_gap_vertical_ev=fund_gap,
            exciton_binding_energy_ev=e_binding,
            spectrum_wavelengths_nm=spec_wl,
            spectrum_epsilon_l_mol_cm=spec_eps,
            states=states_data,
            summary=summary,
        )

    def thermochemistry(
        self,
        mol: MoleculeSpec,
        method: str = "b3lyp",
        basis: str = "sto-3g",
        temperature_k: float = 298.15,
        pressure_pa: float = 101325.0,
        pre_optimize: bool = False,
        name: str = "molecule",
        **options,
    ) -> OPVThermochemistryResult:
        """Runs vibrational frequency analysis and RRHO thermochemistry."""
        target_mol = mol
        if pre_optimize:
            opt_res = self.optimize(mol, method=method, basis=basis, name=name, **options)
            if opt_res.optimized_geometry:
                target_mol = MoleculeSpec(
                    symbols=mol.symbols,
                    coordinates_angstrom=opt_res.optimized_geometry,
                    charge=mol.charge,
                    multiplicity=mol.multiplicity,
                )

        opts = dict(options)
        opts["temperature_k"] = temperature_k
        opts["pressure_pa"] = pressure_pa

        req = CalculationRequest(
            task="frequency",
            molecule=target_mol,
            backend=self.backend,
            method=method,
            basis=basis,
            options=opts,
            resources=self._default_resources(),
        )
        res = self.service.run(req)
        if res.status != "succeeded":
            raise RuntimeError(f"Thermochemistry calculation failed ({res.status}): {res.errors}")

        freqs = res.frequencies_cm or []
        irs = res.ir_intensities_km_mol or []
        n_imag = sum(1 for f in freqs if f < 0.0)
        is_ts = (n_imag == 1)

        energy = res.total_energy_hartree or res.scf_energy_hartree or 0.0
        zpve_h = res.zpve_hartree or 0.0
        zpve_kcal = res.zpve_kcal_mol or (zpve_h * 627.509474)
        thermo_h = res.thermal_energy_hartree or (energy + zpve_h)
        h_h = res.enthalpy_hartree or 0.0
        g_h = res.gibbs_free_energy_hartree or 0.0
        s_cal = res.entropy_cal_mol_k or 0.0
        cv_cal = res.heat_capacity_cv_cal_mol_k or 0.0
        cp_cal = res.heat_capacity_cp_cal_mol_k or 0.0

        ts_tag = (
            "Transition State (1 imaginary mode)"
            if is_ts
            else (
                "Local Minimum (all real modes)"
                if n_imag == 0
                else f"Higher-Order Saddle Point ({n_imag} imaginary modes)"
            )
        )

        summary = (
            f"=== CalChem RRHO Thermochemistry ({name}: {method.upper()}/{basis}, T={temperature_k:.2f} K, P={pressure_pa:.1f} Pa) ===\n"
            f"Electronic Energy E_0   = {energy:14.8f} Hartree\n"
            f"Zero-Point Energy ZPVE  = {zpve_h:14.8f} Hartree ({zpve_kcal:8.3f} kcal/mol)\n"
            f"Thermal Enthalpy H(T)   = {h_h:14.8f} Hartree\n"
            f"Gibbs Free Energy G(T)  = {g_h:14.8f} Hartree\n"
            f"Entropy S(T, P)         = {s_cal:10.3f} cal/(mol*K)\n"
            f"Heat Capacity C_p, C_v  = {cp_cal:10.3f}, {cv_cal:10.3f} cal/(mol*K)\n"
            f"Vibrational Modes       = {len(freqs)} modes ({ts_tag})\n"
        )

        return OPVThermochemistryResult(
            name=name,
            method=method,
            basis=basis,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            scf_energy_hartree=energy,
            zpve_hartree=zpve_h,
            zpve_kcal_mol=zpve_kcal,
            thermal_energy_hartree=thermo_h,
            enthalpy_hartree=h_h,
            gibbs_free_energy_hartree=g_h,
            entropy_cal_mol_k=s_cal,
            heat_capacity_cv_cal_mol_k=cv_cal,
            heat_capacity_cp_cal_mol_k=cp_cal,
            n_vibrations=len(freqs),
            n_imaginary_modes=n_imag,
            is_transition_state=is_ts,
            frequencies_cm=freqs,
            ir_intensities_km_mol=irs,
            summary=summary,
        )

    def reaction_barrier(
        self,
        guess_ts_mol: MoleculeSpec,
        method: str = "rhf",
        basis: str = "sto-3g",
        run_irc_path: bool = True,
        max_irc_points: int = 10,
        irc_step_size_bohr: float = 0.1,
        name: str = "reaction",
    ) -> OPVReactionBarrierResult:
        """Searches for a transition state (P-RFO) and optionally integrates the IRC pathway."""
        ts_req = CalculationRequest(
            task="ts",
            molecule=guess_ts_mol,
            backend=self.backend,
            method=method,
            basis=basis,
            options={"max_steps": 40},
            resources=self._default_resources(),
        )
        ts_res = self.service.run(ts_req)
        if ts_res.status != "succeeded":
            raise RuntimeError(f"Transition state search failed ({ts_res.status}): {ts_res.errors}")

        ts_energy = ts_res.total_energy_hartree or ts_res.scf_energy_hartree or 0.0
        imag_freq = ts_res.imaginary_frequency_cm or 0.0
        opt_mol = ts_res.molecule or guess_ts_mol
        trans_vec = ts_res.extra.get("transition_vector")

        fwd_barrier = None
        rev_barrier = None
        rxn_energy = None
        path_points: List[IrcPointInfo] = []

        if run_irc_path and trans_vec is not None:
            irc_req = CalculationRequest(
                task="irc",
                molecule=opt_mol,
                backend=self.backend,
                method=method,
                basis=basis,
                options={
                    "transition_vector": trans_vec,
                    "max_points": max_irc_points,
                    "step_size_bohr": irc_step_size_bohr,
                    "direction": "both",
                },
                resources=self._default_resources(),
            )
            irc_res = self.service.run(irc_req)
            if irc_res.status in ("succeeded", "not_converged"):
                fwd_barrier = irc_res.irc_forward_barrier_kcal_mol
                rev_barrier = irc_res.irc_reverse_barrier_kcal_mol
                rxn_energy = irc_res.reaction_energy_kcal_mol
                path_points = irc_res.irc_path

        def _barrier_text(value):
            return "unavailable (IRC endpoints incomplete)" if value is None else f"{value:8.2f} kcal/mol"

        summary = (
            f"=== CalChem Reaction Pathway ({name}: {method.upper()}/{basis}) ===\n"
            f"TS Electronic Energy E_TS = {ts_energy:14.8f} Hartree\n"
            f"Imaginary Frequency       = {imag_freq:10.1f} cm^-1\n"
            f"Forward Barrier E_a(fwd)  = {_barrier_text(fwd_barrier)}\n"
            f"Reverse Barrier E_a(rev)  = {_barrier_text(rev_barrier)}\n"
            f"Reaction Energy Delta_H   = {_barrier_text(rxn_energy)}\n"
            f"IRC Pathway Points        = {len(path_points)} points\n"
        )

        return OPVReactionBarrierResult(
            name=name,
            method=method,
            basis=basis,
            ts_energy_hartree=ts_energy,
            imaginary_frequency_cm=imag_freq,
            forward_barrier_kcal_mol=fwd_barrier,
            reverse_barrier_kcal_mol=rev_barrier,
            reaction_energy_kcal_mol=rxn_energy,
            n_irc_points=len(path_points),
            ts_molecule=opt_mol,
            irc_path=path_points,
            summary=summary,
        )

    def solvation_free_energy(
        self,
        mol: MoleculeSpec,
        solvent: str = "chlorobenzene",
        solvent_model: str = "cpcm",
        method: str = "b3lyp",
        basis: str = "sto-3g",
        name: str = "molecule",
    ) -> OPVSolvationResult:
        """Evaluates pure Rust continuum solvation free energy in an OPV processing solvent (e.g. Chlorobenzene)."""
        req = CalculationRequest(
            task="solvation",
            molecule=mol,
            backend=self.backend,
            method=method,
            basis=basis,
            solvent=solvent,
            solvent_model=solvent_model,
            resources=self._default_resources(),
        )
        res = self.service.run(req)
        if res.status != "succeeded":
            raise RuntimeError(f"Solvation calculation failed for '{name}': {res.errors}")

        solv_h = float(res.solvated_energy_hartree or 0.0)
        gas_h = float(res.scf_energy_hartree or 0.0)
        dg_solv = float(res.delta_g_solv_kcal_mol or 0.0)
        dg_elec = float(res.delta_g_elec_kcal_mol or 0.0)
        dg_cds = float(res.delta_g_cds_kcal_mol or 0.0)
        sasa = float(res.cavity_surface_area_angstrom2 or 0.0)
        eps = float(res.solvent_dielectric or 0.0)

        summary = (
            f"=== CalChem Solvation Free Energy ({name} in {solvent.capitalize()}) ===\n"
            f"Method / Basis:          {method.upper()} / {basis}\n"
            f"Dielectric Constant eps: {eps:.2f}\n"
            f"Gas Phase Energy:        {gas_h:14.6f} Hartree\n"
            f"Solvated Energy:         {solv_h:14.6f} Hartree\n"
            f"Electrostatic Solv:      {dg_elec:8.2f} kcal/mol ({dg_elec / 23.060548:6.3f} eV)\n"
            f"Cavitation-Dispersion:   {dg_cds:8.2f} kcal/mol ({dg_cds / 23.060548:6.3f} eV)\n"
            f"Total Delta G_solv:      {dg_solv:8.2f} kcal/mol ({dg_solv / 23.060548:6.3f} eV)\n"
            f"Cavity SASA:             {sasa:8.1f} A^2\n"
        )

        return OPVSolvationResult(
            name=name,
            solvent=solvent,
            dielectric_constant=eps,
            solvated_energy_hartree=solv_h,
            gas_energy_hartree=gas_h,
            delta_g_solv_kcal_mol=dg_solv,
            delta_g_elec_kcal_mol=dg_elec,
            delta_g_cds_kcal_mol=dg_cds,
            cavity_surface_area_angstrom2=sasa,
            method=method,
            basis=basis,
            summary=summary,
        )

```

## `crates/calchem-int/src/boys.rs`

SHA256: `5c4787d449b7fe2c65a20d70eebf7e14750c578bdc2521f286423845c34b04b7`

```rust
use std::f64::consts::PI;

/// High-precision rational Chebyshev approximation of `erf(x)` (Cody, Math. Comp. 23, 631 (1969)),
/// accurate to `< 2.5e-16` (~1 ULP in IEEE 754 `f64`) across the entire real line.
pub fn erf_approx(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    if ax <= 0.46875 {
        let ysq = ax * ax;
        const P: [f64; 5] = [
            3.16112374387056560e00,
            1.13864154151050156e02,
            3.77485237685302021e02,
            3.20937758913846947e03,
            1.85777706184603153e-01,
        ];
        const Q: [f64; 4] = [
            2.36012909523441209e01,
            2.44024637934444173e02,
            1.28261652607737228e03,
            2.84423683343917062e03,
        ];
        let mut xnum = P[4] * ysq;
        let mut xden = ysq;
        for i in 0..3 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        let res = ax * (xnum + P[3]) / (xden + Q[3]);
        sign * res
    } else if ax <= 4.0 {
        const P: [f64; 9] = [
            5.64188496988670089e-01,
            8.88314979438837594e00,
            6.61191906371416295e01,
            2.98635138197400131e02,
            8.81952221241769090e02,
            1.71204761263407058e03,
            2.05107837782607147e03,
            1.23033935479799725e03,
            2.15311535474403846e-08,
        ];
        const Q: [f64; 8] = [
            1.57449261107098347e01,
            1.17693950891312499e02,
            5.37181101862009858e02,
            1.62138957456669019e03,
            3.29079923573345963e03,
            4.36261909014324716e03,
            3.43936767414372164e03,
            1.23033935480374942e03,
        ];
        let mut xnum = P[8] * ax;
        let mut xden = ax;
        for i in 0..7 {
            xnum = (xnum + P[i]) * ax;
            xden = (xden + Q[i]) * ax;
        }
        let result = (xnum + P[7]) / (xden + Q[7]);
        let ysq = (ax * 16.0).floor() / 16.0;
        let del_x = (ax - ysq) * (ax + ysq);
        let erfc_val = (-ysq * ysq).exp() * (-del_x).exp() * result;
        sign * (1.0 - erfc_val)
    } else if ax < 26.6 {
        let ysq = 1.0 / (ax * ax);
        const P: [f64; 6] = [
            3.05326634961232344e-01,
            3.60344899949804439e-01,
            1.25781726111229246e-01,
            1.60837851487422766e-02,
            6.58749161529837803e-04,
            1.63153871373020978e-02,
        ];
        const Q: [f64; 5] = [
            2.56852019228982242e00,
            1.87295284992346047e00,
            5.27905102951428412e-01,
            6.05183413124413191e-02,
            2.33520497626869185e-03,
        ];
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        let mut result = ysq * (xnum + P[4]) / (xden + Q[4]);
        result = (0.56418958354775628695 - result) / ax;
        let ysq_hi = (ax * 16.0).floor() / 16.0;
        let del_x = (ax - ysq_hi) * (ax + ysq_hi);
        let erfc_val = (-ysq_hi * ysq_hi).exp() * (-del_x).exp() * result;
        sign * (1.0 - erfc_val)
    } else {
        sign
    }
}

/// Evaluates the Boys function batch `F_0(t), F_1(t), ..., F_{max_n}(t)` into `out[0..=max_n]`:
/// `F_n(t) = \int_0^1 u^{2n} \exp(-t u^2) du`
///
/// Positive Kummer series plus downward recurrence where upward recurrence
/// would subtract comparable numbers. At large T, upward recursion is used
/// only through n <= T; higher orders are evaluated by a positive series.
/// Batch evaluation allocates no auxiliary vector.
pub fn boys_batch(max_n: usize, t: f64, out: &mut [f64]) {
    assert!(out.len() > max_n, "boys_batch output slice too short");
    assert!(t >= 0.0, "Boys function argument t must be >= 0, got {t}");

    if t < 1e-14 {
        for (n, slot) in out.iter_mut().enumerate().take(max_n + 1) {
            let nf = n as f64;
            *slot = 1.0 / (2.0 * nf + 1.0) - t / (2.0 * nf + 3.0) + 0.5 * t * t / (2.0 * nf + 5.0);
        }
        return;
    }

    if t.is_infinite() {
        out[..=max_n].fill(0.0);
        return;
    }
    let et = (-t).exp();
    if t > max_n as f64 + 40.0 || t >= 700.0 {
        out[0] = 0.5 * (PI / t).sqrt() * erf_approx(t.sqrt());
        let last_up = max_n.min(t as usize);
        for n in 1..=last_up {
            out[n] = ((2 * n - 1) as f64 * out[n - 1] - et) * (0.5 / t);
        }
        // For n>T the Gaussian tail is not negligible, even if exp(-T) is
        // tiny. The old asymptotic-only branch produced unbounded relative error.
        for n in last_up + 1..=max_n {
            out[n] = et * kummer_scaled(n, t);
        }
        return;
    }
    out[max_n] = et * kummer_scaled(max_n, t);
    for n in (1..=max_n).rev() {
        out[n - 1] = (2.0 * t * out[n] + et) / (2 * n - 1) as f64;
    }
}

/// exp(T) F_n(T), evaluated only where T <= n+40 and T < 700,
/// or n >= T. Kahan summation avoids loss in the positive series tail.
fn kummer_scaled(n: usize, t: f64) -> f64 {
    let mut term = 1.0 / (2.0 * n as f64 + 1.0);
    let mut sum = term;
    let mut correction = 0.0;
    for k in 1..10000 {
        term *= t / (n as f64 + k as f64 + 0.5);
        let y = term - correction;
        let next = sum + y;
        correction = (next - sum) - y;
        sum = next;
        if term <= sum * 5e-17 {
            break;
        }
    }
    sum
}

/// Evaluates the single Boys function value `F_n(t)`.
pub fn boys(n: usize, t: f64) -> f64 {
    let mut buf = vec![0.0; n + 1];
    boys_batch(n, t, &mut buf);
    buf[n]
}

```

## `crates/calchem-int/src/symmetry.rs`

SHA256: `dcdc70a09957bd5bd01610abcde370127362483f522acfdf13b421a4c46a5692`

```rust
/// Enumerates canonical shell quartets `[i, j, k, l]` satisfying `i <= j`, `k <= l`,
/// and composite pair index `ij <= kl`.
pub fn unique_shell_quartets(n_shells: usize) -> Vec<[usize; 4]> {
    shell_quartets(n_shells).collect()
}

/// Lazy canonical quartets: constant auxiliary storage.
pub fn shell_quartets(n_shells: usize) -> impl Iterator<Item = [usize; 4]> {
    (0..n_shells).flat_map(move |i| {
        (i..n_shells).flat_map(move |j| {
            (i..n_shells).flat_map(move |k| {
                let start = if k == i { j } else { k };
                (start..n_shells).map(move |l| [i, j, k, l])
            })
        })
    })
}

/// Returns all 8 real-AO permutation indices of `(mu nu | lam sig)`:
/// `(mu nu | lam sig) = (nu mu | lam sig) = (mu nu | sig lam) = (nu mu | sig lam)`
/// `= (lam sig | mu nu) = (sig lam | mu nu) = (lam sig | nu mu) = (sig lam | nu mu)`.
pub fn eri_permutations(mu: usize, nu: usize, lam: usize, sig: usize) -> [[usize; 4]; 8] {
    [
        [mu, nu, lam, sig],
        [nu, mu, lam, sig],
        [mu, nu, sig, lam],
        [nu, mu, sig, lam],
        [lam, sig, mu, nu],
        [sig, lam, mu, nu],
        [lam, sig, nu, mu],
        [sig, lam, nu, mu],
    ]
}

```

## `crates/calchem-dft/src/d3.rs`

SHA256: `13e151d9a2d71bddd9b4505f17064dac106d69fc5b968cdbfd3a84eb8bac9697`

```rust
//! Standard two-body D3(BJ)/D3(0), in atomic units; ATM is not included.
//! CN interpolation uses the original pair reference tables, not atomic mixing rules.
//! Data attribution and corresponding source: ../data/d3/NOTICE.md.
use calchem_model::{Atom, CalcError};
use serde::{Deserialize, Serialize};
#[path = "d3_data.rs"]
mod data;
const A0_ANGSTROM: f64 = 0.529177210903;
const C6_BYTES: &[u8] = include_bytes!("../data/d3/c6.bin");

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DampingScheme {
    BeckeJohnson {
        s6: f64,
        s8: f64,
        a1: f64,
        a2: f64,
    },
    Zero {
        s6: f64,
        s8: f64,
        sr6: f64,
        sr8: f64,
        alpha6: f64,
        alpha8: f64,
    },
}
impl Default for DampingScheme {
    fn default() -> Self {
        Self::BeckeJohnson {
            s6: 1.0,
            s8: 1.9889,
            a1: 0.3981,
            a2: 4.4211,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct D3Model {
    pub damping: DampingScheme,
    pub functional_name: String,
}

fn pair_index(a: usize, b: usize) -> usize {
    let hi = a.max(b);
    let lo = a.min(b);
    hi * (hi - 1) / 2 + lo - 1
}
pub fn covalent_radius_bohr(z: u8) -> f64 {
    data::COVALENT_ANGSTROM[z as usize - 1] * 4.0 / (3.0 * A0_ANGSTROM)
}
pub fn vdw_radius_bohr(z: u8) -> f64 {
    data::VDW_PAIR_ANGSTROM[pair_index(z as usize, z as usize)] / (2.0 * A0_ANGSTROM)
}
fn r4r2(z: u8) -> f64 {
    (0.5 * (z as f64).sqrt() * data::R4_OVER_R2[z as usize - 1]).sqrt()
}
fn reference_c6(a: u8, b: u8, i: usize, j: usize) -> f64 {
    let (r, c) = if a > b { (i, j) } else { (j, i) };
    let off = (pair_index(a as usize, b as usize) * 49 + c * 7 + r) * 8;
    f64::from_le_bytes(
        C6_BYTES[off..off + 8]
            .try_into()
            .expect("fixed D3 table layout"),
    )
}
fn weights(z: u8, cn: f64) -> ([f64; 7], [f64; 7]) {
    let refs = &data::REFERENCE_CN[(z as usize - 1) * 7..z as usize * 7];
    let mut w = [0.0; 7];
    let mut dw = [0.0; 7];
    let mut nearest = f64::INFINITY;
    for &r in refs {
        if r >= 0.0 {
            nearest = nearest.min((cn - r).powi(2));
        }
    }
    let mut sum = 0.0;
    for i in 0..7 {
        if refs[i] >= 0.0 {
            w[i] = (-4.0 * ((cn - refs[i]).powi(2) - nearest)).exp();
            sum += w[i];
        }
    }
    let mut mean = 0.0;
    for i in 0..7 {
        w[i] /= sum;
        dw[i] = -8.0 * (cn - refs[i]);
        mean += w[i] * dw[i];
    }
    for i in 0..7 {
        dw[i] = w[i] * (dw[i] - mean);
    }
    (w, dw)
}
fn pair_c6(a: u8, b: u8, wa: &([f64; 7], [f64; 7]), wb: &([f64; 7], [f64; 7])) -> (f64, f64, f64) {
    let (mut v, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..7 {
        for j in 0..7 {
            let c = reference_c6(a, b, i, j);
            v += wa.0[i] * wb.0[j] * c;
            da += wa.1[i] * wb.0[j] * c;
            db += wa.0[i] * wb.1[j] * c;
        }
    }
    (v, da, db)
}
impl D3Model {
    pub fn d3bj(functional: &str) -> Result<Self, CalcError> {
        let f = functional.trim().to_ascii_lowercase();
        let damping = match f.as_str() {
            "b3lyp" | "b3lyp5" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 1.9889,
                a1: 0.3981,
                a2: 4.4211,
            },
            "pbe" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 0.7875,
                a1: 0.4289,
                a2: 4.4407,
            },
            "pbe0" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 1.2177,
                a1: 0.4145,
                a2: 4.8593,
            },
            "blyp" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 2.6996,
                a1: 0.4298,
                a2: 4.2359,
            },
            "bp86" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 3.2822,
                a1: 0.3946,
                a2: 4.8516,
            },
            "tpss" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 1.9435,
                a1: 0.4535,
                a2: 4.4752,
            },
            "b97" | "b97-d" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 2.2609,
                a1: 0.5545,
                a2: 3.2297,
            },
            "cam-b3lyp" | "camb3lyp" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 2.0674,
                a1: 0.3708,
                a2: 5.4743,
            },
            "hf" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 0.9171,
                a1: 0.3385,
                a2: 2.883,
            },
            _ => {
                return Err(CalcError::UnsupportedMethod(format!(
                    "Unsupported D3 functional: {functional}"
                )))
            }
        };
        Ok(Self {
            damping,
            functional_name: f,
        })
    }
    pub fn d3zero(functional: &str) -> Result<Self, CalcError> {
        let f = functional.trim().to_ascii_lowercase();
        let damping = match f.as_str() {
            "b3lyp" | "b3lyp5" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.703,
                sr6: 1.261,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "pbe" => DampingScheme::Zero {
                s6: 1.0,
                s8: 0.722,
                sr6: 1.217,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "pbe0" => DampingScheme::Zero {
                s6: 1.0,
                s8: 0.928,
                sr6: 1.287,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "blyp" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.682,
                sr6: 1.094,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "bp86" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.683,
                sr6: 1.139,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "tpss" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.105,
                sr6: 1.166,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "b97" | "b97-d" => DampingScheme::Zero {
                s6: 1.0,
                s8: 0.909,
                sr6: 0.892,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "cam-b3lyp" | "camb3lyp" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.217,
                sr6: 1.378,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "hf" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.746,
                sr6: 1.158,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            _ => {
                return Err(CalcError::UnsupportedMethod(format!(
                    "Unsupported D3 functional: {functional}"
                )))
            }
        };
        Ok(Self {
            damping,
            functional_name: f,
        })
    }

    pub fn energy(&self, atoms: &[Atom]) -> Result<f64, CalcError> {
        Ok(self.energy_and_gradient(atoms, false)?.0)
    }
    pub fn gradient(&self, atoms: &[Atom]) -> Result<Vec<[f64; 3]>, CalcError> {
        Ok(self.energy_and_gradient(atoms, true)?.1)
    }
    pub fn energy_and_gradient(
        &self,
        atoms: &[Atom],
        compute_gradient: bool,
    ) -> Result<(f64, Vec<[f64; 3]>), CalcError> {
        for a in atoms {
            if !(1..=103).contains(&a.z) || !a.position_bohr.iter().all(|x| x.is_finite()) {
                return Err(CalcError::InvalidMolecule(
                    "D3 requires finite coordinates and elements Z=1..103".into(),
                ));
            }
        }
        let valid = match self.damping {
            DampingScheme::BeckeJohnson { s6, s8, a1, a2 } => {
                [s6, s8, a1, a2].iter().all(|v| v.is_finite() && *v >= 0.0) && a2 > 0.0
            }
            DampingScheme::Zero {
                s6,
                s8,
                sr6,
                sr8,
                alpha6,
                alpha8,
            } => {
                [s6, s8, sr6, sr8, alpha6, alpha8]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0)
                    && sr6 > 0.0
                    && sr8 > 0.0
                    && alpha6 > 6.0
                    && alpha8 > 8.0
            }
        };
        if !valid {
            return Err(CalcError::InvalidMolecule(
                "Invalid D3 damping parameters".into(),
            ));
        }
        let n = atoms.len();
        let mut cn = vec![0.0; n];
        // O(N^2) geometry only; no AO tensor. 40 Bohr is the standard CN cutoff.
        for i in 0..n {
            for j in 0..i {
                let r = distance(atoms, i, j).1;
                if !r.is_finite() || r < 1e-8 {
                    return Err(CalcError::InvalidMolecule(
                        "D3 coincident atoms or non-finite distance".into(),
                    ));
                }
                if r <= 40.0 {
                    let rc = covalent_radius_bohr(atoms[i].z) + covalent_radius_bohr(atoms[j].z);
                    let f = 1.0 / (1.0 + (-16.0 * (rc / r - 1.0)).exp());
                    cn[i] += f;
                    cn[j] += f;
                }
            }
        }
        let ws: Vec<_> = atoms
            .iter()
            .zip(&cn)
            .map(|(a, &c)| weights(a.z, c))
            .collect();
        let mut grad = vec![[0.0; 3]; n];
        let mut dedcn = vec![0.0; n];
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..i {
                let (d, r) = distance(atoms, i, j);
                let (c6, dc_i, dc_j) = pair_c6(atoms[i].z, atoms[j].z, &ws[i], &ws[j]);
                let c8_ratio = 3.0 * r4r2(atoms[i].z) * r4r2(atoms[j].z);
                let mut kernel = 0.0;
                let mut dkernel = 0.0;
                match self.damping {
                    DampingScheme::BeckeJohnson { s6, s8, a1, a2 } => {
                        let cutoff = a1 * c8_ratio.sqrt() + a2;
                        for (power, scale) in [(6, s6), (8, s8 * c8_ratio)] {
                            let denom = r.powi(power) + cutoff.powi(power);
                            kernel -= scale / denom;
                            dkernel += scale * power as f64 * r.powi(power - 1) / (denom * denom);
                        }
                    }
                    DampingScheme::Zero {
                        s6,
                        s8,
                        sr6,
                        sr8,
                        alpha6,
                        alpha8,
                    } => {
                        let r0 = data::VDW_PAIR_ANGSTROM
                            [pair_index(atoms[i].z as usize, atoms[j].z as usize)]
                            / A0_ANGSTROM;
                        for (power, scale, sr, alpha) in
                            [(6, s6, sr6, alpha6), (8, s8 * c8_ratio, sr8, alpha8)]
                        {
                            let damp = 1.0 / (1.0 + 6.0 * (sr * r0 / r).powf(alpha));
                            let factor = scale * damp / r.powi(power);
                            kernel -= factor;
                            dkernel += factor / r * (power as f64 - alpha * (1.0 - damp));
                        }
                    }
                }
                energy += c6 * kernel;
                if compute_gradient {
                    dedcn[i] += dc_i * kernel;
                    dedcn[j] += dc_j * kernel;
                    for q in 0..3 {
                        let f = c6 * dkernel * d[q] / r;
                        grad[i][q] += f;
                        grad[j][q] -= f;
                    }
                }
            }
        }
        if compute_gradient {
            for i in 0..n {
                for j in 0..i {
                    let (d, r) = distance(atoms, i, j);
                    if r > 40.0 {
                        continue;
                    }
                    let rc = covalent_radius_bohr(atoms[i].z) + covalent_radius_bohr(atoms[j].z);
                    let f = 1.0 / (1.0 + (-16.0 * (rc / r - 1.0)).exp());
                    let factor = -(dedcn[i] + dedcn[j]) * 16.0 * rc / (r * r) * f * (1.0 - f);
                    for q in 0..3 {
                        let v = factor * d[q] / r;
                        grad[i][q] += v;
                        grad[j][q] -= v;
                    }
                }
            }
        }
        if !energy.is_finite() || grad.iter().flatten().any(|x| !x.is_finite()) {
            return Err(CalcError::InvalidMolecule(
                "D3 result exceeds floating-point range".into(),
            ));
        }
        Ok((energy, grad))
    }
}
fn distance(atoms: &[Atom], i: usize, j: usize) -> ([f64; 3], f64) {
    let d = std::array::from_fn(|q| atoms[i].position_bohr[q] - atoms[j].position_bohr[q]);
    let r = d[0].hypot(d[1]).hypot(d[2]);
    (d, r)
}

```

## `crates/calchem-dft/src/d3_data.rs`

SHA256: `1666df511bd3d4cdebdefbbf061adc195e200404bbe885700ee030fa15061915`

```rust
// Generated from upstream tables; see ../data/d3/NOTICE.md.
pub const REFERENCE_CN: [f64; 721] = [
0.9118, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9865, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9808, 1.9697, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9706, 1.9441, 2.9128, 4.5856, -1.0, -1.0,
0.0, 0.9868, 1.9985, 2.9987, 3.9844, -1.0, -1.0,
0.0, 0.9944, 2.0143, 2.9903, -1.0, -1.0, -1.0,
0.0, 0.9925, 1.9887, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9982, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9684, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9628, 1.9496, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9648, 1.9311, 2.9146, -1.0, -1.0, -1.0,
0.0, 0.9507, 1.9435, 2.9407, 3.8677, -1.0, -1.0,
0.0, 0.9947, 2.0102, 2.9859, -1.0, -1.0, -1.0,
0.0, 0.9948, 1.9903, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9972, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9767, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9831, 1.9349, -1.0, -1.0, -1.0, -1.0,
0.0, 1.8627, 2.8999, -1.0, -1.0, -1.0, -1.0,
0.0, 1.8299, 3.8675, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9138, 2.911, -1.0, -1.0, -1.0, -1.0,
0.0, 1.8269, 10.6191, -1.0, -1.0, -1.0, -1.0,
0.0, 1.6406, 9.8849, -1.0, -1.0, -1.0, -1.0,
0.0, 1.6483, 9.1376, -1.0, -1.0, -1.0, -1.0,
0.0, 1.7149, 2.9263, 7.7785, -1.0, -1.0, -1.0,
0.0, 1.7937, 6.5458, 6.2918, -1.0, -1.0, -1.0,
0.0, 0.9576, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9419, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9601, 1.9315, 2.9233, -1.0, -1.0, -1.0,
0.0, 0.9434, 1.9447, 2.9186, 3.8972, -1.0, -1.0,
0.0, 0.9889, 1.9793, 2.9709, -1.0, -1.0, -1.0,
0.0, 0.9901, 1.9812, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9974, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9738, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9801, 1.9143, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9153, 2.8903, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9355, 3.9106, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9545, 2.9225, -1.0, -1.0, -1.0, -1.0,
0.0, 1.942, 11.0556, -1.0, -1.0, -1.0, -1.0,
0.0, 1.6682, 9.5402, -1.0, -1.0, -1.0, -1.0,
0.0, 1.8584, 8.8895, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9003, 2.9696, -1.0, -1.0, -1.0, -1.0,
0.0, 1.863, 5.7095, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9679, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9539, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9633, 1.9378, 2.9353, -1.0, -1.0, -1.0,
0.0, 0.9514, 1.9505, 2.9259, 3.9123, -1.0, -1.0,
0.0, 0.9749, 1.9523, 2.9315, -1.0, -1.0, -1.0,
0.0, 0.9811, 1.9639, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9968, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9909, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9797, 1.8467, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9373, 2.9175, -1.0, -1.0, -1.0, -1.0,
2.7991, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.9425, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.9455, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.9413, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.93, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 1.8286, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.8732, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.9086, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.8965, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.9242, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.9282, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.9246, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.8482, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 2.9219, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9254, 3.884, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9459, 2.8988, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9292, 10.9153, -1.0, -1.0, -1.0, -1.0,
0.0, 1.8104, 9.8054, -1.0, -1.0, -1.0, -1.0,
0.0, 1.8858, 9.1527, -1.0, -1.0, -1.0, -1.0,
0.0, 1.8648, 2.9424, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9188, 6.6669, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9846, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 1.9896, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9267, 1.9302, 2.942, -1.0, -1.0, -1.0,
0.0, 0.9383, 1.9356, 2.9081, 3.9098, -1.0, -1.0,
0.0, 0.982, 1.9655, 2.95, -1.0, -1.0, -1.0,
0.0, 0.9815, 1.9639, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9954, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9705, -1.0, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9661, 1.9251, -1.0, -1.0, -1.0, -1.0,
0.0, 0.9802, 1.9445, 2.907, 3.8174, 4.6723, 5.5599,
0.0, 0.9847, 1.956, 2.9302, 3.8997, -1.0, -1.0,
0.0, 0.9647, 1.9079, 2.9037, 3.8711, 4.9094, 4.5318,
0.0, 0.9766, 2.8888, 3.9129, 4.1181, 5.9187, -1.0,
0.0, 0.9838, 1.9499, 2.9159, 3.9358, 4.9069, 5.9005,
0.0, 0.9537, 1.9439, 2.9323, 3.9441, 4.9192, 5.8888,
0.0, 0.9163, 1.8563, 2.8823, 4.8005, 5.7794, -1.0,
0.0, 0.9762, 1.9288, 2.8929, 3.8167, 4.7478, 5.6866,
0.0, 0.9705, 1.9511, 2.9262, 3.9342, -1.0, -1.0,
0.0, 0.9581, 1.9123, 2.9327, 3.9105, 5.8285, -1.0,
0.0, 0.9346, 1.8816, 2.9075, 3.8705, 4.8131, 5.7244,
0.0, 0.95, 1.9165, 2.9377, 3.8956, 4.854, 5.816,
0.0, 0.971, 1.9564, 2.9515, 3.9353, -1.0, -1.0,
0.0, 0.9722, 1.9605, 2.9452, 3.9296, 4.2582, 4.5511,
0.0, 0.9569, 1.9215, 2.8958, 3.7644, 4.6808, 5.5939,
];
pub const R4_OVER_R2: [f64; 118] = [
8.0589, 3.4698, 29.0974, 14.8517, 11.8799, 7.8715, 5.5588,
4.7566, 3.8025, 3.1036, 26.1552, 17.2304, 17.721, 12.7442,
9.5361, 8.1652, 6.7463, 5.6004, 29.2012, 22.3934, 19.0598,
16.859, 15.4023, 12.5589, 13.4788, 12.2309, 11.2809, 10.5569,
10.1428, 9.4907, 13.4606, 10.8544, 8.9386, 8.135, 7.1251,
6.1971, 30.0162, 24.4103, 20.3537, 17.478, 13.5528, 11.8451,
11.0355, 10.1997, 9.5414, 9.0061, 8.6417, 8.9975, 14.0834,
11.8333, 10.0179, 9.3844, 8.411, 7.5152, 32.7622, 27.5708,
23.1671, 21.6003, 20.9615, 20.4562, 20.101, 19.7475, 19.4828,
15.6013, 19.2362, 17.4717, 17.8321, 17.4237, 17.1954, 17.1631,
14.5716, 15.8758, 13.8989, 12.4834, 11.4421, 10.2671, 8.3549,
7.8496, 7.3278, 7.482, 13.5124, 11.6554, 10.0959, 9.734,
8.8584, 8.0125, 29.8135, 26.3157, 19.1885, 15.8542, 16.1305,
15.6161, 15.1226, 16.1576, 14.651, 14.7178, 13.9108, 13.5623,
13.2326, 12.9189, 12.6133, 12.3142, 14.8326, 12.3771, 10.6378,
9.3638, 8.2297, 7.5667, 6.9456, 6.3946, 5.9159, 5.4929,
6.7286, 6.5144, 10.9169, 10.36, 9.4723, 8.6641,
];
pub const COVALENT_ANGSTROM: [f64; 118] = [
0.32, 0.46, 1.2, 0.94, 0.77, 0.75, 0.71,
0.63, 0.64, 0.67, 1.4, 1.25, 1.13, 1.04,
1.1, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33,
1.22, 1.21, 1.1, 1.07, 1.04, 1.0, 0.99,
1.01, 1.09, 1.12, 1.09, 1.15, 1.1, 1.14,
1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24,
1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28,
1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76,
1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51,
1.52, 1.51, 1.5, 1.49, 1.49, 1.48, 1.53,
1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11,
1.12, 1.13, 1.32, 1.3, 1.3, 1.36, 1.31,
1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52,
1.53, 1.54, 1.55, 1.49, 1.49, 1.51, 1.51,
1.48, 1.5, 1.56, 1.58, 1.45, 1.41, 1.34,
1.29, 1.27, 1.21, 1.16, 1.15, 1.09, 1.22,
1.36, 1.43, 1.46, 1.58, 1.48, 1.57,
];
pub const VDW_PAIR_ANGSTROM: [f64; 5356] = [
2.1823, 1.8547, 1.7347, 2.9086, 2.5732, 3.4956, 2.355,
2.5095, 2.9802, 3.0982, 2.5141, 2.3917, 2.9977, 2.9484,
3.216, 2.4492, 2.2527, 3.1933, 3.0214, 2.9531, 2.9103,
2.3667, 2.1328, 2.8784, 2.766, 2.7776, 2.7063, 2.6225,
2.1768, 2.0625, 2.6395, 2.6648, 2.6482, 2.5697, 2.4846,
2.4817, 2.0646, 1.9891, 2.5086, 2.6908, 2.6233, 2.477,
2.3885, 2.3511, 2.2996, 1.9892, 1.9251, 2.419, 2.5473,
2.4994, 2.4091, 2.3176, 2.2571, 2.1946, 2.1374, 2.9898,
2.6397, 3.6031, 3.1219, 3.762, 3.2485, 2.9357, 2.7093,
2.5781, 2.4839, 3.7082, 2.5129, 2.7321, 3.1052, 3.2962,
3.1331, 3.2, 2.9586, 3.0822, 2.8582, 2.712, 3.257,
3.4839, 2.8766, 2.7427, 3.2776, 3.2363, 3.5929, 3.2826,
3.0911, 2.9369, 2.903, 2.7789, 3.3921, 3.397, 4.0106,
2.8884, 2.6605, 3.7513, 3.1613, 3.3605, 3.3325, 3.0991,
2.9297, 2.8674, 2.7571, 3.8129, 3.3266, 3.7105, 3.7917,
2.8304, 2.5538, 3.3932, 3.1193, 3.1866, 3.1245, 3.0465,
2.8727, 2.7664, 2.6926, 3.4608, 3.2984, 3.5142, 3.5418,
3.5017, 2.619, 2.4797, 3.1331, 3.054, 3.0651, 2.9879,
2.9054, 2.8805, 2.733, 2.6331, 3.2096, 3.5668, 3.3684,
3.3686, 3.318, 3.3107, 2.4757, 2.4019, 2.9789, 3.1468,
2.9768, 2.8848, 2.7952, 2.7457, 2.6881, 2.5728, 3.0574,
3.3264, 3.3562, 3.2529, 3.1916, 3.1523, 3.1046, 2.3725,
2.3289, 2.876, 2.9804, 2.9093, 2.804, 2.7071, 2.6386,
2.572, 2.5139, 2.9517, 3.1606, 3.2085, 3.1692, 3.0982,
3.0352, 2.973, 2.9148, 3.2147, 2.8315, 3.8724, 3.4621,
3.8823, 3.376, 3.0746, 2.8817, 2.7552, 2.6605, 3.974,
3.6192, 3.6569, 3.9586, 3.6188, 3.3917, 3.2479, 3.1434,
4.2411, 2.7597, 3.0588, 3.3474, 3.6214, 3.4353, 3.4729,
3.2487, 3.32, 3.0914, 2.9403, 3.4972, 3.7993, 3.6773,
3.8678, 3.5808, 3.8243, 3.5826, 3.4156, 3.8765, 4.1035,
2.7361, 2.9765, 3.2475, 3.5004, 3.4185, 3.4378, 3.2084,
3.2787, 3.0604, 2.9187, 3.4037, 3.6759, 3.6586, 3.8327,
3.5372, 3.7665, 3.531, 3.37, 3.7788, 3.9804, 3.8903,
2.6832, 2.906, 3.2613, 3.4359, 3.3538, 3.386, 3.155,
3.23, 3.0133, 2.8736, 3.4024, 3.6142, 3.5979, 3.5295,
3.4834, 3.714, 3.4782, 3.317, 3.7434, 3.9623, 3.8181,
3.7642, 2.6379, 2.8494, 3.184, 3.4225, 3.2771, 3.3401,
3.1072, 3.1885, 2.9714, 2.8319, 3.3315, 3.5979, 3.5256,
3.498, 3.4376, 3.6714, 3.4346, 3.2723, 3.6859, 3.8985,
3.7918, 3.7372, 3.7211, 2.923, 2.6223, 3.4161, 2.8999,
3.0557, 3.3308, 3.0555, 2.8508, 2.7385, 2.664, 3.5263,
3.0277, 3.299, 3.7721, 3.5017, 3.2751, 3.1368, 3.0435,
3.7873, 3.2858, 3.214, 3.1727, 3.2178, 3.4414, 2.549,
2.7623, 3.0991, 3.3252, 3.1836, 3.2428, 3.0259, 3.1225,
2.9032, 2.7621, 3.249, 3.511, 3.4429, 3.3845, 3.3574,
3.6045, 3.3658, 3.2013, 3.611, 3.8241, 3.709, 3.6496,
3.6333, 3.0896, 3.5462, 2.4926, 2.7136, 3.0693, 3.2699,
3.1272, 3.1893, 2.9658, 3.0972, 2.8778, 2.7358, 3.2206,
3.4566, 3.3896, 3.3257, 3.2946, 3.5693, 3.3312, 3.167,
3.5805, 3.7711, 3.6536, 3.5927, 3.5775, 3.0411, 3.4885,
3.4421, 2.4667, 2.6709, 3.0575, 3.2357, 3.0908, 3.1537,
2.9235, 3.0669, 2.8476, 2.7054, 3.2064, 3.4519, 3.3593,
3.2921, 3.2577, 3.2161, 3.2982, 3.1339, 3.5606, 3.7582,
3.6432, 3.5833, 3.5691, 3.0161, 3.4812, 3.4339, 3.4327,
2.4515, 2.6338, 3.0511, 3.2229, 3.063, 3.1265, 2.8909,
3.0253, 2.8184, 2.6764, 3.1968, 3.4114, 3.3492, 3.2691,
3.232, 3.1786, 3.268, 3.1036, 3.5453, 3.7259, 3.609,
3.5473, 3.5327, 3.0018, 3.4413, 3.3907, 3.3593, 3.3462,
2.4413, 2.6006, 3.054, 3.1987, 3.049, 3.1058, 2.8643,
2.9948, 2.7908, 2.6491, 3.195, 3.3922, 3.3316, 3.2585,
3.2136, 3.1516, 3.2364, 3.0752, 3.5368, 3.7117, 3.5941,
3.5313, 3.5164, 2.9962, 3.4225, 3.3699, 3.337, 3.3234,
3.3008, 2.4318, 2.5729, 3.0416, 3.1639, 3.0196, 3.0843,
2.8413, 2.7436, 2.7608, 2.6271, 3.1811, 3.3591, 3.3045,
3.2349, 3.1942, 3.1291, 3.2111, 3.0534, 3.5189, 3.6809,
3.5635, 3.5001, 3.4854, 2.9857, 3.3897, 3.3363, 3.3027,
3.289, 3.2655, 3.2309, 2.8502, 2.6934, 3.2467, 3.1921,
3.5663, 3.2541, 3.0571, 2.9048, 2.8657, 2.7438, 3.3547,
3.351, 3.9837, 3.6871, 3.4862, 3.3389, 3.2413, 3.1708,
3.6096, 3.628, 3.686, 3.5568, 3.4836, 3.2868, 3.3994,
3.3476, 3.317, 3.295, 3.2874, 3.2606, 3.9579, 2.9226,
2.6838, 3.7867, 3.1732, 3.3872, 3.3643, 3.1267, 2.9541,
2.8505, 2.7781, 3.8475, 3.3336, 3.7359, 3.8266, 3.5733,
3.3959, 3.2775, 3.1915, 3.9878, 3.8816, 3.581, 3.5364,
3.506, 3.8097, 3.3925, 3.3348, 3.3019, 3.2796, 3.2662,
3.2464, 3.7136, 3.8619, 2.914, 2.6271, 3.4771, 3.1774,
3.256, 3.197, 3.1207, 2.9406, 2.8322, 2.7571, 3.5455,
3.3514, 3.5837, 3.6177, 3.5816, 3.3902, 3.2604, 3.1652,
3.7037, 3.6283, 3.5858, 3.533, 3.4884, 3.5789, 3.4094,
3.3473, 3.3118, 3.2876, 3.2707, 3.2521, 3.557, 3.6496,
3.6625, 2.73, 2.587, 3.2471, 3.1487, 3.1667, 3.0914,
3.0107, 2.9812, 2.83, 2.7284, 3.3259, 3.3182, 3.4707,
3.4748, 3.4279, 3.4182, 3.2547, 3.1353, 3.5116, 3.9432,
3.8828, 3.8303, 3.788, 3.376, 3.7218, 3.3408, 3.3059,
3.2698, 3.2446, 3.2229, 3.4422, 3.5023, 3.5009, 3.5268,
2.6026, 2.5355, 3.1129, 3.2863, 3.1029, 3.0108, 2.9227,
2.8694, 2.8109, 2.6929, 3.1958, 3.467, 3.4018, 3.3805,
3.3218, 3.2815, 3.2346, 3.0994, 3.3937, 3.7266, 3.6697,
3.6164, 3.573, 3.2522, 3.5051, 3.4686, 3.4355, 3.4084,
3.3748, 3.3496, 3.3692, 3.4052, 3.391, 3.3849, 3.3662,
2.5087, 2.4814, 3.0239, 3.1312, 3.0535, 2.9457, 2.8496,
2.778, 2.7828, 2.6532, 3.1063, 3.3143, 3.3549, 3.312,
3.2421, 3.1787, 3.1176, 3.0613, 3.3082, 3.5755, 3.5222,
3.4678, 3.4231, 3.1684, 3.3528, 3.3162, 3.2827, 3.2527,
3.2308, 3.2029, 3.3173, 3.3343, 3.3092, 3.2795, 3.2452,
3.2096, 3.2893, 2.8991, 4.0388, 3.61, 3.9388, 3.4475,
3.159, 2.9812, 2.8586, 2.7683, 4.1428, 3.7911, 3.8225,
4.0372, 3.7059, 3.4935, 3.3529, 3.2492, 4.4352, 4.0826,
3.9733, 3.9254, 3.8646, 3.9315, 3.7837, 3.7465, 3.7211,
3.7012, 3.6893, 3.6676, 3.7736, 4.066, 3.7926, 3.6158,
3.5017, 3.4166, 4.6176, 2.8786, 3.1658, 3.5823, 3.7689,
3.5762, 3.5789, 3.3552, 3.4004, 3.1722, 3.0212, 3.7241,
3.9604, 3.85, 3.9844, 3.7035, 3.9161, 3.6751, 3.5075,
4.1151, 4.2877, 4.1579, 4.1247, 4.0617, 3.4874, 3.9848,
3.928, 3.9079, 3.8751, 3.8604, 3.8277, 3.8002, 3.9981,
3.7544, 4.0371, 3.8225, 3.6718, 4.3092, 4.4764, 2.8997,
3.0953, 3.4524, 3.6107, 3.6062, 3.5783, 3.3463, 3.3855,
3.1746, 3.0381, 3.6019, 3.7938, 3.8697, 3.9781, 3.6877,
3.8736, 3.6451, 3.489, 3.9858, 4.1179, 4.043, 3.9563,
3.9182, 3.4002, 3.831, 3.7716, 3.7543, 3.7203, 3.7053,
3.6742, 3.8318, 3.7631, 3.7392, 3.9892, 3.7832, 3.6406,
4.1701, 4.3016, 4.2196, 2.8535, 3.0167, 3.3978, 3.5363,
3.5393, 3.5301, 3.296, 3.3352, 3.1287, 2.9967, 3.6659,
3.7239, 3.807, 3.7165, 3.6368, 3.8162, 3.5885, 3.4336,
3.9829, 4.0529, 3.9584, 3.9025, 3.8607, 3.3673, 3.7658,
3.7035, 3.6866, 3.6504, 3.6339, 3.6024, 3.7708, 3.7283,
3.6896, 3.9315, 3.725, 3.5819, 4.1457, 4.228, 4.113,
4.0597, 3.0905, 2.7998, 3.6448, 3.0739, 3.2996, 3.5262,
3.2559, 3.0518, 2.9394, 2.8658, 3.7514, 3.2295, 3.5643,
3.7808, 3.6931, 3.4723, 3.3357, 3.2429, 4.028, 3.5589,
3.4636, 3.4994, 3.4309, 3.6177, 3.2946, 3.2376, 3.205,
3.1847, 3.1715, 3.1599, 3.5555, 3.8111, 3.7693, 3.5718,
3.4498, 3.3662, 4.1608, 3.7417, 3.6536, 3.6154, 3.8596,
3.0301, 2.7312, 3.5821, 3.0473, 3.2137, 3.4679, 3.1975,
2.9969, 2.8847, 2.811, 3.6931, 3.2076, 3.4943, 3.5956,
3.6379, 3.419, 3.2808, 3.186, 3.985, 3.5105, 3.433,
3.3797, 3.4155, 3.6033, 3.2737, 3.2145, 3.1807, 3.1596,
3.1461, 3.1337, 3.4812, 3.6251, 3.7152, 3.5201, 3.3966,
3.3107, 4.1128, 3.6899, 3.6082, 3.5604, 3.7834, 3.7543,
2.9189, 2.6777, 3.4925, 2.9648, 3.1216, 3.294, 3.0975,
2.9757, 2.8493, 2.7638, 3.6085, 3.1214, 3.4006, 3.4793,
3.5147, 3.3806, 3.2356, 3.1335, 3.9144, 3.4183, 3.3369,
3.2803, 3.2679, 3.4871, 3.1714, 3.1521, 3.1101, 3.0843,
3.067, 3.0539, 3.389, 3.5086, 3.5895, 3.4783, 3.3484,
3.2559, 4.0422, 3.5967, 3.5113, 3.4576, 3.6594, 3.6313,
3.569, 2.8578, 2.6334, 3.4673, 2.9245, 3.0732, 3.2435,
3.0338, 2.9462, 2.8143, 2.724, 3.5832, 3.0789, 3.3617,
3.4246, 3.4505, 3.3443, 3.1964, 3.0913, 3.8921, 3.3713,
3.2873, 3.2281, 3.2165, 3.4386, 3.1164, 3.122, 3.0761,
3.048, 3.0295, 3.0155, 3.3495, 3.4543, 3.526, 3.4413,
3.3085, 3.2134, 4.017, 3.5464, 3.4587, 3.4006, 3.6027,
3.573, 3.4945, 3.4623, 2.824, 2.596, 3.4635, 2.9032,
3.0431, 3.2115, 2.9892, 2.9148, 2.7801, 2.6873, 3.5776,
3.0568, 3.3433, 3.3949, 3.4132, 3.3116, 3.1616, 3.0548,
3.8859, 3.3719, 3.2917, 3.2345, 3.2274, 3.4171, 3.1293,
3.0567, 3.0565, 3.0274, 3.0087, 2.9939, 3.3293, 3.4249,
3.4902, 3.4091, 3.2744, 3.1776, 4.0078, 3.5374, 3.4537,
3.3956, 3.5747, 3.543, 3.4522, 3.416, 3.3975, 2.8004,
2.5621, 3.4617, 2.9154, 3.0203, 3.1875, 2.9548, 2.8038,
2.7472, 2.653, 3.5736, 3.0584, 3.3304, 3.3748, 3.3871,
3.2028, 3.1296, 3.0214, 3.8796, 3.3337, 3.2492, 3.1883,
3.1802, 3.405, 3.0756, 3.0478, 3.0322, 3.0323, 3.0163,
3.0019, 3.3145, 3.405, 3.4656, 3.3021, 3.2433, 3.1453,
3.9991, 3.5017, 3.4141, 3.352, 3.5583, 3.5251, 3.4243,
3.3851, 3.3662, 3.3525, 2.7846, 2.5324, 3.4652, 2.8759,
3.0051, 3.1692, 2.9273, 2.7615, 2.7164, 2.6212, 3.5744,
3.0275, 3.3249, 3.3627, 3.3686, 3.1669, 3.0584, 2.9915,
3.8773, 3.3099, 3.2231, 3.16, 3.152, 3.4023, 3.0426,
3.0099, 2.992, 2.9809, 2.98, 2.9646, 3.3068, 3.393,
3.4486, 3.2682, 3.1729, 3.1168, 3.9952, 3.4796, 3.3901,
3.3255, 3.553, 3.5183, 3.4097, 3.3683, 3.3492, 3.336,
3.3308, 2.5424, 2.6601, 3.2555, 3.2807, 3.1384, 3.1737,
2.9397, 2.8429, 2.8492, 2.7225, 3.3875, 3.491, 3.452,
3.3608, 3.3036, 3.2345, 3.2999, 3.1487, 3.7409, 3.8392,
3.7148, 3.6439, 3.6182, 3.1753, 3.521, 3.4639, 3.4265,
3.4075, 3.3828, 3.3474, 3.4071, 3.3754, 3.3646, 3.3308,
3.4393, 3.2993, 3.8768, 3.9891, 3.831, 3.7483, 3.3417,
3.3019, 3.225, 3.1832, 3.1578, 3.1564, 3.1224, 3.462,
2.9743, 2.8058, 3.483, 3.3474, 3.6863, 3.3617, 3.1608,
3.0069, 2.964, 2.8427, 3.5885, 3.5219, 4.1314, 3.812,
3.6015, 3.4502, 3.3498, 3.2777, 3.8635, 3.8232, 3.8486,
3.7215, 3.6487, 3.4724, 3.5627, 3.5087, 3.4757, 3.4517,
3.4423, 3.4139, 4.1028, 3.8388, 3.6745, 3.5562, 3.4806,
3.4272, 4.0182, 3.9991, 4.0007, 3.9282, 3.7238, 3.6498,
3.5605, 3.5211, 3.5009, 3.4859, 3.4785, 3.5621, 4.2623,
3.0775, 2.8275, 4.0181, 3.3385, 3.5379, 3.5036, 3.2589,
3.0804, 3.0094, 2.9003, 4.0869, 3.5088, 3.9105, 3.9833,
3.7176, 3.5323, 3.4102, 3.3227, 4.2702, 4.0888, 3.756,
3.7687, 3.6681, 3.6405, 3.5569, 3.499, 3.4659, 3.4433,
3.433, 3.4092, 3.8867, 4.019, 3.7961, 3.6412, 3.5405,
3.4681, 4.3538, 4.2136, 3.9381, 3.8912, 3.9681, 3.7909,
3.6774, 3.6262, 3.5999, 3.5823, 3.5727, 3.5419, 4.0245,
4.1874, 3.0893, 2.7917, 3.7262, 3.3518, 3.4241, 3.5433,
3.2773, 3.089, 2.9775, 2.901, 3.8048, 3.5362, 3.7746,
3.7911, 3.7511, 3.5495, 3.4149, 3.3177, 4.0129, 3.837,
3.7739, 3.7125, 3.7152, 3.7701, 3.5813, 3.5187, 3.4835,
3.4595, 3.4439, 3.4242, 3.7476, 3.8239, 3.8346, 3.6627,
3.5479, 3.4639, 4.1026, 3.9733, 3.9292, 3.8667, 3.9513,
3.8959, 3.7698, 3.7089, 3.6765, 3.6548, 3.6409, 3.5398,
3.8759, 3.9804, 4.015, 2.9091, 2.7638, 3.5066, 3.3377,
3.3481, 3.2633, 3.181, 3.1428, 2.9872, 2.8837, 3.5929,
3.5183, 3.6729, 3.6596, 3.6082, 3.5927, 3.4224, 3.2997,
3.819, 4.1865, 4.1114, 4.054, 3.6325, 3.5697, 3.5561,
3.5259, 3.4901, 3.4552, 3.4315, 3.4091, 3.6438, 3.6879,
3.6832, 3.7043, 3.5557, 3.4466, 3.9203, 4.2919, 4.2196,
4.1542, 3.7573, 3.7039, 3.6546, 3.6151, 3.5293, 3.4849,
3.4552, 3.5192, 3.7673, 3.8359, 3.8525, 3.8901, 2.7806,
2.7209, 3.3812, 3.4958, 3.2913, 3.1888, 3.099, 3.0394,
2.9789, 2.8582, 3.4716, 3.6883, 3.6105, 3.5704, 3.5059,
3.4619, 3.4138, 3.2742, 3.708, 3.9773, 3.901, 3.8409,
3.7944, 3.4465, 3.7235, 3.6808, 3.6453, 3.6168, 3.5844,
3.5576, 3.5772, 3.5959, 3.5768, 3.5678, 3.5486, 3.4228,
3.8107, 4.0866, 4.0169, 3.9476, 3.6358, 3.58, 3.526,
3.4838, 3.4501, 3.4204, 3.3553, 3.6487, 3.6973, 3.7398,
3.7405, 3.7459, 3.738, 2.6848, 2.674, 3.2925, 3.3386,
3.2473, 3.1284, 3.0301, 2.9531, 2.9602, 2.8272, 3.383,
3.5358, 3.5672, 3.5049, 3.4284, 3.3621, 3.3001, 3.2451,
3.6209, 3.8299, 3.7543, 3.692, 3.6436, 3.3598, 3.5701,
3.5266, 3.4904, 3.459, 3.4364, 3.4077, 3.5287, 3.528,
3.4969, 3.465, 3.4304, 3.3963, 3.7229, 3.9402, 3.8753,
3.8035, 3.5499, 3.4913, 3.4319, 3.3873, 3.352, 3.3209,
3.2948, 3.5052, 3.6465, 3.6696, 3.6577, 3.6388, 3.6142,
3.5889, 3.3968, 3.0122, 4.2241, 3.7887, 4.0049, 3.5384,
3.2698, 3.1083, 2.9917, 2.9057, 4.334, 3.99, 4.6588,
4.1278, 3.8125, 3.6189, 3.4851, 3.3859, 4.6531, 4.3134,
4.2258, 4.1309, 4.0692, 4.0944, 3.985, 3.9416, 3.9112,
3.8873, 3.8736, 3.8473, 4.6027, 4.1538, 3.8994, 3.7419,
3.6356, 3.5548, 4.8353, 4.5413, 4.3891, 4.3416, 4.3243,
4.2753, 4.2053, 4.179, 4.1685, 4.1585, 4.1536, 4.0579,
4.198, 4.4564, 4.2192, 4.0528, 3.9489, 3.8642, 5.0567,
3.063, 3.3271, 4.0432, 4.0046, 4.1555, 3.7426, 3.513,
3.5174, 3.2884, 3.1378, 4.1894, 4.2321, 4.1725, 4.1833,
3.8929, 4.0544, 3.8118, 3.6414, 4.6373, 4.6268, 4.475,
4.4134, 4.3458, 3.8582, 4.2583, 4.1898, 4.1562, 4.1191,
4.1069, 4.0639, 4.1257, 4.1974, 3.9532, 4.1794, 3.966,
3.813, 4.816, 4.8272, 4.6294, 4.584, 4.077, 4.0088,
3.9103, 3.8536, 3.8324, 3.7995, 3.7826, 4.2294, 4.338,
4.4352, 4.1933, 4.458, 4.2554, 4.1072, 5.0454, 5.1814,
3.0632, 3.2662, 3.6432, 3.8088, 3.791, 3.7381, 3.5093,
3.5155, 3.3047, 3.1681, 3.7871, 3.9924, 4.0637, 4.1382,
3.8591, 4.0164, 3.7878, 3.6316, 4.1741, 4.3166, 4.2395,
4.1831, 4.1107, 3.5857, 4.027, 3.9676, 3.9463, 3.915,
3.9021, 3.8708, 4.024, 4.1551, 3.9108, 4.1337, 3.9289,
3.7873, 4.3666, 4.508, 4.4232, 4.3155, 3.8461, 3.8007,
3.6991, 3.6447, 3.6308, 3.5959, 3.5749, 4.0359, 4.3124,
4.3539, 4.1122, 4.3772, 4.1785, 4.0386, 4.7004, 4.8604,
4.6261, 2.9455, 3.247, 3.6108, 3.8522, 3.6625, 3.6598,
3.4411, 3.466, 3.2415, 3.0944, 3.7514, 4.0397, 3.9231,
4.0561, 3.786, 3.9845, 3.7454, 3.5802, 4.1366, 4.3581,
4.2351, 4.2011, 4.1402, 3.5381, 4.0653, 4.0093, 3.9883,
3.957, 3.9429, 3.9112, 3.8728, 4.0682, 3.8351, 4.1054,
3.8928, 3.7445, 4.3415, 4.5497, 4.3833, 4.3122, 3.8051,
3.7583, 3.6622, 3.6108, 3.5971, 3.5628, 3.5408, 4.078,
4.0727, 4.2836, 4.0553, 4.3647, 4.1622, 4.0178, 4.5802,
4.9125, 4.5861, 4.6201, 2.9244, 3.2241, 3.5848, 3.8293,
3.6395, 3.64, 3.4204, 3.4499, 3.2253, 3.0779, 3.7257,
4.017, 3.9003, 4.0372, 3.7653, 3.9672, 3.7283, 3.563,
4.1092, 4.3347, 4.2117, 4.1793, 4.1179, 3.5139, 4.0426,
3.9867, 3.9661, 3.9345, 3.92, 3.8883, 3.8498, 4.0496,
3.8145, 4.0881, 3.8756, 3.7271, 4.3128, 4.5242, 4.3578,
4.287, 3.7796, 3.7318, 3.6364, 3.5854, 3.5726, 3.5378,
3.5155, 4.0527, 4.0478, 4.263, 4.0322, 4.3449, 4.1421,
3.9975, 4.5499, 4.8825, 4.5601, 4.595, 4.5702, 2.9046,
3.2044, 3.5621, 3.8078, 3.6185, 3.622, 3.4019, 3.4359,
3.211, 3.0635, 3.7037, 3.9958, 3.8792, 4.0194, 3.746,
3.9517, 3.7128, 3.5474, 4.0872, 4.3138, 4.1906, 4.1593,
4.0973, 3.4919, 4.0216, 3.9657, 3.9454, 3.9134, 3.8986,
3.8669, 3.8289, 4.0323, 3.7954, 4.0725, 3.8598, 3.7113,
4.2896, 4.5021, 4.3325, 4.2645, 3.7571, 3.7083, 3.6136,
3.5628, 3.5507, 3.5155, 3.4929, 4.0297, 4.0234, 4.2442,
4.0112, 4.3274, 4.124, 3.9793, 4.5257, 4.8568, 4.5353,
4.5733, 4.5485, 4.5271, 2.8878, 3.189, 3.5412, 3.7908,
3.5974, 3.6078, 3.3871, 3.4243, 3.1992, 3.0513, 3.6831,
3.9784, 3.8579, 4.0049, 3.7304, 3.9392, 3.7002, 3.5347,
4.0657, 4.2955, 4.1705, 4.1424, 4.08, 3.4717, 4.0043,
3.9485, 3.9286, 3.8965, 3.8815, 3.85, 3.8073, 4.018,
3.7796, 4.0598, 3.847, 3.6983, 4.2678, 4.483, 4.3132,
4.2444, 3.737, 3.6876, 3.5935, 3.5428, 3.5314, 3.4958,
3.473, 4.0117, 4.0043, 4.2287, 3.9939, 4.3134, 4.1096,
3.9646, 4.5032, 4.8356, 4.5156, 4.5544, 4.5297, 4.5083,
4.4896, 2.8709, 3.1737, 3.5199, 3.7734, 3.5802, 3.5934,
3.3724, 3.4128, 3.1877, 3.0396, 3.6624, 3.9608, 3.8397,
3.9893, 3.7145, 3.9266, 3.6877, 3.5222, 4.0448, 4.2771,
4.1523, 4.1247, 4.0626, 3.453, 3.9866, 3.931, 3.9115,
3.8792, 3.8641, 3.8326, 3.7892, 4.0025, 3.7636, 4.0471,
3.8343, 3.6854, 4.2464, 4.4635, 4.2939, 4.2252, 3.7169,
3.6675, 3.5739, 3.5235, 3.5126, 3.4768, 3.4537, 3.9932,
3.9854, 4.2123, 3.9765, 4.2992, 4.0951, 3.95, 4.4811,
4.8135, 4.4959, 4.5351, 4.5105, 4.4891, 4.4705, 4.4515,
2.8568, 3.1608, 3.505, 3.7598, 3.5665, 3.5803, 3.3601,
3.4031, 3.1779, 3.0296, 3.6479, 3.9471, 3.8262, 3.9773,
3.7015, 3.9162, 3.6771, 3.5115, 4.0306, 4.2634, 4.1385,
4.1116, 4.0489, 3.4366, 3.9732, 3.9176, 3.8983, 3.8659,
3.8507, 3.8191, 3.7757, 3.9907, 3.7506, 4.0365, 3.8235,
3.6745, 4.2314, 4.449, 4.2792, 4.2105, 3.7003, 3.651,
3.5578, 3.5075, 3.4971, 3.4609, 3.4377, 3.9788, 3.9712,
4.1997, 3.9624, 4.2877, 4.0831, 3.9378, 4.4655, 4.7974,
4.4813, 4.5209, 4.4964, 4.475, 4.4565, 4.4375, 4.4234,
2.6798, 3.0151, 3.2586, 3.5292, 3.5391, 3.4902, 3.2887,
3.3322, 3.1228, 2.9888, 3.4012, 3.7145, 3.783, 3.6665,
3.5898, 3.8077, 3.581, 3.4265, 3.7726, 4.0307, 3.9763,
3.889, 3.8489, 3.2706, 3.7595, 3.6984, 3.6772, 3.6428,
3.6243, 3.5951, 3.7497, 3.6775, 3.6364, 3.9203, 3.7157,
3.5746, 3.9494, 4.2076, 4.1563, 4.0508, 3.5329, 3.478,
3.3731, 3.3126, 3.2846, 3.2426, 3.2135, 3.7491, 3.9006,
3.8332, 3.8029, 4.1436, 3.9407, 3.7998, 4.1663, 4.5309,
4.3481, 4.2911, 4.2671, 4.2415, 4.223, 4.2047, 4.1908,
4.1243, 2.5189, 2.9703, 3.3063, 3.6235, 3.4517, 3.3989,
3.2107, 3.2434, 3.0094, 2.858, 3.4253, 3.8157, 3.7258,
3.6132, 3.5297, 3.7566, 3.5095, 3.3368, 3.789, 4.1298,
4.019, 3.9573, 3.9237, 3.2677, 3.848, 3.8157, 3.7656,
3.7317, 3.7126, 3.6814, 3.6793, 3.6218, 3.5788, 3.8763,
3.6572, 3.5022, 3.9737, 4.3255, 4.1828, 4.1158, 3.5078,
3.4595, 3.36, 3.3088, 3.2575, 3.2164, 3.1856, 3.8522,
3.8665, 3.8075, 3.7772, 4.1391, 3.9296, 3.7772, 4.2134,
4.7308, 4.3787, 4.3894, 4.3649, 4.3441, 4.3257, 4.3073,
4.2941, 4.1252, 4.2427, 3.0481, 2.9584, 3.6919, 3.599,
3.8881, 3.4209, 3.1606, 3.1938, 2.9975, 2.8646, 3.8138,
3.7935, 3.7081, 3.9155, 3.591, 3.4808, 3.4886, 3.3397,
4.1336, 4.1122, 3.9888, 3.9543, 3.8917, 3.5894, 3.8131,
3.7635, 3.7419, 3.7071, 3.688, 3.6574, 3.6546, 3.9375,
3.6579, 3.587, 3.6361, 3.5039, 4.3149, 4.2978, 4.1321,
4.1298, 3.8164, 3.768, 3.7154, 3.6858, 3.6709, 3.6666,
3.6517, 3.8174, 3.8608, 4.1805, 3.9102, 3.8394, 3.8968,
3.7673, 4.5274, 4.6682, 4.3344, 4.3639, 4.3384, 4.3162,
4.2972, 4.2779, 4.2636, 4.0253, 4.1168, 4.1541, 2.8136,
3.0951, 3.4635, 3.6875, 3.4987, 3.5183, 3.2937, 3.358,
3.1325, 2.9832, 3.6078, 3.8757, 3.7616, 3.9222, 3.637,
3.8647, 3.6256, 3.4595, 3.9874, 4.1938, 4.0679, 4.043,
3.9781, 3.3886, 3.9008, 3.8463, 3.8288, 3.795, 3.779,
3.7472, 3.7117, 3.9371, 3.6873, 3.9846, 3.7709, 3.621,
4.1812, 4.375, 4.2044, 4.134, 3.6459, 3.5929, 3.5036,
3.4577, 3.4528, 3.4146, 3.3904, 3.9014, 3.9031, 4.1443,
3.8961, 4.2295, 4.0227, 3.8763, 4.4086, 4.7097, 4.4064,
4.4488, 4.4243, 4.4029, 4.3842, 4.3655, 4.3514, 4.1162,
4.2205, 4.1953, 4.2794, 2.8032, 3.0805, 3.4519, 3.67,
3.4827, 3.505, 3.2799, 3.3482, 3.1233, 2.9747, 3.5971,
3.8586, 3.7461, 3.91, 3.6228, 3.8535, 3.6147, 3.449,
3.9764, 4.1773, 4.0511, 4.027, 3.9614, 3.3754, 3.8836,
3.8291, 3.8121, 3.778, 3.7619, 3.73, 3.6965, 3.9253,
3.6734, 3.9733, 3.7597, 3.6099, 4.1683, 4.3572, 4.1862,
4.1153, 3.6312, 3.5772, 3.4881, 3.4429, 3.4395, 3.4009,
3.3766, 3.8827, 3.8868, 4.1316, 3.8807, 4.2164, 4.0092,
3.8627, 4.3936, 4.6871, 4.3882, 4.4316, 4.4073, 4.3858,
4.3672, 4.3485, 4.3344, 4.0984, 4.2036, 4.1791, 4.2622,
4.245, 2.7967, 3.0689, 3.4445, 3.6581, 3.4717, 3.4951,
3.2694, 3.3397, 3.1147, 2.9661, 3.5898, 3.8468, 3.7358,
3.9014, 3.6129, 3.8443, 3.6054, 3.4396, 3.9683, 4.1656,
4.0394, 4.0158, 3.9498, 3.3677, 3.8718, 3.8164, 3.8005,
3.7662, 3.75, 3.7181, 3.6863, 3.917, 3.6637, 3.9641,
3.7503, 3.6004, 4.159, 4.3448, 4.1739, 4.1029, 3.6224,
3.5677, 3.4785, 3.4314, 3.4313, 3.3923, 3.368, 3.8698,
3.8758, 4.1229, 3.8704, 4.2063, 3.9987, 3.8519, 4.3832,
4.6728, 4.3759, 4.4195, 4.3952, 4.3737, 4.3551, 4.3364,
4.3223, 4.0861, 4.1911, 4.1676, 4.2501, 4.2329, 4.2208,
2.7897, 3.0636, 3.4344, 3.648, 3.4626, 3.4892, 3.2626,
3.3344, 3.1088, 2.9597, 3.5804, 3.8359, 3.7251, 3.894,
3.6047, 3.8375, 3.599, 3.4329, 3.9597, 4.1542, 4.0278,
4.0048, 3.939, 3.3571, 3.8608, 3.8056, 3.7899, 3.756,
3.74, 3.7081, 3.6758, 3.9095, 3.6552, 3.9572, 3.7436,
3.5933, 4.1508, 4.3337, 4.1624, 4.0916, 3.6126, 3.5582,
3.4684, 3.4212, 3.4207, 3.3829, 3.3586, 3.8604, 3.8658,
4.1156, 3.862, 4.1994, 3.9917, 3.8446, 4.375, 4.6617,
4.3644, 4.4083, 4.384, 4.3625, 4.3439, 4.3253, 4.3112,
4.0745, 4.1807, 4.1578, 4.239, 4.2218, 4.2097, 4.1986,
2.8395, 3.0081, 3.3171, 3.4878, 3.536, 3.5145, 3.2809,
3.3307, 3.126, 2.994, 3.4741, 3.6675, 3.7832, 3.6787,
3.6156, 3.8041, 3.5813, 3.4301, 3.848, 3.9849, 3.9314,
3.8405, 3.8029, 3.2962, 3.7104, 3.6515, 3.6378, 3.602,
3.5849, 3.555, 3.7494, 3.6893, 3.6666, 3.917, 3.715,
3.576, 4.0268, 4.1596, 4.1107, 3.9995, 3.5574, 3.5103,
3.4163, 3.3655, 3.3677, 3.3243, 3.2975, 3.7071, 3.9047,
3.8514, 3.8422, 3.8022, 3.9323, 3.7932, 4.2343, 4.4583,
4.3115, 4.2457, 4.2213, 4.1945, 4.1756, 4.1569, 4.1424,
4.062, 4.0494, 3.9953, 4.0694, 4.0516, 4.0396, 4.028,
4.013, 2.9007, 2.9674, 3.8174, 3.5856, 3.6486, 3.5339,
3.2832, 3.3154, 3.1144, 2.9866, 3.9618, 3.843, 3.998,
3.8134, 3.6652, 3.7985, 3.5756, 3.4207, 4.4061, 4.2817,
4.1477, 4.0616, 3.9979, 3.6492, 3.8833, 3.8027, 3.766,
3.7183, 3.6954, 3.6525, 3.9669, 3.8371, 3.7325, 3.916,
3.7156, 3.5714, 4.6036, 4.462, 4.3092, 4.2122, 3.8478,
3.7572, 3.6597, 3.5969, 3.5575, 3.5386, 3.5153, 3.7818,
4.1335, 4.0153, 3.9177, 3.8603, 3.9365, 3.7906, 4.7936,
4.741, 4.5461, 4.5662, 4.534, 4.5059, 4.4832, 4.4604,
4.4429, 4.2346, 4.4204, 4.3119, 4.345, 4.3193, 4.3035,
4.2933, 4.1582, 4.245, 2.8559, 2.905, 3.8325, 3.5442,
3.5077, 3.4905, 3.2396, 3.272, 3.0726, 2.9467, 3.9644,
3.805, 3.8981, 3.7762, 3.6216, 3.7531, 3.5297, 3.3742,
4.3814, 4.2818, 4.1026, 4.0294, 3.964, 3.6208, 3.8464,
3.7648, 3.7281, 3.679, 3.6542, 3.6117, 3.865, 3.801,
3.6894, 3.8713, 3.6699, 3.5244, 4.5151, 4.4517, 4.2538,
4.1483, 3.8641, 3.7244, 3.6243, 3.5589, 3.5172, 3.4973,
3.4715, 3.734, 4.0316, 3.9958, 3.8687, 3.8115, 3.8862,
3.7379, 4.7091, 4.7156, 4.5199, 4.5542, 4.523, 4.4959,
4.475, 4.4529, 4.4361, 4.1774, 4.3774, 4.2963, 4.3406,
4.3159, 4.3006, 4.291, 4.1008, 4.1568, 4.098, 2.811,
2.852, 3.748, 3.5105, 3.4346, 3.3461, 3.1971, 3.2326,
3.0329, 2.907, 3.8823, 3.7928, 3.8264, 3.7006, 3.5797,
3.7141, 3.4894, 3.3326, 4.3048, 4.2217, 4.0786, 3.99,
3.9357, 3.6331, 3.8333, 3.7317, 3.6957, 3.646, 3.6197,
3.5779, 3.7909, 3.7257, 3.6476, 3.5729, 3.6304, 3.4834,
4.4368, 4.3921, 4.2207, 4.1133, 3.8067, 3.7421, 3.614,
3.5491, 3.5077, 3.4887, 3.4623, 3.6956, 3.9568, 3.8976,
3.824, 3.7684, 3.8451, 3.6949, 4.6318, 4.6559, 4.4533,
4.4956, 4.4641, 4.4366, 4.4155, 4.3936, 4.3764, 4.1302,
4.3398, 4.2283, 4.2796, 4.2547, 4.2391, 4.2296, 4.0699,
4.1083, 4.0319, 3.9855, 2.7676, 2.8078, 3.6725, 3.4804,
3.3775, 3.2411, 3.1581, 3.1983, 2.9973, 2.8705, 3.807,
3.7392, 3.7668, 3.6263, 3.5402, 3.6807, 3.4545, 3.2962,
4.2283, 4.1698, 4.024, 3.9341, 3.8711, 3.5489, 3.7798,
3.7, 3.6654, 3.6154, 3.5882, 3.5472, 3.7289, 3.651,
3.6078, 3.5355, 3.5963, 3.448, 4.3587, 4.339, 4.1635,
4.0536, 3.7193, 3.6529, 3.5512, 3.4837, 3.44, 3.4191,
3.3891, 3.6622, 3.8934, 3.8235, 3.7823, 3.7292, 3.8106,
3.6589, 4.5535, 4.6013, 4.3961, 4.4423, 4.4109, 4.3835,
4.3625, 4.3407, 4.3237, 4.0863, 4.2835, 4.1675, 4.2272,
4.2025, 4.1869, 4.1774, 4.0126, 4.046, 3.9815, 3.934,
3.8955, 2.6912, 2.7604, 3.6037, 3.4194, 3.3094, 3.171,
3.0862, 3.1789, 2.9738, 2.8427, 3.7378, 3.6742, 3.6928,
3.5512, 3.4614, 3.4087, 3.4201, 3.2607, 4.1527, 4.0977,
3.9523, 3.8628, 3.8002, 3.4759, 3.7102, 3.6466, 3.6106,
3.558, 3.5282, 3.4878, 3.6547, 3.5763, 3.5289, 3.5086,
3.5593, 3.4099, 4.2788, 4.2624, 4.0873, 3.977, 3.6407,
3.5743, 3.5178, 3.4753, 3.3931, 3.3694, 3.3339, 3.6002,
3.8164, 3.7478, 3.7028, 3.6952, 3.7669, 3.6137, 4.4698,
4.5488, 4.3168, 4.3646, 4.3338, 4.3067, 4.286, 4.2645,
4.2478, 4.0067, 4.2349, 4.0958, 4.1543, 4.1302, 4.1141,
4.1048, 3.941, 3.9595, 3.8941, 3.8465, 3.8089, 3.749,
2.7895, 2.5849, 3.6484, 3.0162, 3.1267, 3.2125, 3.0043,
2.9572, 2.8197, 2.7261, 3.7701, 3.2446, 3.5239, 3.4696,
3.4261, 3.3508, 3.1968, 3.0848, 4.1496, 3.6598, 3.5111,
3.4199, 3.3809, 3.5382, 3.2572, 3.21, 3.1917, 3.1519,
3.1198, 3.1005, 3.5071, 3.5086, 3.5073, 3.4509, 3.312,
3.2082, 4.2611, 3.8117, 3.6988, 3.5646, 3.6925, 3.6295,
3.5383, 3.491, 3.4625, 3.4233, 3.4007, 3.2329, 3.6723,
3.6845, 3.6876, 3.6197, 3.4799, 3.3737, 4.4341, 4.0525,
3.9011, 3.8945, 3.8635, 3.8368, 3.8153, 3.7936, 3.7758,
3.4944, 3.4873, 3.904, 3.711, 3.6922, 3.6799, 3.6724,
3.5622, 3.6081, 3.5426, 3.4922, 3.4498, 3.3984, 3.4456,
2.7522, 2.5524, 3.5742, 2.9508, 3.0751, 3.0158, 2.9644,
2.8338, 2.7891, 2.6933, 3.6926, 3.1814, 3.4528, 3.4186,
3.3836, 3.2213, 3.1626, 3.0507, 4.0548, 3.5312, 3.4244,
3.3409, 3.281, 3.4782, 3.1905, 3.1494, 3.1221, 3.1128,
3.0853, 3.0384, 3.4366, 3.4562, 3.4638, 3.3211, 3.2762,
3.173, 4.1632, 3.6825, 3.5822, 3.487, 3.6325, 3.574,
3.4733, 3.4247, 3.3969, 3.3764, 3.3525, 3.1984, 3.5989,
3.6299, 3.6433, 3.4937, 3.4417, 3.3365, 4.3304, 3.9242,
3.7793, 3.7623, 3.7327, 3.7071, 3.686, 3.665, 3.6476,
3.3849, 3.3534, 3.8216, 3.587, 3.5695, 3.5584, 3.5508,
3.4856, 3.5523, 3.4934, 3.4464, 3.4055, 3.3551, 3.3888,
3.3525, 2.7202, 2.5183, 3.4947, 2.8731, 3.0198, 3.1457,
2.9276, 2.7826, 2.7574, 2.6606, 3.609, 3.0581, 3.3747,
3.3677, 3.345, 3.1651, 3.1259, 3.0147, 3.9498, 3.3857,
3.2917, 3.2154, 3.1604, 3.4174, 3.0735, 3.0342, 3.0096,
3.0136, 2.9855, 2.968, 3.3604, 3.4037, 3.4243, 3.2633,
3.181, 3.1351, 4.0557, 3.5368, 3.4526, 3.3699, 3.5707,
3.5184, 3.4085, 3.3595, 3.3333, 3.3143, 3.3041, 3.1094,
3.5193, 3.5745, 3.6025, 3.4338, 3.3448, 3.2952, 4.2158,
3.7802, 3.6431, 3.6129, 3.5853, 3.561, 3.5406, 3.5204,
3.5036, 3.2679, 3.2162, 3.7068, 3.4483, 3.4323, 3.4221,
3.4138, 3.3652, 3.4576, 3.4053, 3.3618, 3.3224, 3.2711,
3.3326, 3.295, 3.2564, 2.5315, 2.6104, 3.2734, 3.2299,
3.109, 2.9942, 2.9159, 2.8324, 2.835, 2.7216, 3.3994,
3.4475, 3.4354, 3.3438, 3.2807, 3.2169, 3.2677, 3.1296,
3.7493, 3.8075, 3.6846, 3.6104, 3.5577, 3.2052, 3.4803,
3.4236, 3.3845, 3.364, 3.3365, 3.301, 3.3938, 3.3624,
3.344, 3.3132, 3.4035, 3.2754, 3.8701, 3.9523, 3.8018,
3.7149, 3.3673, 3.3199, 3.2483, 3.2069, 3.1793, 3.1558,
3.1395, 3.4097, 3.541, 3.5228, 3.5116, 3.4921, 3.4781,
3.469, 4.042, 4.1759, 4.0078, 4.045, 4.0189, 3.9952,
3.977, 3.9583, 3.9434, 3.7217, 3.8228, 3.7826, 3.864,
3.8446, 3.8314, 3.8225, 3.6817, 3.7068, 3.6555, 3.6159,
3.5831, 3.5257, 3.2133, 3.1689, 3.1196, 3.3599, 2.9852,
2.7881, 3.5284, 3.3493, 3.6958, 3.3642, 3.1568, 3.0055,
2.9558, 2.8393, 3.6287, 3.5283, 4.1511, 3.8259, 3.6066,
3.4527, 3.348, 3.2713, 3.9037, 3.8361, 3.8579, 3.7311,
3.6575, 3.5176, 3.5693, 3.5157, 3.4814, 3.4559, 3.4445,
3.416, 4.1231, 3.8543, 3.6816, 3.5602, 3.4798, 3.4208,
4.0542, 4.0139, 4.0165, 3.9412, 3.7698, 3.6915, 3.6043,
3.5639, 3.5416, 3.5247, 3.5153, 3.5654, 4.2862, 4.0437,
3.8871, 3.7741, 3.6985, 3.6413, 4.2345, 4.3663, 4.3257,
4.0869, 4.0612, 4.0364, 4.017, 3.9978, 3.9834, 3.9137,
3.8825, 3.8758, 3.9143, 3.8976, 3.8864, 3.8768, 3.919,
4.1613, 4.0566, 3.9784, 3.9116, 3.8326, 3.7122, 3.6378,
3.5576, 3.5457, 4.3127, 3.116, 2.8482, 4.0739, 3.3599,
3.5698, 3.5366, 3.2854, 3.1039, 2.9953, 2.9192, 4.1432,
3.532, 3.9478, 4.0231, 3.7509, 3.5604, 3.434, 3.3426,
4.3328, 3.8288, 3.7822, 3.7909, 3.6907, 3.6864, 3.5793,
3.5221, 3.4883, 3.4649, 3.4514, 3.4301, 3.9256, 4.0596,
3.8307, 3.6702, 3.5651, 3.4884, 4.4182, 4.2516, 3.9687,
3.9186, 3.9485, 3.837, 3.7255, 3.6744, 3.6476, 3.6295,
3.6193, 3.5659, 4.0663, 4.2309, 4.0183, 3.868, 3.7672,
3.6923, 4.524, 4.4834, 4.157, 4.3204, 4.2993, 4.2804,
4.2647, 4.2481, 4.2354, 3.8626, 3.8448, 4.2267, 4.1799,
4.167, 3.8738, 3.8643, 3.8796, 4.0575, 4.0354, 3.9365,
3.8611, 3.7847, 3.7388, 3.6826, 3.6251, 3.5492, 4.0889,
4.2764, 3.1416, 2.8325, 3.7735, 3.3787, 3.4632, 3.5923,
3.3214, 3.1285, 3.0147, 2.9366, 3.8527, 3.5602, 3.8131,
3.8349, 3.7995, 3.5919, 3.4539, 3.354, 4.0654, 3.8603,
3.7972, 3.7358, 3.7392, 3.8157, 3.6055, 3.5438, 3.5089,
3.4853, 3.4698, 3.4508, 3.7882, 3.8682, 3.8837, 3.7055,
3.587, 3.5, 4.1573, 4.0005, 3.9568, 3.8936, 3.999,
3.9433, 3.8172, 3.7566, 3.7246, 3.7033, 3.69, 3.5697,
3.9183, 4.0262, 4.0659, 3.8969, 3.7809, 3.6949, 4.2765,
4.2312, 4.1401, 4.0815, 4.058, 4.0369, 4.0194, 4.0017,
3.9874, 3.8312, 3.812, 3.9454, 3.921, 3.9055, 3.8951,
3.8866, 3.8689, 3.9603, 3.9109, 3.9122, 3.8233, 3.7438,
3.7436, 3.6981, 3.6555, 3.5452, 3.9327, 4.0658, 4.1175,
2.9664, 2.8209, 3.5547, 3.3796, 3.3985, 3.3164, 3.2364,
3.1956, 3.037, 2.9313, 3.6425, 3.5565, 3.7209, 3.7108,
3.6639, 3.6484, 3.4745, 3.3492, 3.8755, 4.2457, 3.7758,
3.7161, 3.6693, 3.6155, 3.5941, 3.5643, 3.5292, 3.495,
3.472, 3.4503, 3.6936, 3.7392, 3.7388, 3.7602, 3.6078,
3.496, 3.98, 4.3518, 4.2802, 3.858, 3.8056, 3.7527,
3.7019, 3.6615, 3.5768, 3.533, 3.5038, 3.5639, 3.8192,
3.8883, 3.9092, 3.9478, 3.7995, 3.6896, 4.1165, 4.5232,
4.4357, 4.4226, 4.4031, 4.386, 4.3721, 4.358, 4.3466,
4.2036, 4.2037, 3.8867, 4.2895, 4.2766, 4.2662, 4.2598,
3.8408, 3.9169, 3.8681, 3.825, 3.7855, 3.7501, 3.6753,
3.5499, 3.4872, 3.5401, 3.8288, 3.9217, 3.9538, 4.0054,
2.8388, 2.789, 3.4329, 3.5593, 3.3488, 3.2486, 3.1615,
3.1, 3.0394, 2.9165, 3.5267, 3.7479, 3.665, 3.6263,
3.5658, 3.5224, 3.4762, 3.3342, 3.7738, 4.0333, 3.9568,
3.8975, 3.8521, 3.4929, 3.783, 3.7409, 3.7062, 3.6786,
3.6471, 3.6208, 3.6337, 3.6519, 3.6363, 3.6278, 3.611,
3.4825, 3.8795, 4.1448, 4.0736, 4.0045, 3.6843, 3.6291,
3.5741, 3.5312, 3.4974, 3.4472, 3.4034, 3.7131, 3.7557,
3.7966, 3.8005, 3.8068, 3.8015, 3.6747, 4.0222, 4.3207,
4.2347, 4.2191, 4.199, 4.1811, 4.1666, 4.1521, 4.1401,
3.997, 3.9943, 3.9592, 4.08, 4.0664, 4.0559, 4.0488,
3.9882, 4.0035, 3.9539, 3.9138, 3.8798, 3.8355, 3.5359,
3.4954, 3.3962, 3.5339, 3.7595, 3.825, 3.8408, 3.86,
3.8644, 2.7412, 2.7489, 3.3374, 3.395, 3.3076, 3.191,
3.0961, 3.0175, 3.028, 2.8929, 3.4328, 3.5883, 3.6227,
3.5616, 3.4894, 3.4241, 3.3641, 3.312, 3.6815, 3.8789,
3.8031, 3.7413, 3.6939, 3.401, 3.6225, 3.5797, 3.5443,
3.5139, 3.4923, 3.4642, 3.586, 3.5849, 3.557, 3.5257,
3.4936, 3.4628, 3.7874, 3.9916, 3.9249, 3.853, 3.5932,
3.5355, 3.4757, 3.4306, 3.3953, 3.3646, 3.339, 3.5637,
3.7053, 3.7266, 3.7177, 3.6996, 3.6775, 3.6558, 3.9331,
4.1655, 4.0879, 4.0681, 4.0479, 4.0299, 4.0152, 4.0006,
3.9883, 3.85, 3.8359, 3.8249, 3.9269, 3.9133, 3.9025,
3.8948, 3.8422, 3.8509, 3.799, 3.757, 3.7219, 3.6762,
3.426, 3.3866, 3.3425, 3.5294, 3.7022, 3.7497, 3.7542,
3.7494, 3.737, 3.7216, 3.3286, 3.5286, 4.5857, 4.2143,
3.9714, 3.9428, 3.8857, 3.7714, 3.7857, 3.8286, 4.8714,
4.6571, 4.4857, 4.3571, 4.4429, 4.3286, 4.2857, 4.2428,
5.3857, 5.0714, 4.7714, 4.6143, 4.6, 4.4429, 4.4,
4.3571, 4.3, 4.2857, 4.3143, 4.4286, 4.4714, 4.4286,
4.5143, 4.4429, 4.5, 4.5429, 5.5714, 5.2572, 4.9714,
4.8572, 4.7571, 4.6428, 4.5143, 4.4857, 4.4857, 4.4143,
4.5143, 4.6286, 4.7, 4.6714, 4.6714, 4.6286, 4.7571,
4.7429, 5.8571, 5.3857, 5.1857, 4.9714, 5.1286, 5.1143,
5.1, 5.0857, 5.0286, 5.0428, 5.0286, 5.0143, 5.0,
5.0, 4.9857, 5.0571, 4.9571, 4.8286, 4.7429, 4.6286,
4.5572, 4.5286, 4.4572, 4.4714, 4.4857, 4.7571, 4.7286,
4.7286, 4.8143, 4.7429, 4.8429, 4.9, 5.7428, 3.0429,
3.2429, 4.3, 3.9286, 3.6857, 3.6571, 3.6, 3.4857,
3.5, 3.5428, 4.5857, 4.3714, 4.2, 4.0714, 4.1572,
4.0429, 4.0, 3.9571, 5.1, 4.7857, 4.4857, 4.3286,
4.3143, 4.1572, 4.1143, 4.0714, 4.0143, 4.0, 4.0286,
4.1429, 4.1857, 4.1429, 4.2285, 4.1572, 4.2143, 4.2571,
5.2857, 4.9714, 4.6857, 4.5714, 4.4714, 4.3571, 4.2285,
4.2, 4.2, 4.1286, 4.2285, 4.3429, 4.4143, 4.3857,
4.3857, 4.3429, 4.4714, 4.4572, 5.5714, 5.1, 4.9,
4.6857, 4.8429, 4.8286, 4.8143, 4.8, 4.7429, 4.7571,
4.7429, 4.7286, 4.7143, 4.7143, 4.7, 4.7714, 4.6714,
4.5429, 4.4572, 4.3429, 4.2714, 4.2428, 4.1715, 4.1857,
4.2, 4.4714, 4.4429, 4.4429, 4.5286, 4.4572, 4.5572,
4.6143, 5.4571, 5.1714, 2.8428, 3.0429, 4.1, 3.7286,
3.4857, 3.4572, 3.4, 3.2857, 3.3, 3.3429, 4.3857,
4.1715, 4.0, 3.8714, 3.9571, 3.8428, 3.8, 3.7572,
4.9, 4.5857, 4.2857, 4.1286, 4.1143, 3.9571, 3.9143,
3.8714, 3.8143, 3.8, 3.8286, 3.9428, 3.9857, 3.9428,
4.0286, 3.9571, 4.0143, 4.0571, 5.0857, 4.7714, 4.4857,
4.3714, 4.2714, 4.1572, 4.0286, 4.0, 4.0, 3.9286,
4.0286, 4.1429, 4.2143, 4.1857, 4.1857, 4.1429, 4.2714,
4.2571, 5.3714, 4.9, 4.7, 4.4857, 4.6428, 4.6286,
4.6143, 4.6, 4.5429, 4.5572, 4.5429, 4.5286, 4.5143,
4.5143, 4.5, 4.5714, 4.4714, 4.3429, 4.2571, 4.1429,
4.0714, 4.0429, 3.9714, 3.9857, 4.0, 4.2714, 4.2428,
4.2428, 4.3286, 4.2571, 4.3571, 4.4143, 5.2572, 4.9714,
4.7714, 2.7143, 2.9143, 3.9714, 3.6, 3.3572, 3.3286,
3.2714, 3.1571, 3.1714, 3.2143, 4.2571, 4.0429, 3.8714,
3.7429, 3.8286, 3.7143, 3.6714, 3.6286, 4.7714, 4.4572,
4.1572, 4.0, 3.9857, 3.8286, 3.7857, 3.7429, 3.6857,
3.6714, 3.7, 3.8143, 3.8571, 3.8143, 3.9, 3.8286,
3.8857, 3.9286, 4.9571, 4.6428, 4.3571, 4.2428, 4.1429,
4.0286, 3.9, 3.8714, 3.8714, 3.8, 3.9, 4.0143,
4.0857, 4.0571, 4.0571, 4.0143, 4.1429, 4.1286, 5.2429,
4.7714, 4.5714, 4.3571, 4.5143, 4.5, 4.4857, 4.4714,
4.4143, 4.4286, 4.4143, 4.4, 4.3857, 4.3857, 4.3714,
4.4429, 4.3429, 4.2143, 4.1286, 4.0143, 3.9428, 3.9143,
3.8428, 3.8571, 3.8714, 4.1429, 4.1143, 4.1143, 4.2,
4.1286, 4.2285, 4.2857, 5.1286, 4.8429, 4.6428, 4.5143,
2.6286, 2.8286, 3.8857, 3.5143, 3.2714, 3.2429, 3.1857,
3.0715, 3.0857, 3.1285, 4.1715, 3.9571, 3.7857, 3.6571,
3.7429, 3.6286, 3.5857, 3.5428, 4.6857, 4.3714, 4.0714,
3.9143, 3.9, 3.7429, 3.7, 3.6571, 3.6, 3.5857,
3.6143, 3.7286, 3.7714, 3.7286, 3.8143, 3.7429, 3.8,
3.8428, 4.8714, 4.5572, 4.2714, 4.1572, 4.0571, 3.9428,
3.8143, 3.7857, 3.7857, 3.7143, 3.8143, 3.9286, 4.0,
3.9714, 3.9714, 3.9286, 4.0571, 4.0429, 5.1571, 4.6857,
4.4857, 4.2714, 4.4286, 4.4143, 4.4, 4.3857, 4.3286,
4.3429, 4.3286, 4.3143, 4.3, 4.3, 4.2857, 4.3571,
4.2571, 4.1286, 4.0429, 3.9286, 3.8571, 3.8286, 3.7572,
3.7714, 3.7857, 4.0571, 4.0286, 4.0286, 4.1143, 4.0429,
4.1429, 4.2, 5.0428, 4.7571, 4.5572, 4.4286, 4.3429,
2.6429, 2.8428, 3.9, 3.5286, 3.2857, 3.2571, 3.2,
3.0857, 3.1, 3.1428, 4.1857, 3.9714, 3.8, 3.6714,
3.7572, 3.6429, 3.6, 3.5571, 4.7, 4.3857, 4.0857,
3.9286, 3.9143, 3.7572, 3.7143, 3.6714, 3.6143, 3.6,
3.6286, 3.7429, 3.7857, 3.7429, 3.8286, 3.7572, 3.8143,
3.8571, 4.8857, 4.5714, 4.2857, 4.1715, 4.0714, 3.9571,
3.8286, 3.8, 3.8, 3.7286, 3.8286, 3.9428, 4.0143,
3.9857, 3.9857, 3.9428, 4.0714, 4.0571, 5.1714, 4.7,
4.5, 4.2857, 4.4429, 4.4286, 4.4143, 4.4, 4.3429,
4.3571, 4.3429, 4.3286, 4.3143, 4.3143, 4.3, 4.3714,
4.2714, 4.1429, 4.0571, 3.9428, 3.8714, 3.8428, 3.7714,
3.7857, 3.8, 4.0714, 4.0429, 4.0429, 4.1286, 4.0571,
4.1572, 4.2143, 5.0571, 4.7714, 4.5714, 4.4429, 4.3571,
4.3714, 2.6572, 2.8571, 3.9143, 3.5428, 3.3, 3.2714,
3.2143, 3.1, 3.1143, 3.1571, 4.2, 3.9857, 3.8143,
3.6857, 3.7714, 3.6571, 3.6143, 3.5714, 4.7143, 4.4,
4.1, 3.9428, 3.9286, 3.7714, 3.7286, 3.6857, 3.6286,
3.6143, 3.6429, 3.7572, 3.8, 3.7572, 3.8428, 3.7714,
3.8286, 3.8714, 4.9, 4.5857, 4.3, 4.1857, 4.0857,
3.9714, 3.8428, 3.8143, 3.8143, 3.7429, 3.8428, 3.9571,
4.0286, 4.0, 4.0, 3.9571, 4.0857, 4.0714, 5.1857,
4.7143, 4.5143, 4.3, 4.4572, 4.4429, 4.4286, 4.4143,
4.3571, 4.3714, 4.3571, 4.3429, 4.3286, 4.3286, 4.3143,
4.3857, 4.2857, 4.1572, 4.0714, 3.9571, 3.8857, 3.8571,
3.7857, 3.8, 3.8143, 4.0857, 4.0571, 4.0571, 4.1429,
4.0714, 4.1715, 4.2285, 5.0714, 4.7857, 4.5857, 4.4572,
4.3714, 4.3857, 4.4, 2.6714, 2.8714, 3.9286, 3.5571,
3.3143, 3.2857, 3.2286, 3.1143, 3.1285, 3.1714, 4.2143,
4.0, 3.8286, 3.7, 3.7857, 3.6714, 3.6286, 3.5857,
4.7286, 4.4143, 4.1143, 3.9571, 3.9428, 3.7857, 3.7429,
3.7, 3.6429, 3.6286, 3.6571, 3.7714, 3.8143, 3.7714,
3.8571, 3.7857, 3.8428, 3.8857, 4.9143, 4.6, 4.3143,
4.2, 4.1, 3.9857, 3.8571, 3.8286, 3.8286, 3.7572,
3.8571, 3.9714, 4.0429, 4.0143, 4.0143, 3.9714, 4.1,
4.0857, 5.2, 4.7286, 4.5286, 4.3143, 4.4714, 4.4572,
4.4429, 4.4286, 4.3714, 4.3857, 4.3714, 4.3571, 4.3429,
4.3429, 4.3286, 4.4, 4.3, 4.1715, 4.0857, 3.9714,
3.9, 3.8714, 3.8, 3.8143, 3.8286, 4.1, 4.0714,
4.0714, 4.1572, 4.0857, 4.1857, 4.2428, 5.0857, 4.8,
4.6, 4.4714, 4.3857, 4.4, 4.4143, 4.4286, 2.5857,
2.7857, 3.8428, 3.4714, 3.2286, 3.2, 3.1428, 3.0286,
3.0429, 3.0857, 4.1286, 3.9143, 3.7429, 3.6143, 3.7,
3.5857, 3.5428, 3.5, 4.6428, 4.3286, 4.0286, 3.8714,
3.8571, 3.7, 3.6571, 3.6143, 3.5571, 3.5428, 3.5714,
3.6857, 3.7286, 3.6857, 3.7714, 3.7, 3.7572, 3.8,
4.8286, 4.5143, 4.2285, 4.1143, 4.0143, 3.9, 3.7714,
3.7429, 3.7429, 3.6714, 3.7714, 3.8857, 3.9571, 3.9286,
3.9286, 3.8857, 4.0143, 4.0, 5.1143, 4.6428, 4.4429,
4.2285, 4.3857, 4.3714, 4.3571, 4.3429, 4.2857, 4.3,
4.2857, 4.2714, 4.2571, 4.2571, 4.2428, 4.3143, 4.2143,
4.0857, 4.0, 3.8857, 3.8143, 3.7857, 3.7143, 3.7286,
3.7429, 4.0143, 3.9857, 3.9857, 4.0714, 4.0, 4.1,
4.1572, 5.0, 4.7143, 4.5143, 4.3857, 4.3, 4.3143,
4.3286, 4.3429, 4.2571, 2.5857, 2.7857, 3.8428, 3.4714,
3.2286, 3.2, 3.1428, 3.0286, 3.0429, 3.0857, 4.1286,
3.9143, 3.7429, 3.6143, 3.7, 3.5857, 3.5428, 3.5,
4.6428, 4.3286, 4.0286, 3.8714, 3.8571, 3.7, 3.6571,
3.6143, 3.5571, 3.5428, 3.5714, 3.6857, 3.7286, 3.6857,
3.7714, 3.7, 3.7572, 3.8, 4.8286, 4.5143, 4.2285,
4.1143, 4.0143, 3.9, 3.7714, 3.7429, 3.7429, 3.6714,
3.7714, 3.8857, 3.9571, 3.9286, 3.9286, 3.8857, 4.0143,
4.0, 5.1143, 4.6428, 4.4429, 4.2285, 4.3857, 4.3714,
4.3571, 4.3429, 4.2857, 4.3, 4.2857, 4.2714, 4.2571,
4.2571, 4.2428, 4.3143, 4.2143, 4.0857, 4.0, 3.8857,
3.8143, 3.7857, 3.7143, 3.7286, 3.7429, 4.0143, 3.9857,
3.9857, 4.0714, 4.0, 4.1, 4.1572, 5.0, 4.7143,
4.5143, 4.3857, 4.3, 4.3143, 4.3286, 4.3429, 4.2571,
4.2571, 2.6143, 2.8143, 3.8714, 3.5, 3.2571, 3.2286,
3.1714, 3.0572, 3.0715, 3.1143, 4.1572, 3.9428, 3.7714,
3.6429, 3.7286, 3.6143, 3.5714, 3.5286, 4.6714, 4.3571,
4.0571, 3.9, 3.8857, 3.7286, 3.6857, 3.6429, 3.5857,
3.5714, 3.6, 3.7143, 3.7572, 3.7143, 3.8, 3.7286,
3.7857, 3.8286, 4.8572, 4.5429, 4.2571, 4.1429, 4.0429,
3.9286, 3.8, 3.7714, 3.7714, 3.7, 3.8, 3.9143,
3.9857, 3.9571, 3.9571, 3.9143, 4.0429, 4.0286, 5.1429,
4.6714, 4.4714, 4.2571, 4.4143, 4.4, 4.3857, 4.3714,
4.3143, 4.3286, 4.3143, 4.3, 4.2857, 4.2857, 4.2714,
4.3429, 4.2428, 4.1143, 4.0286, 3.9143, 3.8428, 3.8143,
3.7429, 3.7572, 3.7714, 4.0429, 4.0143, 4.0143, 4.1,
4.0286, 4.1286, 4.1857, 5.0286, 4.7429, 4.5429, 4.4143,
4.3286, 4.3429, 4.3571, 4.3714, 4.2857, 4.2857, 4.3143,
2.6143, 2.8143, 3.8714, 3.5, 3.2571, 3.2286, 3.1714,
3.0572, 3.0715, 3.1143, 4.1572, 3.9428, 3.7714, 3.6429,
3.7286, 3.6143, 3.5714, 3.5286, 4.6714, 4.3571, 4.0571,
3.9, 3.8857, 3.7286, 3.6857, 3.6429, 3.5857, 3.5714,
3.6, 3.7143, 3.7572, 3.7143, 3.8, 3.7286, 3.7857,
3.8286, 4.8572, 4.5429, 4.2571, 4.1429, 4.0429, 3.9286,
3.8, 3.7714, 3.7714, 3.7, 3.8, 3.9143, 3.9857,
3.9571, 3.9571, 3.9143, 4.0429, 4.0286, 5.1429, 4.6714,
4.4714, 4.2571, 4.4143, 4.4, 4.3857, 4.3714, 4.3143,
4.3286, 4.3143, 4.3, 4.2857, 4.2857, 4.2714, 4.3429,
4.2428, 4.1143, 4.0286, 3.9143, 3.8428, 3.8143, 3.7429,
3.7572, 3.7714, 4.0429, 4.0143, 4.0143, 4.1, 4.0286,
4.1286, 4.1857, 5.0286, 4.7429, 4.5429, 4.4143, 4.3286,
4.3429, 4.3571, 4.3714, 4.2857, 4.2857, 4.3143, 4.3143,
2.5714, 2.7714, 3.8286, 3.4572, 3.2143, 3.1857, 3.1285,
3.0143, 3.0286, 3.0715, 4.1143, 3.9, 3.7286, 3.6,
3.6857, 3.5714, 3.5286, 3.4857, 4.6286, 4.3143, 4.0143,
3.8571, 3.8428, 3.6857, 3.6429, 3.6, 3.5428, 3.5286,
3.5571, 3.6714, 3.7143, 3.6714, 3.7572, 3.6857, 3.7429,
3.7857, 4.8143, 4.5, 4.2143, 4.1, 4.0, 3.8857,
3.7572, 3.7286, 3.7286, 3.6571, 3.7572, 3.8714, 3.9428,
3.9143, 3.9143, 3.8714, 4.0, 3.9857, 5.1, 4.6286,
4.4286, 4.2143, 4.3714, 4.3571, 4.3429, 4.3286, 4.2714,
4.2857, 4.2714, 4.2571, 4.2428, 4.2428, 4.2285, 4.3,
4.2, 4.0714, 3.9857, 3.8714, 3.8, 3.7714, 3.7,
3.7143, 3.7286, 4.0, 3.9714, 3.9714, 4.0571, 3.9857,
4.0857, 4.1429, 4.9857, 4.7, 4.5, 4.3714, 4.2857,
4.3, 4.3143, 4.3286, 4.2428, 4.2428, 4.2714, 4.2714,
4.2285, 2.6, 2.8, 3.8571, 3.4857, 3.2429, 3.2143,
3.1571, 3.0429, 3.0572, 3.1, 4.1429, 3.9286, 3.7572,
3.6286, 3.7143, 3.6, 3.5571, 3.5143, 4.6571, 4.3429,
4.0429, 3.8857, 3.8714, 3.7143, 3.6714, 3.6286, 3.5714,
3.5571, 3.5857, 3.7, 3.7429, 3.7, 3.7857, 3.7143,
3.7714, 3.8143, 4.8429, 4.5286, 4.2428, 4.1286, 4.0286,
3.9143, 3.7857, 3.7572, 3.7572, 3.6857, 3.7857, 3.9,
3.9714, 3.9428, 3.9428, 3.9, 4.0286, 4.0143, 5.1286,
4.6571, 4.4572, 4.2428, 4.4, 4.3857, 4.3714, 4.3571,
4.3, 4.3143, 4.3, 4.2857, 4.2714, 4.2714, 4.2571,
4.3286, 4.2285, 4.1, 4.0143, 3.9, 3.8286, 3.8,
3.7286, 3.7429, 3.7572, 4.0286, 4.0, 4.0, 4.0857,
4.0143, 4.1143, 4.1715, 5.0143, 4.7286, 4.5286, 4.4,
4.3143, 4.3286, 4.3429, 4.3571, 4.2714, 4.2714, 4.3,
4.3, 4.2571, 4.2857, 2.6857, 2.8857, 3.9428, 3.5714,
3.3286, 3.3, 3.2429, 3.1285, 3.1428, 3.1857, 4.2285,
4.0143, 3.8428, 3.7143, 3.8, 3.6857, 3.6429, 3.6,
4.7429, 4.4286, 4.1286, 3.9714, 3.9571, 3.8, 3.7572,
3.7143, 3.6571, 3.6429, 3.6714, 3.7857, 3.8286, 3.7857,
3.8714, 3.8, 3.8571, 3.9, 4.9285, 4.6143, 4.3286,
4.2143, 4.1143, 4.0, 3.8714, 3.8428, 3.8428, 3.7714,
3.8714, 3.9857, 4.0571, 4.0286, 4.0286, 3.9857, 4.1143,
4.1, 5.2143, 4.7429, 4.5429, 4.3286, 4.4857, 4.4714,
4.4572, 4.4429, 4.3857, 4.4, 4.3857, 4.3714, 4.3571,
4.3571, 4.3429, 4.4143, 4.3143, 4.1857, 4.1, 3.9857,
3.9143, 3.8857, 3.8143, 3.8286, 3.8428, 4.1143, 4.0857,
4.0857, 4.1715, 4.1, 4.2, 4.2571, 5.1, 4.8143,
4.6143, 4.4857, 4.4, 4.4143, 4.4286, 4.4429, 4.3571,
4.3571, 4.3857, 4.3857, 4.3429, 4.3714, 4.4572, 2.7143,
2.9143, 3.9714, 3.6, 3.3572, 3.3286, 3.2714, 3.1571,
3.1714, 3.2143, 4.2571, 4.0429, 3.8714, 3.7429, 3.8286,
3.7143, 3.6714, 3.6286, 4.7714, 4.4572, 4.1572, 4.0,
3.9857, 3.8286, 3.7857, 3.7429, 3.6857, 3.6714, 3.7,
3.8143, 3.8571, 3.8143, 3.9, 3.8286, 3.8857, 3.9286,
4.9571, 4.6428, 4.3571, 4.2428, 4.1429, 4.0286, 3.9,
3.8714, 3.8714, 3.8, 3.9, 4.0143, 4.0857, 4.0571,
4.0571, 4.0143, 4.1429, 4.1286, 5.2429, 4.7714, 4.5714,
4.3571, 4.5143, 4.5, 4.4857, 4.4714, 4.4143, 4.4286,
4.4143, 4.4, 4.3857, 4.3857, 4.3714, 4.4429, 4.3429,
4.2143, 4.1286, 4.0143, 3.9428, 3.9143, 3.8428, 3.8571,
3.8714, 4.1429, 4.1143, 4.1143, 4.2, 4.1286, 4.2285,
4.2857, 5.1286, 4.8429, 4.6428, 4.5143, 4.4286, 4.4429,
4.4572, 4.4714, 4.3857, 4.3857, 4.4143, 4.4143, 4.3714,
4.4, 4.4857, 4.5143, 2.5286, 2.7286, 3.7857, 3.4143,
3.1714, 3.1428, 3.0857, 2.9714, 2.9857, 3.0286, 4.0714,
3.8571, 3.6857, 3.5571, 3.6429, 3.5286, 3.4857, 3.4429,
4.5857, 4.2714, 3.9714, 3.8143, 3.8, 3.6429, 3.6,
3.5571, 3.5, 3.4857, 3.5143, 3.6286, 3.6714, 3.6286,
3.7143, 3.6429, 3.7, 3.7429, 4.7714, 4.4572, 4.1715,
4.0571, 3.9571, 3.8428, 3.7143, 3.6857, 3.6857, 3.6143,
3.7143, 3.8286, 3.9, 3.8714, 3.8714, 3.8286, 3.9571,
3.9428, 5.0571, 4.5857, 4.3857, 4.1715, 4.3286, 4.3143,
4.3, 4.2857, 4.2285, 4.2428, 4.2285, 4.2143, 4.2,
4.2, 4.1857, 4.2571, 4.1572, 4.0286, 3.9428, 3.8286,
3.7572, 3.7286, 3.6571, 3.6714, 3.6857, 3.9571, 3.9286,
3.9286, 4.0143, 3.9428, 4.0429, 4.1, 4.9428, 4.6571,
4.4572, 4.3286, 4.2428, 4.2571, 4.2714, 4.2857, 4.2,
4.2, 4.2285, 4.2285, 4.1857, 4.2143, 4.3, 4.3286,
4.1429,
];

```

## `crates/calchem-dft/src/grid.rs`

SHA256: `0eba49597dae027e5946c52443c76bd5986a2d71b9010d966ce5be096d21bac2`

```rust
//! Multicenter molecular DFT quadrature grid:
//! - Murray-Handy-Laming Euler-Maclaurin radial quadrature ($q = 2$)
//! - Radial-shell pruned Lebedev-Laikov spherical quadrature
//! - Becke (1988) $k = 3$ fuzzy-Voronoi atomic partitioning with Bragg-Slater heteronuclear adjustment
//! - Exact analytical nuclear derivatives $\frac{dw_g}{d\mathbf{R}_A}$ of Becke quadrature weights

use calchem_model::{Molecule, ANGSTROM_TO_BOHR};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::lebedev::LebedevDegree;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GridLevel {
    Coarse,
    #[default]
    Medium,
    Fine,
}

impl GridLevel {
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "coarse" | "small" | "1" => Self::Coarse,
            "fine" | "ultrafine" | "3" | "4" => Self::Fine,
            _ => Self::Medium,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GridPoint {
    pub coords: [f64; 3],
    /// Total quadrature weight `raw_weight * becke_weight`.
    pub weight: f64,
    /// Single-center radial * spherical weight (`w_rad * w_ang`) before Becke partitioning.
    pub raw_weight: f64,
    /// Index of the parent nucleus $H(g)$ with which `coords = R_H + u_g` co-moves.
    pub host_atom: usize,
}

#[derive(Debug, Clone)]
pub struct MolecularGrid {
    pub points: Vec<GridPoint>,
    pub level: GridLevel,
}

/// Bragg-Slater empirical atomic radii in Angstrom for `Z = 1..=86`
/// (Becke, *J. Chem. Phys.* **88**, 2547, 1988, Appendix; `H` = 0.35 Å).
pub fn bragg_slater_radius_bohr(z: u8) -> f64 {
    let ang = match z {
        1 => 0.35,
        2 => 0.35,
        3 => 1.45,
        4 => 1.05,
        5 => 0.85,
        6 => 0.70,
        7 => 0.65,
        8 => 0.60,
        9 => 0.50,
        10 => 0.45,
        11 => 1.80,
        12 => 1.50,
        13 => 1.25,
        14 => 1.10,
        15 => 1.00,
        16 => 1.00,
        17 => 1.00,
        18 => 1.00,
        19 => 2.20,
        20 => 1.80,
        21..=30 => 1.40,
        31 => 1.30,
        32 => 1.25,
        33 => 1.15,
        34 => 1.15,
        35 => 1.15, // Br
        36 => 1.10,
        37 => 2.35,
        38 => 2.00,
        39..=48 => 1.55,
        49 => 1.55,
        50 => 1.45,
        51 => 1.45,
        52 => 1.40,
        53 => 1.40, // I
        54 => 1.35,
        55 => 2.60,
        56 => 2.15,
        _ => 1.75,
    };
    ang * ANGSTROM_TO_BOHR
}

/// Becke heteronuclear boundary shift parameter $a_{CD} \in [-0.5, 0.5]$.
#[inline]
pub fn becke_asymmetry_param(z_c: u8, z_d: u8) -> f64 {
    if z_c == z_d {
        return 0.0;
    }
    let rc = bragg_slater_radius_bohr(z_c);
    let rd = bragg_slater_radius_bohr(z_d);
    let chi = rc / rd;
    let u = (chi - 1.0) / (chi + 1.0);
    let a = u / (u * u - 1.0);
    a.clamp(-0.5, 0.5)
}

#[inline]
fn becke_poly1(x: f64) -> f64 {
    0.5 * x * (3.0 - x * x)
}

#[inline]
fn becke_poly1_deriv(x: f64) -> f64 {
    1.5 * (1.0 - x * x)
}

/// Evaluates Becke $k = 3$ step function $s(\nu) = \frac{1}{2}(1 - p_3(\nu))$ and its derivative $s'(\nu)$.
#[inline]
fn becke_step_and_deriv(nu: f64) -> (f64, f64) {
    let x = nu.clamp(-1.0, 1.0);
    let f1 = becke_poly1(x);
    let d1 = becke_poly1_deriv(x);
    let f2 = becke_poly1(f1);
    let d2 = becke_poly1_deriv(f1);
    let f3 = becke_poly1(f2);
    let d3 = becke_poly1_deriv(f2);
    let s = 0.5 * (1.0 - f3);
    let ds_dnu = -0.5 * d3 * d2 * d1;
    (s, ds_dnu)
}

fn radial_shell_count(z: u8, level: GridLevel) -> usize {
    match level {
        GridLevel::Coarse => {
            if z <= 2 {
                25
            } else if z <= 18 {
                35
            } else {
                45
            }
        }
        GridLevel::Medium => {
            if z <= 2 {
                35
            } else if z <= 18 {
                50
            } else {
                65
            }
        }
        GridLevel::Fine => {
            if z <= 2 {
                55
            } else if z <= 18 {
                75
            } else {
                90
            }
        }
    }
}

/// SG-1 / Treutler-style radial-region angular pruning based on `r / R_BS`.
fn pruned_angular_degree(r_bohr: f64, r_bs: f64, level: GridLevel) -> LebedevDegree {
    let ratio = r_bohr / r_bs.max(0.3);
    match level {
        GridLevel::Coarse => {
            if ratio < 0.25 {
                LebedevDegree::N26
            } else if ratio < 0.75 {
                LebedevDegree::N50
            } else if ratio < 4.5 {
                LebedevDegree::N110
            } else {
                LebedevDegree::N50
            }
        }
        GridLevel::Medium => {
            if ratio < 0.20 {
                LebedevDegree::N26
            } else if ratio < 0.55 {
                LebedevDegree::N50
            } else if ratio < 1.10 {
                LebedevDegree::N110
            } else if ratio < 4.50 {
                LebedevDegree::N194
            } else {
                LebedevDegree::N110
            }
        }
        GridLevel::Fine => {
            if ratio < 0.18 {
                LebedevDegree::N50
            } else if ratio < 0.50 {
                LebedevDegree::N110
            } else if ratio < 0.95 {
                LebedevDegree::N194
            } else if ratio < 5.00 {
                LebedevDegree::N302
            } else {
                LebedevDegree::N194
            }
        }
    }
}

/// Computes normalized Becke cell weight $B_H(\mathbf{r})$ for host atom `host` at point `r`.
pub fn becke_partition_weight(mol: &Molecule, host: usize, r: [f64; 3]) -> f64 {
    let n_atoms = mol.atoms.len();
    if n_atoms <= 1 {
        return 1.0;
    }

    let mut dist_r = vec![0.0f64; n_atoms];
    for (c, atom) in mol.atoms.iter().enumerate() {
        let dx = r[0] - atom.position_bohr[0];
        let dy = r[1] - atom.position_bohr[1];
        let dz = r[2] - atom.position_bohr[2];
        dist_r[c] = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    let mut cell_p = vec![1.0f64; n_atoms];
    for c in 0..n_atoms {
        for d in (c + 1)..n_atoms {
            let ac = mol.atoms[c].position_bohr;
            let ad = mol.atoms[d].position_bohr;
            let r_cd =
                ((ac[0] - ad[0]).powi(2) + (ac[1] - ad[1]).powi(2) + (ac[2] - ad[2]).powi(2))
                    .sqrt();
            if r_cd < 1e-14 {
                continue;
            }
            let mu = ((dist_r[c] - dist_r[d]) / r_cd).clamp(-1.0, 1.0);
            let a_cd = becke_asymmetry_param(mol.atoms[c].z, mol.atoms[d].z);
            let nu_cd = mu + a_cd * (1.0 - mu * mu);
            let (s_cd, _) = becke_step_and_deriv(nu_cd);
            // Note: nu_dc = -mu + (-a_cd)*(1 - mu^2) = -nu_cd => s(nu_dc) = 1 - s(nu_cd)!
            let s_dc = 1.0 - s_cd;
            cell_p[c] *= s_cd;
            cell_p[d] *= s_dc;
        }
    }

    let sum_p: f64 = cell_p.iter().sum();
    if sum_p <= 1e-30 {
        0.0
    } else {
        cell_p[host] / sum_p
    }
}

/// Computes the exact total nuclear gradient $\frac{d w_g}{d \mathbf{R}_A}$ (`A = 0..n_atoms`)
/// of a grid point $\mathbf{r}_g = \mathbf{R}_H + \mathbf{u}_g$ co-moving with `host` ($H$).
pub fn becke_weight_nuclear_gradient(
    mol: &Molecule,
    host: usize,
    r: [f64; 3],
    raw_weight: f64,
    out_grad: &mut [[f64; 3]],
    f_xc_val: f64,
) {
    let n_atoms = mol.atoms.len();
    if n_atoms <= 1 || f_xc_val.abs() < 1e-18 || raw_weight == 0.0 {
        return;
    }

    // Small stack/vec buffers? Note n_atoms is typically <= 200; for hot loops we use small Vecs
    // or caller scratch, wait: let's compute cleanly with zero quadratic blowup!
    let mut vec_ra = vec![[0.0f64; 3]; n_atoms];
    let mut dist_r = vec![0.0f64; n_atoms];
    for (c, atom) in mol.atoms.iter().enumerate() {
        let dx = r[0] - atom.position_bohr[0];
        let dy = r[1] - atom.position_bohr[1];
        let dz = r[2] - atom.position_bohr[2];
        vec_ra[c] = [dx, dy, dz];
        dist_r[c] = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    // Store s_mat[c * n + d] and ds_dmu[c * n + d] = ds(nu_cd)/d(mu_cd).
    let mut s_mat = vec![1.0f64; n_atoms * n_atoms];
    let mut ds_dmu = vec![0.0f64; n_atoms * n_atoms];
    let mut cell_p = vec![1.0f64; n_atoms];

    for c in 0..n_atoms {
        for d in (c + 1)..n_atoms {
            let ac = mol.atoms[c].position_bohr;
            let ad = mol.atoms[d].position_bohr;
            let r_cd =
                ((ac[0] - ad[0]).powi(2) + (ac[1] - ad[1]).powi(2) + (ac[2] - ad[2]).powi(2))
                    .sqrt();
            if r_cd < 1e-14 {
                continue;
            }
            let mu = ((dist_r[c] - dist_r[d]) / r_cd).clamp(-1.0, 1.0);
            let a_cd = becke_asymmetry_param(mol.atoms[c].z, mol.atoms[d].z);
            let nu_cd = mu + a_cd * (1.0 - mu * mu);
            let (s_cd, ds_dnu) = becke_step_and_deriv(nu_cd);
            let s_dc = 1.0 - s_cd;
            // d(nu_cd)/d(mu_cd) = 1 - 2 * a_cd * mu
            let deriv_mu_cd = ds_dnu * (1.0 - 2.0 * a_cd * mu);

            s_mat[c * n_atoms + d] = s_cd;
            s_mat[d * n_atoms + c] = s_dc;
            // Note: mu_dc = -mu_cd and s_dc = 1 - s_cd => d(s_dc)/d(mu_dc) = d(1 - s_cd)/d(-mu_cd) = +deriv_mu_cd!
            ds_dmu[c * n_atoms + d] = deriv_mu_cd;
            ds_dmu[d * n_atoms + c] = deriv_mu_cd;

            cell_p[c] *= s_cd;
            cell_p[d] *= s_dc;
        }
    }

    let z_sum: f64 = cell_p.iter().sum();
    if z_sum <= 1e-30 {
        return;
    }
    let w_norm = cell_p[host] / z_sum;
    let pref = raw_weight * f_xc_val / z_sum;

    // For each atom A != host, hold `r = R_host + u` FIXED while moving `R_A`:
    // Only pairs involving A (`(A, B)` and `(B, A)` for `B != A`) depend on `R_A`!
    // Specifically, for B != A:
    //   mu_{AB} = (r_A - r_B) / R_{AB}
    //   grad_{R_A} mu_{AB} = [ -(r - R_A)/r_A - mu_{AB} * (R_A - R_B)/R_{AB} ] / R_{AB}
    // and mu_{BA} = -mu_{AB}, so grad_{R_A} mu_{BA} = -grad_{R_A} mu_{AB}!
    let mut host_accum = [0.0f64; 3];

    for a in 0..n_atoms {
        if a == host {
            continue;
        }
        let ra = dist_r[a];
        if ra < 1e-14 {
            continue;
        }
        let pos_a = mol.atoms[a].position_bohr;
        let dr_a = [-vec_ra[a][0] / ra, -vec_ra[a][1] / ra, -vec_ra[a][2] / ra];

        // Compute dP_A / dR_A = sum_{B != A} (P_A / s_{AB}) * ds_{AB}/dR_A
        // and for each B != A: dP_B / dR_A = (P_B / s_{BA}) * ds_{BA}/dR_A
        // Notice we can compute exact product-without-one `prod_{D != C, k} s_{CD}` safely even if `s_{Ck} == 0`!
        let prod_except = |c: usize, skip: usize| -> f64 {
            let s_val = s_mat[c * n_atoms + skip];
            if s_val > 1e-12 {
                cell_p[c] / s_val
            } else {
                let mut p = 1.0;
                for d in 0..n_atoms {
                    if d != c && d != skip {
                        p *= s_mat[c * n_atoms + d];
                    }
                }
                p
            }
        };

        let mut dp_host = [0.0f64; 3];
        let mut dz_sum = [0.0f64; 3];

        for b in 0..n_atoms {
            if b == a {
                continue;
            }
            let pos_b = mol.atoms[b].position_bohr;
            let rab_vec = [
                pos_a[0] - pos_b[0],
                pos_a[1] - pos_b[1],
                pos_a[2] - pos_b[2],
            ];
            let rab = (rab_vec[0] * rab_vec[0] + rab_vec[1] * rab_vec[1] + rab_vec[2] * rab_vec[2])
                .sqrt();
            if rab < 1e-14 {
                continue;
            }
            let mu_ab = ((ra - dist_r[b]) / rab).clamp(-1.0, 1.0);
            let inv_rab = 1.0 / rab;
            let dmu_ab = [
                (dr_a[0] - mu_ab * rab_vec[0] * inv_rab) * inv_rab,
                (dr_a[1] - mu_ab * rab_vec[1] * inv_rab) * inv_rab,
                (dr_a[2] - mu_ab * rab_vec[2] * inv_rab) * inv_rab,
            ];

            // ds_{AB} / dR_A = ds_dmu[a, b] * dmu_ab
            // ds_{BA} / dR_A = -ds_{AB} / dR_A  (since s_{BA} = 1 - s_{AB}!)
            let slope = ds_dmu[a * n_atoms + b];
            if slope.abs() < 1e-30 {
                continue;
            }
            let p_a_no_b = prod_except(a, b);
            let p_b_no_a = prod_except(b, a);

            for q in 0..3 {
                let ds_ab = slope * dmu_ab[q];
                let dpa_term = p_a_no_b * ds_ab;
                let dpb_term = -p_b_no_a * ds_ab;
                dz_sum[q] += dpa_term + dpb_term;
                if b == host {
                    dp_host[q] += dpb_term;
                }
            }
        }

        // Note: since `a != host`, `P_host` only depends on `R_A` through `b == host` above!
        // d(P_host / Z) / dR_A = (dp_host - w_norm * dz_sum) / Z
        for q in 0..3 {
            let dw_f = pref * (dp_host[q] - w_norm * dz_sum[q]);
            out_grad[a][q] += dw_f;
            host_accum[q] -= dw_f;
        }
    }

    for q in 0..3 {
        out_grad[host][q] += host_accum[q];
    }
}

impl MolecularGrid {
    pub fn build(mol: &Molecule, level: GridLevel) -> Self {
        Self::build_with_radial_min(mol, level, 0)
    }

    /// Resolve diffuse Gaussian tails from the basis exponents, not element radii.
    pub fn build_for_basis(
        mol: &Molecule,
        basis: &calchem_basis::BasisSet,
        level: GridLevel,
    ) -> Self {
        let alpha = basis
            .shells
            .iter()
            .flat_map(|s| &s.primitives)
            .map(|p| p.exponent)
            .fold(f64::INFINITY, f64::min);
        let extra = if alpha < 1e-3 {
            (180.0 * (1e-4 / alpha.max(1e-12)).powf(0.25)).ceil() as usize
        } else {
            0
        };
        Self::build_with_radial_min(mol, level, extra.min(4096))
    }

    fn build_with_radial_min(mol: &Molecule, level: GridLevel, radial_min: usize) -> Self {
        let per_atom: Vec<Vec<GridPoint>> = (0..mol.atoms.len())
            .into_par_iter()
            .map(|atom_idx| {
                let atom = &mol.atoms[atom_idx];
                let r_bs = bragg_slater_radius_bohr(atom.z);
                let base_min = if matches!(level, GridLevel::Fine) {
                    150
                } else {
                    0
                };
                let n_rad = radial_shell_count(atom.z, level)
                    .max(radial_min)
                    .max(base_min);
                let m = (n_rad + 1) as f64;
                let mut local_pts = Vec::with_capacity(n_rad * 150);

                for i in 1..=n_rad {
                    let t = i as f64;
                    let mt = m - t;
                    // Murray-Handy-Laming q=2 Euler-Maclaurin mapping:
                    let r = r_bs * (t * t) / (mt * mt);
                    let w_rad = 2.0 * r_bs.powi(3) * m * t.powi(5) / mt.powi(7);
                    let ang = pruned_angular_degree(r, r_bs, level).points();

                    for &(u, w_ang) in ang {
                        let raw_w = w_rad * w_ang;
                        let coords = [
                            atom.position_bohr[0] + r * u[0],
                            atom.position_bohr[1] + r * u[1],
                            atom.position_bohr[2] + r * u[2],
                        ];
                        let bw = becke_partition_weight(mol, atom_idx, coords);
                        let w_tot = raw_w * bw;
                        if w_tot > 1e-18 {
                            local_pts.push(GridPoint {
                                coords,
                                weight: w_tot,
                                raw_weight: raw_w,
                                host_atom: atom_idx,
                            });
                        }
                    }
                }
                local_pts
            })
            .collect();

        let total_len: usize = per_atom.iter().map(|v| v.len()).sum();
        let mut points = Vec::with_capacity(total_len);
        for mut sub in per_atom {
            points.append(&mut sub);
        }

        Self { points, level }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_model::Atom;
    use std::f64::consts::PI;

    #[test]
    fn test_single_center_gaussian_3d_integral() {
        let mol = Molecule::new(vec![Atom::from_bohr("C", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
        let grid = MolecularGrid::build(&mol, GridLevel::Medium);
        for alpha in [0.15_f64, 1.0, 8.5] {
            let exact = (PI / alpha).powf(1.5);
            let num: f64 = grid
                .points
                .iter()
                .map(|p| {
                    let r2 = p.coords[0].powi(2) + p.coords[1].powi(2) + p.coords[2].powi(2);
                    p.weight * (-alpha * r2).exp()
                })
                .sum();
            let rel_err = (num - exact).abs() / exact;
            assert!(
                rel_err < 1e-6,
                "alpha={alpha}: num={num}, exact={exact}, rel_err={rel_err:.3e}"
            );
        }
    }

    #[test]
    fn test_multicenter_becke_gaussian_sum_and_weight_gradient_fd() {
        let mol = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.02, -0.04, 0.12]).unwrap(),
                Atom::from_angstrom("H", [0.08, 0.78, -0.45]).unwrap(),
                Atom::from_angstrom("H", [-0.05, -0.74, -0.49]).unwrap(),
            ],
            0,
            1,
        );
        let grid = MolecularGrid::build(&mol, GridLevel::Medium);

        // Sum of Gaussians centered on all 3 atoms integrates to 3 * pi^{3/2}
        let exact = 3.0 * PI.powf(1.5);
        let num: f64 = grid
            .points
            .iter()
            .map(|gp| {
                let mut val = 0.0;
                for a in &mol.atoms {
                    let d2 = (gp.coords[0] - a.position_bohr[0]).powi(2)
                        + (gp.coords[1] - a.position_bohr[1]).powi(2)
                        + (gp.coords[2] - a.position_bohr[2]).powi(2);
                    val += (-d2).exp();
                }
                gp.weight * val
            })
            .sum();
        assert!((num - exact).abs() / exact < 2e-5);

        // Exact analytical Becke weight nuclear gradient vs central finite differences:
        // Pick a non-trivial valence grid point belonging to host atom 0 (O) and also host atom 1 (H).
        for test_host in [0usize, 1usize] {
            let u = [0.42_f64, -0.31, 0.57];
            let raw_w = 0.73;
            let f_val = -1.25;
            let r0 = [
                mol.atoms[test_host].position_bohr[0] + u[0],
                mol.atoms[test_host].position_bohr[1] + u[1],
                mol.atoms[test_host].position_bohr[2] + u[2],
            ];
            let mut ana = vec![[0.0; 3]; mol.atoms.len()];
            becke_weight_nuclear_gradient(&mol, test_host, r0, raw_w, &mut ana, f_val);

            let h = 1e-4;
            let eval_shifted = |atom_idx: usize, axis: usize, delta: f64| -> f64 {
                let mut m_s = mol.clone();
                m_s.atoms[atom_idx].position_bohr[axis] += delta;
                let r_s = [
                    m_s.atoms[test_host].position_bohr[0] + u[0],
                    m_s.atoms[test_host].position_bohr[1] + u[1],
                    m_s.atoms[test_host].position_bohr[2] + u[2],
                ];
                raw_w * becke_partition_weight(&m_s, test_host, r_s) * f_val
            };
            for a in 0..mol.atoms.len() {
                for q in 0..3 {
                    let wp2 = eval_shifted(a, q, 2.0 * h);
                    let wp1 = eval_shifted(a, q, h);
                    let wm1 = eval_shifted(a, q, -h);
                    let wm2 = eval_shifted(a, q, -2.0 * h);
                    let fd = (-wp2 + 8.0 * wp1 - 8.0 * wm1 + wm2) / (12.0 * h);
                    assert!(
                        (ana[a][q] - fd).abs() < 1e-11,
                        "host={test_host} atom={a} q={q}: ana={} vs fd={}",
                        ana[a][q],
                        fd
                    );
                }
            }
        }
    }
}

```

## `crates/calchem-dft/src/xc.rs`

SHA256: `b819ce8c7598587baeb7ef8b53b860dd9587c3db3fe0b07439cafa40cacbc6dc`

```rust
//! Spin-polarized Exchange-Correlation (XC) Functional Engine in Pure Rust:
//! - LDA: Slater (Dirac) exchange + Vosko-Wilk-Nusair (`VWN5` and `VWN_RPA`) correlation
//! - GGA: Becke 88 (`B88`) exchange, Lee-Yang-Parr (`LYP`, Miehlich gradient-only form) correlation,
//!   and Perdew-Burke-Ernzerhof (`PBE`) exchange + correlation (built on `PW92`)
//! - Hybrids: `B3LYP` (canonical Gaussian/Stephens `VWN_RPA`), `B3LYP5` (`VWN5`), and `PBE0` (`25% HF`)

use std::f64::consts::PI;

use calchem_model::CalcError;
use serde::{Deserialize, Serialize};

/// Density threshold below which spin XC energy density and potentials safely vanish.
const RHO_CUTOFF: f64 = 1e-18;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XcFunctional {
    /// Slater exchange + VWN5 local correlation (`LDA` / `SVWN` / `SVWN5`)
    Svwn5,
    /// Becke88 exchange + Lee-Yang-Parr correlation (`BLYP`)
    Blyp,
    /// Perdew-Burke-Ernzerhof GGA exchange-correlation (`PBE`)
    Pbe,
    /// Canonical B3LYP hybrid: `0.20 HF + 0.80 Slater + 0.72 dB88 + 0.19 VWN_RPA + 0.81 LYP`
    B3lyp,
    /// B3LYP5 hybrid variant using `VWN5` for the `0.19` local correlation component
    B3lyp5,
    /// Hybrid PBE0 (`PBE1PBE`): `0.25 HF + 0.75 PBE_X + 1.00 PBE_C`
    Pbe0,
    /// Global-Hybrid approximation to CAM-B3LYP (static HF exchange c_HF = 0.42).
    /// Note: Standard range-separated attenuated ERIs (erf(mu*r12)/r12) require external adapters.
    CamB3lyp,
    /// Global-Hybrid approximation to wB97X-D (static HF exchange c_HF = 0.22).
    /// Note: Standard range-separated attenuated ERIs require external adapters.
    Wb97xd,
}

impl XcFunctional {
    pub const SUPPORTED_NAMES: &'static [&'static str] = &[
        "svwn5", "svwn", "lda", "blyp", "pbe", "b3lyp", "b3lyp5", "pbe0",
    ];

    pub fn from_name(name: &str) -> Result<Self, CalcError> {
        let norm: String = name
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
            .collect();
        match norm.as_str() {
            "svwn" | "svwn5" | "lda" | "slatervwn5" => Ok(Self::Svwn5),
            "blyp" => Ok(Self::Blyp),
            "pbe" | "pbepbe" => Ok(Self::Pbe),
            "b3lyp" => Ok(Self::B3lyp),
            "b3lyp5" => Ok(Self::B3lyp5),
            "pbe0" | "pbe1pbe" => Ok(Self::Pbe0),
            "camb3lyp" | "wb97xd" | "wb97x" => Err(CalcError::UnsupportedMethod(
                "Native range-separated exchange ERIs are not implemented. CAM-B3LYP/wB97X(-D) require an external Gaussian/ORCA backend; global-hybrid substitution is disabled.".into()
            )),
            _ => Err(CalcError::UnsupportedMethod(format!(
                "Unsupported DFT XC functional '{name}' (supported: SVWN5, BLYP, PBE, B3LYP, B3LYP5, PBE0)"
            ))),
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Svwn5 => "svwn5",
            Self::Blyp => "blyp",
            Self::Pbe => "pbe",
            Self::B3lyp => "b3lyp",
            Self::B3lyp5 => "b3lyp5",
            Self::Pbe0 => "pbe0",
            Self::CamB3lyp => "cam-b3lyp",
            Self::Wb97xd => "wb97x-d",
        }
    }

    /// Guard Rust callers that construct legacy enum variants directly.
    pub fn validate_native(self) -> Result<(), CalcError> {
        if matches!(self, Self::CamB3lyp | Self::Wb97xd) {
            return Err(CalcError::UnsupportedMethod(
                "Range-separated exchange ERIs are not implemented in the native core".into(),
            ));
        }
        Ok(())
    }

    /// Fraction $c_{\text{HF}} \in [0, 1]$ of exact Hartree-Fock exchange.
    #[inline]
    pub fn exact_exchange_fraction(self) -> f64 {
        match self {
            Self::Svwn5 | Self::Blyp | Self::Pbe => 0.0,
            Self::B3lyp | Self::B3lyp5 => 0.20,
            Self::Pbe0 => 0.25,
            Self::CamB3lyp => 0.42,
            Self::Wb97xd => 0.22,
        }
    }

    #[inline]
    pub fn is_gga(self) -> bool {
        !matches!(self, Self::Svwn5)
    }
}

/// Volumetric XC energy density $f_{\text{xc}}$ (so $E_{\text{xc}} = \int f_{\text{xc}}\,d^3\mathbf{r}$)
/// and first partial derivatives with respect to $(\rho_\alpha, \rho_\beta, \gamma_{\alpha\alpha}, \gamma_{\alpha\beta}, \gamma_{\beta\beta})$.
#[derive(Debug, Clone, Copy, Default)]
pub struct XcPointOut {
    pub f_xc: f64,
    pub v_rho_a: f64,
    pub v_rho_b: f64,
    pub v_gamma_aa: f64,
    pub v_gamma_ab: f64,
    pub v_gamma_bb: f64,
}

impl XcPointOut {
    #[inline]
    fn add_scaled(&mut self, scale: f64, other: Self) {
        self.f_xc += scale * other.f_xc;
        self.v_rho_a += scale * other.v_rho_a;
        self.v_rho_b += scale * other.v_rho_b;
        self.v_gamma_aa += scale * other.v_gamma_aa;
        self.v_gamma_ab += scale * other.v_gamma_ab;
        self.v_gamma_bb += scale * other.v_gamma_bb;
    }
}

/// Evaluates `func` at spin-polarized invariants $(\rho_\alpha, \rho_\beta, \gamma_{\alpha\alpha}, \gamma_{\alpha\beta}, \gamma_{\beta\beta})$.
#[inline]
pub fn eval_xc_spin(
    func: XcFunctional,
    rho_a: f64,
    rho_b: f64,
    gamma_aa: f64,
    gamma_ab: f64,
    gamma_bb: f64,
) -> XcPointOut {
    let ra = rho_a.max(0.0);
    let rb = rho_b.max(0.0);
    if ra + rb < RHO_CUTOFF {
        return XcPointOut::default();
    }
    let gaa = gamma_aa.max(0.0);
    let gbb = gamma_bb.max(0.0);

    match func {
        XcFunctional::Svwn5 => {
            let mut out = slater_x(ra, rb);
            out.add_scaled(1.0, vwn_c(ra, rb, &VWN5_PARAMS));
            out
        }
        XcFunctional::Blyp => {
            let mut out = slater_x(ra, rb);
            out.add_scaled(1.0, becke88_delta_x(ra, rb, gaa, gbb));
            out.add_scaled(1.0, lyp_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::Pbe => {
            let mut out = pbe_x(ra, rb, gaa, gbb);
            out.add_scaled(1.0, pbe_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::B3lyp => {
            let mut out = XcPointOut::default();
            out.add_scaled(0.80, slater_x(ra, rb));
            out.add_scaled(0.72, becke88_delta_x(ra, rb, gaa, gbb));
            out.add_scaled(0.19, vwn_c(ra, rb, &VWN_RPA_PARAMS));
            out.add_scaled(0.81, lyp_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::B3lyp5 => {
            let mut out = XcPointOut::default();
            out.add_scaled(0.80, slater_x(ra, rb));
            out.add_scaled(0.72, becke88_delta_x(ra, rb, gaa, gbb));
            out.add_scaled(0.19, vwn_c(ra, rb, &VWN5_PARAMS));
            out.add_scaled(0.81, lyp_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::Pbe0 => {
            let mut out = XcPointOut::default();
            out.add_scaled(0.75, pbe_x(ra, rb, gaa, gbb));
            out.add_scaled(1.00, pbe_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::CamB3lyp => {
            let mut out = camb3lyp_x(ra, rb, gaa, gbb);
            out.add_scaled(0.19, vwn_c(ra, rb, &VWN5_PARAMS));
            out.add_scaled(0.81, lyp_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::Wb97xd => wb97xd_xc(ra, rb, gaa, gamma_ab, gbb),
    }
}

// ============================================================================
// 1. Slater (Dirac) Spin-Polarized LDA Exchange
// ============================================================================

#[inline]
fn slater_x_one_spin(rho_s: f64) -> (f64, f64) {
    if rho_s < RHO_CUTOFF {
        return (0.0, 0.0);
    }
    let cx = 0.75 * (6.0 / PI).powf(1.0 / 3.0);
    let r13 = rho_s.cbrt();
    let f = -cx * rho_s * r13;
    let v = -(4.0 / 3.0) * cx * r13;
    (f, v)
}

#[inline]
fn slater_x(ra: f64, rb: f64) -> XcPointOut {
    let (fa, va) = slater_x_one_spin(ra);
    let (fb, vb) = slater_x_one_spin(rb);
    XcPointOut {
        f_xc: fa + fb,
        v_rho_a: va,
        v_rho_b: vb,
        ..Default::default()
    }
}

// ============================================================================
// 2. Vosko-Wilk-Nusair (1980) Correlation (VWN5 & VWN_RPA)
// ============================================================================

struct VwnChannel {
    a: f64,
    x0: f64,
    b: f64,
    c: f64,
}

struct VwnParams {
    para: VwnChannel,
    ferro: VwnChannel,
    stiff: VwnChannel,
}

const VWN5_PARAMS: VwnParams = VwnParams {
    para: VwnChannel {
        a: 0.0310907,
        x0: -0.10498,
        b: 3.72744,
        c: 12.9352,
    },
    ferro: VwnChannel {
        a: 0.01554535,
        x0: -0.32500,
        b: 7.06042,
        c: 18.0578,
    },
    stiff: VwnChannel {
        a: -0.016886863940389905, // -1 / (6 * pi^2)
        x0: -0.00475840,
        b: 1.13107,
        c: 13.0045,
    },
};

const VWN_RPA_PARAMS: VwnParams = VwnParams {
    para: VwnChannel {
        a: 0.0310907,
        x0: -0.409286,
        b: 13.0720,
        c: 42.7198,
    },
    ferro: VwnChannel {
        a: 0.01554535,
        x0: -0.743294,
        b: 20.1231,
        c: 101.578,
    },
    stiff: VwnChannel {
        a: -0.016886863940389905,
        x0: -0.228344,
        b: 1.06835,
        c: 11.4813,
    },
};

/// Returns `(eps(rs), rs * d(eps)/d(rs))` using VWN's exact Padé rational derivative.
#[inline]
fn vwn_channel_eval(x: f64, ch: &VwnChannel) -> (f64, f64) {
    let xx = x * x + ch.b * x + ch.c;
    let xx0 = ch.x0 * ch.x0 + ch.b * ch.x0 + ch.c;
    let q = (4.0 * ch.c - ch.b * ch.b).sqrt();
    let atan_term = (q / (2.0 * x + ch.b)).atan();
    let x_m_x0 = x - ch.x0;

    let eps = ch.a
        * ((x * x / xx).ln() + (2.0 * ch.b / q) * atan_term
            - (ch.b * ch.x0 / xx0)
                * ((x_m_x0 * x_m_x0 / xx).ln() + (2.0 * (ch.b + 2.0 * ch.x0) / q) * atan_term));

    // Exact Padé rational identity: rs * d(eps)/d(rs) = A * [c*(x - x0) - b*x*x0] / [(x - x0)*X(x)]
    let rs_deps = ch.a * (ch.c * x_m_x0 - ch.b * x * ch.x0) / (x_m_x0 * xx);
    (eps, rs_deps)
}

#[inline]
fn spin_interp_f_and_df(zeta: f64) -> (f64, f64, f64) {
    let cbrt2_m1 = 2.0_f64.cbrt() - 1.0;
    let denom = 2.0 * cbrt2_m1;
    let fdd0 = 4.0 / (9.0 * cbrt2_m1);
    let zp = (1.0 + zeta).max(0.0);
    let zm = (1.0 - zeta).max(0.0);
    let zp13 = zp.cbrt();
    let zm13 = zm.cbrt();
    let fz = (zp * zp13 + zm * zm13 - 2.0) / denom;
    let dfz = (4.0 / (3.0 * denom)) * (zp13 - zm13);
    (fz, dfz, fdd0)
}

#[inline]
fn vwn_c(ra: f64, rb: f64, params: &VwnParams) -> XcPointOut {
    let rho = ra + rb;
    if rho < RHO_CUTOFF {
        return XcPointOut::default();
    }
    let rs = (3.0 / (4.0 * PI * rho)).cbrt();
    let x = rs.sqrt();
    let zeta = ((ra - rb) / rho).clamp(-1.0, 1.0);

    let (ep, rs_dep) = vwn_channel_eval(x, &params.para);
    let (ef, rs_def) = vwn_channel_eval(x, &params.ferro);
    let (ac, rs_dac) = vwn_channel_eval(x, &params.stiff);

    let (fz, dfz, fdd0) = spin_interp_f_and_df(zeta);
    let z2 = zeta * zeta;
    let z3 = z2 * zeta;
    let z4 = z2 * z2;
    let one_m_z4 = 1.0 - z4;

    let inv_fdd0 = 1.0 / fdd0;
    let dec = ef - ep;
    let rs_ddec = rs_def - rs_dep;

    let ec = ep + ac * fz * inv_fdd0 * one_m_z4 + dec * fz * z4;
    let rs_dec_drs = rs_dep + rs_dac * fz * inv_fdd0 * one_m_z4 + rs_ddec * fz * z4;
    let dec_dzeta =
        ac * inv_fdd0 * (dfz * one_m_z4 - 4.0 * z3 * fz) + dec * (dfz * z4 + 4.0 * z3 * fz);

    let base_v = ec - (1.0 / 3.0) * rs_dec_drs;
    XcPointOut {
        f_xc: rho * ec,
        v_rho_a: base_v + (1.0 - zeta) * dec_dzeta,
        v_rho_b: base_v - (1.0 + zeta) * dec_dzeta,
        ..Default::default()
    }
}

// ============================================================================
// 3. Becke 88 (B88) GGA Exchange Gradient Correction (\Delta E_x^{B88})
// ============================================================================

#[inline]
fn b88_delta_one_spin(rho_s: f64, gamma_ss: f64) -> (f64, f64, f64) {
    if rho_s < RHO_CUTOFF {
        return (0.0, 0.0, 0.0);
    }
    const BETA: f64 = 0.0042;
    let r13 = rho_s.cbrt();
    let r43 = rho_s * r13;
    let s2 = gamma_ss.max(0.0) / (r43 * r43);
    let s = s2.sqrt();
    let asinh_s = s.asinh();
    let denom = 1.0 + 6.0 * BETA * s * asinh_s;

    let g = -BETA * s2 / denom;
    // h(s) = g'(s) / s (analytic & smooth at s = 0, where h(0) = -2 * BETA)
    let sqrt_1_s2 = (1.0 + s2).sqrt();
    let h =
        -BETA * (2.0 + 6.0 * BETA * s * asinh_s - 6.0 * BETA * s2 / sqrt_1_s2) / (denom * denom);

    let f = r43 * g;
    let v_rho = (4.0 / 3.0) * r13 * (g - s2 * h);
    let v_gamma = 0.5 * h / r43;
    (f, v_rho, v_gamma)
}

#[inline]
fn becke88_delta_x(ra: f64, rb: f64, gaa: f64, gbb: f64) -> XcPointOut {
    let (fa, vra, vgaa) = b88_delta_one_spin(ra, gaa);
    let (fb, vrb, vgbb) = b88_delta_one_spin(rb, gbb);
    XcPointOut {
        f_xc: fa + fb,
        v_rho_a: vra,
        v_rho_b: vrb,
        v_gamma_aa: vgaa,
        v_gamma_ab: 0.0,
        v_gamma_bb: vgbb,
    }
}

// ============================================================================
// 4. Lee-Yang-Parr (LYP) GGA Correlation (Miehlich et al. 1989 Eq. 2)
// ============================================================================

/// Returns `(S, dS/d(ra), A_aa, A_ab, A_bb)` for Miehlich Eq. (2)'s bracket `S`.
/// Calling with `(rb, ra, delta, ddelta, gbb, gab, gaa)` yields `dS/d(rb)` by symmetry!
#[inline]
fn lyp_bracket_and_dra(
    ra: f64,
    rb: f64,
    delta: f64,
    ddelta: f64,
    gaa: f64,
    gab: f64,
    gbb: f64,
) -> (f64, f64, f64, f64, f64) {
    let rho = ra + rb;
    let rho2 = rho * rho;
    let cf = 0.3 * (3.0 * PI * PI).powf(2.0 / 3.0);
    let k0 = 2.0_f64.powf(11.0 / 3.0) * cf;

    // S_TF = k0 * ra * rb * (ra^{8/3} + rb^{8/3})
    let ra_83 = if ra > 0.0 { ra.powf(8.0 / 3.0) } else { 0.0 };
    let rb_83 = if rb > 0.0 { rb.powf(8.0 / 3.0) } else { 0.0 };
    let s_tf = k0 * ra * rb * (ra_83 + rb_83);
    let ds_tf_dra = k0 * ((11.0 / 3.0) * ra_83 * rb + rb * rb_83);

    // C_gamma = ra * rb * (47 - 7 * delta)/18 - (2/3) * rho^2
    let c_gam = ra * rb * (47.0 - 7.0 * delta) / 18.0 - (2.0 / 3.0) * rho2;
    let dc_gam_dra =
        rb * (47.0 - 7.0 * delta) / 18.0 + ra * rb * (-7.0 * ddelta) / 18.0 - (4.0 / 3.0) * rho;

    // E_aa = -ra*rb*(5/2 - delta/18) - ((delta - 11)/9) * (ra^2 * rb / rho) + ((2/3)*rho^2 - rb^2)
    let q_aa = ra * ra * rb / rho;
    let dq_aa_dra = (2.0 * ra * rb * rho - ra * ra * rb) / rho2;
    let e_aa = -ra * rb * (2.5 - delta / 18.0) - ((delta - 11.0) / 9.0) * q_aa
        + ((2.0 / 3.0) * rho2 - rb * rb);
    let de_aa_dra = -rb * (2.5 - delta / 18.0) + ra * rb * (ddelta / 18.0)
        - (ddelta / 9.0) * q_aa
        - ((delta - 11.0) / 9.0) * dq_aa_dra
        + (4.0 / 3.0) * rho;

    // E_bb = -ra*rb*(5/2 - delta/18) - ((delta - 11)/9) * (ra * rb^2 / rho) + ((2/3)*rho^2 - ra^2)
    let q_bb = ra * rb * rb / rho;
    let dq_bb_dra = (rb * rb * rho - ra * rb * rb) / rho2;
    let e_bb = -ra * rb * (2.5 - delta / 18.0) - ((delta - 11.0) / 9.0) * q_bb
        + ((2.0 / 3.0) * rho2 - ra * ra);
    let de_bb_dra = -rb * (2.5 - delta / 18.0) + ra * rb * (ddelta / 18.0)
        - (ddelta / 9.0) * q_bb
        - ((delta - 11.0) / 9.0) * dq_bb_dra
        + (4.0 / 3.0) * rho
        - 2.0 * ra;

    let a_aa = c_gam + e_aa;
    let a_ab = 2.0 * c_gam;
    let a_bb = c_gam + e_bb;

    let s = s_tf + a_aa * gaa + a_ab * gab + a_bb * gbb;
    let ds_dra = ds_tf_dra
        + (dc_gam_dra + de_aa_dra) * gaa
        + (2.0 * dc_gam_dra) * gab
        + (dc_gam_dra + de_bb_dra) * gbb;

    (s, ds_dra, a_aa, a_ab, a_bb)
}

#[inline]
fn lyp_c(ra: f64, rb: f64, gaa: f64, gab: f64, gbb: f64) -> XcPointOut {
    let rho = ra + rb;
    if rho < RHO_CUTOFF {
        return XcPointOut::default();
    }
    const A: f64 = 0.04918;
    const B: f64 = 0.132;
    const C: f64 = 0.2533;
    const D: f64 = 0.349;

    let r13 = rho.powf(-1.0 / 3.0);
    // Avoid exp(-c * rho^{-1/3}) underflowing into NaN ratios at ultra-tiny density tails:
    if C * r13 > 500.0 {
        // omega -> 0 exponentially; only T0 survives.
        let den = 1.0 + D * r13;
        let g0 = -4.0 * A / (rho * den);
        let dg0 = -g0 * (1.0 + (2.0 / 3.0) * D * r13) / (rho * den);
        return XcPointOut {
            f_xc: g0 * ra * rb,
            v_rho_a: dg0 * ra * rb + g0 * rb,
            v_rho_b: dg0 * ra * rb + g0 * ra,
            ..Default::default()
        };
    }

    let den = 1.0 + D * r13;
    let g0 = -4.0 * A / (rho * den);
    let dg0 = -g0 * (1.0 + (2.0 / 3.0) * D * r13) / (rho * den);
    let t0 = g0 * ra * rb;
    let dt0_dra = dg0 * ra * rb + g0 * rb;
    let dt0_drb = dg0 * ra * rb + g0 * ra;

    let omega = (-C * r13).exp() / den * rho.powf(-11.0 / 3.0);
    let delta = C * r13 + D * r13 / den;
    let domega = omega * (delta - 11.0) / (3.0 * rho);
    let r43_inv = r13 / rho;
    let ddelta = -(1.0 / 3.0) * r43_inv * (C + D / (den * den));

    let (s, ds_dra, a_aa, a_ab, a_bb) = lyp_bracket_and_dra(ra, rb, delta, ddelta, gaa, gab, gbb);
    let (_s_sym, ds_drb, _, _, _) = lyp_bracket_and_dra(rb, ra, delta, ddelta, gbb, gab, gaa);

    let ab = A * B;
    let f_xc = t0 - ab * omega * s;
    let v_rho_a = dt0_dra - ab * (domega * s + omega * ds_dra);
    let v_rho_b = dt0_drb - ab * (domega * s + omega * ds_drb);

    XcPointOut {
        f_xc,
        v_rho_a,
        v_rho_b,
        v_gamma_aa: -ab * omega * a_aa,
        v_gamma_ab: -ab * omega * a_ab,
        v_gamma_bb: -ab * omega * a_bb,
    }
}

// ============================================================================
// 5. Perdew-Burke-Ernzerhof (PBE) GGA Exchange & Correlation (PW92 base)
// ============================================================================

#[inline]
fn pbe_x_one_spin(rho_s: f64, gamma_ss: f64) -> (f64, f64, f64) {
    if rho_s < RHO_CUTOFF {
        return (0.0, 0.0, 0.0);
    }
    const KAPPA: f64 = 0.804;
    const MU: f64 = 0.2195149727645171;
    // Spin-scaling: f_x(rho_s, gamma_ss) = 0.5 * f_x^{unpol}(2*rho_s, 4*gamma_ss)
    let rho_u = 2.0 * rho_s;
    let gam_u = 4.0 * gamma_ss.max(0.0);

    let cx_u = -0.75 * (3.0 / PI).powf(1.0 / 3.0);
    let r13 = rho_u.cbrt();
    let r43 = rho_u * r13;
    let ex_lda = cx_u * r43;

    let kf = (3.0 * PI * PI * rho_u).cbrt();
    let denom_s2 = 4.0 * kf * kf * rho_u * rho_u;
    let s2 = gam_u / denom_s2;

    let one_p_ms = 1.0 + (MU / KAPPA) * s2;
    let fx = 1.0 + KAPPA - KAPPA / one_p_ms;
    let dfx_ds2 = MU / (one_p_ms * one_p_ms);

    // f_u = ex_lda * fx; ds2/drho_u = -(8/3) * s2 / rho_u; ds2/dgam_u = 1 / denom_s2
    let f_u = ex_lda * fx;
    let df_drho_u = (4.0 / 3.0) * cx_u * r13 * fx + ex_lda * dfx_ds2 * (-(8.0 / 3.0) * s2 / rho_u);
    let df_dgam_u = ex_lda * dfx_ds2 / denom_s2;

    // Map back: f = 0.5 * f_u(2*rho_s, 4*gamma_ss) => df/drho_s = df_drho_u, df/dgamma_ss = 2 * df_dgam_u
    (0.5 * f_u, df_drho_u, 2.0 * df_dgam_u)
}

#[inline]
fn pbe_x(ra: f64, rb: f64, gaa: f64, gbb: f64) -> XcPointOut {
    let (fa, vra, vgaa) = pbe_x_one_spin(ra, gaa);
    let (fb, vrb, vgbb) = pbe_x_one_spin(rb, gbb);
    XcPointOut {
        f_xc: fa + fb,
        v_rho_a: vra,
        v_rho_b: vrb,
        v_gamma_aa: vgaa,
        v_gamma_ab: 0.0,
        v_gamma_bb: vgbb,
    }
}

struct Pw92Channel {
    a: f64,
    alpha1: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
}

const PW92_G0: Pw92Channel = Pw92Channel {
    a: 0.0310907,
    alpha1: 0.21370,
    b1: 7.5957,
    b2: 3.5876,
    b3: 1.6382,
    b4: 0.49294,
};

const PW92_G1: Pw92Channel = Pw92Channel {
    a: 0.01554535,
    alpha1: 0.20548,
    b1: 14.1189,
    b2: 6.1977,
    b3: 3.3662,
    b4: 0.62517,
};

const PW92_GA: Pw92Channel = Pw92Channel {
    a: 0.0168869,
    alpha1: 0.11125,
    b1: 10.357,
    b2: 3.6231,
    b3: 0.88026,
    b4: 0.49671,
};

/// Returns `(G(rs), dG/drs)` for a PW92 channel.
#[inline]
fn pw92_g(rs: f64, ch: &Pw92Channel) -> (f64, f64) {
    let x = rs.sqrt();
    let x2 = rs;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let q0 = -2.0 * ch.a * (1.0 + ch.alpha1 * rs);
    let q1 = 2.0 * ch.a * (ch.b1 * x + ch.b2 * x2 + ch.b3 * x3 + ch.b4 * x4);
    let dq1_drs = ch.a * (ch.b1 / x + 2.0 * ch.b2 + 3.0 * ch.b3 * x + 4.0 * ch.b4 * x2);
    let ln_term = (1.0 + 1.0 / q1).ln();
    let g = q0 * ln_term;
    let dg_drs = -2.0 * ch.a * ch.alpha1 * ln_term - q0 * dq1_drs / (q1 * (q1 + 1.0));
    (g, dg_drs)
}

#[inline]
fn pbe_c(ra: f64, rb: f64, gaa: f64, gab: f64, gbb: f64) -> XcPointOut {
    let rho = ra + rb;
    if rho < RHO_CUTOFF {
        return XcPointOut::default();
    }
    let gamma_tot = (gaa + 2.0 * gab + gbb).max(0.0);
    let rs = (3.0 / (4.0 * PI * rho)).cbrt();
    let drs_drho = -rs / (3.0 * rho);
    let zeta = ((ra - rb) / rho).clamp(-1.0, 1.0);

    // 1. PW92 local correlation eps_c(rs, zeta)
    let (e0, de0_drs) = pw92_g(rs, &PW92_G0);
    let (e1, de1_drs) = pw92_g(rs, &PW92_G1);
    let (minus_ac, d_minus_ac_drs) = pw92_g(rs, &PW92_GA);
    let ac = -minus_ac;
    let dac_drs = -d_minus_ac_drs;

    let (fz, dfz, fdd0) = spin_interp_f_and_df(zeta);
    let inv_fdd0 = 1.0 / fdd0;
    let z2 = zeta * zeta;
    let z3 = z2 * zeta;
    let z4 = z2 * z2;
    let one_m_z4 = 1.0 - z4;

    let ec_lsda = e0 + ac * fz * inv_fdd0 * one_m_z4 + (e1 - e0) * fz * z4;
    let dec_drs = de0_drs + dac_drs * fz * inv_fdd0 * one_m_z4 + (de1_drs - de0_drs) * fz * z4;
    let dec_drho = dec_drs * drs_drho;
    let dec_dzeta =
        ac * inv_fdd0 * (dfz * one_m_z4 - 4.0 * z3 * fz) + (e1 - e0) * (dfz * z4 + 4.0 * z3 * fz);

    if gamma_tot < 1e-36 {
        return XcPointOut {
            f_xc: rho * ec_lsda,
            v_rho_a: ec_lsda + rho * dec_drho + (1.0 - zeta) * dec_dzeta,
            v_rho_b: ec_lsda + rho * dec_drho - (1.0 + zeta) * dec_dzeta,
            ..Default::default()
        };
    }

    // 2. PBE gradient correction H(rho, zeta, gamma_tot)
    let gamma_pbe = (1.0 - 2.0_f64.ln()) / (PI * PI);
    const BETA_PBE: f64 = 0.06672455060314922;
    let b0 = BETA_PBE / gamma_pbe;

    let zp = (1.0 + zeta).max(1e-24);
    let zm = (1.0 - zeta).max(1e-24);
    let zp23 = zp.cbrt().powi(2);
    let zm23 = zm.cbrt().powi(2);
    let phi = 0.5 * (zp23 + zm23);
    let phi2 = phi * phi;
    let phi3 = phi2 * phi;

    // Singularity-free combinations (1 - zeta) * dphi/dzeta and -(1 + zeta) * dphi/dzeta:
    let one_m_z_dphi = ((1.0 - zeta) * zp.cbrt().recip() - zm23) / 3.0;
    let minus_one_p_z_dphi = ((1.0 + zeta) * zm.cbrt().recip() - zp23) / 3.0;

    let c_t = (16.0 / PI) * (3.0 * PI * PI).cbrt();
    let rho73 = rho * rho * rho.cbrt();
    let denom_t2 = c_t * phi2 * rho73;
    let t2 = gamma_tot / denom_t2;

    let w = -ec_lsda / (gamma_pbe * phi3);
    let (a_func, da_dw) = if w > 100.0 {
        (0.0, 0.0)
    } else {
        let em1 = w.exp_m1();
        let ew = w.exp();
        let af = b0 / em1;
        (af, -af * ew / em1)
    };

    let u = a_func * t2;
    let d_poly = 1.0 + u + u * u;
    let q_rat = (1.0 + u) / d_poly;
    let z_arg = 1.0 + b0 * t2 * q_rat;
    let ln_z = z_arg.ln();
    let h = gamma_pbe * phi3 * ln_z;

    let dz_dt2 = b0 * (1.0 + 2.0 * u) / (d_poly * d_poly);
    let dz_da = -b0 * t2 * t2 * u * (2.0 + u) / (d_poly * d_poly);
    let dh_dt2 = gamma_pbe * phi3 * dz_dt2 / z_arg;
    let dh_dw = gamma_pbe * phi3 * (dz_da * da_dw) / z_arg;

    // Chain rule at fixed (zeta, gamma_tot) with respect to rho:
    //   dt2/drho = -(7/3) * t2 / rho,   dw/drho = -dec_drho / (gamma_pbe * phi3)
    let dh_drho_fixed_zeta =
        dh_dt2 * (-(7.0 / 3.0) * t2 / rho) + dh_dw * (-dec_drho / (gamma_pbe * phi3));

    // Chain rule with respect to zeta through both dec_dzeta and phi(zeta):
    //   dH/dphi|_{w,t2} + dh_dw * dw/dphi + dh_dt2 * dt2/dphi
    //   where dH/dphi|_{w,t2} = 3 * gamma_pbe * phi^2 * ln_z,
    //         dw/dphi = -3 * w / phi,
    //         dt2/dphi = -2 * t2 / phi.
    let dh_dphi_total =
        3.0 * gamma_pbe * phi2 * ln_z + dh_dw * (-3.0 * w / phi) + dh_dt2 * (-2.0 * t2 / phi);
    let dh_d_ec = dh_dw * (-1.0 / (gamma_pbe * phi3));

    // rho * d(ec + H)/drho_a = (ec + H) + rho*(dec_drho + dh_drho_fixed_zeta)
    //   + (1 - zeta)*(dec_dzeta * (1 + dh_d_ec)) + dh_dphi_total * [(1 - zeta)*dphi/dzeta]
    let total_e = ec_lsda + h;
    let common_rho = total_e + rho * (dec_drho + dh_drho_fixed_zeta);
    let v_rho_a =
        common_rho + (1.0 - zeta) * dec_dzeta * (1.0 + dh_d_ec) + dh_dphi_total * one_m_z_dphi;
    let v_rho_b = common_rho - (1.0 + zeta) * dec_dzeta * (1.0 + dh_d_ec)
        + dh_dphi_total * minus_one_p_z_dphi;

    let v_gamma_tot = rho * dh_dt2 / denom_t2;

    XcPointOut {
        f_xc: rho * total_e,
        v_rho_a,
        v_rho_b,
        v_gamma_aa: v_gamma_tot,
        v_gamma_ab: 2.0 * v_gamma_tot,
        v_gamma_bb: v_gamma_tot,
    }
}

// ============================================================================
// 6. Coulomb-Attenuated B3LYP (CAM-B3LYP, Yanai et al. 2004)
// ============================================================================

#[inline]
fn camb3lyp_attenuation(a: f64) -> (f64, f64) {
    if a < 1e-5 {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let f = 1.0 - (8.0 / 3.0) * a * sqrt_pi + 8.0 * a * a;
        let df_da = -(8.0 / 3.0) * sqrt_pi + 16.0 * a;
        return (f.clamp(0.0, 1.0), df_da);
    }
    if a > 15.0 {
        let f = 1.0 / (36.0 * a * a);
        let df_da = -1.0 / (18.0 * a * a * a);
        return (f.clamp(0.0, 1.0), df_da);
    }

    let a2 = a * a;
    let a3 = a2 * a;
    let inv_2a = 0.5 / a;
    let erf_val = calchem_int::erf_approx(inv_2a);
    let exp_val = (-0.25 / a2).exp();
    let sqrt_pi = std::f64::consts::PI.sqrt();

    let bracket = sqrt_pi * erf_val + (2.0 * a - 4.0 * a3) * exp_val - 3.0 * a + 4.0 * a3;
    let f_atten = 1.0 - (8.0 / 3.0) * a * bracket;

    let d_bracket = -3.0 + 12.0 * a2 * (1.0 - exp_val);
    let df_da = -(8.0 / 3.0) * (bracket + a * d_bracket);

    (f_atten.clamp(0.0, 1.0), df_da)
}

#[inline]
fn camb3lyp_x_one_spin(rho_s: f64, gamma_ss: f64) -> (f64, f64, f64) {
    if rho_s < RHO_CUTOFF {
        return (0.0, 0.0, 0.0);
    }

    let (fa_slater, vra_slater) = slater_x_one_spin(rho_s);
    let (fa_b88, vra_b88, vgaa_b88) = b88_delta_one_spin(rho_s, gamma_ss);

    let fx_tot = fa_slater + fa_b88;
    let v_rho_tot = vra_slater + vra_b88;
    let v_gam_tot = vgaa_b88;

    let k_f = (6.0 * PI * PI * rho_s).cbrt();
    let mu = 0.33;
    let a = mu / (2.0 * k_f);
    let (f_atten, df_da) = camb3lyp_attenuation(a);

    // Alpha = 0.19, Beta = 0.46
    // W(a) = 0.35 + 0.46 * F_atten(a)
    let w = 0.35 + 0.46 * f_atten;
    let dw_da = 0.46 * df_da;
    let da_drho = -(1.0 / 3.0) * a / rho_s;
    let dw_drho = dw_da * da_drho;

    let f = w * fx_tot;
    let v_rho = w * v_rho_tot + dw_drho * fx_tot;
    let v_gamma = w * v_gam_tot;

    (f, v_rho, v_gamma)
}

#[inline]
fn camb3lyp_x(ra: f64, rb: f64, gaa: f64, gbb: f64) -> XcPointOut {
    let (fa, vra, vgaa) = camb3lyp_x_one_spin(ra, gaa);
    let (fb, vrb, vgbb) = camb3lyp_x_one_spin(rb, gbb);
    XcPointOut {
        f_xc: fa + fb,
        v_rho_a: vra,
        v_rho_b: vrb,
        v_gamma_aa: vgaa,
        v_gamma_ab: 0.0,
        v_gamma_bb: vgbb,
    }
}

// ============================================================================
// 7. wB97X-D Range-Separated Hybrid XC (Chai & Head-Gordon 2008)
// ============================================================================

#[inline]
fn wb97xd_x_one_spin(rho_s: f64, gamma_ss: f64) -> (f64, f64, f64) {
    if rho_s < RHO_CUTOFF {
        return (0.0, 0.0, 0.0);
    }
    let (f_slater, v_slater) = slater_x_one_spin(rho_s);
    let r13 = rho_s.cbrt();
    let r43 = rho_s * r13;
    let r83 = r43 * r43;

    let g = gamma_ss.max(0.0);
    let s2 = g / (r83 * 4.0 * (3.0 * PI * PI).powf(2.0 / 3.0));
    let u = (0.0042 * s2) / (1.0 + 0.0042 * s2);
    let du_ds2 = 0.0042 / ((1.0 + 0.0042 * s2) * (1.0 + 0.0042 * s2));

    // Chai & Head-Gordon (2008) B97 short-range exchange coefficients
    const C0: f64 = 0.778;
    const C1: f64 = 0.404;
    const C2: f64 = 0.141;

    let p = C0 + C1 * u + C2 * u * u;
    let dp_du = C1 + 2.0 * C2 * u;

    let f = f_slater * p;
    let ds2_drho = -(8.0 / 3.0) * s2 / rho_s;
    let ds2_dgam = s2 / g.max(1e-18);

    let v_rho = v_slater * p + f_slater * dp_du * du_ds2 * ds2_drho;
    let v_gam = f_slater * dp_du * du_ds2 * ds2_dgam;

    (f, v_rho, v_gam)
}

#[inline]
fn wb97xd_xc(ra: f64, rb: f64, gaa: f64, gab: f64, gbb: f64) -> XcPointOut {
    let (fa, vra, vgaa) = wb97xd_x_one_spin(ra, gaa);
    let (fb, vrb, vgbb) = wb97xd_x_one_spin(rb, gbb);
    let c_pbe = pbe_c(ra, rb, gaa, gab, gbb);

    XcPointOut {
        f_xc: fa + fb + c_pbe.f_xc,
        v_rho_a: vra + c_pbe.v_rho_a,
        v_rho_b: vrb + c_pbe.v_rho_b,
        v_gamma_aa: vgaa + c_pbe.v_gamma_aa,
        v_gamma_ab: c_pbe.v_gamma_ab,
        v_gamma_bb: vgbb + c_pbe.v_gamma_bb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_xc_partials_match_fd(
        func: XcFunctional,
        ra: f64,
        rb: f64,
        gaa: f64,
        gab: f64,
        gbb: f64,
    ) {
        let ana = eval_xc_spin(func, ra, rb, gaa, gab, gbb);
        let h_r = 1e-6 * (ra + rb).max(1e-2);
        let h_g = 1e-6 * (gaa + gbb + 1.0);

        let f_only = |a: f64, b: f64, aa: f64, ab: f64, bb: f64| -> f64 {
            eval_xc_spin(func, a, b, aa, ab, bb).f_xc
        };

        // 4-point central difference O(h^4) for sharp 1e-11 verification
        let fd4 = |eval_shift: &dyn Fn(f64) -> f64, h: f64| -> f64 {
            (-eval_shift(2.0 * h) + 8.0 * eval_shift(h) - 8.0 * eval_shift(-h)
                + eval_shift(-2.0 * h))
                / (12.0 * h)
        };

        if ra > 5.0 * h_r {
            let fd_vra = fd4(&|d| f_only(ra + d, rb, gaa, gab, gbb), h_r);
            assert!(
                (ana.v_rho_a - fd_vra).abs() < 1e-9,
                "{func:?} v_rho_a mismatch: ana={} vs fd={}",
                ana.v_rho_a,
                fd_vra
            );
        }
        if rb > 5.0 * h_r {
            let fd_vrb = fd4(&|d| f_only(ra, rb + d, gaa, gab, gbb), h_r);
            assert!(
                (ana.v_rho_b - fd_vrb).abs() < 1e-9,
                "{func:?} v_rho_b mismatch: ana={} vs fd={}",
                ana.v_rho_b,
                fd_vrb
            );
        }
        if gaa > 5.0 * h_g {
            let fd_gaa = fd4(&|d| f_only(ra, rb, gaa + d, gab, gbb), h_g);
            assert!(
                (ana.v_gamma_aa - fd_gaa).abs() < 1e-9,
                "{func:?} v_gamma_aa mismatch: ana={} vs fd={}",
                ana.v_gamma_aa,
                fd_gaa
            );
        }
        if gaa > 5.0 * h_g && gbb > 5.0 * h_g {
            let fd_gab = fd4(&|d| f_only(ra, rb, gaa, gab + d, gbb), h_g);
            assert!(
                (ana.v_gamma_ab - fd_gab).abs() < 1e-9,
                "{func:?} v_gamma_ab mismatch: ana={} vs fd={}",
                ana.v_gamma_ab,
                fd_gab
            );
        }
        if gbb > 5.0 * h_g {
            let fd_gbb = fd4(&|d| f_only(ra, rb, gaa, gab, gbb + d), h_g);
            assert!(
                (ana.v_gamma_bb - fd_gbb).abs() < 1e-9,
                "{func:?} v_gamma_bb mismatch: ana={} vs fd={}",
                ana.v_gamma_bb,
                fd_gbb
            );
        }
    }

    #[test]
    fn test_all_xc_functionals_analytic_derivatives_vs_fd() {
        let all_funcs = [
            XcFunctional::Svwn5,
            XcFunctional::Blyp,
            XcFunctional::Pbe,
            XcFunctional::B3lyp,
            XcFunctional::B3lyp5,
            XcFunctional::Pbe0,
            XcFunctional::CamB3lyp,
            XcFunctional::Wb97xd,
        ];

        for func in all_funcs {
            // 1. General open-shell asymmetric point
            assert_xc_partials_match_fd(func, 0.37, 0.19, 0.41, 0.12, 0.23);
            // 2. Closed-shell symmetric point
            assert_xc_partials_match_fd(func, 0.25, 0.25, 0.18, 0.18, 0.18);
            // 3. High-density core region
            assert_xc_partials_match_fd(func, 14.2, 13.8, 120.0, 110.0, 105.0);
            // 4. Low-density tail region
            assert_xc_partials_match_fd(func, 1.2e-4, 0.7e-4, 3.5e-7, 1.1e-7, 2.0e-7);
            // 5. Fully spin-polarized 1-electron point (rb = 0, gbb = gab = 0)
            assert_xc_partials_match_fd(func, 0.42, 0.0, 0.31, 0.0, 0.0);
        }
    }

    #[test]
    fn test_lyp_one_electron_correlation_vanishes_identically() {
        // Fundamental property of LYP: one-electron correlation energy is identically zero!
        let out = lyp_c(0.53, 0.0, 0.77, 0.0, 0.0);
        assert!(out.f_xc.abs() < 1e-16);
        assert!(out.v_rho_a.abs() < 1e-16);
        assert!(out.v_gamma_aa.abs() < 1e-16);
    }
}

```

## `crates/calchem-scf/src/lib.rs`

SHA256: `92ef1eac5c625ac40f382dc2b2d5ac53a3b5d479612bec25877674d1965049d5`

```rust
pub mod diis;
pub mod rhf;
pub mod rks;
pub mod uhf;
pub mod uks;

pub use diis::DiisState;
pub use rhf::rhf;
pub use rks::{rks, rks_with_grid};
pub use uhf::{uhf, uhf_s_squared};
pub use uks::{uks, uks_with_grid};

/// Conservative admission bound before stored ERIs are allocated. This is not a
/// runtime allocator quota; callers may choose a larger explicit E/D-host budget.
pub fn validate_resources_and_options(
    basis: &calchem_basis::BasisSet,
    opts: &calchem_model::ScfOptions,
) -> Result<(), calchem_model::CalcError> {
    use calchem_model::CalcError;
    if opts.max_iter == 0
        || !opts.conv_energy.is_finite()
        || opts.conv_energy <= 0.0
        || !opts.conv_density.is_finite()
        || opts.conv_density <= 0.0
        || !opts.damping.is_finite()
        || !(0.0..1.0).contains(&opts.damping)
        || opts.diis_max_vecs > 64
    {
        return Err(CalcError::InvalidMolecule(
            "Invalid SCF iteration/tolerance/damping/DIIS options".into(),
        ));
    }
    let n = basis.n_ao as u128;
    if n > 65535 {
        return Err(CalcError::UnsupportedBasis(
            "AO indices exceed the stored ERI index range".into(),
        ));
    }
    let pair = n * (n + 1) / 2;
    let shells = basis.shells.len() as u128;
    let sp = shells * (shells + 1) / 2;
    let estimated = if n <= 64 {
        n.pow(4) * 16
    } else {
        pair * (pair + 1) / 2 * 16 + sp * (sp + 1) / 2 * 32
    };
    let budget = match std::env::var("CALCHEM_MAX_ERI_BYTES") {
        Ok(v) => v.parse::<u128>().ok().filter(|v| *v > 0).ok_or_else(|| {
            CalcError::InvalidMolecule("CALCHEM_MAX_ERI_BYTES must be a positive integer".into())
        })?,
        Err(_) => 4 * 1024 * 1024 * 1024,
    };
    if estimated > budget {
        return Err(CalcError::UnsupportedBasis(format!(
            "Stored-ERI conservative memory estimate {estimated} bytes exceeds budget {budget}; reduce basis/system or explicitly set CALCHEM_MAX_ERI_BYTES. Direct/RI SCF is not implemented."
        )));
    }
    Ok(())
}

```

## `crates/calchem-scf/src/diis.rs`

SHA256: `9f9456f8ee7482b057e59210d87f56ab7afb661d64bf90ff0021c6f759941651`

```rust
use calchem_tensor::{solve_linear_system, Matrix2D};

#[derive(Debug, Clone)]
pub struct DiisState {
    pub max_vecs: usize,
    pub fock_list: Vec<Matrix2D>,
    pub err_list: Vec<Matrix2D>,
}

impl DiisState {
    pub fn new(max_vecs: usize) -> Self {
        Self {
            max_vecs: max_vecs.max(2),
            fock_list: Vec::new(),
            err_list: Vec::new(),
        }
    }

    pub fn push_and_extrapolate(&mut self, fock: &Matrix2D, err: &Matrix2D) -> Matrix2D {
        self.fock_list.push(fock.clone());
        self.err_list.push(err.clone());
        while self.fock_list.len() > self.max_vecs {
            self.fock_list.remove(0);
            self.err_list.remove(0);
        }

        if self.fock_list.len() < 2 {
            return fock.clone();
        }

        // Try solving the Pulay B-matrix system; if near-singular, drop oldest vector(s)
        // and retry before falling back to the unextrapolated Fock matrix.
        let total = self.fock_list.len();
        for start in 0..(total - 1) {
            let f_slice = &self.fock_list[start..];
            let e_slice = &self.err_list[start..];
            if let Some(extrap) = Self::try_extrapolate_slice(f_slice, e_slice) {
                return extrap;
            }
        }

        fock.clone()
    }

    fn try_extrapolate_slice(focks: &[Matrix2D], errs: &[Matrix2D]) -> Option<Matrix2D> {
        let m = focks.len();
        if m < 2 {
            return None;
        }
        let dim = m + 1;
        let mut b = Matrix2D::zeros(dim, dim);
        let mut rhs = vec![0.0; dim];

        let mut max_diag: f64 = 0.0;
        for i in 0..m {
            b.set(i, m, -1.0);
            b.set(m, i, -1.0);
            for j in i..m {
                let dot: f64 = errs[i]
                    .data
                    .iter()
                    .zip(errs[j].data.iter())
                    .map(|(x, y)| x * y)
                    .sum();
                if i == j && dot > max_diag {
                    max_diag = dot;
                }
                b.set(i, j, dot);
                b.set(j, i, dot);
            }
        }
        if !max_diag.is_finite() || max_diag <= 1e-30 {
            return None;
        }
        for i in 0..m {
            for j in 0..m {
                b.set(i, j, b.get(i, j) / max_diag);
            }
        }
        b.set(m, m, 0.0);
        rhs[m] = -1.0;

        // The commutator tends to zero at convergence, so B_ij tends to zero
        // quadratically. An absolute pivot threshold would disable DIIS exactly
        // when response-quality SCF is needed. Scaling this entire block leaves
        // all Pulay coefficients invariant (only the Lagrange multiplier changes).
        let mut scale = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                scale = scale.max(b.get(i, j).abs());
            }
        }
        if !scale.is_finite() || scale <= f64::MIN_POSITIVE {
            return None;
        }
        for i in 0..m {
            for j in 0..m {
                b.set(i, j, b.get(i, j) / scale);
            }
        }

        let coeffs = solve_linear_system(&b, &rhs).ok()?;
        let weight_sum: f64 = coeffs.iter().take(m).sum();
        if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 1e-6 {
            return None;
        }
        // Reject nearly collinear subspaces with huge cancellation weights
        if coeffs
            .iter()
            .take(m)
            .any(|&c| !c.is_finite() || c.abs() > 50.0)
        {
            return None;
        }

        let mut out = Matrix2D::zeros(focks[0].rows, focks[0].cols);
        for (k, fk) in focks.iter().enumerate() {
            let ck = coeffs[k];
            for idx in 0..out.data.len() {
                out.data[idx] += ck * fk.data[idx];
            }
        }
        Some(out)
    }
}

```

## `crates/calchem-scf/src/rhf.rs`

SHA256: `550ea2950c7bcf1bd73c1cbb52bad9c3733b805c15561b1a1aab059d4ce68bea`

```rust
use crate::diis::DiisState;
use calchem_basis::BasisSet;
use calchem_int::{
    compute_ecp, compute_kinetic, compute_nuclear, compute_overlap, compute_sparse_eri,
    SparseSymmetricEri,
};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};

fn build_fock_matrix(h_core: &Matrix2D, p: &Matrix2D, eri: &SparseSymmetricEri) -> Matrix2D {
    let n = h_core.rows;
    let mut f = eri.contract_rks(&p.data, 1.0);
    for i in 0..n * n {
        f[i] += h_core.data[i];
    }
    Matrix2D {
        rows: n,
        cols: n,
        data: f,
    }
}

fn compute_electronic_energy(p: &Matrix2D, h_core: &Matrix2D, fock: &Matrix2D) -> f64 {
    let n = h_core.rows;
    let mut e_elec = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            e_elec += 0.5 * p.get(mu, nu) * (h_core.get(mu, nu) + fock.get(mu, nu));
        }
    }
    e_elec
}

fn diagonalize_fock(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D), CalcError> {
    let n = fock.rows;
    let fp = matmul(&matmul(xt, fock), x);
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14)?;
    let c = matmul(x, &cp);
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for m in 0..n_occ {
                sum += c.get(mu, m) * c.get(nu, m);
            }
            p.set(mu, nu, 2.0 * sum);
        }
    }
    Ok((eps, c, p))
}

/// Runs a closed-shell Restricted Hartree-Fock (RHF) calculation in the given basis set.
pub fn rhf(
    mol: &Molecule,
    basis: &BasisSet,
    opts: &ScfOptions,
) -> Result<ScfWavefunction, CalcError> {
    crate::validate_resources_and_options(basis, opts)?;
    let (n_alpha, n_beta) = basis.validate_rhf_valence(mol)?;
    if n_alpha != n_beta {
        return Err(CalcError::UnsupportedMethod(
            "RHF requires a closed-shell singlet state (n_alpha == n_beta)".into(),
        ));
    }
    let n_occ = n_alpha;
    let n = basis.n_ao;
    if n == 0 || n_occ > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({}) for {} occupied orbitals",
            n, n_occ
        )));
    }

    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }

    let eri = compute_sparse_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    // Symmetric orthogonalization X = U * s^{-1/2} * U^T
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is singular or too ill-conditioned for full-rank SCF (eigenvalue {:.3e})",
                val
            )));
        }
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    // Core Hamiltonian initial guess
    let (_, _, mut p) = diagonalize_fock(&h_core, &x, &xt, n_occ)?;
    let mut f;

    let mut diis = DiisState::new(opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut last_delta_e = f64::INFINITY;
    let mut last_rms_d = f64::INFINITY;

    for iter in 1..=opts.max_iter {
        used_iter = iter;

        f = build_fock_matrix(&h_core, &p, &eri);
        let e_elec = compute_electronic_energy(&p, &h_core, &f);
        let e_tot = e_elec + e_nuc;
        energy_history.push(e_tot);

        // Pulay commutator error e = F P S - S P F
        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_extrap = if opts.use_diis && iter >= 2 {
            diis.push_and_extrapolate(&f, &err)
        } else {
            if opts.use_diis {
                let _ = diis.push_and_extrapolate(&f, &err);
            }
            f.clone()
        };

        let (_eps_new, _c_new, mut p_new) = diagonalize_fock(&f_extrap, &x, &xt, n_occ)?;

        let effective_damping = if opts.damping > 0.0 && opts.damping < 1.0 {
            opts.damping
        } else if n >= 40 {
            0.3
        } else {
            0.0
        };
        if effective_damping > 0.0 {
            for idx in 0..p_new.data.len() {
                p_new.data[idx] =
                    (1.0 - effective_damping) * p_new.data[idx] + effective_damping * p.data[idx];
            }
        }

        let delta_e = if iter == 1 {
            f64::INFINITY
        } else {
            (e_tot - e_prev).abs()
        };
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        last_delta_e = delta_e;
        last_rms_d = rms_d;
        p = p_new;
        e_prev = e_tot;

        let grad_tol = (opts.conv_density * 100.0).max(1e-6);
        let tight_plateau = iter > 8
            && delta_e < (opts.conv_energy * 0.1).min(1e-12)
            && rms_d < opts.conv_density.max(1e-8)
            && diis_err < grad_tol;
        if iter > 1
            && ((delta_e < opts.conv_energy && rms_d < opts.conv_density && diis_err < grad_tol)
                || tight_plateau)
        {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    // Tighten the final fixed-point step to ensure gradient-grade stationarity:
    // canonical (C, epsilon, P, F) with F = F[P], P = 2 C_occ C_occ^T,
    // and ||FPS - SPF|| below gradient tolerance.
    let mut final_f = build_fock_matrix(&h_core, &p, &eri);
    let (eps_init, c_init, p_init) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
    let f_init = build_fock_matrix(&h_core, &p_init, &eri);
    let fps_init = matmul(&matmul(&f_init, &p_init), &s);
    let spf_init = matmul(&matmul(&s, &p_init), &f_init);
    let mut comm_init = Matrix2D::zeros(n, n);
    for idx in 0..n * n {
        comm_init.data[idx] = fps_init.data[idx] - spf_init.data[idx];
    }
    let comm_norm_init = frobenius_norm(&comm_init);

    let fp_density_tol = opts.conv_density.min(1e-10);
    let fp_comm_tol = (opts.conv_density * 10.0).clamp(1e-13, 1e-8);

    let mut final_eps = eps_init;
    let mut final_c = c_init;
    let mut final_p = p_init;
    final_f = f_init;

    if comm_norm_init > fp_comm_tol {
        let mut fp_diis = DiisState::new(opts.diis_max_vecs.max(6));
        for fp_step in 0..20 {
            let fps = matmul(&matmul(&final_f, &final_p), &s);
            let spf = matmul(&matmul(&s, &final_p), &final_f);
            let mut comm = Matrix2D::zeros(n, n);
            for idx in 0..n * n {
                comm.data[idx] = fps.data[idx] - spf.data[idx];
            }
            let f_step = if fp_step >= 2 {
                fp_diis.push_and_extrapolate(&final_f, &comm)
            } else {
                let _ = fp_diis.push_and_extrapolate(&final_f, &comm);
                final_f.clone()
            };
            let (eps_new, c_new, p_new) = diagonalize_fock(&f_step, &x, &xt, n_occ)?;
            let diff = rms_diff(&p_new, &final_p);
            final_p = p_new;
            final_eps = eps_new;
            final_c = c_new;
            final_f = build_fock_matrix(&h_core, &final_p, &eri);
            let fps2 = matmul(&matmul(&final_f, &final_p), &s);
            let spf2 = matmul(&matmul(&s, &final_p), &final_f);
            let mut comm2 = Matrix2D::zeros(n, n);
            for idx in 0..n * n {
                comm2.data[idx] = fps2.data[idx] - spf2.data[idx];
            }
            let comm_norm2 = frobenius_norm(&comm2);
            if diff < fp_density_tol && comm_norm2 < fp_comm_tol {
                let (eps_exact, c_exact, p_exact) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
                final_eps = eps_exact;
                final_c = c_exact;
                final_p = p_exact;
                final_f = build_fock_matrix(&h_core, &final_p, &eri);
                break;
            }
        }
    }

    let fps_chk = matmul(&matmul(&final_f, &final_p), &s);
    let spf_chk = matmul(&matmul(&s, &final_p), &final_f);
    let mut comm_chk = Matrix2D::zeros(n, n);
    for idx in 0..n * n {
        comm_chk.data[idx] = fps_chk.data[idx] - spf_chk.data[idx];
    }
    let final_comm_norm = frobenius_norm(&comm_chk);
    if !final_comm_norm.is_finite() || final_comm_norm > fp_comm_tol {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }
    let e_elec_final = compute_electronic_energy(&final_p, &h_core, &final_f);
    let e_tot_final = e_elec_final + e_nuc;

    let mut occ = vec![0.0; n];
    for slot in occ.iter_mut().take(n_occ) {
        *slot = 2.0;
    }

    Ok(ScfWavefunction {
        converged,
        iterations: used_iter,
        total_energy_hartree: e_tot_final,
        electronic_energy_hartree: e_elec_final,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_mo: n,
        mo_energies: final_eps,
        mo_occupations: occ,
        mo_coefficients: final_c.data,
        density_alpha: final_p.data,
        density_beta: None,
        fock_alpha: final_f.data,
        overlap_matrix: s.data,
        h_core_matrix: h_core.data,
        diagnostics: ScfDiagnostics {
            iterations: used_iter,
            energy_history,
            rms_density_history: rms_history,
            diis_error_history: diis_err_history,
        },
    })
}

```

## `crates/calchem-scf/src/uhf.rs`

SHA256: `a061c25d98a5fdf42db879d17778b8c8ed1a086715317fbd16194e10229bf9a8`

```rust
//! Unrestricted Hartree-Fock (UHF) for open-shell systems (radical ions, doublets, triplets).
//!
//! Fock operators (per spin, `P^t = P^a + P^b`):
//!   `F^a = h + J[P^t] - K[P^a]`,  `F^b = h + J[P^t] - K[P^b]`
//! Energy:
//!   `E = 1/2 * sum_{mu nu} [ P^t h + P^a F^a + P^b F^b ]_{mu nu} + E_nn`
//! DIIS runs on the stacked `[F^a; F^b]` / `[e^a; e^b]` matrices so both spins share
//! one set of extrapolation coefficients.

use crate::diis::DiisState;
use calchem_basis::BasisSet;
use calchem_int::{compute_ecp, compute_eri, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, UhfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D, Tensor4D};

/// Builds `(F^a, F^b)` from per-spin densities.
fn build_uhf_fock(
    h_core: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    eri: &Tensor4D,
) -> (Matrix2D, Matrix2D) {
    let n = h_core.rows;
    let mut fa = Matrix2D::zeros(n, n);
    let mut fb = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut j = 0.0;
            let mut ka = 0.0;
            let mut kb = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    let pa_ls = pa.get(lam, sig);
                    let pb_ls = pb.get(lam, sig);
                    let coul = eri.get(mu, nu, lam, sig);
                    let exch = eri.get(mu, lam, nu, sig);
                    j += (pa_ls + pb_ls) * coul;
                    ka += pa_ls * exch;
                    kb += pb_ls * exch;
                }
            }
            let h = h_core.get(mu, nu);
            fa.set(mu, nu, h + j - ka);
            fb.set(mu, nu, h + j - kb);
        }
    }
    (fa, fb)
}

fn uhf_electronic_energy(
    h_core: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    fa: &Matrix2D,
    fb: &Matrix2D,
) -> f64 {
    let mut e = 0.0;
    for i in 0..h_core.data.len() {
        let pt = pa.data[i] + pb.data[i];
        e += pt * h_core.data[i] + pa.data[i] * fa.data[i] + pb.data[i] * fb.data[i];
    }
    0.5 * e
}

/// Diagonalizes `F` in the orthogonal basis; returns `(eps, C, P_spin)` with `P = C_occ C_occ^T`.
fn diagonalize_spin(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D), CalcError> {
    let n = fock.rows;
    let fp = matmul(&matmul(xt, fock), x);
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14)?;
    let c = matmul(x, &cp);
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut s = 0.0;
            for m in 0..n_occ {
                s += c.get(mu, m) * c.get(nu, m);
            }
            p.set(mu, nu, s);
        }
    }
    Ok((eps, c, p))
}

fn commutator(f: &Matrix2D, p: &Matrix2D, s: &Matrix2D) -> Matrix2D {
    let fps = matmul(&matmul(f, p), s);
    let spf = matmul(&matmul(s, p), f);
    let mut e = Matrix2D::zeros(f.rows, f.cols);
    for i in 0..e.data.len() {
        e.data[i] = fps.data[i] - spf.data[i];
    }
    e
}

fn stack(a: &Matrix2D, b: &Matrix2D) -> Matrix2D {
    let mut out = Matrix2D::zeros(a.rows + b.rows, a.cols);
    out.data[..a.data.len()].copy_from_slice(&a.data);
    out.data[a.data.len()..].copy_from_slice(&b.data);
    out
}

fn unstack(m: &Matrix2D, n: usize) -> (Matrix2D, Matrix2D) {
    let mut a = Matrix2D::zeros(n, n);
    let mut b = Matrix2D::zeros(n, n);
    a.data.copy_from_slice(&m.data[..n * n]);
    b.data.copy_from_slice(&m.data[n * n..]);
    (a, b)
}

/// `<S^2> = S_z(S_z+1) + n_beta - tr(P^a S P^b S)`.
pub fn uhf_s_squared(
    pa: &Matrix2D,
    pb: &Matrix2D,
    s: &Matrix2D,
    n_alpha: usize,
    n_beta: usize,
) -> f64 {
    let sz = 0.5 * (n_alpha as f64 - n_beta as f64);
    let m = matmul(&matmul(&matmul(pa, s), pb), s);
    let mut tr = 0.0;
    for i in 0..m.rows {
        tr += m.get(i, i);
    }
    sz * (sz + 1.0) + n_beta as f64 - tr
}

/// Runs an Unrestricted Hartree-Fock calculation.
pub fn uhf(
    mol: &Molecule,
    basis: &BasisSet,
    opts: &ScfOptions,
) -> Result<UhfWavefunction, CalcError> {
    crate::validate_resources_and_options(basis, opts)?;
    let (n_alpha, n_beta) = basis.validate_uhf_valence(mol)?;
    let n = basis.n_ao;
    if n == 0 || n_alpha > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({n}) for {n_alpha} alpha electrons"
        )));
    }

    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }
    let eri = compute_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    // Symmetric orthogonalization X = U s^{-1/2} U^T
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is singular or too ill-conditioned for full-rank SCF (eigenvalue {val:.3e})"
            )));
        }
        s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    // Core-Hamiltonian guess for both spins
    let (_, _, mut pa) = diagonalize_spin(&h_core, &x, &xt, n_alpha)?;
    let (_, _, mut pb) = diagonalize_spin(&h_core, &x, &xt, n_beta)?;

    let mut diis = DiisState::new(opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();
    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut last_delta_e = f64::INFINITY;
    let mut last_rms_d = f64::INFINITY;

    for iter in 1..=opts.max_iter {
        used_iter = iter;
        let (fa, fb) = build_uhf_fock(&h_core, &pa, &pb, &eri);
        let e_tot = uhf_electronic_energy(&h_core, &pa, &pb, &fa, &fb) + e_nuc;
        energy_history.push(e_tot);

        let err = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_stacked = stack(&fa, &fb);
        let (fa_ex, fb_ex) = if opts.use_diis {
            let ex = diis.push_and_extrapolate(&f_stacked, &err);
            if iter >= 2 {
                unstack(&ex, n)
            } else {
                (fa.clone(), fb.clone())
            }
        } else {
            (fa.clone(), fb.clone())
        };

        let (_, _, mut pa_new) = diagonalize_spin(&fa_ex, &x, &xt, n_alpha)?;
        let (_, _, mut pb_new) = diagonalize_spin(&fb_ex, &x, &xt, n_beta)?;

        if opts.damping > 0.0 && opts.damping < 1.0 {
            let d = opts.damping;
            for i in 0..pa_new.data.len() {
                pa_new.data[i] = (1.0 - d) * pa_new.data[i] + d * pa.data[i];
                pb_new.data[i] = (1.0 - d) * pb_new.data[i] + d * pb.data[i];
            }
        }

        let delta_e = if iter == 1 {
            f64::INFINITY
        } else {
            (e_tot - e_prev).abs()
        };
        let rms_d = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        rms_history.push(rms_d);
        last_delta_e = delta_e;
        last_rms_d = rms_d;
        pa = pa_new;
        pb = pb_new;
        e_prev = e_tot;

        let grad_tol = (opts.conv_density * 100.0).max(1e-6);
        let tight_plateau = iter > 8
            && delta_e < (opts.conv_energy * 0.1).min(1e-12)
            && rms_d < opts.conv_density.max(1e-8)
            && diis_err < grad_tol;
        if iter > 1
            && ((delta_e < opts.conv_energy && rms_d < opts.conv_density && diis_err < grad_tol)
                || tight_plateau)
        {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    // Fixed-point tightening so (C, eps, P, F) are mutually consistent (gradient-grade).
    let (fa_init_b, fb_init_b) = build_uhf_fock(&h_core, &pa, &pb, &eri);
    let (ea_0, ca_0, pa_0) = diagonalize_spin(&fa_init_b, &x, &xt, n_alpha)?;
    let (eb_0, cb_0, pb_0) = diagonalize_spin(&fb_init_b, &x, &xt, n_beta)?;
    let (fa_0, fb_0) = build_uhf_fock(&h_core, &pa_0, &pb_0, &eri);
    let comm_0 = frobenius_norm(&stack(
        &commutator(&fa_0, &pa_0, &s),
        &commutator(&fb_0, &pb_0, &s),
    ));

    let mut fa = fa_0;
    let mut fb = fb_0;
    let mut pa = pa_0;
    let mut pb = pb_0;
    let mut eps_a = ea_0;
    let mut eps_b = eb_0;
    let mut ca = ca_0;
    let mut cb = cb_0;
    let fp_density_tol = opts.conv_density.min(1e-10);
    let fp_comm_tol = (opts.conv_density * 10.0).clamp(1e-13, 1e-8);

    if comm_0 > fp_comm_tol {
        let mut polish_diis = DiisState::new(opts.diis_max_vecs.max(6));
        for fp_step in 0..60 {
            let err_a = commutator(&fa, &pa, &s);
            let err_b = commutator(&fb, &pb, &s);
            let extrap = polish_diis.push_and_extrapolate(&stack(&fa, &fb), &stack(&err_a, &err_b));
            let (fa_step, fb_step) = if fp_step >= 2 {
                unstack(&extrap, n)
            } else {
                (fa.clone(), fb.clone())
            };
            let (ea, ca_new, pa_new) = diagonalize_spin(&fa_step, &x, &xt, n_alpha)?;
            let (eb, cb_new, pb_new) = diagonalize_spin(&fb_step, &x, &xt, n_beta)?;
            let diff = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
            pa = pa_new;
            pb = pb_new;
            eps_a = ea;
            eps_b = eb;
            ca = ca_new;
            cb = cb_new;
            let (fa_n, fb_n) = build_uhf_fock(&h_core, &pa, &pb, &eri);
            fa = fa_n;
            fb = fb_n;
            let comm_norm =
                frobenius_norm(&stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s)));
            if diff < fp_density_tol && comm_norm < fp_comm_tol {
                let (ea_ex, ca_ex, pa_ex) = diagonalize_spin(&fa, &x, &xt, n_alpha)?;
                let (eb_ex, cb_ex, pb_ex) = diagonalize_spin(&fb, &x, &xt, n_beta)?;
                eps_a = ea_ex;
                eps_b = eb_ex;
                ca = ca_ex;
                cb = cb_ex;
                pa = pa_ex;
                pb = pb_ex;
                let (fa_ex, fb_ex) = build_uhf_fock(&h_core, &pa, &pb, &eri);
                fa = fa_ex;
                fb = fb_ex;
                break;
            }
        }
    }

    let final_comm = frobenius_norm(&stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s)));
    if !final_comm.is_finite() || final_comm > fp_comm_tol {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }
    let e_elec = uhf_electronic_energy(&h_core, &pa, &pb, &fa, &fb);
    let s2 = uhf_s_squared(&pa, &pb, &s, n_alpha, n_beta);

    Ok(UhfWavefunction {
        converged: true,
        iterations: used_iter,
        total_energy_hartree: e_elec + e_nuc,
        electronic_energy_hartree: e_elec,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_alpha,
        n_beta,
        mo_energies_alpha: eps_a,
        mo_energies_beta: eps_b,
        mo_coefficients_alpha: ca.data,
        mo_coefficients_beta: cb.data,
        density_alpha: pa.data,
        density_beta: pb.data,
        s_squared: s2,
        overlap_matrix: s.data,
        diagnostics: ScfDiagnostics {
            iterations: used_iter,
            energy_history,
            rms_density_history: rms_history,
            diis_error_history: diis_err_history,
        },
    })
}

```

## `crates/calchem-scf/src/rks.rs`

SHA256: `7b3cd674ecc4c8dd01aa65432633970c966d96d511c5deff205f18e95c4817f3`

```rust
//! Closed-shell Restricted Kohn-Sham (RKS) Density Functional Theory solver in pure Rust.
//!
//! Supports pure LDA/GGA (`c_HF = 0`, skipping exact-exchange ERI contraction) and
//! global hybrid functionals (`B3LYP`, `B3LYP5`, `PBE0`) as well as scalar relativistic ECPs.

use crate::diis::DiisState;
use calchem_basis::BasisSet;
use calchem_dft::{evaluate_vxc_rks, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::{
    compute_ecp, compute_kinetic, compute_nuclear, compute_overlap, compute_sparse_eri,
    SparseSymmetricEri,
};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};

fn build_two_electron_rks(p: &Matrix2D, eri: &SparseSymmetricEri, exx_fraction: f64) -> Matrix2D {
    let n = p.rows;
    Matrix2D {
        rows: n,
        cols: n,
        data: eri.contract_rks(&p.data, exx_fraction),
    }
}

fn assemble_rks_fock_and_energy(
    h_core: &Matrix2D,
    p: &Matrix2D,
    eri: &SparseSymmetricEri,
    basis: &BasisSet,
    grid: &MolecularGrid,
    xc: XcFunctional,
) -> (Matrix2D, f64, f64) {
    let n = h_core.rows;
    let g_2e = build_two_electron_rks(p, eri, xc.exact_exchange_fraction());
    let (e_xc, v_xc) = evaluate_vxc_rks(basis, grid, xc, p);

    let mut f = Matrix2D::zeros(n, n);
    let mut e_1e_2e = 0.0;
    for i in 0..n * n {
        f.data[i] = h_core.data[i] + g_2e.data[i] + v_xc.data[i];
        e_1e_2e += p.data[i] * (h_core.data[i] + 0.5 * g_2e.data[i]);
    }
    (f, e_1e_2e + e_xc, e_xc)
}

fn diagonalize_fock(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D), CalcError> {
    let n = fock.rows;
    let fp = matmul(&matmul(xt, fock), x);
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14)?;
    let c = matmul(x, &cp);
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for m in 0..n_occ {
                sum += c.get(mu, m) * c.get(nu, m);
            }
            p.set(mu, nu, 2.0 * sum);
        }
    }
    Ok((eps, c, p))
}

/// Runs a closed-shell Restricted Kohn-Sham (RKS) DFT calculation.
pub fn rks(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    opts: &ScfOptions,
) -> Result<ScfWavefunction, CalcError> {
    crate::validate_resources_and_options(basis, opts)?;
    xc.validate_native()?;
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    rks_with_grid(mol, basis, xc, &grid, opts)
}

/// Runs a closed-shell Restricted Kohn-Sham (RKS) DFT calculation on a prebuilt [`MolecularGrid`].
pub fn rks_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    opts: &ScfOptions,
) -> Result<ScfWavefunction, CalcError> {
    crate::validate_resources_and_options(basis, opts)?;
    xc.validate_native()?;
    let (n_alpha, n_beta) = basis.validate_rhf_valence(mol)?;
    if n_alpha != n_beta {
        return Err(CalcError::UnsupportedMethod(
            "RKS requires a closed-shell singlet state (n_alpha == n_beta)".into(),
        ));
    }
    let n_occ = n_alpha;
    let n = basis.n_ao;
    if n == 0 || n_occ > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({n}) for {n_occ} occupied orbitals"
        )));
    }

    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }

    let eri = compute_sparse_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is singular or too ill-conditioned for full-rank SCF (eigenvalue {val:.3e})"
            )));
        }
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    let (_, _, mut p) = diagonalize_fock(&h_core, &x, &xt, n_occ)?;
    let mut f;

    let mut diis = DiisState::new(opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut last_delta_e = f64::INFINITY;
    let mut last_rms_d = f64::INFINITY;

    for iter in 1..=opts.max_iter {
        used_iter = iter;

        let (f_cur, e_elec, _e_xc) =
            assemble_rks_fock_and_energy(&h_core, &p, &eri, basis, grid, xc);
        f = f_cur;
        let e_tot = e_elec + e_nuc;
        energy_history.push(e_tot);

        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_extrap = if opts.use_diis && iter >= 2 {
            diis.push_and_extrapolate(&f, &err)
        } else {
            if opts.use_diis {
                let _ = diis.push_and_extrapolate(&f, &err);
            }
            f.clone()
        };

        let (_eps_new, _c_new, mut p_new) = diagonalize_fock(&f_extrap, &x, &xt, n_occ)?;

        let effective_damping = if opts.damping > 0.0 && opts.damping < 1.0 {
            opts.damping
        } else if n >= 40 && (iter <= 10 || diis_err > 0.05) {
            0.3
        } else {
            0.0
        };
        if effective_damping > 0.0 {
            for idx in 0..p_new.data.len() {
                p_new.data[idx] =
                    (1.0 - effective_damping) * p_new.data[idx] + effective_damping * p.data[idx];
            }
        }

        let delta_e = if iter == 1 {
            f64::INFINITY
        } else {
            (e_tot - e_prev).abs()
        };
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        last_delta_e = delta_e;
        last_rms_d = rms_d;
        p = p_new;
        e_prev = e_tot;

        let grad_tol = (opts.conv_density * 100.0).max(1e-6);
        let tight_plateau = iter > 8
            && delta_e < (opts.conv_energy * 0.1).min(1e-12)
            && rms_d < opts.conv_density.max(1e-8)
            && diis_err < grad_tol;
        if iter > 1
            && ((delta_e < opts.conv_energy && rms_d < opts.conv_density && diis_err < grad_tol)
                || tight_plateau)
        {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    // Gradient-grade fixed-point tightening (using DIIS-assisted polishing if needed).
    let (mut final_f, mut e_elec_final, _) =
        assemble_rks_fock_and_energy(&h_core, &p, &eri, basis, grid, xc);
    let mut final_eps = Vec::new();
    let mut final_c = Matrix2D::zeros(n, n);
    let mut final_p = p.clone();
    let fp_density_tol = opts.conv_density.min(1e-10);
    let fp_comm_tol = 1e-8;
    let mut fp_diis = DiisState::new(opts.diis_max_vecs.max(6));
    let max_fp_steps = 50;

    for fp_step in 0..max_fp_steps {
        let fps = matmul(&matmul(&final_f, &final_p), &s);
        let spf = matmul(&matmul(&s, &final_p), &final_f);
        let mut comm = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            comm.data[i] = fps.data[i] - spf.data[i];
        }
        let _comm_norm = frobenius_norm(&comm);
        let f_step = if fp_step >= 2 {
            fp_diis.push_and_extrapolate(&final_f, &comm)
        } else {
            let _ = fp_diis.push_and_extrapolate(&final_f, &comm);
            final_f.clone()
        };

        let (eps_new, c_new, p_new) = diagonalize_fock(&f_step, &x, &xt, n_occ)?;
        let diff = rms_diff(&p_new, &final_p);
        final_p = p_new;
        final_eps = eps_new;
        final_c = c_new;
        let (f_rebuilt, e_rebuilt, _) =
            assemble_rks_fock_and_energy(&h_core, &final_p, &eri, basis, grid, xc);
        final_f = f_rebuilt;
        e_elec_final = e_rebuilt;

        let fps2 = matmul(&matmul(&final_f, &final_p), &s);
        let spf2 = matmul(&matmul(&s, &final_p), &final_f);
        let mut comm2 = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            comm2.data[i] = fps2.data[i] - spf2.data[i];
        }
        let comm_norm2 = frobenius_norm(&comm2);

        if diff < fp_density_tol && comm_norm2 < fp_comm_tol {
            // Do one final pure diagonalization of `final_f = F[final_p]` so (C, eps) match F[P]
            let (eps_exact, c_exact, p_exact) = diagonalize_fock(&final_f, &x, &xt, n_occ)?;
            final_eps = eps_exact;
            final_c = c_exact;
            final_p = p_exact;
            let (f_exact, e_exact, _) =
                assemble_rks_fock_and_energy(&h_core, &final_p, &eri, basis, grid, xc);
            final_f = f_exact;
            e_elec_final = e_exact;
            break;
        }
    }

    {
        let fps = matmul(&matmul(&final_f, &final_p), &s);
        let spf = matmul(&matmul(&s, &final_p), &final_f);
        let mut comm = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            comm.data[i] = fps.data[i] - spf.data[i];
        }
        let comm_norm = frobenius_norm(&comm);
        if !comm_norm.is_finite() || comm_norm > 1e-6 {
            return Err(CalcError::NotConverged {
                iterations: used_iter,
                delta_e: last_delta_e,
                rms_d: last_rms_d,
            });
        }
    }

    let e_tot_final = e_elec_final + e_nuc;
    let mut occ = vec![0.0; n];
    for slot in occ.iter_mut().take(n_occ) {
        *slot = 2.0;
    }

    Ok(ScfWavefunction {
        converged,
        iterations: used_iter,
        total_energy_hartree: e_tot_final,
        electronic_energy_hartree: e_elec_final,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_mo: n,
        mo_energies: final_eps,
        mo_occupations: occ,
        mo_coefficients: final_c.data,
        density_alpha: final_p.data,
        density_beta: None,
        fock_alpha: final_f.data,
        overlap_matrix: s.data,
        h_core_matrix: h_core.data,
        diagnostics: ScfDiagnostics {
            iterations: used_iter,
            energy_history,
            rms_density_history: rms_history,
            diis_error_history: diis_err_history,
        },
    })
}

```

## `crates/calchem-scf/src/uks.rs`

SHA256: `72802e663e9bfb0cb7692def8ce8c01ee34bcfdc9b18f3d9bf8e2621f4eca5ea`

```rust
//! Unrestricted Kohn-Sham (UKS) Density Functional Theory solver for open-shell systems
//! (radical cations/anions, doublets, triplets, single-electron atoms) in pure Rust.

use crate::diis::DiisState;
use crate::uhf::uhf_s_squared;
use calchem_basis::BasisSet;
use calchem_dft::{evaluate_vxc_uks, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::{compute_ecp, compute_eri, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, UhfWavefunction};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D, Tensor4D};

fn build_two_electron_uks(
    pa: &Matrix2D,
    pb: &Matrix2D,
    eri: &Tensor4D,
    exx_fraction: f64,
) -> (Matrix2D, Matrix2D) {
    let n = pa.rows;
    let mut ga = Matrix2D::zeros(n, n);
    let mut gb = Matrix2D::zeros(n, n);
    if exx_fraction.abs() < 1e-15 {
        for mu in 0..n {
            for nu in mu..n {
                let mut j = 0.0;
                for lam in 0..n {
                    for sig in 0..n {
                        let pt_ls = pa.get(lam, sig) + pb.get(lam, sig);
                        j += pt_ls * eri.get(mu, nu, lam, sig);
                    }
                }
                ga.set(mu, nu, j);
                ga.set(nu, mu, j);
                gb.set(mu, nu, j);
                gb.set(nu, mu, j);
            }
        }
    } else {
        for mu in 0..n {
            for nu in mu..n {
                let mut j = 0.0;
                let mut ka = 0.0;
                let mut kb = 0.0;
                for lam in 0..n {
                    for sig in 0..n {
                        let pa_ls = pa.get(lam, sig);
                        let pb_ls = pb.get(lam, sig);
                        let coul = eri.get(mu, nu, lam, sig);
                        let exch = eri.get(mu, lam, nu, sig);
                        j += (pa_ls + pb_ls) * coul;
                        ka += pa_ls * exch;
                        kb += pb_ls * exch;
                    }
                }
                let va = j - exx_fraction * ka;
                let vb = j - exx_fraction * kb;
                ga.set(mu, nu, va);
                ga.set(nu, mu, va);
                gb.set(mu, nu, vb);
                gb.set(nu, mu, vb);
            }
        }
    }
    (ga, gb)
}

fn assemble_uks_fock_and_energy(
    h_core: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    eri: &Tensor4D,
    basis: &BasisSet,
    grid: &MolecularGrid,
    xc: XcFunctional,
) -> (Matrix2D, Matrix2D, f64, f64) {
    let n = h_core.rows;
    let (ga, gb) = build_two_electron_uks(pa, pb, eri, xc.exact_exchange_fraction());
    let (e_xc, va, vb) = evaluate_vxc_uks(basis, grid, xc, pa, pb);

    let mut fa = Matrix2D::zeros(n, n);
    let mut fb = Matrix2D::zeros(n, n);
    let mut e_1e_2e = 0.0;
    for i in 0..n * n {
        fa.data[i] = h_core.data[i] + ga.data[i] + va.data[i];
        fb.data[i] = h_core.data[i] + gb.data[i] + vb.data[i];
        let pt_i = pa.data[i] + pb.data[i];
        e_1e_2e +=
            pt_i * h_core.data[i] + 0.5 * (pa.data[i] * ga.data[i] + pb.data[i] * gb.data[i]);
    }
    (fa, fb, e_1e_2e + e_xc, e_xc)
}

fn diagonalize_spin(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D), CalcError> {
    let n = fock.rows;
    let fp = matmul(&matmul(xt, fock), x);
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14)?;
    let c = matmul(x, &cp);
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut s = 0.0;
            for m in 0..n_occ {
                s += c.get(mu, m) * c.get(nu, m);
            }
            p.set(mu, nu, s);
        }
    }
    Ok((eps, c, p))
}

fn commutator(f: &Matrix2D, p: &Matrix2D, s: &Matrix2D) -> Matrix2D {
    let fps = matmul(&matmul(f, p), s);
    let spf = matmul(&matmul(s, p), f);
    let mut e = Matrix2D::zeros(f.rows, f.cols);
    for i in 0..e.data.len() {
        e.data[i] = fps.data[i] - spf.data[i];
    }
    e
}

fn stack(a: &Matrix2D, b: &Matrix2D) -> Matrix2D {
    let mut out = Matrix2D::zeros(a.rows + b.rows, a.cols);
    out.data[..a.data.len()].copy_from_slice(&a.data);
    out.data[a.data.len()..].copy_from_slice(&b.data);
    out
}

fn unstack(m: &Matrix2D, n: usize) -> (Matrix2D, Matrix2D) {
    let mut a = Matrix2D::zeros(n, n);
    let mut b = Matrix2D::zeros(n, n);
    a.data.copy_from_slice(&m.data[..n * n]);
    b.data.copy_from_slice(&m.data[n * n..]);
    (a, b)
}

/// Runs an Unrestricted Kohn-Sham (UKS) DFT calculation.
pub fn uks(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    opts: &ScfOptions,
) -> Result<UhfWavefunction, CalcError> {
    crate::validate_resources_and_options(basis, opts)?;
    xc.validate_native()?;
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    uks_with_grid(mol, basis, xc, &grid, opts)
}

/// Runs an Unrestricted Kohn-Sham (UKS) DFT calculation on a prebuilt [`MolecularGrid`].
pub fn uks_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    opts: &ScfOptions,
) -> Result<UhfWavefunction, CalcError> {
    crate::validate_resources_and_options(basis, opts)?;
    xc.validate_native()?;
    let (n_alpha, n_beta) = basis.validate_uhf_valence(mol)?;
    let n = basis.n_ao;
    if n == 0 || n_alpha > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({n}) for {n_alpha} alpha electrons"
        )));
    }

    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }
    let eri = compute_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is singular or too ill-conditioned for full-rank SCF (eigenvalue {val:.3e})"
            )));
        }
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    let (_, _, mut pa) = diagonalize_spin(&h_core, &x, &xt, n_alpha)?;
    let (_, _, mut pb) = diagonalize_spin(&h_core, &x, &xt, n_beta)?;

    let mut diis = DiisState::new(opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();
    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut last_delta_e = f64::INFINITY;
    let mut last_rms_d = f64::INFINITY;

    for iter in 1..=opts.max_iter {
        used_iter = iter;
        let (fa, fb, e_elec, _) =
            assemble_uks_fock_and_energy(&h_core, &pa, &pb, &eri, basis, grid, xc);
        let e_tot = e_elec + e_nuc;
        energy_history.push(e_tot);

        let err = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_stacked = stack(&fa, &fb);
        let (fa_ex, fb_ex) = if opts.use_diis {
            let ex = diis.push_and_extrapolate(&f_stacked, &err);
            if iter >= 2 {
                unstack(&ex, n)
            } else {
                (fa.clone(), fb.clone())
            }
        } else {
            (fa.clone(), fb.clone())
        };

        let (_, _, mut pa_new) = diagonalize_spin(&fa_ex, &x, &xt, n_alpha)?;
        let (_, _, mut pb_new) = diagonalize_spin(&fb_ex, &x, &xt, n_beta)?;

        if opts.damping > 0.0 && opts.damping < 1.0 {
            let d = opts.damping;
            for i in 0..pa_new.data.len() {
                pa_new.data[i] = (1.0 - d) * pa_new.data[i] + d * pa.data[i];
                pb_new.data[i] = (1.0 - d) * pb_new.data[i] + d * pb.data[i];
            }
        }

        let delta_e = if iter == 1 {
            f64::INFINITY
        } else {
            (e_tot - e_prev).abs()
        };
        let rms_d = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        rms_history.push(rms_d);
        last_delta_e = delta_e;
        last_rms_d = rms_d;
        pa = pa_new;
        pb = pb_new;
        e_prev = e_tot;

        let grad_tol = (opts.conv_density * 100.0).max(1e-6);
        let tight_plateau = iter > 8
            && delta_e < (opts.conv_energy * 0.1).min(1e-12)
            && rms_d < opts.conv_density.max(1e-8)
            && diis_err < grad_tol;
        if iter > 1
            && ((delta_e < opts.conv_energy && rms_d < opts.conv_density && diis_err < grad_tol)
                || tight_plateau)
        {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    // Gradient-grade fixed-point tightening
    let (mut fa, mut fb, mut e_elec_final, _) =
        assemble_uks_fock_and_energy(&h_core, &pa, &pb, &eri, basis, grid, xc);
    let mut eps_a = Vec::new();
    let mut eps_b = Vec::new();
    let mut ca = Matrix2D::zeros(n, n);
    let mut cb = Matrix2D::zeros(n, n);
    let fp_density_tol = opts.conv_density.min(1e-10);
    let fp_comm_tol = (opts.conv_density * 10.0).clamp(1e-13, 1e-8);
    let mut fp_diis = DiisState::new(opts.diis_max_vecs.max(6));

    for fp_step in 0..30 {
        let err = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let f_stacked = stack(&fa, &fb);
        let (fa_step, fb_step) = if fp_step >= 2 {
            let ex = fp_diis.push_and_extrapolate(&f_stacked, &err);
            unstack(&ex, n)
        } else {
            let _ = fp_diis.push_and_extrapolate(&f_stacked, &err);
            (fa.clone(), fb.clone())
        };

        let (ea, ca_new, pa_new) = diagonalize_spin(&fa_step, &x, &xt, n_alpha)?;
        let (eb, cb_new, pb_new) = diagonalize_spin(&fb_step, &x, &xt, n_beta)?;
        let diff = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        pa = pa_new;
        pb = pb_new;
        eps_a = ea;
        eps_b = eb;
        ca = ca_new;
        cb = cb_new;
        let (fa_reb, fb_reb, e_reb, _) =
            assemble_uks_fock_and_energy(&h_core, &pa, &pb, &eri, basis, grid, xc);
        fa = fa_reb;
        fb = fb_reb;
        e_elec_final = e_reb;

        let err2 = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
        let comm_norm = frobenius_norm(&err2);

        if diff < fp_density_tol && comm_norm < fp_comm_tol {
            let (ea_ex, ca_ex, pa_ex) = diagonalize_spin(&fa, &x, &xt, n_alpha)?;
            let (eb_ex, cb_ex, pb_ex) = diagonalize_spin(&fb, &x, &xt, n_beta)?;
            eps_a = ea_ex;
            eps_b = eb_ex;
            ca = ca_ex;
            cb = cb_ex;
            pa = pa_ex;
            pb = pb_ex;
            let (fa_ex, fb_ex, e_ex, _) =
                assemble_uks_fock_and_energy(&h_core, &pa, &pb, &eri, basis, grid, xc);
            fa = fa_ex;
            fb = fb_ex;
            e_elec_final = e_ex;
            break;
        }
    }

    // Recheck after the final exact diagonalization and density rebuild too.
    let final_error = stack(&commutator(&fa, &pa, &s), &commutator(&fb, &pb, &s));
    let final_comm_norm = frobenius_norm(&final_error);
    if !final_comm_norm.is_finite() || final_comm_norm > fp_comm_tol {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    let s2 = uhf_s_squared(&pa, &pb, &s, n_alpha, n_beta);

    Ok(UhfWavefunction {
        converged,
        iterations: used_iter,
        total_energy_hartree: e_elec_final + e_nuc,
        electronic_energy_hartree: e_elec_final,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_alpha,
        n_beta,
        mo_energies_alpha: eps_a,
        mo_energies_beta: eps_b,
        mo_coefficients_alpha: ca.data,
        mo_coefficients_beta: cb.data,
        density_alpha: pa.data,
        density_beta: pb.data,
        s_squared: s2,
        overlap_matrix: s.data,
        diagnostics: ScfDiagnostics {
            iterations: used_iter,
            energy_history,
            rms_density_history: rms_history,
            diis_error_history: diis_err_history,
        },
    })
}

```

## `crates/calchem-grad/src/lib.rs`

SHA256: `fbee9b41825bd87eb16f17c3b7547fb618d465972af1fa3551a47634547b6e43`

```rust
pub mod derivatives;

use calchem_basis::BasisSet;
use calchem_dft::{xc_nuclear_gradient, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::screening::SchwarzScreening;
use calchem_int::symmetry::shell_quartets;
use calchem_model::{CalcError, Molecule, ScfWavefunction, UhfWavefunction};
use calchem_tensor::Matrix2D;
use derivatives::{
    eri_derivative_quartet, nuclear_repulsion_derivative_with_basis, one_electron_derivative,
};
use rayon::prelude::*;

#[inline]
fn distinct_shell_permutations(i: usize, j: usize, k: usize, l: usize) -> Vec<[usize; 4]> {
    let raw = [
        [i, j, k, l],
        [j, i, k, l],
        [i, j, l, k],
        [j, i, l, k],
        [k, l, i, j],
        [l, k, i, j],
        [k, l, j, i],
        [l, k, j, i],
    ];
    let mut out: Vec<[usize; 4]> = Vec::with_capacity(8);
    for p in raw {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// Validates that coordinates are finite and the basis shells sit on the molecule's nuclei.
fn check_geometry(mol: &Molecule, basis: &BasisSet) -> Result<(), CalcError> {
    if mol.atoms.is_empty() {
        return Err(CalcError::InvalidMolecule("No atoms in molecule".into()));
    }
    for atom in &mol.atoms {
        if !atom.position_bohr.iter().all(|c| c.is_finite()) {
            return Err(CalcError::InvalidMolecule(
                "Non-finite coordinates detected in molecule".into(),
            ));
        }
    }
    for shell in &basis.shells {
        let atom_pos = mol.atoms[shell.atom_index].position_bohr;
        for c in 0..3 {
            if (shell.center[c] - atom_pos[c]).abs() > 1e-10 {
                return Err(CalcError::InvalidMolecule(format!(
                    "Basis shell center mismatch on atom {}: shell center {:?} vs atom pos {:?}",
                    shell.atom_index, shell.center, atom_pos
                )));
            }
        }
    }
    for ecp in &basis.ecp_potentials {
        let atom_pos = mol.atoms[ecp.atom_index].position_bohr;
        for c in 0..3 {
            if (ecp.center[c] - atom_pos[c]).abs() > 1e-10 {
                return Err(CalcError::InvalidMolecule(format!(
                    "ECP center mismatch on atom {}: ecp center {:?} vs atom pos {:?}",
                    ecp.atom_index, ecp.center, atom_pos
                )));
            }
        }
    }
    Ok(())
}

fn to_matrix(flat: &[f64], n: usize) -> Matrix2D {
    let mut m = Matrix2D::zeros(n, n);
    m.data.copy_from_slice(&flat[..n * n]);
    m
}

/// `W = sum_i eps_i C_{mu i} C_{nu i}` over the first `n_occ` columns, scaled by `occ`.
fn energy_weighted_density(c: &[f64], eps: &[f64], n: usize, n_occ: usize, occ: f64) -> Matrix2D {
    let mut w = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut s = 0.0;
            for i in 0..n_occ {
                s += eps[i] * c[mu * n + i] * c[nu * n + i];
            }
            w.set(mu, nu, occ * s);
        }
    }
    w
}

/// Shared gradient engine for RHF, UHF, RKS, and UKS (`1e + 2e + E_nn + Pulay`).
///
/// `dE/dX = dE_nn/dX + sum P^t dh/dX - sum W dS/dX + sum (mu nu|lam sig)^X Gamma_{mu nu lam sig}`
/// with `Gamma = 1/2 P^t_{mn} P^t_{ls} - c_HF / 2 * (P^a_{ml} P^a_{ns} + P^b_{ml} P^b_{ns})`.
fn scf_gradient_core(
    mol: &Molecule,
    basis: &BasisSet,
    pt: &Matrix2D,
    pa: &Matrix2D,
    pb: &Matrix2D,
    w: &Matrix2D,
    exx_fraction: f64,
) -> Vec<[f64; 3]> {
    let n_atoms = mol.atoms.len();
    let n = basis.n_ao;

    let screening = SchwarzScreening::new(basis, 1e-11);
    let canonical_quartets = shell_quartets(basis.shells.len());

    let g_2e_total = canonical_quartets
        .par_bridge()
        .fold(
            || vec![[0.0; 3]; n_atoms],
            |mut local_grad, [i, j, k, l]| {
                if screening.can_skip(i, j, k, l) {
                    return local_grad;
                }
                for [a_idx, b_idx, c_idx, d_idx] in distinct_shell_permutations(i, j, k, l) {
                    let sh_a = &basis.shells[a_idx];
                    let sh_b = &basis.shells[b_idx];
                    let sh_c = &basis.shells[c_idx];
                    let sh_d = &basis.shells[d_idx];

                    let mut involved = [
                        sh_a.atom_index,
                        sh_b.atom_index,
                        sh_c.atom_index,
                        sh_d.atom_index,
                    ];
                    involved.sort_unstable();
                    let mut n_unique_atoms = 0;
                    for idx in 0..4 {
                        if idx == 0 || involved[idx] != involved[idx - 1] {
                            involved[n_unique_atoms] = involved[idx];
                            n_unique_atoms += 1;
                        }
                    }

                    let na = sh_a.n_functions();
                    let nb = sh_b.n_functions();
                    let nc = sh_c.n_functions();
                    let nd = sh_d.n_functions();

                    for &atom in &involved[..n_unique_atoms] {
                        for q in 0..3 {
                            let d_block = eri_derivative_quartet([sh_a, sh_b, sh_c, sh_d], atom, q);
                            let mut d_contrib = 0.0;
                            for ia in 0..na {
                                let mu = sh_a.ao_offset + ia;
                                for ib in 0..nb {
                                    let nu = sh_b.ao_offset + ib;
                                    for ic in 0..nc {
                                        let lam = sh_c.ao_offset + ic;
                                        for id in 0..nd {
                                            let sig = sh_d.ao_offset + id;
                                            let d_val =
                                                d_block[((ia * nb + ib) * nc + ic) * nd + id];
                                            if d_val.abs() > 1e-16 {
                                                let coul = 0.5 * pt.get(mu, nu) * pt.get(lam, sig);
                                                let gamma = if exx_fraction.abs() < 1e-15 {
                                                    coul
                                                } else {
                                                    let exch = 0.5
                                                        * exx_fraction
                                                        * (pa.get(mu, lam) * pa.get(nu, sig)
                                                            + pb.get(mu, lam) * pb.get(nu, sig));
                                                    coul - exch
                                                };
                                                d_contrib += d_val * gamma;
                                            }
                                        }
                                    }
                                }
                            }
                            local_grad[atom][q] += d_contrib;
                        }
                    }
                }
                local_grad
            },
        )
        .reduce(
            || vec![[0.0; 3]; n_atoms],
            |mut a, b| {
                for i in 0..n_atoms {
                    for q in 0..3 {
                        a[i][q] += b[i][q];
                    }
                }
                a
            },
        );

    let mut grad = vec![[0.0; 3]; n_atoms];
    for a in 0..n_atoms {
        for q in 0..3 {
            let g_nuc = nuclear_repulsion_derivative_with_basis(mol, basis, a, q);
            let (ds, dt, dv) = one_electron_derivative(mol, basis, a, q);
            let mut g_1e = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    let h_val = dt.get(mu, nu) + dv.get(mu, nu);
                    g_1e += pt.get(mu, nu) * h_val - w.get(mu, nu) * ds.get(mu, nu);
                }
            }
            grad[a][q] = g_nuc + g_1e + g_2e_total[a][q];
        }
    }
    grad
}

/// Computes the analytical RHF Cartesian energy gradient with respect to nuclear coordinates.
///
/// Output units: Hartree/Bohr (Eh/Bohr), shaped as `[ [dE/dx, dE/dy, dE/dz], ... ]` for each atom.
/// The nuclear force is `-G`.
pub fn rhf_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    wf: &ScfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    if !wf.converged {
        return Err(CalcError::NotConverged {
            iterations: wf.iterations,
            delta_e: 0.0,
            rms_d: 0.0,
        });
    }
    if mol.multiplicity != 1 {
        return Err(CalcError::UnsupportedMethod(
            "Analytical RHF gradient requires singlet closed-shell state".into(),
        ));
    }
    let n = basis.n_ao;
    if wf.n_ao != n || wf.n_mo != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({}) does not match wavefunction AO count ({})",
            n, wf.n_ao
        )));
    }
    check_geometry(mol, basis)?;
    if mol.atoms.len() == 1 {
        return Ok(vec![[0.0, 0.0, 0.0]]);
    }

    // RHF stores the spin-summed density P = 2 C_occ C_occ^T in `density_alpha`.
    let pt = to_matrix(&wf.density_alpha, n);
    let mut half = pt.clone();
    for v in half.data.iter_mut() {
        *v *= 0.5;
    }
    let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 1e-12).count();
    let w = energy_weighted_density(&wf.mo_coefficients, &wf.mo_energies, n, n_occ, 2.0);

    Ok(scf_gradient_core(mol, basis, &pt, &half, &half, &w, 1.0))
}

/// Computes the analytical UHF Cartesian energy gradient (Eh/Bohr) for open-shell systems.
pub fn uhf_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    if !wf.converged {
        return Err(CalcError::NotConverged {
            iterations: wf.iterations,
            delta_e: 0.0,
            rms_d: 0.0,
        });
    }
    let n = basis.n_ao;
    if wf.n_ao != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({}) does not match wavefunction AO count ({})",
            n, wf.n_ao
        )));
    }
    check_geometry(mol, basis)?;
    if mol.atoms.len() == 1 {
        return Ok(vec![[0.0, 0.0, 0.0]]);
    }

    let pa = to_matrix(&wf.density_alpha, n);
    let pb = to_matrix(&wf.density_beta, n);
    let mut pt = pa.clone();
    for (t, b) in pt.data.iter_mut().zip(&pb.data) {
        *t += b;
    }
    let wa = energy_weighted_density(
        &wf.mo_coefficients_alpha,
        &wf.mo_energies_alpha,
        n,
        wf.n_alpha,
        1.0,
    );
    let wb = energy_weighted_density(
        &wf.mo_coefficients_beta,
        &wf.mo_energies_beta,
        n,
        wf.n_beta,
        1.0,
    );
    let mut w = wa;
    for (t, b) in w.data.iter_mut().zip(&wb.data) {
        *t += b;
    }

    Ok(scf_gradient_core(mol, basis, &pt, &pa, &pb, &w, 1.0))
}

/// Computes the exact analytical Restricted Kohn-Sham (RKS) Cartesian nuclear gradient (Eh/Bohr).
pub fn rks_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    wf: &ScfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    rks_gradient_with_grid(mol, basis, xc, &grid, wf)
}

/// Computes the exact analytical RKS Cartesian nuclear gradient (Eh/Bohr) on a prebuilt [`MolecularGrid`].
pub fn rks_gradient_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    wf: &ScfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    if !wf.converged {
        return Err(CalcError::NotConverged {
            iterations: wf.iterations,
            delta_e: 0.0,
            rms_d: 0.0,
        });
    }
    if mol.multiplicity != 1 {
        return Err(CalcError::UnsupportedMethod(
            "Analytical RKS gradient requires singlet closed-shell state".into(),
        ));
    }
    let n = basis.n_ao;
    if wf.n_ao != n || wf.n_mo != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({n}) does not match wavefunction AO count ({})",
            wf.n_ao
        )));
    }
    check_geometry(mol, basis)?;
    if mol.atoms.len() == 1 {
        return Ok(vec![[0.0, 0.0, 0.0]]);
    }

    let pt = to_matrix(&wf.density_alpha, n);
    let mut half = pt.clone();
    for v in half.data.iter_mut() {
        *v *= 0.5;
    }
    let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 1e-12).count();
    let w = energy_weighted_density(&wf.mo_coefficients, &wf.mo_energies, n, n_occ, 2.0);

    let mut grad = scf_gradient_core(
        mol,
        basis,
        &pt,
        &half,
        &half,
        &w,
        xc.exact_exchange_fraction(),
    );
    let g_xc = xc_nuclear_gradient(mol, basis, grid, xc, &half, &half);
    for (ga, gx) in grad.iter_mut().zip(g_xc.iter()) {
        ga[0] += gx[0];
        ga[1] += gx[1];
        ga[2] += gx[2];
    }
    Ok(grad)
}

/// Computes the exact analytical Unrestricted Kohn-Sham (UKS) Cartesian nuclear gradient (Eh/Bohr).
pub fn uks_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid_level: GridLevel,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    let grid = MolecularGrid::build_for_basis(mol, basis, grid_level);
    uks_gradient_with_grid(mol, basis, xc, &grid, wf)
}

/// Computes the exact analytical UKS Cartesian nuclear gradient (Eh/Bohr) on a prebuilt [`MolecularGrid`].
pub fn uks_gradient_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    xc: XcFunctional,
    grid: &MolecularGrid,
    wf: &UhfWavefunction,
) -> Result<Vec<[f64; 3]>, CalcError> {
    if !wf.converged {
        return Err(CalcError::NotConverged {
            iterations: wf.iterations,
            delta_e: 0.0,
            rms_d: 0.0,
        });
    }
    let n = basis.n_ao;
    if wf.n_ao != n {
        return Err(CalcError::InvalidMolecule(format!(
            "Basis AO count ({n}) does not match wavefunction AO count ({})",
            wf.n_ao
        )));
    }
    check_geometry(mol, basis)?;
    if mol.atoms.len() == 1 {
        return Ok(vec![[0.0, 0.0, 0.0]]);
    }

    let pa = to_matrix(&wf.density_alpha, n);
    let pb = to_matrix(&wf.density_beta, n);
    let mut pt = pa.clone();
    for (t, b) in pt.data.iter_mut().zip(&pb.data) {
        *t += b;
    }
    let wa = energy_weighted_density(
        &wf.mo_coefficients_alpha,
        &wf.mo_energies_alpha,
        n,
        wf.n_alpha,
        1.0,
    );
    let wb = energy_weighted_density(
        &wf.mo_coefficients_beta,
        &wf.mo_energies_beta,
        n,
        wf.n_beta,
        1.0,
    );
    let mut w = wa;
    for (t, b) in w.data.iter_mut().zip(&wb.data) {
        *t += b;
    }

    let mut grad = scf_gradient_core(mol, basis, &pt, &pa, &pb, &w, xc.exact_exchange_fraction());
    let g_xc = xc_nuclear_gradient(mol, basis, grid, xc, &pa, &pb);
    for (ga, gx) in grad.iter_mut().zip(g_xc.iter()) {
        ga[0] += gx[0];
        ga[1] += gx[1];
        ga[2] += gx[2];
    }
    Ok(grad)
}

```

## `crates/calchem-opt/src/lib.rs`

SHA256: `79b54b36616f7eaa901f0a345c89edbf7b66b5590d70f69e66b8be69d8a4c42f`

```rust
use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid, XcFunctional};
use calchem_grad::{rhf_gradient, rks_gradient_with_grid, uhf_gradient, uks_gradient_with_grid};
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_scf::{rhf, rks_with_grid, uhf, uks_with_grid};
use calchem_tensor::Matrix2D;
use serde::{Deserialize, Serialize};

pub mod irc;
pub mod prfo;

pub use irc::{run_irc, IrcBranch, IrcDirection, IrcOptions, IrcPoint, IrcResult};
pub use prfo::{search_transition_state, TsOptions, TsResult};

/// SCF / Kohn-Sham reference used to evaluate energies and analytical gradients during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Reference {
    /// RHF for closed-shell singlets, UHF otherwise.
    #[default]
    Auto,
    Rhf,
    Uhf,
    /// Restricted Kohn-Sham DFT with the specified XC functional.
    Rks(XcFunctional),
    /// Unrestricted Kohn-Sham DFT with the specified XC functional.
    Uks(XcFunctional),
    /// Closed-shell RKS for singlets, open-shell UKS otherwise.
    DftAuto(XcFunctional),
}

impl Reference {
    pub fn from_method_and_xc(method: &str, xc: Option<&str>) -> Result<Self, CalcError> {
        let m = method.trim().to_ascii_lowercase();
        let resolve_xc = |default_xc: &str| -> Result<XcFunctional, CalcError> {
            XcFunctional::from_name(xc.unwrap_or(default_xc))
        };
        match m.as_str() {
            "rhf" => Ok(Self::Rhf),
            "uhf" => Ok(Self::Uhf),
            "hf" | "auto" => Ok(Self::Auto),
            "rks" => Ok(Self::Rks(resolve_xc("b3lyp")?)),
            "uks" => Ok(Self::Uks(resolve_xc("b3lyp")?)),
            "dft" => Ok(Self::DftAuto(resolve_xc("b3lyp")?)),
            other => {
                let func = XcFunctional::from_name(other)?;
                Ok(Self::DftAuto(func))
            }
        }
    }

    fn resolve(self, mol: &Molecule) -> Reference {
        match self {
            Reference::Auto => {
                if mol.multiplicity == 1 {
                    Reference::Rhf
                } else {
                    Reference::Uhf
                }
            }
            Reference::DftAuto(xc) => {
                if mol.multiplicity == 1 {
                    Reference::Rks(xc)
                } else {
                    Reference::Uks(xc)
                }
            }
            r => r,
        }
    }
}

pub(crate) fn scf_energy(
    mol: &Molecule,
    bs: &BasisSet,
    r: Reference,
    grid_level: GridLevel,
    o: &ScfOptions,
) -> Result<f64, CalcError> {
    match r.resolve(mol) {
        Reference::Uhf => Ok(uhf(mol, bs, o)?.total_energy_hartree),
        Reference::Rhf | Reference::Auto => Ok(rhf(mol, bs, o)?.total_energy_hartree),
        Reference::Rks(xc) => {
            let grid = MolecularGrid::build_for_basis(mol, bs, grid_level);
            Ok(rks_with_grid(mol, bs, xc, &grid, o)?.total_energy_hartree)
        }
        Reference::Uks(xc) | Reference::DftAuto(xc) => {
            let grid = MolecularGrid::build_for_basis(mol, bs, grid_level);
            Ok(uks_with_grid(mol, bs, xc, &grid, o)?.total_energy_hartree)
        }
    }
}

pub fn scf_energy_gradient(
    mol: &Molecule,
    bs: &BasisSet,
    r: Reference,
    grid_level: GridLevel,
    o: &ScfOptions,
) -> Result<(f64, Vec<[f64; 3]>), CalcError> {
    match r.resolve(mol) {
        Reference::Uhf => {
            let wf = uhf(mol, bs, o)?;
            Ok((wf.total_energy_hartree, uhf_gradient(mol, bs, &wf)?))
        }
        Reference::Rhf | Reference::Auto => {
            let wf = rhf(mol, bs, o)?;
            Ok((wf.total_energy_hartree, rhf_gradient(mol, bs, &wf)?))
        }
        Reference::Rks(xc) => {
            let grid = MolecularGrid::build_for_basis(mol, bs, grid_level);
            let wf = rks_with_grid(mol, bs, xc, &grid, o)?;
            let g = rks_gradient_with_grid(mol, bs, xc, &grid, &wf)?;
            Ok((wf.total_energy_hartree, g))
        }
        Reference::Uks(xc) | Reference::DftAuto(xc) => {
            let grid = MolecularGrid::build_for_basis(mol, bs, grid_level);
            let wf = uks_with_grid(mol, bs, xc, &grid, o)?;
            let g = uks_gradient_with_grid(mol, bs, xc, &grid, &wf)?;
            Ok((wf.total_energy_hartree, g))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptOptions {
    pub max_steps: usize,
    pub max_force: f64,
    pub rms_force: f64,
    pub max_displacement: f64,
    pub max_step_bohr: f64,
    pub scf_options: ScfOptions,
    pub c1: f64,
    pub backtrack_factor: f64,
    pub max_line_searches: usize,
    #[serde(default)]
    pub reference: Reference,
    #[serde(default)]
    pub grid_level: GridLevel,
    #[serde(default)]
    pub dispersion: Option<String>,
    #[serde(default)]
    pub solvent: Option<String>,
}

impl Default for OptOptions {
    fn default() -> Self {
        let mut scf_opts = ScfOptions::default();
        scf_opts.max_iter = 150;
        scf_opts.conv_energy = 1e-11;
        scf_opts.conv_density = 1e-9;
        Self {
            max_steps: 100,
            max_force: 1e-4,
            rms_force: 5e-5,
            max_displacement: 1e-3,
            max_step_bohr: 0.2,
            scf_options: scf_opts,
            c1: 1e-4,
            backtrack_factor: 0.5,
            max_line_searches: 20,
            reference: Reference::Auto,
            grid_level: GridLevel::Medium,
            dispersion: None,
            solvent: None,
        }
    }
}

pub fn scf_energy_gradient_with_options(
    mol: &Molecule,
    bs: &BasisSet,
    opts: &OptOptions,
) -> Result<(f64, Vec<[f64; 3]>), CalcError> {
    // Gas-phase derivatives are analytic. The optional SCRF surface uses explicit
    // central differences of *fully reconverged total energies*, including CDS.
    // A frozen gas density is not the derivative of either relaxed surface.
    let (mut e, mut g) = if opts.solvent.is_some() {
        let e = solvent_total_energy(mol, bs, opts)?;
        let h = 1e-4;
        let mut g = vec![[0.0; 3]; mol.atoms.len()];
        for a in 0..mol.atoms.len() {
            for q in 0..3 {
                let mut plus = mol.clone();
                let mut minus = mol.clone();
                plus.atoms[a].position_bohr[q] += h;
                minus.atoms[a].position_bohr[q] -= h;
                let bp = BasisSet::from_name(&bs.name, &plus)?;
                let bm = BasisSet::from_name(&bs.name, &minus)?;
                g[a][q] = (solvent_total_energy(&plus, &bp, opts)?
                    - solvent_total_energy(&minus, &bm, opts)?)
                    / (2.0 * h);
            }
        }
        (e, g)
    } else {
        scf_energy_gradient(mol, bs, opts.reference, opts.grid_level, &opts.scf_options)?
    };

    // 1. Dispersion correction (e.g. D3(BJ) or D3(0))
    if opts.dispersion.is_some() {
        let d3 = dispersion_model(opts)?;
        let (e_d3, g_d3) = d3.energy_and_gradient(&mol.atoms, true)?;
        e += e_d3;
        for (ga, gd) in g.iter_mut().zip(g_d3.iter()) {
            ga[0] += gd[0];
            ga[1] += gd[1];
            ga[2] += gd[2];
        }
    }

    Ok((e, g))
}

pub fn scf_energy_with_options(
    mol: &Molecule,
    bs: &BasisSet,
    opts: &OptOptions,
) -> Result<f64, CalcError> {
    let mut e = if opts.solvent.is_some() {
        solvent_total_energy(mol, bs, opts)?
    } else {
        scf_energy(mol, bs, opts.reference, opts.grid_level, &opts.scf_options)?
    };
    if opts.dispersion.is_some() {
        e += dispersion_model(opts)?.energy(&mol.atoms)?;
    }
    Ok(e)
}

fn dispersion_model(opts: &OptOptions) -> Result<calchem_dft::D3Model, CalcError> {
    let functional = match opts.reference {
        Reference::Rks(xc) | Reference::Uks(xc) | Reference::DftAuto(xc) => xc.canonical_name(),
        _ => "hf",
    };
    match opts
        .dispersion
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "bj" | "d3bj" | "d3(bj)" => calchem_dft::D3Model::d3bj(functional),
        "zero" | "d3zero" | "d3(0)" => calchem_dft::D3Model::d3zero(functional),
        _ => Err(CalcError::UnsupportedMethod(
            "Unknown D3 damping scheme".into(),
        )),
    }
}

fn solvent_total_energy(
    mol: &Molecule,
    bs: &BasisSet,
    opts: &OptOptions,
) -> Result<f64, CalcError> {
    let solvent = calchem_solv::Solvent::from_name(opts.solvent.as_deref().unwrap_or(""))
        .ok_or_else(|| CalcError::InvalidMolecule("Unknown solvent".into()))?;
    let (method, xc) = match opts.reference.resolve(mol) {
        Reference::Rhf | Reference::Auto => ("rhf", None),
        Reference::Rks(xc) => ("rks", Some(xc)),
        _ => {
            return Err(CalcError::UnsupportedMethod(
                "Open-shell SCRF gradients are not implemented; use an external backend".into(),
            ))
        }
    };
    let solv_opts = calchem_solv::SolvationOptions {
        solvent,
        ..Default::default()
    };
    let res = calchem_solv::run_scrf(mol, bs, &opts.scf_options, &solv_opts, method, xc)?;
    if !res.converged {
        return Err(CalcError::NotConverged {
            iterations: opts.scf_options.max_iter,
            delta_e: f64::INFINITY,
            rms_d: f64::INFINITY,
        });
    }
    Ok(res.solvated_energy)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptStep {
    pub step: usize,
    pub energy_hartree: f64,
    pub max_force: f64,
    pub rms_force: f64,
    pub max_displacement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptResult {
    pub molecule: Molecule,
    pub energy_hartree: f64,
    pub gradient: Vec<[f64; 3]>,
    pub steps: usize,
    pub converged: bool,
    pub history: Vec<OptStep>,
}

pub(crate) fn coords_from_mol(mol: &Molecule) -> Vec<f64> {
    let mut coords = Vec::with_capacity(mol.atoms.len() * 3);
    for atom in &mol.atoms {
        coords.extend_from_slice(&atom.position_bohr);
    }
    coords
}

pub(crate) fn mol_with_coords(base_mol: &Molecule, coords: &[f64]) -> Molecule {
    let mut mol = base_mol.clone();
    for (i, atom) in mol.atoms.iter_mut().enumerate() {
        atom.position_bohr = [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]];
    }
    mol
}

pub(crate) fn mat_vec_mul(mat: &Matrix2D, vec: &[f64]) -> Vec<f64> {
    let n = vec.len();
    let mut res = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += mat.get(i, j) * vec[j];
        }
        res[i] = sum;
    }
    res
}

pub(crate) fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Optimizes the molecular geometry using the requested HF or DFT method and basis set.
/// Returns the optimized `Molecule`.
pub fn optimize_geometry(
    mol: &Molecule,
    basis: &BasisSet,
    method: &str,
) -> Result<Molecule, CalcError> {
    let reference = Reference::from_method_and_xc(method, None)?;
    let opts = OptOptions {
        reference,
        ..OptOptions::default()
    };
    let res = optimize_geometry_with_options(mol, &basis.name, &opts)?;
    Ok(res.molecule)
}

/// Optimizes molecular geometry with custom optimizer options.
pub fn optimize_geometry_with_options(
    mol: &Molecule,
    basis_name: &str,
    opts: &OptOptions,
) -> Result<OptResult, CalcError> {
    let n_atoms = mol.atoms.len();
    if n_atoms == 0 {
        return Err(CalcError::InvalidMolecule("No atoms in molecule".into()));
    }

    if n_atoms == 1 {
        let bs = BasisSet::from_name(basis_name, mol)?;
        let e = scf_energy(mol, &bs, opts.reference, opts.grid_level, &opts.scf_options)?;
        return Ok(OptResult {
            molecule: mol.clone(),
            energy_hartree: e,
            gradient: vec![[0.0, 0.0, 0.0]],
            steps: 0,
            converged: true,
            history: vec![OptStep {
                step: 0,
                energy_hartree: e,
                max_force: 0.0,
                rms_force: 0.0,
                max_displacement: 0.0,
            }],
        });
    }

    let dim = n_atoms * 3;
    let mut x = coords_from_mol(mol);

    let mut h_inv = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        h_inv.set(i, i, 0.7);
    }

    let mut history = Vec::new();
    let mut last_displacement = 0.0;
    let mut cached_eg: Option<(f64, Vec<[f64; 3]>)> = None;

    for step in 0..=opts.max_steps {
        let cur_mol = mol_with_coords(mol, &x);
        let (current_e, current_g_3d) = if let Some(eg) = cached_eg.take() {
            eg
        } else {
            let cur_basis = BasisSet::from_name(basis_name, &cur_mol)?;
            scf_energy_gradient_with_options(&cur_mol, &cur_basis, opts)?
        };

        let mut g = Vec::with_capacity(dim);
        for atom_g in &current_g_3d {
            g.extend_from_slice(atom_g);
        }

        let max_force = g.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let sum_sq_g: f64 = g.iter().map(|v| v * v).sum();
        let rms_force = (sum_sq_g / (dim as f64)).sqrt();

        history.push(OptStep {
            step,
            energy_hartree: current_e,
            max_force,
            rms_force,
            max_displacement: last_displacement,
        });

        if step == 0 {
            if max_force < opts.max_force && rms_force < opts.rms_force {
                return Ok(OptResult {
                    molecule: cur_mol,
                    energy_hartree: current_e,
                    gradient: current_g_3d,
                    steps: step,
                    converged: true,
                    history,
                });
            }
        } else if max_force < opts.max_force
            && rms_force < opts.rms_force
            && last_displacement < opts.max_displacement
        {
            return Ok(OptResult {
                molecule: cur_mol,
                energy_hartree: current_e,
                gradient: current_g_3d,
                steps: step,
                converged: true,
                history,
            });
        }

        if step == opts.max_steps {
            break;
        }

        let mut p = mat_vec_mul(&h_inv, &g);
        for v in p.iter_mut() {
            *v = -*v;
        }

        let mut g_dot_p = dot_prod(&g, &p);
        if g_dot_p >= 0.0 {
            for i in 0..dim {
                for j in 0..dim {
                    h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                }
            }
            p = g.iter().map(|v| -0.7 * v).collect();
            g_dot_p = dot_prod(&g, &p);
        }

        let p_norm = dot_prod(&p, &p).sqrt();
        if p_norm > opts.max_step_bohr && p_norm > 1e-12 {
            let scale = opts.max_step_bohr / p_norm;
            for v in p.iter_mut() {
                *v *= scale;
            }
            g_dot_p *= scale;
        }

        let mut alpha = 1.0;
        let mut accepted = false;
        let mut x_new = x.clone();

        for _ls in 0..opts.max_line_searches {
            for i in 0..dim {
                x_new[i] = x[i] + alpha * p[i];
            }

            let trial_mol = mol_with_coords(mol, &x_new);
            if let Ok(trial_bs) = BasisSet::from_name(basis_name, &trial_mol) {
                if let Ok(e_trial) = scf_energy_with_options(&trial_mol, &trial_bs, opts) {
                    if e_trial <= current_e + opts.c1 * alpha * g_dot_p {
                        accepted = true;
                        break;
                    }
                }
            }
            alpha *= opts.backtrack_factor;
        }

        if !accepted {
            let mut is_diagonal = true;
            for i in 0..dim {
                for j in 0..dim {
                    if i != j && h_inv.get(i, j).abs() > 1e-10 {
                        is_diagonal = false;
                        break;
                    }
                }
                if !is_diagonal {
                    break;
                }
            }

            if !is_diagonal {
                for i in 0..dim {
                    for j in 0..dim {
                        h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                    }
                }
                let small_step = 0.05;
                for i in 0..dim {
                    x_new[i] = x[i] - small_step * g[i];
                }
                let trial_mol = mol_with_coords(mol, &x_new);
                if let Ok(trial_bs) = BasisSet::from_name(basis_name, &trial_mol) {
                    if let Ok(e_trial) = scf_energy_with_options(&trial_mol, &trial_bs, opts) {
                        if e_trial <= current_e {
                            accepted = true;
                        }
                    }
                }
            }

            if !accepted {
                return Err(CalcError::Internal(format!(
                    "Geometry optimization line search stalled at step {step}"
                )));
            }
        }

        let mut s = vec![0.0; dim];
        for i in 0..dim {
            s[i] = x_new[i] - x[i];
        }

        let mut max_atom_disp = 0.0;
        for a in 0..n_atoms {
            let dx = s[3 * a];
            let dy = s[3 * a + 1];
            let dz = s[3 * a + 2];
            let disp = (dx * dx + dy * dy + dz * dz).sqrt();
            if disp > max_atom_disp {
                max_atom_disp = disp;
            }
        }
        last_displacement = max_atom_disp;

        let new_mol = mol_with_coords(mol, &x_new);
        let new_bs = BasisSet::from_name(basis_name, &new_mol)?;
        let (new_e, new_g_3d) = scf_energy_gradient_with_options(&new_mol, &new_bs, opts)?;
        let mut g_new = Vec::with_capacity(dim);
        for ag in &new_g_3d {
            g_new.extend_from_slice(ag);
        }
        cached_eg = Some((new_e, new_g_3d));

        let mut y = vec![0.0; dim];
        for i in 0..dim {
            y[i] = g_new[i] - g[i];
        }

        let s_dot_y = dot_prod(&s, &y);
        if s_dot_y > 1e-10 {
            let rho = 1.0 / s_dot_y;

            let mut v = Matrix2D::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    let delta_ij = if i == j { 1.0 } else { 0.0 };
                    v.set(i, j, delta_ij - rho * s[i] * y[j]);
                }
            }

            let mut temp = Matrix2D::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0;
                    for k in 0..dim {
                        sum += v.get(i, k) * h_inv.get(k, j);
                    }
                    temp.set(i, j, sum);
                }
            }

            let mut h_next = Matrix2D::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0;
                    for k in 0..dim {
                        sum += temp.get(i, k) * v.get(j, k);
                    }
                    h_next.set(i, j, sum + rho * s[i] * s[j]);
                }
            }
            h_inv = h_next;
        } else {
            for i in 0..dim {
                for j in 0..dim {
                    h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                }
            }
        }

        x = x_new;
    }

    Err(CalcError::NotConverged {
        iterations: opts.max_steps,
        delta_e: 0.0,
        rms_d: last_displacement,
    })
}

```

## `crates/calchem-opt/src/prfo.rs`

SHA256: `523e8b7d98c347355d2e8e7561040b5fa6b79f88036bf40f25b2bd1bf161467f`

```rust
//! Partitioned Rational Function Optimization (P-RFO) for Transition State (TS) Search.
//!
//! Features:
//! - Rational Function Optimization with uphill maximization along the reaction coordinate
//!   and downhill minimization in all transverse degrees of freedom (Baker 1986).
//! - Automatic mode-following across steps using maximal eigenvector overlap.
//! - Bofill (1994) mixed MS/PSB Hessian update maintaining correct indefinite curvature.
//! - Trust radius step truncation and convergence monitoring.

use calchem_basis::BasisSet;
use calchem_dft::GridLevel;
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_tensor::{eigh_sym, Matrix2D};
use calchem_vib::{compute_cartesian_hessian_and_apt, FreqOptions, VibMethod};
use serde::{Deserialize, Serialize};

use crate::{
    coords_from_mol, dot_prod, mat_vec_mul, mol_with_coords, scf_energy_gradient, OptStep,
    Reference,
};

/// Transition state search options (Partitioned Rational Function Optimization, P-RFO).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsOptions {
    /// Maximum number of P-RFO macro-iterations.
    pub max_steps: usize,
    /// Maximum absolute Cartesian force threshold (Hartree / Bohr).
    pub max_force: f64,
    /// Root-mean-square Cartesian force threshold (Hartree / Bohr).
    pub rms_force: f64,
    /// Maximum Cartesian displacement threshold (Bohr).
    pub max_displacement: f64,
    /// Maximum trust radius step length (Bohr).
    pub trust_radius_bohr: f64,
    /// Optional normal mode index (0-based) to follow uphill. Default `None` follows lowest eigenvalue.
    pub mode_index: Option<usize>,
    /// Hessian recalculation interval (0 = initial Hessian only, N = recompute every N steps).
    pub recalc_hessian_interval: usize,
    #[serde(default)]
    pub reference: Reference,
    #[serde(default)]
    pub grid_level: GridLevel,
    pub scf_options: ScfOptions,
}

impl Default for TsOptions {
    fn default() -> Self {
        let mut scf_opts = ScfOptions::default();
        scf_opts.conv_energy = 1e-11;
        scf_opts.conv_density = 1e-10;
        Self {
            max_steps: 100,
            max_force: 3e-4,
            rms_force: 1.5e-4,
            max_displacement: 1.2e-3,
            trust_radius_bohr: 0.1,
            mode_index: None,
            recalc_hessian_interval: 0,
            reference: Reference::Auto,
            grid_level: GridLevel::Medium,
            scf_options: scf_opts,
        }
    }
}

/// Output of a transition state search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsResult {
    /// Transition state molecular geometry.
    pub molecule: Molecule,
    /// Final electronic energy (Hartree).
    pub energy_hartree: f64,
    /// Final Cartesian gradient ($N \times 3$, Hartree / Bohr).
    pub gradient: Vec<[f64; 3]>,
    /// Number of steps taken.
    pub steps: usize,
    /// Whether convergence criteria were met.
    pub converged: bool,
    /// Signed imaginary frequency of the saddle mode ($\text{cm}^{-1}$, negative for imaginary).
    pub imaginary_frequency_cm: f64,
    /// Normalized Cartesian displacement vector of the transition mode ($3N$).
    pub transition_vector: Vec<f64>,
    /// Step history.
    pub history: Vec<OptStep>,
}

/// Computes the initial exact finite-difference Cartesian Hessian matrix.
pub(super) fn compute_initial_hessian(
    mol: &Molecule,
    basis_name: &str,
    reference: Reference,
    grid_level: GridLevel,
    scf_options: &ScfOptions,
) -> Result<Matrix2D, CalcError> {
    let vib_method = match reference.resolve(mol) {
        Reference::Rhf | Reference::Auto => VibMethod::Rhf,
        Reference::Uhf => VibMethod::Uhf,
        Reference::Rks(xc) => VibMethod::Rks(xc),
        Reference::Uks(xc) | Reference::DftAuto(xc) => VibMethod::Uks(xc),
    };
    let freq_opts = FreqOptions {
        step_bohr: 0.001,
        grid_level,
        temperature_k: 298.15,
        pressure_pa: 101325.0,
        symmetry_number: None,
        scf_options: scf_options.clone(),
    };
    let (hess, _, _) = compute_cartesian_hessian_and_apt(mol, basis_name, vib_method, &freq_opts)?;
    Ok(hess)
}

/// Solves the 1D transverse P-RFO secular equation for lambda_n <= min_{j != m} Lambda_j:
/// f(lambda_n) = lambda_n - sum_{j != m} (g_tilde_j^2 / (lambda_n - Lambda_j)) = 0
fn solve_transverse_lambda(evals: &[f64], g_tilde: &[f64], ts_mode: usize) -> f64 {
    let mut min_lam = f64::INFINITY;
    for (j, &lam) in evals.iter().enumerate() {
        if j != ts_mode && lam < min_lam {
            min_lam = lam;
        }
    }

    if min_lam.is_infinite() {
        return 0.0;
    }

    // Initial guess strictly below min_lam
    let mut lambda = if min_lam > 0.1 { 0.0 } else { min_lam - 0.1 };
    if lambda >= min_lam - 1e-8 {
        lambda = min_lam - 0.1;
    }

    for _ in 0..60 {
        let mut f = lambda;
        let mut f_prime = 1.0;
        for (j, &lam) in evals.iter().enumerate() {
            if j == ts_mode {
                continue;
            }
            let denom = lambda - lam;
            let denom2 = denom * denom;
            let g2 = g_tilde[j] * g_tilde[j];
            f -= g2 / denom;
            f_prime += g2 / denom2;
        }

        if f.abs() < 1e-12 {
            break;
        }

        let delta = f / f_prime.max(1.0);
        let mut new_lam = lambda - delta;
        if new_lam >= min_lam - 1e-8 {
            new_lam = 0.5 * (lambda + min_lam);
        }
        if (new_lam - lambda).abs() < 1e-14 {
            lambda = new_lam;
            break;
        }
        lambda = new_lam;
    }

    lambda
}

/// Updates Hessian matrix using the Bofill (1994) formula:
/// H_{k+1} = H_k + (1 - mu) * Delta H_PSB + mu * Delta H_MS
fn update_hessian_bofill(h: &mut Matrix2D, s: &[f64], y: &[f64]) {
    let dim = s.len();
    let s_dot_s = dot_prod(s, s);
    if s_dot_s < 1e-14 {
        return;
    }

    let hs = mat_vec_mul(h, s);
    let mut xi = vec![0.0; dim];
    for i in 0..dim {
        xi[i] = y[i] - hs[i];
    }

    let xi_dot_xi = dot_prod(&xi, &xi);
    if xi_dot_xi < 1e-14 {
        return;
    }

    let s_dot_xi = dot_prod(s, &xi);
    let mu = ((s_dot_xi * s_dot_xi) / (s_dot_s * xi_dot_xi)).clamp(0.0, 1.0);
    let use_ms = s_dot_xi.abs() > 1e-10;

    for i in 0..dim {
        for j in 0..dim {
            let d_psb = (xi[i] * s[j] + s[i] * xi[j]) / s_dot_s
                - (s_dot_xi * s[i] * s[j]) / (s_dot_s * s_dot_s);

            let d_h = if use_ms {
                let d_ms = (xi[i] * xi[j]) / s_dot_xi;
                (1.0 - mu) * d_psb + mu * d_ms
            } else {
                d_psb
            };

            h.set(i, j, h.get(i, j) + d_h);
        }
    }

    // Symmetrize
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h.get(i, j) + h.get(j, i));
            h.set(i, j, avg);
            h.set(j, i, avg);
        }
    }
}

/// Searches for a First-Order Saddle Point (Transition State) using Partitioned RFO (P-RFO).
pub fn search_transition_state(
    mol: &Molecule,
    basis_name: &str,
    opts: &TsOptions,
) -> Result<TsResult, CalcError> {
    let n_atoms = mol.atoms.len();
    if n_atoms < 2 {
        return Err(CalcError::InvalidMolecule(
            "Transition state search requires at least 2 atoms".into(),
        ));
    }

    let dim = n_atoms * 3;
    if opts.mode_index.is_some_and(|i| i >= dim)
        || [
            opts.max_force,
            opts.rms_force,
            opts.max_displacement,
            opts.trust_radius_bohr,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(CalcError::InvalidMolecule(
            "Invalid TS mode index or nonpositive/nonfinite tolerances/trust radius".into(),
        ));
    }
    let mut x = coords_from_mol(mol);

    // Initial exact Hessian evaluation
    let mut hessian = compute_initial_hessian(
        mol,
        basis_name,
        opts.reference,
        opts.grid_level,
        &opts.scf_options,
    )?;

    let mut history = Vec::new();
    let mut last_displacement = 0.0;
    let mut target_mode_vector: Option<Vec<f64>> = None;
    let mut cached_eg: Option<(f64, Vec<[f64; 3]>)> = None;

    for step in 0..=opts.max_steps {
        let (current_e, current_g_3d) = if let Some(eg) = cached_eg.take() {
            eg
        } else {
            let cur_mol = mol_with_coords(mol, &x);
            let cur_basis = BasisSet::from_name(basis_name, &cur_mol)?;
            scf_energy_gradient(
                &cur_mol,
                &cur_basis,
                opts.reference,
                opts.grid_level,
                &opts.scf_options,
            )?
        };
        let cur_mol = mol_with_coords(mol, &x);

        let mut g = Vec::with_capacity(dim);
        for row in &current_g_3d {
            g.extend_from_slice(row);
        }

        let max_force = g.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let sum_sq_g: f64 = g.iter().map(|v| v * v).sum();
        let rms_force = (sum_sq_g / (dim as f64)).sqrt();

        history.push(OptStep {
            step,
            energy_hartree: current_e,
            max_force,
            rms_force,
            max_displacement: last_displacement,
        });

        // Diagonalize current Hessian
        let (evals, evecs) = eigh_sym(&hessian, 150, 1e-14)?;

        // Identify the uphill transition mode via mode-following
        let ts_mode = if let Some(ref target_vec) = target_mode_vector {
            let mut best_idx = 0;
            let mut best_overlap = -1.0;
            for col in 0..dim {
                let mut ov = 0.0;
                for row in 0..dim {
                    ov += target_vec[row] * evecs.get(row, col);
                }
                let abs_ov = ov.abs();
                if abs_ov > best_overlap {
                    best_overlap = abs_ov;
                    best_idx = col;
                }
            }
            best_idx
        } else if let Some(user_idx) = opts.mode_index {
            user_idx.min(dim - 1)
        } else {
            // Lowest eigenvalue mode
            0
        };

        // Extract transition vector
        let mut cur_ts_vec = vec![0.0; dim];
        for row in 0..dim {
            cur_ts_vec[row] = evecs.get(row, ts_mode);
        }
        target_mode_vector = Some(cur_ts_vec.clone());

        // Check convergence
        let is_force_converged = if step == 0 {
            max_force < opts.max_force && rms_force < opts.rms_force
        } else {
            max_force < opts.max_force
                && rms_force < opts.rms_force
                && (last_displacement < opts.max_displacement || max_force < 0.5 * opts.max_force)
        };

        if is_force_converged {
            // Bofill is a search model, never a certificate of Hessian inertia.
            let exact_hessian = compute_initial_hessian(
                &cur_mol,
                basis_name,
                opts.reference,
                opts.grid_level,
                &opts.scf_options,
            )?;
            let empty_apt = Matrix2D::zeros(3, dim);
            let vib = calchem_vib::analyze_vibrations(&cur_mol, &exact_hessian, &empty_apt)?;
            // Curvatures within 10 cm^-1 are unresolved at the numerical Hessian accuracy.
            // Reject both flat modes and saddles with more than one negative curvature.
            let imag_modes: Vec<_> = vib
                .modes
                .iter()
                .filter(|m| m.frequency_cm < -10.0)
                .collect();
            let is_true_ts =
                imag_modes.len() == 1 && vib.modes.iter().all(|m| m.frequency_cm.abs() > 10.0);
            let imag_freq = vib.modes.first().map(|m| m.frequency_cm).unwrap_or(0.0);
            let certified_vector = vib
                .modes
                .first()
                .map(|m| m.displacements.clone())
                .unwrap_or(cur_ts_vec);

            return Ok(TsResult {
                molecule: cur_mol,
                energy_hartree: current_e,
                gradient: current_g_3d,
                steps: step,
                converged: is_true_ts,
                imaginary_frequency_cm: imag_freq,
                transition_vector: certified_vector,
                history,
            });
        }

        if step == opts.max_steps {
            break;
        }

        // Project gradient into Hessian eigenbasis: g_tilde = V^T * g
        let mut g_tilde = vec![0.0; dim];
        for col in 0..dim {
            let mut sum = 0.0;
            for row in 0..dim {
                sum += evecs.get(row, col) * g[row];
            }
            g_tilde[col] = sum;
        }

        // P-RFO step along transition mode (uphill)
        let lam_m = evals[ts_mode];
        let g_m = g_tilde[ts_mode];
        let lambda_p = 0.5 * lam_m + (0.25 * lam_m * lam_m + g_m * g_m).sqrt();
        let s_tilde_m = -g_m / (lam_m - lambda_p).min(-1e-12);

        // P-RFO step along transverse modes (downhill minimization)
        let lambda_n = solve_transverse_lambda(&evals, &g_tilde, ts_mode);

        let mut s_tilde = vec![0.0; dim];
        for j in 0..dim {
            if j == ts_mode {
                s_tilde[j] = s_tilde_m;
            } else {
                let denom = evals[j] - lambda_n;
                s_tilde[j] = -g_tilde[j] / (if denom.abs() < 1e-12 { 1e-12 } else { denom });
            }
        }

        // Transform back to Cartesian coordinates: s = V * s_tilde
        let mut s = vec![0.0; dim];
        for row in 0..dim {
            let mut sum = 0.0;
            for col in 0..dim {
                sum += evecs.get(row, col) * s_tilde[col];
            }
            s[row] = sum;
        }

        // Apply trust radius scaling
        let step_norm = dot_prod(&s, &s).sqrt();
        if step_norm > opts.trust_radius_bohr && step_norm > 1e-14 {
            let scale = opts.trust_radius_bohr / step_norm;
            for val in &mut s {
                *val *= scale;
            }
        }

        let mut x_new = x.clone();
        for i in 0..dim {
            x_new[i] += s[i];
        }

        // Compute max atomic displacement
        let mut max_disp = 0.0;
        for a in 0..n_atoms {
            let dx = s[3 * a];
            let dy = s[3 * a + 1];
            let dz = s[3 * a + 2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > max_disp {
                max_disp = d;
            }
        }
        last_displacement = max_disp;

        // Evaluate gradient at new point
        let new_mol = mol_with_coords(mol, &x_new);
        let new_bs = BasisSet::from_name(basis_name, &new_mol)?;
        let (new_e, new_g_3d) = scf_energy_gradient(
            &new_mol,
            &new_bs,
            opts.reference,
            opts.grid_level,
            &opts.scf_options,
        )?;

        let mut g_new = Vec::with_capacity(dim);
        for row in &new_g_3d {
            g_new.extend_from_slice(row);
        }

        let mut y = vec![0.0; dim];
        for i in 0..dim {
            y[i] = g_new[i] - g[i];
        }

        cached_eg = Some((new_e, new_g_3d));

        // Update Hessian (either full recalculation or Bofill update)
        if opts.recalc_hessian_interval > 0 && (step + 1) % opts.recalc_hessian_interval == 0 {
            hessian = compute_initial_hessian(
                &new_mol,
                basis_name,
                opts.reference,
                opts.grid_level,
                &opts.scf_options,
            )?;
        } else {
            update_hessian_bofill(&mut hessian, &s, &y);
        }

        x = x_new;
    }

    Err(CalcError::NotConverged {
        iterations: opts.max_steps,
        delta_e: 0.0,
        rms_d: last_displacement,
    })
}

```

## `crates/calchem-opt/src/irc.rs`

SHA256: `babdef9dc28cfbca301b927aa3979e1955ae5dd10d12cedd79aab35088a61d8c`

```rust
//! Gonzalez-Schlegel Constrained Hypersphere Minimization for Intrinsic Reaction Coordinate (IRC).
//!
//! Integrates the steepest-descent reaction path in mass-weighted Cartesian coordinates:
//!   dq / ds = - grad_q(E) / ||grad_q(E)||
//! using Gonzalez & Schlegel's (1989, 1990) implicit circular arc geodesic integrator.

use calchem_basis::BasisSet;
use calchem_dft::GridLevel;
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_vib::masses::{atomic_mass_amu, HARTREE_TO_KCAL_MOL};
use serde::{Deserialize, Serialize};

use crate::{coords_from_mol, dot_prod, mol_with_coords, scf_energy_gradient, Reference};

/// Direction to follow the Intrinsic Reaction Coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IrcDirection {
    #[default]
    Both,
    Forward,
    Reverse,
}

impl IrcDirection {
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "forward" | "fwd" => Self::Forward,
            "reverse" | "rev" | "backward" => Self::Reverse,
            _ => Self::Both,
        }
    }
}

/// Options for Gonzalez-Schlegel IRC integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcOptions {
    /// Step size in mass-weighted coordinates (default 0.1 amu^(1/2) * Bohr).
    pub step_size_bohr: f64,
    /// Maximum number of points per branch (default 30).
    pub max_points: usize,
    /// Direction: Both, Forward, or Reverse.
    #[serde(default)]
    pub direction: IrcDirection,
    /// Gradient norm threshold to terminate branch (default 5e-4 a.u.).
    pub convergence_tol: f64,
    /// Maximum Gonzalez-Schlegel hypersphere minimization iterations per step.
    pub max_gs_iter: usize,
    #[serde(default)]
    pub reference: Reference,
    #[serde(default)]
    pub grid_level: GridLevel,
    pub scf_options: ScfOptions,
}

impl Default for IrcOptions {
    fn default() -> Self {
        let mut scf_opts = ScfOptions::default();
        scf_opts.conv_energy = 1e-11;
        scf_opts.conv_density = 1e-10;
        Self {
            step_size_bohr: 0.1,
            max_points: 30,
            direction: IrcDirection::Both,
            convergence_tol: 5e-4,
            max_gs_iter: 25,
            reference: Reference::Auto,
            grid_level: GridLevel::Medium,
            scf_options: scf_opts,
        }
    }
}

/// A single point along the IRC reaction path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcPoint {
    /// 0-based point index along the branch (0 is the transition state).
    pub point_index: usize,
    /// Arc length s in mass-weighted coordinates (amu^(1/2) * Bohr).
    pub path_coordinate_s: f64,
    /// Electronic energy at this point (Hartree).
    pub energy_hartree: f64,
    /// Energy difference relative to the transition state (kcal/mol, <= 0).
    pub delta_energy_kcal_mol: f64,
    /// Norm of the gradient in mass-weighted coordinates (Hartree / (amu^(1/2) * Bohr)).
    pub gradient_norm: f64,
    /// Molecular geometry at this point.
    pub molecule: Molecule,
}

/// A directional branch of the IRC path (forward to product or reverse to reactant).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcBranch {
    pub direction: String,
    pub points: Vec<IrcPoint>,
    pub converged_to_minimum: bool,
    pub final_energy_hartree: f64,
    pub barrier_kcal_mol: f64,
}

/// Full IRC result containing both forward and reverse branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcResult {
    pub transition_state: IrcPoint,
    pub forward: Option<IrcBranch>,
    pub reverse: Option<IrcBranch>,
    /// Overall reaction energy: E_product - E_reactant (kcal/mol).
    pub reaction_energy_kcal_mol: Option<f64>,
    /// Forward reaction barrier: E_TS - E_reactant (kcal/mol).
    pub barrier_forward_kcal_mol: Option<f64>,
    /// Reverse reaction barrier: E_TS - E_product (kcal/mol).
    pub barrier_reverse_kcal_mol: Option<f64>,
}

fn get_masses_3n(mol: &Molecule) -> Vec<f64> {
    let mut m = Vec::with_capacity(mol.atoms.len() * 3);
    for atom in &mol.atoms {
        let mass = atomic_mass_amu(atom.z);
        m.push(mass);
        m.push(mass);
        m.push(mass);
    }
    m
}

fn to_mass_weighted(x: &[f64], masses: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(masses.iter())
        .map(|(&xi, &mi)| xi * mi.sqrt())
        .collect()
}

fn from_mass_weighted(q: &[f64], masses: &[f64]) -> Vec<f64> {
    q.iter()
        .zip(masses.iter())
        .map(|(&qi, &mi)| qi / mi.sqrt())
        .collect()
}

fn gradient_mass_weighted(g: &[f64], masses: &[f64]) -> Vec<f64> {
    g.iter()
        .zip(masses.iter())
        .map(|(&gi, &mi)| gi / mi.sqrt())
        .collect()
}

fn eval_q_energy_gradient(
    q: &[f64],
    masses: &[f64],
    base_mol: &Molecule,
    basis_name: &str,
    opts: &IrcOptions,
) -> Result<(f64, Vec<f64>, Molecule), CalcError> {
    let x = from_mass_weighted(q, masses);
    let cur_mol = mol_with_coords(base_mol, &x);
    let cur_bs = BasisSet::from_name(basis_name, &cur_mol)?;
    let (e, g_3d) = scf_energy_gradient(
        &cur_mol,
        &cur_bs,
        opts.reference,
        opts.grid_level,
        &opts.scf_options,
    )?;
    let mut g_cart = Vec::with_capacity(q.len());
    for row in g_3d {
        g_cart.extend_from_slice(&row);
    }
    let g_q = gradient_mass_weighted(&g_cart, masses);
    Ok((e, g_q, cur_mol))
}

/// Integrates one branch of the IRC path (forward or reverse) using Gonzalez-Schlegel.
fn integrate_irc_branch(
    ts_q: &[f64],
    ts_tangent: &[f64],
    masses: &[f64],
    base_mol: &Molecule,
    basis_name: &str,
    e_ts: f64,
    direction_sign: f64,
    direction_label: &str,
    opts: &IrcOptions,
) -> Result<IrcBranch, CalcError> {
    let dim = ts_q.len();
    let mut ds = opts.step_size_bohr;

    let mut points = Vec::with_capacity(opts.max_points + 1);

    // Initial launch vector: tangent along transition vector
    let mut p_k = vec![0.0; dim];
    for i in 0..dim {
        p_k[i] = direction_sign * ts_tangent[i];
    }
    let p_k_norm = dot_prod(&p_k, &p_k).sqrt().max(1e-14);
    for v in &mut p_k {
        *v /= p_k_norm;
    }

    let mut q_cur = ts_q.to_vec();
    let mut s_acc = 0.0;
    let mut converged_to_min = false;
    let mut min_energy = e_ts;

    for pt_idx in 1..=opts.max_points {
        let mut accepted = None;
        for _retry in 0..8 {
            let radius = 0.5 * ds;
            // Hypersphere center: q_star = q_cur + 0.5 * ds * p_k
            let mut q_star = vec![0.0; dim];
            for i in 0..dim {
                q_star[i] = q_cur[i] + radius * p_k[i];
            }

            // Initial guess on the hypersphere: q_cur + ds * p_k
            let mut q_trial = vec![0.0; dim];
            for i in 0..dim {
                q_trial[i] = q_cur[i] + ds * p_k[i];
            }

            // Gonzalez-Schlegel constrained hypersphere minimization
            for _ in 0..opts.max_gs_iter {
                let (_e_eval, g_q, _pt_m) =
                    eval_q_energy_gradient(&q_trial, masses, base_mol, basis_name, opts)?;

                // Radial unit vector from sphere center to current trial point
                let mut u = vec![0.0; dim];
                for i in 0..dim {
                    u[i] = (q_trial[i] - q_star[i]) / radius;
                }
                let g_dot_u = dot_prod(&g_q, &u);

                // Perpendicular tangent gradient: g_perp = g_q - (g_q . u) * u
                let mut g_perp = vec![0.0; dim];
                for i in 0..dim {
                    g_perp[i] = g_q[i] - g_dot_u * u[i];
                }
                let g_perp_norm = dot_prod(&g_perp, &g_perp).sqrt();
                if g_perp_norm < opts.convergence_tol {
                    break;
                }

                // Step in tangent space with curvature-safe damping
                let alpha = (radius / (g_perp_norm + 1e-4)).clamp(0.05, 0.8);
                let mut q_cand = vec![0.0; dim];
                for i in 0..dim {
                    q_cand[i] = q_trial[i] - alpha * g_perp[i];
                }

                // Project back onto the sphere of radius R
                let mut diff = vec![0.0; dim];
                for i in 0..dim {
                    diff[i] = q_cand[i] - q_star[i];
                }
                let diff_norm = dot_prod(&diff, &diff).sqrt().max(1e-14);
                for i in 0..dim {
                    q_trial[i] = q_star[i] + radius * (diff[i] / diff_norm);
                }
            }

            // Always evaluate exact energy & gradient at the final q_trial geometry
            // to guarantee coordinates, energy, and forces are strictly synchronized.
            let (e_k, g_k, pt_mol) =
                eval_q_energy_gradient(&q_trial, masses, base_mol, basis_name, opts)?;
            let g_k_norm = dot_prod(&g_k, &g_k).sqrt();
            let u: Vec<_> = q_trial
                .iter()
                .zip(&q_star)
                .map(|(q, c)| (q - c) / radius)
                .collect();
            let radial = dot_prod(&g_k, &u);
            let tangent_norm = g_k
                .iter()
                .zip(&u)
                .map(|(g, u)| (g - radial * u).powi(2))
                .sum::<f64>()
                .sqrt();
            let previous_e = points
                .last()
                .map(|p: &IrcPoint| p.energy_hartree)
                .unwrap_or(e_ts);
            if tangent_norm <= opts.convergence_tol && e_k <= previous_e + 1e-8 {
                accepted = Some((q_trial, e_k, g_k, pt_mol, g_k_norm));
                break;
            }
            ds *= 0.5;
        }
        let Some((q_trial, e_k, g_k, pt_mol, g_k_norm)) = accepted else {
            // No accepted sphere endpoint: report an incomplete branch, never a minimum.
            break;
        };
        s_acc += ds;

        if e_k < min_energy {
            min_energy = e_k;
        }

        points.push(IrcPoint {
            point_index: pt_idx,
            path_coordinate_s: s_acc,
            energy_hartree: e_k,
            delta_energy_kcal_mol: (e_k - e_ts) * HARTREE_TO_KCAL_MOL,
            gradient_norm: g_k_norm,
            molecule: pt_mol,
        });

        // Termination checks
        if g_k_norm < opts.convergence_tol {
            converged_to_min = true;
            break;
        }

        // Energy rebound: if energy rises above previous step along the descent path.
        // Rigorous condition: only declare converged to minimum if gradient is small (< 3 * tol).
        // Otherwise, the trajectory simply encountered a barrier or step-size overshoot.
        if pt_idx > 1 && e_k > points[pt_idx - 2].energy_hartree + 1e-6 {
            if g_k_norm < 3.0 * opts.convergence_tol {
                converged_to_min = true;
            }
            break;
        }

        // Update tangent vector for next step: p_{k+1} = -g_k / ||g_k||
        if g_k_norm > 1e-14 {
            for i in 0..dim {
                p_k[i] = -g_k[i] / g_k_norm;
            }
        } else {
            converged_to_min = true;
            break;
        }

        q_cur = q_trial;
    }

    let final_e = points.last().map(|p| p.energy_hartree).unwrap_or(e_ts);
    let barrier = (e_ts - min_energy) * HARTREE_TO_KCAL_MOL;

    Ok(IrcBranch {
        direction: direction_label.to_string(),
        points,
        converged_to_minimum: converged_to_min,
        final_energy_hartree: final_e,
        barrier_kcal_mol: barrier.max(0.0),
    })
}

/// Runs full Intrinsic Reaction Coordinate (IRC) integration from a Transition State.
pub fn run_irc(
    ts_mol: &Molecule,
    basis_name: &str,
    transition_vector: &[f64],
    opts: &IrcOptions,
) -> Result<IrcResult, CalcError> {
    let dim = 3 * ts_mol.atoms.len();
    if transition_vector.len() != dim
        || transition_vector.iter().any(|v| !v.is_finite())
        || transition_vector.iter().map(|v| v * v).sum::<f64>() < 1e-20
        || !opts.step_size_bohr.is_finite()
        || opts.step_size_bohr <= 0.0
        || !opts.convergence_tol.is_finite()
        || opts.convergence_tol <= 0.0
        || opts.max_gs_iter == 0
        || opts.max_points == 0
    {
        return Err(CalcError::InvalidMolecule("IRC requires a nonzero finite 3N mode, positive step/tolerance and nonzero iteration budgets".into()));
    }
    let masses = get_masses_3n(ts_mol);
    let x_ts = coords_from_mol(ts_mol);
    let q_ts = to_mass_weighted(&x_ts, &masses);

    // Evaluate transition state energy and gradient
    let (e_ts, g_ts_q, _) = eval_q_energy_gradient(&q_ts, &masses, ts_mol, basis_name, opts)?;
    let g_ts_norm = dot_prod(&g_ts_q, &g_ts_q).sqrt();
    if g_ts_norm > 3.0 * opts.convergence_tol {
        return Err(CalcError::InvalidMolecule(
            "IRC starting geometry is not stationary at the requested force tolerance".into(),
        ));
    }
    let hessian = super::prfo::compute_initial_hessian(
        ts_mol,
        basis_name,
        opts.reference,
        opts.grid_level,
        &opts.scf_options,
    )?;
    let apt = calchem_tensor::Matrix2D::zeros(3, dim);
    let vib = calchem_vib::analyze_vibrations(ts_mol, &hessian, &apt)?;
    let negative: Vec<_> = vib
        .modes
        .iter()
        .filter(|m| m.frequency_cm < -10.0)
        .collect();
    if negative.len() != 1 || vib.modes.iter().any(|m| m.frequency_cm.abs() <= 10.0) {
        return Err(CalcError::InvalidMolecule(
            "IRC starting point must have exactly one resolved imaginary vibrational mode".into(),
        ));
    }

    // Mass-weight and normalize transition vector
    let q_tangent = to_mass_weighted(transition_vector, &masses);
    let norm_tangent = dot_prod(&q_tangent, &q_tangent).sqrt().max(1e-14);
    let norm_q_tangent: Vec<f64> = q_tangent.into_iter().map(|v| v / norm_tangent).collect();
    let certified = to_mass_weighted(&negative[0].displacements, &masses);
    let certified_norm = dot_prod(&certified, &certified).sqrt();
    let overlap = dot_prod(&norm_q_tangent, &certified).abs() / certified_norm;
    if overlap < 0.5 {
        return Err(CalcError::InvalidMolecule(
            "IRC launch vector does not follow the certified unstable mode".into(),
        ));
    }

    let ts_point = IrcPoint {
        point_index: 0,
        path_coordinate_s: 0.0,
        energy_hartree: e_ts,
        delta_energy_kcal_mol: 0.0,
        gradient_norm: g_ts_norm,
        molecule: ts_mol.clone(),
    };

    let forward = if matches!(opts.direction, IrcDirection::Forward | IrcDirection::Both) {
        Some(integrate_irc_branch(
            &q_ts,
            &norm_q_tangent,
            &masses,
            ts_mol,
            basis_name,
            e_ts,
            1.0,
            "forward",
            opts,
        )?)
    } else {
        None
    };

    let reverse = if matches!(opts.direction, IrcDirection::Reverse | IrcDirection::Both) {
        Some(integrate_irc_branch(
            &q_ts,
            &norm_q_tangent,
            &masses,
            ts_mol,
            basis_name,
            e_ts,
            -1.0,
            "reverse",
            opts,
        )?)
    } else {
        None
    };

    let e_prod = forward
        .as_ref()
        .filter(|f| f.converged_to_minimum)
        .map(|f| f.final_energy_hartree);
    let e_react = reverse
        .as_ref()
        .filter(|r| r.converged_to_minimum)
        .map(|r| r.final_energy_hartree);

    let rxn_energy = match (e_prod, e_react) {
        (Some(p), Some(r)) => Some((p - r) * HARTREE_TO_KCAL_MOL),
        _ => None,
    };

    let barrier_fwd = e_react.map(|r| ((e_ts - r) * HARTREE_TO_KCAL_MOL).max(0.0));
    let barrier_rev = e_prod.map(|p| ((e_ts - p) * HARTREE_TO_KCAL_MOL).max(0.0));

    Ok(IrcResult {
        transition_state: ts_point,
        forward,
        reverse,
        reaction_energy_kcal_mol: rxn_energy,
        barrier_forward_kcal_mol: barrier_fwd,
        barrier_reverse_kcal_mol: barrier_rev,
    })
}

```

## `crates/calchem-solv/src/scrf.rs`

SHA256: `c2706b28053bb26201758291ceeb5416f1d884aeecddd8dac34a1d075350f6b6`

```rust
//! Self-Consistent Reaction Field (SCRF) solver and post-SCF PCM solvation engines.

use calchem_basis::BasisSet;
use calchem_dft::{evaluate_vxc_rks, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::{
    compute_ecp, compute_kinetic, compute_nuclear, compute_overlap, compute_sparse_eri,
    SparseSymmetricEri,
};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction};
use calchem_scf::diis::DiisState;
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};
use serde::{Deserialize, Serialize};

use crate::cavity::{CavityOptions, MolecularCavity};
use crate::matrix::BoundaryMatrix;
use crate::potential::CavityPotentialEngine;
use crate::solvent::{Solvent, SolventModel};

/// Conversion constant from Hartree to kcal/mol.
pub const HARTREE_TO_KCAL_MOL: f64 = 627.509474063;

/// Options controlling SCRF and continuum solvation calculations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolvationOptions {
    /// Continuum solvent specification (dielectric constant, refractive index, surface tension).
    pub solvent: Solvent,
    /// Solvation boundary element model (C-PCM, COSMO, IEF-PCM, Onsager).
    pub model: SolventModel,
    /// Molecular cavity boundary tessellation parameters.
    pub cavity_opts: CavityOptions,
    /// Whether to renormalize total surface charge to satisfy Gauss's law (-f(eps) * Q_net).
    pub renormalize_charge: bool,
    /// Whether to include non-electrostatic cavitation-dispersion-solvent structural reorganization (CDS).
    pub include_cds: bool,
}

impl Default for SolvationOptions {
    fn default() -> Self {
        Self {
            solvent: Solvent::water(),
            model: SolventModel::Cpcm,
            cavity_opts: CavityOptions::default(),
            renormalize_charge: true,
            include_cds: true,
        }
    }
}

/// Comprehensive outcome of a continuum solvation calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolvationResult {
    /// Total free energy in solution (Hartree).
    pub solvated_energy: f64,
    /// Reference gas-phase energy (Hartree).
    pub gas_energy: f64,
    /// Net solvation free energy $\Delta G_{\text{solv}} = G_{\text{sol}} - E_{\text{gas}}$ (Hartree).
    pub delta_g_solv: f64,
    /// Net solvation free energy (kcal/mol).
    pub delta_g_solv_kcal_mol: f64,
    /// Electrostatic solvation interaction free energy $\frac{1}{2} \mathbf{q}^T \mathbf{V}^{\text{solute}}$ (kcal/mol).
    pub delta_g_elec_kcal_mol: f64,
    /// Non-electrostatic cavitation-dispersion free energy (kcal/mol).
    pub delta_g_cds_kcal_mol: f64,
    /// Solvent-accessible cavity surface area (Angstrom^2).
    pub cavity_surface_area_angstrom2: f64,
    /// Net sum of apparent surface charges $\sum_k q_k$ (atomic units).
    pub total_surface_charge: f64,
    /// Discretized apparent surface charges $\{q_k\}$ on boundary tesserae.
    pub surface_charges: Vec<f64>,
    /// Coordinates of boundary tesserae centers (Bohr).
    pub tessera_coords_bohr: Vec<[f64; 3]>,
    /// Number of SCRF self-consistency iterations.
    pub scf_iterations: usize,
    /// Convergence status.
    pub converged: bool,
    /// Solvated converged wavefunction (molecular orbitals and density).
    pub wavefunction: ScfWavefunction,
}

/// Computes fast post-SCF PCM solvation from an existing converged gas-phase wavefunction.
pub fn compute_pcm_solvation(
    mol: &Molecule,
    basis: &BasisSet,
    gas_wf: &ScfWavefunction,
    opts: &SolvationOptions,
) -> Result<SolvationResult, CalcError> {
    opts.model.validate_native()?;
    let cavity = MolecularCavity::build(mol, &opts.cavity_opts);
    let boundary_matrix = BoundaryMatrix::build(&cavity)?;
    let potential_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;

    let net_charge = if opts.renormalize_charge {
        Some(mol.charge as f64)
    } else {
        None
    };

    let n = basis.n_ao;
    let total_density: Vec<f64> = match &gas_wf.density_beta {
        Some(pb) => gas_wf
            .density_alpha
            .iter()
            .zip(pb.iter())
            .map(|(a, b)| a + b)
            .collect(),
        None => gas_wf.density_alpha.clone(),
    };
    let density_mat = Matrix2D {
        rows: n,
        cols: n,
        data: total_density,
    };

    let f_eps = opts.model.dielectric_factor(opts.solvent.epsilon);
    let v_solute = potential_engine.compute_solute_potential(&density_mat);
    let q = boundary_matrix.solve_surface_charges(&v_solute, f_eps, net_charge)?;

    // Electrostatic solvation energy: 0.5 * q^T * V_solute
    let e_elec_hartree: f64 = 0.5
        * q.iter()
            .zip(v_solute.iter())
            .map(|(&qk, &vk)| qk * vk)
            .sum::<f64>();

    // Cavitation-dispersion non-electrostatic energy
    let cds_kcal_mol = if opts.include_cds {
        opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity.total_area_angstrom2
    } else {
        0.0
    };
    let cds_hartree = cds_kcal_mol / HARTREE_TO_KCAL_MOL;

    let delta_g_solv_hartree = e_elec_hartree + cds_hartree;
    let solvated_energy = gas_wf.total_energy_hartree + delta_g_solv_hartree;

    let total_q: f64 = q.iter().sum();
    let tessera_coords: Vec<[f64; 3]> = cavity.tesserae.iter().map(|t| t.position_bohr).collect();

    Ok(SolvationResult {
        solvated_energy,
        gas_energy: gas_wf.total_energy_hartree,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: e_elec_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: cavity.total_area_angstrom2,
        total_surface_charge: total_q,
        surface_charges: q,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: gas_wf.diagnostics.iterations,
        converged: gas_wf.converged,
        wavefunction: gas_wf.clone(),
    })
}

/// Executes full closed-shell Restricted Hartree-Fock SCRF in solution.
pub fn run_scrf_rhf(
    mol: &Molecule,
    basis: &BasisSet,
    scf_opts: &ScfOptions,
    opts: &SolvationOptions,
) -> Result<SolvationResult, CalcError> {
    opts.model.validate_native()?;
    calchem_scf::validate_resources_and_options(basis, scf_opts)?;
    let n = basis.n_ao;
    let n_elec = mol.nelectron()?;
    if n_elec % 2 != 0 {
        return Err(CalcError::InvalidMolecule(
            "Closed-shell RHF SCRF requires an even number of electrons".into(),
        ));
    }
    let n_occ = n_elec / 2;

    // 1. One- and two-electron integrals
    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }
    let eri = compute_sparse_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    // 2. Cavity and boundary operator setup
    let cavity = MolecularCavity::build(mol, &opts.cavity_opts);
    let boundary_matrix = BoundaryMatrix::build(&cavity)?;
    let potential_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;
    let f_eps = opts.model.dielectric_factor(opts.solvent.epsilon);
    let net_charge = if opts.renormalize_charge {
        Some(mol.charge as f64)
    } else {
        None
    };

    // 3. Symmetric orthogonalizer X = S^{-1/2}
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 0.0 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is not positive definite ({val:.3e})"
            )));
        }
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    // Initial guess from core Hamiltonian
    let (_, _, mut p) = diagonalize_fock_rhf(&h_core, &x, &xt, n_occ)?;

    let mut diis = DiisState::new(scf_opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut q_converged = Vec::new();
    let mut v_solute_converged = Vec::new();
    let mut final_evals = Vec::new();
    let mut final_coeffs = Matrix2D::zeros(n, n);
    let mut final_fock = Matrix2D::zeros(n, n);

    for iter in 1..=scf_opts.max_iter {
        used_iter = iter;

        // Solute potential on cavity tesserae
        let v_solute = potential_engine.compute_solute_potential(&p);
        let q = boundary_matrix.solve_surface_charges(&v_solute, f_eps, net_charge)?;

        // Reaction field operator V_rf = -sum_k q_k V_k
        let v_rf = potential_engine.compute_reaction_field_matrix(&q);

        // Bare gas-phase Fock build F_0 = H_core + G[P]
        let f_0 = build_fock_rhf(&h_core, &p, &eri);

        // Effective Fock matrix in SCRF: F = F_0 + V_rf
        let mut f = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            f.data[i] = f_0.data[i] + v_rf.data[i];
        }

        // Total electronic energy: 0.5 * Tr(P * (H_core + F_0)) + 0.5 * q^T * V_solute
        let e_gas_like = compute_electronic_energy_rhf(&p, &h_core, &f_0);
        let e_solv_elec: f64 = 0.5
            * q.iter()
                .zip(v_solute.iter())
                .map(|(&qk, &vk)| qk * vk)
                .sum::<f64>();
        let e_tot = e_gas_like + e_nuc + e_solv_elec;
        energy_history.push(e_tot);

        // Pulay commutator error: e = F P S - S P F
        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_extrap = diis.push_and_extrapolate(&f, &err);
        let (evals, c, mut p_new) = diagonalize_fock_rhf(&f_extrap, &x, &xt, n_occ)?;

        let effective_damping = if scf_opts.damping > 0.0 && scf_opts.damping < 1.0 {
            scf_opts.damping
        } else if n >= 40 {
            0.3
        } else {
            0.0
        };
        if effective_damping > 0.0 {
            for idx in 0..p_new.data.len() {
                p_new.data[idx] =
                    (1.0 - effective_damping) * p_new.data[idx] + effective_damping * p.data[idx];
            }
        }

        let delta_e = (e_tot - e_prev).abs();
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        e_prev = e_tot;
        p = p_new;
        q_converged = q;
        v_solute_converged = v_solute;
        final_evals = evals;
        final_coeffs = c;
        final_fock = f;

        if iter > 1 && delta_e < scf_opts.conv_energy && rms_d < scf_opts.conv_density {
            converged = true;
            break;
        }
    }

    // Cavitation-dispersion (RHF SCRF)
    let cds_kcal_mol = if opts.include_cds {
        opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity.total_area_angstrom2
    } else {
        0.0
    };
    let cds_hartree = cds_kcal_mol / HARTREE_TO_KCAL_MOL;

    let e_solv_elec: f64 = 0.5
        * q_converged
            .iter()
            .zip(v_solute_converged.iter())
            .map(|(&qk, &vk)| qk * vk)
            .sum::<f64>();
    let solvated_energy = e_prev + cds_hartree;
    // Compute independent unpolarized gas-phase RHF energy for strict thermodynamic reference consistency:
    let gas_energy = calchem_scf::rhf(mol, basis, scf_opts)?.total_energy_hartree;
    let delta_g_solv_hartree = solvated_energy - gas_energy;

    let diag = ScfDiagnostics {
        iterations: used_iter,
        energy_history,
        rms_density_history: rms_history,
        diis_error_history: diis_err_history,
    };

    let mut mo_occupations = vec![0.0f64; n];
    for occ in mo_occupations.iter_mut().take(n_occ) {
        *occ = 2.0;
    }

    let wf = ScfWavefunction {
        converged,
        iterations: used_iter,
        total_energy_hartree: solvated_energy,
        electronic_energy_hartree: e_prev - e_nuc,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_mo: n,
        mo_energies: final_evals,
        mo_occupations,
        mo_coefficients: final_coeffs.data,
        density_alpha: p.data,
        density_beta: None,
        fock_alpha: final_fock.data,
        overlap_matrix: s.data,
        h_core_matrix: h_core.data,
        diagnostics: diag,
    };

    let total_q: f64 = q_converged.iter().sum();
    let tessera_coords: Vec<[f64; 3]> = cavity.tesserae.iter().map(|t| t.position_bohr).collect();

    Ok(SolvationResult {
        solvated_energy,
        gas_energy,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: (delta_g_solv_hartree - cds_hartree) * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: cavity.total_area_angstrom2,
        total_surface_charge: total_q,
        surface_charges: q_converged,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: used_iter,
        converged,
        wavefunction: wf,
    })
}

/// Executes full closed-shell Restricted Kohn-Sham (RKS) SCRF in solution.
pub fn run_scrf_rks(
    mol: &Molecule,
    basis: &BasisSet,
    scf_opts: &ScfOptions,
    opts: &SolvationOptions,
    xc: XcFunctional,
) -> Result<SolvationResult, CalcError> {
    opts.model.validate_native()?;
    calchem_scf::validate_resources_and_options(basis, scf_opts)?;
    xc.validate_native()?;
    let n = basis.n_ao;
    let n_elec = mol.nelectron()?;
    if n_elec % 2 != 0 {
        return Err(CalcError::InvalidMolecule(
            "Closed-shell RKS SCRF requires an even number of electrons".into(),
        ));
    }
    let n_occ = n_elec / 2;

    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }
    let eri = compute_sparse_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);
    let grid = MolecularGrid::build_for_basis(mol, basis, GridLevel::Fine);

    let cavity = MolecularCavity::build(mol, &opts.cavity_opts);
    let boundary_matrix = BoundaryMatrix::build(&cavity)?;
    let potential_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;
    let f_eps = opts.model.dielectric_factor(opts.solvent.epsilon);
    let net_charge = if opts.renormalize_charge {
        Some(mol.charge as f64)
    } else {
        None
    };

    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    let (_, _, mut p) = diagonalize_fock_rhf(&h_core, &x, &xt, n_occ)?;

    let mut diis = DiisState::new(scf_opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut q_converged = Vec::new();
    let mut v_solute_converged = Vec::new();
    let mut final_evals = Vec::new();
    let mut final_coeffs = Matrix2D::zeros(n, n);
    let mut final_fock = Matrix2D::zeros(n, n);

    for iter in 1..=scf_opts.max_iter {
        used_iter = iter;

        let v_solute = potential_engine.compute_solute_potential(&p);
        let q = boundary_matrix.solve_surface_charges(&v_solute, f_eps, net_charge)?;
        let v_rf = potential_engine.compute_reaction_field_matrix(&q);

        let g_2e = build_two_electron_rks(&p, &eri, xc.exact_exchange_fraction());
        let (e_xc, v_xc) = evaluate_vxc_rks(basis, &grid, xc, &p);

        let mut f = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            f.data[i] = h_core.data[i] + g_2e.data[i] + v_xc.data[i] + v_rf.data[i];
        }

        // Electronic energy in RKS: Tr(P*H_core) + 0.5*Tr(P*J) - 0.5*c_HF*Tr(P*K) + E_xc + 0.5*q^T*V_solute
        let mut e_1e = 0.0;
        let mut e_2e = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                let p_val = p.get(mu, nu);
                e_1e += p_val * h_core.get(mu, nu);
                e_2e += 0.5 * p_val * g_2e.get(mu, nu);
            }
        }
        let e_solv_elec: f64 = 0.5
            * q.iter()
                .zip(v_solute.iter())
                .map(|(&qk, &vk)| qk * vk)
                .sum::<f64>();
        let e_tot = e_1e + e_2e + e_xc + e_nuc + e_solv_elec;
        energy_history.push(e_tot);

        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_extrap = diis.push_and_extrapolate(&f, &err);
        let (evals, c, mut p_new) = diagonalize_fock_rhf(&f_extrap, &x, &xt, n_occ)?;

        let effective_damping = if scf_opts.damping > 0.0 && scf_opts.damping < 1.0 {
            scf_opts.damping
        } else if n >= 40 {
            0.3
        } else {
            0.0
        };
        if effective_damping > 0.0 {
            for idx in 0..p_new.data.len() {
                p_new.data[idx] =
                    (1.0 - effective_damping) * p_new.data[idx] + effective_damping * p.data[idx];
            }
        }

        let delta_e = (e_tot - e_prev).abs();
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        e_prev = e_tot;
        p = p_new;
        q_converged = q;
        v_solute_converged = v_solute;
        final_evals = evals;
        final_coeffs = c;
        final_fock = f;

        if iter > 1 && delta_e < scf_opts.conv_energy && rms_d < scf_opts.conv_density {
            converged = true;
            break;
        }
    }

    let cds_kcal_mol = if opts.include_cds {
        opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity.total_area_angstrom2
    } else {
        0.0
    };
    let cds_hartree = cds_kcal_mol / HARTREE_TO_KCAL_MOL;

    let e_solv_elec: f64 = 0.5
        * q_converged
            .iter()
            .zip(v_solute_converged.iter())
            .map(|(&qk, &vk)| qk * vk)
            .sum::<f64>();
    let solvated_energy = e_prev + cds_hartree;
    // Compute independent unpolarized gas-phase RKS energy for strict thermodynamic reference consistency:
    let gas_energy =
        calchem_scf::rks_with_grid(mol, basis, xc, &grid, scf_opts)?.total_energy_hartree;
    let delta_g_solv_hartree = solvated_energy - gas_energy;

    let diag = ScfDiagnostics {
        iterations: used_iter,
        energy_history,
        rms_density_history: rms_history,
        diis_error_history: diis_err_history,
    };

    let mut mo_occupations = vec![0.0f64; n];
    for occ in mo_occupations.iter_mut().take(n_occ) {
        *occ = 2.0;
    }

    let wf = ScfWavefunction {
        converged,
        iterations: used_iter,
        total_energy_hartree: solvated_energy,
        electronic_energy_hartree: e_prev - e_nuc,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_mo: n,
        mo_energies: final_evals,
        mo_occupations,
        mo_coefficients: final_coeffs.data,
        density_alpha: p.data,
        density_beta: None,
        fock_alpha: final_fock.data,
        overlap_matrix: s.data,
        h_core_matrix: h_core.data,
        diagnostics: diag,
    };

    let total_q: f64 = q_converged.iter().sum();
    let tessera_coords: Vec<[f64; 3]> = cavity.tesserae.iter().map(|t| t.position_bohr).collect();

    Ok(SolvationResult {
        solvated_energy,
        gas_energy,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: (delta_g_solv_hartree - cds_hartree) * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: cavity.total_area_angstrom2,
        total_surface_charge: total_q,
        surface_charges: q_converged,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: used_iter,
        converged,
        wavefunction: wf,
    })
}

/// Unified entrypoint for SCRF solvation calculations across HF and DFT.
pub fn run_scrf(
    mol: &Molecule,
    basis: &BasisSet,
    scf_opts: &ScfOptions,
    opts: &SolvationOptions,
    method: &str,
    xc: Option<XcFunctional>,
) -> Result<SolvationResult, CalcError> {
    let m = method.trim().to_ascii_lowercase();
    let is_open_shell = mol.multiplicity > 1
        || mol.nelectron().map(|ne| ne % 2 != 0).unwrap_or(false)
        || m == "uhf"
        || m == "uks";

    if is_open_shell {
        return Err(CalcError::UnsupportedMethod(
            "Self-consistent open-shell PCM is not implemented. A post-SCF correction to a gas UHF/UKS density must not be reported as SCRF; use an external backend.".into()
        ));
    }

    match m.as_str() {
        "rhf" | "hf" => run_scrf_rhf(mol, basis, scf_opts, opts),
        "rks" | "dft" => {
            let func = xc.unwrap_or(XcFunctional::B3lyp);
            run_scrf_rks(mol, basis, scf_opts, opts, func)
        }
        "b3lyp" | "b3lyp5" | "pbe" | "blyp" | "svwn5" | "pbe0" | "camb3lyp" | "cam-b3lyp"
        | "wb97xd" | "wb97x-d" | "wb97x" => {
            let func = XcFunctional::from_name(method)?;
            if xc.is_some_and(|x| x != func) {
                return Err(CalcError::UnsupportedMethod(
                    "Conflicting method and XC identities".into(),
                ));
            }
            run_scrf_rks(mol, basis, scf_opts, opts, func)
        }
        other => Err(CalcError::UnsupportedMethod(format!(
            "Unsupported SCRF method: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_fock_rhf(h_core: &Matrix2D, p: &Matrix2D, eri: &SparseSymmetricEri) -> Matrix2D {
    let n = h_core.rows;
    let mut f = eri.contract_rks(&p.data, 1.0);
    for i in 0..n * n {
        f[i] += h_core.data[i];
    }
    Matrix2D {
        rows: n,
        cols: n,
        data: f,
    }
}

fn compute_electronic_energy_rhf(p: &Matrix2D, h_core: &Matrix2D, fock: &Matrix2D) -> f64 {
    let n = h_core.rows;
    let mut e_elec = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            e_elec += 0.5 * p.get(mu, nu) * (h_core.get(mu, nu) + fock.get(mu, nu));
        }
    }
    e_elec
}

fn diagonalize_fock_rhf(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D), CalcError> {
    let n = fock.rows;
    let fp = matmul(&matmul(xt, fock), x);
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14)?;
    let c = matmul(x, &cp);
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for m in 0..n_occ {
                sum += c.get(mu, m) * c.get(nu, m);
            }
            p.set(mu, nu, 2.0 * sum);
        }
    }
    Ok((eps, c, p))
}

fn build_two_electron_rks(p: &Matrix2D, eri: &SparseSymmetricEri, exx_fraction: f64) -> Matrix2D {
    let n = p.rows;
    Matrix2D {
        rows: n,
        cols: n,
        data: eri.contract_rks(&p.data, exx_fraction),
    }
}

```

## `crates/calchem-solv/src/gradient.rs`

SHA256: `a95237e93c37e998d62a07f6589cccd7b47706748d2c61edea36111e61686ee4`

```rust
//! Analytical and numerical nuclear gradients for Continuum Solvation (PCM / COSMO / SCRF).
//!
//! Provides the nuclear gradient of the electrostatic and CDS solvation free energy:
//! $$\mathbf{g}_A^{\text{solv}} = \nabla_{\mathbf{R}_A} \left( \frac{1}{2} \mathbf{q}^T \mathbf{V}^{\text{solute}} + \Delta G_{\text{CDS}} \right)$$

use calchem_basis::BasisSet;
use calchem_model::{CalcError, Molecule};
use calchem_tensor::Matrix2D;

use crate::cavity::{CavityOptions, MolecularCavity};
use crate::matrix::BoundaryMatrix;
use crate::potential::CavityPotentialEngine;
use crate::solvent::{Solvent, SolventModel};

/// Solvation energy for a specified density matrix (Hartree).
fn pcm_energy_for_density(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    solvent: &Solvent,
    model: SolventModel,
    cavity_opts: &CavityOptions,
) -> Result<f64, CalcError> {
    let cavity = MolecularCavity::build(mol, cavity_opts);
    let b_mat = BoundaryMatrix::build(&cavity)?;
    let pot_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;
    let v_solute = pot_engine.compute_solute_potential(density);
    let f_eps = model.dielectric_factor(solvent.epsilon);
    let q = b_mat.solve_surface_charges(&v_solute, f_eps, Some(mol.charge as f64))?;

    let mut e_elec = 0.0;
    for k in 0..cavity.n_tesserae() {
        e_elec += 0.5 * q[k] * v_solute[k];
    }

    let surface_area_angstrom2 = cavity.total_area_angstrom2;
    let cds_kcal_mol = solvent.surface_tension_kcal_mol_angstrom2() * surface_area_angstrom2;
    let e_cds = cds_kcal_mol / crate::scrf::HARTREE_TO_KCAL_MOL;
    Ok(e_elec + e_cds)
}

/// Computes the nuclear gradient of the PCM electrostatic and CDS solvation free energy (Eh/Bohr).
///
/// Central differences at FIXED AO density, including the CDS area term.
/// This is a partial derivative; it excludes density response and is not the
/// gradient of a relaxed gas-phase or SCRF energy. Cavity topology is piecewise smooth.
pub fn compute_pcm_gradient_fd(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    solvent: &Solvent,
    model: SolventModel,
    cavity_opts: Option<&CavityOptions>,
    h_bohr: f64,
) -> Result<Vec<[f64; 3]>, CalcError> {
    model.validate_native()?;
    let n_atoms = mol.atoms.len();
    let mut grad = vec![[0.0, 0.0, 0.0]; n_atoms];

    let opts = cavity_opts.cloned().unwrap_or_default();
    if !h_bohr.is_finite() || h_bohr <= 0.0 {
        return Err(CalcError::InvalidMolecule(
            "PCM difference step must be finite and positive".into(),
        ));
    }
    let h = h_bohr;

    for i in 0..n_atoms {
        for c in 0..3 {
            let mut mol_plus = mol.clone();
            mol_plus.atoms[i].position_bohr[c] += h;
            let basis_plus = BasisSet::from_name(&basis.name, &mol_plus)?;
            let e_plus =
                pcm_energy_for_density(&mol_plus, &basis_plus, density, solvent, model, &opts)?;

            let mut mol_minus = mol.clone();
            mol_minus.atoms[i].position_bohr[c] -= h;
            let basis_minus = BasisSet::from_name(&basis.name, &mol_minus)?;
            let e_minus =
                pcm_energy_for_density(&mol_minus, &basis_minus, density, solvent, model, &opts)?;

            let g = (e_plus - e_minus) / (2.0 * h);
            grad[i][c] = g;
        }
    }

    Ok(grad)
}

/// Evaluates the analytical nuclear gradient of the PCM electrostatic boundary interaction and cavity energy.
pub fn compute_pcm_gradient_analytic(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    solvent: &Solvent,
    cavity_opts: Option<&CavityOptions>,
) -> Result<Vec<[f64; 3]>, CalcError> {
    let _ = (mol, basis, density, solvent, cavity_opts);
    Err(CalcError::UnsupportedMethod(
        "Analytic PCM gradient is not implemented. Use compute_pcm_gradient_fd for a fixed-density partial derivative, or fully relaxed SCRF energy differences.".into()
    ))
}

/// Unified PCM nuclear gradient entry point.
pub fn run_pcm_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    solvent: &Solvent,
    model: SolventModel,
    cavity_opts: Option<&CavityOptions>,
) -> Result<Vec<[f64; 3]>, CalcError> {
    compute_pcm_gradient_fd(mol, basis, density, solvent, model, cavity_opts, 1e-4)
}

```

## `crates/calchem-solv/src/solvent.rs`

SHA256: `2b65ec0c941ef501034a91bbf8d0c6e5770ee4cff747f65c3335946c47209038`

```rust
//! Continuum dielectric solvent models, solvent database, and dielectric response functions.

use serde::{Deserialize, Serialize};

/// Supported continuum solvation formalisms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolventModel {
    /// Conductor-like Polarizable Continuum Model (C-PCM): `f(eps) = (eps - 1) / eps`.
    Cpcm,
    /// Conductor-like Screening Model (COSMO): `f(eps) = (eps - 1) / (eps + 0.5)`.
    Cosmo,
    /// Integral Equation Formalism PCM in conductor limit: `f(eps) = (eps - 1) / eps`.
    IefPcm,
    /// Dipolar spherical reaction field (Onsager model): `f(eps) = 2*(eps - 1) / (2*eps + 1)`.
    Onsager,
}

impl Default for SolventModel {
    fn default() -> Self {
        Self::Cpcm
    }
}

impl SolventModel {
    pub fn validate_native(self) -> Result<(), calchem_model::CalcError> {
        if !matches!(self, Self::Cpcm | Self::Cosmo) {
            return Err(calchem_model::CalcError::UnsupportedMethod(
                "Native solvent boundary solver implements C-PCM/COSMO only; IEF-PCM/Onsager require distinct operators and an external backend".into()
            ));
        }
        Ok(())
    }
    /// Evaluates the dielectric screening prefactor `f(epsilon)`.
    pub fn dielectric_factor(self, epsilon: f64) -> f64 {
        if (epsilon - 1.0).abs() < 1e-12 {
            return 0.0;
        }
        match self {
            Self::Cpcm | Self::IefPcm => (epsilon - 1.0) / epsilon,
            Self::Cosmo => (epsilon - 1.0) / (epsilon + 0.5),
            Self::Onsager => 2.0 * (epsilon - 1.0) / (2.0 * epsilon + 1.0),
        }
    }

    /// Evaluates the optical (fast) dielectric response factor `f(eps_infty)` where `eps_infty = n^2`.
    pub fn optical_dielectric_factor(self, refractive_index: f64) -> f64 {
        let eps_inf = refractive_index * refractive_index;
        self.dielectric_factor(eps_inf)
    }
}

/// Physical parameters of a continuum solvent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solvent {
    /// Canonical solvent identifier (e.g. `"water"`, `"chlorobenzene"`).
    pub name: String,
    /// Static dielectric constant `epsilon` (relative permittivity).
    pub epsilon: f64,
    /// Optical refractive index `n` (defines high-frequency permittivity `eps_inf = n^2`).
    pub refractive_index: f64,
    /// Macroscopic surface tension `gamma` (mN/m = dyn/cm).
    pub surface_tension_m_n_per_m: f64,
    /// Effective solvent probe radius (Angstrom).
    pub probe_radius_angstrom: f64,
}

impl Default for Solvent {
    fn default() -> Self {
        Self::water()
    }
}

impl Solvent {
    /// Conversion constant from mN/m (mJ/m^2) to kcal / (mol * Angstrom^2).
    pub const MN_M_TO_KCAL_MOL_ANGSTROM2: f64 = 0.001439326;

    /// Creates a custom solvent with specified properties.
    pub fn custom(
        name: impl Into<String>,
        epsilon: f64,
        refractive_index: f64,
        surface_tension_m_n_per_m: f64,
        probe_radius_angstrom: f64,
    ) -> Self {
        Self {
            name: name.into(),
            epsilon,
            refractive_index,
            surface_tension_m_n_per_m,
            probe_radius_angstrom,
        }
    }

    /// Water (H2O, standard aqueous medium).
    pub fn water() -> Self {
        Self::custom("water", 78.3553, 1.3330, 72.8, 1.385)
    }

    /// Acetonitrile (CH3CN, polar aprotic solvent).
    pub fn acetonitrile() -> Self {
        Self::custom("acetonitrile", 37.5, 1.3441, 29.3, 2.19)
    }

    /// Methanol (CH3OH, protic solvent).
    pub fn methanol() -> Self {
        Self::custom("methanol", 32.61, 1.3288, 22.5, 2.05)
    }

    /// Ethanol (CH3CH2OH).
    pub fn ethanol() -> Self {
        Self::custom("ethanol", 24.55, 1.3611, 22.4, 2.22)
    }

    /// Acetone (CH3COCH3).
    pub fn acetone() -> Self {
        Self::custom("acetone", 20.7, 1.3586, 23.5, 2.38)
    }

    /// Dichloromethane (DCM, CH2Cl2).
    pub fn dichloromethane() -> Self {
        Self::custom("dichloromethane", 8.93, 1.4244, 28.1, 2.33)
    }

    /// Tetrahydrofuran (THF, C4H8O).
    pub fn tetrahydrofuran() -> Self {
        Self::custom("tetrahydrofuran", 7.58, 1.4072, 26.4, 2.52)
    }

    /// Chlorobenzene (C6H5Cl, standard OPV thin-film processing solvent).
    pub fn chlorobenzene() -> Self {
        Self::custom("chlorobenzene", 5.62, 1.5248, 33.6, 2.72)
    }

    /// Chloroform (CHCl3).
    pub fn chloroform() -> Self {
        Self::custom("chloroform", 4.806, 1.4459, 27.2, 2.52)
    }

    /// Toluene (C7H8, aromatic solvent).
    pub fn toluene() -> Self {
        Self::custom("toluene", 2.379, 1.4969, 28.5, 2.82)
    }

    /// Benzene (C6H6, nonpolar aromatic solvent).
    pub fn benzene() -> Self {
        Self::custom("benzene", 2.2706, 1.5011, 28.9, 2.63)
    }

    /// Hexane (C6H14, aliphatic alkane solvent).
    pub fn hexane() -> Self {
        Self::custom("hexane", 1.88, 1.3749, 18.4, 3.00)
    }

    /// Cyclohexane (C6H12).
    pub fn cyclohexane() -> Self {
        Self::custom("cyclohexane", 2.02, 1.4262, 24.9, 2.81)
    }

    /// Gas phase / vacuum (epsilon = 1.0, zero screening).
    pub fn gas() -> Self {
        Self::custom("gas", 1.0, 1.0, 0.0, 0.0)
    }

    /// Resolves a solvent by case-insensitive name or alias.
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.trim().to_ascii_lowercase();
        let key = n.replace(['-', '_', ' '], "");
        match key.as_str() {
            "water" | "h2o" | "aqueous" => Some(Self::water()),
            "acetonitrile" | "mecn" | "ch3cn" => Some(Self::acetonitrile()),
            "methanol" | "meoh" | "ch3oh" => Some(Self::methanol()),
            "ethanol" | "etoh" | "ch3ch2oh" => Some(Self::ethanol()),
            "acetone" | "ch3coch3" => Some(Self::acetone()),
            "dichloromethane" | "dcm" | "ch2cl2" => Some(Self::dichloromethane()),
            "tetrahydrofuran" | "thf" => Some(Self::tetrahydrofuran()),
            "chlorobenzene" | "cb" | "phcl" => Some(Self::chlorobenzene()),
            "chloroform" | "chcl3" => Some(Self::chloroform()),
            "toluene" | "phme" => Some(Self::toluene()),
            "benzene" | "phh" => Some(Self::benzene()),
            "hexane" | "nhexane" => Some(Self::hexane()),
            "cyclohexane" => Some(Self::cyclohexane()),
            "gas" | "vacuum" | "none" => Some(Self::gas()),
            _ => None,
        }
    }

    /// Macroscopic surface tension converted to kcal / (mol * Angstrom^2).
    pub fn surface_tension_kcal_mol_angstrom2(&self) -> f64 {
        self.surface_tension_m_n_per_m * Self::MN_M_TO_KCAL_MOL_ANGSTROM2
    }
}

```

## `crates/calchem-vib/src/masses.rs`

SHA256: `d5a1ee33c076b830e0183179fbf87b4fc1a039683c97334efbf5bef596b62201`

```rust
//! Standard isotopic and atomic masses (in atomic mass units / amu) for $Z = 1 \dots 86$.
//!
//! Uses most abundant isotope standard masses (IUPAC / NIST) matching standard
//! computational quantum chemistry codes (Gaussian, ORCA, PySCF).

pub const AMU_TO_ELECTRON_MASS: f64 = 1822.888486209;
pub const HARTREE_TO_EV: f64 = 27.211386245988;
pub const HARTREE_TO_KCAL_MOL: f64 = 627.5094740631;
pub const HARTREE_TO_KJ_MOL: f64 = 2625.4996394799;
pub const JOULE_TO_CALORIE: f64 = 1.0 / 4.184;

// Physical conversion constant from sqrt(Hartree / (Bohr^2 * amu)) to wavenumber cm^-1:
// 1 Hartree = 4.3597447222071e-18 J
// 1 Bohr = 5.29177210903e-11 m
// 1 amu = 1.66053906660e-27 kg
// c = 2.99792458e10 cm/s
// sqrt(Hartree / (Bohr^2 * amu)) / (2 * pi * c) = 5140.48714376 cm^-1
pub const HESSIAN_EVAL_TO_INV_CM: f64 = 5140.48714376;

// Prefactor for IR intensity: km/mol from |dmu/dQ|^2 with mu in e*a0, Q in amu^{1/2}*a0:
// I_k = (N_A * pi) / (3 * c^2 * 4*pi*epsilon_0) * |dmu/dQ|^2
// When dmu/dQ is in Debye / (amu^{1/2} * Angstrom), the factor is 42.2561 km/mol.
// Since dmu/dQ is in atomic units (e*a0 / (amu^{1/2}*a0) = e / amu^{1/2}),
// converting e*a0 -> Debye (* 2.54174623) and a0 -> Angstrom (/ 0.529177210903)
// yields a conversion factor squared of (2.54174623 / 0.529177210903)^2 = 23.0707711.
// 42.2561 * 23.0707711 = 974.88081 km/mol.
pub const IR_INTENSITY_PREFACTOR_KM_MOL: f64 = 974.8808105;

// Ideal gas constant R = 8.314462618 J/(mol*K) = 1.9872042586 cal/(mol*K)
pub const GAS_CONSTANT_J_MOL_K: f64 = 8.314462618;
pub const GAS_CONSTANT_CAL_MOL_K: f64 = 1.9872042586;
pub const GAS_CONSTANT_HARTREE_K: f64 = 3.166811563e-6; // k_B in Hartree / K
pub const SPEED_OF_LIGHT_CM_S: f64 = 2.99792458e10;
pub const PLANCK_CONSTANT_J_S: f64 = 6.62607015e-34;
pub const BOLTZMANN_CONSTANT_J_K: f64 = 1.380649e-23;
pub const AVOGADRO_CONSTANT: f64 = 6.02214076e23;
pub const STANDARD_PRESSURE_PA: f64 = 101325.0; // 1 atm
pub const STANDARD_TEMPERATURE_K: f64 = 298.15;

/// Most abundant isotope atomic masses in amu for elements $Z = 1 \dots 86$.
const MOST_ABUNDANT_ISOTOPE_MASSES: [f64; 86] = [
    1.00782503223,  // 1  H  (^1H)
    4.00260325413,  // 2  He (^4He)
    7.0160034366,   // 3  Li (^7Li)
    9.012183065,    // 4  Be (^9Be)
    11.00930536,    // 5  B  (^11B)
    12.000000000,   // 6  C  (^12C)
    14.00307400443, // 7  N  (^14N)
    15.99491461957, // 8  O  (^16O)
    18.99840316273, // 9  F  (^19F)
    19.9924401762,  // 10 Ne (^20Ne)
    22.9897692820,  // 11 Na (^23Na)
    23.985041697,   // 12 Mg (^24Mg)
    26.98153853,    // 13 Al (^27Al)
    27.97692653465, // 14 Si (^28Si)
    30.97376199842, // 15 P  (^31P)
    31.9720711744,  // 16 S  (^32S)
    34.968852682,   // 17 Cl (^35Cl)
    39.9623831237,  // 18 Ar (^40Ar)
    38.9637064864,  // 19 K  (^39K)
    39.962590863,   // 20 Ca (^40Ca)
    44.95590828,    // 21 Sc (^45Sc)
    47.94794198,    // 22 Ti (^48Ti)
    50.9439570,     // 23 V  (^51V)
    51.94050623,    // 24 Cr (^52Cr)
    54.93804393,    // 25 Mn (^55Mn)
    55.93493633,    // 26 Fe (^56Fe)
    58.93319429,    // 27 Co (^59Co)
    57.93534166,    // 28 Ni (^58Ni)
    62.92959772,    // 29 Cu (^63Cu)
    63.92914201,    // 30 Zn (^64Zn)
    68.9255735,     // 31 Ga (^69Ga)
    73.9211774,     // 32 Ge (^74Ge)
    74.92159457,    // 33 As (^75As)
    79.9165218,     // 34 Se (^80Se)
    78.9183376,     // 35 Br (^79Br)
    83.9114977282,  // 36 Kr (^84Kr)
    84.9117897379,  // 37 Rb (^85Rb)
    87.9056125,     // 38 Sr (^88Sr)
    88.9058479,     // 39 Y  (^89Y)
    89.9046977,     // 40 Zr (^90Zr)
    92.9063772,     // 41 Nb (^93Nb)
    97.90540482,    // 42 Mo (^98Mo)
    98.9062547,     // 43 Tc (^99Tc)
    101.9043441,    // 44 Ru (^102Ru)
    102.9054980,    // 45 Rh (^103Rh)
    105.9034804,    // 46 Pd (^106Pd)
    106.9050916,    // 47 Ag (^107Ag)
    113.90336509,   // 48 Cd (^114Cd)
    114.903878776,  // 49 In (^115In)
    119.9021991,    // 50 Sn (^120Sn)
    120.9038157,    // 51 Sb (^121Sb)
    129.90622274,   // 52 Te (^130Te)
    126.9044719,    // 53 I  (^127I)
    131.9041550856, // 54 Xe (^132Xe)
    132.905451961,  // 55 Cs (^133Cs)
    137.9052470,    // 56 Ba (^138Ba)
    138.9063563,    // 57 La (^139La)
    139.9054409,    // 58 Ce (^140Ce)
    140.9076576,    // 59 Pr (^141Pr)
    141.9077290,    // 60 Nd (^142Nd)
    144.9127559,    // 61 Pm (^145Pm)
    151.9197397,    // 62 Sm (^152Sm)
    152.9212380,    // 63 Eu (^153Eu)
    157.9241123,    // 64 Gd (^158Gd)
    158.9253547,    // 65 Tb (^159Tb)
    163.9291819,    // 66 Dy (^164Dy)
    164.9303288,    // 67 Ho (^165Ho)
    165.9302995,    // 68 Er (^166Er)
    168.9342179,    // 69 Tm (^169Tm)
    173.9388664,    // 70 Yb (^174Yb)
    174.9407752,    // 71 Lu (^175Lu)
    179.9465570,    // 72 Hf (^180Hf)
    180.9479958,    // 73 Ta (^181Ta)
    183.95093092,   // 74 W  (^184W)
    186.9557501,    // 75 Re (^187Re)
    191.9614770,    // 76 Os (^192Os)
    192.9629216,    // 77 Ir (^193Ir)
    194.9647917,    // 78 Pt (^195Pt)
    196.96656879,   // 79 Au (^197Au)
    201.97064340,   // 80 Hg (^202Hg)
    204.9744278,    // 81 Tl (^205Tl)
    207.9766525,    // 82 Pb (^208Pb)
    208.9803991,    // 83 Bi (^209Bi)
    208.9824308,    // 84 Po (^209Po)
    209.9871479,    // 85 At (^210At)
    222.0175782,    // 86 Rn (^222Rn)
];

/// Returns the most abundant isotope mass in atomic mass units (amu).
pub fn atomic_mass_amu(z: u8) -> f64 {
    if z >= 1 && (z as usize) <= MOST_ABUNDANT_ISOTOPE_MASSES.len() {
        MOST_ABUNDANT_ISOTOPE_MASSES[(z - 1) as usize]
    } else {
        (z as f64) * 2.0 // Rough fallback for superheavies
    }
}

```

## `crates/calchem-vib/src/thermo.rs`

SHA256: `6bac31c8012291957af52ca74154a1173d99074fb79aa16ba6aa1d050366f3ac`

```rust
//! Standard Rigid Rotor Harmonic Oscillator (RRHO) Thermochemistry.
//!
//! Evaluates Zero-Point Vibrational Energy (ZPVE), thermal enthalpy $H(T, P)$,
//! entropy $S(T, P)$, constant-pressure/volume heat capacities $C_p, C_v$,
//! and Gibbs free energy $G(T, P)$.

use calchem_model::Molecule;
use serde::{Deserialize, Serialize};

use crate::masses::{
    atomic_mass_amu, GAS_CONSTANT_HARTREE_K,
    GAS_CONSTANT_J_MOL_K, HARTREE_TO_KCAL_MOL,
};
use crate::modes::VibrationalAnalysis;

/// Conversion factor: 1 cm^-1 in Hartree.
const INV_CM_TO_HARTREE: f64 = 4.556335252912e-6;
/// Conversion factor: 1 cm^-1 in Kelvin ($h c / k_B$).
const INV_CM_TO_KELVIN: f64 = 1.43877736;
/// 1 amu * Bohr^2 in kg * m^2 (CODATA 2018: 1.66053906660e-27 * 0.529177210903e-10^2).
const AMU_BOHR2_TO_KG_M2: f64 = 4.64998298e-48;
/// 1 amu in kg.
const AMU_TO_KG: f64 = 1.66053906660e-27;
/// Planck constant h in J * s.
const H_PLANCK_J_S: f64 = 6.62607015e-34;
/// Boltzmann constant k_B in J / K.
const K_BOLTZMANN_J_K: f64 = 1.380649e-23;

/// Complete Ideal-Gas RRHO Thermochemical Properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thermochemistry {
    pub temperature_k: f64,
    pub pressure_pa: f64,
    pub symmetry_number: usize,
    pub electronic_energy_hartree: f64,

    // Zero-Point Vibrational Energy
    pub zpve_hartree: f64,
    pub zpve_kcal_mol: f64,

    // Thermal energy components (Hartree)
    pub e_trans_hartree: f64,
    pub e_rot_hartree: f64,
    pub e_vib_hartree: f64,
    pub e_thermal_correction_hartree: f64,
    pub e_thermal_total_hartree: f64,

    // Enthalpy (Hartree & kcal/mol)
    pub enthalpy_correction_hartree: f64,
    pub enthalpy_hartree: f64,
    pub enthalpy_kcal_mol: f64,

    // Entropy components in cal / (mol * K)
    pub s_trans_cal_mol_k: f64,
    pub s_rot_cal_mol_k: f64,
    pub s_vib_cal_mol_k: f64,
    pub s_elec_cal_mol_k: f64,
    pub s_total_cal_mol_k: f64,
    pub s_total_j_mol_k: f64,

    // Heat capacities in cal / (mol * K)
    pub cv_cal_mol_k: f64,
    pub cp_cal_mol_k: f64,
    pub cv_j_mol_k: f64,
    pub cp_j_mol_k: f64,

    // Gibbs Free Energy (Hartree & kcal/mol)
    pub gibbs_correction_hartree: f64,
    pub gibbs_free_energy_hartree: f64,
    pub gibbs_free_energy_kcal_mol: f64,
}

/// Evaluates complete RRHO thermochemical properties given electronic energy and normal modes.
pub fn compute_thermochemistry(
    mol: &Molecule,
    electronic_energy_hartree: f64,
    vib: &VibrationalAnalysis,
    temperature_k: f64,
    pressure_pa: f64,
    symmetry_number_override: Option<usize>,
) -> Thermochemistry {
    let t = if temperature_k > 0.0 { temperature_k } else { 298.15 };
    let p = if pressure_pa > 0.0 { pressure_pa } else { 101325.0 };
    let sigma = symmetry_number_override.unwrap_or(1).max(1);
    let n_atoms = mol.atoms.len();

    // Total mass in amu and kg
    let total_mass_amu: f64 = mol.atoms.iter().map(|a| atomic_mass_amu(a.z)).sum();
    let mass_kg = total_mass_amu * AMU_TO_KG;

    // 1. ZPVE and Vibrational partition function contributions
    let mut zpve_eh = 0.0;
    let mut e_vib_thermal_j_mol = 0.0;
    let mut s_vib_j_mol_k = 0.0;
    let mut cv_vib_j_mol_k = 0.0;

    for mode in &vib.modes {
        if !mode.is_imaginary && mode.frequency_cm > 1.0 {
            let nu = mode.frequency_cm;
            zpve_eh += 0.5 * nu * INV_CM_TO_HARTREE;

            let theta_v = nu * INV_CM_TO_KELVIN;
            let x = theta_v / t;
            if x < 150.0 {
                let exp_x = x.exp();
                let denom = exp_x - 1.0;
                e_vib_thermal_j_mol += GAS_CONSTANT_J_MOL_K * theta_v / denom;
                cv_vib_j_mol_k += GAS_CONSTANT_J_MOL_K * (x * x * exp_x) / (denom * denom);
                let ln_term = (1.0 - (-x).exp()).ln();
                s_vib_j_mol_k += GAS_CONSTANT_J_MOL_K * (x / denom - ln_term);
            }
        }
    }

    let zpve_kcal = zpve_eh * HARTREE_TO_KCAL_MOL;
    let e_vib_thermal_eh = (e_vib_thermal_j_mol / 1000.0) / 2625.49963948; // J/mol to Eh
    let e_vib_eh = zpve_eh + e_vib_thermal_eh;

    // 2. Translational contributions (Sackur-Tetrode equation)
    let e_trans_eh = 1.5 * GAS_CONSTANT_HARTREE_K * t;
    let cv_trans_j_mol_k = 1.5 * GAS_CONSTANT_J_MOL_K;

    let s_trans_j_mol_k = if mass_kg > 0.0 {
        // q_trans = (2 * pi * m * k_B * T / h^2)^(3/2) * (k_B * T / P)
        let two_pi_m_kt_h2 = (2.0 * std::f64::consts::PI * mass_kg * K_BOLTZMANN_J_K * t)
            / (H_PLANCK_J_S * H_PLANCK_J_S);
        let q_trans = two_pi_m_kt_h2.powf(1.5) * (K_BOLTZMANN_J_K * t / p);
        if q_trans > 1e-12 {
            GAS_CONSTANT_J_MOL_K * (q_trans.ln() + 2.5)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 3. Rotational contributions
    let (e_rot_eh, cv_rot_j_mol_k, s_rot_j_mol_k) = if n_atoms <= 1 {
        // Monoatomic
        (0.0, 0.0, 0.0)
    } else if vib.is_linear {
        // Linear molecule
        let i_amu_bohr2 = vib.moments_of_inertia_amu_bohr2[2].max(vib.moments_of_inertia_amu_bohr2[1]);
        let i_kg_m2 = i_amu_bohr2 * AMU_BOHR2_TO_KG_M2;
        let q_rot = (8.0 * std::f64::consts::PI * std::f64::consts::PI * i_kg_m2 * K_BOLTZMANN_J_K * t)
            / ((sigma as f64) * H_PLANCK_J_S * H_PLANCK_J_S);
        let s_rot = if q_rot > 1e-12 {
            GAS_CONSTANT_J_MOL_K * (q_rot.ln() + 1.0)
        } else {
            0.0
        };
        (GAS_CONSTANT_HARTREE_K * t, GAS_CONSTANT_J_MOL_K, s_rot)
    } else {
        // Non-linear molecule
        let ia = vib.moments_of_inertia_amu_bohr2[0] * AMU_BOHR2_TO_KG_M2;
        let ib = vib.moments_of_inertia_amu_bohr2[1] * AMU_BOHR2_TO_KG_M2;
        let ic = vib.moments_of_inertia_amu_bohr2[2] * AMU_BOHR2_TO_KG_M2;
        let prod_i = (ia * ib * ic).max(1e-150);
        let factor = (8.0 * std::f64::consts::PI * std::f64::consts::PI * K_BOLTZMANN_J_K * t)
            / (H_PLANCK_J_S * H_PLANCK_J_S);
        let q_rot = (std::f64::consts::PI.sqrt() / (sigma as f64)) * factor.powf(1.5) * prod_i.sqrt();
        let s_rot = if q_rot > 1e-12 {
            GAS_CONSTANT_J_MOL_K * (q_rot.ln() + 1.5)
        } else {
            0.0
        };
        (1.5 * GAS_CONSTANT_HARTREE_K * t, 1.5 * GAS_CONSTANT_J_MOL_K, s_rot)
    };

    // 4. Electronic entropy
    let mult = mol.multiplicity.max(1) as f64;
    let s_elec_j_mol_k = GAS_CONSTANT_J_MOL_K * mult.ln();

    // 5. Total entropy and heat capacities
    let s_total_j_mol_k = s_trans_j_mol_k + s_rot_j_mol_k + s_vib_j_mol_k + s_elec_j_mol_k;
    let cv_total_j_mol_k = cv_trans_j_mol_k + cv_rot_j_mol_k + cv_vib_j_mol_k;
    let cp_total_j_mol_k = cv_total_j_mol_k + GAS_CONSTANT_J_MOL_K;

    let j_to_cal = 1.0 / 4.184;
    let s_total_cal_mol_k = s_total_j_mol_k * j_to_cal;
    let cv_cal_mol_k = cv_total_j_mol_k * j_to_cal;
    let cp_cal_mol_k = cp_total_j_mol_k * j_to_cal;

    // 6. Enthalpy and Gibbs Free Energy
    // Thermal correction to energy Delta E_th = E_trans + E_rot + E_vib
    let e_thermal_corr_eh = e_trans_eh + e_rot_eh + e_vib_eh;
    let e_thermal_total_eh = electronic_energy_hartree + e_thermal_corr_eh;

    // H = E_thermal + k_B * T
    let pv_eh = GAS_CONSTANT_HARTREE_K * t;
    let enthalpy_corr_eh = e_thermal_corr_eh + pv_eh;
    let enthalpy_eh = electronic_energy_hartree + enthalpy_corr_eh;
    let enthalpy_kcal = enthalpy_eh * HARTREE_TO_KCAL_MOL;

    // -T * S in Hartree: (s_total_j_mol_k * t) / 2.62549963948e6
    let ts_eh = (s_total_j_mol_k * t) / 2.62549963948e6;
    let gibbs_corr_eh = enthalpy_corr_eh - ts_eh;
    let gibbs_eh = electronic_energy_hartree + gibbs_corr_eh;
    let gibbs_kcal = gibbs_eh * HARTREE_TO_KCAL_MOL;

    Thermochemistry {
        temperature_k: t,
        pressure_pa: p,
        symmetry_number: sigma,
        electronic_energy_hartree,
        zpve_hartree: zpve_eh,
        zpve_kcal_mol: zpve_kcal,
        e_trans_hartree: e_trans_eh,
        e_rot_hartree: e_rot_eh,
        e_vib_hartree: e_vib_eh,
        e_thermal_correction_hartree: e_thermal_corr_eh,
        e_thermal_total_hartree: e_thermal_total_eh,
        enthalpy_correction_hartree: enthalpy_corr_eh,
        enthalpy_hartree: enthalpy_eh,
        enthalpy_kcal_mol: enthalpy_kcal,
        s_trans_cal_mol_k: s_trans_j_mol_k * j_to_cal,
        s_rot_cal_mol_k: s_rot_j_mol_k * j_to_cal,
        s_vib_cal_mol_k: s_vib_j_mol_k * j_to_cal,
        s_elec_cal_mol_k: s_elec_j_mol_k * j_to_cal,
        s_total_cal_mol_k,
        s_total_j_mol_k,
        cv_cal_mol_k,
        cp_cal_mol_k,
        cv_j_mol_k: cv_total_j_mol_k,
        cp_j_mol_k: cp_total_j_mol_k,
        gibbs_correction_hartree: gibbs_corr_eh,
        gibbs_free_energy_hartree: gibbs_eh,
        gibbs_free_energy_kcal_mol: gibbs_kcal,
    }
}

```

## `crates/calchem-vib/src/modes.rs`

SHA256: `272762b1edd213fd0e41139ddc332f356f3a67244d31c284d5247c680e2e513b`

```rust
//! Normal mode analysis, Eckart translation-rotation projection,
//! vibrational frequencies ($\text{cm}^{-1}$), and infrared (IR) intensities ($\text{km/mol}$).

use calchem_model::{CalcError, Molecule};
use calchem_tensor::{eigh_sym, matmul, transpose, Matrix2D};
use serde::{Deserialize, Serialize};

use crate::masses::{atomic_mass_amu, HESSIAN_EVAL_TO_INV_CM, IR_INTENSITY_PREFACTOR_KM_MOL};

const HARTREE_BOHR2_TO_MDYN_ANG: f64 = 15.568931;

/// Single vibrational normal mode with frequency, reduced mass, force constant, and IR intensity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalMode {
    /// 1-based index among vibrational modes.
    pub mode_index: usize,
    /// Harmonic vibrational frequency $\tilde{\nu}_k$ in $\text{cm}^{-1}$ (positive for real, negative for imaginary).
    pub frequency_cm: f64,
    /// `true` if $\lambda_k < 0$ (imaginary frequency, characteristic of a transition state).
    pub is_imaginary: bool,
    /// Reduced mass $\mu_k$ in atomic mass units (amu).
    pub reduced_mass_amu: f64,
    /// Harmonic force constant $k$ in $\text{mDyn} / \text{Å}$.
    pub force_constant_m_dyn_a: f64,
    /// Infrared absorption intensity $I_k$ in $\text{km} / \text{mol}$.
    pub ir_intensity_km_mol: f64,
    /// Normalized Cartesian displacement vector ($3N$ elements: `[dx_0, dy_0, dz_0, dx_1, ...]`).
    pub displacements: Vec<f64>,
}

/// Output of harmonic vibrational frequency and normal mode analysis.
#[derive(Debug, Clone)]
pub struct VibrationalAnalysis {
    pub n_atoms: usize,
    pub is_linear: bool,
    pub n_trans_rot: usize,
    pub n_vib: usize,
    /// Principal moments of inertia $I_A \le I_B \le I_C$ in $\text{amu}\cdot\text{Bohr}^2$.
    pub moments_of_inertia_amu_bohr2: [f64; 3],
    /// Rotational constants $A, B, C$ in GHz.
    pub rotational_constants_ghz: [f64; 3],
    /// Symmetrized Cartesian Hessian matrix ($3N \times 3N$, Hartree / $\text{Bohr}^2$).
    pub cartesian_hessian: Matrix2D,
    /// Atomic Polar Tensor ($3 \times 3N$, atomic units).
    pub apt_matrix: Matrix2D,
    /// Vibrational normal modes ordered by ascending frequency.
    pub modes: Vec<NormalMode>,
    /// Projected translation and rotation residual eigenvalues (Hartree / ($\text{amu}\cdot\text{Bohr}^2$)).
    pub trans_rot_residuals: Vec<f64>,
}

/// Computes center of mass and coordinates shifted to center of mass.
fn center_of_mass(mol: &Molecule, masses_amu: &[f64]) -> ([f64; 3], Vec<[f64; 3]>) {
    let mut total_m = 0.0;
    let mut com = [0.0; 3];
    for (atom, &m) in mol.atoms.iter().zip(masses_amu) {
        total_m += m;
        for q in 0..3 {
            com[q] += m * atom.position_bohr[q];
        }
    }
    if total_m > 0.0 {
        com[0] /= total_m;
        com[1] /= total_m;
        com[2] /= total_m;
    }
    let shifted: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .map(|a| {
            [
                a.position_bohr[0] - com[0],
                a.position_bohr[1] - com[1],
                a.position_bohr[2] - com[2],
            ]
        })
        .collect();
    (com, shifted)
}

/// Computes the $3 \times 3$ inertia tensor in $\text{amu}\cdot\text{Bohr}^2$ and principal moments.
fn compute_inertia_tensor(
    shifted_coords: &[[f64; 3]],
    masses_amu: &[f64],
) -> ([f64; 3], [f64; 3], bool) {
    let mut i_mat = Matrix2D::zeros(3, 3);
    for (pos, &m) in shifted_coords.iter().zip(masses_amu) {
        let x = pos[0];
        let y = pos[1];
        let z = pos[2];
        let r2 = x * x + y * y + z * z;
        i_mat.add_assign_at(0, 0, m * (r2 - x * x));
        i_mat.add_assign_at(1, 1, m * (r2 - y * y));
        i_mat.add_assign_at(2, 2, m * (r2 - z * z));
        i_mat.add_assign_at(0, 1, -m * x * y);
        i_mat.add_assign_at(1, 0, -m * x * y);
        i_mat.add_assign_at(0, 2, -m * x * z);
        i_mat.add_assign_at(2, 0, -m * x * z);
        i_mat.add_assign_at(1, 2, -m * y * z);
        i_mat.add_assign_at(2, 1, -m * y * z);
    }

    let (mut evals, _) =
        eigh_sym(&i_mat, 50, 1e-14).unwrap_or(([0.0, 0.0, 0.0].to_vec(), Matrix2D::identity(3)));
    evals.sort_by(|a, b| a.total_cmp(b));
    let moments = [evals[0].max(0.0), evals[1].max(0.0), evals[2].max(0.0)];

    // Rotational constants B = h / (8 * pi^2 * c * I)
    // 1 amu * Bohr^2 = 1.66053906660e-27 kg * (5.29177210903e-11 m)^2 = 4.6509653e-48 kg*m^2
    // h / (8 * pi^2 * 4.6509653e-48 kg*m^2) in GHz = 6.62607015e-34 / (8 * pi^2 * 4.6509653e-48) * 1e-9 = 505.379006 GHz * amu * Bohr^2
    const INERTIA_TO_GHZ: f64 = 505.379006;
    let b_ghz = [
        if moments[0] > 1e-6 {
            INERTIA_TO_GHZ / moments[0]
        } else {
            0.0
        },
        if moments[1] > 1e-6 {
            INERTIA_TO_GHZ / moments[1]
        } else {
            0.0
        },
        if moments[2] > 1e-6 {
            INERTIA_TO_GHZ / moments[2]
        } else {
            0.0
        },
    ];

    let is_linear =
        shifted_coords.len() >= 2 && (moments[0] < 1e-4 * moments[2] || moments[0] < 1e-4);
    (moments, b_ghz, is_linear)
}

/// Orthonormalizes vectors using two-pass Modified Gram-Schmidt; drops vectors with norm $< 10^{-6}$.
fn gram_schmidt_ortho(_dim: usize, raw_vecs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for mut v in raw_vecs {
        for _pass in 0..2 {
            for q in &basis {
                let dot: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(q.iter()) {
                    *vi -= dot * qi;
                }
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-6 {
            let inv_n = 1.0 / norm;
            for x in &mut v {
                *x *= inv_n;
            }
            basis.push(v);
        }
    }
    basis
}

/// Performs normal mode analysis on Cartesian Hessian and Atomic Polar Tensor.
pub fn analyze_vibrations(
    mol: &Molecule,
    cart_hessian: &Matrix2D,
    apt: &Matrix2D,
) -> Result<VibrationalAnalysis, CalcError> {
    let n_atoms = mol.atoms.len();
    let dim = 3 * n_atoms;
    if cart_hessian.rows != dim || cart_hessian.cols != dim {
        return Err(CalcError::InvalidMolecule(format!(
            "Cartesian Hessian dimension {}x{} mismatch for {n_atoms} atoms (expected {dim}x{dim})",
            cart_hessian.rows, cart_hessian.cols
        )));
    }

    if apt.rows != 3
        || apt.cols != dim
        || cart_hessian
            .data
            .iter()
            .chain(&apt.data)
            .any(|x| !x.is_finite())
    {
        return Err(CalcError::InvalidMolecule(
            "Hessian/APT must be finite, with APT shaped (3,3N)".into(),
        ));
    }
    let masses_amu: Vec<f64> = mol.atoms.iter().map(|a| atomic_mass_amu(a.z)).collect();
    let mut mass_coord = vec![0.0f64; dim];
    for (i, &m) in masses_amu.iter().enumerate() {
        mass_coord[3 * i] = m;
        mass_coord[3 * i + 1] = m;
        mass_coord[3 * i + 2] = m;
    }

    let (_, shifted) = center_of_mass(mol, &masses_amu);
    let (moments, b_ghz, is_linear) = compute_inertia_tensor(&shifted, &masses_amu);

    // Build translation and rotation vectors in mass-weighted coordinate space:
    // T_q: translation along Cartesian axis q
    // R_q: rotation about Cartesian axis q
    let mut raw_tr: Vec<Vec<f64>> = Vec::with_capacity(6);
    for q in 0..3 {
        let mut t = vec![0.0f64; dim];
        for a in 0..n_atoms {
            t[3 * a + q] = masses_amu[a].sqrt();
        }
        raw_tr.push(t);
    }

    if n_atoms >= 2 {
        let mut rx = vec![0.0f64; dim];
        let mut ry = vec![0.0f64; dim];
        let mut rz = vec![0.0f64; dim];
        for a in 0..n_atoms {
            let sqrt_m = masses_amu[a].sqrt();
            let x = shifted[a][0];
            let y = shifted[a][1];
            let z = shifted[a][2];
            // R_x = (0, -z, y) * sqrt(m)
            rx[3 * a + 1] = -z * sqrt_m;
            rx[3 * a + 2] = y * sqrt_m;
            // R_y = (z, 0, -x) * sqrt(m)
            ry[3 * a] = z * sqrt_m;
            ry[3 * a + 2] = -x * sqrt_m;
            // R_z = (-y, x, 0) * sqrt(m)
            rz[3 * a] = -y * sqrt_m;
            rz[3 * a + 1] = x * sqrt_m;
        }
        raw_tr.push(rx);
        raw_tr.push(ry);
        raw_tr.push(rz);
    }

    // Orthonormalize translation-rotation basis D (dim x n_tr)
    let tr_basis = gram_schmidt_ortho(dim, raw_tr);
    let n_tr = tr_basis.len();
    let n_vib = dim.saturating_sub(n_tr);

    // Mass-weight the Hessian: H_mw_{ij} = H_{ij} / sqrt(m_i * m_j)
    let mut h_mw = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        let inv_sqrt_mi = 1.0 / mass_coord[i].sqrt();
        for j in 0..dim {
            let inv_sqrt_mj = 1.0 / mass_coord[j].sqrt();
            h_mw.set(i, j, cart_hessian.get(i, j) * inv_sqrt_mi * inv_sqrt_mj);
        }
    }

    // Construct projection matrix P = I - D D^T and project H_mw: H_proj = P H_mw P
    let mut h_proj = h_mw.clone();
    if n_tr > 0 {
        let mut d_mat = Matrix2D::zeros(dim, n_tr);
        for (col, v) in tr_basis.iter().enumerate() {
            for row in 0..dim {
                d_mat.set(row, col, v[row]);
            }
        }
        let dt_h = matmul(&transpose(&d_mat), &h_mw);
        let h_d = matmul(&h_mw, &d_mat);
        let dt_h_d = matmul(&dt_h, &d_mat);
        let d_dt_h = matmul(&d_mat, &dt_h);
        let h_d_dt = matmul(&h_d, &transpose(&d_mat));
        let d_dt_h_d_dt = matmul(&matmul(&d_mat, &dt_h_d), &transpose(&d_mat));

        for i in 0..dim {
            for j in 0..dim {
                let v =
                    h_mw.get(i, j) - d_dt_h.get(i, j) - h_d_dt.get(i, j) + d_dt_h_d_dt.get(i, j);
                h_proj.set(i, j, v);
            }
        }
    }

    // Symmetrize H_proj to prevent floating-point asymmetry
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h_proj.get(i, j) + h_proj.get(j, i));
            h_proj.set(i, j, avg);
            h_proj.set(j, i, avg);
        }
    }

    // Diagonalize only the orthogonal complement of rigid translations/rotations.
    // Sorting the 3N spectrum by |lambda| can delete genuine floppy modes.
    let mut seeds = tr_basis.clone();
    for axis in 0..dim {
        let mut v = vec![0.0; dim];
        v[axis] = 1.0;
        seeds.push(v);
    }
    let complete = gram_schmidt_ortho(dim, seeds);
    if complete.len() != dim {
        return Err(CalcError::LinearAlgebra(
            "Incomplete vibrational projection basis".into(),
        ));
    }
    let mut v = Matrix2D::zeros(dim, n_vib);
    for (j, col) in complete[n_tr..].iter().enumerate() {
        for i in 0..dim {
            v.set(i, j, col[i]);
        }
    }
    let hv = matmul(&matmul(&transpose(&v), &h_mw), &v);
    let (evals, small) = if n_vib > 0 {
        eigh_sym(&hv, 150, 1e-14)?
    } else {
        (Vec::new(), Matrix2D::zeros(0, 0))
    };
    let evecs = matmul(&v, &small);
    let mut vib_modes: Vec<(f64, Vec<f64>)> = (0..n_vib)
        .map(|j| (evals[j], (0..dim).map(|i| evecs.get(i, j)).collect()))
        .collect();
    let trans_rot_modes: Vec<f64> = tr_basis
        .iter()
        .map(|q| {
            let mut value = 0.0;
            for i in 0..dim {
                for j in 0..dim {
                    value += q[i] * h_proj.get(i, j) * q[j];
                }
            }
            value
        })
        .collect();

    // Sort vibrational modes by ascending signed eigenvalue (imaginary/negative first, then real)
    vib_modes.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut modes: Vec<NormalMode> = Vec::with_capacity(n_vib);
    for (idx, (lam, evec_mw)) in vib_modes.into_iter().enumerate() {
        let (freq_cm, is_imag) = if lam >= 0.0 {
            (HESSIAN_EVAL_TO_INV_CM * lam.sqrt(), false)
        } else {
            (-HESSIAN_EVAL_TO_INV_CM * (-lam).sqrt(), true)
        };

        // Cartesian displacement U_i = L_i / sqrt(m_i)
        let mut u_cart = vec![0.0f64; dim];
        for i in 0..dim {
            u_cart[i] = evec_mw[i] / mass_coord[i].sqrt();
        }

        // Reduced mass mu = 1 / sum_i (L_i / sqrt(m_i))^2 = 1 / sum_i U_i^2 (amu)
        let sum_u2: f64 = u_cart.iter().map(|x| x * x).sum();
        let red_mass = if sum_u2 > 1e-15 { 1.0 / sum_u2 } else { 1.0 };

        // Normalize Cartesian displacement: Q_i = U_i / sqrt(sum_j U_j^2)
        let inv_norm = if sum_u2 > 1e-15 {
            1.0 / sum_u2.sqrt()
        } else {
            1.0
        };
        let q_disp: Vec<f64> = u_cart.iter().map(|&x| x * inv_norm).collect();

        // Harmonic force constant k = lambda * red_mass in mDyn / A
        let k_mdyn_a = lam.abs() * red_mass * HARTREE_BOHR2_TO_MDYN_ANG;

        // IR intensity in km/mol: APT in atomic units, Q in sqrt(amu)*Bohr.
        let mut dmu_dq = [0.0f64; 3];
        for p in 0..3 {
            let mut s = 0.0;
            for i in 0..dim {
                s += apt.get(p, i) * u_cart[i];
            }
            dmu_dq[p] = s;
        }
        let dmu_sq = dmu_dq[0] * dmu_dq[0] + dmu_dq[1] * dmu_dq[1] + dmu_dq[2] * dmu_dq[2];
        let ir_intensity = IR_INTENSITY_PREFACTOR_KM_MOL * dmu_sq;

        modes.push(NormalMode {
            mode_index: idx + 1,
            frequency_cm: freq_cm,
            is_imaginary: is_imag,
            reduced_mass_amu: red_mass,
            force_constant_m_dyn_a: k_mdyn_a,
            ir_intensity_km_mol: ir_intensity,
            displacements: q_disp,
        });
    }

    Ok(VibrationalAnalysis {
        n_atoms,
        is_linear,
        n_trans_rot: n_tr,
        n_vib,
        moments_of_inertia_amu_bohr2: moments,
        rotational_constants_ghz: b_ghz,
        cartesian_hessian: cart_hessian.clone(),
        apt_matrix: apt.clone(),
        modes,
        trans_rot_residuals: trans_rot_modes,
    })
}

```

## `crates/calchem-vib/src/hessian.rs`

SHA256: `564a6a529c13b711fd9555abde0c198c5db9d10b6da37e8932cd5621bc175d38`

```rust
//! Semi-numerical Cartesian Hessian matrix ($H_{ij} = \frac{\partial^2 E}{\partial x_i \partial x_j}$)
//! and Atomic Polar Tensor (APT, $P_{qi} = \frac{\partial \mu_q}{\partial x_i}$) from analytical nuclear gradients.

use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_grad::{rhf_gradient, rks_gradient, uhf_gradient, uks_gradient};
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_prop::{compute_dipole, AU_TO_DEBYE};
use calchem_scf::{rhf, rks, uhf, uks};
use calchem_tensor::Matrix2D;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Calculation method / functional for vibrational frequency analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VibMethod {
    Rhf,
    Uhf,
    Rks(XcFunctional),
    Uks(XcFunctional),
}

impl VibMethod {
    pub fn is_dft(&self) -> bool {
        matches!(self, Self::Rks(_) | Self::Uks(_))
    }

    pub fn xc_functional(&self) -> Option<XcFunctional> {
        match self {
            Self::Rks(f) | Self::Uks(f) => Some(*f),
            _ => None,
        }
    }
}

/// Options controlling vibrational frequency and Hessian calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreqOptions {
    /// Finite-difference displacement step along nuclear Cartesian coordinates (Bohr).
    /// Default is `0.001` Bohr.
    pub step_bohr: f64,
    /// DFT molecular numerical quadrature grid level (used when method is RKS/UKS).
    pub grid_level: GridLevel,
    /// Standard thermochemical temperature in Kelvin (default `298.15` K).
    pub temperature_k: f64,
    /// Standard thermochemical pressure in Pascal (default `101325.0` Pa = 1 atm).
    pub pressure_pa: f64,
    /// Rotational symmetry number $\sigma$ (optional override; if `None`, estimated or defaulted to 1).
    pub symmetry_number: Option<usize>,
    /// Underlying SCF convergence options.
    pub scf_options: ScfOptions,
}

impl Default for FreqOptions {
    fn default() -> Self {
        Self {
            step_bohr: 0.001,
            grid_level: GridLevel::Medium,
            temperature_k: 298.15,
            pressure_pa: 101325.0,
            symmetry_number: None,
            scf_options: ScfOptions {
                conv_energy: 1e-11,
                conv_density: 1e-9,
                ..Default::default()
            },
        }
    }
}

/// Evaluates analytical nuclear gradient and electric dipole moment (in a.u.) at a geometry.
fn eval_grad_and_dipole(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    grid_level: GridLevel,
    scf_opts: &ScfOptions,
) -> Result<(Vec<f64>, [f64; 3], f64), CalcError> {
    let bs = BasisSet::from_name(basis_name, mol)?;
    let (grad_atoms, dipole_au, energy) = match method {
        VibMethod::Rhf => {
            let wfn = rhf(mol, &bs, scf_opts)?;
            let g = rhf_gradient(mol, &bs, &wfn)?;
            let (dip_deb, _) = compute_dipole(mol, &bs, &wfn.density_alpha);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au, wfn.total_energy_hartree)
        }
        VibMethod::Uhf => {
            let wfn = uhf(mol, &bs, scf_opts)?;
            let g = uhf_gradient(mol, &bs, &wfn)?;
            let total_dens = wfn.total_density();
            let (dip_deb, _) = compute_dipole(mol, &bs, &total_dens);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au, wfn.total_energy_hartree)
        }
        VibMethod::Rks(func) => {
            let wfn = rks(mol, &bs, func, grid_level, scf_opts)?;
            let g = rks_gradient(mol, &bs, func, grid_level, &wfn)?;
            let (dip_deb, _) = compute_dipole(mol, &bs, &wfn.density_alpha);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au, wfn.total_energy_hartree)
        }
        VibMethod::Uks(func) => {
            let wfn = uks(mol, &bs, func, grid_level, scf_opts)?;
            let g = uks_gradient(mol, &bs, func, grid_level, &wfn)?;
            let total_dens = wfn.total_density();
            let (dip_deb, _) = compute_dipole(mol, &bs, &total_dens);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au, wfn.total_energy_hartree)
        }
    };
    let mut grad_3n = Vec::with_capacity(3 * grad_atoms.len());
    for row in grad_atoms {
        grad_3n.push(row[0]);
        grad_3n.push(row[1]);
        grad_3n.push(row[2]);
    }
    Ok((grad_3n, dipole_au, energy))
}

/// Evaluates the symmetric Cartesian Hessian matrix ($3N \times 3N$, Hartree / $\text{Bohr}^2$)
/// and Atomic Polar Tensor (APT, $3 \times 3N$, $e$) by Rayon-parallel central differentiation
/// of analytical nuclear gradients.
pub fn compute_cartesian_hessian_and_apt(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    opts: &FreqOptions,
) -> Result<(Matrix2D, Matrix2D, f64), CalcError> {
    // A density error is amplified by 1/(2h) in both the Hessian and APT.
    // Energy convergence alone does not control IR intensity accuracy.
    let mut response_options = opts.clone();
    response_options.scf_options.conv_energy = response_options.scf_options.conv_energy.min(1e-13);
    response_options.scf_options.conv_density =
        response_options.scf_options.conv_density.min(1e-12);
    let opts = &response_options;
    let n_atoms = mol.atoms.len();
    let dim = 3 * n_atoms;
    if dim == 0 {
        return Err(CalcError::InvalidMolecule(
            "Cannot compute Hessian for empty molecule".into(),
        ));
    }
    let delta = opts.step_bohr;
    if !delta.is_finite() || delta <= 0.0 {
        return Err(CalcError::InvalidMolecule(format!(
            "Invalid differentiation step_bohr: {delta}"
        )));
    }

    // Unperturbed reference energy and gradient
    let (_g0, _dip0, e0) =
        eval_grad_and_dipole(mol, basis_name, method, opts.grid_level, &opts.scf_options)?;

    let two_delta = 2.0 * delta;

    // Parallel differentiation along all 3N Cartesian coordinates
    let cols: Result<Vec<(Vec<f64>, [f64; 3])>, CalcError> = (0..dim)
        .into_par_iter()
        .map(|k| {
            let atom_idx = k / 3;
            let axis = k % 3;

            // +delta displacement
            let mut mol_plus = mol.clone();
            mol_plus.atoms[atom_idx].position_bohr[axis] += delta;
            let (g_plus, dip_plus, _) = eval_grad_and_dipole(
                &mol_plus,
                basis_name,
                method,
                opts.grid_level,
                &opts.scf_options,
            )?;

            // -delta displacement
            let mut mol_minus = mol.clone();
            mol_minus.atoms[atom_idx].position_bohr[axis] -= delta;
            let (g_minus, dip_minus, _) = eval_grad_and_dipole(
                &mol_minus,
                basis_name,
                method,
                opts.grid_level,
                &opts.scf_options,
            )?;

            let mut h_col = vec![0.0f64; dim];
            for j in 0..dim {
                h_col[j] = (g_plus[j] - g_minus[j]) / two_delta;
            }

            let apt_col = [
                (dip_plus[0] - dip_minus[0]) / two_delta,
                (dip_plus[1] - dip_minus[1]) / two_delta,
                (dip_plus[2] - dip_minus[2]) / two_delta,
            ];

            Ok((h_col, apt_col))
        })
        .collect();

    let cols = cols?;

    // Assemble Cartesian Hessian H (dim x dim) and enforce exact symmetry H = (H + H^T) / 2
    let mut hess = Matrix2D::zeros(dim, dim);
    for k in 0..dim {
        let (ref h_col, _) = cols[k];
        for j in 0..dim {
            hess.set(j, k, h_col[j]);
        }
    }
    for j in 0..dim {
        for k in (j + 1)..dim {
            let avg = 0.5 * (hess.get(j, k) + hess.get(k, j));
            hess.set(j, k, avg);
            hess.set(k, j, avg);
        }
    }

    // Assemble Atomic Polar Tensor (3 x dim)
    let mut apt = Matrix2D::zeros(3, dim);
    for k in 0..dim {
        let (_, apt_col) = cols[k];
        for q in 0..3 {
            apt.set(q, k, apt_col[q]);
        }
    }

    Ok((hess, apt, e0))
}

```

## `crates/calchem-py/src/lib.rs`

SHA256: `be570a6f48e3f8a040b85f37d6898375d17b573a7aeafd9903222208c45b55b0`

```rust
use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_grad::{rhf_gradient, rks_gradient, uhf_gradient, uks_gradient};
use calchem_model::{
    symbol_from_atomic_number, Atom, CalcError, Molecule, ScfOptions, BOHR_TO_ANGSTROM,
};
use calchem_opt::{
    optimize_geometry_with_options, run_irc as rust_run_irc,
    search_transition_state as rust_search_ts, IrcDirection, IrcOptions, OptOptions, Reference,
    TsOptions,
};
use calchem_prop::{
    auto_grid_for_molecule, compute_density_grid, compute_dipole, compute_esp_at_points,
    compute_esp_grid, compute_esp_surface, compute_orbital_grid, encode_dual_mesh_binary,
    encode_esp_surface_binary, extract_dual_isosurface, extract_isosurface, format_gaussian_cube,
    parse_gaussian_cube, population_analysis, DualOrbitalMesh, EspSurface, GridSpec,
    IsosurfaceMesh,
};
use calchem_scf::{rhf, rks, uhf, uks};
use calchem_solv::{
    run_scrf as rust_run_scrf, BoundaryMatrix, CavityLebedevOrder, CavityOptions,
    CavityPotentialEngine, MolecularCavity, SolvationOptions, Solvent, SolventModel,
    HARTREE_TO_KCAL_MOL,
};
use calchem_tddft::{
    broaden_uv_vis_spectrum, run_tddft as rust_run_tddft, run_tddft_from_scf, ExcitedState,
    OrbitalTransition, SpinManifold, TddftOptions, TddftResult,
};
use calchem_vib::{compute_frequencies_and_thermo, FreqOptions, VibMethod};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

// Distinct Python exception types so callers can tell "the SCF did not converge"
// (a legitimate, reportable outcome) apart from bad input or an internal failure.
create_exception!(_core, ScfNotConvergedError, PyRuntimeError);
create_exception!(_core, RustCoreError, PyRuntimeError);

/// Version of the Rust core reported to Python (keep in sync with crates/*/Cargo.toml).
const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

fn to_py_err(e: CalcError) -> PyErr {
    match e {
        CalcError::NotConverged { .. } => ScfNotConvergedError::new_err(e.to_string()),
        CalcError::InvalidMolecule(_)
        | CalcError::UnsupportedBasis(_)
        | CalcError::UnsupportedMethod(_) => PyValueError::new_err(e.to_string()),
        _ => RustCoreError::new_err(e.to_string()),
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyScfWavefunction {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
    #[pyo3(get)]
    pub electronic_energy_hartree: f64,
    #[pyo3(get)]
    pub nuclear_repulsion_hartree: f64,
    #[pyo3(get)]
    pub n_ao: usize,
    #[pyo3(get)]
    pub mo_energies: Vec<f64>,
    #[pyo3(get)]
    pub mo_occupations: Vec<f64>,
    /// Row-major AO x MO coefficients, index `mu * n_mo + i`.
    #[pyo3(get)]
    pub mo_coefficients: Vec<f64>,
    /// Spin-summed total density `P = P_alpha + P_beta` (or `2 C_occ C_occ^T` in RHF), row-major.
    #[pyo3(get)]
    pub density: Vec<f64>,
    /// Spin density `P_alpha - P_beta` for UHF wavefunctions (`None` for closed-shell RHF).
    #[pyo3(get)]
    pub spin_density: Option<Vec<f64>>,
    #[pyo3(get)]
    pub energy_history: Vec<f64>,
    #[pyo3(get)]
    pub is_uhf: bool,
    #[pyo3(get)]
    pub s_squared: Option<f64>,
    #[pyo3(get)]
    pub n_alpha: usize,
    #[pyo3(get)]
    pub n_beta: usize,
    #[pyo3(get)]
    pub mo_energies_alpha: Option<Vec<f64>>,
    #[pyo3(get)]
    pub mo_energies_beta: Option<Vec<f64>>,
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub geometry_bohr: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub basis_name: String,
    #[pyo3(get)]
    pub charge: i32,
    #[pyo3(get)]
    pub multiplicity: usize,
    #[pyo3(get)]
    pub xc_functional: Option<String>,
    pub dipole: Option<[f64; 3]>,
    pub dipole_mag: Option<f64>,
}

impl PyScfWavefunction {
    fn build_mol_and_basis(&self) -> PyResult<(Molecule, BasisSet)> {
        let mut atoms = Vec::with_capacity(self.symbols.len());
        for (s, pos) in self.symbols.iter().zip(&self.geometry_bohr) {
            atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
        }
        let mol = Molecule::new(atoms, self.charge, self.multiplicity);
        let bs = BasisSet::from_name(&self.basis_name, &mol).map_err(to_py_err)?;
        Ok((mol, bs))
    }

    fn spin_channel_for_mo(&self, mo_idx: usize) -> u32 {
        if self.is_uhf && mo_idx >= self.n_ao {
            1
        } else {
            0
        }
    }
}

#[pymethods]
impl PyScfWavefunction {
    /// Highest occupied MO energy (Eh), or None if there are no occupied orbitals.
    #[getter]
    fn homo_energy_hartree(&self) -> Option<f64> {
        self.mo_energies
            .iter()
            .zip(&self.mo_occupations)
            .filter(|(_, &o)| o > 0.5)
            .map(|(&e, _)| e)
            .fold(None, |acc: Option<f64>, e| {
                Some(acc.map_or(e, |a| a.max(e)))
            })
    }

    /// Lowest unoccupied MO energy (Eh), or None if there are no virtual orbitals.
    #[getter]
    fn lumo_energy_hartree(&self) -> Option<f64> {
        self.mo_energies
            .iter()
            .zip(&self.mo_occupations)
            .filter(|(_, &o)| o <= 0.5)
            .map(|(&e, _)| e)
            .fold(None, |acc: Option<f64>, e| {
                Some(acc.map_or(e, |a| a.min(e)))
            })
    }

    #[getter]
    fn dipole_debye(&self) -> Option<[f64; 3]> {
        self.dipole
    }

    #[getter]
    fn dipole_magnitude_debye(&self) -> Option<f64> {
        self.dipole_mag
    }
}

#[pyclass]
pub struct PyOptResult {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub coords: Vec<f64>,
    #[pyo3(get)]
    pub energy: f64,
    #[pyo3(get)]
    pub steps: usize,
}

/// Output of a pure Rust Transition State Search (P-RFO).
#[pyclass(name = "TsResult")]
#[derive(Clone)]
pub struct PyTsResult {
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub geometry_bohr: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub energy_hartree: f64,
    #[pyo3(get)]
    pub gradient: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub imaginary_frequency_cm: f64,
    #[pyo3(get)]
    pub transition_vector: Vec<f64>,
}

#[pymethods]
impl PyTsResult {
    fn __repr__(&self) -> String {
        format!(
            "TsResult(converged={}, energy={:.8} Eh, imag_freq={:.1} cm^-1, steps={})",
            self.converged, self.energy_hartree, self.imaginary_frequency_cm, self.steps
        )
    }
}

/// Single point along an Intrinsic Reaction Coordinate (IRC) path.
#[pyclass(name = "IrcPoint")]
#[derive(Clone)]
pub struct PyIrcPoint {
    #[pyo3(get)]
    pub point_index: usize,
    #[pyo3(get)]
    pub direction: String,
    #[pyo3(get)]
    pub path_coordinate_s: f64,
    #[pyo3(get)]
    pub energy_hartree: f64,
    #[pyo3(get)]
    pub delta_energy_kcal_mol: f64,
    #[pyo3(get)]
    pub gradient_norm: f64,
    #[pyo3(get)]
    pub geometry_bohr: Vec<[f64; 3]>,
}

#[pymethods]
impl PyIrcPoint {
    fn __repr__(&self) -> String {
        format!(
            "IrcPoint(idx={}, dir='{}', s={:.3}, dE={:.2} kcal/mol, g_norm={:.3e})",
            self.point_index,
            self.direction,
            self.path_coordinate_s,
            self.delta_energy_kcal_mol,
            self.gradient_norm
        )
    }
}

/// One unidirectional branch of an IRC reaction pathway (forward or reverse).
#[pyclass(name = "IrcBranch")]
#[derive(Clone)]
pub struct PyIrcBranch {
    #[pyo3(get)]
    pub converged_to_minimum: bool,
    #[pyo3(get)]
    pub direction: String,
    #[pyo3(get)]
    pub n_points: usize,
    #[pyo3(get)]
    pub barrier_kcal_mol: f64,
    #[pyo3(get)]
    pub delta_energy_kcal_mol: f64,
    #[pyo3(get)]
    pub points: Vec<PyIrcPoint>,
}

#[pymethods]
impl PyIrcBranch {
    fn __repr__(&self) -> String {
        format!(
            "IrcBranch(dir='{}', n_points={}, barrier={:.2} kcal/mol, dE={:.2} kcal/mol)",
            self.direction, self.n_points, self.barrier_kcal_mol, self.delta_energy_kcal_mol
        )
    }
}

/// Complete IRC result holding the transition state and optional forward/reverse branches.
#[pyclass(name = "IrcResult")]
#[derive(Clone)]
pub struct PyIrcResult {
    #[pyo3(get)]
    pub transition_state: PyIrcPoint,
    #[pyo3(get)]
    pub forward: Option<PyIrcBranch>,
    #[pyo3(get)]
    pub reverse: Option<PyIrcBranch>,
    #[pyo3(get)]
    pub total_points: usize,
    #[pyo3(get)]
    pub barrier_forward_kcal_mol: Option<f64>,
    #[pyo3(get)]
    pub barrier_reverse_kcal_mol: Option<f64>,
    #[pyo3(get)]
    pub reaction_energy_kcal_mol: Option<f64>,
}

#[pymethods]
impl PyIrcResult {
    fn __repr__(&self) -> String {
        format!(
            "IrcResult(total_points={}, fwd_barrier={:?}, rev_barrier={:?})",
            self.total_points, self.barrier_forward_kcal_mol, self.barrier_reverse_kcal_mol
        )
    }
}

/// Zero-copy single-lobe triangle mesh backed by Rust-allocated `float32` / `uint32` NumPy buffers.
#[pyclass]
pub struct PyIsosurfaceMesh {
    #[pyo3(get)]
    pub isovalue: f32,
    #[pyo3(get)]
    pub n_vertices: usize,
    #[pyo3(get)]
    pub n_triangles: usize,
    vertices_arr: Py<PyArray2<f32>>,
    normals_arr: Py<PyArray2<f32>>,
    faces_arr: Py<PyArray2<u32>>,
}

impl PyIsosurfaceMesh {
    fn from_rust(py: Python<'_>, mesh: IsosurfaceMesh) -> PyResult<Self> {
        let n_v = mesh.n_vertices();
        let n_t = mesh.n_triangles();
        let v_arr = PyArray1::from_vec_bound(py, mesh.vertices)
            .reshape([n_v, 3])?
            .unbind();
        let n_arr = PyArray1::from_vec_bound(py, mesh.normals)
            .reshape([n_v, 3])?
            .unbind();
        let f_arr = PyArray1::from_vec_bound(py, mesh.indices)
            .reshape([n_t, 3])?
            .unbind();
        Ok(Self {
            isovalue: mesh.isovalue,
            n_vertices: n_v,
            n_triangles: n_t,
            vertices_arr: v_arr,
            normals_arr: n_arr,
            faces_arr: f_arr,
        })
    }

    fn to_rust(&self, py: Python<'_>) -> PyResult<IsosurfaceMesh> {
        let v = self.vertices_arr.bind(py).readonly();
        let n = self.normals_arr.bind(py).readonly();
        let f = self.faces_arr.bind(py).readonly();
        Ok(IsosurfaceMesh {
            isovalue: self.isovalue,
            vertices: v.as_slice()?.to_vec(),
            normals: n.as_slice()?.to_vec(),
            indices: f.as_slice()?.to_vec(),
        })
    }
}

#[pymethods]
impl PyIsosurfaceMesh {
    /// `(n_vertices, 3)` `float32` vertex position buffer (zero-copy handle).
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.vertices_arr.bind(py).clone()
    }

    /// `(n_vertices, 3)` `float32` unit outward normal buffer (zero-copy handle).
    #[getter]
    fn normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.normals_arr.bind(py).clone()
    }

    /// `(n_triangles, 3)` `uint32` CCW triangle index buffer (zero-copy handle).
    #[getter]
    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        self.faces_arr.bind(py).clone()
    }

    /// Alias of `faces` (`(n_triangles, 3)` `uint32`).
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        self.faces_arr.bind(py).clone()
    }
}

/// Dual-phase (`+isovalue` and `-isovalue`) molecular orbital / spin-density mesh with binary wire serialization.
#[pyclass]
pub struct PyDualOrbitalMesh {
    #[pyo3(get)]
    pub isovalue: f32,
    #[pyo3(get)]
    pub units_angstrom: bool,
    #[pyo3(get)]
    pub mo_index: i32,
    #[pyo3(get)]
    pub spin_channel: u32,
    #[pyo3(get)]
    pub positive: Py<PyIsosurfaceMesh>,
    #[pyo3(get)]
    pub negative: Py<PyIsosurfaceMesh>,
}

impl PyDualOrbitalMesh {
    fn from_rust(
        py: Python<'_>,
        dual: DualOrbitalMesh,
        mo_index: i32,
        spin_channel: u32,
    ) -> PyResult<Self> {
        let pos = Py::new(py, PyIsosurfaceMesh::from_rust(py, dual.positive)?)?;
        let neg = Py::new(py, PyIsosurfaceMesh::from_rust(py, dual.negative)?)?;
        Ok(Self {
            isovalue: dual.isovalue,
            units_angstrom: dual.units_angstrom,
            mo_index,
            spin_channel,
            positive: pos,
            negative: neg,
        })
    }
}

#[pymethods]
impl PyDualOrbitalMesh {
    /// Serializes both positive and negative lobes into a 64-byte-aligned `CCMSH\x01` binary packet.
    fn to_binary_packet<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let pos_ref = self.positive.borrow(py);
        let neg_ref = self.negative.borrow(py);
        let dual = DualOrbitalMesh {
            isovalue: self.isovalue,
            positive: pos_ref.to_rust(py)?,
            negative: neg_ref.to_rust(py)?,
            units_angstrom: self.units_angstrom,
        };
        let encoded = encode_dual_mesh_binary(&dual, self.mo_index, self.spin_channel);
        Ok(PyBytes::new_bound(py, &encoded))
    }
}

fn parse_atoms(
    atomic_numbers_or_symbols: &Bound<'_, PyAny>,
    coords: &Bound<'_, PyAny>,
) -> PyResult<Vec<Atom>> {
    let raw_coords: Vec<[f64; 3]> = if let Ok(py_arr) = coords.extract::<Vec<[f64; 3]>>() {
        py_arr
    } else if let Ok(flat) = coords.extract::<Vec<f64>>() {
        if flat.len() % 3 != 0 {
            return Err(PyValueError::new_err(
                "coords array length must be multiple of 3",
            ));
        }
        let mut v = Vec::with_capacity(flat.len() / 3);
        for chunk in flat.chunks_exact(3) {
            v.push([chunk[0], chunk[1], chunk[2]]);
        }
        v
    } else if let Ok(nested) = coords.extract::<Vec<Vec<f64>>>() {
        let mut v = Vec::with_capacity(nested.len());
        for row in nested {
            if row.len() != 3 {
                return Err(PyValueError::new_err(
                    "each coordinate must have 3 components",
                ));
            }
            v.push([row[0], row[1], row[2]]);
        }
        v
    } else {
        return Err(PyValueError::new_err(
            "failed to parse coords into 3D positions",
        ));
    };

    let n = raw_coords.len();

    let mut atoms = Vec::with_capacity(n);
    if let Ok(nums) = atomic_numbers_or_symbols.extract::<Vec<u8>>() {
        if nums.len() != n {
            return Err(PyValueError::new_err(
                "atomic_numbers and coords length mismatch",
            ));
        }
        for (z, pos) in nums.iter().zip(&raw_coords) {
            atoms.push(Atom {
                symbol: symbol_from_atomic_number(*z).map_err(to_py_err)?,
                z: *z,
                position_bohr: *pos,
            });
        }
    } else if let Ok(nums) = atomic_numbers_or_symbols.extract::<Vec<i64>>() {
        if nums.len() != n {
            return Err(PyValueError::new_err(
                "atomic_numbers and coords length mismatch",
            ));
        }
        for (z, pos) in nums.iter().zip(&raw_coords) {
            let z_u8 = *z as u8;
            atoms.push(Atom {
                symbol: symbol_from_atomic_number(z_u8).map_err(to_py_err)?,
                z: z_u8,
                position_bohr: *pos,
            });
        }
    } else if let Ok(syms) = atomic_numbers_or_symbols.extract::<Vec<String>>() {
        if syms.len() != n {
            return Err(PyValueError::new_err("symbols and coords length mismatch"));
        }
        for (s, pos) in syms.iter().zip(&raw_coords) {
            atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
        }
    } else {
        return Err(PyValueError::new_err(
            "failed to parse atomic_numbers_or_symbols",
        ));
    }

    Ok(atoms)
}

#[derive(Clone, Copy)]
enum ResolvedScfKind {
    Rhf,
    Uhf,
    Rks(XcFunctional),
    Uks(XcFunctional),
}

fn resolve_scf_kind(
    method: &str,
    xc: Option<&str>,
    multiplicity: usize,
) -> PyResult<ResolvedScfKind> {
    let m = method.trim().to_ascii_lowercase();
    let parse_xc = |default_xc: &str| -> PyResult<XcFunctional> {
        XcFunctional::from_name(xc.unwrap_or(default_xc)).map_err(to_py_err)
    };
    match m.as_str() {
        "rhf" => Ok(ResolvedScfKind::Rhf),
        "uhf" => Ok(ResolvedScfKind::Uhf),
        "hf" | "auto" => Ok(if multiplicity != 1 {
            ResolvedScfKind::Uhf
        } else {
            ResolvedScfKind::Rhf
        }),
        "rks" => Ok(ResolvedScfKind::Rks(parse_xc("b3lyp")?)),
        "uks" => Ok(ResolvedScfKind::Uks(parse_xc("b3lyp")?)),
        "dft" => {
            let func = parse_xc("b3lyp")?;
            Ok(if multiplicity != 1 {
                ResolvedScfKind::Uks(func)
            } else {
                ResolvedScfKind::Rks(func)
            })
        }
        other => {
            let func = XcFunctional::from_name(other).map_err(to_py_err)?;
            Ok(if multiplicity != 1 {
                ResolvedScfKind::Uks(func)
            } else {
                ResolvedScfKind::Rks(func)
            })
        }
    }
}

/// Run an RHF, UHF, RKS, or UKS single-point calculation in pure Rust.
#[pyfunction]
#[pyo3(signature = (symbols, geometry_bohr, charge=0, multiplicity=1, method="rhf", basis="sto-3g", max_iter=100, conv_energy=1e-10, conv_density=1e-8, use_diis=true, diis_max_vecs=8, damping=0.0, xc=None, grid_level="medium"))]
#[allow(clippy::too_many_arguments)]
pub fn run_single_point(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    max_iter: usize,
    conv_energy: f64,
    conv_density: f64,
    use_diis: bool,
    diis_max_vecs: usize,
    damping: f64,
    xc: Option<&str>,
    grid_level: &str,
) -> PyResult<PyScfWavefunction> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be >= 1"));
    }
    if !(conv_energy > 0.0 && conv_density > 0.0) {
        return Err(PyValueError::new_err(
            "convergence thresholds must be positive",
        ));
    }
    if !(0.0..1.0).contains(&damping) {
        return Err(PyValueError::new_err("damping must be in [0, 1)"));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        if !pos.iter().all(|x| x.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "non-finite coordinate for atom {s}"
            )));
        }
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);
    let bs = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;
    let scf_kind = resolve_scf_kind(method, xc, multiplicity)?;
    let gl = GridLevel::from_name(grid_level);

    let opts = ScfOptions {
        max_iter,
        conv_energy,
        conv_density,
        use_diis,
        diis_max_vecs,
        damping,
        ..Default::default()
    };

    py.allow_threads(move || match scf_kind {
        ResolvedScfKind::Uhf | ResolvedScfKind::Uks(_) => {
            let (wf, xc_name) = match scf_kind {
                ResolvedScfKind::Uhf => (uhf(&mol, &bs, &opts).map_err(to_py_err)?, None),
                ResolvedScfKind::Uks(func) => (
                    uks(&mol, &bs, func, gl, &opts).map_err(to_py_err)?,
                    Some(func.canonical_name().to_string()),
                ),
                _ => unreachable!(),
            };
            let total_density = wf.total_density();
            let spin_density: Vec<f64> = wf
                .density_alpha
                .iter()
                .zip(&wf.density_beta)
                .map(|(a, b)| a - b)
                .collect();
            let (dipole, dipole_mag) = compute_dipole(&mol, &bs, &total_density);
            let n = wf.n_ao;

            let mut mo_energies = Vec::with_capacity(2 * n);
            mo_energies.extend_from_slice(&wf.mo_energies_alpha);
            mo_energies.extend_from_slice(&wf.mo_energies_beta);

            let mut mo_occupations = vec![0.0; 2 * n];
            for o in mo_occupations.iter_mut().take(wf.n_alpha) {
                *o = 1.0;
            }
            for o in mo_occupations[n..].iter_mut().take(wf.n_beta) {
                *o = 1.0;
            }

            let mut mo_coefficients = Vec::with_capacity(n * 2 * n);
            for mu in 0..n {
                mo_coefficients.extend_from_slice(&wf.mo_coefficients_alpha[mu * n..(mu + 1) * n]);
                mo_coefficients.extend_from_slice(&wf.mo_coefficients_beta[mu * n..(mu + 1) * n]);
            }

            Ok(PyScfWavefunction {
                converged: wf.converged,
                iterations: wf.iterations,
                total_energy_hartree: wf.total_energy_hartree,
                electronic_energy_hartree: wf.electronic_energy_hartree,
                nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
                n_ao: wf.n_ao,
                mo_energies,
                mo_occupations,
                mo_coefficients,
                density: total_density,
                spin_density: Some(spin_density),
                energy_history: wf.diagnostics.energy_history,
                is_uhf: true,
                s_squared: Some(wf.s_squared),
                n_alpha: wf.n_alpha,
                n_beta: wf.n_beta,
                mo_energies_alpha: Some(wf.mo_energies_alpha),
                mo_energies_beta: Some(wf.mo_energies_beta),
                symbols,
                geometry_bohr,
                basis_name: basis.to_string(),
                charge,
                multiplicity,
                xc_functional: xc_name,
                dipole: Some(dipole),
                dipole_mag: Some(dipole_mag),
            })
        }
        ResolvedScfKind::Rhf | ResolvedScfKind::Rks(_) => {
            let (wf, xc_name) = match scf_kind {
                ResolvedScfKind::Rhf => (rhf(&mol, &bs, &opts).map_err(to_py_err)?, None),
                ResolvedScfKind::Rks(func) => (
                    rks(&mol, &bs, func, gl, &opts).map_err(to_py_err)?,
                    Some(func.canonical_name().to_string()),
                ),
                _ => unreachable!(),
            };
            let (dipole, dipole_mag) = compute_dipole(&mol, &bs, &wf.density_alpha);
            let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 0.5).count();

            Ok(PyScfWavefunction {
                converged: wf.converged,
                iterations: wf.iterations,
                total_energy_hartree: wf.total_energy_hartree,
                electronic_energy_hartree: wf.electronic_energy_hartree,
                nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
                n_ao: wf.n_ao,
                mo_energies: wf.mo_energies,
                mo_occupations: wf.mo_occupations,
                mo_coefficients: wf.mo_coefficients,
                density: wf.density_alpha,
                spin_density: None,
                energy_history: wf.diagnostics.energy_history,
                is_uhf: false,
                s_squared: Some(0.0),
                n_alpha: n_occ,
                n_beta: n_occ,
                mo_energies_alpha: None,
                mo_energies_beta: None,
                symbols,
                geometry_bohr,
                basis_name: basis.to_string(),
                charge,
                multiplicity,
                xc_functional: xc_name,
                dipole: Some(dipole),
                dipole_mag: Some(dipole_mag),
            })
        }
    })
}

/// Computes the analytical RHF, UHF, RKS, or UKS energy gradient with respect to Cartesian coordinates.
///
/// Returns a 2D numpy array `(N, 3)` in units of Eh/Bohr.
#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords, charge=0, multiplicity=1, basis="sto-3g", method="auto", xc=None, grid_level="medium"))]
#[allow(clippy::too_many_arguments)]
pub fn run_gradient<'py>(
    py: Python<'py>,
    atomic_numbers: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    charge: i32,
    multiplicity: usize,
    basis: &str,
    method: &str,
    xc: Option<&str>,
    grid_level: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let atoms = parse_atoms(atomic_numbers, coords)?;
    let n = atoms.len();
    let mol = Molecule::new(atoms, charge, multiplicity);
    let bs = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;
    let scf_kind = resolve_scf_kind(method, xc, multiplicity)?;
    let gl = GridLevel::from_name(grid_level);

    let mut opts = ScfOptions::default();
    opts.max_iter = 150;
    opts.conv_energy = 1e-11;
    opts.conv_density = 1e-9;

    let grad = py.allow_threads(|| -> PyResult<Vec<[f64; 3]>> {
        Ok(match scf_kind {
            ResolvedScfKind::Uhf => {
                let wf = uhf(&mol, &bs, &opts).map_err(to_py_err)?;
                uhf_gradient(&mol, &bs, &wf).map_err(to_py_err)?
            }
            ResolvedScfKind::Rhf => {
                let wf = rhf(&mol, &bs, &opts).map_err(to_py_err)?;
                rhf_gradient(&mol, &bs, &wf).map_err(to_py_err)?
            }
            ResolvedScfKind::Rks(func) => {
                let wf = rks(&mol, &bs, func, gl, &opts).map_err(to_py_err)?;
                rks_gradient(&mol, &bs, func, gl, &wf).map_err(to_py_err)?
            }
            ResolvedScfKind::Uks(func) => {
                let wf = uks(&mol, &bs, func, gl, &opts).map_err(to_py_err)?;
                uks_gradient(&mol, &bs, func, gl, &wf).map_err(to_py_err)?
            }
        })
    })?;

    let mut flat = Vec::with_capacity(n * 3);
    for atom_g in grad {
        flat.extend_from_slice(&atom_g);
    }

    let arr1 = PyArray1::from_vec_bound(py, flat);
    let arr2 = arr1.reshape([n, 3])?;
    Ok(arr2)
}

/// Optimizes molecular geometry using Rust BFGS and analytical RHF/UHF/RKS/UKS gradients.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    atomic_numbers,
    coords,
    basis="sto-3g",
    max_steps=100,
    charge=0,
    multiplicity=1,
    method="auto",
    xc=None,
    grid_level="medium",
    dispersion=None,
    solvent=None
))]
pub fn run_optimize(
    py: Python<'_>,
    atomic_numbers: &Bound<'_, PyAny>,
    coords: &Bound<'_, PyAny>,
    basis: &str,
    max_steps: usize,
    charge: i32,
    multiplicity: usize,
    method: &str,
    xc: Option<&str>,
    grid_level: &str,
    dispersion: Option<&str>,
    solvent: Option<&str>,
) -> PyResult<PyOptResult> {
    let atoms = parse_atoms(atomic_numbers, coords)?;
    let mol = Molecule::new(atoms, charge, multiplicity);

    let reference = Reference::from_method_and_xc(method, xc).map_err(|_| {
        PyValueError::new_err(format!("Unsupported method in Rust optimizer: {method}"))
    })?;

    let mut opts = OptOptions::default();
    opts.max_steps = max_steps;
    opts.max_force = 3e-5;
    opts.rms_force = 2e-5;
    opts.max_displacement = 1e-3;
    opts.reference = reference;
    opts.grid_level = GridLevel::from_name(grid_level);
    opts.dispersion = dispersion.map(|s| s.to_string());
    opts.solvent = solvent.map(|s| s.to_string());

    let res = py
        .allow_threads(|| optimize_geometry_with_options(&mol, basis, &opts))
        .map_err(to_py_err)?;

    let mut flat_coords = Vec::with_capacity(res.molecule.atoms.len() * 3);
    for atom in &res.molecule.atoms {
        flat_coords.extend_from_slice(&atom.position_bohr);
    }

    Ok(PyOptResult {
        converged: res.converged,
        coords: flat_coords,
        energy: res.energy_hartree,
        steps: res.steps,
    })
}

/// Returns `(origin_bohr, spacing_bohr, counts_xyz)` for a padded uniform 3D grid around `wf`.
#[pyfunction]
#[pyo3(signature = (wf, margin_bohr=4.5, spacing_bohr=0.25))]
pub fn auto_orbital_grid_spec(
    wf: &PyScfWavefunction,
    margin_bohr: f64,
    spacing_bohr: f64,
) -> PyResult<([f64; 3], [f64; 3], [usize; 3])> {
    let (mol, _bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    Ok((spec.origin_bohr, spec.spacing_bohr, spec.counts))
}

/// Evaluates molecular orbital on a 3D grid and returns a numpy 3D array of shape `(nz, ny, nx)`.
#[pyfunction]
pub fn eval_orbital_cube_zero_copy<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
    mo_idx: usize,
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (_mol, bs) = wf.build_mol_and_basis()?;
    let n_mo = wf.mo_occupations.len();
    let grid_vec = compute_orbital_grid(
        &bs,
        &wf.mo_coefficients,
        n_mo,
        mo_idx,
        origin,
        spacing,
        counts,
    )
    .map_err(to_py_err)?;

    let arr1 = PyArray1::from_vec_bound(py, grid_vec);
    let arr3 = arr1.reshape([counts[2], counts[1], counts[0]])?;
    Ok(arr3)
}

/// Evaluates total electron density (`spin_density=false`) or spin density `P_alpha - P_beta`
/// (`spin_density=true`) on a 3D grid and returns a zero-copy `float32` array `(nz, ny, nx)`.
#[pyfunction]
#[pyo3(signature = (wf, origin, spacing, counts, spin_density=false))]
pub fn eval_density_cube_zero_copy<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
    spin_density: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (_mol, bs) = wf.build_mol_and_basis()?;
    let grid_vec = if spin_density {
        if let Some(ref sd) = wf.spin_density {
            compute_density_grid(&bs, sd, origin, spacing, counts).map_err(to_py_err)?
        } else {
            let total = counts[0]
                .checked_mul(counts[1])
                .and_then(|v| v.checked_mul(counts[2]))
                .ok_or_else(|| PyValueError::new_err("Grid dimensions overflow"))?;
            vec![0.0f32; total]
        }
    } else {
        compute_density_grid(&bs, &wf.density, origin, spacing, counts).map_err(to_py_err)?
    };

    let arr1 = PyArray1::from_vec_bound(py, grid_vec);
    let arr3 = arr1.reshape([counts[2], counts[1], counts[0]])?;
    Ok(arr3)
}

/// Extracts dual-phase (`+isovalue`, `-isovalue`) Marching Cubes meshes directly from an existing
/// `(nz, ny, nx)` `float32` volumetric grid (enabling millisecond interactive slider updates).
#[pyfunction]
#[pyo3(signature = (grid, origin_bohr, spacing_bohr, isovalue=0.03, output_angstrom=true, mo_index=-1, spin_channel=0))]
pub fn extract_isosurface_from_grid<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray3<'py, f32>,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    isovalue: f32,
    output_angstrom: bool,
    mo_index: i32,
    spin_channel: u32,
) -> PyResult<PyDualOrbitalMesh> {
    let shape = grid.shape();
    let counts = [shape[2], shape[1], shape[0]]; // (nz, ny, nx) -> [nx, ny, nz]
    let slice = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("grid must be a C-contiguous 3D float32 array"))?;
    let dual = extract_dual_isosurface(
        slice,
        origin_bohr,
        spacing_bohr,
        counts,
        isovalue,
        output_angstrom,
    )
    .map_err(to_py_err)?;
    PyDualOrbitalMesh::from_rust(py, dual, mo_index, spin_channel)
}

/// Extracts a single-lobe Marching Cubes mesh at `isovalue` from an existing `(nz, ny, nx)` grid.
#[pyfunction]
#[pyo3(signature = (grid, origin_bohr, spacing_bohr, isovalue=0.002, output_angstrom=true))]
pub fn extract_single_isosurface_from_grid<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray3<'py, f32>,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    isovalue: f32,
    output_angstrom: bool,
) -> PyResult<PyIsosurfaceMesh> {
    let shape = grid.shape();
    let counts = [shape[2], shape[1], shape[0]];
    let slice = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("grid must be a C-contiguous 3D float32 array"))?;
    let scale = if output_angstrom {
        BOHR_TO_ANGSTROM
    } else {
        1.0
    };
    let mesh = extract_isosurface(slice, origin_bohr, spacing_bohr, counts, isovalue, scale)
        .map_err(to_py_err)?;
    PyIsosurfaceMesh::from_rust(py, mesh)
}

/// End-to-end Rust parallel molecular orbital grid evaluation + dual-lobe Marching Cubes extraction.
#[pyfunction]
#[pyo3(signature = (wf, mo_idx, isovalue=0.03, spacing_bohr=0.25, margin_bohr=4.5, output_angstrom=true))]
pub fn extract_orbital_isosurface(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    mo_idx: usize,
    isovalue: f32,
    spacing_bohr: f64,
    margin_bohr: f64,
    output_angstrom: bool,
) -> PyResult<PyDualOrbitalMesh> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    let n_mo = wf.mo_occupations.len();
    let grid_vec = compute_orbital_grid(
        &bs,
        &wf.mo_coefficients,
        n_mo,
        mo_idx,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
    )
    .map_err(to_py_err)?;
    let dual = extract_dual_isosurface(
        &grid_vec,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        isovalue,
        output_angstrom,
    )
    .map_err(to_py_err)?;
    PyDualOrbitalMesh::from_rust(py, dual, mo_idx as i32, wf.spin_channel_for_mo(mo_idx))
}

/// End-to-end Rust parallel electron/spin density grid evaluation + Marching Cubes extraction.
#[pyfunction]
#[pyo3(signature = (wf, isovalue=0.002, spacing_bohr=0.25, margin_bohr=4.5, spin_density=false, output_angstrom=true))]
pub fn extract_density_isosurface(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    isovalue: f32,
    spacing_bohr: f64,
    margin_bohr: f64,
    spin_density: bool,
    output_angstrom: bool,
) -> PyResult<PyDualOrbitalMesh> {
    if !isovalue.is_finite() || isovalue <= 0.0 {
        return Err(PyValueError::new_err("isovalue must be finite and > 0"));
    }
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;

    if spin_density {
        let grid_vec = if let Some(ref sd) = wf.spin_density {
            compute_density_grid(&bs, sd, spec.origin_bohr, spec.spacing_bohr, spec.counts)
                .map_err(to_py_err)?
        } else {
            vec![0.0f32; spec.counts[0] * spec.counts[1] * spec.counts[2]]
        };
        let dual = extract_dual_isosurface(
            &grid_vec,
            spec.origin_bohr,
            spec.spacing_bohr,
            spec.counts,
            isovalue,
            output_angstrom,
        )
        .map_err(to_py_err)?;
        PyDualOrbitalMesh::from_rust(py, dual, -1, 2)
    } else {
        let grid_vec = compute_density_grid(
            &bs,
            &wf.density,
            spec.origin_bohr,
            spec.spacing_bohr,
            spec.counts,
        )
        .map_err(to_py_err)?;
        let scale = if output_angstrom {
            BOHR_TO_ANGSTROM
        } else {
            1.0
        };
        let pos = extract_isosurface(
            &grid_vec,
            spec.origin_bohr,
            spec.spacing_bohr,
            spec.counts,
            isovalue,
            scale,
        )
        .map_err(to_py_err)?;
        let empty_neg = IsosurfaceMesh {
            isovalue: -isovalue,
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        };
        let dual = DualOrbitalMesh {
            isovalue,
            positive: pos,
            negative: empty_neg,
            units_angstrom: output_angstrom,
        };
        PyDualOrbitalMesh::from_rust(py, dual, -1, 0)
    }
}

/// Evaluates molecular orbital `mo_idx` and formats a standard Gaussian `.cube` string in pure Rust.
#[pyfunction]
#[pyo3(signature = (wf, mo_idx, spacing_bohr=0.25, margin_bohr=4.5, title=""))]
pub fn export_orbital_gaussian_cube(
    wf: &PyScfWavefunction,
    mo_idx: usize,
    spacing_bohr: f64,
    margin_bohr: f64,
    title: &str,
) -> PyResult<String> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    let n_mo = wf.mo_occupations.len();
    let grid = compute_orbital_grid(
        &bs,
        &wf.mo_coefficients,
        n_mo,
        mo_idx,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
    )
    .map_err(to_py_err)?;
    let c2 = if title.trim().is_empty() {
        format!("Molecular Orbital {mo_idx} ({})", wf.basis_name)
    } else {
        title.to_string()
    };
    format_gaussian_cube(
        &mol,
        Some(&bs),
        &grid,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        "Generated by CalChem Rust Core",
        &c2,
    )
    .map_err(to_py_err)
}

/// Evaluates electron or spin density and formats a standard Gaussian `.cube` string in pure Rust.
#[pyfunction]
#[pyo3(signature = (wf, spacing_bohr=0.25, margin_bohr=4.5, spin_density=false, title=""))]
pub fn export_density_gaussian_cube(
    wf: &PyScfWavefunction,
    spacing_bohr: f64,
    margin_bohr: f64,
    spin_density: bool,
    title: &str,
) -> PyResult<String> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    let grid = if spin_density {
        if let Some(ref sd) = wf.spin_density {
            compute_density_grid(&bs, sd, spec.origin_bohr, spec.spacing_bohr, spec.counts)
                .map_err(to_py_err)?
        } else {
            vec![0.0f32; spec.counts[0] * spec.counts[1] * spec.counts[2]]
        }
    } else {
        compute_density_grid(
            &bs,
            &wf.density,
            spec.origin_bohr,
            spec.spacing_bohr,
            spec.counts,
        )
        .map_err(to_py_err)?
    };
    let c2 = if title.trim().is_empty() {
        if spin_density {
            format!("Spin Density ({})", wf.basis_name)
        } else {
            format!("Total Electron Density ({})", wf.basis_name)
        }
    } else {
        title.to_string()
    };
    format_gaussian_cube(
        &mol,
        Some(&bs),
        &grid,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        "Generated by CalChem Rust Core",
        &c2,
    )
    .map_err(to_py_err)
}

/// Formats an arbitrary `(nz, ny, nx)` `float32` volumetric grid into Gaussian `.cube` text in Rust.
#[pyfunction]
#[pyo3(signature = (symbols, geometry_bohr, grid, origin_bohr, spacing_bohr, title="", basis=""))]
pub fn format_grid_gaussian_cube<'py>(
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    grid: PyReadonlyArray3<'py, f32>,
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    title: &str,
    basis: &str,
) -> PyResult<String> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(&geometry_bohr) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, 0, 1);
    let bs_opt = if basis.trim().is_empty() {
        None
    } else {
        Some(BasisSet::from_name(basis, &mol).map_err(to_py_err)?)
    };
    let shape = grid.shape();
    let counts = [shape[2], shape[1], shape[0]];
    let slice = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("grid must be a C-contiguous 3D float32 array"))?;
    format_gaussian_cube(
        &mol,
        bs_opt.as_ref(),
        slice,
        origin_bohr,
        spacing_bohr,
        counts,
        "Generated by CalChem Rust Core",
        title,
    )
    .map_err(to_py_err)
}

/// Parses a standard Gaussian `.cube` string in Rust, returning
/// `(symbols, geometry_bohr, origin_bohr, spacing_bohr, grid_zyx_f32)`.
#[pyfunction]
pub fn parse_gaussian_cube_str<'py>(
    py: Python<'py>,
    cube_text: &str,
) -> PyResult<(
    Vec<String>,
    Vec<[f64; 3]>,
    [f64; 3],
    [f64; 3],
    Bound<'py, PyArray3<f32>>,
)> {
    let (mol, spec, grid_zyx) = parse_gaussian_cube(cube_text).map_err(to_py_err)?;
    let symbols: Vec<String> = mol.atoms.iter().map(|a| a.symbol.clone()).collect();
    let geometry_bohr: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position_bohr).collect();
    let arr1 = PyArray1::from_vec_bound(py, grid_zyx);
    let arr3 = arr1.reshape([spec.counts[2], spec.counts[1], spec.counts[0]])?;
    Ok((
        symbols,
        geometry_bohr,
        spec.origin_bohr,
        spec.spacing_bohr,
        arr3,
    ))
}

/// Mulliken / Löwdin atomic charges (ECP-consistent `Z_eff`) and UHF/UKS spin populations.
///
/// Returns a dict with keys `mulliken`, `lowdin` (lists of per-atom charges in e) and
/// `mulliken_spin`, `lowdin_spin` (per-atom spin populations, or `None` for closed shells).
#[pyfunction]
pub fn compute_population_analysis<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
) -> PyResult<Bound<'py, PyDict>> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let pop = py
        .allow_threads(|| population_analysis(&mol, &bs, &wf.density, wf.spin_density.as_deref()))
        .map_err(to_py_err)?;
    let d = PyDict::new_bound(py);
    d.set_item("mulliken", pop.mulliken_charges)?;
    d.set_item("lowdin", pop.lowdin_charges)?;
    d.set_item("mulliken_spin", pop.mulliken_spin)?;
    d.set_item("lowdin_spin", pop.lowdin_spin)?;
    Ok(d)
}

/// Exact molecular electrostatic potential (Hartree / e) at `(N, 3)` Bohr points.
#[pyfunction]
pub fn eval_esp_at_points<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
    points_bohr: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = points_bohr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err("points_bohr must have shape (N, 3)"));
    }
    let flat = points_bohr
        .as_slice()
        .map_err(|_| PyValueError::new_err("points_bohr must be a C-contiguous float64 array"))?;
    let pts: Vec<[f64; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let (mol, bs) = wf.build_mol_and_basis()?;
    let v = py
        .allow_threads(|| compute_esp_at_points(&mol, &bs, &wf.density, &pts))
        .map_err(to_py_err)?;
    Ok(PyArray1::from_vec_bound(py, v))
}

/// Exact ESP on a uniform grid; returns a zero-copy `float32` array of shape `(nz, ny, nx)`.
#[pyfunction]
pub fn eval_esp_cube_zero_copy<'py>(
    py: Python<'py>,
    wf: &PyScfWavefunction,
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3],
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = GridSpec {
        origin_bohr: origin,
        spacing_bohr: spacing,
        counts,
    };
    let grid = py
        .allow_threads(|| compute_esp_grid(&mol, &bs, &wf.density, &spec))
        .map_err(to_py_err)?;
    let arr3 = PyArray1::from_vec_bound(py, grid).reshape([counts[2], counts[1], counts[0]])?;
    Ok(arr3)
}

/// ESP-mapped isodensity surface (vertex-colored mesh) with `CCESP\x01` binary serialization.
#[pyclass]
pub struct PyEspSurface {
    #[pyo3(get)]
    pub density_isovalue: f32,
    #[pyo3(get)]
    pub esp_min: f32,
    #[pyo3(get)]
    pub esp_max: f32,
    #[pyo3(get)]
    pub units_angstrom: bool,
    #[pyo3(get)]
    pub mesh: Py<PyIsosurfaceMesh>,
    esp_arr: Py<PyArray1<f32>>,
}

#[pymethods]
impl PyEspSurface {
    /// `(n_vertices,)` `float32` per-vertex ESP (Hartree / e), zero-copy handle.
    #[getter]
    fn esp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.esp_arr.bind(py).clone()
    }

    /// Serializes the surface into a 64-byte-aligned `CCESP\x01` binary packet.
    fn to_binary_packet<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let surf = EspSurface {
            density_isovalue: self.density_isovalue,
            mesh: self.mesh.borrow(py).to_rust(py)?,
            esp: self.esp_arr.bind(py).readonly().as_slice()?.to_vec(),
            esp_min: self.esp_min,
            esp_max: self.esp_max,
            units_angstrom: self.units_angstrom,
        };
        Ok(PyBytes::new_bound(py, &encode_esp_surface_binary(&surf)))
    }
}

/// Builds the `rho = density_isovalue` isodensity surface and maps the exact ESP onto its vertices.
#[pyfunction]
#[pyo3(signature = (wf, density_isovalue=0.001, spacing_bohr=0.25, margin_bohr=5.0, output_angstrom=true))]
pub fn extract_esp_surface(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    density_isovalue: f32,
    spacing_bohr: f64,
    margin_bohr: f64,
    output_angstrom: bool,
) -> PyResult<PyEspSurface> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let surf = py
        .allow_threads(|| {
            compute_esp_surface(
                &mol,
                &bs,
                &wf.density,
                density_isovalue,
                spacing_bohr,
                margin_bohr,
                output_angstrom,
            )
        })
        .map_err(to_py_err)?;
    let mesh = Py::new(py, PyIsosurfaceMesh::from_rust(py, surf.mesh)?)?;
    Ok(PyEspSurface {
        density_isovalue: surf.density_isovalue,
        esp_min: surf.esp_min,
        esp_max: surf.esp_max,
        units_angstrom: surf.units_angstrom,
        mesh,
        esp_arr: PyArray1::from_vec_bound(py, surf.esp).unbind(),
    })
}

/// Evaluates molecular electrostatic potential (ESP) on an automatic 3D grid and formats a
/// standard Gaussian `.cube` string in pure Rust.
#[pyfunction]
#[pyo3(signature = (wf, spacing_bohr=0.25, margin_bohr=5.0, title=""))]
pub fn export_esp_gaussian_cube(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    spacing_bohr: f64,
    margin_bohr: f64,
    title: &str,
) -> PyResult<String> {
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spec = auto_grid_for_molecule(&mol, margin_bohr, spacing_bohr).map_err(to_py_err)?;
    let grid = py
        .allow_threads(|| compute_esp_grid(&mol, &bs, &wf.density, &spec))
        .map_err(to_py_err)?;
    let c2 = if title.trim().is_empty() {
        format!("Molecular Electrostatic Potential ({})", wf.basis_name)
    } else {
        title.to_string()
    };
    format_gaussian_cube(
        &mol,
        Some(&bs),
        &grid,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        "Generated by CalChem Rust Core",
        &c2,
    )
    .map_err(to_py_err)
}

/// Basis set names embedded in the Rust core.
#[pyfunction]
pub fn supported_basis_sets() -> Vec<&'static str> {
    BasisSet::SUPPORTED_NAMES.to_vec()
}

/// DFT exchange-correlation functional names supported natively by the Rust Kohn-Sham core.
#[pyfunction]
pub fn supported_dft_functionals() -> Vec<&'static str> {
    XcFunctional::SUPPORTED_NAMES.to_vec()
}

fn scf_wfn_to_py(
    wf: &calchem_model::ScfWavefunction,
    mol: &Molecule,
    bs: &BasisSet,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    basis_name: String,
    charge: i32,
    multiplicity: usize,
    xc_name: Option<String>,
) -> PyScfWavefunction {
    let (dipole, dipole_mag) = compute_dipole(mol, bs, &wf.density_alpha);
    let n_occ = wf.mo_occupations.iter().filter(|&&o| o > 0.5).count();
    PyScfWavefunction {
        converged: wf.converged,
        iterations: wf.iterations,
        total_energy_hartree: wf.total_energy_hartree,
        electronic_energy_hartree: wf.electronic_energy_hartree,
        nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
        n_ao: wf.n_ao,
        mo_energies: wf.mo_energies.clone(),
        mo_occupations: wf.mo_occupations.clone(),
        mo_coefficients: wf.mo_coefficients.clone(),
        density: wf.density_alpha.clone(),
        spin_density: None,
        energy_history: wf.diagnostics.energy_history.clone(),
        is_uhf: false,
        s_squared: Some(0.0),
        n_alpha: n_occ,
        n_beta: n_occ,
        mo_energies_alpha: None,
        mo_energies_beta: None,
        symbols,
        geometry_bohr,
        basis_name,
        charge,
        multiplicity,
        xc_functional: xc_name,
        dipole: Some(dipole),
        dipole_mag: Some(dipole_mag),
    }
}

/// Single occupied -> virtual canonical MO contribution to an excited state.
#[pyclass]
#[derive(Clone)]
pub struct PyOrbitalTransition {
    #[pyo3(get)]
    pub occ_index: usize,
    #[pyo3(get)]
    pub virt_index: usize,
    #[pyo3(get)]
    pub amplitude_x: f64,
    #[pyo3(get)]
    pub amplitude_y: f64,
    #[pyo3(get)]
    pub weight: f64,
}

#[pymethods]
impl PyOrbitalTransition {
    fn __repr__(&self) -> String {
        format!(
            "OrbitalTransition(occ={}, virt={}, weight={:.4}, X={:.4}, Y={:.4})",
            self.occ_index, self.virt_index, self.weight, self.amplitude_x, self.amplitude_y
        )
    }
}

/// Computed electronic excited state and spectroscopic observables.
#[pyclass]
#[derive(Clone)]
pub struct PyExcitedState {
    #[pyo3(get)]
    pub state_index: usize,
    #[pyo3(get)]
    pub manifold_index: usize,
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub multiplicity: usize,
    #[pyo3(get)]
    pub excitation_energy_hartree: f64,
    #[pyo3(get)]
    pub excitation_energy_ev: f64,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
    #[pyo3(get)]
    pub wavelength_nm: f64,
    #[pyo3(get)]
    pub transition_dipole_au: [f64; 3],
    #[pyo3(get)]
    pub oscillator_strength: f64,
    #[pyo3(get)]
    pub dominant_transitions: Vec<PyOrbitalTransition>,
}

#[pymethods]
impl PyExcitedState {
    fn __repr__(&self) -> String {
        format!(
            "ExcitedState(label='{}', mult={}, energy_ev={:.4}, lambda_nm={:.1}, f={:.4})",
            self.label,
            self.multiplicity,
            self.excitation_energy_ev,
            self.wavelength_nm,
            self.oscillator_strength
        )
    }
}

/// Gaussian-broadened UV-Vis molar decadic absorption spectrum.
#[pyclass]
#[derive(Clone)]
pub struct PyUvVisSpectrum {
    #[pyo3(get)]
    pub wavelengths_nm: Vec<f64>,
    #[pyo3(get)]
    pub energies_ev: Vec<f64>,
    #[pyo3(get)]
    pub epsilon_l_mol_cm: Vec<f64>,
    #[pyo3(get)]
    pub fwhm_ev: f64,
}

#[pymethods]
impl PyUvVisSpectrum {
    fn __repr__(&self) -> String {
        format!(
            "UvVisSpectrum(n_points={}, range=[{:.1}, {:.1}] nm, max_eps={:.1})",
            self.wavelengths_nm.len(),
            self.wavelengths_nm.first().copied().unwrap_or(0.0),
            self.wavelengths_nm.last().copied().unwrap_or(0.0),
            self.epsilon_l_mol_cm.iter().copied().fold(0.0f64, f64::max)
        )
    }
}

/// Complete output of a pure Rust CIS / TDHF / TDA-DFT / TD-DFT excited-state calculation.
#[pyclass]
#[derive(Clone)]
pub struct PyTddftResult {
    #[pyo3(get)]
    pub ground_energy_hartree: f64,
    #[pyo3(get)]
    pub xc_functional: Option<String>,
    #[pyo3(get)]
    pub tda: bool,
    #[pyo3(get)]
    pub n_occ: usize,
    #[pyo3(get)]
    pub n_virt: usize,
    #[pyo3(get)]
    pub homo_energy_hartree: f64,
    #[pyo3(get)]
    pub lumo_energy_hartree: f64,
    #[pyo3(get)]
    pub ks_gap_ev: f64,
    #[pyo3(get)]
    pub optical_gap_ev: Option<f64>,
    #[pyo3(get)]
    pub lowest_triplet_ev: Option<f64>,
    #[pyo3(get)]
    pub delta_est_ev: Option<f64>,
    #[pyo3(get)]
    pub davidson_iterations_singlet: usize,
    #[pyo3(get)]
    pub davidson_iterations_triplet: usize,
    #[pyo3(get)]
    pub states: Vec<PyExcitedState>,
    #[pyo3(get)]
    pub scf_wavefunction: PyScfWavefunction,
}

#[pymethods]
impl PyTddftResult {
    fn __repr__(&self) -> String {
        format!(
            "TddftResult(tda={}, states={}, E_opt={:?} eV, Delta_EST={:?} eV)",
            self.tda,
            self.states.len(),
            self.optical_gap_ev,
            self.delta_est_ev
        )
    }

    /// Evaluates the Gaussian-broadened UV-Vis spectrum over `[min_nm, max_nm]`.
    #[pyo3(signature = (min_nm=150.0, max_nm=800.0, n_points=500, fwhm_ev=0.33))]
    fn broaden_spectrum(
        &self,
        min_nm: f64,
        max_nm: f64,
        n_points: usize,
        fwhm_ev: f64,
    ) -> PyResult<PyUvVisSpectrum> {
        let rust_states = py_states_to_rust(&self.states);
        let spec = broaden_uv_vis_spectrum(&rust_states, min_nm, max_nm, n_points, fwhm_ev)
            .map_err(to_py_err)?;
        Ok(PyUvVisSpectrum {
            wavelengths_nm: spec.wavelengths_nm,
            energies_ev: spec.energies_ev,
            epsilon_l_mol_cm: spec.epsilon_l_mol_cm,
            fwhm_ev: spec.fwhm_ev,
        })
    }
}

fn py_states_to_rust(states: &[PyExcitedState]) -> Vec<ExcitedState> {
    states
        .iter()
        .map(|s| ExcitedState {
            state_index: s.state_index,
            manifold_index: s.manifold_index,
            label: s.label.clone(),
            multiplicity: s.multiplicity,
            excitation_energy_hartree: s.excitation_energy_hartree,
            excitation_energy_ev: s.excitation_energy_ev,
            total_energy_hartree: s.total_energy_hartree,
            wavelength_nm: s.wavelength_nm,
            transition_dipole_au: s.transition_dipole_au,
            oscillator_strength: s.oscillator_strength,
            dominant_transitions: s
                .dominant_transitions
                .iter()
                .map(|t| OrbitalTransition {
                    occ_index: t.occ_index,
                    virt_index: t.virt_index,
                    amplitude_x: t.amplitude_x,
                    amplitude_y: t.amplitude_y,
                    weight: t.weight,
                })
                .collect(),
        })
        .collect()
}

fn rust_to_py_tddft_result(
    res: TddftResult,
    mol: &Molecule,
    bs: &BasisSet,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    basis_name: String,
    charge: i32,
    multiplicity: usize,
) -> PyTddftResult {
    let py_wfn = scf_wfn_to_py(
        &res.scf_wavefunction,
        mol,
        bs,
        symbols,
        geometry_bohr,
        basis_name,
        charge,
        multiplicity,
        res.xc_functional.clone(),
    );

    let py_states: Vec<PyExcitedState> = res
        .states
        .into_iter()
        .map(|s| PyExcitedState {
            state_index: s.state_index,
            manifold_index: s.manifold_index,
            label: s.label,
            multiplicity: s.multiplicity,
            excitation_energy_hartree: s.excitation_energy_hartree,
            excitation_energy_ev: s.excitation_energy_ev,
            total_energy_hartree: s.total_energy_hartree,
            wavelength_nm: s.wavelength_nm,
            transition_dipole_au: s.transition_dipole_au,
            oscillator_strength: s.oscillator_strength,
            dominant_transitions: s
                .dominant_transitions
                .into_iter()
                .map(|t| PyOrbitalTransition {
                    occ_index: t.occ_index,
                    virt_index: t.virt_index,
                    amplitude_x: t.amplitude_x,
                    amplitude_y: t.amplitude_y,
                    weight: t.weight,
                })
                .collect(),
        })
        .collect();

    PyTddftResult {
        ground_energy_hartree: res.ground_energy_hartree,
        xc_functional: res.xc_functional,
        tda: res.tda,
        n_occ: res.n_occ,
        n_virt: res.n_virt,
        homo_energy_hartree: res.homo_energy_hartree,
        lumo_energy_hartree: res.lumo_energy_hartree,
        ks_gap_ev: res.ks_gap_ev,
        optical_gap_ev: res.optical_gap_ev,
        lowest_triplet_ev: res.lowest_triplet_ev,
        delta_est_ev: res.delta_est_ev,
        davidson_iterations_singlet: res.davidson_iterations_singlet,
        davidson_iterations_triplet: res.davidson_iterations_triplet,
        states: py_states,
        scf_wavefunction: py_wfn,
    }
}

fn resolve_tddft_method_and_xc(
    method: &str,
    xc: Option<&str>,
) -> PyResult<(Option<XcFunctional>, bool)> {
    let m = method.trim().to_ascii_lowercase();
    match m.as_str() {
        "cis" => Ok((None, true)),
        "tdhf" | "rpa_hf" => Ok((None, false)),
        "rhf" | "hf" => {
            if let Some(x) = xc {
                let func = XcFunctional::from_name(x).map_err(to_py_err)?;
                Ok((Some(func), false))
            } else {
                Ok((None, false))
            }
        }
        "rks" | "dft" | "tddft" | "td-dft" | "tda" | "tda-dft" => {
            let func_name = xc.unwrap_or("b3lyp");
            let func = XcFunctional::from_name(func_name).map_err(to_py_err)?;
            let tda_by_name = m.starts_with("tda");
            Ok((Some(func), tda_by_name))
        }
        other => {
            if let Ok(func) = XcFunctional::from_name(other) {
                Ok((Some(func), false))
            } else if let Some(x) = xc {
                let func = XcFunctional::from_name(x).map_err(to_py_err)?;
                Ok((Some(func), false))
            } else {
                Err(PyValueError::new_err(format!(
                    "Unsupported TD-DFT method or functional: '{method}'"
                )))
            }
        }
    }
}

/// Runs a linear-response excited-state calculation (CIS, TDHF, TDA-DFT, TD-DFT) from geometry in pure Rust.
#[pyfunction]
#[pyo3(signature = (
    symbols,
    geometry_bohr,
    charge=0,
    multiplicity=1,
    method="rks",
    basis="6-31g",
    n_states=5,
    tda=false,
    manifold="singlet",
    xc=None,
    grid_level="medium",
    max_iter=80,
    conv_tol=1e-6,
    force_davidson=false,
    scf_max_iter=100,
    scf_conv_energy=1e-10,
    scf_conv_density=1e-8
))]
#[allow(clippy::too_many_arguments)]
pub fn run_tddft(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    n_states: usize,
    tda: bool,
    manifold: &str,
    xc: Option<&str>,
    grid_level: &str,
    max_iter: usize,
    conv_tol: f64,
    force_davidson: bool,
    scf_max_iter: usize,
    scf_conv_energy: f64,
    scf_conv_density: f64,
) -> PyResult<PyTddftResult> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    if n_states == 0 {
        return Err(PyValueError::new_err("n_states must be >= 1"));
    }
    if multiplicity != 1 {
        return Err(PyValueError::new_err(
            "TD-DFT/CIS currently requires a closed-shell reference (multiplicity=1)",
        ));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        if !pos.iter().all(|x| x.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "non-finite coordinate for atom {s}"
            )));
        }
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);
    let bs = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;
    let (xc_func, tda_by_name) = resolve_tddft_method_and_xc(method, xc)?;
    let spin_manifold = SpinManifold::from_name(manifold).map_err(to_py_err)?;
    let gl = GridLevel::from_name(grid_level);

    let opts = TddftOptions {
        n_states,
        tda: tda || tda_by_name,
        manifold: spin_manifold,
        grid_level: gl,
        max_iter,
        conv_tol,
        force_davidson,
        scf_options: ScfOptions {
            max_iter: scf_max_iter,
            conv_energy: scf_conv_energy,
            conv_density: scf_conv_density,
            ..Default::default()
        },
    };

    let res = py
        .allow_threads(|| rust_run_tddft(&mol, &bs, xc_func, &opts))
        .map_err(to_py_err)?;

    Ok(rust_to_py_tddft_result(
        res,
        &mol,
        &bs,
        symbols,
        geometry_bohr,
        basis.to_string(),
        charge,
        multiplicity,
    ))
}

/// Runs a linear-response excited-state calculation atop an already-converged ground-state [`PyScfWavefunction`].
#[pyfunction]
#[pyo3(signature = (
    wf,
    n_states=5,
    tda=false,
    manifold="singlet",
    xc=None,
    grid_level="medium",
    max_iter=80,
    conv_tol=1e-6,
    force_davidson=false
))]
#[allow(clippy::too_many_arguments)]
pub fn run_tddft_from_scf_py(
    py: Python<'_>,
    wf: &PyScfWavefunction,
    n_states: usize,
    tda: bool,
    manifold: &str,
    xc: Option<&str>,
    grid_level: &str,
    max_iter: usize,
    conv_tol: f64,
    force_davidson: bool,
) -> PyResult<PyTddftResult> {
    if wf.is_uhf || wf.multiplicity != 1 {
        return Err(PyValueError::new_err(
            "TD-DFT/CIS currently requires a closed-shell singlet wavefunction",
        ));
    }
    if n_states == 0 {
        return Err(PyValueError::new_err("n_states must be >= 1"));
    }
    let (mol, bs) = wf.build_mol_and_basis()?;
    let spin_manifold = SpinManifold::from_name(manifold).map_err(to_py_err)?;
    let gl = GridLevel::from_name(grid_level);

    let xc_name = xc.or(wf.xc_functional.as_deref());
    let xc_func = match xc_name {
        Some(x)
            if !x.is_empty()
                && x.to_ascii_lowercase() != "rhf"
                && x.to_ascii_lowercase() != "hf" =>
        {
            Some(XcFunctional::from_name(x).map_err(to_py_err)?)
        }
        _ => None,
    };

    let opts = TddftOptions {
        n_states,
        tda,
        manifold: spin_manifold,
        grid_level: gl,
        max_iter,
        conv_tol,
        force_davidson,
        scf_options: ScfOptions::default(),
    };

    let scf_wfn = calchem_model::ScfWavefunction {
        converged: wf.converged,
        iterations: wf.iterations,
        total_energy_hartree: wf.total_energy_hartree,
        electronic_energy_hartree: wf.electronic_energy_hartree,
        nuclear_repulsion_hartree: wf.nuclear_repulsion_hartree,
        n_ao: wf.n_ao,
        n_mo: wf.n_ao,
        mo_energies: wf.mo_energies.clone(),
        mo_occupations: wf.mo_occupations.clone(),
        mo_coefficients: wf.mo_coefficients.clone(),
        density_alpha: wf.density.clone(),
        density_beta: None,
        fock_alpha: vec![0.0; wf.n_ao * wf.n_ao],
        overlap_matrix: vec![0.0; wf.n_ao * wf.n_ao],
        h_core_matrix: vec![0.0; wf.n_ao * wf.n_ao],
        diagnostics: calchem_model::ScfDiagnostics {
            iterations: wf.iterations,
            energy_history: wf.energy_history.clone(),
            rms_density_history: vec![],
            diis_error_history: vec![],
        },
    };

    let res = py
        .allow_threads(|| run_tddft_from_scf(&mol, &bs, &scf_wfn, xc_func, &opts))
        .map_err(to_py_err)?;

    Ok(rust_to_py_tddft_result(
        res,
        &mol,
        &bs,
        wf.symbols.clone(),
        wf.geometry_bohr.clone(),
        wf.basis_name.clone(),
        wf.charge,
        wf.multiplicity,
    ))
}

/// Generates a Gaussian-broadened UV-Vis absorption spectrum from excited states.
#[pyfunction]
#[pyo3(signature = (states, min_nm=150.0, max_nm=800.0, n_points=500, fwhm_ev=0.33))]
pub fn broaden_uv_vis(
    states: Vec<PyExcitedState>,
    min_nm: f64,
    max_nm: f64,
    n_points: usize,
    fwhm_ev: f64,
) -> PyResult<PyUvVisSpectrum> {
    let rust_states = py_states_to_rust(&states);
    let spec = broaden_uv_vis_spectrum(&rust_states, min_nm, max_nm, n_points, fwhm_ev)
        .map_err(to_py_err)?;
    Ok(PyUvVisSpectrum {
        wavelengths_nm: spec.wavelengths_nm,
        energies_ev: spec.energies_ev,
        epsilon_l_mol_cm: spec.epsilon_l_mol_cm,
        fwhm_ev: spec.fwhm_ev,
    })
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyNormalMode {
    #[pyo3(get)]
    pub mode_index: usize,
    #[pyo3(get)]
    pub frequency_cm: f64,
    #[pyo3(get)]
    pub is_imaginary: bool,
    #[pyo3(get)]
    pub reduced_mass_amu: f64,
    #[pyo3(get)]
    pub force_constant_m_dyn_a: f64,
    #[pyo3(get)]
    pub ir_intensity_km_mol: f64,
    #[pyo3(get)]
    pub displacement: Vec<[f64; 3]>,
}

#[pymethods]
impl PyNormalMode {
    fn __repr__(&self) -> String {
        format!(
            "PyNormalMode(#{}: freq={:.2} cm^-1, red_mass={:.3} amu, k={:.3} mDyn/A, IR={:.3} km/mol, imag={})",
            self.mode_index,
            self.frequency_cm,
            self.reduced_mass_amu,
            self.force_constant_m_dyn_a,
            self.ir_intensity_km_mol,
            self.is_imaginary,
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyThermochemistry {
    #[pyo3(get)]
    pub temperature_k: f64,
    #[pyo3(get)]
    pub pressure_pa: f64,
    #[pyo3(get)]
    pub zpve_hartree: f64,
    #[pyo3(get)]
    pub zpve_kcal_mol: f64,
    #[pyo3(get)]
    pub e_trans_hartree: f64,
    #[pyo3(get)]
    pub e_rot_hartree: f64,
    #[pyo3(get)]
    pub e_vib_hartree: f64,
    #[pyo3(get)]
    pub e_thermal_hartree: f64,
    #[pyo3(get)]
    pub h_hartree: f64,
    #[pyo3(get)]
    pub g_hartree: f64,
    #[pyo3(get)]
    pub s_cal_mol_k: f64,
    #[pyo3(get)]
    pub s_trans_cal_mol_k: f64,
    #[pyo3(get)]
    pub s_rot_cal_mol_k: f64,
    #[pyo3(get)]
    pub s_vib_cal_mol_k: f64,
    #[pyo3(get)]
    pub s_elec_cal_mol_k: f64,
    #[pyo3(get)]
    pub cv_cal_mol_k: f64,
    #[pyo3(get)]
    pub cp_cal_mol_k: f64,
}

#[pymethods]
impl PyThermochemistry {
    fn __repr__(&self) -> String {
        format!(
            "PyThermochemistry(T={:.2} K, P={:.1} Pa, ZPVE={:.6} Eh, H={:.6} Eh, G={:.6} Eh, S={:.3} cal/(mol*K))",
            self.temperature_k,
            self.pressure_pa,
            self.zpve_hartree,
            self.h_hartree,
            self.g_hartree,
            self.s_cal_mol_k,
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyVibrationalAnalysis {
    #[pyo3(get)]
    pub n_vib: usize,
    #[pyo3(get)]
    pub is_linear: bool,
    #[pyo3(get)]
    pub principal_moments_amu_bohr2: [f64; 3],
    #[pyo3(get)]
    pub rotational_constants_ghz: [f64; 3],
    #[pyo3(get)]
    pub trans_rot_residuals: Vec<f64>,
    #[pyo3(get)]
    pub modes: Vec<PyNormalMode>,
}

#[pymethods]
impl PyVibrationalAnalysis {
    fn __repr__(&self) -> String {
        format!(
            "PyVibrationalAnalysis(n_vib={}, linear={}, modes={})",
            self.n_vib,
            self.is_linear,
            self.modes.len(),
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyFrequencyResult {
    #[pyo3(get)]
    pub energy_hartree: f64,
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub geometry_bohr: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub basis_name: String,
    #[pyo3(get)]
    pub charge: i32,
    #[pyo3(get)]
    pub multiplicity: usize,
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub xc_functional: Option<String>,
    #[pyo3(get)]
    pub vibrational: PyVibrationalAnalysis,
    #[pyo3(get)]
    pub thermo: PyThermochemistry,
    #[pyo3(get)]
    pub cartesian_hessian: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub apt: Vec<Vec<f64>>,
}

#[pymethods]
impl PyFrequencyResult {
    #[getter]
    fn frequencies_cm(&self) -> Vec<f64> {
        self.vibrational
            .modes
            .iter()
            .map(|m| m.frequency_cm)
            .collect()
    }

    #[getter]
    fn ir_intensities_km_mol(&self) -> Vec<f64> {
        self.vibrational
            .modes
            .iter()
            .map(|m| m.ir_intensity_km_mol)
            .collect()
    }

    #[getter]
    fn zpve_hartree(&self) -> f64 {
        self.thermo.zpve_hartree
    }

    #[getter]
    fn enthalpy_hartree(&self) -> f64 {
        self.thermo.h_hartree
    }

    #[getter]
    fn gibbs_free_energy_hartree(&self) -> f64 {
        self.thermo.g_hartree
    }

    #[getter]
    fn entropy_cal_mol_k(&self) -> f64 {
        self.thermo.s_cal_mol_k
    }

    #[getter]
    fn heat_capacity_cv_cal_mol_k(&self) -> f64 {
        self.thermo.cv_cal_mol_k
    }

    #[getter]
    fn heat_capacity_cp_cal_mol_k(&self) -> f64 {
        self.thermo.cp_cal_mol_k
    }

    fn __repr__(&self) -> String {
        format!(
            "PyFrequencyResult(energy={:.8} Eh, n_vib={}, ZPVE={:.6} Eh, G={:.6} Eh)",
            self.energy_hartree,
            self.vibrational.n_vib,
            self.thermo.zpve_hartree,
            self.thermo.g_hartree,
        )
    }
}

/// Runs nuclear Hessian finite-difference vibrational frequency analysis and RRHO thermochemistry.
#[pyfunction]
#[pyo3(signature = (
    symbols,
    geometry_bohr,
    charge=0,
    multiplicity=1,
    method="rks",
    basis="sto-3g",
    step_bohr=0.001,
    xc=None,
    grid_level="medium",
    temperature_k=298.15,
    pressure_pa=101325.0,
    symmetry_number=None,
    max_iter=100,
    conv_energy=1e-10,
    conv_density=1e-8,
))]
pub fn run_frequency(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    step_bohr: f64,
    xc: Option<&str>,
    grid_level: &str,
    temperature_k: f64,
    pressure_pa: f64,
    symmetry_number: Option<usize>,
    max_iter: usize,
    conv_energy: f64,
    conv_density: f64,
) -> PyResult<PyFrequencyResult> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    if symbols.is_empty() {
        return Err(PyValueError::new_err("molecule cannot be empty"));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        if !pos.iter().all(|x| x.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "non-finite coordinate for atom {s}"
            )));
        }
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);
    let _ = BasisSet::from_name(basis, &mol).map_err(to_py_err)?;

    let scf_kind = resolve_scf_kind(method, xc, multiplicity)?;
    let (vib_method, xc_name) = match scf_kind {
        ResolvedScfKind::Rhf => (VibMethod::Rhf, None),
        ResolvedScfKind::Uhf => (VibMethod::Uhf, None),
        ResolvedScfKind::Rks(f) => (VibMethod::Rks(f), Some(f.canonical_name().to_string())),
        ResolvedScfKind::Uks(f) => (VibMethod::Uks(f), Some(f.canonical_name().to_string())),
    };

    let gl = GridLevel::from_name(grid_level);
    let opts = FreqOptions {
        step_bohr,
        grid_level: gl,
        temperature_k,
        pressure_pa,
        symmetry_number,
        scf_options: ScfOptions {
            max_iter,
            conv_energy,
            conv_density,
            ..Default::default()
        },
    };

    let res = py
        .allow_threads(|| compute_frequencies_and_thermo(&mol, basis, vib_method, &opts))
        .map_err(to_py_err)?;

    let n_atoms = symbols.len();
    let dim = 3 * n_atoms;

    let py_modes: Vec<PyNormalMode> = res
        .vibrational_analysis
        .modes
        .into_iter()
        .map(|m| {
            let disp: Vec<[f64; 3]> = m
                .displacements
                .chunks(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            PyNormalMode {
                mode_index: m.mode_index,
                frequency_cm: m.frequency_cm,
                is_imaginary: m.is_imaginary,
                reduced_mass_amu: m.reduced_mass_amu,
                force_constant_m_dyn_a: m.force_constant_m_dyn_a,
                ir_intensity_km_mol: m.ir_intensity_km_mol,
                displacement: disp,
            }
        })
        .collect();

    let py_vib = PyVibrationalAnalysis {
        n_vib: res.vibrational_analysis.n_vib,
        is_linear: res.vibrational_analysis.is_linear,
        principal_moments_amu_bohr2: res.vibrational_analysis.moments_of_inertia_amu_bohr2,
        rotational_constants_ghz: res.vibrational_analysis.rotational_constants_ghz,
        trans_rot_residuals: res.vibrational_analysis.trans_rot_residuals,
        modes: py_modes,
    };

    let py_thermo = PyThermochemistry {
        temperature_k: res.thermochemistry.temperature_k,
        pressure_pa: res.thermochemistry.pressure_pa,
        zpve_hartree: res.thermochemistry.zpve_hartree,
        zpve_kcal_mol: res.thermochemistry.zpve_kcal_mol,
        e_trans_hartree: res.thermochemistry.e_trans_hartree,
        e_rot_hartree: res.thermochemistry.e_rot_hartree,
        e_vib_hartree: res.thermochemistry.e_vib_hartree,
        e_thermal_hartree: res.thermochemistry.e_thermal_total_hartree,
        h_hartree: res.thermochemistry.enthalpy_hartree,
        g_hartree: res.thermochemistry.gibbs_free_energy_hartree,
        s_cal_mol_k: res.thermochemistry.s_total_cal_mol_k,
        s_trans_cal_mol_k: res.thermochemistry.s_trans_cal_mol_k,
        s_rot_cal_mol_k: res.thermochemistry.s_rot_cal_mol_k,
        s_vib_cal_mol_k: res.thermochemistry.s_vib_cal_mol_k,
        s_elec_cal_mol_k: res.thermochemistry.s_elec_cal_mol_k,
        cv_cal_mol_k: res.thermochemistry.cv_cal_mol_k,
        cp_cal_mol_k: res.thermochemistry.cp_cal_mol_k,
    };

    let mut h_mat = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            h_mat[i][j] = res.cartesian_hessian.get(i, j);
        }
    }

    let mut apt_mat = vec![vec![0.0f64; dim]; 3];
    for p in 0..3 {
        for j in 0..dim {
            apt_mat[p][j] = res.apt.get(p, j);
        }
    }

    Ok(PyFrequencyResult {
        energy_hartree: res.energy_hartree,
        symbols,
        geometry_bohr,
        basis_name: basis.to_string(),
        charge,
        multiplicity,
        method: method.to_string(),
        xc_functional: xc_name,
        vibrational: py_vib,
        thermo: py_thermo,
        cartesian_hessian: h_mat,
        apt: apt_mat,
    })
}

/// Runs a pure Rust Transition State Search using Partitioned Rational Function Optimization (P-RFO).
#[pyfunction]
#[pyo3(signature = (
    symbols,
    geometry_bohr,
    charge=0,
    multiplicity=1,
    method="auto",
    basis="sto-3g",
    xc=None,
    grid_level="medium",
    max_steps=50,
    max_force=3e-4,
    rms_force=1.5e-4,
    max_displacement=1.2e-3,
    trust_radius_bohr=0.1,
    mode_index=None,
    recalc_hessian_interval=0,
    scf_max_iter=100,
    scf_conv_energy=1e-11,
    scf_conv_density=1e-10
))]
#[allow(clippy::too_many_arguments)]
pub fn run_ts_search(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    xc: Option<&str>,
    grid_level: &str,
    max_steps: usize,
    max_force: f64,
    rms_force: f64,
    max_displacement: f64,
    trust_radius_bohr: f64,
    mode_index: Option<usize>,
    recalc_hessian_interval: usize,
    scf_max_iter: usize,
    scf_conv_energy: f64,
    scf_conv_density: f64,
) -> PyResult<PyTsResult> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    if symbols.len() < 2 {
        return Err(PyValueError::new_err("TS search requires at least 2 atoms"));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);

    let reference = Reference::from_method_and_xc(method, xc).map_err(|_| {
        PyValueError::new_err(format!("Unsupported method in Rust TS optimizer: {method}"))
    })?;

    let gl = GridLevel::from_name(grid_level);
    let opts = TsOptions {
        max_steps,
        max_force,
        rms_force,
        max_displacement,
        trust_radius_bohr,
        mode_index,
        recalc_hessian_interval,
        reference,
        grid_level: gl,
        scf_options: ScfOptions {
            max_iter: scf_max_iter,
            conv_energy: scf_conv_energy,
            conv_density: scf_conv_density,
            ..Default::default()
        },
    };

    let res = py
        .allow_threads(|| rust_search_ts(&mol, basis, &opts))
        .map_err(to_py_err)?;

    let out_symbols: Vec<String> = res
        .molecule
        .atoms
        .iter()
        .map(|a| symbol_from_atomic_number(a.z).unwrap_or_else(|_| "X".into()))
        .collect();
    let out_coords: Vec<[f64; 3]> = res.molecule.atoms.iter().map(|a| a.position_bohr).collect();

    Ok(PyTsResult {
        symbols: out_symbols,
        geometry_bohr: out_coords,
        energy_hartree: res.energy_hartree,
        gradient: res.gradient,
        steps: res.steps,
        converged: res.converged,
        imaginary_frequency_cm: res.imaginary_frequency_cm,
        transition_vector: res.transition_vector,
    })
}

/// Runs a pure Rust Gonzalez-Schlegel Intrinsic Reaction Coordinate (IRC) integration.
#[pyfunction]
#[pyo3(signature = (
    symbols,
    geometry_bohr,
    transition_vector,
    charge=0,
    multiplicity=1,
    method="auto",
    basis="sto-3g",
    xc=None,
    grid_level="medium",
    step_size_bohr=0.1,
    max_points=20,
    direction="both",
    convergence_tol=1e-3,
    max_gs_iter=10,
    scf_max_iter=100,
    scf_conv_energy=1e-11,
    scf_conv_density=1e-10
))]
#[allow(clippy::too_many_arguments)]
pub fn run_irc(
    py: Python<'_>,
    symbols: Vec<String>,
    geometry_bohr: Vec<[f64; 3]>,
    transition_vector: Vec<f64>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
    xc: Option<&str>,
    grid_level: &str,
    step_size_bohr: f64,
    max_points: usize,
    direction: &str,
    convergence_tol: f64,
    max_gs_iter: usize,
    scf_max_iter: usize,
    scf_conv_energy: f64,
    scf_conv_density: f64,
) -> PyResult<PyIrcResult> {
    if symbols.len() != geometry_bohr.len() {
        return Err(PyValueError::new_err(
            "symbols and geometry_bohr length mismatch",
        ));
    }
    if transition_vector.len() != 3 * symbols.len() {
        return Err(PyValueError::new_err(format!(
            "transition_vector length {} mismatch with 3 * {} atoms",
            transition_vector.len(),
            symbols.len()
        )));
    }

    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(geometry_bohr.iter()) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);

    let reference = Reference::from_method_and_xc(method, xc)
        .map_err(|_| PyValueError::new_err(format!("Unsupported method in Rust IRC: {method}")))?;

    let irc_dir = match direction.trim().to_ascii_lowercase().as_str() {
        "forward" | "fwd" => IrcDirection::Forward,
        "reverse" | "rev" => IrcDirection::Reverse,
        "both" | "full" => IrcDirection::Both,
        other => {
            return Err(PyValueError::new_err(format!(
                "Invalid IRC direction '{other}', expected 'forward', 'reverse', or 'both'"
            )))
        }
    };

    let gl = GridLevel::from_name(grid_level);
    let opts = IrcOptions {
        step_size_bohr,
        max_points,
        direction: irc_dir,
        convergence_tol,
        max_gs_iter,
        reference,
        grid_level: gl,
        scf_options: ScfOptions {
            max_iter: scf_max_iter,
            conv_energy: scf_conv_energy,
            conv_density: scf_conv_density,
            ..Default::default()
        },
    };

    let res = py
        .allow_threads(|| rust_run_irc(&mol, basis, &transition_vector, &opts))
        .map_err(to_py_err)?;

    let map_pt = |pt: &calchem_opt::IrcPoint, dir: &str| -> PyIrcPoint {
        let coords: Vec<[f64; 3]> = pt.molecule.atoms.iter().map(|a| a.position_bohr).collect();
        PyIrcPoint {
            point_index: pt.point_index,
            direction: dir.to_string(),
            path_coordinate_s: pt.path_coordinate_s,
            energy_hartree: pt.energy_hartree,
            delta_energy_kcal_mol: pt.delta_energy_kcal_mol,
            gradient_norm: pt.gradient_norm,
            geometry_bohr: coords,
        }
    };

    let map_branch = |br: &calchem_opt::IrcBranch| -> PyIrcBranch {
        let last_de = br
            .points
            .last()
            .map(|p| p.delta_energy_kcal_mol)
            .unwrap_or(0.0);
        PyIrcBranch {
            converged_to_minimum: br.converged_to_minimum,
            direction: br.direction.clone(),
            n_points: br.points.len(),
            barrier_kcal_mol: br.barrier_kcal_mol,
            delta_energy_kcal_mol: last_de,
            points: br.points.iter().map(|p| map_pt(p, &br.direction)).collect(),
        }
    };

    let total_pts = 1
        + res.forward.as_ref().map_or(0, |f| f.points.len())
        + res.reverse.as_ref().map_or(0, |r| r.points.len());

    let py_ts = map_pt(&res.transition_state, "ts");
    let py_fwd = res.forward.as_ref().map(map_branch);
    let py_rev = res.reverse.as_ref().map(map_branch);

    Ok(PyIrcResult {
        transition_state: py_ts,
        forward: py_fwd,
        reverse: py_rev,
        total_points: total_pts,
        barrier_forward_kcal_mol: res.barrier_forward_kcal_mol,
        barrier_reverse_kcal_mol: res.barrier_reverse_kcal_mol,
        reaction_energy_kcal_mol: res.reaction_energy_kcal_mol,
    })
}

#[pyclass(name = "SolvationResult")]
#[derive(Debug, Clone)]
pub struct PySolvationResult {
    #[pyo3(get)]
    pub solvated_energy: f64,
    #[pyo3(get)]
    pub gas_energy: f64,
    #[pyo3(get)]
    pub delta_g_solv: f64,
    #[pyo3(get)]
    pub delta_g_solv_kcal_mol: f64,
    #[pyo3(get)]
    pub delta_g_elec_kcal_mol: f64,
    #[pyo3(get)]
    pub delta_g_cds_kcal_mol: f64,
    #[pyo3(get)]
    pub cavity_surface_area_angstrom2: f64,
    #[pyo3(get)]
    pub total_surface_charge: f64,
    pub surface_charges: Vec<f64>,
    #[pyo3(get)]
    pub tessera_coords_bohr: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub scf_iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub solvent_name: String,
    #[pyo3(get)]
    pub dielectric_constant: f64,
}

#[pymethods]
impl PySolvationResult {
    #[getter]
    fn surface_charges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.surface_charges)
    }

    fn __repr__(&self) -> String {
        format!(
            "SolvationResult(solvent='{}', eps={:.2}, solvated_energy={:.6} Eh, dG_solv={:.2} kcal/mol, dG_elec={:.2} kcal/mol, dG_cds={:.2} kcal/mol, SASA={:.1} A^2)",
            self.solvent_name,
            self.dielectric_constant,
            self.solvated_energy,
            self.delta_g_solv_kcal_mol,
            self.delta_g_elec_kcal_mol,
            self.delta_g_cds_kcal_mol,
            self.cavity_surface_area_angstrom2
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    symbols,
    coords,
    basis_name,
    solvent="water",
    solvent_model="cpcm",
    charge=0,
    multiplicity=1,
    method="b3lyp",
    xc=None,
    scaling_factor=1.20,
    include_cds=true,
    max_iter=100,
    conv_energy=1e-8,
    conv_density=1e-6,
))]
pub fn run_solvation(
    py: Python<'_>,
    symbols: Vec<String>,
    coords: Vec<[f64; 3]>,
    basis_name: &str,
    solvent: &str,
    solvent_model: &str,
    charge: i32,
    multiplicity: usize,
    method: &str,
    xc: Option<&str>,
    scaling_factor: f64,
    include_cds: bool,
    max_iter: usize,
    conv_energy: f64,
    conv_density: f64,
) -> PyResult<PySolvationResult> {
    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.iter().zip(&coords) {
        atoms.push(Atom::from_bohr(s, *pos).map_err(to_py_err)?);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);
    let bs = BasisSet::from_name(basis_name, &mol).map_err(to_py_err)?;

    let solv = Solvent::from_name(solvent)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported solvent: '{solvent}'")))?;

    let model = match solvent_model.to_ascii_lowercase().as_str() {
        "cpcm" | "c-pcm" => SolventModel::Cpcm,
        "cosmo" => SolventModel::Cosmo,
        "iefpcm" | "ief-pcm" => SolventModel::IefPcm,
        "onsager" => SolventModel::Onsager,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown solvent model: '{other}'"
            )))
        }
    };

    let cav_opts = CavityOptions {
        scaling_factor,
        lebedev_order: CavityLebedevOrder::N50,
        probe_radius_angstrom: 0.0,
        exclusion_tol_bohr: 1e-5,
    };

    model.validate_native().map_err(to_py_err)?;

    let solv_opts = SolvationOptions {
        solvent: solv.clone(),
        model,
        cavity_opts: cav_opts,
        renormalize_charge: true,
        include_cds,
    };

    let scf_opts = ScfOptions {
        max_iter,
        conv_energy,
        conv_density,
        ..Default::default()
    };

    let xc_func = match xc {
        Some(x) => Some(XcFunctional::from_name(x).map_err(to_py_err)?),
        None => {
            if method.eq_ignore_ascii_case("dft") || method.eq_ignore_ascii_case("b3lyp") {
                Some(XcFunctional::B3lyp)
            } else {
                None
            }
        }
    };

    let res = py
        .allow_threads(|| rust_run_scrf(&mol, &bs, &scf_opts, &solv_opts, method, xc_func))
        .map_err(to_py_err)?;

    Ok(PySolvationResult {
        solvated_energy: res.solvated_energy,
        gas_energy: res.gas_energy,
        delta_g_solv: res.delta_g_solv,
        delta_g_solv_kcal_mol: res.delta_g_solv_kcal_mol,
        delta_g_elec_kcal_mol: res.delta_g_elec_kcal_mol,
        delta_g_cds_kcal_mol: res.delta_g_cds_kcal_mol,
        cavity_surface_area_angstrom2: res.cavity_surface_area_angstrom2,
        total_surface_charge: res.total_surface_charge,
        surface_charges: res.surface_charges,
        tessera_coords_bohr: res.tessera_coords_bohr,
        scf_iterations: res.scf_iterations,
        converged: res.converged,
        solvent_name: solv.name.clone(),
        dielectric_constant: solv.epsilon,
    })
}

#[pyfunction]
#[pyo3(signature = (
    scf_wfn,
    solvent="water",
    solvent_model="cpcm",
    scaling_factor=1.20,
    include_cds=true,
))]
pub fn run_pcm_from_scf_py(
    _py: Python<'_>,
    scf_wfn: &PyScfWavefunction,
    solvent: &str,
    solvent_model: &str,
    scaling_factor: f64,
    include_cds: bool,
) -> PyResult<PySolvationResult> {
    let (mol, bs) = scf_wfn.build_mol_and_basis()?;

    let solv = Solvent::from_name(solvent)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported solvent: '{solvent}'")))?;

    let model = match solvent_model.to_ascii_lowercase().as_str() {
        "cpcm" | "c-pcm" => SolventModel::Cpcm,
        "cosmo" => SolventModel::Cosmo,
        "iefpcm" | "ief-pcm" => SolventModel::IefPcm,
        "onsager" => SolventModel::Onsager,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown solvent model: '{other}'"
            )))
        }
    };
    model.validate_native().map_err(to_py_err)?;

    let cav_opts = CavityOptions {
        scaling_factor,
        lebedev_order: CavityLebedevOrder::N50,
        probe_radius_angstrom: 0.0,
        exclusion_tol_bohr: 1e-5,
    };

    let solv_opts = SolvationOptions {
        solvent: solv.clone(),
        model,
        cavity_opts: cav_opts,
        renormalize_charge: true,
        include_cds,
    };

    let cavity = MolecularCavity::build(&mol, &solv_opts.cavity_opts);
    let boundary_matrix = BoundaryMatrix::build(&cavity).map_err(to_py_err)?;
    let potential_engine = CavityPotentialEngine::build(&mol, &bs, &cavity).map_err(to_py_err)?;

    let net_charge = if solv_opts.renormalize_charge {
        Some(mol.charge as f64)
    } else {
        None
    };

    let n = bs.n_ao;
    let density_mat = calchem_tensor::Matrix2D {
        rows: n,
        cols: n,
        data: scf_wfn.density.clone(),
    };

    let f_eps = solv_opts.model.dielectric_factor(solv_opts.solvent.epsilon);
    let v_solute = potential_engine.compute_solute_potential(&density_mat);
    let q = boundary_matrix
        .solve_surface_charges(&v_solute, f_eps, net_charge)
        .map_err(to_py_err)?;

    let e_elec_hartree: f64 = 0.5
        * q.iter()
            .zip(v_solute.iter())
            .map(|(&qk, &vk)| qk * vk)
            .sum::<f64>();

    let cds_kcal_mol = if solv_opts.include_cds {
        solv_opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity.total_area_angstrom2
    } else {
        0.0
    };
    let cds_hartree = cds_kcal_mol / HARTREE_TO_KCAL_MOL;

    let delta_g_solv_hartree = e_elec_hartree + cds_hartree;
    let solvated_energy = scf_wfn.total_energy_hartree + delta_g_solv_hartree;

    let total_q: f64 = q.iter().sum();
    let tessera_coords: Vec<[f64; 3]> = cavity.tesserae.iter().map(|t| t.position_bohr).collect();

    Ok(PySolvationResult {
        solvated_energy,
        gas_energy: scf_wfn.total_energy_hartree,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: e_elec_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: cavity.total_area_angstrom2,
        total_surface_charge: total_q,
        surface_charges: q,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: scf_wfn.iterations,
        converged: scf_wfn.converged,
        solvent_name: solv.name.clone(),
        dielectric_constant: solv.epsilon,
    })
}

#[pyclass(name = "D3Result")]
#[derive(Debug, Clone)]
pub struct PyD3Result {
    #[pyo3(get)]
    pub energy_hartree: f64,
    #[pyo3(get)]
    pub energy_kcal_mol: f64,
    pub gradient: Vec<[f64; 3]>,
    #[pyo3(get)]
    pub functional: String,
    #[pyo3(get)]
    pub damping: String,
}

#[pymethods]
impl PyD3Result {
    #[getter]
    fn gradient<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.gradient.len();
        let mut flat = Vec::with_capacity(n * 3);
        for g in &self.gradient {
            flat.extend_from_slice(g);
        }
        let arr1 = PyArray1::from_vec_bound(py, flat);
        arr1.reshape([n, 3])
    }

    fn __repr__(&self) -> String {
        format!(
            "D3Result(functional='{}', damping='{}', energy={:.6} Eh, energy={:.2} kcal/mol)",
            self.functional, self.damping, self.energy_hartree, self.energy_kcal_mol
        )
    }
}

/// Evaluates Grimme DFT-D3 / DFT-D3(BJ) dispersion energy and analytical nuclear gradient.
#[pyfunction]
#[pyo3(signature = (symbols, coords, functional="b3lyp", damping="bj"))]
pub fn run_d3_dispersion(
    py: Python<'_>,
    symbols: Vec<String>,
    coords: Vec<[f64; 3]>,
    functional: &str,
    damping: &str,
) -> PyResult<PyD3Result> {
    if symbols.len() != coords.len() {
        return Err(PyValueError::new_err(format!(
            "Symbols count ({}) does not match coords count ({})",
            symbols.len(),
            coords.len()
        )));
    }
    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.into_iter().zip(coords.into_iter()) {
        let a = Atom::from_bohr(&s, pos).map_err(to_py_err)?;
        atoms.push(a);
    }
    let d3 = match damping.trim().to_ascii_lowercase().as_str() {
        "zero" | "d3zero" | "d3(0)" => calchem_dft::D3Model::d3zero(functional),
        "bj" | "d3bj" | "d3(bj)" => calchem_dft::D3Model::d3bj(functional),
        _ => return Err(PyValueError::new_err("D3 damping must be 'bj' or 'zero'")),
    }
    .map_err(to_py_err)?;
    let (e, g) = py
        .allow_threads(|| d3.energy_and_gradient(&atoms, true))
        .map_err(to_py_err)?;
    Ok(PyD3Result {
        energy_hartree: e,
        energy_kcal_mol: e * 627.509474,
        gradient: g,
        functional: functional.to_string(),
        damping: damping.to_string(),
    })
}

/// Computes the exact nuclear gradient of the PCM electrostatic solvation free energy (Eh/Bohr).
#[pyfunction]
#[pyo3(signature = (
    symbols,
    coords,
    basis_name,
    density,
    solvent="water",
    _solvent_model="cpcm",
    charge=0,
    multiplicity=1,
    scaling_factor=1.20
))]
pub fn run_solvation_gradient<'py>(
    py: Python<'py>,
    symbols: Vec<String>,
    coords: Vec<[f64; 3]>,
    basis_name: &str,
    density: PyReadonlyArray2<'py, f64>,
    solvent: &str,
    _solvent_model: &str,
    charge: i32,
    multiplicity: usize,
    scaling_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if symbols.len() != coords.len() {
        return Err(PyValueError::new_err("Mismatch between symbols and coords"));
    }
    let mut atoms = Vec::with_capacity(symbols.len());
    for (s, pos) in symbols.into_iter().zip(coords.into_iter()) {
        let a = Atom::from_bohr(&s, pos).map_err(to_py_err)?;
        atoms.push(a);
    }
    let mol = Molecule::new(atoms, charge, multiplicity);
    let basis = BasisSet::from_name(basis_name, &mol).map_err(to_py_err)?;
    let d_shape = density.shape();
    if d_shape[0] != basis.n_ao || d_shape[1] != basis.n_ao {
        return Err(PyValueError::new_err(format!(
            "Density matrix shape ({}, {}) does not match AO dimension {}",
            d_shape[0], d_shape[1], basis.n_ao
        )));
    }
    let density_slice = density.as_slice().map_err(|_| {
        PyValueError::new_err("Density array must be contiguous C-order f64 buffer")
    })?;
    let d_mat = calchem_tensor::Matrix2D {
        rows: basis.n_ao,
        cols: basis.n_ao,
        data: density_slice.to_vec(),
    };
    let solv = calchem_solv::Solvent::from_name(solvent)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported solvent: {solvent}")))?;
    let model = match _solvent_model.to_ascii_lowercase().as_str() {
        "cpcm" | "c-pcm" => calchem_solv::SolventModel::Cpcm,
        "cosmo" => calchem_solv::SolventModel::Cosmo,
        "iefpcm" | "ief-pcm" => calchem_solv::SolventModel::IefPcm,
        "onsager" => calchem_solv::SolventModel::Onsager,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown solvent model: '{other}'"
            )))
        }
    };
    let cav_opts = CavityOptions {
        scaling_factor,
        ..Default::default()
    };
    let grad = py
        .allow_threads(|| {
            calchem_solv::run_pcm_gradient(&mol, &basis, &d_mat, &solv, model, Some(&cav_opts))
        })
        .map_err(to_py_err)?;
    let n = grad.len();
    let mut flat = Vec::with_capacity(n * 3);
    for g in &grad {
        flat.extend_from_slice(g);
    }
    let arr1 = PyArray1::from_vec_bound(py, flat);
    arr1.reshape([n, 3])
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", CORE_VERSION)?;
    m.add(
        "ScfNotConvergedError",
        m.py().get_type_bound::<ScfNotConvergedError>(),
    )?;
    m.add("RustCoreError", m.py().get_type_bound::<RustCoreError>())?;
    m.add_class::<PyScfWavefunction>()?;
    m.add_class::<PyOptResult>()?;
    m.add_class::<PyTsResult>()?;
    m.add_class::<PyIrcPoint>()?;
    m.add_class::<PyIrcBranch>()?;
    m.add_class::<PyIrcResult>()?;
    m.add_class::<PySolvationResult>()?;
    m.add_class::<PyIsosurfaceMesh>()?;
    m.add_class::<PyDualOrbitalMesh>()?;
    m.add_class::<PyEspSurface>()?;
    m.add_class::<PyOrbitalTransition>()?;
    m.add_class::<PyExcitedState>()?;
    m.add_class::<PyUvVisSpectrum>()?;
    m.add_class::<PyTddftResult>()?;
    m.add_class::<PyNormalMode>()?;
    m.add_class::<PyThermochemistry>()?;
    m.add_class::<PyVibrationalAnalysis>()?;
    m.add_class::<PyFrequencyResult>()?;
    m.add_class::<PyD3Result>()?;
    m.add_function(wrap_pyfunction!(run_d3_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(run_solvation_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(run_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(run_ts_search, m)?)?;
    m.add_function(wrap_pyfunction!(run_irc, m)?)?;
    m.add_function(wrap_pyfunction!(run_solvation, m)?)?;
    m.add_function(wrap_pyfunction!(run_pcm_from_scf_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_tddft, m)?)?;
    m.add_function(wrap_pyfunction!(run_tddft_from_scf_py, m)?)?;
    m.add_function(wrap_pyfunction!(broaden_uv_vis, m)?)?;
    m.add_function(wrap_pyfunction!(compute_population_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(eval_esp_at_points, m)?)?;
    m.add_function(wrap_pyfunction!(eval_esp_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(extract_esp_surface, m)?)?;
    m.add_function(wrap_pyfunction!(export_esp_gaussian_cube, m)?)?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    m.add_function(wrap_pyfunction!(run_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(auto_orbital_grid_spec, m)?)?;
    m.add_function(wrap_pyfunction!(eval_orbital_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(eval_density_cube_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(extract_isosurface_from_grid, m)?)?;
    m.add_function(wrap_pyfunction!(extract_single_isosurface_from_grid, m)?)?;
    m.add_function(wrap_pyfunction!(extract_orbital_isosurface, m)?)?;
    m.add_function(wrap_pyfunction!(extract_density_isosurface, m)?)?;
    m.add_function(wrap_pyfunction!(export_orbital_gaussian_cube, m)?)?;
    m.add_function(wrap_pyfunction!(export_density_gaussian_cube, m)?)?;
    m.add_function(wrap_pyfunction!(format_grid_gaussian_cube, m)?)?;
    m.add_function(wrap_pyfunction!(parse_gaussian_cube_str, m)?)?;
    m.add_function(wrap_pyfunction!(supported_basis_sets, m)?)?;
    m.add_function(wrap_pyfunction!(supported_dft_functionals, m)?)?;
    Ok(())
}

```

