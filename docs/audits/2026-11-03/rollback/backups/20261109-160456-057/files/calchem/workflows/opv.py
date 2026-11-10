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


# Authoritative Gaussian 16 reference ground-truth dataset from E:\YDW\ChemDatas
CHEMDATAS_REFERENCE_DATA: Dict[str, dict] = {
    "y6_ic_ff": {
        "source_file": r"E:\YDW\ChemDatas\ydw01\IC\ydw-IC_FF.log",
        "fchk_file": r"E:\YDW\ChemDatas\ydw01\IC\ydw-IC_FF.fchk",
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
        "source_file": r"E:\YDW\ChemDatas\ydw01\BTP\ydw-BTP.log",
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
        "source_file": r"E:\YDW\ChemDatas\ydw01\BTP-series\ydw-BTP-4F.log",
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
        "source_file": r"E:\YDW\ChemDatas\Gjf\ydw-PM6-opt01.gjf",
        "chemical_name": "PM6 (PBDB-T-2F) full donor repeat monomer (BDT-2F + 2 Thiophene + BDD + alkyl chains)",
        "formula": "C68H78F2O2S8",
        "n_atoms": 158,
    },
    "pm6_l8bo_cluster": {
        "source_file": r"E:\YDW\ChemDatas\TH\00cluster_PM6-L8BO-2ac_rank1_donor-BDD1_Acceptor2-NC1.log",
        "gjf_file": r"E:\YDW\ChemDatas\TH\00cluster_PM6-L8BO-2ac_rank1_donor-BDD1_Acceptor2-NC1(2).gjf",
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
        mol = load_chemdatas_molecule(ref["source_file"], name="Y6_BTP_Core")
        mol.reference_data = {**ref, **(mol.reference_data or {})}
        return mol

    if key in ("y6_full", "btp_4f", "y6"):
        ref = dict(CHEMDATAS_REFERENCE_DATA["y6_full"])
        mol = load_chemdatas_molecule(ref["source_file"], name="Y6_BTP-4F_Full")
        mol.reference_data = {**ref, **(mol.reference_data or {})}
        return mol

    if key in ("pm6_monomer", "pm6"):
        ref = dict(CHEMDATAS_REFERENCE_DATA["pm6_monomer"])
        mol = load_chemdatas_molecule(ref["source_file"], name="PM6_Monomer_Full")
        mol.reference_data = {**ref, **(mol.reference_data or {})}
        return mol

    if key in ("pm6_l8bo_cluster", "pm6_y6_md_cluster", "da_cluster"):
        ref = dict(CHEMDATAS_REFERENCE_DATA["pm6_l8bo_cluster"])
        log_p = Path(ref["source_file"])
        gjf_p = Path(ref["gjf_file"])
        target = log_p if log_p.is_file() else gjf_p
        mol = load_chemdatas_molecule(target, name="PM6_L8BO_MD_Cluster")
        mol.reference_data = {**ref, **(mol.reference_data or {})}
        return mol

    if key in ("donor_bdd1", "pm6_bdd1"):
        ref_cluster = CHEMDATAS_REFERENCE_DATA["pm6_l8bo_cluster"]
        gjf_p = Path(ref_cluster["gjf_file"])
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
        gjf_p = Path(ref_cluster["gjf_file"])
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

    raise ValueError(
        f"Unknown OPV system '{system_name}'. Supported: "
        "'y6_ic_ff', 'y6_btp_core', 'y6_full', 'pm6_monomer', 'pm6_l8bo_cluster', 'donor_bdd1', 'acceptor_nc1'."
    )


def get_pm6_fragment(
    fluorinated: bool = True,
    variant: Literal["donor_bdd1", "monomer"] = "donor_bdd1",
) -> OPVMolecule:
    """Returns an authentic PM6 (PBDB-T-2F) donor molecular structure from `E:\\YDW\\ChemDatas`.

    Variants:
    - `'donor_bdd1'` (default): Authentic 32-atom T-BDD-T conjugated donor unit (`C18H8O2S4`)
      extracted from `E:\\YDW\\ChemDatas\\TH\\00cluster_PM6-L8BO-2ac_rank1_donor-BDD1_Acceptor2-NC1(2).gjf`,
      rigidly aligned to its principal molecular XY plane for stacking/alignment workflows.
    - `'monomer'`: Complete 158-atom PM6 polymer repeat monomer (`C68H78F2O2S8`)
      loaded directly from `E:\\YDW\\ChemDatas\\Gjf\\ydw-PM6-opt01.gjf`.
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
    """Returns an authentic Y6 (BTP-4F) non-fullerene acceptor unit from `E:\\YDW\\ChemDatas`.

    Variants:
    - `'ic2f'` (default): Authentic 22-atom Gaussian-optimized IC-2F acceptor terminal unit (`C13H4F2N2O`)
      loaded directly from `E:\\YDW\\ChemDatas\\ydw01\\IC\\ydw-IC_FF.log` (`wB97X-D/6-31G(d,p)`).
    - `'btp_core'`: Authentic 159-atom Gaussian-optimized BTP central core (`C60H90N4S5`)
      loaded directly from `E:\\YDW\\ChemDatas\\ydw01\\BTP\\ydw-BTP.log` (`wB97X-D/6-31G(d,p)`).
    - `'full'`: Authentic 199-atom Gaussian-optimized full Y6 molecule (`C86H94F4N8O2S5`)
      loaded directly from `E:\\YDW\\ChemDatas\\ydw01\\BTP-series\\ydw-BTP-4F.log` (`B3LYP-D3(BJ)/6-311G(d,p)`).
    - `'acceptor_nc1'`: Authentic 28-atom Acceptor2-NC1 terminal fragment (`C17H6F2N2O`) from the MD cluster.
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
            if irc_res.status == "succeeded":
                fwd_barrier = irc_res.irc_forward_barrier_kcal_mol
                rev_barrier = irc_res.irc_reverse_barrier_kcal_mol
                rxn_energy = irc_res.reaction_energy_kcal_mol
                path_points = irc_res.irc_path

        summary = (
            f"=== CalChem Reaction Pathway ({name}: {method.upper()}/{basis}) ===\n"
            f"TS Electronic Energy E_TS = {ts_energy:14.8f} Hartree\n"
            f"Imaginary Frequency       = {imag_freq:10.1f} cm^-1\n"
            f"Forward Barrier E_a(fwd)  = {fwd_barrier or 0.0:8.2f} kcal/mol\n"
            f"Reverse Barrier E_a(rev)  = {rev_barrier or 0.0:8.2f} kcal/mol\n"
            f"Reaction Energy Delta_H   = {rxn_energy or 0.0:8.2f} kcal/mol\n"
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
