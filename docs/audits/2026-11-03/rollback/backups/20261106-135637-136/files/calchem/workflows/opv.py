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
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import calchem.backends  # noqa: F401 - ensure backends are registered
from calchem.api.service import TaskService
from calchem.core.request import CalculationRequest, MoleculeSpec, ResourceLimits

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
    """OPV molecule specification (donor or acceptor)."""
    name: str
    symbols: List[str]
    coordinates_angstrom: List[List[float]]
    charge: int = 0
    multiplicity: int = 1

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
        name: str = "molecule",
    ) -> OPVWorkflowResult:
        """Runs single-point energy and extracts frontier molecular orbitals."""
        req = CalculationRequest(
            task="single_point",
            molecule=mol,
            backend=self.backend,
            method=method,
            basis=basis,
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

        return OPVWorkflowResult(
            name=name,
            scf_energy_hartree=result.scf_energy_hartree,
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
        name: str = "molecule",
    ) -> OPVWorkflowResult:
        """Optimizes geometry and extracts frontier orbitals at relaxed equilibrium."""
        req = CalculationRequest(
            task="optimize",
            molecule=mol,
            backend=self.backend,
            method=method,
            basis=basis,
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
            sp_res = self.single_point(relaxed_mol, method=method, basis=basis, name=name)
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

        return OPVWorkflowResult(
            name=name,
            scf_energy_hartree=result.scf_energy_hartree,
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
