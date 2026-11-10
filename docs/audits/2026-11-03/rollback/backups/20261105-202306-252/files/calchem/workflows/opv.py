"""CalChem OPV (Organic Photovoltaics) Workflow.

Provides standardized computational pipelines for organic solar cells (OSC/OPV):
1. Molecular preparation (geometry, charge, multiplicity)
2. Native Rust-accelerated RHF geometry optimization & single-point energy
3. Frontier molecular orbital (FMO) extraction (HOMO, LUMO, Gap in eV)
4. Donor-Acceptor (D-A) energy level alignment & open-circuit voltage (Voc) estimation (Scharber model)
5. Electric dipole moment evaluation for molecular packing & electrostatic potential analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import calchem.backends  # noqa: F401 - ensure backends are registered
from calchem.api.service import TaskService
from calchem.core.request import CalculationRequest, MoleculeSpec, ResourceLimits

HARTREE_TO_EV = 27.211386245988


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
    optimized_geometry: Optional[List[List[float]]] = None
    converged: bool = False
    errors: List[str] = field(default_factory=list)


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


class OPVWorkflow:
    """OPV standard computational pipeline."""

    def __init__(self, service: Optional[TaskService] = None, backend: str = "native"):
        self.service = service or TaskService()
        self.backend = backend

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
            resources=ResourceLimits(max_cpu_cores=8, max_memory_mb=4000, timeout_seconds=300),
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
            resources=ResourceLimits(max_cpu_cores=8, max_memory_mb=4000, timeout_seconds=300),
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
        else:
            dipole = result.dipole_debye
            dipole_mag = result.dipole_magnitude_debye

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
            optimized_geometry=opt_geo,
            converged=result.status == "succeeded",
            errors=result.errors,
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
