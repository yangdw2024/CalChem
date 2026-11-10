"""Unit tests for pure Rust Transition State Search (P-RFO) and Gonzalez-Schlegel IRC."""

import math
from calchem import _core as rust_core
from calchem.backends.native import NativeCalchem
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.workflows.opv import OPVWorkflow, OPVReactionBarrierResult

BOHR_TO_ANGSTROM = 0.529177210903


def test_rust_core_ts_search_hcn():
    """Tests pure Rust P-RFO transition state search for HCN <-> HNC isomerization."""
    # Perturbed near-TS geometry in Bohr:
    symbols = ["C", "N", "H"]
    geometry_bohr = [
        [0.0, 0.0, 0.0],
        [2.310, 0.0, 0.0],
        [0.685, 2.155, 0.0],
    ]

    res = rust_core.run_ts_search(
        symbols=symbols,
        geometry_bohr=geometry_bohr,
        charge=0,
        multiplicity=1,
        method="rhf",
        basis="sto-3g",
        max_steps=15,
        max_force=8e-4,
        rms_force=5e-4,
        max_displacement=2e-3,
        trust_radius_bohr=0.05,
    )

    assert res.converged is True
    assert res.energy_hartree < -91.5
    assert res.imaginary_frequency_cm < -500.0, f"Expected imaginary frequency < -500 cm^-1, got {res.imaginary_frequency_cm}"
    assert len(res.transition_vector) == 9

    # Literature R_CN is ~2.308 Bohr
    c_pos = res.geometry_bohr[0]
    n_pos = res.geometry_bohr[1]
    r_cn = abs(n_pos[0] - c_pos[0])
    assert abs(r_cn - 2.308) < 0.03


def test_rust_core_run_irc_hcn():
    """Tests pure Rust Gonzalez-Schlegel IRC integration from converged HCN TS."""
    # Literature TS geometry in Bohr
    symbols = ["C", "N", "H"]
    geometry_bohr = [
        [0.0, 0.0, 0.0],
        [2.308, 0.0, 0.0],
        [0.672, 2.171, 0.0],
    ]

    # Obtain TS and transition vector
    ts_res = rust_core.run_ts_search(
        symbols=symbols,
        geometry_bohr=geometry_bohr,
        charge=0,
        multiplicity=1,
        method="rhf",
        basis="sto-3g",
        max_steps=5,
    )
    assert ts_res.converged is True

    irc_res = rust_core.run_irc(
        symbols=symbols,
        geometry_bohr=ts_res.geometry_bohr,
        transition_vector=ts_res.transition_vector,
        charge=0,
        multiplicity=1,
        method="rhf",
        basis="sto-3g",
        step_size_bohr=0.1,
        max_points=4,
        direction="both",
        convergence_tol=1e-3,
        max_gs_iter=6,
    )

    assert irc_res.transition_state.point_index == 0
    assert irc_res.transition_state.path_coordinate_s == 0.0
    assert irc_res.transition_state.delta_energy_kcal_mol == 0.0
    assert irc_res.total_points >= 3

    # Forward branch descends into well
    assert irc_res.forward is not None
    assert len(irc_res.forward.points) > 0
    last_fwd = irc_res.forward.points[-1]
    assert last_fwd.delta_energy_kcal_mol < 0.0
    assert irc_res.forward.barrier_kcal_mol > 0.0

    # Reverse branch descends into opposite well
    assert irc_res.reverse is not None
    assert len(irc_res.reverse.points) > 0
    last_rev = irc_res.reverse.points[-1]
    assert last_rev.delta_energy_kcal_mol < 0.0
    assert irc_res.reverse.barrier_kcal_mol > 0.0


def test_native_backend_ts_search():
    """Tests NativeCalchem task='ts' execution dispatch."""
    mol = MoleculeSpec(
        symbols=["C", "N", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.0],
            [2.310 * BOHR_TO_ANGSTROM, 0.0, 0.0],
            [0.685 * BOHR_TO_ANGSTROM, 2.155 * BOHR_TO_ANGSTROM, 0.0],
        ],
        charge=0,
        multiplicity=1,
    )
    req = CalculationRequest(
        task="ts",
        molecule=mol,
        backend="native",
        method="rhf",
        basis="sto-3g",
        options={"max_steps": 15, "trust_radius_bohr": 0.05},
    )

    calc = NativeCalchem()
    res = calc.run(req)

    assert res.status == "succeeded"
    assert res.scf_converged is True
    assert res.is_transition_state is True
    assert res.imaginary_frequency_cm is not None
    assert res.imaginary_frequency_cm < -500.0
    assert res.optimized_coordinates_angstrom is not None
    assert "transition_vector" in res.extra


def test_native_backend_irc():
    """Tests NativeCalchem task='irc' execution dispatch."""
    mol = MoleculeSpec(
        symbols=["C", "N", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.0],
            [2.308 * BOHR_TO_ANGSTROM, 0.0, 0.0],
            [0.672 * BOHR_TO_ANGSTROM, 2.171 * BOHR_TO_ANGSTROM, 0.0],
        ],
        charge=0,
        multiplicity=1,
    )
    req = CalculationRequest(
        task="irc",
        molecule=mol,
        backend="native",
        method="rhf",
        basis="sto-3g",
        options={"max_points": 3, "step_size_bohr": 0.1, "max_gs_iter": 5},
    )

    calc = NativeCalchem()
    res = calc.run(req)

    assert res.status == "succeeded"
    assert res.is_transition_state is True
    assert res.irc_forward_barrier_kcal_mol is not None
    assert res.irc_reverse_barrier_kcal_mol is not None
    assert len(res.irc_path) >= 3


class _InMemoryNativeService:
    def __init__(self):
        self._backend = NativeCalchem()

    def run(self, request):
        return self._backend.run(request)


def test_opv_workflow_reaction_barrier():
    """Tests OPVWorkflow.reaction_barrier high-level facade."""
    mol = MoleculeSpec(
        symbols=["C", "N", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.0],
            [2.308 * BOHR_TO_ANGSTROM, 0.0, 0.0],
            [0.672 * BOHR_TO_ANGSTROM, 2.171 * BOHR_TO_ANGSTROM, 0.0],
        ],
        charge=0,
        multiplicity=1,
    )

    service = _InMemoryNativeService()
    wf = OPVWorkflow(service=service, backend="native")
    result: OPVReactionBarrierResult = wf.reaction_barrier(
        mol,
        method="rhf",
        basis="sto-3g",
        run_irc_path=True,
        max_irc_points=3,
        irc_step_size_bohr=0.1,
        name="HCN_Isomerization",
    )

    assert result.name == "HCN_Isomerization"
    assert result.imaginary_frequency_cm < -500.0
    assert result.forward_barrier_kcal_mol is not None
    assert result.reverse_barrier_kcal_mol is not None
    assert result.n_irc_points >= 3
    assert "HCN_Isomerization" in result.summary
    assert "Imaginary Frequency" in result.summary
