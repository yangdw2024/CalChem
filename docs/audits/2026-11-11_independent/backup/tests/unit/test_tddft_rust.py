"""Unit tests for pure Rust Linear-Response TD-DFT / TDA / CIS Excited-State Engine,
UV-Vis Optical Absorption Spectra, and Exciton Binding Energy E_b."""

from __future__ import annotations

import math
import numpy as np
import pytest

from calchem.backends.native import NativeCalchem
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.workflows.opv import OPVWorkflow


class _InMemoryNativeService:
    def __init__(self) -> None:
        self._backend = NativeCalchem()

    def run(self, request: CalculationRequest):
        return self._backend.run(request)


def _water_spec() -> MoleculeSpec:
    return MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
        charge=0,
        multiplicity=1,
    )


def _ethylene_spec() -> MoleculeSpec:
    return MoleculeSpec(
        symbols=["C", "C", "H", "H", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.6695],
            [0.0, 0.0, -0.6695],
            [0.0, 0.9289, 1.2321],
            [0.0, -0.9289, 1.2321],
            [0.0, 0.9289, -1.2321],
            [0.0, -0.9289, -1.2321],
        ],
        charge=0,
        multiplicity=1,
    )


def _formaldehyde_spec() -> MoleculeSpec:
    return MoleculeSpec(
        symbols=["C", "O", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, -0.5290],
            [0.0, 0.0, 0.6770],
            [0.0, 0.9400, -1.1160],
            [0.0, -0.9400, -1.1160],
        ],
        charge=0,
        multiplicity=1,
    )


def test_tddft_water_cis_and_b3lyp_native_result():
    backend = NativeCalchem()
    w = _water_spec()

    # 1. CIS (TDA Hartree-Fock)
    res_cis = backend.run(
        CalculationRequest(
            task="cis",
            molecule=w,
            backend="native",
            method="cis",
            basis="6-31g",
            options={"n_states": 4, "manifold": "both"},
        )
    )
    assert res_cis.status == "succeeded"
    assert res_cis.optical_gap_ev is not None
    assert res_cis.lowest_triplet_ev is not None
    assert res_cis.delta_est_ev is not None
    assert res_cis.optical_gap_ev > res_cis.lowest_triplet_ev
    assert res_cis.delta_est_ev > 0.0
    assert len(res_cis.excited_states) == 8

    # 2. Full linear-response TD-B3LYP (Casida RPA)
    res_td = backend.run(
        CalculationRequest(
            task="tddft",
            molecule=w,
            backend="native",
            method="b3lyp",
            basis="6-31g",
            options={"n_states": 3, "tda": False, "manifold": "both", "grid_level": "coarse"},
        )
    )
    assert res_td.status == "succeeded"
    assert res_td.optical_gap_ev is not None
    assert res_td.optical_gap_ev > 6.0
    assert res_td.lowest_triplet_ev < res_td.optical_gap_ev
    assert len(res_td.excited_states) == 6

    # Verify transition dipole and oscillator strength consistency
    for st in res_td.excited_states:
        if st.multiplicity == 1:
            assert st.oscillator_strength >= 0.0
            assert len(st.transition_dipole_au) == 3
        else:
            assert st.oscillator_strength == 0.0
            assert all(abs(d) < 1e-12 for d in st.transition_dipole_au)


def test_tddft_from_scf_wavefunction_zero_recomputation():
    from calchem import _core

    w = _water_spec()
    b2a = 0.529177210903
    coords_bohr = [[c / b2a for c in row] for row in w.coordinates_angstrom]

    wf = _core.run_single_point(
        list(w.symbols),
        coords_bohr,
        w.charge,
        w.multiplicity,
        method="b3lyp",
        basis="6-31g",
        grid_level="coarse",
    )
    assert wf.converged

    # Run TD-DFT directly atop the Python-held PyScfWavefunction
    td_res = _core.run_tddft_from_scf_py(
        wf,
        n_states=3,
        tda=False,
        manifold="both",
        grid_level="coarse",
    )
    assert abs(td_res.ground_energy_hartree - wf.total_energy_hartree) < 1e-12
    assert td_res.optical_gap_ev is not None
    assert td_res.lowest_triplet_ev is not None
    assert td_res.delta_est_ev > 0.0


def test_spectroscopy_formaldehyde_dark_and_ethylene_bright():
    backend = NativeCalchem()

    # Formaldehyde: symmetry-forbidden dark S1 (n -> pi*)
    ch2o = _formaldehyde_spec()
    res_ch2o = backend.run(
        CalculationRequest(
            task="tddft",
            molecule=ch2o,
            backend="native",
            method="b3lyp",
            basis="6-31g",
            options={"n_states": 3, "manifold": "singlet", "grid_level": "coarse"},
        )
    )
    assert res_ch2o.status == "succeeded"
    s1_ch2o = res_ch2o.excited_states[0]
    assert 3.5 <= s1_ch2o.excitation_energy_ev <= 4.6
    assert s1_ch2o.oscillator_strength < 1e-4

    # Ethylene: bright pi -> pi* with large oscillator strength
    c2h4 = _ethylene_spec()
    res_c2h4 = backend.run(
        CalculationRequest(
            task="tddft",
            molecule=c2h4,
            backend="native",
            method="b3lyp",
            basis="6-31g",
            options={"n_states": 4, "manifold": "singlet", "grid_level": "coarse"},
        )
    )
    assert res_c2h4.status == "succeeded"
    bright_s = max(res_c2h4.excited_states, key=lambda s: s.oscillator_strength)
    assert bright_s.oscillator_strength > 0.25
    assert 7.2 <= bright_s.excitation_energy_ev <= 8.6


def test_uv_vis_broadening_and_spectrum():
    from calchem import _core

    backend = NativeCalchem()
    c2h4 = _ethylene_spec()
    res = backend.run(
        CalculationRequest(
            task="tddft",
            molecule=c2h4,
            backend="native",
            method="b3lyp",
            basis="6-31g",
            options={"n_states": 4, "manifold": "singlet", "grid_level": "coarse"},
        )
    )
    td_obj = res.extra["tddft_result"]

    spec = td_obj.broaden_spectrum(min_nm=120.0, max_nm=300.0, n_points=361, fwhm_ev=0.4)
    assert len(spec.wavelengths_nm) == 361
    assert len(spec.epsilon_l_mol_cm) == 361
    peak_idx = int(np.argmax(spec.epsilon_l_mol_cm))
    peak_wl = spec.wavelengths_nm[peak_idx]
    peak_eps = spec.epsilon_l_mol_cm[peak_idx]

    bright_s = max(res.excited_states, key=lambda s: s.oscillator_strength)
    expected_peak_wl = 1239.841984 / bright_s.excitation_energy_ev
    assert abs(peak_wl - expected_peak_wl) < 1.0
    assert peak_eps > 1.0e4


def test_opv_workflow_optical_absorption_and_exciton_binding():
    service = _InMemoryNativeService()
    wf = OPVWorkflow(service=service)
    c2h4 = _ethylene_spec()

    abs_res = wf.optical_absorption(
        c2h4,
        method="b3lyp",
        basis="6-31g",
        n_states=4,
        tda=False,
        manifold="both",
        compute_exciton_binding=True,
        name="ethylene",
    )

    assert abs_res.optical_gap_ev > 6.0
    assert abs_res.lowest_triplet_ev is not None
    assert abs_res.lowest_triplet_ev < abs_res.optical_gap_ev
    assert abs_res.delta_est_ev > 2.0
    assert abs_res.oscillator_strength_max > 0.25
    assert len(abs_res.spectrum_wavelengths_nm) > 0
    assert len(abs_res.spectrum_epsilon_l_mol_cm) > 0

    # Exciton binding energy check
    assert abs_res.fundamental_gap_vertical_ev is not None
    assert abs_res.exciton_binding_energy_ev is not None
    # For ethylene, fundamental gap IP_v - EA_v is significantly larger than optical gap (Eb > 1.5 eV)
    assert abs_res.fundamental_gap_vertical_ev > abs_res.optical_gap_ev
    assert abs_res.exciton_binding_energy_ev > 1.5
    assert "Exciton Binding Energy E_b" in abs_res.summary
