"""Tests for Gaussian and ORCA adapters using synthetic logs (no external executable needed)."""
import pytest
from unittest.mock import patch
from datetime import datetime

from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.backends.gaussian import GaussianBackend, _parse_g16_log, _parse_frequencies, _parse_final_geometry
from calchem.backends.orca import ORCABackend, _parse_orca_output


_H2O_MOL = MoleculeSpec(
    symbols=["O", "H", "H"],
    coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],
    charge=0,
    multiplicity=1,
)


def test_gaussian_opt_takes_last_scf_energy():
    """Q01: Gaussian Opt log with multiple SCF steps must take the final energy, not the initial."""
    synth_log = """
 Entering Gaussian System, Link 0=g16
 # B3LYP/6-31G(d) Opt
 ...
 Cycle 1:
 SCF Done:  E(RB3LYP) =  -76.4000000000     A.U. after   10 cycles
 ...
 Step number   2
 SCF Done:  E(RB3LYP) =  -76.4250000000     A.U. after    8 cycles
 ...
 Stationary point found.
 Normal termination of Gaussian 16
"""
    parsed = _parse_g16_log(synth_log)
    assert parsed["normal_termination"] is True
    assert parsed["opt_converged"] is True
    # Must be -76.425, NOT the initial -76.400!
    assert parsed["energy_hartree"] == pytest.approx(-76.425)
    assert parsed["scf_energy_hartree"] == pytest.approx(-76.425)


def test_gaussian_post_hf_does_not_substitute_scf_energy():
    """Q01: For MP2, if MP2 energy is missing, it must not falsely return SCF energy."""
    synth_log_no_mp2 = """
 # MP2/6-31G SP
 SCF Done:  E(RHF) =  -76.0100000000     A.U. after    8 cycles
 Error termination in link 9999
"""
    parsed = _parse_g16_log(synth_log_no_mp2, method="mp2")
    assert parsed["scf_energy_hartree"] == pytest.approx(-76.01)
    # Total MP2 energy must be None, NOT -76.01!
    assert parsed["energy_hartree"] is None


def test_gaussian_freq_parsing_and_result():
    """Q02: Frequencies must be returned in Result.frequencies_cm, not None."""
    synth_log = """
 Entering Gaussian System
 # RHF/STO-3G Freq
 SCF Done:  E(RHF) =  -74.9600000000     A.U. after    7 cycles
 Harmonic frequencies (cm**-1), IR intensities (KM/Mole)
                      1                      2                      3
                     A1                     B2                     A1
 Frequencies --   1700.5000              3650.2000              3750.8000
 Red. masses --     1.0800                 1.0200                 1.0500
 Normal termination of Gaussian 16
"""
    backend = GaussianBackend(executable="mock_g16")
    req = CalculationRequest(
        task="frequency",
        molecule=_H2O_MOL,
        backend="gaussian",
        method="rhf",
        basis="sto-3g",
    )
    with patch.object(backend, "capabilities") as mock_caps, \
         patch.object(backend, "_run_command", return_value=(0, synth_log, "")):
        from calchem.core.capabilities import BackendCapabilities
        mock_caps.return_value = BackendCapabilities(name="gaussian", available=True)
        res = backend.frequency(req)

    assert res.status == "succeeded"
    assert res.frequencies_cm is not None
    assert len(res.frequencies_cm) == 3
    assert res.frequencies_cm[0] == pytest.approx(1700.5)
    assert res.frequencies_cm[1] == pytest.approx(3650.2)
    assert res.frequencies_cm[2] == pytest.approx(3750.8)


def test_orca_no_name_error_and_correct_dipole():
    """Q03: ORCA output parsing must not raise NameError on dipole_mag and must extract magnitude."""
    synth_output = """
                                 *****************
                                 * O   R   C   A *
                                 *****************
           Version 6.1.1
 SCF CONVERGED AFTER   9 CYCLES
 FINAL SINGLE POINT ENERGY       -76.412345678901
 ...
 -------------
 DIPOLE MOMENT
 -------------
                                X            Y            Z
 Electronic contribution:      0.00000      0.00000     -0.78912
 Nuclear contribution   :      0.00000      0.00000      0.54321
 Total Dipole Moment    :      0.00000      0.00000     -0.24591
 Magnitude (Debye)      :      0.62514

 ****ORCA TERMINATED NORMALLY****
"""
    backend = ORCABackend(executable="mock_orca")
    req = CalculationRequest(
        task="single_point",
        molecule=_H2O_MOL,
        backend="orca",
        method="b3lyp",
        basis="def2-svp",
    )
    with patch.object(backend, "capabilities") as mock_caps, \
         patch.object(backend, "_run_command", return_value=(0, synth_output, "")):
        from calchem.core.capabilities import BackendCapabilities
        mock_caps.return_value = BackendCapabilities(name="orca", available=True)
        res = backend.single_point(req)

    assert res.status == "succeeded"
    assert res.total_energy_hartree == pytest.approx(-76.412345678901)
    assert res.scf_iterations == 9
    assert res.dipole_magnitude_debye == pytest.approx(0.62514)
