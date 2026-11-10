"""
CalChem 高级 API。

提供单点计算、几何优化、频率分析等核心功能的封装。

所有方法均为自研实现，无需外部依赖。
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List

import numpy as np

from ..mol.molecule import Molecule
from ..io.input import read_molecule
from ..io.output import Results
from ..utils.logging_utils import get_logger
from ..utils.exceptions import InputError

logger = get_logger(__name__)


def single_point(
    mol_or_file,
    method: str = "hf",
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    charge: int = 0,
    multiplicity: int = 1,
    **kwargs,
) -> Results:
    """Run a single-point energy calculation.

    Parameters
    ----------
    mol_or_file : Molecule or str
        Molecule object or path to input file
    method : str
        Method: 'hf', 'dft', 'rhf', 'uhf', 'rks', 'uks', 'mp2', 'ccsd'
    basis : str
        Basis set name
    xc : str
        DFT functional (for DFT methods)
    charge : int
        Molecular charge
    multiplicity : int
        Spin multiplicity

    Returns
    -------
    Results
        Calculation results
    """
    if isinstance(mol_or_file, str):
        mol = read_molecule(mol_or_file, charge=charge, multiplicity=multiplicity)
    elif isinstance(mol_or_file, Molecule):
        mol = mol_or_file
    else:
        raise InputError(f"Invalid molecule input: {type(mol_or_file)}")

    method_lower = method.lower()

    if method_lower in ("hf", "rhf"):
        from ..scf.hf import RHF
        calc = RHF(mol, basis=basis, **kwargs)
    elif method_lower == "uhf":
        from ..scf.hf import UHF
        calc = UHF(mol, basis=basis, **kwargs)
    elif method_lower == "rohf":
        from ..scf.hf import ROHF
        calc = ROHF(mol, basis=basis, **kwargs)
    elif method_lower in ("dft", "rks"):
        from ..scf.dft import RKS
        calc = RKS(mol, basis=basis, xc=xc, **kwargs)
    elif method_lower == "uks":
        from ..scf.dft import UKS
        calc = UKS(mol, basis=basis, xc=xc, **kwargs)
    elif method_lower == "mp2":
        from ..post_hf.mp2 import MP2
        calc = MP2(mol, basis=basis, **kwargs)
    elif method_lower == "ccsd":
        from ..post_hf.ccsd import CCSD
        calc = CCSD(mol, basis=basis, **kwargs)
    elif method_lower in ("ccsd(t)", "ccsdt"):
        from ..post_hf.ccsd_t import CCSDT
        calc = CCSDT(mol, basis=basis, **kwargs)
    elif method_lower == "mp4":
        from ..post_hf.mp4 import MP4
        calc = MP4(mol, basis=basis, **kwargs)
    elif method_lower == "ccd":
        from ..post_hf.ccd import CCD
        calc = CCD(mol, basis=basis, **kwargs)
    elif method_lower == "casscf":
        from ..post_hf.casscf import CASSCF
        n_active = kwargs.pop("n_active", 2)
        n_active_elec = kwargs.pop("n_active_elec", 2)
        calc = CASSCF(mol, basis=basis, n_active=n_active, n_active_elec=n_active_elec, **kwargs)
    else:
        raise InputError(f"Unknown method: {method}")

    calc.kernel()

    return calc.get_results()


def optimize(
    mol_or_file,
    method: str = "hf",
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    charge: int = 0,
    multiplicity: int = 1,
    optimizer: str = "bfgs",
    use_numerical_grad: bool = True,
    **kwargs,
) -> Tuple[Molecule, float]:
    """Run a geometry optimization.

    Parameters
    ----------
    mol_or_file : Molecule or str
        Molecule object or path to input file
    method : str
        Method: 'hf', 'dft', etc.
    basis : str
        Basis set name
    xc : str
        DFT functional
    charge : int
        Molecular charge
    multiplicity : int
        Spin multiplicity
    optimizer : str
        Optimizer: 'bfgs', 'berny'
    use_numerical_grad : bool
        If True, use numerical gradient

    Returns
    -------
    tuple
        (optimized_molecule, final_energy)
    """
    if isinstance(mol_or_file, str):
        mol = read_molecule(mol_or_file, charge=charge, multiplicity=multiplicity)
    elif isinstance(mol_or_file, Molecule):
        mol = mol_or_file
    else:
        raise InputError(f"Invalid molecule input: {type(mol_or_file)}")

    def grad_func(current_mol):
        result = single_point(
            current_mol,
            method=method,
            basis=basis,
            xc=xc,
            charge=charge,
            multiplicity=multiplicity,
        )
        if use_numerical_grad:
            grad = _numerical_gradient(
                current_mol,
                method=method,
                basis=basis,
                xc=xc,
                charge=charge,
                multiplicity=multiplicity,
            )
        else:
            try:
                if method.lower() in ("hf", "rhf"):
                    from ..scf.hf import RHF
                    from ..grad.hf_grad import HFGradient
                    calc = RHF(current_mol, basis=basis)
                    calc.kernel()
                    grad = HFGradient(calc).kernel()
                elif method.lower() in ("dft", "rks"):
                    from ..scf.dft import RKS
                    from ..grad.dft_grad import DFTGradient
                    calc = RKS(current_mol, basis=basis, xc=xc)
                    calc.kernel()
                    grad = DFTGradient(calc).kernel()
                else:
                    grad = _numerical_gradient(
                        current_mol, method=method, basis=basis, xc=xc,
                        charge=charge, multiplicity=multiplicity
                    )
            except Exception as e:
                logger.warning(f"Analytical gradient failed: {e}, falling back to numerical")
                grad = _numerical_gradient(
                    current_mol, method=method, basis=basis, xc=xc,
                    charge=charge, multiplicity=multiplicity
                )
        return result.energy, grad

    opt_lower = optimizer.lower()
    opt_kwargs = {}
    for k, v in kwargs.items():
        if k in ("max_cycles", "conv_tol"):
            opt_kwargs[k] = v

    if opt_lower == "bfgs":
        from ..opt.bfgs import BFGS
        opt = BFGS(mol, grad_func, **opt_kwargs)
    elif opt_lower == "berny":
        try:
            from ..opt.berny import Berny
            opt = Berny(mol, grad_func, **kwargs)
        except ImportError:
            raise RuntimeError("Berny optimizer requires Berny package: pip install berny")
    else:
        raise InputError(f"Unknown optimizer: {optimizer}")

    final_mol = opt.optimize()

    return final_mol, opt.final_energy


def frequency(
    mol_or_file,
    method: str = "hf",
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    charge: int = 0,
    multiplicity: int = 1,
    temperature: float = 298.15,
    pressure: float = 1.0,
    use_numerical_hessian: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run a frequency analysis.

    Parameters
    ----------
    mol_or_file : Molecule or str
        Molecule object or path to input file
    method : str
        Method: 'hf', 'dft', etc.
    basis : str
        Basis set name
    xc : str
        DFT functional
    charge : int
        Molecular charge
    multiplicity : int
        Spin multiplicity
    temperature : float
        Temperature in Kelvin
    pressure : float
        Pressure in atm
    use_numerical_hessian : bool
        If True, use numerical Hessian

    Returns
    -------
    tuple
        (frequencies, thermochemistry)
    """
    if isinstance(mol_or_file, str):
        mol = read_molecule(mol_or_file, charge=charge, multiplicity=multiplicity)
    elif isinstance(mol_or_file, Molecule):
        mol = mol_or_file
    else:
        raise InputError(f"Invalid molecule input: {type(mol_or_file)}")

    calc = None
    method_lower = method.lower()

    if method_lower in ("hf", "rhf"):
        from ..scf.hf import RHF
        calc = RHF(mol, basis=basis)
    elif method_lower in ("dft", "rks"):
        from ..scf.dft import RKS
        calc = RKS(mol, basis=basis, xc=xc)
    else:
        raise InputError(f"Frequency analysis not supported for method: {method}")

    calc.kernel()

    if use_numerical_hessian:
        hessian = _numerical_hessian(
            mol, method=method, basis=basis, xc=xc,
            charge=charge, multiplicity=multiplicity
        )
        freq_analysis = _NumericalFrequencyAnalysis(calc, hessian)
    else:
        try:
            from ..grad.freq import FrequencyAnalysis
            freq_analysis = FrequencyAnalysis(calc)
        except ImportError:
            logger.warning("Analytical Hessian not available, using numerical Hessian")
            hessian = _numerical_hessian(
                mol, method=method, basis=basis, xc=xc,
                charge=charge, multiplicity=multiplicity
            )
            freq_analysis = _NumericalFrequencyAnalysis(calc, hessian)

    freqs = freq_analysis.kernel(temperature=temperature, pressure=pressure)

    return freqs, freq_analysis.thermo or {}


def excited(
    mol_or_file,
    method: str = "cis",
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    nstates: int = 10,
    singlet: bool = True,
    charge: int = 0,
    multiplicity: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """Run an excited state calculation.

    Parameters
    ----------
    mol_or_file : Molecule or str
        Molecule object or path to input file
    method : str
        Method: 'cis', 'tddft', 'tdhf'
    basis : str
        Basis set name
    xc : str
        DFT functional (for TDDFT)
    nstates : int
        Number of excited states
    singlet : bool
        If True, compute singlet excited states
    charge : int
        Molecular charge
    multiplicity : int
        Spin multiplicity

    Returns
    -------
    dict
        Excited state results
    """
    if isinstance(mol_or_file, str):
        mol = read_molecule(mol_or_file, charge=charge, multiplicity=multiplicity)
    elif isinstance(mol_or_file, Molecule):
        mol = mol_or_file
    else:
        raise InputError(f"Invalid molecule input: {type(mol_or_file)}")

    method_lower = method.lower()

    if method_lower == "cis":
        from ..scf.hf import RHF
        from ..excited.cis import CIS
        hf = RHF(mol, basis=basis, **kwargs)
        hf.kernel()
        cis = CIS(hf, nstates=nstates, singlet=singlet)
        e_exc = cis.kernel()
        return {
            "method": "CIS",
            "excitation_energies": e_exc,
            "oscillator_strengths": cis.oscillator_strengths,
            "transition_dipoles": cis.transition_dipoles,
            "converged": cis.converged,
            "ground_state_energy": hf.e_tot,
        }
    elif method_lower == "tddft":
        from ..scf.dft import RKS
        from ..excited.tddft_native import TDDFT
        rks = RKS(mol, basis=basis, xc=xc, **kwargs)
        rks.kernel()
        tddft = TDDFT(rks, nstates=nstates, singlet=singlet)
        e_exc = tddft.kernel()
        return {
            "method": "TDDFT",
            "excitation_energies": e_exc,
            "oscillator_strengths": tddft.oscillator_strengths,
            "transition_dipoles": tddft.transition_dipoles,
            "converged": tddft.converged,
            "ground_state_energy": rks.e_tot,
        }
    else:
        raise InputError(f"Excited state method not supported: {method}")


def properties(
    mol_or_file,
    method: str = "hf",
    basis: str = "sto-3g",
    xc: str = "b3lyp",
    charge: int = 0,
    multiplicity: int = 1,
    props: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Compute molecular properties.

    Parameters
    ----------
    mol_or_file : Molecule or str
        Molecule object or path to input file
    method : str
        Method: 'hf', 'dft', etc.
    basis : str
        Basis set name
    xc : str
        DFT functional
    charge : int
        Molecular charge
    multiplicity : int
        Spin multiplicity
    props : list
        List of properties to compute: 'dipole', 'quadrupole', 'polarizability'

    Returns
    -------
    dict
        Computed properties
    """
    if isinstance(mol_or_file, str):
        mol = read_molecule(mol_or_file, charge=charge, multiplicity=multiplicity)
    elif isinstance(mol_or_file, Molecule):
        mol = mol_or_file
    else:
        raise InputError(f"Invalid molecule input: {type(mol_or_file)}")

    if props is None:
        props = ['dipole']

    method_lower = method.lower()

    if method_lower in ("hf", "rhf"):
        from ..scf.hf import RHF
        calc = RHF(mol, basis=basis, **kwargs)
    elif method_lower in ("dft", "rks"):
        from ..scf.dft import RKS
        calc = RKS(mol, basis=basis, xc=xc, **kwargs)
    else:
        raise InputError(f"Properties calculation not supported for method: {method}")

    calc.kernel()

    results = {}

    if 'dipole' in props:
        from ..props.multipole_native import compute_dipole_moment
        results['dipole'] = compute_dipole_moment(calc)

    if 'quadrupole' in props:
        from ..props.multipole_native import compute_quadrupole_moment
        results['quadrupole'] = compute_quadrupole_moment(calc)

    results['energy'] = calc.e_tot
    results['method'] = method
    results['basis'] = basis

    return results


def _numerical_gradient(mol, method="hf", basis="sto-3g", xc="b3lyp",
                       charge=0, multiplicity=1, step=1e-5):
    """Compute numerical gradient using central differences."""
    coords = mol.coordinates.copy()
    n_atoms = mol.natom
    grad = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        for j in range(3):
            coords[i, j] += step
            mol_plus = Molecule(mol.symbols, coords, units="bohr")
            e_plus = single_point(mol_plus, method=method, basis=basis,
                                 xc=xc, charge=charge, multiplicity=multiplicity).energy

            coords[i, j] -= 2 * step
            mol_minus = Molecule(mol.symbols, coords, units="bohr")
            e_minus = single_point(mol_minus, method=method, basis=basis,
                                  xc=xc, charge=charge, multiplicity=multiplicity).energy

            coords[i, j] += step
            grad[i, j] = (e_plus - e_minus) / (2 * step)

    return grad


def _numerical_hessian(mol, method="hf", basis="sto-3g", xc="b3lyp",
                       charge=0, multiplicity=1, step=1e-5):
    """Compute numerical Hessian using finite differences of gradients."""
    coords = mol.coordinates.copy()
    n_atoms = mol.natom
    hess = np.zeros((3 * n_atoms, 3 * n_atoms))

    for i in range(n_atoms):
        for j in range(3):
            coords[i, j] += step
            mol_plus = Molecule(mol.symbols, coords, units="bohr")
            grad_plus = _numerical_gradient(mol_plus, method=method, basis=basis,
                                            xc=xc, charge=charge, multiplicity=multiplicity)

            coords[i, j] -= 2 * step
            mol_minus = Molecule(mol.symbols, coords, units="bohr")
            grad_minus = _numerical_gradient(mol_minus, method=method, basis=basis,
                                             xc=xc, charge=charge, multiplicity=multiplicity)

            coords[i, j] += step
            dgrad = (grad_plus - grad_minus) / (2 * step)
            hess[i * 3 + j] = dgrad.flatten()

    return (hess + hess.T) / 2


class _NumericalFrequencyAnalysis:
    """Frequency analysis using numerical Hessian."""

    def __init__(self, scf_calc, hessian):
        self.scf_calc = scf_calc
        self.hessian = hessian
        self.thermo = {}

    def kernel(self, temperature=298.15, pressure=1.0):
        mol = self.scf_calc.mol
        n_atoms = mol.natom
        coords = mol.coordinates

        masses = []
        from ..utils.constants import ATOMIC_MASSES
        for sym in mol.symbols:
            masses.append(ATOMIC_MASSES.get(sym, 1.0))

        mass_matrix = np.zeros((3 * n_atoms, 3 * n_atoms))
        for i in range(n_atoms):
            for j in range(3):
                mass_matrix[i * 3 + j, i * 3 + j] = masses[i]

        mass_sqrt = np.diag(np.sqrt(np.diag(mass_matrix)))
        mass_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(mass_matrix)))

        hess_weighted = mass_inv_sqrt @ self.hessian @ mass_inv_sqrt

        eigvals, eigvecs = np.linalg.eigh(hess_weighted)

        freqs = np.sqrt(np.maximum(eigvals, 0.0))

        freqs_cm1 = freqs * 219474.63

        self._compute_thermo(freqs_cm1, temperature, pressure, n_atoms)

        return freqs_cm1

    def _compute_thermo(self, freqs_cm1, temperature, pressure, n_atoms):
        from ..utils.constants import KB, HARTREE_TO_JOULE, AVOGADRO, R

        n_modes = len(freqs_cm1)
        n_trans = 3
        n_rot = 3 if n_atoms > 1 else 0
        n_vib = n_modes - n_trans - n_rot

        vib_freqs = freqs_cm1[n_trans + n_rot:]

        zpe = 0.0
        for nu in vib_freqs:
            if nu > 0:
                zpe += 0.5 * nu * 4.556335e-6

        self.thermo = {
            'zpe': zpe,
            'temperature': temperature,
            'pressure': pressure,
            'n_atoms': n_atoms,
            'n_vibrational_modes': n_vib,
        }