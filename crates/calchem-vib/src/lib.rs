//! `calchem-vib`: Pure Rust Vibrational Frequencies, Hessian Analysis, IR Intensities,
//! and Ideal-Gas Rigid-Rotor Harmonic-Oscillator (RRHO) Thermochemistry ($H, S, C_p, G$).
//!
//! Capabilities:
//! - Semi-numerical Cartesian Hessian matrix ($3N \times 3N$) via parallel central differentiation of analytical nuclear gradients
//! - Atomic Polar Tensor (APT, $3 \times 3N$) and length-gauge dipole derivatives
//! - Eckart frame translation and rotation projection ($3N - 6$ or $3N - 5$ vibrational modes)
//! - Harmonic vibrational frequencies $\tilde{\nu}_k$ ($\text{cm}^{-1}$), reduced masses (amu), force constants ($\text{mDyn}/Å$),
//!   and infrared absorption intensities $I_k$ ($\text{km}/\text{mol}$)
//! - Real vs imaginary mode discrimination ($\tilde{\nu} < 0$ for transition state characterization)
//! - Complete RRHO thermochemical properties: Zero-Point Vibrational Energy (ZPVE), thermal enthalpy $H(T, P)$,
//!   entropy $S(T, P)$, heat capacities $C_p, C_v$, and Gibbs free energy $G(T, P)$

pub mod hessian;
pub mod masses;
pub mod modes;
pub mod thermo;

pub use hessian::{FreqOptions, VibMethod, compute_cartesian_hessian_and_apt};
pub use masses::{
    atomic_mass_amu, GAS_CONSTANT_CAL_MOL_K, GAS_CONSTANT_HARTREE_K, GAS_CONSTANT_J_MOL_K,
    HARTREE_TO_EV, HARTREE_TO_KCAL_MOL, HESSIAN_EVAL_TO_INV_CM, IR_INTENSITY_PREFACTOR_KM_MOL,
};
pub use modes::{NormalMode, VibrationalAnalysis, analyze_vibrations};
pub use thermo::{Thermochemistry, compute_thermochemistry};

use calchem_model::{CalcError, Molecule};


use calchem_tensor::Matrix2D;

/// High-level result combining vibrational normal mode analysis and thermochemistry.
#[derive(Debug, Clone)]
pub struct FrequencyResult {
    pub energy_hartree: f64,
    pub cartesian_hessian: Matrix2D,
    pub apt: Matrix2D,
    pub vibrational_analysis: VibrationalAnalysis,
    pub thermochemistry: Thermochemistry,
}

/// Runs full vibrational frequency analysis and RRHO thermochemistry on a molecule.
pub fn compute_frequencies_and_thermo(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    opts: &FreqOptions,
) -> Result<FrequencyResult, CalcError> {
    let (cart_hess, apt, e0) = compute_cartesian_hessian_and_apt(mol, basis_name, method, opts)?;
    let vib = analyze_vibrations(mol, &cart_hess, &apt)?;
    let thermo = compute_thermochemistry(
        mol,
        e0,
        &vib,
        opts.temperature_k,
        opts.pressure_pa,
        opts.symmetry_number,
    );
    Ok(FrequencyResult {
        energy_hartree: e0,
        cartesian_hessian: cart_hess,
        apt,
        vibrational_analysis: vib,
        thermochemistry: thermo,
    })
}
