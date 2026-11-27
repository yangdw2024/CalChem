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

pub use hessian::{compute_cartesian_hessian_and_apt, FreqOptions, VibMethod};
pub use masses::{
    atomic_mass_amu, GAS_CONSTANT_CAL_MOL_K, GAS_CONSTANT_HARTREE_K, GAS_CONSTANT_J_MOL_K,
    HARTREE_TO_EV, HARTREE_TO_KCAL_MOL, HESSIAN_EVAL_TO_INV_CM, IR_INTENSITY_PREFACTOR_KM_MOL,
};
pub use modes::{analyze_vibrations, NormalMode, VibrationalAnalysis};
pub use thermo::{compute_thermochemistry, compute_thermochemistry_checked, Thermochemistry};

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
    pub gpu_kernel_invocations: usize,
    pub gpu_fallback_invocations: usize,
}

/// Runs full vibrational frequency analysis and RRHO thermochemistry on a molecule.
pub fn compute_frequencies_and_thermo(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    opts: &FreqOptions,
) -> Result<FrequencyResult, CalcError> {
    Ok(compute_frequencies_with_model(mol, basis_name, method, opts, None, 1)?.0)
}

pub fn compute_frequencies_with_model(
    mol: &Molecule, basis_name: &str, method: VibMethod, opts: &FreqOptions,
    dispersion: Option<&calchem_dft::D3Model>, displacement_jobs: usize,
) -> Result<(FrequencyResult, calchem_model::ScfSummary, f64), CalcError> {
    compute_frequencies_with_checkpoint(mol, basis_name, method, opts, dispersion, displacement_jobs, None, None)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_frequencies_with_checkpoint(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    opts: &FreqOptions,
    dispersion: Option<&calchem_dft::D3Model>,
    displacement_jobs: usize,
    checkpoint_dir: Option<&std::path::Path>,
    model_fingerprint: Option<&str>,
) -> Result<(FrequencyResult, calchem_model::ScfSummary, f64), CalcError> {
    let (cart_hess, apt, center, correction, gpu_kernel_invocations, gpu_fallback_invocations) =
        hessian::compute_hessian_with_checkpoint_and_telemetry(
            mol,
            basis_name,
            method,
            opts,
            dispersion,
            displacement_jobs,
            checkpoint_dir,
            model_fingerprint,
        )?;
    let e0 = center.total_energy_hartree + correction;
    let vib = analyze_vibrations(mol, &cart_hess, &apt)?;
    let thermo = compute_thermochemistry_checked(
        mol,
        e0,
        &vib,
        opts.temperature_k,
        opts.pressure_pa,
        opts.symmetry_number,
    )?;
    Ok((FrequencyResult {
        energy_hartree: e0,
        cartesian_hessian: cart_hess,
        apt,
        vibrational_analysis: vib,
        thermochemistry: thermo,
        gpu_kernel_invocations,
        gpu_fallback_invocations,
    }, center, correction))
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_model::Atom;
    use std::sync::Mutex;

    static VIB_ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_freq_gpu_telemetry_accumulates_displacements_and_disable_gpu_fallbacks() {
        let _guard = VIB_ENV_LOCK.lock().unwrap();
        let h2 = Molecule::new(
            vec![
                Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
            ],
            0,
            1,
        );
        let mut opts = FreqOptions::default();
        opts.scf_options.use_gpu = true;

        let (res_gpu, center, _) =
            compute_frequencies_with_model(&h2, "6-31g(d)", VibMethod::Rhf, &opts, None, 1)
                .unwrap();
        if res_gpu.gpu_kernel_invocations > 0 {
            assert_eq!(res_gpu.gpu_fallback_invocations, 0);
            assert!(
                res_gpu.gpu_kernel_invocations > center.diagnostics.gpu_kernel_invocations,
                "Expected total freq GPU kernels ({}) > center SCF kernels ({})",
                res_gpu.gpu_kernel_invocations,
                center.diagnostics.gpu_kernel_invocations
            );
        } else {
            assert!(res_gpu.gpu_fallback_invocations > 0);
        }

        let prev = std::env::var("CALCHEM_DISABLE_GPU").ok();
        std::env::set_var("CALCHEM_DISABLE_GPU", "1");
        let res_fb =
            compute_frequencies_with_model(&h2, "6-31g(d)", VibMethod::Rhf, &opts, None, 1);
        match prev {
            Some(v) => std::env::set_var("CALCHEM_DISABLE_GPU", v),
            None => std::env::remove_var("CALCHEM_DISABLE_GPU"),
        }
        let (res_fb, _, _) = res_fb.unwrap();
        assert_eq!(res_fb.gpu_kernel_invocations, 0);
        assert!(res_fb.gpu_fallback_invocations > 0);
    }
}
