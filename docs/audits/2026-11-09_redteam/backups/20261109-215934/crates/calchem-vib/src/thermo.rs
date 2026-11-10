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
/// 1 amu * Bohr^2 in kg * m^2.
const AMU_BOHR2_TO_KG_M2: f64 = 4.6509653e-48;
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
