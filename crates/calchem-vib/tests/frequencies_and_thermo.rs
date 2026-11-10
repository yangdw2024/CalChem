use calchem_dft::{GridLevel, XcFunctional};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_vib::{
    FreqOptions, VibMethod, compute_frequencies_and_thermo,
};

fn water_mol() -> Molecule {
    // Near-equilibrium H2O geometry
    Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
        ],
        0,
        1,
    )
}

#[test]
fn test_water_b3lyp_vibrational_frequencies_and_thermo() {
    let mol = water_mol();
    let opts = FreqOptions {
        step_bohr: 0.002,
        grid_level: GridLevel::Coarse,
        temperature_k: 298.15,
        pressure_pa: 101325.0,
        symmetry_number: Some(2), // C2v water
        scf_options: ScfOptions {
            conv_energy: 1e-11,
            conv_density: 1e-9,
            ..Default::default()
        },
    };

    let res = compute_frequencies_and_thermo(
        &mol,
        "6-31g",
        VibMethod::Rks(XcFunctional::B3lyp),
        &opts,
    )
    .unwrap();

    let vib = &res.vibrational_analysis;
    let thermo = &res.thermochemistry;

    // Water is non-linear: exactly 3N - 6 = 3 vibrational modes
    assert!(!vib.is_linear);
    assert_eq!(vib.n_trans_rot, 6);
    assert_eq!(vib.n_vib, 3);
    assert_eq!(vib.modes.len(), 3);

    for m in &vib.modes {
        println!(
            "Mode #{}: freq = {:.2} cm^-1, red_mass = {:.3} amu, k = {:.3} mDyn/A, IR = {:.3} km/mol",
            m.mode_index, m.frequency_cm, m.reduced_mass_amu, m.force_constant_m_dyn_a, m.ir_intensity_km_mol
        );
    }

    // Verify 6 translation/rotation modes have tiny residual eigenvalues
    for &res_lam in &vib.trans_rot_residuals {
        assert!(
            res_lam.abs() < 5e-4,
            "Translation/rotation residual eigenvalue too large: {res_lam:.3e}"
        );
    }

    // Verify all 3 modes are real (local minimum on PES)
    for mode in &vib.modes {
        assert!(!mode.is_imaginary, "Mode {} should be real, got freq={}", mode.mode_index, mode.frequency_cm);
        assert!(mode.frequency_cm > 1000.0, "Frequency should be > 1000 cm^-1, got {}", mode.frequency_cm);
        assert!(mode.reduced_mass_amu > 0.9 && mode.reduced_mass_amu < 2.5);
        assert!(mode.ir_intensity_km_mol >= 0.0);
    }

    // Mode 1: H-O-H bend (expected ~1600 - 1800 cm^-1)
    let bend = &vib.modes[0];
    assert!(
        (1500.0..=1900.0).contains(&bend.frequency_cm),
        "Bending mode frequency expected in [1500, 1900] cm^-1, got {:.1}",
        bend.frequency_cm
    );
    assert!(bend.ir_intensity_km_mol > 1.0, "Water bending mode has strong IR intensity");
    // Modes 2 & 3: Symmetric & Asymmetric stretches (expected ~3700 - 4100 cm^-1)
    let sym_str = &vib.modes[1];
    let asym_str = &vib.modes[2];
    assert!(
        bend.ir_intensity_km_mol > sym_str.ir_intensity_km_mol,
        "Water bending mode IR intensity should exceed symmetric stretch"
    );
    assert!(
        (3500.0..=4200.0).contains(&sym_str.frequency_cm),
        "Symmetric stretch expected in [3500, 4200] cm^-1, got {:.1}",
        sym_str.frequency_cm
    );
    assert!(
        (3600.0..=4300.0).contains(&asym_str.frequency_cm),
        "Asymmetric stretch expected in [3600, 4300] cm^-1, got {:.1}",
        asym_str.frequency_cm
    );
    assert!(asym_str.frequency_cm >= sym_str.frequency_cm);

    // Verify RRHO Thermochemistry
    assert!(thermo.zpve_hartree > 0.018 && thermo.zpve_hartree < 0.026);
    assert!(thermo.zpve_kcal_mol > 11.0 && thermo.zpve_kcal_mol < 16.0);
    assert!(thermo.e_thermal_correction_hartree > thermo.zpve_hartree);
    assert!(thermo.enthalpy_correction_hartree > thermo.e_thermal_correction_hartree);
    assert!(thermo.s_total_cal_mol_k > 40.0 && thermo.s_total_cal_mol_k < 55.0);
    assert!(thermo.cp_cal_mol_k > thermo.cv_cal_mol_k);
    assert!((thermo.cp_cal_mol_k - thermo.cv_cal_mol_k - 1.9872).abs() < 1e-3);
    assert!(thermo.gibbs_free_energy_hartree < thermo.enthalpy_hartree);
}

#[test]
fn test_diatomic_linear_hbr_ecp_frequencies() {
    // HBr / LANL2DZ with scalar relativistic ECP on Br
    let hbr = Molecule::new(
        vec![
            Atom::from_angstrom("Br", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.0, 1.414]).unwrap(),
        ],
        0,
        1,
    );

    let opts = FreqOptions {
        step_bohr: 0.002,
        grid_level: GridLevel::Coarse,
        temperature_k: 298.15,
        pressure_pa: 101325.0,
        symmetry_number: Some(1),
        scf_options: ScfOptions {
            conv_energy: 1e-11,
            conv_density: 1e-9,
            ..Default::default()
        },
    };

    let res = compute_frequencies_and_thermo(
        &hbr,
        "lanl2dz",
        VibMethod::Rks(XcFunctional::B3lyp),
        &opts,
    )
    .unwrap();

    let vib = &res.vibrational_analysis;
    assert!(vib.is_linear);
    assert_eq!(vib.n_trans_rot, 5);
    assert_eq!(vib.n_vib, 1);
    assert_eq!(vib.modes.len(), 1);

    let stretch = &vib.modes[0];
    assert!(!stretch.is_imaginary);
    // H-Br stretch expected in 2300 - 2800 cm^-1
    assert!(
        (2200.0..=2900.0).contains(&stretch.frequency_cm),
        "HBr stretch expected in [2200, 2900] cm^-1, got {:.1}",
        stretch.frequency_cm
    );
    // Reduced mass of H-Br should be close to H mass (~1.0 amu)
    assert!((stretch.reduced_mass_amu - 1.0).abs() < 0.05);
}
