use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rhf, rks, uks};

#[test]
fn test_rij_density_fitting_rks_pure_dft_and_b3lyp_on_water() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();

    // 1. Pure DFT (LDA XcFunctional::Lda)
    let opts_exact = ScfOptions {
        conv_energy: 1e-8,
        conv_density: 1e-6,
        density_fitting: false,
        ..Default::default()
    };
    let opts_rij = ScfOptions {
        conv_energy: 1e-8,
        conv_density: 1e-6,
        density_fitting: false,
        ..Default::default()
    };

    let wf_exact_lda = rks(&mol, &bs, XcFunctional::Svwn5, GridLevel::Coarse, &opts_exact).unwrap();
    let wf_rij_lda = rks(&mol, &bs, XcFunctional::Svwn5, GridLevel::Coarse, &opts_rij).unwrap();

    assert!(wf_exact_lda.converged);
    assert!(wf_rij_lda.converged);

    let d_e_lda = (wf_rij_lda.total_energy_hartree - wf_exact_lda.total_energy_hartree).abs();
    assert!(
        d_e_lda < 1e-3,
        "RI-J LDA energy diff vs exact 4-center should be < 1e-3 Eh, got {d_e_lda:.3e}"
    );

    // 2. Hybrid B3LYP (RI-J Coulomb + Direct Exchange)
    let opts_b3lyp = ScfOptions {
        conv_energy: 1e-6,
        conv_density: 1e-6,
        density_fitting: true,
        ..Default::default()
    };
    let wf_exact_b3lyp = rks(&mol, &bs, XcFunctional::B3lyp, GridLevel::Coarse, &opts_exact).unwrap();
    let wf_rij_b3lyp = rks(&mol, &bs, XcFunctional::B3lyp, GridLevel::Coarse, &opts_b3lyp).unwrap();

    assert!(wf_exact_b3lyp.converged);
    assert!(wf_rij_b3lyp.converged);

    let d_e_b3lyp = (wf_rij_b3lyp.total_energy_hartree - wf_exact_b3lyp.total_energy_hartree).abs();
    assert!(
        d_e_b3lyp < 2e-3,
        "RI-J B3LYP energy diff vs exact 4-center should be < 2e-3 Eh, got {d_e_b3lyp:.3e}"
    );
}

#[test]
fn test_rij_density_fitting_rhf_and_uks_radical() {
    let mol_singlet = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let bs_h2 = BasisSet::from_name("sto-3g", &mol_singlet).unwrap();

    let opts_rij = ScfOptions {
        conv_energy: 1e-8,
        conv_density: 1e-6,
        density_fitting: true,
        ..Default::default()
    };

    let wf_rhf = rhf(&mol_singlet, &bs_h2, &opts_rij).unwrap();
    assert!(wf_rhf.converged);
    assert!((wf_rhf.total_energy_hartree - (-1.1167)).abs() < 2e-3);

    // Open-shell radical doublet: water cation H2O+ (mult 2, charge 1)
    let mol_cat = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        1,
        2,
    );
    let bs_cat = BasisSet::from_name("sto-3g", &mol_cat).unwrap();
    let wf_cat = uks(&mol_cat, &bs_cat, XcFunctional::Svwn5, GridLevel::Coarse, &opts_rij).unwrap();
    assert!(wf_cat.converged);
    assert!(
        wf_cat.total_energy_hartree < -74.0,
        "Expected H2O+ energy < -74.0, got {}",
        wf_cat.total_energy_hartree
    );
}
