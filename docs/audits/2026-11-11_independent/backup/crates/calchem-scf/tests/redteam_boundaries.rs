use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_model::{Atom, Molecule, ScfOptions};

#[test]
fn diis_coefficients_are_invariant_under_residual_scale() {
    use calchem_tensor::Matrix2D;
    let mut values = Vec::new();
    for scale in [1.0_f64, 1e-12] {
        let mut state = calchem_scf::DiisState::new(4);
        let mut f1 = Matrix2D::zeros(3, 3);
        f1.set(0, 0, 1.0);
        let mut f2 = Matrix2D::zeros(3, 3);
        f2.set(0, 0, 3.0);
        let mut e1 = Matrix2D::zeros(3, 3);
        e1.set(0, 1, scale);
        e1.set(1, 0, -scale);
        let mut e2 = Matrix2D::zeros(3, 3);
        e2.set(0, 2, scale);
        e2.set(2, 0, -scale);
        state.push_and_extrapolate(&f1, &e1);
        values.push(state.push_and_extrapolate(&f2, &e2).get(0, 0));
    }
    assert!((values[0] - 2.0).abs() < 1e-14);
    assert!((values[1] - values[0]).abs() < 1e-14);
}

#[test]
fn range_separated_cam_b3lyp_and_wb97xd_succeed() {
    let mol = Molecule::new(vec![Atom::from_bohr("He", [0.0; 3]).unwrap()], 0, 1);
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
    for xc in [XcFunctional::CamB3lyp, XcFunctional::Wb97xd] {
        let res_rks = calchem_scf::rks(&mol, &bs, xc, GridLevel::Coarse, &ScfOptions::default())
            .expect("Range-separated RKS should converge on He");
        assert!(res_rks.converged);
        assert!(res_rks.total_energy_hartree < -2.7 && res_rks.total_energy_hartree > -3.1);

        let res_uks = calchem_scf::uks(&mol, &bs, xc, GridLevel::Coarse, &ScfOptions::default())
            .expect("Range-separated UKS should converge on He");
        assert!(res_uks.converged);
        assert!((res_rks.total_energy_hartree - res_uks.total_energy_hartree).abs() < 1e-6);
    }
}

#[test]
fn memory_and_invalid_scf_options_are_rejected_before_integrals() {
    let mol = Molecule::new(
        (0..256)
            .map(|i| Atom::from_bohr("He", [i as f64 * 4.0, 0.0, 0.0]).unwrap())
            .collect(),
        0,
        1,
    );
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
    let error = calchem_scf::validate_resources_and_options(&bs, &ScfOptions::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("memory estimate"), "{error}");
    let invalid = ScfOptions {
        conv_energy: f64::INFINITY,
        ..Default::default()
    };
    assert!(calchem_scf::validate_resources_and_options(&bs, &invalid).is_err());
}

#[test]
fn singular_basis_is_rejected_instead_of_zero_energy_ghost_orbitals() {
    let mol = Molecule::new(vec![Atom::from_bohr("He", [0.0; 3]).unwrap()], 0, 1);
    let bs=BasisSet::from_json_str(r#"{"name":"duplicate-s","elements":{"He":[{"shell_type":"s","exponents":[1.0],"coefficients":[1.0]},{"shell_type":"s","exponents":[1.0],"coefficients":[1.0]}]}}"#,&mol).unwrap();
    assert!(calchem_scf::rhf(&mol, &bs, &ScfOptions::default()).is_err());
}
