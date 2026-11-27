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
fn cam_b3lyp_and_wb97xd_succeed_natively() {
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
    let auto = ScfOptions::default();
    assert!(calchem_scf::should_use_direct(&bs, &auto));
    assert!(calchem_scf::validate_resources_and_options(&bs, &auto).is_ok());
    let stored = ScfOptions { direct: Some(false), ..auto };
    let error = calchem_scf::validate_resources_and_options(&bs, &stored).unwrap_err();
    assert!(matches!(error, calchem_model::CalcError::ResourceLimit(_)), "{error}");
    let invalid = ScfOptions {
        conv_energy: f64::INFINITY,
        ..Default::default()
    };
    assert!(calchem_scf::validate_resources_and_options(&bs, &invalid).is_err());
}

#[test]
fn dense_open_shell_memory_is_not_admitted_using_sparse_rks_estimate() {
    let mol = Molecule::new(vec![Atom::from_bohr("He", [0.0; 3]).unwrap()], 0, 1);
    let mut basis = BasisSet::from_name("sto-3g", &mol).unwrap();
    // Dimension-only admission probe: no large tensor or integral is allocated.
    basis.n_ao = 200;
    assert!(calchem_scf::validate_resources_and_options(&basis, &ScfOptions::default()).is_ok());
    assert!(calchem_scf::validate_open_shell_stored_resources(&basis, &ScfOptions::default())
        .unwrap_err().to_string().contains("dense stored-ERI"));
}

#[test]
fn singular_basis_is_rejected_instead_of_zero_energy_ghost_orbitals() {
    let mol = Molecule::new(vec![Atom::from_bohr("He", [0.0; 3]).unwrap()], 0, 1);
    let bs=BasisSet::from_json_str(r#"{"name":"duplicate-s","elements":{"He":[{"shell_type":"s","exponents":[1.0],"coefficients":[1.0]},{"shell_type":"s","exponents":[1.0],"coefficients":[1.0]}]}}"#,&mol).unwrap();
    assert!(calchem_scf::rhf(&mol, &bs, &ScfOptions::default()).is_err());
}

#[test]
fn disk_diis_io_failure_is_propagated_by_scf_solvers() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
    let opts = ScfOptions {
        diis_disk: true,
        diis_dir: Some("NUL:\\invalid\\calchem_scf_diis_fail".into()),
        ..Default::default()
    };
    let err_rhf = calchem_scf::rhf(&mol, &bs, &opts).unwrap_err();
    let err_uhf = calchem_scf::uhf(&mol, &bs, &opts).unwrap_err();
    let err_rks =
        calchem_scf::rks(&mol, &bs, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap_err();
    let err_uks =
        calchem_scf::uks(&mol, &bs, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap_err();
    for (name, err) in [
        ("RHF", err_rhf),
        ("UHF", err_uhf),
        ("RKS", err_rks),
        ("UKS", err_uks),
    ] {
        assert!(
            err.to_string().contains("DIIS disk spill I/O error"),
            "Expected DIIS I/O error from {name}, got: {err}"
        );
    }
}

#[test]
fn incremental_fock_drift_reset_and_stage_transition_match_full_build() {
    use calchem_scf::{RksFockAccumulator, RksFockBuilder, UksFockAccumulator, UksFockBuilder};
    use calchem_tensor::Matrix2D;

    let mol = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.757, 0.586]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.757, 0.586]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let n = bs.n_ao;
    let r_builder = RksFockBuilder::new_with_options(&mol, &bs, true, false, false, None).unwrap();
    let u_builder = UksFockBuilder::new_with_options(&mol, &bs, true, false, false, None).unwrap();

    let mut p = Matrix2D::zeros(n, n);
    for i in 0..n {
        p.set(i, i, 0.2);
    }
    let mut acc = RksFockAccumulator::new(n, &p, 1e-8);
    assert!(!acc.in_tight_pass());
    for iter in 1..=10usize {
        if iter > 1 {
            p.set(0, 0, p.get(0, 0) + 1e-5);
        }
        let tight_base = acc.tight_base();
        let g = acc.update(&r_builder, &bs, &p, 0.20, None, iter, true, 1e-3, 1e-2);
        let g_ref = r_builder.build_2e_with_thresh(&bs, &p, 0.20, None, tight_base);
        let max_err = g.data.iter().zip(&g_ref.data).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-6, "RKS iter {iter} drift err={max_err:.3e}");
    }
    let tight_base = acc.tight_base();
    let g_tight = acc.update(&r_builder, &bs, &p, 0.20, None, 11, true, 1e-5, 1e-4).clone();
    assert!(acc.in_tight_pass());
    let g_ref = r_builder.build_2e_with_thresh(&bs, &p, 0.20, None, tight_base);
    let tight_err = g_tight.data.iter().zip(&g_ref.data).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(tight_err < 1e-12, "Stage-1 tight err={tight_err:.3e}");

    let mut pa = Matrix2D::zeros(n, n);
    let mut pb = Matrix2D::zeros(n, n);
    for i in 0..n {
        pa.set(i, i, 0.12);
        pb.set(i, i, 0.08);
    }
    let mut u_acc = UksFockAccumulator::new(n, &pa, &pb, 1e-8);
    for iter in 1..=10usize {
        if iter > 1 {
            pa.set(0, 0, pa.get(0, 0) + 1e-5);
        }
        let u_tight_base = u_acc.tight_base();
        let (ga, gb) = u_acc.update(&u_builder, &bs, &pa, &pb, 0.20, None, iter, true, 1e-3, 1e-2);
        let (ga_ref, gb_ref) = u_builder.build_2e_with_thresh(&bs, &pa, &pb, 0.20, None, u_tight_base);
        let err_a = ga.data.iter().zip(&ga_ref.data).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let err_b = gb.data.iter().zip(&gb_ref.data).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err_a < 1e-6 && err_b < 1e-6, "UKS iter {iter} drift err=({err_a:.3e},{err_b:.3e})");
    }
}

#[test]
fn sad_guess_rejects_zero_electrons_and_handles_no_diis_scf() {
    let h2 = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("sto-3g", &h2).unwrap();
    let s = calchem_int::compute_overlap(&bs);
    assert!(calchem_scf::sad_guess(&h2, &bs, &s, 0).is_err());
    assert!(calchem_scf::sad_guess_uhf(&h2, &bs, &s, 0, 0).is_err());

    let no_diis = ScfOptions {
        use_diis: false,
        damping: 0.2,
        ..Default::default()
    };
    let wfn = calchem_scf::rks(&h2, &bs, XcFunctional::Pbe, GridLevel::Coarse, &no_diis).unwrap();
    assert!(wfn.converged);
}
