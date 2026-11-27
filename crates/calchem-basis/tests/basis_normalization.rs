use calchem_basis::{
    normalized_contraction, normalized_primitive_self_overlap, primitive_norm, BasisSet, ShellType,
};
use calchem_model::{Atom, CalcError, Molecule};
use std::f64::consts::PI;

#[test]
fn test_primitive_normalization_s_p_d() {
    let alpha = 1.25_f64;

    // s: (0,0,0) -> (2*alpha/pi)^0.75
    let ns_expected = (2.0 * alpha / PI).powf(0.75);
    assert!((primitive_norm(alpha, [0, 0, 0]) - ns_expected).abs() < 1e-15);

    // p: (1,0,0), (0,1,0), (0,0,1) -> (2*alpha/pi)^0.75 * sqrt(4*alpha)
    let np_expected = (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).sqrt();
    for &lmn in &[[1, 0, 0], [0, 1, 0], [0, 0, 1]] {
        assert!((primitive_norm(alpha, lmn) - np_expected).abs() < 1e-15);
    }

    // d axial: (2,0,0), (0,2,0), (0,0,2) -> (2*alpha/pi)^0.75 * (4*alpha) / sqrt(3)
    let nd_axial_expected = (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha) / 3.0_f64.sqrt();
    for &lmn in &[[2, 0, 0], [0, 2, 0], [0, 0, 2]] {
        assert!((primitive_norm(alpha, lmn) - nd_axial_expected).abs() < 1e-15);
    }

    // d off-diagonal: (1,1,0), (1,0,1), (0,1,1) -> (2*alpha/pi)^0.75 * (4*alpha)
    let nd_off_expected = (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha);
    for &lmn in &[[1, 1, 0], [1, 0, 1], [0, 1, 1]] {
        assert!((primitive_norm(alpha, lmn) - nd_off_expected).abs() < 1e-15);
    }

    // Every normalized primitive has exact unit self-overlap <chi | chi> = 1
    for &lmn in &[
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 2, 0],
        [0, 1, 1],
        [0, 0, 2],
    ] {
        let s_self = normalized_primitive_self_overlap(alpha, alpha, lmn);
        assert!(
            (s_self - 1.0).abs() < 1e-15,
            "lmn={:?}, s_self={}",
            lmn,
            s_self
        );
    }
}

#[test]
fn test_unknown_shell_type_rejected() {
    assert_eq!(ShellType::from_str("f").unwrap(), ShellType::F);
    assert!(matches!(
        ShellType::from_str("g"),
        Err(CalcError::UnsupportedBasis(_))
    ));
    assert!(matches!(
        ShellType::from_str("sp"),
        Err(CalcError::UnsupportedBasis(_))
    ));
    assert!(matches!(
        ShellType::from_str("unknown"),
        Err(CalcError::UnsupportedBasis(_))
    ));
}

#[test]
fn test_sto3g_and_631g_contraction_self_overlap_is_one() {
    let symbols = ["H", "He", "C", "N", "O", "F", "Ne"];
    for sym in symbols {
        let mol = Molecule::new(vec![Atom::from_bohr(sym, [0.0, 0.0, 0.0]).unwrap()], 0, 1);
        for basis_name in ["sto-3g", "6-31g"] {
            let bs = BasisSet::from_name(basis_name, &mol).unwrap();
            for shell in &bs.shells {
                for &lmn in shell.components() {
                    let mut s_self = 0.0;
                    for p in &shell.primitives {
                        for q in &shell.primitives {
                            s_self += p.coeff
                                * q.coeff
                                * normalized_primitive_self_overlap(p.exponent, q.exponent, lmn);
                        }
                    }
                    assert!(
                        (s_self - 1.0).abs() < 1e-14,
                        "Self-overlap != 1 for {} in {}: {:?}",
                        sym,
                        basis_name,
                        s_self
                    );
                }
            }
        }
    }
}

#[test]
fn test_d_shell_contraction_self_overlap() {
    let exps = [1.2_f64, 0.45_f64];
    let raw_coeffs = [0.6_f64, 0.5_f64];
    let norm_coeffs = normalized_contraction(&exps, &raw_coeffs, [2, 0, 0]).unwrap();
    for &lmn in ShellType::D.components() {
        let mut s_self = 0.0;
        for i in 0..exps.len() {
            for j in 0..exps.len() {
                s_self += norm_coeffs[i]
                    * norm_coeffs[j]
                    * normalized_primitive_self_overlap(exps[i], exps[j], lmn);
            }
        }
        assert!((s_self - 1.0).abs() < 1e-14);
    }
}

#[test]
fn test_f_shell_primitive_and_contraction_normalization() {
    let alpha = 1.15_f64;
    let nf_axial = (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).powf(1.5) / 15.0_f64.sqrt();
    for &lmn in &[[3, 0, 0], [0, 3, 0], [0, 0, 3]] {
        assert!((primitive_norm(alpha, lmn) - nf_axial).abs() < 1e-15);
    }
    let nf_210 = (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).powf(1.5) / 3.0_f64.sqrt();
    assert!((primitive_norm(alpha, [2, 1, 0]) - nf_210).abs() < 1e-15);
    let nf_111 = (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).powf(1.5);
    assert!((primitive_norm(alpha, [1, 1, 1]) - nf_111).abs() < 1e-15);

    for &lmn in ShellType::F.components() {
        let s_self = normalized_primitive_self_overlap(alpha, alpha, lmn);
        assert!((s_self - 1.0).abs() < 1e-15);
    }

    let exps = [1.8_f64, 0.55_f64];
    let raw_coeffs = [0.7_f64, 0.4_f64];
    let norm_coeffs = normalized_contraction(&exps, &raw_coeffs, [3, 0, 0]).unwrap();
    for &lmn in ShellType::F.components() {
        let mut s_self = 0.0;
        for i in 0..exps.len() {
            for j in 0..exps.len() {
                s_self += norm_coeffs[i]
                    * norm_coeffs[j]
                    * normalized_primitive_self_overlap(exps[i], exps[j], lmn);
            }
        }
        assert!((s_self - 1.0).abs() < 1e-14);
    }
}

#[test]
fn test_basis_aliases_unknown_missing_element_and_cartesian_6d_10f_counts() {
    let h2o = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    assert_eq!(BasisSet::canonical_name(" 6-31G* "), Some("6-31g(d)"));
    assert_eq!(BasisSet::canonical_name("6-31g**"), Some("6-31g(d,p)"));
    assert_eq!(BasisSet::canonical_name("lanl2dz*"), Some("lanl2dzdp"));
    assert_eq!(BasisSet::canonical_name("cc-pvtz"), None);

    assert!(matches!(
        BasisSet::from_name("cc-pvtz", &h2o),
        Err(CalcError::UnsupportedBasis(_))
    ));
    let bs_gdp = BasisSet::from_name("6-31g**", &h2o).unwrap();
    assert_eq!(bs_gdp.n_ao, 25);

    let fe = Molecule::new(vec![Atom::from_bohr("Fe", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
    let bs_fe = BasisSet::from_name("6-31g(d)", &fe).unwrap();
    assert!(bs_fe.shells.iter().any(|s| s.shell_type == ShellType::F && s.n_functions() == 10));

    let xe = Molecule::new(vec![Atom::from_bohr("Xe", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
    assert!(matches!(
        BasisSet::from_name("6-31g(d,p)", &xe),
        Err(CalcError::UnsupportedBasis(_))
    ));
    let au = Molecule::new(vec![Atom::from_bohr("Au", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
    assert!(matches!(
        BasisSet::from_name("sto-3g", &au),
        Err(CalcError::UnsupportedBasis(_))
    ));
}

#[test]
fn test_load_def2_jfit_elements_and_out_of_range() {
    use calchem_basis::load_def2_jfit_for_element;

    for z in [1u8, 6, 7, 8, 9, 16, 17, 35, 53] {
        let shells = load_def2_jfit_for_element(z).expect("element must exist in def2-universal-jfit");
        assert!(!shells.is_empty());
        assert!(shells.iter().all(|(st, exps, coeffs)| {
            st.angular_momentum() <= 3 && !exps.is_empty() && exps.len() == coeffs.len()
        }));
    }
    for z in [0u8, 2, 26, 255] {
        assert!(load_def2_jfit_for_element(z).is_none(), "Z={z} should be None");
    }
}

#[test]
fn test_valence_ecp_and_geometry_validation_errors() {
    let empty_mol = Molecule::new(vec![], 0, 1);
    assert!(matches!(
        BasisSet::from_name("sto-3g", &empty_mol),
        Err(CalcError::InvalidMolecule(_))
    ));

    let close_h2 = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 5e-4]).unwrap(),
        ],
        0,
        1,
    );
    let bs_close = BasisSet::sto_3g(&close_h2).unwrap();
    assert!(matches!(
        bs_close.validate_rhf_valence(&close_h2),
        Err(CalcError::InvalidMolecule(_))
    ));

    let hi = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("I", [0.0, 0.0, 3.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hi = BasisSet::from_name("lanl2dzdp", &hi).unwrap();
    assert!(bs_hi.has_ecp());
    assert_eq!(bs_hi.effective_nuclear_charge(&hi, 0), 1.0);
    assert_eq!(bs_hi.effective_nuclear_charge(&hi, 1), 7.0);
    assert!((bs_hi.nuclear_repulsion_energy(&hi) - 7.0 / 3.0).abs() < 1e-14);
    assert_eq!(bs_hi.validate_rhf_valence(&hi).unwrap(), (4, 4));

    let bad_ecp_json = r#"{
        "name": "bad-ecp",
        "elements": { "H": [{"shell_type": "s", "exponents": [1.0], "coefficients": [1.0]}] },
        "ecp": { "H": {"core_electrons": 1, "l_max": 0, "local": []} }
    }"#;
    let h_atom = Molecule::new(vec![Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap()], 0, 2);
    assert!(matches!(
        BasisSet::from_json_str(bad_ecp_json, &h_atom),
        Err(CalcError::InvalidMolecule(_))
    ));
}
