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
        assert!((s_self - 1.0).abs() < 1e-15, "lmn={:?}, s_self={}", lmn, s_self);
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
