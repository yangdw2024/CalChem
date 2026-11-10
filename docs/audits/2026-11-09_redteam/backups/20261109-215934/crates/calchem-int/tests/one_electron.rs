use calchem_basis::{ normalized_contraction, primitive_norm, BasisSet, Primitive, Shell, ShellType };
use calchem_int::{
    compute_kinetic, compute_nuclear, compute_overlap, shell_kinetic_block, shell_nuclear_block,
    shell_overlap_block,
};
use calchem_model::{Atom, Molecule};

#[test]
fn test_h2_sto3g_one_electron_reference() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let s = compute_overlap(&basis);
    let t = compute_kinetic(&basis);
    let v = compute_nuclear(&basis, &mol);

    assert!((s.get(0, 0) - 1.0).abs() < 1e-14);
    assert!((s.get(1, 1) - 1.0).abs() < 1e-14);
    assert!((s.get(0, 1) - 0.659318206134864).abs() < 1e-12);
    assert!((s.get(1, 0) - 0.659318206134864).abs() < 1e-12);

    assert!((t.get(0, 0) - 0.760031883566609).abs() < 1e-12);
    // Closed-form s/s kinetic integral, mpmath dps=50 (tools/reference/gen_integral_refs.py).
    // The former reference 0.236454655845882 was off by 1.34e-10.
    assert!((t.get(0, 1) - 0.2364546559796740044).abs() < 1e-12);

    // Closed-form s/s nuclear attraction (both nuclei), mpmath dps=50 with quadrature Boys
    // (tools/reference/gen_integral_refs.py). Former values -1.880440891908111 /
    // -1.194834622962271 deviated by 5.7e-10 / 2.6e-9 (legacy Python Boys error).
    assert!((v.get(0, 0) - (-1.8804408924734455503)).abs() < 1e-12);
    assert!((v.get(0, 1) - (-1.1948346203692893879)).abs() < 1e-12);
}

#[test]
fn test_h2o_sto3g_one_electron_symmetry_and_reference() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let s = compute_overlap(&basis);
    let t = compute_kinetic(&basis);
    let v = compute_nuclear(&basis, &mol);

    assert_eq!(basis.n_ao, 7);
    assert!(s.sym_frobenius_error() < 1e-14);
    assert!(t.sym_frobenius_error() < 1e-14);
    assert!(v.sym_frobenius_error() < 1e-13);

    for i in 0..basis.n_ao {
        assert!((s.get(i, i) - 1.0).abs() < 1e-13, "S[{i},{i}] = {}", s.get(i, i));
        assert!(t.get(i, i) > 0.0, "T[{i},{i}] = {}", t.get(i, i));
        assert!(v.get(i, i) < 0.0, "V[{i},{i}] = {}", v.get(i, i));
    }

    // Compare against exact Python reference values for H2O/STO-3G:
    let s_row0_ref = [
        1.000000000000000,
        0.236703926750692,
        0.000000000000000,
        0.000000000000000,
        0.000000000000000,
        0.085690824055376,
        0.097488180540617,
    ];
    for (j, &expected) in s_row0_ref.iter().enumerate() {
        assert!((s.get(0, j) - expected).abs() < 1e-12);
    }

    let t_diag_ref = [
        29.003204305138183,
        0.808127955339426,
        2.528731199156927,
        2.528731199156927,
        2.528731199156928,
        0.760031883566609,
        0.760031883566609,
    ];
    for (i, &expected) in t_diag_ref.iter().enumerate() {
        assert!(
            (t.get(i, i) - expected).abs() < 1e-11,
            "T[{i},{i}] got {}, expected {}",
            t.get(i, i),
            expected
        );
    }

    let v_diag_ref = [
        -62.072885242882819,
        -10.430686918351427,
        -10.375796985370741,
        -10.480786072935665,
        -10.219140223130537,
        -6.779615288341706,
        -7.085094157869087,
    ];
    for (i, &expected) in v_diag_ref.iter().enumerate() {
        assert!(
            (v.get(i, i) - expected).abs() < 1e-11,
            "V[{i},{i}] got {}, expected {}",
            v.get(i, i),
            expected
        );
    }

    // Off-diagonal p-s nuclear attraction matrix elements
    assert!((v.get(2, 5) - (-0.084699603336947)).abs() < 1e-12);
    assert!((v.get(3, 5) - (-3.643299760159863)).abs() < 1e-12);
}

#[test]
fn test_cartesian_d_shell_one_electron_reference() {
    let exps_d = [0.8_f64];
    let coeffs_d = normalized_contraction(&exps_d, &[1.0], [2, 0, 0]).unwrap();
    let sh_d = Shell {
        atom_index: 0,
        center: [0.0, 0.0, 0.0],
        shell_type: ShellType::D,
        primitives: vec![Primitive {
            exponent: 0.8,
            coeff: coeffs_d[0],
            raw_coeff: 1.0,
            norm: primitive_norm(0.8, [2, 0, 0]),
        }],
        ao_offset: 0,
    };

    let exps_s = [0.5_f64];
    let coeffs_s = normalized_contraction(&exps_s, &[1.0], [0, 0, 0]).unwrap();
    let sh_s = Shell {
        atom_index: 1,
        center: [0.0, 0.0, 1.2],
        shell_type: ShellType::S,
        primitives: vec![Primitive {
            exponent: 0.5,
            coeff: coeffs_s[0],
            raw_coeff: 1.0,
            norm: primitive_norm(0.5, [0, 0, 0]),
        }],
        ao_offset: 6,
    };

    let s_dd = shell_overlap_block(&sh_d, &sh_d);
    for i in 0..6 {
        assert!((s_dd.get(i, i) - 1.0).abs() < 1e-14);
    }
    // <d_xx | d_yy> = 1/3 for concentric Cartesian d orbitals
    assert!((s_dd.get(0, 3) - (1.0 / 3.0)).abs() < 1e-14);

    let t_dd = shell_kinetic_block(&sh_d, &sh_d);
    let t_dd_diag_ref = [
        1.733333333333334,
        2.800000000000001,
        2.800000000000001,
        1.733333333333334,
        2.800000000000001,
        1.733333333333334,
    ];
    for (i, &expected) in t_dd_diag_ref.iter().enumerate() {
        assert!((t_dd.get(i, i) - expected).abs() < 1e-13);
    }

    let s_ds = shell_overlap_block(&sh_d, &sh_s);
    let s_ds_ref = [
        0.437890071945269,
        0.0,
        0.0,
        0.437890071945269,
        0.0,
        0.680413804099572,
    ];
    for (i, &expected) in s_ds_ref.iter().enumerate() {
        assert!((s_ds.get(i, 0) - expected).abs() < 1e-13);
    }

    let t_ds = shell_kinetic_block(&sh_d, &sh_s);
    let t_ds_ref = [
        0.116390662909950,
        0.0,
        0.0,
        0.116390662909950,
        0.0,
        0.572622289693949,
    ];
    for (i, &expected) in t_ds_ref.iter().enumerate() {
        assert!((t_ds.get(i, 0) - expected).abs() < 1e-13);
    }

    let mol_o = Molecule::new(vec![Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
    let v_ds = shell_nuclear_block(&sh_d, &sh_s, &mol_o);
    let v_ds_ref = [
        -2.847606831141998,
        0.0,
        0.0,
        -2.847606831141998,
        0.0,
        -4.128554081538142,
    ];
    for (i, &expected) in v_ds_ref.iter().enumerate() {
        assert!((v_ds.get(i, 0) - expected).abs() < 1e-12);
    }
}
