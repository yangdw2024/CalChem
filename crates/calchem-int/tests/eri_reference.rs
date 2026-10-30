use calchem_basis::BasisSet;
use calchem_int::compute_eri;
use calchem_model::{Atom, Molecule};

#[test]
fn test_h2_sto3g_eri_reference() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let eri = compute_eri(&basis);

    assert!((eri.get(0, 0, 0, 0) - 0.7746059439198977).abs() < 1e-12);
    assert!((eri.get(1, 1, 1, 1) - 0.7746059439198977).abs() < 1e-12);
    assert!((eri.get(0, 0, 1, 1) - 0.5696759256037445).abs() < 1e-12);
    assert!((eri.get(0, 1, 0, 1) - 0.29702854027693154).abs() < 1e-12);
    assert!((eri.get(0, 0, 0, 1) - 0.44410765803196034).abs() < 1e-12);
}

#[test]
fn test_h2o_sto3g_eri_reference_quartets() {
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
    let eri = compute_eri(&basis);

    // Compare ssss, ppss, psps, pppp, and spss quartets against Python reference:
    let cases = [
        ([0usize, 0, 0, 0], 4.785065765661066_f64),
        ([1, 1, 1, 1], 0.817206321588127_f64),
        ([2, 2, 2, 2], 0.880159093300264_f64),
        ([2, 3, 2, 3], 0.047444445101798_f64),
        ([1, 2, 5, 5], 0.000000000000000_f64),
        ([2, 5, 3, 6], -0.004937516143074_f64),
        ([5, 6, 5, 6], 0.072470787680740_f64),
    ];

    for ([i, j, k, l], expected) in cases {
        let got = eri.get(i, j, k, l);
        assert!(
            (got - expected).abs() < 1e-12,
            "ERI({i},{j},{k},{l}): got {got:.15f}, expected {expected:.15f}"
        );
    }
}
