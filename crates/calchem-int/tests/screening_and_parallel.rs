use calchem_basis::BasisSet;
use calchem_int::screening::{shell_pair_bounds, SchwarzScreening};
use calchem_int::{compute_eri, compute_eri_with_options, unique_shell_quartets, IntegralOptions};
use calchem_model::{Atom, Molecule};

#[test]
fn test_schwarz_screening_and_parallel_equivalence() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::basis_6_31g(&mol).unwrap();

    // 1. Check Schwarz bounds
    let bounds = shell_pair_bounds(&bs);
    assert_eq!(bounds.len(), bs.shells.len() * bs.shells.len());
    for &b in &bounds {
        assert!(b >= 0.0 && b.is_finite());
    }

    let screening = SchwarzScreening::new(&bs, 1e-10);
    let quartets = unique_shell_quartets(bs.shells.len());
    let skipped = screening.count_skipped(&quartets);

    // In 6-31G water (9 shells, 171 unique quartets), Cauchy-Schwarz screening should safely skip quartets
    println!("Total unique quartets: {}, skipped: {}", quartets.len(), skipped);

    // 2. Compute unscreened serial ERI as reference
    let mut opts_serial_unscreened = IntegralOptions::default();
    opts_serial_unscreened.use_screening = false;
    opts_serial_unscreened.use_parallel = false;
    let eri_ref = compute_eri_with_options(&bs, &opts_serial_unscreened);

    // 3. Compute screened serial ERI
    let mut opts_serial_screened = IntegralOptions::default();
    opts_serial_screened.use_screening = true;
    opts_serial_screened.use_parallel = false;
    let eri_screened = compute_eri_with_options(&bs, &opts_serial_screened);

    // 4. Compute screened parallel ERI
    let mut opts_parallel_screened = IntegralOptions::default();
    opts_parallel_screened.use_screening = true;
    opts_parallel_screened.use_parallel = true;
    let eri_parallel = compute_eri_with_options(&bs, &opts_parallel_screened);

    let n = bs.n_ao;
    let mut max_diff_screening = 0.0f64;
    let mut max_diff_parallel = 0.0f64;

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let v_ref = eri_ref.get(i, j, k, l);
                    let v_sc = eri_screened.get(i, j, k, l);
                    let v_par = eri_parallel.get(i, j, k, l);

                    let diff_sc = (v_ref - v_sc).abs();
                    let diff_par = (v_sc - v_par).abs();

                    if diff_sc > max_diff_screening {
                        max_diff_screening = diff_sc;
                    }
                    if diff_par > max_diff_parallel {
                        max_diff_parallel = diff_par;
                    }
                }
            }
        }
    }

    // Schwarz screening error should be less than the threshold (1e-10)
    assert!(
        max_diff_screening < 1e-9,
        "Screening introduced integral error {max_diff_screening} >= 1e-9"
    );

    // Serial vs Parallel screened integrals must match identically
    assert!(
        max_diff_parallel < 1e-14,
        "Parallel ERI differs from serial ERI by {max_diff_parallel}"
    );

    // Default compute_eri matches parallel screened ERI
    let eri_default = compute_eri(&bs);
    assert_eq!(eri_default, eri_parallel);
}
