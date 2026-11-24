use calchem_basis::BasisSet;
use calchem_int::direct_fock::DirectFockEngine;
use calchem_int::{compute_eri, compute_sparse_eri};
use calchem_model::{Atom, Molecule};

fn make_water_631gd() -> (Molecule, BasisSet) {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    (mol, bs)
}

fn make_test_density(n: usize) -> Vec<f64> {
    let mut p = vec![0.0f64; n * n];
    for i in 0..n {
        p[i * n + i] = 1.0 / (i as f64 + 1.0);
        for j in 0..i {
            let v = 0.1 / ((i + j) as f64 + 1.0);
            p[i * n + j] = v;
            p[j * n + i] = v;
        }
    }
    p
}

#[test]
fn test_unscreened_direct_fock_strictly_matches_stored_eri_rks() {
    let (_mol, bs) = make_water_631gd();
    let n = bs.n_ao;
    let p = make_test_density(n);

    let sparse_eri = compute_sparse_eri(&bs);
    // Unscreened direct engine (threshold = 0.0)
    let engine_unscreened = DirectFockEngine::new(&bs, 0.0);

    // Test across multiple exact exchange fractions (pure DFT 0.0, B3LYP 0.20, HF 1.0)
    for &exx in &[0.0, 0.20, 1.0] {
        let g_stored = sparse_eri.contract_rks(&p, exx);
        let g_direct = engine_unscreened.build_rks(&bs, &p, exx, 0.0);

        let mut max_diff = 0.0f64;
        let mut frob_diff_sq = 0.0f64;
        for i in 0..n * n {
            let d = (g_stored[i] - g_direct[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
            frob_diff_sq += d * d;
        }
        let frob_diff = frob_diff_sq.sqrt();

        println!(
            "Unscreened Direct vs Stored 6-31G(d) [exx={exx}]: max_diff = {max_diff:.3e}, frob_diff = {frob_diff:.3e}"
        );
        assert!(
            max_diff < 1e-12,
            "Unscreened Direct vs Stored max error {max_diff:.3e} exceeds 1e-12 at exx={exx}"
        );
        assert!(
            frob_diff < 1e-11,
            "Unscreened Direct vs Stored Frobenius error {frob_diff:.3e} exceeds 1e-11 at exx={exx}"
        );
    }
}

#[test]
fn test_unscreened_direct_fock_strictly_matches_stored_eri_uks() {
    let (_mol, bs) = make_water_631gd();
    let n = bs.n_ao;
    let pa = make_test_density(n);
    let mut pb = vec![0.0f64; n * n];
    for idx in 0..n * n {
        pb[idx] = pa[idx] * 0.85;
    }

    let eri_dense = compute_eri(&bs);
    let engine_unscreened = DirectFockEngine::new(&bs, 0.0);

    for &exx in &[0.0, 0.20, 1.0] {
        // Reference UKS contraction from dense Tensor4D
        let mut ga_ref = vec![0.0f64; n * n];
        let mut gb_ref = vec![0.0f64; n * n];
        for mu in 0..n {
            for nu in 0..n {
                let mut j = 0.0;
                let mut ka = 0.0;
                let mut kb = 0.0;
                for lam in 0..n {
                    for sig in 0..n {
                        let pt = pa[lam * n + sig] + pb[lam * n + sig];
                        let coul = eri_dense.get(mu, nu, lam, sig);
                        let exch = eri_dense.get(mu, lam, nu, sig);
                        j += pt * coul;
                        ka += pa[lam * n + sig] * exch;
                        kb += pb[lam * n + sig] * exch;
                    }
                }
                ga_ref[mu * n + nu] = j - exx * ka;
                gb_ref[mu * n + nu] = j - exx * kb;
            }
        }

        let (ga_dir, gb_dir) = engine_unscreened.build_uks(&bs, &pa, &pb, exx, 0.0);

        let mut max_diff_a = 0.0f64;
        let mut max_diff_b = 0.0f64;
        for i in 0..n * n {
            let da = (ga_ref[i] - ga_dir[i]).abs();
            let db = (gb_ref[i] - gb_dir[i]).abs();
            if da > max_diff_a {
                max_diff_a = da;
            }
            if db > max_diff_b {
                max_diff_b = db;
            }
        }
        println!(
            "Unscreened Direct vs Dense UKS 6-31G(d) [exx={exx}]: max_diff_alpha = {max_diff_a:.3e}, max_diff_beta = {max_diff_b:.3e}"
        );
        assert!(
            max_diff_a < 1e-12,
            "UKS alpha channel max diff {max_diff_a:.3e} exceeds 1e-12"
        );
        assert!(
            max_diff_b < 1e-12,
            "UKS beta channel max diff {max_diff_b:.3e} exceeds 1e-12"
        );
    }
}

#[test]
fn test_screening_statistics_and_rigorous_cumulative_bound() {
    let (_mol, bs) = make_water_631gd();
    let n = bs.n_ao;
    let p = make_test_density(n);
    let exx = 0.20; // B3LYP

    let engine = DirectFockEngine::new(&bs, 0.0);
    let (g_exact, stats_exact) = engine.build_rks_with_stats(&bs, &p, exx, 0.0);
    assert_eq!(stats_exact.skipped_quartets, 0);
    assert_eq!(
        stats_exact.computed_quartets,
        stats_exact.total_quartet_candidates
    );

    let thresholds = [1e-8, 1e-10, 1e-12, 1e-14];

    for &thresh in &thresholds {
        let (g_screened, stats) = engine.build_rks_with_stats(&bs, &p, exx, thresh);

        let mut max_err = 0.0f64;
        let mut frob_err_sq = 0.0f64;
        for i in 0..n * n {
            let d = (g_exact[i] - g_screened[i]).abs();
            if d > max_err {
                max_err = d;
            }
            frob_err_sq += d * d;
        }
        let frob_err = frob_err_sq.sqrt();

        println!(
            "Thresh = {thresh:.1e} | Total = {} | Computed = {} | Skipped = {} | MaxBound = {:.3e} | CumBound = {:.3e} | ActualMaxErr = {:.3e} | ActualFrobErr = {:.3e}",
            stats.total_quartet_candidates,
            stats.computed_quartets,
            stats.skipped_quartets,
            stats.max_skipped_bound,
            stats.cumulative_bound_error,
            max_err,
            frob_err
        );

        // Verification of the rigorous Cauchy-Schwarz upper bound:
        // 1. The maximum individual skipped quartet bound must be strictly less than threshold
        if stats.skipped_quartets > 0 {
            assert!(
                stats.max_skipped_bound <= thresh * 1.0000000001,
                "Max skipped bound {:.3e} exceeded threshold {:.3e}",
                stats.max_skipped_bound,
                thresh
            );
        }

        // 2. The actual max error must be strictly bounded by the cumulative upper bound:
        // max_err <= cumulative_bound_error (within floating point precision)
        assert!(
            max_err <= stats.cumulative_bound_error + 1e-14,
            "Actual error {max_err:.3e} exceeded cumulative bound {:.3e}",
            stats.cumulative_bound_error
        );
    }
}

#[test]
fn test_screening_distance_scaling_he_dimer() {
    // Two helium atoms separated by varying distances (2.0 to 20.0 Bohr)
    let distances = [2.0, 5.0, 10.0, 20.0];
    let exx = 1.0;
    let thresh = 1e-10;

    for &r in &distances {
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("He", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("He", [0.0, 0.0, r]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::basis_6_31g(&mol).unwrap();
        let n = bs.n_ao;
        let p = make_test_density(n);

        let engine = DirectFockEngine::new(&bs, thresh);
        let (g_screened, stats) = engine.build_rks_with_stats(&bs, &p, exx, thresh);

        let engine_exact = DirectFockEngine::new(&bs, 0.0);
        let (g_exact, _) = engine_exact.build_rks_with_stats(&bs, &p, exx, 0.0);

        let mut max_err = 0.0f64;
        for i in 0..n * n {
            let d = (g_exact[i] - g_screened[i]).abs();
            if d > max_err {
                max_err = d;
            }
        }

        let skip_ratio = (stats.skipped_quartets as f64) / (stats.total_quartet_candidates as f64);
        println!(
            "He2 distance R = {:4.1} Bohr: Computed = {:2}, Skipped = {:2} ({:5.1}%), MaxErr = {:.3e} (threshold = {:.1e})",
            r,
            stats.computed_quartets,
            stats.skipped_quartets,
            skip_ratio * 100.0,
            max_err,
            thresh
        );

        assert!(
            max_err <= stats.cumulative_bound_error + 1e-14,
            "Max error {max_err:.3e} exceeded cumulative bound"
        );
        assert!(
            max_err <= thresh * 10.0,
            "Max error {max_err:.3e} exceeded tolerance at R={r}"
        );
    }
}

#[test]
fn test_direct_k_general_matches_sparse_sym_anti_and_asymmetric() {
    let (_mol, bs) = make_water_631gd();
    let n = bs.n_ao;
    let mut x_sym = vec![0.0; n * n];
    let mut x_anti = vec![0.0; n * n];
    let mut x_gen = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let v_ij = ((i as f64 + 1.3) * 0.37 + (j as f64 + 0.7) * 0.19).sin() * 0.25;
            let v_ji = ((j as f64 + 1.3) * 0.37 + (i as f64 + 0.7) * 0.19).sin() * 0.25;
            x_sym[i * n + j] = 0.5 * (v_ij + v_ji);
            x_anti[i * n + j] = 0.5 * (v_ij - v_ji);
            x_gen[i * n + j] = v_ij;
        }
    }

    let engine = DirectFockEngine::new(&bs, 1e-13);
    for omega in [None, Some(0.33)] {
        let eri_sparse = calchem_int::compute_sparse_eri_with_omega(
            &bs,
            &calchem_int::IntegralOptions::default(),
            omega,
        );
        for (label, x) in [("sym", &x_sym), ("anti", &x_anti), ("gen", &x_gen)] {
            let k_ref = eri_sparse.contract_k_gen(x);
            let k_dir = engine.build_k_general_with_omega(&bs, x, omega, 1e-13);
            let max_diff = k_ref
                .iter()
                .zip(&k_dir)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_diff < 1e-11,
                "Direct K general mismatch for {label} (omega={omega:?}): max_diff={max_diff:.3e}"
            );
        }
    }
}
