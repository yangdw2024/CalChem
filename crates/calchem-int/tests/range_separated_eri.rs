use calchem_basis::BasisSet;
use calchem_int::{
    compute_sparse_eri, compute_sparse_eri_with_omega, eri_primitive,
    eri_primitive_with_omega, IntegralOptions,
};
use calchem_model::{Atom, Molecule};

#[test]
fn test_range_separated_eri_asymptotic_limits() {
    let r_a = [0.0, 0.0, 0.0];
    let r_b = [0.0, 0.0, 0.0];
    let r_c = [0.0, 0.0, 1.2];
    let r_d = [0.0, 0.0, 1.2];

    let alpha = 1.5;
    let beta = 1.2;
    let gamma = 0.8;
    let delta = 0.9;

    let p = alpha + beta;
    let q = gamma + delta;
    let rho = p * q / (p + q);

    // 1. Standard unattenuated Coulomb integral
    let val_std = eri_primitive([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], alpha, r_a, beta, r_b, gamma, r_c, delta, r_d);

    // 2. Large omega limit -> should match standard Coulomb to high precision
    let val_large_w = eri_primitive_with_omega([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], alpha, r_a, beta, r_b, gamma, r_c, delta, r_d, Some(1e6));
    assert!((val_std - val_large_w).abs() < 1e-11, "Large omega limit failed: diff={}", (val_std - val_large_w).abs());

    // 3. Small omega limit -> should smoothly vanish to 0
    let val_small_w = eri_primitive_with_omega([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], alpha, r_a, beta, r_b, gamma, r_c, delta, r_d, Some(1e-6));
    assert!(val_small_w.abs() < 1e-5, "Small omega limit failed: val={}", val_small_w);

    // 4. Concentric s-orbitals exact analytical ratio:
    // (ss | erf(w*r12)/r12 | ss) / (ss | 1/r12 | ss) == theta == w / sqrt(rho + w^2)
    let val_conc_std = eri_primitive([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], alpha, r_a, beta, r_a, gamma, r_a, delta, r_a);
    let omega = 0.33; // CAM-B3LYP parameter
    let val_conc_w = eri_primitive_with_omega([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], alpha, r_a, beta, r_a, gamma, r_a, delta, r_a, Some(omega));
    let analytical_theta = omega / (rho + omega * omega).sqrt();
    let computed_ratio = val_conc_w / val_conc_std;
    assert!((computed_ratio - analytical_theta).abs() < 1e-12, "Concentric ratio mismatch: computed={}, expected={}", computed_ratio, analytical_theta);
}

#[test]
fn test_range_separated_eri_8fold_symmetry_and_stability() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.43, -1.10]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.43, -1.10]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
    let opts = IntegralOptions::default();

    // Standard vs CAM-B3LYP (omega = 0.33)
    let eri_std = compute_sparse_eri(&bs);
    let eri_cam = compute_sparse_eri_with_omega(&bs, &opts, Some(0.33));

    assert_eq!(eri_std.n_ao, bs.n_ao);
    assert_eq!(eri_cam.n_ao, bs.n_ao);

    // Every attenuated integral should be strictly bounded by standard Coulomb: 0 <= (mu nu|lam sig)_w <= (mu nu|lam sig)
    if let (Some(d_std), Some(d_cam)) = (&eri_std.dense, &eri_cam.dense) {
        let n = bs.n_ao;
        for mu in 0..n {
            for nu in 0..n {
                for lam in 0..n {
                    for sig in 0..n {
                        let v_std = d_std.get(mu, nu, lam, sig);
                        let v_cam = d_cam.get(mu, nu, lam, sig);
                        // Positivity of diagonal
                        if mu == nu && lam == sig && mu == lam {
                            assert!(v_cam > 0.0);
                            assert!(v_cam <= v_std + 1e-14);
                        }
                        // 8-fold symmetry
                        assert_eq!(v_cam, d_cam.get(nu, mu, lam, sig));
                        assert_eq!(v_cam, d_cam.get(mu, nu, sig, lam));
                        assert_eq!(v_cam, d_cam.get(lam, sig, mu, nu));
                    }
                }
            }
        }
    }
}
