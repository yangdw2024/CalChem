use calchem_int::{boys, boys_batch, erf_approx};

#[test]
fn test_erf_approx_precision() {
    let cases = [
        (0.0_f64, 0.0_f64),
        (0.5_f64, 0.5204998778130465_f64),
        (1.0_f64, 0.8427007929497149_f64),
        (2.0_f64, 0.9953222650189527_f64),
        (5.0_f64, 0.9999999999984625_f64),
        (-1.0_f64, -0.8427007929497149_f64),
    ];
    for (x, expected) in cases {
        let got = erf_approx(x);
        assert!(
            (got - expected).abs() < 1e-15,
            "x={x}: got={got}, expected={expected}, diff={:e}",
            (got - expected).abs()
        );
    }
}

#[test]
fn test_boys_known_values_and_stability() {
    let cases = [
        (0usize, 0.0_f64, 1.0_f64),
        (1usize, 0.0_f64, 1.0_f64 / 3.0_f64),
        (2usize, 0.0_f64, 1.0_f64 / 5.0_f64),
        (4usize, 0.0_f64, 1.0_f64 / 9.0_f64),
        (0usize, 1e-8_f64, 0.9999999966666667_f64),
        (0usize, 1.0_f64, 0.7468241328124271_f64),
        (1usize, 1.0_f64, 0.18947234582049235_f64),
        (2usize, 1.0_f64, 0.10026879814501737_f64),
        // Critical medium-T regime (15 <= T <= 35) where alternating Taylor series fails.
        // References regenerated 2026-11-03 by tools/reference/gen_integral_refs.py
        // (mpmath dps=50 quadrature of the defining integral). The previous values
        // 0.1772453850905516 (= sqrt(pi)/10, i.e. erf(5) taken as 1), 2.555e-4 and
        // 0.16360991294575365 were wrong; see docs/audits/2026-11-03/evidence.
        (0usize, 25.0_f64, 0.17724538509027909508_f64),
        (4usize, 25.0_f64, 2.977722148749626937e-6_f64),
        (0usize, 29.34071349_f64, 0.16360990987453259531_f64),
        (0usize, 50.0_f64, 0.12533141373155003_f64),
    ];

    for (n, t, expected) in cases {
        let got = boys(n, t);
        assert!(
            (got - expected).abs() < 1e-14,
            "n={n}, t={t}: got={got:.16e}, expected={expected:.16e}, diff={:e}",
            (got - expected).abs()
        );
    }
}

#[test]
fn test_boys_batch_recurrence_consistency() {
    for &t in &[1e-12_f64, 0.05, 1.2, 8.5, 18.0, 26.42683569, 29.34071349, 35.0, 42.0, 80.0] {
        let mut out = [0.0; 9];
        boys_batch(8, t, &mut out);
        let exp_neg_t = (-t).exp();
        for n in 1..=8 {
            // Exact identity: (2n - 1) F_{n-1}(t) - 2 t F_n(t) = exp(-t)
            let lhs = (2.0 * (n as f64) - 1.0) * out[n - 1] - 2.0 * t * out[n];
            assert!(
                (lhs - exp_neg_t).abs() < 1e-13,
                "Recurrence identity failed at t={t}, n={n}: lhs={lhs:.16e}, exp(-t)={exp_neg_t:.16e}"
            );
            assert!(out[n] > 0.0 && out[n] < out[n - 1]);
        }
    }
}
