use calchem_tddft::davidson::{solve_rpa, solve_tda};

#[test]
fn rpa_obeys_original_equations_and_symplectic_normalization() {
    for iterative in [false, true] {
        let (modes, _) = solve_rpa(&[0.8, 1.3], 2, 30, 1e-10, iterative, |vectors| {
            let apb = vectors
                .iter()
                .map(|v| vec![0.95 * v[0] + 0.08 * v[1], 0.08 * v[0] + 1.4 * v[1]])
                .collect();
            let amb = vectors
                .iter()
                .map(|v| vec![0.65 * v[0] + 0.02 * v[1], 0.02 * v[0] + 1.2 * v[1]])
                .collect();
            (apb, amb)
        })
        .unwrap();
        for mode in modes {
            let p = [mode.x[0] + mode.y[0], mode.x[1] + mode.y[1]];
            let m = [mode.x[0] - mode.y[0], mode.x[1] - mode.y[1]];
            let residual = [
                0.95 * p[0] + 0.08 * p[1] - mode.omega * m[0],
                0.08 * p[0] + 1.4 * p[1] - mode.omega * m[1],
                0.65 * m[0] + 0.02 * m[1] - mode.omega * p[0],
                0.02 * m[0] + 1.2 * m[1] - mode.omega * p[1],
            ];
            assert!(residual.iter().all(|v| v.abs() < 1e-10));
            assert!((p[0] * m[0] + p[1] * m[1] - 1.0).abs() < 1e-10);
        }
    }
}

#[test]
fn unconverged_exhausted_subspace_is_not_reported_successful() {
    let result = solve_tda(&[0.8, 1.3], 1, 30, 1e-30, true, |vectors| {
        vectors
            .iter()
            .map(|v| vec![0.8 * v[0] + 0.123 * v[1], 0.123 * v[0] + 1.3 * v[1]])
            .collect()
    });
    assert!(result.is_err());
}
