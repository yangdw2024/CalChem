use calchem_tddft::davidson::{
    solve_rpa, solve_rpa_with_screening, solve_tda, solve_tda_with_screening, ScreeningStage,
};
use calchem_tensor::{eigh_sym, Matrix2D};

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

fn build_synthetic_sym_mat(dim: usize, diag_base: f64, diag_step: f64, couple_scale: f64) -> (Vec<f64>, Matrix2D) {
    let mut diag = Vec::with_capacity(dim);
    let mut mat = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        let d = diag_base + diag_step * (i as f64);
        diag.push(d);
        mat.set(i, i, d);
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            // Increasing coupling for higher low-lying roots (i = 0..5) so lower roots converge faster and lock first
            let row_boost = if i < 6 { 0.05 + 0.22 * (i as f64) } else { 0.35 };
            let dist = (j - i) as f64;
            let val = couple_scale * row_boost * (((i * 7 + j * 13 + 1) as f64).sin()) / (1.0 + 0.25 * dist * dist);
            mat.set(i, j, val);
            mat.set(j, i, val);
        }
    }
    (diag, mat)
}

fn mat_vec_mul(mat: &Matrix2D, v: &[f64]) -> Vec<f64> {
    let n = mat.rows;
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let row = &mat.data[i * n..(i + 1) * n];
        let mut s = 0.0;
        for j in 0..n {
            s += row[j] * v[j];
        }
        out[i] = s;
    }
    out
}

#[test]
fn root_locking_shrinks_batch_and_matches_dense() {
    use std::cell::RefCell;

    let dim = 100usize;
    let n_roots = 5usize;
    let conv_tol = 1e-10;
    let (diag, a_mat) = build_synthetic_sym_mat(dim, 0.5, 0.06, 0.045);
    let (dense_evals, _) = eigh_sym(&a_mat, 200, 1e-14).unwrap();

    // TDA root locking
    let tda_batches: RefCell<Vec<usize>> = RefCell::new(Vec::new());
    let (tda_modes, tda_iters) = solve_tda_with_screening(
        &diag,
        n_roots,
        60,
        conv_tol,
        true,
        false,
        |vecs, stage| {
            assert_eq!(stage, ScreeningStage::Tight);
            tda_batches.borrow_mut().push(vecs.len());
            vecs.iter().map(|v| mat_vec_mul(&a_mat, v)).collect()
        },
    )
    .unwrap();
    assert!(tda_iters >= 4, "Expected >= 4 TDA iters to observe root locking, got {tda_iters}");
    for r in 0..n_roots {
        assert!(
            (tda_modes[r].omega - dense_evals[r]).abs() < 1e-10,
            "TDA root {r} mismatch: {} vs {}",
            tda_modes[r].omega,
            dense_evals[r]
        );
        assert!(tda_modes[r].residual_norm <= conv_tol);
    }
    let tb = tda_batches.borrow();
    assert!(tb.len() >= 3);
    assert!(
        tb.windows(2).skip(1).all(|w| w[1] <= w[0]),
        "TDA batch size after initial guess must shrink monotonically via root locking: {:?}",
        *tb
    );
    assert!(
        *tb.last().unwrap() < tb[1],
        "TDA final batch size must be strictly smaller than iter-2 batch size: {:?}",
        *tb
    );

    // RPA root locking
    let (_, b_mat_raw) = build_synthetic_sym_mat(dim, 0.05, 0.002, 0.015);
    let mut apb = Matrix2D::zeros(dim, dim);
    let mut amb = Matrix2D::zeros(dim, dim);
    for idx in 0..(dim * dim) {
        apb.data[idx] = a_mat.data[idx] + b_mat_raw.data[idx];
        amb.data[idx] = a_mat.data[idx] - b_mat_raw.data[idx];
    }
    // Dense RPA reference
    let (lam_m, q_m) = eigh_sym(&amb, 200, 1e-14).unwrap();
    let mut sqrt_m = Matrix2D::zeros(dim, dim);
    for (k, &ev) in lam_m.iter().enumerate() {
        let s = ev.sqrt();
        for i in 0..dim {
            let qik = q_m.get(i, k);
            for j in 0..dim {
                sqrt_m.add_assign_at(i, j, s * qik * q_m.get(j, k));
            }
        }
    }
    let omega_sq = calchem_tensor::matmul(&calchem_tensor::matmul(&sqrt_m, &apb), &sqrt_m);
    let (w2_dense, _) = eigh_sym(&omega_sq, 200, 1e-14).unwrap();

    let rpa_batches: RefCell<Vec<usize>> = RefCell::new(Vec::new());
    let (rpa_modes, rpa_iters) = solve_rpa_with_screening(
        &diag,
        n_roots,
        60,
        conv_tol,
        true,
        false,
        |vecs, stage| {
            assert_eq!(stage, ScreeningStage::Tight);
            rpa_batches.borrow_mut().push(vecs.len());
            let wp = vecs.iter().map(|v| mat_vec_mul(&apb, v)).collect();
            let wm = vecs.iter().map(|v| mat_vec_mul(&amb, v)).collect();
            (wp, wm)
        },
    )
    .unwrap();
    assert!(rpa_iters >= 4);
    for r in 0..n_roots {
        let dense_w = w2_dense[r].sqrt();
        assert!(
            (rpa_modes[r].omega - dense_w).abs() < 1e-10,
            "RPA root {r} mismatch: {} vs {dense_w}",
            rpa_modes[r].omega
        );
        assert!(rpa_modes[r].residual_norm <= conv_tol);
    }
    let rb = rpa_batches.borrow();
    assert!(
        *rb.last().unwrap() < rb[1],
        "RPA final batch size must shrink via root locking: {:?}",
        *rb
    );
}

#[test]
fn two_stage_screening_unlocks_and_verifies_tight() {
    use std::cell::RefCell;

    let dim = 80usize;
    let n_roots = 3usize;
    let conv_tol = 1e-8;
    let (diag, a_tight) = build_synthetic_sym_mat(dim, 0.4, 0.08, 0.035);
    let (_, pert) = build_synthetic_sym_mat(dim, 1.0, 0.0, 0.05);
    let mut a_coarse = a_tight.clone();
    for idx in 0..(dim * dim) {
        a_coarse.data[idx] += 8e-4 * pert.data[idx];
    }

    // TDA two-stage check
    let stages_tda: RefCell<Vec<ScreeningStage>> = RefCell::new(Vec::new());
    let (tda_two, _) = solve_tda_with_screening(
        &diag,
        n_roots,
        60,
        conv_tol,
        true,
        true,
        |vecs, stage| {
            stages_tda.borrow_mut().push(stage);
            let op = match stage {
                ScreeningStage::Coarse => &a_coarse,
                ScreeningStage::Tight => &a_tight,
            };
            vecs.iter().map(|v| mat_vec_mul(op, v)).collect()
        },
    )
    .unwrap();
    let (tda_ref, _) = solve_tda_with_screening(
        &diag,
        n_roots,
        60,
        conv_tol,
        true,
        false,
        |vecs, _| vecs.iter().map(|v| mat_vec_mul(&a_tight, v)).collect(),
    )
    .unwrap();

    let st = stages_tda.borrow();
    assert!(st.contains(&ScreeningStage::Coarse), "Must start in Coarse stage");
    assert!(st.contains(&ScreeningStage::Tight), "Must transition to Tight stage");
    let first_tight = st.iter().position(|&s| s == ScreeningStage::Tight).unwrap();
    assert!(
        st[first_tight..].iter().all(|&s| s == ScreeningStage::Tight),
        "Must stay in Tight after Coarse -> Tight transition"
    );
    for (m2, mr) in tda_two.iter().zip(&tda_ref) {
        assert!((m2.omega - mr.omega).abs() < 1e-10);
        assert!(m2.residual_norm <= conv_tol);
    }

    // RPA two-stage check
    let mut apb_tight = a_tight.clone();
    let mut amb_tight = a_tight.clone();
    for i in 0..dim {
        apb_tight.add_assign_at(i, i, 0.04);
        amb_tight.add_assign_at(i, i, -0.04);
    }
    let mut apb_coarse = apb_tight.clone();
    for idx in 0..(dim * dim) {
        apb_coarse.data[idx] += 8e-4 * pert.data[idx];
    }

    let stages_rpa: RefCell<Vec<ScreeningStage>> = RefCell::new(Vec::new());
    let (rpa_two, _) = solve_rpa_with_screening(
        &diag,
        n_roots,
        60,
        conv_tol,
        true,
        true,
        |vecs, stage| {
            stages_rpa.borrow_mut().push(stage);
            let op_p = match stage {
                ScreeningStage::Coarse => &apb_coarse,
                ScreeningStage::Tight => &apb_tight,
            };
            let wp = vecs.iter().map(|v| mat_vec_mul(op_p, v)).collect();
            let wm = vecs.iter().map(|v| mat_vec_mul(&amb_tight, v)).collect();
            (wp, wm)
        },
    )
    .unwrap();
    let (rpa_ref, _) = solve_rpa_with_screening(
        &diag,
        n_roots,
        60,
        conv_tol,
        true,
        false,
        |vecs, _| {
            let wp = vecs.iter().map(|v| mat_vec_mul(&apb_tight, v)).collect();
            let wm = vecs.iter().map(|v| mat_vec_mul(&amb_tight, v)).collect();
            (wp, wm)
        },
    )
    .unwrap();
    let sr = stages_rpa.borrow();
    assert!(sr.contains(&ScreeningStage::Coarse) && sr.contains(&ScreeningStage::Tight));
    for (m2, mr) in rpa_two.iter().zip(&rpa_ref) {
        assert!((m2.omega - mr.omega).abs() < 1e-10);
        assert!(m2.residual_norm <= conv_tol);
    }
}

#[test]
fn degenerate_boundary_root_locking() {
    let dim = 80usize;
    let k_roots = 4usize;
    let conv_tol = 1e-10;
    let mut diag = vec![0.0f64; dim];
    let mut a_mat = Matrix2D::zeros(dim, dim);
    // Roots 0..3 well below 0.70, index 3 at 0.700, index 4 (buffer root) at 0.708
    diag[0] = 0.30;
    diag[1] = 0.42;
    diag[2] = 0.55;
    diag[3] = 0.700;
    diag[4] = 0.708;
    for i in 5..dim {
        diag[i] = 1.2 + 0.05 * ((i - 5) as f64);
    }
    for i in 0..dim {
        a_mat.set(i, i, diag[i]);
    }
    // Strong off-diagonal repulsion on index 4 drops its dressed eigenvalue below 0.700 (~0.68),
    // so if buffer root 4 were skipped, the 4th root would be missed!
    a_mat.set(4, 15, 0.22);
    a_mat.set(15, 4, 0.22);
    a_mat.set(4, 16, -0.18);
    a_mat.set(16, 4, -0.18);

    let (dense_evals, _) = eigh_sym(&a_mat, 200, 1e-14).unwrap();
    let (modes, _) = solve_tda_with_screening(
        &diag,
        k_roots,
        50,
        conv_tol,
        true,
        false,
        |vecs, _| vecs.iter().map(|v| mat_vec_mul(&a_mat, v)).collect(),
    )
    .unwrap();
    for r in 0..k_roots {
        assert!(
            (modes[r].omega - dense_evals[r]).abs() < 1e-10,
            "Missed near-boundary root {r}: got {} vs dense {}",
            modes[r].omega,
            dense_evals[r]
        );
    }
}

#[test]
fn rpa_linear_collapse_zero_extra_evals() {
    use std::cell::RefCell;

    // dim=68, n_roots=1 => n_track=5, max_subspace=64, m_init=6 -> forces collapse before convergence
    let dim = 68usize;
    let n_roots = 1usize;
    let conv_tol = 1e-10;
    let mut diag = Vec::with_capacity(dim);
    let mut apb = Matrix2D::zeros(dim, dim);
    let mut amb = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        let d = 0.40 + 0.006 * (i as f64);
        diag.push(d);
        apb.set(i, i, d + 0.05);
        amb.set(i, i, d - 0.02);
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            let phase = ((i * 5 + j * 9 + 1) as f64).cos();
            let dist = (j - i) as f64;
            let vp = 0.048 * phase / (1.0 + 0.08 * dist);
            let vm = 0.015 * (((i * 3 + j * 7) as f64).sin()) / (1.0 + 0.2 * dist);
            apb.set(i, j, vp);
            apb.set(j, i, vp);
            amb.set(i, j, vm);
            amb.set(j, i, vm);
        }
    }

    let call_batches: RefCell<Vec<usize>> = RefCell::new(Vec::new());
    let (modes, iters) = solve_rpa_with_screening(
        &diag,
        n_roots,
        50,
        conv_tol,
        true,
        false,
        |vecs, _| {
            call_batches.borrow_mut().push(vecs.len());
            let wp = vecs.iter().map(|v| mat_vec_mul(&apb, v)).collect();
            let wm = vecs.iter().map(|v| mat_vec_mul(&amb, v)).collect();
            (wp, wm)
        },
    )
    .unwrap();

    let calls = call_batches.borrow();
    assert!(
        iters >= 12,
        "Expected >= 12 iters to exercise RPA subspace collapse (m + cands > 64), got {iters} (calls={:?})",
        *calls
    );
    assert_eq!(
        calls.len(),
        iters,
        "Zero-ERI linear collapse must invoke apply_apb_amb strictly once per iter (calls={}, iters={iters})",
        calls.len()
    );
    assert!(modes[0].residual_norm <= conv_tol);
    let xpy_dot_xmy: f64 = modes[0]
        .x
        .iter()
        .zip(&modes[0].y)
        .map(|(x, y)| (x + y) * (x - y))
        .sum();
    assert!(
        (xpy_dot_xmy - 1.0).abs() < 1e-10,
        "Casida normalization violated after linear collapse: {xpy_dot_xmy}"
    );
}

fn dense_rpa_evals(apb: &Matrix2D, amb: &Matrix2D) -> Vec<f64> {
    let dim = apb.rows;
    let (lam_m, q_m) = eigh_sym(amb, 200, 1e-14).unwrap();
    let mut sqrt_m = Matrix2D::zeros(dim, dim);
    for (k, &ev) in lam_m.iter().enumerate() {
        assert!(ev > 1e-10, "A-B must be positive definite");
        let s = ev.sqrt();
        for i in 0..dim {
            let qik = q_m.get(i, k);
            for j in 0..dim {
                sqrt_m.add_assign_at(i, j, s * qik * q_m.get(j, k));
            }
        }
    }
    let tmp = calchem_tensor::matmul(&sqrt_m, apb);
    let mut omega_sq = calchem_tensor::matmul(&tmp, &sqrt_m);
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (omega_sq.get(i, j) + omega_sq.get(j, i));
            omega_sq.set(i, j, avg);
            omega_sq.set(j, i, avg);
        }
    }
    let (w2, _) = eigh_sym(&omega_sq, 200, 1e-14).unwrap();
    w2.into_iter().map(|v| v.sqrt()).collect()
}

#[test]
fn delayed_buffer_root_flip_across_gap_cliff_tda_and_rpa() {
    let dim = 70usize;
    let k_roots = 3usize;
    let conv_tol = 1e-10;

    // Case 2a: buffer root r = 3 starts +0.025 Eh (>= 0.02) above diag[2] = 0.620
    let mut diag_2a = vec![0.0f64; dim];
    diag_2a[0] = 0.50;
    diag_2a[1] = 0.55;
    diag_2a[2] = 0.62;
    diag_2a[3] = 0.645;
    for i in 4..dim {
        diag_2a[i] = 0.75 + 0.03 * ((i - 4) as f64);
    }
    let mut a_2a = Matrix2D::zeros(dim, dim);
    let mut b_2a = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        a_2a.set(i, i, diag_2a[i]);
        b_2a.set(i, i, 0.012);
    }
    a_2a.set(1, 12, 0.04);
    a_2a.set(12, 1, 0.04);
    a_2a.set(2, 13, 0.08);
    a_2a.set(13, 2, 0.08);
    b_2a.set(1, 12, 0.010);
    b_2a.set(12, 1, 0.010);
    b_2a.set(2, 13, 0.015);
    b_2a.set(13, 2, 0.015);
    for j in 20..25 {
        a_2a.set(3, j, 0.165);
        a_2a.set(j, 3, 0.165);
        b_2a.set(3, j, 0.030);
        b_2a.set(j, 3, 0.030);
    }

    let (dense_tda_2a, _) = eigh_sym(&a_2a, 200, 1e-14).unwrap();
    let (tda_2a, _) = solve_tda_with_screening(&diag_2a, k_roots, 50, conv_tol, true, false, |vecs, _| {
        vecs.iter().map(|v| mat_vec_mul(&a_2a, v)).collect()
    })
    .unwrap();
    for r in 0..k_roots {
        assert!(
            (tda_2a[r].omega - dense_tda_2a[r]).abs() < 1e-10,
            "TDA 2a root {r} mismatch: got {} vs dense {}",
            tda_2a[r].omega,
            dense_tda_2a[r]
        );
        assert!(tda_2a[r].residual_norm <= conv_tol);
    }

    let mut apb_2a = Matrix2D::zeros(dim, dim);
    let mut amb_2a = Matrix2D::zeros(dim, dim);
    for idx in 0..(dim * dim) {
        apb_2a.data[idx] = a_2a.data[idx] + b_2a.data[idx];
        amb_2a.data[idx] = a_2a.data[idx] - b_2a.data[idx];
    }
    let dense_rpa_2a = dense_rpa_evals(&apb_2a, &amb_2a);
    let (rpa_2a, _) = solve_rpa_with_screening(&diag_2a, k_roots, 50, conv_tol, true, false, |vecs, _| {
        (
            vecs.iter().map(|v| mat_vec_mul(&apb_2a, v)).collect(),
            vecs.iter().map(|v| mat_vec_mul(&amb_2a, v)).collect(),
        )
    })
    .unwrap();
    for r in 0..k_roots {
        assert!(
            (rpa_2a[r].omega - dense_rpa_2a[r]).abs() < 1e-10,
            "RPA 2a root {r} mismatch: got {} vs dense {}",
            rpa_2a[r].omega,
            dense_rpa_2a[r]
        );
        assert!(rpa_2a[r].residual_norm <= conv_tol);
    }

    // Case 2c: diag[3] starts +0.010 Eh (< 0.02) above diag[2], but root 2 relaxes inside m_init,
    // widening gap to >= 0.025 Eh before r = 3 relaxes (tested with Coarse->Tight unlock too).
    let mut diag_2c = vec![0.0f64; dim];
    diag_2c[0] = 0.50;
    diag_2c[1] = 0.56;
    diag_2c[2] = 0.62;
    diag_2c[3] = 0.630;
    for i in 4..dim {
        diag_2c[i] = 0.68 + 0.03 * ((i - 4) as f64);
    }
    let mut a_2c = Matrix2D::zeros(dim, dim);
    let mut b_2c = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        a_2c.set(i, i, diag_2c[i]);
        b_2c.set(i, i, 0.012);
    }
    a_2c.set(1, 7, 0.05);
    a_2c.set(7, 1, 0.05);
    a_2c.set(2, 8, 0.05);
    a_2c.set(8, 2, 0.05);
    a_2c.set(2, 13, 0.03);
    a_2c.set(13, 2, 0.03);
    b_2c.set(1, 7, 0.008);
    b_2c.set(7, 1, 0.008);
    b_2c.set(2, 8, 0.010);
    b_2c.set(8, 2, 0.010);
    for j in 20..25 {
        a_2c.set(3, j, 0.165);
        a_2c.set(j, 3, 0.165);
        b_2c.set(3, j, 0.030);
        b_2c.set(j, 3, 0.030);
    }

    let (dense_tda_2c, _) = eigh_sym(&a_2c, 200, 1e-14).unwrap();
    for enable_coarse in [false, true] {
        let (tda_2c, _) = solve_tda_with_screening(
            &diag_2c,
            k_roots,
            50,
            conv_tol,
            true,
            enable_coarse,
            |vecs, _| vecs.iter().map(|v| mat_vec_mul(&a_2c, v)).collect(),
        )
        .unwrap();
        for r in 0..k_roots {
            assert!(
                (tda_2c[r].omega - dense_tda_2c[r]).abs() < 1e-10,
                "TDA 2c (coarse={enable_coarse}) root {r} mismatch: got {} vs dense {}",
                tda_2c[r].omega,
                dense_tda_2c[r]
            );
            assert!(tda_2c[r].residual_norm <= conv_tol);
        }
    }

    let mut apb_2c = Matrix2D::zeros(dim, dim);
    let mut amb_2c = Matrix2D::zeros(dim, dim);
    for idx in 0..(dim * dim) {
        apb_2c.data[idx] = a_2c.data[idx] + b_2c.data[idx];
        amb_2c.data[idx] = a_2c.data[idx] - b_2c.data[idx];
    }
    let dense_rpa_2c = dense_rpa_evals(&apb_2c, &amb_2c);
    let (rpa_2c, _) = solve_rpa_with_screening(&diag_2c, k_roots, 50, conv_tol, true, true, |vecs, _| {
        (
            vecs.iter().map(|v| mat_vec_mul(&apb_2c, v)).collect(),
            vecs.iter().map(|v| mat_vec_mul(&amb_2c, v)).collect(),
        )
    })
    .unwrap();
    for r in 0..k_roots {
        assert!(
            (rpa_2c[r].omega - dense_rpa_2c[r]).abs() < 1e-10,
            "RPA 2c root {r} mismatch: got {} vs dense {}",
            rpa_2c[r].omega,
            dense_rpa_2c[r]
        );
        assert!(rpa_2c[r].residual_norm <= conv_tol);
    }
}

#[test]
fn buffer_root_beyond_k_plus_two_is_not_blind_spotted() {
    let dim = 70usize;
    let k_roots = 2usize;
    let conv_tol = 1e-10;

    for (label, coupled_idx, low_diags) in [
        ("k_roots+2", 4usize, [0.50, 0.60, 0.605, 0.610, 0.615, 0.620]),
        ("k_roots+3", 5usize, [0.50, 0.60, 0.604, 0.608, 0.612, 0.618]),
    ] {
        let mut diag = vec![0.0f64; dim];
        diag[..6].copy_from_slice(&low_diags);
        for i in 6..dim {
            diag[i] = 0.75 + 0.03 * ((i - 6) as f64);
        }
        let mut a_mat = Matrix2D::zeros(dim, dim);
        let mut b_mat = Matrix2D::zeros(dim, dim);
        for i in 0..dim {
            a_mat.set(i, i, diag[i]);
            b_mat.set(i, i, 0.012);
        }
        a_mat.set(1, 12, 0.05);
        a_mat.set(12, 1, 0.05);
        b_mat.set(1, 12, 0.010);
        b_mat.set(12, 1, 0.010);
        for j in 20..25 {
            a_mat.set(coupled_idx, j, 0.165);
            a_mat.set(j, coupled_idx, 0.165);
            b_mat.set(coupled_idx, j, 0.030);
            b_mat.set(j, coupled_idx, 0.030);
        }

        let (dense_tda, _) = eigh_sym(&a_mat, 200, 1e-14).unwrap();
        let (tda_modes, _) = solve_tda_with_screening(&diag, k_roots, 50, conv_tol, true, false, |vecs, _| {
            vecs.iter().map(|v| mat_vec_mul(&a_mat, v)).collect()
        })
        .unwrap();
        for r in 0..k_roots {
            assert!(
                (tda_modes[r].omega - dense_tda[r]).abs() < 1e-10,
                "TDA {label} root {r} mismatch: got {} vs dense {}",
                tda_modes[r].omega,
                dense_tda[r]
            );
            assert!(tda_modes[r].residual_norm <= conv_tol);
        }

        let mut apb = Matrix2D::zeros(dim, dim);
        let mut amb = Matrix2D::zeros(dim, dim);
        for idx in 0..(dim * dim) {
            apb.data[idx] = a_mat.data[idx] + b_mat.data[idx];
            amb.data[idx] = a_mat.data[idx] - b_mat.data[idx];
        }
        let dense_rpa = dense_rpa_evals(&apb, &amb);
        let (rpa_modes, _) = solve_rpa_with_screening(&diag, k_roots, 50, conv_tol, true, false, |vecs, _| {
            (
                vecs.iter().map(|v| mat_vec_mul(&apb, v)).collect(),
                vecs.iter().map(|v| mat_vec_mul(&amb, v)).collect(),
            )
        })
        .unwrap();
        for r in 0..k_roots {
            assert!(
                (rpa_modes[r].omega - dense_rpa[r]).abs() < 1e-10,
                "RPA {label} root {r} mismatch: got {} vs dense {}",
                rpa_modes[r].omega,
                dense_rpa[r]
            );
            assert!(rpa_modes[r].residual_norm <= conv_tol);
        }
    }
}

#[test]
fn quasi_degenerate_boundary_root_uses_effective_buffer_tol() {
    let dim = 70usize;
    let k_roots = 2usize;
    let conv_tol = 1e-10;

    // m_init = 8 (indices 0..8). Index 2 sits at gap = 4e-4 Eh above boundary diag[1] = 0.5000
    // and couples to index 8 (outside m_init) with r_norm = 9e-4 < 1e-3 (fixed buffer_tol),
    // yet its dressed eigenvalue drops below 0.5000. Only eff_buffer_tol = 0.25 * gap = 1e-4 catches it.
    let mut diag = vec![0.0f64; dim];
    diag[0] = 0.4000;
    diag[1] = 0.5000;
    diag[2] = 0.5004;
    diag[3] = 0.5006;
    diag[4] = 0.5008;
    diag[5] = 0.5010;
    diag[6] = 0.5012;
    diag[7] = 0.5014;
    diag[8] = 0.5016;
    for i in 9..dim {
        diag[i] = 0.75 + 0.03 * ((i - 9) as f64);
    }

    let mut a_mat = Matrix2D::zeros(dim, dim);
    let mut b_mat = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        a_mat.set(i, i, diag[i]);
        b_mat.set(i, i, 0.012);
    }
    a_mat.set(2, 8, 9.0e-4);
    a_mat.set(8, 2, 9.0e-4);
    b_mat.set(2, 8, 2.0e-4);
    b_mat.set(8, 2, 2.0e-4);

    let (dense_tda, _) = eigh_sym(&a_mat, 200, 1e-14).unwrap();
    assert!(
        dense_tda[1] < 0.5000,
        "Test setup must produce root flip below 0.5000, got {}",
        dense_tda[1]
    );
    let (tda_modes, _) = solve_tda_with_screening(&diag, k_roots, 50, conv_tol, true, false, |vecs, _| {
        vecs.iter().map(|v| mat_vec_mul(&a_mat, v)).collect()
    })
    .unwrap();
    for r in 0..k_roots {
        assert!(
            (tda_modes[r].omega - dense_tda[r]).abs() < 1e-10,
            "TDA quasi-degenerate root {r} mismatch: got {} vs dense {}",
            tda_modes[r].omega,
            dense_tda[r]
        );
        assert!(tda_modes[r].residual_norm <= conv_tol);
    }

    let mut apb = Matrix2D::zeros(dim, dim);
    let mut amb = Matrix2D::zeros(dim, dim);
    for idx in 0..(dim * dim) {
        apb.data[idx] = a_mat.data[idx] + b_mat.data[idx];
        amb.data[idx] = a_mat.data[idx] - b_mat.data[idx];
    }
    let dense_rpa = dense_rpa_evals(&apb, &amb);
    let (rpa_modes, _) = solve_rpa_with_screening(&diag, k_roots, 50, conv_tol, true, false, |vecs, _| {
        (
            vecs.iter().map(|v| mat_vec_mul(&apb, v)).collect(),
            vecs.iter().map(|v| mat_vec_mul(&amb, v)).collect(),
        )
    })
    .unwrap();
    for r in 0..k_roots {
        assert!(
            (rpa_modes[r].omega - dense_rpa[r]).abs() < 1e-10,
            "RPA quasi-degenerate root {r} mismatch: got {} vs dense {}",
            rpa_modes[r].omega,
            dense_rpa[r]
        );
        assert!(rpa_modes[r].residual_norm <= conv_tol);
    }
}
