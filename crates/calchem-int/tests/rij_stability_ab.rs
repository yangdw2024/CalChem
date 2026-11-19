use calchem_basis::BasisSet;
use calchem_int::density_fitting::{compute_2center_coulomb_metric, RiJEngine};
use calchem_model::{Atom, Molecule};
use calchem_tensor::{Matrix2D, SpdFactor};

fn make_water() -> Molecule {
    Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.12]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.43, -0.96]).unwrap(),
            Atom::from_bohr("H", [0.15, -1.41, -0.92]).unwrap(),
        ],
        0,
        1,
    )
}

fn synthetic_density(n: usize) -> Matrix2D {
    let mut p = Matrix2D::zeros(n, n);
    for i in 0..n {
        p.set(i, i, 0.4 + 0.02 * (i as f64));
        for j in 0..i {
            let v = 0.03 * (((i * 7 + j * 5 + 1) as f64).sin());
            p.set(i, j, v);
            p.set(j, i, v);
        }
    }
    p
}

#[test]
fn fixed_input_cpu_vs_gpu_contraction_and_cholesky_vs_explicit_inverse_ab() {
    let mol = make_water();
    let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let n_ao = basis.n_ao;
    let n_ao_sq = n_ao * n_ao;
    let p = synthetic_density(n_ao);

    let engine = RiJEngine::new(&mol, &basis).unwrap();
    let b_tensor = engine.cached_b.as_ref().expect("cached_b expected for H2O");
    let n_aux = engine.aux_basis.n_aux;
    let v = compute_2center_coulomb_metric(&engine.aux_basis);
    let factor = SpdFactor::from_matrix(&v, 1e-12).unwrap();
    let kappa_unscaled = SpdFactor::unscaled_kappa(&v);
    let v_inv = engine.metric_factor.to_explicit_inverse();

    // 1. Compute v_vec = B * P
    let mut v_vec = vec![0.0; n_aux];
    for q in 0..n_aux {
        let slice = &b_tensor[q * n_ao_sq..(q + 1) * n_ao_sq];
        let mut s = 0.0;
        for k in 0..n_ao_sq {
            s += slice[k] * p.data[k];
        }
        v_vec[q] = s;
    }

    // 2. Compare x_explicit = V^-1 v vs x_chol = SpdFactor::solve(v)
    let mut x_explicit = vec![0.0; n_aux];
    for i in 0..n_aux {
        let mut s = 0.0;
        for j in 0..n_aux {
            s += v_inv.get(i, j) * v_vec[j];
        }
        x_explicit[i] = s;
    }
    let x_chol = factor.solve(&v_vec);

    let max_res_explicit = (0..n_aux)
        .map(|i| {
            let vx: f64 = (0..n_aux).map(|j| v.get(i, j) * x_explicit[j]).sum();
            (vx - v_vec[i]).abs()
        })
        .fold(0.0_f64, f64::max);
    let max_res_chol = (0..n_aux)
        .map(|i| {
            let vx: f64 = (0..n_aux).map(|j| v.get(i, j) * x_chol[j]).sum();
            (vx - v_vec[i]).abs()
        })
        .fold(0.0_f64, f64::max);
    let max_v_inv = v_inv
        .data
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);

    let contract_cpu = |x: &[f64]| -> Vec<f64> {
        let mut j = vec![0.0; n_ao_sq];
        for idx in 0..n_ao_sq {
            let mut s = 0.0;
            for q in 0..n_aux {
                s += x[q] * b_tensor[q * n_ao_sq + idx];
            }
            j[idx] = s;
        }
        j
    };
    let ej_from_j = |j: &[f64]| -> f64 {
        0.5 * (0..n_ao_sq).map(|k| p.data[k] * j[k]).sum::<f64>()
    };

    let j_exp = contract_cpu(&x_explicit);
    let j_chol = contract_cpu(&x_chol);
    let max_dj_chol_exp = (0..n_ao_sq)
        .map(|k| (j_chol[k] - j_exp[k]).abs())
        .fold(0.0_f64, f64::max);
    let de_chol_exp = (ej_from_j(&j_chol) - ej_from_j(&j_exp)).abs();

    println!(
        "[COND] H2O 6-31G(d): n_ao={} n_aux={} kappa_2(V)={:.3e} kappa_2(V_norm)={:.3e} max|V^-1|={:.3e}",
        n_ao, n_aux, kappa_unscaled, factor.kappa_scaled, max_v_inv
    );
    println!(
        "[SOLVE] max|Vx-v|_explicit={:.3e} max|Vx-v|_chol={:.3e} max|J_chol-J_exp|={:.3e} |dE_J|={:.3e}",
        max_res_explicit, max_res_chol, max_dj_chol_exp, de_chol_exp
    );

    // 3. Fixed B and x: CPU vs GPU contraction parity + 5x repeatability
    if calchem_cuda::is_cuda_available() {
        let mut max_j_diff = 0.0_f64;
        let mut max_e_diff = 0.0_f64;
        let mut j_gpu_runs: Vec<Vec<f64>> = Vec::new();
        for _ in 0..5 {
            let mut j_gpu = vec![0.0; n_ao_sq];
            let ok = calchem_cuda::cuda_rij_contract_coulomb(
                b_tensor,
                &x_chol,
                &mut j_gpu,
                n_aux,
                n_ao_sq,
            );
            assert!(ok, "CUDA contraction failed when is_cuda_available() is true");
            let dj = (0..n_ao_sq)
                .map(|k| (j_gpu[k] - j_chol[k]).abs())
                .fold(0.0_f64, f64::max);
            let de = (ej_from_j(&j_gpu) - ej_from_j(&j_chol)).abs();
            max_j_diff = max_j_diff.max(dj);
            max_e_diff = max_e_diff.max(de);
            j_gpu_runs.push(j_gpu);
        }
        let mut gpu_repeat_jitter = 0.0_f64;
        for r in 1..j_gpu_runs.len() {
            for k in 0..n_ao_sq {
                gpu_repeat_jitter = gpu_repeat_jitter.max((j_gpu_runs[r][k] - j_gpu_runs[0][k]).abs());
            }
        }
        println!(
            "[GPU-FIXED] H2O 6-31G(d): max|J_gpu-J_cpu|={:.3e} |dE_J|={:.3e} gpu_5x_repeat_jitter={:.3e}",
            max_j_diff, max_e_diff, gpu_repeat_jitter
        );
        assert!(max_j_diff < 1e-13, "Fixed-input GPU vs CPU J mismatch: {max_j_diff:.3e}");
        assert!(gpu_repeat_jitter == 0.0, "GPU contraction non-deterministic: {gpu_repeat_jitter:.3e}");
    }

    // 4. Compare FD h=1e-4 noise for explicit V^-1 vs Cholesky solve vs analytical gradient
    let grad_ana = engine.compute_coulomb_gradient(mol.atoms.len(), &basis, &p).unwrap();
    let h = 1e-4;
    let ej_ab = |m: &Molecule, use_chol: bool| -> f64 {
        let bs = BasisSet::from_name("6-31g(d)", m).unwrap();
        let eng = RiJEngine::new(m, &bs).unwrap();
        let bt = eng.cached_b.as_ref().unwrap();
        let na = eng.aux_basis.n_aux;
        let mut vv = vec![0.0; na];
        for q in 0..na {
            let sl = &bt[q * n_ao_sq..(q + 1) * n_ao_sq];
            vv[q] = (0..n_ao_sq).map(|k| sl[k] * p.data[k]).sum();
        }
        let xv = if use_chol {
            let vm = compute_2center_coulomb_metric(&eng.aux_basis);
            SpdFactor::from_matrix(&vm, 1e-12).unwrap().solve(&vv)
        } else {
            let vi = eng.metric_factor.to_explicit_inverse();
            (0..na)
                .map(|i| (0..na).map(|j| vi.get(i, j) * vv[j]).sum())
                .collect()
        };
        // E_J = 0.5 * v^T x
        0.5 * (0..na).map(|q| vv[q] * xv[q]).sum::<f64>()
    };

    let mut max_fd_err_exp = 0.0_f64;
    let mut max_fd_err_chol = 0.0_f64;
    for atom_idx in 0..mol.atoms.len() {
        for axis in 0..3 {
            let mut mp = mol.clone();
            mp.atoms[atom_idx].position_bohr[axis] += h;
            let mut mm = mol.clone();
            mm.atoms[atom_idx].position_bohr[axis] -= h;
            let fd_exp = (ej_ab(&mp, false) - ej_ab(&mm, false)) / (2.0 * h);
            let fd_chol = (ej_ab(&mp, true) - ej_ab(&mm, true)) / (2.0 * h);
            let ana = grad_ana[atom_idx][axis];
            max_fd_err_exp = max_fd_err_exp.max((ana - fd_exp).abs());
            max_fd_err_chol = max_fd_err_chol.max((ana - fd_chol).abs());
        }
    }
    println!(
        "[FD h=1e-4] max|ana - fd_explicit|={:.3e} max|ana - fd_chol|={:.3e}",
        max_fd_err_exp, max_fd_err_chol
    );
}
