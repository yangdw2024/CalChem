use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid, XcFunctional};
use calchem_int::density_fitting::RiJEngine;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::rks;

#[test]
fn water_b3lyp_631gd_rij_vs_exact_probe() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ], 0, 1,
    );
    for bname in ["sto-3g", "6-31g", "6-31g(d)", "6-31g(d,p)"] {
        let basis = BasisSet::from_name(bname, &mol).unwrap();
        let engine = RiJEngine::new(&mol, &basis).unwrap();
        let v = calchem_int::density_fitting::compute_2center_coulomb_metric(&engine.aux_basis);
        let v_inv = engine.metric_factor.to_explicit_inverse();
        let vv_inv = calchem_tensor::matmul(&v, &v_inv);
        let mut max_id_err = 0.0_f64;
        for i in 0..engine.aux_basis.n_aux {
            for j in 0..engine.aux_basis.n_aux {
                let target = if i == j { 1.0 } else { 0.0 };
                max_id_err = max_id_err.max((vv_inv.get(i, j) - target).abs());
            }
        }
        let max_v_inv = v_inv.data.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
        let opts = ScfOptions { conv_energy: 1e-7, conv_density: 1e-6, ..Default::default() };
        let exact = rks(&mol, &basis, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap();
        let fitted = rks(&mol, &basis, XcFunctional::B3lyp, GridLevel::Coarse,
            &ScfOptions { density_fitting: true, ..opts }).unwrap();
        eprintln!(
            "{bname:10}: n_ao={} n_aux={} trunc={} max|V*V^-1-I|={:.2e} max|V^-1|={:.2e} dE={:+.3e}",
            basis.n_ao, engine.aux_basis.n_aux, engine.metric_truncated_modes,
            max_id_err, max_v_inv,
            fitted.total_energy_hartree - exact.total_energy_hartree
        );
        assert!(exact.converged && fitted.converged);
        assert!((fitted.total_energy_hartree - exact.total_energy_hartree).abs() < 1.5e-4);
    }

    // Verify J_RI vs exact 4-center J matrix on H2O / 6-31G(d)
    let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let wf_df = calchem_scf::rhf(
        &mol,
        &basis,
        &ScfOptions {
            max_iter: 60,
            conv_energy: 1e-7,
            conv_density: 1e-7,
            density_fitting: true,
            ..Default::default()
        },
    )
    .unwrap();
    let eng = RiJEngine::new(&mol, &basis).unwrap();
    let p_mat = calchem_tensor::Matrix2D {
        rows: basis.n_ao,
        cols: basis.n_ao,
        data: wf_df.density_alpha.clone(),
    };
    let j1 = eng.build_coulomb(&basis, &p_mat);
    let eri = calchem_int::compute_eri(&basis);
    let n = basis.n_ao;
    let mut j_exact = vec![0.0; n * n];
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    sum += eri.get(mu, nu, lam, sig) * p_mat.get(lam, sig);
                }
            }
            j_exact[mu * n + nu] = sum;
        }
    }
    let j_max_diff: f64 = j1
        .data
        .iter()
        .zip(&j_exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let j_rms_diff: f64 = (j1
        .data
        .iter()
        .zip(&j_exact)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / ((n * n) as f64))
        .sqrt();
    let e_j_ri: f64 = 0.5 * p_mat.data.iter().zip(&j1.data).map(|(p, j)| p * j).sum::<f64>();
    let e_j_exact: f64 = 0.5 * p_mat.data.iter().zip(&j_exact).map(|(p, j)| p * j).sum::<f64>();
    eprintln!(
        "H2O 6-31G(d) J_RI vs J_exact: max|dJ|={j_max_diff:.3e} rms|dJ|={j_rms_diff:.3e} E_J(RI)={e_j_ri:.10} E_J(exact)={e_j_exact:.10} dE_J={:+.3e}",
        e_j_ri - e_j_exact
    );
    assert!(j_max_diff < 2e-4, "max|J_RI - J_exact|={j_max_diff:.3e}");
    assert!(j_rms_diff < 5e-5, "rms|J_RI - J_exact|={j_rms_diff:.3e}");
}

#[test]
fn sulfur_631gd_auto_jfit_full_rank_and_accurate() {
    // Regression for 3rd-row element (Sulfur) conditioning and accuracy under per-L AutoAux:
    // Must stay full-rank (metric_truncated_modes == 0, required for analytical RI-J gradients)
    // and match exact 4c B3LYP/6-31G(d) within 1.5e-4 Eh.
    let h2s = Molecule::new(
        vec![
            Atom::from_angstrom("S", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.96, 0.92]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.96, 0.92]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("6-31g(d)", &h2s).unwrap();
    let engine = RiJEngine::new(&h2s, &basis).unwrap();
    assert_eq!(
        engine.metric_truncated_modes, 0,
        "H2S / 6-31G(d) auto-jfit must be full-rank (trunc == 0), got {}",
        engine.metric_truncated_modes
    );
    let opts = ScfOptions {
        conv_energy: 1e-8,
        conv_density: 1e-7,
        ..Default::default()
    };
    let exact = rks(&h2s, &basis, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap();
    let fitted = rks(
        &h2s,
        &basis,
        XcFunctional::B3lyp,
        GridLevel::Coarse,
        &ScfOptions {
            density_fitting: true,
            ..opts
        },
    )
    .unwrap();
    let de = (fitted.total_energy_hartree - exact.total_energy_hartree).abs();
    assert!(
        de < 1.5e-4,
        "H2S / 6-31G(d) auto-jfit |dE| = {de:.3e} Eh exceeds 1.5e-4 Eh"
    );
}

#[test]
fn gpu_rij_polarized_aux_f_shell_executes_on_gpu_and_matches_cpu() {
    let h2o = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
    let grid = MolecularGrid::build_for_basis(&h2o, &basis, GridLevel::Fine);
    for choice in [
        calchem_model::JFitBasis::Auto,
        calchem_model::JFitBasis::Def2UpToF,
    ] {
        let opts_cpu = ScfOptions {
            density_fitting: true,
            jfit_basis: choice,
            use_gpu: false,
            ..Default::default()
        };
        let opts_gpu = ScfOptions {
            density_fitting: true,
            jfit_basis: choice,
            use_gpu: true,
            ..Default::default()
        };
        let opts_exact = ScfOptions {
            density_fitting: false,
            use_gpu: false,
            ..Default::default()
        };
        let cpu = calchem_scf::rks_with_grid(&h2o, &basis, XcFunctional::B3lyp, &grid, &opts_cpu).unwrap();
        let gpu = calchem_scf::rks_with_grid(&h2o, &basis, XcFunctional::B3lyp, &grid, &opts_gpu).unwrap();
        let exact = calchem_scf::rks_with_grid(&h2o, &basis, XcFunctional::B3lyp, &grid, &opts_exact).unwrap();
        assert!(
            calchem_int::RiJEngine::with_basis(&h2o, &basis, false, choice)
                .unwrap()
                .aux_basis
                .shells
                .iter()
                .any(|s| s.shell_type.angular_momentum() == 3),
            "Expected polarized 6-31G(d) auxiliary basis ({choice:?}) to contain F-shells (L=3)"
        );
        let de_cpu_gpu = (gpu.total_energy_hartree - cpu.total_energy_hartree).abs();
        assert!(
            de_cpu_gpu < 1e-10,
            "H2O B3LYP/6-31G(d) RI-J GPU vs CPU ({choice:?}) |dE| = {de_cpu_gpu:.3e} Eh >= 1e-10 Eh"
        );
        let de_exact = (gpu.total_energy_hartree - exact.total_energy_hartree).abs();
        let exact_tol = match choice {
            calchem_model::JFitBasis::Auto => 1.5e-4,
            calchem_model::JFitBasis::Def2UpToF => 2.0e-4,
        };
        assert!(
            de_exact < exact_tol,
            "H2O B3LYP/6-31G(d) RI-J GPU vs Exact4c ({choice:?}) |dE| = {de_exact:.3e} Eh >= {exact_tol:.1e} Eh"
        );
        if calchem_int::is_cuda_available() {
            assert!(
                gpu.diagnostics.gpu_kernel_invocations >= 1,
                "Expected GPU F-shell RI-J kernel invocations >= 1 under {choice:?}, got {}",
                gpu.diagnostics.gpu_kernel_invocations
            );
            assert_eq!(
                gpu.diagnostics.gpu_fallback_invocations, 0,
                "Expected zero GPU fallback under {choice:?}, got {}",
                gpu.diagnostics.gpu_fallback_invocations
            );
        }
    }
}
