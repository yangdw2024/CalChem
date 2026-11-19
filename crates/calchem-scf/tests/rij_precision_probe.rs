use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
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
