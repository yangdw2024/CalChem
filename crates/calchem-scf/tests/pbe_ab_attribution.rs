use calchem_basis::BasisSet;
use calchem_dft::vxc::evaluate_vxc_rks_pbe_ab;
use calchem_dft::xc::PbeAbParams;
use calchem_dft::{GridLevel, MolecularGrid, XcFunctional};
use calchem_int::{
    compute_kinetic, compute_nuclear, compute_overlap, compute_sparse_eri,
};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rks, DiisState};
use calchem_tensor::{eigh_sym, matmul, transpose, Matrix2D};

fn water_bench() -> Molecule {
    Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    )
}

fn run_rks_ab_scf(
    mol: &Molecule,
    basis: &BasisSet,
    grid: &MolecularGrid,
    params: PbeAbParams,
    exx_frac: f64,
) -> f64 {
    let n = basis.n_ao;
    let n_occ = mol.nelectron().unwrap() / 2;
    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v_nuc = compute_nuclear(basis, mol);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v_nuc.data[i];
    }
    let eri = compute_sparse_eri(basis);
    let (s_evals, s_evecs) = eigh_sym(&s, 200, 1e-14).unwrap();
    let mut x_mat = Matrix2D::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += s_evecs.get(i, k) * s_evecs.get(j, k) / s_evals[k].sqrt();
            }
            x_mat.set(i, j, sum);
        }
    }
    let x_t = transpose(&x_mat);
    let v_nn = mol.nuclear_repulsion_energy();

    let f0 = matmul(&x_t, &matmul(&h_core, &x_mat));
    let (_, c0_p) = eigh_sym(&f0, 200, 1e-13).unwrap();
    let c0 = matmul(&x_mat, &c0_p);
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for m in 0..n_occ {
                sum += 2.0 * c0.get(mu, m) * c0.get(nu, m);
            }
            p.set(mu, nu, sum);
        }
    }

    let mut diis = DiisState::new(8);
    let mut e_prev = 0.0;
    for iter in 0..80 {
        let jk = eri.contract_rks(&p.data, exx_frac);
        let (_, _, e_xc, v_xc) =
            evaluate_vxc_rks_pbe_ab(basis, grid, params, 1.0 - exx_frac, &p);

        let mut f = Matrix2D::zeros(n, n);
        let mut e_one = 0.0;
        let mut e_two = 0.0;
        for i in 0..n * n {
            f.data[i] = h_core.data[i] + jk[i] + v_xc.data[i];
            e_one += p.data[i] * h_core.data[i];
            e_two += 0.5 * p.data[i] * jk[i];
        }
        let e_tot = v_nn + e_one + e_two + e_xc;

        // DIIS commutator FDS - SDF
        let fds = matmul(&f, &matmul(&p, &s));
        let sdf = matmul(&s, &matmul(&p, &f));
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fds.data[i] - sdf.data[i];
        }
        let f_diis = diis.push_and_extrapolate(&f, &err);
        let f_prime = matmul(&x_t, &matmul(&f_diis, &x_mat));
        let (_, c_p) = eigh_sym(&f_prime, 200, 1e-13).unwrap();
        let c = matmul(&x_mat, &c_p);

        let mut p_new = Matrix2D::zeros(n, n);
        let mut rms_dp = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                let mut sum = 0.0;
                for m in 0..n_occ {
                    sum += 2.0 * c.get(mu, m) * c.get(nu, m);
                }
                let diff = sum - p.get(mu, nu);
                rms_dp += diff * diff;
                p_new.set(mu, nu, sum);
            }
        }
        rms_dp = (rms_dp / ((n * n) as f64)).sqrt();
        if iter > 0 && (e_tot - e_prev).abs() < 1e-11 && rms_dp < 1e-9 {
            return e_tot;
        }
        e_prev = e_tot;
        p = p_new;
    }
    e_prev
}

#[test]
fn pbe_and_pbe0_gaussian_parameter_ab_attribution() {
    let mol = water_bench();
    let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let grid = MolecularGrid::build_streaming_for_basis(&mol, &basis, GridLevel::UltraFine);
    let opts = ScfOptions {
        conv_energy: 1e-11,
        conv_density: 1e-9,
        max_iter: 120,
        ..Default::default()
    };

    // 1. Fixed density from converged CalChem PBE / 6-31G(d) UltraFine SCF
    let wf_pbe = rks(&mol, &basis, XcFunctional::Pbe, GridLevel::UltraFine, &opts).unwrap();
    let wf_pbe0 = rks(&mol, &basis, XcFunctional::Pbe0, GridLevel::UltraFine, &opts).unwrap();
    let mut p_fixed = Matrix2D::zeros(basis.n_ao, basis.n_ao);
    for (dst, &src) in p_fixed.data.iter_mut().zip(wf_pbe.density_alpha.iter()) {
        *dst = 2.0 * src;
    }
    let p_fixed = &p_fixed;

    let g16_pbe = -76.1130610886;
    let g16_pbe0 = -76.1248351966;

    // Variants (Lehtola & Marques arXiv:2307.07474v2 §V.1 for G16 PBE X)
    let p_analytic = PbeAbParams::CALCHEM_ANALYTIC;
    let p_g16_x = PbeAbParams {
        kappa: 0.804000423825475,
        mu: 0.219510240580611,
        ..p_analytic
    };
    let p_g16_c_bg = PbeAbParams {
        beta: 0.066725,
        gamma: 0.031091,
        ..p_analytic
    };
    let p_pw92_031091 = PbeAbParams {
        pw92_a0: 0.031091,
        pw92_a1: 0.0155455,
        ..p_analytic
    };
    let p_pw92_exact_gamma = PbeAbParams {
        pw92_a0: 0.0310906908696549,
        pw92_a1: 0.5 * 0.0310906908696549,
        ..p_analytic
    };
    let p_g16_comb_pw92_1992 = PbeAbParams {
        kappa: 0.804000423825475,
        mu: 0.219510240580611,
        beta: 0.066725,
        gamma: 0.031091,
        pw92_a0: 0.0310907,
        pw92_a1: 0.01554535,
    };
    let p_g16_comb_pw92_031091 = PbeAbParams {
        kappa: 0.804000423825475,
        mu: 0.219510240580611,
        beta: 0.066725,
        gamma: 0.031091,
        pw92_a0: 0.031091,
        pw92_a1: 0.0155455,
    };

    let (ex0, ec0, exc0, _) = evaluate_vxc_rks_pbe_ab(&basis, &grid, p_analytic, 1.0, p_fixed);
    println!(
        "[FIXED-DENSITY PBE] Baseline analytic: Ex={:.10} Ec={:.10} Exc={:.10}",
        ex0, ec0, exc0
    );
    for (label, var) in [
        ("G16_X_only (LM2023 kappa/mu)", p_g16_x),
        ("G16_C_only (beta=0.066725,gamma=0.031091)", p_g16_c_bg),
        ("PW92_a0=0.031091 only", p_pw92_031091),
        ("PW92_a0=(1-ln2)/pi^2 only", p_pw92_exact_gamma),
        ("Combined G16_X + G16_C + PW92(1992)", p_g16_comb_pw92_1992),
        ("Combined G16_X + G16_C + PW92(0.031091)", p_g16_comb_pw92_031091),
    ] {
        let (ex, ec, exc, _) = evaluate_vxc_rks_pbe_ab(&basis, &grid, var, 1.0, p_fixed);
        println!(
            "[FIXED-DENSITY] {:<42}: dEx={:+.3e} dEc={:+.3e} dExc={:+.3e}",
            label,
            ex - ex0,
            ec - ec0,
            exc - exc0
        );
    }

    // 2. Full SCF comparison for PBE and PBE0
    println!(
        "[SCF BASELINE] CalChem PBE={:.10} (vs G16={:.10}, dE={:+.3e})",
        wf_pbe.total_energy_hartree,
        g16_pbe,
        wf_pbe.total_energy_hartree - g16_pbe
    );
    println!(
        "[SCF BASELINE] CalChem PBE0={:.10} (vs G16={:.10}, dE={:+.3e})",
        wf_pbe0.total_energy_hartree,
        g16_pbe0,
        wf_pbe0.total_energy_hartree - g16_pbe0
    );

    for (label, var) in [
        ("G16_X + G16_C + PW92(1992)", p_g16_comb_pw92_1992),
        ("G16_X + G16_C + PW92(0.031091)", p_g16_comb_pw92_031091),
    ] {
        let e_pbe = run_rks_ab_scf(&mol, &basis, &grid, var, 0.0);
        let e_pbe0 = run_rks_ab_scf(&mol, &basis, &grid, var, 0.25);
        println!(
            "[SCF {}] PBE={:.10} (vs G16 dE={:+.3e}) | PBE0={:.10} (vs G16 dE={:+.3e})",
            label,
            e_pbe,
            e_pbe - g16_pbe,
            e_pbe0,
            e_pbe0 - g16_pbe0
        );
    }
}
