use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid, XcFunctional, evaluate_vxc_rks, evaluate_vxc_uks};
use calchem_int::{compute_ecp, compute_eri, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{DiisState, rhf, rks_with_grid};
use calchem_tddft::{
    EV_TO_NM, SpinManifold, TddftOptions, broaden_uv_vis_spectrum, compute_ao_dipole_matrices,
    orbital_hessian_apb_quad, run_tddft, run_tddft_from_scf,
};
use calchem_tensor::{Matrix2D, Tensor4D, eigh_sym, matmul, rms_diff, transpose};

fn water_mol(shift_ang: [f64; 3]) -> Molecule {
    Molecule::new(
        vec![
            Atom::from_angstrom("O", [shift_ang[0], shift_ang[1], 0.1173 + shift_ang[2]]).unwrap(),
            Atom::from_angstrom(
                "H",
                [shift_ang[0], 0.7572 + shift_ang[1], -0.4692 + shift_ang[2]],
            )
            .unwrap(),
            Atom::from_angstrom(
                "H",
                [shift_ang[0], -0.7572 + shift_ang[1], -0.4692 + shift_ang[2]],
            )
            .unwrap(),
        ],
        0,
        1,
    )
}

#[test]
fn test_davidson_matches_dense_tda_and_rpa_singlet_and_triplet() {
    let mol = water_mol([0.0, 0.0, 0.0]);
    let bs = BasisSet::from_name("6-31g", &mol).unwrap();

    for xc in [None, Some(XcFunctional::B3lyp)] {
        for tda in [true, false] {
            let mut opts_dense = TddftOptions {
                n_states: 4,
                tda,
                manifold: SpinManifold::Both,
                grid_level: GridLevel::Coarse,
                max_iter: 80,
                conv_tol: 1e-10,
                force_davidson: false,
                scf_options: ScfOptions {
                    conv_energy: 1e-12,
                    conv_density: 1e-10,
                    ..ScfOptions::default()
                },
            };
            let res_dense = run_tddft(&mol, &bs, xc, &opts_dense).unwrap();

            opts_dense.force_davidson = true;
            let res_dav = run_tddft(&mol, &bs, xc, &opts_dense).unwrap();

            assert!(
                res_dav.davidson_iterations_singlet >= 2,
                "Forced Davidson should run iteratively"
            );
            assert_eq!(res_dense.states.len(), res_dav.states.len());

            for (sd, sv) in res_dense.states.iter().zip(res_dav.states.iter()) {
                assert_eq!(sd.multiplicity, sv.multiplicity);
                let de = (sd.excitation_energy_hartree - sv.excitation_energy_hartree).abs();
                let df = (sd.oscillator_strength - sv.oscillator_strength).abs();
                assert!(
                    de < 1e-9,
                    "xc={xc:?} tda={tda} state {} energy mismatch: dense={} vs dav={} (err={de:.3e})",
                    sd.label,
                    sd.excitation_energy_hartree,
                    sv.excitation_energy_hartree
                );
                assert!(
                    df < 1e-8,
                    "xc={xc:?} tda={tda} state {} osc mismatch: dense={} vs dav={} (err={df:.3e})",
                    sd.label,
                    sd.oscillator_strength,
                    sv.oscillator_strength
                );
            }
        }
    }
}

fn eval_uks_total_energy(
    mol: &Molecule,
    bs: &BasisSet,
    grid: &MolecularGrid,
    xc: XcFunctional,
    h_core: &Matrix2D,
    eri: &Tensor4D,
    pa: &Matrix2D,
    pb: &Matrix2D,
) -> f64 {
    let n = bs.n_ao;
    let c_hf = xc.exact_exchange_fraction();
    let e_nuc = bs.nuclear_repulsion_energy(mol);
    let mut e_1e = 0.0;
    let mut e_2e = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            let pt_mn = pa.get(mu, nu) + pb.get(mu, nu);
            e_1e += pt_mn * h_core.get(mu, nu);
            let mut j = 0.0;
            let mut ka = 0.0;
            let mut kb = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    let coul = eri.get(mu, nu, lam, sig);
                    j += (pa.get(lam, sig) + pb.get(lam, sig)) * coul;
                    if c_hf.abs() > 1e-15 {
                        let exch = eri.get(mu, lam, nu, sig);
                        ka += pa.get(lam, sig) * exch;
                        kb += pb.get(lam, sig) * exch;
                    }
                }
            }
            e_2e += 0.5
                * (pa.get(mu, nu) * (j - c_hf * ka) + pb.get(mu, nu) * (j - c_hf * kb));
        }
    }
    let (e_xc, _, _) = evaluate_vxc_uks(bs, grid, xc, pa, pb);
    e_nuc + e_1e + e_2e + e_xc
}

#[test]
fn test_thouless_orbital_rotation_hessian_singlet_and_triplet() {
    let mol = water_mol([0.0, 0.0, 0.0]);
    let bs = BasisSet::from_name("6-31g", &mol).unwrap();
    let grid_level = GridLevel::Coarse;
    let grid = MolecularGrid::build(&mol, grid_level);
    let n = bs.n_ao;

    let t = compute_kinetic(&bs);
    let v = compute_nuclear(&bs, &mol);
    let ecp = compute_ecp(&bs);
    let mut h_core = Matrix2D::zeros(n, n);
    for idx in 0..(n * n) {
        h_core.data[idx] = t.data[idx] + v.data[idx] + ecp.data[idx];
    }
    let eri = compute_eri(&bs);

    let scf_opts = ScfOptions {
        max_iter: 120,
        conv_energy: 1e-13,
        conv_density: 1e-11,
        ..ScfOptions::default()
    };

    for func in [XcFunctional::Pbe, XcFunctional::B3lyp] {
        let wfn = rks_with_grid(&mol, &bs, func, &grid, &scf_opts).unwrap();
        let n_occ = 5usize;
        let n_virt = n - n_occ;

        // Build normalized superposition phi_u = sum_i alpha_i phi_i, phi_v = sum_a beta_a phi_a
        // so u_ov = alpha (x) beta tests both diagonal and off-diagonal (A + B) couplings!
        let mut alpha = vec![0.35, -0.45, 0.50, -0.40, 0.52];
        let norm_a: f64 = alpha.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut alpha {
            *x /= norm_a;
        }
        let mut beta: Vec<f64> = (0..n_virt)
            .map(|a| (a as f64 * 1.3 + 0.7).cos())
            .collect();
        let norm_b: f64 = beta.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut beta {
            *x /= norm_b;
        }

        let mut phi_u = vec![0.0f64; n];
        let mut phi_v = vec![0.0f64; n];
        for mu in 0..n {
            for i in 0..n_occ {
                phi_u[mu] += alpha[i] * wfn.mo_coefficients[mu * n + i];
            }
            for a in 0..n_virt {
                phi_v[mu] += beta[a] * wfn.mo_coefficients[mu * n + (n_occ + a)];
            }
        }

        let mut u_ov = vec![0.0f64; n_occ * n_virt];
        for i in 0..n_occ {
            for a in 0..n_virt {
                u_ov[i * n_virt + a] = alpha[i] * beta[a];
            }
        }

        let build_rotated_density = |tau: f64| -> Matrix2D {
            let s = tau.sin();
            let c = tau.cos();
            let sc = s * c;
            let s2 = s * s;
            let mut p = Matrix2D::zeros(n, n);
            for mu in 0..n {
                for nu in 0..n {
                    let p0 = 0.5 * wfn.density_alpha[mu * n + nu];
                    let lin = sc * (phi_u[mu] * phi_v[nu] + phi_v[mu] * phi_u[nu]);
                    let quad = s2 * (phi_v[mu] * phi_v[nu] - phi_u[mu] * phi_u[nu]);
                    p.set(mu, nu, p0 + lin + quad);
                }
            }
            p
        };

        let p0 = build_rotated_density(0.0);
        let e0 = eval_uks_total_energy(&mol, &bs, &grid, func, &h_core, &eri, &p0, &p0);
        assert!(
            (e0 - wfn.total_energy_hartree).abs() < 1e-11,
            "UKS energy at theta=0 must match RKS energy"
        );

        let theta = 5e-3;
        for (triplet, sign_b) in [(false, 1.0), (true, -1.0)] {
            let apb_ana = orbital_hessian_apb_quad(
                &mol,
                &bs,
                &wfn,
                Some(func),
                grid_level,
                &u_ov,
                triplet,
            )
            .unwrap();

            let eval_e = |th: f64| -> f64 {
                let ta = th / std::f64::consts::SQRT_2;
                let tb = sign_b * ta;
                let pa = build_rotated_density(ta);
                let pb = build_rotated_density(tb);
                eval_uks_total_energy(&mol, &bs, &grid, func, &h_core, &eri, &pa, &pb)
            };

            let em2 = eval_e(-2.0 * theta);
            let em1 = eval_e(-theta);
            let ep1 = eval_e(theta);
            let ep2 = eval_e(2.0 * theta);

            // Exact 4th-order coefficient of theta^2 in E(theta) = E(0) + [U^T(A+B)U] theta^2 + O(theta^4)
            let apb_fd = (-em2 + 16.0 * em1 - 30.0 * e0 + 16.0 * ep1 - ep2) / (24.0 * theta * theta);
            let err = (apb_ana - apb_fd).abs();
            assert!(
                err < 1e-8,
                "Thouless Hessian mismatch for {func:?} (triplet={triplet}): ana={apb_ana:.10} vs fd={apb_fd:.10} (err={err:.3e})"
            );
        }
    }
}

/// Self-consistent finite-field closed-shell RHF / RKS solver with $H_{\text{core}}(\mathcal{E}_q) = H_{\text{core}} + \mathcal{E}_q d_q^{\text{AO}}$.
fn run_finite_field_scf(
    mol: &Molecule,
    bs: &BasisSet,
    xc: Option<(XcFunctional, &MolecularGrid)>,
    field_axis: usize,
    field_au: f64,
) -> f64 {
    let n = bs.n_ao;
    let n_occ = bs.validate_rhf_valence(mol).unwrap().0;
    let s = compute_overlap(bs);
    let t = compute_kinetic(bs);
    let v = compute_nuclear(bs, mol);
    let ecp = compute_ecp(bs);
    let dips = compute_ao_dipole_matrices(bs);
    let eri = compute_eri(bs);
    let e_nuc = bs.nuclear_repulsion_energy(mol);

    let mut h_core = Matrix2D::zeros(n, n);
    for idx in 0..(n * n) {
        h_core.data[idx] =
            t.data[idx] + v.data[idx] + ecp.data[idx] + field_au * dips[field_axis].data[idx];
    }

    let (s_vals, s_vecs) = eigh_sym(&s, 120, 1e-14).unwrap();
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    let diag_p = |fock: &Matrix2D| -> Matrix2D {
        let fp = matmul(&matmul(&xt, fock), &x);
        let (_, cp) = eigh_sym(&fp, 120, 1e-14).unwrap();
        let c = matmul(&x, &cp);
        let mut p = Matrix2D::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                let mut sum = 0.0;
                for i in 0..n_occ {
                    sum += c.get(mu, i) * c.get(nu, i);
                }
                p.set(mu, nu, 2.0 * sum);
            }
        }
        p
    };

    let c_hf = xc.map_or(1.0, |(f, _)| f.exact_exchange_fraction());
    let mut p = diag_p(&h_core);
    let mut diis = DiisState::new(8);
    let mut e_prev = 0.0;

    for _iter in 0..120 {
        let mut g_2e = Matrix2D::zeros(n, n);
        for mu in 0..n {
            for nu in mu..n {
                let mut g = 0.0;
                for lam in 0..n {
                    for sig in 0..n {
                        let pls = p.get(lam, sig);
                        let j = eri.get(mu, nu, lam, sig);
                        let k = if c_hf.abs() > 1e-15 {
                            eri.get(mu, lam, nu, sig)
                        } else {
                            0.0
                        };
                        g += pls * (j - 0.5 * c_hf * k);
                    }
                }
                g_2e.set(mu, nu, g);
                g_2e.set(nu, mu, g);
            }
        }

        let (e_xc, v_xc) = match xc {
            None => (0.0, Matrix2D::zeros(n, n)),
            Some((func, grid)) => evaluate_vxc_rks(bs, grid, func, &p),
        };

        let mut fock = Matrix2D::zeros(n, n);
        let mut e_elec = e_xc;
        for idx in 0..(n * n) {
            fock.data[idx] = h_core.data[idx] + g_2e.data[idx] + v_xc.data[idx];
            e_elec += p.data[idx] * (h_core.data[idx] + 0.5 * g_2e.data[idx]);
        }
        let e_tot = e_nuc + e_elec;

        let fps = matmul(&matmul(&fock, &p), &s);
        let spf = matmul(&matmul(&s, &p), &fock);
        let mut err_ao = Matrix2D::zeros(n, n);
        for idx in 0..(n * n) {
            err_ao.data[idx] = fps.data[idx] - spf.data[idx];
        }
        let err_orth = matmul(&matmul(&xt, &err_ao), &x);
        let f_extrap = diis.push_and_extrapolate(&fock, &err_orth);
        let p_new = diag_p(&f_extrap);
        let dp = rms_diff(&p_new, &p);
        p = p_new;
        if (e_tot - e_prev).abs() < 1e-13 && dp < 1e-11 {
            return e_tot;
        }
        e_prev = e_tot;
    }
    panic!("Finite-field SCF failed to converge");
}

#[test]
fn test_rpa_exact_static_polarizability_sum_rule_and_origin_invariance() {
    let mol = water_mol([0.0, 0.0, 0.0]);
    let bs = BasisSet::from_name("6-31g", &mol).unwrap();
    let grid = MolecularGrid::build(&mol, GridLevel::Coarse);
    let n_ov = 5 * (bs.n_ao - 5);

    let scf_opts = ScfOptions {
        max_iter: 120,
        conv_energy: 1e-13,
        conv_density: 1e-11,
        ..ScfOptions::default()
    };

    for xc in [None, Some(XcFunctional::B3lyp)] {
        let wfn = match xc {
            None => rhf(&mol, &bs, &scf_opts).unwrap(),
            Some(f) => rks_with_grid(&mol, &bs, f, &grid, &scf_opts).unwrap(),
        };
        let td_opts = TddftOptions {
            n_states: n_ov,
            tda: false,
            manifold: SpinManifold::Singlet,
            grid_level: GridLevel::Coarse,
            max_iter: 80,
            conv_tol: 1e-10,
            force_davidson: false,
            scf_options: scf_opts.clone(),
        };
        let res = run_tddft_from_scf(&mol, &bs, &wfn, xc, &td_opts).unwrap();
        assert_eq!(res.states.len(), n_ov);

        let xc_grid = xc.map(|f| (f, &grid));
        let e0 = run_finite_field_scf(&mol, &bs, xc_grid, 2, 0.0);
        let eps_f = 5e-4;

        for axis in [1usize, 2usize] {
            let ep = run_finite_field_scf(&mol, &bs, xc_grid, axis, eps_f);
            let em = run_finite_field_scf(&mol, &bs, xc_grid, axis, -eps_f);
            let alpha_ff = -(ep + em - 2.0 * e0) / (eps_f * eps_f);

            let alpha_sos: f64 = res
                .states
                .iter()
                .map(|st| {
                    let mu_q = st.transition_dipole_au[axis];
                    2.0 * (mu_q * mu_q) / st.excitation_energy_hartree
                })
                .sum();

            let rel_err = (alpha_sos - alpha_ff).abs() / alpha_ff.abs();
            assert!(
                rel_err < 1e-5,
                "Static polarizability sum rule failed for xc={xc:?} axis={axis}: SOS={alpha_sos:.8} vs FF={alpha_ff:.8} (rel_err={rel_err:.3e})"
            );
        }
    }

    // Origin / translation invariance of vertical excitation energies and transition dipoles
    let mol_shifted = water_mol([3.4, -2.1, 4.7]);
    let bs_shifted = BasisSet::from_name("6-31g", &mol_shifted).unwrap();
    let opts_inv = TddftOptions {
        n_states: 5,
        tda: false,
        manifold: SpinManifold::Singlet,
        grid_level: GridLevel::Coarse,
        conv_tol: 1e-9,
        scf_options: scf_opts,
        ..TddftOptions::default()
    };
    let r0 = run_tddft(&mol, &bs, Some(XcFunctional::B3lyp), &opts_inv).unwrap();
    let r1 = run_tddft(&mol_shifted, &bs_shifted, Some(XcFunctional::B3lyp), &opts_inv).unwrap();
    for (s0, s1) in r0.states.iter().zip(r1.states.iter()) {
        assert!(
            (s0.excitation_energy_ev - s1.excitation_energy_ev).abs() < 1e-7,
            "Translated excitation energy must be invariant"
        );
        assert!(
            (s0.oscillator_strength - s1.oscillator_strength).abs() < 1e-7,
            "Translated oscillator strength must be invariant"
        );
    }
}

#[test]
fn test_spectroscopy_benchmarks_ethylene_formaldehyde_ecp_and_uv_vis() {
    // 1. Ethylene (C2H4 / 6-31G, n_ov = 8 * 14 = 112 > 64 -> iterative Block Davidson!)
    let c2h4 = Molecule::new(
        vec![
            Atom::from_angstrom("C", [0.0, 0.0, 0.6695]).unwrap(),
            Atom::from_angstrom("C", [0.0, 0.0, -0.6695]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.9289, 1.2321]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.9289, 1.2321]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.9289, -1.2321]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.9289, -1.2321]).unwrap(),
        ],
        0,
        1,
    );
    let bs_eth = BasisSet::from_name("6-31g", &c2h4).unwrap();
    let opts_eth = TddftOptions {
        n_states: 4,
        tda: false,
        manifold: SpinManifold::Both,
        grid_level: GridLevel::Coarse,
        conv_tol: 1e-6,
        ..TddftOptions::default()
    };
    let eth_res = run_tddft(&c2h4, &bs_eth, Some(XcFunctional::B3lyp), &opts_eth).unwrap();

    let t1 = eth_res
        .states
        .iter()
        .find(|s| s.multiplicity == 3 && s.manifold_index == 1)
        .unwrap();
    assert!(
        (3.8..=4.9).contains(&t1.excitation_energy_ev),
        "Ethylene T1 (pi->pi*) expected in [3.8, 4.9] eV, got {:.3} eV",
        t1.excitation_energy_ev
    );
    assert_eq!(t1.oscillator_strength, 0.0);

    let bright_s = eth_res
        .states
        .iter()
        .filter(|s| s.multiplicity == 1)
        .max_by(|a, b| a.oscillator_strength.total_cmp(&b.oscillator_strength))
        .unwrap();
    assert!(
        bright_s.oscillator_strength > 0.25,
        "Ethylene bright pi->pi* singlet should have f > 0.25, got {:.4}",
        bright_s.oscillator_strength
    );
    assert!(
        (7.2..=8.6).contains(&bright_s.excitation_energy_ev),
        "Ethylene bright pi->pi* singlet expected in [7.2, 8.6] eV, got {:.3} eV",
        bright_s.excitation_energy_ev
    );
    assert!(
        eth_res.delta_est_ev.unwrap() > 2.5,
        "Ethylene Delta E_ST must be strongly positive (> 2.5 eV)"
    );

    // 2. Formaldehyde (CH2O / 6-31G): symmetry-forbidden dark n -> pi* S1!
    let ch2o = Molecule::new(
        vec![
            Atom::from_angstrom("C", [0.0, 0.0, -0.5290]).unwrap(),
            Atom::from_angstrom("O", [0.0, 0.0, 0.6770]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.9400, -1.1160]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.9400, -1.1160]).unwrap(),
        ],
        0,
        1,
    );
    let bs_ch2o = BasisSet::from_name("6-31g", &ch2o).unwrap();
    let ch2o_res = run_tddft(
        &ch2o,
        &bs_ch2o,
        Some(XcFunctional::B3lyp),
        &TddftOptions {
            n_states: 3,
            tda: false,
            manifold: SpinManifold::Singlet,
            grid_level: GridLevel::Coarse,
            ..TddftOptions::default()
        },
    )
    .unwrap();
    let ch2o_s1 = &ch2o_res.states[0];
    assert!(
        (3.5..=4.6).contains(&ch2o_s1.excitation_energy_ev),
        "Formaldehyde S1 (n->pi*) expected in [3.5, 4.6] eV, got {:.3} eV",
        ch2o_s1.excitation_energy_ev
    );
    assert!(
        ch2o_s1.oscillator_strength < 1e-5,
        "Formaldehyde C2v A2 n->pi* S1 must be electric-dipole forbidden (f < 1e-5), got {:.3e}",
        ch2o_s1.oscillator_strength
    );

    // 3. Heavy-element scalar relativistic ECP TD-DFT (HBr / LANL2DZ) + UV-Vis broadening
    let hbr = Molecule::new(
        vec![
            Atom::from_angstrom("Br", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.0, 1.414]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hbr = BasisSet::from_name("lanl2dz", &hbr).unwrap();
    let hbr_res = run_tddft(
        &hbr,
        &bs_hbr,
        Some(XcFunctional::B3lyp),
        &TddftOptions {
            n_states: 4,
            tda: true,
            manifold: SpinManifold::Singlet,
            grid_level: GridLevel::Coarse,
            ..TddftOptions::default()
        },
    )
    .unwrap();
    assert_eq!(hbr_res.n_occ, 4);
    assert!(hbr_res.optical_gap_ev.unwrap() > 4.0);

    let spec = broaden_uv_vis_spectrum(&eth_res.states, 120.0, 300.0, 361, 0.4).unwrap();
    let (peak_idx, &peak_eps) = spec
        .epsilon_l_mol_cm
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    assert!(peak_eps > 1.0e4);
    let expected_peak_nm = EV_TO_NM / bright_s.excitation_energy_ev;
    assert!(
        (spec.wavelengths_nm[peak_idx] - expected_peak_nm).abs() < 1.0,
        "UV-Vis peak wavelength {:.2} nm should match bright state {:.2} nm",
        spec.wavelengths_nm[peak_idx],
        expected_peak_nm
    );
}
