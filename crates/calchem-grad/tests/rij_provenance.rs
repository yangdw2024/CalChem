use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_int::RiJEngine;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_tensor::Matrix2D;

#[test]
fn ri_j_fixed_density_coulomb_gradient_matches_central_finite_difference() {
    let coords = [
        ("O", [0.0, 0.0, 0.12]),
        ("H", [0.0, 1.43, -0.96]),
        ("H", [0.15, -1.41, -0.92]),
    ];
    let atoms: Vec<Atom> = coords
        .iter()
        .map(|(s, r)| Atom::from_bohr(s, *r).unwrap())
        .collect();
    let mol = Molecule::new(atoms, 0, 1);
    let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let n = basis.n_ao;

    let mut p = Matrix2D::zeros(n, n);
    for i in 0..n {
        p.set(i, i, 0.4 + 0.02 * (i as f64));
        for j in 0..i {
            let v = 0.03 * (((i * 7 + j * 5 + 1) as f64).sin());
            p.set(i, j, v);
            p.set(j, i, v);
        }
    }

    let rij = RiJEngine::new(&mol, &basis).unwrap();
    assert_eq!(rij.metric_truncated_modes, 0);
    let grad_ana = rij
        .compute_coulomb_gradient(mol.atoms.len(), &basis, &p)
        .unwrap();

    // Check translational invariance
    for q in 0..3 {
        let sum_q: f64 = grad_ana.iter().map(|g| g[q]).sum();
        assert!(
            sum_q.abs() < 1e-11,
            "RI-J Coulomb gradient translational invariance violated on axis {q}: {sum_q:.3e}"
        );
    }

    let h = 1e-4;
    let ej_at = |m: &Molecule| -> f64 {
        let bs = BasisSet::from_name("6-31g(d)", m).unwrap();
        let eng = RiJEngine::new(m, &bs).unwrap();
        let j = eng.build_coulomb(&bs, &p);
        let mut e = 0.0;
        for idx in 0..n * n {
            e += 0.5 * p.data[idx] * j.data[idx];
        }
        e
    };

    for atom_idx in 0..mol.atoms.len() {
        for axis in 0..3 {
            let mut mp = mol.clone();
            mp.atoms[atom_idx].position_bohr[axis] += h;
            let mut mm = mol.clone();
            mm.atoms[atom_idx].position_bohr[axis] -= h;

            let fd = (ej_at(&mp) - ej_at(&mm)) / (2.0 * h);
            let ana = grad_ana[atom_idx][axis];
            let err = (ana - fd).abs();
            assert!(
                err < 1e-6,
                "RI-J fixed-density Coulomb gradient mismatch at atom {atom_idx} axis {axis}: ana={ana:.10e} fd={fd:.10e} err={err:.3e}"
            );
        }
    }
}

#[test]
fn ri_wavefunctions_compute_consistent_analytical_gradients() {
    let make_mol = |z: f64| {
        Molecule::new(
            vec![
                Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, z]).unwrap(),
            ],
            0,
            1,
        )
    };
    let z0 = 1.45;
    let h = 1e-4;
    let mol = make_mol(z0);
    let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let opts = ScfOptions {
        density_fitting: true,
        conv_energy: 1e-10,
        conv_density: 1e-8,
        ..Default::default()
    };

    // 1. RHF + RI-J
    let wf_rhf = calchem_scf::rhf(&mol, &basis, &opts).unwrap();
    assert!(wf_rhf.density_fitting);
    let g_rhf = calchem_grad::rhf_gradient(&mol, &basis, &wf_rhf).unwrap();
    let mol_p = make_mol(z0 + h);
    let mol_m = make_mol(z0 - h);
    let bs_p = BasisSet::from_name("6-31g(d)", &mol_p).unwrap();
    let bs_m = BasisSet::from_name("6-31g(d)", &mol_m).unwrap();
    let e_p = calchem_scf::rhf(&mol_p, &bs_p, &opts)
        .unwrap()
        .total_energy_hartree;
    let e_m = calchem_scf::rhf(&mol_m, &bs_m, &opts)
        .unwrap()
        .total_energy_hartree;
    let fd_rhf = (e_p - e_m) / (2.0 * h);
    assert!(
        (g_rhf[1][2] - fd_rhf).abs() < 1e-6,
        "RHF RI-J grad mismatch: ana={} fd={}",
        g_rhf[1][2],
        fd_rhf
    );

    // 2. RKS (PBE + RI-J, pure DFT with 0 exact exchange)
    let wf_rks = calchem_scf::rks(&mol, &basis, XcFunctional::Pbe, GridLevel::Fine, &opts).unwrap();
    let g_rks =
        calchem_grad::rks_gradient(&mol, &basis, XcFunctional::Pbe, GridLevel::Fine, &wf_rks)
            .unwrap();
    let e_p_rks = calchem_scf::rks(&mol_p, &bs_p, XcFunctional::Pbe, GridLevel::Fine, &opts)
        .unwrap()
        .total_energy_hartree;
    let e_m_rks = calchem_scf::rks(&mol_m, &bs_m, XcFunctional::Pbe, GridLevel::Fine, &opts)
        .unwrap()
        .total_energy_hartree;
    let fd_rks = (e_p_rks - e_m_rks) / (2.0 * h);
    assert!(
        (g_rks[1][2] - fd_rks).abs() < 1e-6,
        "RKS RI-J grad mismatch: ana={} fd={}",
        g_rks[1][2],
        fd_rks
    );

    // 3. UHF + RI-J
    let wf_uhf = calchem_scf::uhf(&mol, &basis, &opts).unwrap();
    let g_uhf = calchem_grad::uhf_gradient(&mol, &basis, &wf_uhf).unwrap();
    assert!((g_uhf[1][2] - fd_rhf).abs() < 1e-6);

    // 4. UKS + RI-J
    let wf_uks = calchem_scf::uks(&mol, &basis, XcFunctional::Pbe, GridLevel::Fine, &opts).unwrap();
    let g_uks =
        calchem_grad::uks_gradient(&mol, &basis, XcFunctional::Pbe, GridLevel::Fine, &wf_uks)
            .unwrap();
    assert!((g_uks[1][2] - fd_rks).abs() < 1e-6);
}

#[test]
fn ri_j_gradient_rejects_rank_truncated_pseudoinverse_metric() {
    use calchem_basis::{primitive_norm, ShellType};
    use calchem_int::{AuxiliaryBasisSet, AuxiliaryShell};

    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("sto-3g", &mol).unwrap();

    // Construct a genuinely linearly dependent auxiliary basis set (two identical s-shells on atom 0)
    // so that invert_coulomb_metric_with_rank_info naturally truncates 1 singular mode (lambda = 0 < 1e-12 * lambda_max).
    let mut aux_basis = AuxiliaryBasisSet::build_auto_jfit(&mol, &basis);
    let dup_exp = 1.0;
    let dup_norm = primitive_norm(dup_exp, [0, 0, 0]);
    for _ in 0..2 {
        let ao_off = aux_basis.n_aux;
        aux_basis.shells.push(AuxiliaryShell {
            atom_index: 0,
            center: [0.0, 0.0, 0.0],
            shell_type: ShellType::S,
            exponents: vec![dup_exp],
            coeffs: vec![dup_norm],
            ao_offset: ao_off,
        });
        aux_basis.n_aux += 1;
    }

    let rij = RiJEngine::with_auxiliary_basis(&basis, aux_basis).unwrap();
    assert!(
        rij.metric_truncated_modes >= 1,
        "Linearly dependent auxiliary shells must produce natural rank truncation (got {})",
        rij.metric_truncated_modes
    );

    let p = Matrix2D::identity(basis.n_ao);
    let err = rij
        .compute_coulomb_gradient(mol.atoms.len(), &basis, &p)
        .expect_err("rank-truncated metric must be rejected in analytical RI-J gradient");
    assert!(format!("{err}").contains("truncated"));
}
