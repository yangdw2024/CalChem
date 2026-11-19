use calchem_basis::BasisSet;
use calchem_grad::derivatives::one_electron_derivative;
use calchem_grad::rhf_gradient;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::rhf;

#[test]
fn test_single_atom_gradient_vanishes() {
    let mol = Molecule::new(vec![Atom::from_bohr("He", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let wf = rhf(&mol, &basis, &ScfOptions::default()).unwrap();
    let grad = rhf_gradient(&mol, &basis, &wf).unwrap();
    assert_eq!(grad.len(), 1);
    for q in 0..3 {
        assert_eq!(grad[0][q], 0.0);
    }
}

#[test]
fn test_one_electron_derivatives_finite_difference() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let (ds, dt, dv) = one_electron_derivative(&mol, &basis, 1, 2);

    // Finite difference on atom 1 along z (axis 2)
    let h = 0.0005;
    let mol_plus = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4 + h]).unwrap(),
        ],
        0,
        1,
    );
    let basis_plus = BasisSet::sto_3g(&mol_plus).unwrap();
    let s_plus = calchem_int::compute_overlap(&basis_plus);
    let t_plus = calchem_int::compute_kinetic(&basis_plus);
    let v_plus = calchem_int::compute_nuclear(&basis_plus, &mol_plus);

    let mol_minus = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4 - h]).unwrap(),
        ],
        0,
        1,
    );
    let basis_minus = BasisSet::sto_3g(&mol_minus).unwrap();
    let s_minus = calchem_int::compute_overlap(&basis_minus);
    let t_minus = calchem_int::compute_kinetic(&basis_minus);
    let v_minus = calchem_int::compute_nuclear(&basis_minus, &mol_minus);

    for i in 0..basis.n_ao {
        for j in 0..basis.n_ao {
            let num_ds = (s_plus.get(i, j) - s_minus.get(i, j)) / (2.0 * h);
            let num_dt = (t_plus.get(i, j) - t_minus.get(i, j)) / (2.0 * h);
            let num_dv = (v_plus.get(i, j) - v_minus.get(i, j)) / (2.0 * h);

            assert!(
                (ds.get(i, j) - num_ds).abs() < 1e-6,
                "dS mismatch at ({i}, {j}): analytic {} vs fd {}",
                ds.get(i, j),
                num_ds
            );
            assert!(
                (dt.get(i, j) - num_dt).abs() < 1e-6,
                "dT mismatch at ({i}, {j}): analytic {} vs fd {}",
                dt.get(i, j),
                num_dt
            );
            assert!(
                (dv.get(i, j) - num_dv).abs() < 1e-6,
                "dV mismatch at ({i}, {j}): analytic {} vs fd {}",
                dv.get(i, j),
                num_dv
            );
        }
    }
}

#[test]
fn test_h2_sto3g_rhf_gradient() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let mut opts = ScfOptions::default();
    opts.conv_energy = 1e-12;
    opts.conv_density = 1e-11;
    let wf = rhf(&mol, &basis, &opts).unwrap();
    assert!(wf.converged);

    let g = rhf_gradient(&mol, &basis, &wf).unwrap();

    // Baseline reference: G_z on H1 = -0.02845406, on H2 = +0.02845406
    let expected_gz = 0.02845406;
    assert!(
        (g[0][2] - (-expected_gz)).abs() < 1e-6,
        "g[0][2] = {} vs {}",
        g[0][2],
        -expected_gz
    );
    assert!(
        (g[1][2] - expected_gz).abs() < 1e-6,
        "g[1][2] = {} vs {}",
        g[1][2],
        expected_gz
    );

    // Translational invariance sum_A G_A = 0
    for q in 0..3 {
        assert!((g[0][q] + g[1][q]).abs() < 1e-10);
    }

    // Finite difference check at h=0.002 Bohr
    let h = 0.002;
    let mol_p = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4 + h]).unwrap(),
        ],
        0,
        1,
    );
    let bs_p = BasisSet::sto_3g(&mol_p).unwrap();
    let wf_p = rhf(&mol_p, &bs_p, &opts).unwrap();

    let mol_m = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4 - h]).unwrap(),
        ],
        0,
        1,
    );
    let bs_m = BasisSet::sto_3g(&mol_m).unwrap();
    let wf_m = rhf(&mol_m, &bs_m, &opts).unwrap();

    let num_g_z = (wf_p.total_energy_hartree - wf_m.total_energy_hartree) / (2.0 * h);
    assert!(
        (g[1][2] - num_g_z).abs() < 1e-5,
        "Analytic {} vs FD {}",
        g[1][2],
        num_g_z
    );
}

#[test]
fn test_h2o_sto3g_rhf_gradient() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let mut opts = ScfOptions::default();
    opts.conv_energy = 1e-12;
    opts.conv_density = 1e-11;
    let wf = rhf(&mol, &basis, &opts).unwrap();
    assert!(wf.converged);

    let grad = rhf_gradient(&mol, &basis, &wf).unwrap();

    // 1. Translational invariance: sum_A G_A approx 0
    let mut sum_g = [0.0; 3];
    for a in 0..3 {
        for q in 0..3 {
            sum_g[q] += grad[a][q];
        }
    }
    for q in 0..3 {
        assert!(
            sum_g[q].abs() < 1e-6,
            "Translational invariance violated on axis {q}: sum = {}",
            sum_g[q]
        );
    }

    // 2. Rotational invariance: sum_A (R_A x G_A) approx 0
    let mut torque = [0.0; 3];
    for a in 0..3 {
        let r = mol.atoms[a].position_bohr;
        let g = grad[a];
        torque[0] += r[1] * g[2] - r[2] * g[1];
        torque[1] += r[2] * g[0] - r[0] * g[2];
        torque[2] += r[0] * g[1] - r[1] * g[0];
    }
    for q in 0..3 {
        assert!(
            torque[q].abs() < 1e-5,
            "Rotational invariance violated on axis {q}: torque = {}",
            torque[q]
        );
    }

    // 3. Central difference check for each atom and Cartesian coordinate: h=0.002
    let h = 0.002;
    for a in 0..3 {
        for q in 0..3 {
            let mut pos_p = mol.atoms.clone();
            pos_p[a].position_bohr[q] += h;
            let mol_p = Molecule::new(pos_p, 0, 1);
            let bs_p = BasisSet::sto_3g(&mol_p).unwrap();
            let wf_p = rhf(&mol_p, &bs_p, &opts).unwrap();

            let mut pos_m = mol.atoms.clone();
            pos_m[a].position_bohr[q] -= h;
            let mol_m = Molecule::new(pos_m, 0, 1);
            let bs_m = BasisSet::sto_3g(&mol_m).unwrap();
            let wf_m = rhf(&mol_m, &bs_m, &opts).unwrap();

            let fd_g = (wf_p.total_energy_hartree - wf_m.total_energy_hartree) / (2.0 * h);
            let diff = (grad[a][q] - fd_g).abs();
            assert!(
                diff < 1e-4,
                "Gradient mismatch for atom {a} axis {q}: analytic {} vs fd {} (diff: {:e})",
                grad[a][q],
                fd_g,
                diff
            );
        }
    }
}
