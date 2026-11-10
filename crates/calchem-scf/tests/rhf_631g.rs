use calchem_basis::BasisSet;
use calchem_grad::rhf_gradient;
use calchem_int::compute_overlap;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::rhf;
use calchem_tensor::eigh_sym;

#[test]
fn test_h2o_631g_shells_and_normalization() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::basis_6_31g(&mol).unwrap();

    // H2O with 6-31G: O has 5 shells (s, s, p, s, p => 1+1+3+1+3 = 9 AOs)
    // Each H has 2 s shells (1+1 = 2 AOs)
    // Total = 9 + 2 + 2 = 13 Cartesian AOs
    assert_eq!(bs.n_ao, 13, "6-31G H2O must have exactly 13 AOs");
    assert_eq!(bs.shells.len(), 9, "6-31G H2O must have 9 shells");

    // Overlap matrix checks:
    let s = compute_overlap(&bs);
    assert_eq!(s.rows, 13);
    assert_eq!(s.cols, 13);

    // Diagonal elements S_ii must be normalized to ~1.0
    for i in 0..13 {
        let s_ii = s.get(i, i);
        assert!(
            (s_ii - 1.0).abs() < 1e-12,
            "S_{{{i},{i}}} = {s_ii} != 1.0 (normalization error)"
        );
    }

    // Overlap matrix must be strictly positive definite
    let (evals, _) = eigh_sym(&s, 100, 1e-14).unwrap();
    for &val in &evals {
        assert!(
            val > 1e-4,
            "Overlap eigenvalue {val} is not strictly positive definite"
        );
    }
}

#[test]
fn test_h2o_631g_energy_and_gradient() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::basis_6_31g(&mol).unwrap();

    let mut opts = ScfOptions::default();
    opts.conv_energy = 1e-12;
    opts.conv_density = 1e-11;

    let wf = rhf(&mol, &bs, &opts).unwrap();
    assert!(wf.converged);

    // Golden mathematical RHF/6-31G energy: -75.800277509588 Eh
    let ref_energy = -75.800277509588_f64;
    assert!(
        (wf.total_energy_hartree - ref_energy).abs() < 1e-8,
        "Energy mismatch: got {} vs ref {}",
        wf.total_energy_hartree,
        ref_energy
    );

    // Analytical gradient evaluation
    let grad = rhf_gradient(&mol, &bs, &wf).unwrap();

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
            "6-31G translational invariance violated on axis {q}: sum = {}",
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
            "6-31G rotational invariance violated on axis {q}: torque = {}",
            torque[q]
        );
    }

    // 3. Central difference check on displaced atom 1 along y (axis 1)
    let h = 0.002;
    let mut pos_p = mol.atoms.clone();
    pos_p[1].position_bohr[1] += h;
    let mol_p = Molecule::new(pos_p, 0, 1);
    let bs_p = BasisSet::basis_6_31g(&mol_p).unwrap();
    let wf_p = rhf(&mol_p, &bs_p, &opts).unwrap();

    let mut pos_m = mol.atoms.clone();
    pos_m[1].position_bohr[1] -= h;
    let mol_m = Molecule::new(pos_m, 0, 1);
    let bs_m = BasisSet::basis_6_31g(&mol_m).unwrap();
    let wf_m = rhf(&mol_m, &bs_m, &opts).unwrap();

    let fd_g1_y = (wf_p.total_energy_hartree - wf_m.total_energy_hartree) / (2.0 * h);
    let diff = (grad[1][1] - fd_g1_y).abs();
    assert!(
        diff < 1e-4,
        "6-31G gradient mismatch for H1 axis y: analytic {} vs fd {} (diff: {:e})",
        grad[1][1],
        fd_g1_y,
        diff
    );
}
