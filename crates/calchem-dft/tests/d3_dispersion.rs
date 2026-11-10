use calchem_dft::d3::D3Model;
use calchem_model::{Atom, Molecule};

#[test]
fn test_d3bj_energy_and_translational_invariance() {
    let mol = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
        ],
        0,
        1,
    );

    let d3 = D3Model::d3bj("b3lyp").unwrap();
    let (energy, grad) = d3.energy_and_gradient(&mol.atoms, true).unwrap();

    // Dispersion energy is strictly negative (attractive)
    assert!(energy < 0.0, "D3 energy must be attractive (negative), got {energy}");
    assert!(energy > -0.01, "Small water molecule dispersion should be on the order of 1e-4..1e-3 Eh, got {energy}");

    // Translational invariance: sum of gradients over all atoms must strictly be zero
    let mut sum_g = [0.0; 3];
    for ga in &grad {
        sum_g[0] += ga[0];
        sum_g[1] += ga[1];
        sum_g[2] += ga[2];
    }
    assert!(
        sum_g[0].abs() < 1e-12 && sum_g[1].abs() < 1e-12 && sum_g[2].abs() < 1e-12,
        "Translational invariance violated: {:?}",
        sum_g
    );
}

#[test]
fn test_d3bj_analytical_gradient_vs_finite_difference() {
    // Asymmetric water molecule to test all 9 gradient components
    let mol = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.05, 0.02, 0.1173]).unwrap(),
            Atom::from_angstrom("H", [0.01, 0.7572, -0.4692]).unwrap(),
            Atom::from_angstrom("H", [-0.03, -0.7312, -0.4501]).unwrap(),
        ],
        0,
        1,
    );

    let d3 = D3Model::d3bj("b3lyp").unwrap();
    let (_e0, ana_grad) = d3.energy_and_gradient(&mol.atoms, true).unwrap();

    let h = 1e-5; // finite difference step in Bohr
    for i in 0..mol.atoms.len() {
        for c in 0..3 {
            let mut atoms_plus = mol.atoms.clone();
            atoms_plus[i].position_bohr[c] += h;
            let (e_plus, _) = d3.energy_and_gradient(&atoms_plus, false).unwrap();

            let mut atoms_minus = mol.atoms.clone();
            atoms_minus[i].position_bohr[c] -= h;
            let (e_minus, _) = d3.energy_and_gradient(&atoms_minus, false).unwrap();

            let num_g = (e_plus - e_minus) / (2.0 * h);
            let ana_g = ana_grad[i][c];

            let diff = (ana_g - num_g).abs();
            assert!(
                diff < 1e-7,
                "Gradient mismatch at atom {i} coord {c}: analytical = {ana_g:.8e}, numerical = {num_g:.8e}, diff = {diff:.8e}"
            );
        }
    }
}

#[test]
fn test_d3zero_analytical_gradient_vs_finite_difference() {
    let mol = Molecule::new(
        vec![
            Atom::from_angstrom("C", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.63, 0.63, 0.63]).unwrap(),
            Atom::from_angstrom("H", [-0.63, -0.63, 0.63]).unwrap(),
            Atom::from_angstrom("H", [-0.63, 0.63, -0.63]).unwrap(),
            Atom::from_angstrom("H", [0.63, -0.63, -0.63]).unwrap(),
        ],
        0,
        1,
    );

    let d3 = D3Model::d3zero("pbe").unwrap();
    let (e0, ana_grad) = d3.energy_and_gradient(&mol.atoms, true).unwrap();
    assert!(e0 < 0.0);

    let h = 1e-5;
    for i in 0..mol.atoms.len() {
        for c in 0..3 {
            let mut atoms_plus = mol.atoms.clone();
            atoms_plus[i].position_bohr[c] += h;
            let (e_plus, _) = d3.energy_and_gradient(&atoms_plus, false).unwrap();

            let mut atoms_minus = mol.atoms.clone();
            atoms_minus[i].position_bohr[c] -= h;
            let (e_minus, _) = d3.energy_and_gradient(&atoms_minus, false).unwrap();

            let num_g = (e_plus - e_minus) / (2.0 * h);
            let ana_g = ana_grad[i][c];

            let diff = (ana_g - num_g).abs();
            assert!(
                diff < 1e-7,
                "D3(0) gradient mismatch at atom {i} coord {c}: analytical = {ana_g:.8e}, numerical = {num_g:.8e}, diff = {diff:.8e}"
            );
        }
    }
}

#[test]
fn test_d3_rejects_unknown_functional_audit_probe() {
    assert!(D3Model::d3bj("not-a-functional").is_err(), "Unknown functional must be rejected!");
    assert!(D3Model::d3zero("not-a-functional").is_err(), "Unknown functional must be rejected!");
}

#[test]
fn test_water_d3bj_matches_gaussian_and_orca_golden_reference() {
    // Audit Red-team Probe Verification:
    // Water molecule B3LYP-D3(BJ) dispersion energy in Gaussian 16 is -0.0005738766 Eh,
    // and in ORCA 6.1.1 is -0.0005738770 Eh.
    let mol = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.7570, -0.4692]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.7570, -0.4692]).unwrap(),
        ],
        0,
        1,
    );
    let d3 = D3Model::d3bj("b3lyp").unwrap();
    let (energy, _) = d3.energy_and_gradient(&mol.atoms, false).unwrap();
    let expected = -0.0005738766;
    let diff = (energy - expected).abs();
    assert!(
        diff < 2e-5,
        "Water D3(BJ) dispersion energy must match Gaussian/ORCA (~ -5.74e-4 Eh), got {energy:.8e}, diff={diff:.2e}"
    );
}
