use calchem_basis::BasisSet;
use calchem_model::{Atom, Molecule};
use calchem_opt::{optimize_geometry, optimize_geometry_with_options, OptOptions};

#[test]
fn test_single_atom_optimization() {
    let mol = Molecule::new(vec![Atom::from_bohr("He", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let opt_mol = optimize_geometry(&mol, &basis, "rhf").unwrap();
    assert_eq!(opt_mol.atoms.len(), 1);
    assert_eq!(opt_mol.atoms[0].position_bohr, [0.0, 0.0, 0.0]);
}

#[test]
fn test_h2_optimization_from_1_4_bohr() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let mut opts = OptOptions::default();
    opts.max_force = 1e-4;
    opts.rms_force = 5e-5;
    opts.max_displacement = 1e-3;

    let res = optimize_geometry_with_options(&mol, "sto-3g", &opts).unwrap();
    assert!(res.converged);
    assert!(res.steps > 0);

    let c0 = res.molecule.atoms[0].position_bohr;
    let c1 = res.molecule.atoms[1].position_bohr;
    let bond = ((c1[0] - c0[0]).powi(2) + (c1[1] - c0[1]).powi(2) + (c1[2] - c0[2]).powi(2)).sqrt();

    // RHF/STO-3G H2 equilibrium bond is 1.346138 Bohr
    assert!(
        (bond - 1.346138).abs() < 1e-3,
        "Optimized bond length = {} Bohr vs 1.346138",
        bond
    );

    // Energy reference: -1.1175058714 Eh
    assert!(
        (res.energy_hartree - (-1.1175058714)).abs() < 1e-7,
        "Optimized energy = {} Eh vs -1.1175058714",
        res.energy_hartree
    );

    // Forces must be below tolerance
    for g in &res.gradient {
        for q in 0..3 {
            assert!(g[q].abs() <= opts.max_force * 1.5);
        }
    }
}

#[test]
fn test_h2_optimization_from_compressed_1_2_bohr() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.2]).unwrap(),
        ],
        0,
        1,
    );
    let mut opts = OptOptions::default();
    opts.max_force = 1e-4;
    opts.rms_force = 5e-5;
    opts.max_displacement = 1e-3;

    let res = optimize_geometry_with_options(&mol, "sto-3g", &opts).unwrap();
    assert!(res.converged);

    let c0 = res.molecule.atoms[0].position_bohr;
    let c1 = res.molecule.atoms[1].position_bohr;
    let bond = ((c1[0] - c0[0]).powi(2) + (c1[1] - c0[1]).powi(2) + (c1[2] - c0[2]).powi(2)).sqrt();

    assert!(
        (bond - 1.346138).abs() < 1e-3,
        "Optimized bond length = {} Bohr vs 1.346138",
        bond
    );
    assert!(
        (res.energy_hartree - (-1.1175058714)).abs() < 1e-7,
        "Optimized energy = {} Eh vs -1.1175058714",
        res.energy_hartree
    );
}
