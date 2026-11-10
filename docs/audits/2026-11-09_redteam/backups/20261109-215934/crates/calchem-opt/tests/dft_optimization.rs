use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_model::{Atom, BOHR_TO_ANGSTROM, Molecule};
use calchem_opt::{OptOptions, Reference, optimize_geometry, optimize_geometry_with_options};

#[test]
fn test_water_and_cation_b3lyp_geometry_optimization() {
    // Distorted H2O -> B3LYP/6-31G(d) equilibrium ~ R_OH = 0.969 A, HOH = 103.6 deg
    let mol = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.85, -0.52]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.82, -0.56]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let opt_mol = optimize_geometry(&mol, &bs, "b3lyp").unwrap();

    let o = opt_mol.atoms[0].position_bohr;
    let h1 = opt_mol.atoms[1].position_bohr;
    let h2 = opt_mol.atoms[2].position_bohr;
    let v1 = [h1[0] - o[0], h1[1] - o[1], h1[2] - o[2]];
    let v2 = [h2[0] - o[0], h2[1] - o[1], h2[2] - o[2]];
    let r1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt() * BOHR_TO_ANGSTROM;
    let r2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt() * BOHR_TO_ANGSTROM;
    let cos_ang = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
        / ((r1 / BOHR_TO_ANGSTROM) * (r2 / BOHR_TO_ANGSTROM));
    let ang_deg = cos_ang.clamp(-1.0, 1.0).acos().to_degrees();

    assert!((r1 - r2).abs() < 1e-4, "Symmetric OH bonds expected: {r1} vs {r2}");
    assert!(
        (0.960..=0.980).contains(&r1),
        "Unexpected B3LYP/6-31G(d) H2O R_OH: {r1:.4} A"
    );
    assert!(
        (102.5..=105.0).contains(&ang_deg),
        "Unexpected B3LYP/6-31G(d) H2O angle: {ang_deg:.2} deg"
    );

    // Open-shell H2+ cation optimization via DftAuto(Pbe0)
    let h2_cat = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.5]).unwrap(),
        ],
        1,
        2,
    );
    let opts = OptOptions {
        reference: Reference::DftAuto(XcFunctional::Pbe0),
        grid_level: GridLevel::Coarse,
        ..OptOptions::default()
    };
    let res = optimize_geometry_with_options(&h2_cat, "6-31g", &opts).unwrap();
    assert!(res.converged);
    let r_cat_bohr = (res.molecule.atoms[1].position_bohr[2]
        - res.molecule.atoms[0].position_bohr[2])
        .abs();
    assert!(
        r_cat_bohr > 1.85,
        "Expected stretched H2+ bond length > 1.85 Bohr, got {r_cat_bohr:.4}"
    );
}
