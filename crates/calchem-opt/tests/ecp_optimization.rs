use calchem_basis::BasisSet;
use calchem_model::{Atom, Molecule, BOHR_TO_ANGSTROM};
use calchem_opt::optimize_geometry;

#[test]
fn test_ecp_geometry_optimization_hbr_and_hi_equilibrium_bond_lengths() {
    let hbr0 = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("Br", [0.0, 0.0, 1.30]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hbr0 = BasisSet::from_name("lanl2dz", &hbr0).unwrap();
    let hbr_opt = optimize_geometry(&hbr0, &bs_hbr0, "rhf").unwrap();
    let r_hbr_ang = ((hbr_opt.atoms[1].position_bohr[2] - hbr_opt.atoms[0].position_bohr[2]).abs())
        * BOHR_TO_ANGSTROM;
    // Wadt & Hay (1985) Table VIII RHF/LANL2DZ Re(HBr) ~ 1.44 A
    assert!(
        (1.40..=1.48).contains(&r_hbr_ang),
        "Unexpected optimized HBr/LANL2DZ bond length: {r_hbr_ang:.4} A"
    );

    let hi0 = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("I", [0.0, 0.0, 1.50]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hi0 = BasisSet::from_name("lanl2dz", &hi0).unwrap();
    let hi_opt = optimize_geometry(&hi0, &bs_hi0, "rhf").unwrap();
    let r_hi_ang = ((hi_opt.atoms[1].position_bohr[2] - hi_opt.atoms[0].position_bohr[2]).abs())
        * BOHR_TO_ANGSTROM;
    // Wadt & Hay (1985) Table VIII RHF/LANL2DZ Re(HI) ~ 1.64 A (and > Re(HBr))
    assert!(
        (1.60..=1.68).contains(&r_hi_ang) && r_hi_ang > r_hbr_ang + 0.15,
        "Unexpected optimized HI/LANL2DZ bond length: {r_hi_ang:.4} A (HBr={r_hbr_ang:.4} A)"
    );
}
