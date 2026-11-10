//! UHF geometry optimization for open-shell species (radical cations).

use calchem_basis::BasisSet;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_opt::{optimize_geometry_with_options, OptOptions, Reference};
use calchem_scf::uhf;

fn h2_cation(r: f64) -> Molecule {
    Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, r]).unwrap(),
        ],
        1,
        2,
    )
}

fn e_uhf(mol: &Molecule) -> f64 {
    let bs = BasisSet::from_name("sto-3g", mol).unwrap();
    let o = ScfOptions { conv_energy: 1e-12, conv_density: 1e-10, ..Default::default() };
    uhf(mol, &bs, &o).unwrap().total_energy_hartree
}

#[test]
fn test_h2_cation_uhf_optimization_reaches_minimum() {
    let start = h2_cation(1.4);
    let e_start = e_uhf(&start);
    let opts = OptOptions { max_force: 3e-5, rms_force: 2e-5, ..OptOptions::default() };
    assert_eq!(opts.reference, Reference::Auto);
    let res = optimize_geometry_with_options(&start, "sto-3g", &opts).unwrap();
    assert!(res.converged);
    assert!(res.energy_hartree < e_start, "optimization must lower the energy");

    let a = res.molecule.atoms[0].position_bohr;
    let b = res.molecule.atoms[1].position_bohr;
    let r_eq = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    // Removing a bonding electron lengthens the bond relative to neutral H2 (1.346 Bohr).
    assert!(r_eq > 1.6, "H2+ bond length {r_eq} Bohr should exceed neutral H2");

    // True minimum along the bond coordinate: both displacements raise the energy.
    let e_eq = e_uhf(&h2_cation(r_eq));
    assert!((e_eq - res.energy_hartree).abs() < 1e-9);
    assert!(e_uhf(&h2_cation(r_eq + 0.01)) > e_eq);
    assert!(e_uhf(&h2_cation(r_eq - 0.01)) > e_eq);
    for g in &res.gradient {
        for v in g {
            assert!(v.abs() < 3e-5, "residual gradient {v}");
        }
    }
}

#[test]
fn test_h2o_cation_uhf_optimization_converges() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.43, 1.11]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.43, 1.11]).unwrap(),
        ],
        1,
        2,
    );
    let e0 = e_uhf(&mol);
    let res = optimize_geometry_with_options(&mol, "sto-3g", &OptOptions::default()).unwrap();
    assert!(res.converged);
    assert!(res.energy_hartree < e0);
    let max_g = res.gradient.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(max_g < 1e-4, "max residual gradient {max_g}");
}
