//! UHF analytic gradient validation: finite differences, translational invariance,
//! and exact agreement with RHF for closed shells.

use calchem_basis::BasisSet;
use calchem_grad::{rhf_gradient, uhf_gradient};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rhf, uhf};

fn tight() -> ScfOptions {
    ScfOptions {
        conv_energy: 1e-12,
        conv_density: 1e-10,
        max_iter: 300,
        ..Default::default()
    }
}

fn build(symbols: &[&str], coords: &[[f64; 3]], charge: i32, mult: usize) -> Molecule {
    Molecule::new(
        symbols
            .iter()
            .zip(coords)
            .map(|(s, c)| Atom::from_bohr(s, *c).unwrap())
            .collect(),
        charge,
        mult,
    )
}

fn uhf_energy(symbols: &[&str], coords: &[[f64; 3]], charge: i32, mult: usize, basis: &str) -> f64 {
    let mol = build(symbols, coords, charge, mult);
    let bs = BasisSet::from_name(basis, &mol).unwrap();
    uhf(&mol, &bs, &tight()).unwrap().total_energy_hartree
}

fn check_fd(symbols: &[&str], coords: &[[f64; 3]], charge: i32, mult: usize, basis: &str, tol: f64) {
    let mol = build(symbols, coords, charge, mult);
    let bs = BasisSet::from_name(basis, &mol).unwrap();
    let wf = uhf(&mol, &bs, &tight()).unwrap();
    let g = uhf_gradient(&mol, &bs, &wf).unwrap();

    // Translational invariance: total force vanishes.
    for q in 0..3 {
        let sum: f64 = g.iter().map(|a| a[q]).sum();
        assert!(sum.abs() < 1e-8, "{basis}: sum of gradient axis {q} = {sum:e}");
    }

    let h = 1e-4;
    for a in 0..coords.len() {
        for q in 0..3 {
            let mut plus = coords.to_vec();
            let mut minus = coords.to_vec();
            plus[a][q] += h;
            minus[a][q] -= h;
            let fd = (uhf_energy(symbols, &plus, charge, mult, basis)
                - uhf_energy(symbols, &minus, charge, mult, basis))
                / (2.0 * h);
            assert!(
                (g[a][q] - fd).abs() < tol,
                "{basis} atom {a} axis {q}: analytic {:.10} vs FD {:.10}",
                g[a][q],
                fd
            );
        }
    }
}

/// H2+ (one electron, doublet) away from equilibrium.
#[test]
fn test_h2_cation_gradient_fd() {
    check_fd(&["H", "H"], &[[0.0, 0.0, 0.0], [0.1, -0.05, 1.6]], 1, 2, "sto-3g", 1e-6);
}

/// H2O+ radical cation, distorted geometry, both basis sets.
#[test]
fn test_h2o_cation_gradient_fd() {
    let coords = [[0.0, 0.05, -0.02], [0.0, 1.50, 1.05], [0.10, -1.40, 1.12]];
    check_fd(&["O", "H", "H"], &coords, 1, 2, "sto-3g", 1e-6);
    check_fd(&["O", "H", "H"], &coords, 1, 2, "6-31g", 1e-6);
}

/// H2O- radical anion (doublet with an electron in a virtual-like orbital).
#[test]
fn test_h2o_anion_gradient_fd() {
    let coords = [[0.0, 0.0, 0.0], [0.0, 1.45, 1.10], [0.0, -1.45, 1.10]];
    check_fd(&["O", "H", "H"], &coords, -1, 2, "sto-3g", 1e-6);
}

/// Closed-shell UHF gradient must equal the RHF gradient.
#[test]
fn test_closed_shell_uhf_gradient_equals_rhf() {
    let coords = [[0.0, 0.0, 0.0], [0.0, 1.432, 0.0], [1.112, -0.716, 0.0]];
    let mol = build(&["O", "H", "H"], &coords, 0, 1);
    for basis in ["sto-3g", "6-31g"] {
        let bs = BasisSet::from_name(basis, &mol).unwrap();
        let g_r = rhf_gradient(&mol, &bs, &rhf(&mol, &bs, &tight()).unwrap()).unwrap();
        let g_u = uhf_gradient(&mol, &bs, &uhf(&mol, &bs, &tight()).unwrap()).unwrap();
        for a in 0..3 {
            for q in 0..3 {
                assert!(
                    (g_r[a][q] - g_u[a][q]).abs() < 1e-8,
                    "{basis} atom {a} axis {q}: RHF {} vs UHF {}",
                    g_r[a][q],
                    g_u[a][q]
                );
            }
        }
    }
}
