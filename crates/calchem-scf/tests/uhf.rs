//! UHF validation against exact identities and literature values.

use calchem_basis::BasisSet;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rhf, uhf};

fn tight() -> ScfOptions {
    ScfOptions {
        conv_energy: 1e-11,
        conv_density: 1e-10,
        max_iter: 200,
        ..Default::default()
    }
}

fn h2o(charge: i32, mult: usize) -> Molecule {
    Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        charge,
        mult,
    )
}

/// One-electron system: no electron-electron interaction, so UHF must reproduce
/// the lowest eigenvalue of the core Hamiltonian. STO-3G H atom literature value:
/// E = -0.466581850 Eh (Szabo & Ostlund).
#[test]
fn test_h_atom_sto3g_exact_one_electron() {
    let mol = Molecule::new(vec![Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap()], 0, 2);
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
    let wf = uhf(&mol, &bs, &tight()).unwrap();
    assert!(wf.converged);
    assert_eq!((wf.n_alpha, wf.n_beta), (1, 0));
    assert!(
        (wf.total_energy_hartree - (-0.466581850)).abs() < 1e-8,
        "H atom STO-3G UHF energy {} != -0.466581850",
        wf.total_energy_hartree
    );
    // A single unpaired electron is a pure doublet: <S^2> = 0.75 exactly.
    assert!((wf.s_squared - 0.75).abs() < 1e-12, "<S^2> = {}", wf.s_squared);
}

/// For a closed-shell singlet the UHF solution (from an identical alpha/beta guess)
/// must coincide with RHF: same energy and <S^2> = 0.
#[test]
fn test_closed_shell_uhf_equals_rhf() {
    for basis in ["sto-3g", "6-31g"] {
        let mol = h2o(0, 1);
        let bs = BasisSet::from_name(basis, &mol).unwrap();
        let e_rhf = rhf(&mol, &bs, &tight()).unwrap().total_energy_hartree;
        let wf = uhf(&mol, &bs, &tight()).unwrap();
        assert!(
            (wf.total_energy_hartree - e_rhf).abs() < 1e-9,
            "{basis}: UHF {} vs RHF {}",
            wf.total_energy_hartree,
            e_rhf
        );
        assert!(wf.s_squared.abs() < 1e-9, "{basis}: <S^2> = {}", wf.s_squared);
        let pa = &wf.density_alpha;
        let pb = &wf.density_beta;
        let max_spin_density = pa.iter().zip(pb).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(max_spin_density < 1e-7, "{basis}: residual spin density {max_spin_density}");
    }
}

/// Radical cation H2O+ (doublet): converged, physically sensible.
/// - Ionization lowers stability: E(H2O+) > E(H2O).
/// - Koopmans-consistent ordering: vertical IP is positive and below -eps_HOMO(neutral)
///   (orbital relaxation always lowers the cation energy relative to the frozen-orbital estimate).
/// - Spin contamination of a UHF doublet is small but non-negative: 0.75 <= <S^2> < 0.80.
#[test]
fn test_h2o_cation_doublet() {
    let neutral = h2o(0, 1);
    let cation = h2o(1, 2);
    let bs0 = BasisSet::from_name("6-31g", &neutral).unwrap();
    let bs1 = BasisSet::from_name("6-31g", &cation).unwrap();
    let wf0 = rhf(&neutral, &bs0, &tight()).unwrap();
    let wf1 = uhf(&cation, &bs1, &tight()).unwrap();
    assert!(wf1.converged);
    assert_eq!((wf1.n_alpha, wf1.n_beta), (5, 4));

    let ip_vertical = wf1.total_energy_hartree - wf0.total_energy_hartree;
    let homo0 = wf0.mo_energies[4];
    assert!(ip_vertical > 0.0, "IP must be positive, got {ip_vertical}");
    assert!(
        ip_vertical < -homo0,
        "Delta-SCF IP {ip_vertical} must be below Koopmans IP {}",
        -homo0
    );
    assert!(
        wf1.s_squared >= 0.75 - 1e-10 && wf1.s_squared < 0.80,
        "<S^2> = {}",
        wf1.s_squared
    );
    // Densities integrate to electron counts: tr(P^s S) = n_s.
    let n = wf1.n_ao;
    let s = &wf1.overlap_matrix;
    let (mut na, mut nb) = (0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            na += wf1.density_alpha[i * n + j] * s[j * n + i];
            nb += wf1.density_beta[i * n + j] * s[j * n + i];
        }
    }
    assert!((na - 5.0).abs() < 1e-10 && (nb - 4.0).abs() < 1e-10, "tr(PS) = {na}, {nb}");
}

/// Triplet O2-like test on a small system: H2 triplet (both electrons alpha) has no exchange
/// partner in beta, <S^2> must equal S(S+1) = 2 exactly in a minimal basis.
#[test]
fn test_h2_triplet_pure_spin() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        3,
    );
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
    let wf = uhf(&mol, &bs, &tight()).unwrap();
    assert_eq!((wf.n_alpha, wf.n_beta), (2, 0));
    assert!((wf.s_squared - 2.0).abs() < 1e-12, "<S^2> = {}", wf.s_squared);
    // With both orbitals of a 2-function basis occupied, the energy is basis-determined;
    // it must lie above the singlet ground state.
    let singlet = Molecule::new(mol.atoms.clone(), 0, 1);
    let e_s = rhf(&singlet, &bs, &tight()).unwrap().total_energy_hartree;
    assert!(wf.total_energy_hartree > e_s);
}
