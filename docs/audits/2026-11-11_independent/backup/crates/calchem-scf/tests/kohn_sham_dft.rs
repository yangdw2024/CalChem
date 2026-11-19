use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rhf, rks, uks};

const HARTREE_TO_EV: f64 = 27.211_386_245_988;

fn water_standard() -> Molecule {
    let r = 0.9572_f64;
    let half_theta = (104.52_f64 / 2.0).to_radians();
    let hy = r * half_theta.sin();
    let hz = -r * half_theta.cos();
    Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, hy, hz]).unwrap(),
            Atom::from_angstrom("H", [0.0, -hy, hz]).unwrap(),
        ],
        0,
        1,
    )
}

fn ethylene(charge: i32, multiplicity: usize) -> Molecule {
    Molecule::new(
        vec![
            Atom::from_angstrom("C", [0.0, 0.0, 0.6695]).unwrap(),
            Atom::from_angstrom("C", [0.0, 0.0, -0.6695]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.9289, 1.2321]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.9289, 1.2321]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.9289, -1.2321]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.9289, -1.2321]).unwrap(),
        ],
        charge,
        multiplicity,
    )
}

#[test]
fn test_closed_shell_uks_matches_rks_across_all_functionals_and_water_b3lyp_benchmark() {
    let mol = water_standard();
    let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let opts = ScfOptions::default();

    for xc in [
        XcFunctional::Svwn5,
        XcFunctional::Blyp,
        XcFunctional::Pbe,
        XcFunctional::B3lyp,
        XcFunctional::B3lyp5,
        XcFunctional::Pbe0,
    ] {
        let wf_r = rks(&mol, &basis, xc, GridLevel::Medium, &opts).unwrap();
        let wf_u = uks(&mol, &basis, xc, GridLevel::Medium, &opts).unwrap();
        assert!(wf_r.converged && wf_u.converged);
        assert!(
            (wf_r.total_energy_hartree - wf_u.total_energy_hartree).abs() < 1e-10,
            "{xc:?}: RKS={} vs UKS={}",
            wf_r.total_energy_hartree,
            wf_u.total_energy_hartree
        );
        assert!(wf_u.s_squared.abs() < 1e-10);
    }

    // Canonical B3LYP / 6-31G(d) (Cartesian 6D) NIST CCCBDB reference for H2O is ~ -76.40895 Eh.
    let wf_med = rks(&mol, &basis, XcFunctional::B3lyp, GridLevel::Medium, &opts).unwrap();
    let wf_fine = rks(&mol, &basis, XcFunctional::B3lyp, GridLevel::Fine, &opts).unwrap();
    assert!(
        (wf_med.total_energy_hartree - wf_fine.total_energy_hartree).abs() < 5e-5,
        "Grid convergence Medium={} vs Fine={}",
        wf_med.total_energy_hartree,
        wf_fine.total_energy_hartree
    );
    assert!(
        (wf_fine.total_energy_hartree - (-76.40895)).abs() < 1.5e-3,
        "H2O B3LYP/6-31G(d) energy {} drifted from NIST reference -76.40895 Eh",
        wf_fine.total_energy_hartree
    );
}

#[test]
fn test_ethylene_dft_frontier_gap_and_vertical_ionization_potential() {
    let mol0 = ethylene(0, 1);
    let mol_cat = ethylene(1, 2);
    let bs0 = BasisSet::from_name("6-31g(d)", &mol0).unwrap();
    let bs_cat = BasisSet::from_name("6-31g(d)", &mol_cat).unwrap();
    let opts = ScfOptions::default();

    let wf_rhf = rhf(&mol0, &bs0, &opts).unwrap();
    let wf_b3lyp = rks(&mol0, &bs0, XcFunctional::B3lyp, GridLevel::Medium, &opts).unwrap();
    let wf_pbe = rks(&mol0, &bs0, XcFunctional::Pbe, GridLevel::Medium, &opts).unwrap();

    // C2H4 has 16 electrons -> 8 doubly occupied MOs (HOMO = idx 7, LUMO = idx 8)
    let gap_rhf_ev = (wf_rhf.mo_energies[8] - wf_rhf.mo_energies[7]) * HARTREE_TO_EV;
    let gap_b3lyp_ev = (wf_b3lyp.mo_energies[8] - wf_b3lyp.mo_energies[7]) * HARTREE_TO_EV;
    let gap_pbe_ev = (wf_pbe.mo_energies[8] - wf_pbe.mo_energies[7]) * HARTREE_TO_EV;

    assert!(gap_rhf_ev > 14.0);
    assert!(
        (6.0..8.2).contains(&gap_b3lyp_ev),
        "Unexpected C2H4 B3LYP HOMO-LUMO gap: {gap_b3lyp_ev:.3} eV"
    );
    assert!(
        (4.8..6.8).contains(&gap_pbe_ev) && gap_pbe_ev < gap_b3lyp_ev,
        "Unexpected C2H4 PBE HOMO-LUMO gap: {gap_pbe_ev:.3} eV"
    );

    // Open-shell UKS B3LYP on C2H4+ radical cation -> vertical IP near experimental ~10.5 eV
    let wf_cat = uks(
        &mol_cat,
        &bs_cat,
        XcFunctional::B3lyp,
        GridLevel::Medium,
        &opts,
    )
    .unwrap();
    assert!((wf_cat.s_squared - 0.75).abs() < 0.03);
    let vip_ev = (wf_cat.total_energy_hartree - wf_b3lyp.total_energy_hartree) * HARTREE_TO_EV;
    assert!(
        (9.8..11.2).contains(&vip_ev),
        "Unexpected C2H4 vertical IP at B3LYP/6-31G(d): {vip_ev:.3} eV"
    );
}

#[test]
fn test_halogen_ecp_hbr_lanl2dz_b3lyp_rks_and_uks() {
    let hbr = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("Br", [0.0, 0.0, 1.414]).unwrap(),
        ],
        0,
        1,
    );
    let hbr_cat = Molecule::new(hbr.atoms.clone(), 1, 2);
    let bs0 = BasisSet::from_name("lanl2dz", &hbr).unwrap();
    let bs_cat = BasisSet::from_name("lanl2dz", &hbr_cat).unwrap();
    let opts = ScfOptions::default();

    let wf0 = rks(&hbr, &bs0, XcFunctional::B3lyp, GridLevel::Medium, &opts).unwrap();
    let wf_cat = uks(
        &hbr_cat,
        &bs_cat,
        XcFunctional::B3lyp,
        GridLevel::Medium,
        &opts,
    )
    .unwrap();

    assert!(wf0.converged && wf_cat.converged);
    // H (1e) + Br LANL2DZ valence (7e) = 8 valence electrons -> energy around -13.6 Eh
    assert!(
        (-14.5..-13.0).contains(&wf0.total_energy_hartree),
        "Unexpected HBr B3LYP/LANL2DZ energy: {}",
        wf0.total_energy_hartree
    );
    let vip_ev = (wf_cat.total_energy_hartree - wf0.total_energy_hartree) * HARTREE_TO_EV;
    assert!(
        (10.5..13.0).contains(&vip_ev),
        "Unexpected HBr B3LYP/LANL2DZ vertical IP: {vip_ev:.3} eV"
    );
}
