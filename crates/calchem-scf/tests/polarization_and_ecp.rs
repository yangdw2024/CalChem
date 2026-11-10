use calchem_basis::BasisSet;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rhf, uhf};

fn water_angstrom() -> Molecule {
    // Standard experimental geometry: R(OH) = 0.9572 A, angle(HOH) = 104.52 deg
    let r = 0.9572_f64;
    let half_theta = (104.52_f64 * std::f64::consts::PI / 180.0) * 0.5;
    let y = r * half_theta.sin();
    let z = -r * half_theta.cos();
    Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, y, z]).unwrap(),
            Atom::from_angstrom("H", [0.0, -y, z]).unwrap(),
        ],
        0,
        1,
    )
}

#[test]
fn test_water_631gd_and_631gdp_variational_hierarchy_and_literature_energy() {
    let mol = water_angstrom();
    let opts = ScfOptions::default();

    let bs_631g = BasisSet::from_name("6-31g", &mol).unwrap();
    let bs_631gd = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let bs_star = BasisSet::from_name("6-31g*", &mol).unwrap();
    let bs_631gdp = BasisSet::from_name("6-31g(d,p)", &mol).unwrap();

    assert_eq!(bs_631g.n_ao, 13);
    assert_eq!(bs_631gd.n_ao, 19); // +6 Cartesian d on O
    assert_eq!(bs_star.n_ao, 19);
    assert_eq!(bs_631gdp.n_ao, 25); // +3 p on each H

    let wf_631g = rhf(&mol, &bs_631g, &opts).unwrap();
    let wf_631gd = rhf(&mol, &bs_631gd, &opts).unwrap();
    let wf_star = rhf(&mol, &bs_star, &opts).unwrap();
    let wf_631gdp = rhf(&mol, &bs_631gdp, &opts).unwrap();

    assert!(wf_631gd.converged && wf_631gdp.converged);
    assert!((wf_631gd.total_energy_hartree - wf_star.total_energy_hartree).abs() < 1e-13);

    // Variational principle: adding polarization functions strictly lowers RHF energy
    assert!(
        wf_631gd.total_energy_hartree < wf_631g.total_energy_hartree - 0.02,
        "Expected 6-31G(d) ({}) < 6-31G ({})",
        wf_631gd.total_energy_hartree,
        wf_631g.total_energy_hartree
    );
    assert!(
        wf_631gdp.total_energy_hartree < wf_631gd.total_energy_hartree - 0.01,
        "Expected 6-31G(d,p) ({}) < 6-31G(d) ({})",
        wf_631gdp.total_energy_hartree,
        wf_631gd.total_energy_hartree
    );

    // Standard Pople RHF/6-31G(d) (6D Cartesian) water energy at experimental geometry ~ -76.01075 Eh
    assert!(
        (wf_631gd.total_energy_hartree - (-76.01075)).abs() < 5e-4,
        "Unexpected RHF/6-31G(d) water energy: {}",
        wf_631gd.total_energy_hartree
    );
}

#[test]
fn test_h2s_631gd_and_all_electron_halogens_sto3g() {
    let opts = ScfOptions::default();

    // H2S in 6-31G(d) (3rd-row sulfur polarization d=0.65)
    let h2s = Molecule::new(
        vec![
            Atom::from_angstrom("S", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.96, -0.94]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.96, -0.94]).unwrap(),
        ],
        0,
        1,
    );
    let bs_h2s_631g = BasisSet::from_name("6-31g", &h2s).unwrap();
    let bs_h2s_631gd = BasisSet::from_name("6-31g(d)", &h2s).unwrap();
    assert_eq!(bs_h2s_631gd.n_ao, bs_h2s_631g.n_ao + 6);
    let wf_h2s_base = rhf(&h2s, &bs_h2s_631g, &opts).unwrap();
    let wf_h2s_pol = rhf(&h2s, &bs_h2s_631gd, &opts).unwrap();
    assert!(wf_h2s_pol.total_energy_hartree < wf_h2s_base.total_energy_hartree - 0.02);

    // All-electron STO-3G on HBr (Z=35) and HI (Z=53)
    let hbr = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("Br", [0.0, 0.0, 1.414]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hbr = BasisSet::from_name("sto-3g", &hbr).unwrap();
    assert!(!bs_hbr.has_ecp());
    let wf_hbr = rhf(&hbr, &bs_hbr, &opts).unwrap();
    assert!(wf_hbr.converged);
    assert!(wf_hbr.total_energy_hartree < -2500.0);

    let hi = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("I", [0.0, 0.0, 1.609]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hi = BasisSet::from_name("sto-3g", &hi).unwrap();
    assert!(!bs_hi.has_ecp());
    let wf_hi = rhf(&hi, &bs_hi, &opts).unwrap();
    assert!(wf_hi.converged);
    assert!(wf_hi.total_energy_hartree < -6800.0);
}

#[test]
fn test_lanl2dz_and_lanl2dzdp_ecp_invariance_and_open_shell_halogens() {
    let opts = ScfOptions::default();

    // HBr in LANL2DZ: H (0 core) + Br (28 core -> 7 valence) => 8 valence electrons (4 pairs)
    let hbr_z = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("Br", [0.0, 0.0, 1.44]).unwrap(),
        ],
        0,
        1,
    );
    let bs_z = BasisSet::from_name("lanl2dz", &hbr_z).unwrap();
    assert!(bs_z.has_ecp());
    assert_eq!(bs_z.core_electrons, vec![0, 28]);
    assert_eq!(bs_z.validate_rhf_valence(&hbr_z).unwrap(), (4, 4));

    let wf_z = rhf(&hbr_z, &bs_z, &opts).unwrap();
    assert!(wf_z.converged);

    // Physical bound-state check: E(HBr, RHF) < E(H, UHF) + E(Br ^2P, UHF) - 0.05 Eh
    let h_atom = Molecule::new(vec![Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap()], 0, 2);
    let br_atom = Molecule::new(vec![Atom::from_angstrom("Br", [0.0, 0.0, 0.0]).unwrap()], 0, 2);
    let e_h = uhf(&h_atom, &BasisSet::from_name("lanl2dz", &h_atom).unwrap(), &opts)
        .unwrap()
        .total_energy_hartree;
    let e_br = uhf(&br_atom, &BasisSet::from_name("lanl2dz", &br_atom).unwrap(), &opts)
        .unwrap()
        .total_energy_hartree;
    let d_e_hbr = (e_h + e_br) - wf_z.total_energy_hartree;
    assert!(
        d_e_hbr > 0.05 && d_e_hbr < 0.25,
        "Unexpected HBr LANL2DZ dissociation energy De = {d_e_hbr:.6} Eh (E_mol={}, E_H={e_h}, E_Br={e_br})",
        wf_z.total_energy_hartree
    );

    // Rotated + translated HBr must give identical energy!
    let r = 1.44_f64;
    let dir = [0.36_f64, -0.48_f64, 0.80_f64];
    let shift = [1.7_f64, -2.3_f64, 0.9_f64];
    let hbr_rot = Molecule::new(
        vec![
            Atom::from_angstrom("H", shift).unwrap(),
            Atom::from_angstrom(
                "Br",
                [
                    shift[0] + r * dir[0],
                    shift[1] + r * dir[1],
                    shift[2] + r * dir[2],
                ],
            )
            .unwrap(),
        ],
        0,
        1,
    );
    let bs_rot = BasisSet::from_name("lanl2dz", &hbr_rot).unwrap();
    let wf_rot = rhf(&hbr_rot, &bs_rot, &opts).unwrap();

    assert!(
        (wf_z.total_energy_hartree - wf_rot.total_energy_hartree).abs() < 1e-11,
        "ECP rotational/translational energy variance: {} vs {}",
        wf_z.total_energy_hartree,
        wf_rot.total_energy_hartree
    );

    // HI in LANL2DZ and LANL2DZdp (Iodine Z=53, 46 core -> 7 valence)
    let hi = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("I", [0.0, 0.0, 1.63]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hi_dz = BasisSet::from_name("lanl2dz", &hi).unwrap();
    let bs_hi_dp = BasisSet::from_name("lanl2dzdp", &hi).unwrap();
    assert_eq!(bs_hi_dz.core_electrons, vec![0, 46]);
    assert_eq!(bs_hi_dp.core_electrons, vec![0, 46]);
    let wf_hi_dz = rhf(&hi, &bs_hi_dz, &opts).unwrap();
    let wf_hi_dp = rhf(&hi, &bs_hi_dp, &opts).unwrap();
    assert!(wf_hi_dz.converged && wf_hi_dp.converged);
    // LANL2DZdp adds d,p polarization + diffuse -> strictly lower variational energy
    assert!(wf_hi_dp.total_energy_hartree < wf_hi_dz.total_energy_hartree - 0.01);

    // Open-shell UHF on HI+ radical cation (charge=+1, mult=2 -> 7 valence e-: 4 alpha, 3 beta)
    let hi_cat = Molecule::new(hi.atoms.clone(), 1, 2);
    let wf_hi_cat = uhf(&hi_cat, &bs_hi_dz, &opts).unwrap();
    assert!(wf_hi_cat.converged);
    assert_eq!((wf_hi_cat.n_alpha, wf_hi_cat.n_beta), (4, 3));
    assert!(wf_hi_cat.total_energy_hartree > wf_hi_dz.total_energy_hartree); // positive IP!
    assert!((wf_hi_cat.s_squared - 0.75).abs() < 0.05);
}
