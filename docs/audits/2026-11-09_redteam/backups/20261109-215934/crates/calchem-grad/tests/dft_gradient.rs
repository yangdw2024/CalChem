use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_grad::{rks_gradient, uks_gradient};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rks, uks};

fn tight_scf_opts() -> ScfOptions {
    ScfOptions {
        conv_energy: 1e-12,
        conv_density: 1e-10,
        ..Default::default()
    }
}

#[test]
fn test_rks_and_uks_analytical_gradients_match_central_finite_differences() {
    let mol = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.03, -0.02, 0.08]).unwrap(),
            Atom::from_angstrom("H", [0.84, 0.56, -0.35]).unwrap(),
            Atom::from_angstrom("H", [-0.71, 0.65, -0.41]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let opts = tight_scf_opts();
    let h = 1e-4;

    for xc in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp] {
        let wf = rks(&mol, &bs, xc, GridLevel::Coarse, &opts).unwrap();
        let ana = rks_gradient(&mol, &bs, xc, GridLevel::Coarse, &wf).unwrap();

        for q in 0..3 {
            let sum_q: f64 = ana.iter().map(|g| g[q]).sum();
            assert!(
                sum_q.abs() < 1e-8,
                "{xc:?} translational invariance violated: sum={sum_q:.3e}"
            );
        }

        for a in 0..mol.atoms.len() {
            for q in 0..3 {
                let mut mp = mol.clone();
                let mut mm = mol.clone();
                mp.atoms[a].position_bohr[q] += h;
                mm.atoms[a].position_bohr[q] -= h;
                let bsp = BasisSet::from_name("6-31g(d)", &mp).unwrap();
                let bsm = BasisSet::from_name("6-31g(d)", &mm).unwrap();
                let ep = rks(&mp, &bsp, xc, GridLevel::Coarse, &opts)
                    .unwrap()
                    .total_energy_hartree;
                let em = rks(&mm, &bsm, xc, GridLevel::Coarse, &opts)
                    .unwrap()
                    .total_energy_hartree;
                let fd = (ep - em) / (2.0 * h);
                let diff = (ana[a][q] - fd).abs();
                assert!(
                    diff < 5e-6,
                    "{xc:?} RKS atom={a} q={q}: ana={:.9e} fd={:.9e} diff={diff:.3e}",
                    ana[a][q],
                    fd
                );
            }
        }
    }

    // Open-shell UKS gradient on H2O+ radical cation (`charge=1, mult=2`)
    let mol_cat = Molecule::new(mol.atoms.clone(), 1, 2);
    let bs_cat = BasisSet::from_name("6-31g(d)", &mol_cat).unwrap();
    let wf_cat = uks(&mol_cat, &bs_cat, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap();
    let ana_cat = uks_gradient(&mol_cat, &bs_cat, XcFunctional::B3lyp, GridLevel::Coarse, &wf_cat).unwrap();

    for a in 0..mol_cat.atoms.len() {
        for q in 0..3 {
            let mut mp = mol_cat.clone();
            let mut mm = mol_cat.clone();
            mp.atoms[a].position_bohr[q] += h;
            mm.atoms[a].position_bohr[q] -= h;
            let bsp = BasisSet::from_name("6-31g(d)", &mp).unwrap();
            let bsm = BasisSet::from_name("6-31g(d)", &mm).unwrap();
            let ep = uks(&mp, &bsp, XcFunctional::B3lyp, GridLevel::Coarse, &opts)
                .unwrap()
                .total_energy_hartree;
            let em = uks(&mm, &bsm, XcFunctional::B3lyp, GridLevel::Coarse, &opts)
                .unwrap()
                .total_energy_hartree;
            let fd = (ep - em) / (2.0 * h);
            let diff = (ana_cat[a][q] - fd).abs();
            assert!(
                diff < 5e-6,
                "B3LYP UKS H2O+ atom={a} q={q}: ana={:.9e} fd={:.9e} diff={diff:.3e}",
                ana_cat[a][q],
                fd
            );
        }
    }

    // Halogen ECP gradient on stretched HBr / LANL2DZ with B3LYP
    let hbr = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("Br", [0.0, 0.0, 1.52]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hbr = BasisSet::from_name("lanl2dz", &hbr).unwrap();
    let wf_hbr = rks(&hbr, &bs_hbr, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap();
    let ana_hbr = rks_gradient(&hbr, &bs_hbr, XcFunctional::B3lyp, GridLevel::Coarse, &wf_hbr).unwrap();

    let mut mp = hbr.clone();
    let mut mm = hbr.clone();
    mp.atoms[1].position_bohr[2] += h;
    mm.atoms[1].position_bohr[2] -= h;
    let ep = rks(
        &mp,
        &BasisSet::from_name("lanl2dz", &mp).unwrap(),
        XcFunctional::B3lyp,
        GridLevel::Coarse,
        &opts,
    )
    .unwrap()
    .total_energy_hartree;
    let em = rks(
        &mm,
        &BasisSet::from_name("lanl2dz", &mm).unwrap(),
        XcFunctional::B3lyp,
        GridLevel::Coarse,
        &opts,
    )
    .unwrap()
    .total_energy_hartree;
    let fd_z = (ep - em) / (2.0 * h);
    assert!(
        (ana_hbr[1][2] - fd_z).abs() < 5e-6,
        "HBr B3LYP/LANL2DZ grad_z: ana={} fd={fd_z}",
        ana_hbr[1][2]
    );
}
