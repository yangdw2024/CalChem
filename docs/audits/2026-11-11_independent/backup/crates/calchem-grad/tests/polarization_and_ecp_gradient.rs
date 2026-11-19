use calchem_basis::BasisSet;
use calchem_grad::{rhf_gradient, uhf_gradient};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rhf, uhf};

fn tight_scf_opts() -> ScfOptions {
    let mut opts = ScfOptions::default();
    opts.conv_energy = 1e-12;
    opts.conv_density = 1e-11;
    opts.max_iter = 150;
    opts
}

fn central_fd_rhf(mol: &Molecule, basis_name: &str, h: f64) -> Vec<[f64; 3]> {
    let opts = tight_scf_opts();
    let n = mol.atoms.len();
    let mut fd = vec![[0.0; 3]; n];
    for a in 0..n {
        for q in 0..3 {
            let mut m_plus = mol.clone();
            m_plus.atoms[a].position_bohr[q] += h;
            let bs_p = BasisSet::from_name(basis_name, &m_plus).unwrap();
            let ep = rhf(&m_plus, &bs_p, &opts).unwrap().total_energy_hartree;

            let mut m_minus = mol.clone();
            m_minus.atoms[a].position_bohr[q] -= h;
            let bs_m = BasisSet::from_name(basis_name, &m_minus).unwrap();
            let em = rhf(&m_minus, &bs_m, &opts).unwrap().total_energy_hartree;

            fd[a][q] = (ep - em) / (2.0 * h);
        }
    }
    fd
}

fn central_fd_uhf(mol: &Molecule, basis_name: &str, h: f64) -> Vec<[f64; 3]> {
    let opts = tight_scf_opts();
    let n = mol.atoms.len();
    let mut fd = vec![[0.0; 3]; n];
    for a in 0..n {
        for q in 0..3 {
            let mut m_plus = mol.clone();
            m_plus.atoms[a].position_bohr[q] += h;
            let bs_p = BasisSet::from_name(basis_name, &m_plus).unwrap();
            let ep = uhf(&m_plus, &bs_p, &opts).unwrap().total_energy_hartree;

            let mut m_minus = mol.clone();
            m_minus.atoms[a].position_bohr[q] -= h;
            let bs_m = BasisSet::from_name(basis_name, &m_minus).unwrap();
            let em = uhf(&m_minus, &bs_m, &opts).unwrap().total_energy_hartree;

            fd[a][q] = (ep - em) / (2.0 * h);
        }
    }
    fd
}

#[test]
fn test_analytic_gradient_631gd_rhf_and_uhf_matches_finite_difference() {
    let opts = tight_scf_opts();

    // Asymmetric distorted H2O in 6-31G(d) (tests d-shell 1e & 2e derivatives up to f-primitives)
    let h2o = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.05, -0.02, 0.10]).unwrap(),
            Atom::from_bohr("H", [0.10, 1.52, -0.95]).unwrap(),
            Atom::from_bohr("H", [-0.15, -1.38, -1.05]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
    let wf = rhf(&h2o, &bs, &opts).unwrap();
    let g_ana = rhf_gradient(&h2o, &bs, &wf).unwrap();
    let g_fd = central_fd_rhf(&h2o, "6-31g(d)", 1e-4);

    for a in 0..h2o.atoms.len() {
        for q in 0..3 {
            assert!(
                (g_ana[a][q] - g_fd[a][q]).abs() < 2e-6,
                "6-31G(d) RHF grad mismatch atom {a} axis {q}: ana={} vs fd={}",
                g_ana[a][q],
                g_fd[a][q]
            );
        }
    }

    // Open-shell H2O+ radical cation in 6-31G(d)
    let h2o_cat = Molecule::new(h2o.atoms.clone(), 1, 2);
    let wf_u = uhf(&h2o_cat, &bs, &opts).unwrap();
    let gu_ana = uhf_gradient(&h2o_cat, &bs, &wf_u).unwrap();
    let gu_fd = central_fd_uhf(&h2o_cat, "6-31g(d)", 1e-4);
    for a in 0..h2o_cat.atoms.len() {
        for q in 0..3 {
            assert!(
                (gu_ana[a][q] - gu_fd[a][q]).abs() < 2e-6,
                "6-31G(d) UHF grad mismatch atom {a} axis {q}: ana={} vs fd={}",
                gu_ana[a][q],
                gu_fd[a][q]
            );
        }
    }
}

#[test]
fn test_ecp_analytic_gradient_lanl2dz_and_lanl2dzdp_halogens() {
    let opts = tight_scf_opts();

    // 1. Compressed HBr in LANL2DZ (RHF + UHF cation)
    let hbr = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.1, -0.2, 0.05]).unwrap(),
            Atom::from_angstrom("Br", [0.4, 0.3, 1.25]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hbr = BasisSet::from_name("lanl2dz", &hbr).unwrap();
    let wf_hbr = rhf(&hbr, &bs_hbr, &opts).unwrap();
    let g_hbr = rhf_gradient(&hbr, &bs_hbr, &wf_hbr).unwrap();
    let fd_hbr = central_fd_rhf(&hbr, "lanl2dz", 1e-4);
    for a in 0..2 {
        for q in 0..3 {
            assert!(
                (g_hbr[a][q] - fd_hbr[a][q]).abs() < 2e-6,
                "HBr LANL2DZ RHF grad mismatch ({a},{q}): ana={} vs fd={}",
                g_hbr[a][q],
                fd_hbr[a][q]
            );
        }
    }

    let hbr_cat = Molecule::new(hbr.atoms.clone(), 1, 2);
    let wf_hbr_u = uhf(&hbr_cat, &bs_hbr, &opts).unwrap();
    let g_hbr_u = uhf_gradient(&hbr_cat, &bs_hbr, &wf_hbr_u).unwrap();
    let fd_hbr_u = central_fd_uhf(&hbr_cat, "lanl2dz", 1e-4);
    for a in 0..2 {
        for q in 0..3 {
            assert!(
                (g_hbr_u[a][q] - fd_hbr_u[a][q]).abs() < 2e-6,
                "HBr+ LANL2DZ UHF grad mismatch ({a},{q}): ana={} vs fd={}",
                g_hbr_u[a][q],
                fd_hbr_u[a][q]
            );
        }
    }

    // 2. Stretched HI in LANL2DZdp (Iodine 46-core ECP + d,p polarization + diffuse)
    let hi = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("I", [0.0, 0.0, 1.80]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hi = BasisSet::from_name("lanl2dzdp", &hi).unwrap();
    let wf_hi = rhf(&hi, &bs_hi, &opts).unwrap();
    let g_hi = rhf_gradient(&hi, &bs_hi, &wf_hi).unwrap();
    let fd_hi = central_fd_rhf(&hi, "lanl2dzdp", 1e-4);
    assert!(
        (g_hi[1][2] - fd_hi[1][2]).abs() < 2e-6,
        "HI LANL2DZdp grad mismatch: ana={} vs fd={}",
        g_hi[1][2],
        fd_hi[1][2]
    );

    // 3. Non-collinear 3-center HOBr in LANL2DZ (verifies A != B != C 3-center ECP gradients + translational sum = 0)
    let hobr = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.00, 0.05, 0.00]).unwrap(),
            Atom::from_angstrom("H", [0.95, -0.15, 0.25]).unwrap(),
            Atom::from_angstrom("Br", [-0.35, 1.75, -0.20]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hobr = BasisSet::from_name("lanl2dz", &hobr).unwrap();
    let wf_hobr = rhf(&hobr, &bs_hobr, &opts).unwrap();
    let g_hobr = rhf_gradient(&hobr, &bs_hobr, &wf_hobr).unwrap();
    let fd_hobr = central_fd_rhf(&hobr, "lanl2dz", 1e-4);

    for q in 0..3 {
        let sum_q: f64 = g_hobr.iter().map(|g| g[q]).sum();
        assert!(
            sum_q.abs() < 1e-8,
            "HOBr translational invariance violated along axis {q}: sum={sum_q}"
        );
    }
    for a in 0..3 {
        for q in 0..3 {
            assert!(
                (g_hobr[a][q] - fd_hobr[a][q]).abs() < 3e-6,
                "HOBr 3-center ECP grad mismatch ({a},{q}): ana={} vs fd={}",
                g_hobr[a][q],
                fd_hobr[a][q]
            );
        }
    }
}
