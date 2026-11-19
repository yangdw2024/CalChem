use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::{rhf, rks, uhf, uks};

#[test]
fn test_rhf_direct_scf_matches_stored_eri_energy_and_orbitals() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("6-31g", &mol).unwrap();

    let opts_stored = ScfOptions {
        conv_energy: 1e-11,
        conv_density: 1e-9,
        direct: Some(false),
        ..Default::default()
    };
    let opts_direct = ScfOptions {
        conv_energy: 1e-11,
        conv_density: 1e-9,
        direct: Some(true),
        ..Default::default()
    };

    let wf_stored = rhf(&mol, &bs, &opts_stored).unwrap();
    let wf_direct = rhf(&mol, &bs, &opts_direct).unwrap();

    assert!(wf_stored.converged);
    assert!(wf_direct.converged);

    let d_e = (wf_stored.total_energy_hartree - wf_direct.total_energy_hartree).abs();
    assert!(
        d_e < 1e-10,
        "RHF Direct vs Stored energy difference {d_e:.3e} exceeds tolerance"
    );

    for i in 0..wf_stored.mo_energies.len() {
        let d_mo = (wf_stored.mo_energies[i] - wf_direct.mo_energies[i]).abs();
        assert!(
            d_mo < 1e-8,
            "MO energy {i} differs by {d_mo:.3e}: stored={}, direct={}",
            wf_stored.mo_energies[i],
            wf_direct.mo_energies[i]
        );
    }
}

#[test]
fn test_rks_b3lyp_direct_scf_matches_stored_eri() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();

    let opts_stored = ScfOptions {
        conv_energy: 1e-11,
        conv_density: 1e-9,
        direct: Some(false),
        ..Default::default()
    };
    let opts_direct = ScfOptions {
        conv_energy: 1e-11,
        conv_density: 1e-9,
        direct: Some(true),
        ..Default::default()
    };

    let wf_stored = rks(
        &mol,
        &bs,
        XcFunctional::B3lyp,
        GridLevel::Coarse,
        &opts_stored,
    )
    .unwrap();
    let wf_direct = rks(
        &mol,
        &bs,
        XcFunctional::B3lyp,
        GridLevel::Coarse,
        &opts_direct,
    )
    .unwrap();

    assert!(wf_stored.converged);
    assert!(wf_direct.converged);

    let d_e = (wf_stored.total_energy_hartree - wf_direct.total_energy_hartree).abs();
    assert!(
        d_e < 1e-10,
        "RKS B3LYP Direct vs Stored energy difference {d_e:.3e} exceeds tolerance"
    );
}

#[test]
fn test_uhf_and_uks_direct_scf_matches_stored() {
    // Open-shell water cation doublet H2O+
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        1,
        2,
    );
    let bs = BasisSet::from_name("sto-3g", &mol).unwrap();

    let opts_stored = ScfOptions {
        conv_energy: 1e-11,
        conv_density: 1e-9,
        direct: Some(false),
        ..Default::default()
    };
    let opts_direct = ScfOptions {
        conv_energy: 1e-11,
        conv_density: 1e-9,
        direct: Some(true),
        ..Default::default()
    };

    // UHF comparison
    let wf_uhf_stored = uhf(&mol, &bs, &opts_stored).unwrap();
    let wf_uhf_direct = uhf(&mol, &bs, &opts_direct).unwrap();
    assert!(wf_uhf_stored.converged);
    assert!(wf_uhf_direct.converged);

    let d_uhf_e = (wf_uhf_stored.total_energy_hartree - wf_uhf_direct.total_energy_hartree).abs();
    assert!(
        d_uhf_e < 1e-10,
        "UHF Direct vs Stored energy diff {d_uhf_e:.3e}"
    );

    // UKS B3LYP comparison
    let wf_uks_stored = uks(
        &mol,
        &bs,
        XcFunctional::B3lyp,
        GridLevel::Coarse,
        &opts_stored,
    )
    .unwrap();
    let wf_uks_direct = uks(
        &mol,
        &bs,
        XcFunctional::B3lyp,
        GridLevel::Coarse,
        &opts_direct,
    )
    .unwrap();
    assert!(wf_uks_stored.converged);
    assert!(wf_uks_direct.converged);

    let d_uks_e = (wf_uks_stored.total_energy_hartree - wf_uks_direct.total_energy_hartree).abs();
    assert!(
        d_uks_e < 1e-10,
        "UKS B3LYP Direct vs Stored energy diff {d_uks_e:.3e}"
    );
}

#[test]
fn test_large_molecule_direct_scf_runs_without_memory_barrier() {
    // Construct a linear helium array with 24 atoms (24 AOs in STO-3G) or 70 AOs in 6-31G
    // Let's use 10 He atoms in 6-31G which gives 20 shells and 20 basis functions, or
    // Neon decamer (10 Ne atoms in 6-31G gives 90 basis functions > 64 threshold!)
    let atoms: Vec<Atom> = (0..8)
        .map(|i| Atom::from_angstrom("Ne", [i as f64 * 3.5, 0.0, 0.0]).unwrap())
        .collect();
    let mol = Molecule::new(atoms, 0, 1);
    let bs = BasisSet::from_name("6-31g", &mol).unwrap();

    assert!(
        bs.n_ao >= 72,
        "Ne8 with 6-31G should have >= 72 AOs, got {}",
        bs.n_ao
    );

    let opts = ScfOptions {
        max_iter: 25,
        conv_energy: 1e-8,
        conv_density: 1e-6,
        direct: Some(true),
        ..Default::default()
    };

    let wf = rhf(&mol, &bs, &opts).unwrap();
    assert!(
        wf.converged,
        "Direct SCF on 72+ AO system converged cleanly"
    );
    assert!(wf.total_energy_hartree < -1000.0);
}
