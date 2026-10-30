use calchem_basis::BasisSet;
use calchem_model::{Atom, CalcError, Molecule, ScfOptions};
use calchem_scf::rhf;

#[test]
fn test_h2_sto3g_rhf() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let wf = rhf(&mol, &basis, &ScfOptions::default()).unwrap();
    assert!(wf.converged);
    assert!((wf.total_energy_hartree - (-1.116714325063)).abs() < 1e-10);
}

#[test]
fn test_he_sto3g_rhf() {
    let mol = Molecule::new(vec![Atom::from_bohr("He", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let wf = rhf(&mol, &basis, &ScfOptions::default()).unwrap();
    assert!(wf.converged);
    assert!((wf.total_energy_hartree - (-2.807783957540)).abs() < 1e-10);
}

#[test]
fn test_ne_sto3g_rhf() {
    let mol = Molecule::new(vec![Atom::from_bohr("Ne", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let wf = rhf(&mol, &basis, &ScfOptions::default()).unwrap();
    assert!(wf.converged);
    assert!((wf.total_energy_hartree - (-126.6045291715)).abs() < 1e-9);
}

#[test]
fn test_h2o_sto3g_rhf_szabo_geometry() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let wf = rhf(&mol, &basis, &ScfOptions::default()).unwrap();
    assert!(wf.converged);

    // Exact mathematical RHF/STO-3G energy at Szabo geometry with stable Kummer Boys function:
    //   E_exact = -74.69531911423618 Eh (verified with 60-digit arbitrary precision).
    // Note: Legacy Python BASELINE.md recorded -74.6953191771 Eh because python_reference/one_electron.py
    // used an alternating Taylor series for Boys F_n(T) up to T = 30.0, which suffered -2.53e-7
    // floating-point cancellation at T = 29.34071349 (O 1s primitives with H nuclei), shifting V_00 by
    // -3.028e-8 Eh and total energy by -6.286e-8 Eh.
    let exact_ref = -74.69531911423618_f64;
    let legacy_py_baseline = -74.6953191771_f64;
    assert!(
        (wf.total_energy_hartree - exact_ref).abs() < 1e-10,
        "H2O/STO-3G energy {} differs from exact mathematical reference {} by {:e}",
        wf.total_energy_hartree,
        exact_ref,
        (wf.total_energy_hartree - exact_ref).abs()
    );
    assert!(
        (wf.total_energy_hartree - legacy_py_baseline).abs() < 1e-7,
        "H2O/STO-3G energy {} differs from legacy Python baseline {}",
        wf.total_energy_hartree,
        legacy_py_baseline
    );
}

#[test]
fn test_631g_rhf_benchmarks() {
    let h2 = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let bs_h2 = BasisSet::basis_6_31g(&h2).unwrap();
    let wf_h2 = rhf(&h2, &bs_h2, &ScfOptions::default()).unwrap();
    assert!(wf_h2.converged);
    assert!((wf_h2.total_energy_hartree - (-1.126742704452)).abs() < 1e-9);

    let h2o = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let bs_h2o = BasisSet::basis_6_31g(&h2o).unwrap();
    let wf_h2o = rhf(&h2o, &bs_h2o, &ScfOptions::default()).unwrap();
    assert!(wf_h2o.converged);
    assert!((wf_h2o.total_energy_hartree - (-75.800277509588)).abs() < 1e-9);
}

#[test]
fn test_rhf_returns_not_converged_error_when_max_iter_is_one() {
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
            Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::sto_3g(&mol).unwrap();
    let opts = ScfOptions {
        max_iter: 1,
        ..Default::default()
    };
    let res = rhf(&mol, &basis, &opts);
    assert!(matches!(res, Err(CalcError::NotConverged { .. })));
}
