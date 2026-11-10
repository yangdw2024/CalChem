use calchem_basis::BasisSet;
use calchem_int::compute_eri;
use calchem_model::{Atom, Molecule};

#[test]
fn test_eri_8fold_symmetry_h2o_sto3g() {
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
    let eri = compute_eri(&basis);
    let n = basis.n_ao;

    for mu in 0..n {
        for nu in 0..n {
            for lam in 0..n {
                for sig in 0..n {
                    let v = eri.get(mu, nu, lam, sig);
                    assert!((v - eri.get(nu, mu, lam, sig)).abs() < 1e-13);
                    assert!((v - eri.get(mu, nu, sig, lam)).abs() < 1e-13);
                    assert!((v - eri.get(nu, mu, sig, lam)).abs() < 1e-13);
                    assert!((v - eri.get(lam, sig, mu, nu)).abs() < 1e-13);
                    assert!((v - eri.get(sig, lam, mu, nu)).abs() < 1e-13);
                    assert!((v - eri.get(lam, sig, nu, mu)).abs() < 1e-13);
                    assert!((v - eri.get(sig, lam, nu, mu)).abs() < 1e-13);
                }
            }
        }
    }

    // Schwarz inequality check: |(mu nu | lam sig)| <= sqrt((mu nu | mu nu) * (lam sig | lam sig))
    for mu in 0..n {
        for nu in 0..n {
            let d_mn = eri.get(mu, nu, mu, nu);
            assert!(d_mn >= -1e-14, "Diagonal ERI ({mu}{nu}|{mu}{nu}) = {d_mn} < 0");
            for lam in 0..n {
                for sig in 0..n {
                    let d_ls = eri.get(lam, sig, lam, sig);
                    let bound = (d_mn.max(0.0) * d_ls.max(0.0)).sqrt();
                    let val = eri.get(mu, nu, lam, sig).abs();
                    assert!(
                        val <= bound + 1e-12,
                        "Schwarz violation at ({mu},{nu}|{lam},{sig}): |val|={val} > bound={bound}"
                    );
                }
            }
        }
    }
}
