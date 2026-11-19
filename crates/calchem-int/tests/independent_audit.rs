use calchem_int::{density_fitting::invert_coulomb_metric, eri_primitive_with_omega};
use calchem_tensor::{matmul, Matrix2D};

fn integral(l: [[u8; 3]; 4], r: [[f64; 3]; 4], w: f64) -> f64 {
    eri_primitive_with_omega(
        l[0],
        l[1],
        l[2],
        l[3],
        0.7,
        r[0],
        1.1,
        r[1],
        0.9,
        r[2],
        1.3,
        r[3],
        Some(w),
    )
}

#[test]
fn erf_zero_large_omega_and_high_angular_center_derivatives() {
    let r = [
        [0.1, -0.2, 0.3],
        [0.4, 0.2, -0.1],
        [-0.3, 0.5, 0.9],
        [0.7, -0.4, 0.6],
    ];
    for a in 0..=3 {
        let l = [[a, 0, 0], [0, 1, 0], [0, 0, a], [0, 0, 0]];
        assert_eq!(integral(l, r, 0.0), 0.0);
        let full = calchem_int::eri_primitive(
            l[0], l[1], l[2], l[3], 0.7, r[0], 1.1, r[1], 0.9, r[2], 1.3, r[3],
        );
        assert!((integral(l, r, 1e200) - full).abs() < 1e-12);
        for w in [0.03, 0.2, 0.33, 3.0] {
            for center in 0..4 {
                for axis in 0..3 {
                    let mut plus = l;
                    plus[center][axis] += 1;
                    let mut analytic = 2.0 * [0.7, 1.1, 0.9, 1.3][center] * integral(plus, r, w);
                    if l[center][axis] > 0 {
                        let mut minus = l;
                        minus[center][axis] -= 1;
                        analytic -= l[center][axis] as f64 * integral(minus, r, w);
                    }
                    let h = 2e-5;
                    let mut rp = r;
                    let mut rm = r;
                    rp[center][axis] += h;
                    rm[center][axis] -= h;
                    let fd = (integral(l, rp, w) - integral(l, rm, w)) / (2.0 * h);
                    assert!(
                        (fd - analytic).abs() < 3e-9,
                        "l={l:?}, w={w}, center={center}, axis={axis}: {analytic} vs {fd}"
                    );
                }
            }
        }
    }
}

#[test]
fn ri_metric_rank_protection_and_normalization_invariance() {
    for scale in [1e-6_f64, 1.0, 1e6] {
        let mut v = Matrix2D::zeros(2, 2);
        v.data = vec![scale * scale, scale, scale, 1.0]; // duplicate auxiliary functions
        let inv = invert_coulomb_metric(&v).unwrap();
        let recovered = matmul(&matmul(&v, &inv), &v);
        for k in 0..4 {
            assert!((recovered.data[k] - v.data[k]).abs() < 1e-12 * v.data[k].abs().max(1.0));
        }
        assert!(inv.data.iter().all(|x| x.is_finite()));
    }
    let mut bad = Matrix2D::identity(2);
    bad.set(0, 1, 2.0);
    bad.set(1, 0, 2.0);
    assert!(invert_coulomb_metric(&bad).is_err());
}

#[test]
fn cached_and_on_the_fly_ri_coulomb_agree() {
    use calchem_basis::BasisSet;
    use calchem_int::density_fitting::RiJEngine;
    use calchem_model::{Atom, Molecule};
    let mol = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0; 3]).unwrap(),
            Atom::from_bohr("H", [0.2, 0.1, 1.4]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("6-31g", &mol).unwrap();
    let mut engine = RiJEngine::new(&mol, &basis).unwrap();
    assert!(engine.cached_b.is_some());
    let mut p = Matrix2D::identity(basis.n_ao);
    p.set(0, 1, 0.1);
    p.set(1, 0, 0.1);
    let cached = engine.build_coulomb(&basis, &p);
    engine.cached_b = None;
    let streaming = engine.build_coulomb(&basis, &p);
    let error = cached
        .data
        .iter()
        .zip(&streaming.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("RI-J cached versus shell-on-the-fly max error={error:.3e}");
    assert!(error < 1e-10);
}
