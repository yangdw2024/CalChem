use std::collections::HashMap;

use calchem_basis::BasisSet;
use calchem_int::one_electron::shell_nuclear_block_with_charges;
use calchem_model::{Atom, Molecule, ScfOptions, ANGSTROM_TO_BOHR};
use calchem_prop::{
    compute_dipole, compute_esp_at_points, compute_esp_grid, compute_esp_surface,
    decode_esp_surface_binary, encode_esp_surface_binary, population_analysis, EspEvaluator,
    GridSpec, IsosurfaceMesh, AU_TO_DEBYE,
};
use calchem_scf::{rhf, uhf};

fn water(charge: i32, mult: usize) -> Molecule {
    Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
        ],
        charge,
        mult,
    )
}

/// Reference electronic potential `sum P_{mu nu} <mu|1/|r'-r||nu>` from the SCF nuclear-attraction
/// shell blocks with a unit point charge placed at `r` (fully independent of `EspEvaluator`).
fn reference_electronic_potential(bs: &BasisSet, p: &[f64], r: [f64; 3]) -> f64 {
    let probe = Molecule::new(vec![Atom::from_bohr("H", r).unwrap()], 0, 2);
    let n = bs.n_ao;
    let mut v = 0.0;
    for sh_a in &bs.shells {
        for sh_b in &bs.shells {
            let blk = shell_nuclear_block_with_charges(sh_a, sh_b, &probe, &[1.0]);
            for ia in 0..blk.rows {
                for ib in 0..blk.cols {
                    // block = -<a|1/|r'-r||b>
                    v -= p[(sh_a.ao_offset + ia) * n + sh_b.ao_offset + ib] * blk.get(ia, ib);
                }
            }
        }
    }
    v
}

fn euler_characteristic(mesh: &IsosurfaceMesh) -> isize {
    let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
    for t in mesh.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    assert!(edges.values().all(|&c| c == 2), "mesh must be a closed 2-manifold");
    mesh.n_vertices() as isize - edges.len() as isize + mesh.n_triangles() as isize
}

#[test]
fn test_esp_matches_independent_nuclear_attraction_integrals_and_ecp_screening() {
    let mol = water(0, 1);
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let wf = rhf(&mol, &bs, &ScfOptions::default()).unwrap();
    let ev = EspEvaluator::new(&mol, &bs, &wf.density_alpha).unwrap();
    let mut buf = vec![0.0; 16];

    let o = mol.atoms[0].position_bohr;
    for r in [
        [0.3, -0.2, o[2] + 0.25],
        [1.1, 0.7, -0.4],
        [-2.0, 1.5, 2.5],
        [0.0, 0.0, -6.0],
        [4.0, -3.0, 1.0],
    ] {
        let v_ref = reference_electronic_potential(&bs, &wf.density_alpha, r);
        let v = ev.electronic_potential(r, &mut buf);
        assert!(
            (v - v_ref).abs() < 1e-10 * v_ref.abs().max(1.0),
            "electronic ESP mismatch at {r:?}: {v} vs {v_ref}"
        );
    }

    // Exactly at a nucleus the potential is singular and must be rejected.
    assert!(compute_esp_at_points(&mol, &bs, &wf.density_alpha, &[o]).is_err());

    // ECP: HBr / LANL2DZ uses Z_eff(Br) = 7 and valence density -> neutral far-field (V ~ mu/R^2).
    let hbr = Molecule::new(
        vec![
            Atom::from_angstrom("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("Br", [0.0, 0.0, 1.414]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hbr = BasisSet::from_name("lanl2dz", &hbr).unwrap();
    let wf_hbr = rhf(&hbr, &bs_hbr, &ScfOptions::default()).unwrap();
    let r_far = 80.0;
    let v = compute_esp_at_points(
        &hbr,
        &bs_hbr,
        &wf_hbr.density_alpha,
        &[[r_far, 0.0, 0.0], [0.0, r_far, 0.0]],
    )
    .unwrap();
    // Perpendicular to the dipole axis the monopole/dipole terms vanish: only O(1/R^3) remains.
    for x in v {
        assert!(x.abs() < 1e-4, "ECP far-field ESP must vanish for neutral HBr, got {x}");
    }
}

#[test]
fn test_esp_far_field_multipole_limits_neutral_dipole_and_cation_charge() {
    // Neutral H2O: antisymmetric part (V(+R n) - V(-R n)) / 2 = mu . n / R^2 + O(R^-4).
    let mol = water(0, 1);
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let wf = rhf(&mol, &bs, &ScfOptions::default()).unwrap();
    let (mu_debye, _) = compute_dipole(&mol, &bs, &wf.density_alpha);
    let mu_au = mu_debye.map(|m| m / AU_TO_DEBYE);

    let n = [0.3, -0.2, (1.0f64 - 0.13).sqrt()];
    let mu_n = mu_au[0] * n[0] + mu_au[1] * n[1] + mu_au[2] * n[2];
    // Relative residual vs. mu.n/R^2 is the O(R^-2) octupole correction: it must shrink 4x per doubling.
    let rel_residual = |r: f64| -> f64 {
        let pts = [
            [r * n[0], r * n[1], r * n[2]],
            [-r * n[0], -r * n[1], -r * n[2]],
        ];
        let v = compute_esp_at_points(&mol, &bs, &wf.density_alpha, &pts).unwrap();
        let anti = 0.5 * (v[0] - v[1]);
        (anti - mu_n / (r * r)) / (mu_n / (r * r))
    };
    let e40 = rel_residual(40.0);
    let e80 = rel_residual(80.0);
    let e160 = rel_residual(160.0);
    assert!(e160.abs() < 1e-3, "dipolar far field relative residual {e160:.3e} at R=160");
    assert!((e40 / e80 - 4.0).abs() < 0.2, "octupole R^-2 scaling: {e40:.3e} / {e80:.3e}");
    assert!((e80 / e160 - 4.0).abs() < 0.2, "octupole R^-2 scaling: {e80:.3e} / {e160:.3e}");

    // H2O+ (UHF doublet): symmetric part (V(+R n) + V(-R n)) / 2 = q / R + O(R^-3).
    let cat = water(1, 2);
    let bs_c = BasisSet::from_name("6-31g(d)", &cat).unwrap();
    let wf_c = uhf(&cat, &bs_c, &ScfOptions::default()).unwrap();
    let p_tot: Vec<f64> = wf_c
        .density_alpha
        .iter()
        .zip(&wf_c.density_beta)
        .map(|(a, b)| a + b)
        .collect();
    let r = 100.0;
    let pts = [
        [r * n[0], r * n[1], r * n[2]],
        [-r * n[0], -r * n[1], -r * n[2]],
    ];
    let v = compute_esp_at_points(&cat, &bs_c, &p_tot, &pts).unwrap();
    let q_eff = 0.5 * (v[0] + v[1]) * r;
    assert!((q_eff - 1.0).abs() < 1e-3, "cation monopole q_eff = {q_eff}");

    // Grid ESP equals point ESP on identical points.
    let spec = GridSpec {
        origin_bohr: [-3.1, -2.9, -3.3],
        spacing_bohr: [1.7, 1.9, 2.1],
        counts: [3, 4, 3],
    };
    let grid = compute_esp_grid(&mol, &bs, &wf.density_alpha, &spec).unwrap();
    let probe = [
        spec.origin_bohr[0] + 2.0 * 1.7,
        spec.origin_bohr[1] + 1.0 * 1.9,
        spec.origin_bohr[2] + 2.0 * 2.1,
    ];
    let v_probe = compute_esp_at_points(&mol, &bs, &wf.density_alpha, &[probe]).unwrap()[0];
    let idx = (2 * 4 + 1) * 3 + 2;
    assert!((grid[idx] as f64 - v_probe).abs() < 1e-6 * v_probe.abs().max(1e-3));
}

#[test]
fn test_mulliken_lowdin_charges_conservation_symmetry_spin_and_ecp() {
    let mol = water(0, 1);
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let wf = rhf(&mol, &bs, &ScfOptions::default()).unwrap();
    let pop = population_analysis(&mol, &bs, &wf.density_alpha, None).unwrap();
    for q in [&pop.mulliken_charges, &pop.lowdin_charges] {
        assert!(q.iter().sum::<f64>().abs() < 1e-8, "charges must sum to 0: {q:?}");
        assert!(q[0] < -0.3, "O must be negative: {q:?}");
        assert!(q[1] > 0.15 && (q[1] - q[2]).abs() < 1e-8, "H symmetric positive: {q:?}");
    }
    // Literature RHF/6-31G(d) Mulliken charge of O in water ~ -0.87 e.
    assert!((pop.mulliken_charges[0] + 0.87).abs() < 0.08);
    assert!(pop.mulliken_spin.is_none());

    // H2O+ UHF doublet: charges sum to +1 and spin populations sum to exactly 1.
    let cat = water(1, 2);
    let bs_c = BasisSet::from_name("6-31g(d)", &cat).unwrap();
    let wf_c = uhf(&cat, &bs_c, &ScfOptions::default()).unwrap();
    let p_tot: Vec<f64> = wf_c.density_alpha.iter().zip(&wf_c.density_beta).map(|(a, b)| a + b).collect();
    let p_spin: Vec<f64> = wf_c.density_alpha.iter().zip(&wf_c.density_beta).map(|(a, b)| a - b).collect();
    let pc = population_analysis(&cat, &bs_c, &p_tot, Some(&p_spin)).unwrap();
    for q in [&pc.mulliken_charges, &pc.lowdin_charges] {
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-8);
    }
    for s in [pc.mulliken_spin.as_ref().unwrap(), pc.lowdin_spin.as_ref().unwrap()] {
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-8);
        // 1b1 hole is an O 2p_x lone pair: spin is localized on oxygen.
        assert!(s[0] > 0.8, "spin on O {s:?}");
    }

    // ECP-consistent charges (Z_eff(Br) = 7) for HBr / LANL2DZ.
    let hbr = Molecule::new(
        vec![
            Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("Br", [0.0, 0.0, 1.414 * ANGSTROM_TO_BOHR]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hbr = BasisSet::from_name("lanl2dz", &hbr).unwrap();
    let wf_hbr = rhf(&hbr, &bs_hbr, &ScfOptions::default()).unwrap();
    let ph = population_analysis(&hbr, &bs_hbr, &wf_hbr.density_alpha, None).unwrap();
    assert!(ph.mulliken_charges.iter().sum::<f64>().abs() < 1e-8);
    assert!(ph.lowdin_charges.iter().sum::<f64>().abs() < 1e-8);
    assert!(ph.mulliken_charges[1] < 0.0 && ph.mulliken_charges[0] > 0.0);
}

#[test]
fn test_water_esp_mapped_isodensity_surface_and_ccesp_roundtrip() {
    let mol = water(0, 1);
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let wf = rhf(&mol, &bs, &ScfOptions::default()).unwrap();
    let surf = compute_esp_surface(&mol, &bs, &wf.density_alpha, 0.001, 0.3, 5.0, false).unwrap();

    assert_eq!(euler_characteristic(&surf.mesh), 2);
    assert_eq!(surf.esp.len(), surf.mesh.n_vertices());
    // Water 0.001 a.u. surface: lone-pair minimum ~ -0.06 Eh/e, O-H hydrogen maximum ~ +0.07 Eh/e.
    assert!(surf.esp_min < -0.03 && surf.esp_min > -0.12, "esp_min {}", surf.esp_min);
    assert!(surf.esp_max > 0.04 && surf.esp_max < 0.15, "esp_max {}", surf.esp_max);

    let dist = |v: [f64; 3], a: usize| -> f64 {
        let p = mol.atoms[a].position_bohr;
        ((v[0] - p[0]).powi(2) + (v[1] - p[1]).powi(2) + (v[2] - p[2]).powi(2)).sqrt()
    };
    let vtx = |i: usize| -> [f64; 3] {
        [
            surf.mesh.vertices[3 * i] as f64,
            surf.mesh.vertices[3 * i + 1] as f64,
            surf.mesh.vertices[3 * i + 2] as f64,
        ]
    };
    let i_min = (0..surf.esp.len()).min_by(|&a, &b| surf.esp[a].total_cmp(&surf.esp[b])).unwrap();
    let i_max = (0..surf.esp.len()).max_by(|&a, &b| surf.esp[a].total_cmp(&surf.esp[b])).unwrap();
    let vmin = vtx(i_min);
    let vmax = vtx(i_max);
    assert!(dist(vmin, 0) < dist(vmin, 1).min(dist(vmin, 2)), "ESP minimum must sit on O side");
    assert!(dist(vmax, 0) > dist(vmax, 1).min(dist(vmax, 2)), "ESP maximum must sit on H side");

    let packet = encode_esp_surface_binary(&surf);
    let back = decode_esp_surface_binary(&packet).unwrap();
    assert_eq!(back, surf);
    assert!(decode_esp_surface_binary(&packet[..packet.len() - 4]).is_err());
}
