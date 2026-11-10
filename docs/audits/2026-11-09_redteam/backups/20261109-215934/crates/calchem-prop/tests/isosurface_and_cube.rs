use std::collections::HashSet;

use calchem_basis::BasisSet;
use calchem_model::{Atom, Molecule, ScfOptions, BOHR_TO_ANGSTROM};
use calchem_prop::{
    auto_grid_for_molecule, compute_density_grid, compute_orbital_grid, decode_dual_mesh_binary,
    encode_dual_mesh_binary, extract_dual_isosurface, extract_isosurface, format_gaussian_cube,
    parse_gaussian_cube, IsosurfaceMesh,
};
use calchem_scf::{rhf, uhf};

/// Asserts that `mesh` is a strictly closed, consistently outward-oriented 2-manifold
/// (every directed half-edge `(u, v)` appears at most once and its opposite `(v, u)` also exists),
/// and returns `(V, E, F, euler_characteristic, total_area, signed_volume)`.
fn verify_closed_manifold(mesh: &IsosurfaceMesh) -> (usize, usize, usize, isize, f64, f64) {
    let n_v = mesh.n_vertices();
    let n_f = mesh.n_triangles();
    assert_eq!(mesh.vertices.len(), n_v * 3);
    assert_eq!(mesh.normals.len(), n_v * 3);
    assert_eq!(mesh.indices.len(), n_f * 3);
    assert!(n_v >= 4 && n_f >= 4, "mesh must be non-empty");

    let mut half_edges: HashSet<(u32, u32)> = HashSet::with_capacity(n_f * 3);
    let mut area = 0.0f64;
    let mut signed_vol = 0.0f64;

    for t in 0..n_f {
        let i0 = mesh.indices[t * 3];
        let i1 = mesh.indices[t * 3 + 1];
        let i2 = mesh.indices[t * 3 + 2];
        assert!((i0 as usize) < n_v && (i1 as usize) < n_v && (i2 as usize) < n_v);
        assert!(i0 != i1 && i1 != i2 && i0 != i2, "degenerate triangle indices");

        assert!(
            half_edges.insert((i0, i1)),
            "duplicate directed half-edge ({i0}, {i1}) -> non-manifold or inconsistent winding"
        );
        assert!(
            half_edges.insert((i1, i2)),
            "duplicate directed half-edge ({i1}, {i2}) -> non-manifold or inconsistent winding"
        );
        assert!(
            half_edges.insert((i2, i0)),
            "duplicate directed half-edge ({i2}, {i0}) -> non-manifold or inconsistent winding"
        );

        let v0 = [
            mesh.vertices[i0 as usize * 3] as f64,
            mesh.vertices[i0 as usize * 3 + 1] as f64,
            mesh.vertices[i0 as usize * 3 + 2] as f64,
        ];
        let v1 = [
            mesh.vertices[i1 as usize * 3] as f64,
            mesh.vertices[i1 as usize * 3 + 1] as f64,
            mesh.vertices[i1 as usize * 3 + 2] as f64,
        ];
        let v2 = [
            mesh.vertices[i2 as usize * 3] as f64,
            mesh.vertices[i2 as usize * 3 + 1] as f64,
            mesh.vertices[i2 as usize * 3 + 2] as f64,
        ];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let double_area = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        area += 0.5 * double_area;

        // Signed volume via divergence theorem: (1/6) sum v0 . (v1 x v2)
        let v1_x_v2 = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        signed_vol += (v0[0] * v1_x_v2[0] + v0[1] * v1_x_v2[1] + v0[2] * v1_x_v2[2]) / 6.0;
    }

    for &(u, v) in &half_edges {
        assert!(
            half_edges.contains(&(v, u)),
            "open boundary crack: half-edge ({u}, {v}) missing opposite ({v}, {u})"
        );
    }

    let n_e = half_edges.len() / 2;
    let euler = (n_v as isize) - (n_e as isize) + (n_f as isize);
    (n_v, n_e, n_f, euler, area, signed_vol)
}

#[test]
fn test_analytic_gaussian_sphere_manifold_geometry_and_normals() {
    let n = 61usize;
    let spacing = [0.1f64, 0.1, 0.1];
    let origin = [-3.0f64, -3.0, -3.0];
    let counts = [n, n, n];
    let radius = 1.5f64;
    let iso = (-radius * radius).exp() as f32;

    let mut grid = vec![0.0f32; n * n * n];
    for iz in 0..n {
        let z = origin[2] + (iz as f64) * spacing[2];
        for iy in 0..n {
            let y = origin[1] + (iy as f64) * spacing[1];
            for ix in 0..n {
                let x = origin[0] + (ix as f64) * spacing[0];
                let r2 = x * x + y * y + z * z;
                grid[(iz * n + iy) * n + ix] = (-r2).exp() as f32;
            }
        }
    }

    let mesh = extract_isosurface(&grid, origin, spacing, counts, iso, 1.0).unwrap();
    let (_v, _e, _f, euler, area, signed_vol) = verify_closed_manifold(&mesh);
    assert_eq!(euler, 2, "single sphere must have Euler characteristic V - E + F = 2");

    let exact_area = 4.0 * std::f64::consts::PI * radius * radius;
    let exact_vol = (4.0 / 3.0) * std::f64::consts::PI * radius * radius * radius;
    assert!(
        (area - exact_area).abs() / exact_area < 0.015,
        "sphere area {area:.4} deviates >1.5% from 4*pi*R^2 = {exact_area:.4}"
    );
    assert!(
        signed_vol > 0.0 && (signed_vol - exact_vol).abs() / exact_vol < 0.015,
        "signed volume {signed_vol:.4} must be positive (outward CCW) and within 1.5% of {exact_vol:.4}"
    );

    for i in 0..mesh.n_vertices() {
        let vx = mesh.vertices[i * 3] as f64;
        let vy = mesh.vertices[i * 3 + 1] as f64;
        let vz = mesh.vertices[i * 3 + 2] as f64;
        let r = (vx * vx + vy * vy + vz * vz).sqrt();
        assert!(
            (r - radius).abs() < 0.01,
            "vertex radius {r:.5} deviates from exact {radius}"
        );

        let nx = mesh.normals[i * 3] as f64;
        let ny = mesh.normals[i * 3 + 1] as f64;
        let nz = mesh.normals[i * 3 + 2] as f64;
        let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
        assert!((n_len - 1.0).abs() < 1e-5, "normal must be unit length");
        let cos_angle = (nx * vx + ny * vy + nz * vz) / r;
        assert!(
            cos_angle > 0.998,
            "outward normal alignment cos={cos_angle:.6} too low"
        );
    }
}

#[test]
fn test_f_xyz_octant_nodal_saddle_disambiguation_watertight() {
    // Field f(x,y,z) = x * y * z * exp(-0.6 * r^2) has 8 octant lobes (+4, -4) meeting at a
    // high-order saddle at the origin. Verify both + and - dual lobes are 100% crack-free.
    let n = 50usize;
    let spacing = [0.16f64, 0.16, 0.16];
    let origin = [-3.92f64, -3.92, -3.92]; // avoids landing exactly on 0.0 to exercise general cuts
    let counts = [n, n, n];

    let mut grid = vec![0.0f32; n * n * n];
    for iz in 0..n {
        let z = origin[2] + (iz as f64) * spacing[2];
        for iy in 0..n {
            let y = origin[1] + (iy as f64) * spacing[1];
            for ix in 0..n {
                let x = origin[0] + (ix as f64) * spacing[0];
                let r2 = x * x + y * y + z * z;
                grid[(iz * n + iy) * n + ix] = (x * y * z * (-0.6 * r2).exp()) as f32;
            }
        }
    }

    let dual = extract_dual_isosurface(&grid, origin, spacing, counts, 0.04, true).unwrap();
    let (_, _, _, euler_pos, _, vol_pos) = verify_closed_manifold(&dual.positive);
    let (_, _, _, euler_neg, _, vol_neg) = verify_closed_manifold(&dual.negative);
    // 4 disjoint topological spheres per phase => V - E + F = 4 * 2 = 8
    assert_eq!(euler_pos, 8);
    assert_eq!(euler_neg, 8);
    assert!(vol_pos > 0.0 && vol_neg > 0.0);
}

#[test]
fn test_water_631gd_homo_dual_mesh_density_cube_and_binary_protocol() {
    let mol = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
        ],
        0,
        1,
    );
    let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
    let wf = rhf(&mol, &bs, &ScfOptions::default()).unwrap();
    assert!(wf.converged);

    let spec = auto_grid_for_molecule(&mol, 4.5, 0.22).unwrap();
    // Water has 10 electrons -> 5 doubly occupied MOs -> HOMO is index 4 (out-of-plane 1b1 = 2p_x).
    let homo_idx = 4usize;
    let homo_grid = compute_orbital_grid(
        &bs,
        &wf.mo_coefficients,
        wf.n_ao,
        homo_idx,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
    )
    .unwrap();

    // Verify orbital normalization on 3D grid: integral |psi|^2 dV ~ 1.0
    let dv = spec.spacing_bohr[0] * spec.spacing_bohr[1] * spec.spacing_bohr[2];
    let norm_sq: f64 = homo_grid.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() * dv;
    assert!(
        (norm_sq - 1.0).abs() < 0.005,
        "HOMO grid norm^2 = {norm_sq:.6}, expected ~1.0"
    );

    let dual = extract_dual_isosurface(
        &homo_grid,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        0.04,
        true,
    )
    .unwrap();
    assert!(dual.units_angstrom);

    let (_, _, _, euler_p, area_p, vol_p) = verify_closed_manifold(&dual.positive);
    let (_, _, _, euler_n, area_n, vol_n) = verify_closed_manifold(&dual.negative);
    assert_eq!(euler_p, 2);
    assert_eq!(euler_n, 2);
    assert!(
        (area_p - area_n).abs() / area_p < 1e-4,
        "1b1 +/- lobes must be symmetric in area: {area_p} vs {area_n}"
    );
    assert!(
        (vol_p - vol_n).abs() / vol_p < 1e-4,
        "1b1 +/- lobes must be symmetric in volume: {vol_p} vs {vol_n}"
    );

    // Opposite X centroids in Angstrom across the YZ molecular mirror plane:
    let mean_x = |m: &IsosurfaceMesh| -> f64 {
        let nv = m.n_vertices();
        (0..nv).map(|i| m.vertices[i * 3] as f64).sum::<f64>() / (nv as f64)
    };
    let cx_pos = mean_x(&dual.positive);
    let cx_neg = mean_x(&dual.negative);
    assert!(cx_pos * cx_neg < 0.0, "lobes must lie on opposite sides of X=0");
    assert!((cx_pos + cx_neg).abs() < 1e-3, "centroids must mirror across X=0");
    assert!(cx_pos.abs() > 0.35 && cx_pos.abs() < 0.85);

    // Verify CCMSH\x01 binary wire round-trip:
    let packet = encode_dual_mesh_binary(&dual, homo_idx as i32, 0);
    let (decoded, dec_mo, dec_spin) = decode_dual_mesh_binary(&packet).unwrap();
    assert_eq!(dec_mo, homo_idx as i32);
    assert_eq!(dec_spin, 0);
    assert_eq!(decoded, dual);

    // Verify Gaussian .cube export and parse round-trip:
    let cube_txt = format_gaussian_cube(
        &mol,
        Some(&bs),
        &homo_grid,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        "Water 6-31G(d)",
        "MO 4 (HOMO)",
    )
    .unwrap();
    let (parsed_mol, parsed_spec, parsed_grid) = parse_gaussian_cube(&cube_txt).unwrap();
    assert_eq!(parsed_mol.atoms.len(), 3);
    assert_eq!(parsed_spec.counts, spec.counts);
    for q in 0..3 {
        assert!((parsed_spec.origin_bohr[q] - spec.origin_bohr[q]).abs() < 1e-5);
        assert!((parsed_spec.spacing_bohr[q] - spec.spacing_bohr[q]).abs() < 1e-5);
    }
    let max_diff = homo_grid
        .iter()
        .zip(&parsed_grid)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 2e-5, "Gaussian cube round-trip max diff {max_diff}");
}

#[test]
fn test_uhf_spin_density_and_ecp_valence_density_grids() {
    // 1. Water radical cation (H2O+, doublet): spin density P_alpha - P_beta integrates to 1.0 e.
    let h2o_cat = Molecule::new(
        vec![
            Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
            Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
        ],
        1,
        2,
    );
    let bs_cat = BasisSet::from_name("6-31g(d)", &h2o_cat).unwrap();
    let wf_cat = uhf(&h2o_cat, &bs_cat, &ScfOptions::default()).unwrap();
    assert!(wf_cat.converged);

    let spec_cat = auto_grid_for_molecule(&h2o_cat, 4.5, 0.22).unwrap();
    let spin_density_mat: Vec<f64> = wf_cat
        .density_alpha
        .iter()
        .zip(&wf_cat.density_beta)
        .map(|(a, b)| a - b)
        .collect();
    let spin_grid = compute_density_grid(
        &bs_cat,
        &spin_density_mat,
        spec_cat.origin_bohr,
        spec_cat.spacing_bohr,
        spec_cat.counts,
    )
    .unwrap();
    let dv_cat = spec_cat.spacing_bohr[0] * spec_cat.spacing_bohr[1] * spec_cat.spacing_bohr[2];
    let integrated_spin: f64 = spin_grid.iter().map(|&v| v as f64).sum::<f64>() * dv_cat;
    assert!(
        (integrated_spin - 1.0).abs() < 0.005,
        "UHF spin density integral = {integrated_spin:.6}, expected 1.0"
    );

    let spin_mesh = extract_isosurface(
        &spin_grid,
        spec_cat.origin_bohr,
        spec_cat.spacing_bohr,
        spec_cat.counts,
        0.01,
        BOHR_TO_ANGSTROM,
    )
    .unwrap();
    let (_, _, _, euler_spin, _, vol_spin) = verify_closed_manifold(&spin_mesh);
    assert!(euler_spin >= 2 && euler_spin % 2 == 0);
    assert!(vol_spin > 0.0);

    // 2. HBr with LANL2DZ ECP: smooth nodeless valence density integrates to 8.0 valence electrons,
    //    and Gaussian .cube atom line records Br effective nuclear charge Z_eff = 7.000000.
    let hbr = Molecule::new(
        vec![
            Atom::from_angstrom("Br", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [0.0, 0.0, 1.414]).unwrap(),
        ],
        0,
        1,
    );
    let bs_hbr = BasisSet::from_name("lanl2dz", &hbr).unwrap();
    let wf_hbr = rhf(&hbr, &bs_hbr, &ScfOptions::default()).unwrap();
    assert!(wf_hbr.converged);

    let spec_hbr = auto_grid_for_molecule(&hbr, 5.0, 0.25).unwrap();
    let rho_hbr = compute_density_grid(
        &bs_hbr,
        &wf_hbr.density_alpha,
        spec_hbr.origin_bohr,
        spec_hbr.spacing_bohr,
        spec_hbr.counts,
    )
    .unwrap();
    let dv_hbr = spec_hbr.spacing_bohr[0] * spec_hbr.spacing_bohr[1] * spec_hbr.spacing_bohr[2];
    let valence_electrons: f64 = rho_hbr.iter().map(|&v| v as f64).sum::<f64>() * dv_hbr;
    assert!(
        (valence_electrons - 8.0).abs() < 0.01,
        "HBr/LANL2DZ valence density integral = {valence_electrons:.6}, expected 8.0"
    );

    let hbr_cube = format_gaussian_cube(
        &hbr,
        Some(&bs_hbr),
        &rho_hbr,
        spec_hbr.origin_bohr,
        spec_hbr.spacing_bohr,
        spec_hbr.counts,
        "HBr LANL2DZ",
        "Valence Electron Density",
    )
    .unwrap();
    let br_atom_line = hbr_cube.lines().nth(6).unwrap();
    let cols: Vec<&str> = br_atom_line.split_whitespace().collect();
    assert_eq!(cols[0], "35");
    assert!((cols[1].parse::<f64>().unwrap() - 7.0).abs() < 1e-6);
}
