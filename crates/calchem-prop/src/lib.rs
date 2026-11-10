pub mod esp;
pub mod marching_cubes;

use std::fmt::Write as _;

use calchem_basis::{BasisSet, Shell};
use calchem_int::primitive::{overlap_primitive, GaussianPair};
use calchem_model::{Atom, CalcError, Molecule, ANGSTROM_TO_BOHR};
use calchem_tensor::Matrix2D;
use rayon::prelude::*;

pub use esp::{
    compute_esp_at_points, compute_esp_grid, compute_esp_surface, decode_esp_surface_binary,
    encode_esp_surface_binary, population_analysis, EspEvaluator, EspSurface, PopulationAnalysis,
    CCESP_HEADER_BYTES, CCESP_MAGIC,
};
pub use marching_cubes::{
    decode_dual_mesh_binary, encode_dual_mesh_binary, extract_dual_isosurface, extract_isosurface,
    DualOrbitalMesh, IsosurfaceMesh, CCMSH_HEADER_BYTES, CCMSH_MAGIC,
};

pub const AU_TO_DEBYE: f64 = 2.54174623;

/// Uniform 3D Cartesian grid specification in atomic units (Bohr).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSpec {
    pub origin_bohr: [f64; 3],
    pub spacing_bohr: [f64; 3],
    /// Grid point counts `[nx, ny, nz]` along X, Y, Z.
    pub counts: [usize; 3],
}

/// Automatically constructs a padded uniform 3D grid enclosing `mol` in Bohr.
pub fn auto_grid_for_molecule(
    mol: &Molecule,
    margin_bohr: f64,
    spacing_bohr: f64,
) -> Result<GridSpec, CalcError> {
    if mol.atoms.is_empty() {
        return Err(CalcError::InvalidMolecule(
            "Cannot build volumetric grid for an empty molecule".into(),
        ));
    }
    if !margin_bohr.is_finite() || margin_bohr <= 0.0 {
        return Err(CalcError::InvalidMolecule(format!(
            "Grid margin_bohr must be finite and positive, got {margin_bohr}"
        )));
    }
    if !spacing_bohr.is_finite() || spacing_bohr <= 0.0 {
        return Err(CalcError::InvalidMolecule(format!(
            "Grid spacing_bohr must be finite and positive, got {spacing_bohr}"
        )));
    }

    let mut min_c = [f64::INFINITY; 3];
    let mut max_c = [f64::NEG_INFINITY; 3];
    for atom in &mol.atoms {
        for q in 0..3 {
            min_c[q] = min_c[q].min(atom.position_bohr[q]);
            max_c[q] = max_c[q].max(atom.position_bohr[q]);
        }
    }

    let mut origin_bohr = [0.0; 3];
    let mut counts = [0usize; 3];
    for q in 0..3 {
        let span = (max_c[q] - min_c[q]) + 2.0 * margin_bohr;
        let n_intervals = (span / spacing_bohr).ceil().max(2.0) as usize;
        let actual_span = (n_intervals as f64) * spacing_bohr;
        let center = 0.5 * (min_c[q] + max_c[q]);
        origin_bohr[q] = center - 0.5 * actual_span;
        counts[q] = n_intervals + 1;
    }

    Ok(GridSpec {
        origin_bohr,
        spacing_bohr: [spacing_bohr; 3],
        counts,
    })
}

/// Computes the electric dipole moment vector (in Debye) and its magnitude (in Debye).
pub fn compute_dipole(
    mol: &Molecule,
    basis: &BasisSet,
    density_alpha: &[f64],
) -> ([f64; 3], f64) {
    let n = basis.n_ao;
    let mut mu_elec = [0.0; 3];

    // Compute electronic dipole integrals <mu | r_q | nu>
    for q in 0..3 {
        let mut d_mat = Matrix2D::zeros(n, n);
        for sh_a in &basis.shells {
            let comps_a = sh_a.components();
            for sh_b in &basis.shells {
                let comps_b = sh_b.components();
                for (ia, &ca) in comps_a.iter().enumerate() {
                    let mu = sh_a.ao_offset + ia;
                    for (ib, &cb) in comps_b.iter().enumerate() {
                        let nu = sh_b.ao_offset + ib;

                        let mut sum_prim = 0.0;
                        for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
                            let norm_a = sh_a.primitive_norm_for(pa_idx, ca);
                            for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
                                let norm_b = sh_b.primitive_norm_for(pb_idx, cb);
                                let weight = pa.coeff * pb.coeff * norm_a * norm_b;
                                let pair = GaussianPair::new(
                                    pa.exponent,
                                    sh_a.center,
                                    pb.exponent,
                                    sh_b.center,
                                );

                                // <g_a | r_q | g_b> = <g_{a+e_q} | g_b> + A_q * <g_a | g_b>
                                let mut ca_plus = ca.map(|v| v as i32);
                                ca_plus[q] += 1;
                                let cb_i32 = cb.map(|v| v as i32);
                                let ca_i32 = ca.map(|v| v as i32);

                                let term_plus = overlap_primitive(ca_plus, cb_i32, &pair);
                                let term_zero = overlap_primitive(ca_i32, cb_i32, &pair);

                                sum_prim += weight * (term_plus + sh_a.center[q] * term_zero);
                            }
                        }
                        d_mat.set(mu, nu, sum_prim);
                    }
                }
            }
        }

        // Contract with spin-summed density matrix P (stored in density_alpha)
        let mut sum_q = 0.0;
        for i in 0..n {
            for j in 0..n {
                sum_q += density_alpha[i * n + j] * d_mat.get(i, j);
            }
        }
        mu_elec[q] = sum_q;
    }

    // Nuclear dipole moment: sum_A Z_eff(A) * R_A
    let mut mu_nuc = [0.0; 3];
    for (idx, atom) in mol.atoms.iter().enumerate() {
        let z = basis.effective_nuclear_charge(mol, idx);
        for q in 0..3 {
            mu_nuc[q] += z * atom.position_bohr[q];
        }
    }

    // Total dipole moment in Debye: (mu_nuc - mu_elec) * AU_TO_DEBYE
    let mut dipole_debye = [0.0; 3];
    for q in 0..3 {
        dipole_debye[q] = (mu_nuc[q] - mu_elec[q]) * AU_TO_DEBYE;
    }
    let magnitude = (dipole_debye[0] * dipole_debye[0]
        + dipole_debye[1] * dipole_debye[1]
        + dipole_debye[2] * dipole_debye[2])
        .sqrt();

    (dipole_debye, magnitude)
}

#[derive(Clone, Copy)]
struct PrimEval {
    exponent: f64,
    weighted_axial_norm: f64,
}

struct PrecomputedShell {
    center: [f64; 3],
    ao_offset: usize,
    r2_cutoff: f64,
    prims: Vec<PrimEval>,
    comps: &'static [[u8; 3]],
    /// Ratio `primitive_norm(alpha, comp) / primitive_norm(alpha, [L,0,0])` per component.
    comp_scales: Vec<f64>,
}

impl PrecomputedShell {
    fn from_shell(sh: &Shell) -> Self {
        let comps = sh.components();
        let l_ang = sh.shell_type.angular_momentum() as f64;
        let alpha_min = sh
            .primitives
            .iter()
            .map(|p| p.exponent)
            .fold(f64::INFINITY, f64::min)
            .max(1e-12);
        // exp(-(36 + 4L)) < 2.3e-16 ensures radial tail truncation error is well below f32/f64 precision.
        let r2_cutoff = (36.0 + 4.0 * l_ang) / alpha_min;

        let prims = sh
            .primitives
            .iter()
            .map(|p| PrimEval {
                exponent: p.exponent,
                weighted_axial_norm: p.coeff * p.norm,
            })
            .collect();

        let lead_norm = sh.primitives.first().map(|p| p.norm).unwrap_or(1.0);
        let comp_scales = comps
            .iter()
            .map(|&c| {
                if sh.primitives.is_empty() || lead_norm == 0.0 {
                    1.0
                } else {
                    sh.primitive_norm_for(0, c) / lead_norm
                }
            })
            .collect();

        Self {
            center: sh.center,
            ao_offset: sh.ao_offset,
            r2_cutoff,
            prims,
            comps,
            comp_scales,
        }
    }

    #[inline]
    fn eval_radial(&self, r2: f64) -> f64 {
        let mut s = 0.0;
        for p in &self.prims {
            s += p.weighted_axial_norm * (-p.exponent * r2).exp();
        }
        s
    }
}

#[inline]
fn pow_u8(val: f64, exp: u8) -> f64 {
    match exp {
        0 => 1.0,
        1 => val,
        2 => val * val,
        3 => val * val * val,
        _ => val.powi(exp as i32),
    }
}

#[inline]
fn cartesian_monomial(dx: f64, dy: f64, dz: f64, c: [u8; 3]) -> f64 {
    pow_u8(dx, c[0]) * pow_u8(dy, c[1]) * pow_u8(dz, c[2])
}

fn validate_grid_args(
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3],
) -> Result<usize, CalcError> {
    if counts.contains(&0) {
        return Err(CalcError::InvalidMolecule(format!(
            "Grid counts must be >= 1 along every axis, got {counts:?}"
        )));
    }
    if !origin.iter().all(|x| x.is_finite()) || !spacing.iter().all(|&s| s.is_finite() && s > 0.0) {
        return Err(CalcError::InvalidMolecule(
            "Grid origin must be finite and spacing must be strictly positive".into(),
        ));
    }
    counts[0]
        .checked_mul(counts[1])
        .and_then(|v| v.checked_mul(counts[2]))
        .ok_or_else(|| CalcError::InvalidMolecule("Grid size overflow".into()))
}

/// Evaluates molecular orbital `mo_idx` on a uniform 3D grid in parallel across Z-slices.
///
/// Output array is indexed in `(nz, ny, nx)` order (`(iz * ny + iy) * nx + ix`), matching
/// VTK `vtkImageData` native scalar layout with zero transpose.
pub fn compute_orbital_grid(
    basis: &BasisSet,
    mo_coefficients: &[f64],
    n_mo: usize,
    mo_idx: usize,
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
) -> Result<Vec<f32>, CalcError> {
    let total_pts = validate_grid_args(origin, spacing, counts)?;
    let [nx, ny, _nz] = counts;

    if mo_idx >= n_mo {
        return Err(CalcError::InvalidMolecule(format!(
            "Orbital index {mo_idx} out of range (n_mo = {n_mo})"
        )));
    }
    if mo_coefficients.len() < basis.n_ao * n_mo {
        return Err(CalcError::InvalidMolecule(format!(
            "MO coefficient matrix length {} smaller than n_ao * n_mo ({} * {})",
            mo_coefficients.len(),
            basis.n_ao,
            n_mo
        )));
    }

    // Extract active MO coefficients and precompute per-shell radial evaluators.
    let mut mo_col = vec![0.0f64; basis.n_ao];
    for mu in 0..basis.n_ao {
        mo_col[mu] = mo_coefficients[mu * n_mo + mo_idx];
    }

    let shells: Vec<(PrecomputedShell, bool)> = basis
        .shells
        .iter()
        .map(|sh| {
            let pre = PrecomputedShell::from_shell(sh);
            let active = (0..pre.comps.len()).any(|ic| mo_col[pre.ao_offset + ic].abs() >= 1e-16);
            (pre, active)
        })
        .collect();

    let mut grid = vec![0.0f32; total_pts];
    let slice_len = ny * nx;

    grid.par_chunks_mut(slice_len)
        .enumerate()
        .for_each(|(iz, z_slice)| {
            let z = origin[2] + (iz as f64) * spacing[2];
            for iy in 0..ny {
                let y = origin[1] + (iy as f64) * spacing[1];
                let row_off = iy * nx;
                for ix in 0..nx {
                    let x = origin[0] + (ix as f64) * spacing[0];
                    let mut psi = 0.0f64;

                    for (sh, active) in &shells {
                        if !*active {
                            continue;
                        }
                        let dx = x - sh.center[0];
                        let dy = y - sh.center[1];
                        let dz = z - sh.center[2];
                        let r2 = dx * dx + dy * dy + dz * dz;
                        if r2 > sh.r2_cutoff {
                            continue;
                        }

                        let radial = sh.eval_radial(r2);
                        if radial.abs() < 1e-18 {
                            continue;
                        }

                        for (ic, &c) in sh.comps.iter().enumerate() {
                            let c_mu = mo_col[sh.ao_offset + ic];
                            if c_mu.abs() < 1e-16 {
                                continue;
                            }
                            let ang = cartesian_monomial(dx, dy, dz, c);
                            psi += c_mu * radial * sh.comp_scales[ic] * ang;
                        }
                    }

                    z_slice[row_off + ix] = psi as f32;
                }
            }
        });

    Ok(grid)
}

/// Evaluates a real-space density field `\rho(r) = \sum_{\mu,\nu} P_{\mu\nu} \phi_\mu(r) \phi_\nu(r)`
/// (either total electron density `P = P^\alpha + P^\beta` or spin density `P^s = P^\alpha - P^\beta`)
/// on a uniform 3D grid in parallel across Z-slices.
///
/// Output array is indexed in `(nz, ny, nx)` order.
pub fn compute_density_grid(
    basis: &BasisSet,
    density_matrix: &[f64],
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
) -> Result<Vec<f32>, CalcError> {
    let total_pts = validate_grid_args(origin, spacing, counts)?;
    let n_ao = basis.n_ao;
    if density_matrix.len() != n_ao * n_ao {
        return Err(CalcError::InvalidMolecule(format!(
            "Density matrix length {} does not match n_ao^2 ({n_ao}^2)",
            density_matrix.len()
        )));
    }

    let [nx, ny, _nz] = counts;
    let shells: Vec<PrecomputedShell> = basis
        .shells
        .iter()
        .map(PrecomputedShell::from_shell)
        .collect();

    let mut grid = vec![0.0f32; total_pts];
    let slice_len = ny * nx;

    grid.par_chunks_mut(slice_len)
        .enumerate()
        .for_each_init(
            || Vec::<(usize, f64)>::with_capacity(n_ao),
            |active_aos, (iz, z_slice)| {
                let z = origin[2] + (iz as f64) * spacing[2];
                for iy in 0..ny {
                    let y = origin[1] + (iy as f64) * spacing[1];
                    let row_off = iy * nx;
                    for ix in 0..nx {
                        let x = origin[0] + (ix as f64) * spacing[0];
                        active_aos.clear();

                        for sh in &shells {
                            let dx = x - sh.center[0];
                            let dy = y - sh.center[1];
                            let dz = z - sh.center[2];
                            let r2 = dx * dx + dy * dy + dz * dz;
                            if r2 > sh.r2_cutoff {
                                continue;
                            }
                            let radial = sh.eval_radial(r2);
                            if radial.abs() < 1e-18 {
                                continue;
                            }
                            for (ic, &c) in sh.comps.iter().enumerate() {
                                let phi = radial * sh.comp_scales[ic] * cartesian_monomial(dx, dy, dz, c);
                                if phi.abs() >= 1e-15 {
                                    active_aos.push((sh.ao_offset + ic, phi));
                                }
                            }
                        }

                        let mut rho = 0.0f64;
                        for (a_idx, &(mu, phi_mu)) in active_aos.iter().enumerate() {
                            rho += density_matrix[mu * n_ao + mu] * phi_mu * phi_mu;
                            for &(nu, phi_nu) in &active_aos[(a_idx + 1)..] {
                                let p_sym =
                                    density_matrix[mu * n_ao + nu] + density_matrix[nu * n_ao + mu];
                                rho += p_sym * phi_mu * phi_nu;
                            }
                        }

                        z_slice[row_off + ix] = rho as f32;
                    }
                }
            },
        );

    Ok(grid)
}

/// Formats a `(nz, ny, nx)` volumetric grid into standard Gaussian `.cube` text (Bohr coordinates).
///
/// Gaussian `.cube` files store scalar values in X-outer, Y-middle, Z-inner order with up to 6
/// `13.5E` values per line and a line break at the end of every Z-column (`iz == nz - 1`).
pub fn format_gaussian_cube(
    mol: &Molecule,
    basis: Option<&BasisSet>,
    grid_zyx: &[f32],
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
    comment1: &str,
    comment2: &str,
) -> Result<String, CalcError> {
    let total_pts = validate_grid_args(origin_bohr, spacing_bohr, counts)?;
    if grid_zyx.len() != total_pts {
        return Err(CalcError::InvalidMolecule(format!(
            "Grid length {} does not match counts {:?} (expected {total_pts})",
            grid_zyx.len(),
            counts
        )));
    }
    let [nx, ny, nz] = counts;

    let clean_line = |s: &str, fallback: &str| -> String {
        let trimmed = s.lines().next().unwrap_or("").trim();
        if trimmed.is_empty() {
            fallback.to_string()
        } else {
            trimmed.to_string()
        }
    };

    // Preallocate ~15 bytes per scalar value + header.
    let mut out = String::with_capacity(256 + mol.atoms.len() * 64 + total_pts * 15);
    let _ = writeln!(out, "{}", clean_line(comment1, "Generated by CalChem Rust Core"));
    let _ = writeln!(
        out,
        "{}",
        clean_line(comment2, "Volumetric Grid (atomic units / Bohr)")
    );
    let _ = writeln!(
        out,
        "{:5} {:12.6} {:12.6} {:12.6}",
        mol.atoms.len(),
        origin_bohr[0],
        origin_bohr[1],
        origin_bohr[2]
    );
    let _ = writeln!(out, "{:5} {:12.6} {:12.6} {:12.6}", nx, spacing_bohr[0], 0.0, 0.0);
    let _ = writeln!(out, "{:5} {:12.6} {:12.6} {:12.6}", ny, 0.0, spacing_bohr[1], 0.0);
    let _ = writeln!(out, "{:5} {:12.6} {:12.6} {:12.6}", nz, 0.0, 0.0, spacing_bohr[2]);

    for (idx, atom) in mol.atoms.iter().enumerate() {
        let z_charge = basis
            .map(|bs| bs.effective_nuclear_charge(mol, idx))
            .unwrap_or(atom.z as f64);
        let p = atom.position_bohr;
        let _ = writeln!(
            out,
            "{:5} {:12.6} {:12.6} {:12.6} {:12.6}",
            atom.z, z_charge, p[0], p[1], p[2]
        );
    }

    let stride_y = nx;
    let stride_z = nx * ny;
    for ix in 0..nx {
        for iy in 0..ny {
            let base_xy = iy * stride_y + ix;
            for iz in 0..nz {
                let val = grid_zyx[iz * stride_z + base_xy];
                let _ = write!(out, " {:12.5E}", val);
                if (iz + 1) % 6 == 0 || iz + 1 == nz {
                    out.push('\n');
                }
            }
        }
    }

    Ok(out)
}

/// Parses a standard single-field Gaussian `.cube` text into `(Molecule, GridSpec, Vec<f32>)`
/// where the returned scalar buffer is converted into `(nz, ny, nx)` memory order and Bohr units.
pub fn parse_gaussian_cube(cube_text: &str) -> Result<(Molecule, GridSpec, Vec<f32>), CalcError> {
    let mut lines = cube_text.lines();
    let _c1 = lines
        .next()
        .ok_or_else(|| CalcError::InvalidMolecule("Empty Gaussian cube text".into()))?;
    let _c2 = lines
        .next()
        .ok_or_else(|| CalcError::InvalidMolecule("Truncated Gaussian cube header".into()))?;

    let line3 = lines
        .next()
        .ok_or_else(|| CalcError::InvalidMolecule("Missing Gaussian cube origin line".into()))?;
    let p3: Vec<&str> = line3.split_whitespace().collect();
    if p3.len() < 4 {
        return Err(CalcError::InvalidMolecule(
            "Malformed Gaussian cube atom/origin line".into(),
        ));
    }
    let raw_natom: isize = p3[0]
        .parse()
        .map_err(|_| CalcError::InvalidMolecule("Invalid atom count in cube".into()))?;
    let has_mo_header = raw_natom < 0;
    let natom = raw_natom.unsigned_abs();
    let mut origin_bohr = [0.0f64; 3];
    for q in 0..3 {
        origin_bohr[q] = p3[q + 1]
            .parse()
            .map_err(|_| CalcError::InvalidMolecule("Invalid origin coordinate in cube".into()))?;
    }

    let mut counts = [0usize; 3];
    let mut spacing_bohr = [0.0f64; 3];
    let mut axis_in_angstrom = false;
    for axis in 0..3 {
        let ax_line = lines
            .next()
            .ok_or_else(|| CalcError::InvalidMolecule("Missing cube axis line".into()))?;
        let pa: Vec<&str> = ax_line.split_whitespace().collect();
        if pa.len() < 4 {
            return Err(CalcError::InvalidMolecule("Malformed cube axis line".into()));
        }
        let n_signed: isize = pa[0]
            .parse()
            .map_err(|_| CalcError::InvalidMolecule("Invalid grid count in cube".into()))?;
        if n_signed == 0 {
            return Err(CalcError::InvalidMolecule("Cube grid count cannot be 0".into()));
        }
        if n_signed < 0 {
            axis_in_angstrom = true;
        }
        counts[axis] = n_signed.unsigned_abs();
        let step_raw: f64 = pa[axis + 1]
            .parse()
            .map_err(|_| CalcError::InvalidMolecule("Invalid grid step in cube".into()))?;
        spacing_bohr[axis] = if n_signed < 0 {
            step_raw.abs() * ANGSTROM_TO_BOHR
        } else {
            step_raw.abs()
        };
    }

    if axis_in_angstrom {
        for q in 0..3 {
            origin_bohr[q] *= ANGSTROM_TO_BOHR;
        }
    }

    let mut atoms = Vec::with_capacity(natom);
    for _ in 0..natom {
        let aline = lines
            .next()
            .ok_or_else(|| CalcError::InvalidMolecule("Truncated atom list in cube".into()))?;
        let pa: Vec<&str> = aline.split_whitespace().collect();
        if pa.len() < 5 {
            return Err(CalcError::InvalidMolecule("Malformed atom line in cube".into()));
        }
        let z: u8 = pa[0]
            .parse()
            .map_err(|_| CalcError::InvalidMolecule("Invalid atomic number in cube".into()))?;
        let sym = calchem_model::symbol_from_atomic_number(z)?;
        let mut pos = [0.0f64; 3];
        for q in 0..3 {
            pos[q] = pa[q + 2]
                .parse()
                .map_err(|_| CalcError::InvalidMolecule("Invalid atom coordinate in cube".into()))?;
            if axis_in_angstrom {
                pos[q] *= ANGSTROM_TO_BOHR;
            }
        }
        atoms.push(Atom::from_bohr(&sym, pos)?);
    }

    if has_mo_header {
        // Multi-orbital header line: first integer `n_mo`, followed by `n_mo` indices.
        let mo_line = lines
            .next()
            .ok_or_else(|| CalcError::InvalidMolecule("Missing MO header line in cube".into()))?;
        let pm: Vec<&str> = mo_line.split_whitespace().collect();
        let n_mo_cube: usize = pm
            .first()
            .ok_or_else(|| CalcError::InvalidMolecule("Empty MO header line in cube".into()))?
            .parse()
            .map_err(|_| CalcError::InvalidMolecule("Invalid n_mo in cube header".into()))?;
        if n_mo_cube != 1 {
            return Err(CalcError::InvalidMolecule(format!(
                "parse_gaussian_cube supports single-field cubes (got n_mo={n_mo_cube})"
            )));
        }
    }

    let [nx, ny, nz] = counts;
    let total_pts = validate_grid_args(origin_bohr, spacing_bohr, counts)?;
    let mut vals_xyz = Vec::with_capacity(total_pts);
    for line in lines {
        for tok in line.split_whitespace() {
            let v: f32 = tok
                .parse()
                .map_err(|_| CalcError::InvalidMolecule(format!("Invalid float '{tok}' in cube")))?;
            vals_xyz.push(v);
        }
    }
    if vals_xyz.len() != total_pts {
        return Err(CalcError::InvalidMolecule(format!(
            "Cube scalar count {} does not match nx*ny*nz = {total_pts}",
            vals_xyz.len()
        )));
    }

    let mut grid_zyx = vec![0.0f32; total_pts];
    let stride_y = nx;
    let stride_z = nx * ny;
    let mut src = 0usize;
    for ix in 0..nx {
        for iy in 0..ny {
            let base_xy = iy * stride_y + ix;
            for iz in 0..nz {
                grid_zyx[iz * stride_z + base_xy] = vals_xyz[src];
                src += 1;
            }
        }
    }

    let mol = Molecule::new(atoms, 0, 1);
    let spec = GridSpec {
        origin_bohr,
        spacing_bohr,
        counts,
    };
    Ok((mol, spec, grid_zyx))
}
