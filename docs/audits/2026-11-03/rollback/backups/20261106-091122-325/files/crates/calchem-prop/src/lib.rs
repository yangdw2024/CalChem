use calchem_basis::BasisSet;
use calchem_int::primitive::{overlap_primitive, GaussianPair};
use calchem_model::{CalcError, Molecule};
use calchem_tensor::Matrix2D;

pub const AU_TO_DEBYE: f64 = 2.54174623;

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

/// Evaluates molecular orbital `mo_idx` on a uniform 3D grid.
///
/// Output array is indexed in `(nz, ny, nx)` order with total size `nx * ny * nz`.
pub fn compute_orbital_grid(
    basis: &BasisSet,
    mo_coefficients: &[f64],
    n_mo: usize,
    mo_idx: usize,
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
) -> Result<Vec<f32>, CalcError> {
    let nx = counts[0];
    let ny = counts[1];
    let nz = counts[2];

    if mo_idx >= n_mo {
        return Err(CalcError::InvalidMolecule(format!(
            "Orbital index {mo_idx} out of range (n_mo = {n_mo})"
        )));
    }

    let mut grid = vec![0.0f32; nz * ny * nx];

    for iz in 0..nz {
        let z = origin[2] + (iz as f64) * spacing[2];
        for iy in 0..ny {
            let y = origin[1] + (iy as f64) * spacing[1];
            for ix in 0..nx {
                let x = origin[0] + (ix as f64) * spacing[0];
                let r = [x, y, z];

                let mut psi = 0.0f64;

                for shell in &basis.shells {
                    let dx = r[0] - shell.center[0];
                    let dy = r[1] - shell.center[1];
                    let dz = r[2] - shell.center[2];
                    let r2 = dx * dx + dy * dy + dz * dz;

                    let comps = shell.components();
                    for (ic, &c) in comps.iter().enumerate() {
                        let mu = shell.ao_offset + ic;
                        let c_mu = mo_coefficients[mu * n_mo + mo_idx];
                        if c_mu.abs() < 1e-16 {
                            continue;
                        }

                        let ang_val = dx.powi(c[0] as i32)
                            * dy.powi(c[1] as i32)
                            * dz.powi(c[2] as i32);

                        let mut ao_val = 0.0;
                        for (p_idx, p) in shell.primitives.iter().enumerate() {
                            let norm = shell.primitive_norm_for(p_idx, c);
                            ao_val += p.coeff * norm * (-p.exponent * r2).exp();
                        }
                        ao_val *= ang_val;

                        psi += c_mu * ao_val;
                    }
                }

                // Layout: (iz * ny + iy) * nx + ix
                let idx = (iz * ny + iy) * nx + ix;
                grid[idx] = psi as f32;
            }
        }
    }

    Ok(grid)
}
