use calchem_basis::{AtomEcp, BasisSet, Shell};
use calchem_int::boys::boys_batch;
use calchem_int::ecp::ecp_atom_primitive;
use calchem_int::eri::EriPairData;
use calchem_int::primitive::{kinetic_primitive, nuclear_vrr, overlap_primitive, GaussianPair};
use calchem_model::Molecule;
use calchem_tensor::Matrix2D;

pub fn nuclear_repulsion_derivative(mol: &Molecule, atom: usize, axis: usize) -> f64 {
    let mut grad = 0.0;
    let r_a = mol.atoms[atom].position_bohr;
    let z_a = mol.atoms[atom].z as f64;

    for (b_idx, atom_b) in mol.atoms.iter().enumerate() {
        if b_idx == atom {
            continue;
        }
        let r_b = atom_b.position_bohr;
        let z_b = atom_b.z as f64;
        let dx = r_a[0] - r_b[0];
        let dy = r_a[1] - r_b[1];
        let dz = r_a[2] - r_b[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > 1e-12 {
            let diff_axis = r_a[axis] - r_b[axis];
            grad -= z_a * z_b * diff_axis / (dist * dist * dist);
        }
    }
    grad
}

pub fn nuclear_repulsion_derivative_with_basis(
    mol: &Molecule,
    basis: &BasisSet,
    atom: usize,
    axis: usize,
) -> f64 {
    let mut grad = 0.0;
    let r_a = mol.atoms[atom].position_bohr;
    let z_a = basis.effective_nuclear_charge(mol, atom);

    for (b_idx, atom_b) in mol.atoms.iter().enumerate() {
        if b_idx == atom {
            continue;
        }
        let r_b = atom_b.position_bohr;
        let z_b = basis.effective_nuclear_charge(mol, b_idx);
        let dx = r_a[0] - r_b[0];
        let dy = r_a[1] - r_b[1];
        let dz = r_a[2] - r_b[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > 1e-12 {
            let diff_axis = r_a[axis] - r_b[axis];
            grad -= z_a * z_b * diff_axis / (dist * dist * dist);
        }
    }
    grad
}

#[inline]
fn prim_overlap_deriv_a(
    ca: [u8; 3],
    cb: [u8; 3],
    alpha: f64,
    axis: usize,
    pair: &GaussianPair,
) -> f64 {
    let mut ca_plus = ca.map(|v| v as i32);
    ca_plus[axis] += 1;
    let b_i32 = cb.map(|v| v as i32);
    let mut val = 2.0 * alpha * overlap_primitive(ca_plus, b_i32, pair);
    if ca[axis] > 0 {
        let mut ca_minus = ca.map(|v| v as i32);
        ca_minus[axis] -= 1;
        val -= (ca[axis] as f64) * overlap_primitive(ca_minus, b_i32, pair);
    }
    val
}

#[inline]
fn prim_overlap_deriv_b(
    ca: [u8; 3],
    cb: [u8; 3],
    beta: f64,
    axis: usize,
    pair: &GaussianPair,
) -> f64 {
    let a_i32 = ca.map(|v| v as i32);
    let mut cb_plus = cb.map(|v| v as i32);
    cb_plus[axis] += 1;
    let mut val = 2.0 * beta * overlap_primitive(a_i32, cb_plus, pair);
    if cb[axis] > 0 {
        let mut cb_minus = cb.map(|v| v as i32);
        cb_minus[axis] -= 1;
        val -= (cb[axis] as f64) * overlap_primitive(a_i32, cb_minus, pair);
    }
    val
}

#[inline]
fn prim_kinetic_deriv_a(
    ca: [u8; 3],
    cb: [u8; 3],
    alpha: f64,
    beta: f64,
    axis: usize,
    pair: &GaussianPair,
) -> f64 {
    let mut ca_plus = ca.map(|v| v as i32);
    ca_plus[axis] += 1;
    let b_i32 = cb.map(|v| v as i32);
    let mut val = 2.0 * alpha * kinetic_primitive(ca_plus, b_i32, beta, pair);
    if ca[axis] > 0 {
        let mut ca_minus = ca.map(|v| v as i32);
        ca_minus[axis] -= 1;
        val -= (ca[axis] as f64) * kinetic_primitive(ca_minus, b_i32, beta, pair);
    }
    val
}

#[inline]
fn prim_kinetic_deriv_b(
    ca: [u8; 3],
    cb: [u8; 3],
    _alpha: f64,
    beta: f64,
    axis: usize,
    pair: &GaussianPair,
) -> f64 {
    let a_i32 = ca.map(|v| v as i32);
    let mut cb_plus = cb.map(|v| v as i32);
    cb_plus[axis] += 1;
    let mut val = 2.0 * beta * kinetic_primitive(a_i32, cb_plus, beta, pair);
    if cb[axis] > 0 {
        let mut cb_minus = cb.map(|v| v as i32);
        cb_minus[axis] -= 1;
        val -= (cb[axis] as f64) * kinetic_primitive(a_i32, cb_minus, beta, pair);
    }
    val
}

#[inline]
fn prim_nuclear_deriv_a(
    ca: [u8; 3],
    cb: [u8; 3],
    alpha: f64,
    axis: usize,
    pair: &GaussianPair,
    c_k: [f64; 3],
    boys_vals: &[f64],
) -> f64 {
    let mut ca_plus = ca;
    ca_plus[axis] += 1;
    let mut val = 2.0 * alpha * nuclear_vrr(ca_plus, cb, 0, pair, c_k, boys_vals);
    if ca[axis] > 0 {
        let mut ca_minus = ca;
        ca_minus[axis] -= 1;
        val -= (ca[axis] as f64) * nuclear_vrr(ca_minus, cb, 0, pair, c_k, boys_vals);
    }
    val
}

#[inline]
fn prim_nuclear_deriv_b(
    ca: [u8; 3],
    cb: [u8; 3],
    beta: f64,
    axis: usize,
    pair: &GaussianPair,
    c_k: [f64; 3],
    boys_vals: &[f64],
) -> f64 {
    let mut cb_plus = cb;
    cb_plus[axis] += 1;
    let mut val = 2.0 * beta * nuclear_vrr(ca, cb_plus, 0, pair, c_k, boys_vals);
    if cb[axis] > 0 {
        let mut cb_minus = cb;
        cb_minus[axis] -= 1;
        val -= (cb[axis] as f64) * nuclear_vrr(ca, cb_minus, 0, pair, c_k, boys_vals);
    }
    val
}

#[inline]
fn prim_ecp_deriv_a(
    ca: [u8; 3],
    cb: [u8; 3],
    alpha: f64,
    r_a: [f64; 3],
    beta: f64,
    r_b: [f64; 3],
    axis: usize,
    ecp: &AtomEcp,
) -> f64 {
    let mut ca_plus = ca;
    ca_plus[axis] += 1;
    let mut val = 2.0 * alpha * ecp_atom_primitive(ca_plus, cb, alpha, r_a, beta, r_b, ecp);
    if ca[axis] > 0 {
        let mut ca_minus = ca;
        ca_minus[axis] -= 1;
        val -= (ca[axis] as f64) * ecp_atom_primitive(ca_minus, cb, alpha, r_a, beta, r_b, ecp);
    }
    val
}

#[inline]
fn prim_ecp_deriv_b(
    ca: [u8; 3],
    cb: [u8; 3],
    alpha: f64,
    r_a: [f64; 3],
    beta: f64,
    r_b: [f64; 3],
    axis: usize,
    ecp: &AtomEcp,
) -> f64 {
    let mut cb_plus = cb;
    cb_plus[axis] += 1;
    let mut val = 2.0 * beta * ecp_atom_primitive(ca, cb_plus, alpha, r_a, beta, r_b, ecp);
    if cb[axis] > 0 {
        let mut cb_minus = cb;
        cb_minus[axis] -= 1;
        val -= (cb[axis] as f64) * ecp_atom_primitive(ca, cb_minus, alpha, r_a, beta, r_b, ecp);
    }
    val
}

/// Evaluates one-electron integral derivatives `(dS, dT, dV_eff_plus_ecp)` for differentiation of
/// atom `atom` along coordinate axis `axis` (0=x, 1=y, 2=z).
pub fn one_electron_derivative(
    mol: &Molecule,
    basis: &BasisSet,
    atom: usize,
    axis: usize,
) -> (Matrix2D, Matrix2D, Matrix2D) {
    let n = basis.n_ao;
    let mut ds = Matrix2D::zeros(n, n);
    let mut dt = Matrix2D::zeros(n, n);
    let mut dv = Matrix2D::zeros(n, n);

    for sh_a in &basis.shells {
        let comps_a = sh_a.components();
        let sh_a_is_atom = sh_a.atom_index == atom;

        for sh_b in &basis.shells {
            let comps_b = sh_b.components();
            let sh_b_is_atom = sh_b.atom_index == atom;

            // Maximum angular momentum needed for Boys function:
            // Since derivative raises angular momentum by 1, we need max_m = la + lb + 1.
            let max_m = (sh_a.shell_type.angular_momentum() + sh_b.shell_type.angular_momentum())
                as usize
                + 1;
            let mut boys_vals = vec![0.0; max_m + 1];

            for (ia, &ca) in comps_a.iter().enumerate() {
                let mu = sh_a.ao_offset + ia;
                for (ib, &cb) in comps_b.iter().enumerate() {
                    let nu = sh_b.ao_offset + ib;

                    let mut s_deriv = 0.0;
                    let mut t_deriv = 0.0;
                    let mut v_deriv = 0.0;

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

                            // Overlap derivative:
                            if sh_a_is_atom {
                                s_deriv += weight
                                    * prim_overlap_deriv_a(ca, cb, pa.exponent, axis, &pair);
                            }
                            if sh_b_is_atom {
                                s_deriv += weight
                                    * prim_overlap_deriv_b(ca, cb, pb.exponent, axis, &pair);
                            }

                            // Kinetic derivative:
                            if sh_a_is_atom {
                                t_deriv += weight
                                    * prim_kinetic_deriv_a(
                                        ca,
                                        cb,
                                        pa.exponent,
                                        pb.exponent,
                                        axis,
                                        &pair,
                                    );
                            }
                            if sh_b_is_atom {
                                t_deriv += weight
                                    * prim_kinetic_deriv_b(
                                        ca,
                                        cb,
                                        pa.exponent,
                                        pb.exponent,
                                        axis,
                                        &pair,
                                    );
                            }

                            // Nuclear attraction derivative (using effective core charge Z_eff):
                            for (k_idx, nuc) in mol.atoms.iter().enumerate() {
                                let k_is_atom = k_idx == atom;
                                if (sh_a_is_atom == k_is_atom) && (sh_b_is_atom == k_is_atom) {
                                    continue;
                                }
                                let c_k = nuc.position_bohr;
                                let z_k = basis.effective_nuclear_charge(mol, k_idx);
                                let pc2 = (pair.p[0] - c_k[0]).powi(2)
                                    + (pair.p[1] - c_k[1]).powi(2)
                                    + (pair.p[2] - c_k[2]).powi(2);
                                boys_batch(max_m, pair.gamma * pc2, &mut boys_vals);

                                let da = prim_nuclear_deriv_a(
                                    ca,
                                    cb,
                                    pa.exponent,
                                    axis,
                                    &pair,
                                    c_k,
                                    &boys_vals,
                                );
                                let db = prim_nuclear_deriv_b(
                                    ca,
                                    cb,
                                    pb.exponent,
                                    axis,
                                    &pair,
                                    c_k,
                                    &boys_vals,
                                );

                                if sh_a_is_atom {
                                    v_deriv -= z_k * weight * da;
                                }
                                if sh_b_is_atom {
                                    v_deriv -= z_k * weight * db;
                                }
                                if k_is_atom {
                                    v_deriv += z_k * weight * (da + db);
                                }
                            }

                            // Scalar ECP derivative (Type-1 local + Type-2 semilocal):
                            for ecp in &basis.ecp_potentials {
                                let c_is_atom = ecp.atom_index == atom;
                                if (sh_a_is_atom == c_is_atom) && (sh_b_is_atom == c_is_atom) {
                                    continue;
                                }
                                let ecp_da = prim_ecp_deriv_a(
                                    ca,
                                    cb,
                                    pa.exponent,
                                    sh_a.center,
                                    pb.exponent,
                                    sh_b.center,
                                    axis,
                                    ecp,
                                );
                                let ecp_db = prim_ecp_deriv_b(
                                    ca,
                                    cb,
                                    pa.exponent,
                                    sh_a.center,
                                    pb.exponent,
                                    sh_b.center,
                                    axis,
                                    ecp,
                                );
                                if sh_a_is_atom {
                                    v_deriv += weight * ecp_da;
                                }
                                if sh_b_is_atom {
                                    v_deriv += weight * ecp_db;
                                }
                                if c_is_atom {
                                    v_deriv -= weight * (ecp_da + ecp_db);
                                }
                            }
                        }
                    }

                    if sh_a_is_atom || sh_b_is_atom {
                        ds.set(mu, nu, s_deriv);
                        dt.set(mu, nu, t_deriv);
                    }
                    dv.set(mu, nu, v_deriv);
                }
            }
        }
    }

    (ds, dt, dv)
}

/// Evaluates the contracted ERI derivatives for quartet `(sh_a sh_b | sh_c sh_d)`
/// differentiated with respect to the coordinate of atom `atom` along `axis`.
///
/// Output length is `na * nb * nc * nd` in row-major layout `((ia * nb + ib) * nc + ic) * nd + id`.
pub fn eri_derivative_quartet(shells: [&Shell; 4], atom: usize, axis: usize) -> Vec<f64> {
    let [sh_a, sh_b, sh_c, sh_d] = shells;
    let comps_a = sh_a.components();
    let comps_b = sh_b.components();
    let comps_c = sh_c.components();
    let comps_d = sh_d.components();
    let na = comps_a.len();
    let nb = comps_b.len();
    let nc = comps_c.len();
    let nd = comps_d.len();
    let total_len = na * nb * nc * nd;

    let sh_a_is_atom = sh_a.atom_index == atom;
    let sh_b_is_atom = sh_b.atom_index == atom;
    let sh_c_is_atom = sh_c.atom_index == atom;
    let sh_d_is_atom = sh_d.atom_index == atom;

    if !sh_a_is_atom && !sh_b_is_atom && !sh_c_is_atom && !sh_d_is_atom {
        return vec![0.0; total_len];
    }

    let max_ab = (sh_a.shell_type.angular_momentum() + sh_b.shell_type.angular_momentum()) as usize
        + if sh_a_is_atom || sh_b_is_atom { 1 } else { 0 };
    let max_cd = (sh_c.shell_type.angular_momentum() + sh_d.shell_type.angular_momentum()) as usize
        + if sh_c_is_atom || sh_d_is_atom { 1 } else { 0 };

    let mut out = vec![0.0; total_len];

    for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
        for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
            for (pc_idx, pc) in sh_c.primitives.iter().enumerate() {
                for (pd_idx, pd) in sh_d.primitives.iter().enumerate() {
                    let c_abcd = pa.coeff * pb.coeff * pc.coeff * pd.coeff;
                    let mut rec = EriPairData::new(
                        pa.exponent,
                        sh_a.center,
                        pb.exponent,
                        sh_b.center,
                        pc.exponent,
                        sh_c.center,
                        pd.exponent,
                        sh_d.center,
                        max_ab,
                        max_cd,
                    );

                    for (ia, &ca) in comps_a.iter().enumerate() {
                        let norm_a = sh_a.primitive_norm_for(pa_idx, ca);
                        for (ib, &cb) in comps_b.iter().enumerate() {
                            let norm_ab = norm_a * sh_b.primitive_norm_for(pb_idx, cb);
                            for (ic, &cc) in comps_c.iter().enumerate() {
                                let norm_abc = norm_ab * sh_c.primitive_norm_for(pc_idx, cc);
                                for (id, &cd) in comps_d.iter().enumerate() {
                                    let weight =
                                        c_abcd * norm_abc * sh_d.primitive_norm_for(pd_idx, cd);
                                    let idx = ((ia * nb + ib) * nc + ic) * nd + id;

                                    let mut d_eri = 0.0;

                                    // Center A derivative
                                    if sh_a_is_atom {
                                        let mut ca_plus = ca;
                                        ca_plus[axis] += 1;
                                        let eri_plus = rec.hrr_bra(ca_plus, cb, cc, cd);
                                        d_eri += 2.0 * pa.exponent * eri_plus;
                                        if ca[axis] > 0 {
                                            let mut ca_minus = ca;
                                            ca_minus[axis] -= 1;
                                            let eri_minus = rec.hrr_bra(ca_minus, cb, cc, cd);
                                            d_eri -= (ca[axis] as f64) * eri_minus;
                                        }
                                    }

                                    // Center B derivative
                                    if sh_b_is_atom {
                                        let mut cb_plus = cb;
                                        cb_plus[axis] += 1;
                                        let eri_plus = rec.hrr_bra(ca, cb_plus, cc, cd);
                                        d_eri += 2.0 * pb.exponent * eri_plus;
                                        if cb[axis] > 0 {
                                            let mut cb_minus = cb;
                                            cb_minus[axis] -= 1;
                                            let eri_minus = rec.hrr_bra(ca, cb_minus, cc, cd);
                                            d_eri -= (cb[axis] as f64) * eri_minus;
                                        }
                                    }

                                    // Center C derivative
                                    if sh_c_is_atom {
                                        let mut cc_plus = cc;
                                        cc_plus[axis] += 1;
                                        let eri_plus = rec.hrr_bra(ca, cb, cc_plus, cd);
                                        d_eri += 2.0 * pc.exponent * eri_plus;
                                        if cc[axis] > 0 {
                                            let mut cc_minus = cc;
                                            cc_minus[axis] -= 1;
                                            let eri_minus = rec.hrr_bra(ca, cb, cc_minus, cd);
                                            d_eri -= (cc[axis] as f64) * eri_minus;
                                        }
                                    }

                                    // Center D derivative
                                    if sh_d_is_atom {
                                        let mut cd_plus = cd;
                                        cd_plus[axis] += 1;
                                        let eri_plus = rec.hrr_bra(ca, cb, cc, cd_plus);
                                        d_eri += 2.0 * pd.exponent * eri_plus;
                                        if cd[axis] > 0 {
                                            let mut cd_minus = cd;
                                            cd_minus[axis] -= 1;
                                            let eri_minus = rec.hrr_bra(ca, cb, cc, cd_minus);
                                            d_eri -= (cd[axis] as f64) * eri_minus;
                                        }
                                    }

                                    out[idx] += weight * d_eri;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    out
}
