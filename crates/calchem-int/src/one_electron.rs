use crate::boys::boys_batch;
use crate::primitive::{kinetic_primitive, nuclear_vrr, overlap_primitive, GaussianPair};
use calchem_basis::{BasisSet, Shell};
use calchem_model::Molecule;
use calchem_tensor::Matrix2D;

pub fn shell_overlap_block(sh_a: &Shell, sh_b: &Shell) -> Matrix2D {
    let comps_a = sh_a.components();
    let comps_b = sh_b.components();
    let mut block = Matrix2D::zeros(comps_a.len(), comps_b.len());

    for (ia, &ca) in comps_a.iter().enumerate() {
        let a_i32 = ca.map(|v| v as i32);
        for (ib, &cb) in comps_b.iter().enumerate() {
            let b_i32 = cb.map(|v| v as i32);
            let mut sum = 0.0;
            for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
                let norm_a = sh_a.primitive_norm_for(pa_idx, ca);
                for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
                    let norm_b = sh_b.primitive_norm_for(pb_idx, cb);
                    let pair =
                        GaussianPair::new(pa.exponent, sh_a.center, pb.exponent, sh_b.center);
                    let s = overlap_primitive(a_i32, b_i32, &pair);
                    sum += pa.coeff * pb.coeff * norm_a * norm_b * s;
                }
            }
            block.set(ia, ib, sum);
        }
    }
    block
}

pub fn shell_kinetic_block(sh_a: &Shell, sh_b: &Shell) -> Matrix2D {
    let comps_a = sh_a.components();
    let comps_b = sh_b.components();
    let mut block = Matrix2D::zeros(comps_a.len(), comps_b.len());

    for (ia, &ca) in comps_a.iter().enumerate() {
        let a_i32 = ca.map(|v| v as i32);
        for (ib, &cb) in comps_b.iter().enumerate() {
            let b_i32 = cb.map(|v| v as i32);
            let mut sum = 0.0;
            for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
                let norm_a = sh_a.primitive_norm_for(pa_idx, ca);
                for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
                    let norm_b = sh_b.primitive_norm_for(pb_idx, cb);
                    let pair =
                        GaussianPair::new(pa.exponent, sh_a.center, pb.exponent, sh_b.center);
                    let t = kinetic_primitive(a_i32, b_i32, pb.exponent, &pair);
                    sum += pa.coeff * pb.coeff * norm_a * norm_b * t;
                }
            }
            block.set(ia, ib, sum);
        }
    }
    block
}

pub fn shell_nuclear_block(sh_a: &Shell, sh_b: &Shell, mol: &Molecule) -> Matrix2D {
    let charges: Vec<f64> = mol.atoms.iter().map(|a| a.z as f64).collect();
    shell_nuclear_block_with_charges(sh_a, sh_b, mol, &charges)
}

pub fn shell_nuclear_block_with_charges(
    sh_a: &Shell,
    sh_b: &Shell,
    mol: &Molecule,
    charges: &[f64],
) -> Matrix2D {
    let comps_a = sh_a.components();
    let comps_b = sh_b.components();
    let max_m = (sh_a.shell_type.angular_momentum() + sh_b.shell_type.angular_momentum()) as usize;
    let mut boys_vals = vec![0.0; max_m + 1];
    let mut block = Matrix2D::zeros(comps_a.len(), comps_b.len());

    for (ia, &ca) in comps_a.iter().enumerate() {
        for (ib, &cb) in comps_b.iter().enumerate() {
            let mut sum = 0.0;
            for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
                let norm_a = sh_a.primitive_norm_for(pa_idx, ca);
                for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
                    let norm_b = sh_b.primitive_norm_for(pb_idx, cb);
                    let pair =
                        GaussianPair::new(pa.exponent, sh_a.center, pb.exponent, sh_b.center);
                    let weight = pa.coeff * pb.coeff * norm_a * norm_b;
                    for (atom, &z_eff) in mol.atoms.iter().zip(charges.iter()) {
                        let c = atom.position_bohr;
                        let pc2 = (pair.p[0] - c[0]).powi(2)
                            + (pair.p[1] - c[1]).powi(2)
                            + (pair.p[2] - c[2]).powi(2);
                        boys_batch(max_m, pair.gamma * pc2, &mut boys_vals);
                        let v = nuclear_vrr(ca, cb, 0, &pair, c, &boys_vals);
                        sum -= z_eff * weight * v;
                    }
                }
            }
            block.set(ia, ib, sum);
        }
    }
    block
}

pub fn compute_overlap(basis: &BasisSet) -> Matrix2D {
    let mut s_mat = Matrix2D::zeros(basis.n_ao, basis.n_ao);
    for sh_a in &basis.shells {
        for sh_b in &basis.shells {
            let block = shell_overlap_block(sh_a, sh_b);
            for ia in 0..block.rows {
                for ib in 0..block.cols {
                    s_mat.set(sh_a.ao_offset + ia, sh_b.ao_offset + ib, block.get(ia, ib));
                }
            }
        }
    }
    s_mat
}

pub fn compute_kinetic(basis: &BasisSet) -> Matrix2D {
    let mut t_mat = Matrix2D::zeros(basis.n_ao, basis.n_ao);
    for sh_a in &basis.shells {
        for sh_b in &basis.shells {
            let block = shell_kinetic_block(sh_a, sh_b);
            for ia in 0..block.rows {
                for ib in 0..block.cols {
                    t_mat.set(sh_a.ao_offset + ia, sh_b.ao_offset + ib, block.get(ia, ib));
                }
            }
        }
    }
    t_mat
}

pub fn compute_nuclear(basis: &BasisSet, mol: &Molecule) -> Matrix2D {
    let charges: Vec<f64> = (0..mol.atoms.len())
        .map(|i| basis.effective_nuclear_charge(mol, i))
        .collect();
    let mut v_mat = Matrix2D::zeros(basis.n_ao, basis.n_ao);
    for sh_a in &basis.shells {
        for sh_b in &basis.shells {
            let block = shell_nuclear_block_with_charges(sh_a, sh_b, mol, &charges);
            for ia in 0..block.rows {
                for ib in 0..block.cols {
                    v_mat.set(sh_a.ao_offset + ia, sh_b.ao_offset + ib, block.get(ia, ib));
                }
            }
        }
    }
    v_mat
}
