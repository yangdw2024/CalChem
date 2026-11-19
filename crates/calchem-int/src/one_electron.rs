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
    let na = comps_a.len();
    let nb = comps_b.len();
    let max_m = (sh_a.shell_type.angular_momentum() + sh_b.shell_type.angular_momentum()) as usize;
    let mut boys_vals = [0.0f64; 16];
    let mut block = Matrix2D::zeros(na, nb);

    for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
        let mut wa = [0.0f64; 10];
        for (ia, &ca) in comps_a.iter().enumerate() {
            wa[ia] = pa.coeff * sh_a.primitive_norm_for(pa_idx, ca);
        }
        for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
            let pair = GaussianPair::new(pa.exponent, sh_a.center, pb.exponent, sh_b.center);
            if pair.kab < 1e-18 {
                continue;
            }
            let mut wb = [0.0f64; 10];
            for (ib, &cb) in comps_b.iter().enumerate() {
                wb[ib] = pb.coeff * sh_b.primitive_norm_for(pb_idx, cb);
            }
            for (atom, &z_eff) in mol.atoms.iter().zip(charges.iter()) {
                if z_eff == 0.0 {
                    continue;
                }
                let c = atom.position_bohr;
                let dx = pair.p[0] - c[0];
                let dy = pair.p[1] - c[1];
                let dz = pair.p[2] - c[2];
                let pc2 = dx * dx + dy * dy + dz * dz;
                boys_batch(max_m, pair.gamma * pc2, &mut boys_vals[..=max_m]);
                for (ia, &ca) in comps_a.iter().enumerate() {
                    let z_wa = z_eff * wa[ia];
                    for (ib, &cb) in comps_b.iter().enumerate() {
                        let v = nuclear_vrr(ca, cb, 0, &pair, c, &boys_vals[..=max_m]);
                        block.add_assign_at(ia, ib, -z_wa * wb[ib] * v);
                    }
                }
            }
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
    use rayon::prelude::*;
    let rows: Vec<_> = basis.shells.par_iter().enumerate().map(|(i, a)| {
        basis.shells[..=i].iter().map(|b| shell_nuclear_block_with_charges(a,b,mol,&charges))
            .collect::<Vec<_>>()
    }).collect();
    for (i,row) in rows.into_iter().enumerate() {
        for (j,block) in row.into_iter().enumerate() {
            for ia in 0..block.rows {
                for ib in 0..block.cols {
                    let u = basis.shells[i].ao_offset+ia;
                    let v = basis.shells[j].ao_offset+ib;
                    v_mat.set(u,v,block.get(ia,ib));
                    v_mat.set(v,u,block.get(ia,ib));
                }
            }
        }
    }
    v_mat
}
