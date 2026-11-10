//! Exact quantum-mechanical solute potential and reaction field operator evaluation.

use calchem_basis::BasisSet;
use calchem_int::boys::boys_batch;
use calchem_int::primitive::{nuclear_vrr, GaussianPair};
use calchem_model::{CalcError, Molecule};
use calchem_tensor::Matrix2D;
use rayon::prelude::*;

use crate::cavity::MolecularCavity;

/// Precomputed cavity potential operator engine.
///
/// Stores nuclear potentials $V_k^{\text{nuc}}$ and the 3-index one-electron potential
/// tensor $V_{k, \mu\nu} = \langle \mu | \frac{1}{\|\mathbf{r} - \mathbf{s}_k\|} | \nu \rangle$.
#[derive(Debug, Clone)]
pub struct CavityPotentialEngine {
    pub n_tesserae: usize,
    pub n_ao: usize,
    /// Nuclear electrostatic potential at each tessera (Hartree / e).
    pub v_nuc: Vec<f64>,
    /// Flattened 3-index tensor of size `n_tesserae * n_ao * n_ao`:
    /// `v_tessera_ao[k * n_ao * n_ao + mu * n_ao + nu]`.
    pub v_tessera_ao: Vec<f64>,
}

impl CavityPotentialEngine {
    /// Builds the potential engine for a given molecule, basis set, and cavity.
    pub fn build(mol: &Molecule, basis: &BasisSet, cavity: &MolecularCavity) -> Result<Self, CalcError> {
        let n_tesserae = cavity.n_tesserae();
        let n_ao = basis.n_ao;

        if n_tesserae == 0 {
            return Err(CalcError::InvalidMolecule("Empty cavity tesserae".into()));
        }

        // 1. Nuclear potential at each tessera: V_k^nuc = sum_A Z_eff(A) / |s_k - R_A|
        let nuclei: Vec<([f64; 3], f64)> = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (a.position_bohr, basis.effective_nuclear_charge(mol, i)))
            .collect();

        let v_nuc: Vec<f64> = cavity
            .tesserae
            .iter()
            .map(|t| {
                let s = t.position_bohr;
                let mut v = 0.0;
                for &(r_a, z_a) in &nuclei {
                    let d = ((s[0] - r_a[0]).powi(2)
                        + (s[1] - r_a[1]).powi(2)
                        + (s[2] - r_a[2]).powi(2))
                    .sqrt()
                    .max(1e-10);
                    v += z_a / d;
                }
                v
            })
            .collect();

        // 2. Precompute maximum angular momentum across basis shells
        let max_l = basis
            .shells
            .iter()
            .map(|sh| sh.shell_type.angular_momentum() as usize)
            .max()
            .unwrap_or(0);
        let max_m = 2 * max_l;

        // 3. Evaluate one-electron potential integrals V_{k, mu nu} in parallel across tesserae
        let v_tessera_ao_blocks: Vec<Vec<f64>> = (0..n_tesserae)
            .into_par_iter()
            .map(|k| {
                let s_k = cavity.tesserae[k].position_bohr;
                let mut blk = vec![0.0f64; n_ao * n_ao];
                let mut boys_buf = vec![0.0f64; max_m + 1];

                for (ia_sh, sh_a) in basis.shells.iter().enumerate() {
                    let comps_a = sh_a.components();
                    let l_a = sh_a.shell_type.angular_momentum() as usize;

                    for sh_b in basis.shells.iter().skip(ia_sh) {
                        let comps_b = sh_b.components();
                        let l_b = sh_b.shell_type.angular_momentum() as usize;
                        let sh_max_m = l_a + l_b;
                        let same_shell = std::ptr::eq(sh_a, sh_b);

                        for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
                            for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
                                let pair = GaussianPair::new(
                                    pa.exponent,
                                    sh_a.center,
                                    pb.exponent,
                                    sh_b.center,
                                );
                                let pc2 = (pair.p[0] - s_k[0]).powi(2)
                                    + (pair.p[1] - s_k[1]).powi(2)
                                    + (pair.p[2] - s_k[2]).powi(2);

                                boys_batch(sh_max_m, pair.gamma * pc2, &mut boys_buf);

                                for (ia, &ca) in comps_a.iter().enumerate() {
                                    let mu = sh_a.ao_offset + ia;
                                    let norm_a = sh_a.primitive_norm_for(pa_idx, ca);

                                    for (ib, &cb) in comps_b.iter().enumerate() {
                                        let nu = sh_b.ao_offset + ib;
                                        let norm_b = sh_b.primitive_norm_for(pb_idx, cb);
                                        let weight = pa.coeff * pb.coeff * norm_a * norm_b;

                                        let val = nuclear_vrr(
                                            ca,
                                            cb,
                                            0,
                                            &pair,
                                            s_k,
                                            &boys_buf,
                                        );

                                        let term = weight * val;
                                        blk[mu * n_ao + nu] += term;
                                        if !same_shell {
                                            blk[nu * n_ao + mu] += term;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                blk
            })
            .collect();

        // Flatten blocks into single continuous array
        let mut v_tessera_ao = Vec::with_capacity(n_tesserae * n_ao * n_ao);
        for blk in v_tessera_ao_blocks {
            v_tessera_ao.extend_from_slice(&blk);
        }

        Ok(Self {
            n_tesserae,
            n_ao,
            v_nuc,
            v_tessera_ao,
        })
    }

    /// Evaluates the electronic potential $V_k^{\text{el}}[P] = \sum_{\mu\nu} P_{\mu\nu} V_{k, \mu\nu}$
    /// at all tesserae from the total density matrix `p`.
    pub fn compute_electronic_potential(&self, p: &Matrix2D) -> Vec<f64> {
        let n_ao = self.n_ao;
        let n_tess = self.n_tesserae;
        let p_data = &p.data;

        let mut v_el = vec![0.0f64; n_tess];
        let n2 = n_ao * n_ao;

        for k in 0..n_tess {
            let offset = k * n2;
            let blk = &self.v_tessera_ao[offset..(offset + n2)];
            let mut sum = 0.0;
            for i in 0..n2 {
                sum += p_data[i] * blk[i];
            }
            v_el[k] = sum;
        }

        v_el
    }

    /// Evaluates total solute electrostatic potential $V_k^{\text{solute}} = V_k^{\text{nuc}} - V_k^{\text{el}}[P]$.
    pub fn compute_solute_potential(&self, p: &Matrix2D) -> Vec<f64> {
        let v_el = self.compute_electronic_potential(p);
        self.v_nuc
            .iter()
            .zip(v_el.iter())
            .map(|(&vn, &ve)| vn - ve)
            .collect()
    }

    /// Builds the reaction field Fock matrix $V^{\text{RF}}_{\mu\nu} = -\sum_k q_k V_{k, \mu\nu}$.
    pub fn compute_reaction_field_matrix(&self, q: &[f64]) -> Matrix2D {
        let n_ao = self.n_ao;
        let n_tess = self.n_tesserae;
        let n2 = n_ao * n_ao;
        let mut v_rf_data = vec![0.0f64; n2];

        for k in 0..n_tess {
            let q_k = q[k];
            if q_k.abs() < 1e-18 {
                continue;
            }
            let offset = k * n2;
            let blk = &self.v_tessera_ao[offset..(offset + n2)];
            for i in 0..n2 {
                v_rf_data[i] -= q_k * blk[i];
            }
        }

        Matrix2D {
            rows: n_ao,
            cols: n_ao,
            data: v_rf_data,
        }
    }
}
