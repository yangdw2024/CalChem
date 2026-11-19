//! Density Fitting (Resolution-of-the-Identity, RI-J) Coulomb Engine.
//!
//! Replaces four-center two-electron Coulomb integral evaluation:
//!     J_μν = ∑_{λσ} (μν|λσ) P_λσ
//! with three-center and two-center contractions:
//!     v_P = ∑_{μν} (P|μν) P_μν
//!     x_P = ∑_Q (P|Q)^{-1} v_Q
//!     J_μν = ∑_P (P|μν) x_P
//!
//! Reduces computational scaling from O(N_ao^4) to O(N_aux * N_ao^2) with
//! zero four-center ERI memory footprint.

use crate::boys::boys_batch;
use crate::primitive::GaussianPair;
use crate::screening::shell_pair_bounds;
use calchem_basis::{load_def2_jfit_for_element, primitive_norm, BasisSet, Shell, ShellType};
use calchem_model::{CalcError, Molecule};
use calchem_tensor::{Matrix2D, SpdFactor};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Auxiliary Gaussian basis shell used for density fitting.
#[derive(Debug, Clone)]
pub struct AuxiliaryShell {
    pub atom_index: usize,
    pub center: [f64; 3],
    pub shell_type: ShellType,
    pub exponents: Vec<f64>,
    pub coeffs: Vec<f64>,
    pub ao_offset: usize,
}

impl AuxiliaryShell {
    pub fn n_functions(&self) -> usize {
        self.shell_type.n_ao()
    }

    pub fn components(&self) -> &'static [[u8; 3]] {
        self.shell_type.components()
    }
}

/// Auxiliary basis set container for RI-J density fitting.
#[derive(Debug, Clone)]
pub struct AuxiliaryBasisSet {
    pub name: String,
    pub shells: Vec<AuxiliaryShell>,
    pub n_aux: usize,
}

impl AuxiliaryBasisSet {
    /// Constructs an uncontracted even-tempered RI-J auxiliary basis set
    /// optimized for Coulomb expansion from the primary orbital basis set.
    ///
    /// For each atom, spans the exponent range of orbital shell pairs
    /// [alpha_min, 2 * alpha_max] with geometric ratio 2.0.
    /// Angular coverage uses the explicit valence/steep heuristics below;
    /// it is not a complete product basis and has no universal fitting error bound.
    pub fn build_auto_jfit(mol: &Molecule, primary: &BasisSet) -> Self {
        let mut shells = Vec::new();
        let mut ao_offset = 0;

        for (atom_idx, _atom) in mol.atoms.iter().enumerate() {
            let atom_shells: Vec<&Shell> = primary
                .shells
                .iter()
                .filter(|s| s.atom_index == atom_idx)
                .collect();
            if atom_shells.is_empty() {
                continue;
            }

            let center = atom_shells[0].center;
            let mut min_exp = f64::INFINITY;
            let mut max_exp = 0.0_f64;
            let mut max_l: u8 = 0;

            for s in &atom_shells {
                let l = s.shell_type.angular_momentum();
                if l > max_l {
                    max_l = l;
                }
                for p in &s.primitives {
                    if p.exponent < min_exp {
                        min_exp = p.exponent;
                    }
                    if p.exponent > max_exp {
                        max_exp = p.exponent;
                    }
                }
            }

            if !min_exp.is_finite() || min_exp <= 0.0 {
                min_exp = 0.1;
            }
            if max_exp <= min_exp {
                max_exp = min_exp * 5.0;
            }

            // Target exponent span for auxiliary fitting functions:
            // Orbital products phi_mu * phi_nu have exponents alpha_mu + alpha_nu in [2*min_exp, 2*max_exp].
            // Include the diffuse tail of inter-center products. The metric eigensolver
            // removes near-dependent directions if needed.
            let fit_min = min_exp * 1.5;
            let fit_max = max_exp * 2.0;

            // Generate geometric progression of exponents:
            // Compact stride for large molecules with S/P primary bases keeps N_aux bounded.
            let compact_large = mol.atoms.len() > 30 && max_l <= 1;
            let ratio = if compact_large {
                2.45_f64.max((fit_max / fit_min).powf(1.0 / 9.0))
            } else {
                1.95_f64.max((fit_max / fit_min).powf(1.0 / 13.0))
            };
            let mut exps = Vec::new();
            let mut cur = fit_min;
            while cur <= fit_max * 1.0001 {
                exps.push(cur);
                cur *= ratio;
            }
            if exps.len() < 3 {
                exps = vec![fit_min, (fit_min * fit_max).sqrt(), fit_max];
            }

            // Angular momentum coverage:
            // Core s-only above 80.0; P covers 2p/3p and 1s*2p up to 80.0;
            // Cartesian D covers 2p*2p and polarization products in [0.25, 45.0] ([0.4, 8.0] for large SP);
            // Cartesian F covers d*p/d*d polarization products in [0.5, 8.0].
            for &exp in &exps {
                let max_l_for_exp = if exp > 80.0 {
                    0
                } else if compact_large {
                    if max_l >= 1 && (0.4..=8.0).contains(&exp) {
                        2
                    } else {
                        1.min(max_l + 1)
                    }
                } else if max_l >= 2 && (0.5..=8.0).contains(&exp) {
                    3
                } else if max_l >= 1 && (0.25..=45.0).contains(&exp) {
                    2
                } else {
                    1.min(max_l + 1)
                };

                for l in 0..=max_l_for_exp {
                    let (st, l_exp) = match l {
                        0 => (ShellType::S, exp),
                        1 => (ShellType::P, exp * 1.25),
                        2 => (ShellType::D, exp * 1.65),
                        _ => (ShellType::F, exp * 1.65),
                    };
                    shells.push(AuxiliaryShell {
                        atom_index: atom_idx,
                        center,
                        shell_type: st,
                        exponents: vec![l_exp],
                        coeffs: vec![1.0],
                        ao_offset,
                    });
                    ao_offset += st.n_ao();
                }
            }
        }

        Self {
            name: format!("{}-auto-jfit", primary.name),
            shells,
            n_aux: ao_offset,
        }
    }

    /// Constructs the contracted MolSSI BSE `def2-universal-jfit` (Weigend 2006) auxiliary
    /// basis set up to F-shells (`l <= 3`), falling back to `auto-jfit` for unsupported elements.
    pub fn build_def2_jfit(mol: &Molecule, primary: &BasisSet) -> Self {
        let mut shells = Vec::new();
        let mut ao_offset = 0;
        let mut used_fallback = false;

        for (atom_idx, atom) in mol.atoms.iter().enumerate() {
            if let Some(atom_jfit) = load_def2_jfit_for_element(atom.z as u8) {
                for (st, exps, coeffs) in atom_jfit {
                    shells.push(AuxiliaryShell {
                        atom_index: atom_idx,
                        center: atom.position_bohr,
                        shell_type: st,
                        exponents: exps,
                        coeffs,
                        ao_offset,
                    });
                    ao_offset += st.n_ao();
                }
            } else {
                used_fallback = true;
                let single_mol = Molecule::new(vec![atom.clone()], 0, 1);
                let atom_shells: Vec<Shell> = primary
                    .shells
                    .iter()
                    .filter(|s| s.atom_index == atom_idx)
                    .cloned()
                    .map(|mut s| {
                        s.atom_index = 0;
                        s
                    })
                    .collect();
                if !atom_shells.is_empty() {
                    let sub_bs = BasisSet {
                        name: primary.name.clone(),
                        shells: atom_shells,
                        n_ao: 0,
                        core_electrons: Vec::new(),
                        ecp_potentials: Vec::new(),
                    };
                    let fb = Self::build_auto_jfit(&single_mol, &sub_bs);
                    for mut sh in fb.shells {
                        sh.atom_index = atom_idx;
                        sh.ao_offset = ao_offset;
                        ao_offset += sh.n_functions();
                        shells.push(sh);
                    }
                }
            }
        }

        let name = if used_fallback {
            format!("{}-def2-jfit-hybrid", primary.name)
        } else {
            format!("{}-def2-jfit", primary.name)
        };
        Self {
            name,
            shells,
            n_aux: ao_offset,
        }
    }
}

// ---------------------------------------------------------------------------
// 2-Center Coulomb Metric: (P | Q)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Eri2CenterData {
    p: f64,
    q: f64,
    zeta: f64,
    rho: f64,
    wp: [f64; 3],
    wq: [f64; 3],
    prefactor: f64,
    boys: Vec<f64>,
    max_p: usize,
    max_q: usize,
    max_m: usize,
    vrr_cache: Vec<f64>,
}

impl Eri2CenterData {
    fn new(
        gamma_p: f64,
        r_p: [f64; 3],
        gamma_q: f64,
        r_q: [f64; 3],
        max_p: usize,
        max_q: usize,
    ) -> Self {
        let p = gamma_p;
        let q = gamma_q;
        let zeta = p + q;
        let rho = p * q / zeta;
        let w = [
            (p * r_p[0] + q * r_q[0]) / zeta,
            (p * r_p[1] + q * r_q[1]) / zeta,
            (p * r_p[2] + q * r_q[2]) / zeta,
        ];
        let wp = [w[0] - r_p[0], w[1] - r_p[1], w[2] - r_p[2]];
        let wq = [w[0] - r_q[0], w[1] - r_q[1], w[2] - r_q[2]];
        let pq2 = (r_p[0] - r_q[0]).powi(2) + (r_p[1] - r_q[1]).powi(2) + (r_p[2] - r_q[2]).powi(2);
        let max_m = max_p + max_q;
        let mut boys = vec![0.0; max_m + 1];
        boys_batch(max_m, rho * pq2, &mut boys);
        let prefactor = 2.0 * PI.powf(2.5) / (p * q * zeta.sqrt());

        let dp = max_p + 1;
        let dq = max_q + 1;
        let dm = max_m + 1;
        let cache_len = if max_m == 0 {
            0
        } else {
            dp * dp * dp * dq * dq * dq * dm
        };

        Self {
            p,
            q,
            zeta,
            rho,
            wp,
            wq,
            prefactor,
            boys,
            max_p,
            max_q,
            max_m,
            vrr_cache: vec![f64::NAN; cache_len],
        }
    }

    #[inline]
    fn vrr_index(&self, p: [u8; 3], q: [u8; 3], m: usize) -> usize {
        let dp = self.max_p + 1;
        let dq = self.max_q + 1;
        let dm = self.max_m + 1;
        let ip = ((p[0] as usize) * dp + (p[1] as usize)) * dp + (p[2] as usize);
        let iq = ((q[0] as usize) * dq + (q[1] as usize)) * dq + (q[2] as usize);
        (ip * (dq * dq * dq) + iq) * dm + m
    }

    fn vrr(&mut self, p: [u8; 3], q: [u8; 3], m: usize) -> f64 {
        if p == [0, 0, 0] && q == [0, 0, 0] {
            return self.prefactor * self.boys[m];
        }

        let idx = self.vrr_index(p, q, m);
        let cached = self.vrr_cache[idx];
        if !cached.is_nan() {
            return cached;
        }

        let p_exp = self.p;
        let q_exp = self.q;
        let zeta = self.zeta;
        let rho = self.rho;
        let wp = self.wp;
        let wq = self.wq;

        for axis in 0..3 {
            if p[axis] > 0 {
                let mut p1 = p;
                p1[axis] -= 1;
                let v_m1 = self.vrr(p1, q, m + 1);
                let mut val = wp[axis] * v_m1;

                if p1[axis] > 0 {
                    let mut p2 = p1;
                    p2[axis] -= 1;
                    let v2_m = self.vrr(p2, q, m);
                    let v2_m1 = self.vrr(p2, q, m + 1);
                    val += (p1[axis] as f64) / (2.0 * p_exp) * (v2_m - (rho / p_exp) * v2_m1);
                }

                if q[axis] > 0 {
                    let mut q1 = q;
                    q1[axis] -= 1;
                    let vq_m1 = self.vrr(p1, q1, m + 1);
                    val += (q[axis] as f64) / (2.0 * zeta) * vq_m1;
                }

                self.vrr_cache[idx] = val;
                return val;
            }
        }

        for axis in 0..3 {
            if q[axis] > 0 {
                let mut q1 = q;
                q1[axis] -= 1;
                let v_m1 = self.vrr(p, q1, m + 1);
                let mut val = wq[axis] * v_m1;

                if q1[axis] > 0 {
                    let mut q2 = q1;
                    q2[axis] -= 1;
                    let v2_m = self.vrr(p, q2, m);
                    let v2_m1 = self.vrr(p, q2, m + 1);
                    val += (q1[axis] as f64) / (2.0 * q_exp) * (v2_m - (rho / q_exp) * v2_m1);
                }

                self.vrr_cache[idx] = val;
                return val;
            }
        }

        unreachable!()
    }
}

pub fn eri_2center_primitive(
    p_lmn: [u8; 3],
    q_lmn: [u8; 3],
    gamma_p: f64,
    r_p: [f64; 3],
    gamma_q: f64,
    r_q: [f64; 3],
) -> f64 {
    let max_p = (p_lmn[0] + p_lmn[1] + p_lmn[2]) as usize;
    let max_q = (q_lmn[0] + q_lmn[1] + q_lmn[2]) as usize;
    let mut data = Eri2CenterData::new(gamma_p, r_p, gamma_q, r_q, max_p, max_q);
    data.vrr(p_lmn, q_lmn, 0)
}

/// Evaluates the 2-center Coulomb metric block (P | Q) between two auxiliary shells.
pub fn shell_2center_block(p_shell: &AuxiliaryShell, q_shell: &AuxiliaryShell) -> Matrix2D {
    let np = p_shell.n_functions();
    let nq = q_shell.n_functions();
    let mut block = Matrix2D::zeros(np, nq);
    let p_comps = p_shell.components();
    let q_comps = q_shell.components();

    for (p_idx, &p_lmn) in p_comps.iter().enumerate() {
        for (q_idx, &q_lmn) in q_comps.iter().enumerate() {
            let mut sum = 0.0;
            for (p_prim, &gamma_p) in p_shell.exponents.iter().enumerate() {
                let cp = p_shell.coeffs[p_prim] * primitive_norm(gamma_p, p_lmn);
                for (q_prim, &gamma_q) in q_shell.exponents.iter().enumerate() {
                    let cq = q_shell.coeffs[q_prim] * primitive_norm(gamma_q, q_lmn);
                    let prim_val = eri_2center_primitive(
                        p_lmn,
                        q_lmn,
                        gamma_p,
                        p_shell.center,
                        gamma_q,
                        q_shell.center,
                    );
                    sum += cp * cq * prim_val;
                }
            }
            block.set(p_idx, q_idx, sum);
        }
    }
    block
}

/// Computes the full 2-center Coulomb metric matrix V_PQ = (P | Q).
pub fn compute_2center_coulomb_metric(aux_basis: &AuxiliaryBasisSet) -> Matrix2D {
    let n_aux = aux_basis.n_aux;
    let n_shells = aux_basis.shells.len();

    let shell_pairs: Vec<(usize, usize)> = (0..n_shells)
        .flat_map(|p| (0..=p).map(move |q| (p, q)))
        .collect();

    let blocks: Vec<((usize, usize), Matrix2D)> = shell_pairs
        .into_par_iter()
        .map(|(p, q)| {
            let blk = shell_2center_block(&aux_basis.shells[p], &aux_basis.shells[q]);
            ((p, q), blk)
        })
        .collect();

    let mut v = Matrix2D::zeros(n_aux, n_aux);
    for ((p, q), blk) in blocks {
        let p_off = aux_basis.shells[p].ao_offset;
        let q_off = aux_basis.shells[q].ao_offset;
        let np = aux_basis.shells[p].n_functions();
        let nq = aux_basis.shells[q].n_functions();

        for i in 0..np {
            for j in 0..nq {
                let val = blk.get(i, j);
                v.set(p_off + i, q_off + j, val);
                v.set(q_off + j, p_off + i, val);
            }
        }
    }
    v
}

/// Inverts the symmetric positive-definite 2-center metric matrix V = (P | Q)
/// using spectral thresholding to handle near-linear dependencies cleanly.
pub fn invert_coulomb_metric(v: &Matrix2D) -> Result<Matrix2D, CalcError> {
    let (v_inv, _) = invert_coulomb_metric_with_rank_info(v)?;
    Ok(v_inv)
}

/// Inverts the 2-center Coulomb metric matrix V = (P | Q) and also returns the number
/// of truncated near-zero eigenmodes (`0` when V is full rank).
pub fn invert_coulomb_metric_with_rank_info(v: &Matrix2D) -> Result<(Matrix2D, usize), CalcError> {
    calchem_tensor::invert_spd_with_rank_info(v, 1e-12)
}

// ---------------------------------------------------------------------------
// 3-Center Coulomb Integrals: (P | μ ν)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Eri3CenterData {
    #[allow(dead_code)]
    pair_cd: GaussianPair,
    p: f64,
    q: f64,
    zeta: f64,
    rho: f64,
    qc: [f64; 3],
    cd: [f64; 3],
    wp: [f64; 3],
    wq: [f64; 3],
    prefactor: f64,
    boys: Vec<f64>,
    max_p: usize,
    max_ab: usize,
    max_m: usize,
    vrr_cache: Vec<f64>,
}

impl Eri3CenterData {
    fn new(
        gamma_p: f64,
        r_p: [f64; 3],
        alpha: f64,
        r_a: [f64; 3],
        beta: f64,
        r_b: [f64; 3],
        max_p: usize,
        max_ab: usize,
    ) -> Self {
        let pair_cd = GaussianPair::new(alpha, r_a, beta, r_b);
        let p = gamma_p;
        let q = pair_cd.gamma;
        let zeta = p + q;
        let rho = p * q / zeta;
        let w = [
            (p * r_p[0] + q * pair_cd.p[0]) / zeta,
            (p * r_p[1] + q * pair_cd.p[1]) / zeta,
            (p * r_p[2] + q * pair_cd.p[2]) / zeta,
        ];
        let wp = [w[0] - r_p[0], w[1] - r_p[1], w[2] - r_p[2]];
        let wq = [
            w[0] - pair_cd.p[0],
            w[1] - pair_cd.p[1],
            w[2] - pair_cd.p[2],
        ];
        let pq2 = (r_p[0] - pair_cd.p[0]).powi(2)
            + (r_p[1] - pair_cd.p[1]).powi(2)
            + (r_p[2] - pair_cd.p[2]).powi(2);
        let max_m = max_p + max_ab;
        let mut boys = vec![0.0; max_m + 1];
        boys_batch(max_m, rho * pq2, &mut boys);
        let prefactor = 2.0 * PI.powf(2.5) / (p * q * zeta.sqrt()) * pair_cd.kab;

        let dp = max_p + 1;
        let dab = max_ab + 1;
        let dm = max_m + 1;
        let cache_len = if max_m == 0 {
            0
        } else {
            dp * dp * dp * dab * dab * dab * dm
        };

        Self {
            qc: pair_cd.pa,
            cd: pair_cd.ab,
            pair_cd,
            p,
            q,
            zeta,
            rho,
            wp,
            wq,
            prefactor,
            boys,
            max_p,
            max_ab,
            max_m,
            vrr_cache: vec![f64::NAN; cache_len],
        }
    }

    #[inline]
    fn vrr_index(&self, p: [u8; 3], c: [u8; 3], m: usize) -> usize {
        let dp = self.max_p + 1;
        let dc = self.max_ab + 1;
        let dm = self.max_m + 1;
        let ip = ((p[0] as usize) * dp + (p[1] as usize)) * dp + (p[2] as usize);
        let ic = ((c[0] as usize) * dc + (c[1] as usize)) * dc + (c[2] as usize);
        (ip * (dc * dc * dc) + ic) * dm + m
    }

    fn vrr(&mut self, p: [u8; 3], c: [u8; 3], m: usize) -> f64 {
        if p == [0, 0, 0] && c == [0, 0, 0] {
            return self.prefactor * self.boys[m];
        }

        let idx = self.vrr_index(p, c, m);
        let cached = self.vrr_cache[idx];
        if !cached.is_nan() {
            return cached;
        }

        let p_exp = self.p;
        let q_exp = self.q;
        let zeta = self.zeta;
        let rho = self.rho;
        let wp = self.wp;
        let qc = self.qc;
        let wq = self.wq;

        for axis in 0..3 {
            if p[axis] > 0 {
                let mut p1 = p;
                p1[axis] -= 1;
                let v_m1 = self.vrr(p1, c, m + 1);
                let mut val = wp[axis] * v_m1;

                if p1[axis] > 0 {
                    let mut p2 = p1;
                    p2[axis] -= 1;
                    let v2_m = self.vrr(p2, c, m);
                    let v2_m1 = self.vrr(p2, c, m + 1);
                    val += (p1[axis] as f64) / (2.0 * p_exp) * (v2_m - (rho / p_exp) * v2_m1);
                }

                if c[axis] > 0 {
                    let mut c1 = c;
                    c1[axis] -= 1;
                    let vc_m1 = self.vrr(p1, c1, m + 1);
                    val += (c[axis] as f64) / (2.0 * zeta) * vc_m1;
                }

                self.vrr_cache[idx] = val;
                return val;
            }
        }

        for axis in 0..3 {
            if c[axis] > 0 {
                let mut c1 = c;
                c1[axis] -= 1;
                let v_m = self.vrr(p, c1, m);
                let v_m1 = self.vrr(p, c1, m + 1);
                let mut val = qc[axis] * v_m + wq[axis] * v_m1;

                if c1[axis] > 0 {
                    let mut c2 = c1;
                    c2[axis] -= 1;
                    let v2_m = self.vrr(p, c2, m);
                    let v2_m1 = self.vrr(p, c2, m + 1);
                    val += (c1[axis] as f64) / (2.0 * q_exp) * (v2_m - (rho / q_exp) * v2_m1);
                }

                if p[axis] > 0 {
                    let mut p1 = p;
                    p1[axis] -= 1;
                    let vp_m1 = self.vrr(p1, c1, m + 1);
                    val += (p[axis] as f64) / (2.0 * zeta) * vp_m1;
                }

                self.vrr_cache[idx] = val;
                return val;
            }
        }

        unreachable!()
    }

    fn hrr_ket(&mut self, p: [u8; 3], c: [u8; 3], d: [u8; 3]) -> f64 {
        for axis in 0..3 {
            if d[axis] > 0 {
                let mut d1 = d;
                d1[axis] -= 1;
                let mut c_plus = c;
                c_plus[axis] += 1;
                let cd_axis = self.cd[axis];
                let t1 = self.hrr_ket(p, c_plus, d1);
                let t2 = self.hrr_ket(p, c, d1);
                return t1 + cd_axis * t2;
            }
        }
        self.vrr(p, c, 0)
    }
}

pub fn eri_3center_primitive(
    p_lmn: [u8; 3],
    a_lmn: [u8; 3],
    b_lmn: [u8; 3],
    gamma_p: f64,
    r_p: [f64; 3],
    alpha: f64,
    r_a: [f64; 3],
    beta: f64,
    r_b: [f64; 3],
) -> f64 {
    let max_p = (p_lmn[0] + p_lmn[1] + p_lmn[2]) as usize;
    let max_ab = (a_lmn[0] + a_lmn[1] + a_lmn[2] + b_lmn[0] + b_lmn[1] + b_lmn[2]) as usize;
    let mut data = Eri3CenterData::new(gamma_p, r_p, alpha, r_a, beta, r_b, max_p, max_ab);
    data.hrr_ket(p_lmn, a_lmn, b_lmn)
}

/// Evaluates a 3-center contracted shell block: (P | i j) where P is an auxiliary shell
/// and (i, j) are orbital shells, writing into `out` (resized to `np * ni * nj`) in `[P][i][j]` order.
pub fn shell_3center_block_into(
    p_shell: &AuxiliaryShell,
    i_shell: &Shell,
    j_shell: &Shell,
    out: &mut Vec<f64>,
) {
    let np = p_shell.n_functions();
    let ni = i_shell.n_functions();
    let nj = j_shell.n_functions();
    out.resize(np * ni * nj, 0.0);
    out.fill(0.0);

    let p_comps = p_shell.components();
    let i_comps = i_shell.components();
    let j_comps = j_shell.components();
    let max_p = p_shell.shell_type.angular_momentum() as usize;
    let max_ab = i_shell.shell_type.angular_momentum() as usize
        + j_shell.shell_type.angular_momentum() as usize;

    for (p_prim, &gamma_p) in p_shell.exponents.iter().enumerate() {
        let p_c_base = p_shell.coeffs[p_prim];
        for (i_prim, prim_i) in i_shell.primitives.iter().enumerate() {
            for (j_prim, prim_j) in j_shell.primitives.iter().enumerate() {
                let mut data = Eri3CenterData::new(
                    gamma_p,
                    p_shell.center,
                    prim_i.exponent,
                    i_shell.center,
                    prim_j.exponent,
                    j_shell.center,
                    max_p,
                    max_ab,
                );
                if data.pair_cd.kab < 1e-18 {
                    continue;
                }
                if max_p == 0 && max_ab == 0 {
                    let val = data.prefactor * data.boys[0];
                    let cp = p_c_base * primitive_norm(gamma_p, p_comps[0]);
                    let ci = prim_i.coeff * i_shell.primitive_norm_for(i_prim, i_comps[0]);
                    let cj = prim_j.coeff * j_shell.primitive_norm_for(j_prim, j_comps[0]);
                    out[0] += cp * ci * cj * val;
                    continue;
                }
                for (p_idx, &p_lmn) in p_comps.iter().enumerate() {
                    let cp = p_c_base * primitive_norm(gamma_p, p_lmn);
                    for (i_idx, &i_lmn) in i_comps.iter().enumerate() {
                        let cpi = cp * prim_i.coeff * i_shell.primitive_norm_for(i_prim, i_lmn);
                        for (j_idx, &j_lmn) in j_comps.iter().enumerate() {
                            let cj = prim_j.coeff * j_shell.primitive_norm_for(j_prim, j_lmn);
                            let val = data.hrr_ket(p_lmn, i_lmn, j_lmn);
                            out[(p_idx * ni + i_idx) * nj + j_idx] += cpi * cj * val;
                        }
                    }
                }
            }
        }
    }
}

/// Evaluates a 3-center contracted shell block: (P | i j) where P is an auxiliary shell
/// and (i, j) are orbital shells. Returns a 3D block flattened as [P][i][j].
pub fn shell_3center_block(p_shell: &AuxiliaryShell, i_shell: &Shell, j_shell: &Shell) -> Vec<f64> {
    let mut out = Vec::new();
    shell_3center_block_into(p_shell, i_shell, j_shell, &mut out);
    out
}

// ---------------------------------------------------------------------------
// RI-J Coulomb Matrix Builder
// ---------------------------------------------------------------------------

/// RI-J Coulomb Engine managing auxiliary basis, metric inversion, and
/// fast O(N^3) Coulomb matrix accumulation with optional 3-center tensor caching
/// and Schwarz screening.
pub struct RiJEngine {
    pub use_gpu: bool,
    pub aux_basis: AuxiliaryBasisSet,
    pub metric_factor: SpdFactor,
    pub metric_truncated_modes: usize,
    pub cached_b: Option<Vec<f64>>,
    pub n_ao: usize,
    pub aux_diag_bounds: Vec<f64>,
    pub gpu_kernel_invocations: AtomicUsize,
    pub gpu_fallback_invocations: AtomicUsize,
}

impl RiJEngine {
    /// Builds the RI-J engine for a given molecule and primary basis.
    pub fn new(mol: &Molecule, primary: &BasisSet) -> Result<Self, CalcError> {
        let aux_basis = AuxiliaryBasisSet::build_auto_jfit(mol, primary);
        let budget = std::env::var("CALCHEM_MAX_MEMORY_BYTES")
            .ok()
            .and_then(|v| v.parse::<u128>().ok())
            .unwrap_or(8 * 1024 * 1024 * 1024);
        // Dense metric Cholesky/eigensolver working estimate in N_aux^2 f64s.
        let metric_bytes = 6 * (aux_basis.n_aux as u128).pow(2) * 8;
        if metric_bytes > budget {
            return Err(CalcError::UnsupportedBasis(format!(
                "RI-J metric workspace estimate {metric_bytes} bytes exceeds budget {budget}"
            )));
        }
        let v = compute_2center_coulomb_metric(&aux_basis);
        let metric_factor = SpdFactor::from_matrix(&v, 1e-12)?;
        let metric_truncated_modes = metric_factor.n_truncated;

        let n_ao = primary.n_ao;
        let n_aux = aux_basis.n_aux;
        let n_ao_sq = n_ao * n_ao;
        let req_bytes = n_aux
            .checked_mul(n_ao_sq)
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| CalcError::UnsupportedBasis("RI-J tensor dimensions overflow".into()))?;

        let max_df_bytes: usize = std::env::var("CALCHEM_MAX_DF_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2560 * 1024 * 1024); // 2.5 GiB default on 16 GiB workstations

        let aux_diag_bounds: Vec<f64> = aux_basis
            .shells
            .iter()
            .map(|p_sh| {
                let mut max_d = 0.0f64;
                for p_fn in 0..p_sh.n_functions() {
                    let p = p_sh.ao_offset + p_fn;
                    let v_diag = v.get(p, p).max(0.0);
                    if v_diag > max_d {
                        max_d = v_diag;
                    }
                }
                max_d.sqrt()
            })
            .collect();

        let cached_b = if req_bytes <= max_df_bytes {
            let primary_bounds = shell_pair_bounds(primary);
            let n_primary_shells = primary.shells.len();

            let mut b_flat = vec![0.0; n_aux * n_ao_sq];
            let mut remainder = b_flat.as_mut_slice();
            let mut slices = Vec::with_capacity(aux_basis.shells.len());
            for shell in &aux_basis.shells {
                let (rows, rest) = remainder.split_at_mut(shell.n_functions() * n_ao_sq);
                slices.push(rows);
                remainder = rest;
            }
            aux_basis
                .shells
                .par_iter()
                .zip(slices.into_par_iter())
                .enumerate()
                .for_each_init(Vec::new, |blk, (p_idx, (p_shell, shell_rows))| {
                    let np = p_shell.n_functions();
                    let p_b = aux_diag_bounds[p_idx];

                    for i_idx in 0..n_primary_shells {
                        let i_shell = &primary.shells[i_idx];
                        let i_off = i_shell.ao_offset;
                        let ni = i_shell.n_functions();

                        for j_idx in 0..=i_idx {
                            let q_ij = primary_bounds[i_idx * n_primary_shells + j_idx];
                            if p_b * q_ij < 1e-12 {
                                continue;
                            }

                            let j_shell = &primary.shells[j_idx];
                            let j_off = j_shell.ao_offset;
                            let nj = j_shell.n_functions();

                            shell_3center_block_into(p_shell, i_shell, j_shell, blk);
                            for p_fn in 0..np {
                                let p_row_offset = p_fn * n_ao_sq;
                                for mu in 0..ni {
                                    let r_mu = i_off + mu;
                                    for nu in 0..nj {
                                        let c_nu = j_off + nu;
                                        let val = blk[(p_fn * ni + mu) * nj + nu];
                                        shell_rows[p_row_offset + r_mu * n_ao + c_nu] = val;
                                        shell_rows[p_row_offset + c_nu * n_ao + r_mu] = val;
                                    }
                                }
                            }
                        }
                    }
                });

            Some(b_flat)
        } else {
            None
        };

        Ok(Self {
            use_gpu: false,
            aux_basis,
            metric_factor,
            metric_truncated_modes,
            cached_b,
            n_ao,
            aux_diag_bounds,
            gpu_kernel_invocations: AtomicUsize::new(0),
            gpu_fallback_invocations: AtomicUsize::new(0),
        })
    }

    /// Builds the RI-J engine using a caller-supplied custom auxiliary basis set.
    pub fn with_auxiliary_basis(
        primary: &BasisSet,
        aux_basis: AuxiliaryBasisSet,
    ) -> Result<Self, CalcError> {
        let v = compute_2center_coulomb_metric(&aux_basis);
        let metric_factor = SpdFactor::from_matrix(&v, 1e-12)?;
        let metric_truncated_modes = metric_factor.n_truncated;
        let n_ao = primary.n_ao;
        let aux_diag_bounds: Vec<f64> = aux_basis
            .shells
            .iter()
            .map(|p_sh| {
                let mut max_d = 0.0f64;
                for p_fn in 0..p_sh.n_functions() {
                    let p = p_sh.ao_offset + p_fn;
                    let v_diag = v.get(p, p).max(0.0);
                    if v_diag > max_d {
                        max_d = v_diag;
                    }
                }
                max_d.sqrt()
            })
            .collect();

        Ok(Self {
            use_gpu: false,
            aux_basis,
            metric_factor,
            metric_truncated_modes,
            cached_b: None,
            n_ao,
            aux_diag_bounds,
            gpu_kernel_invocations: AtomicUsize::new(0),
            gpu_fallback_invocations: AtomicUsize::new(0),
        })
    }

    /// Evaluates the fitted Coulomb matrix J_μν = ∑_P (P|μν) x_P
    /// using either precomputed cached tensor or screened parallel accumulation.
    pub fn build_coulomb(&self, primary: &BasisSet, density: &Matrix2D) -> Matrix2D {
        assert_eq!(
            primary.n_ao, self.n_ao,
            "RI-J orbital basis dimension mismatch"
        );
        assert_eq!(
            (density.rows, density.cols, density.data.len()),
            (self.n_ao, self.n_ao, self.n_ao * self.n_ao),
            "RI-J density dimension mismatch"
        );
        let n_ao = primary.n_ao;
        let n_aux = self.aux_basis.n_aux;
        let n_ao_sq = n_ao * n_ao;

        if let Some(ref b_tensor) = self.cached_b {
            // Fast O(N^3) GPU/parallel path with precomputed 3-center tensor
            let gpu_ready = self.use_gpu && calchem_cuda::is_cuda_available();

            // 1. v_P = ∑_{μν} B_{P, μν} P_{μν}
            let mut v_vec = vec![0.0; n_aux];
            let mut used_gpu_v = false;
            if gpu_ready {
                used_gpu_v = calchem_cuda::cuda_rij_contract_v(
                    b_tensor,
                    &density.data,
                    &mut v_vec,
                    n_aux,
                    n_ao_sq,
                );
                if used_gpu_v {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                }
            }
            if !used_gpu_v {
                v_vec = (0..n_aux)
                    .into_par_iter()
                    .map(|p| {
                        let b_slice = &b_tensor[p * n_ao_sq..(p + 1) * n_ao_sq];
                        let mut sum = 0.0;
                        for k in 0..n_ao_sq {
                            sum += b_slice[k] * density.data[k];
                        }
                        sum
                    })
                    .collect();
            }

            // 2. Solve V x = v via Cholesky factor (or pseudoinverse fallback)
            let x_vec = self.metric_factor.solve(&v_vec);

            // 3. J_{μν} = ∑_P x_P B_{P, μν}
            let mut j_data = vec![0.0; n_ao_sq];
            let mut used_gpu = false;
            if self.use_gpu {
                if gpu_ready {
                    used_gpu = calchem_cuda::cuda_rij_contract_coulomb(
                        b_tensor,
                        &x_vec,
                        &mut j_data,
                        n_aux,
                        n_ao_sq,
                    );
                }
                if used_gpu {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                }
            }

            if !used_gpu {
                let chunk_size = 512;
                j_data
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(c_idx, chunk)| {
                        let base = c_idx * chunk_size;
                        for (k, val) in chunk.iter_mut().enumerate() {
                            let idx = base + k;
                            let mut sum = 0.0;
                            for p in 0..n_aux {
                                sum += x_vec[p] * b_tensor[p * n_ao_sq + idx];
                            }
                            *val = sum;
                        }
                    });
            }

            return Matrix2D {
                rows: n_ao,
                cols: n_ao,
                data: j_data,
            };
        }

        // Screened parallel on-the-fly path for large systems
        if self.use_gpu {
            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
        }
        let n_primary_shells = primary.shells.len();
        let primary_bounds = shell_pair_bounds(primary);

        // 1. Pass 1: compute v_P = ∑_{μν} (P|μν) P_μν
        let v_vec: Vec<f64> = self
            .aux_basis
            .shells
            .par_iter()
            .enumerate()
            .map(|(p_idx, p_shell)| {
                let p_b = self.aux_diag_bounds[p_idx];
                let np = p_shell.n_functions();
                let mut v_local = vec![0.0; np];
                for i_idx in 0..n_primary_shells {
                    let i_shell = &primary.shells[i_idx];
                    let i_off = i_shell.ao_offset;
                    let ni = i_shell.n_functions();

                    for j_idx in 0..=i_idx {
                        let q_ij = primary_bounds[i_idx * n_primary_shells + j_idx];
                        if p_b * q_ij < 1e-12 {
                            continue;
                        }

                        let j_shell = &primary.shells[j_idx];
                        let j_off = j_shell.ao_offset;
                        let nj = j_shell.n_functions();
                        let deg = if i_idx == j_idx { 1.0 } else { 2.0 };

                        let blk = shell_3center_block(p_shell, i_shell, j_shell);
                        for p_fn in 0..np {
                            let mut pair_sum = 0.0;
                            for mu in 0..ni {
                                for nu in 0..nj {
                                    let d_val = density.get(i_off + mu, j_off + nu);
                                    let int_val = blk[(p_fn * ni + mu) * nj + nu];
                                    pair_sum += d_val * int_val;
                                }
                            }
                            v_local[p_fn] += deg * pair_sum;
                        }
                    }
                }
                v_local
            })
            .flatten()
            .collect();

        // 2. Solve V x = v via Cholesky factor (or pseudoinverse fallback)
        let x_vec = self.metric_factor.solve(&v_vec);

        // 3. Pass 2: accumulate J_μν = ∑_P (P|μν) x_P
        let primary_pairs: Vec<(usize, usize)> = (0..n_primary_shells)
            .flat_map(|i| (0..=i).map(move |j| (i, j)))
            .collect();

        let pair_contributions: Vec<((usize, usize), Vec<f64>)> = primary_pairs
            .into_par_iter()
            .map(|(i_idx, j_idx)| {
                let q_ij = primary_bounds[i_idx * n_primary_shells + j_idx];
                let i_shell = &primary.shells[i_idx];
                let j_shell = &primary.shells[j_idx];
                let ni = i_shell.n_functions();
                let nj = j_shell.n_functions();
                let mut j_blk = vec![0.0; ni * nj];

                for (p_idx, p_shell) in self.aux_basis.shells.iter().enumerate() {
                    let p_b = self.aux_diag_bounds[p_idx];
                    if p_b * q_ij < 1e-12 {
                        continue;
                    }

                    let p_off = p_shell.ao_offset;
                    let np = p_shell.n_functions();
                    let blk = shell_3center_block(p_shell, i_shell, j_shell);

                    for p_fn in 0..np {
                        let xp = x_vec[p_off + p_fn];
                        if xp.abs() < 1e-15 {
                            continue;
                        }
                        for mu in 0..ni {
                            for nu in 0..nj {
                                j_blk[mu * nj + nu] += xp * blk[(p_fn * ni + mu) * nj + nu];
                            }
                        }
                    }
                }
                ((i_idx, j_idx), j_blk)
            })
            .collect();

        let mut j_mat = Matrix2D::zeros(n_ao, n_ao);
        for ((i_idx, j_idx), blk) in pair_contributions {
            let i_off = primary.shells[i_idx].ao_offset;
            let j_off = primary.shells[j_idx].ao_offset;
            let ni = primary.shells[i_idx].n_functions();
            let nj = primary.shells[j_idx].n_functions();

            for mu in 0..ni {
                for nu in 0..nj {
                    let val = blk[mu * nj + nu];
                    j_mat.set(i_off + mu, j_off + nu, val);
                    j_mat.set(j_off + nu, i_off + mu, val);
                }
            }
        }

        j_mat
    }

    /// Computes the exact analytical nuclear gradient of the RI-J Coulomb energy:
    ///
    /// `dE_J^{RI}/dR_{A,q} = sum_{P,mu,nu} x_P P_{mu,nu} d(P|mu nu)/dR_{A,q} - 1/2 sum_{P,Q} x_P x_Q d(P|Q)/dR_{A,q}`
    ///
    /// where `x_P = sum_Q (P|Q)^{-1} sum_{mu,nu} (Q|mu nu) P_{mu,nu}`.
    ///
    /// Explicitly rejects rank-truncated pseudoinverse metrics (`metric_truncated_modes > 0`),
    /// where the full-rank inverse derivative identity `d(V^{-1}) = -V^{-1} (dV) V^{-1}` does not hold.
    pub fn compute_coulomb_gradient(
        &self,
        n_atoms: usize,
        primary: &BasisSet,
        density: &Matrix2D,
    ) -> Result<Vec<[f64; 3]>, CalcError> {
        if self.metric_truncated_modes > 0 {
            return Err(CalcError::LinearAlgebra(format!(
                "Analytical RI-J nuclear gradient is undefined when Coulomb metric inversion truncated {} near-singular eigenmode(s); full-rank metric required",
                self.metric_truncated_modes
            )));
        }
        assert_eq!(
            primary.n_ao, self.n_ao,
            "RI-J orbital basis dimension mismatch"
        );
        assert_eq!(
            (density.rows, density.cols, density.data.len()),
            (self.n_ao, self.n_ao, self.n_ao * self.n_ao),
            "RI-J density dimension mismatch"
        );
        let n_ao = primary.n_ao;
        let n_aux = self.aux_basis.n_aux;
        let n_ao_sq = n_ao * n_ao;
        let n_primary_shells = primary.shells.len();
        let primary_bounds = shell_pair_bounds(primary);

        // 1. Compute v_P = sum_{mu,nu} (P|mu nu) P_{mu,nu}
        let v_vec: Vec<f64> = if let Some(ref b_tensor) = self.cached_b {
            (0..n_aux)
                .into_par_iter()
                .map(|p| {
                    let b_slice = &b_tensor[p * n_ao_sq..(p + 1) * n_ao_sq];
                    let mut sum = 0.0;
                    for k in 0..n_ao_sq {
                        sum += b_slice[k] * density.data[k];
                    }
                    sum
                })
                .collect()
        } else {
            self.aux_basis
                .shells
                .par_iter()
                .enumerate()
                .map(|(p_idx, p_shell)| {
                    let p_b = self.aux_diag_bounds[p_idx];
                    let np = p_shell.n_functions();
                    let mut v_local = vec![0.0; np];
                    for i_idx in 0..n_primary_shells {
                        let i_shell = &primary.shells[i_idx];
                        let i_off = i_shell.ao_offset;
                        let ni = i_shell.n_functions();

                        for j_idx in 0..=i_idx {
                            let q_ij = primary_bounds[i_idx * n_primary_shells + j_idx];
                            if p_b * q_ij < 1e-12 {
                                continue;
                            }

                            let j_shell = &primary.shells[j_idx];
                            let j_off = j_shell.ao_offset;
                            let nj = j_shell.n_functions();
                            let deg = if i_idx == j_idx { 1.0 } else { 2.0 };

                            let blk = shell_3center_block(p_shell, i_shell, j_shell);
                            for p_fn in 0..np {
                                let mut pair_sum = 0.0;
                                for mu in 0..ni {
                                    for nu in 0..nj {
                                        let d_val = density.get(i_off + mu, j_off + nu);
                                        let int_val = blk[(p_fn * ni + mu) * nj + nu];
                                        pair_sum += d_val * int_val;
                                    }
                                }
                                v_local[p_fn] += deg * pair_sum;
                            }
                        }
                    }
                    v_local
                })
                .flatten()
                .collect()
        };

        // 2. Solve V x = v via Cholesky factor (or pseudoinverse fallback)
        let x_vec = self.metric_factor.solve(&v_vec);

        // 3. 3-center derivative contribution: sum_{P,mu,nu} x_P P_{mu,nu} d(P|mu nu)/dR
        let mut grad = self
            .aux_basis
            .shells
            .par_iter()
            .enumerate()
            .fold(
                || vec![[0.0; 3]; n_atoms],
                |mut local_grad, (p_idx, p_shell)| {
                    let p_b = self.aux_diag_bounds[p_idx];
                    let p_off = p_shell.ao_offset;
                    let p_comps = p_shell.components();
                    let max_p = p_shell.shell_type.angular_momentum() as usize + 1;

                    for i_idx in 0..n_primary_shells {
                        let i_shell = &primary.shells[i_idx];
                        let i_off = i_shell.ao_offset;
                        let i_comps = i_shell.components();

                        for j_idx in 0..=i_idx {
                            let q_ij = primary_bounds[i_idx * n_primary_shells + j_idx];
                            if p_b * q_ij < 1e-12 {
                                continue;
                            }

                            let j_shell = &primary.shells[j_idx];
                            if p_shell.atom_index == i_shell.atom_index
                                && i_shell.atom_index == j_shell.atom_index
                            {
                                // Single-center 3-center integral is translationally invariant.
                                continue;
                            }
                            let j_off = j_shell.ao_offset;
                            let j_comps = j_shell.components();
                            let deg = if i_idx == j_idx { 1.0 } else { 2.0 };
                            let max_ab = (i_shell.shell_type.angular_momentum()
                                + j_shell.shell_type.angular_momentum())
                                as usize
                                + 1;

                            for (p_prim, &gamma_p) in p_shell.exponents.iter().enumerate() {
                                let cp_raw = p_shell.coeffs[p_prim];
                                for (i_prim, prim_i) in i_shell.primitives.iter().enumerate() {
                                    let alpha = prim_i.exponent;
                                    for (j_prim, prim_j) in j_shell.primitives.iter().enumerate() {
                                        let beta = prim_j.exponent;
                                        let mut data = Eri3CenterData::new(
                                            gamma_p,
                                            p_shell.center,
                                            alpha,
                                            i_shell.center,
                                            beta,
                                            j_shell.center,
                                            max_p,
                                            max_ab,
                                        );

                                        for (p_fn, &p_lmn) in p_comps.iter().enumerate() {
                                            let xp = x_vec[p_off + p_fn];
                                            if xp.abs() < 1e-18 {
                                                continue;
                                            }
                                            let cp = cp_raw * primitive_norm(gamma_p, p_lmn);
                                            for (mu, &i_lmn) in i_comps.iter().enumerate() {
                                                let ci = prim_i.coeff
                                                    * i_shell.primitive_norm_for(i_prim, i_lmn);
                                                for (nu, &j_lmn) in j_comps.iter().enumerate() {
                                                    let d_mn =
                                                        density.get(i_off + mu, j_off + nu);
                                                    let cj = prim_j.coeff
                                                        * j_shell.primitive_norm_for(j_prim, j_lmn);
                                                    let w = deg * xp * d_mn * cp * ci * cj;
                                                    if w.abs() < 1e-18 {
                                                        continue;
                                                    }

                                                    for axis in 0..3 {
                                                        let mut p_plus = p_lmn;
                                                        p_plus[axis] += 1;
                                                        let mut dp = 2.0
                                                            * gamma_p
                                                            * data.hrr_ket(p_plus, i_lmn, j_lmn);
                                                        if p_lmn[axis] > 0 {
                                                            let mut p_minus = p_lmn;
                                                            p_minus[axis] -= 1;
                                                            dp -= (p_lmn[axis] as f64)
                                                                * data.hrr_ket(
                                                                    p_minus, i_lmn, j_lmn,
                                                                );
                                                        }

                                                        let mut i_plus = i_lmn;
                                                        i_plus[axis] += 1;
                                                        let mut di = 2.0
                                                            * alpha
                                                            * data.hrr_ket(p_lmn, i_plus, j_lmn);
                                                        if i_lmn[axis] > 0 {
                                                            let mut i_minus = i_lmn;
                                                            i_minus[axis] -= 1;
                                                            di -= (i_lmn[axis] as f64)
                                                                * data.hrr_ket(
                                                                    p_lmn, i_minus, j_lmn,
                                                                );
                                                        }

                                                        let mut j_plus = j_lmn;
                                                        j_plus[axis] += 1;
                                                        let mut dj = 2.0
                                                            * beta
                                                            * data.hrr_ket(p_lmn, i_lmn, j_plus);
                                                        if j_lmn[axis] > 0 {
                                                            let mut j_minus = j_lmn;
                                                            j_minus[axis] -= 1;
                                                            dj -= (j_lmn[axis] as f64)
                                                                * data.hrr_ket(
                                                                    p_lmn, i_lmn, j_minus,
                                                                );
                                                        }

                                                        local_grad[p_shell.atom_index][axis] +=
                                                            w * dp;
                                                        local_grad[i_shell.atom_index][axis] +=
                                                            w * di;
                                                        local_grad[j_shell.atom_index][axis] +=
                                                            w * dj;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    local_grad
                },
            )
            .reduce(
                || vec![[0.0; 3]; n_atoms],
                |mut a, b| {
                    for idx in 0..n_atoms {
                        for axis in 0..3 {
                            a[idx][axis] += b[idx][axis];
                        }
                    }
                    a
                },
            );

        // 4. 2-center metric derivative contribution: -1/2 sum_{P,Q} x_P x_Q d(P|Q)/dR
        let n_aux_shells = self.aux_basis.shells.len();
        for p_idx in 0..n_aux_shells {
            let p_shell = &self.aux_basis.shells[p_idx];
            let p_off = p_shell.ao_offset;
            let p_comps = p_shell.components();
            let max_p = p_shell.shell_type.angular_momentum() as usize + 1;

            for q_idx in 0..p_idx {
                let q_shell = &self.aux_basis.shells[q_idx];
                if p_shell.atom_index == q_shell.atom_index {
                    continue;
                }
                let q_off = q_shell.ao_offset;
                let q_comps = q_shell.components();
                let max_q = q_shell.shell_type.angular_momentum() as usize + 1;

                for (p_prim, &gamma_p) in p_shell.exponents.iter().enumerate() {
                    let cp_raw = p_shell.coeffs[p_prim];
                    for (q_prim, &gamma_q) in q_shell.exponents.iter().enumerate() {
                        let cq_raw = q_shell.coeffs[q_prim];
                        let mut data = Eri2CenterData::new(
                            gamma_p,
                            p_shell.center,
                            gamma_q,
                            q_shell.center,
                            max_p,
                            max_q,
                        );

                        for (p_fn, &p_lmn) in p_comps.iter().enumerate() {
                            let xp = x_vec[p_off + p_fn];
                            if xp.abs() < 1e-18 {
                                continue;
                            }
                            let cp = cp_raw * primitive_norm(gamma_p, p_lmn);
                            for (q_fn, &q_lmn) in q_comps.iter().enumerate() {
                                let xq = x_vec[q_off + q_fn];
                                let cq = cq_raw * primitive_norm(gamma_q, q_lmn);
                                // Factor of 2 for off-diagonal (P,Q) and (Q,P) cancels the 1/2 in -1/2 x_P x_Q dV_{PQ}.
                                let w_2c = -xp * xq * cp * cq;
                                if w_2c.abs() < 1e-18 {
                                    continue;
                                }

                                for axis in 0..3 {
                                    let mut p_plus = p_lmn;
                                    p_plus[axis] += 1;
                                    let mut dp = 2.0 * gamma_p * data.vrr(p_plus, q_lmn, 0);
                                    if p_lmn[axis] > 0 {
                                        let mut p_minus = p_lmn;
                                        p_minus[axis] -= 1;
                                        dp -= (p_lmn[axis] as f64) * data.vrr(p_minus, q_lmn, 0);
                                    }

                                    let mut q_plus = q_lmn;
                                    q_plus[axis] += 1;
                                    let mut dq = 2.0 * gamma_q * data.vrr(p_lmn, q_plus, 0);
                                    if q_lmn[axis] > 0 {
                                        let mut q_minus = q_lmn;
                                        q_minus[axis] -= 1;
                                        dq -= (q_lmn[axis] as f64) * data.vrr(p_lmn, q_minus, 0);
                                    }

                                    grad[p_shell.atom_index][axis] += w_2c * dp;
                                    grad[q_shell.atom_index][axis] += w_2c * dq;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_model::Atom;
    use calchem_tensor::eigh_sym;

    #[test]
    fn test_2center_metric_positivity_and_symmetry() {
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
        let aux = AuxiliaryBasisSet::build_auto_jfit(&mol, &bs);
        assert!(aux.n_aux > 0);

        let v = compute_2center_coulomb_metric(&aux);
        assert_eq!(v.rows, aux.n_aux);
        assert_eq!(v.cols, aux.n_aux);

        // Check symmetry V_PQ == V_QP
        for i in 0..v.rows {
            for j in 0..v.cols {
                assert!((v.get(i, j) - v.get(j, i)).abs() < 1e-12);
            }
        }

        // Check positive-definiteness: all eigenvalues > 0
        let (evals, _) = eigh_sym(&v, 200, 1e-14).unwrap();
        for &ev in &evals {
            assert!(ev > 1e-12, "Metric eigenvalue must be positive: {ev}");
        }
    }

    #[test]
    fn test_ri_j_coulomb_matrix_matches_exact_four_center_on_water() {
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
        let n_ao = bs.n_ao;

        // Arbitrary symmetric positive test density matrix
        let mut d = Matrix2D::zeros(n_ao, n_ao);
        for i in 0..n_ao {
            d.set(i, i, 1.5 / (i + 1) as f64);
            for j in 0..i {
                let v = 0.1 / ((i + j + 1) as f64);
                d.set(i, j, v);
                d.set(j, i, v);
            }
        }

        // Build RI-J Coulomb
        let rij = RiJEngine::new(&mol, &bs).unwrap();
        let j_ri = rij.build_coulomb(&bs, &d);

        // Build Exact 4-center Coulomb
        let direct_eng = crate::direct_fock::DirectFockEngine::new(&bs, 0.0);
        let j_exact_vec = direct_eng.build_rks(&bs, &d.data, 0.0, 0.0);

        // RI Coulomb should approximate exact Coulomb within density fitting accuracy (< 1e-3 a.u. for minimal STO-3G)
        let mut max_diff = 0.0;
        for i in 0..n_ao {
            for j in 0..n_ao {
                let diff = (j_ri.get(i, j) - j_exact_vec[i * n_ao + j]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 5e-3,
            "RI-J Coulomb max diff vs exact 4-center should be < 5e-3, got {max_diff:.3e}"
        );
    }
}
