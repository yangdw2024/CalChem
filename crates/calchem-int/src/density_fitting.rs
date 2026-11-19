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
            let compact_large = mol.atoms.len() > 30;
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
            // Default cap L_aux <= 2 guarantees 100% GPU kernel residence on CUDA devices.
            let aux_max_l_cap: u8 = std::env::var("CALCHEM_AUTO_AUX_MAX_L")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2);

            for &exp in &exps {
                let max_l_raw = if exp > 80.0 {
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
                let max_l_for_exp = max_l_raw.min(aux_max_l_cap);

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

    for (pp,&alpha) in p_shell.exponents.iter().enumerate() {
        for (qq,&beta) in q_shell.exponents.iter().enumerate() {
            let mut data = Eri2CenterData::new(alpha,p_shell.center,beta,q_shell.center,
                p_shell.shell_type.angular_momentum() as usize,q_shell.shell_type.angular_momentum() as usize);
            for (i,&a) in p_comps.iter().enumerate() {
                let ca = p_shell.coeffs[pp]*primitive_norm(alpha,a);
                for (j,&b) in q_comps.iter().enumerate() {
                    let cb = q_shell.coeffs[qq]*primitive_norm(beta,b);
                    block.add_assign_at(i,j,ca*cb*data.vrr(a,b,0));
                }
            }
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
struct Eri3CenterData<'a> {
    p: f64,
    q: f64,
    zeta: f64,
    rho: f64,
    qc: [f64; 3],
    cd: [f64; 3],
    wp: [f64; 3],
    wq: [f64; 3],
    prefactor: f64,
    boys: [f64; 16],
    max_p: usize,
    max_ab: usize,
    max_m: usize,
    vrr_cache: &'a mut Vec<f64>,
}

/// Precomputed primitive pair for primary orbital shells `(i, j)` reused across all auxiliary shells `P`.
#[derive(Debug, Clone)]
struct PrecomputedPrimPair3C {
    q: f64,
    p_center: [f64; 3],
    qc: [f64; 3],
    qd: [f64; 3],
    cd: [f64; 3],
    two_pi25_kab_over_q: f64,
    ci_norm: [f64; 10],
    cj_norm: [f64; 10],
}

fn precompute_prim_pairs_3c(i_shell: &Shell, j_shell: &Shell) -> Vec<PrecomputedPrimPair3C> {
    let i_comps = i_shell.components();
    let j_comps = j_shell.components();
    let ni = i_comps.len();
    let nj = j_comps.len();
    let mut pairs = Vec::with_capacity(i_shell.primitives.len() * j_shell.primitives.len());
    let two_pi25 = 2.0 * PI.powf(2.5);

    for (i_prim, prim_i) in i_shell.primitives.iter().enumerate() {
        let mut ci_norm = [0.0f64; 10];
        for (ii, &i_lmn) in i_comps.iter().enumerate() {
            ci_norm[ii] = prim_i.coeff * i_shell.primitive_norm_for(i_prim, i_lmn);
        }
        for (j_prim, prim_j) in j_shell.primitives.iter().enumerate() {
            let pair_cd = GaussianPair::new(
                prim_i.exponent,
                i_shell.center,
                prim_j.exponent,
                j_shell.center,
            );
            if pair_cd.kab < 1e-18 {
                continue;
            }
            let mut cj_norm = [0.0f64; 10];
            for (jj, &j_lmn) in j_comps.iter().enumerate() {
                cj_norm[jj] = prim_j.coeff * j_shell.primitive_norm_for(j_prim, j_lmn);
            }
            let qd = [
                pair_cd.pa[0] + pair_cd.ab[0],
                pair_cd.pa[1] + pair_cd.ab[1],
                pair_cd.pa[2] + pair_cd.ab[2],
            ];
            let _ = (ni, nj);
            pairs.push(PrecomputedPrimPair3C {
                q: pair_cd.gamma,
                p_center: pair_cd.p,
                qc: pair_cd.pa,
                qd,
                cd: pair_cd.ab,
                two_pi25_kab_over_q: two_pi25 * pair_cd.kab / pair_cd.gamma,
                ci_norm,
                cj_norm,
            });
        }
    }
    pairs
}

impl<'a> Eri3CenterData<'a> {
    #[allow(clippy::too_many_arguments)]
    fn init_from_pair(
        gamma_p: f64,
        r_p: [f64; 3],
        pp: &PrecomputedPrimPair3C,
        max_p: usize,
        max_ab: usize,
        vrr_cache: &'a mut Vec<f64>,
    ) -> Self {
        let p = gamma_p;
        let q = pp.q;
        let zeta = p + q;
        let inv_zeta = 1.0 / zeta;
        let rho = p * q * inv_zeta;
        let w = [
            (p * r_p[0] + q * pp.p_center[0]) * inv_zeta,
            (p * r_p[1] + q * pp.p_center[1]) * inv_zeta,
            (p * r_p[2] + q * pp.p_center[2]) * inv_zeta,
        ];
        let wp = [w[0] - r_p[0], w[1] - r_p[1], w[2] - r_p[2]];
        let wq = [
            w[0] - pp.p_center[0],
            w[1] - pp.p_center[1],
            w[2] - pp.p_center[2],
        ];
        let dx = r_p[0] - pp.p_center[0];
        let dy = r_p[1] - pp.p_center[1];
        let dz = r_p[2] - pp.p_center[2];
        let pq2 = dx * dx + dy * dy + dz * dz;
        let max_m = max_p + max_ab;
        let mut boys = [0.0f64; 16];
        boys_batch(max_m, rho * pq2, &mut boys[..=max_m]);
        let prefactor = pp.two_pi25_kab_over_q / (p * zeta.sqrt());

        let dp = max_p + 1;
        let dab = max_ab + 1;
        let dm = max_m + 1;
        let cache_len = if max_m == 0 {
            0
        } else {
            dp * dp * dp * dab * dab * dab * dm
        };
        if vrr_cache.len() < cache_len {
            vrr_cache.resize(cache_len, f64::NAN);
        }
        vrr_cache[..cache_len].fill(f64::NAN);

        Self {
            p,
            q,
            zeta,
            rho,
            qc: pp.qc,
            cd: pp.cd,
            wp,
            wq,
            prefactor,
            boys,
            max_p,
            max_ab,
            max_m,
            vrr_cache,
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
    let pair_cd = GaussianPair::new(alpha, r_a, beta, r_b);
    let qd = [
        pair_cd.pa[0] + pair_cd.ab[0],
        pair_cd.pa[1] + pair_cd.ab[1],
        pair_cd.pa[2] + pair_cd.ab[2],
    ];
    let pp = PrecomputedPrimPair3C {
        q: pair_cd.gamma,
        p_center: pair_cd.p,
        qc: pair_cd.pa,
        qd,
        cd: pair_cd.ab,
        two_pi25_kab_over_q: 2.0 * PI.powf(2.5) * pair_cd.kab / pair_cd.gamma,
        ci_norm: [1.0; 10],
        cj_norm: [1.0; 10],
    };
    let mut vrr_cache = Vec::new();
    let mut data = Eri3CenterData::init_from_pair(gamma_p, r_p, &pp, max_p, max_ab, &mut vrr_cache);
    data.hrr_ket(p_lmn, a_lmn, b_lmn)
}

/// Evaluates a 3-center contracted shell block `(P | i j)` using precomputed `(i, j)` primitive
/// pairs and a reusable thread-local `vrr_cache`, with zero heap allocations and closed-form
/// fast paths for S/P angular momenta.
fn shell_3center_block_precomputed(
    p_shell: &AuxiliaryShell,
    i_shell: &Shell,
    j_shell: &Shell,
    prim_pairs: &[PrecomputedPrimPair3C],
    out: &mut Vec<f64>,
    vrr_cache: &mut Vec<f64>,
) {
    let np = p_shell.n_functions();
    let ni = i_shell.n_functions();
    let nj = j_shell.n_functions();
    out.resize(np * ni * nj, 0.0);
    out.fill(0.0);
    if prim_pairs.is_empty() {
        return;
    }

    let p_comps = p_shell.components();
    let i_comps = i_shell.components();
    let j_comps = j_shell.components();
    let lp = p_shell.shell_type.angular_momentum() as usize;
    let li = i_shell.shell_type.angular_momentum() as usize;
    let lj = j_shell.shell_type.angular_momentum() as usize;
    let max_ab = li + lj;
    let r_p = p_shell.center;

    for (p_prim, &gamma_p) in p_shell.exponents.iter().enumerate() {
        let p_c_base = p_shell.coeffs[p_prim];
        let mut cp_norms = [0.0f64; 15];
        for (p_idx, &p_lmn) in p_comps.iter().enumerate() {
            cp_norms[p_idx] = p_c_base * primitive_norm(gamma_p, p_lmn);
        }

        // Fast closed-form paths for lp <= 1, li <= 1, lj <= 1 (covers all S/P combinations!)
        if lp <= 1 && li <= 1 && lj <= 1 {
            let max_m = lp + li + lj;
            let mut boys = [0.0f64; 4];
            for pp in prim_pairs {
                let p = gamma_p;
                let q = pp.q;
                let zeta = p + q;
                let inv_zeta = 1.0 / zeta;
                let rho = p * q * inv_zeta;
                let dx = r_p[0] - pp.p_center[0];
                let dy = r_p[1] - pp.p_center[1];
                let dz = r_p[2] - pp.p_center[2];
                let pq2 = dx * dx + dy * dy + dz * dz;
                boys_batch(max_m, rho * pq2, &mut boys[..=max_m]);
                let pref = pp.two_pi25_kab_over_q / (p * zeta.sqrt());

                let t0 = pref * boys[0];
                let t1 = if max_m >= 1 { pref * boys[1] } else { 0.0 };
                let t2 = if max_m >= 2 { pref * boys[2] } else { 0.0 };
                let t3 = if max_m >= 3 { pref * boys[3] } else { 0.0 };

                match (lp, li, lj) {
                    (0, 0, 0) => {
                        out[0] += cp_norms[0] * pp.ci_norm[0] * pp.cj_norm[0] * t0;
                    }
                    (0, 1, 0) => {
                        let wx = (p * r_p[0] + q * pp.p_center[0]) * inv_zeta;
                        let wy = (p * r_p[1] + q * pp.p_center[1]) * inv_zeta;
                        let wz = (p * r_p[2] + q * pp.p_center[2]) * inv_zeta;
                        let wq = [wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]];
                        let scale = cp_norms[0] * pp.cj_norm[0];
                        for a in 0..3 {
                            out[a] += scale * pp.ci_norm[a] * (pp.qc[a] * t0 + wq[a] * t1);
                        }
                    }
                    (0, 0, 1) => {
                        let wx = (p * r_p[0] + q * pp.p_center[0]) * inv_zeta;
                        let wy = (p * r_p[1] + q * pp.p_center[1]) * inv_zeta;
                        let wz = (p * r_p[2] + q * pp.p_center[2]) * inv_zeta;
                        let wq = [wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]];
                        let scale = cp_norms[0] * pp.ci_norm[0];
                        for b in 0..3 {
                            out[b] += scale * pp.cj_norm[b] * (pp.qd[b] * t0 + wq[b] * t1);
                        }
                    }
                    (1, 0, 0) => {
                        let wx = (p * r_p[0] + q * pp.p_center[0]) * inv_zeta;
                        let wy = (p * r_p[1] + q * pp.p_center[1]) * inv_zeta;
                        let wz = (p * r_p[2] + q * pp.p_center[2]) * inv_zeta;
                        let wp = [wx - r_p[0], wy - r_p[1], wz - r_p[2]];
                        let scale = pp.ci_norm[0] * pp.cj_norm[0];
                        for k in 0..3 {
                            out[k] += cp_norms[k] * scale * (wp[k] * t1);
                        }
                    }
                    (1, 1, 0) => {
                        let wx = (p * r_p[0] + q * pp.p_center[0]) * inv_zeta;
                        let wy = (p * r_p[1] + q * pp.p_center[1]) * inv_zeta;
                        let wz = (p * r_p[2] + q * pp.p_center[2]) * inv_zeta;
                        let wp = [wx - r_p[0], wy - r_p[1], wz - r_p[2]];
                        let wq = [wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]];
                        let half_iz = 0.5 * inv_zeta * t1;
                        let i1 = [
                            pp.qc[0] * t1 + wq[0] * t2,
                            pp.qc[1] * t1 + wq[1] * t2,
                            pp.qc[2] * t1 + wq[2] * t2,
                        ];
                        let cj0 = pp.cj_norm[0];
                        for k in 0..3 {
                            let cpk = cp_norms[k] * cj0;
                            let wpk = wp[k];
                            for a in 0..3 {
                                let mut v = wpk * i1[a];
                                if a == k {
                                    v += half_iz;
                                }
                                out[k * 3 + a] += cpk * pp.ci_norm[a] * v;
                            }
                        }
                    }
                    (1, 0, 1) => {
                        let wx = (p * r_p[0] + q * pp.p_center[0]) * inv_zeta;
                        let wy = (p * r_p[1] + q * pp.p_center[1]) * inv_zeta;
                        let wz = (p * r_p[2] + q * pp.p_center[2]) * inv_zeta;
                        let wp = [wx - r_p[0], wy - r_p[1], wz - r_p[2]];
                        let wq = [wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]];
                        let half_iz = 0.5 * inv_zeta * t1;
                        let j1 = [
                            pp.qd[0] * t1 + wq[0] * t2,
                            pp.qd[1] * t1 + wq[1] * t2,
                            pp.qd[2] * t1 + wq[2] * t2,
                        ];
                        let ci0 = pp.ci_norm[0];
                        for k in 0..3 {
                            let cpk = cp_norms[k] * ci0;
                            let wpk = wp[k];
                            for b in 0..3 {
                                let mut v = wpk * j1[b];
                                if b == k {
                                    v += half_iz;
                                }
                                out[k * 3 + b] += cpk * pp.cj_norm[b] * v;
                            }
                        }
                    }
                    (0, 1, 1) => {
                        let wx = (p * r_p[0] + q * pp.p_center[0]) * inv_zeta;
                        let wy = (p * r_p[1] + q * pp.p_center[1]) * inv_zeta;
                        let wz = (p * r_p[2] + q * pp.p_center[2]) * inv_zeta;
                        let wq = [wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]];
                        let diag_term = (t0 - (rho / q) * t1) / (2.0 * q);
                        let b0 = [
                            pp.qd[0] * t0 + wq[0] * t1,
                            pp.qd[1] * t0 + wq[1] * t1,
                            pp.qd[2] * t0 + wq[2] * t1,
                        ];
                        let b1 = [
                            pp.qd[0] * t1 + wq[0] * t2,
                            pp.qd[1] * t1 + wq[1] * t2,
                            pp.qd[2] * t1 + wq[2] * t2,
                        ];
                        let cp0 = cp_norms[0];
                        for a in 0..3 {
                            let cpa = cp0 * pp.ci_norm[a];
                            for b in 0..3 {
                                let mut v = pp.qc[a] * b0[b] + wq[a] * b1[b];
                                if a == b {
                                    v += diag_term;
                                }
                                out[a * 3 + b] += cpa * pp.cj_norm[b] * v;
                            }
                        }
                    }
                    (1, 1, 1) => {
                        let wx = (p * r_p[0] + q * pp.p_center[0]) * inv_zeta;
                        let wy = (p * r_p[1] + q * pp.p_center[1]) * inv_zeta;
                        let wz = (p * r_p[2] + q * pp.p_center[2]) * inv_zeta;
                        let wp = [wx - r_p[0], wy - r_p[1], wz - r_p[2]];
                        let wq = [wx - pp.p_center[0], wy - pp.p_center[1], wz - pp.p_center[2]];
                        let half_iz = 0.5 * inv_zeta;
                        let diag1 = (t1 - (rho / q) * t2) / (2.0 * q);
                        let i1 = [
                            pp.qc[0] * t1 + wq[0] * t2,
                            pp.qc[1] * t1 + wq[1] * t2,
                            pp.qc[2] * t1 + wq[2] * t2,
                        ];
                        let j1 = [
                            pp.qd[0] * t1 + wq[0] * t2,
                            pp.qd[1] * t1 + wq[1] * t2,
                            pp.qd[2] * t1 + wq[2] * t2,
                        ];
                        let j2 = [
                            pp.qd[0] * t2 + wq[0] * t3,
                            pp.qd[1] * t2 + wq[1] * t3,
                            pp.qd[2] * t2 + wq[2] * t3,
                        ];
                        let mut ab1 = [[0.0f64; 3]; 3];
                        for a in 0..3 {
                            for b in 0..3 {
                                let mut v = pp.qc[a] * j1[b] + wq[a] * j2[b];
                                if a == b {
                                    v += diag1;
                                }
                                ab1[a][b] = v;
                            }
                        }
                        for k in 0..3 {
                            let cpk = cp_norms[k];
                            let wpk = wp[k];
                            for a in 0..3 {
                                let cpka = cpk * pp.ci_norm[a];
                                let a_eq_k = a == k;
                                let i1_a = i1[a];
                                for b in 0..3 {
                                    let mut v = wpk * ab1[a][b];
                                    if a_eq_k {
                                        v += half_iz * j1[b];
                                    }
                                    if b == k {
                                        v += half_iz * i1_a;
                                    }
                                    out[(k * 3 + a) * 3 + b] += cpka * pp.cj_norm[b] * v;
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            continue;
        }

        // General zero-allocation Obara-Saika VRR/HRR for D/F/G shells
        for pp in prim_pairs {
            let mut data = Eri3CenterData::init_from_pair(
                gamma_p, r_p, pp, lp, max_ab, vrr_cache,
            );
            for (p_idx, &p_lmn) in p_comps.iter().enumerate() {
                let cp = cp_norms[p_idx];
                for (i_idx, &i_lmn) in i_comps.iter().enumerate() {
                    let cpi = cp * pp.ci_norm[i_idx];
                    for (j_idx, &j_lmn) in j_comps.iter().enumerate() {
                        let val = data.hrr_ket(p_lmn, i_lmn, j_lmn);
                        out[(p_idx * ni + i_idx) * nj + j_idx] += cpi * pp.cj_norm[j_idx] * val;
                    }
                }
            }
        }
    }
}

/// Evaluates a 3-center contracted shell block: (P | i j) where P is an auxiliary shell
/// and (i, j) are orbital shells, writing into `out` (resized to `np * ni * nj`) in `[P][i][j]` order.
pub fn shell_3center_block_into(
    p_shell: &AuxiliaryShell,
    i_shell: &Shell,
    j_shell: &Shell,
    out: &mut Vec<f64>,
) {
    let prim_pairs = precompute_prim_pairs_3c(i_shell, j_shell);
    let mut vrr_cache = Vec::new();
    shell_3center_block_precomputed(p_shell, i_shell, j_shell, &prim_pairs, out, &mut vrr_cache);
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
    packed_b: Option<calchem_cuda::ResidentRijTensor>,
    // One entry per independent AO product, indexing the full density and its transpose.
    packed_pairs: Vec<(usize, usize)>,
    pub n_ao: usize,
    pub aux_diag_bounds: Vec<f64>,
    pub gpu_kernel_invocations: AtomicUsize,
    pub gpu_fallback_invocations: AtomicUsize,
}

impl RiJEngine {
    /// Builds the RI-J engine for a given molecule and primary basis (CPU builder).
    pub fn new(mol: &Molecule, primary: &BasisSet) -> Result<Self, CalcError> {
        Self::new_with_gpu(mol, primary, false)
    }

    /// Builds the RI-J engine, optionally accelerating 3-center integral evaluation
    /// and contractions directly on the CUDA GPU.
    pub fn new_with_gpu(
        mol: &Molecule,
        primary: &BasisSet,
        use_gpu: bool,
    ) -> Result<Self, CalcError> {
        let cache_limit = std::env::var("CALCHEM_MAX_DF_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize << 30);
        Self::with_cache_limit_gpu(mol, primary, cache_limit, use_gpu)
    }

    #[allow(dead_code)]
    fn with_cache_limit(mol: &Molecule, primary: &BasisSet, cache_limit: usize) -> Result<Self, CalcError> {
        Self::with_cache_limit_gpu(mol, primary, cache_limit, false)
    }

    fn with_cache_limit_gpu(
        mol: &Molecule,
        primary: &BasisSet,
        cache_limit: usize,
        use_gpu: bool,
    ) -> Result<Self, CalcError> {
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
        let profile = std::env::var_os("CALCHEM_PROFILE").is_some();
        let t_m0 = std::time::Instant::now();
        let v = compute_2center_coulomb_metric(&aux_basis);
        let d_m2c = t_m0.elapsed();
        let t_f0 = std::time::Instant::now();
        let metric_factor = SpdFactor::for_solve(&v, 1e-12)?;
        let d_mfac = t_f0.elapsed();
        let metric_truncated_modes = metric_factor.n_truncated;

        let n_ao = primary.n_ao;
        let n_aux = aux_basis.n_aux;
        let n_ao_sq = n_ao * n_ao;
        let req_bytes = n_aux
            .checked_mul(n_ao_sq)
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| CalcError::UnsupportedBasis("RI-J tensor dimensions overflow".into()))?;

        let max_df_bytes = cache_limit.min(budget.saturating_sub(metric_bytes).min(usize::MAX as u128) as usize);

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

        let primary_bounds = shell_pair_bounds(primary);
        let n_primary_shells = primary.shells.len();
        let mut packed_pairs = Vec::new();
        let mut shell_offsets = vec![usize::MAX; n_primary_shells*n_primary_shells];
        let packed = req_bytes > max_df_bytes.min(256 << 20);
        let max_bound = aux_diag_bounds.iter().copied().fold(0.0_f64, f64::max);
        let mut sig_ij_list: Vec<(usize, usize, f64, usize, Vec<PrecomputedPrimPair3C>)> = Vec::new();
        for (i,a) in primary.shells.iter().enumerate() {
            for (j,b) in primary.shells[..=i].iter().enumerate() {
                let ij = i*n_primary_shells+j;
                let q_ij = primary_bounds[ij];
                if max_bound * q_ij < 1e-12 { continue; }
                let pps = precompute_prim_pairs_3c(a, b);
                if pps.is_empty() { continue; }
                let offset = if packed {
                    let off = packed_pairs.len();
                    shell_offsets[ij] = off;
                    for mu in 0..a.n_functions() {
                        let last = if i==j { mu+1 } else { b.n_functions() };
                        for nu in 0..last {
                            let (u,v)=(a.ao_offset+mu,b.ao_offset+nu);
                            packed_pairs.push((u*n_ao+v,v*n_ao+u));
                        }
                    }
                    off
                } else {
                    0
                };
                sig_ij_list.push((i, j, q_ij, offset, pps));
            }
        }
        let width = if packed { packed_pairs.len() } else { n_ao_sq };
        let bytes = n_aux as u128 * width as u128 * 8;
        if profile {
            eprintln!(
                "[rij] n_aux={n_aux} width={width} dense_width={n_ao_sq} bytes={bytes} limit={max_df_bytes} sig_ij={} metric_2c={:.3}s metric_fac={:.3}s",
                sig_ij_list.len(),
                d_m2c.as_secs_f64(),
                d_mfac.as_secs_f64()
            );
        }
        let t_3c0 = std::time::Instant::now();
        let gpu_eligible = use_gpu
            && packed
            && width > 0
            && bytes <= max_df_bytes as u128
            && calchem_cuda::is_cuda_available()
            && aux_basis
                .shells
                .iter()
                .all(|s| s.shell_type.angular_momentum() <= 2)
            && primary
                .shells
                .iter()
                .all(|s| s.shell_type.angular_momentum() <= 2);

        let mut gpu_packed_b: Option<calchem_cuda::ResidentRijTensor> = None;
        if gpu_eligible {
            let mut gpu_aux_shells = Vec::with_capacity(aux_basis.shells.len());
            let mut gpu_aux_prims = Vec::new();
            for (p_idx, p_shell) in aux_basis.shells.iter().enumerate() {
                let prim_start = gpu_aux_prims.len() as i32;
                let p_comps = p_shell.components();
                for (pp_i, &gamma_p) in p_shell.exponents.iter().enumerate() {
                    let c_base = p_shell.coeffs[pp_i];
                    let mut cp_norm = [0.0f64; 6];
                    for (pf, &lmn) in p_comps.iter().enumerate() {
                        cp_norm[pf] = c_base * primitive_norm(gamma_p, lmn);
                    }
                    gpu_aux_prims.push(calchem_cuda::AuxPrimGpu { gamma_p, cp_norm });
                }
                gpu_aux_shells.push(calchem_cuda::AuxShellGpu {
                    center: p_shell.center,
                    bound: aux_diag_bounds[p_idx],
                    lp: p_shell.shell_type.angular_momentum() as i32,
                    ao_offset: p_shell.ao_offset as i32,
                    prim_start,
                    n_prims: p_shell.exponents.len() as i32,
                });
            }

            let mut gpu_shell_pairs = Vec::with_capacity(sig_ij_list.len());
            let mut gpu_prim_pairs = Vec::new();
            for &(i_idx, j_idx, q_ij, shell_off, ref pps) in &sig_ij_list {
                let prim_start = gpu_prim_pairs.len() as i32;
                let ni = primary.shells[i_idx].n_functions();
                let nj = primary.shells[j_idx].n_functions();
                for pp in pps {
                    let mut ci_norm = [0.0f64; 6];
                    ci_norm[..ni].copy_from_slice(&pp.ci_norm[..ni]);
                    let mut cj_norm = [0.0f64; 6];
                    cj_norm[..nj].copy_from_slice(&pp.cj_norm[..nj]);
                    gpu_prim_pairs.push(calchem_cuda::PrimPair3CGpu {
                        q: pp.q,
                        p_center: pp.p_center,
                        qc: pp.qc,
                        qd: pp.qd,
                        two_pi25_kab_over_q: pp.two_pi25_kab_over_q,
                        ci_norm,
                        cj_norm,
                    });
                }
                gpu_shell_pairs.push(calchem_cuda::ShellPair3CGpu {
                    q_ij,
                    li: primary.shells[i_idx].shell_type.angular_momentum() as i32,
                    lj: primary.shells[j_idx].shell_type.angular_momentum() as i32,
                    same_shell: if i_idx == j_idx { 1 } else { 0 },
                    col_offset: shell_off as i32,
                    prim_start,
                    n_prims: pps.len() as i32,
                });
            }

            gpu_packed_b = calchem_cuda::ResidentRijTensor::build_eri3c_gpu(
                &gpu_aux_shells,
                &gpu_aux_prims,
                &gpu_shell_pairs,
                &gpu_prim_pairs,
                n_aux,
                width,
            );
        }

        let (cached_b, packed_b) = if let Some(tensor) = gpu_packed_b {
            if profile {
                eprintln!("[rij] eri_3c_build_gpu={:.3}s", t_3c0.elapsed().as_secs_f64());
            }
            (None, Some(tensor))
        } else {
            let cache = if width > 0 && bytes <= max_df_bytes as u128 {
                let mut b_flat = vec![0.0; n_aux * width];
                let mut remainder = b_flat.as_mut_slice();
                let mut slices = Vec::with_capacity(aux_basis.shells.len());
                for shell in &aux_basis.shells {
                    let (rows, rest) = remainder.split_at_mut(shell.n_functions() * width);
                    slices.push(rows);
                    remainder = rest;
                }
                aux_basis
                    .shells
                    .par_iter()
                    .zip(slices.into_par_iter())
                    .enumerate()
                    .for_each_init(
                        || (Vec::<f64>::new(), Vec::<f64>::new()),
                        |(blk, vrr_cache), (p_idx, (p_shell, shell_rows))| {
                            let np = p_shell.n_functions();
                            let p_b = aux_diag_bounds[p_idx];

                            for &(i_idx, j_idx, q_ij, shell_off, ref pps) in &sig_ij_list {
                                if p_b * q_ij < 1e-12 {
                                    continue;
                                }
                                let i_shell = &primary.shells[i_idx];
                                let j_shell = &primary.shells[j_idx];
                                let i_off = i_shell.ao_offset;
                                let j_off = j_shell.ao_offset;
                                let ni = i_shell.n_functions();
                                let nj = j_shell.n_functions();

                                shell_3center_block_precomputed(
                                    p_shell, i_shell, j_shell, pps, blk, vrr_cache,
                                );
                                if packed {
                                    for p_fn in 0..np {
                                        let base = p_fn * width + shell_off;
                                        let p_blk = p_fn * ni * nj;
                                        if i_idx == j_idx {
                                            let mut pos = 0;
                                            for mu in 0..ni {
                                                for nu in 0..=mu {
                                                    shell_rows[base + pos] = blk[p_blk + mu * nj + nu];
                                                    pos += 1;
                                                }
                                            }
                                        } else {
                                            let len = ni * nj;
                                            shell_rows[base..base + len]
                                                .copy_from_slice(&blk[p_blk..p_blk + len]);
                                        }
                                    }
                                } else {
                                    for p_fn in 0..np {
                                        let p_row_offset = p_fn * width;
                                        let p_blk = p_fn * ni * nj;
                                        for mu in 0..ni {
                                            let r_mu = i_off + mu;
                                            for nu in 0..nj {
                                                let c_nu = j_off + nu;
                                                let val = blk[p_blk + mu * nj + nu];
                                                shell_rows[p_row_offset + r_mu * n_ao + c_nu] = val;
                                                shell_rows[p_row_offset + c_nu * n_ao + r_mu] = val;
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    );

                Some(b_flat)
            } else {
                None
            };
            if profile {
                eprintln!("[rij] eri_3c_build_cpu={:.3}s", t_3c0.elapsed().as_secs_f64());
            }
            if packed {
                (None, cache.map(calchem_cuda::ResidentRijTensor::new))
            } else {
                (cache, None)
            }
        };
        Ok(Self {
            use_gpu,
            aux_basis,
            metric_factor,
            metric_truncated_modes,
            cached_b, packed_b, packed_pairs,
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
        let metric_factor = SpdFactor::for_solve(&v, 1e-12)?;
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
            cached_b: None, packed_b: None, packed_pairs: Vec::new(),
            n_ao,
            aux_diag_bounds,
            gpu_kernel_invocations: AtomicUsize::new(0),
            gpu_fallback_invocations: AtomicUsize::new(0),
        })
    }

    fn pack_density(&self, density: &Matrix2D) -> Vec<f64> {
        self.packed_pairs.iter().map(|&(uv,vu)| if uv==vu {
            density.data[uv]
        } else { density.data[uv]+density.data[vu] }).collect()
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

        if let Some(b_tensor) = self.cached_b.as_deref().or_else(|| self.packed_b.as_ref().map(|b| b.as_slice())) {
            let gpu_tensor = self.packed_b.as_ref().map(|b| b.view())
                .unwrap_or_else(|| calchem_cuda::RijTensorView::from(b_tensor));
            let compact;
            let density_data = if self.packed_b.is_some() {
                compact = self.pack_density(density);
                compact.as_slice()
            } else { density.data.as_slice() };
            let width = density_data.len();
            // Fast O(N^3) GPU/parallel path with precomputed 3-center tensor
            let gpu_ready = self.use_gpu && calchem_cuda::is_cuda_available();

            // 1. v_P = ∑_{μν} B_{P, μν} P_{μν}
            let mut v_vec = vec![0.0; n_aux];
            let mut used_gpu_v = false;
            if gpu_ready {
                used_gpu_v = gpu_tensor.contract_v(
                    density_data,
                    &mut v_vec,
                    n_aux,
                    width,
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
                        let b_slice = &b_tensor[p * width..(p + 1) * width];
                        let mut sum = 0.0;
                        for k in 0..width {
                            sum += b_slice[k] * density_data[k];
                        }
                        sum
                    })
                    .collect();
            }

            // 2. Solve V x = v via Cholesky factor (or pseudoinverse fallback)
            let x_vec = self.metric_factor.solve(&v_vec);

            // 3. J_{μν} = ∑_P x_P B_{P, μν}
            let mut j_data = vec![0.0; width];
            let mut used_gpu = false;
            if self.use_gpu {
                if gpu_ready {
                    used_gpu = gpu_tensor.contract_coulomb(
                        &x_vec,
                        &mut j_data,
                        n_aux,
                        width,
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
                                sum += x_vec[p] * b_tensor[p * width + idx];
                            }
                            *val = sum;
                        }
                    });
            }

            if self.packed_b.is_some() {
                let mut j = Matrix2D::zeros(n_ao,n_ao);
                for (&(uv,vu),value) in self.packed_pairs.iter().zip(j_data) {
                    j.data[uv]=value;
                    j.data[vu]=value;
                }
                return j;
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
        let v_vec: Vec<f64> = if let Some(packed) = &self.packed_b {
            let d = self.pack_density(density);
            packed.as_slice().par_chunks_exact(d.len()).map(|b| b.iter().zip(&d).map(|(x,y)|x*y).sum()).collect()
        } else if let Some(ref b_tensor) = self.cached_b {
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

                            let mut grad_vrr_cache = Vec::new();
                            for (p_prim, &gamma_p) in p_shell.exponents.iter().enumerate() {
                                let cp_raw = p_shell.coeffs[p_prim];
                                for (i_prim, prim_i) in i_shell.primitives.iter().enumerate() {
                                    let alpha = prim_i.exponent;
                                    for (j_prim, prim_j) in j_shell.primitives.iter().enumerate() {
                                        let beta = prim_j.exponent;
                                        let pair_cd = GaussianPair::new(
                                            alpha,
                                            i_shell.center,
                                            beta,
                                            j_shell.center,
                                        );
                                        let qd = [
                                            pair_cd.pa[0] + pair_cd.ab[0],
                                            pair_cd.pa[1] + pair_cd.ab[1],
                                            pair_cd.pa[2] + pair_cd.ab[2],
                                        ];
                                        let pp = PrecomputedPrimPair3C {
                                            q: pair_cd.gamma,
                                            p_center: pair_cd.p,
                                            qc: pair_cd.pa,
                                            qd,
                                            cd: pair_cd.ab,
                                            two_pi25_kab_over_q: 2.0 * PI.powf(2.5) * pair_cd.kab
                                                / pair_cd.gamma,
                                            ci_norm: [1.0; 10],
                                            cj_norm: [1.0; 10],
                                        };
                                        let mut data = Eri3CenterData::init_from_pair(
                                            gamma_p,
                                            p_shell.center,
                                            &pp,
                                            max_p,
                                            max_ab,
                                            &mut grad_vrr_cache,
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
    fn packed_coulomb_matches_dense_for_mixed_shells() {
        let mol = Molecule::new(vec![
            Atom::from_bohr("O",[0.0;3]).unwrap(),
            Atom::from_bohr("H",[1.4,0.1,0.0]).unwrap(),
            Atom::from_bohr("H",[0.0,1.4,0.1]).unwrap(),
        ],0,1);
        let basis = BasisSet::from_name("6-31g(d)",&mol).unwrap();
        let n = basis.n_ao;
        let dense = RiJEngine::with_cache_limit(&mol,&basis,usize::MAX).unwrap();
        let mut packed = RiJEngine::with_cache_limit(&mol,&basis,dense.aux_basis.n_aux*n*(n+1)*4).unwrap();
        assert!(packed.packed_b.is_some());
        let mut density=Matrix2D::identity(n);
        density.set(0,2,0.12);
        density.set(2,0,-0.02);
        let expected=dense.build_coulomb(&basis,&density);
        for gpu in [false,true] {
            packed.use_gpu=gpu;
            let actual=packed.build_coulomb(&basis,&density);
            let error=actual.data.iter().zip(&expected.data).fold(0.0_f64,|a,(x,y)| a.max((x-y).abs()));
            assert!(error<1e-9,"packed J error {error}");
        }
    }

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

    #[test]
    fn test_gpu_eri3c_matches_cpu_to_machine_precision() {
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("S", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("C", [3.2, 0.1, -0.2]).unwrap(),
                Atom::from_bohr("F", [5.6, 0.4, 0.3]).unwrap(),
                Atom::from_bohr("H", [3.1, -1.9, 0.5]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("sto-3g", &mol).unwrap();
        // Force packed path via small cache_limit that still fits this molecule
        let cpu = RiJEngine::with_cache_limit_gpu(&mol, &bs, 4usize << 30, false).unwrap();
        // Build packed CPU vs GPU explicitly by calling GPU kernel on the same packed layout
        let aux_basis = &cpu.aux_basis;
        let n_aux = aux_basis.n_aux;
        let n_ao = bs.n_ao;
        let primary_bounds = shell_pair_bounds(&bs);
        let n_sh = bs.shells.len();
        let max_b = cpu.aux_diag_bounds.iter().copied().fold(0.0_f64, f64::max);

        let mut sig_ij = Vec::new();
        let mut col_off = 0usize;
        for (i, a) in bs.shells.iter().enumerate() {
            for (j, b) in bs.shells[..=i].iter().enumerate() {
                let q_ij = primary_bounds[i * n_sh + j];
                if max_b * q_ij < 1e-12 { continue; }
                let pps = precompute_prim_pairs_3c(a, b);
                if pps.is_empty() { continue; }
                let n_cols = if i == j {
                    a.n_functions() * (a.n_functions() + 1) / 2
                } else {
                    a.n_functions() * b.n_functions()
                };
                sig_ij.push((i, j, q_ij, col_off, pps));
                col_off += n_cols;
            }
        }
        let width = col_off;

        // CPU reference packed tensor
        let mut cpu_b = vec![0.0f64; n_aux * width];
        let mut blk = Vec::new();
        let mut vrr = Vec::new();
        for (p_idx, p_sh) in aux_basis.shells.iter().enumerate() {
            let np = p_sh.n_functions();
            let pb = cpu.aux_diag_bounds[p_idx];
            for &(i, j, q_ij, off, ref pps) in &sig_ij {
                if pb * q_ij < 1e-12 { continue; }
                let (a, b) = (&bs.shells[i], &bs.shells[j]);
                let (ni, nj) = (a.n_functions(), b.n_functions());
                shell_3center_block_precomputed(p_sh, a, b, pps, &mut blk, &mut vrr);
                for pf in 0..np {
                    let row = (p_sh.ao_offset + pf) * width + off;
                    let p_blk = pf * ni * nj;
                    if i == j {
                        let mut pos = 0;
                        for mu in 0..ni {
                            for nu in 0..=mu {
                                cpu_b[row + pos] = blk[p_blk + mu * nj + nu];
                                pos += 1;
                            }
                        }
                    } else {
                        cpu_b[row..row + ni * nj].copy_from_slice(&blk[p_blk..p_blk + ni * nj]);
                    }
                }
            }
        }

        // GPU packed tensor
        let mut gpu_aux_sh = Vec::new();
        let mut gpu_aux_pr = Vec::new();
        for (p_idx, p_sh) in aux_basis.shells.iter().enumerate() {
            let prim_start = gpu_aux_pr.len() as i32;
            let comps = p_sh.components();
            for (pp_i, &gp) in p_sh.exponents.iter().enumerate() {
                let c_base = p_sh.coeffs[pp_i];
                let mut cp_norm = [0.0f64; 6];
                for (pf, &lmn) in comps.iter().enumerate() {
                    cp_norm[pf] = c_base * primitive_norm(gp, lmn);
                }
                gpu_aux_pr.push(calchem_cuda::AuxPrimGpu { gamma_p: gp, cp_norm });
            }
            gpu_aux_sh.push(calchem_cuda::AuxShellGpu {
                center: p_sh.center,
                bound: cpu.aux_diag_bounds[p_idx],
                lp: p_sh.shell_type.angular_momentum() as i32,
                ao_offset: p_sh.ao_offset as i32,
                prim_start,
                n_prims: p_sh.exponents.len() as i32,
            });
        }
        let mut gpu_sp = Vec::new();
        let mut gpu_pp = Vec::new();
        for &(i, j, q_ij, off, ref pps) in &sig_ij {
            let prim_start = gpu_pp.len() as i32;
            let ni = bs.shells[i].n_functions();
            let nj = bs.shells[j].n_functions();
            for pp in pps {
                let mut ci_norm = [0.0f64; 6];
                ci_norm[..ni].copy_from_slice(&pp.ci_norm[..ni]);
                let mut cj_norm = [0.0f64; 6];
                cj_norm[..nj].copy_from_slice(&pp.cj_norm[..nj]);
                gpu_pp.push(calchem_cuda::PrimPair3CGpu {
                    q: pp.q,
                    p_center: pp.p_center,
                    qc: pp.qc,
                    qd: pp.qd,
                    two_pi25_kab_over_q: pp.two_pi25_kab_over_q,
                    ci_norm,
                    cj_norm,
                });
            }
            gpu_sp.push(calchem_cuda::ShellPair3CGpu {
                q_ij,
                li: bs.shells[i].shell_type.angular_momentum() as i32,
                lj: bs.shells[j].shell_type.angular_momentum() as i32,
                same_shell: if i == j { 1 } else { 0 },
                col_offset: off as i32,
                prim_start,
                n_prims: pps.len() as i32,
            });
        }

        let gpu_tensor = calchem_cuda::ResidentRijTensor::build_eri3c_gpu(
            &gpu_aux_sh, &gpu_aux_pr, &gpu_sp, &gpu_pp, n_aux, width,
        ).expect("GPU 3-center ERI builder should succeed on CUDA device");
        let gpu_b = gpu_tensor.as_slice();
        let _ = n_ao;

        let mut max_err = 0.0f64;
        let mut worst = (0usize, 0usize, 0.0f64, 0.0f64);
        for p in 0..n_aux {
            for c in 0..width {
                let idx = p * width + c;
                let err = (gpu_b[idx] - cpu_b[idx]).abs();
                if err > max_err {
                    max_err = err;
                    worst = (p, c, gpu_b[idx], cpu_b[idx]);
                }
            }
        }
        assert!(
            max_err < 1e-12,
            "GPU vs CPU 3-center ERI mismatch: max_err={max_err:.3e} at (p={}, c={}) gpu={:.12e} cpu={:.12e}",
            worst.0, worst.1, worst.2, worst.3
        );
    }

    #[test]
    fn test_gpu_eri3c_d_orbitals_matches_cpu_to_machine_precision() {
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("S", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("C", [3.2, 0.1, -0.2]).unwrap(),
                Atom::from_bohr("F", [5.6, 0.4, 0.3]).unwrap(),
                Atom::from_bohr("H", [3.1, -1.9, 0.5]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        // Verify primary basis contains D-shells
        assert!(bs.shells.iter().any(|s| s.shell_type.angular_momentum() == 2));

        let cpu = RiJEngine::with_cache_limit_gpu(&mol, &bs, 4usize << 30, false).unwrap();
        let aux_basis = &cpu.aux_basis;
        let n_aux = aux_basis.n_aux;
        let n_ao = bs.n_ao;
        let primary_bounds = shell_pair_bounds(&bs);
        let n_sh = bs.shells.len();
        let max_b = cpu.aux_diag_bounds.iter().copied().fold(0.0_f64, f64::max);

        let mut sig_ij = Vec::new();
        let mut col_off = 0usize;
        for (i, a) in bs.shells.iter().enumerate() {
            for (j, b) in bs.shells[..=i].iter().enumerate() {
                let q_ij = primary_bounds[i * n_sh + j];
                if max_b * q_ij < 1e-12 { continue; }
                let pps = precompute_prim_pairs_3c(a, b);
                if pps.is_empty() { continue; }
                let n_cols = if i == j {
                    a.n_functions() * (a.n_functions() + 1) / 2
                } else {
                    a.n_functions() * b.n_functions()
                };
                sig_ij.push((i, j, q_ij, col_off, pps));
                col_off += n_cols;
            }
        }
        let width = col_off;

        // CPU reference packed tensor
        let mut cpu_b = vec![0.0f64; n_aux * width];
        let mut blk = Vec::new();
        let mut vrr = Vec::new();
        for (p_idx, p_sh) in aux_basis.shells.iter().enumerate() {
            let np = p_sh.n_functions();
            let pb = cpu.aux_diag_bounds[p_idx];
            for &(i, j, q_ij, off, ref pps) in &sig_ij {
                if pb * q_ij < 1e-12 { continue; }
                let (a, b) = (&bs.shells[i], &bs.shells[j]);
                let (ni, nj) = (a.n_functions(), b.n_functions());
                shell_3center_block_precomputed(p_sh, a, b, pps, &mut blk, &mut vrr);
                for pf in 0..np {
                    let row = (p_sh.ao_offset + pf) * width + off;
                    let p_blk = pf * ni * nj;
                    if i == j {
                        let mut pos = 0;
                        for mu in 0..ni {
                            for nu in 0..=mu {
                                cpu_b[row + pos] = blk[p_blk + mu * nj + nu];
                                pos += 1;
                            }
                        }
                    } else {
                        cpu_b[row..row + ni * nj].copy_from_slice(&blk[p_blk..p_blk + ni * nj]);
                    }
                }
            }
        }

        // GPU packed tensor
        let mut gpu_aux_sh = Vec::new();
        let mut gpu_aux_pr = Vec::new();
        for (p_idx, p_sh) in aux_basis.shells.iter().enumerate() {
            let prim_start = gpu_aux_pr.len() as i32;
            let comps = p_sh.components();
            for (pp_i, &gp) in p_sh.exponents.iter().enumerate() {
                let c_base = p_sh.coeffs[pp_i];
                let mut cp_norm = [0.0f64; 6];
                for (pf, &lmn) in comps.iter().enumerate() {
                    cp_norm[pf] = c_base * primitive_norm(gp, lmn);
                }
                gpu_aux_pr.push(calchem_cuda::AuxPrimGpu { gamma_p: gp, cp_norm });
            }
            gpu_aux_sh.push(calchem_cuda::AuxShellGpu {
                center: p_sh.center,
                bound: cpu.aux_diag_bounds[p_idx],
                lp: p_sh.shell_type.angular_momentum() as i32,
                ao_offset: p_sh.ao_offset as i32,
                prim_start,
                n_prims: p_sh.exponents.len() as i32,
            });
        }
        let mut gpu_sp = Vec::new();
        let mut gpu_pp = Vec::new();
        for &(i, j, q_ij, off, ref pps) in &sig_ij {
            let prim_start = gpu_pp.len() as i32;
            let ni = bs.shells[i].n_functions();
            let nj = bs.shells[j].n_functions();
            for pp in pps {
                let mut ci_norm = [0.0f64; 6];
                ci_norm[..ni].copy_from_slice(&pp.ci_norm[..ni]);
                let mut cj_norm = [0.0f64; 6];
                cj_norm[..nj].copy_from_slice(&pp.cj_norm[..nj]);
                gpu_pp.push(calchem_cuda::PrimPair3CGpu {
                    q: pp.q,
                    p_center: pp.p_center,
                    qc: pp.qc,
                    qd: pp.qd,
                    two_pi25_kab_over_q: pp.two_pi25_kab_over_q,
                    ci_norm,
                    cj_norm,
                });
            }
            gpu_sp.push(calchem_cuda::ShellPair3CGpu {
                q_ij,
                li: bs.shells[i].shell_type.angular_momentum() as i32,
                lj: bs.shells[j].shell_type.angular_momentum() as i32,
                same_shell: if i == j { 1 } else { 0 },
                col_offset: off as i32,
                prim_start,
                n_prims: pps.len() as i32,
            });
        }

        let gpu_tensor = calchem_cuda::ResidentRijTensor::build_eri3c_gpu(
            &gpu_aux_sh, &gpu_aux_pr, &gpu_sp, &gpu_pp, n_aux, width,
        ).expect("GPU 3-center ERI builder should succeed for D orbitals on CUDA device");
        let gpu_b = gpu_tensor.as_slice();
        let _ = n_ao;

        let mut max_err = 0.0f64;
        let mut worst = (0usize, 0usize, 0.0f64, 0.0f64);
        for p in 0..n_aux {
            for c in 0..width {
                let idx = p * width + c;
                let err = (gpu_b[idx] - cpu_b[idx]).abs();
                if err > max_err {
                    max_err = err;
                    worst = (p, c, gpu_b[idx], cpu_b[idx]);
                }
            }
        }
        println!("[test_gpu_eri3c_d_orbitals] width={width} n_aux={n_aux} max_err={max_err:.3e}");
        assert!(
            max_err < 1e-11,
            "GPU vs CPU 3-center ERI mismatch for D orbitals: max_err={max_err:.3e} at (p={}, c={}) gpu={:.12e} cpu={:.12e}",
            worst.0, worst.1, worst.2, worst.3
        );
    }
}
