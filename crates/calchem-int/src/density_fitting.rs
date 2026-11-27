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
        let aux_max_l_cap: u8 = std::env::var("CALCHEM_AUTO_AUX_MAX_L")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        Self::build_auto_jfit_with_max_l(mol, primary, aux_max_l_cap)
    }

    /// Constructs an uncontracted per-angular-momentum even-tempered RI-J auxiliary basis
    /// (AutoAux-style) with an explicit maximum auxiliary angular momentum cap (`aux_max_l_cap`,
    /// e.g. `3` for up to Cartesian F-shells, `2` for S/P/D-only GPU ERI3C kernels).
    pub fn build_auto_jfit_with_max_l(mol: &Molecule, primary: &BasisSet, aux_max_l_cap: u8) -> Self {
        let mut shells = Vec::new();
        let mut ao_offset = 0;

        let mut push_ladder = |atom_idx: usize,
                               center: [f64; 3],
                               st: ShellType,
                               lo: f64,
                               hi: f64,
                               ratio: f64| {
            if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && hi >= lo && ratio > 1.05) {
                return;
            }
            let mut cur = lo;
            while cur <= hi * 1.0001 {
                shells.push(AuxiliaryShell {
                    atom_index: atom_idx,
                    center,
                    shell_type: st,
                    exponents: vec![cur],
                    coeffs: vec![1.0],
                    ao_offset,
                });
                ao_offset += st.n_ao();
                cur *= ratio;
            }
        };

        let primary_max_l: u8 = primary
            .shells
            .iter()
            .map(|s| s.shell_type.angular_momentum())
            .max()
            .unwrap_or(0);

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
            let mut l_min = [f64::INFINITY; 4];
            let mut l_max = [0.0_f64; 4];

            for s in &atom_shells {
                let l = s.shell_type.angular_momentum();
                if l > max_l {
                    max_l = l;
                }
                let l_idx = (l as usize).min(3);
                for p in &s.primitives {
                    if p.exponent < min_exp {
                        min_exp = p.exponent;
                    }
                    if p.exponent > max_exp {
                        max_exp = p.exponent;
                    }
                    if p.exponent < l_min[l_idx] {
                        l_min[l_idx] = p.exponent;
                    }
                    if p.exponent > l_max[l_idx] {
                        l_max[l_idx] = p.exponent;
                    }
                }
            }

            if !min_exp.is_finite() || min_exp <= 0.0 {
                min_exp = 0.1;
            }
            if max_exp <= min_exp {
                max_exp = min_exp * 5.0;
            }

            // Compact stride for >30-atom globally unpolarized S/P molecules keeps N_aux bounded for GPU ERI3C.
            let compact_large = mol.atoms.len() > 30 && primary_max_l <= 1;

            // L = 0 (S): spans full orbital-product exponent range [1.55*min_exp, 2.0*max_exp].
            let s_lo = min_exp * 1.55;
            let s_hi = max_exp * 2.0;
            let ratio_s = if compact_large {
                2.45_f64.max((s_hi / s_lo).powf(1.0 / 9.0))
            } else {
                2.08_f64
            };
            push_ladder(atom_idx, center, ShellType::S, s_lo, s_hi, ratio_s);

            // L = 1 (P): fits s*p and p*d products (plus compact valence P on pure-S atoms like H).
            if aux_max_l_cap >= 1 {
                if max_l == 0 {
                    let p_lo = (min_exp * 1.80).max(0.35);
                    let p_hi = (max_exp * 1.50).min(15.0).max(p_lo);
                    let ratio_p = if compact_large { 2.80 } else { 2.45 };
                    push_ladder(atom_idx, center, ShellType::P, p_lo, p_hi, ratio_p);
                } else {
                    let p_min_prim = l_min[1].min(min_exp);
                    let p_lo = (p_min_prim * 1.85).max(0.35);
                    let p_hi_raw = (l_max[1] * 2.2).max(l_max[1] + l_max[0].min(60.0));
                    let p_hi = if compact_large {
                        p_hi_raw.min(45.0)
                    } else {
                        p_hi_raw.min(160.0)
                    }
                    .max(p_lo);
                    let ratio_p = if compact_large { 2.80 } else { 2.35 };
                    push_ladder(atom_idx, center, ShellType::P, p_lo, p_hi, ratio_p);
                }
            }

            // L = 2 (Cartesian D): fits p*p and s*d products. Geometric ratio 2.55, floor >= 0.42,
            // and cap <= 120.0 prevent Cartesian r^2 S-trace linear dependence against the S ladder
            // (keeping trunc == 0 on 3rd-row elements like S and on 48-atom pi-stacked dimers).
            if max_l >= 1 && aux_max_l_cap >= 2 && l_min[1].is_finite() {
                let d_lo = (l_min[1] * 1.90).max(0.42);
                let d_hi_raw = if l_max[2] > 0.0 {
                    (l_max[1] * 2.0).max(l_max[2] * 2.5)
                } else {
                    l_max[1] * 2.0
                };
                let d_hi = if compact_large {
                    d_hi_raw.min(12.0)
                } else {
                    d_hi_raw.min(120.0)
                }
                .max(d_lo);
                let ratio_d = if compact_large { 2.90 } else { 2.55 };
                push_ladder(atom_idx, center, ShellType::D, d_lo, d_hi, ratio_d);
            }

            // L = 3 (Cartesian F): fits p*d polarization products whenever primary has d-shells.
            // Unpolarized S/P bases (max_l <= 1) automatically stay at L_aux <= 2 for GPU ERI3C.
            if max_l >= 2 && aux_max_l_cap >= 3 && l_min[2].is_finite() {
                let p_min_eff = if l_min[1].is_finite() { l_min[1] } else { min_exp };
                let p_max_eff = if l_max[1] > 0.0 { l_max[1].min(12.0) } else { 4.0 };
                let f_lo = (p_min_eff + l_min[2]).max(0.75);
                let f_hi = ((p_max_eff + l_max[2]) * 1.20).max(f_lo * 2.50);
                push_ladder(atom_idx, center, ShellType::F, f_lo, f_hi, 2.60);
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

struct GpuRijStreamState {
    streamer: calchem_cuda::GpuRijStreamer,
    chunks: Vec<(usize, usize, usize, usize)>,
}

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
    gpu_streamer: Option<GpuRijStreamState>,
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
        Self::with_basis(mol, primary, use_gpu, calchem_model::JFitBasis::Auto)
    }

    pub fn with_basis(mol: &Molecule, primary: &BasisSet, use_gpu: bool, choice: calchem_model::JFitBasis) -> Result<Self,CalcError> {
        let cache_limit = std::env::var("CALCHEM_MAX_DF_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize << 30);
        Self::build(mol, primary, cache_limit, use_gpu, choice, true)
    }

    /// Lightweight RI-J engine constructor for standalone gradient calls: skips building
    /// or caching the full 3-center `B` tensor in host/GPU memory while keeping full metric
    /// factorization and GPU 3c gradient support.
    pub fn with_basis_for_gradient(mol: &Molecule, primary: &BasisSet, use_gpu: bool, choice: calchem_model::JFitBasis) -> Result<Self, CalcError> {
        Self::build(mol, primary, 0, use_gpu, choice, false)
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
        Self::build(mol, primary, cache_limit, use_gpu, calchem_model::JFitBasis::Auto, true)
    }

    fn build(mol: &Molecule, primary: &BasisSet, cache_limit: usize, use_gpu: bool, choice: calchem_model::JFitBasis, build_3c_cache: bool) -> Result<Self, CalcError> {
        let aux_basis = match choice {
            calchem_model::JFitBasis::Auto => AuxiliaryBasisSet::build_auto_jfit(mol, primary),
            calchem_model::JFitBasis::Def2UpToF => {
                let aux = AuxiliaryBasisSet::build_def2_jfit(mol, primary);
                if aux.name.ends_with("-hybrid") {
                    return Err(CalcError::UnsupportedBasis("Requested def2-jfit-f has missing elements".into()));
                }
                aux
            }
        };
        let budget = std::env::var("CALCHEM_MAX_MEMORY_BYTES")
            .ok()
            .and_then(|v| v.parse::<u128>().ok())
            .unwrap_or(8 * 1024 * 1024 * 1024);
        // Dense metric Cholesky/eigensolver working estimate in N_aux^2 f64s.
        let metric_bytes = 6 * (aux_basis.n_aux as u128).pow(2) * 8;
        if metric_bytes > budget {
            return Err(CalcError::ResourceLimit(format!(
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

        if !build_3c_cache {
            return Ok(Self {
                use_gpu,
                aux_basis,
                metric_factor,
                metric_truncated_modes,
                cached_b: None,
                packed_b: None,
                gpu_streamer: None,
                packed_pairs: Vec::new(),
                n_ao,
                aux_diag_bounds,
                gpu_kernel_invocations: AtomicUsize::new(0),
                gpu_fallback_invocations: AtomicUsize::new(0),
            });
        }

        let has_unsupported_gpu_l = aux_basis
            .shells
            .iter()
            .any(|s| s.shell_type.angular_momentum() > 3)
            || primary
                .shells
                .iter()
                .any(|s| s.shell_type.angular_momentum() > 2);
        let cuda_avail = calchem_cuda::is_cuda_available();

        let primary_bounds = shell_pair_bounds(primary);
        let n_primary_shells = primary.shells.len();
        let mut packed_pairs = Vec::new();
        let mut shell_offsets = vec![usize::MAX; n_primary_shells*n_primary_shells];
        let packed = (use_gpu && cuda_avail && !has_unsupported_gpu_l)
            || req_bytes > max_df_bytes.min(256 << 20);
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
            && cuda_avail
            && !has_unsupported_gpu_l;

        let mut gpu_packed_b: Option<calchem_cuda::ResidentRijTensor> = None;
        let mut gpu_streamer: Option<GpuRijStreamState> = None;
        if gpu_eligible {
            let mut gpu_aux_shells = Vec::with_capacity(aux_basis.shells.len());
            let mut gpu_aux_prims = Vec::new();
            for (p_idx, p_shell) in aux_basis.shells.iter().enumerate() {
                let prim_start = gpu_aux_prims.len() as i32;
                let p_comps = p_shell.components();
                for (pp_i, &gamma_p) in p_shell.exponents.iter().enumerate() {
                    let c_base = p_shell.coeffs[pp_i];
                    let mut cp_norm = [0.0f64; 10];
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

            let gpu_limit = calchem_cuda::resolve_gpu_memory_limit();
            let vram_res_cap = (gpu_limit.saturating_sub(512usize << 20))
                .max(256usize << 20)
                .min(gpu_limit);
            let resident_cap = (max_df_bytes as u128).min(vram_res_cap as u128);
            if bytes <= resident_cap {
                gpu_packed_b = calchem_cuda::ResidentRijTensor::build_eri3c_gpu_resident(
                    &gpu_aux_shells,
                    &gpu_aux_prims,
                    &gpu_shell_pairs,
                    &gpu_prim_pairs,
                    n_aux,
                    width,
                );
            }
            if gpu_packed_b.is_none() {
                let row_bytes = width * 8;
                let max_sh_n = aux_basis
                    .shells
                    .iter()
                    .map(|s| s.n_functions())
                    .max()
                    .unwrap_or(1);
                let chunk_cap = (512usize << 20)
                    .min((gpu_limit / 4).max(16usize << 20))
                    .min(max_df_bytes.max(row_bytes * max_sh_n));
                let mut chunks = Vec::new();
                let mut sh_start = 0usize;
                let mut ao_start = 0usize;
                let mut cur_sh = 0usize;
                let mut cur_aux = 0usize;
                for sh in &aux_basis.shells {
                    let nf = sh.n_functions();
                    if cur_sh > 0 && (cur_aux + nf) * row_bytes > chunk_cap {
                        chunks.push((sh_start, cur_sh, ao_start, cur_aux));
                        sh_start += cur_sh;
                        ao_start += cur_aux;
                        cur_sh = 0;
                        cur_aux = 0;
                    }
                    cur_sh += 1;
                    cur_aux += nf;
                }
                if cur_sh > 0 {
                    chunks.push((sh_start, cur_sh, ao_start, cur_aux));
                }
                let max_chunk_n_aux = chunks.iter().map(|c| c.3).max().unwrap_or(0);
                if max_chunk_n_aux > 0 {
                    if let Some(streamer) = calchem_cuda::GpuRijStreamer::upload(
                        &gpu_aux_shells,
                        &gpu_aux_prims,
                        &gpu_shell_pairs,
                        &gpu_prim_pairs,
                        max_chunk_n_aux,
                        width,
                    ) {
                        if profile {
                            eprintln!(
                                "[rij] gpu_streamer chunks={} max_chunk_n_aux={} width={}",
                                chunks.len(),
                                max_chunk_n_aux,
                                width
                            );
                        }
                        gpu_streamer = Some(GpuRijStreamState { streamer, chunks });
                    }
                }
            }
        }

        let (cached_b, packed_b, initial_gpu_kernels, initial_gpu_fallbacks) =
            if let Some(tensor) = gpu_packed_b {
                if profile {
                    eprintln!("[rij] eri_3c_build_gpu={:.3}s", t_3c0.elapsed().as_secs_f64());
                }
                (None, Some(tensor), 1usize, 0usize)
            } else if gpu_streamer.is_some() {
                (None, None, 0usize, 0usize)
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
                let fallback_inc = if use_gpu
                    && width > 0
                    && (has_unsupported_gpu_l
                        || gpu_eligible
                        || (packed && !cuda_avail))
                {
                    1usize
                } else {
                    0usize
                };
                if packed {
                    (
                        None,
                        cache.map(calchem_cuda::ResidentRijTensor::new),
                        0usize,
                        fallback_inc,
                    )
                } else {
                    (cache, None, 0usize, fallback_inc)
                }
            };
        Ok(Self {
            use_gpu,
            aux_basis,
            metric_factor,
            metric_truncated_modes,
            cached_b, packed_b, gpu_streamer, packed_pairs,
            n_ao,
            aux_diag_bounds,
            gpu_kernel_invocations: AtomicUsize::new(initial_gpu_kernels),
            gpu_fallback_invocations: AtomicUsize::new(initial_gpu_fallbacks),
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
            cached_b: None, packed_b: None, gpu_streamer: None, packed_pairs: Vec::new(),
            n_ao,
            aux_diag_bounds,
            gpu_kernel_invocations: AtomicUsize::new(0),
            gpu_fallback_invocations: AtomicUsize::new(0),
        })
    }

    #[inline]
    pub fn has_cached_tensor(&self) -> bool {
        self.cached_b.is_some() || self.packed_b.is_some() || self.gpu_streamer.is_some()
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

        if let Some(ref packed_tensor) = self.packed_b {
            let compact = self.pack_density(density);
            let width = compact.len();
            let gpu_ready = self.use_gpu && calchem_cuda::is_cuda_available();
            let mut v_vec = vec![0.0; n_aux];
            let mut used_gpu_v = false;
            let mut fallback_recorded = false;
            if gpu_ready {
                used_gpu_v = packed_tensor.view().contract_v(&compact, &mut v_vec, n_aux, width);
                if used_gpu_v {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                    fallback_recorded = true;
                }
            } else if self.use_gpu && packed_tensor.is_device_only() {
                self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                fallback_recorded = true;
            }
            if !used_gpu_v && !packed_tensor.is_device_only() {
                let b_tensor = packed_tensor.as_slice();
                v_vec = (0..n_aux)
                    .into_par_iter()
                    .map(|p| {
                        let b_slice = &b_tensor[p * width..(p + 1) * width];
                        let mut sum = 0.0;
                        for k in 0..width {
                            sum += b_slice[k] * compact[k];
                        }
                        sum
                    })
                    .collect();
                used_gpu_v = true; // host slice valid
            }
            if used_gpu_v {
                let x_vec = self.metric_factor.solve(&v_vec);
                let mut j_data = vec![0.0; width];
                let mut used_gpu_j = false;
                if self.use_gpu {
                    if gpu_ready {
                        used_gpu_j = packed_tensor.view().contract_coulomb(&x_vec, &mut j_data, n_aux, width);
                    }
                    if used_gpu_j {
                        self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                    } else if !fallback_recorded {
                        self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                        fallback_recorded = true;
                    }
                }
                if !used_gpu_j && !packed_tensor.is_device_only() {
                    let b_tensor = packed_tensor.as_slice();
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
                    used_gpu_j = true;
                }
                if used_gpu_j {
                    let mut j = Matrix2D::zeros(n_ao, n_ao);
                    for (&(uv, vu), value) in self.packed_pairs.iter().zip(j_data) {
                        j.data[uv] = value;
                        j.data[vu] = value;
                    }
                    return j;
                }
            }
            if self.use_gpu && !fallback_recorded {
                self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
            }
        } else if let Some(ref stream_state) = self.gpu_streamer {
            let compact = self.pack_density(density);
            let width = compact.len();
            let mut v_vec = vec![0.0; n_aux];
            let mut ok = stream_state.streamer.pass1_begin(&compact);
            if ok {
                for &(sh_start, sh_len, ao_start, chunk_n_aux) in &stream_state.chunks {
                    if stream_state.streamer.pass1_v_resident(
                        sh_start,
                        sh_len,
                        ao_start,
                        &mut v_vec[ao_start..ao_start + chunk_n_aux],
                    ) {
                        self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                    } else {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                let x_vec = self.metric_factor.solve(&v_vec);
                let mut j_data = vec![0.0; width];
                if stream_state.streamer.pass2_begin() {
                    for &(sh_start, sh_len, ao_start, chunk_n_aux) in &stream_state.chunks {
                        if stream_state.streamer.pass2_accum(
                            &x_vec[ao_start..ao_start + chunk_n_aux],
                            sh_start,
                            sh_len,
                            ao_start,
                        ) {
                            self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok && stream_state.streamer.pass2_finish(&mut j_data) {
                        let mut j = Matrix2D::zeros(n_ao, n_ao);
                        for (&(uv, vu), value) in self.packed_pairs.iter().zip(j_data) {
                            j.data[uv] = value;
                            j.data[vu] = value;
                        }
                        return j;
                    }
                }
            }
            if self.use_gpu {
                self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
            }
        } else if let Some(b_tensor) = self.cached_b.as_deref() {
            let density_data = density.data.as_slice();
            let width = density_data.len();
            let gpu_tensor = calchem_cuda::RijTensorView::from(b_tensor);
            let gpu_ready = self.use_gpu && calchem_cuda::is_cuda_available();

            let mut v_vec = vec![0.0; n_aux];
            let mut used_gpu_v = false;
            let mut fallback_recorded = false;
            if gpu_ready {
                used_gpu_v = gpu_tensor.contract_v(density_data, &mut v_vec, n_aux, width);
                if used_gpu_v {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                    fallback_recorded = true;
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

            let x_vec = self.metric_factor.solve(&v_vec);
            let mut j_data = vec![0.0; width];
            let mut used_gpu = false;
            if self.use_gpu {
                if gpu_ready {
                    used_gpu = gpu_tensor.contract_coulomb(&x_vec, &mut j_data, n_aux, width);
                }
                if used_gpu {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                } else if !fallback_recorded {
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

            return Matrix2D {
                rows: n_ao,
                cols: n_ao,
                data: j_data,
            };
        } else if self.use_gpu {
            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
        }

        // Screened parallel on-the-fly path for large systems
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
        Ok(self.compute_coulomb_gradient_with_telemetry(n_atoms, primary, density)?.0)
    }

    /// Evaluates the RI-J Coulomb nuclear gradient and returns `(grad, gpu_kernels, gpu_fallbacks)`.
    pub fn compute_coulomb_gradient_with_telemetry(
        &self,
        n_atoms: usize,
        primary: &BasisSet,
        density: &Matrix2D,
    ) -> Result<(Vec<[f64; 3]>, usize, usize), CalcError> {
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

        let mut gpu_k = 0usize;
        let mut gpu_fb = 0usize;
        let mut fallback_recorded = false;
        let max_aux_bound = self.aux_diag_bounds.iter().copied().fold(0.0_f64, f64::max);
        let has_grad3c_work = !self.aux_basis.shells.is_empty()
            && (0..n_primary_shells).any(|i_idx| {
                (0..=i_idx).any(|j_idx| max_aux_bound * primary_bounds[i_idx * n_primary_shells + j_idx] >= 1e-12)
            });
        let has_unsupported_gpu_l = self
            .aux_basis
            .shells
            .iter()
            .any(|s| s.shell_type.angular_momentum() > 3)
            || primary
                .shells
                .iter()
                .any(|s| s.shell_type.angular_momentum() > 2);
        let cuda_avail = calchem_cuda::is_cuda_available();

        // Build Grad3c GPU metadata lazily if GPU is active and angular momenta are supported.
        struct Grad3cGpuMeta {
            aux_shells: Vec<calchem_cuda::Grad3cAuxShellGpu>,
            aux_prims: Vec<calchem_cuda::Grad3cAuxPrimGpu>,
            shell_pairs: Vec<calchem_cuda::Grad3cShellPairGpu>,
            prim_pairs: Vec<calchem_cuda::Grad3cPrimPairGpu>,
        }
        let grad3c_meta: Option<Grad3cGpuMeta> = if self.use_gpu && cuda_avail && !has_unsupported_gpu_l && n_atoms > 1 {
            let max_bound = self.aux_diag_bounds.iter().copied().fold(0.0_f64, f64::max);
            let mut aux_shells = Vec::with_capacity(self.aux_basis.shells.len());
            let mut aux_prims = Vec::new();
            for (p_idx, p_shell) in self.aux_basis.shells.iter().enumerate() {
                let prim_start = aux_prims.len() as i32;
                let lp = p_shell.shell_type.angular_momentum();
                let p_comps = p_shell.components();
                let lead_norm_1 = primitive_norm(1.0, [lp, 0, 0]);
                let mut comp_scales = [0.0f64; 10];
                for (pf, &lmn) in p_comps.iter().enumerate() {
                    comp_scales[pf] = primitive_norm(1.0, lmn) / lead_norm_1;
                }
                for (pp_i, &gamma_p) in p_shell.exponents.iter().enumerate() {
                    let cp_lead = p_shell.coeffs[pp_i] * primitive_norm(gamma_p, [lp, 0, 0]);
                    aux_prims.push(calchem_cuda::Grad3cAuxPrimGpu { gamma_p, cp_lead });
                }
                aux_shells.push(calchem_cuda::Grad3cAuxShellGpu {
                    center: p_shell.center,
                    bound: self.aux_diag_bounds[p_idx],
                    comp_scales,
                    lp: lp as i32,
                    ao_offset: p_shell.ao_offset as i32,
                    prim_start,
                    n_prims: p_shell.exponents.len() as i32,
                    atom_index: p_shell.atom_index as i32,
                    _pad: 0,
                });
            }

            let two_pi25 = 2.0 * PI.powf(2.5);
            let mut shell_pairs = Vec::new();
            let mut prim_pairs = Vec::new();
            for i_idx in 0..n_primary_shells {
                let i_shell = &primary.shells[i_idx];
                let li = i_shell.shell_type.angular_momentum();
                let mut comp_scales_i = [0.0f64; 6];
                if let Some(p0) = i_shell.primitives.first() {
                    if p0.norm > 0.0 {
                        for (ii, &c) in i_shell.components().iter().enumerate() {
                            comp_scales_i[ii] = i_shell.primitive_norm_for(0, c) / p0.norm;
                        }
                    }
                }
                for j_idx in 0..=i_idx {
                    let q_ij = primary_bounds[i_idx * n_primary_shells + j_idx];
                    if max_bound * q_ij < 1e-12 {
                        continue;
                    }
                    let j_shell = &primary.shells[j_idx];
                    let lj = j_shell.shell_type.angular_momentum();
                    let mut comp_scales_j = [0.0f64; 6];
                    if let Some(p0) = j_shell.primitives.first() {
                        if p0.norm > 0.0 {
                            for (jj, &c) in j_shell.components().iter().enumerate() {
                                comp_scales_j[jj] = j_shell.primitive_norm_for(0, c) / p0.norm;
                            }
                        }
                    }
                    let prim_start = prim_pairs.len() as i32;
                    for prim_i in &i_shell.primitives {
                        let wi = prim_i.coeff * prim_i.norm;
                        for prim_j in &j_shell.primitives {
                            let pair_cd = GaussianPair::new(
                                prim_i.exponent,
                                i_shell.center,
                                prim_j.exponent,
                                j_shell.center,
                            );
                            if pair_cd.kab < 1e-18 {
                                continue;
                            }
                            let wj = prim_j.coeff * prim_j.norm;
                            prim_pairs.push(calchem_cuda::Grad3cPrimPairGpu {
                                q: pair_cd.gamma,
                                alpha: prim_i.exponent,
                                beta: prim_j.exponent,
                                two_pi25_kab_w_over_q: two_pi25 * pair_cd.kab * wi * wj / pair_cd.gamma,
                                p_center: pair_cd.p,
                                qc: pair_cd.pa,
                            });
                        }
                    }
                    let n_prims = prim_pairs.len() as i32 - prim_start;
                    if n_prims > 0 {
                        shell_pairs.push(calchem_cuda::Grad3cShellPairGpu {
                            q_ij,
                            ab_vec: [
                                i_shell.center[0] - j_shell.center[0],
                                i_shell.center[1] - j_shell.center[1],
                                i_shell.center[2] - j_shell.center[2],
                            ],
                            comp_scales_i,
                            comp_scales_j,
                            li: li as i32,
                            lj: lj as i32,
                            same_shell: if i_idx == j_idx { 1 } else { 0 },
                            atom_i: i_shell.atom_index as i32,
                            atom_j: j_shell.atom_index as i32,
                            ao_off_i: i_shell.ao_offset as i32,
                            ao_off_j: j_shell.ao_offset as i32,
                            prim_start,
                            n_prims,
                            _pad: 0,
                        });
                    }
                }
            }
            if !aux_shells.is_empty() && !aux_prims.is_empty() && !shell_pairs.is_empty() && !prim_pairs.is_empty() {
                Some(Grad3cGpuMeta {
                    aux_shells,
                    aux_prims,
                    shell_pairs,
                    prim_pairs,
                })
            } else {
                None
            }
        } else {
            None
        };

        // 1. Compute v_P = sum_{mu,nu} (P|mu nu) P_{mu,nu}
        let mut gpu_v_opt: Option<Vec<f64>> = None;
        if let Some(packed) = &self.packed_b {
            let d = self.pack_density(density);
            if packed.is_device_only() {
                if self.use_gpu && cuda_avail {
                    let mut v_gpu = vec![0.0; n_aux];
                    if packed.view().contract_v(&d, &mut v_gpu, n_aux, d.len()) {
                        gpu_k += 1;
                        gpu_v_opt = Some(v_gpu);
                    } else {
                        gpu_fb += 1;
                        fallback_recorded = true;
                    }
                } else if self.use_gpu {
                    gpu_fb += 1;
                    fallback_recorded = true;
                }
            } else {
                if self.use_gpu && cuda_avail {
                    let mut v_gpu = vec![0.0; n_aux];
                    if packed.view().contract_v(&d, &mut v_gpu, n_aux, d.len()) {
                        gpu_k += 1;
                        gpu_v_opt = Some(v_gpu);
                    } else {
                        gpu_fb += 1;
                        fallback_recorded = true;
                    }
                } else if self.use_gpu {
                    gpu_fb += 1;
                    fallback_recorded = true;
                }
                if gpu_v_opt.is_none() {
                    gpu_v_opt = Some(
                        packed
                            .as_slice()
                            .par_chunks_exact(d.len())
                            .map(|b| b.iter().zip(&d).map(|(x, y)| x * y).sum())
                            .collect(),
                    );
                }
            }
        } else if let Some(ref stream_state) = self.gpu_streamer {
            if self.use_gpu && cuda_avail {
                let d = self.pack_density(density);
                let mut v_gpu = vec![0.0; n_aux];
                let mut ok = stream_state.streamer.pass1_begin(&d);
                let mut chunk_k = 0usize;
                if ok {
                    for &(sh_start, sh_len, ao_start, chunk_n_aux) in &stream_state.chunks {
                        if !stream_state.streamer.pass1_v_resident(
                            sh_start,
                            sh_len,
                            ao_start,
                            &mut v_gpu[ao_start..ao_start + chunk_n_aux],
                        ) {
                            ok = false;
                            break;
                        }
                        chunk_k += 1;
                    }
                }
                if ok {
                    gpu_k += chunk_k;
                    gpu_v_opt = Some(v_gpu);
                } else {
                    gpu_fb += 1;
                    fallback_recorded = true;
                }
            } else if self.use_gpu {
                gpu_fb += 1;
                fallback_recorded = true;
            }
        } else if let Some(ref meta) = grad3c_meta {
            let mut v_gpu = vec![0.0; n_aux];
            if calchem_cuda::cuda_rij_grad_v(
                &meta.aux_shells,
                &meta.aux_prims,
                &meta.shell_pairs,
                &meta.prim_pairs,
                &density.data,
                &mut v_gpu,
                n_ao,
            ) {
                gpu_k += 1;
                gpu_v_opt = Some(v_gpu);
            } else {
                gpu_fb += 1;
                fallback_recorded = true;
            }
        } else if self.use_gpu && n_atoms > 1 && has_grad3c_work {
            gpu_fb += 1;
            fallback_recorded = true;
        }
        let v_vec: Vec<f64> = if let Some(v_gpu) = gpu_v_opt {
            v_gpu
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
        let mut grad_3c_opt: Option<Vec<[f64; 3]>> = None;
        if let Some(ref meta) = grad3c_meta {
            let mut grad_flat = vec![0.0f64; n_atoms * 3];
            if calchem_cuda::cuda_rij_grad_3c(
                &meta.aux_shells,
                &meta.aux_prims,
                &meta.shell_pairs,
                &meta.prim_pairs,
                &x_vec,
                &density.data,
                &mut grad_flat,
                n_ao,
            ) {
                gpu_k += 1;
                grad_3c_opt = Some(grad_flat.as_chunks::<3>().0.to_vec());
            } else if !fallback_recorded {
                gpu_fb += 1;
            }
        } else if self.use_gpu && n_atoms > 1 && has_grad3c_work && !fallback_recorded {
            gpu_fb += 1;
        }

        let grad_3c = if let Some(g) = grad_3c_opt {
            g
        } else {
            self.aux_basis
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
            )
        };

        // 4. 2-center metric derivative contribution: -1/2 sum_{P,Q} x_P x_Q d(P|Q)/dR (parallelized on CPU)
        let n_aux_shells = self.aux_basis.shells.len();
        let grad_2c = (0..n_aux_shells)
            .into_par_iter()
            .fold(
                || vec![[0.0; 3]; n_atoms],
                |mut local_grad, p_idx| {
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

                                            local_grad[p_shell.atom_index][axis] += w_2c * dp;
                                            local_grad[q_shell.atom_index][axis] += w_2c * dq;
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

        let mut grad = grad_3c;
        for idx in 0..n_atoms {
            for axis in 0..3 {
                grad[idx][axis] += grad_2c[idx][axis];
            }
        }

        self.gpu_kernel_invocations.fetch_add(gpu_k, Ordering::Relaxed);
        self.gpu_fallback_invocations.fetch_add(gpu_fb, Ordering::Relaxed);
        Ok((grad, gpu_k, gpu_fb))
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
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
                let mut cp_norm = [0.0f64; 10];
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
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

        let aux_l2 = AuxiliaryBasisSet::build_auto_jfit_with_max_l(&mol, &bs, 2);
        let cpu = RiJEngine::with_auxiliary_basis(&bs, aux_l2).unwrap();
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
                let mut cp_norm = [0.0f64; 10];
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

    #[test]
    fn test_gpu_eri3c_f_shell_aux_matches_cpu_to_machine_precision() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        for choice in [
            calchem_model::JFitBasis::Auto,
            calchem_model::JFitBasis::Def2UpToF,
        ] {
            let cpu = RiJEngine::with_basis(&mol, &bs, false, choice).unwrap();
            let aux_basis = &cpu.aux_basis;
            assert!(
                aux_basis.shells.iter().any(|s| s.shell_type.angular_momentum() == 3),
                "Expected aux F-shell (L=3) under {choice:?}"
            );
            let n_aux = aux_basis.n_aux;
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

            let mut gpu_aux_sh = Vec::new();
            let mut gpu_aux_pr = Vec::new();
            for (p_idx, p_sh) in aux_basis.shells.iter().enumerate() {
                let prim_start = gpu_aux_pr.len() as i32;
                let comps = p_sh.components();
                for (pp_i, &gp) in p_sh.exponents.iter().enumerate() {
                    let c_base = p_sh.coeffs[pp_i];
                    let mut cp_norm = [0.0f64; 10];
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
            ).expect("GPU F-shell 3c ERI builder should succeed");
            let gpu_b = gpu_tensor.as_slice();
            let mut max_err = 0.0f64;
            for idx in 0..n_aux * width {
                max_err = max_err.max((gpu_b[idx] - cpu_b[idx]).abs());
            }
            assert!(
                max_err < 1e-11,
                "GPU vs CPU F-shell ({choice:?}) 3c ERI max_err={max_err:.3e} >= 1e-11"
            );
        }
    }

    #[test]
    fn test_gpu_rij_chunked_streaming_matches_dense_cpu() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let cpu = RiJEngine::with_cache_limit_gpu(&mol, &bs, 4usize << 30, false).unwrap();
        // Force aux-chunked GPU streaming by capping resident cache below full B size (1 MiB)
        let gpu_stream = RiJEngine::with_cache_limit_gpu(&mol, &bs, 1usize << 20, true).unwrap();
        assert!(gpu_stream.gpu_streamer.is_some(), "Expected GpuRijStreamer to activate");

        let n = bs.n_ao;
        let mut p = Matrix2D::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = 0.05 / (1.0 + ((i as f64 - j as f64).abs()));
                p.set(i, j, v);
                p.set(j, i, v);
            }
        }
        let j_cpu = cpu.build_coulomb(&bs, &p);
        let j_gpu = gpu_stream.build_coulomb(&bs, &p);
        let mut max_err = 0.0f64;
        for idx in 0..n * n {
            max_err = max_err.max((j_cpu.data[idx] - j_gpu.data[idx]).abs());
        }
        assert!(max_err < 1e-11, "Chunked GPU RI-J max_err={max_err:.3e} >= 1e-11");
        let k = gpu_stream.gpu_kernel_invocations.load(Ordering::Relaxed);
        let fb = gpu_stream.gpu_fallback_invocations.load(Ordering::Relaxed);
        assert!(k > 0 && fb == 0, "Expected GPU streaming telemetry k>0 fb==0, got k={k} fb={fb}");
    }

    #[test]
    fn test_gpu_rij_coulomb_gradient_parity_and_disable_gpu_telemetry() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let n = bs.n_ao;
        let mut p = Matrix2D::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = 0.03 / (1.0 + (i as f64 - j as f64).abs());
                p.set(i, j, v);
                p.set(j, i, v);
            }
        }

        if calchem_cuda::is_cuda_available() {
            for choice in [
                calchem_model::JFitBasis::Auto,
                calchem_model::JFitBasis::Def2UpToF,
            ] {
                let cpu = RiJEngine::with_basis_for_gradient(&mol, &bs, false, choice).unwrap();
                let (g_cpu, kc, fbc) = cpu.compute_coulomb_gradient_with_telemetry(mol.atoms.len(), &bs, &p).unwrap();
                assert_eq!((kc, fbc), (0, 0));

                let gpu = RiJEngine::with_basis_for_gradient(&mol, &bs, true, choice).unwrap();
                let (g_gpu, kg, fbg) = gpu.compute_coulomb_gradient_with_telemetry(mol.atoms.len(), &bs, &p).unwrap();
                assert!(kg > 0 && fbg == 0, "{choice:?}: expected k>0 fb==0, got ({kg}, {fbg})");

                // Translational invariance check
                let mut sum_g = [0.0f64; 3];
                let mut max_diff = 0.0f64;
                for (ga, gb) in g_cpu.iter().zip(&g_gpu) {
                    for q in 0..3 {
                        sum_g[q] += gb[q];
                        max_diff = max_diff.max((ga[q] - gb[q]).abs());
                    }
                }
                let trans_err = sum_g.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                assert!(max_diff < 1e-10, "{choice:?}: GPU vs CPU RI-J grad max_diff={max_diff:.3e}");
                assert!(trans_err < 1e-11, "{choice:?}: GPU RI-J grad translational sum={trans_err:.3e}");
            }
        }

        // Disabled GPU telemetry check
        struct ResetDisable(Option<std::ffi::OsString>);
        impl Drop for ResetDisable {
            fn drop(&mut self) {
                if let Some(ref v) = self.0 {
                    std::env::set_var("CALCHEM_DISABLE_GPU", v);
                } else {
                    std::env::remove_var("CALCHEM_DISABLE_GPU");
                }
            }
        }
        let _reset = ResetDisable(std::env::var_os("CALCHEM_DISABLE_GPU"));
        std::env::set_var("CALCHEM_DISABLE_GPU", "1");
        let dis = RiJEngine::with_basis_for_gradient(&mol, &bs, true, calchem_model::JFitBasis::Auto).unwrap();
        let (_, kd, fbd) = dis.compute_coulomb_gradient_with_telemetry(mol.atoms.len(), &bs, &p).unwrap();
        assert_eq!(kd, 0);
        assert!(fbd > 0, "Expected fallback > 0 under CALCHEM_DISABLE_GPU=1");
    }

    struct DfEnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }
    impl DfEnvGuard {
        fn new(keys: &[&'static str]) -> Self {
            Self {
                saved: keys.iter().map(|&k| (k, std::env::var(k).ok())).collect(),
            }
        }
    }
    impl Drop for DfEnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    #[test]
    fn test_gpu_rij_primary_fshell_and_fallback_telemetry_no_double_count() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _env = DfEnvGuard::new(&["CALCHEM_DISABLE_GPU", "CALCHEM_MAX_GPU_MEM_MB"]);

        let mol = Molecule::new(
            vec![
                Atom::from_bohr("Fe", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 3.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        assert!(bs.shells.iter().any(|s| s.shell_type.angular_momentum() == 3));

        let n = bs.n_ao;
        let mut p = Matrix2D::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = 0.02 / (1.0 + (i as f64 - j as f64).abs());
                p.set(i, j, v);
                p.set(j, i, v);
            }
        }

        let cpu = RiJEngine::new_with_gpu(&mol, &bs, false).unwrap();
        let gpu_f = RiJEngine::new_with_gpu(&mol, &bs, true).unwrap();

        let j_cpu = cpu.build_coulomb(&bs, &p);
        let j_gpu = gpu_f.build_coulomb(&bs, &p);
        let max_j_err = j_cpu
            .data
            .iter()
            .zip(&j_gpu.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_j_err < 1e-11, "Primary F-shell RI-J J mismatch: {max_j_err:.3e}");

        let (g_cpu, kc, fbc) = cpu.compute_coulomb_gradient_with_telemetry(2, &bs, &p).unwrap();
        assert_eq!((kc, fbc), (0, 0));
        let (g_gpu, kg, fbg) = gpu_f.compute_coulomb_gradient_with_telemetry(2, &bs, &p).unwrap();
        let max_g_err = g_cpu
            .iter()
            .zip(&g_gpu)
            .flat_map(|(a, b)| (0..3).map(move |q| (a[q] - b[q]).abs()))
            .fold(0.0f64, f64::max);
        assert!(max_g_err < 1e-11, "Primary F-shell RI-J grad mismatch: {max_g_err:.3e}");
        assert_eq!(kg, 0);
        assert_eq!(fbg, 1, "Expected exactly 1 fallback (no double count), got {fbg}");

        if calchem_cuda::is_cuda_available() {
            let h2o = Molecule::new(
                vec![
                    Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                    Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                    Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
                ],
                0,
                1,
            );
            let bs_w = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
            let nw = bs_w.n_ao;
            let pw = Matrix2D::identity(nw);
            let rij_dev = RiJEngine::new_with_gpu(&h2o, &bs_w, true).unwrap();
            std::env::set_var("CALCHEM_DISABLE_GPU", "1");
            let fb_before = rij_dev.gpu_fallback_invocations.load(Ordering::Relaxed);
            let j_fb = rij_dev.build_coulomb(&bs_w, &pw);
            let fb_after = rij_dev.gpu_fallback_invocations.load(Ordering::Relaxed);
            assert_eq!(fb_after - fb_before, 1, "Device-only build_coulomb fallback must count once");
            let j_ref = RiJEngine::new_with_gpu(&h2o, &bs_w, false).unwrap().build_coulomb(&bs_w, &pw);
            let err = j_fb.data.iter().zip(&j_ref.data).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(err < 1e-11);
        }
    }

    #[test]
    fn test_rij_tight_vram_streaming_fallback_def2_missing_element_and_auto_aux_cap() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _env = DfEnvGuard::new(&["CALCHEM_MAX_GPU_MEM_MB", "CALCHEM_AUTO_AUX_MAX_L"]);

        let he = Molecule::new(vec![Atom::from_bohr("He", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
        let bs_he = BasisSet::from_name("6-31g(d)", &he).unwrap();
        let err = RiJEngine::with_basis(&he, &bs_he, false, calchem_model::JFitBasis::Def2UpToF)
            .err()
            .expect("He must fail Def2UpToF");
        assert!(matches!(err, CalcError::UnsupportedBasis(_)));

        std::env::set_var("CALCHEM_AUTO_AUX_MAX_L", "2");
        let h2o = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs_w = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
        let aux_d = AuxiliaryBasisSet::build_auto_jfit(&h2o, &bs_w);
        assert!(aux_d.shells.iter().all(|s| s.shell_type.angular_momentum() <= 2));
        std::env::remove_var("CALCHEM_AUTO_AUX_MAX_L");

        if calchem_cuda::is_cuda_available() {
            std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "0");
            let oom = RiJEngine::with_cache_limit_gpu(&h2o, &bs_w, 0, true).unwrap();
            assert!(!oom.has_cached_tensor());
            let pw = Matrix2D::identity(bs_w.n_ao);
            let j_oom = oom.build_coulomb(&bs_w, &pw);
            let j_cpu = RiJEngine::new(&h2o, &bs_w).unwrap().build_coulomb(&bs_w, &pw);
            let err = j_oom.data.iter().zip(&j_cpu.data).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(err < 1e-11, "OOM on-the-fly fallback mismatch: {err:.3e}");
            assert!(oom.gpu_fallback_invocations.load(Ordering::Relaxed) >= 1);
        }
    }
}
