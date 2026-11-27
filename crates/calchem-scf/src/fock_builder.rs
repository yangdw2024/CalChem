use calchem_basis::BasisSet;
use calchem_int::{
    compute_eri, compute_eri_with_omega, compute_sparse_eri, compute_sparse_eri_with_omega,
    DirectFockEngine, IntegralOptions, RiJEngine, SparseSymmetricEri,
};
use calchem_model::{CalcError, Molecule};
use calchem_tensor::{rms_diff, Matrix2D, Tensor4D};
use std::sync::{Arc, OnceLock};

/// Resolves the tight Stage-1 Direct ERI screening base threshold from `CALCHEM_DIRECT_THRESH`
/// or `conv_density`: `(conv_density * 1e-3).clamp(1e-12, 1e-11)`.
pub fn resolve_tight_base(conv_density: f64) -> f64 {
    if let Some(val) = std::env::var("CALCHEM_DIRECT_THRESH")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        return val;
    }
    (conv_density * 1e-3).clamp(1e-12, 1e-11)
}

/// Resolves the coarse Stage-0 Direct ERI screening base threshold.
/// Returns `tight_base` if `CALCHEM_DIRECT_THRESH` is explicitly pinned.
pub fn resolve_coarse_base(tight_base: f64) -> f64 {
    if std::env::var("CALCHEM_DIRECT_THRESH")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .is_some()
    {
        return tight_base;
    }
    (tight_base * 1000.0).min(1e-8)
}

/// Returns the Direct-K screening threshold scaled inversely by the EXX coefficient magnitude
/// so small hybrid mixing fractions skip negligible quartets without losing target energy accuracy.
fn direct_k_thresh_with_base(base: f64, scale: f64) -> f64 {
    base / scale.abs().max(0.05)
}

/// Fock matrix builder supporting Stored-ERI (for small molecules), Direct SCF (for large molecules),
/// and Density Fitting (RI-J) for accelerated O(N^3) Coulomb evaluation, with full native support
/// for range-separated hybrid exchange (CAM-B3LYP, wB97X-D).
#[derive(Clone)]
pub enum RksFockBuilder {
    Stored(SparseSymmetricEri, Option<SparseSymmetricEri>),
    Direct(DirectFockEngine),
    DensityFitting(Arc<RiJEngine>, Arc<OnceLock<DirectFockEngine>>),
}

impl RksFockBuilder {
    #[allow(dead_code)]
    pub(crate) fn new(basis: &BasisSet, use_direct: bool) -> Self {
        if use_direct {
            Self::Direct(DirectFockEngine::new(basis, 1e-11))
        } else {
            Self::Stored(compute_sparse_eri(basis), None)
        }
    }

    pub fn new_with_options(
        mol: &Molecule,
        basis: &BasisSet,
        use_direct: bool,
        use_df: bool,
        use_gpu: bool,
        range_sep: Option<(f64, f64, f64)>,
    ) -> Result<Self, CalcError> {
        Self::with_jfit_basis(mol,basis,use_direct,use_df,use_gpu,range_sep,calchem_model::JFitBasis::Auto)
    }

    pub fn with_jfit_basis(mol: &Molecule, basis: &BasisSet, use_direct: bool, use_df: bool, use_gpu: bool, range_sep: Option<(f64,f64,f64)>, jfit: calchem_model::JFitBasis) -> Result<Self, CalcError> {
        if use_df {
            let rij = RiJEngine::with_basis(mol,basis,use_gpu,jfit)?;
            let rij = Arc::new(rij);
            let engine = Arc::new(OnceLock::new());
            Ok(Self::DensityFitting(rij, engine))
        } else if use_direct {
            Ok(Self::Direct(DirectFockEngine::new_with_gpu(basis, 1e-12, use_gpu)))
        } else {
            let eri = compute_sparse_eri(basis);
            let lr_eri = if let Some((_, _, omega)) = range_sep {
                Some(compute_sparse_eri_with_omega(
                    basis,
                    &IntegralOptions::default(),
                    Some(omega),
                ))
            } else {
                None
            };
            Ok(Self::Stored(eri, lr_eri))
        }
    }

    pub fn build_2e(
        &self,
        basis: &BasisSet,
        p: &Matrix2D,
        exx: f64,
        range_sep: Option<(f64, f64, f64)>,
    ) -> Matrix2D {
        self.build_2e_with_thresh(basis, p, exx, range_sep, resolve_tight_base(1e-9))
    }

    pub fn build_2e_with_thresh(
        &self,
        basis: &BasisSet,
        p: &Matrix2D,
        exx: f64,
        range_sep: Option<(f64, f64, f64)>,
        base_thresh: f64,
    ) -> Matrix2D {
        let n = p.rows;
        match self {
            Self::Stored(eri, ref lr_eri) => {
                if let Some((alpha, beta, _)) = range_sep {
                    let mut data = eri.contract_rks(&p.data, alpha);
                    if let Some(ref lr) = lr_eri {
                        let k_lr = lr.contract_k_gen(&p.data);
                        let half_beta = 0.5 * beta;
                        for i in 0..n * n {
                            data[i] -= half_beta * k_lr[i];
                        }
                    }
                    Matrix2D {
                        rows: n,
                        cols: n,
                        data,
                    }
                } else {
                    let data = eri.contract_rks(&p.data, exx);
                    Matrix2D {
                        rows: n,
                        cols: n,
                        data,
                    }
                }
            }
            Self::Direct(engine) => {
                if let Some((alpha, beta, omega)) = range_sep {
                    let mut data = engine.build_rks(basis, &p.data, alpha, base_thresh);
                    let workers = engine.determine_worker_count(false, None);
                    let k_lr = engine.build_k_with_omega_bounded(
                        basis,
                        &p.data,
                        Some(omega),
                        direct_k_thresh_with_base(base_thresh, beta),
                        workers,
                    );
                    let half_beta = 0.5 * beta;
                    for i in 0..n * n {
                        data[i] -= half_beta * k_lr[i];
                    }
                    Matrix2D {
                        rows: n,
                        cols: n,
                        data,
                    }
                } else {
                    let data = engine.build_rks(basis, &p.data, exx, base_thresh);
                    Matrix2D {
                        rows: n,
                        cols: n,
                        data,
                    }
                }
            }
            Self::DensityFitting(rij, _) => {
                let mut g = rij.build_coulomb(basis, p);
                if let Some(k_w) = self.build_df_k_weighted(basis, p, exx, range_sep, base_thresh)
                {
                    for i in 0..n * n {
                        g.data[i] -= k_w.data[i];
                    }
                }
                g
            }
        }
    }

    /// Computes only the EXX-weighted Direct-K matrix (`0.5 * exx * K[P]`, or
    /// `0.5 * alpha * K_std[P] + 0.5 * beta * K_lr[P]`) for `DensityFitting` builders.
    /// Returns `None` for pure DFT (`exx == 0` and `range_sep.is_none()`) or non-DF builders.
    pub fn build_df_k_weighted(
        &self,
        basis: &BasisSet,
        p: &Matrix2D,
        exx: f64,
        range_sep: Option<(f64, f64, f64)>,
        base_thresh: f64,
    ) -> Option<Matrix2D> {
        let Self::DensityFitting(rij, engine_lock) = self else {
            return None;
        };
        let n = p.rows;
        if let Some((alpha, beta, omega)) = range_sep {
            let engine = engine_lock
                .get_or_init(|| DirectFockEngine::new_with_gpu(basis, 1e-11, rij.use_gpu));
            let workers = engine.determine_worker_count(false, None);
            let mut k_weighted = vec![0.0f64; n * n];
            if alpha.abs() >= 1e-15 {
                let k_std = engine.build_k_with_omega_bounded(
                    basis,
                    &p.data,
                    None,
                    direct_k_thresh_with_base(base_thresh, alpha),
                    workers,
                );
                let half_alpha = 0.5 * alpha;
                for i in 0..n * n {
                    k_weighted[i] += half_alpha * k_std[i];
                }
            }
            if beta.abs() >= 1e-15 {
                let k_lr = engine.build_k_with_omega_bounded(
                    basis,
                    &p.data,
                    Some(omega),
                    direct_k_thresh_with_base(base_thresh, beta),
                    workers,
                );
                let half_beta = 0.5 * beta;
                for i in 0..n * n {
                    k_weighted[i] += half_beta * k_lr[i];
                }
            }
            Some(Matrix2D {
                rows: n,
                cols: n,
                data: k_weighted,
            })
        } else if exx.abs() < 1e-15 {
            None
        } else {
            let engine = engine_lock
                .get_or_init(|| DirectFockEngine::new_with_gpu(basis, 1e-11, rij.use_gpu));
            let workers = engine.determine_worker_count(false, None);
            let mut k_std = engine.build_k_with_omega_bounded(
                basis,
                &p.data,
                None,
                direct_k_thresh_with_base(base_thresh, exx),
                workers,
            );
            let half_exx = 0.5 * exx;
            for val in &mut k_std {
                *val *= half_exx;
            }
            Some(Matrix2D {
                rows: n,
                cols: n,
                data: k_std,
            })
        }
    }

    #[allow(dead_code)]
    pub(crate) fn build_fock(
        &self,
        h_core: &Matrix2D,
        basis: &BasisSet,
        p: &Matrix2D,
        exx: f64,
    ) -> Matrix2D {
        self.build_fock_with_thresh(h_core, basis, p, exx, resolve_tight_base(1e-9))
    }

    pub(crate) fn build_fock_with_thresh(
        &self,
        h_core: &Matrix2D,
        basis: &BasisSet,
        p: &Matrix2D,
        exx: f64,
        base_thresh: f64,
    ) -> Matrix2D {
        let n = h_core.rows;
        let g = self.build_2e_with_thresh(basis, p, exx, None, base_thresh);
        let mut f = Matrix2D::zeros(n, n);
        for idx in 0..n * n {
            f.data[idx] = h_core.data[idx] + g.data[idx];
        }
        f
    }

    pub fn telemetry(&self) -> (usize, usize, usize) {
        use std::sync::atomic::Ordering;
        match self {
            Self::Direct(engine) => (
                engine.gpu_kernel_invocations.load(Ordering::Relaxed),
                engine.gpu_fallback_invocations.load(Ordering::Relaxed),
                0,
            ),
            Self::DensityFitting(rij, engine_lock) => {
                let mut k = rij.gpu_kernel_invocations.load(Ordering::Relaxed);
                let mut fb = rij.gpu_fallback_invocations.load(Ordering::Relaxed);
                if let Some(engine) = engine_lock.get() {
                    k += engine.gpu_kernel_invocations.load(Ordering::Relaxed);
                    fb += engine.gpu_fallback_invocations.load(Ordering::Relaxed);
                }
                (k, fb, rij.metric_truncated_modes)
            }
            Self::Stored(_, _) => (0, 0, 0),
        }
    }
}

/// Open-shell Fock matrix builder supporting Stored-ERI (Tensor4D), Direct SCF,
/// and Density Fitting (RI-J) with range-separated hybrid exchange.
#[derive(Clone)]
pub enum UksFockBuilder {
    Stored(Tensor4D, Option<Tensor4D>),
    Direct(DirectFockEngine),
    DensityFitting(Arc<RiJEngine>, Arc<OnceLock<DirectFockEngine>>),
}

impl UksFockBuilder {
    #[allow(dead_code)]
    pub(crate) fn new(basis: &BasisSet, use_direct: bool) -> Self {
        if use_direct {
            Self::Direct(DirectFockEngine::new(basis, 1e-11))
        } else {
            Self::Stored(compute_eri(basis), None)
        }
    }

    pub fn new_with_options(
        mol: &Molecule,
        basis: &BasisSet,
        use_direct: bool,
        use_df: bool,
        use_gpu: bool,
        range_sep: Option<(f64, f64, f64)>,
    ) -> Result<Self, CalcError> {
        Self::with_jfit_basis(mol,basis,use_direct,use_df,use_gpu,range_sep,calchem_model::JFitBasis::Auto)
    }

    pub fn with_jfit_basis(mol: &Molecule, basis: &BasisSet, use_direct: bool, use_df: bool, use_gpu: bool, range_sep: Option<(f64,f64,f64)>, jfit: calchem_model::JFitBasis) -> Result<Self, CalcError> {
        if use_df {
            let rij = RiJEngine::with_basis(mol,basis,use_gpu,jfit)?;
            let rij = Arc::new(rij);
            let engine = Arc::new(OnceLock::new());
            Ok(Self::DensityFitting(rij, engine))
        } else if use_direct {
            Ok(Self::Direct(DirectFockEngine::new_with_gpu(basis, 1e-12, use_gpu)))
        } else {
            let eri = compute_eri(basis);
            let lr_eri = if let Some((_, _, omega)) = range_sep {
                Some(compute_eri_with_omega(
                    basis,
                    &IntegralOptions::default(),
                    Some(omega),
                ))
            } else {
                None
            };
            Ok(Self::Stored(eri, lr_eri))
        }
    }

    pub fn build_2e(
        &self,
        basis: &BasisSet,
        pa: &Matrix2D,
        pb: &Matrix2D,
        exx_fraction: f64,
        range_sep: Option<(f64, f64, f64)>,
    ) -> (Matrix2D, Matrix2D) {
        self.build_2e_with_thresh(
            basis,
            pa,
            pb,
            exx_fraction,
            range_sep,
            resolve_tight_base(1e-9),
        )
    }

    pub fn build_2e_with_thresh(
        &self,
        basis: &BasisSet,
        pa: &Matrix2D,
        pb: &Matrix2D,
        exx_fraction: f64,
        range_sep: Option<(f64, f64, f64)>,
        base_thresh: f64,
    ) -> (Matrix2D, Matrix2D) {
        let n = pa.rows;
        match self {
            Self::Stored(eri, ref lr_eri) => {
                let mut ga = Matrix2D::zeros(n, n);
                let mut gb = Matrix2D::zeros(n, n);
                let eff_exx = if let Some((alpha, _, _)) = range_sep {
                    alpha
                } else {
                    exx_fraction
                };

                if eff_exx.abs() < 1e-15 && range_sep.is_none() {
                    for mu in 0..n {
                        for nu in mu..n {
                            let mut j = 0.0;
                            for lam in 0..n {
                                for sig in 0..n {
                                    let pt_ls = pa.get(lam, sig) + pb.get(lam, sig);
                                    j += pt_ls * eri.get(mu, nu, lam, sig);
                                }
                            }
                            ga.set(mu, nu, j);
                            ga.set(nu, mu, j);
                            gb.set(mu, nu, j);
                            gb.set(nu, mu, j);
                        }
                    }
                } else {
                    for mu in 0..n {
                        for nu in mu..n {
                            let mut j = 0.0;
                            let mut ka = 0.0;
                            let mut kb = 0.0;
                            for lam in 0..n {
                                for sig in 0..n {
                                    let pa_ls = pa.get(lam, sig);
                                    let pb_ls = pb.get(lam, sig);
                                    let coul = eri.get(mu, nu, lam, sig);
                                    let exch = eri.get(mu, lam, nu, sig);
                                    j += (pa_ls + pb_ls) * coul;
                                    ka += pa_ls * exch;
                                    kb += pb_ls * exch;
                                }
                            }
                            let ga_val = j - eff_exx * ka;
                            let gb_val = j - eff_exx * kb;
                            ga.set(mu, nu, ga_val);
                            ga.set(nu, mu, ga_val);
                            gb.set(mu, nu, gb_val);
                            gb.set(nu, mu, gb_val);
                        }
                    }
                    if let Some((_, beta, _)) = range_sep {
                        if let Some(ref lr) = lr_eri {
                            for mu in 0..n {
                                for nu in 0..n {
                                    let mut ka_lr = 0.0;
                                    let mut kb_lr = 0.0;
                                    for lam in 0..n {
                                        for sig in 0..n {
                                            let exch_lr = lr.get(mu, lam, nu, sig);
                                            ka_lr += pa.get(lam, sig) * exch_lr;
                                            kb_lr += pb.get(lam, sig) * exch_lr;
                                        }
                                    }
                                    ga.set(mu, nu, ga.get(mu, nu) - beta * ka_lr);
                                    gb.set(mu, nu, gb.get(mu, nu) - beta * kb_lr);
                                }
                            }
                        }
                    }
                }
                (ga, gb)
            }
            Self::Direct(engine) => {
                if let Some((alpha, beta, omega)) = range_sep {
                    let (mut da, mut db) =
                        engine.build_uks(basis, &pa.data, &pb.data, alpha, base_thresh);
                    let workers = engine.determine_worker_count(false, None);
                    let k_thresh = direct_k_thresh_with_base(base_thresh, beta);
                    let (ka_lr, kb_lr) = engine.build_k_pair_with_omega_bounded(
                        basis,
                        &pa.data,
                        &pb.data,
                        Some(omega),
                        k_thresh,
                        workers,
                    );
                    for i in 0..n * n {
                        da[i] -= beta * ka_lr[i];
                        db[i] -= beta * kb_lr[i];
                    }
                    (
                        Matrix2D {
                            rows: n,
                            cols: n,
                            data: da,
                        },
                        Matrix2D {
                            rows: n,
                            cols: n,
                            data: db,
                        },
                    )
                } else {
                    let (da, db) =
                        engine.build_uks(basis, &pa.data, &pb.data, exx_fraction, base_thresh);
                    (
                        Matrix2D {
                            rows: n,
                            cols: n,
                            data: da,
                        },
                        Matrix2D {
                            rows: n,
                            cols: n,
                            data: db,
                        },
                    )
                }
            }
            Self::DensityFitting(rij, _) => {
                let mut p_tot = Matrix2D::zeros(n, n);
                for i in 0..n * n {
                    p_tot.data[i] = pa.data[i] + pb.data[i];
                }
                let j_ri = rij.build_coulomb(basis, &p_tot);
                let mut ga = j_ri.clone();
                let mut gb = j_ri;
                if let Some((ka_w, kb_w)) = self.build_df_k_weighted(
                    basis,
                    pa,
                    pb,
                    exx_fraction,
                    range_sep,
                    base_thresh,
                ) {
                    for i in 0..n * n {
                        ga.data[i] -= ka_w.data[i];
                        gb.data[i] -= kb_w.data[i];
                    }
                }
                (ga, gb)
            }
        }
    }

    pub fn build_df_k_weighted(
        &self,
        basis: &BasisSet,
        pa: &Matrix2D,
        pb: &Matrix2D,
        exx_fraction: f64,
        range_sep: Option<(f64, f64, f64)>,
        base_thresh: f64,
    ) -> Option<(Matrix2D, Matrix2D)> {
        let Self::DensityFitting(rij, engine_lock) = self else {
            return None;
        };
        let n = pa.rows;
        if let Some((alpha, beta, omega)) = range_sep {
            let engine = engine_lock
                .get_or_init(|| DirectFockEngine::new_with_gpu(basis, 1e-11, rij.use_gpu));
            let workers = engine.determine_worker_count(false, None);
            let mut ka_w = vec![0.0f64; n * n];
            let mut kb_w = vec![0.0f64; n * n];
            if alpha.abs() >= 1e-15 {
                let t_a = direct_k_thresh_with_base(base_thresh, alpha);
                let (ka_std, kb_std) = engine.build_k_pair_with_omega_bounded(
                    basis,
                    &pa.data,
                    &pb.data,
                    None,
                    t_a,
                    workers,
                );
                for i in 0..n * n {
                    ka_w[i] += alpha * ka_std[i];
                    kb_w[i] += alpha * kb_std[i];
                }
            }
            if beta.abs() >= 1e-15 {
                let t_b = direct_k_thresh_with_base(base_thresh, beta);
                let (ka_lr, kb_lr) = engine.build_k_pair_with_omega_bounded(
                    basis,
                    &pa.data,
                    &pb.data,
                    Some(omega),
                    t_b,
                    workers,
                );
                for i in 0..n * n {
                    ka_w[i] += beta * ka_lr[i];
                    kb_w[i] += beta * kb_lr[i];
                }
            }
            Some((
                Matrix2D {
                    rows: n,
                    cols: n,
                    data: ka_w,
                },
                Matrix2D {
                    rows: n,
                    cols: n,
                    data: kb_w,
                },
            ))
        } else if exx_fraction.abs() < 1e-15 {
            None
        } else {
            let engine = engine_lock
                .get_or_init(|| DirectFockEngine::new_with_gpu(basis, 1e-11, rij.use_gpu));
            let workers = engine.determine_worker_count(false, None);
            let t_exx = direct_k_thresh_with_base(base_thresh, exx_fraction);
            let (mut ka_std, mut kb_std) = engine.build_k_pair_with_omega_bounded(
                basis,
                &pa.data,
                &pb.data,
                None,
                t_exx,
                workers,
            );
            for i in 0..n * n {
                ka_std[i] *= exx_fraction;
                kb_std[i] *= exx_fraction;
            }
            Some((
                Matrix2D {
                    rows: n,
                    cols: n,
                    data: ka_std,
                },
                Matrix2D {
                    rows: n,
                    cols: n,
                    data: kb_std,
                },
            ))
        }
    }

    pub fn telemetry(&self) -> (usize, usize, usize) {
        use std::sync::atomic::Ordering;
        match self {
            Self::Direct(engine) => (
                engine.gpu_kernel_invocations.load(Ordering::Relaxed),
                engine.gpu_fallback_invocations.load(Ordering::Relaxed),
                0,
            ),
            Self::DensityFitting(rij, engine_lock) => {
                let mut k = rij.gpu_kernel_invocations.load(Ordering::Relaxed);
                let mut fb = rij.gpu_fallback_invocations.load(Ordering::Relaxed);
                if let Some(engine) = engine_lock.get() {
                    k += engine.gpu_kernel_invocations.load(Ordering::Relaxed);
                    fb += engine.gpu_fallback_invocations.load(Ordering::Relaxed);
                }
                (k, fb, rij.metric_truncated_modes)
            }
            Self::Stored(_, _) => (0, 0, 0),
        }
    }
}

/// Drift-bounded incremental two-electron Fock accumulator (`ΔP`) with two-stage screening
/// for closed-shell RHF/RKS and SCRF loops.
pub struct RksFockAccumulator {
    g_2e: Matrix2D,
    j_ri_cur: Matrix2D,
    k_df_cur: Matrix2D,
    p_prev: Matrix2D,
    cum_dp_rms: f64,
    cum_dp_max: f64,
    inc_steps: usize,
    pass: u8,
    pending_tight_rebuild: bool,
    tight_base: f64,
    coarse_base: f64,
}

impl RksFockAccumulator {
    pub fn new(n: usize, p_init: &Matrix2D, conv_density: f64) -> Self {
        let tight_base = resolve_tight_base(conv_density);
        let coarse_base = resolve_coarse_base(tight_base);
        let pass = if coarse_base <= tight_base { 1 } else { 0 };
        Self {
            g_2e: Matrix2D::zeros(n, n),
            j_ri_cur: Matrix2D::zeros(n, n),
            k_df_cur: Matrix2D::zeros(n, n),
            p_prev: p_init.clone(),
            cum_dp_rms: 0.0,
            cum_dp_max: 0.0,
            inc_steps: 0,
            pass,
            pending_tight_rebuild: false,
            tight_base,
            coarse_base,
        }
    }

    #[inline]
    pub fn in_tight_pass(&self) -> bool {
        self.pass == 1 && !self.pending_tight_rebuild
    }

    #[inline]
    pub fn request_tight_pass(&mut self) {
        self.pass = 1;
        self.pending_tight_rebuild = true;
    }

    #[inline]
    pub fn tight_base(&self) -> f64 {
        self.tight_base
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        builder: &RksFockBuilder,
        basis: &BasisSet,
        p: &Matrix2D,
        exx: f64,
        range_sep: Option<(f64, f64, f64)>,
        iter: usize,
        use_direct: bool,
        last_rms_d: f64,
        prev_diis_err: f64,
    ) -> &Matrix2D {
        if self.pass == 0 && iter >= 2 && last_rms_d < 1e-4 && prev_diis_err < 1e-3 {
            self.pass = 1;
            self.pending_tight_rebuild = true;
        }

        let mut force_full = iter == 1 || !use_direct || self.pending_tight_rebuild;
        let base_thresh = if self.pass == 1 {
            self.tight_base
        } else {
            self.coarse_base
        };

        let n = p.rows;
        let mut delta_p = Matrix2D::zeros(n, n);
        let mut dp_max = 0.0f64;
        let mut dp_rms = 0.0f64;
        if !force_full {
            let mut sum_sq = 0.0f64;
            for idx in 0..n * n {
                let d = p.data[idx] - self.p_prev.data[idx];
                delta_p.data[idx] = d;
                let a = d.abs();
                if a > dp_max {
                    dp_max = a;
                }
                sum_sq += d * d;
            }
            dp_rms = (sum_sq / ((n * n).max(1) as f64)).sqrt();
            if dp_rms >= 0.01
                || dp_max >= 0.15
                || self.cum_dp_rms + dp_rms >= 0.05
                || self.cum_dp_max + dp_max >= 0.25
                || self.inc_steps >= 8
            {
                force_full = true;
            }
        }

        if force_full {
            match builder {
                RksFockBuilder::DensityFitting(rij, _) => {
                    self.j_ri_cur = rij.build_coulomb(basis, p);
                    if let Some(k_w) =
                        builder.build_df_k_weighted(basis, p, exx, range_sep, base_thresh)
                    {
                        self.k_df_cur = k_w;
                        for idx in 0..n * n {
                            self.g_2e.data[idx] =
                                self.j_ri_cur.data[idx] - self.k_df_cur.data[idx];
                        }
                    } else {
                        self.k_df_cur = Matrix2D::zeros(n, n);
                        self.g_2e = self.j_ri_cur.clone();
                    }
                }
                _ => {
                    self.g_2e = builder.build_2e_with_thresh(basis, p, exx, range_sep, base_thresh);
                }
            }
            self.cum_dp_rms = 0.0;
            self.cum_dp_max = 0.0;
            self.inc_steps = 0;
            self.pending_tight_rebuild = false;
        } else if dp_max > 0.0 {
            match builder {
                RksFockBuilder::DensityFitting(rij, _) => {
                    let j_cur = if rij.has_cached_tensor() {
                        rij.build_coulomb(basis, p)
                    } else {
                        let dj = rij.build_coulomb(basis, &delta_p);
                        for idx in 0..n * n {
                            self.j_ri_cur.data[idx] += dj.data[idx];
                        }
                        self.j_ri_cur.clone()
                    };
                    if let Some(dk) =
                        builder.build_df_k_weighted(basis, &delta_p, exx, range_sep, base_thresh)
                    {
                        for idx in 0..n * n {
                            self.k_df_cur.data[idx] += dk.data[idx];
                            self.g_2e.data[idx] = j_cur.data[idx] - self.k_df_cur.data[idx];
                        }
                    } else {
                        self.g_2e = j_cur;
                    }
                }
                _ => {
                    let dg =
                        builder.build_2e_with_thresh(basis, &delta_p, exx, range_sep, base_thresh);
                    for idx in 0..n * n {
                        self.g_2e.data[idx] += dg.data[idx];
                    }
                }
            }
            self.cum_dp_rms += dp_rms;
            self.cum_dp_max += dp_max;
            self.inc_steps += 1;
        }

        self.p_prev = p.clone();
        &self.g_2e
    }
}

/// Drift-bounded incremental two-electron Fock accumulator (`ΔP`) with two-stage screening
/// for open-shell UHF/UKS and unrestricted SCRF loops.
pub struct UksFockAccumulator {
    ga_cur: Matrix2D,
    gb_cur: Matrix2D,
    j_ri_cur: Matrix2D,
    ka_df_cur: Matrix2D,
    kb_df_cur: Matrix2D,
    pa_prev: Matrix2D,
    pb_prev: Matrix2D,
    cum_dp_rms: f64,
    cum_dp_max: f64,
    inc_steps: usize,
    pass: u8,
    pending_tight_rebuild: bool,
    tight_base: f64,
    coarse_base: f64,
}

impl UksFockAccumulator {
    pub fn new(n: usize, pa_init: &Matrix2D, pb_init: &Matrix2D, conv_density: f64) -> Self {
        let tight_base = resolve_tight_base(conv_density);
        let coarse_base = resolve_coarse_base(tight_base);
        let pass = if coarse_base <= tight_base { 1 } else { 0 };
        Self {
            ga_cur: Matrix2D::zeros(n, n),
            gb_cur: Matrix2D::zeros(n, n),
            j_ri_cur: Matrix2D::zeros(n, n),
            ka_df_cur: Matrix2D::zeros(n, n),
            kb_df_cur: Matrix2D::zeros(n, n),
            pa_prev: pa_init.clone(),
            pb_prev: pb_init.clone(),
            cum_dp_rms: 0.0,
            cum_dp_max: 0.0,
            inc_steps: 0,
            pass,
            pending_tight_rebuild: false,
            tight_base,
            coarse_base,
        }
    }

    #[inline]
    pub fn in_tight_pass(&self) -> bool {
        self.pass == 1 && !self.pending_tight_rebuild
    }

    #[inline]
    pub fn request_tight_pass(&mut self) {
        self.pass = 1;
        self.pending_tight_rebuild = true;
    }

    #[inline]
    pub fn tight_base(&self) -> f64 {
        self.tight_base
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        builder: &UksFockBuilder,
        basis: &BasisSet,
        pa: &Matrix2D,
        pb: &Matrix2D,
        exx_fraction: f64,
        range_sep: Option<(f64, f64, f64)>,
        iter: usize,
        use_direct: bool,
        last_rms_d: f64,
        prev_diis_err: f64,
    ) -> (&Matrix2D, &Matrix2D) {
        if self.pass == 0 && iter >= 2 && last_rms_d < 1e-4 && prev_diis_err < 1e-3 {
            self.pass = 1;
            self.pending_tight_rebuild = true;
        }

        let mut force_full = iter == 1 || !use_direct || self.pending_tight_rebuild;
        let base_thresh = if self.pass == 1 {
            self.tight_base
        } else {
            self.coarse_base
        };

        let n = pa.rows;
        let mut dpa = Matrix2D::zeros(n, n);
        let mut dpb = Matrix2D::zeros(n, n);
        let mut dp_max = 0.0f64;
        let mut dp_rms = 0.0f64;
        if !force_full {
            for idx in 0..n * n {
                let da = pa.data[idx] - self.pa_prev.data[idx];
                let db = pb.data[idx] - self.pb_prev.data[idx];
                dpa.data[idx] = da;
                dpb.data[idx] = db;
                let m = da.abs().max(db.abs());
                if m > dp_max {
                    dp_max = m;
                }
            }
            dp_rms = rms_diff(pa, &self.pa_prev).max(rms_diff(pb, &self.pb_prev));
            if dp_rms >= 0.01
                || dp_max >= 0.15
                || self.cum_dp_rms + dp_rms >= 0.05
                || self.cum_dp_max + dp_max >= 0.25
                || self.inc_steps >= 8
            {
                force_full = true;
            }
        }

        if force_full {
            match builder {
                UksFockBuilder::DensityFitting(rij, _) => {
                    let mut p_tot = Matrix2D::zeros(n, n);
                    for idx in 0..n * n {
                        p_tot.data[idx] = pa.data[idx] + pb.data[idx];
                    }
                    self.j_ri_cur = rij.build_coulomb(basis, &p_tot);
                    if let Some((ka, kb)) = builder.build_df_k_weighted(
                        basis,
                        pa,
                        pb,
                        exx_fraction,
                        range_sep,
                        base_thresh,
                    ) {
                        self.ka_df_cur = ka;
                        self.kb_df_cur = kb;
                        for idx in 0..n * n {
                            let j = self.j_ri_cur.data[idx];
                            self.ga_cur.data[idx] = j - self.ka_df_cur.data[idx];
                            self.gb_cur.data[idx] = j - self.kb_df_cur.data[idx];
                        }
                    } else {
                        self.ka_df_cur = Matrix2D::zeros(n, n);
                        self.kb_df_cur = Matrix2D::zeros(n, n);
                        self.ga_cur = self.j_ri_cur.clone();
                        self.gb_cur = self.j_ri_cur.clone();
                    }
                }
                _ => {
                    let (ga, gb) = builder.build_2e_with_thresh(
                        basis,
                        pa,
                        pb,
                        exx_fraction,
                        range_sep,
                        base_thresh,
                    );
                    self.ga_cur = ga;
                    self.gb_cur = gb;
                }
            }
            self.cum_dp_rms = 0.0;
            self.cum_dp_max = 0.0;
            self.inc_steps = 0;
            self.pending_tight_rebuild = false;
        } else if dp_max > 0.0 {
            match builder {
                UksFockBuilder::DensityFitting(rij, _) => {
                    let j_cur = if rij.has_cached_tensor() {
                        let mut p_tot = Matrix2D::zeros(n, n);
                        for idx in 0..n * n {
                            p_tot.data[idx] = pa.data[idx] + pb.data[idx];
                        }
                        rij.build_coulomb(basis, &p_tot)
                    } else {
                        let mut dpt = Matrix2D::zeros(n, n);
                        for idx in 0..n * n {
                            dpt.data[idx] = dpa.data[idx] + dpb.data[idx];
                        }
                        let dj = rij.build_coulomb(basis, &dpt);
                        for idx in 0..n * n {
                            self.j_ri_cur.data[idx] += dj.data[idx];
                        }
                        self.j_ri_cur.clone()
                    };
                    if let Some((dka, dkb)) = builder.build_df_k_weighted(
                        basis,
                        &dpa,
                        &dpb,
                        exx_fraction,
                        range_sep,
                        base_thresh,
                    ) {
                        for idx in 0..n * n {
                            self.ka_df_cur.data[idx] += dka.data[idx];
                            self.kb_df_cur.data[idx] += dkb.data[idx];
                            let j = j_cur.data[idx];
                            self.ga_cur.data[idx] = j - self.ka_df_cur.data[idx];
                            self.gb_cur.data[idx] = j - self.kb_df_cur.data[idx];
                        }
                    } else {
                        self.ga_cur = j_cur.clone();
                        self.gb_cur = j_cur;
                    }
                }
                _ => {
                    let (dga, dgb) = builder.build_2e_with_thresh(
                        basis,
                        &dpa,
                        &dpb,
                        exx_fraction,
                        range_sep,
                        base_thresh,
                    );
                    for idx in 0..n * n {
                        self.ga_cur.data[idx] += dga.data[idx];
                        self.gb_cur.data[idx] += dgb.data[idx];
                    }
                }
            }
            self.cum_dp_rms += dp_rms;
            self.cum_dp_max += dp_max;
            self.inc_steps += 1;
        }

        self.pa_prev = pa.clone();
        self.pb_prev = pb.clone();
        (&self.ga_cur, &self.gb_cur)
    }
}
