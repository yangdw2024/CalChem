use calchem_basis::BasisSet;
use calchem_int::{
    compute_eri, compute_eri_with_omega, compute_sparse_eri, compute_sparse_eri_with_omega,
    DirectFockEngine, IntegralOptions, RiJEngine, SparseSymmetricEri,
};
use calchem_model::{CalcError, Molecule};
use calchem_tensor::{Matrix2D, Tensor4D};
use std::sync::Arc;

/// Fock matrix builder supporting Stored-ERI (for small molecules), Direct SCF (for large molecules),
/// and Density Fitting (RI-J) for accelerated O(N^3) Coulomb evaluation, with full native support
/// for range-separated hybrid exchange (CAM-B3LYP, wB97X-D).
#[derive(Clone)]
pub(crate) enum RksFockBuilder {
    Stored(SparseSymmetricEri, Option<SparseSymmetricEri>),
    Direct(DirectFockEngine),
    DensityFitting(Arc<RiJEngine>, DirectFockEngine),
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

    pub(crate) fn new_with_options(
        mol: &Molecule,
        basis: &BasisSet,
        use_direct: bool,
        use_df: bool,
        range_sep: Option<(f64, f64, f64)>,
    ) -> Result<Self, CalcError> {
        if use_df {
            let rij = Arc::new(RiJEngine::new(mol, basis)?);
            let engine = DirectFockEngine::new(basis, 1e-11);
            Ok(Self::DensityFitting(rij, engine))
        } else if use_direct {
            Ok(Self::Direct(DirectFockEngine::new(basis, 1e-11)))
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

    pub(crate) fn build_2e(
        &self,
        basis: &BasisSet,
        p: &Matrix2D,
        exx: f64,
        range_sep: Option<(f64, f64, f64)>,
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
                    let mut data = engine.build_rks(basis, &p.data, alpha, 1e-11);
                    let k_lr = engine.build_k_with_omega(basis, &p.data, Some(omega));
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
                    let thresh = std::env::var("CALCHEM_DIRECT_THRESH")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-10);
                    let data = engine.build_rks(basis, &p.data, exx, thresh);
                    Matrix2D {
                        rows: n,
                        cols: n,
                        data,
                    }
                }
            }
            Self::DensityFitting(rij, engine) => {
                let j_ri = rij.build_coulomb(basis, p);
                if let Some((alpha, beta, omega)) = range_sep {
                    let k_std = engine.build_k(basis, &p.data);
                    let k_lr = engine.build_k_with_omega(basis, &p.data, Some(omega));
                    let half_alpha = 0.5 * alpha;
                    let half_beta = 0.5 * beta;
                    let mut g = j_ri;
                    for i in 0..n * n {
                        g.data[i] -= half_alpha * k_std[i] + half_beta * k_lr[i];
                    }
                    g
                } else if exx.abs() < 1e-15 {
                    j_ri
                } else {
                    let k_std = engine.build_k(basis, &p.data);
                    let half_exx = 0.5 * exx;
                    let mut g = j_ri;
                    for i in 0..n * n {
                        g.data[i] -= half_exx * k_std[i];
                    }
                    g
                }
            }
        }
    }

    pub(crate) fn build_fock(
        &self,
        h_core: &Matrix2D,
        basis: &BasisSet,
        p: &Matrix2D,
        exx: f64,
    ) -> Matrix2D {
        let n = h_core.rows;
        let g = self.build_2e(basis, p, exx, None);
        let mut f = Matrix2D::zeros(n, n);
        for idx in 0..n * n {
            f.data[idx] = h_core.data[idx] + g.data[idx];
        }
        f
    }
}

/// Open-shell Fock matrix builder supporting Stored-ERI (Tensor4D), Direct SCF,
/// and Density Fitting (RI-J) with range-separated hybrid exchange.
#[derive(Clone)]
pub(crate) enum UksFockBuilder {
    Stored(Tensor4D, Option<Tensor4D>),
    Direct(DirectFockEngine),
    DensityFitting(Arc<RiJEngine>, DirectFockEngine),
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

    pub(crate) fn new_with_options(
        mol: &Molecule,
        basis: &BasisSet,
        use_direct: bool,
        use_df: bool,
        range_sep: Option<(f64, f64, f64)>,
    ) -> Result<Self, CalcError> {
        if use_df {
            let rij = Arc::new(RiJEngine::new(mol, basis)?);
            let engine = DirectFockEngine::new(basis, 1e-11);
            Ok(Self::DensityFitting(rij, engine))
        } else if use_direct {
            Ok(Self::Direct(DirectFockEngine::new(basis, 1e-11)))
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

    pub(crate) fn build_2e(
        &self,
        basis: &BasisSet,
        pa: &Matrix2D,
        pb: &Matrix2D,
        exx_fraction: f64,
        range_sep: Option<(f64, f64, f64)>,
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
                        engine.build_uks(basis, &pa.data, &pb.data, alpha, 1e-11);
                    let ka_lr = engine.build_k_with_omega(basis, &pa.data, Some(omega));
                    let kb_lr = engine.build_k_with_omega(basis, &pb.data, Some(omega));
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
                    let thresh = std::env::var("CALCHEM_DIRECT_THRESH")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-10);
                    let (da, db) =
                        engine.build_uks(basis, &pa.data, &pb.data, exx_fraction, thresh);
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
            Self::DensityFitting(rij, engine) => {
                let mut p_tot = Matrix2D::zeros(n, n);
                for i in 0..n * n {
                    p_tot.data[i] = pa.data[i] + pb.data[i];
                }
                let j_ri = rij.build_coulomb(basis, &p_tot);
                if let Some((alpha, beta, omega)) = range_sep {
                    let ka_std = engine.build_k(basis, &pa.data);
                    let kb_std = engine.build_k(basis, &pb.data);
                    let ka_lr = engine.build_k_with_omega(basis, &pa.data, Some(omega));
                    let kb_lr = engine.build_k_with_omega(basis, &pb.data, Some(omega));
                    let mut ga = j_ri.clone();
                    let mut gb = j_ri;
                    for i in 0..n * n {
                        ga.data[i] -= alpha * ka_std[i] + beta * ka_lr[i];
                        gb.data[i] -= alpha * kb_std[i] + beta * kb_lr[i];
                    }
                    (ga, gb)
                } else if exx_fraction.abs() < 1e-15 {
                    (j_ri.clone(), j_ri)
                } else {
                    let ka_std = engine.build_k(basis, &pa.data);
                    let kb_std = engine.build_k(basis, &pb.data);
                    let mut ga = j_ri.clone();
                    let mut gb = j_ri;
                    for i in 0..n * n {
                        ga.data[i] -= exx_fraction * ka_std[i];
                        gb.data[i] -= exx_fraction * kb_std[i];
                    }
                    (ga, gb)
                }
            }
        }
    }
}
