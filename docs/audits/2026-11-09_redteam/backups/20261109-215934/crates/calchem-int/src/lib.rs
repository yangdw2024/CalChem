pub mod boys;
pub mod ecp;
pub mod eri;
pub mod one_electron;
pub mod primitive;
pub mod screening;
pub mod symmetry;

use calchem_basis::BasisSet;
use calchem_tensor::Tensor4D;
use rayon::prelude::*;

pub use boys::{boys, boys_batch, erf_approx};
pub use ecp::{
    compute_ecp, ecp_atom_primitive, ecp_local_primitive, ecp_semilocal_primitive,
    scaled_mod_sph_bessel_batch, shell_ecp_block,
};
pub use eri::{eri_primitive, eri_shell_quartet, EriPairData};
pub use one_electron::{
    compute_kinetic, compute_nuclear, compute_overlap, shell_kinetic_block, shell_nuclear_block,
    shell_nuclear_block_with_charges, shell_overlap_block,
};
pub use primitive::{kinetic_primitive, nuclear_vrr, overlap_1d, overlap_primitive, GaussianPair};
pub use screening::{shell_pair_bounds, SchwarzScreening};
pub use symmetry::{eri_permutations, unique_shell_quartets};

/// Options controlling electron repulsion integral evaluation (screening, parallelism).
#[derive(Debug, Clone)]
pub struct IntegralOptions {
    pub screen_threshold: f64,
    pub use_screening: bool,
    pub use_parallel: bool,
    pub parallel_threshold: usize,
}

impl Default for IntegralOptions {
    fn default() -> Self {
        Self {
            screen_threshold: 1e-10,
            use_screening: true,
            use_parallel: true,
            parallel_threshold: 4,
        }
    }
}

/// Backward-compatible alias for `erf_approx`.
pub fn erf(x: f64) -> f64 {
    erf_approx(x)
}

#[inline]
fn scatter_quartet_block(
    sh_a: &calchem_basis::Shell,
    sh_b: &calchem_basis::Shell,
    sh_c: &calchem_basis::Shell,
    sh_d: &calchem_basis::Shell,
    block: &[f64],
    eri: &mut Tensor4D,
) {
    let na = sh_a.n_functions();
    let nb = sh_b.n_functions();
    let nc = sh_c.n_functions();
    let nd = sh_d.n_functions();

    for ia in 0..na {
        let mu = sh_a.ao_offset + ia;
        for ib in 0..nb {
            let nu = sh_b.ao_offset + ib;
            for ic in 0..nc {
                let lam = sh_c.ao_offset + ic;
                for id in 0..nd {
                    let sig = sh_d.ao_offset + id;
                    let val = block[((ia * nb + ib) * nc + ic) * nd + id];
                    if val.abs() > 1e-16 {
                        for [p, q, r, s] in eri_permutations(mu, nu, lam, sig) {
                            eri.set(p, q, r, s, val);
                        }
                    }
                }
            }
        }
    }
}

/// Generates screened canonical shell quartets `[i, j, k, l]` without materializing
/// skipped quartets in memory.
pub fn screened_shell_quartets(basis: &BasisSet, opts: &IntegralOptions) -> Vec<[usize; 4]> {
    let n_shells = basis.shells.len();
    if !opts.use_screening {
        return unique_shell_quartets(n_shells);
    }
    let sc = SchwarzScreening::new(basis, opts.screen_threshold);
    let b_max = sc.bounds.iter().copied().fold(0.0f64, f64::max);
    if b_max == 0.0 {
        return Vec::new();
    }
    let mut sig_pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n_shells {
        for j in i..n_shells {
            let b_ij = sc.bound(i, j);
            if b_ij * b_max >= sc.threshold {
                sig_pairs.push((i, j, b_ij));
            }
        }
    }
    let mut quartets = Vec::new();
    for p_idx in 0..sig_pairs.len() {
        let (i, j, b_ij) = sig_pairs[p_idx];
        for &(k, l, b_kl) in &sig_pairs[p_idx..] {
            if b_ij * b_kl >= sc.threshold {
                quartets.push([i, j, k, l]);
            }
        }
    }
    quartets
}

/// Evaluates the full 4-index AO electron repulsion integral tensor `(mu nu | lam sig)`
/// with options for Schwarz screening and Rayon multithreading.
pub fn compute_eri_with_options(basis: &BasisSet, opts: &IntegralOptions) -> Tensor4D {
    let n = basis.n_ao;
    let mut eri = Tensor4D::zeros(n);
    let quartets = screened_shell_quartets(basis, opts);

    if opts.use_parallel && basis.shells.len() >= opts.parallel_threshold {
        let blocks: Vec<([usize; 4], Vec<f64>)> = quartets
            .into_par_iter()
            .map(|[i, j, k, l]| {
                let block = eri_shell_quartet(
                    &basis.shells[i],
                    &basis.shells[j],
                    &basis.shells[k],
                    &basis.shells[l],
                );
                ([i, j, k, l], block)
            })
            .collect();

        for ([i, j, k, l], block) in blocks {
            scatter_quartet_block(
                &basis.shells[i],
                &basis.shells[j],
                &basis.shells[k],
                &basis.shells[l],
                &block,
                &mut eri,
            );
        }
    } else {
        for [i, j, k, l] in quartets {
            let block = eri_shell_quartet(
                &basis.shells[i],
                &basis.shells[j],
                &basis.shells[k],
                &basis.shells[l],
            );
            scatter_quartet_block(
                &basis.shells[i],
                &basis.shells[j],
                &basis.shells[k],
                &basis.shells[l],
                &block,
                &mut eri,
            );
        }
    }

    eri
}

/// Evaluates the full 4-index AO electron repulsion integral tensor `(mu nu | lam sig)`
/// using the default screening and parallel settings.
pub fn compute_eri(basis: &BasisSet) -> Tensor4D {
    compute_eri_with_options(basis, &IntegralOptions::default())
}

/// Symmetry-compressed 4-center electron repulsion integral table for memory-efficient
/// O(N_nonzero) Fock matrix construction on large molecules without O(N^4) dense allocation,
/// while preserving exact row-major dense summation for small systems (`n_ao <= 64`).
#[derive(Debug, Clone)]
pub struct SparseSymmetricEri {
    pub n_ao: usize,
    pub dense: Option<Tensor4D>,
    /// Canonical `(mu, nu, lam, sig, weighted_val)` entries with `mu <= nu`, `lam <= sig`,
    /// `(mu, nu) <= (lam, sig)`, where `weighted_val = eri_val / stabilizer_order`.
    pub entries: Vec<(u16, u16, u16, u16, f64)>,
}

impl SparseSymmetricEri {
    /// Contracts the ERIs with a symmetric closed-shell density matrix `p` (`n_ao * n_ao`)
    /// to form `G_{mu,nu} = sum_{lam,sig} P_{lam,sig} [(mu nu|lam sig) - 0.5 * exx * (mu lam|nu sig)]`.
    pub fn contract_rks(&self, p: &[f64], exx_fraction: f64) -> Vec<f64> {
        let n = self.n_ao;
        if let Some(ref eri) = self.dense {
            let mut g = vec![0.0f64; n * n];
            if exx_fraction.abs() < 1e-15 {
                for mu in 0..n {
                    for nu in mu..n {
                        let mut j_sum = 0.0;
                        for lam in 0..n {
                            for sig in 0..n {
                                j_sum += p[lam * n + sig] * eri.get(mu, nu, lam, sig);
                            }
                        }
                        g[mu * n + nu] = j_sum;
                        g[nu * n + mu] = j_sum;
                    }
                }
            } else {
                let half_c = 0.5 * exx_fraction;
                for mu in 0..n {
                    for nu in mu..n {
                        let mut g_munu = 0.0;
                        for lam in 0..n {
                            for sig in 0..n {
                                let p_ls = p[lam * n + sig];
                                let j = eri.get(mu, nu, lam, sig);
                                let k = eri.get(mu, lam, nu, sig);
                                g_munu += p_ls * (j - half_c * k);
                            }
                        }
                        g[mu * n + nu] = g_munu;
                        g[nu * n + mu] = g_munu;
                    }
                }
            }
            return g;
        }
        let mut g = vec![0.0f64; n * n];
        if exx_fraction.abs() < 1e-15 {
            for &(mu, nu, lam, sig, v) in &self.entries {
                let mu = mu as usize;
                let nu = nu as usize;
                let lam = lam as usize;
                let sig = sig as usize;
                let p_ls = 2.0 * v * p[lam * n + sig];
                let p_mn = 2.0 * v * p[mu * n + nu];
                g[mu * n + nu] += p_ls;
                g[nu * n + mu] += p_ls;
                g[lam * n + sig] += p_mn;
                g[sig * n + lam] += p_mn;
            }
        } else {
            let half_c = 0.5 * exx_fraction;
            for &(mu, nu, lam, sig, v) in &self.entries {
                let mu = mu as usize;
                let nu = nu as usize;
                let lam = lam as usize;
                let sig = sig as usize;
                let p_ls = 2.0 * v * p[lam * n + sig];
                let p_mn = 2.0 * v * p[mu * n + nu];
                g[mu * n + nu] += p_ls;
                g[nu * n + mu] += p_ls;
                g[lam * n + sig] += p_mn;
                g[sig * n + lam] += p_mn;

                let vc = half_c * v;
                let k1 = vc * p[nu * n + sig];
                let k2 = vc * p[mu * n + sig];
                let k3 = vc * p[nu * n + lam];
                let k4 = vc * p[mu * n + lam];
                g[mu * n + lam] -= k1;
                g[lam * n + mu] -= k1;
                g[nu * n + lam] -= k2;
                g[lam * n + nu] -= k2;
                g[mu * n + sig] -= k3;
                g[sig * n + mu] -= k3;
                g[nu * n + sig] -= k4;
                g[sig * n + nu] -= k4;
            }
        }
        g
    }

    /// Contracts the ERIs with a general (non-symmetric) matrix `x` (`n_ao * n_ao`)
    /// to form the exact-exchange matrix `K_{mu,nu}[X] = sum_{lam,sig} (mu lam|nu sig) X_{lam,sig}`.
    pub fn contract_k_gen(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n_ao;
        if let Some(ref eri) = self.dense {
            let mut k = vec![0.0f64; n * n];
            for mu in 0..n {
                for nu in 0..n {
                    let mut acc = 0.0f64;
                    for lam in 0..n {
                        for sig in 0..n {
                            acc += eri.get(mu, lam, nu, sig) * x[lam * n + sig];
                        }
                    }
                    k[mu * n + nu] = acc;
                }
            }
            return k;
        }
        let mut k = vec![0.0f64; n * n];
        for &(mu, nu, lam, sig, v) in &self.entries {
            let mu = mu as usize;
            let nu = nu as usize;
            let lam = lam as usize;
            let sig = sig as usize;
            // All 8 permutations [a, b, c, d] of (mu, nu, lam, sig) contribute v * x[b * n + d] to k[a * n + c]
            k[mu * n + lam] += v * x[nu * n + sig];
            k[nu * n + lam] += v * x[mu * n + sig];
            k[mu * n + sig] += v * x[nu * n + lam];
            k[nu * n + sig] += v * x[mu * n + lam];
            k[lam * n + mu] += v * x[sig * n + nu];
            k[sig * n + mu] += v * x[lam * n + nu];
            k[lam * n + nu] += v * x[sig * n + mu];
            k[sig * n + nu] += v * x[lam * n + mu];
        }
        k
    }
}

#[inline]
fn extract_canonical_quartets(
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    sh_a: &calchem_basis::Shell,
    sh_b: &calchem_basis::Shell,
    sh_c: &calchem_basis::Shell,
    sh_d: &calchem_basis::Shell,
    block: &[f64],
    out: &mut Vec<(u16, u16, u16, u16, f64)>,
) {
    let na = sh_a.n_functions();
    let nb = sh_b.n_functions();
    let nc = sh_c.n_functions();
    let nd = sh_d.n_functions();

    for ia in 0..na {
        let ib_end = if i == j { ia + 1 } else { nb };
        let mu0 = sh_a.ao_offset + ia;
        for ib in 0..ib_end {
            let nu0 = sh_b.ao_offset + ib;
            let (mu, nu) = if mu0 <= nu0 { (mu0, nu0) } else { (nu0, mu0) };
            for ic in 0..nc {
                let id_end = if k == l { ic + 1 } else { nd };
                let lam0 = sh_c.ao_offset + ic;
                for id in 0..id_end {
                    let sig0 = sh_d.ao_offset + id;
                    let (lam, sig) = if lam0 <= sig0 { (lam0, sig0) } else { (sig0, lam0) };
                    if i == k && j == l && (ia, ib) < (ic, id) {
                        continue;
                    }
                    let val = block[((ia * nb + ib) * nc + ic) * nd + id];
                    if val.abs() > 1e-16 {
                        let s1 = if mu == nu { 2.0 } else { 1.0 };
                        let s2 = if lam == sig { 2.0 } else { 1.0 };
                        let s3 = if mu == lam && nu == sig { 2.0 } else { 1.0 };
                        out.push((
                            mu as u16,
                            nu as u16,
                            lam as u16,
                            sig as u16,
                            val / (s1 * s2 * s3),
                        ));
                    }
                }
            }
        }
    }
}

/// Evaluates the symmetry-compressed sparse ERI list without allocating a dense `O(N^4)` tensor
/// for large molecules (`n_ao > 64`), while using contiguous row-major `Tensor4D` for `n_ao <= 64`.
pub fn compute_sparse_eri_with_options(
    basis: &BasisSet,
    opts: &IntegralOptions,
) -> SparseSymmetricEri {
    let n = basis.n_ao;
    if n <= 64 {
        return SparseSymmetricEri {
            n_ao: n,
            dense: Some(compute_eri_with_options(basis, opts)),
            entries: Vec::new(),
        };
    }
    let quartets = screened_shell_quartets(basis, opts);

    let entries = if opts.use_parallel && basis.shells.len() >= opts.parallel_threshold {
        quartets
            .into_par_iter()
            .fold(
                Vec::new,
                |mut local_vec, [i, j, k, l]| {
                    let block = eri_shell_quartet(
                        &basis.shells[i],
                        &basis.shells[j],
                        &basis.shells[k],
                        &basis.shells[l],
                    );
                    extract_canonical_quartets(
                        i,
                        j,
                        k,
                        l,
                        &basis.shells[i],
                        &basis.shells[j],
                        &basis.shells[k],
                        &basis.shells[l],
                        &block,
                        &mut local_vec,
                    );
                    local_vec
                },
            )
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            })
    } else {
        let mut out = Vec::new();
        for [i, j, k, l] in quartets {
            let block = eri_shell_quartet(
                &basis.shells[i],
                &basis.shells[j],
                &basis.shells[k],
                &basis.shells[l],
            );
            extract_canonical_quartets(
                i,
                j,
                k,
                l,
                &basis.shells[i],
                &basis.shells[j],
                &basis.shells[k],
                &basis.shells[l],
                &block,
                &mut out,
            );
        }
        out
    };

    SparseSymmetricEri {
        n_ao: n,
        dense: None,
        entries,
    }
}

/// Evaluates the symmetry-compressed sparse ERI list using default screening and parallel settings.
pub fn compute_sparse_eri(basis: &BasisSet) -> SparseSymmetricEri {
    compute_sparse_eri_with_options(basis, &IntegralOptions::default())
}
