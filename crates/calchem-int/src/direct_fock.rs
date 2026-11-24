use rayon::prelude::*;
use crate::eri::{collect_egp_pairs, eri_from_pairs, eri_shell_quartet_with_omega_into};
use std::sync::{Arc, Mutex, OnceLock};
use crate::screening::SchwarzScreening;
use calchem_basis::{BasisSet, Shell};

/// Pre-screened shell pair with its Cauchy-Schwarz bound.
#[derive(Debug, Clone, Copy)]
pub struct ShellPair {
    pub shell_i: usize,
    pub shell_j: usize,
    pub bound: f64,
}

/// Computes the shell-pair entrywise L1 density matrix norm:
/// `D_{1, ij} = \sum_{\mu \in i, \nu \in j} |P_{\mu\nu}|`.
///
/// This provides a mathematically rigorous Cauchy-Schwarz upper bound on the contraction
/// of any 2-electron integral block with the density matrix:
/// `|\sum_{\lambda \in k, \sigma \in l} (\mu\nu|\lambda\sigma) P_{\lambda\sigma}| \le b_{ij} b_{kl} D_{1, kl}`.
pub fn compute_shell_pair_l1_density(basis: &BasisSet, density: &[f64]) -> Vec<f64> {
    let n_shells = basis.shells.len();
    let n_ao = basis.n_ao;
    let mut d = vec![0.0f64; n_shells * n_shells];

    for i in 0..n_shells {
        let sh_i = &basis.shells[i];
        let off_i = sh_i.ao_offset;
        let na = sh_i.n_functions();

        for j in 0..=i {
            let sh_j = &basis.shells[j];
            let off_j = sh_j.ao_offset;
            let nb = sh_j.n_functions();

            let mut sum_val = 0.0f64;
            for ia in 0..na {
                let mu = off_i + ia;
                let row_off = mu * n_ao;
                for ib in 0..nb {
                    let nu = off_j + ib;
                    sum_val += density[row_off + nu].abs();
                }
            }
            d[i * n_shells + j] = sum_val;
            d[j * n_shells + i] = sum_val;
        }
    }
    d
}

/// Computes the shell-pair maximum absolute density matrix element:
/// `D_{ij} = \max_{\mu \in i, \nu \in j} |P_{\mu\nu}|`.
pub fn compute_shell_pair_max_density(basis: &BasisSet, density: &[f64]) -> Vec<f64> {
    let n_shells = basis.shells.len();
    let n_ao = basis.n_ao;
    let mut d = vec![0.0f64; n_shells * n_shells];

    for i in 0..n_shells {
        let sh_i = &basis.shells[i];
        let off_i = sh_i.ao_offset;
        let na = sh_i.n_functions();

        for j in 0..=i {
            let sh_j = &basis.shells[j];
            let off_j = sh_j.ao_offset;
            let nb = sh_j.n_functions();

            let mut max_val = 0.0f64;
            for ia in 0..na {
                let mu = off_i + ia;
                let row_off = mu * n_ao;
                for ib in 0..nb {
                    let nu = off_j + ib;
                    let v = density[row_off + nu].abs();
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            d[i * n_shells + j] = max_val;
            d[j * n_shells + i] = max_val;
        }
    }
    d
}

/// Execution statistics and cumulative error accounting for a Direct Fock build.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DirectFockStats {
    /// Number of significant shell pairs considered.
    pub significant_shell_pairs: usize,
    /// Total number of unique shell quartet candidates: `N_pairs * (N_pairs + 1) / 2`.
    pub total_quartet_candidates: usize,
    /// Number of shell quartets actually evaluated via Obara-Saika recurrence.
    pub computed_quartets: usize,
    /// Number of shell quartets safely skipped by Cauchy-Schwarz screening.
    pub skipped_quartets: usize,
    /// Conservative upper bound on the cumulative entrywise Fock error:
    /// `\sum_{(ij|kl) \in \text{skipped}} B(ij|kl)`.
    pub cumulative_bound_error: f64,
    /// Maximum upper bound among all individual skipped quartets.
    pub max_skipped_bound: f64,
}

/// Filters all shell pairs `(i >= j)` whose upper bound times `b_max` satisfies the screening threshold.
/// When `threshold <= 0.0`, all shell pairs are retained (exact reference mode).
pub fn find_significant_shell_pairs(
    basis: &BasisSet,
    screening: &SchwarzScreening,
    threshold: f64,
) -> Vec<ShellPair> {
    let n_shells = basis.shells.len();
    let b_max = screening.bounds.iter().copied().fold(0.0f64, f64::max);
    if b_max <= 0.0 && threshold > 0.0 {
        return Vec::new();
    }
    let mut sig_pairs = Vec::new();
    for i in 0..n_shells {
        for j in 0..=i {
            let b_ij = screening.bound(i, j);
            if threshold <= 0.0 || b_ij * b_max * 16.0 >= threshold {
                sig_pairs.push(ShellPair {
                    shell_i: i,
                    shell_j: j,
                    bound: b_ij,
                });
            }
        }
    }
    sig_pairs
}

/// Contracts a computed shell quartet block `(sh_a sh_b | sh_c sh_d)` on-the-fly into a
/// thread-local closed-shell two-electron Fock matrix `G = J - 0.5 * exx_fraction * K`.
#[inline]
pub fn contract_quartet_rks(
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    sh_a: &Shell,
    sh_b: &Shell,
    sh_c: &Shell,
    sh_d: &Shell,
    block: &[f64],
    p: &[f64],
    n: usize,
    exx_fraction: f64,
    g: &mut [f64],
) {
    let na = sh_a.n_functions();
    let nb = sh_b.n_functions();
    let nc = sh_c.n_functions();
    let nd = sh_d.n_functions();

    let has_exchange = exx_fraction.abs() > 1e-15;
    let half_c = 0.5 * exx_fraction;

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
                    let (lam, sig) = if lam0 <= sig0 {
                        (lam0, sig0)
                    } else {
                        (sig0, lam0)
                    };
                    if i == k && j == l && (ia, ib) < (ic, id) {
                        continue;
                    }
                    let val = block[((ia * nb + ib) * nc + ic) * nd + id];
                    if val != 0.0 {
                        let s1 = if mu == nu { 2.0 } else { 1.0 };
                        let s2 = if lam == sig { 2.0 } else { 1.0 };
                        let s3 = if mu == lam && nu == sig { 2.0 } else { 1.0 };
                        let v = val / (s1 * s2 * s3);

                        let p_ls = 2.0 * v * p[lam * n + sig];
                        let p_mn = 2.0 * v * p[mu * n + nu];
                        g[mu * n + nu] += p_ls;
                        g[nu * n + mu] += p_ls;
                        g[lam * n + sig] += p_mn;
                        g[sig * n + lam] += p_mn;

                        if has_exchange {
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
                }
            }
        }
    }
}

/// Contracts a computed shell quartet block `(sh_a sh_b | sh_c sh_d)` on-the-fly into a
/// thread-local exact-exchange matrix `K[P]`.
#[inline]
pub fn contract_quartet_k(
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    sh_a: &Shell,
    sh_b: &Shell,
    sh_c: &Shell,
    sh_d: &Shell,
    block: &[f64],
    p: &[f64],
    n: usize,
    k_mat: &mut [f64],
) {
    let na = sh_a.n_functions();
    let nb = sh_b.n_functions();
    let nc = sh_c.n_functions();
    let nd = sh_d.n_functions();

    // Fast path: distinct shell pairs (i != j, k != l, (i,j) != (k,l)) — ~99% of quartets!
    // Zero symmetry divisions, half the memory writes (caller symmetrizes K + K^T once at end).
    if i != j && k != l && (i != k || j != l) {
        let ao_a = sh_a.ao_offset;
        let ao_b = sh_b.ao_offset;
        let ao_c = sh_c.ao_offset;
        let ao_d = sh_d.ao_offset;
        for ia in 0..na {
            let mu = ao_a + ia;
            let mu_n = mu * n;
            let ia_nb = ia * nb;
            for ib in 0..nb {
                let nu = ao_b + ib;
                let nu_n = nu * n;
                let ab_nc = (ia_nb + ib) * nc;
                for ic in 0..nc {
                    let lam = ao_c + ic;
                    let p_nu_lam = p[nu_n + lam];
                    let p_mu_lam = p[mu_n + lam];
                    let abc_nd = (ab_nc + ic) * nd;
                    let mut acc_mu_lam = 0.0;
                    let mut acc_nu_lam = 0.0;
                    for id in 0..nd {
                        let v = block[abc_nd + id];
                        if v != 0.0 {
                            let sig = ao_d + id;
                            acc_mu_lam += v * p[nu_n + sig];
                            acc_nu_lam += v * p[mu_n + sig];
                            k_mat[mu_n + sig] += v * p_nu_lam;
                            k_mat[nu_n + sig] += v * p_mu_lam;
                        }
                    }
                    k_mat[mu_n + lam] += acc_mu_lam;
                    k_mat[nu_n + lam] += acc_nu_lam;
                }
            }
        }
        return;
    }

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
                    let (lam, sig) = if lam0 <= sig0 {
                        (lam0, sig0)
                    } else {
                        (sig0, lam0)
                    };
                    if i == k && j == l && (ia, ib) < (ic, id) {
                        continue;
                    }
                    let val = block[((ia * nb + ib) * nc + ic) * nd + id];
                    if val != 0.0 {
                        let s1 = if mu == nu { 2.0 } else { 1.0 };
                        let s2 = if lam == sig { 2.0 } else { 1.0 };
                        let s3 = if mu == lam && nu == sig { 2.0 } else { 1.0 };
                        let v = val / (s1 * s2 * s3);

                        k_mat[mu * n + lam] += v * p[nu * n + sig];
                        k_mat[nu * n + lam] += v * p[mu * n + sig];
                        k_mat[mu * n + sig] += v * p[nu * n + lam];
                        k_mat[nu * n + sig] += v * p[mu * n + lam];
                    }
                }
            }
        }
    }
}

/// Contracts a computed shell quartet block `(sh_a sh_b | sh_c sh_d)` into a general
/// (possibly non-symmetric or antisymmetric) exact-exchange matrix
/// `K_{mu,nu}[X] = sum_{lam,sig} (mu lam | nu sig) X_{lam,sig}`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn contract_quartet_k_general(
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    sh_a: &Shell,
    sh_b: &Shell,
    sh_c: &Shell,
    sh_d: &Shell,
    block: &[f64],
    x: &[f64],
    n: usize,
    k_mat: &mut [f64],
) {
    let na = sh_a.n_functions();
    let nb = sh_b.n_functions();
    let nc = sh_c.n_functions();
    let nd = sh_d.n_functions();

    if i != j && k != l && (i != k || j != l) {
        let ao_a = sh_a.ao_offset;
        let ao_b = sh_b.ao_offset;
        let ao_c = sh_c.ao_offset;
        let ao_d = sh_d.ao_offset;
        for ia in 0..na {
            let mu = ao_a + ia;
            let mu_n = mu * n;
            let ia_nb = ia * nb;
            for ib in 0..nb {
                let nu = ao_b + ib;
                let nu_n = nu * n;
                let ab_nc = (ia_nb + ib) * nc;
                for ic in 0..nc {
                    let lam = ao_c + ic;
                    let lam_n = lam * n;
                    let x_nu_lam = x[nu_n + lam];
                    let x_mu_lam = x[mu_n + lam];
                    let x_lam_nu = x[lam_n + nu];
                    let x_lam_mu = x[lam_n + mu];
                    let abc_nd = (ab_nc + ic) * nd;
                    let mut acc_mu_lam = 0.0;
                    let mut acc_nu_lam = 0.0;
                    let mut acc_lam_mu = 0.0;
                    let mut acc_lam_nu = 0.0;
                    for id in 0..nd {
                        let v = block[abc_nd + id];
                        if v != 0.0 {
                            let sig = ao_d + id;
                            let sig_n = sig * n;
                            acc_mu_lam += v * x[nu_n + sig];
                            acc_nu_lam += v * x[mu_n + sig];
                            k_mat[mu_n + sig] += v * x_nu_lam;
                            k_mat[nu_n + sig] += v * x_mu_lam;
                            acc_lam_mu += v * x[sig_n + nu];
                            k_mat[sig_n + mu] += v * x_lam_nu;
                            acc_lam_nu += v * x[sig_n + mu];
                            k_mat[sig_n + nu] += v * x_lam_mu;
                        }
                    }
                    k_mat[mu_n + lam] += acc_mu_lam;
                    k_mat[nu_n + lam] += acc_nu_lam;
                    k_mat[lam_n + mu] += acc_lam_mu;
                    k_mat[lam_n + nu] += acc_lam_nu;
                }
            }
        }
        return;
    }

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
                    let (lam, sig) = if lam0 <= sig0 {
                        (lam0, sig0)
                    } else {
                        (sig0, lam0)
                    };
                    if i == k && j == l && (ia, ib) < (ic, id) {
                        continue;
                    }
                    let val = block[((ia * nb + ib) * nc + ic) * nd + id];
                    if val != 0.0 {
                        let s1 = if mu == nu { 2.0 } else { 1.0 };
                        let s2 = if lam == sig { 2.0 } else { 1.0 };
                        let s3 = if mu == lam && nu == sig { 2.0 } else { 1.0 };
                        let v = val / (s1 * s2 * s3);

                        k_mat[mu * n + lam] += v * x[nu * n + sig];
                        k_mat[nu * n + lam] += v * x[mu * n + sig];
                        k_mat[mu * n + sig] += v * x[nu * n + lam];
                        k_mat[nu * n + sig] += v * x[mu * n + lam];
                        k_mat[lam * n + mu] += v * x[sig * n + nu];
                        k_mat[sig * n + mu] += v * x[lam * n + nu];
                        k_mat[lam * n + nu] += v * x[sig * n + mu];
                        k_mat[sig * n + nu] += v * x[lam * n + mu];
                    }
                }
            }
        }
    }
}

/// Contracts a computed shell quartet block `(sh_a sh_b | sh_c sh_d)` on-the-fly into
/// open-shell two-electron Fock matrices `G_alpha = J[P^t] - exx * K[P^\alpha]` and
/// `G_beta = J[P^t] - exx * K[P^\beta]`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn contract_quartet_uks(
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    sh_a: &Shell,
    sh_b: &Shell,
    sh_c: &Shell,
    sh_d: &Shell,
    block: &[f64],
    pa: &[f64],
    pb: &[f64],
    pt: &[f64],
    n: usize,
    exx_fraction: f64,
    ga: &mut [f64],
    gb: &mut [f64],
) {
    let na = sh_a.n_functions();
    let nb = sh_b.n_functions();
    let nc = sh_c.n_functions();
    let nd = sh_d.n_functions();

    let has_exchange = exx_fraction.abs() > 1e-15;

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
                    let (lam, sig) = if lam0 <= sig0 {
                        (lam0, sig0)
                    } else {
                        (sig0, lam0)
                    };
                    if i == k && j == l && (ia, ib) < (ic, id) {
                        continue;
                    }
                    let val = block[((ia * nb + ib) * nc + ic) * nd + id];
                    if val != 0.0 {
                        let s1 = if mu == nu { 2.0 } else { 1.0 };
                        let s2 = if lam == sig { 2.0 } else { 1.0 };
                        let s3 = if mu == lam && nu == sig { 2.0 } else { 1.0 };
                        let v = val / (s1 * s2 * s3);

                        // Coulomb contraction with total density pt
                        let p_ls = 2.0 * v * pt[lam * n + sig];
                        let p_mn = 2.0 * v * pt[mu * n + nu];

                        ga[mu * n + nu] += p_ls;
                        ga[nu * n + mu] += p_ls;
                        ga[lam * n + sig] += p_mn;
                        ga[sig * n + lam] += p_mn;

                        gb[mu * n + nu] += p_ls;
                        gb[nu * n + mu] += p_ls;
                        gb[lam * n + sig] += p_mn;
                        gb[sig * n + lam] += p_mn;

                        // Exchange contraction
                        if has_exchange {
                            let vc = exx_fraction * v;

                            // Alpha exchange
                            let ka1 = vc * pa[nu * n + sig];
                            let ka2 = vc * pa[mu * n + sig];
                            let ka3 = vc * pa[nu * n + lam];
                            let ka4 = vc * pa[mu * n + lam];
                            ga[mu * n + lam] -= ka1;
                            ga[lam * n + mu] -= ka1;
                            ga[nu * n + lam] -= ka2;
                            ga[lam * n + nu] -= ka2;
                            ga[mu * n + sig] -= ka3;
                            ga[sig * n + mu] -= ka3;
                            ga[nu * n + sig] -= ka4;
                            ga[sig * n + nu] -= ka4;

                            // Beta exchange
                            let kb1 = vc * pb[nu * n + sig];
                            let kb2 = vc * pb[mu * n + sig];
                            let kb3 = vc * pb[nu * n + lam];
                            let kb4 = vc * pb[mu * n + lam];
                            gb[mu * n + lam] -= kb1;
                            gb[lam * n + mu] -= kb1;
                            gb[nu * n + lam] -= kb2;
                            gb[lam * n + nu] -= kb2;
                            gb[mu * n + sig] -= kb3;
                            gb[sig * n + mu] -= kb3;
                            gb[nu * n + sig] -= kb4;
                            gb[sig * n + nu] -= kb4;
                        }
                    }
                }
            }
        }
    }
}

/// High-performance Direct SCF Fock Matrix Engine.
///
/// Computes Coulomb and exact-exchange contributions on-the-fly without storing
/// the O(N^4) ERI tensor, supporting closed-shell and open-shell systems with
/// dynamic density-weighted Schwarz screening and lock-free Rayon parallelism.
#[derive(Debug, Clone)]
pub struct DirectFockEngine {
    pub n_ao: usize,
    pub n_shells: usize,
    pub screening: SchwarzScreening,
    pub sig_pairs: Vec<ShellPair>,
    exchange_rows: Arc<Vec<OnceLock<Mutex<Vec<f64>>>>>,
    pair_prefix: Arc<Vec<usize>>,
    cached_kets: Arc<Vec<usize>>,
}

impl DirectFockEngine {
    /// Initializes a Direct Fock engine with precomputed Schwarz screening bounds and significant shell pairs.
    pub fn new(basis: &BasisSet, threshold: f64) -> Self {
        let screening = SchwarzScreening::new(basis, threshold);
        // A density-independent pair cutoff cannot bound a later contraction with
        // an arbitrary density. Retain pairs and screen complete quartets using D.
        let mut sig_pairs = find_significant_shell_pairs(basis, &screening, 0.0);
        sig_pairs.sort_by(|a,b| b.bound.total_cmp(&a.bound));
        let bytes = std::env::var("CALCHEM_MAX_K_CACHE_BYTES").ok()
            .and_then(|v| v.parse::<usize>().ok()).unwrap_or(1536 << 20);
        // Reserve the highest Schwarz products first. Prefix offsets allow a lookup
        // without hashing or one allocation per quartet. Cache policy changes no integral cutoff.
        let mut prefix = vec![0usize];
        for p in &sig_pairs {
            prefix.push(prefix.last().unwrap() + basis.shells[p.shell_i].n_functions()
                * basis.shells[p.shell_j].n_functions());
        }
        let extent = |pi: usize, cutoff: f64| sig_pairs[..=pi].partition_point(|q|
            sig_pairs[pi].bound*q.bound > cutoff);
        let fits = |cutoff| {
            let mut elements = 0u128;
            for pi in 0..sig_pairs.len() {
                elements += (prefix[pi+1]-prefix[pi]) as u128 * prefix[extent(pi,cutoff)] as u128;
                if elements*8 > bytes as u128 { return false; }
            }
            true
        };
        let mut low = -36.0;
        let mut high = sig_pairs.first().map_or(0.0, |p| (p.bound*p.bound).max(1.0).log10()+1.0);
        if !fits(10f64.powf(low)) {
            for _ in 0..40 {
                let mid = (low+high)*0.5;
                if fits(10f64.powf(mid)) {high=mid;} else {low=mid;}
            }
        } else { high=low; }
        let cutoff=10f64.powf(high);
        let cached_kets=Arc::new((0..sig_pairs.len()).map(|pi| extent(pi,cutoff)).collect());
        let pair_prefix=Arc::new(prefix);
        let exchange_rows = Arc::new((0..sig_pairs.len()).map(|_| OnceLock::new()).collect());
        Self {
            n_ao: basis.n_ao,
            n_shells: basis.shells.len(),
            screening,
            sig_pairs,
            exchange_rows, pair_prefix, cached_kets,
        }
    }

    /// Determines the optimal bounded number of worker buffers based on available memory budget.
    pub fn determine_worker_count(&self, is_open_shell: bool, max_workers: Option<usize>) -> usize {
        let n = self.n_ao;
        let channels = if is_open_shell { 2 } else { 1 };
        let buffer_bytes = n * n * 8 * channels;
        if buffer_bytes == 0 {
            return 1;
        }

        let total_threads = rayon::current_num_threads().max(1);
        let requested_threads = max_workers.unwrap_or(total_threads).clamp(1, total_threads);

        // Budget from env var CALCHEM_MAX_DIRECT_BYTES, default 4 GiB dedicated to accumulators
        let budget_bytes: usize = std::env::var("CALCHEM_MAX_DIRECT_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4 * 1024 * 1024 * 1024);

        let max_affordable = (budget_bytes / buffer_bytes).max(1);
        requested_threads.min(max_affordable)
    }

    /// Evaluates the closed-shell two-electron Fock matrix `G = J[P] - 0.5 * exx * K[P]`
    /// on-the-fly using a strictly bounded pool of `num_workers` thread-local accumulators,
    /// eliminating unbounded task allocation and providing deterministic reduction.
    pub fn build_rks_bounded(
        &self,
        basis: &BasisSet,
        p: &[f64],
        exx_fraction: f64,
        threshold: f64,
        num_workers: usize,
    ) -> (Vec<f64>, DirectFockStats) {
        let n = self.n_ao;
        let n_shells = self.n_shells;
        let d = compute_shell_pair_l1_density(basis, p);
        let has_exchange = exx_fraction.abs() > 1e-15;

        let num_pairs = self.sig_pairs.len();
        if num_pairs == 0 {
            return (vec![0.0f64; n * n], DirectFockStats::default());
        }

        let k_workers = self.determine_worker_count(false, Some(num_workers)).min(num_pairs);
        let total_candidates = num_pairs * (num_pairs + 1) / 2;

        let mut worker_buffers: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_stats: Vec<DirectFockStats> = vec![DirectFockStats::default(); k_workers];

        if k_workers == 1 {
            let local_g = &mut worker_buffers[0];
            let local_stats = &mut worker_stats[0];
            let mut block = Vec::with_capacity(1296);
            for p_idx in 0..num_pairs {
                let pair_p = &self.sig_pairs[p_idx];
                let i = pair_p.shell_i;
                let j = pair_p.shell_j;
                let b_ij = pair_p.bound;
                let d_ij = d[i * n_shells + j];
                let sh_a = &basis.shells[i];
                let sh_b = &basis.shells[j];

                for q_idx in 0..=p_idx {
                    local_stats.total_quartet_candidates += 1;
                    let pair_q = &self.sig_pairs[q_idx];
                    let k = pair_q.shell_i;
                    let l = pair_q.shell_j;
                    let b_kl = pair_q.bound;
                    let b_ijkl = b_ij * b_kl;

                    let d_kl = d[k * n_shells + l];
                    let bound_j = 4.0 * b_ijkl * d_ij.max(d_kl);
                    let bound = if has_exchange {
                        let d_ik = d[i * n_shells + k];
                        let d_il = d[i * n_shells + l];
                        let d_jk = d[j * n_shells + k];
                        let d_jl = d[j * n_shells + l];
                        let d_exch = d_ik.max(d_il).max(d_jk).max(d_jl);
                        let bound_k = 4.0 * exx_fraction.abs() * b_ijkl * d_exch;
                        bound_j + bound_k
                    } else {
                        bound_j
                    };

                    if threshold > 0.0 && bound < threshold {
                        local_stats.skipped_quartets += 1;
                        local_stats.cumulative_bound_error += bound;
                        if bound > local_stats.max_skipped_bound {
                            local_stats.max_skipped_bound = bound;
                        }
                        continue;
                    }

                    local_stats.computed_quartets += 1;
                    let sh_c = &basis.shells[k];
                    let sh_d = &basis.shells[l];
                    eri_shell_quartet_with_omega_into(sh_a, sh_b, sh_c, sh_d, None, &mut block);

                    contract_quartet_rks(
                        i,
                        j,
                        k,
                        l,
                        sh_a,
                        sh_b,
                        sh_c,
                        sh_d,
                        &block,
                        p,
                        n,
                        exx_fraction,
                        local_g,
                    );
                }
            }
        } else {
            let d_ref = &d;
            rayon::scope(|s| {
                for (worker, (local_g, local_stats)) in worker_buffers.iter_mut().zip(worker_stats.iter_mut()).enumerate()
                {
                    s.spawn(move |_| {
                        let mut block = Vec::with_capacity(1296);
                        for p_idx in (worker..num_pairs).step_by(k_workers) {
                        let pair_p = &self.sig_pairs[p_idx];
                        let i = pair_p.shell_i;
                        let j = pair_p.shell_j;
                        let b_ij = pair_p.bound;
                        let d_ij = d_ref[i * n_shells + j];
                        let sh_a = &basis.shells[i];
                        let sh_b = &basis.shells[j];

                        for q_idx in 0..=p_idx {
                            local_stats.total_quartet_candidates += 1;
                            let pair_q = &self.sig_pairs[q_idx];
                            let k = pair_q.shell_i;
                            let l = pair_q.shell_j;
                            let b_kl = pair_q.bound;
                            let b_ijkl = b_ij * b_kl;

                            let d_kl = d_ref[k * n_shells + l];
                            let bound_j = 4.0 * b_ijkl * d_ij.max(d_kl);
                            let bound = if has_exchange {
                                let d_ik = d_ref[i * n_shells + k];
                                let d_il = d_ref[i * n_shells + l];
                                let d_jk = d_ref[j * n_shells + k];
                                let d_jl = d_ref[j * n_shells + l];
                                let d_exch = d_ik.max(d_il).max(d_jk).max(d_jl);
                                let bound_k = 4.0 * exx_fraction.abs() * b_ijkl * d_exch;
                                bound_j + bound_k
                            } else {
                                bound_j
                            };

                            if threshold > 0.0 && bound < threshold {
                                local_stats.skipped_quartets += 1;
                                local_stats.cumulative_bound_error += bound;
                                if bound > local_stats.max_skipped_bound {
                                    local_stats.max_skipped_bound = bound;
                                }
                                continue;
                            }

                            local_stats.computed_quartets += 1;
                            let sh_c = &basis.shells[k];
                            let sh_d = &basis.shells[l];
                            eri_shell_quartet_with_omega_into(
                                sh_a,
                                sh_b,
                                sh_c,
                                sh_d,
                                None,
                                &mut block,
                            );

                            contract_quartet_rks(
                                i,
                                j,
                                k,
                                l,
                                sh_a,
                                sh_b,
                                sh_c,
                                sh_d,
                                &block,
                                p,
                                n,
                                exx_fraction,
                                local_g,
                            );
                        }
                        }
                    });
                }
            });
        }

        let mut final_g = worker_buffers.remove(0);
        let mut final_stats = worker_stats.remove(0);

        for (other_buf, other_stat) in worker_buffers.into_iter().zip(worker_stats.into_iter()) {
            for idx in 0..final_g.len() {
                final_g[idx] += other_buf[idx];
            }
            final_stats.computed_quartets += other_stat.computed_quartets;
            final_stats.skipped_quartets += other_stat.skipped_quartets;
            final_stats.cumulative_bound_error += other_stat.cumulative_bound_error;
            if other_stat.max_skipped_bound > final_stats.max_skipped_bound {
                final_stats.max_skipped_bound = other_stat.max_skipped_bound;
            }
        }

        final_stats.significant_shell_pairs = num_pairs;
        final_stats.total_quartet_candidates = total_candidates;
        (final_g, final_stats)
    }

    /// Evaluates the closed-shell two-electron Fock matrix `G = J[P] - 0.5 * exx * K[P]`
    /// on-the-fly directly contracting with density `p` using automatically bounded parallelism,
    /// returning both the matrix and rigorous screening statistics.
    pub fn build_rks_with_stats(
        &self,
        basis: &BasisSet,
        p: &[f64],
        exx_fraction: f64,
        threshold: f64,
    ) -> (Vec<f64>, DirectFockStats) {
        let workers = self.determine_worker_count(false, None);
        self.build_rks_bounded(basis, p, exx_fraction, threshold, workers)
    }

    /// Evaluates the closed-shell two-electron Fock matrix `G = J[P] - 0.5 * exx * K[P]`
    /// on-the-fly directly contracting with density `p` using Rayon parallelization.
    pub fn build_rks(
        &self,
        basis: &BasisSet,
        p: &[f64],
        exx_fraction: f64,
        threshold: f64,
    ) -> Vec<f64> {
        self.build_rks_with_stats(basis, p, exx_fraction, threshold)
            .0
    }

    /// Evaluates exact exchange matrix `K_{mu,nu}[P] = sum_{lam,sig} (mu lam|nu sig)_omega P_{lam,sig}`
    /// on-the-fly with optional range-separation parameter `omega` using a bounded worker pool.
    pub fn build_k_with_omega_bounded(
        &self,
        basis: &BasisSet,
        p: &[f64],
        omega: Option<f64>,
        threshold: f64,
        num_workers: usize,
    ) -> Vec<f64> {
        let n = self.n_ao;
        let n_shells = self.n_shells;
        let d = compute_shell_pair_l1_density(basis, p);
        let num_pairs = self.sig_pairs.len();
        if num_pairs == 0 {
            return vec![0.0f64; n * n];
        }

        let k_workers = self.determine_worker_count(false, Some(num_workers)).min(num_pairs);
        let mut worker_buffers: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();

        let max_density = d.iter().copied().fold(0.0_f64, f64::max);
        let largest = self.sig_pairs[0].bound;
        let num_pairs = self.sig_pairs.partition_point(|q| threshold <= 0.0 ||
            4.0*(largest*q.bound)*max_density >= threshold);
        let primitive_pairs: Vec<_> = self.sig_pairs[..num_pairs].iter().map(|p|
            collect_egp_pairs(&basis.shells[p.shell_i], &basis.shells[p.shell_j])).collect();
        let row_max: Vec<_> = d.chunks_exact(n_shells).map(|row|
            row.iter().copied().fold(0.0_f64,f64::max)).collect();
        worker_buffers.par_iter_mut().enumerate().for_each(|(worker, local_k)| {
            let mut block = Vec::new();
            for pi in (worker..num_pairs).step_by(k_workers) {
                let bra = &self.sig_pairs[pi];
                let (i,j) = (bra.shell_i,bra.shell_j);
                let (a,b) = (&basis.shells[i], &basis.shells[j]);
                let candidates = &self.sig_pairs[..=pi];
                let end = candidates.partition_point(|q|
                    threshold <= 0.0 || 4.0 * (bra.bound * q.bound) * row_max[i].max(row_max[j]) >= threshold);
                let significant = |q: &ShellPair| {
                    let (k,l) = (q.shell_i,q.shell_j);
                    let dm = d[i*n_shells+k].max(d[i*n_shells+l]).max(d[j*n_shells+k]).max(d[j*n_shells+l]);
                    threshold <= 0.0 || 4.0 * (bra.bound * q.bound) * dm >= threshold
                };
                let evaluate = |qi: usize, out: &mut Vec<f64>| {
                    let q = &candidates[qi];
                    eri_from_pairs(a,b,&basis.shells[q.shell_i],&basis.shells[q.shell_j],
                        &primitive_pairs[pi],&primitive_pairs[qi],omega,out);
                };
                let bra_width = a.n_functions()*b.n_functions();
                let count = if omega.is_none() { self.cached_kets[pi] } else { 0 };
                let mut cache = if count > 0 && end > 0 {
                    Some(self.exchange_rows[pi].get_or_init(|| Mutex::new(vec![f64::NAN;
                        bra_width*self.pair_prefix[count]])).lock().unwrap())
                } else { None };
                for (qi,q) in candidates[..end].iter().enumerate() {
                    if !significant(q) { continue; }
                    let vals = if qi < count {
                        let start = self.pair_prefix[qi]*bra_width;
                        let stop = self.pair_prefix[qi+1]*bra_width;
                        let slot = &mut cache.as_mut().unwrap()[start..stop];
                        if slot[0].is_nan() {
                            evaluate(qi,&mut block);
                            slot.copy_from_slice(&block);
                        }
                        &slot[..]
                    } else {
                        evaluate(qi,&mut block);
                        &block[..]
                    };
                    contract_quartet_k(i,j,q.shell_i,q.shell_j,a,b,&basis.shells[q.shell_i],
                        &basis.shells[q.shell_j],vals,p,n,local_k);
                }
            }
        });
        let mut final_k = worker_buffers.remove(0);
        for w_buf in worker_buffers {
            for idx in 0..n * n {
                final_k[idx] += w_buf[idx];
            }
        }
        // Symmetrize K = K_raw + K_raw^T (matches the paired writes hoisted out of contract_quartet_k)
        for r in 0..n {
            final_k[r * n + r] *= 2.0;
            for c in (r + 1)..n {
                let s = final_k[r * n + c] + final_k[c * n + r];
                final_k[r * n + c] = s;
                final_k[c * n + r] = s;
            }
        }
        final_k
    }

    /// Evaluates exact exchange matrix `K[P]` on-the-fly with optional range-separation parameter `omega`.
    pub fn build_k_with_omega(&self, basis: &BasisSet, p: &[f64], omega: Option<f64>) -> Vec<f64> {
        let workers = self.determine_worker_count(false, None);
        let thresh = std::env::var("CALCHEM_DIRECT_THRESH")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-9);
        self.build_k_with_omega_bounded(basis, p, omega, thresh, workers)
    }

    /// Evaluates general (non-symmetric, symmetric, or antisymmetric) exact exchange matrix
    /// `K_{mu,nu}[X] = sum_{lam,sig} (mu lam | nu sig)_omega X_{lam,sig}` on-the-fly with
    /// optional range-separation parameter `omega`, for AO-direct TD-DFT/TDA response.
    pub fn build_k_general_with_omega(
        &self,
        basis: &BasisSet,
        x: &[f64],
        omega: Option<f64>,
        threshold: f64,
    ) -> Vec<f64> {
        let n = self.n_ao;
        let n_shells = self.n_shells;
        // Symmetrized L1 shell-pair bound max(|X_ab|, |X_ba|) guarantees rigorous screening for general X.
        let mut d = vec![0.0f64; n_shells * n_shells];
        for (i, sh_i) in basis.shells.iter().enumerate() {
            let off_i = sh_i.ao_offset;
            let ni = sh_i.n_functions();
            for (j, sh_j) in basis.shells.iter().enumerate().skip(i) {
                let off_j = sh_j.ao_offset;
                let nj = sh_j.n_functions();
                let mut l1 = 0.0f64;
                for a in 0..ni {
                    let mu = off_i + a;
                    for b in 0..nj {
                        let nu = off_j + b;
                        let m = x[mu * n + nu].abs().max(x[nu * n + mu].abs());
                        l1 += m;
                    }
                }
                d[i * n_shells + j] = l1;
                d[j * n_shells + i] = l1;
            }
        }
        let num_pairs = self.sig_pairs.len();
        if num_pairs == 0 {
            return vec![0.0f64; n * n];
        }

        let workers = self.determine_worker_count(false, None);
        let k_workers = workers.min(num_pairs).max(1);
        let mut worker_buffers: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();

        let max_density = d.iter().copied().fold(0.0_f64, f64::max);
        if max_density == 0.0 {
            return vec![0.0f64; n * n];
        }
        let largest = self.sig_pairs[0].bound;
        let active_pairs = self.sig_pairs.partition_point(|q| {
            threshold <= 0.0 || 4.0 * (largest * q.bound) * max_density >= threshold
        });
        if active_pairs == 0 {
            return vec![0.0f64; n * n];
        }
        let primitive_pairs: Vec<_> = self.sig_pairs[..active_pairs]
            .iter()
            .map(|p| collect_egp_pairs(&basis.shells[p.shell_i], &basis.shells[p.shell_j]))
            .collect();
        let row_max: Vec<_> = d
            .chunks_exact(n_shells)
            .map(|row| row.iter().copied().fold(0.0_f64, f64::max))
            .collect();

        worker_buffers
            .par_iter_mut()
            .enumerate()
            .for_each(|(worker, local_k)| {
                let mut block = Vec::new();
                for pi in (worker..active_pairs).step_by(k_workers) {
                    let bra = &self.sig_pairs[pi];
                    let (i, j) = (bra.shell_i, bra.shell_j);
                    let (a, b) = (&basis.shells[i], &basis.shells[j]);
                    let candidates = &self.sig_pairs[..=pi];
                    let end = candidates.partition_point(|q| {
                        threshold <= 0.0
                            || 4.0 * (bra.bound * q.bound) * row_max[i].max(row_max[j]) >= threshold
                    });
                    let bra_width = a.n_functions() * b.n_functions();
                    let count = if omega.is_none() { self.cached_kets[pi] } else { 0 };
                    let mut cache = if count > 0 && end > 0 {
                        Some(
                            self.exchange_rows[pi]
                                .get_or_init(|| {
                                    Mutex::new(vec![f64::NAN; bra_width * self.pair_prefix[count]])
                                })
                                .lock()
                                .unwrap(),
                        )
                    } else {
                        None
                    };
                    for (qi, q) in candidates[..end].iter().enumerate() {
                        let (k, l) = (q.shell_i, q.shell_j);
                        let dm = d[i * n_shells + k]
                            .max(d[i * n_shells + l])
                            .max(d[j * n_shells + k])
                            .max(d[j * n_shells + l]);
                        if threshold > 0.0 && 4.0 * (bra.bound * q.bound) * dm < threshold {
                            continue;
                        }
                        let vals = if qi < count {
                            let start = self.pair_prefix[qi] * bra_width;
                            let stop = self.pair_prefix[qi + 1] * bra_width;
                            let slot = &mut cache.as_mut().unwrap()[start..stop];
                            if slot[0].is_nan() {
                                eri_from_pairs(
                                    a,
                                    b,
                                    &basis.shells[k],
                                    &basis.shells[l],
                                    &primitive_pairs[pi],
                                    &primitive_pairs[qi],
                                    omega,
                                    &mut block,
                                );
                                slot.copy_from_slice(&block);
                            }
                            &slot[..]
                        } else {
                            eri_from_pairs(
                                a,
                                b,
                                &basis.shells[k],
                                &basis.shells[l],
                                &primitive_pairs[pi],
                                &primitive_pairs[qi],
                                omega,
                                &mut block,
                            );
                            &block[..]
                        };
                        contract_quartet_k_general(
                            i,
                            j,
                            k,
                            l,
                            a,
                            b,
                            &basis.shells[k],
                            &basis.shells[l],
                            vals,
                            x,
                            n,
                            local_k,
                        );
                    }
                }
            });

        let mut final_k = worker_buffers.remove(0);
        for w_buf in worker_buffers {
            for idx in 0..n * n {
                final_k[idx] += w_buf[idx];
            }
        }
        final_k
    }

    pub fn build_k(&self, basis: &BasisSet, p: &[f64]) -> Vec<f64> {
        self.build_k_with_omega(basis, p, None)
    }

    /// Evaluates open-shell two-electron Fock matrices `(G_alpha, G_beta)`
    /// on-the-fly directly contracting with `pa` and `pb` in a single pass using a strictly
    /// bounded pool of `num_workers` thread-local accumulators.
    pub fn build_uks_bounded(
        &self,
        basis: &BasisSet,
        pa: &[f64],
        pb: &[f64],
        exx_fraction: f64,
        threshold: f64,
        num_workers: usize,
    ) -> (Vec<f64>, Vec<f64>, DirectFockStats) {
        let n = self.n_ao;
        let n_shells = self.n_shells;
        let mut pt = vec![0.0f64; n * n];
        for idx in 0..n * n {
            pt[idx] = pa[idx] + pb[idx];
        }

        let dt = compute_shell_pair_l1_density(basis, &pt);
        let da = compute_shell_pair_l1_density(basis, pa);
        let db = compute_shell_pair_l1_density(basis, pb);
        let has_exchange = exx_fraction.abs() > 1e-15;

        let num_pairs = self.sig_pairs.len();
        if num_pairs == 0 {
            return (
                vec![0.0f64; n * n],
                vec![0.0f64; n * n],
                DirectFockStats::default(),
            );
        }

        let k_workers = self.determine_worker_count(true, Some(num_workers)).min(num_pairs);
        let total_candidates = num_pairs * (num_pairs + 1) / 2;

        let mut worker_buffers_a: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_buffers_b: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_stats: Vec<DirectFockStats> = vec![DirectFockStats::default(); k_workers];

        if k_workers == 1 {
            let local_ga = &mut worker_buffers_a[0];
            let local_gb = &mut worker_buffers_b[0];
            let local_stats = &mut worker_stats[0];
            let mut block = Vec::with_capacity(1296);

            for p_idx in 0..num_pairs {
                let pair_p = &self.sig_pairs[p_idx];
                let i = pair_p.shell_i;
                let j = pair_p.shell_j;
                let b_ij = pair_p.bound;
                let dt_ij = dt[i * n_shells + j];
                let sh_a = &basis.shells[i];
                let sh_b = &basis.shells[j];

                for q_idx in 0..=p_idx {
                    local_stats.total_quartet_candidates += 1;
                    let pair_q = &self.sig_pairs[q_idx];
                    let k = pair_q.shell_i;
                    let l = pair_q.shell_j;
                    let b_kl = pair_q.bound;
                    let b_ijkl = b_ij * b_kl;

                    let dt_kl = dt[k * n_shells + l];
                    let bound_j = 4.0 * b_ijkl * dt_ij.max(dt_kl);
                    let bound = if has_exchange {
                        let da_exch = da[i * n_shells + k]
                            .max(da[i * n_shells + l])
                            .max(da[j * n_shells + k])
                            .max(da[j * n_shells + l]);
                        let db_exch = db[i * n_shells + k]
                            .max(db[i * n_shells + l])
                            .max(db[j * n_shells + k])
                            .max(db[j * n_shells + l]);
                        let bound_k = 4.0 * exx_fraction.abs() * b_ijkl * da_exch.max(db_exch);
                        bound_j + bound_k
                    } else {
                        bound_j
                    };

                    if threshold > 0.0 && bound < threshold {
                        local_stats.skipped_quartets += 1;
                        local_stats.cumulative_bound_error += bound;
                        if bound > local_stats.max_skipped_bound {
                            local_stats.max_skipped_bound = bound;
                        }
                        continue;
                    }

                    local_stats.computed_quartets += 1;
                    let sh_c = &basis.shells[k];
                    let sh_d = &basis.shells[l];
                    eri_shell_quartet_with_omega_into(sh_a, sh_b, sh_c, sh_d, None, &mut block);

                    contract_quartet_uks(
                        i,
                        j,
                        k,
                        l,
                        sh_a,
                        sh_b,
                        sh_c,
                        sh_d,
                        &block,
                        pa,
                        pb,
                        &pt,
                        n,
                        exx_fraction,
                        local_ga,
                        local_gb,
                    );
                }
            }
        } else {
            let dt_ref = &dt;
            let da_ref = &da;
            let db_ref = &db;
            let pt_ref = &pt;
            rayon::scope(|s| {
                for (worker, ((local_ga, local_gb), local_stats)) in worker_buffers_a
                    .iter_mut()
                    .zip(worker_buffers_b.iter_mut())
                    .zip(worker_stats.iter_mut()).enumerate()
                {
                    s.spawn(move |_| {
                        let mut block = Vec::with_capacity(1296);
                        for p_idx in (worker..num_pairs).step_by(k_workers) {
                        let pair_p = &self.sig_pairs[p_idx];
                        let i = pair_p.shell_i;
                        let j = pair_p.shell_j;
                        let b_ij = pair_p.bound;
                        let dt_ij = dt_ref[i * n_shells + j];
                        let sh_a = &basis.shells[i];
                        let sh_b = &basis.shells[j];

                        for q_idx in 0..=p_idx {
                            local_stats.total_quartet_candidates += 1;
                            let pair_q = &self.sig_pairs[q_idx];
                            let k = pair_q.shell_i;
                            let l = pair_q.shell_j;
                            let b_kl = pair_q.bound;
                            let b_ijkl = b_ij * b_kl;

                            let dt_kl = dt_ref[k * n_shells + l];
                            let bound_j = 4.0 * b_ijkl * dt_ij.max(dt_kl);
                            let bound = if has_exchange {
                                let da_exch = da_ref[i * n_shells + k]
                                    .max(da_ref[i * n_shells + l])
                                    .max(da_ref[j * n_shells + k])
                                    .max(da_ref[j * n_shells + l]);
                                let db_exch = db_ref[i * n_shells + k]
                                    .max(db_ref[i * n_shells + l])
                                    .max(db_ref[j * n_shells + k])
                                    .max(db_ref[j * n_shells + l]);
                                let bound_k =
                                    4.0 * exx_fraction.abs() * b_ijkl * da_exch.max(db_exch);
                                bound_j + bound_k
                            } else {
                                bound_j
                            };

                            if threshold > 0.0 && bound < threshold {
                                local_stats.skipped_quartets += 1;
                                local_stats.cumulative_bound_error += bound;
                                if bound > local_stats.max_skipped_bound {
                                    local_stats.max_skipped_bound = bound;
                                }
                                continue;
                            }

                            local_stats.computed_quartets += 1;
                            let sh_c = &basis.shells[k];
                            let sh_d = &basis.shells[l];
                            eri_shell_quartet_with_omega_into(
                                sh_a,
                                sh_b,
                                sh_c,
                                sh_d,
                                None,
                                &mut block,
                            );

                            contract_quartet_uks(
                                i,
                                j,
                                k,
                                l,
                                sh_a,
                                sh_b,
                                sh_c,
                                sh_d,
                                &block,
                                pa,
                                pb,
                                pt_ref,
                                n,
                                exx_fraction,
                                local_ga,
                                local_gb,
                            );
                        }
                        }
                    });
                }
            });
        }

        let mut final_ga = worker_buffers_a.remove(0);
        let mut final_gb = worker_buffers_b.remove(0);
        let mut final_stats = worker_stats.remove(0);

        for ((b_a, b_b), other_stat) in worker_buffers_a
            .into_iter()
            .zip(worker_buffers_b.into_iter())
            .zip(worker_stats.into_iter())
        {
            for idx in 0..final_ga.len() {
                final_ga[idx] += b_a[idx];
                final_gb[idx] += b_b[idx];
            }
            final_stats.computed_quartets += other_stat.computed_quartets;
            final_stats.skipped_quartets += other_stat.skipped_quartets;
            final_stats.cumulative_bound_error += other_stat.cumulative_bound_error;
            if other_stat.max_skipped_bound > final_stats.max_skipped_bound {
                final_stats.max_skipped_bound = other_stat.max_skipped_bound;
            }
        }

        final_stats.significant_shell_pairs = num_pairs;
        final_stats.total_quartet_candidates = total_candidates;
        (final_ga, final_gb, final_stats)
    }

    /// Evaluates open-shell two-electron Fock matrices `(G_alpha, G_beta)`
    /// on-the-fly directly contracting with `pa` and `pb` in a single pass,
    /// returning both matrices and rigorous screening statistics.
    pub fn build_uks_with_stats(
        &self,
        basis: &BasisSet,
        pa: &[f64],
        pb: &[f64],
        exx_fraction: f64,
        threshold: f64,
    ) -> (Vec<f64>, Vec<f64>, DirectFockStats) {
        let workers = self.determine_worker_count(true, None);
        self.build_uks_bounded(basis, pa, pb, exx_fraction, threshold, workers)
    }

    /// Evaluates open-shell two-electron Fock matrices `(G_alpha, G_beta)`
    /// on-the-fly directly contracting with `pa` and `pb` in a single pass.
    pub fn build_uks(
        &self,
        basis: &BasisSet,
        pa: &[f64],
        pb: &[f64],
        exx_fraction: f64,
        threshold: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let (ga, gb, _) = self.build_uks_with_stats(basis, pa, pb, exx_fraction, threshold);
        (ga, gb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_sparse_eri;
    use calchem_model::{Atom, Molecule};

    #[test]
    fn test_direct_fock_rks_matches_sparse_eri_on_water() {
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
        let n = bs.n_ao;

        // Dummy symmetric density matrix
        let mut p = vec![0.0f64; n * n];
        for i in 0..n {
            p[i * n + i] = 1.0;
            for j in 0..i {
                let val = 0.1 / ((i + j + 1) as f64);
                p[i * n + j] = val;
                p[j * n + i] = val;
            }
        }

        let sparse_eri = compute_sparse_eri(&bs);
        let direct_engine = DirectFockEngine::new(&bs, 1e-12);

        for exx in [0.0, 0.2, 1.0] {
            let g_sparse = sparse_eri.contract_rks(&p, exx);
            let g_direct = direct_engine.build_rks(&bs, &p, exx, 1e-12);

            let mut max_diff = 0.0f64;
            for idx in 0..n * n {
                let diff = (g_sparse[idx] - g_direct[idx]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            assert!(
                max_diff < 1e-11,
                "Direct vs Sparse ERI diff {max_diff} exceeded tolerance for exx={exx}"
            );
        }
    }

    #[test]
    fn test_direct_fock_uks_matches_dense_tensor_on_water() {
        use crate::compute_eri;

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
        let n = bs.n_ao;

        let mut pa = vec![0.0f64; n * n];
        let mut pb = vec![0.0f64; n * n];
        for i in 0..n {
            pa[i * n + i] = 0.6;
            pb[i * n + i] = 0.4;
            for j in 0..i {
                let val = 0.05 / ((i + j + 1) as f64);
                pa[i * n + j] = val * 1.1;
                pa[j * n + i] = val * 1.1;
                pb[i * n + j] = val * 0.9;
                pb[j * n + i] = val * 0.9;
            }
        }

        let eri = compute_eri(&bs);
        let direct_engine = DirectFockEngine::new(&bs, 1e-12);

        for exx in [0.0, 0.2, 1.0] {
            let (ga_dir, gb_dir) = direct_engine.build_uks(&bs, &pa, &pb, exx, 1e-12);

            // Reference from Tensor4D:
            for mu in 0..n {
                for nu in 0..n {
                    let mut j = 0.0;
                    let mut ka = 0.0;
                    let mut kb = 0.0;
                    for lam in 0..n {
                        for sig in 0..n {
                            let pt_ls = pa[lam * n + sig] + pb[lam * n + sig];
                            let coul = eri.get(mu, nu, lam, sig);
                            let exch = eri.get(mu, lam, nu, sig);
                            j += pt_ls * coul;
                            ka += pa[lam * n + sig] * exch;
                            kb += pb[lam * n + sig] * exch;
                        }
                    }
                    let ga_ref = j - exx * ka;
                    let gb_ref = j - exx * kb;

                    let diff_a = (ga_dir[mu * n + nu] - ga_ref).abs();
                    let diff_b = (gb_dir[mu * n + nu] - gb_ref).abs();
                    assert!(
                        diff_a < 1e-11,
                        "UKS Alpha diff {diff_a} at ({mu},{nu}) exceeded tolerance for exx={exx}"
                    );
                    assert!(
                        diff_b < 1e-11,
                        "UKS Beta diff {diff_b} at ({mu},{nu}) exceeded tolerance for exx={exx}"
                    );
                }
            }
        }
    }
}
