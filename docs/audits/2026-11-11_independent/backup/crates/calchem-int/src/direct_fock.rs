use crate::eri::{eri_shell_quartet, eri_shell_quartet_with_omega};
use crate::screening::SchwarzScreening;
use calchem_basis::{BasisSet, Shell};
use rayon::prelude::*;

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
                    if val.abs() > 1e-16 {
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
                    if val.abs() > 1e-16 {
                        let s1 = if mu == nu { 2.0 } else { 1.0 };
                        let s2 = if lam == sig { 2.0 } else { 1.0 };
                        let s3 = if mu == lam && nu == sig { 2.0 } else { 1.0 };
                        let v = val / (s1 * s2 * s3);

                        let k1 = v * p[nu * n + sig];
                        let k2 = v * p[mu * n + sig];
                        let k3 = v * p[nu * n + lam];
                        let k4 = v * p[mu * n + lam];
                        k_mat[mu * n + lam] += k1;
                        k_mat[lam * n + mu] += k1;
                        k_mat[nu * n + lam] += k2;
                        k_mat[lam * n + nu] += k2;
                        k_mat[mu * n + sig] += k3;
                        k_mat[sig * n + mu] += k3;
                        k_mat[nu * n + sig] += k4;
                        k_mat[sig * n + nu] += k4;
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
                    if val.abs() > 1e-16 {
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
}

impl DirectFockEngine {
    /// Initializes a Direct Fock engine with precomputed Schwarz screening bounds and significant shell pairs.
    pub fn new(basis: &BasisSet, threshold: f64) -> Self {
        let screening = SchwarzScreening::new(basis, threshold);
        let sig_pairs = find_significant_shell_pairs(basis, &screening, threshold);
        Self {
            n_ao: basis.n_ao,
            n_shells: basis.shells.len(),
            screening,
            sig_pairs,
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

        let k_workers = num_workers.max(1);
        let total_candidates = num_pairs * (num_pairs + 1) / 2;

        let mut worker_buffers: Vec<Vec<f64>> = (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_stats: Vec<DirectFockStats> = vec![DirectFockStats::default(); k_workers];

        if k_workers == 1 {
            let local_g = &mut worker_buffers[0];
            let local_stats = &mut worker_stats[0];
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
                        let bound_k = exx_fraction * b_ijkl * d_exch;
                        bound_j.max(bound_k)
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
                    let block = eri_shell_quartet(sh_a, sh_b, sh_c, sh_d);

                    contract_quartet_rks(
                        i, j, k, l, sh_a, sh_b, sh_c, sh_d, &block, p, n, exx_fraction, local_g,
                    );
                }
            }
        } else {
            let next_pair = std::sync::atomic::AtomicUsize::new(0);
            let d_ref = &d;
            rayon::scope(|s| {
                for (local_g, local_stats) in worker_buffers.iter_mut().zip(worker_stats.iter_mut()) {
                    let next_pair = &next_pair;
                    s.spawn(move |_| {
                        loop {
                            let p_idx = next_pair.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if p_idx >= num_pairs {
                                break;
                            }
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
                                    let bound_k = exx_fraction * b_ijkl * d_exch;
                                    bound_j.max(bound_k)
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
                                let block = eri_shell_quartet(sh_a, sh_b, sh_c, sh_d);

                                contract_quartet_rks(
                                    i, j, k, l, sh_a, sh_b, sh_c, sh_d, &block, p, n, exx_fraction, local_g,
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
        self.build_rks_with_stats(basis, p, exx_fraction, threshold).0
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

        let k_workers = num_workers.max(1);
        let mut worker_buffers: Vec<Vec<f64>> = (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();

        if k_workers == 1 {
            let local_k = &mut worker_buffers[0];
            for p_idx in 0..num_pairs {
                let pair_p = &self.sig_pairs[p_idx];
                let i = pair_p.shell_i;
                let j = pair_p.shell_j;
                let b_ij = pair_p.bound;
                let sh_a = &basis.shells[i];
                let sh_b = &basis.shells[j];

                for q_idx in 0..=p_idx {
                    let pair_q = &self.sig_pairs[q_idx];
                    let k = pair_q.shell_i;
                    let l = pair_q.shell_j;
                    let b_kl = pair_q.bound;
                    let b_ijkl = b_ij * b_kl;

                    let d_ik = d[i * n_shells + k];
                    let d_il = d[i * n_shells + l];
                    let d_jk = d[j * n_shells + k];
                    let d_jl = d[j * n_shells + l];
                    let d_exch = d_ik.max(d_il).max(d_jk).max(d_jl);
                    let bound = b_ijkl * d_exch;
                    if threshold > 0.0 && bound < threshold {
                        continue;
                    }

                    let sh_c = &basis.shells[k];
                    let sh_d = &basis.shells[l];
                    let block = eri_shell_quartet_with_omega(sh_a, sh_b, sh_c, sh_d, omega);
                    contract_quartet_k(i, j, k, l, sh_a, sh_b, sh_c, sh_d, &block, p, n, local_k);
                }
            }
        } else {
            let next_pair = std::sync::atomic::AtomicUsize::new(0);
            let d_ref = &d;
            rayon::scope(|s| {
                for local_k in worker_buffers.iter_mut() {
                    let next_pair = &next_pair;
                    s.spawn(move |_| {
                        loop {
                            let p_idx = next_pair.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if p_idx >= num_pairs {
                                break;
                            }
                            let pair_p = &self.sig_pairs[p_idx];
                            let i = pair_p.shell_i;
                            let j = pair_p.shell_j;
                            let b_ij = pair_p.bound;
                            let sh_a = &basis.shells[i];
                            let sh_b = &basis.shells[j];

                            for q_idx in 0..=p_idx {
                                let pair_q = &self.sig_pairs[q_idx];
                                let k = pair_q.shell_i;
                                let l = pair_q.shell_j;
                                let b_kl = pair_q.bound;
                                let b_ijkl = b_ij * b_kl;

                                let d_ik = d_ref[i * n_shells + k];
                                let d_il = d_ref[i * n_shells + l];
                                let d_jk = d_ref[j * n_shells + k];
                                let d_jl = d_ref[j * n_shells + l];
                                let d_exch = d_ik.max(d_il).max(d_jk).max(d_jl);
                                let bound = b_ijkl * d_exch;
                                if threshold > 0.0 && bound < threshold {
                                    continue;
                                }

                                let sh_c = &basis.shells[k];
                                let sh_d = &basis.shells[l];
                                let block = eri_shell_quartet_with_omega(sh_a, sh_b, sh_c, sh_d, omega);
                                contract_quartet_k(i, j, k, l, sh_a, sh_b, sh_c, sh_d, &block, p, n, local_k);
                            }
                        }
                    });
                }
            });
        }

        let mut final_k = vec![0.0f64; n * n];
        for w_buf in worker_buffers {
            for idx in 0..n * n {
                final_k[idx] += w_buf[idx];
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
            return (vec![0.0f64; n * n], vec![0.0f64; n * n], DirectFockStats::default());
        }

        let k_workers = num_workers.max(1);
        let total_candidates = num_pairs * (num_pairs + 1) / 2;

        let mut worker_buffers_a: Vec<Vec<f64>> = (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_buffers_b: Vec<Vec<f64>> = (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_stats: Vec<DirectFockStats> = vec![DirectFockStats::default(); k_workers];

        if k_workers == 1 {
            let local_ga = &mut worker_buffers_a[0];
            let local_gb = &mut worker_buffers_b[0];
            let local_stats = &mut worker_stats[0];

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
                        let bound_k = exx_fraction * b_ijkl * da_exch.max(db_exch);
                        bound_j.max(bound_k)
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
                    let block = eri_shell_quartet(sh_a, sh_b, sh_c, sh_d);

                    contract_quartet_uks(
                        i, j, k, l, sh_a, sh_b, sh_c, sh_d, &block, pa, pb, &pt, n, exx_fraction, local_ga, local_gb,
                    );
                }
            }
        } else {
            let next_pair = std::sync::atomic::AtomicUsize::new(0);
            let dt_ref = &dt;
            let da_ref = &da;
            let db_ref = &db;
            let pt_ref = &pt;
            rayon::scope(|s| {
                for ((local_ga, local_gb), local_stats) in worker_buffers_a
                    .iter_mut()
                    .zip(worker_buffers_b.iter_mut())
                    .zip(worker_stats.iter_mut())
                {
                    let next_pair = &next_pair;
                    s.spawn(move |_| {
                        loop {
                            let p_idx = next_pair.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if p_idx >= num_pairs {
                                break;
                            }
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
                                    let bound_k = exx_fraction * b_ijkl * da_exch.max(db_exch);
                                    bound_j.max(bound_k)
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
                                let block = eri_shell_quartet(sh_a, sh_b, sh_c, sh_d);

                                contract_quartet_uks(
                                    i, j, k, l, sh_a, sh_b, sh_c, sh_d, &block, pa, pb, pt_ref, n, exx_fraction, local_ga, local_gb,
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
