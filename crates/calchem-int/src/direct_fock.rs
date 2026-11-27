use rayon::prelude::*;
use crate::eri::{collect_egp_pairs, eri_from_pairs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use crate::screening::SchwarzScreening;
use calchem_basis::{BasisSet, Shell};
use calchem_tensor::Matrix2D;

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

#[inline]
fn symmetrize_in_place(mat: &mut [f64], n: usize) {
    for r in 0..n {
        mat[r * n + r] *= 2.0;
        for c in (r + 1)..n {
            let s = mat[r * n + c] + mat[c * n + r];
            mat[r * n + c] = s;
            mat[c * n + r] = s;
        }
    }
}

#[inline]
fn antisymmetrize_in_place(mat: &mut [f64], n: usize) {
    for r in 0..n {
        mat[r * n + r] = 0.0;
        for c in (r + 1)..n {
            let d = mat[r * n + c] - mat[c * n + r];
            mat[r * n + c] = d;
            mat[c * n + r] = -d;
        }
    }
}

#[inline]
fn compute_shell_pair_l1_abs_sym(basis: &BasisSet, x: &[f64]) -> Vec<f64> {
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
                let row_mu = mu * n_ao;
                for ib in 0..nb {
                    let nu = off_j + ib;
                    sum_val += x[row_mu + nu].abs().max(x[nu * n_ao + mu].abs());
                }
            }
            d[i * n_shells + j] = sum_val;
            d[j * n_shells + i] = sum_val;
        }
    }
    d
}

/// Internal half-write RKS quartet contraction used by `build_rks_bounded`.
/// Writes only one triangle/side of each symmetric contribution; caller must call
/// `symmetrize_in_place(&mut g_raw, n)` once after reducing worker buffers.
#[allow(clippy::too_many_arguments)]
#[inline]
fn contract_quartet_rks_raw(
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
    g_raw: &mut [f64],
) {
    let na = sh_a.n_functions();
    let nb = sh_b.n_functions();
    let nc = sh_c.n_functions();
    let nd = sh_d.n_functions();

    let has_exchange = exx_fraction.abs() > 1e-15;
    let half_c = 0.5 * exx_fraction;

    if i != j && k != l && (i != k || j != l) {
        let ao_a = sh_a.ao_offset;
        let ao_b = sh_b.ao_offset;
        let ao_c = sh_c.ao_offset;
        let ao_d = sh_d.ao_offset;

        if !has_exchange {
            for ia in 0..na {
                let mu = ao_a + ia;
                let mu_n = mu * n;
                let ia_nb = ia * nb;
                for ib in 0..nb {
                    let nu = ao_b + ib;
                    let p_mn2 = 2.0 * p[mu_n + nu];
                    let ab_nc = (ia_nb + ib) * nc;
                    let mut acc_mn = 0.0;
                    for ic in 0..nc {
                        let lam_n = (ao_c + ic) * n;
                        let abc_nd = (ab_nc + ic) * nd;
                        for id in 0..nd {
                            let v = block[abc_nd + id];
                            if v != 0.0 {
                                let sig = ao_d + id;
                                acc_mn += v * p[lam_n + sig];
                                g_raw[lam_n + sig] += v * p_mn2;
                            }
                        }
                    }
                    g_raw[mu_n + nu] += 2.0 * acc_mn;
                }
            }
            return;
        }

        for ia in 0..na {
            let mu = ao_a + ia;
            let mu_n = mu * n;
            let ia_nb = ia * nb;
            for ib in 0..nb {
                let nu = ao_b + ib;
                let nu_n = nu * n;
                let p_mn2 = 2.0 * p[mu_n + nu];
                let ab_nc = (ia_nb + ib) * nc;
                let mut acc_mn = 0.0;
                for ic in 0..nc {
                    let lam = ao_c + ic;
                    let lam_n = lam * n;
                    let c_nu_lam = half_c * p[nu_n + lam];
                    let c_mu_lam = half_c * p[mu_n + lam];
                    let abc_nd = (ab_nc + ic) * nd;
                    let mut acc_mu_lam = 0.0;
                    let mut acc_nu_lam = 0.0;
                    for id in 0..nd {
                        let v = block[abc_nd + id];
                        if v != 0.0 {
                            let sig = ao_d + id;
                            acc_mn += v * p[lam_n + sig];
                            g_raw[lam_n + sig] += v * p_mn2;
                            acc_mu_lam += v * p[nu_n + sig];
                            acc_nu_lam += v * p[mu_n + sig];
                            g_raw[mu_n + sig] -= v * c_nu_lam;
                            g_raw[nu_n + sig] -= v * c_mu_lam;
                        }
                    }
                    g_raw[mu_n + lam] -= half_c * acc_mu_lam;
                    g_raw[nu_n + lam] -= half_c * acc_nu_lam;
                }
                g_raw[mu_n + nu] += 2.0 * acc_mn;
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

                        g_raw[mu * n + nu] += 2.0 * v * p[lam * n + sig];
                        g_raw[lam * n + sig] += 2.0 * v * p[mu * n + nu];

                        if has_exchange {
                            let vc = half_c * v;
                            g_raw[mu * n + lam] -= vc * p[nu * n + sig];
                            g_raw[nu * n + lam] -= vc * p[mu * n + sig];
                            g_raw[mu * n + sig] -= vc * p[nu * n + lam];
                            g_raw[nu * n + sig] -= vc * p[mu * n + lam];
                        }
                    }
                }
            }
        }
    }
}

/// Internal half-write J+K quartet contraction for symmetric `p`.
/// Accumulates unscaled `+J[P]` into `j_raw` and unscaled `+K_raw[P]` into `k_raw`
/// in one pass over `block`; caller must call `symmetrize_in_place` on both outputs.
#[allow(clippy::too_many_arguments)]
#[inline]
fn contract_quartet_jk_sym_raw(
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
    j_raw: &mut [f64],
    k_raw: &mut [f64],
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
                let p_mn2 = 2.0 * p[mu_n + nu];
                let ab_nc = (ia_nb + ib) * nc;
                let mut acc_mn = 0.0;
                for ic in 0..nc {
                    let lam = ao_c + ic;
                    let lam_n = lam * n;
                    let p_nu_lam = p[nu_n + lam];
                    let p_mu_lam = p[mu_n + lam];
                    let abc_nd = (ab_nc + ic) * nd;
                    let mut acc_mu_lam = 0.0;
                    let mut acc_nu_lam = 0.0;
                    for id in 0..nd {
                        let v = block[abc_nd + id];
                        if v != 0.0 {
                            let sig = ao_d + id;
                            acc_mn += v * p[lam_n + sig];
                            j_raw[lam_n + sig] += v * p_mn2;
                            acc_mu_lam += v * p[nu_n + sig];
                            acc_nu_lam += v * p[mu_n + sig];
                            k_raw[mu_n + sig] += v * p_nu_lam;
                            k_raw[nu_n + sig] += v * p_mu_lam;
                        }
                    }
                    k_raw[mu_n + lam] += acc_mu_lam;
                    k_raw[nu_n + lam] += acc_nu_lam;
                }
                j_raw[mu_n + nu] += 2.0 * acc_mn;
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

                        j_raw[mu * n + nu] += 2.0 * v * p[lam * n + sig];
                        j_raw[lam * n + sig] += 2.0 * v * p[mu * n + nu];

                        k_raw[mu * n + lam] += v * p[nu * n + sig];
                        k_raw[nu * n + lam] += v * p[mu * n + sig];
                        k_raw[mu * n + sig] += v * p[nu * n + lam];
                        k_raw[nu * n + sig] += v * p[mu * n + lam];
                    }
                }
            }
        }
    }
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

/// Internal half-write UKS quartet contraction used by `build_uks_bounded`.
/// Caller must call `symmetrize_in_place` on `ga_raw` and `gb_raw` once after reduction.
#[allow(clippy::too_many_arguments)]
#[inline]
fn contract_quartet_uks_raw(
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
    ga_raw: &mut [f64],
    gb_raw: &mut [f64],
) {
    let na = sh_a.n_functions();
    let nb = sh_b.n_functions();
    let nc = sh_c.n_functions();
    let nd = sh_d.n_functions();

    let has_exchange = exx_fraction.abs() > 1e-15;

    if i != j && k != l && (i != k || j != l) {
        let ao_a = sh_a.ao_offset;
        let ao_b = sh_b.ao_offset;
        let ao_c = sh_c.ao_offset;
        let ao_d = sh_d.ao_offset;

        if !has_exchange {
            for ia in 0..na {
                let mu = ao_a + ia;
                let mu_n = mu * n;
                let ia_nb = ia * nb;
                for ib in 0..nb {
                    let nu = ao_b + ib;
                    let pt_mn2 = 2.0 * pt[mu_n + nu];
                    let ab_nc = (ia_nb + ib) * nc;
                    let mut acc_mn = 0.0;
                    for ic in 0..nc {
                        let lam_n = (ao_c + ic) * n;
                        let abc_nd = (ab_nc + ic) * nd;
                        for id in 0..nd {
                            let v = block[abc_nd + id];
                            if v != 0.0 {
                                let sig = ao_d + id;
                                acc_mn += v * pt[lam_n + sig];
                                let j_ls = v * pt_mn2;
                                ga_raw[lam_n + sig] += j_ls;
                                gb_raw[lam_n + sig] += j_ls;
                            }
                        }
                    }
                    let j_mn = 2.0 * acc_mn;
                    ga_raw[mu_n + nu] += j_mn;
                    gb_raw[mu_n + nu] += j_mn;
                }
            }
            return;
        }

        for ia in 0..na {
            let mu = ao_a + ia;
            let mu_n = mu * n;
            let ia_nb = ia * nb;
            for ib in 0..nb {
                let nu = ao_b + ib;
                let nu_n = nu * n;
                let pt_mn2 = 2.0 * pt[mu_n + nu];
                let ab_nc = (ia_nb + ib) * nc;
                let mut acc_mn = 0.0;
                for ic in 0..nc {
                    let lam = ao_c + ic;
                    let lam_n = lam * n;
                    let ca_nu_lam = exx_fraction * pa[nu_n + lam];
                    let ca_mu_lam = exx_fraction * pa[mu_n + lam];
                    let cb_nu_lam = exx_fraction * pb[nu_n + lam];
                    let cb_mu_lam = exx_fraction * pb[mu_n + lam];
                    let abc_nd = (ab_nc + ic) * nd;
                    let mut acc_a_mu_lam = 0.0;
                    let mut acc_a_nu_lam = 0.0;
                    let mut acc_b_mu_lam = 0.0;
                    let mut acc_b_nu_lam = 0.0;
                    for id in 0..nd {
                        let v = block[abc_nd + id];
                        if v != 0.0 {
                            let sig = ao_d + id;
                            acc_mn += v * pt[lam_n + sig];
                            let j_ls = v * pt_mn2;
                            ga_raw[lam_n + sig] += j_ls;
                            gb_raw[lam_n + sig] += j_ls;

                            acc_a_mu_lam += v * pa[nu_n + sig];
                            acc_a_nu_lam += v * pa[mu_n + sig];
                            ga_raw[mu_n + sig] -= v * ca_nu_lam;
                            ga_raw[nu_n + sig] -= v * ca_mu_lam;

                            acc_b_mu_lam += v * pb[nu_n + sig];
                            acc_b_nu_lam += v * pb[mu_n + sig];
                            gb_raw[mu_n + sig] -= v * cb_nu_lam;
                            gb_raw[nu_n + sig] -= v * cb_mu_lam;
                        }
                    }
                    ga_raw[mu_n + lam] -= exx_fraction * acc_a_mu_lam;
                    ga_raw[nu_n + lam] -= exx_fraction * acc_a_nu_lam;
                    gb_raw[mu_n + lam] -= exx_fraction * acc_b_mu_lam;
                    gb_raw[nu_n + lam] -= exx_fraction * acc_b_nu_lam;
                }
                let j_mn = 2.0 * acc_mn;
                ga_raw[mu_n + nu] += j_mn;
                gb_raw[mu_n + nu] += j_mn;
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

                        let p_ls = 2.0 * v * pt[lam * n + sig];
                        let p_mn = 2.0 * v * pt[mu * n + nu];

                        ga_raw[mu * n + nu] += p_ls;
                        ga_raw[lam * n + sig] += p_mn;
                        gb_raw[mu * n + nu] += p_ls;
                        gb_raw[lam * n + sig] += p_mn;

                        if has_exchange {
                            let vc = exx_fraction * v;
                            ga_raw[mu * n + lam] -= vc * pa[nu * n + sig];
                            ga_raw[nu * n + lam] -= vc * pa[mu * n + sig];
                            ga_raw[mu * n + sig] -= vc * pa[nu * n + lam];
                            ga_raw[nu * n + sig] -= vc * pa[mu * n + lam];

                            gb_raw[mu * n + lam] -= vc * pb[nu * n + sig];
                            gb_raw[nu * n + lam] -= vc * pb[mu * n + sig];
                            gb_raw[mu * n + sig] -= vc * pb[nu * n + lam];
                            gb_raw[nu * n + sig] -= vc * pb[mu * n + lam];
                        }
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
    pub use_gpu: bool,
    pub screening: SchwarzScreening,
    pub sig_pairs: Vec<ShellPair>,
    exchange_rows: Arc<Vec<OnceLock<Mutex<Vec<f64>>>>>,
    pair_prefix: Arc<Vec<usize>>,
    cached_kets: Arc<Vec<usize>>,
    gpu_engine: Arc<OnceLock<Option<calchem_cuda::GpuDirectEngine>>>,
    pub gpu_kernel_invocations: Arc<AtomicUsize>,
    pub gpu_fallback_invocations: Arc<AtomicUsize>,
}

fn resolve_gpu_fp32_thresh() -> f64 {
    std::env::var("CALCHEM_GPU_FP32_THRESH")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&x| x.is_finite() && x >= 0.0)
        .unwrap_or(1e-5)
}

impl DirectFockEngine {
    /// Initializes a CPU Direct Fock engine with precomputed Schwarz screening bounds and significant shell pairs.
    pub fn new(basis: &BasisSet, threshold: f64) -> Self {
        Self::new_with_gpu(basis, threshold, false)
    }

    /// Initializes a Direct Fock engine, optionally accelerating Direct-JK / Direct-K on CUDA GPU
    /// (skipping the 1.5 GiB host `exchange_rows` cache when CUDA is active).
    pub fn new_with_gpu(basis: &BasisSet, threshold: f64, use_gpu: bool) -> Self {
        let cuda_active = use_gpu && calchem_cuda::is_cuda_available();
        let bytes = if cuda_active {
            0
        } else {
            std::env::var("CALCHEM_MAX_K_CACHE_BYTES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| {
                    let default_cap = 1536usize << 20;
                    if let Some(max_mem) = std::env::var("CALCHEM_MAX_MEMORY_BYTES")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&m| m > 0)
                    {
                        default_cap.min(max_mem / 3)
                    } else {
                        default_cap
                    }
                })
        };
        Self::new_with_cache_budget_and_gpu(basis, threshold, bytes, use_gpu)
    }

    /// Initializes a Direct Fock engine with an explicit `exchange_rows` cache byte budget.
    pub fn new_with_cache_budget(basis: &BasisSet, threshold: f64, max_k_cache_bytes: usize) -> Self {
        Self::new_with_cache_budget_and_gpu(basis, threshold, max_k_cache_bytes, false)
    }

    fn new_with_cache_budget_and_gpu(
        basis: &BasisSet,
        threshold: f64,
        max_k_cache_bytes: usize,
        use_gpu: bool,
    ) -> Self {
        let screening = SchwarzScreening::new(basis, threshold);
        // A density-independent pair cutoff cannot bound a later contraction with
        // an arbitrary density. Retain pairs and screen complete quartets using D.
        let mut sig_pairs = find_significant_shell_pairs(basis, &screening, 0.0);
        sig_pairs.sort_by(|a,b| b.bound.total_cmp(&a.bound));
        let bytes = max_k_cache_bytes;
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
            use_gpu,
            screening,
            sig_pairs,
            exchange_rows, pair_prefix, cached_kets,
            gpu_engine: Arc::new(OnceLock::new()),
            gpu_kernel_invocations: Arc::new(AtomicUsize::new(0)),
            gpu_fallback_invocations: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Returns `(gpu_kernel_invocations, gpu_fallback_invocations)` recorded by this engine.
    pub fn gpu_telemetry(&self) -> (usize, usize) {
        (
            self.gpu_kernel_invocations.load(Ordering::Relaxed),
            self.gpu_fallback_invocations.load(Ordering::Relaxed),
        )
    }

    fn ensure_gpu_engine(&self, basis: &BasisSet) -> Option<&calchem_cuda::GpuDirectEngine> {
        if !self.use_gpu || self.sig_pairs.is_empty() || !calchem_cuda::is_cuda_available() {
            return None;
        }
        self.gpu_engine
            .get_or_init(|| {
                let mut gpu_shells = Vec::with_capacity(basis.shells.len());
                for sh in &basis.shells {
                    let mut comp_scales = [1.0f64; 10];
                    if let Some(p0) = sh.primitives.first() {
                        if p0.norm > 0.0 {
                            for (i, &c) in sh.components().iter().enumerate() {
                                comp_scales[i] = sh.primitive_norm_for(0, c) / p0.norm;
                            }
                        }
                    }
                    gpu_shells.push(calchem_cuda::DirectShellGpu {
                        center: sh.center,
                        comp_scales,
                        l: sh.shell_type.angular_momentum() as i32,
                        ao_offset: sh.ao_offset as i32,
                        n_funcs: sh.n_functions() as i32,
                        _pad: 0,
                    });
                }

                let mut gpu_prims = Vec::new();
                let mut pair_prim_ranges = Vec::with_capacity(self.sig_pairs.len());
                for sp in &self.sig_pairs {
                    let start = gpu_prims.len() as i32;
                    let pps = collect_egp_pairs(&basis.shells[sp.shell_i], &basis.shells[sp.shell_j]);
                    for pp in &pps {
                        gpu_prims.push(calchem_cuda::DirectPrimPairGpu {
                            gamma: pp.pair.gamma,
                            kab_w: pp.pair.kab * pp.w,
                            p: pp.pair.p,
                            pa: pp.pair.pa,
                        });
                    }
                    pair_prim_ranges.push((start, pps.len() as i32));
                }

                let pair_prefix_u64: Vec<u64> = self.pair_prefix.iter().map(|&v| v as u64).collect();
                let max_cap = std::env::var("CALCHEM_MAX_GPU_K_CACHE_BYTES")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(3usize << 30);
                let rec_bytes = calchem_cuda::GpuDirectEngine::recommended_cache_bytes(max_cap);
                let meta_est = gpu_shells.len() * std::mem::size_of::<calchem_cuda::DirectShellGpu>()
                    + self.sig_pairs.len() * std::mem::size_of::<calchem_cuda::DirectShellPairGpu>()
                    + gpu_prims.len() * std::mem::size_of::<calchem_cuda::DirectPrimPairGpu>()
                    + pair_prefix_u64.len() * 8
                    + 4 * self.n_ao * self.n_ao * 8
                    + 3 * self.n_shells * self.n_shells * 8
                    + (128usize << 20);
                let cache_bytes_cap = rec_bytes.saturating_sub(meta_est);

                let extent = |pi: usize, cutoff: f64| {
                    self.sig_pairs[..=pi]
                        .partition_point(|q| self.sig_pairs[pi].bound * q.bound > cutoff)
                };
                let fits = |cutoff: f64| {
                    let mut elements = 0u128;
                    for pi in 0..self.sig_pairs.len() {
                        let bra_w = self.pair_prefix[pi + 1] - self.pair_prefix[pi];
                        elements += bra_w as u128 * self.pair_prefix[extent(pi, cutoff)] as u128;
                        if elements * 8 > cache_bytes_cap as u128 {
                            return false;
                        }
                    }
                    true
                };
                let gpu_cached_kets: Vec<usize> = if cache_bytes_cap == 0 {
                    vec![0; self.sig_pairs.len()]
                } else {
                    let mut low = -36.0;
                    let mut high = self
                        .sig_pairs
                        .first()
                        .map_or(0.0, |p| (p.bound * p.bound).max(1.0).log10() + 1.0);
                    if !fits(10f64.powf(low)) {
                        for _ in 0..40 {
                            let mid = (low + high) * 0.5;
                            if fits(10f64.powf(mid)) {
                                high = mid;
                            } else {
                                low = mid;
                            }
                        }
                    } else {
                        high = low;
                    }
                    let cutoff = 10f64.powf(high);
                    (0..self.sig_pairs.len()).map(|pi| extent(pi, cutoff)).collect()
                };

                let mut cache_elements = 0usize;
                let mut valid_slots = 0usize;
                let mut gpu_pairs = Vec::with_capacity(self.sig_pairs.len());
                for (pi, sp) in self.sig_pairs.iter().enumerate() {
                    let ck = gpu_cached_kets[pi];
                    let bra_w = self.pair_prefix[pi + 1] - self.pair_prefix[pi];
                    let (p_start, p_len) = pair_prim_ranges[pi];
                    gpu_pairs.push(calchem_cuda::DirectShellPairGpu {
                        shell_i: sp.shell_i as i32,
                        shell_j: sp.shell_j as i32,
                        bound: sp.bound,
                        prim_pair_start: p_start,
                        n_prim_pairs: p_len,
                        cached_kets: ck as i32,
                        cache_valid_offset: valid_slots as i32,
                        cache_data_offset: cache_elements as u64,
                    });
                    valid_slots += ck;
                    cache_elements += bra_w * self.pair_prefix[ck];
                }

                if let Some(eng) = calchem_cuda::GpuDirectEngine::upload(
                    &gpu_shells,
                    &gpu_pairs,
                    &gpu_prims,
                    &pair_prefix_u64,
                    self.n_ao,
                    cache_elements,
                    valid_slots,
                ) {
                    return Some(eng);
                }

                // Retry once with zero quartet cache if VRAM was tight/fragmented
                if cache_elements > 0 {
                    for gp in &mut gpu_pairs {
                        gp.cached_kets = 0;
                        gp.cache_valid_offset = 0;
                        gp.cache_data_offset = 0;
                    }
                    return calchem_cuda::GpuDirectEngine::upload(
                        &gpu_shells,
                        &gpu_pairs,
                        &gpu_prims,
                        &pair_prefix_u64,
                        self.n_ao,
                        0,
                        0,
                    );
                }
                None
            })
            .as_ref()
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
        let num_pairs = self.sig_pairs.len();
        if num_pairs == 0 {
            return (vec![0.0f64; n * n], DirectFockStats::default());
        }

        let d = compute_shell_pair_l1_density(basis, p);
        let has_exchange = exx_fraction.abs() > 1e-15;
        let exx_scale = if has_exchange { exx_fraction.abs() } else { 0.0 };
        let total_candidates = num_pairs * (num_pairs + 1) / 2;

        let mut bound_prefix = Vec::with_capacity(num_pairs + 1);
        bound_prefix.push(0.0f64);
        let mut prefix_max_d = Vec::with_capacity(num_pairs);
        let mut cur_max_d = 0.0f64;
        for pair in &self.sig_pairs {
            bound_prefix.push(bound_prefix.last().unwrap() + pair.bound);
            let d_pair = d[pair.shell_i * n_shells + pair.shell_j];
            if d_pair > cur_max_d {
                cur_max_d = d_pair;
            }
            prefix_max_d.push(cur_max_d);
        }

        let max_d = d.iter().copied().fold(0.0f64, f64::max);
        let global_d_eff = (1.0 + exx_scale) * max_d;
        let largest = self.sig_pairs[0].bound;
        let active_pairs = self.sig_pairs.partition_point(|q| {
            threshold <= 0.0 || 4.0 * (largest * q.bound) * global_d_eff >= threshold
        });

        if active_pairs == 0 {
            let mut stats = DirectFockStats {
                significant_shell_pairs: num_pairs,
                total_quartet_candidates: total_candidates,
                computed_quartets: 0,
                skipped_quartets: total_candidates,
                cumulative_bound_error: 0.0,
                max_skipped_bound: 0.0,
            };
            if max_d > 0.0 {
                let mut outer_sum = 0.0f64;
                for (pi, pair) in self.sig_pairs.iter().enumerate() {
                    outer_sum += 4.0 * pair.bound * global_d_eff * bound_prefix[pi + 1];
                }
                stats.cumulative_bound_error = outer_sum;
                stats.max_skipped_bound = 4.0 * (largest * largest) * global_d_eff;
            }
            return (vec![0.0f64; n * n], stats);
        }

        let row_max: Vec<f64> = if has_exchange {
            d.chunks_exact(n_shells)
                .map(|row| row.iter().copied().fold(0.0f64, f64::max))
                .collect()
        } else {
            Vec::new()
        };

        if self.use_gpu {
            if let Some(gpu) = self.ensure_gpu_engine(basis) {
                let ends: Vec<i32> = (0..active_pairs)
                    .map(|pi| {
                        let bra = &self.sig_pairs[pi];
                        let (i, j) = (bra.shell_i, bra.shell_j);
                        let b_ij = bra.bound;
                        let d_ij = d[i * n_shells + j];
                        let bra_d_eff = d_ij.max(prefix_max_d[pi])
                            + if has_exchange {
                                exx_scale * row_max[i].max(row_max[j])
                            } else {
                                0.0
                            };
                        self.sig_pairs[..=pi].partition_point(|q| {
                            threshold <= 0.0 || 4.0 * (b_ij * q.bound) * bra_d_eff >= threshold
                        }) as i32
                    })
                    .collect();
                let mut g_raw = vec![0.0f64; n * n];
                if let Some(gpu_stats) = gpu.contract_rks(
                    p,
                    &d,
                    active_pairs,
                    &ends,
                    exx_fraction,
                    threshold,
                    resolve_gpu_fp32_thresh(),
                    &mut g_raw,
                ) {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                    symmetrize_in_place(&mut g_raw, n);
                    let mut stats = DirectFockStats {
                        significant_shell_pairs: num_pairs,
                        total_quartet_candidates: total_candidates,
                        computed_quartets: gpu_stats.computed_quartets as usize,
                        skipped_quartets: gpu_stats.skipped_quartets as usize,
                        cumulative_bound_error: 0.0,
                        max_skipped_bound: 0.0,
                    };
                    for pi in 0..active_pairs {
                        let end = ends[pi] as usize;
                        if end <= pi {
                            stats.skipped_quartets += (pi + 1) - end;
                        }
                    }
                    if active_pairs < num_pairs {
                        stats.skipped_quartets += total_candidates - active_pairs * (active_pairs + 1) / 2;
                    }
                    return (g_raw, stats);
                }
            }
            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
        }

        let primitive_pairs: Vec<_> = self.sig_pairs[..active_pairs]
            .iter()
            .map(|sp| collect_egp_pairs(&basis.shells[sp.shell_i], &basis.shells[sp.shell_j]))
            .collect();

        let k_workers = self
            .determine_worker_count(false, Some(num_workers))
            .min(active_pairs);

        let mut worker_buffers: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_stats: Vec<DirectFockStats> = vec![DirectFockStats::default(); k_workers];

        let run_worker = |worker: usize, local_g: &mut Vec<f64>, local_stats: &mut DirectFockStats| {
            let mut block = Vec::with_capacity(1296);
            for pi in (worker..active_pairs).step_by(k_workers) {
                let bra = &self.sig_pairs[pi];
                let (i, j) = (bra.shell_i, bra.shell_j);
                let b_ij = bra.bound;
                let d_ij = d[i * n_shells + j];
                let sh_a = &basis.shells[i];
                let sh_b = &basis.shells[j];

                let bra_d_eff = d_ij.max(prefix_max_d[pi])
                    + if has_exchange {
                        exx_scale * row_max[i].max(row_max[j])
                    } else {
                        0.0
                    };
                let candidates = &self.sig_pairs[..=pi];
                let end = candidates.partition_point(|q| {
                    threshold <= 0.0 || 4.0 * (b_ij * q.bound) * bra_d_eff >= threshold
                });

                if end <= pi {
                    let n_tail = (pi + 1) - end;
                    let tail_bound_sum =
                        4.0 * b_ij * bra_d_eff * (bound_prefix[pi + 1] - bound_prefix[end]).max(0.0);
                    let first_tail_bound = 4.0 * (b_ij * candidates[end].bound) * bra_d_eff;
                    local_stats.skipped_quartets += n_tail;
                    local_stats.cumulative_bound_error += tail_bound_sum;
                    if first_tail_bound > local_stats.max_skipped_bound {
                        local_stats.max_skipped_bound = first_tail_bound;
                    }
                }

                let bra_width = sh_a.n_functions() * sh_b.n_functions();
                let count = self.cached_kets[pi];
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

                for (qi, pair_q) in candidates[..end].iter().enumerate() {
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
                        let bound_k = 4.0 * exx_scale * b_ijkl * d_exch;
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
                    let vals = if qi < count {
                        let start = self.pair_prefix[qi] * bra_width;
                        let stop = self.pair_prefix[qi + 1] * bra_width;
                        let slot = &mut cache.as_mut().unwrap()[start..stop];
                        if slot[0].is_nan() {
                            eri_from_pairs(
                                sh_a,
                                sh_b,
                                sh_c,
                                sh_d,
                                &primitive_pairs[pi],
                                &primitive_pairs[qi],
                                None,
                                &mut block,
                            );
                            slot.copy_from_slice(&block);
                        }
                        &slot[..]
                    } else {
                        eri_from_pairs(
                            sh_a,
                            sh_b,
                            sh_c,
                            sh_d,
                            &primitive_pairs[pi],
                            &primitive_pairs[qi],
                            None,
                            &mut block,
                        );
                        &block[..]
                    };

                    contract_quartet_rks_raw(
                        i,
                        j,
                        k,
                        l,
                        sh_a,
                        sh_b,
                        sh_c,
                        sh_d,
                        vals,
                        p,
                        n,
                        exx_fraction,
                        local_g,
                    );
                }
            }
        };

        if k_workers == 1 {
            run_worker(0, &mut worker_buffers[0], &mut worker_stats[0]);
        } else {
            worker_buffers
                .par_iter_mut()
                .zip(worker_stats.par_iter_mut())
                .enumerate()
                .for_each(|(worker, (local_g, local_stats))| {
                    run_worker(worker, local_g, local_stats);
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

        symmetrize_in_place(&mut final_g, n);

        if active_pairs < num_pairs {
            let n_outer = total_candidates - active_pairs * (active_pairs + 1) / 2;
            final_stats.skipped_quartets += n_outer;
            if max_d > 0.0 {
                let mut outer_sum = 0.0f64;
                for pi in active_pairs..num_pairs {
                    outer_sum +=
                        4.0 * self.sig_pairs[pi].bound * global_d_eff * bound_prefix[pi + 1];
                }
                final_stats.cumulative_bound_error += outer_sum;
                let max_outer = 4.0 * (largest * self.sig_pairs[active_pairs].bound) * global_d_eff;
                if max_outer > final_stats.max_skipped_bound {
                    final_stats.max_skipped_bound = max_outer;
                }
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

        let max_density = d.iter().copied().fold(0.0_f64, f64::max);
        let largest = self.sig_pairs[0].bound;
        let num_pairs = self.sig_pairs.partition_point(|q| threshold <= 0.0 ||
            4.0*(largest*q.bound)*max_density >= threshold);
        if num_pairs == 0 {
            return vec![0.0f64; n * n];
        }
        let row_max: Vec<_> = d.chunks_exact(n_shells).map(|row|
            row.iter().copied().fold(0.0_f64,f64::max)).collect();

        if self.use_gpu {
            if let Some(gpu) = self.ensure_gpu_engine(basis) {
                let ends: Vec<i32> = (0..num_pairs)
                    .map(|pi| {
                        let bra = &self.sig_pairs[pi];
                        let (i, j) = (bra.shell_i, bra.shell_j);
                        let rm = row_max[i].max(row_max[j]);
                        self.sig_pairs[..=pi].partition_point(|q| {
                            threshold <= 0.0 || 4.0 * (bra.bound * q.bound) * rm >= threshold
                        }) as i32
                    })
                    .collect();
                let mut k_raw = vec![0.0f64; n * n];
                if gpu
                    .contract_k_only(
                        p,
                        &d,
                        num_pairs,
                        &ends,
                        omega,
                        threshold,
                        resolve_gpu_fp32_thresh(),
                        &mut k_raw,
                    )
                    .is_some()
                {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                    symmetrize_in_place(&mut k_raw, n);
                    return k_raw;
                }
            }
            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
        }

        let k_workers = self.determine_worker_count(false, Some(num_workers)).min(num_pairs);
        let mut worker_buffers: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let primitive_pairs: Vec<_> = self.sig_pairs[..num_pairs].iter().map(|p|
            collect_egp_pairs(&basis.shells[p.shell_i], &basis.shells[p.shell_j])).collect();
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

    /// Evaluates both spin exact-exchange matrices `(K[Pa], K[Pb])` in a single sweep
    /// (GPU `contract_k_pair` when `--gpu` is enabled, else CPU fallback).
    pub fn build_k_pair_with_omega_bounded(
        &self,
        basis: &BasisSet,
        pa: &[f64],
        pb: &[f64],
        omega: Option<f64>,
        threshold: f64,
        num_workers: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_ao;
        let n_shells = self.n_shells;
        if self.sig_pairs.is_empty() {
            return (vec![0.0f64; n * n], vec![0.0f64; n * n]);
        }
        let da = compute_shell_pair_l1_density(basis, pa);
        let db = compute_shell_pair_l1_density(basis, pb);
        let max_da = da.iter().copied().fold(0.0_f64, f64::max);
        let max_db = db.iter().copied().fold(0.0_f64, f64::max);
        let max_density = max_da.max(max_db);
        let largest = self.sig_pairs[0].bound;
        let active_pairs = self.sig_pairs.partition_point(|q| {
            threshold <= 0.0 || 4.0 * (largest * q.bound) * max_density >= threshold
        });
        if active_pairs == 0 {
            return (vec![0.0f64; n * n], vec![0.0f64; n * n]);
        }

        if self.use_gpu {
            if let Some(gpu) = self.ensure_gpu_engine(basis) {
                let row_max_ab: Vec<f64> = da
                    .chunks_exact(n_shells)
                    .zip(db.chunks_exact(n_shells))
                    .map(|(ra, rb)| {
                        ra.iter()
                            .copied()
                            .fold(0.0f64, f64::max)
                            .max(rb.iter().copied().fold(0.0f64, f64::max))
                    })
                    .collect();
                let ends: Vec<i32> = (0..active_pairs)
                    .map(|pi| {
                        let bra = &self.sig_pairs[pi];
                        let rm = row_max_ab[bra.shell_i].max(row_max_ab[bra.shell_j]);
                        self.sig_pairs[..=pi].partition_point(|q| {
                            threshold <= 0.0 || 4.0 * (bra.bound * q.bound) * rm >= threshold
                        }) as i32
                    })
                    .collect();
                let mut ka_raw = vec![0.0f64; n * n];
                let mut kb_raw = vec![0.0f64; n * n];
                if gpu
                    .contract_k_pair(
                        pa,
                        pb,
                        &da,
                        &db,
                        active_pairs,
                        &ends,
                        omega,
                        threshold,
                        resolve_gpu_fp32_thresh(),
                        &mut ka_raw,
                        &mut kb_raw,
                    )
                    .is_some()
                {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                    symmetrize_in_place(&mut ka_raw, n);
                    symmetrize_in_place(&mut kb_raw, n);
                    return (ka_raw, kb_raw);
                }
            }
            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
        }

        // CPU fallback (preserve exact semantics without double-counting fallback)
        let ka = self.build_k_with_omega_bounded_cpu(basis, pa, omega, threshold, num_workers);
        let kb = self.build_k_with_omega_bounded_cpu(basis, pb, omega, threshold, num_workers);
        (ka, kb)
    }

    fn build_k_with_omega_bounded_cpu(
        &self,
        basis: &BasisSet,
        p: &[f64],
        omega: Option<f64>,
        threshold: f64,
        num_workers: usize,
    ) -> Vec<f64> {
        let mut cpu_view = self.clone();
        cpu_view.use_gpu = false;
        cpu_view.build_k_with_omega_bounded(basis, p, omega, threshold, num_workers)
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
        if self.use_gpu && n > 0 {
            if calchem_cuda::is_cuda_available() {
                let mut x_sym = vec![0.0f64; n * n];
                let mut x_anti = vec![0.0f64; n * n];
                for r in 0..n {
                    x_sym[r * n + r] = x[r * n + r];
                    for c in (r + 1)..n {
                        let a = x[r * n + c];
                        let b = x[c * n + r];
                        let s = 0.5 * (a + b);
                        let d = 0.5 * (a - b);
                        x_sym[r * n + c] = s;
                        x_sym[c * n + r] = s;
                        x_anti[r * n + c] = d;
                        x_anti[c * n + r] = -d;
                    }
                }
                let res = self.build_jk_batch_sym_anti_bounded(
                    basis,
                    &[&x_sym],
                    &[&x_anti],
                    DirectJkBatchOptions {
                        compute_j: false,
                        compute_k_sym: true,
                        compute_k_anti: true,
                        j_screen_weight: 0.0,
                        k_screen_weight: 1.0,
                        omega,
                        threshold,
                    },
                    self.determine_worker_count(false, None),
                );
                let mut out = res
                    .k_sym
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| vec![0.0f64; n * n]);
                if let Some(ka) = res.k_anti.into_iter().next() {
                    for idx in 0..(n * n) {
                        out[idx] += ka[idx];
                    }
                }
                return out;
            } else {
                self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
            }
        }
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

        if exx_fraction.abs() <= 1e-15 {
            let (j_tot, stats) = self.build_rks_bounded(basis, &pt, 0.0, threshold, num_workers);
            return (j_tot.clone(), j_tot, stats);
        }

        let num_pairs = self.sig_pairs.len();
        if num_pairs == 0 {
            return (
                vec![0.0f64; n * n],
                vec![0.0f64; n * n],
                DirectFockStats::default(),
            );
        }

        let dt = compute_shell_pair_l1_density(basis, &pt);
        let da = compute_shell_pair_l1_density(basis, pa);
        let db = compute_shell_pair_l1_density(basis, pb);
        let exx_scale = exx_fraction.abs();
        let total_candidates = num_pairs * (num_pairs + 1) / 2;

        let mut bound_prefix = Vec::with_capacity(num_pairs + 1);
        bound_prefix.push(0.0f64);
        let mut prefix_max_dt = Vec::with_capacity(num_pairs);
        let mut cur_max_dt = 0.0f64;
        for pair in &self.sig_pairs {
            bound_prefix.push(bound_prefix.last().unwrap() + pair.bound);
            let dt_pair = dt[pair.shell_i * n_shells + pair.shell_j];
            if dt_pair > cur_max_dt {
                cur_max_dt = dt_pair;
            }
            prefix_max_dt.push(cur_max_dt);
        }

        let max_dt = dt.iter().copied().fold(0.0f64, f64::max);
        let max_da = da.iter().copied().fold(0.0f64, f64::max);
        let max_db = db.iter().copied().fold(0.0f64, f64::max);
        let global_d_eff = max_dt + exx_scale * max_da.max(max_db);
        let largest = self.sig_pairs[0].bound;
        let active_pairs = self.sig_pairs.partition_point(|q| {
            threshold <= 0.0 || 4.0 * (largest * q.bound) * global_d_eff >= threshold
        });

        if active_pairs == 0 {
            let mut stats = DirectFockStats {
                significant_shell_pairs: num_pairs,
                total_quartet_candidates: total_candidates,
                computed_quartets: 0,
                skipped_quartets: total_candidates,
                cumulative_bound_error: 0.0,
                max_skipped_bound: 0.0,
            };
            if global_d_eff > 0.0 {
                let mut outer_sum = 0.0f64;
                for (pi, pair) in self.sig_pairs.iter().enumerate() {
                    outer_sum += 4.0 * pair.bound * global_d_eff * bound_prefix[pi + 1];
                }
                stats.cumulative_bound_error = outer_sum;
                stats.max_skipped_bound = 4.0 * (largest * largest) * global_d_eff;
            }
            return (vec![0.0f64; n * n], vec![0.0f64; n * n], stats);
        }

        let row_max_ab: Vec<f64> = da
            .chunks_exact(n_shells)
            .zip(db.chunks_exact(n_shells))
            .map(|(ra, rb)| {
                let ma = ra.iter().copied().fold(0.0f64, f64::max);
                let mb = rb.iter().copied().fold(0.0f64, f64::max);
                ma.max(mb)
            })
            .collect();

        if self.use_gpu {
            if let Some(gpu) = self.ensure_gpu_engine(basis) {
                let ends: Vec<i32> = (0..active_pairs)
                    .map(|pi| {
                        let bra = &self.sig_pairs[pi];
                        let (i, j) = (bra.shell_i, bra.shell_j);
                        let dt_ij = dt[i * n_shells + j];
                        let bra_d_eff = dt_ij.max(prefix_max_dt[pi])
                            + exx_scale * row_max_ab[i].max(row_max_ab[j]);
                        self.sig_pairs[..=pi].partition_point(|q| {
                            threshold <= 0.0 || 4.0 * (bra.bound * q.bound) * bra_d_eff >= threshold
                        }) as i32
                    })
                    .collect();
                let mut ga_raw = vec![0.0f64; n * n];
                let mut gb_raw = vec![0.0f64; n * n];
                if let Some(gpu_stats) = gpu.contract_uks(
                    pa,
                    pb,
                    &dt,
                    &da,
                    &db,
                    active_pairs,
                    &ends,
                    exx_fraction,
                    threshold,
                    resolve_gpu_fp32_thresh(),
                    &mut ga_raw,
                    &mut gb_raw,
                ) {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                    symmetrize_in_place(&mut ga_raw, n);
                    symmetrize_in_place(&mut gb_raw, n);
                    let mut stats = DirectFockStats {
                        significant_shell_pairs: num_pairs,
                        total_quartet_candidates: total_candidates,
                        computed_quartets: gpu_stats.computed_quartets as usize,
                        skipped_quartets: gpu_stats.skipped_quartets as usize,
                        cumulative_bound_error: 0.0,
                        max_skipped_bound: 0.0,
                    };
                    for pi in 0..active_pairs {
                        let end = ends[pi] as usize;
                        if end <= pi {
                            stats.skipped_quartets += (pi + 1) - end;
                        }
                    }
                    if active_pairs < num_pairs {
                        stats.skipped_quartets += total_candidates - active_pairs * (active_pairs + 1) / 2;
                    }
                    return (ga_raw, gb_raw, stats);
                }
            }
            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
        }

        let primitive_pairs: Vec<_> = self.sig_pairs[..active_pairs]
            .iter()
            .map(|sp| collect_egp_pairs(&basis.shells[sp.shell_i], &basis.shells[sp.shell_j]))
            .collect();

        let k_workers = self
            .determine_worker_count(true, Some(num_workers))
            .min(active_pairs);

        let mut worker_buffers_a: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_buffers_b: Vec<Vec<f64>> =
            (0..k_workers).map(|_| vec![0.0f64; n * n]).collect();
        let mut worker_stats: Vec<DirectFockStats> = vec![DirectFockStats::default(); k_workers];

        let run_worker = |worker: usize,
                          local_ga: &mut Vec<f64>,
                          local_gb: &mut Vec<f64>,
                          local_stats: &mut DirectFockStats| {
            let mut block = Vec::with_capacity(1296);
            for pi in (worker..active_pairs).step_by(k_workers) {
                let bra = &self.sig_pairs[pi];
                let (i, j) = (bra.shell_i, bra.shell_j);
                let b_ij = bra.bound;
                let dt_ij = dt[i * n_shells + j];
                let sh_a = &basis.shells[i];
                let sh_b = &basis.shells[j];

                let bra_d_eff =
                    dt_ij.max(prefix_max_dt[pi]) + exx_scale * row_max_ab[i].max(row_max_ab[j]);
                let candidates = &self.sig_pairs[..=pi];
                let end = candidates.partition_point(|q| {
                    threshold <= 0.0 || 4.0 * (b_ij * q.bound) * bra_d_eff >= threshold
                });

                if end <= pi {
                    let n_tail = (pi + 1) - end;
                    let tail_bound_sum =
                        4.0 * b_ij * bra_d_eff * (bound_prefix[pi + 1] - bound_prefix[end]).max(0.0);
                    let first_tail_bound = 4.0 * (b_ij * candidates[end].bound) * bra_d_eff;
                    local_stats.skipped_quartets += n_tail;
                    local_stats.cumulative_bound_error += tail_bound_sum;
                    if first_tail_bound > local_stats.max_skipped_bound {
                        local_stats.max_skipped_bound = first_tail_bound;
                    }
                }

                let bra_width = sh_a.n_functions() * sh_b.n_functions();
                let count = self.cached_kets[pi];
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

                for (qi, pair_q) in candidates[..end].iter().enumerate() {
                    let k = pair_q.shell_i;
                    let l = pair_q.shell_j;
                    let b_kl = pair_q.bound;
                    let b_ijkl = b_ij * b_kl;

                    let dt_kl = dt[k * n_shells + l];
                    let bound_j = 4.0 * b_ijkl * dt_ij.max(dt_kl);
                    let da_exch = da[i * n_shells + k]
                        .max(da[i * n_shells + l])
                        .max(da[j * n_shells + k])
                        .max(da[j * n_shells + l]);
                    let db_exch = db[i * n_shells + k]
                        .max(db[i * n_shells + l])
                        .max(db[j * n_shells + k])
                        .max(db[j * n_shells + l]);
                    let bound_k = 4.0 * exx_scale * b_ijkl * da_exch.max(db_exch);
                    let bound = bound_j + bound_k;

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
                    let vals = if qi < count {
                        let start = self.pair_prefix[qi] * bra_width;
                        let stop = self.pair_prefix[qi + 1] * bra_width;
                        let slot = &mut cache.as_mut().unwrap()[start..stop];
                        if slot[0].is_nan() {
                            eri_from_pairs(
                                sh_a,
                                sh_b,
                                sh_c,
                                sh_d,
                                &primitive_pairs[pi],
                                &primitive_pairs[qi],
                                None,
                                &mut block,
                            );
                            slot.copy_from_slice(&block);
                        }
                        &slot[..]
                    } else {
                        eri_from_pairs(
                            sh_a,
                            sh_b,
                            sh_c,
                            sh_d,
                            &primitive_pairs[pi],
                            &primitive_pairs[qi],
                            None,
                            &mut block,
                        );
                        &block[..]
                    };

                    contract_quartet_uks_raw(
                        i,
                        j,
                        k,
                        l,
                        sh_a,
                        sh_b,
                        sh_c,
                        sh_d,
                        vals,
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
        };

        if k_workers == 1 {
            run_worker(
                0,
                &mut worker_buffers_a[0],
                &mut worker_buffers_b[0],
                &mut worker_stats[0],
            );
        } else {
            worker_buffers_a
                .par_iter_mut()
                .zip(worker_buffers_b.par_iter_mut())
                .zip(worker_stats.par_iter_mut())
                .enumerate()
                .for_each(|(worker, ((local_ga, local_gb), local_stats))| {
                    run_worker(worker, local_ga, local_gb, local_stats);
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

        symmetrize_in_place(&mut final_ga, n);
        symmetrize_in_place(&mut final_gb, n);

        if active_pairs < num_pairs {
            let n_outer = total_candidates - active_pairs * (active_pairs + 1) / 2;
            final_stats.skipped_quartets += n_outer;
            if global_d_eff > 0.0 {
                let mut outer_sum = 0.0f64;
                for pi in active_pairs..num_pairs {
                    outer_sum +=
                        4.0 * self.sig_pairs[pi].bound * global_d_eff * bound_prefix[pi + 1];
                }
                final_stats.cumulative_bound_error += outer_sum;
                let max_outer = 4.0 * (largest * self.sig_pairs[active_pairs].bound) * global_d_eff;
                if max_outer > final_stats.max_skipped_bound {
                    final_stats.max_skipped_bound = max_outer;
                }
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

    fn determine_batch_workers_and_chunk(
        &self,
        total_channels: usize,
        n_batches: usize,
        max_workers: Option<usize>,
    ) -> (usize, usize) {
        let n = self.n_ao;
        if n == 0 || total_channels == 0 || n_batches == 0 {
            return (1, n_batches.max(1));
        }
        let bytes_per_vec_per_worker = total_channels
            .saturating_mul(n)
            .saturating_mul(n)
            .saturating_mul(8);
        if bytes_per_vec_per_worker == 0 {
            return (1, n_batches.max(1));
        }

        let batch_budget: usize = std::env::var("CALCHEM_MAX_DIRECT_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&m| m > 0)
            .unwrap_or_else(|| {
                let default_cap = 512usize << 20;
                if let Some(max_mem) = std::env::var("CALCHEM_MAX_MEMORY_BYTES")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&m| m > 0)
                {
                    default_cap.min((max_mem / 4).max(1))
                } else {
                    default_cap
                }
            });

        let total_threads = rayon::current_num_threads().max(1);
        let requested_threads = max_workers.unwrap_or(total_threads).clamp(1, total_threads);
        let k_workers = requested_threads.min((batch_budget / bytes_per_vec_per_worker).max(1));
        let bytes_per_vec = k_workers.saturating_mul(bytes_per_vec_per_worker).max(1);
        let max_b_chunk = (batch_budget / bytes_per_vec).max(1).min(n_batches);
        (k_workers, max_b_chunk)
    }

    /// Contracts `J[D_sym_b]`, symmetric `K[D_sym_b]`, and antisymmetric `K[D_anti_b]`
    /// across a batch of symmetric and antisymmetric trial densities in a single 4c
    /// shell-quartet sweep per memory-bounded chunk.
    pub fn build_jk_batch_sym_anti_bounded(
        &self,
        basis: &BasisSet,
        d_sym: &[&[f64]],
        d_anti: &[&[f64]],
        opts: DirectJkBatchOptions,
        num_workers: usize,
    ) -> DirectJkBatchOutput {
        self.build_jk_batch_sym_anti_bounded_impl(basis, d_sym, d_anti, opts, num_workers, None)
    }

    fn build_jk_batch_sym_anti_bounded_impl(
        &self,
        basis: &BasisSet,
        d_sym: &[&[f64]],
        d_anti: &[&[f64]],
        opts: DirectJkBatchOptions,
        num_workers: usize,
        chunk_cap_override: Option<usize>,
    ) -> DirectJkBatchOutput {
        let n = self.n_ao;
        let nn = n * n;
        for m in d_sym {
            assert_eq!(m.len(), nn, "d_sym matrix length mismatch");
        }
        for m in d_anti {
            assert_eq!(m.len(), nn, "d_anti matrix length mismatch");
        }
        #[cfg(debug_assertions)]
        {
            for m in d_sym {
                for r in 0..n {
                    for c in (r + 1)..n {
                        debug_assert!(
                            (m[r * n + c] - m[c * n + r]).abs() <= 1e-10,
                            "d_sym input must be symmetric"
                        );
                    }
                }
            }
            for m in d_anti {
                for r in 0..n {
                    debug_assert!(m[r * n + r].abs() <= 1e-10, "d_anti diagonal must be zero");
                    for c in (r + 1)..n {
                        debug_assert!(
                            (m[r * n + c] + m[c * n + r]).abs() <= 1e-10,
                            "d_anti input must be antisymmetric"
                        );
                    }
                }
            }
        }

        let do_j = opts.compute_j && !d_sym.is_empty();
        let do_k_sym = opts.compute_k_sym && !d_sym.is_empty();
        let do_k_anti = opts.compute_k_anti && !d_anti.is_empty();

        let b_sym = if do_j || do_k_sym { d_sym.len() } else { 0 };
        let b_anti = if do_k_anti { d_anti.len() } else { 0 };
        let b_total = b_sym.max(b_anti);

        let num_pairs = self.sig_pairs.len();
        if b_total == 0 || num_pairs == 0 || n == 0 {
            return DirectJkBatchOutput {
                j_sym: if opts.compute_j {
                    vec![vec![0.0f64; nn]; d_sym.len()]
                } else {
                    Vec::new()
                },
                k_sym: if opts.compute_k_sym {
                    vec![vec![0.0f64; nn]; d_sym.len()]
                } else {
                    Vec::new()
                },
                k_anti: if opts.compute_k_anti {
                    vec![vec![0.0f64; nn]; d_anti.len()]
                } else {
                    Vec::new()
                },
                stats: DirectFockStats::default(),
            };
        }

        let total_channels = (do_j as usize) + (do_k_sym as usize) + (do_k_anti as usize);
        let (k_workers_base, mut max_b_chunk) =
            self.determine_batch_workers_and_chunk(total_channels, b_total, Some(num_workers));
        let gpu_engine = if self.use_gpu {
            self.ensure_gpu_engine(basis)
        } else {
            None
        };
        if let Some(gpu) = gpu_engine {
            let mats_per_vec = ((do_j || do_k_sym) as usize)
                + (do_j as usize)
                + (do_k_sym as usize)
                + 2 * (do_k_anti as usize);
            let bytes_per_vec = mats_per_vec.saturating_mul(nn).saturating_mul(8).max(1);
            let td_reusable = gpu.td_scratch_bytes();
            let alloc_other = calchem_cuda::total_allocated_gpu_bytes().saturating_sub(td_reusable);
            let lim_rem = calchem_cuda::resolve_gpu_memory_limit().saturating_sub(alloc_other);
            let free_rem = calchem_cuda::get_cuda_memory_info()
                .map(|(free, _)| free.saturating_add(td_reusable).saturating_sub(128usize << 20))
                .unwrap_or(lim_rem);
            let gpu_budget = lim_rem.min(free_rem).min(256usize << 20).max(bytes_per_vec);
            max_b_chunk = (gpu_budget / bytes_per_vec).max(1).min(b_total);
        }
        if let Some(cap) = chunk_cap_override {
            max_b_chunk = max_b_chunk.min(cap.max(1));
        }

        let n_shells = self.n_shells;
        let ss = n_shells * n_shells;
        let total_candidates = num_pairs * (num_pairs + 1) / 2;
        let largest = self.sig_pairs[0].bound;

        let d_sym_l1_all: Vec<Vec<f64>> = d_sym[..b_sym]
            .iter()
            .map(|m| compute_shell_pair_l1_abs_sym(basis, m))
            .collect();
        let d_anti_l1_all: Vec<Vec<f64>> = d_anti[..b_anti]
            .iter()
            .map(|m| compute_shell_pair_l1_abs_sym(basis, m))
            .collect();

        let mut bound_prefix = Vec::with_capacity(num_pairs + 1);
        bound_prefix.push(0.0f64);
        for pair in &self.sig_pairs {
            bound_prefix.push(bound_prefix.last().unwrap() + pair.bound);
        }

        let mut out = DirectJkBatchOutput {
            j_sym: if opts.compute_j {
                Vec::with_capacity(d_sym.len())
            } else {
                Vec::new()
            },
            k_sym: if opts.compute_k_sym {
                Vec::with_capacity(d_sym.len())
            } else {
                Vec::new()
            },
            k_anti: if opts.compute_k_anti {
                Vec::with_capacity(d_anti.len())
            } else {
                Vec::new()
            },
            stats: DirectFockStats {
                significant_shell_pairs: num_pairs,
                ..DirectFockStats::default()
            },
        };

        let mut n_chunks = 0usize;
        let mut chunk_start = 0usize;
        while chunk_start < b_total {
            let chunk_end = (chunk_start + max_b_chunk).min(b_total);
            let sym_start = chunk_start.min(b_sym);
            let sym_end = chunk_end.min(b_sym);
            let b_sym_c = sym_end - sym_start;
            let anti_start = chunk_start.min(b_anti);
            let anti_end = chunk_end.min(b_anti);
            let b_anti_c = anti_end - anti_start;

            let chunk_do_j = do_j && b_sym_c > 0;
            let chunk_do_k_sym = do_k_sym && b_sym_c > 0;
            let chunk_do_k_anti = do_k_anti && b_anti_c > 0;

            let d_j_env: Vec<f64> = if chunk_do_j {
                let mut env = vec![0.0f64; ss];
                for d_b in &d_sym_l1_all[sym_start..sym_end] {
                    for idx in 0..ss {
                        if d_b[idx] > env[idx] {
                            env[idx] = d_b[idx];
                        }
                    }
                }
                env
            } else {
                Vec::new()
            };

            let d_k_env: Vec<f64> = if chunk_do_k_sym || chunk_do_k_anti {
                let mut env = vec![0.0f64; ss];
                if chunk_do_k_sym {
                    for d_b in &d_sym_l1_all[sym_start..sym_end] {
                        for idx in 0..ss {
                            if d_b[idx] > env[idx] {
                                env[idx] = d_b[idx];
                            }
                        }
                    }
                }
                if chunk_do_k_anti {
                    for d_b in &d_anti_l1_all[anti_start..anti_end] {
                        for idx in 0..ss {
                            if d_b[idx] > env[idx] {
                                env[idx] = d_b[idx];
                            }
                        }
                    }
                }
                env
            } else {
                Vec::new()
            };

            let j_w = if chunk_do_j {
                opts.j_screen_weight.abs()
            } else {
                0.0
            };
            let k_w = if chunk_do_k_sym || chunk_do_k_anti {
                opts.k_screen_weight.abs()
            } else {
                0.0
            };

            let max_dj = if chunk_do_j {
                d_j_env.iter().copied().fold(0.0f64, f64::max)
            } else {
                0.0
            };
            let max_dk = if chunk_do_k_sym || chunk_do_k_anti {
                d_k_env.iter().copied().fold(0.0f64, f64::max)
            } else {
                0.0
            };
            let global_d_eff = j_w * max_dj + k_w * max_dk;

            if opts.threshold > 0.0 && global_d_eff == 0.0 {
                n_chunks += 1;
                if chunk_do_j {
                    out.j_sym.extend((0..b_sym_c).map(|_| vec![0.0f64; nn]));
                }
                if chunk_do_k_sym {
                    out.k_sym.extend((0..b_sym_c).map(|_| vec![0.0f64; nn]));
                }
                if chunk_do_k_anti {
                    out.k_anti.extend((0..b_anti_c).map(|_| vec![0.0f64; nn]));
                }
                out.stats.skipped_quartets += total_candidates;
                chunk_start = chunk_end;
                continue;
            }

            let prefix_max_dj: Vec<f64> = if chunk_do_j {
                let mut pmax = Vec::with_capacity(num_pairs);
                let mut cur = 0.0f64;
                for pair in &self.sig_pairs {
                    let dv = d_j_env[pair.shell_i * n_shells + pair.shell_j];
                    if dv > cur {
                        cur = dv;
                    }
                    pmax.push(cur);
                }
                pmax
            } else {
                Vec::new()
            };

            let active_pairs = self.sig_pairs.partition_point(|q| {
                opts.threshold <= 0.0 || 4.0 * (largest * q.bound) * global_d_eff >= opts.threshold
            });

            if active_pairs == 0 {
                n_chunks += 1;
                if chunk_do_j {
                    out.j_sym.extend((0..b_sym_c).map(|_| vec![0.0f64; nn]));
                }
                if chunk_do_k_sym {
                    out.k_sym.extend((0..b_sym_c).map(|_| vec![0.0f64; nn]));
                }
                if chunk_do_k_anti {
                    out.k_anti.extend((0..b_anti_c).map(|_| vec![0.0f64; nn]));
                }
                out.stats.skipped_quartets += total_candidates;
                let mut outer_sum = 0.0f64;
                for (pi, pair) in self.sig_pairs.iter().enumerate() {
                    outer_sum += 4.0 * pair.bound * global_d_eff * bound_prefix[pi + 1];
                }
                out.stats.cumulative_bound_error += outer_sum;
                let max_outer = 4.0 * (largest * largest) * global_d_eff;
                if max_outer > out.stats.max_skipped_bound {
                    out.stats.max_skipped_bound = max_outer;
                }
                chunk_start = chunk_end;
                continue;
            }

            let row_max_k: Vec<f64> = if k_w > 0.0 {
                d_k_env
                    .chunks_exact(n_shells)
                    .map(|row| row.iter().copied().fold(0.0f64, f64::max))
                    .collect()
            } else {
                Vec::new()
            };

            if self.use_gpu {
                if let Some(gpu) = gpu_engine {
                    let ends: Vec<i32> = (0..active_pairs)
                        .map(|pi| {
                            let bra = &self.sig_pairs[pi];
                            let (i, j) = (bra.shell_i, bra.shell_j);
                            let b_ij = bra.bound;
                            let d_j_ij = if chunk_do_j {
                                d_j_env[i * n_shells + j]
                            } else {
                                0.0
                            };
                            let bra_d_eff = if chunk_do_j {
                                j_w * d_j_ij.max(prefix_max_dj[pi])
                            } else {
                                0.0
                            } + if k_w > 0.0 {
                                k_w * row_max_k[i].max(row_max_k[j])
                            } else {
                                0.0
                            };
                            self.sig_pairs[..=pi].partition_point(|q| {
                                opts.threshold <= 0.0
                                    || 4.0 * (b_ij * q.bound) * bra_d_eff >= opts.threshold
                            }) as i32
                        })
                        .collect();

                    let mut d_sym_packed = if chunk_do_j || chunk_do_k_sym {
                        Vec::with_capacity(b_sym_c * nn)
                    } else {
                        Vec::new()
                    };
                    if chunk_do_j || chunk_do_k_sym {
                        for m in &d_sym[sym_start..sym_end] {
                            d_sym_packed.extend_from_slice(m);
                        }
                    }
                    let mut d_anti_packed = if chunk_do_k_anti {
                        Vec::with_capacity(b_anti_c * nn)
                    } else {
                        Vec::new()
                    };
                    if chunk_do_k_anti {
                        for m in &d_anti[anti_start..anti_end] {
                            d_anti_packed.extend_from_slice(m);
                        }
                    }

                    let mut out_j_packed = if chunk_do_j {
                        vec![0.0f64; b_sym_c * nn]
                    } else {
                        Vec::new()
                    };
                    let mut out_ks_packed = if chunk_do_k_sym {
                        vec![0.0f64; b_sym_c * nn]
                    } else {
                        Vec::new()
                    };
                    let mut out_ka_packed = if chunk_do_k_anti {
                        vec![0.0f64; b_anti_c * nn]
                    } else {
                        Vec::new()
                    };
                    let eff_fp32_thresh = if opts.threshold <= 0.0 {
                        0.0
                    } else {
                        resolve_gpu_fp32_thresh().min(opts.threshold * 1e6)
                    };

                    match gpu.contract_tddft_batch_with_code(
                        b_sym_c,
                        b_anti_c,
                        chunk_do_j,
                        chunk_do_k_sym,
                        chunk_do_k_anti,
                        &d_sym_packed,
                        &d_anti_packed,
                        &d_j_env,
                        &d_k_env,
                        active_pairs,
                        &ends,
                        j_w,
                        k_w,
                        opts.omega,
                        opts.threshold,
                        eff_fp32_thresh,
                        &mut out_j_packed,
                        &mut out_ks_packed,
                        &mut out_ka_packed,
                    ) {
                        Ok(gpu_stats) => {
                            n_chunks += 1;
                            self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                            if chunk_do_j {
                                for b in 0..b_sym_c {
                                    let mut mat = out_j_packed[b * nn..(b + 1) * nn].to_vec();
                                    symmetrize_in_place(&mut mat, n);
                                    out.j_sym.push(mat);
                                }
                            }
                            if chunk_do_k_sym {
                                for b in 0..b_sym_c {
                                    let mut mat = out_ks_packed[b * nn..(b + 1) * nn].to_vec();
                                    symmetrize_in_place(&mut mat, n);
                                    out.k_sym.push(mat);
                                }
                            }
                            if chunk_do_k_anti {
                                for b in 0..b_anti_c {
                                    let mut mat = out_ka_packed[b * nn..(b + 1) * nn].to_vec();
                                    antisymmetrize_in_place(&mut mat, n);
                                    out.k_anti.push(mat);
                                }
                            }
                            out.stats.computed_quartets += gpu_stats.computed_quartets as usize;
                            out.stats.skipped_quartets += gpu_stats.skipped_quartets as usize;
                            for (pi, &end_val) in ends.iter().enumerate() {
                                let end = end_val as usize;
                                if end <= pi {
                                    out.stats.skipped_quartets += (pi + 1) - end;
                                }
                            }
                            if active_pairs < num_pairs {
                                out.stats.skipped_quartets +=
                                    total_candidates - active_pairs * (active_pairs + 1) / 2;
                            }
                            chunk_start = chunk_end;
                            continue;
                        }
                        Err(code) if (-4..=-2).contains(&code) && (chunk_end - chunk_start) > 1 => {
                            max_b_chunk = ((chunk_end - chunk_start) / 2).max(1);
                            continue;
                        }
                        Err(_) => {
                            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                } else {
                    self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                }
            }

            n_chunks += 1;
            let primitive_pairs: Vec<_> = self.sig_pairs[..active_pairs]
                .iter()
                .map(|sp| collect_egp_pairs(&basis.shells[sp.shell_i], &basis.shells[sp.shell_j]))
                .collect();

            let k_workers = k_workers_base.min(active_pairs).max(1);
            let sym_slice = &d_sym[sym_start..sym_end];
            let anti_slice = &d_anti[anti_start..anti_end];
            let sym_l1_slice = &d_sym_l1_all[sym_start..sym_end];
            let anti_l1_slice = &d_anti_l1_all[anti_start..anti_end];

            struct BatchWorkerAcc {
                j_sym: Vec<Vec<f64>>,
                k_sym: Vec<Vec<f64>>,
                k_anti: Vec<Vec<f64>>,
                stats: DirectFockStats,
            }

            let mut worker_accs: Vec<BatchWorkerAcc> = (0..k_workers)
                .map(|_| BatchWorkerAcc {
                    j_sym: if chunk_do_j {
                        (0..b_sym_c).map(|_| vec![0.0f64; nn]).collect()
                    } else {
                        Vec::new()
                    },
                    k_sym: if chunk_do_k_sym {
                        (0..b_sym_c).map(|_| vec![0.0f64; nn]).collect()
                    } else {
                        Vec::new()
                    },
                    k_anti: if chunk_do_k_anti {
                        (0..b_anti_c).map(|_| vec![0.0f64; nn]).collect()
                    } else {
                        Vec::new()
                    },
                    stats: DirectFockStats::default(),
                })
                .collect();

            let run_worker = |worker: usize, acc: &mut BatchWorkerAcc| {
                let mut block = Vec::with_capacity(1296);
                for pi in (worker..active_pairs).step_by(k_workers) {
                    let bra = &self.sig_pairs[pi];
                    let (i, j) = (bra.shell_i, bra.shell_j);
                    let b_ij = bra.bound;
                    let sh_a = &basis.shells[i];
                    let sh_b = &basis.shells[j];

                    let d_j_ij = if chunk_do_j {
                        d_j_env[i * n_shells + j]
                    } else {
                        0.0
                    };
                    let bra_d_eff = if chunk_do_j {
                        j_w * d_j_ij.max(prefix_max_dj[pi])
                    } else {
                        0.0
                    } + if k_w > 0.0 {
                        k_w * row_max_k[i].max(row_max_k[j])
                    } else {
                        0.0
                    };

                    let candidates = &self.sig_pairs[..=pi];
                    let end = candidates.partition_point(|q| {
                        opts.threshold <= 0.0 || 4.0 * (b_ij * q.bound) * bra_d_eff >= opts.threshold
                    });

                    if end <= pi {
                        let n_tail = (pi + 1) - end;
                        let tail_bound_sum = 4.0
                            * b_ij
                            * bra_d_eff
                            * (bound_prefix[pi + 1] - bound_prefix[end]).max(0.0);
                        let first_tail_bound = 4.0 * (b_ij * candidates[end].bound) * bra_d_eff;
                        acc.stats.skipped_quartets += n_tail;
                        acc.stats.cumulative_bound_error += tail_bound_sum;
                        if first_tail_bound > acc.stats.max_skipped_bound {
                            acc.stats.max_skipped_bound = first_tail_bound;
                        }
                    }

                    let bra_width = sh_a.n_functions() * sh_b.n_functions();
                    let count = if opts.omega.is_none() {
                        self.cached_kets[pi]
                    } else {
                        0
                    };
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

                    for (qi, pair_q) in candidates[..end].iter().enumerate() {
                        let k = pair_q.shell_i;
                        let l = pair_q.shell_j;
                        let b_kl = pair_q.bound;
                        let b_ijkl = b_ij * b_kl;

                        let bound_j = if chunk_do_j {
                            4.0 * j_w * b_ijkl * d_j_ij.max(d_j_env[k * n_shells + l])
                        } else {
                            0.0
                        };
                        let bound_k = if k_w > 0.0 {
                            let d_exch = d_k_env[i * n_shells + k]
                                .max(d_k_env[i * n_shells + l])
                                .max(d_k_env[j * n_shells + k])
                                .max(d_k_env[j * n_shells + l]);
                            4.0 * k_w * b_ijkl * d_exch
                        } else {
                            0.0
                        };
                        let bound = bound_j + bound_k;

                        if opts.threshold > 0.0 && bound < opts.threshold {
                            acc.stats.skipped_quartets += 1;
                            acc.stats.cumulative_bound_error += bound;
                            if bound > acc.stats.max_skipped_bound {
                                acc.stats.max_skipped_bound = bound;
                            }
                            continue;
                        }

                        acc.stats.computed_quartets += 1;
                        let sh_c = &basis.shells[k];
                        let sh_d = &basis.shells[l];
                        let vals = if qi < count {
                            let start = self.pair_prefix[qi] * bra_width;
                            let stop = self.pair_prefix[qi + 1] * bra_width;
                            let slot = &mut cache.as_mut().unwrap()[start..stop];
                            if slot[0].is_nan() {
                                eri_from_pairs(
                                    sh_a,
                                    sh_b,
                                    sh_c,
                                    sh_d,
                                    &primitive_pairs[pi],
                                    &primitive_pairs[qi],
                                    None,
                                    &mut block,
                                );
                                slot.copy_from_slice(&block);
                            }
                            &slot[..]
                        } else {
                            eri_from_pairs(
                                sh_a,
                                sh_b,
                                sh_c,
                                sh_d,
                                &primitive_pairs[pi],
                                &primitive_pairs[qi],
                                opts.omega,
                                &mut block,
                            );
                            &block[..]
                        };

                        if chunk_do_j || chunk_do_k_sym {
                            for b in 0..b_sym_c {
                                let d_b = &sym_l1_slice[b];
                                let do_j_b = chunk_do_j
                                    && (opts.threshold <= 0.0
                                        || 4.0
                                            * j_w
                                            * b_ijkl
                                            * d_b[i * n_shells + j].max(d_b[k * n_shells + l])
                                            >= opts.threshold);
                                let do_k_b = chunk_do_k_sym
                                    && (opts.threshold <= 0.0
                                        || 4.0
                                            * k_w
                                            * b_ijkl
                                            * d_b[i * n_shells + k]
                                                .max(d_b[i * n_shells + l])
                                                .max(d_b[j * n_shells + k])
                                                .max(d_b[j * n_shells + l])
                                            >= opts.threshold);
                                if do_j_b && do_k_b {
                                    contract_quartet_jk_sym_raw(
                                        i,
                                        j,
                                        k,
                                        l,
                                        sh_a,
                                        sh_b,
                                        sh_c,
                                        sh_d,
                                        vals,
                                        sym_slice[b],
                                        n,
                                        &mut acc.j_sym[b],
                                        &mut acc.k_sym[b],
                                    );
                                } else if do_j_b {
                                    contract_quartet_rks_raw(
                                        i,
                                        j,
                                        k,
                                        l,
                                        sh_a,
                                        sh_b,
                                        sh_c,
                                        sh_d,
                                        vals,
                                        sym_slice[b],
                                        n,
                                        0.0,
                                        &mut acc.j_sym[b],
                                    );
                                } else if do_k_b {
                                    contract_quartet_k(
                                        i,
                                        j,
                                        k,
                                        l,
                                        sh_a,
                                        sh_b,
                                        sh_c,
                                        sh_d,
                                        vals,
                                        sym_slice[b],
                                        n,
                                        &mut acc.k_sym[b],
                                    );
                                }
                            }
                        }

                        if chunk_do_k_anti {
                            for b in 0..b_anti_c {
                                let d_b = &anti_l1_slice[b];
                                let do_k_b = opts.threshold <= 0.0
                                    || 4.0
                                        * k_w
                                        * b_ijkl
                                        * d_b[i * n_shells + k]
                                            .max(d_b[i * n_shells + l])
                                            .max(d_b[j * n_shells + k])
                                            .max(d_b[j * n_shells + l])
                                        >= opts.threshold;
                                if do_k_b {
                                    contract_quartet_k(
                                        i,
                                        j,
                                        k,
                                        l,
                                        sh_a,
                                        sh_b,
                                        sh_c,
                                        sh_d,
                                        vals,
                                        anti_slice[b],
                                        n,
                                        &mut acc.k_anti[b],
                                    );
                                }
                            }
                        }
                    }
                }
            };

            if k_workers == 1 {
                run_worker(0, &mut worker_accs[0]);
            } else {
                worker_accs
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(worker, acc)| run_worker(worker, acc));
            }

            let mut chunk_acc = worker_accs.remove(0);
            for other in worker_accs {
                for (dst, src) in chunk_acc.j_sym.iter_mut().zip(other.j_sym.iter()) {
                    for idx in 0..nn {
                        dst[idx] += src[idx];
                    }
                }
                for (dst, src) in chunk_acc.k_sym.iter_mut().zip(other.k_sym.iter()) {
                    for idx in 0..nn {
                        dst[idx] += src[idx];
                    }
                }
                for (dst, src) in chunk_acc.k_anti.iter_mut().zip(other.k_anti.iter()) {
                    for idx in 0..nn {
                        dst[idx] += src[idx];
                    }
                }
                chunk_acc.stats.computed_quartets += other.stats.computed_quartets;
                chunk_acc.stats.skipped_quartets += other.stats.skipped_quartets;
                chunk_acc.stats.cumulative_bound_error += other.stats.cumulative_bound_error;
                if other.stats.max_skipped_bound > chunk_acc.stats.max_skipped_bound {
                    chunk_acc.stats.max_skipped_bound = other.stats.max_skipped_bound;
                }
            }

            for mat in &mut chunk_acc.j_sym {
                symmetrize_in_place(mat, n);
            }
            for mat in &mut chunk_acc.k_sym {
                symmetrize_in_place(mat, n);
            }
            for mat in &mut chunk_acc.k_anti {
                antisymmetrize_in_place(mat, n);
            }

            out.j_sym.extend(chunk_acc.j_sym);
            out.k_sym.extend(chunk_acc.k_sym);
            out.k_anti.extend(chunk_acc.k_anti);

            out.stats.computed_quartets += chunk_acc.stats.computed_quartets;
            out.stats.skipped_quartets += chunk_acc.stats.skipped_quartets;
            out.stats.cumulative_bound_error += chunk_acc.stats.cumulative_bound_error;
            if chunk_acc.stats.max_skipped_bound > out.stats.max_skipped_bound {
                out.stats.max_skipped_bound = chunk_acc.stats.max_skipped_bound;
            }

            if active_pairs < num_pairs {
                let n_outer = total_candidates - active_pairs * (active_pairs + 1) / 2;
                out.stats.skipped_quartets += n_outer;
                let mut outer_sum = 0.0f64;
                for pi in active_pairs..num_pairs {
                    outer_sum +=
                        4.0 * self.sig_pairs[pi].bound * global_d_eff * bound_prefix[pi + 1];
                }
                out.stats.cumulative_bound_error += outer_sum;
                let max_outer = 4.0 * (largest * self.sig_pairs[active_pairs].bound) * global_d_eff;
                if max_outer > out.stats.max_skipped_bound {
                    out.stats.max_skipped_bound = max_outer;
                }
            }
            chunk_start = chunk_end;
        }

        out.stats.total_quartet_candidates = total_candidates * n_chunks.max(1);
        out
    }

    /// Convenience wrapper for `calchem-tddft`: contracts batched symmetric and antisymmetric
    /// trial densities in one ERI pass (or memory-bounded chunked passes) and returns
    /// `(J[D_sym], K[D_sym], K[D_anti])` as `Matrix2D` vectors.
    #[allow(clippy::too_many_arguments)]
    pub fn build_tddft_response_batch(
        &self,
        basis: &BasisSet,
        d_sym: &[Matrix2D],
        d_anti: &[Matrix2D],
        compute_j: bool,
        compute_k: bool,
        omega: Option<f64>,
        threshold: f64,
    ) -> (Vec<Matrix2D>, Vec<Matrix2D>, Vec<Matrix2D>) {
        let n = self.n_ao;
        let d_sym_refs: Vec<&[f64]> = d_sym.iter().map(|m| m.data.as_slice()).collect();
        let d_anti_refs: Vec<&[f64]> = d_anti.iter().map(|m| m.data.as_slice()).collect();
        let opts = DirectJkBatchOptions {
            compute_j,
            compute_k_sym: compute_k && !d_sym_refs.is_empty(),
            compute_k_anti: compute_k && !d_anti_refs.is_empty(),
            j_screen_weight: 1.0,
            k_screen_weight: 1.0,
            omega,
            threshold,
        };
        let workers = rayon::current_num_threads().max(1);
        let res = self.build_jk_batch_sym_anti_bounded(basis, &d_sym_refs, &d_anti_refs, opts, workers);
        let to_mats = |vecs: Vec<Vec<f64>>| -> Vec<Matrix2D> {
            vecs.into_iter()
                .map(|data| Matrix2D {
                    rows: n,
                    cols: n,
                    data,
                })
                .collect()
        };
        (to_mats(res.j_sym), to_mats(res.k_sym), to_mats(res.k_anti))
    }
}

/// Configuration options for batched symmetric + antisymmetric Direct J/K contraction.
#[derive(Debug, Clone, Copy)]
pub struct DirectJkBatchOptions {
    pub compute_j: bool,
    pub compute_k_sym: bool,
    pub compute_k_anti: bool,
    pub j_screen_weight: f64,
    pub k_screen_weight: f64,
    pub omega: Option<f64>,
    pub threshold: f64,
}

impl Default for DirectJkBatchOptions {
    fn default() -> Self {
        Self {
            compute_j: true,
            compute_k_sym: true,
            compute_k_anti: true,
            j_screen_weight: 1.0,
            k_screen_weight: 1.0,
            omega: None,
            threshold: 1e-12,
        }
    }
}

/// Batched output buffers and screening statistics from [`DirectFockEngine::build_jk_batch_sym_anti_bounded`].
#[derive(Debug, Clone, Default)]
pub struct DirectJkBatchOutput {
    pub j_sym: Vec<Vec<f64>>,
    pub k_sym: Vec<Vec<f64>>,
    pub k_anti: Vec<Vec<f64>>,
    pub stats: DirectFockStats,
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

    #[test]
    fn test_direct_fock_localized_asymmetric_delta_p_preserves_nonlocal_coulomb() {
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [10.0, 0.0, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let last_atom_start = bs
            .shells
            .iter()
            .find(|s| s.atom_index == 2)
            .unwrap()
            .ao_offset;

        // Localized mixed-sign ΔP nonzero ONLY on the distant H atom (shells on O, H1 have row_max == 0)
        let mut dp = vec![0.0f64; n * n];
        for i in last_atom_start..n {
            dp[i * n + i] = if i % 2 == 0 { 0.4 } else { -0.25 };
            for j in last_atom_start..i {
                let v = if (i + j) % 2 == 0 { 0.08 } else { -0.06 };
                dp[i * n + j] = v;
                dp[j * n + i] = v;
            }
        }

        let sparse_eri = compute_sparse_eri(&bs);
        let direct_engine = DirectFockEngine::new(&bs, 1e-12);
        for exx in [0.0, 0.2] {
            let g_ref = sparse_eri.contract_rks(&dp, exx);
            let (g_dir, stats) = direct_engine.build_rks_with_stats(&bs, &dp, exx, 1e-11);
            let max_diff = g_ref
                .iter()
                .zip(&g_dir)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_diff <= stats.cumulative_bound_error + 1e-12,
                "Localized ΔP Coulomb diff {max_diff:.3e} exceeded bound {:.3e} at exx={exx}",
                stats.cumulative_bound_error
            );
            assert!(
                max_diff < 1e-9,
                "Localized ΔP Coulomb diff {max_diff:.3e} too large at exx={exx}"
            );
            // Verify zero-density fast exit too
            let (g_zero, stats_zero) =
                direct_engine.build_rks_with_stats(&bs, &vec![0.0; n * n], exx, 1e-11);
            assert!(g_zero.iter().all(|&v| v == 0.0));
            assert_eq!(stats_zero.computed_quartets, 0);
            assert_eq!(stats_zero.skipped_quartets, stats_zero.total_quartet_candidates);
        }
    }

    #[test]
    fn test_direct_fock_fshell_and_ecp_rks_uks_parity() {
        // Synthetic HBr-like molecule with LANL2DZ (ECP) and an added F shell to exercise L=3 raw contraction
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("Br", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 2.7]).unwrap(),
            ],
            0,
            1,
        );
        let mut bs = BasisSet::from_name("lanl2dz", &mol).unwrap();
        bs.shells.push(Shell {
            atom_index: 1,
            center: [0.0, 0.0, 2.7],
            shell_type: calchem_basis::ShellType::F,
            primitives: vec![calchem_basis::Primitive {
                exponent: 1.1,
                coeff: 1.0,
                raw_coeff: 1.0,
                norm: calchem_basis::primitive_norm(1.1, [3, 0, 0]),
            }],
            ao_offset: bs.n_ao,
        });
        bs.n_ao += 10;

        let n = bs.n_ao;
        let mut p = vec![0.0f64; n * n];
        let mut pa = vec![0.0f64; n * n];
        let mut pb = vec![0.0f64; n * n];
        for i in 0..n {
            p[i * n + i] = if i % 3 == 0 { 0.3 } else { -0.1 };
            pa[i * n + i] = 0.2;
            pb[i * n + i] = -0.05;
            for j in 0..i {
                let v = if (i + j) % 2 == 0 { 0.03 } else { -0.02 };
                p[i * n + j] = v;
                p[j * n + i] = v;
                pa[i * n + j] = v * 0.7;
                pa[j * n + i] = v * 0.7;
                pb[i * n + j] = -v * 0.3;
                pb[j * n + i] = -v * 0.3;
            }
        }

        let sparse_eri = compute_sparse_eri(&bs);
        let direct_engine = DirectFockEngine::new(&bs, 0.0);

        for exx in [0.0, 0.25] {
            let g_ref = sparse_eri.contract_rks(&p, exx);
            let g_dir = direct_engine.build_rks(&bs, &p, exx, 0.0);
            let max_diff = g_ref
                .iter()
                .zip(&g_dir)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_diff < 1e-11,
                "F-shell/ECP RKS diff {max_diff:.3e} exceeded tolerance for exx={exx}"
            );

            // Check UKS vs RKS identity when pa == pb == 0.5 * p
            let half_p: Vec<f64> = p.iter().map(|v| 0.5 * v).collect();
            let (ga, gb, _) = direct_engine.build_uks_bounded(&bs, &half_p, &half_p, exx, 0.0, 4);
            let max_uks_diff = g_ref
                .iter()
                .zip(ga.iter().zip(&gb))
                .map(|(r, (a, b))| (r - a).abs().max((r - b).abs()))
                .fold(0.0f64, f64::max);
            assert!(
                max_uks_diff < 1e-11,
                "F-shell/ECP UKS vs RKS diff {max_uks_diff:.3e} exceeded tolerance for exx={exx}"
            );
        }
    }

    fn build_test_sym_anti_pair(n: usize, seed: usize, zero_diag_shift: bool) -> (Vec<f64>, Vec<f64>) {
        let mut sym = vec![0.0f64; n * n];
        let mut anti = vec![0.0f64; n * n];
        for i in 0..n {
            if !zero_diag_shift {
                sym[i * n + i] = ((i * 3 + seed) as f64 * 0.41).sin() * 0.25;
            }
            for j in 0..i {
                let s_val = ((i * 7 + j * 11 + seed * 5 + 1) as f64 * 0.37).cos() * 0.08;
                let a_val = ((i * 13 + j * 5 + seed * 3 + 2) as f64 * 0.29).sin() * 0.07;
                sym[i * n + j] = s_val;
                sym[j * n + i] = s_val;
                anti[i * n + j] = a_val;
                anti[j * n + i] = -a_val;
            }
        }
        (sym, anti)
    }

    #[test]
    fn test_batched_jk_sym_anti_matches_unbatched_exact_and_screened() {
        let mol = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
                Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
                Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let last_atom_off = bs.shells.iter().find(|s| s.atom_index == 2).unwrap().ao_offset;

        let mut d_syms = Vec::new();
        let mut d_antis = Vec::new();
        // b=0,1: dense sym/anti
        for seed in 0..2 {
            let (s, a) = build_test_sym_anti_pair(n, seed, false);
            d_syms.push(s);
            d_antis.push(a);
        }
        // b=2: exact zero
        d_syms.push(vec![0.0; n * n]);
        d_antis.push(vec![0.0; n * n]);
        // b=3: localized single-atom on H2
        let mut loc_s = vec![0.0; n * n];
        let mut loc_a = vec![0.0; n * n];
        for i in last_atom_off..n {
            loc_s[i * n + i] = 0.35;
            for j in last_atom_off..i {
                loc_s[i * n + j] = 0.05;
                loc_s[j * n + i] = 0.05;
                loc_a[i * n + j] = 0.04;
                loc_a[j * n + i] = -0.04;
            }
        }
        d_syms.push(loc_s);
        d_antis.push(loc_a);
        // b=4: another dense sym/anti
        let (s4, a4) = build_test_sym_anti_pair(n, 7, false);
        d_syms.push(s4);
        d_antis.push(a4);

        let d_sym_refs: Vec<&[f64]> = d_syms.iter().map(|v| v.as_slice()).collect();
        let d_anti_refs: Vec<&[f64]> = d_antis.iter().map(|v| v.as_slice()).collect();
        let engine = DirectFockEngine::new(&bs, 1e-12);
        let sparse_lr = crate::compute_sparse_eri_with_omega(
            &bs,
            &crate::IntegralOptions {
                screen_threshold: 0.0,
                use_screening: false,
                ..crate::IntegralOptions::default()
            },
            Some(0.33),
        );

        for omega in [None, Some(0.33)] {
            for thresh in [0.0, 1e-12] {
                for workers in [1, 4] {
                    let out = engine.build_jk_batch_sym_anti_bounded(
                        &bs,
                        &d_sym_refs,
                        &d_anti_refs,
                        DirectJkBatchOptions {
                            compute_j: true,
                            compute_k_sym: true,
                            compute_k_anti: true,
                            j_screen_weight: 1.0,
                            k_screen_weight: 1.0,
                            omega,
                            threshold: thresh,
                        },
                        workers,
                    );
                    assert_eq!(out.j_sym.len(), 5);
                    assert_eq!(out.k_sym.len(), 5);
                    assert_eq!(out.k_anti.len(), 5);

                    for b in 0..5 {
                        let j_ref = if omega.is_none() {
                            engine.build_rks(&bs, d_sym_refs[b], 0.0, thresh)
                        } else {
                            sparse_lr.contract_rks(d_sym_refs[b], 0.0)
                        };
                        let ks_ref =
                            engine.build_k_general_with_omega(&bs, d_sym_refs[b], omega, thresh);
                        let ka_ref =
                            engine.build_k_general_with_omega(&bs, d_anti_refs[b], omega, thresh);

                        let mut max_j = 0.0f64;
                        let mut max_ks = 0.0f64;
                        let mut max_ka = 0.0f64;
                        let mut max_sym_err = 0.0f64;
                        let mut max_anti_err = 0.0f64;
                        for r in 0..n {
                            max_anti_err = max_anti_err.max(out.k_anti[b][r * n + r].abs());
                            for c in 0..n {
                                max_j = max_j.max((out.j_sym[b][r * n + c] - j_ref[r * n + c]).abs());
                                max_ks = max_ks.max((out.k_sym[b][r * n + c] - ks_ref[r * n + c]).abs());
                                max_ka =
                                    max_ka.max((out.k_anti[b][r * n + c] - ka_ref[r * n + c]).abs());
                                max_sym_err = max_sym_err
                                    .max((out.j_sym[b][r * n + c] - out.j_sym[b][c * n + r]).abs())
                                    .max((out.k_sym[b][r * n + c] - out.k_sym[b][c * n + r]).abs());
                                max_anti_err = max_anti_err
                                    .max((out.k_anti[b][r * n + c] + out.k_anti[b][c * n + r]).abs());
                            }
                        }
                        assert!(
                            max_j < 1e-12 && max_ks < 1e-12 && max_ka < 1e-12,
                            "Batched mismatch (omega={omega:?}, thresh={thresh}, w={workers}, b={b}): j={max_j:.3e}, ks={max_ks:.3e}, ka={max_ka:.3e}"
                        );
                        assert!(
                            max_sym_err < 1e-14 && max_anti_err < 1e-14,
                            "Sym/anti violation: sym={max_sym_err:.3e}, anti={max_anti_err:.3e}"
                        );

                        // Verify J[D_anti] == 0 identically
                        let j_anti_ref = sparse_lr.contract_rks(d_anti_refs[b], 0.0);
                        let max_j_anti = j_anti_ref.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                        assert!(max_j_anti < 1e-14, "J[D_anti] must vanish, got {max_j_anti:.3e}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_batched_jk_cache_consistency_omega_none_then_some() {
        use crate::{compute_sparse_eri_with_omega, IntegralOptions};

        let mol = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
                Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
                Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let (s0, a0) = build_test_sym_anti_pair(n, 1, false);
        let (s1, a1) = build_test_sym_anti_pair(n, 2, false);
        let d_sym_mats = vec![
            Matrix2D { rows: n, cols: n, data: s0 },
            Matrix2D { rows: n, cols: n, data: s1 },
        ];
        let d_anti_mats = vec![
            Matrix2D { rows: n, cols: n, data: a0 },
            Matrix2D { rows: n, cols: n, data: a1 },
        ];

        // 1. Fresh engine running omega = Some(0.33) MUST NOT initialize exchange_rows
        let lr_only_engine = DirectFockEngine::new(&bs, 1e-12);
        let (_, ks_lr_fresh, ka_lr_fresh) = lr_only_engine.build_tddft_response_batch(
            &bs,
            &d_sym_mats,
            &d_anti_mats,
            false,
            true,
            Some(0.33),
            1e-12,
        );
        assert!(
            lr_only_engine.exchange_rows.iter().all(|row| row.get().is_none()),
            "omega=Some(..) must never touch exchange_rows cache"
        );

        // 2. Shared engine: None -> Some(0.33) -> None
        let shared = DirectFockEngine::new(&bs, 1e-12);
        let (j_1, ks_1, ka_1) = shared.build_tddft_response_batch(
            &bs,
            &d_sym_mats,
            &d_anti_mats,
            true,
            true,
            None,
            1e-12,
        );
        let (j_lr, ks_lr, ka_lr) = shared.build_tddft_response_batch(
            &bs,
            &d_sym_mats,
            &d_anti_mats,
            false,
            true,
            Some(0.33),
            1e-12,
        );
        assert!(j_lr.is_empty());
        let (j_2, ks_2, ka_2) = shared.build_tddft_response_batch(
            &bs,
            &d_sym_mats,
            &d_anti_mats,
            true,
            true,
            None,
            1e-12,
        );

        let sparse_none = compute_sparse_eri_with_omega(
            &bs,
            &IntegralOptions {
                screen_threshold: 0.0,
                use_screening: false,
                ..IntegralOptions::default()
            },
            None,
        );
        let sparse_lr = compute_sparse_eri_with_omega(
            &bs,
            &IntegralOptions {
                screen_threshold: 0.0,
                use_screening: false,
                ..IntegralOptions::default()
            },
            Some(0.33),
        );

        for b in 0..2 {
            let j_sp = sparse_none.contract_rks(&d_sym_mats[b].data, 0.0);
            let ks_sp = sparse_none.contract_k_gen(&d_sym_mats[b].data);
            let ka_sp = sparse_none.contract_k_gen(&d_anti_mats[b].data);
            let ks_lr_sp = sparse_lr.contract_k_gen(&d_sym_mats[b].data);
            let ka_lr_sp = sparse_lr.contract_k_gen(&d_anti_mats[b].data);

            for idx in 0..(n * n) {
                assert!((j_1[b].data[idx] - j_2[b].data[idx]).abs() < 1e-14);
                assert!((ks_1[b].data[idx] - ks_2[b].data[idx]).abs() < 1e-14);
                assert!((ka_1[b].data[idx] - ka_2[b].data[idx]).abs() < 1e-14);
                assert!((ks_lr[b].data[idx] - ks_lr_fresh[b].data[idx]).abs() < 1e-14);
                assert!((ka_lr[b].data[idx] - ka_lr_fresh[b].data[idx]).abs() < 1e-14);

                assert!((j_1[b].data[idx] - j_sp[idx]).abs() < 1e-12);
                assert!((ks_1[b].data[idx] - ks_sp[idx]).abs() < 1e-12);
                assert!((ka_1[b].data[idx] - ka_sp[idx]).abs() < 1e-12);
                assert!((ks_lr[b].data[idx] - ks_lr_sp[idx]).abs() < 1e-12);
                assert!((ka_lr[b].data[idx] - ka_lr_sp[idx]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_batched_jk_edge_cases_and_chunking() {
        let mol = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
                Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
                Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let engine = DirectFockEngine::new(&bs, 1e-12);

        // B = 0
        let empty = engine.build_jk_batch_sym_anti_bounded(
            &bs,
            &[],
            &[],
            DirectJkBatchOptions::default(),
            4,
        );
        assert!(empty.j_sym.is_empty() && empty.k_sym.is_empty() && empty.k_anti.is_empty());

        let mut d_syms = Vec::new();
        let mut d_antis = Vec::new();
        for seed in 0..5 {
            let (s, a) = build_test_sym_anti_pair(n, seed + 10, false);
            d_syms.push(s);
            d_antis.push(a);
        }
        let d_sym_refs: Vec<&[f64]> = d_syms.iter().map(|v| v.as_slice()).collect();
        let d_anti_refs: Vec<&[f64]> = d_antis.iter().map(|v| v.as_slice()).collect();

        // TDA-like: d_anti empty, compute_k_anti = false
        let tda_out = engine.build_jk_batch_sym_anti_bounded(
            &bs,
            &d_sym_refs,
            &[],
            DirectJkBatchOptions {
                compute_j: true,
                compute_k_sym: true,
                compute_k_anti: false,
                ..DirectJkBatchOptions::default()
            },
            4,
        );
        assert_eq!(tda_out.j_sym.len(), 5);
        assert_eq!(tda_out.k_sym.len(), 5);
        assert!(tda_out.k_anti.is_empty());

        // d_sym empty, compute_k_anti = true
        let anti_only = engine.build_jk_batch_sym_anti_bounded(
            &bs,
            &[],
            &d_anti_refs,
            DirectJkBatchOptions {
                compute_j: false,
                compute_k_sym: false,
                compute_k_anti: true,
                ..DirectJkBatchOptions::default()
            },
            4,
        );
        assert!(anti_only.j_sym.is_empty() && anti_only.k_sym.is_empty());
        assert_eq!(anti_only.k_anti.len(), 5);

        // Forced small chunking (max_b_chunk = 2 on B = 5) vs unchunked
        let full = engine.build_jk_batch_sym_anti_bounded(
            &bs,
            &d_sym_refs,
            &d_anti_refs,
            DirectJkBatchOptions::default(),
            4,
        );
        let chunked = engine.build_jk_batch_sym_anti_bounded_impl(
            &bs,
            &d_sym_refs,
            &d_anti_refs,
            DirectJkBatchOptions::default(),
            4,
            Some(2),
        );
        for b in 0..5 {
            for idx in 0..(n * n) {
                assert!((full.j_sym[b][idx] - chunked.j_sym[b][idx]).abs() < 1e-14);
                assert!((full.k_sym[b][idx] - chunked.k_sym[b][idx]).abs() < 1e-14);
                assert!((full.k_anti[b][idx] - chunked.k_anti[b][idx]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_batched_jk_fshell_and_ecp_parity() {
        use crate::{compute_sparse_eri_with_omega, IntegralOptions};

        let mol = Molecule::new(
            vec![
                Atom::from_bohr("Br", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 2.7]).unwrap(),
            ],
            0,
            1,
        );
        let mut bs = BasisSet::from_name("lanl2dz", &mol).unwrap();
        bs.shells.push(Shell {
            atom_index: 1,
            center: [0.0, 0.0, 2.7],
            shell_type: calchem_basis::ShellType::F,
            primitives: vec![calchem_basis::Primitive {
                exponent: 1.1,
                coeff: 1.0,
                raw_coeff: 1.0,
                norm: calchem_basis::primitive_norm(1.1, [3, 0, 0]),
            }],
            ao_offset: bs.n_ao,
        });
        bs.n_ao += 10;
        let n = bs.n_ao;

        let mut d_syms = Vec::new();
        let mut d_antis = Vec::new();
        for seed in 0..3 {
            let (s, a) = build_test_sym_anti_pair(n, seed + 20, false);
            d_syms.push(s);
            d_antis.push(a);
        }
        let d_sym_refs: Vec<&[f64]> = d_syms.iter().map(|v| v.as_slice()).collect();
        let d_anti_refs: Vec<&[f64]> = d_antis.iter().map(|v| v.as_slice()).collect();
        let engine = DirectFockEngine::new(&bs, 0.0);

        for omega in [None, Some(0.30)] {
            let sparse = compute_sparse_eri_with_omega(
                &bs,
                &IntegralOptions {
                    screen_threshold: 0.0,
                    use_screening: false,
                    ..IntegralOptions::default()
                },
                omega,
            );
            let out = engine.build_jk_batch_sym_anti_bounded(
                &bs,
                &d_sym_refs,
                &d_anti_refs,
                DirectJkBatchOptions {
                    compute_j: omega.is_none(),
                    compute_k_sym: true,
                    compute_k_anti: true,
                    j_screen_weight: 1.0,
                    k_screen_weight: 1.0,
                    omega,
                    threshold: 0.0,
                },
                4,
            );
            for b in 0..3 {
                if omega.is_none() {
                    let j_ref = sparse.contract_rks(d_sym_refs[b], 0.0);
                    for (idx, &jr) in j_ref.iter().enumerate() {
                        assert!(
                            (out.j_sym[b][idx] - jr).abs() < 1e-11,
                            "F-shell/ECP J mismatch at b={b}"
                        );
                    }
                }
                let ks_ref = sparse.contract_k_gen(d_sym_refs[b]);
                let ka_ref = sparse.contract_k_gen(d_anti_refs[b]);
                for idx in 0..(n * n) {
                    assert!(
                        (out.k_sym[b][idx] - ks_ref[idx]).abs() < 1e-11,
                        "F-shell/ECP K_sym mismatch at omega={omega:?}, b={b}"
                    );
                    assert!(
                        (out.k_anti[b][idx] - ka_ref[idx]).abs() < 1e-11,
                        "F-shell/ECP K_anti mismatch at omega={omega:?}, b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_partial_exchange_rows_cache_coarse_tight_and_omega_parity() {
        use crate::{compute_sparse_eri_with_omega, IntegralOptions};

        let mol = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [6.0, 0.0, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let last_atom_off = bs.shells.iter().find(|s| s.atom_index == 2).unwrap().ao_offset;

        let full = DirectFockEngine::new_with_cache_budget(&bs, 1e-12, usize::MAX);
        let full_cache_bytes: usize = (0..full.sig_pairs.len())
            .map(|pi| {
                let bra_w = full.pair_prefix[pi + 1] - full.pair_prefix[pi];
                bra_w * full.pair_prefix[full.cached_kets[pi]] * 8
            })
            .sum();
        let partial_budget = (full_cache_bytes / 4).max(1);
        let partial = DirectFockEngine::new_with_cache_budget(&bs, 1e-12, partial_budget);
        let uncached = DirectFockEngine::new_with_cache_budget(&bs, 1e-12, 0);

        assert!(
            partial
                .cached_kets
                .iter()
                .enumerate()
                .any(|(pi, &c)| c > 0 && c < pi + 1),
            "Expected partial ket caching (0 < cached_kets[pi] < pi + 1), got {:?}",
            *partial.cached_kets
        );
        assert!(uncached.cached_kets.iter().all(|&c| c == 0));

        let (s0, a0) = build_test_sym_anti_pair(n, 31, false);
        let mut loc_s = vec![0.0; n * n];
        let mut loc_a = vec![0.0; n * n];
        for i in last_atom_off..n {
            loc_s[i * n + i] = 0.35;
            for j in last_atom_off..i {
                loc_s[i * n + j] = 0.05;
                loc_s[j * n + i] = 0.05;
                loc_a[i * n + j] = 0.04;
                loc_a[j * n + i] = -0.04;
            }
        }
        let d_syms = [s0, vec![0.0; n * n], loc_s];
        let d_antis = [a0, vec![0.0; n * n], loc_a];
        let d_sym_refs: Vec<&[f64]> = d_syms.iter().map(|v| v.as_slice()).collect();
        let d_anti_refs: Vec<&[f64]> = d_antis.iter().map(|v| v.as_slice()).collect();

        // 1. Coarse pass (omega = None, threshold = 1e-4) leaves screened slots as NaN
        let coarse_opts = DirectJkBatchOptions {
            compute_j: true,
            compute_k_sym: true,
            compute_k_anti: true,
            j_screen_weight: 1.0,
            k_screen_weight: 1.0,
            omega: None,
            threshold: 1e-4,
        };
        let p_coarse = partial.build_jk_batch_sym_anti_bounded(&bs, &d_sym_refs, &d_anti_refs, coarse_opts, 4);
        let u_coarse = uncached.build_jk_batch_sym_anti_bounded(&bs, &d_sym_refs, &d_anti_refs, coarse_opts, 4);
        for b in 0..d_syms.len() {
            for idx in 0..(n * n) {
                assert!((p_coarse.j_sym[b][idx] - u_coarse.j_sym[b][idx]).abs() < 1e-14);
                assert!((p_coarse.k_sym[b][idx] - u_coarse.k_sym[b][idx]).abs() < 1e-14);
                assert!((p_coarse.k_anti[b][idx] - u_coarse.k_anti[b][idx]).abs() < 1e-14);
            }
        }
        let (rks_c_p, _) = partial.build_rks_bounded(&bs, d_sym_refs[0], 0.25, 1e-4, 4);
        let (rks_c_u, _) = uncached.build_rks_bounded(&bs, d_sym_refs[0], 0.25, 1e-4, 4);
        for idx in 0..(n * n) {
            assert!((rks_c_p[idx] - rks_c_u[idx]).abs() < 1e-14);
        }

        // 2. Intermediate omega = Some(0.33) Tight pass must bypass partially-populated exchange_rows
        let lr_opts = DirectJkBatchOptions {
            compute_j: false,
            compute_k_sym: true,
            compute_k_anti: true,
            j_screen_weight: 1.0,
            k_screen_weight: 1.0,
            omega: Some(0.33),
            threshold: 1e-12,
        };
        let p_lr = partial.build_jk_batch_sym_anti_bounded(&bs, &d_sym_refs, &d_anti_refs, lr_opts, 4);
        let u_lr = uncached.build_jk_batch_sym_anti_bounded(&bs, &d_sym_refs, &d_anti_refs, lr_opts, 4);
        let sparse_lr = compute_sparse_eri_with_omega(
            &bs,
            &IntegralOptions { screen_threshold: 0.0, use_screening: false, ..IntegralOptions::default() },
            Some(0.33),
        );
        for b in 0..d_syms.len() {
            let ks_sp = sparse_lr.contract_k_gen(d_sym_refs[b]);
            let ka_sp = sparse_lr.contract_k_gen(d_anti_refs[b]);
            for idx in 0..(n * n) {
                assert!((p_lr.k_sym[b][idx] - u_lr.k_sym[b][idx]).abs() < 1e-14);
                assert!((p_lr.k_anti[b][idx] - u_lr.k_anti[b][idx]).abs() < 1e-14);
                assert!((p_lr.k_sym[b][idx] - ks_sp[idx]).abs() < 1e-12);
                assert!((p_lr.k_anti[b][idx] - ka_sp[idx]).abs() < 1e-12);
            }
        }

        // 3. Tight pass (omega = None, threshold = 1e-12) lazily fills NaN holes + uncached tail
        let tight_opts = DirectJkBatchOptions {
            compute_j: true,
            compute_k_sym: true,
            compute_k_anti: true,
            j_screen_weight: 1.0,
            k_screen_weight: 1.0,
            omega: None,
            threshold: 1e-12,
        };
        let sparse_none = compute_sparse_eri_with_omega(
            &bs,
            &IntegralOptions { screen_threshold: 0.0, use_screening: false, ..IntegralOptions::default() },
            None,
        );
        for workers in [1, 4] {
            let p_tight = partial.build_jk_batch_sym_anti_bounded(&bs, &d_sym_refs, &d_anti_refs, tight_opts, workers);
            let u_tight = uncached.build_jk_batch_sym_anti_bounded(&bs, &d_sym_refs, &d_anti_refs, tight_opts, workers);
            for b in 0..d_syms.len() {
                let j_sp = sparse_none.contract_rks(d_sym_refs[b], 0.0);
                let ks_sp = sparse_none.contract_k_gen(d_sym_refs[b]);
                let ka_sp = sparse_none.contract_k_gen(d_anti_refs[b]);
                for idx in 0..(n * n) {
                    assert!((p_tight.j_sym[b][idx] - u_tight.j_sym[b][idx]).abs() < 1e-14);
                    assert!((p_tight.k_sym[b][idx] - u_tight.k_sym[b][idx]).abs() < 1e-14);
                    assert!((p_tight.k_anti[b][idx] - u_tight.k_anti[b][idx]).abs() < 1e-14);
                    assert!((p_tight.j_sym[b][idx] - j_sp[idx]).abs() < 1e-12);
                    assert!((p_tight.k_sym[b][idx] - ks_sp[idx]).abs() < 1e-12);
                    assert!((p_tight.k_anti[b][idx] - ka_sp[idx]).abs() < 1e-12);
                }
            }
        }

        // Cross-check SCF RKS/UKS & K paths on the same partially cached engine
        let (rks_p, _) = partial.build_rks_bounded(&bs, d_sym_refs[0], 0.25, 1e-12, 4);
        let (rks_u, _) = uncached.build_rks_bounded(&bs, d_sym_refs[0], 0.25, 1e-12, 4);
        let (ga_p, gb_p, _) = partial.build_uks_bounded(&bs, d_sym_refs[0], d_sym_refs[2], 0.25, 1e-12, 4);
        let (ga_u, gb_u, _) = uncached.build_uks_bounded(&bs, d_sym_refs[0], d_sym_refs[2], 0.25, 1e-12, 4);
        let k_sym_p = partial.build_k_with_omega_bounded(&bs, d_sym_refs[0], None, 1e-12, 4);
        let k_sym_u = uncached.build_k_with_omega_bounded(&bs, d_sym_refs[0], None, 1e-12, 4);
        let k_gen_p = partial.build_k_general_with_omega(&bs, d_anti_refs[0], None, 1e-12);
        let k_gen_u = uncached.build_k_general_with_omega(&bs, d_anti_refs[0], None, 1e-12);
        for idx in 0..(n * n) {
            assert!((rks_p[idx] - rks_u[idx]).abs() < 1e-14);
            assert!((ga_p[idx] - ga_u[idx]).abs() < 1e-14);
            assert!((gb_p[idx] - gb_u[idx]).abs() < 1e-14);
            assert!((k_sym_p[idx] - k_sym_u[idx]).abs() < 1e-14);
            assert!((k_gen_p[idx] - k_gen_u[idx]).abs() < 1e-14);
        }

        // Verify a partial row was genuinely allocated and total bytes respect the budget
        assert!(
            partial.exchange_rows.iter().enumerate().any(|(pi, lock)| {
                lock.get().is_some()
                    && partial.cached_kets[pi] > 0
                    && partial.cached_kets[pi] < pi + 1
            }),
            "At least one partially cached exchange_rows slot must be initialized"
        );
        let alloc_bytes: usize = partial
            .exchange_rows
            .iter()
            .filter_map(|lock| lock.get().map(|m| m.lock().unwrap().len() * 8))
            .sum();
        assert!(
            alloc_bytes <= partial_budget,
            "exchange_rows allocated {alloc_bytes} bytes > budget {partial_budget}"
        );
    }

    #[test]
    fn test_gpu_direct_rks_uks_k_omega_parity() {
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
        let n = bs.n_ao;
        let mut pa = vec![0.0f64; n * n];
        let mut pb = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let va = 0.04 / (1.0 + (i as f64 - j as f64).abs());
                let vb = 0.03 / (1.0 + 1.5 * (i as f64 - j as f64).abs());
                pa[i * n + j] = va;
                pa[j * n + i] = va;
                pb[i * n + j] = vb;
                pb[j * n + i] = vb;
            }
        }

        let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
        let gpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);

        // 1. RKS JK (exx=0.0, 0.20, 1.0) — run twice to verify VRAM quartet cache hit!
        for exx in [0.0, 0.20, 1.0] {
            let (g_cpu, _) = cpu.build_rks_bounded(&bs, &pa, exx, 1e-12, 4);
            let (g_gpu1, _) = gpu.build_rks_bounded(&bs, &pa, exx, 1e-12, 4);
            let (g_gpu2, _) = gpu.build_rks_bounded(&bs, &pa, exx, 1e-12, 4);
            let mut max_err1 = 0.0f64;
            let mut max_err2 = 0.0f64;
            for idx in 0..n * n {
                max_err1 = max_err1.max((g_gpu1[idx] - g_cpu[idx]).abs());
                max_err2 = max_err2.max((g_gpu2[idx] - g_cpu[idx]).abs());
            }
            assert!(max_err1 < 1e-7, "GPU RKS exx={exx} miss/eval max_err={max_err1:.3e}");
            assert!(max_err2 < 1e-7, "GPU RKS exx={exx} cached max_err={max_err2:.3e}");
        }

        // 2. UKS JK
        let (ga_cpu, gb_cpu, _) = cpu.build_uks_bounded(&bs, &pa, &pb, 0.25, 1e-12, 4);
        let (ga_gpu, gb_gpu, _) = gpu.build_uks_bounded(&bs, &pa, &pb, 0.25, 1e-12, 4);
        let mut max_uks = 0.0f64;
        for idx in 0..n * n {
            max_uks = max_uks
                .max((ga_gpu[idx] - ga_cpu[idx]).abs())
                .max((gb_gpu[idx] - gb_cpu[idx]).abs());
        }
        assert!(max_uks < 1e-7, "GPU UKS max_err={max_uks:.3e}");

        // 3. K-only & K-pair with omega = None and omega = Some(0.33)
        for omega in [None, Some(0.33)] {
            let k_cpu = cpu.build_k_with_omega_bounded(&bs, &pa, omega, 1e-12, 4);
            let k_gpu = gpu.build_k_with_omega_bounded(&bs, &pa, omega, 1e-12, 4);
            let (ka_cpu, kb_cpu) = cpu.build_k_pair_with_omega_bounded(&bs, &pa, &pb, omega, 1e-12, 4);
            let (ka_gpu, kb_gpu) = gpu.build_k_pair_with_omega_bounded(&bs, &pa, &pb, omega, 1e-12, 4);
            let mut max_k = 0.0f64;
            for idx in 0..n * n {
                max_k = max_k
                    .max((k_gpu[idx] - k_cpu[idx]).abs())
                    .max((ka_gpu[idx] - ka_cpu[idx]).abs())
                    .max((kb_gpu[idx] - kb_cpu[idx]).abs());
            }
            assert!(max_k < 1e-7, "GPU K/K_pair omega={omega:?} max_err={max_k:.3e}");
        }

        let k_cnt = gpu.gpu_kernel_invocations.load(Ordering::Relaxed);
        let fb_cnt = gpu.gpu_fallback_invocations.load(Ordering::Relaxed);
        assert!(k_cnt >= 10 && fb_cnt == 0, "Expected k>=10 fb==0, got k={k_cnt} fb={fb_cnt}");
    }

    #[test]
    fn test_gpu_direct_primary_f_shell_parity() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        // Fe + H in 6-31G(d) exercises primary F-shell (L=3) F_BLOCK on GPU
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
        let mut p = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = 0.02 / (1.0 + (i as f64 - j as f64).abs());
                p[i * n + j] = v;
                p[j * n + i] = v;
            }
        }
        let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
        let gpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
        let (g_cpu, _) = cpu.build_rks_bounded(&bs, &p, 0.20, 1e-12, 4);
        let (g_gpu, _) = gpu.build_rks_bounded(&bs, &p, 0.20, 1e-12, 4);
        let mut max_err = 0.0f64;
        for idx in 0..n * n {
            max_err = max_err.max((g_gpu[idx] - g_cpu[idx]).abs());
        }
        assert!(max_err < 1e-11, "GPU F_BLOCK vs CPU max_err={max_err:.3e}");
        assert_eq!(gpu.gpu_fallback_invocations.load(Ordering::Relaxed), 0);
    }

    struct EnvVarGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvVarGuard {
        fn new(keys: &[&'static str]) -> Self {
            let saved = keys
                .iter()
                .map(|&k| (k, std::env::var(k).ok()))
                .collect();
            Self { saved }
        }
    }

    impl Drop for EnvVarGuard {
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
    fn test_gpu_direct_dd_dd_and_dd_ff_parity() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let _env = EnvVarGuard::new(&["CALCHEM_GPU_FP32_THRESH"]);
        std::env::set_var("CALCHEM_GPU_FP32_THRESH", "0");

        // ZnO 6-31G(d) exercises both (d,d|d,d) in eval_d_warp_impl (n_a==35, max_cd==4)
        // and (d,d|f,f) in direct_eval_f_f64_kernel (WL_F_64).
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("Zn", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("O", [0.0, 0.0, 3.2]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        assert!(bs.shells.iter().any(|s| s.shell_type.angular_momentum() == 2));
        assert!(bs.shells.iter().any(|s| s.shell_type.angular_momentum() == 3));

        let n = bs.n_ao;
        let mut pa = vec![0.0f64; n * n];
        let mut pb = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let va = 0.03 * ((i * 7 + j * 3 + 1) as f64 * 0.23).cos();
                let vb = 0.025 * ((i * 5 + j * 11 + 2) as f64 * 0.19).sin();
                pa[i * n + j] = va;
                pa[j * n + i] = va;
                pb[i * n + j] = vb;
                pb[j * n + i] = vb;
            }
        }

        let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
        let gpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);

        for exx in [0.0, 0.20, 1.0] {
            let (g_cpu, _) = cpu.build_rks_bounded(&bs, &pa, exx, 1e-12, 4);
            let (g_gpu_miss, _) = gpu.build_rks_bounded(&bs, &pa, exx, 1e-12, 4);
            let (g_gpu_hit, _) = gpu.build_rks_bounded(&bs, &pa, exx, 1e-12, 4);
            let mut err_miss = 0.0f64;
            let mut err_hit = 0.0f64;
            for idx in 0..n * n {
                err_miss = err_miss.max((g_gpu_miss[idx] - g_cpu[idx]).abs());
                err_hit = err_hit.max((g_gpu_hit[idx] - g_cpu[idx]).abs());
            }
            assert!(err_miss < 1e-11, "ZnO RKS miss exx={exx} err={err_miss:.3e}");
            assert!(err_hit < 1e-11, "ZnO RKS hit exx={exx} err={err_hit:.3e}");
        }

        let (ga_cpu, gb_cpu, _) = cpu.build_uks_bounded(&bs, &pa, &pb, 0.25, 1e-12, 4);
        let (ga_gpu, gb_gpu, _) = gpu.build_uks_bounded(&bs, &pa, &pb, 0.25, 1e-12, 4);
        let mut err_uks = 0.0f64;
        for idx in 0..n * n {
            err_uks = err_uks
                .max((ga_gpu[idx] - ga_cpu[idx]).abs())
                .max((gb_gpu[idx] - gb_cpu[idx]).abs());
        }
        assert!(err_uks < 1e-11, "ZnO UKS err={err_uks:.3e}");

        for omega in [None, Some(0.33)] {
            let k_cpu = cpu.build_k_with_omega_bounded(&bs, &pa, omega, 1e-12, 4);
            let k_gpu = gpu.build_k_with_omega_bounded(&bs, &pa, omega, 1e-12, 4);
            let (ka_cpu, kb_cpu) = cpu.build_k_pair_with_omega_bounded(&bs, &pa, &pb, omega, 1e-12, 4);
            let (ka_gpu, kb_gpu) = gpu.build_k_pair_with_omega_bounded(&bs, &pa, &pb, omega, 1e-12, 4);
            let mut err_k = 0.0f64;
            for idx in 0..n * n {
                err_k = err_k
                    .max((k_gpu[idx] - k_cpu[idx]).abs())
                    .max((ka_gpu[idx] - ka_cpu[idx]).abs())
                    .max((kb_gpu[idx] - kb_cpu[idx]).abs());
            }
            assert!(err_k < 1e-11, "ZnO K/K_pair omega={omega:?} err={err_k:.3e}");
        }

        assert!(gpu.gpu_kernel_invocations.load(Ordering::Relaxed) >= 11);
        assert_eq!(gpu.gpu_fallback_invocations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_gpu_direct_forced_small_wl_stride_and_tight_vram_f_scratch() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let _env = EnvVarGuard::new(&[
            "CALCHEM_GPU_DIR_WL_STRIDE",
            "CALCHEM_MAX_GPU_MEM_MB",
            "CALCHEM_MAX_GPU_K_CACHE_BYTES",
            "CALCHEM_GPU_FP32_THRESH",
        ]);
        std::env::set_var("CALCHEM_GPU_FP32_THRESH", "0");

        let mol = Molecule::new(
            vec![
                Atom::from_bohr("Zn", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("O", [0.0, 0.0, 3.2]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let mut p = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = 0.025 / (1.0 + (i as f64 - j as f64).abs());
                p[i * n + j] = v;
                p[j * n + i] = v;
            }
        }

        let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
        let (g_cpu, cpu_stats) = cpu.build_rks_bounded(&bs, &p, 0.20, 1e-12, 4);
        let k_cpu = cpu.build_k_with_omega_bounded(&bs, &p, Some(0.33), 1e-12, 4);

        // 1. Force tiny dir_wl_stride = 16 so single-bra ket range sub-splitting (h_ends[pi] > 16) runs
        std::env::set_var("CALCHEM_GPU_DIR_WL_STRIDE", "16");
        {
            let gpu_split = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
            assert!(gpu_split.sig_pairs.len() > 16);
            let (g_gpu1, stats1) = gpu_split.build_rks_bounded(&bs, &p, 0.20, 1e-12, 4);
            let (g_gpu2, stats2) = gpu_split.build_rks_bounded(&bs, &p, 0.20, 1e-12, 4);
            let k_gpu = gpu_split.build_k_with_omega_bounded(&bs, &p, Some(0.33), 1e-12, 4);

            let max_g1 = g_gpu1.iter().zip(&g_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            let max_g2 = g_gpu2.iter().zip(&g_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            let max_k = k_gpu.iter().zip(&k_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(max_g1 < 1e-11, "WL split miss err={max_g1:.3e}");
            assert!(max_g2 < 1e-11, "WL split hit err={max_g2:.3e}");
            assert!(max_k < 1e-11, "WL split K(omega) err={max_k:.3e}");
            assert_eq!(stats1.computed_quartets, cpu_stats.computed_quartets);
            assert_eq!(stats2.computed_quartets, cpu_stats.computed_quartets);
            assert_eq!(gpu_split.gpu_fallback_invocations.load(Ordering::Relaxed), 0);
            assert_eq!(gpu_split.gpu_kernel_invocations.load(Ordering::Relaxed), 3);
        }

        // 2. Tight VRAM budget (10 MiB < 44.79 MiB for f_grid=64): forces adaptive f_grid shrink on WL_F_64
        std::env::remove_var("CALCHEM_GPU_DIR_WL_STRIDE");
        std::env::set_var("CALCHEM_MAX_GPU_K_CACHE_BYTES", "0");
        std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "10");
        {
            let gpu_tight = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
            let (g_tight, _) = gpu_tight.build_rks_bounded(&bs, &p, 0.20, 1e-12, 4);
            let max_err = g_tight.iter().zip(&g_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(max_err < 1e-11, "Tight 10 MiB WL_F_64 shrink err={max_err:.3e}");
            assert_eq!(gpu_tight.gpu_fallback_invocations.load(Ordering::Relaxed), 0);
            assert_eq!(gpu_tight.gpu_kernel_invocations.load(Ordering::Relaxed), 1);
            assert!(
                calchem_cuda::total_allocated_gpu_bytes() <= 10 * 1024 * 1024,
                "GPU allocations exceeded 10 MiB budget: {}",
                calchem_cuda::total_allocated_gpu_bytes()
            );
        }

        // 3. Ultra-tight VRAM budget (1 MiB): even f_grid=1 cannot fit -> clean -4 CPU fallback
        std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "1");
        {
            let gpu_oom = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
            let (g_oom, _) = gpu_oom.build_rks_bounded(&bs, &p, 0.20, 1e-12, 4);
            let max_err = g_oom.iter().zip(&g_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(max_err < 1e-11, "1 MiB fallback err={max_err:.3e}");
            assert_eq!(gpu_oom.gpu_fallback_invocations.load(Ordering::Relaxed), 1);
            let alloc_now = calchem_cuda::total_allocated_gpu_bytes();
            assert!(
                alloc_now <= 1024 * 1024,
                "GPU allocations exceeded 1 MiB budget on OOM fallback: {alloc_now} bytes"
            );
        }
    }

    #[test]
    fn test_gpu_build_k_general_with_omega_disable_gpu_records_fallback() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvVarGuard::new(&["CALCHEM_DISABLE_GPU"]);
        std::env::set_var("CALCHEM_DISABLE_GPU", "1");

        let mol = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.43, -1.11]).unwrap(),
                Atom::from_bohr("H", [0.0, -1.43, -1.11]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let mut x = vec![0.0f64; n * n];
        for r in 0..n {
            for c in 0..n {
                x[r * n + c] = 0.01 * ((r + 1) as f64 - 0.5 * (c as f64)) / (1.0 + (r as f64 - c as f64).abs());
            }
        }
        let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
        let k_cpu = cpu.build_k_general_with_omega(&bs, &x, Some(0.33), 1e-12);
        let gpu_dis = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
        let k_dis = gpu_dis.build_k_general_with_omega(&bs, &x, Some(0.33), 1e-12);

        assert_eq!(gpu_dis.gpu_telemetry(), (0, 1));
        let max_err = k_dis.iter().zip(&k_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_err < 1e-14, "Disabled GPU build_k_general_with_omega mismatch: {max_err:.3e}");
    }

    #[test]
    fn test_gpu_multi_context_coexistence() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let _env = EnvVarGuard::new(&["CALCHEM_GPU_FP32_THRESH", "CALCHEM_MAX_GPU_K_CACHE_BYTES"]);
        std::env::set_var("CALCHEM_GPU_FP32_THRESH", "0");
        std::env::set_var("CALCHEM_MAX_GPU_K_CACHE_BYTES", "16777216");

        let mol1 = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs1 = BasisSet::from_name("6-31g(d)", &mol1).unwrap();
        let n1 = bs1.n_ao;
        let mut p1 = vec![0.0f64; n1 * n1];
        for i in 0..n1 {
            for j in 0..=i {
                let v = 0.05 / (1.0 + (i as f64 - j as f64).abs());
                p1[i * n1 + j] = v;
                p1[j * n1 + i] = v;
            }
        }

        let mol2 = Molecule::new(
            vec![
                Atom::from_bohr("Fe", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 3.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs2 = BasisSet::from_name("6-31g(d)", &mol2).unwrap();
        let n2 = bs2.n_ao;
        let mut p2 = vec![0.0f64; n2 * n2];
        for i in 0..n2 {
            for j in 0..=i {
                let v = 0.02 / (1.0 + (i as f64 - j as f64).abs());
                p2[i * n2 + j] = v;
                p2[j * n2 + i] = v;
            }
        }

        let cpu1 = DirectFockEngine::new_with_gpu(&bs1, 1e-12, false);
        let cpu2 = DirectFockEngine::new_with_gpu(&bs2, 1e-12, false);
        let (ref1, _) = cpu1.build_rks_bounded(&bs1, &p1, 0.20, 1e-12, 4);
        let (ref2, _) = cpu2.build_rks_bounded(&bs2, &p2, 0.20, 1e-12, 4);

        let gpu1 = DirectFockEngine::new_with_gpu(&bs1, 1e-12, true);
        let (g1_a, _) = gpu1.build_rks_bounded(&bs1, &p1, 0.20, 1e-12, 4);

        // Upload a second live GpuDirectEngine AND a live GpuVxcGrid while gpu1 stays alive
        let gpu2 = DirectFockEngine::new_with_gpu(&bs2, 1e-12, true);
        let (g2_a, _) = gpu2.build_rks_bounded(&bs2, &p2, 0.20, 1e-12, 4);

        let vxc_grid = calchem_cuda::GpuVxcGrid::upload(
            &[calchem_cuda::AoShellGpu {
                center: [0.0, 0.0, 0.0],
                r2_cutoff: 100.0,
                ao_offset: 0,
                n_funcs: 1,
                prim_start: 0,
                n_prims: 1,
                comp_start: 0,
                atom_index: 0,
            }],
            &[calchem_cuda::AoPrimGpu {
                exponent: 1.0,
                weighted_norm: 1.0,
            }],
            &[calchem_cuda::AoCompGpu {
                scale: 1.0,
                lx: 0,
                ly: 0,
                lz: 0,
                _pad: 0,
            }],
            &[calchem_cuda::GridPtGpu {
                x: 0.1,
                y: 0.2,
                z: 0.3,
                batch_id: 0,
                host_atom: 0,
            }],
            &[0, 1],
            &[0],
            n1,
        )
        .expect("GpuVxcGrid upload should coexist with 2 live GpuDirectEngines");

        // Re-run BOTH gpu1 and gpu2 plus vxc_grid: none may evict another!
        let (g1_b, _) = gpu1.build_rks_bounded(&bs1, &p1, 0.20, 1e-12, 4);
        let (g2_b, _) = gpu2.build_rks_bounded(&bs2, &p2, 0.20, 1e-12, 4);
        let mut rg = [0.0f64; 4];
        assert!(vxc_grid.eval_rho_grad(&p1, &mut rg));

        let err1 = g1_a
            .iter()
            .chain(g1_b.iter())
            .zip(ref1.iter().chain(ref1.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let err2 = g2_a
            .iter()
            .chain(g2_b.iter())
            .zip(ref2.iter().chain(ref2.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(err1 < 1e-11, "gpu1 coexistence err={err1:.3e}");
        assert!(err2 < 1e-11, "gpu2 coexistence err={err2:.3e}");
        assert_eq!(gpu1.gpu_telemetry(), (2, 0));
        assert_eq!(gpu2.gpu_telemetry(), (2, 0));
    }

    #[test]
    fn test_gpu_tddft_batch_parity_d_and_f_shells() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let _env = EnvVarGuard::new(&[
            "CALCHEM_GPU_FP32_THRESH",
            "CALCHEM_GPU_DIR_WL_STRIDE",
            "CALCHEM_DISABLE_GPU",
        ]);

        let h2o = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let zno = Molecule::new(
            vec![
                Atom::from_bohr("Zn", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("O", [0.0, 0.0, 3.2]).unwrap(),
            ],
            0,
            1,
        );

        for (label, mol, basis_name, fp32_val, wl_stride) in [
            ("h2o_fp64", &h2o, "6-31g(d)", "0", "0"),
            ("zno_fp64", &zno, "6-31g(d)", "0", "0"),
            ("zno_fp32_split", &zno, "6-31g(d)", "1e-5", "16"),
        ] {
            std::env::set_var("CALCHEM_GPU_FP32_THRESH", fp32_val);
            if wl_stride != "0" {
                std::env::set_var("CALCHEM_GPU_DIR_WL_STRIDE", wl_stride);
            } else {
                std::env::remove_var("CALCHEM_GPU_DIR_WL_STRIDE");
            }
            let bs = BasisSet::from_name(basis_name, mol).unwrap();
            let n = bs.n_ao;
            let b_sym_cnt = 5usize;
            let b_anti_cnt = 3usize;
            let mut d_sym = Vec::with_capacity(b_sym_cnt);
            for b in 0..b_sym_cnt {
                let mut m = Matrix2D::zeros(n, n);
                for i in 0..n {
                    for j in 0..=i {
                        let v = 0.02 * (((b + 1) * (i * 5 + j * 3 + 1)) as f64 * 0.17).cos();
                        m.set(i, j, v);
                        m.set(j, i, v);
                    }
                }
                d_sym.push(m);
            }
            let mut d_anti = Vec::with_capacity(b_anti_cnt);
            for b in 0..b_anti_cnt {
                let mut m = Matrix2D::zeros(n, n);
                for i in 0..n {
                    for j in (i + 1)..n {
                        let v = 0.015 * (((b + 1) * (i * 7 + j * 11 + 2)) as f64 * 0.21).sin();
                        m.set(i, j, v);
                        m.set(j, i, -v);
                    }
                }
                d_anti.push(m);
            }

            let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
            let gpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
            let tol = if fp32_val == "0" { 1e-11 } else { 1e-7 };

            for omega in [None, Some(0.33)] {
                let (j_cpu, ks_cpu, ka_cpu) =
                    cpu.build_tddft_response_batch(&bs, &d_sym, &d_anti, true, true, omega, 1e-12);
                // First pass (cache miss for omega=None) and second pass (cache hit for omega=None)
                for pass in 0..2 {
                    let (j_gpu, ks_gpu, ka_gpu) = gpu.build_tddft_response_batch(
                        &bs, &d_sym, &d_anti, true, true, omega, 1e-12,
                    );
                    let mut err_j = 0.0f64;
                    let mut err_ks = 0.0f64;
                    let mut err_ka = 0.0f64;
                    let mut worst_j = (0usize, 0usize, 0.0f64, 0.0f64);
                    for b in 0..b_sym_cnt {
                        for idx in 0..n * n {
                            let dj = (j_gpu[b].data[idx] - j_cpu[b].data[idx]).abs();
                            if dj > err_j {
                                err_j = dj;
                                worst_j = (idx / n, idx % n, j_gpu[b].data[idx], j_cpu[b].data[idx]);
                            }
                            err_ks = err_ks.max((ks_gpu[b].data[idx] - ks_cpu[b].data[idx]).abs());
                        }
                    }
                    for b in 0..b_anti_cnt {
                        for idx in 0..n * n {
                            err_ka = err_ka.max((ka_gpu[b].data[idx] - ka_cpu[b].data[idx]).abs());
                        }
                    }
                    let max_err = err_j.max(err_ks).max(err_ka);
                    assert!(
                        max_err < tol,
                        "{label} omega={omega:?} pass={pass} err_j={err_j:.3e} worst_j={worst_j:?} err_ks={err_ks:.3e} err_ka={err_ka:.3e} >= {tol:.3e}"
                    );
                }
            }

            // Also verify forced small chunk_cap_override = 2
            let d_sym_refs: Vec<&[f64]> = d_sym.iter().map(|m| m.data.as_slice()).collect();
            let d_anti_refs: Vec<&[f64]> = d_anti.iter().map(|m| m.data.as_slice()).collect();
            let opts = DirectJkBatchOptions {
                compute_j: true,
                compute_k_sym: true,
                compute_k_anti: true,
                j_screen_weight: 1.0,
                k_screen_weight: 1.0,
                omega: None,
                threshold: 1e-12,
            };
            let ref_chunk = cpu.build_jk_batch_sym_anti_bounded(&bs, &d_sym_refs, &d_anti_refs, opts, 4);
            let gpu_chunk = gpu.build_jk_batch_sym_anti_bounded_impl(
                &bs,
                &d_sym_refs,
                &d_anti_refs,
                opts,
                4,
                Some(2),
            );
            for b in 0..b_sym_cnt {
                for idx in 0..n * n {
                    assert!((gpu_chunk.j_sym[b][idx] - ref_chunk.j_sym[b][idx]).abs() < tol);
                    assert!((gpu_chunk.k_sym[b][idx] - ref_chunk.k_sym[b][idx]).abs() < tol);
                }
            }
            for b in 0..b_anti_cnt {
                for idx in 0..n * n {
                    assert!((gpu_chunk.k_anti[b][idx] - ref_chunk.k_anti[b][idx]).abs() < tol);
                }
            }
            assert_eq!(gpu.gpu_fallback_invocations.load(Ordering::Relaxed), 0);
            assert!(gpu.gpu_kernel_invocations.load(Ordering::Relaxed) >= 7);
        }

        // Verify CALCHEM_DISABLE_GPU=1 triggers clean fallback telemetry
        std::env::set_var("CALCHEM_DISABLE_GPU", "1");
        let bs = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
        let n = bs.n_ao;
        let ds = vec![Matrix2D::zeros(n, n)];
        let mut d_nonzero = Matrix2D::zeros(n, n);
        d_nonzero.set(0, 0, 0.5);
        let da = vec![Matrix2D::zeros(n, n)];
        let gpu_dis = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
        let _ = gpu_dis.build_tddft_response_batch(&bs, &[d_nonzero], &da, true, true, None, 1e-12);
        let _ = ds;
        assert_eq!(gpu_dis.gpu_telemetry(), (0, 1));
    }

    #[test]
    fn test_gpu_direct_delta_p_zero_density_and_uks_pure_dft_parity() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;

        let mut dp = vec![0.0f64; n * n];
        let mut dp_b = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let va = 0.03 * (((i * 11 + j * 5) as f64) * 0.37).sin();
                let vb = -0.02 * (((i * 7 + j * 13) as f64) * 0.29).cos();
                dp[i * n + j] = va;
                dp[j * n + i] = va;
                dp_b[i * n + j] = vb;
                dp_b[j * n + i] = vb;
            }
        }

        let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
        // Verify MED-03: threshold == 0.0 with j_screen_weight == 0.0 must NOT skip quartets!
        let opts_unscreened = DirectJkBatchOptions {
            compute_j: true,
            compute_k_sym: false,
            compute_k_anti: false,
            j_screen_weight: 0.0,
            k_screen_weight: 0.0,
            omega: None,
            threshold: 0.0,
        };
        let out_unscreened = cpu.build_jk_batch_sym_anti_bounded(&bs, &[&dp], &[], opts_unscreened, 2);
        let max_j_unscreened = out_unscreened.j_sym[0].iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(max_j_unscreened > 1e-3, "threshold=0.0 j_screen_weight=0.0 must not skip quartets");

        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let _env = EnvVarGuard::new(&["CALCHEM_GPU_FP32_THRESH"]);
        std::env::set_var("CALCHEM_GPU_FP32_THRESH", "0");

        let gpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);

        let (g_cpu, _) = cpu.build_rks_bounded(&bs, &dp, 0.20, 1e-12, 4);
        let (g_gpu, _) = gpu.build_rks_bounded(&bs, &dp, 0.20, 1e-12, 4);
        let err_rks = g_cpu.iter().zip(&g_gpu).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(err_rks < 1e-11, "GPU delta_P RKS mismatch: {err_rks:.3e}");

        let (ga_cpu, gb_cpu, _) = cpu.build_uks_bounded(&bs, &dp, &dp_b, 0.0, 1e-12, 4);
        let (ga_gpu, gb_gpu, _) = gpu.build_uks_bounded(&bs, &dp, &dp_b, 0.0, 1e-12, 4);
        let err_uks0 = ga_cpu
            .iter()
            .chain(&gb_cpu)
            .zip(ga_gpu.iter().chain(&gb_gpu))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(err_uks0 < 1e-11, "GPU pure-DFT UKS mismatch: {err_uks0:.3e}");

        let zeros = vec![0.0f64; n * n];
        let (gz, _) = gpu.build_rks_bounded(&bs, &zeros, 0.25, 1e-12, 4);
        assert!(gz.iter().all(|&v| v == 0.0));
        assert_eq!(gpu.gpu_fallback_invocations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_gpu_direct_k_general_omega_and_vram_fallback_parity() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let _env = EnvVarGuard::new(&["CALCHEM_GPU_FP32_THRESH", "CALCHEM_MAX_GPU_MEM_MB"]);
        std::env::set_var("CALCHEM_GPU_FP32_THRESH", "0");

        let mol = Molecule::new(
            vec![
                Atom::from_bohr("Zn", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("O", [0.0, 0.0, 3.2]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let mut x = vec![0.0f64; n * n];
        let mut pa = vec![0.0f64; n * n];
        let mut pb = vec![0.0f64; n * n];
        for r in 0..n {
            for c in 0..n {
                x[r * n + c] = 0.015 * (((r * 5 + c * 9 + 1) as f64) * 0.21).sin()
                    + 0.01 * (((r + 1) as f64) - 0.7 * (c as f64)) / (1.0 + (r as f64 - c as f64).abs());
            }
            for c in 0..=r {
                let va = 0.02 / (1.0 + (r as f64 - c as f64).abs());
                let vb = 0.015 / (1.0 + 1.3 * (r as f64 - c as f64).abs());
                pa[r * n + c] = va;
                pa[c * n + r] = va;
                pb[r * n + c] = vb;
                pb[c * n + r] = vb;
            }
        }

        let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
        let gpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
        for omega in [None, Some(0.33)] {
            let k_cpu = cpu.build_k_general_with_omega(&bs, &x, omega, 1e-12);
            let k_gpu = gpu.build_k_general_with_omega(&bs, &x, omega, 1e-12);
            let err = k_cpu.iter().zip(&k_gpu).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(err < 1e-11, "GPU build_k_general_with_omega ({omega:?}) err={err:.3e}");
        }
        assert_eq!(gpu.gpu_fallback_invocations.load(Ordering::Relaxed), 0);

        std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "1");
        let gpu_oom = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
        let (ka_cpu, kb_cpu) = cpu.build_k_pair_with_omega_bounded(&bs, &pa, &pb, None, 1e-12, 4);
        let (ka_oom, kb_oom) = gpu_oom.build_k_pair_with_omega_bounded(&bs, &pa, &pb, None, 1e-12, 4);
        let err_kp = ka_cpu
            .iter()
            .chain(&kb_cpu)
            .zip(ka_oom.iter().chain(&kb_oom))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(err_kp < 1e-11);
        assert_eq!(gpu_oom.gpu_telemetry(), (0, 1), "K_pair fallback must not double-count");
    }

    #[test]
    fn test_gpu_tddft_batch_anti_only_asymmetric_chunking_and_single_atom_ecp() {
        let _gpu_guard = crate::GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let _env = EnvVarGuard::new(&["CALCHEM_GPU_FP32_THRESH"]);
        std::env::set_var("CALCHEM_GPU_FP32_THRESH", "0");

        let mol = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let n = bs.n_ao;
        let (ds, _) = build_test_sym_anti_pair(n, 1, false);
        let d_antis: Vec<Vec<f64>> = (0..4).map(|s| build_test_sym_anti_pair(n, s + 2, true).1).collect();
        let da_refs: Vec<&[f64]> = d_antis.iter().map(|v| v.as_slice()).collect();

        let cpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, false);
        let gpu = DirectFockEngine::new_with_gpu(&bs, 1e-12, true);
        let opts = DirectJkBatchOptions {
            compute_j: true,
            compute_k_sym: true,
            compute_k_anti: true,
            j_screen_weight: 1.0,
            k_screen_weight: 1.0,
            omega: None,
            threshold: 1e-12,
        };
        let ref_asym = cpu.build_jk_batch_sym_anti_bounded(&bs, &[&ds], &da_refs, opts, 4);
        let gpu_asym = gpu.build_jk_batch_sym_anti_bounded_impl(&bs, &[&ds], &da_refs, opts, 4, Some(2));
        for b in 0..4 {
            let err = ref_asym.k_anti[b]
                .iter()
                .zip(&gpu_asym.k_anti[b])
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f64, f64::max);
            assert!(err < 1e-11, "Asymmetric chunked k_anti[{b}] err={err:.3e}");
        }

        let opts_anti = DirectJkBatchOptions {
            compute_j: false,
            compute_k_sym: false,
            compute_k_anti: true,
            ..opts
        };
        let ref_anti = cpu.build_jk_batch_sym_anti_bounded(&bs, &[], &da_refs, opts_anti, 4);
        let gpu_anti = gpu.build_jk_batch_sym_anti_bounded(&bs, &[], &da_refs, opts_anti, 4);
        for b in 0..4 {
            let err = ref_anti.k_anti[b]
                .iter()
                .zip(&gpu_anti.k_anti[b])
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f64, f64::max);
            assert!(err < 1e-11);
        }

        let h_atom = Molecule::new(vec![Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap()], 0, 2);
        let bs_h = BasisSet::from_name("sto-3g", &h_atom).unwrap();
        let cpu_h = DirectFockEngine::new_with_gpu(&bs_h, 1e-12, false);
        let gpu_h = DirectFockEngine::new_with_gpu(&bs_h, 1e-12, true);
        let (gh_c, _) = cpu_h.build_rks_bounded(&bs_h, &[1.0], 0.25, 1e-12, 1);
        let (gh_g, _) = gpu_h.build_rks_bounded(&bs_h, &[1.0], 0.25, 1e-12, 1);
        assert!((gh_c[0] - gh_g[0]).abs() < 1e-12);

        let hi = Molecule::new(
            vec![
                Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("I", [0.0, 0.0, 3.1]).unwrap(),
            ],
            0,
            1,
        );
        let bs_hi = BasisSet::from_name("lanl2dzdp", &hi).unwrap();
        let nhi = bs_hi.n_ao;
        let phi: Vec<f64> = (0..nhi * nhi).map(|k| if k / nhi == k % nhi { 0.5 } else { 0.01 }).collect();
        let cpu_hi = DirectFockEngine::new_with_gpu(&bs_hi, 1e-12, false);
        let gpu_hi = DirectFockEngine::new_with_gpu(&bs_hi, 1e-12, true);
        let (ghi_c, _) = cpu_hi.build_rks_bounded(&bs_hi, &phi, 0.20, 1e-12, 4);
        let (ghi_g, _) = gpu_hi.build_rks_bounded(&bs_hi, &phi, 0.20, 1e-12, 4);
        let err_hi = ghi_c.iter().zip(&ghi_g).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(err_hi < 1e-11, "GPU Direct ECP HI err={err_hi:.3e}");
    }
}
