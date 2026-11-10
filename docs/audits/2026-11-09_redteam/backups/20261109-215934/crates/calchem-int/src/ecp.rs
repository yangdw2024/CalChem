//! Scalar Effective Core Potential (ECP) integrals (Type-1 local + Type-2 semilocal projectors).
//!
//! Evaluates `<chi_a | U_ECP | chi_b>` where on each ECP center `C`:
//!   `U_C(r_C) = U_{L_max}(r_C) + sum_{l=0}^{L_max - 1} \Delta U_l(r_C) \hat{P}_l(\hat{r}_C)`
//! with radial Gaussian expansions `\sum_k d_k r^{n_k - 2} exp(-\zeta_k r^2)`.
//!
//! Numerical formulation:
//! - Exact Funk-Hecke spherical harmonic expansion with scaled modified spherical Bessel functions
//!   `M_\lambda(z) = exp(-z) i_\lambda(z) \in [0, 1]`, combined analytically with Gaussian factors
//!   `exp(-\alpha(r - |C - A|)^2)` so overflow/underflow cancellation never occurs for any exponent.
//! - Exact degree-31 spherical polynomial quadrature (`16 x 32` Gauss-Legendre x uniform-\phi)
//!   computed once (`OnceLock`) outside radial loops, guaranteeing machine-precision rotational
//!   invariance.
//! - Per-term Gaussian-windowed 48-point Gauss-Legendre radial quadrature matched to the exact
//!   localization scale `p_tot = \alpha + \beta + \zeta_k`.

use crate::primitive::GaussianPair;
use calchem_basis::{AtomEcp, BasisSet, EcpTerm, Shell};
use calchem_tensor::Matrix2D;
use std::f64::consts::PI;
use std::sync::OnceLock;

const N_ANG_THETA: usize = 16;
const N_ANG_PHI: usize = 32;
const N_RAD_GL: usize = 48;
const MAX_SPH_L: usize = 12;

struct EcpGrids {
    /// Unit vectors `(ux, uy, uz)` and quadrature weights `w` on `S^2` (`sum w = 4 * pi`).
    ang_dirs: Vec<[f64; 3]>,
    ang_weights: Vec<f64>,
    /// Real orthonormal spherical harmonics `S_{l, m_idx}(u)` at each angular point:
    /// indexed as `[pt_idx * NUM_YLM + l * l + m_offset]` where `m_offset in 0..(2*l + 1)`.
    ylm_table: Vec<f64>,
    /// 48-point Gauss-Legendre nodes `x in [-1, 1]` and weights `w`.
    rad_nodes: Vec<f64>,
    rad_weights: Vec<f64>,
}

const NUM_YLM: usize = (MAX_SPH_L + 1) * (MAX_SPH_L + 1);

static ECP_GRIDS: OnceLock<EcpGrids> = OnceLock::new();

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = (n + 1) / 2;
    let nf = n as f64;

    for i in 0..m {
        let mut z = (PI * ((i as f64) + 0.75) / (nf + 0.5)).cos();
        let mut pp = 0.0;
        for _ in 0..12 {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf - 1.0) * z * p2 - (jf - 1.0) * p3) / jf;
            }
            pp = nf * (z * p1 - p2) / (z * z - 1.0);
            let dz = p1 / pp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

/// Evaluates real orthonormal spherical harmonics `S_{l,m}` (`0 <= l <= MAX_SPH_L`) at
/// `(cos_theta, sin_theta, phi)`.
fn eval_real_ylm_point(cos_t: f64, sin_t: f64, phi: f64, out: &mut [f64]) {
    // Associated Legendre polynomials P_l^m(cos_t) for 0 <= m <= l <= MAX_SPH_L
    let mut plm = [[0.0f64; MAX_SPH_L + 1]; MAX_SPH_L + 1];
    plm[0][0] = 1.0;
    for m in 1..=MAX_SPH_L {
        plm[m][m] = ((2 * m - 1) as f64) * sin_t * plm[m - 1][m - 1];
    }
    for m in 0..MAX_SPH_L {
        plm[m + 1][m] = ((2 * m + 1) as f64) * cos_t * plm[m][m];
    }
    for m in 0..=MAX_SPH_L {
        for l in (m + 2)..=MAX_SPH_L {
            plm[l][m] = (((2 * l - 1) as f64) * cos_t * plm[l - 1][m]
                - ((l + m - 1) as f64) * plm[l - 2][m])
                / ((l - m) as f64);
        }
    }

    for l in 0..=MAX_SPH_L {
        let base = l * l;
        let pref0 = (((2 * l + 1) as f64) / (4.0 * PI)).sqrt();
        out[base + l] = pref0 * plm[l][0];

        let mut ratio_fact = 1.0f64; // (l - m)! / (l + m)!
        for m in 1..=l {
            ratio_fact /= ((l - m + 1) * (l + m)) as f64;
            let norm = (((2 * l + 1) as f64) / (2.0 * PI) * ratio_fact).sqrt();
            let mf = m as f64;
            out[base + l + m] = norm * plm[l][m] * (mf * phi).cos();
            out[base + l - m] = norm * plm[l][m] * (mf * phi).sin();
        }
    }
}

fn grids() -> &'static EcpGrids {
    ECP_GRIDS.get_or_init(|| {
        let (theta_nodes, theta_weights) = gauss_legendre(N_ANG_THETA);
        let dphi = 2.0 * PI / (N_ANG_PHI as f64);
        let n_ang = N_ANG_THETA * N_ANG_PHI;
        let mut ang_dirs = Vec::with_capacity(n_ang);
        let mut ang_weights = Vec::with_capacity(n_ang);
        let mut ylm_table = vec![0.0; n_ang * NUM_YLM];

        let mut idx = 0;
        for (cos_t, wt) in theta_nodes.into_iter().zip(theta_weights.into_iter()) {
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            for j in 0..N_ANG_PHI {
                let phi = (j as f64) * dphi;
                let ux = sin_t * phi.cos();
                let uy = sin_t * phi.sin();
                let uz = cos_t;
                ang_dirs.push([ux, uy, uz]);
                ang_weights.push(wt * dphi);
                eval_real_ylm_point(
                    cos_t,
                    sin_t,
                    phi,
                    &mut ylm_table[idx * NUM_YLM..(idx + 1) * NUM_YLM],
                );
                idx += 1;
            }
        }

        let (rad_nodes, rad_weights) = gauss_legendre(N_RAD_GL);
        EcpGrids {
            ang_dirs,
            ang_weights,
            ylm_table,
            rad_nodes,
            rad_weights,
        }
    })
}

/// Evaluates `M_\lambda(z) = exp(-z) i_\lambda(z)` for `\lambda = 0..=max_l` and `z >= 0`.
/// Strictly bounded in `[0, 1]` with zero overflow/underflow cancellation across `[0, \infty)`.
pub fn scaled_mod_sph_bessel_batch(max_l: usize, z: f64, out: &mut [f64]) {
    debug_assert!(out.len() > max_l);
    if z <= 0.0 {
        out[0] = 1.0;
        for slot in out.iter_mut().take(max_l + 1).skip(1) {
            *slot = 0.0;
        }
        return;
    }

    if z <= 14.0 {
        let exp_neg_z = (-z).exp();
        let half_z2 = 0.5 * z * z;
        let mut z_pow = 1.0;
        let mut dfact = 1.0; // (2l + 1)!!
        for (l, slot) in out.iter_mut().enumerate().take(max_l + 1) {
            if l > 0 {
                z_pow *= z;
                dfact *= (2 * l + 1) as f64;
            }
            let mut term = z_pow / dfact;
            let mut sum = term;
            for k in 1..60 {
                term *= half_z2 / ((k as f64) * ((2 * l + 2 * k + 1) as f64));
                sum += term;
                if term <= sum * 1e-16 {
                    break;
                }
            }
            *slot = exp_neg_z * sum;
        }
        return;
    }

    // Exact finite Rayleigh-Hankel closed form for z > 14.0:
    // M_l(z) = 1/(2z) * [ sum_{k=0}^l (-1)^k c_{l,k} (2z)^{-k}
    //                   + (-1)^{l+1} exp(-2z) sum_{k=0}^l c_{l,k} (2z)^{-k} ]
    let inv_2z = 0.5 / z;
    let exp_neg_2z = (-2.0 * z).exp();
    for (l, slot) in out.iter_mut().enumerate().take(max_l + 1) {
        let mut c_lk = 1.0;
        let mut inv_2z_k = 1.0;
        let mut sum_pos = 1.0;
        let mut sum_alt = 1.0;
        for k in 1..=l {
            c_lk *= (((l + k) * (l + 1 - k)) as f64) / (k as f64);
            inv_2z_k *= inv_2z;
            let mag = c_lk * inv_2z_k;
            sum_pos += mag;
            if k % 2 == 0 {
                sum_alt += mag;
            } else {
                sum_alt -= mag;
            }
        }
        let sign_tail = if (l + 1) % 2 == 0 { 1.0 } else { -1.0 };
        *slot = inv_2z * (sum_alt + sign_tail * exp_neg_2z * sum_pos);
    }
}

/// Evaluates Legendre polynomials `P_0(x)..=P_{max_l}(x)` into `out`.
#[inline]
fn legendre_batch(max_l: usize, x: f64, out: &mut [f64]) {
    out[0] = 1.0;
    if max_l == 0 {
        return;
    }
    out[1] = x;
    for l in 1..max_l {
        let lf = l as f64;
        out[l + 1] = ((2.0 * lf + 1.0) * x * out[l] - lf * out[l - 1]) / (lf + 1.0);
    }
}

/// Expands `(y + c)^a` into coefficients `coeffs[0..=a]` of `y^p`.
#[inline]
fn expand_binomial_1d(a: u8, c: f64) -> ([f64; 9], usize) {
    let mut poly = [0.0; 9];
    poly[0] = 1.0;
    let mut deg = 0usize;
    for _ in 0..a {
        for p in (0..=deg).rev() {
            poly[p + 1] += poly[p];
            poly[p] *= c;
        }
        deg += 1;
    }
    (poly, deg)
}

/// Expands `(y + ca)^a * (y + cb)^b` into coefficients `coeffs[0..=(a+b)]` of `y^p`.
#[inline]
fn expand_pair_1d(a: u8, ca: f64, b: u8, cb: f64) -> ([f64; 9], usize) {
    let (pa, da) = expand_binomial_1d(a, ca);
    let (pb, db) = expand_binomial_1d(b, cb);
    let mut out = [0.0; 9];
    for i in 0..=da {
        for j in 0..=db {
            out[i + j] += pa[i] * pb[j];
        }
    }
    (out, da + db)
}

#[inline]
fn unit_neg_dir(v: [f64; 3]) -> (f64, [f64; 3]) {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm > 1e-15 {
        (norm, [-v[0] / norm, -v[1] / norm, -v[2] / norm])
    } else {
        (0.0, [0.0, 0.0, 1.0])
    }
}

/// Evaluates the unnormalized primitive Type-1 local ECP integral
/// `<phi_a(r - A) | sum_k d_k |r - C|^{n_k - 2} exp(-zeta_k |r - C|^2) | phi_b(r - B)>`.
pub fn ecp_local_primitive(
    a: [u8; 3],
    b: [u8; 3],
    alpha: f64,
    r_a: [f64; 3],
    beta: f64,
    r_b: [f64; 3],
    r_c: [f64; 3],
    terms: &[EcpTerm],
) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let pair = GaussianPair::new(alpha, r_a, beta, r_b);
    if pair.kab < 1e-36 {
        return 0.0;
    }

    let g = grids();
    let ca = [r_c[0] - r_a[0], r_c[1] - r_a[1], r_c[2] - r_a[2]];
    let cb = [r_c[0] - r_b[0], r_c[1] - r_b[1], r_c[2] - r_b[2]];
    let cp = [r_c[0] - pair.p[0], r_c[1] - pair.p[1], r_c[2] - pair.p[2]];
    let (rp, k_hat) = unit_neg_dir(cp);

    let (fx, dx) = expand_pair_1d(a[0], ca[0], b[0], cb[0]);
    let (fy, dy) = expand_pair_1d(a[1], ca[1], b[1], cb[1]);
    let (fz, dz) = expand_pair_1d(a[2], ca[2], b[2], cb[2]);
    let max_p = dx + dy + dz;
    debug_assert!(max_p <= MAX_SPH_L);

    // omega[p][lam] = (2*lam + 1) * \oint_{S^2} \Phi_p(u) P_lam(k_hat . u) du
    let mut omega = [[0.0f64; MAX_SPH_L + 1]; MAX_SPH_L + 1];
    let mut leg = [0.0f64; MAX_SPH_L + 1];
    let mut ux_pow = [1.0f64; 9];
    let mut uy_pow = [1.0f64; 9];
    let mut uz_pow = [1.0f64; 9];

    for (u, &w_ang) in g.ang_dirs.iter().zip(g.ang_weights.iter()) {
        for p in 1..=dx {
            ux_pow[p] = ux_pow[p - 1] * u[0];
        }
        for p in 1..=dy {
            uy_pow[p] = uy_pow[p - 1] * u[1];
        }
        for p in 1..=dz {
            uz_pow[p] = uz_pow[p - 1] * u[2];
        }

        let mut phi_p = [0.0f64; MAX_SPH_L + 1];
        for ix in 0..=dx {
            let vx = fx[ix] * ux_pow[ix];
            if vx == 0.0 {
                continue;
            }
            for iy in 0..=dy {
                let vxy = vx * fy[iy] * uy_pow[iy];
                if vxy == 0.0 {
                    continue;
                }
                for iz in 0..=dz {
                    phi_p[ix + iy + iz] += vxy * fz[iz] * uz_pow[iz];
                }
            }
        }

        let cos_ku = k_hat[0] * u[0] + k_hat[1] * u[1] + k_hat[2] * u[2];
        legendre_batch(max_p, cos_ku, &mut leg);

        for p in 0..=max_p {
            let val_p = phi_p[p] * w_ang;
            if val_p == 0.0 {
                continue;
            }
            let mut lam = p % 2;
            while lam <= p {
                omega[p][lam] += val_p * leg[lam];
                lam += 2;
            }
        }
    }
    for p in 0..=max_p {
        let mut lam = p % 2;
        while lam <= p {
            omega[p][lam] *= (2 * lam + 1) as f64;
            lam += 2;
        }
    }

    let gamma = pair.gamma;
    let mut total = 0.0;
    let mut bessel = [0.0f64; MAX_SPH_L + 1];

    for term in terms {
        let zeta = term.exponent;
        let p_tot = gamma + zeta;
        let q_min = zeta * gamma * rp * rp / p_tot;
        if q_min > 85.0 {
            continue;
        }
        let r0 = gamma * rp / p_tot;
        let sigma = (1.0 / p_tot).sqrt();
        let n_pow = (term.r_power as usize) + max_p;
        let r_peak = 0.5 * (r0 + (r0 * r0 + 2.0 * (n_pow as f64) / p_tot).sqrt());
        let r_lo = (r0 - 10.0 * sigma).max(0.0);
        let r_hi = r_peak + 10.0 * sigma;
        let half_len = 0.5 * (r_hi - r_lo);
        let mid = 0.5 * (r_hi + r_lo);

        let mut rad_sum = 0.0;
        for (&x, &w_rad) in g.rad_nodes.iter().zip(g.rad_weights.iter()) {
            let r = mid + half_len * x;
            let exp_env = (-zeta * r * r - gamma * (r - rp) * (r - rp)).exp();
            if exp_env < 1e-40 {
                continue;
            }
            scaled_mod_sph_bessel_batch(max_p, 2.0 * gamma * rp * r, &mut bessel);

            let mut ang_r = 0.0;
            let mut r_pow_p = 1.0;
            for p in 0..=max_p {
                if p > 0 {
                    r_pow_p *= r;
                }
                let mut inner = 0.0;
                let mut lam = p % 2;
                while lam <= p {
                    inner += omega[p][lam] * bessel[lam];
                    lam += 2;
                }
                ang_r += r_pow_p * inner;
            }

            rad_sum += w_rad * r.powi(term.r_power as i32) * exp_env * ang_r;
        }

        total += term.coeff * pair.kab * half_len * rad_sum;
    }

    total
}

/// Precomputes single-center angular projection weights `\omega^{(A)}_{m, p, \lambda}` for
/// projector angular momentum `l` and primitive `(a, A)`.
/// Layout: `((m_idx * (la + 1) + p) * (max_lam + 1)) + lam`.
fn precompute_semilocal_center_omega(
    a: [u8; 3],
    ca: [f64; 3],
    k_hat: [f64; 3],
    l: usize,
) -> (Vec<f64>, usize, usize) {
    let g = grids();
    let (fx, dx) = expand_binomial_1d(a[0], ca[0]);
    let (fy, dy) = expand_binomial_1d(a[1], ca[1]);
    let (fz, dz) = expand_binomial_1d(a[2], ca[2]);
    let la = dx + dy + dz;
    let max_lam = la + l;
    debug_assert!(max_lam <= MAX_SPH_L);

    let n_m = 2 * l + 1;
    let stride_p = max_lam + 1;
    let stride_m = (la + 1) * stride_p;
    let mut omega = vec![0.0f64; n_m * stride_m];

    let mut leg = [0.0f64; MAX_SPH_L + 1];
    let mut ux_pow = [1.0f64; 9];
    let mut uy_pow = [1.0f64; 9];
    let mut uz_pow = [1.0f64; 9];
    let ylm_base = l * l;

    for (pt_idx, (u, &w_ang)) in g.ang_dirs.iter().zip(g.ang_weights.iter()).enumerate() {
        for p in 1..=dx {
            ux_pow[p] = ux_pow[p - 1] * u[0];
        }
        for p in 1..=dy {
            uy_pow[p] = uy_pow[p - 1] * u[1];
        }
        for p in 1..=dz {
            uz_pow[p] = uz_pow[p - 1] * u[2];
        }

        let mut phi_p = [0.0f64; 9];
        for ix in 0..=dx {
            let vx = fx[ix] * ux_pow[ix];
            if vx == 0.0 {
                continue;
            }
            for iy in 0..=dy {
                let vxy = vx * fy[iy] * uy_pow[iy];
                if vxy == 0.0 {
                    continue;
                }
                for iz in 0..=dz {
                    phi_p[ix + iy + iz] += vxy * fz[iz] * uz_pow[iz];
                }
            }
        }

        let cos_ku = k_hat[0] * u[0] + k_hat[1] * u[1] + k_hat[2] * u[2];
        legendre_batch(max_lam, cos_ku, &mut leg);
        let ylm_slice = &g.ylm_table[pt_idx * NUM_YLM + ylm_base..pt_idx * NUM_YLM + ylm_base + n_m];

        for m_idx in 0..n_m {
            let y_w = ylm_slice[m_idx] * w_ang;
            if y_w == 0.0 {
                continue;
            }
            for p in 0..=la {
                let val = phi_p[p] * y_w;
                if val == 0.0 {
                    continue;
                }
                let p_max_lam = p + l;
                let mut lam = p_max_lam % 2;
                let row_off = m_idx * stride_m + p * stride_p;
                while lam <= p_max_lam {
                    omega[row_off + lam] += val * leg[lam];
                    lam += 2;
                }
            }
        }
    }

    for m_idx in 0..n_m {
        for p in 0..=la {
            let p_max_lam = p + l;
            let mut lam = p_max_lam % 2;
            let row_off = m_idx * stride_m + p * stride_p;
            while lam <= p_max_lam {
                omega[row_off + lam] *= (2 * lam + 1) as f64;
                lam += 2;
            }
        }
    }

    (omega, la, max_lam)
}

#[inline]
fn eval_alm_without_gaussian(
    omega: &[f64],
    la: usize,
    max_lam: usize,
    l: usize,
    r: f64,
    bessel: &[f64],
    out_alm: &mut [f64],
) {
    let n_m = 2 * l + 1;
    let stride_p = max_lam + 1;
    let stride_m = (la + 1) * stride_p;

    let mut r_pows = [1.0f64; 9];
    for p in 1..=la {
        r_pows[p] = r_pows[p - 1] * r;
    }

    for (m_idx, slot) in out_alm.iter_mut().enumerate().take(n_m) {
        let mut acc = 0.0;
        for p in 0..=la {
            let p_max_lam = p + l;
            let mut lam = p_max_lam % 2;
            let row_off = m_idx * stride_m + p * stride_p;
            let mut s = 0.0;
            while lam <= p_max_lam {
                s += omega[row_off + lam] * bessel[lam];
                lam += 2;
            }
            acc += r_pows[p] * s;
        }
        *slot = acc;
    }
}

/// Evaluates the unnormalized primitive Type-2 semilocal projector integral
/// `<phi_a(r - A) | (\sum_k d_k |r - C|^{n_k - 2} exp(-zeta_k |r - C|^2)) \hat{P}_l(\hat{r}_C) | phi_b(r - B)>`.
#[allow(clippy::too_many_arguments)]
pub fn ecp_semilocal_primitive(
    a: [u8; 3],
    b: [u8; 3],
    alpha: f64,
    r_a: [f64; 3],
    beta: f64,
    r_b: [f64; 3],
    r_c: [f64; 3],
    l: u8,
    terms: &[EcpTerm],
) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let l_usize = l as usize;
    let ca = [r_c[0] - r_a[0], r_c[1] - r_a[1], r_c[2] - r_a[2]];
    let cb = [r_c[0] - r_b[0], r_c[1] - r_b[1], r_c[2] - r_b[2]];
    let (ra_dist, ka_hat) = unit_neg_dir(ca);
    let (rb_dist, kb_hat) = unit_neg_dir(cb);

    // Fast radial overlap screening before angular precomputation:
    let min_zeta = terms
        .iter()
        .map(|t| t.exponent)
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let q_screen = (alpha * beta * (ra_dist - rb_dist).powi(2)
        + min_zeta * (alpha * ra_dist * ra_dist + beta * rb_dist * rb_dist))
        / (alpha + beta + min_zeta);
    if q_screen > 85.0 {
        return 0.0;
    }

    // One-center angular selection rule shortcut: if A == C and l > |a|, A_{lm}(r) == 0!
    let la_tot = (a[0] + a[1] + a[2]) as usize;
    let lb_tot = (b[0] + b[1] + b[2]) as usize;
    if ra_dist < 1e-15 && (l_usize > la_tot || (la_tot - l_usize) % 2 != 0) {
        return 0.0;
    }
    if rb_dist < 1e-15 && (l_usize > lb_tot || (lb_tot - l_usize) % 2 != 0) {
        return 0.0;
    }

    let (omega_a, la, max_lam_a) = precompute_semilocal_center_omega(a, ca, ka_hat, l_usize);
    let (omega_b, lb, max_lam_b) = precompute_semilocal_center_omega(b, cb, kb_hat, l_usize);
    let n_m = 2 * l_usize + 1;

    let g = grids();
    let mut bessel_a = [0.0f64; MAX_SPH_L + 1];
    let mut bessel_b = [0.0f64; MAX_SPH_L + 1];
    let mut alm_a = [0.0f64; 2 * MAX_SPH_L + 1];
    let mut alm_b = [0.0f64; 2 * MAX_SPH_L + 1];

    let mut total = 0.0;
    for term in terms {
        let zeta = term.exponent;
        let p_tot = alpha + beta + zeta;
        let q_min = (alpha * beta * (ra_dist - rb_dist).powi(2)
            + zeta * (alpha * ra_dist * ra_dist + beta * rb_dist * rb_dist))
            / p_tot;
        if q_min > 85.0 {
            continue;
        }
        let r0 = (alpha * ra_dist + beta * rb_dist) / p_tot;
        let sigma = (1.0 / p_tot).sqrt();
        let n_pow = (term.r_power as usize) + la + lb;
        let r_peak = 0.5 * (r0 + (r0 * r0 + 2.0 * (n_pow as f64) / p_tot).sqrt());
        let r_lo = (r0 - 10.0 * sigma).max(0.0);
        let r_hi = r_peak + 10.0 * sigma;
        let half_len = 0.5 * (r_hi - r_lo);
        let mid = 0.5 * (r_hi + r_lo);

        let mut rad_sum = 0.0;
        for (&x, &w_rad) in g.rad_nodes.iter().zip(g.rad_weights.iter()) {
            let r = mid + half_len * x;
            let exp_env = (-zeta * r * r
                - alpha * (r - ra_dist) * (r - ra_dist)
                - beta * (r - rb_dist) * (r - rb_dist))
                .exp();
            if exp_env < 1e-40 {
                continue;
            }

            scaled_mod_sph_bessel_batch(max_lam_a, 2.0 * alpha * ra_dist * r, &mut bessel_a);
            scaled_mod_sph_bessel_batch(max_lam_b, 2.0 * beta * rb_dist * r, &mut bessel_b);

            eval_alm_without_gaussian(&omega_a, la, max_lam_a, l_usize, r, &bessel_a, &mut alm_a);
            eval_alm_without_gaussian(&omega_b, lb, max_lam_b, l_usize, r, &bessel_b, &mut alm_b);

            let mut dot_m = 0.0;
            for m_idx in 0..n_m {
                dot_m += alm_a[m_idx] * alm_b[m_idx];
            }
            rad_sum += w_rad * r.powi(term.r_power as i32) * exp_env * dot_m;
        }

        total += term.coeff * half_len * rad_sum;
    }

    total
}

/// Evaluates the full unnormalized primitive ECP matrix element `<phi_a | U_C | phi_b>`
/// (Type-1 local + all Type-2 semilocal channels) for a single ECP atom `ecp`.
pub fn ecp_atom_primitive(
    a: [u8; 3],
    b: [u8; 3],
    alpha: f64,
    r_a: [f64; 3],
    beta: f64,
    r_b: [f64; 3],
    ecp: &AtomEcp,
) -> f64 {
    let mut val = ecp_local_primitive(a, b, alpha, r_a, beta, r_b, ecp.center, &ecp.local_terms);
    for ch in &ecp.semilocal_channels {
        val += ecp_semilocal_primitive(
            a,
            b,
            alpha,
            r_a,
            beta,
            r_b,
            ecp.center,
            ch.angular_momentum,
            &ch.terms,
        );
    }
    val
}

/// Evaluates the contracted shell block `<sh_a | U_ECP | sh_b>` summed over all ECP centers.
pub fn shell_ecp_block(sh_a: &Shell, sh_b: &Shell, ecps: &[AtomEcp]) -> Matrix2D {
    let comps_a = sh_a.components();
    let comps_b = sh_b.components();
    let mut block = Matrix2D::zeros(comps_a.len(), comps_b.len());
    if ecps.is_empty() {
        return block;
    }

    for (ia, &ca) in comps_a.iter().enumerate() {
        for (ib, &cb) in comps_b.iter().enumerate() {
            let mut sum = 0.0;
            for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
                let norm_a = sh_a.primitive_norm_for(pa_idx, ca);
                for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
                    let norm_b = sh_b.primitive_norm_for(pb_idx, cb);
                    let weight = pa.coeff * pb.coeff * norm_a * norm_b;
                    for ecp in ecps {
                        sum += weight
                            * ecp_atom_primitive(
                                ca,
                                cb,
                                pa.exponent,
                                sh_a.center,
                                pb.exponent,
                                sh_b.center,
                                ecp,
                            );
                    }
                }
            }
            block.set(ia, ib, sum);
        }
    }
    block
}

/// Evaluates the full symmetric AO scalar ECP matrix `V_ECP` (`n_ao x n_ao`).
pub fn compute_ecp(basis: &BasisSet) -> Matrix2D {
    let n = basis.n_ao;
    let mut v_ecp = Matrix2D::zeros(n, n);
    if !basis.has_ecp() {
        return v_ecp;
    }

    let n_sh = basis.shells.len();
    for i in 0..n_sh {
        let sh_a = &basis.shells[i];
        for j in i..n_sh {
            let sh_b = &basis.shells[j];
            let block = shell_ecp_block(sh_a, sh_b, &basis.ecp_potentials);
            for ia in 0..block.rows {
                let mu = sh_a.ao_offset + ia;
                for ib in 0..block.cols {
                    let nu = sh_b.ao_offset + ib;
                    let val = block.get(ia, ib);
                    v_ecp.set(mu, nu, val);
                    if i != j {
                        v_ecp.set(nu, mu, val);
                    }
                }
            }
        }
    }
    v_ecp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boys::boys_batch;
    use crate::primitive::nuclear_vrr;

    #[test]
    fn test_spherical_harmonics_orthonormality_and_addition_theorem() {
        let g = grids();
        for l1 in 0..=6 {
            for m1 in 0..(2 * l1 + 1) {
                for l2 in 0..=6 {
                    for m2 in 0..(2 * l2 + 1) {
                        let mut sum = 0.0;
                        for (pt, &w) in g.ang_weights.iter().enumerate() {
                            let y1 = g.ylm_table[pt * NUM_YLM + l1 * l1 + m1];
                            let y2 = g.ylm_table[pt * NUM_YLM + l2 * l2 + m2];
                            sum += w * y1 * y2;
                        }
                        let expected = if l1 == l2 && m1 == m2 { 1.0 } else { 0.0 };
                        assert!(
                            (sum - expected).abs() < 1e-14,
                            "Ylm ortho failed ({l1},{m1}) vs ({l2},{m2}): got {sum}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_bessel_continuity_at_crossover() {
        let mut lo = [0.0; 9];
        let mut hi = [0.0; 9];
        scaled_mod_sph_bessel_batch(8, 14.0 - 1e-12, &mut lo);
        scaled_mod_sph_bessel_batch(8, 14.0 + 1e-12, &mut hi);
        for l in 0..=8 {
            assert!(
                (lo[l] - hi[l]).abs() < 1e-11,
                "Bessel discontinuity at l={l}: {} vs {}",
                lo[l],
                hi[l]
            );
        }
    }

    #[test]
    fn test_3center_local_coulomb_matches_boys_nuclear_vrr() {
        let r_a = [0.25, -0.35, 0.55];
        let r_b = [-0.45, 0.60, -0.15];
        let r_c = [0.70, 0.15, -0.80];
        let alpha = 1.35;
        let beta = 0.85;

        let pair = GaussianPair::new(alpha, r_a, beta, r_b);
        let pc2 = (pair.p[0] - r_c[0]).powi(2)
            + (pair.p[1] - r_c[1]).powi(2)
            + (pair.p[2] - r_c[2]).powi(2);
        let mut boys = [0.0; 10];
        boys_batch(8, pair.gamma * pc2, &mut boys);

        // r_power = 1 -> r^{1-2} = 1/r, exponent = 0.0, coeff = 1.0
        let terms = [EcpTerm {
            r_power: 1,
            exponent: 0.0,
            coeff: 1.0,
        }];

        let cases: &[([u8; 3], [u8; 3])] = &[
            ([0, 0, 0], [0, 0, 0]),
            ([1, 0, 0], [0, 1, 0]),
            ([2, 0, 0], [1, 1, 0]),
            ([1, 0, 1], [0, 2, 0]),
            ([2, 1, 0], [0, 1, 1]),
        ];

        for &(ca, cb) in cases {
            let exact = nuclear_vrr(ca, cb, 0, &pair, r_c, &boys);
            let ecp_val = ecp_local_primitive(ca, cb, alpha, r_a, beta, r_b, r_c, &terms);
            assert!(
                (ecp_val - exact).abs() < 1e-12,
                "3-center 1/r mismatch for {:?}-{:?}: ecp={} vs boys={}",
                ca,
                cb,
                ecp_val,
                exact
            );
        }
    }

    #[test]
    fn test_semilocal_completeness_sum_equals_local_3center() {
        let r_a = [0.2, -0.1, 0.4];
        let r_b = [-0.3, 0.25, -0.2];
        let r_c = [0.1, 0.05, -0.15];
        let alpha = 0.9;
        let beta = 1.1;

        for r_power in [0u8, 1, 2] {
            let terms = [EcpTerm {
                r_power,
                exponent: 1.7,
                coeff: 2.5,
            }];
            let ca = [1, 1, 0];
            let cb = [0, 1, 1];
            let loc = ecp_local_primitive(ca, cb, alpha, r_a, beta, r_b, r_c, &terms);
            let mut sum_proj = 0.0;
            for l in 0..=8 {
                sum_proj += ecp_semilocal_primitive(ca, cb, alpha, r_a, beta, r_b, r_c, l, &terms);
            }
            assert!(
                (sum_proj - loc).abs() < 1e-11,
                "Projector completeness failed for n={r_power}: sum_proj={sum_proj} vs loc={loc}"
            );
        }
    }
}
