use crate::boys::boys_batch;
use crate::primitive::GaussianPair;
use calchem_basis::Shell;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct EriPairData {
    pub pair_ab: GaussianPair,
    pub pair_cd: GaussianPair,
    pub p: f64,
    pub q: f64,
    pub zeta: f64,
    pub rho: f64,
    pub w: [f64; 3],
    pub wp: [f64; 3],
    pub wq: [f64; 3],
    pub qc: [f64; 3],
    pub qd: [f64; 3],
    pub cd: [f64; 3],
    pub prefactor: f64,
    pub boys: Vec<f64>,
    pub rho_over_p: f64,
    pub rho_over_q: f64,
    pub inv_2zeta: f64,
    max_ab: usize,
    max_cd: usize,
    max_m: usize,
    vrr_cache: Vec<f64>,
}

impl EriPairData {
    pub fn new(
        alpha: f64,
        a: [f64; 3],
        beta: f64,
        b: [f64; 3],
        gamma: f64,
        c: [f64; 3],
        delta: f64,
        d: [f64; 3],
        max_ab: usize,
        max_cd: usize,
    ) -> Self {
        Self::new_with_omega(alpha, a, beta, b, gamma, c, delta, d, max_ab, max_cd, None)
    }

    pub fn new_with_omega(
        alpha: f64,
        a: [f64; 3],
        beta: f64,
        b: [f64; 3],
        gamma: f64,
        c: [f64; 3],
        delta: f64,
        d: [f64; 3],
        max_ab: usize,
        max_cd: usize,
        omega: Option<f64>,
    ) -> Self {
        let pair_ab = GaussianPair::new(alpha, a, beta, b);
        let pair_cd = GaussianPair::new(gamma, c, delta, d);
        let p = pair_ab.gamma;
        let q = pair_cd.gamma;
        let zeta = p + q;
        let rho = p * q / zeta;
        let w = [
            (p * pair_ab.p[0] + q * pair_cd.p[0]) / zeta,
            (p * pair_ab.p[1] + q * pair_cd.p[1]) / zeta,
            (p * pair_ab.p[2] + q * pair_cd.p[2]) / zeta,
        ];
        let wp = [
            w[0] - pair_ab.p[0],
            w[1] - pair_ab.p[1],
            w[2] - pair_ab.p[2],
        ];
        let wq = [
            w[0] - pair_cd.p[0],
            w[1] - pair_cd.p[1],
            w[2] - pair_cd.p[2],
        ];
        let pq2 = (pair_ab.p[0] - pair_cd.p[0]).powi(2)
            + (pair_ab.p[1] - pair_cd.p[1]).powi(2)
            + (pair_ab.p[2] - pair_cd.p[2]).powi(2);
        let max_m = max_ab + max_cd;
        let mut boys = vec![0.0; max_m + 1];

        let (t_arg, theta) = match omega {
            Some(w_val) => {
                assert!(
                    w_val.is_finite() && w_val >= 0.0,
                    "omega must be finite and nonnegative"
                );
                // hypot avoids overflow at large omega; omega=0 gives the zero operator.
                let theta = w_val / w_val.hypot(rho.sqrt());
                let theta_sq = theta * theta;
                (rho * theta_sq * pq2, Some((theta, theta_sq)))
            }
            None => (rho * pq2, None),
        };

        boys_batch(max_m, t_arg, &mut boys);
        let prefactor = 2.0 * PI.powf(2.5) / (p * q * zeta.sqrt()) * pair_ab.kab * pair_cd.kab;

        let (rho_over_p, rho_over_q, inv_2zeta) = if let Some((theta, theta_sq)) = theta {
            let mut curr_theta = theta;
            let theta2 = theta_sq;
            for m in 0..=max_m {
                boys[m] *= curr_theta;
                curr_theta *= theta2;
            }
            (rho / p, rho / q, 1.0 / (2.0 * zeta))
        } else {
            (rho / p, rho / q, 1.0 / (2.0 * zeta))
        };

        let da = max_ab + 1;
        let dc = max_cd + 1;
        let dm = max_m + 1;
        let cache_len = if max_m == 0 {
            0
        } else {
            da * da * da * dc * dc * dc * dm
        };

        Self {
            pair_ab,
            qc: pair_cd.pa,
            qd: pair_cd.pb,
            cd: pair_cd.ab,
            pair_cd,
            p,
            q,
            zeta,
            rho,
            w,
            wp,
            wq,
            prefactor,
            boys,
            rho_over_p,
            rho_over_q,
            inv_2zeta,
            max_ab,
            max_cd,
            max_m,
            vrr_cache: vec![f64::NAN; cache_len],
        }
    }

    #[inline]
    fn vrr_index(&self, a: [u8; 3], c: [u8; 3], m: usize) -> usize {
        let da = self.max_ab + 1;
        let dc = self.max_cd + 1;
        let dm = self.max_m + 1;
        let ia = ((a[0] as usize) * da + (a[1] as usize)) * da + (a[2] as usize);
        let ic = ((c[0] as usize) * dc + (c[1] as usize)) * dc + (c[2] as usize);
        (ia * (dc * dc * dc) + ic) * dm + m
    }

    /// Obara-Saika Vertical Recurrence Relation (VRR) for `[a 0 | c 0]^{(m)}` with flat cache.
    pub fn vrr(&mut self, a: [u8; 3], c: [u8; 3], m: usize) -> f64 {
        if a == [0, 0, 0] && c == [0, 0, 0] {
            return self.prefactor * self.boys[m];
        }

        let cache_idx = self.vrr_index(a, c, m);
        let cached = self.vrr_cache[cache_idx];
        if !cached.is_nan() {
            return cached;
        }

        let p = self.p;
        let q = self.q;
        let pa = self.pair_ab.pa;
        let wp = self.wp;
        let qc = self.qc;
        let wq = self.wq;
        let rho_over_p = self.rho_over_p;
        let rho_over_q = self.rho_over_q;
        let inv_2zeta = self.inv_2zeta;

        for axis in 0..3 {
            if a[axis] > 0 {
                let mut a1 = a;
                a1[axis] -= 1;

                let v_m = self.vrr(a1, c, m);
                let v_m1 = self.vrr(a1, c, m + 1);
                let mut val = pa[axis] * v_m + wp[axis] * v_m1;

                if a1[axis] > 0 {
                    let mut a2 = a1;
                    a2[axis] -= 1;
                    let v2_m = self.vrr(a2, c, m);
                    let v2_m1 = self.vrr(a2, c, m + 1);
                    val += (a1[axis] as f64) / (2.0 * p) * (v2_m - rho_over_p * v2_m1);
                }

                if c[axis] > 0 {
                    let mut c1 = c;
                    c1[axis] -= 1;
                    let vc_m1 = self.vrr(a1, c1, m + 1);
                    val += (c[axis] as f64) * inv_2zeta * vc_m1;
                }

                self.vrr_cache[cache_idx] = val;
                return val;
            }
        }

        for axis in 0..3 {
            if c[axis] > 0 {
                let mut c1 = c;
                c1[axis] -= 1;

                let v_m = self.vrr(a, c1, m);
                let v_m1 = self.vrr(a, c1, m + 1);
                let mut val = qc[axis] * v_m + wq[axis] * v_m1;

                if c1[axis] > 0 {
                    let mut c2 = c1;
                    c2[axis] -= 1;
                    let v2_m = self.vrr(a, c2, m);
                    let v2_m1 = self.vrr(a, c2, m + 1);
                    val += (c1[axis] as f64) / (2.0 * q) * (v2_m - rho_over_q * v2_m1);
                }

                self.vrr_cache[cache_idx] = val;
                return val;
            }
        }

        unreachable!()
    }

    /// Obara-Saika Horizontal Recurrence Relation (HRR) on ket centers `(c, d)`:
    /// `[a 0 | c d] = [a 0 | c + 1_i, d - 1_i] + (C_i - D_i) [a 0 | c, d - 1_i]`.
    pub fn hrr_ket(&mut self, a: [u8; 3], c: [u8; 3], d: [u8; 3]) -> f64 {
        for axis in 0..3 {
            if d[axis] > 0 {
                let mut d1 = d;
                d1[axis] -= 1;
                let mut c_plus = c;
                c_plus[axis] += 1;
                let cd_axis = self.cd[axis];
                let t1 = self.hrr_ket(a, c_plus, d1);
                let t2 = self.hrr_ket(a, c, d1);
                return t1 + cd_axis * t2;
            }
        }
        self.vrr(a, c, 0)
    }

    /// Obara-Saika Horizontal Recurrence Relation (HRR) on bra centers `(a, b)`:
    /// `[a b | c d] = [a + 1_i, b - 1_i | c d] + (A_i - B_i) [a, b - 1_i | c d]`.
    pub fn hrr_bra(&mut self, a: [u8; 3], b: [u8; 3], c: [u8; 3], d: [u8; 3]) -> f64 {
        for axis in 0..3 {
            if b[axis] > 0 {
                let mut b1 = b;
                b1[axis] -= 1;
                let mut a_plus = a;
                a_plus[axis] += 1;
                let ab_axis = self.pair_ab.ab[axis];
                let t1 = self.hrr_bra(a_plus, b1, c, d);
                let t2 = self.hrr_bra(a, b1, c, d);
                return t1 + ab_axis * t2;
            }
        }
        self.hrr_ket(a, c, d)
    }
}

/// Evaluates a single unnormalized primitive 4-center ERI `(ab|cd)` for general Cartesian
/// angular momenta `a, b, c, d`, with optional range-separation attenuation parameter `omega`.
#[allow(clippy::too_many_arguments)]
pub fn eri_primitive_with_omega(
    a: [u8; 3],
    b: [u8; 3],
    c: [u8; 3],
    d: [u8; 3],
    alpha: f64,
    r_a: [f64; 3],
    beta: f64,
    r_b: [f64; 3],
    gamma: f64,
    r_c: [f64; 3],
    delta: f64,
    r_d: [f64; 3],
    omega: Option<f64>,
) -> f64 {
    let max_ab = (a[0] + a[1] + a[2] + b[0] + b[1] + b[2]) as usize;
    let max_cd = (c[0] + c[1] + c[2] + d[0] + d[1] + d[2]) as usize;
    let mut data = EriPairData::new_with_omega(
        alpha, r_a, beta, r_b, gamma, r_c, delta, r_d, max_ab, max_cd, omega,
    );
    data.hrr_bra(a, b, c, d)
}

#[allow(clippy::too_many_arguments)]
pub fn eri_primitive(
    a: [u8; 3],
    b: [u8; 3],
    c: [u8; 3],
    d: [u8; 3],
    alpha: f64,
    r_a: [f64; 3],
    beta: f64,
    r_b: [f64; 3],
    gamma: f64,
    r_c: [f64; 3],
    delta: f64,
    r_d: [f64; 3],
) -> f64 {
    eri_primitive_with_omega(
        a, b, c, d, alpha, r_a, beta, r_b, gamma, r_c, delta, r_d, None,
    )
}

/// Stack-allocated primitive pair with contraction * leading radial norm weight.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EgpPair {
    pub(crate) pair: GaussianPair,
    pub(crate) w: f64,
}

pub(crate) fn collect_egp_pairs(sh_a: &Shell, sh_b: &Shell) -> Vec<EgpPair> {
    let mut pairs = Vec::with_capacity(sh_a.primitives.len() * sh_b.primitives.len());
    for pa in &sh_a.primitives {
        let wa = pa.coeff * pa.norm;
        for pb in &sh_b.primitives {
            let pair = GaussianPair::new(pa.exponent, sh_a.center, pb.exponent, sh_b.center);
            if pair.kab >= 1e-18 {
                pairs.push(EgpPair {
                    pair,
                    w: wa * pb.coeff * pb.norm,
                });
            }
        }
    }
    pairs
}

#[inline]
fn shell_comp_scales(sh: &Shell) -> [f64; 10] {
    let mut scales = [1.0f64; 10];
    if let Some(p0) = sh.primitives.first() {
        if p0.norm > 0.0 {
            for (i, &c) in sh.components().iter().enumerate() {
                scales[i] = sh.primitive_norm_for(0, c) / p0.norm;
            }
        }
    }
    scales
}

#[allow(clippy::too_many_arguments)]
fn stack_vrr(
    a: [u8; 3],
    c: [u8; 3],
    m: usize,
    da: usize,
    dc: usize,
    dm: usize,
    pa: &[f64; 3],
    qc: &[f64; 3],
    wp: &[f64; 3],
    wq: &[f64; 3],
    p_exp: f64,
    q_exp: f64,
    rho_over_p: f64,
    rho_over_q: f64,
    inv_2zeta: f64,
    prefactor: f64,
    boys: &[f64],
    cache: &mut [f64],
) -> f64 {
    if a == [0, 0, 0] && c == [0, 0, 0] {
        return prefactor * boys[m];
    }
    let ia = ((a[0] as usize) * da + (a[1] as usize)) * da + (a[2] as usize);
    let ic = ((c[0] as usize) * dc + (c[1] as usize)) * dc + (c[2] as usize);
    let cache_idx = (ia * (dc * dc * dc) + ic) * dm + m;
    let cached = cache[cache_idx];
    if !cached.is_nan() {
        return cached;
    }

    for axis in 0..3 {
        if a[axis] > 0 {
            let mut a1 = a;
            a1[axis] -= 1;
            let v_m = stack_vrr(
                a1, c, m, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p, rho_over_q,
                inv_2zeta, prefactor, boys, cache,
            );
            let v_m1 = stack_vrr(
                a1, c, m + 1, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p, rho_over_q,
                inv_2zeta, prefactor, boys, cache,
            );
            let mut val = pa[axis] * v_m + wp[axis] * v_m1;

            if a1[axis] > 0 {
                let mut a2 = a1;
                a2[axis] -= 1;
                let v2_m = stack_vrr(
                    a2, c, m, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p, rho_over_q,
                    inv_2zeta, prefactor, boys, cache,
                );
                let v2_m1 = stack_vrr(
                    a2, c, m + 1, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p, rho_over_q,
                    inv_2zeta, prefactor, boys, cache,
                );
                val += (a1[axis] as f64) / (2.0 * p_exp) * (v2_m - rho_over_p * v2_m1);
            }

            if c[axis] > 0 {
                let mut c1 = c;
                c1[axis] -= 1;
                let vc_m1 = stack_vrr(
                    a1, c1, m + 1, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p,
                    rho_over_q, inv_2zeta, prefactor, boys, cache,
                );
                val += (c[axis] as f64) * inv_2zeta * vc_m1;
            }

            cache[cache_idx] = val;
            return val;
        }
    }

    for axis in 0..3 {
        if c[axis] > 0 {
            let mut c1 = c;
            c1[axis] -= 1;
            let v_m = stack_vrr(
                a, c1, m, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p, rho_over_q,
                inv_2zeta, prefactor, boys, cache,
            );
            let v_m1 = stack_vrr(
                a, c1, m + 1, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p, rho_over_q,
                inv_2zeta, prefactor, boys, cache,
            );
            let mut val = qc[axis] * v_m + wq[axis] * v_m1;

            if c1[axis] > 0 {
                let mut c2 = c1;
                c2[axis] -= 1;
                let v2_m = stack_vrr(
                    a, c2, m, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p, rho_over_q,
                    inv_2zeta, prefactor, boys, cache,
                );
                let v2_m1 = stack_vrr(
                    a, c2, m + 1, da, dc, dm, pa, qc, wp, wq, p_exp, q_exp, rho_over_p,
                    rho_over_q, inv_2zeta, prefactor, boys, cache,
                );
                val += (c1[axis] as f64) / (2.0 * q_exp) * (v2_m - rho_over_q * v2_m1);
            }

            cache[cache_idx] = val;
            return val;
        }
    }

    unreachable!()
}

#[inline]
fn hrr_ket_contracted(
    a: [u8; 3],
    c: [u8; 3],
    d: [u8; 3],
    da: usize,
    dc: usize,
    cd_vec: &[f64; 3],
    vrr_cont: &[f64],
) -> f64 {
    for axis in 0..3 {
        if d[axis] > 0 {
            let mut d1 = d;
            d1[axis] -= 1;
            let mut c_plus = c;
            c_plus[axis] += 1;
            return hrr_ket_contracted(a, c_plus, d1, da, dc, cd_vec, vrr_cont)
                + cd_vec[axis] * hrr_ket_contracted(a, c, d1, da, dc, cd_vec, vrr_cont);
        }
    }
    let ia = ((a[0] as usize) * da + (a[1] as usize)) * da + (a[2] as usize);
    let ic = ((c[0] as usize) * dc + (c[1] as usize)) * dc + (c[2] as usize);
    vrr_cont[ia * (dc * dc * dc) + ic]
}

#[inline]
fn hrr_bra_contracted(
    a: [u8; 3],
    b: [u8; 3],
    c: [u8; 3],
    d: [u8; 3],
    da: usize,
    dc: usize,
    ab_vec: &[f64; 3],
    cd_vec: &[f64; 3],
    vrr_cont: &[f64],
) -> f64 {
    for axis in 0..3 {
        if b[axis] > 0 {
            let mut b1 = b;
            b1[axis] -= 1;
            let mut a_plus = a;
            a_plus[axis] += 1;
            return hrr_bra_contracted(a_plus, b1, c, d, da, dc, ab_vec, cd_vec, vrr_cont)
                + ab_vec[axis] * hrr_bra_contracted(a, b1, c, d, da, dc, ab_vec, cd_vec, vrr_cont);
        }
    }
    hrr_ket_contracted(a, c, d, da, dc, cd_vec, vrr_cont)
}

/// Evaluates the contracted shell quartet `(sh_a sh_b | sh_c sh_d)` into `out`
/// (resized to `na * nb * nc * nd` in `((ia * nb + ib) * nc + ic) * nd + id` order),
/// using Head-Gordon–Pople early contraction and zero-heap stack VRR for S/P shells.
pub fn eri_shell_quartet_with_omega_into(
    sh_a: &Shell,
    sh_b: &Shell,
    sh_c: &Shell,
    sh_d: &Shell,
    omega: Option<f64>,
    out: &mut Vec<f64>,
) {
    let ab = collect_egp_pairs(sh_a, sh_b);
    let cd = collect_egp_pairs(sh_c, sh_d);
    eri_from_pairs(sh_a, sh_b, sh_c, sh_d, &ab, &cd, omega, out);
}

fn angular_tuples(l: u8, max_l: usize) -> &'static [([u8;3], usize)] {
    type Tuples = Vec<([u8;3],usize)>;
    // Current orbital shells are S through F, hence a combined maximum of 6.
    static TABLE: std::sync::OnceLock<[[Tuples;7];4]> = std::sync::OnceLock::new();
    &TABLE.get_or_init(|| std::array::from_fn(|min| std::array::from_fn(|max| {
        let mut tuples = Vec::new();
        for x in 0..=max { for y in 0..=max-x { for z in 0..=max-x-y {
            if x+y+z >= min {
                tuples.push(([x as u8,y as u8,z as u8],(x*(max+1)+y)*(max+1)+z));
            }
        } } }
        tuples
    })))[l as usize][max_l]
}

pub(crate) fn eri_from_pairs(sh_a: &Shell, sh_b: &Shell, sh_c: &Shell, sh_d: &Shell,
    pairs_ab: &[EgpPair], pairs_cd: &[EgpPair], omega: Option<f64>, out: &mut Vec<f64>) {
    let comps_a = sh_a.components();
    let comps_b = sh_b.components();
    let comps_c = sh_c.components();
    let comps_d = sh_d.components();
    let na = comps_a.len();
    let nb = comps_b.len();
    let nc = comps_c.len();
    let nd = comps_d.len();
    let total_len = na * nb * nc * nd;
    out.resize(total_len, 0.0);
    out.fill(0.0);

    let la = sh_a.shell_type.angular_momentum();
    let lb = sh_b.shell_type.angular_momentum();
    let lc = sh_c.shell_type.angular_momentum();
    let ld = sh_d.shell_type.angular_momentum();
    let max_ab = (la + lb) as usize;
    let max_cd = (lc + ld) as usize;

    if pairs_ab.is_empty() {
        return;
    }
    if pairs_cd.is_empty() {
        return;
    }

    if max_ab == 0 && max_cd == 0 {
        let mut sum = 0.0;
        let two_pi25 = 2.0 * PI.powf(2.5);
        for pab in pairs_ab {
            let p = pab.pair.gamma;
            for pcd in pairs_cd {
                let q = pcd.pair.gamma;
                let zeta = p + q;
                let rho = p * q / zeta;
                let pq2 = (pab.pair.p[0] - pcd.pair.p[0]).powi(2)
                    + (pab.pair.p[1] - pcd.pair.p[1]).powi(2)
                    + (pab.pair.p[2] - pcd.pair.p[2]).powi(2);
                let mut b0 = [0.0f64; 1];
                let val = if let Some(w_val) = omega {
                    let theta = w_val / w_val.hypot(rho.sqrt());
                    boys_batch(0, rho * theta * theta * pq2, &mut b0);
                    b0[0] * theta
                } else {
                    boys_batch(0, rho * pq2, &mut b0);
                    b0[0]
                };
                let prefactor = two_pi25 / (p * q * zeta.sqrt()) * pab.pair.kab * pcd.pair.kab;
                sum += pab.w * pcd.w * prefactor * val;
            }
        }
        out[0] = sum;
        return;
    }

    let da = max_ab + 1;
    let dc = max_cd + 1;
    let max_m = max_ab + max_cd;
    let dm = max_m + 1;
    let da3 = da * da * da;
    let dc3 = dc * dc * dc;
    let cont_len = da3 * dc3;
    let cache_len = cont_len * dm;

    // Contracted [a 0 | c 0]^(0) workspace and primitive VRR scratch:
    // Fixed stack arrays for S/P shells (da <= 3, dc <= 3) -> zero heap allocations.
    let mut stack_cont = [0.0f64; 729];
    let mut heap_cont;
    let vrr_cont: &mut [f64] = if da <= 3 && dc <= 3 {
        &mut stack_cont[..cont_len]
    } else {
        heap_cont = vec![0.0f64; cont_len];
        &mut heap_cont[..]
    };

    let mut boys_buf = [0.0f64; 17];
    let two_pi25 = 2.0 * PI.powf(2.5);

    // Fast zero-recursion, zero-NaN-fill bottom-up Obara-Saika VRR for all S/P quartets
    // (max_ab <= 2 && max_cd <= 2). Eliminates 29 KB stack_cache NaN fills and recursive stack_vrr.
    if max_ab <= 2 && max_cd <= 2 {
        const PAIRS2: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        let a_map: [usize; 10] = [
            0,
            da * da,
            da,
            1,
            2 * da * da,
            da * da + da,
            da * da + 1,
            2 * da,
            da + 1,
            2,
        ];
        let c_map: [usize; 10] = [
            0,
            dc * dc,
            dc,
            1,
            2 * dc * dc,
            dc * dc + dc,
            dc * dc + 1,
            2 * dc,
            dc + 1,
            2,
        ];
        let mut comp_cont = [0.0f64; 100]; // 10 x 10 compact [a_compact * 10 + c_compact]
        let na_comp = match max_ab {
            0 => 1,
            1 => 4,
            _ => 10,
        };
        let nc_comp = match max_cd {
            0 => 1,
            1 => 4,
            _ => 10,
        };

        for pab in pairs_ab {
            let p = pab.pair.gamma;
            let inv_2p = 0.5 / p;
            let pa = pab.pair.pa;
            for pcd in pairs_cd {
                let q = pcd.pair.gamma;
                let inv_2q = 0.5 / q;
                let zeta = p + q;
                let inv_zeta = 1.0 / zeta;
                let inv_2zeta = 0.5 * inv_zeta;
                let rho = p * q * inv_zeta;
                let w0 = (p * pab.pair.p[0] + q * pcd.pair.p[0]) * inv_zeta;
                let w1 = (p * pab.pair.p[1] + q * pcd.pair.p[1]) * inv_zeta;
                let w2 = (p * pab.pair.p[2] + q * pcd.pair.p[2]) * inv_zeta;
                let wp = [w0 - pab.pair.p[0], w1 - pab.pair.p[1], w2 - pab.pair.p[2]];
                let wq = [w0 - pcd.pair.p[0], w1 - pcd.pair.p[1], w2 - pcd.pair.p[2]];
                let dx = pab.pair.p[0] - pcd.pair.p[0];
                let dy = pab.pair.p[1] - pcd.pair.p[1];
                let dz = pab.pair.p[2] - pcd.pair.p[2];
                let pq2 = dx * dx + dy * dy + dz * dz;

                let (t_arg, theta) = match omega {
                    Some(w_val) => {
                        let theta = w_val / w_val.hypot(rho.sqrt());
                        (rho * theta * theta * pq2, Some((theta, theta * theta)))
                    }
                    None => (rho * pq2, None),
                };
                let boys = &mut boys_buf[..=max_m];
                boys_batch(max_m, t_arg, boys);
                if let Some((theta, theta2)) = theta {
                    let mut curr = theta;
                    for bm in boys.iter_mut() {
                        *bm *= curr;
                        curr *= theta2;
                    }
                }

                let scale = two_pi25 / (p * q * zeta.sqrt()) * pab.pair.kab * pcd.pair.kab
                    * (pab.w * pcd.w);
                let rho_over_p = rho / p;
                let rho_over_q = rho / q;
                let qc = pcd.pair.pa;

                let mut g00 = [0.0f64; 5];
                for m in 0..=max_m {
                    g00[m] = scale * boys[m];
                }

                let mut g10 = [[0.0f64; 4]; 3];
                if max_ab >= 1 {
                    let m_lim = max_m - 1;
                    for i in 0..3 {
                        let pai = pa[i];
                        let wpi = wp[i];
                        for m in 0..=m_lim {
                            g10[i][m] = pai * g00[m] + wpi * g00[m + 1];
                        }
                    }
                }

                let mut g20 = [[0.0f64; 3]; 6];
                if max_ab >= 2 {
                    let m_lim = max_m - 2;
                    let mut diag_a = [0.0f64; 3];
                    for m in 0..=m_lim {
                        diag_a[m] = inv_2p * (g00[m] - rho_over_p * g00[m + 1]);
                    }
                    for (idx, &(i, j)) in PAIRS2.iter().enumerate() {
                        let pai = pa[i];
                        let wpi = wp[i];
                        if i == j {
                            for m in 0..=m_lim {
                                g20[idx][m] = pai * g10[j][m] + wpi * g10[j][m + 1] + diag_a[m];
                            }
                        } else {
                            for m in 0..=m_lim {
                                g20[idx][m] = pai * g10[j][m] + wpi * g10[j][m + 1];
                            }
                        }
                    }
                }

                // Accumulate c = 0 (m = 0)
                if lc == 0 {
                    comp_cont[0] += g00[0];
                    if max_ab >= 1 {
                        comp_cont[10] += g10[0][0];
                        comp_cont[20] += g10[1][0];
                        comp_cont[30] += g10[2][0];
                    }
                    if max_ab >= 2 {
                        for idx in 0..6 {
                            comp_cont[(4 + idx) * 10] += g20[idx][0];
                        }
                    }
                }

                if max_cd >= 1 {
                    let m_lim_c1 = max_cd - 1; // 0 or 1
                    let mut g01 = [[0.0f64; 2]; 3];
                    for k in 0..3 {
                        for m in 0..=m_lim_c1 {
                            g01[k][m] = qc[k] * g00[m] + wq[k] * g00[m + 1];
                        }
                    }
                    let mut g11 = [[[0.0f64; 2]; 3]; 3];
                    if max_ab >= 1 {
                        for i in 0..3 {
                            for k in 0..3 {
                                if i == k {
                                    for m in 0..=m_lim_c1 {
                                        g11[i][k][m] = qc[k] * g10[i][m]
                                            + wq[k] * g10[i][m + 1]
                                            + inv_2zeta * g00[m + 1];
                                    }
                                } else {
                                    for m in 0..=m_lim_c1 {
                                        g11[i][k][m] =
                                            qc[k] * g10[i][m] + wq[k] * g10[i][m + 1];
                                    }
                                }
                            }
                        }
                    }
                    let mut g21 = [[[0.0f64; 2]; 3]; 6];
                    if max_ab >= 2 {
                        for (idx, &(i, j)) in PAIRS2.iter().enumerate() {
                            for k in 0..3 {
                                for m in 0..=m_lim_c1 {
                                    let mut v = qc[k] * g20[idx][m] + wq[k] * g20[idx][m + 1];
                                    if i == k {
                                        v += inv_2zeta * g10[j][m + 1];
                                    }
                                    if j == k {
                                        v += inv_2zeta * g10[i][m + 1];
                                    }
                                    g21[idx][k][m] = v;
                                }
                            }
                        }
                    }

                    // Accumulate |c| = 1 (m = 0)
                    for k in 0..3 {
                        comp_cont[1 + k] += g01[k][0];
                    }
                    if max_ab >= 1 {
                        for i in 0..3 {
                            for k in 0..3 {
                                comp_cont[(1 + i) * 10 + (1 + k)] += g11[i][k][0];
                            }
                        }
                    }
                    if max_ab >= 2 {
                        for idx in 0..6 {
                            for k in 0..3 {
                                comp_cont[(4 + idx) * 10 + (1 + k)] += g21[idx][k][0];
                            }
                        }
                    }

                    // Accumulate |c| = 2 (m = 0 only!)
                    if max_cd >= 2 {
                        let diag_c0 = inv_2q * (g00[0] - rho_over_q * g00[1]);
                        for (cd_idx, &(k, l)) in PAIRS2.iter().enumerate() {
                            let c_slot = 4 + cd_idx;
                            let mut v0 = qc[k] * g01[l][0] + wq[k] * g01[l][1];
                            if k == l {
                                v0 += diag_c0;
                            }
                            comp_cont[c_slot] += v0;
                        }
                        if max_ab >= 1 {
                            for i in 0..3 {
                                let diag_c1 = inv_2q * (g10[i][0] - rho_over_q * g10[i][1]);
                                let row_off = (1 + i) * 10;
                                for (cd_idx, &(k, l)) in PAIRS2.iter().enumerate() {
                                    let mut v = qc[k] * g11[i][l][0] + wq[k] * g11[i][l][1];
                                    if k == l {
                                        v += diag_c1;
                                    }
                                    if i == k {
                                        v += inv_2zeta * g01[l][1];
                                    }
                                    comp_cont[row_off + 4 + cd_idx] += v;
                                }
                            }
                        }
                        if max_ab >= 2 {
                            for (idx, &(i, j)) in PAIRS2.iter().enumerate() {
                                let diag_c2 = inv_2q * (g20[idx][0] - rho_over_q * g20[idx][1]);
                                let row_off = (4 + idx) * 10;
                                for (cd_idx, &(k, l)) in PAIRS2.iter().enumerate() {
                                    let mut v = qc[k] * g21[idx][l][0] + wq[k] * g21[idx][l][1];
                                    if k == l {
                                        v += diag_c2;
                                    }
                                    if i == k {
                                        v += inv_2zeta * g11[j][l][1];
                                    }
                                    if j == k {
                                        v += inv_2zeta * g11[i][l][1];
                                    }
                                    comp_cont[row_off + 4 + cd_idx] += v;
                                }
                            }
                        }
                    }
                }
            }
        }

        for a_c in 0..na_comp {
            let ia = a_map[a_c];
            let dst_row = &mut vrr_cont[ia * dc3..(ia + 1) * dc3];
            let src_row = &comp_cont[a_c * 10..a_c * 10 + nc_comp];
            for c_c in 0..nc_comp {
                dst_row[c_map[c_c]] = src_row[c_c];
            }
        }
    } else {
        let mut heap_cache = vec![f64::NAN; cache_len];
        let vrr_cache = &mut heap_cache[..];
        // Precompute needed (a, c) pairs for VRR contraction:
        // la <= |a| <= max_ab, lc <= |c| <= max_cd.
        let a_tuples = angular_tuples(la,max_ab);
        let c_tuples = angular_tuples(lc,max_cd);

        for pab in pairs_ab {
            let p = pab.pair.gamma;
            for pcd in pairs_cd {
                let q = pcd.pair.gamma;
                let zeta = p + q;
                let rho = p * q / zeta;
                let w = [
                    (p * pab.pair.p[0] + q * pcd.pair.p[0]) / zeta,
                    (p * pab.pair.p[1] + q * pcd.pair.p[1]) / zeta,
                    (p * pab.pair.p[2] + q * pcd.pair.p[2]) / zeta,
                ];
                let wp = [w[0] - pab.pair.p[0], w[1] - pab.pair.p[1], w[2] - pab.pair.p[2]];
                let wq = [w[0] - pcd.pair.p[0], w[1] - pcd.pair.p[1], w[2] - pcd.pair.p[2]];
                let pq2 = (pab.pair.p[0] - pcd.pair.p[0]).powi(2)
                    + (pab.pair.p[1] - pcd.pair.p[1]).powi(2)
                    + (pab.pair.p[2] - pcd.pair.p[2]).powi(2);

                let (t_arg, theta) = match omega {
                    Some(w_val) => {
                        let theta = w_val / w_val.hypot(rho.sqrt());
                        (rho * theta * theta * pq2, Some((theta, theta * theta)))
                    }
                    None => (rho * pq2, None),
                };
                let boys = &mut boys_buf[..=max_m];
                boys_batch(max_m, t_arg, boys);
                if let Some((theta, theta2)) = theta {
                    let mut curr = theta;
                    for bm in boys.iter_mut() {
                        *bm *= curr;
                        curr *= theta2;
                    }
                }

                let prefactor = two_pi25 / (p * q * zeta.sqrt()) * pab.pair.kab * pcd.pair.kab;
                let w_abcd = pab.w * pcd.w;
                let rho_over_p = rho / p;
                let rho_over_q = rho / q;
                let inv_2zeta = 1.0 / (2.0 * zeta);

                vrr_cache.fill(f64::NAN);

                for &(a_lmn, ia) in a_tuples {
                    let row = &mut vrr_cont[ia * dc3..(ia + 1) * dc3];
                    for &(c_lmn, ic) in c_tuples {
                        let val = stack_vrr(
                            a_lmn,
                            c_lmn,
                            0,
                            da,
                            dc,
                            dm,
                            &pab.pair.pa,
                            &pcd.pair.pa,
                            &wp,
                            &wq,
                            p,
                            q,
                            rho_over_p,
                            rho_over_q,
                            inv_2zeta,
                            prefactor,
                            boys,
                            vrr_cache,
                        );
                        row[ic] += w_abcd * val;
                    }
                }
            }
        }
    }

    // Apply HRR once to the contracted VRR integrals and scale by Cartesian component norms
    let ab_vec = [
        sh_a.center[0] - sh_b.center[0],
        sh_a.center[1] - sh_b.center[1],
        sh_a.center[2] - sh_b.center[2],
    ];
    let cd_vec = [
        sh_c.center[0] - sh_d.center[0],
        sh_c.center[1] - sh_d.center[1],
        sh_c.center[2] - sh_d.center[2],
    ];
    let scale_a = shell_comp_scales(sh_a);
    let scale_b = shell_comp_scales(sh_b);
    let scale_c = shell_comp_scales(sh_c);
    let scale_d = shell_comp_scales(sh_d);

    for (ia, &ca) in comps_a.iter().enumerate() {
        let sa = scale_a[ia];
        for (ib, &cb) in comps_b.iter().enumerate() {
            let sab = sa * scale_b[ib];
            for (ic, &cc) in comps_c.iter().enumerate() {
                let sabc = sab * scale_c[ic];
                for (id, &cd) in comps_d.iter().enumerate() {
                    let val = hrr_bra_contracted(ca, cb, cc, cd, da, dc, &ab_vec, &cd_vec, vrr_cont);
                    let idx = ((ia * nb + ib) * nc + ic) * nd + id;
                    out[idx] = sabc * scale_d[id] * val;
                }
            }
        }
    }
}

/// Evaluates the contracted shell quartet `(sh_a sh_b | sh_c sh_d)` returning a flat buffer
/// of length `na * nb * nc * nd` indexed as `((ia * nb + ib) * nc + ic) * nd + id`, with optional
/// range-separation parameter `omega`.
pub fn eri_shell_quartet_with_omega(
    sh_a: &Shell,
    sh_b: &Shell,
    sh_c: &Shell,
    sh_d: &Shell,
    omega: Option<f64>,
) -> Vec<f64> {
    let mut out = Vec::new();
    eri_shell_quartet_with_omega_into(sh_a, sh_b, sh_c, sh_d, omega, &mut out);
    out
}

pub fn eri_shell_quartet(sh_a: &Shell, sh_b: &Shell, sh_c: &Shell, sh_d: &Shell) -> Vec<f64> {
    eri_shell_quartet_with_omega(sh_a, sh_b, sh_c, sh_d, None)
}
