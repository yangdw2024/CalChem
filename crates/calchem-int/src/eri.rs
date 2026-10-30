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
        boys_batch(max_m, rho * pq2, &mut boys);
        let prefactor = 2.0 * PI.powf(2.5) / (p * q * zeta.sqrt()) * pair_ab.kab * pair_cd.kab;

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
        let zeta = self.zeta;
        let rho = self.rho;
        let pa = self.pair_ab.pa;
        let wp = self.wp;
        let qc = self.qc;
        let wq = self.wq;

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
                    val += (a1[axis] as f64) / (2.0 * p) * (v2_m - (rho / p) * v2_m1);
                }

                if c[axis] > 0 {
                    let mut c1 = c;
                    c1[axis] -= 1;
                    let vc_m1 = self.vrr(a1, c1, m + 1);
                    val += (c[axis] as f64) / (2.0 * zeta) * vc_m1;
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
                    val += (c1[axis] as f64) / (2.0 * q) * (v2_m - (rho / q) * v2_m1);
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
/// angular momenta `a, b, c, d`.
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
    let max_ab = (a[0] + a[1] + a[2] + b[0] + b[1] + b[2]) as usize;
    let max_cd = (c[0] + c[1] + c[2] + d[0] + d[1] + d[2]) as usize;
    let mut data = EriPairData::new(alpha, r_a, beta, r_b, gamma, r_c, delta, r_d, max_ab, max_cd);
    data.hrr_bra(a, b, c, d)
}

/// Evaluates the contracted shell quartet `(sh_a sh_b | sh_c sh_d)` returning a flat buffer
/// of length `na * nb * nc * nd` indexed as `((ia * nb + ib) * nc + ic) * nd + id`.
pub fn eri_shell_quartet(sh_a: &Shell, sh_b: &Shell, sh_c: &Shell, sh_d: &Shell) -> Vec<f64> {
    let comps_a = sh_a.components();
    let comps_b = sh_b.components();
    let comps_c = sh_c.components();
    let comps_d = sh_d.components();
    let na = comps_a.len();
    let nb = comps_b.len();
    let nc = comps_c.len();
    let nd = comps_d.len();

    let max_ab =
        (sh_a.shell_type.angular_momentum() + sh_b.shell_type.angular_momentum()) as usize;
    let max_cd =
        (sh_c.shell_type.angular_momentum() + sh_d.shell_type.angular_momentum()) as usize;

    let mut out = vec![0.0; na * nb * nc * nd];

    for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
        for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
            for (pc_idx, pc) in sh_c.primitives.iter().enumerate() {
                for (pd_idx, pd) in sh_d.primitives.iter().enumerate() {
                    let c_abcd = pa.coeff * pb.coeff * pc.coeff * pd.coeff;
                    let mut rec = EriPairData::new(
                        pa.exponent,
                        sh_a.center,
                        pb.exponent,
                        sh_b.center,
                        pc.exponent,
                        sh_c.center,
                        pd.exponent,
                        sh_d.center,
                        max_ab,
                        max_cd,
                    );

                    for (ia, &ca) in comps_a.iter().enumerate() {
                        let norm_a = sh_a.primitive_norm_for(pa_idx, ca);
                        for (ib, &cb) in comps_b.iter().enumerate() {
                            let norm_ab = norm_a * sh_b.primitive_norm_for(pb_idx, cb);
                            for (ic, &cc) in comps_c.iter().enumerate() {
                                let norm_abc = norm_ab * sh_c.primitive_norm_for(pc_idx, cc);
                                for (id, &cd) in comps_d.iter().enumerate() {
                                    let weight =
                                        c_abcd * norm_abc * sh_d.primitive_norm_for(pd_idx, cd);
                                    let idx = ((ia * nb + ib) * nc + ic) * nd + id;
                                    out[idx] += weight * rec.hrr_bra(ca, cb, cc, cd);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    out
}
