use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct GaussianPair {
    pub gamma: f64,
    pub mu: f64,
    pub rab2: f64,
    pub kab: f64,
    pub p: [f64; 3],
    pub pa: [f64; 3],
    pub pb: [f64; 3],
    pub ab: [f64; 3],
}

impl GaussianPair {
    pub fn new(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3]) -> Self {
        let gamma = alpha + beta;
        let mu = alpha * beta / gamma;
        let ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let rab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let kab = (-mu * rab2).exp();
        let p = [
            (alpha * a[0] + beta * b[0]) / gamma,
            (alpha * a[1] + beta * b[1]) / gamma,
            (alpha * a[2] + beta * b[2]) / gamma,
        ];
        let pa = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
        let pb = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
        Self {
            gamma,
            mu,
            rab2,
            kab,
            p,
            pa,
            pb,
            ab,
        }
    }
}

/// 1D Obara-Saika overlap integral `<la | lb>` without the exponential prefactor `K_AB`.
/// Includes the `(pi / gamma)^{1/2}` base factor so that `S_3D = K_AB * S_x * S_y * S_z`.
pub fn overlap_1d(la: i32, lb: i32, pa: f64, pb: f64, gamma: f64) -> f64 {
    if la < 0 || lb < 0 {
        return 0.0;
    }
    if la == 0 && lb == 0 {
        return (PI / gamma).sqrt();
    }
    let inv_2g = 0.5 / gamma;
    if la > 0 {
        pa * overlap_1d(la - 1, lb, pa, pb, gamma)
            + ((la - 1) as f64) * inv_2g * overlap_1d(la - 2, lb, pa, pb, gamma)
            + (lb as f64) * inv_2g * overlap_1d(la - 1, lb - 1, pa, pb, gamma)
    } else {
        pb * overlap_1d(la, lb - 1, pa, pb, gamma)
            + (la as f64) * inv_2g * overlap_1d(la - 1, lb - 1, pa, pb, gamma)
            + ((lb - 1) as f64) * inv_2g * overlap_1d(la, lb - 2, pa, pb, gamma)
    }
}

/// 3D unnormalized Cartesian GTO primitive overlap integral `<a | b>`.
pub fn overlap_primitive(a: [i32; 3], b: [i32; 3], pair: &GaussianPair) -> f64 {
    if a.iter().any(|&v| v < 0) || b.iter().any(|&v| v < 0) {
        return 0.0;
    }
    let sx = overlap_1d(a[0], b[0], pair.pa[0], pair.pb[0], pair.gamma);
    let sy = overlap_1d(a[1], b[1], pair.pa[1], pair.pb[1], pair.gamma);
    let sz = overlap_1d(a[2], b[2], pair.pa[2], pair.pb[2], pair.gamma);
    pair.kab * sx * sy * sz
}

/// 3D unnormalized Cartesian GTO primitive kinetic integral `<a | -1/2 \nabla^2 | b>`.
///
/// Uses the exact second-derivative identity on the ket primitive:
/// `T_i(a_i, b_i) = -1/2 [ b_i(b_i - 1) S_i(a_i, b_i - 2) - 2\beta(2 b_i + 1) S_i(a_i, b_i) + 4\beta^2 S_i(a_i, b_i + 2) ]`
/// `T(a, b) = K_AB * (T_x S_y S_z + S_x T_y S_z + S_x S_y T_z)`.
pub fn kinetic_primitive(a: [i32; 3], b: [i32; 3], beta: f64, pair: &GaussianPair) -> f64 {
    let sx = overlap_1d(a[0], b[0], pair.pa[0], pair.pb[0], pair.gamma);
    let sy = overlap_1d(a[1], b[1], pair.pa[1], pair.pb[1], pair.gamma);
    let sz = overlap_1d(a[2], b[2], pair.pa[2], pair.pb[2], pair.gamma);

    let kinetic_1d = |axis: usize| -> f64 {
        let ai = a[axis];
        let bi = b[axis];
        let pa = pair.pa[axis];
        let pb = pair.pb[axis];
        let s_m2 = overlap_1d(ai, bi - 2, pa, pb, pair.gamma);
        let s_0 = overlap_1d(ai, bi, pa, pb, pair.gamma);
        let s_p2 = overlap_1d(ai, bi + 2, pa, pb, pair.gamma);

        let term1 = (bi * (bi - 1)) as f64 * s_m2;
        let term2 = -2.0 * beta * (2 * bi + 1) as f64 * s_0;
        let term3 = 4.0 * beta * beta * s_p2;
        -0.5 * (term1 + term2 + term3)
    };

    let tx = kinetic_1d(0);
    let ty = kinetic_1d(1);
    let tz = kinetic_1d(2);

    pair.kab * (tx * sy * sz + sx * ty * sz + sx * sy * tz)
}

/// Obara-Saika Vertical Recurrence Relation (VRR) for the auxiliary nuclear attraction integral
/// `<a | 1 / |r - C| | b>^{(m)}` (positive integral before multiplying by `-Z_C`).
pub fn nuclear_vrr(
    a: [u8; 3],
    b: [u8; 3],
    m: usize,
    pair: &GaussianPair,
    c: [f64; 3],
    boys_vals: &[f64],
) -> f64 {
    if a == [0, 0, 0] && b == [0, 0, 0] {
        return (2.0 * PI / pair.gamma) * pair.kab * boys_vals[m];
    }

    let pc = [pair.p[0] - c[0], pair.p[1] - c[1], pair.p[2] - c[2]];
    let inv_2g = 0.5 / pair.gamma;

    for axis in 0..3 {
        if a[axis] > 0 {
            let mut a1 = a;
            a1[axis] -= 1;

            let v_m = nuclear_vrr(a1, b, m, pair, c, boys_vals);
            let v_m1 = nuclear_vrr(a1, b, m + 1, pair, c, boys_vals);
            let mut val = pair.pa[axis] * v_m - pc[axis] * v_m1;

            if a1[axis] > 0 {
                let mut a2 = a1;
                a2[axis] -= 1;
                let v2_m = nuclear_vrr(a2, b, m, pair, c, boys_vals);
                let v2_m1 = nuclear_vrr(a2, b, m + 1, pair, c, boys_vals);
                val += (a1[axis] as f64) * inv_2g * (v2_m - v2_m1);
            }

            if b[axis] > 0 {
                let mut b1 = b;
                b1[axis] -= 1;
                let vb_m = nuclear_vrr(a1, b1, m, pair, c, boys_vals);
                let vb_m1 = nuclear_vrr(a1, b1, m + 1, pair, c, boys_vals);
                val += (b[axis] as f64) * inv_2g * (vb_m - vb_m1);
            }

            return val;
        }
    }

    for axis in 0..3 {
        if b[axis] > 0 {
            let mut b1 = b;
            b1[axis] -= 1;

            let v_m = nuclear_vrr(a, b1, m, pair, c, boys_vals);
            let v_m1 = nuclear_vrr(a, b1, m + 1, pair, c, boys_vals);
            let mut val = pair.pb[axis] * v_m - pc[axis] * v_m1;

            if b1[axis] > 0 {
                let mut b2 = b1;
                b2[axis] -= 1;
                let v2_m = nuclear_vrr(a, b2, m, pair, c, boys_vals);
                let v2_m1 = nuclear_vrr(a, b2, m + 1, pair, c, boys_vals);
                val += (b1[axis] as f64) * inv_2g * (v2_m - v2_m1);
            }

            return val;
        }
    }

    unreachable!()
}
