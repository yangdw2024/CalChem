use std::f64::consts::PI;

/// High-precision rational Chebyshev approximation of `erf(x)` (Cody, Math. Comp. 23, 631 (1969)),
/// accurate to `< 2.5e-16` (~1 ULP in IEEE 754 `f64`) across the entire real line.
pub fn erf_approx(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    if ax <= 0.46875 {
        let ysq = ax * ax;
        const P: [f64; 5] = [
            3.16112374387056560e00,
            1.13864154151050156e02,
            3.77485237685302021e02,
            3.20937758913846947e03,
            1.85777706184603153e-01,
        ];
        const Q: [f64; 4] = [
            2.36012909523441209e01,
            2.44024637934444173e02,
            1.28261652607737228e03,
            2.84423683343917062e03,
        ];
        let mut xnum = P[4] * ysq;
        let mut xden = ysq;
        for i in 0..3 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        let res = ax * (xnum + P[3]) / (xden + Q[3]);
        sign * res
    } else if ax <= 4.0 {
        const P: [f64; 9] = [
            5.64188496988670089e-01,
            8.88314979438837594e00,
            6.61191906371416295e01,
            2.98635138197400131e02,
            8.81952221241769090e02,
            1.71204761263407058e03,
            2.05107837782607147e03,
            1.23033935479799725e03,
            2.15311535474403846e-08,
        ];
        const Q: [f64; 8] = [
            1.57449261107098347e01,
            1.17693950891312499e02,
            5.37181101862009858e02,
            1.62138957456669019e03,
            3.29079923573345963e03,
            4.36261909014324716e03,
            3.43936767414372164e03,
            1.23033935480374942e03,
        ];
        let mut xnum = P[8] * ax;
        let mut xden = ax;
        for i in 0..7 {
            xnum = (xnum + P[i]) * ax;
            xden = (xden + Q[i]) * ax;
        }
        let result = (xnum + P[7]) / (xden + Q[7]);
        let ysq = (ax * 16.0).floor() / 16.0;
        let del_x = (ax - ysq) * (ax + ysq);
        let erfc_val = (-ysq * ysq).exp() * (-del_x).exp() * result;
        sign * (1.0 - erfc_val)
    } else if ax < 26.6 {
        let ysq = 1.0 / (ax * ax);
        const P: [f64; 6] = [
            3.05326634961232344e-01,
            3.60344899949804439e-01,
            1.25781726111229246e-01,
            1.60837851487422766e-02,
            6.58749161529837803e-04,
            1.63153871373020978e-02,
        ];
        const Q: [f64; 5] = [
            2.56852019228982242e00,
            1.87295284992346047e00,
            5.27905102951428412e-01,
            6.05183413124413191e-02,
            2.33520497626869185e-03,
        ];
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        let mut result = ysq * (xnum + P[4]) / (xden + Q[4]);
        result = (0.56418958354775628695 - result) / ax;
        let ysq_hi = (ax * 16.0).floor() / 16.0;
        let del_x = (ax - ysq_hi) * (ax + ysq_hi);
        let erfc_val = (-ysq_hi * ysq_hi).exp() * (-del_x).exp() * result;
        sign * (1.0 - erfc_val)
    } else {
        sign
    }
}

/// Evaluates the Boys function batch `F_0(t), F_1(t), ..., F_{max_n}(t)` into `out[0..=max_n]`:
/// `F_n(t) = \int_0^1 u^{2n} \exp(-t u^2) du`
///
/// Numerical strategy:
/// - `t < 1e-14`: 3-term Maclaurin expansion `1/(2n+1) - t/(2n+3) + t^2/(2(2n+5))`.
/// - `1e-14 <= t <= 35.0`: computes `F_{max_n}(t)` via the strictly positive Kummer confluent
///   hypergeometric series `F_n(t) = \frac{e^{-t}}{2n+1} \sum_{k=0}^\infty \prod_{j=1}^k \frac{t}{n + j + 1/2}`
///   (completely free of alternating-sign cancellation), followed by unconditionally stable
///   downward recursion `F_{n-1}(t) = (2 t F_n(t) + e^{-t}) / (2n - 1)`.
/// - `t > 35.0`: computes `F_0(t) = \frac{1}{2}\sqrt{\pi/t}\operatorname{erf}(\sqrt{t})` and
///   upward recursion `F_n(t) = ((2n - 1) F_{n-1}(t) - e^{-t}) / (2t)`.
pub fn boys_batch(max_n: usize, t: f64, out: &mut [f64]) {
    assert!(out.len() > max_n, "boys_batch output slice too short");
    assert!(t >= 0.0, "Boys function argument t must be >= 0, got {t}");

    if t < 1e-14 {
        for (n, slot) in out.iter_mut().enumerate().take(max_n + 1) {
            let nf = n as f64;
            *slot = 1.0 / (2.0 * nf + 1.0) - t / (2.0 * nf + 3.0)
                + 0.5 * t * t / (2.0 * nf + 5.0);
        }
        return;
    }

    if t <= 35.0 {
        let exp_neg_t = (-t).exp();
        let nf = max_n as f64;
        let mut term = 1.0 / (2.0 * nf + 1.0);
        let mut acc = term;
        for k in 1..200 {
            term *= t / (nf + (k as f64) + 0.5);
            acc += term;
            if term <= acc * 1e-16 {
                break;
            }
        }
        out[max_n] = exp_neg_t * acc;
        for n in (1..=max_n).rev() {
            let nf = n as f64;
            out[n - 1] = (2.0 * t * out[n] + exp_neg_t) / (2.0 * nf - 1.0);
        }
        return;
    }

    let sqrt_t = t.sqrt();
    out[0] = 0.5 * (PI / t).sqrt() * erf_approx(sqrt_t);
    let exp_neg_t = (-t).exp();
    for n in 1..=max_n {
        let nf = n as f64;
        out[n] = ((2.0 * nf - 1.0) * out[n - 1] - exp_neg_t) / (2.0 * t);
    }
}

/// Evaluates the single Boys function value `F_n(t)`.
pub fn boys(n: usize, t: f64) -> f64 {
    let mut buf = vec![0.0; n + 1];
    boys_batch(n, t, &mut buf);
    buf[n]
}
