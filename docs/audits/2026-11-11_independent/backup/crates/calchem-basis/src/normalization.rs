use calchem_model::CalcError;
use std::f64::consts::PI;

/// Computes the odd double factorial `(2l - 1)!!` with `(-1)!! = 0!! = 1`.
pub fn double_factorial_odd(two_l_minus_one: i32) -> f64 {
    if two_l_minus_one <= 0 {
        return 1.0;
    }
    let mut acc = 1.0;
    let mut k = two_l_minus_one;
    while k > 0 {
        acc *= k as f64;
        k -= 2;
    }
    acc
}

/// Computes the exact normalization constant of a Cartesian GTO primitive:
/// `N(alpha; l, m, n) = (2 * alpha / pi)^(3/4) * sqrt((4 * alpha)^(l+m+n) / ((2l-1)!! (2m-1)!! (2n-1)!!))`
pub fn primitive_norm(alpha: f64, lmn: [u8; 3]) -> f64 {
    let [l, m, n] = lmn.map(|x| x as i32);
    let l_total = l + m + n;
    let radial = (2.0 * alpha / PI).powf(0.75);
    let num = (4.0 * alpha).powi(l_total);
    let den = double_factorial_odd(2 * l - 1)
        * double_factorial_odd(2 * m - 1)
        * double_factorial_odd(2 * n - 1);
    radial * (num / den).sqrt()
}

/// Computes the self-overlap `<chi_i | chi_j>` of two normalized Cartesian GTO primitives
/// sharing the same center and angular momentum `(l, m, n)`.
pub fn normalized_primitive_self_overlap(alpha_i: f64, alpha_j: f64, lmn: [u8; 3]) -> f64 {
    let [l, m, n] = lmn.map(|x| x as i32);
    let gamma = alpha_i + alpha_j;
    let ni = primitive_norm(alpha_i, lmn);
    let nj = primitive_norm(alpha_j, lmn);
    let radial = (PI / gamma).powf(1.5);
    let mut ang = 1.0;
    if l > 0 {
        ang *= double_factorial_odd(2 * l - 1) / (2.0 * gamma).powi(l);
    }
    if m > 0 {
        ang *= double_factorial_odd(2 * m - 1) / (2.0 * gamma).powi(m);
    }
    if n > 0 {
        ang *= double_factorial_odd(2 * n - 1) / (2.0 * gamma).powi(n);
    }
    ni * nj * radial * ang
}

/// Rescales raw contraction coefficients `c_p` so that the contracted basis function
/// `phi = sum_p c_p * N(alpha_p, lmn) * g_p` satisfies `<phi | phi> = 1`.
///
/// Note: Because the angular factor `(2l-1)!!(2m-1)!!(2n-1)!!` cancels identically between
/// `N(alpha_i, lmn) * N(alpha_j, lmn)` and the concentric primitive overlap integral, the
/// contraction normalization factor depends only on `L = l + m + n` and is identical for
/// all Cartesian components within a shell.
pub fn normalized_contraction(
    exponents: &[f64],
    coefficients: &[f64],
    lmn: [u8; 3],
) -> Result<Vec<f64>, CalcError> {
    if exponents.is_empty() || exponents.len() != coefficients.len() {
        return Err(CalcError::UnsupportedBasis(
            "Invalid primitive exponents/coefficients length".into(),
        ));
    }
    if exponents.iter().any(|&a| !a.is_finite() || a <= 0.0) {
        return Err(CalcError::UnsupportedBasis(
            "Primitive exponents must be finite and strictly positive".into(),
        ));
    }
    if coefficients.iter().any(|&c| !c.is_finite()) {
        return Err(CalcError::UnsupportedBasis(
            "Contraction coefficients must be finite".into(),
        ));
    }

    let n_prim = exponents.len();
    let mut self_overlap = 0.0;
    for i in 0..n_prim {
        for j in 0..n_prim {
            let s_ij = normalized_primitive_self_overlap(exponents[i], exponents[j], lmn);
            self_overlap += coefficients[i] * coefficients[j] * s_ij;
        }
    }

    if !self_overlap.is_finite() || self_overlap <= 0.0 {
        return Err(CalcError::UnsupportedBasis(
            "Contracted shell self-overlap is non-positive".into(),
        ));
    }

    let scale = 1.0 / self_overlap.sqrt();
    Ok(coefficients.iter().map(|&c| c * scale).collect())
}
