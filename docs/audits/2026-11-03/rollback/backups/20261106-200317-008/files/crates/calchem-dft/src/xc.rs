//! Spin-polarized Exchange-Correlation (XC) Functional Engine in Pure Rust:
//! - LDA: Slater (Dirac) exchange + Vosko-Wilk-Nusair (`VWN5` and `VWN_RPA`) correlation
//! - GGA: Becke 88 (`B88`) exchange, Lee-Yang-Parr (`LYP`, Miehlich gradient-only form) correlation,
//!   and Perdew-Burke-Ernzerhof (`PBE`) exchange + correlation (built on `PW92`)
//! - Hybrids: `B3LYP` (canonical Gaussian/Stephens `VWN_RPA`), `B3LYP5` (`VWN5`), and `PBE0` (`25% HF`)

use std::f64::consts::PI;

use calchem_model::CalcError;
use serde::{Deserialize, Serialize};

/// Density threshold below which spin XC energy density and potentials safely vanish.
const RHO_CUTOFF: f64 = 1e-18;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XcFunctional {
    /// Slater exchange + VWN5 local correlation (`LDA` / `SVWN` / `SVWN5`)
    Svwn5,
    /// Becke88 exchange + Lee-Yang-Parr correlation (`BLYP`)
    Blyp,
    /// Perdew-Burke-Ernzerhof GGA exchange-correlation (`PBE`)
    Pbe,
    /// Canonical B3LYP hybrid: `0.20 HF + 0.80 Slater + 0.72 dB88 + 0.19 VWN_RPA + 0.81 LYP`
    B3lyp,
    /// B3LYP5 hybrid variant using `VWN5` for the `0.19` local correlation component
    B3lyp5,
    /// Hybrid PBE0 (`PBE1PBE`): `0.25 HF + 0.75 PBE_X + 1.00 PBE_C`
    Pbe0,
}

impl XcFunctional {
    pub const SUPPORTED_NAMES: &'static [&'static str] = &[
        "svwn5", "svwn", "lda", "blyp", "pbe", "b3lyp", "b3lyp5", "pbe0",
    ];

    pub fn from_name(name: &str) -> Result<Self, CalcError> {
        let norm: String = name
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
            .collect();
        match norm.as_str() {
            "svwn" | "svwn5" | "lda" | "slatervwn5" => Ok(Self::Svwn5),
            "blyp" => Ok(Self::Blyp),
            "pbe" | "pbepbe" => Ok(Self::Pbe),
            "b3lyp" => Ok(Self::B3lyp),
            "b3lyp5" => Ok(Self::B3lyp5),
            "pbe0" | "pbe1pbe" => Ok(Self::Pbe0),
            _ => Err(CalcError::UnsupportedMethod(format!(
                "Unsupported DFT XC functional '{name}' (supported: SVWN5, BLYP, PBE, B3LYP, B3LYP5, PBE0)"
            ))),
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Svwn5 => "svwn5",
            Self::Blyp => "blyp",
            Self::Pbe => "pbe",
            Self::B3lyp => "b3lyp",
            Self::B3lyp5 => "b3lyp5",
            Self::Pbe0 => "pbe0",
        }
    }

    /// Fraction $c_{\text{HF}} \in [0, 1]$ of exact Hartree-Fock exchange.
    #[inline]
    pub fn exact_exchange_fraction(self) -> f64 {
        match self {
            Self::Svwn5 | Self::Blyp | Self::Pbe => 0.0,
            Self::B3lyp | Self::B3lyp5 => 0.20,
            Self::Pbe0 => 0.25,
        }
    }

    #[inline]
    pub fn is_gga(self) -> bool {
        !matches!(self, Self::Svwn5)
    }
}

/// Volumetric XC energy density $f_{\text{xc}}$ (so $E_{\text{xc}} = \int f_{\text{xc}}\,d^3\mathbf{r}$)
/// and first partial derivatives with respect to $(\rho_\alpha, \rho_\beta, \gamma_{\alpha\alpha}, \gamma_{\alpha\beta}, \gamma_{\beta\beta})$.
#[derive(Debug, Clone, Copy, Default)]
pub struct XcPointOut {
    pub f_xc: f64,
    pub v_rho_a: f64,
    pub v_rho_b: f64,
    pub v_gamma_aa: f64,
    pub v_gamma_ab: f64,
    pub v_gamma_bb: f64,
}

impl XcPointOut {
    #[inline]
    fn add_scaled(&mut self, scale: f64, other: Self) {
        self.f_xc += scale * other.f_xc;
        self.v_rho_a += scale * other.v_rho_a;
        self.v_rho_b += scale * other.v_rho_b;
        self.v_gamma_aa += scale * other.v_gamma_aa;
        self.v_gamma_ab += scale * other.v_gamma_ab;
        self.v_gamma_bb += scale * other.v_gamma_bb;
    }
}

/// Evaluates `func` at spin-polarized invariants $(\rho_\alpha, \rho_\beta, \gamma_{\alpha\alpha}, \gamma_{\alpha\beta}, \gamma_{\beta\beta})$.
#[inline]
pub fn eval_xc_spin(
    func: XcFunctional,
    rho_a: f64,
    rho_b: f64,
    gamma_aa: f64,
    gamma_ab: f64,
    gamma_bb: f64,
) -> XcPointOut {
    let ra = rho_a.max(0.0);
    let rb = rho_b.max(0.0);
    if ra + rb < RHO_CUTOFF {
        return XcPointOut::default();
    }
    let gaa = gamma_aa.max(0.0);
    let gbb = gamma_bb.max(0.0);

    match func {
        XcFunctional::Svwn5 => {
            let mut out = slater_x(ra, rb);
            out.add_scaled(1.0, vwn_c(ra, rb, &VWN5_PARAMS));
            out
        }
        XcFunctional::Blyp => {
            let mut out = slater_x(ra, rb);
            out.add_scaled(1.0, becke88_delta_x(ra, rb, gaa, gbb));
            out.add_scaled(1.0, lyp_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::Pbe => {
            let mut out = pbe_x(ra, rb, gaa, gbb);
            out.add_scaled(1.0, pbe_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::B3lyp => {
            let mut out = XcPointOut::default();
            out.add_scaled(0.80, slater_x(ra, rb));
            out.add_scaled(0.72, becke88_delta_x(ra, rb, gaa, gbb));
            out.add_scaled(0.19, vwn_c(ra, rb, &VWN_RPA_PARAMS));
            out.add_scaled(0.81, lyp_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::B3lyp5 => {
            let mut out = XcPointOut::default();
            out.add_scaled(0.80, slater_x(ra, rb));
            out.add_scaled(0.72, becke88_delta_x(ra, rb, gaa, gbb));
            out.add_scaled(0.19, vwn_c(ra, rb, &VWN5_PARAMS));
            out.add_scaled(0.81, lyp_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
        XcFunctional::Pbe0 => {
            let mut out = XcPointOut::default();
            out.add_scaled(0.75, pbe_x(ra, rb, gaa, gbb));
            out.add_scaled(1.00, pbe_c(ra, rb, gaa, gamma_ab, gbb));
            out
        }
    }
}

// ============================================================================
// 1. Slater (Dirac) Spin-Polarized LDA Exchange
// ============================================================================

#[inline]
fn slater_x_one_spin(rho_s: f64) -> (f64, f64) {
    if rho_s < RHO_CUTOFF {
        return (0.0, 0.0);
    }
    let cx = 0.75 * (6.0 / PI).powf(1.0 / 3.0);
    let r13 = rho_s.cbrt();
    let f = -cx * rho_s * r13;
    let v = -(4.0 / 3.0) * cx * r13;
    (f, v)
}

#[inline]
fn slater_x(ra: f64, rb: f64) -> XcPointOut {
    let (fa, va) = slater_x_one_spin(ra);
    let (fb, vb) = slater_x_one_spin(rb);
    XcPointOut {
        f_xc: fa + fb,
        v_rho_a: va,
        v_rho_b: vb,
        ..Default::default()
    }
}

// ============================================================================
// 2. Vosko-Wilk-Nusair (1980) Correlation (VWN5 & VWN_RPA)
// ============================================================================

struct VwnChannel {
    a: f64,
    x0: f64,
    b: f64,
    c: f64,
}

struct VwnParams {
    para: VwnChannel,
    ferro: VwnChannel,
    stiff: VwnChannel,
}

const VWN5_PARAMS: VwnParams = VwnParams {
    para: VwnChannel {
        a: 0.0310907,
        x0: -0.10498,
        b: 3.72744,
        c: 12.9352,
    },
    ferro: VwnChannel {
        a: 0.01554535,
        x0: -0.32500,
        b: 7.06042,
        c: 18.0578,
    },
    stiff: VwnChannel {
        a: -0.016886863940389905, // -1 / (6 * pi^2)
        x0: -0.00475840,
        b: 1.13107,
        c: 13.0045,
    },
};

const VWN_RPA_PARAMS: VwnParams = VwnParams {
    para: VwnChannel {
        a: 0.0310907,
        x0: -0.409286,
        b: 13.0720,
        c: 42.7198,
    },
    ferro: VwnChannel {
        a: 0.01554535,
        x0: -0.743294,
        b: 20.1231,
        c: 101.578,
    },
    stiff: VwnChannel {
        a: -0.016886863940389905,
        x0: -0.228344,
        b: 1.06835,
        c: 11.4813,
    },
};

/// Returns `(eps(rs), rs * d(eps)/d(rs))` using VWN's exact Padé rational derivative.
#[inline]
fn vwn_channel_eval(x: f64, ch: &VwnChannel) -> (f64, f64) {
    let xx = x * x + ch.b * x + ch.c;
    let xx0 = ch.x0 * ch.x0 + ch.b * ch.x0 + ch.c;
    let q = (4.0 * ch.c - ch.b * ch.b).sqrt();
    let atan_term = (q / (2.0 * x + ch.b)).atan();
    let x_m_x0 = x - ch.x0;

    let eps = ch.a
        * ((x * x / xx).ln() + (2.0 * ch.b / q) * atan_term
            - (ch.b * ch.x0 / xx0)
                * ((x_m_x0 * x_m_x0 / xx).ln() + (2.0 * (ch.b + 2.0 * ch.x0) / q) * atan_term));

    // Exact Padé rational identity: rs * d(eps)/d(rs) = A * [c*(x - x0) - b*x*x0] / [(x - x0)*X(x)]
    let rs_deps = ch.a * (ch.c * x_m_x0 - ch.b * x * ch.x0) / (x_m_x0 * xx);
    (eps, rs_deps)
}

#[inline]
fn spin_interp_f_and_df(zeta: f64) -> (f64, f64, f64) {
    let cbrt2_m1 = 2.0_f64.cbrt() - 1.0;
    let denom = 2.0 * cbrt2_m1;
    let fdd0 = 4.0 / (9.0 * cbrt2_m1);
    let zp = (1.0 + zeta).max(0.0);
    let zm = (1.0 - zeta).max(0.0);
    let zp13 = zp.cbrt();
    let zm13 = zm.cbrt();
    let fz = (zp * zp13 + zm * zm13 - 2.0) / denom;
    let dfz = (4.0 / (3.0 * denom)) * (zp13 - zm13);
    (fz, dfz, fdd0)
}

#[inline]
fn vwn_c(ra: f64, rb: f64, params: &VwnParams) -> XcPointOut {
    let rho = ra + rb;
    if rho < RHO_CUTOFF {
        return XcPointOut::default();
    }
    let rs = (3.0 / (4.0 * PI * rho)).cbrt();
    let x = rs.sqrt();
    let zeta = ((ra - rb) / rho).clamp(-1.0, 1.0);

    let (ep, rs_dep) = vwn_channel_eval(x, &params.para);
    let (ef, rs_def) = vwn_channel_eval(x, &params.ferro);
    let (ac, rs_dac) = vwn_channel_eval(x, &params.stiff);

    let (fz, dfz, fdd0) = spin_interp_f_and_df(zeta);
    let z2 = zeta * zeta;
    let z3 = z2 * zeta;
    let z4 = z2 * z2;
    let one_m_z4 = 1.0 - z4;

    let inv_fdd0 = 1.0 / fdd0;
    let dec = ef - ep;
    let rs_ddec = rs_def - rs_dep;

    let ec = ep + ac * fz * inv_fdd0 * one_m_z4 + dec * fz * z4;
    let rs_dec_drs = rs_dep + rs_dac * fz * inv_fdd0 * one_m_z4 + rs_ddec * fz * z4;
    let dec_dzeta =
        ac * inv_fdd0 * (dfz * one_m_z4 - 4.0 * z3 * fz) + dec * (dfz * z4 + 4.0 * z3 * fz);

    let base_v = ec - (1.0 / 3.0) * rs_dec_drs;
    XcPointOut {
        f_xc: rho * ec,
        v_rho_a: base_v + (1.0 - zeta) * dec_dzeta,
        v_rho_b: base_v - (1.0 + zeta) * dec_dzeta,
        ..Default::default()
    }
}

// ============================================================================
// 3. Becke 88 (B88) GGA Exchange Gradient Correction (\Delta E_x^{B88})
// ============================================================================

#[inline]
fn b88_delta_one_spin(rho_s: f64, gamma_ss: f64) -> (f64, f64, f64) {
    if rho_s < RHO_CUTOFF {
        return (0.0, 0.0, 0.0);
    }
    const BETA: f64 = 0.0042;
    let r13 = rho_s.cbrt();
    let r43 = rho_s * r13;
    let s2 = gamma_ss.max(0.0) / (r43 * r43);
    let s = s2.sqrt();
    let asinh_s = s.asinh();
    let denom = 1.0 + 6.0 * BETA * s * asinh_s;

    let g = -BETA * s2 / denom;
    // h(s) = g'(s) / s (analytic & smooth at s = 0, where h(0) = -2 * BETA)
    let sqrt_1_s2 = (1.0 + s2).sqrt();
    let h = -BETA * (2.0 + 6.0 * BETA * s * asinh_s - 6.0 * BETA * s2 / sqrt_1_s2)
        / (denom * denom);

    let f = r43 * g;
    let v_rho = (4.0 / 3.0) * r13 * (g - s2 * h);
    let v_gamma = 0.5 * h / r43;
    (f, v_rho, v_gamma)
}

#[inline]
fn becke88_delta_x(ra: f64, rb: f64, gaa: f64, gbb: f64) -> XcPointOut {
    let (fa, vra, vgaa) = b88_delta_one_spin(ra, gaa);
    let (fb, vrb, vgbb) = b88_delta_one_spin(rb, gbb);
    XcPointOut {
        f_xc: fa + fb,
        v_rho_a: vra,
        v_rho_b: vrb,
        v_gamma_aa: vgaa,
        v_gamma_ab: 0.0,
        v_gamma_bb: vgbb,
    }
}

// ============================================================================
// 4. Lee-Yang-Parr (LYP) GGA Correlation (Miehlich et al. 1989 Eq. 2)
// ============================================================================

/// Returns `(S, dS/d(ra), A_aa, A_ab, A_bb)` for Miehlich Eq. (2)'s bracket `S`.
/// Calling with `(rb, ra, delta, ddelta, gbb, gab, gaa)` yields `dS/d(rb)` by symmetry!
#[inline]
fn lyp_bracket_and_dra(
    ra: f64,
    rb: f64,
    delta: f64,
    ddelta: f64,
    gaa: f64,
    gab: f64,
    gbb: f64,
) -> (f64, f64, f64, f64, f64) {
    let rho = ra + rb;
    let rho2 = rho * rho;
    let cf = 0.3 * (3.0 * PI * PI).powf(2.0 / 3.0);
    let k0 = 2.0_f64.powf(11.0 / 3.0) * cf;

    // S_TF = k0 * ra * rb * (ra^{8/3} + rb^{8/3})
    let ra_83 = if ra > 0.0 { ra.powf(8.0 / 3.0) } else { 0.0 };
    let rb_83 = if rb > 0.0 { rb.powf(8.0 / 3.0) } else { 0.0 };
    let s_tf = k0 * ra * rb * (ra_83 + rb_83);
    let ds_tf_dra = k0 * ((11.0 / 3.0) * ra_83 * rb + rb * rb_83);

    // C_gamma = ra * rb * (47 - 7 * delta)/18 - (2/3) * rho^2
    let c_gam = ra * rb * (47.0 - 7.0 * delta) / 18.0 - (2.0 / 3.0) * rho2;
    let dc_gam_dra =
        rb * (47.0 - 7.0 * delta) / 18.0 + ra * rb * (-7.0 * ddelta) / 18.0 - (4.0 / 3.0) * rho;

    // E_aa = -ra*rb*(5/2 - delta/18) - ((delta - 11)/9) * (ra^2 * rb / rho) + ((2/3)*rho^2 - rb^2)
    let q_aa = ra * ra * rb / rho;
    let dq_aa_dra = (2.0 * ra * rb * rho - ra * ra * rb) / rho2;
    let e_aa = -ra * rb * (2.5 - delta / 18.0) - ((delta - 11.0) / 9.0) * q_aa
        + ((2.0 / 3.0) * rho2 - rb * rb);
    let de_aa_dra = -rb * (2.5 - delta / 18.0) + ra * rb * (ddelta / 18.0)
        - (ddelta / 9.0) * q_aa
        - ((delta - 11.0) / 9.0) * dq_aa_dra
        + (4.0 / 3.0) * rho;

    // E_bb = -ra*rb*(5/2 - delta/18) - ((delta - 11)/9) * (ra * rb^2 / rho) + ((2/3)*rho^2 - ra^2)
    let q_bb = ra * rb * rb / rho;
    let dq_bb_dra = (rb * rb * rho - ra * rb * rb) / rho2;
    let e_bb = -ra * rb * (2.5 - delta / 18.0) - ((delta - 11.0) / 9.0) * q_bb
        + ((2.0 / 3.0) * rho2 - ra * ra);
    let de_bb_dra = -rb * (2.5 - delta / 18.0) + ra * rb * (ddelta / 18.0)
        - (ddelta / 9.0) * q_bb
        - ((delta - 11.0) / 9.0) * dq_bb_dra
        + (4.0 / 3.0) * rho
        - 2.0 * ra;

    let a_aa = c_gam + e_aa;
    let a_ab = 2.0 * c_gam;
    let a_bb = c_gam + e_bb;

    let s = s_tf + a_aa * gaa + a_ab * gab + a_bb * gbb;
    let ds_dra = ds_tf_dra
        + (dc_gam_dra + de_aa_dra) * gaa
        + (2.0 * dc_gam_dra) * gab
        + (dc_gam_dra + de_bb_dra) * gbb;

    (s, ds_dra, a_aa, a_ab, a_bb)
}

#[inline]
fn lyp_c(ra: f64, rb: f64, gaa: f64, gab: f64, gbb: f64) -> XcPointOut {
    let rho = ra + rb;
    if rho < RHO_CUTOFF {
        return XcPointOut::default();
    }
    const A: f64 = 0.04918;
    const B: f64 = 0.132;
    const C: f64 = 0.2533;
    const D: f64 = 0.349;

    let r13 = rho.powf(-1.0 / 3.0);
    // Avoid exp(-c * rho^{-1/3}) underflowing into NaN ratios at ultra-tiny density tails:
    if C * r13 > 500.0 {
        // omega -> 0 exponentially; only T0 survives.
        let den = 1.0 + D * r13;
        let g0 = -4.0 * A / (rho * den);
        let dg0 = -g0 * (1.0 + (2.0 / 3.0) * D * r13) / (rho * den);
        return XcPointOut {
            f_xc: g0 * ra * rb,
            v_rho_a: dg0 * ra * rb + g0 * rb,
            v_rho_b: dg0 * ra * rb + g0 * ra,
            ..Default::default()
        };
    }

    let den = 1.0 + D * r13;
    let g0 = -4.0 * A / (rho * den);
    let dg0 = -g0 * (1.0 + (2.0 / 3.0) * D * r13) / (rho * den);
    let t0 = g0 * ra * rb;
    let dt0_dra = dg0 * ra * rb + g0 * rb;
    let dt0_drb = dg0 * ra * rb + g0 * ra;

    let omega = (-C * r13).exp() / den * rho.powf(-11.0 / 3.0);
    let delta = C * r13 + D * r13 / den;
    let domega = omega * (delta - 11.0) / (3.0 * rho);
    let r43_inv = r13 / rho;
    let ddelta = -(1.0 / 3.0) * r43_inv * (C + D / (den * den));

    let (s, ds_dra, a_aa, a_ab, a_bb) = lyp_bracket_and_dra(ra, rb, delta, ddelta, gaa, gab, gbb);
    let (_s_sym, ds_drb, _, _, _) = lyp_bracket_and_dra(rb, ra, delta, ddelta, gbb, gab, gaa);

    let ab = A * B;
    let f_xc = t0 - ab * omega * s;
    let v_rho_a = dt0_dra - ab * (domega * s + omega * ds_dra);
    let v_rho_b = dt0_drb - ab * (domega * s + omega * ds_drb);

    XcPointOut {
        f_xc,
        v_rho_a,
        v_rho_b,
        v_gamma_aa: -ab * omega * a_aa,
        v_gamma_ab: -ab * omega * a_ab,
        v_gamma_bb: -ab * omega * a_bb,
    }
}

// ============================================================================
// 5. Perdew-Burke-Ernzerhof (PBE) GGA Exchange & Correlation (PW92 base)
// ============================================================================

#[inline]
fn pbe_x_one_spin(rho_s: f64, gamma_ss: f64) -> (f64, f64, f64) {
    if rho_s < RHO_CUTOFF {
        return (0.0, 0.0, 0.0);
    }
    const KAPPA: f64 = 0.804;
    const MU: f64 = 0.2195149727645171;
    // Spin-scaling: f_x(rho_s, gamma_ss) = 0.5 * f_x^{unpol}(2*rho_s, 4*gamma_ss)
    let rho_u = 2.0 * rho_s;
    let gam_u = 4.0 * gamma_ss.max(0.0);

    let cx_u = -0.75 * (3.0 / PI).powf(1.0 / 3.0);
    let r13 = rho_u.cbrt();
    let r43 = rho_u * r13;
    let ex_lda = cx_u * r43;

    let kf = (3.0 * PI * PI * rho_u).cbrt();
    let denom_s2 = 4.0 * kf * kf * rho_u * rho_u;
    let s2 = gam_u / denom_s2;

    let one_p_ms = 1.0 + (MU / KAPPA) * s2;
    let fx = 1.0 + KAPPA - KAPPA / one_p_ms;
    let dfx_ds2 = MU / (one_p_ms * one_p_ms);

    // f_u = ex_lda * fx; ds2/drho_u = -(8/3) * s2 / rho_u; ds2/dgam_u = 1 / denom_s2
    let f_u = ex_lda * fx;
    let df_drho_u = (4.0 / 3.0) * cx_u * r13 * fx + ex_lda * dfx_ds2 * (-(8.0 / 3.0) * s2 / rho_u);
    let df_dgam_u = ex_lda * dfx_ds2 / denom_s2;

    // Map back: f = 0.5 * f_u(2*rho_s, 4*gamma_ss) => df/drho_s = df_drho_u, df/dgamma_ss = 2 * df_dgam_u
    (0.5 * f_u, df_drho_u, 2.0 * df_dgam_u)
}

#[inline]
fn pbe_x(ra: f64, rb: f64, gaa: f64, gbb: f64) -> XcPointOut {
    let (fa, vra, vgaa) = pbe_x_one_spin(ra, gaa);
    let (fb, vrb, vgbb) = pbe_x_one_spin(rb, gbb);
    XcPointOut {
        f_xc: fa + fb,
        v_rho_a: vra,
        v_rho_b: vrb,
        v_gamma_aa: vgaa,
        v_gamma_ab: 0.0,
        v_gamma_bb: vgbb,
    }
}

struct Pw92Channel {
    a: f64,
    alpha1: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
}

const PW92_G0: Pw92Channel = Pw92Channel {
    a: 0.0310907,
    alpha1: 0.21370,
    b1: 7.5957,
    b2: 3.5876,
    b3: 1.6382,
    b4: 0.49294,
};

const PW92_G1: Pw92Channel = Pw92Channel {
    a: 0.01554535,
    alpha1: 0.20548,
    b1: 14.1189,
    b2: 6.1977,
    b3: 3.3662,
    b4: 0.62517,
};

const PW92_GA: Pw92Channel = Pw92Channel {
    a: 0.0168869,
    alpha1: 0.11125,
    b1: 10.357,
    b2: 3.6231,
    b3: 0.88026,
    b4: 0.49671,
};

/// Returns `(G(rs), dG/drs)` for a PW92 channel.
#[inline]
fn pw92_g(rs: f64, ch: &Pw92Channel) -> (f64, f64) {
    let x = rs.sqrt();
    let x2 = rs;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let q0 = -2.0 * ch.a * (1.0 + ch.alpha1 * rs);
    let q1 = 2.0 * ch.a * (ch.b1 * x + ch.b2 * x2 + ch.b3 * x3 + ch.b4 * x4);
    let dq1_drs = ch.a * (ch.b1 / x + 2.0 * ch.b2 + 3.0 * ch.b3 * x + 4.0 * ch.b4 * x2);
    let ln_term = (1.0 + 1.0 / q1).ln();
    let g = q0 * ln_term;
    let dg_drs = -2.0 * ch.a * ch.alpha1 * ln_term - q0 * dq1_drs / (q1 * (q1 + 1.0));
    (g, dg_drs)
}

#[inline]
fn pbe_c(ra: f64, rb: f64, gaa: f64, gab: f64, gbb: f64) -> XcPointOut {
    let rho = ra + rb;
    if rho < RHO_CUTOFF {
        return XcPointOut::default();
    }
    let gamma_tot = (gaa + 2.0 * gab + gbb).max(0.0);
    let rs = (3.0 / (4.0 * PI * rho)).cbrt();
    let drs_drho = -rs / (3.0 * rho);
    let zeta = ((ra - rb) / rho).clamp(-1.0, 1.0);

    // 1. PW92 local correlation eps_c(rs, zeta)
    let (e0, de0_drs) = pw92_g(rs, &PW92_G0);
    let (e1, de1_drs) = pw92_g(rs, &PW92_G1);
    let (minus_ac, d_minus_ac_drs) = pw92_g(rs, &PW92_GA);
    let ac = -minus_ac;
    let dac_drs = -d_minus_ac_drs;

    let (fz, dfz, fdd0) = spin_interp_f_and_df(zeta);
    let inv_fdd0 = 1.0 / fdd0;
    let z2 = zeta * zeta;
    let z3 = z2 * zeta;
    let z4 = z2 * z2;
    let one_m_z4 = 1.0 - z4;

    let ec_lsda = e0 + ac * fz * inv_fdd0 * one_m_z4 + (e1 - e0) * fz * z4;
    let dec_drs = de0_drs + dac_drs * fz * inv_fdd0 * one_m_z4 + (de1_drs - de0_drs) * fz * z4;
    let dec_drho = dec_drs * drs_drho;
    let dec_dzeta =
        ac * inv_fdd0 * (dfz * one_m_z4 - 4.0 * z3 * fz) + (e1 - e0) * (dfz * z4 + 4.0 * z3 * fz);

    if gamma_tot < 1e-36 {
        return XcPointOut {
            f_xc: rho * ec_lsda,
            v_rho_a: ec_lsda + rho * dec_drho + (1.0 - zeta) * dec_dzeta,
            v_rho_b: ec_lsda + rho * dec_drho - (1.0 + zeta) * dec_dzeta,
            ..Default::default()
        };
    }

    // 2. PBE gradient correction H(rho, zeta, gamma_tot)
    let gamma_pbe = (1.0 - 2.0_f64.ln()) / (PI * PI);
    const BETA_PBE: f64 = 0.06672455060314922;
    let b0 = BETA_PBE / gamma_pbe;

    let zp = (1.0 + zeta).max(1e-24);
    let zm = (1.0 - zeta).max(1e-24);
    let zp23 = zp.cbrt().powi(2);
    let zm23 = zm.cbrt().powi(2);
    let phi = 0.5 * (zp23 + zm23);
    let phi2 = phi * phi;
    let phi3 = phi2 * phi;

    // Singularity-free combinations (1 - zeta) * dphi/dzeta and -(1 + zeta) * dphi/dzeta:
    let one_m_z_dphi = ((1.0 - zeta) * zp.cbrt().recip() - zm23) / 3.0;
    let minus_one_p_z_dphi = ((1.0 + zeta) * zm.cbrt().recip() - zp23) / 3.0;

    let c_t = (16.0 / PI) * (3.0 * PI * PI).cbrt();
    let rho73 = rho * rho * rho.cbrt();
    let denom_t2 = c_t * phi2 * rho73;
    let t2 = gamma_tot / denom_t2;

    let w = -ec_lsda / (gamma_pbe * phi3);
    let (a_func, da_dw) = if w > 100.0 {
        (0.0, 0.0)
    } else {
        let em1 = w.exp_m1();
        let ew = w.exp();
        let af = b0 / em1;
        (af, -af * ew / em1)
    };

    let u = a_func * t2;
    let d_poly = 1.0 + u + u * u;
    let q_rat = (1.0 + u) / d_poly;
    let z_arg = 1.0 + b0 * t2 * q_rat;
    let ln_z = z_arg.ln();
    let h = gamma_pbe * phi3 * ln_z;

    let dz_dt2 = b0 * (1.0 + 2.0 * u) / (d_poly * d_poly);
    let dz_da = -b0 * t2 * t2 * u * (2.0 + u) / (d_poly * d_poly);
    let dh_dt2 = gamma_pbe * phi3 * dz_dt2 / z_arg;
    let dh_dw = gamma_pbe * phi3 * (dz_da * da_dw) / z_arg;

    // Chain rule at fixed (zeta, gamma_tot) with respect to rho:
    //   dt2/drho = -(7/3) * t2 / rho,   dw/drho = -dec_drho / (gamma_pbe * phi3)
    let dh_drho_fixed_zeta =
        dh_dt2 * (-(7.0 / 3.0) * t2 / rho) + dh_dw * (-dec_drho / (gamma_pbe * phi3));

    // Chain rule with respect to zeta through both dec_dzeta and phi(zeta):
    //   dH/dphi|_{w,t2} + dh_dw * dw/dphi + dh_dt2 * dt2/dphi
    //   where dH/dphi|_{w,t2} = 3 * gamma_pbe * phi^2 * ln_z,
    //         dw/dphi = -3 * w / phi,
    //         dt2/dphi = -2 * t2 / phi.
    let dh_dphi_total =
        3.0 * gamma_pbe * phi2 * ln_z + dh_dw * (-3.0 * w / phi) + dh_dt2 * (-2.0 * t2 / phi);
    let dh_d_ec = dh_dw * (-1.0 / (gamma_pbe * phi3));

    // rho * d(ec + H)/drho_a = (ec + H) + rho*(dec_drho + dh_drho_fixed_zeta)
    //   + (1 - zeta)*(dec_dzeta * (1 + dh_d_ec)) + dh_dphi_total * [(1 - zeta)*dphi/dzeta]
    let total_e = ec_lsda + h;
    let common_rho = total_e + rho * (dec_drho + dh_drho_fixed_zeta);
    let v_rho_a =
        common_rho + (1.0 - zeta) * dec_dzeta * (1.0 + dh_d_ec) + dh_dphi_total * one_m_z_dphi;
    let v_rho_b = common_rho - (1.0 + zeta) * dec_dzeta * (1.0 + dh_d_ec)
        + dh_dphi_total * minus_one_p_z_dphi;

    let v_gamma_tot = rho * dh_dt2 / denom_t2;

    XcPointOut {
        f_xc: rho * total_e,
        v_rho_a,
        v_rho_b,
        v_gamma_aa: v_gamma_tot,
        v_gamma_ab: 2.0 * v_gamma_tot,
        v_gamma_bb: v_gamma_tot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_xc_partials_match_fd(
        func: XcFunctional,
        ra: f64,
        rb: f64,
        gaa: f64,
        gab: f64,
        gbb: f64,
    ) {
        let ana = eval_xc_spin(func, ra, rb, gaa, gab, gbb);
        let h_r = 1e-6 * (ra + rb).max(1e-2);
        let h_g = 1e-6 * (gaa + gbb + 1.0);

        let f_only = |a: f64, b: f64, aa: f64, ab: f64, bb: f64| -> f64 {
            eval_xc_spin(func, a, b, aa, ab, bb).f_xc
        };

        // 4-point central difference O(h^4) for sharp 1e-11 verification
        let fd4 = |eval_shift: &dyn Fn(f64) -> f64, h: f64| -> f64 {
            (-eval_shift(2.0 * h) + 8.0 * eval_shift(h) - 8.0 * eval_shift(-h)
                + eval_shift(-2.0 * h))
                / (12.0 * h)
        };

        if ra > 5.0 * h_r {
            let fd_vra = fd4(&|d| f_only(ra + d, rb, gaa, gab, gbb), h_r);
            assert!(
                (ana.v_rho_a - fd_vra).abs() < 1e-9,
                "{func:?} v_rho_a mismatch: ana={} vs fd={}",
                ana.v_rho_a,
                fd_vra
            );
        }
        if rb > 5.0 * h_r {
            let fd_vrb = fd4(&|d| f_only(ra, rb + d, gaa, gab, gbb), h_r);
            assert!(
                (ana.v_rho_b - fd_vrb).abs() < 1e-9,
                "{func:?} v_rho_b mismatch: ana={} vs fd={}",
                ana.v_rho_b,
                fd_vrb
            );
        }
        if gaa > 5.0 * h_g {
            let fd_gaa = fd4(&|d| f_only(ra, rb, gaa + d, gab, gbb), h_g);
            assert!(
                (ana.v_gamma_aa - fd_gaa).abs() < 1e-9,
                "{func:?} v_gamma_aa mismatch: ana={} vs fd={}",
                ana.v_gamma_aa,
                fd_gaa
            );
        }
        if gaa > 5.0 * h_g && gbb > 5.0 * h_g {
            let fd_gab = fd4(&|d| f_only(ra, rb, gaa, gab + d, gbb), h_g);
            assert!(
                (ana.v_gamma_ab - fd_gab).abs() < 1e-9,
                "{func:?} v_gamma_ab mismatch: ana={} vs fd={}",
                ana.v_gamma_ab,
                fd_gab
            );
        }
        if gbb > 5.0 * h_g {
            let fd_gbb = fd4(&|d| f_only(ra, rb, gaa, gab, gbb + d), h_g);
            assert!(
                (ana.v_gamma_bb - fd_gbb).abs() < 1e-9,
                "{func:?} v_gamma_bb mismatch: ana={} vs fd={}",
                ana.v_gamma_bb,
                fd_gbb
            );
        }
    }

    #[test]
    fn test_all_xc_functionals_analytic_derivatives_vs_fd() {
        let all_funcs = [
            XcFunctional::Svwn5,
            XcFunctional::Blyp,
            XcFunctional::Pbe,
            XcFunctional::B3lyp,
            XcFunctional::B3lyp5,
            XcFunctional::Pbe0,
        ];

        for func in all_funcs {
            // 1. General open-shell asymmetric point
            assert_xc_partials_match_fd(func, 0.37, 0.19, 0.41, 0.12, 0.23);
            // 2. Closed-shell symmetric point
            assert_xc_partials_match_fd(func, 0.25, 0.25, 0.18, 0.18, 0.18);
            // 3. High-density core region
            assert_xc_partials_match_fd(func, 14.2, 13.8, 120.0, 110.0, 105.0);
            // 4. Low-density tail region
            assert_xc_partials_match_fd(func, 1.2e-4, 0.7e-4, 3.5e-7, 1.1e-7, 2.0e-7);
            // 5. Fully spin-polarized 1-electron point (rb = 0, gbb = gab = 0)
            assert_xc_partials_match_fd(func, 0.42, 0.0, 0.31, 0.0, 0.0);
        }
    }

    #[test]
    fn test_lyp_one_electron_correlation_vanishes_identically() {
        // Fundamental property of LYP: one-electron correlation energy is identically zero!
        let out = lyp_c(0.53, 0.0, 0.77, 0.0, 0.0);
        assert!(out.f_xc.abs() < 1e-16);
        assert!(out.v_rho_a.abs() < 1e-16);
        assert!(out.v_gamma_aa.abs() < 1e-16);
    }
}
